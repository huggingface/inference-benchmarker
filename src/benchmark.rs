use crate::requests::{TextGenerationBackend, TextRequestGenerator, TokenizeOptions};
use crate::results::{BenchmarkReport, BenchmarkResults};
use crate::scheduler::{ExecutorType, SchedulerProgress};
use crate::{executors, scheduler};
use log::{debug, info};
use serde::Serialize;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc::{Receiver, Sender};
use tokio::sync::{broadcast, mpsc, Mutex};

const THROUGHPUT_BUDGET: f64 = 1.2; // sweep up to 120% of max throughput

#[derive(Clone, Debug, strum_macros::Display, Serialize)]
pub enum BenchmarkKind {
    Throughput,
    Sweep,
    Rate,
}

pub struct MessageEvent {
    pub message: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub level: log::Level,
}

pub struct BenchmarkEvent {
    pub id: String,
    pub scheduler_type: ExecutorType,
    pub request_throughput: Option<f64>,
    pub progress: f64,
    pub results: Option<BenchmarkResults>,
    pub successful_requests: u64,
    pub failed_requests: u64,
}

pub enum Event {
    BenchmarkStart(BenchmarkEvent),
    BenchmarkProgress(BenchmarkEvent),
    BenchmarkEnd(BenchmarkEvent),
    Message(MessageEvent),
    BenchmarkReportEnd(String),
    BenchmarkError(String),
}

pub struct Benchmark {
    start_time: Option<tokio::time::Instant>,
    end_time: Option<tokio::time::Instant>,
    backend: Box<dyn TextGenerationBackend + Send + Sync>,
    requests: Arc<Mutex<dyn TextRequestGenerator + Send>>,
    report: BenchmarkReport,
    pub(crate) config: BenchmarkConfig,
    event_bus: mpsc::UnboundedSender<Event>,
    stop_sender: broadcast::Sender<()>,
}

#[serde_with::serde_as]
#[derive(Clone, Serialize)]
pub struct BenchmarkConfig {
    pub max_vus: u64,
    #[serde(rename = "duration_secs")]
    #[serde_as(as = "serde_with::DurationSeconds<u64>")]
    pub duration: Duration,
    pub benchmark_kind: BenchmarkKind,
    #[serde(rename = "warmup_duration_secs")]
    #[serde_as(as = "serde_with::DurationSeconds<u64>")]
    pub warmup_duration: Duration,
    pub rates: Option<Vec<f64>>,
    pub num_rates: u64,
    pub prompt_options: Option<TokenizeOptions>,
    pub decode_options: Option<TokenizeOptions>,
    pub tokenizer: String,
    pub model_name: String,
    pub profile: Option<String>,
    #[serde(rename = "meta")]
    pub extra_metadata: Option<HashMap<String, String>>,
    pub run_id: String,
}

impl BenchmarkConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.max_vus == 0 {
            return Err(anyhow::anyhow!("max_vus must be greater than 0"));
        }
        if self.duration.as_secs() == 0 {
            return Err(anyhow::anyhow!("duration must be greater than 0"));
        }
        if self.warmup_duration.as_secs() == 0 {
            return Err(anyhow::anyhow!("warmup_duration must be greater than 0"));
        }
        match self.benchmark_kind {
            BenchmarkKind::Throughput => {
                if self.rates.is_some() {
                    return Err(anyhow::anyhow!(
                        "rates must not be specified for throughput benchmark"
                    ));
                }
            }
            BenchmarkKind::Sweep => {
                if self.rates.is_some() {
                    return Err(anyhow::anyhow!(
                        "rates must not be specified for sweep benchmark"
                    ));
                }
            }
            BenchmarkKind::Rate => {
                if self.rates.is_none() {
                    return Err(anyhow::anyhow!(
                        "rates must be specified for rate benchmark"
                    ));
                }
            }
        }
        Ok(())
    }
}

pub struct BenchmarkProgress {
    id: String,
    progress: SchedulerProgress,
}

impl Benchmark {
    pub fn new(
        config: BenchmarkConfig,
        backend: Box<dyn TextGenerationBackend + Send + Sync>,
        requests: Arc<Mutex<dyn TextRequestGenerator + Send>>,
        event_bus: mpsc::UnboundedSender<Event>,
        stop_sender: broadcast::Sender<()>,
    ) -> Benchmark {
        Benchmark {
            start_time: None,
            end_time: None,
            report: BenchmarkReport::new(),
            config: config.clone(),
            backend,
            requests,
            event_bus,
            stop_sender,
        }
    }

    pub fn get_report(&self) -> BenchmarkReport {
        self.report.clone()
    }

    pub async fn run(&mut self) -> anyhow::Result<BenchmarkReport> {
        self.start_time = Some(tokio::time::Instant::now());
        self.report.start();
        info!("Prewarming backend");
        self.warmup().await?;
        info!("Prewarm complete");
        match self.config.benchmark_kind {
            BenchmarkKind::Throughput => {
                self.run_throughput().await?;
            }
            BenchmarkKind::Sweep => {
                self.run_sweep().await?;
            }
            BenchmarkKind::Rate => {
                self.run_rates().await?;
            }
        }
        self.end_time = Some(tokio::time::Instant::now());
        self.event_bus.send(Event::Message(MessageEvent {
            message: format!(
                "Benchmark complete in {:?}",
                self.duration().expect("duration exists")
            ),
            timestamp: chrono::Utc::now(),
            level: log::Level::Info,
        }))?;
        self.report.end();
        Ok(self.report.clone())
    }

    pub fn duration(&self) -> Option<std::time::Duration> {
        match (self.start_time, self.end_time) {
            (Some(start), Some(end)) => Some(end.duration_since(start)),
            _ => None,
        }
    }

    async fn handle_progress(&self, id: String) -> Sender<Option<SchedulerProgress>> {
        let (tx, mut rx): (
            Sender<Option<SchedulerProgress>>,
            Receiver<Option<SchedulerProgress>>,
        ) = mpsc::channel(8);
        let event_bus = self.event_bus.clone();
        tokio::spawn(async move {
            while let Some(event) = rx.recv().await {
                match event {
                    None => {
                        break;
                    }
                    Some(progress) => {
                        let progress_evt = BenchmarkProgress {
                            id: id.clone(),
                            progress,
                        };
                        let _ = event_bus.send(Event::BenchmarkProgress(BenchmarkEvent {
                            id: progress_evt.id,
                            scheduler_type: ExecutorType::ConstantVUs,
                            request_throughput: Some(progress_evt.progress.requests_throughput),
                            progress: progress_evt.progress.progress,
                            successful_requests: progress_evt.progress.successful_requests,
                            failed_requests: progress_evt.progress.failed_requests,
                            results: None,
                        }));
                    }
                }
            }
        });
        tx
    }

    pub async fn warmup(&mut self) -> anyhow::Result<()> {
        // run a warmup benchmark to prewarm the server

        let id = "warmup".to_string();

        // notify start event
        self.event_bus.send(Event::BenchmarkStart(BenchmarkEvent {
            id: id.to_string(),
            scheduler_type: ExecutorType::ConstantVUs,
            request_throughput: None,
            progress: 0.0,
            results: None,
            successful_requests: 0,
            failed_requests: 0,
        }))?;

        // create progress handler
        let tx = self.handle_progress(id.clone()).await;

        // start scheduler
        let mut scheduler = scheduler::Scheduler::new(
            id,
            self.backend.clone(),
            ExecutorType::ConstantVUs,
            executors::ExecutorConfig {
                max_vus: 1,
                duration: self.config.warmup_duration,
                rate: None,
            },
            self.requests.clone(),
            tx.clone(),
            self.stop_sender.clone(),
        );
        scheduler.run().await?;

        let results = scheduler.get_results().lock().await.clone();
        self.report.add_benchmark_result(results.clone());

        // send None to close the progress handler
        tx.send(None).await.unwrap();

        // notify end event
        self.event_bus.send(Event::BenchmarkEnd(BenchmarkEvent {
            id: "warmup".to_string(),
            scheduler_type: ExecutorType::ConstantVUs,
            request_throughput: results.successful_request_rate().ok(),
            progress: 100.0,
            results: Some(results.clone()),
            successful_requests: results.successful_requests() as u64,
            failed_requests: results.failed_requests() as u64,
        }))?;
        Ok(())
    }

    pub async fn run_throughput(&mut self) -> anyhow::Result<()> {
        info!("Running throughput benchmark");

        let id = "throughput".to_string();

        // notify start event
        self.event_bus.send(Event::BenchmarkStart(BenchmarkEvent {
            id: id.clone(),
            scheduler_type: ExecutorType::ConstantVUs,
            request_throughput: None,
            progress: 0.0,
            results: None,
            successful_requests: 0,
            failed_requests: 0,
        }))?;

        // create progress handler
        let tx = self.handle_progress(id.clone()).await;

        // start scheduler
        let mut scheduler = scheduler::Scheduler::new(
            id.clone(),
            self.backend.clone(),
            ExecutorType::ConstantVUs,
            executors::ExecutorConfig {
                max_vus: self.config.max_vus,
                duration: self.config.duration,
                rate: None,
            },
            self.requests.clone(),
            tx.clone(),
            self.stop_sender.clone(),
        );
        scheduler.run().await?;
        let results = scheduler.get_results().lock().await.clone();
        let rate = results.successful_request_rate().ok();
        self.report.add_benchmark_result(results.clone());

        // send None to close the progress handler
        tx.send(None).await.unwrap();

        // notify end event
        self.event_bus.send(Event::BenchmarkEnd(BenchmarkEvent {
            id: id.clone(),
            scheduler_type: ExecutorType::ConstantVUs,
            request_throughput: rate,
            progress: 100.0,
            results: Some(results.clone()),
            successful_requests: results.successful_requests() as u64,
            failed_requests: results.failed_requests() as u64,
        }))?;
        Ok(())
    }

    pub async fn run_sweep(&mut self) -> anyhow::Result<()> {
        // run a throughput benchmark to retrieve the maximum throughput of server
        self.run_throughput().await?;
        // get the max throughput from the second benchmark result (first is warmup)
        let throughput_results = &self.report.get_results()[1];
        let max_throughput = throughput_results.successful_request_rate()?;
        let max_tokens_throughput = throughput_results.token_throughput_secs()?;
        // notify event bus
        self.event_bus.send(Event::Message(MessageEvent {
            message: format!(
                "Max throughput detected at: {:.2} req/s | {:.2} tokens/s",
                max_throughput, max_tokens_throughput
            ),
            timestamp: chrono::Utc::now(),
            level: log::Level::Info,
        }))?;
        // run a sweep benchmark for 10 different rates from 1req/s to max throughput
        let mut rates = Vec::new();
        let num_rates = self.config.num_rates;
        for i in 1..=num_rates {
            rates.push(i as f64 * max_throughput * THROUGHPUT_BUDGET / num_rates as f64);
        }
        for rate in rates {
            self.run_rate(rate).await?;
        }
        Ok(())
    }

    pub async fn run_rates(&mut self) -> anyhow::Result<()> {
        let rates = self.config.rates.clone().expect("config already validated");
        for rate in rates {
            self.run_rate(rate).await?;
        }
        Ok(())
    }

    pub async fn run_rate(&mut self, rate: f64) -> anyhow::Result<()> {
        debug!("Running benchmark with rate: {} req/s", rate);

        let id = format!("constant@{:.2}req/s", rate);

        // notify start event
        self.event_bus.send(Event::BenchmarkStart(BenchmarkEvent {
            id: id.clone(),
            scheduler_type: ExecutorType::ConstantArrivalRate,
            request_throughput: None,
            progress: 0.0,
            results: None,
            successful_requests: 0,
            failed_requests: 0,
        }))?;

        // create progress handler
        let tx = self.handle_progress(id.clone()).await;

        // start scheduler
        let mut scheduler = scheduler::Scheduler::new(
            id,
            self.backend.clone(),
            scheduler::ExecutorType::ConstantArrivalRate,
            executors::ExecutorConfig {
                max_vus: self.config.max_vus,
                duration: self.config.duration,
                rate: Some(rate),
            },
            self.requests.clone(),
            tx.clone(),
            self.stop_sender.clone(),
        );
        scheduler.run().await?;
        let results = scheduler.get_results().lock().await.clone();
        self.report.add_benchmark_result(results.clone());

        // send None to close the progress handler
        tx.send(None).await.unwrap();

        // notify end event
        self.event_bus.send(Event::BenchmarkEnd(BenchmarkEvent {
            id: format!("constant@{:.2}req/s", rate),
            scheduler_type: ExecutorType::ConstantArrivalRate,
            request_throughput: results.successful_request_rate().ok(),
            progress: 100.0,
            results: Some(results.clone()),
            successful_requests: results.successful_requests() as u64,
            failed_requests: results.failed_requests() as u64,
        }))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::requests::DummyTextGenerationBackend;
    use crate::requests::DummyTextRequestGenerator;
    use std::time::Duration;

    #[tokio::test]
    async fn test_sweep_benchmark_timings() {
        let generation_time = Duration::from_secs(2);
        let (event_tx, mut _event_rx) = tokio::sync::mpsc::unbounded_channel();
        let (stop_sender, _) = tokio::sync::broadcast::channel(1);
        let backend = Box::new(DummyTextGenerationBackend::new(Duration::from_secs(
            generation_time.as_secs(),
        )));
        let requests_generator = Arc::from(Mutex::from(DummyTextRequestGenerator::new()));
        let mut benchmark = Benchmark::new(
            BenchmarkConfig {
                max_vus: 100,
                duration: Duration::from_secs(10),
                benchmark_kind: BenchmarkKind::Sweep,
                warmup_duration: Duration::from_secs(1),
                rates: None,
                num_rates: 2,
                prompt_options: None,
                decode_options: None,
                tokenizer: "gpt2".to_string(),
                model_name: "gpt2".to_string(),
                profile: None,
                extra_metadata: None,
                run_id: "test".to_string(),
            },
            backend,
            requests_generator,
            event_tx,
            stop_sender,
        );
        let report = benchmark.run().await.unwrap();
        assert_eq!(report.get_results().len(), 4);
        let generation_time_per_token_milli = generation_time.as_millis() as i128 / 10;
        for result in report.get_results() {
            let delta_ttft = result.time_to_first_token_avg().unwrap().as_millis() as i128
                - generation_time_per_token_milli; // Dummy backends generates 10 tokens
            let delta_itl = result.inter_token_latency_avg().unwrap().as_millis() as i128
                - generation_time_per_token_milli;
            let delta_e2e = result.e2e_latency_avg().unwrap().as_millis() as i128
                - generation_time.as_millis() as i128;
            let allowed_error_ms = 3; // allow error margin for timing tests
            assert!(
                delta_ttft.abs() <= allowed_error_ms,
                "time_to_first_token_delta: {:?}, expected {:?}",
                delta_ttft.abs(),
                allowed_error_ms
            );
            assert!(
                delta_itl.abs() <= allowed_error_ms,
                "inter_token_latency_delta: {:?}, expected {:?}",
                delta_itl.abs(),
                allowed_error_ms
            );
            assert!(
                delta_e2e.abs() <= allowed_error_ms * 10, // Cumulative error for 10 tokens
                "e2e_latency_delta: {:?}, expected {:?}",
                delta_e2e.abs(),
                allowed_error_ms * 10
            );
        }
    }
}
