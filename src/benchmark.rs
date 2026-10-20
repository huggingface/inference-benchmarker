use crate::cost::GpuCost;
use crate::executors::{Executor, ExecutorConfig, ScheduledExecutor};
use crate::record::RecordSink;
use crate::requests::{
    TextGenerationBackend, TextGenerationRequest, TextRequestGenerator, TokenizeOptions,
};
use crate::results::{BenchmarkReport, BenchmarkResults};
use crate::scheduler::{ExecutorType, Scheduler, SchedulerProgress};
use crate::slo::Slo;
use log::{debug, info};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc::{Receiver, Sender};
use tokio::sync::{broadcast, mpsc, Mutex};

const THROUGHPUT_BUDGET: f64 = 1.2; // sweep up to 120% of max throughput
/// Request timeout of runs without a duration (`Once`, `Replay`).
const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(600);

#[derive(Clone, Debug, Default, PartialEq, strum_macros::Display, Serialize, Deserialize)]
pub enum BenchmarkKind {
    /// Constant concurrency (`max_vus`) for `duration`.
    Throughput,
    /// Throughput, then `num_rates` constant rates up to 120% of the maximum.
    #[default]
    Sweep,
    /// The given constant `rates`.
    Rate,
    /// Every request of the workload exactly once, as fast as `max_vus` allows:
    /// accuracy and conversation suites.
    Once,
    /// Recorded traffic, each request at its recorded offset divided by `replay_speed`.
    Replay,
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

impl BenchmarkEvent {
    fn started(id: &str, scheduler_type: ExecutorType) -> Self {
        Self {
            id: id.to_string(),
            scheduler_type,
            request_throughput: None,
            progress: 0.0,
            results: None,
            successful_requests: 0,
            failed_requests: 0,
        }
    }

    fn ended(id: &str, scheduler_type: ExecutorType, results: &BenchmarkResults) -> Self {
        Self {
            id: id.to_string(),
            scheduler_type,
            request_throughput: results.successful_request_rate().ok(),
            progress: 100.0,
            results: Some(results.clone()),
            successful_requests: results.successful_requests() as u64,
            failed_requests: results.failed_requests() as u64,
        }
    }
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
    /// The chains of a `Once` or `Replay` run (a request, or a conversation's turns).
    schedule: Vec<Vec<Arc<TextGenerationRequest>>>,
    records: Option<RecordSink>,
    report: BenchmarkReport,
    pub(crate) config: BenchmarkConfig,
    event_bus: mpsc::UnboundedSender<Event>,
    stop_sender: broadcast::Sender<()>,
}

fn default_replay_speed() -> f64 {
    1.0
}

#[serde_with::serde_as]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub max_vus: u64,
    #[serde(rename = "duration_secs")]
    #[serde_as(as = "serde_with::DurationSeconds<u64>")]
    pub duration: Duration,
    pub benchmark_kind: BenchmarkKind,
    /// 0 skips the warmup.
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
    /// `Replay` only: 1.0 is real time, 2.0 twice as fast, 0 as fast as possible.
    #[serde(default = "default_replay_speed")]
    pub replay_speed: f64,
    /// Per-request timeout; defaults to `duration`, or 10 minutes without one.
    #[serde(default, rename = "request_timeout_secs")]
    #[serde_as(as = "Option<serde_with::DurationSeconds<u64>>")]
    pub request_timeout: Option<Duration>,
    #[serde(default)]
    pub slo: Option<Slo>,
    #[serde(default)]
    pub gpu_cost: Option<GpuCost>,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            max_vus: 0,
            duration: Duration::ZERO,
            benchmark_kind: BenchmarkKind::default(),
            warmup_duration: Duration::ZERO,
            rates: None,
            num_rates: 0,
            prompt_options: None,
            decode_options: None,
            tokenizer: String::new(),
            model_name: String::new(),
            profile: None,
            extra_metadata: None,
            run_id: String::new(),
            replay_speed: default_replay_speed(),
            request_timeout: None,
            slo: None,
            gpu_cost: None,
        }
    }
}

impl BenchmarkConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.max_vus == 0 {
            return Err(anyhow::anyhow!("max_vus must be greater than 0"));
        }
        let scheduled = matches!(
            self.benchmark_kind,
            BenchmarkKind::Once | BenchmarkKind::Replay
        );
        if !scheduled && self.duration.is_zero() {
            return Err(anyhow::anyhow!("duration must be greater than 0"));
        }
        match self.benchmark_kind {
            BenchmarkKind::Rate => {
                if self.rates.as_ref().is_none_or(|r| r.is_empty()) {
                    return Err(anyhow::anyhow!(
                        "rates must be specified for rate benchmark"
                    ));
                }
                if self.rates.iter().flatten().any(|r| *r <= 0.0) {
                    return Err(anyhow::anyhow!("rates must be greater than 0"));
                }
            }
            ref kind => {
                if self.rates.is_some() {
                    return Err(anyhow::anyhow!(
                        "rates must not be specified for {} benchmark",
                        kind.to_string().to_lowercase()
                    ));
                }
            }
        }
        if !self.replay_speed.is_finite() || self.replay_speed < 0.0 {
            return Err(anyhow::anyhow!("replay_speed must be 0 or more"));
        }
        if let Some(slo) = &self.slo {
            slo.validate()?;
        }
        if let Some(cost) = &self.gpu_cost {
            cost.validate()?;
        }
        Ok(())
    }

    pub fn request_timeout(&self) -> Duration {
        self.request_timeout.unwrap_or(if self.duration.is_zero() {
            DEFAULT_REQUEST_TIMEOUT
        } else {
            self.duration
        })
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
            schedule: Vec::new(),
            records: None,
            event_bus,
            stop_sender,
        }
    }

    /// What a `Once` or `Replay` run sends: chains of one request, or the
    /// turns of one conversation.
    pub fn with_schedule(mut self, schedule: Vec<Vec<Arc<TextGenerationRequest>>>) -> Self {
        self.schedule = schedule;
        self
    }

    /// Send a trace record for every request.
    pub fn with_records(mut self, records: Option<RecordSink>) -> Self {
        self.records = records;
        self
    }

    pub fn get_report(&self) -> BenchmarkReport {
        self.report.clone()
    }

    pub async fn run(&mut self) -> anyhow::Result<BenchmarkReport> {
        self.start_time = Some(tokio::time::Instant::now());
        self.report.start();
        if !self.config.warmup_duration.is_zero() {
            info!("Prewarming backend");
            self.warmup().await?;
            info!("Prewarm complete");
        }
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
            BenchmarkKind::Once => {
                self.run_scheduled("once", None).await?;
            }
            BenchmarkKind::Replay => {
                let speed = self.config.replay_speed;
                self.run_scheduled("replay", Some(speed)).await?;
            }
        }
        self.end_time = Some(tokio::time::Instant::now());
        self.message(format!(
            "Benchmark complete in {:?}",
            self.duration().unwrap_or_default()
        ));
        self.report.end();
        Ok(self.report.clone())
    }

    pub fn duration(&self) -> Option<std::time::Duration> {
        match (self.start_time, self.end_time) {
            (Some(start), Some(end)) => Some(end.duration_since(start)),
            _ => None,
        }
    }

    /// Events are best effort: a caller that stopped listening must not stop the run.
    fn emit(&self, event: Event) {
        let _ = self.event_bus.send(event);
    }

    fn message(&self, message: String) {
        self.emit(Event::Message(MessageEvent {
            message,
            timestamp: chrono::Utc::now(),
            level: log::Level::Info,
        }));
    }

    async fn handle_progress(
        &self,
        id: String,
        scheduler_type: ExecutorType,
    ) -> Sender<Option<SchedulerProgress>> {
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
                            scheduler_type: scheduler_type.clone(),
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

    /// Run one step and add its results to the report. `executor` overrides the
    /// executor `executor_type` would build (scheduled steps bring their own).
    async fn run_step(
        &mut self,
        id: &str,
        executor_type: ExecutorType,
        config: ExecutorConfig,
        executor: Option<Box<dyn Executor + Send>>,
    ) -> anyhow::Result<BenchmarkResults> {
        self.emit(Event::BenchmarkStart(BenchmarkEvent::started(
            id,
            executor_type.clone(),
        )));
        let tx = self
            .handle_progress(id.to_string(), executor_type.clone())
            .await;
        let scheduler = match executor {
            Some(executor) => Ok(Scheduler::with_executor(
                id.to_string(),
                executor,
                executor_type.clone(),
                config,
                self.requests.clone(),
                tx.clone(),
                self.stop_sender.clone(),
            )),
            None => Scheduler::new(
                id.to_string(),
                self.backend.clone(),
                executor_type.clone(),
                config,
                self.requests.clone(),
                tx.clone(),
                self.stop_sender.clone(),
            ),
        };
        let results = match scheduler {
            Ok(scheduler) => scheduler.with_records(self.records.clone()).run().await,
            Err(e) => Err(e),
        };
        // close the progress handler, also when the step failed
        let _ = tx.send(None).await;
        let results = results?;
        self.report.add_benchmark_result(results.clone());
        self.emit(Event::BenchmarkEnd(BenchmarkEvent::ended(
            id,
            executor_type,
            &results,
        )));
        Ok(results)
    }

    pub async fn warmup(&mut self) -> anyhow::Result<()> {
        // run a warmup benchmark to prewarm the server
        let config = ExecutorConfig {
            max_vus: 1,
            duration: self.config.warmup_duration,
            rate: None,
        };
        self.run_step("warmup", ExecutorType::ConstantVUs, config, None)
            .await?;
        Ok(())
    }

    pub async fn run_throughput(&mut self) -> anyhow::Result<BenchmarkResults> {
        info!("Running throughput benchmark");
        let config = ExecutorConfig {
            max_vus: self.config.max_vus,
            duration: self.config.duration,
            rate: None,
        };
        self.run_step("throughput", ExecutorType::ConstantVUs, config, None)
            .await
    }

    pub async fn run_sweep(&mut self) -> anyhow::Result<()> {
        // run a throughput benchmark to retrieve the maximum throughput of server
        let throughput_results = self.run_throughput().await?;
        let max_throughput = throughput_results.successful_request_rate()?;
        let max_tokens_throughput = throughput_results.token_throughput_secs()?;
        self.message(format!(
            "Max throughput detected at: {:.2} req/s | {:.2} tokens/s",
            max_throughput, max_tokens_throughput
        ));
        // run a sweep benchmark for `num_rates` rates from 1 req/s to max throughput
        let num_rates = self.config.num_rates;
        for i in 1..=num_rates {
            let rate = i as f64 * max_throughput * THROUGHPUT_BUDGET / num_rates as f64;
            self.run_rate(rate).await?;
        }
        Ok(())
    }

    pub async fn run_rates(&mut self) -> anyhow::Result<()> {
        let rates = self.config.rates.clone().unwrap_or_default();
        for rate in rates {
            self.run_rate(rate).await?;
        }
        Ok(())
    }

    pub async fn run_rate(&mut self, rate: f64) -> anyhow::Result<()> {
        debug!("Running benchmark with rate: {} req/s", rate);
        let config = ExecutorConfig {
            max_vus: self.config.max_vus,
            duration: self.config.duration,
            rate: Some(rate),
        };
        let id = format!("constant@{:.2}req/s", rate);
        self.run_step(&id, ExecutorType::ConstantArrivalRate, config, None)
            .await?;
        Ok(())
    }

    /// Send every scheduled request once; at recorded times when `speed` is set.
    pub async fn run_scheduled(&mut self, id: &str, speed: Option<f64>) -> anyhow::Result<()> {
        if self.schedule.is_empty() {
            return Err(anyhow::anyhow!(
                "A {id} benchmark needs a list of requests or conversations"
            ));
        }
        info!("Running {id} benchmark of {} chains", self.schedule.len());
        let executor = ScheduledExecutor::new(
            self.backend.clone(),
            self.config.max_vus,
            self.schedule.clone(),
            speed,
        );
        let config = executor.config();
        self.run_step(
            id,
            ExecutorType::Scheduled,
            config,
            Some(Box::new(executor)),
        )
        .await?;
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
                ..Default::default()
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
