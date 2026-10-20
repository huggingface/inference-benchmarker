use crate::executors::{
    ConstantArrivalRateExecutor, ConstantVUsExecutor, Executor, ExecutorConfig,
};
use crate::record::{RecordSink, RequestRecord};
use crate::requests::{
    TextGenerationAggregatedResponse, TextGenerationBackend, TextRequestGenerator,
};
use crate::results::BenchmarkErrors::NoResponses;
use crate::results::BenchmarkResults;
use log::{debug, trace};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::mpsc::{Sender, UnboundedReceiver, UnboundedSender};
use tokio::sync::{broadcast, Mutex};

#[derive(Clone, Debug, PartialEq, strum_macros::Display, Serialize, Deserialize)]
pub enum ExecutorType {
    ConstantVUs,
    ConstantArrivalRate,
    /// Each request of a list once, optionally at recorded times.
    Scheduled,
}

pub struct Scheduler {
    id: String,
    executor: Arc<Mutex<Box<dyn Executor + Send>>>,
    requests_generator: Arc<Mutex<dyn TextRequestGenerator + Send>>,
    results: Arc<Mutex<BenchmarkResults>>,
    progress_tx: Sender<Option<SchedulerProgress>>,
    stop_sender: broadcast::Sender<()>,
    records: Option<RecordSink>,
}

pub struct SchedulerProgress {
    pub progress: f64,
    pub requests_throughput: f64,
    pub successful_requests: u64,
    pub failed_requests: u64,
}

impl Scheduler {
    pub fn new(
        id: String,
        backend: Box<dyn TextGenerationBackend + Send + Sync>,
        executor_type: ExecutorType,
        config: ExecutorConfig,
        requests_generator: Arc<Mutex<dyn TextRequestGenerator + Send>>,
        progress_tx: Sender<Option<SchedulerProgress>>,
        stop_sender: broadcast::Sender<()>,
    ) -> anyhow::Result<Scheduler> {
        let executor: Box<dyn Executor + Send> = match executor_type {
            ExecutorType::ConstantVUs => Box::new(ConstantVUsExecutor::new(
                backend,
                config.max_vus,
                config.duration,
            )),
            ExecutorType::ConstantArrivalRate => {
                let rate = config.rate.ok_or_else(|| {
                    anyhow::anyhow!("Rate must be specified for ConstantArrivalRateExecutor")
                })?;
                Box::new(ConstantArrivalRateExecutor::new(
                    backend,
                    config.max_vus,
                    config.duration,
                    rate,
                ))
            }
            ExecutorType::Scheduled => {
                return Err(anyhow::anyhow!(
                    "A scheduled executor needs its requests: use Scheduler::with_executor"
                ))
            }
        };
        Ok(Self::with_executor(
            id,
            executor,
            executor_type,
            config,
            requests_generator,
            progress_tx,
            stop_sender,
        ))
    }

    /// Run any executor; `executor_type` and `config` describe it in the results.
    pub fn with_executor(
        id: String,
        executor: Box<dyn Executor + Send>,
        executor_type: ExecutorType,
        config: ExecutorConfig,
        requests_generator: Arc<Mutex<dyn TextRequestGenerator + Send>>,
        progress_tx: Sender<Option<SchedulerProgress>>,
        stop_sender: broadcast::Sender<()>,
    ) -> Scheduler {
        Scheduler {
            results: Arc::from(Mutex::from(BenchmarkResults::new(
                id.clone(),
                executor_type,
                config,
            ))),
            id,
            executor: Arc::from(Mutex::from(executor)),
            requests_generator,
            progress_tx,
            stop_sender,
            records: None,
        }
    }

    /// Send a [`RequestRecord`] for every response.
    pub fn with_records(mut self, records: Option<RecordSink>) -> Self {
        self.records = records;
        self
    }

    pub async fn run(&mut self) -> anyhow::Result<BenchmarkResults> {
        debug!("Starting scheduler '{}'", self.id);
        // add responses to the benchmark result as they arrive
        let (tx, mut rx): (
            UnboundedSender<TextGenerationAggregatedResponse>,
            UnboundedReceiver<TextGenerationAggregatedResponse>,
        ) = tokio::sync::mpsc::unbounded_channel();
        let results = self.results.clone();
        let progress_tx = self.progress_tx.clone();
        let mut stop_receiver = self.stop_sender.subscribe();
        let req_gen = self.requests_generator.clone();
        let expected_requests = self.executor.lock().await.expected_requests();
        let records = self.records.clone();
        let step_id = self.id.clone();
        let consumer = tokio::spawn(async move {
            tokio::select! {
                _ = stop_receiver.recv() => {
                    debug!("Received stop signal, stopping benchmark");
                }
                _ = async{
                    while let Some(mut response) = rx.recv().await{
                        trace!("Received response: {:?}", response);
                        if response.ended {
                            return;
                        }
                        if let Some(request) = response.request.clone() {
                            let text = response.response.clone().unwrap_or_default();
                            if !response.failed {
                                response.check = request.check.as_ref().map(|c| c.evaluate(&text));
                                // a failed turn ends its conversation
                                req_gen.lock().await.callback(request, &text);
                            }
                        }
                        if let Some(sink) = &records {
                            let record = RequestRecord::from_response(&step_id, &response, sink.include_text);
                            let _ = sink.tx.send(record).await;
                        }
                        let mut result = results.lock().await;
                        result.add_response(response);
                        let progress = match expected_requests {
                            Some(n) => 100.0 * result.total_requests() as f64 / n.max(1) as f64,
                            None => {
                                let expected_duration = result.executor_config().duration.as_secs_f64();
                                let start_time = result.start_time().unwrap_or(tokio::time::Instant::now());
                                100.0 * (1.0 - (expected_duration - start_time.elapsed().as_secs_f64()) / expected_duration)
                            }
                        };
                        let _ = progress_tx.send(Some(SchedulerProgress {
                            progress: progress.min(100.0),
                            requests_throughput: result.successful_request_rate().unwrap_or_default(),
                            successful_requests: result.successful_requests() as u64,
                            failed_requests: result.failed_requests() as u64,
                        })).await;
                    }
                }=>{}
            }
        });
        self.executor
            .lock()
            .await
            .run(
                self.requests_generator.clone(),
                tx,
                self.stop_sender.clone(),
            )
            .await;
        // every executor ends with an `ended` marker, or the run was stopped
        let _ = consumer.await;
        let results = self.results.lock().await.clone();
        debug!("{:?}", results);
        if results.successful_requests() == 0 {
            Err(anyhow::anyhow!(NoResponses))
        } else {
            Ok(results)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::requests::OpenAITextGenerationBackend;
    use std::time::Duration;
    use tokio::time;

    #[tokio::test]
    async fn test_constant_arrival_rate_scheduler() {
        let (progress_tx, _) = tokio::sync::mpsc::channel(10000);
        let (stop_sender, _) = tokio::sync::broadcast::channel(1);
        let backend = Box::new(crate::requests::DummyTextGenerationBackend::new(
            Duration::from_secs(1),
        ));
        let requests_generator = Arc::from(Mutex::from(
            crate::requests::DummyTextRequestGenerator::new(),
        ));
        let mut scheduler = Scheduler::new(
            "test".to_string(),
            backend,
            ExecutorType::ConstantArrivalRate,
            ExecutorConfig {
                max_vus: 800,
                duration: std::time::Duration::from_secs(10),
                rate: Some(20.0),
            },
            requests_generator,
            progress_tx,
            stop_sender,
        )
        .unwrap();
        let results = scheduler.run().await.unwrap();
        assert_eq!(results.successful_requests(), 180); // 20 requests per second for 10 seconds - 20 requests for last second as the backend has a 1 second delay
    }

    #[tokio::test]
    async fn test_constant_vus_scheduler() {
        let (progress_tx, _) = tokio::sync::mpsc::channel(10000);
        let (stop_sender, _) = broadcast::channel(1);
        let backend = Box::new(crate::requests::DummyTextGenerationBackend::new(
            Duration::from_secs(1),
        ));
        let requests_generator = Arc::from(Mutex::from(
            crate::requests::DummyTextRequestGenerator::new(),
        ));
        let mut scheduler = Scheduler::new(
            "test".to_string(),
            backend,
            ExecutorType::ConstantVUs,
            ExecutorConfig {
                max_vus: 800,
                duration: Duration::from_secs(10),
                rate: None,
            },
            requests_generator,
            progress_tx,
            stop_sender,
        )
        .unwrap();
        let results = scheduler.run().await.unwrap();
        assert!(
            results.successful_requests() > 7200,
            "Expected at least 7200 requests, got {}",
            results.successful_requests()
        );
    }

    #[tokio::test]
    async fn test_constant_arrival_rate_openai_backend() {
        let (progress_tx, _) = tokio::sync::mpsc::channel(10000);
        let (stop_sender, _) = tokio::sync::broadcast::channel(1);
        let mut s = mockito::Server::new_async().await;
        s.mock("POST", "/v1/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_chunked_body(|w| {
                w.write_all(b"data: {\"choices\": [{\"message\": null, \"finish_reason\": null, \"delta\": {\"content\": \"Hello, world!\"}}]}\n\n").unwrap();
                std::thread::sleep(Duration::from_millis(500));
                w.write_all(b"data: {\"choices\": [{\"message\": {\"content\": \"Hello, world!Hello, world!Hello, world!Hello, world!\", \"role\": \"user\"}, \"finish_reason\": \"stop\", \"delta\": {\"content\": \"Hello, world!\"}}]}\n\n").unwrap();
                w.write_all(b"data: [DONE]\n\n")
            })
            .create_async().await;
        let url = s.url().parse().unwrap();
        let tokenizer = crate::test_util::tokenizer();
        let backend = OpenAITextGenerationBackend::try_new(
            "".to_string(),
            url,
            "gpt2".to_string(),
            tokenizer,
            time::Duration::from_secs(10),
        )
        .unwrap();
        let requests_generator = Arc::from(Mutex::from(
            crate::requests::DummyTextRequestGenerator::new(),
        ));
        let mut scheduler = Scheduler::new(
            "test".to_string(),
            Box::new(backend),
            ExecutorType::ConstantArrivalRate,
            ExecutorConfig {
                max_vus: 800,
                duration: Duration::from_secs(10),
                rate: Some(50.0),
            },
            requests_generator,
            progress_tx,
            stop_sender,
        )
        .unwrap();
        let results = scheduler.run().await.unwrap();
        assert_eq!(results.successful_requests(), 475); // 25 expected missing requests due to the 500ms delay in the backend
    }
}
