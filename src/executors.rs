use std::sync::atomic::AtomicI64;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use log::{info, trace, warn};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::{Receiver, Sender, UnboundedSender};
use tokio::sync::{broadcast, Mutex};
use tokio::task::JoinHandle;

use crate::requests::{
    continue_conversation, TextGenerationAggregatedResponse, TextGenerationBackend,
    TextGenerationRequest, TextRequestGenerator,
};

#[serde_with::serde_as]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ExecutorConfig {
    pub max_vus: u64,
    #[serde(rename = "duration_secs")]
    #[serde_as(as = "serde_with::DurationSeconds<u64>")]
    pub duration: Duration,
    pub rate: Option<f64>,
}

#[async_trait]
pub trait Executor {
    async fn run(
        &self,
        requests: Arc<Mutex<dyn TextRequestGenerator + Send>>,
        responses_tx: UnboundedSender<TextGenerationAggregatedResponse>,
        stop_sender: broadcast::Sender<()>,
    );

    /// How many requests the executor will send, when it knows up front.
    /// Progress is then counted in requests instead of time.
    fn expected_requests(&self) -> Option<u64> {
        None
    }
}

pub struct ConstantVUsExecutor {
    config: ExecutorConfig,
    backend: Box<dyn TextGenerationBackend + Send + Sync>,
}

impl ConstantVUsExecutor {
    pub fn new(
        backend: Box<dyn TextGenerationBackend + Send + Sync>,
        max_vus: u64,
        duration: Duration,
    ) -> ConstantVUsExecutor {
        Self {
            backend,
            config: ExecutorConfig {
                max_vus,
                duration,
                rate: None,
            },
        }
    }
}

#[async_trait]
impl Executor for ConstantVUsExecutor {
    async fn run(
        &self,
        requests: Arc<Mutex<dyn TextRequestGenerator + Send>>,
        responses_tx: UnboundedSender<TextGenerationAggregatedResponse>,
        stop_sender: broadcast::Sender<()>,
    ) {
        let start = std::time::Instant::now();
        // channel to handle ending VUs
        let (end_tx, mut end_rx): (Sender<bool>, Receiver<bool>) =
            tokio::sync::mpsc::channel(self.config.max_vus as usize);
        let active_vus = Arc::new(AtomicI64::new(0));
        // start all VUs
        for _ in 0..self.config.max_vus {
            let mut requests_guard = requests.lock().await;
            let request = Arc::from(requests_guard.generate_request());
            drop(requests_guard);
            start_vu(
                self.backend.clone(),
                vec![request],
                responses_tx.clone(),
                end_tx.clone(),
                stop_sender.clone(),
            )
            .await;
            active_vus.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
        let mut stop_receiver = stop_sender.subscribe();
        tokio::select! {
            _ = stop_receiver.recv() => {
                return;
            },
            _ = async {
                // replenish VUs as they finish
                while end_rx.recv().await.is_some() {
                    active_vus.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                    if start.elapsed() > self.config.duration{
                        // signal that the VU work is done
                        let _ = responses_tx.send(TextGenerationAggregatedResponse::new_as_ended());
                        info!("Duration reached, waiting for all VUs to finish...");
                        if active_vus.load(std::sync::atomic::Ordering::SeqCst) == 0 {
                            break;
                        }
                    } else {
                        let mut requests_guard = requests.lock().await;
                        let request = Arc::from(requests_guard.generate_request());
                        drop(requests_guard);
                        active_vus.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        start_vu(self.backend.clone(), vec![request], responses_tx.clone(), end_tx.clone(), stop_sender.clone()).await;
                    }
                }
            }=>{}
        }
    }
}

/// Start a virtual user that sends `chain` in order: one request, or the
/// turns of one conversation, each carrying the model's earlier answers. A
/// failed turn ends the chain.
async fn start_vu(
    backend: Box<dyn TextGenerationBackend + Send + Sync>,
    chain: Vec<Arc<TextGenerationRequest>>,
    responses_tx: UnboundedSender<TextGenerationAggregatedResponse>,
    end_tx: Sender<bool>,
    stop_sender: broadcast::Sender<()>,
) -> JoinHandle<()> {
    let mut stop_receiver = stop_sender.subscribe();
    tokio::spawn(async move {
        tokio::select! {
            _ = stop_receiver.recv() => {
                let _ = end_tx.send(true).await;
            },
            _ = async{
                let mut previous: Option<(Arc<TextGenerationRequest>, TextGenerationAggregatedResponse)> = None;
                for turn in chain {
                    let request = match &previous {
                        None => turn,
                        Some((prev, resp)) => Arc::new(continue_conversation(
                            prev,
                            resp.prompt_tokens(),
                            resp.response.as_deref().unwrap_or_default(),
                            resp.num_generated_tokens,
                            &turn,
                        )),
                    };
                    trace!("VU started with request: {:?}", request);
                    let (tx, mut rx): (Sender<TextGenerationAggregatedResponse>, Receiver<TextGenerationAggregatedResponse>) = tokio::sync::mpsc::channel(1);
                    let req_thread = tokio::spawn({
                        let (backend, request) = (backend.clone(), request.clone());
                        async move { backend.generate(request, tx).await }
                    });
                    // the backend sends one response per request
                    let response = rx.recv().await;
                    let _ = req_thread.await;
                    let Some(response) = response else { break };
                    let failed = response.failed;
                    // ignore errors, if the receiver is gone we want to finish the request
                    // to leave remote server in clean state
                    let _ = responses_tx.send(response.clone());
                    if failed {
                        break;
                    }
                    previous = Some((request, response));
                }
                // signal that the VU work is done
                let _ = end_tx.send(true).await;
            }=>{}
        }
    })
}

pub struct ConstantArrivalRateExecutor {
    config: ExecutorConfig,
    backend: Box<dyn TextGenerationBackend + Send + Sync>,
}

impl ConstantArrivalRateExecutor {
    pub fn new(
        backend: Box<dyn TextGenerationBackend + Send + Sync>,
        max_vus: u64,
        duration: Duration,
        rate: f64,
    ) -> ConstantArrivalRateExecutor {
        Self {
            backend,
            config: ExecutorConfig {
                max_vus,
                duration,
                rate: Some(rate),
            },
        }
    }
}

#[async_trait]
impl Executor for ConstantArrivalRateExecutor {
    async fn run(
        &self,
        requests: Arc<Mutex<dyn TextRequestGenerator + Send>>,
        responses_tx: UnboundedSender<TextGenerationAggregatedResponse>,
        stop_sender: broadcast::Sender<()>,
    ) {
        let start = std::time::Instant::now();
        let active_vus = Arc::new(AtomicI64::new(0));
        // channel to handle ending VUs
        let (end_tx, mut end_rx): (Sender<bool>, Receiver<bool>) =
            tokio::sync::mpsc::channel(self.config.max_vus as usize);
        let rate = self.config.rate.expect("checked in new()");
        // spawn new VUs every `tick_ms` to reach the expected `rate` per second, until the duration is reached
        let tick_ms = 10;
        let mut interval = tokio::time::interval(Duration::from_millis(tick_ms));

        let backend = self.backend.clone();
        let duration = self.config.duration;
        let max_vus = self.config.max_vus;
        let active_vus_thread = active_vus.clone();
        let mut stop_receiver_signal = stop_sender.subscribe();
        let vu_thread = tokio::spawn(async move {
            tokio::select! {
                _ = stop_receiver_signal.recv() => {},
                _= async {
                    let mut spawn_queue = 0.; // start with at least one VU
                    while start.elapsed() < duration {
                        spawn_queue += rate * (tick_ms as f64) / 1000.0;
                        // delay spawning if we can't spawn a full VU yet
                        if spawn_queue < 1.0 {
                            interval.tick().await;
                            continue;
                        }
                        // spawn VUs, keep track of the fraction of VU to spawn for the next iteration
                        let to_spawn = spawn_queue.floor() as u64;
                        spawn_queue -= to_spawn as f64;
                        for _ in 0..to_spawn {
                            if active_vus_thread.load(std::sync::atomic::Ordering::SeqCst) < max_vus as i64 {
                                let mut requests_guard = requests.lock().await;
                                let request = Arc::from(requests_guard.generate_request());
                                start_vu(backend.clone(), vec![request.clone()], responses_tx.clone(), end_tx.clone(),stop_sender.clone()).await;
                                active_vus_thread.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                            } else {
                                warn!("Max VUs reached, skipping request");
                                break;
                            }
                        }
                        interval.tick().await;
                    }
                    // signal that the VU work is done
                    info!("Duration reached, waiting for all VUs to finish...");
                    let _ = responses_tx.send(TextGenerationAggregatedResponse::new_as_ended());
                }=>{}
            }
        });
        while end_rx.recv().await.is_some() {
            active_vus.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
            // wait for all VUs to finish
            if start.elapsed() > self.config.duration
                && active_vus.load(std::sync::atomic::Ordering::SeqCst) == 0
            {
                break;
            }
        }
        // wait for the VU thread to finish
        let _ = vu_thread.await;
    }
}

/// Sends each chain of a fixed list exactly once, then stops. A chain is one
/// request, or the turns of one conversation sent in order.
///
/// With a `speed`, request `i` is sent at `offset_i / speed` after the start:
/// a replay of recorded traffic (2.0 = twice as fast). Without one, requests
/// go out as fast as `max_vus` concurrent users allow: accuracy and
/// conversation suites. A request whose slot is busy waits for a free user, so
/// none are dropped; the trace records when it really started.
pub struct ScheduledExecutor {
    config: ExecutorConfig,
    backend: Box<dyn TextGenerationBackend + Send + Sync>,
    chains: Vec<Vec<Arc<TextGenerationRequest>>>,
    speed: Option<f64>,
}

impl ScheduledExecutor {
    pub fn new(
        backend: Box<dyn TextGenerationBackend + Send + Sync>,
        max_vus: u64,
        mut chains: Vec<Vec<Arc<TextGenerationRequest>>>,
        speed: Option<f64>,
    ) -> ScheduledExecutor {
        chains.retain(|c| !c.is_empty());
        let speed = speed.filter(|s| *s > 0.0);
        let offset = |c: &Vec<Arc<TextGenerationRequest>>| c[0].offset;
        if speed.is_some() {
            // stable: requests recorded at the same moment keep their order
            chains.sort_by_key(offset);
        }
        let duration = speed
            .and_then(|s| Some(chains.iter().filter_map(offset).max()?.div_f64(s)))
            .unwrap_or_default();
        Self {
            backend,
            config: ExecutorConfig {
                max_vus,
                duration,
                rate: None,
            },
            chains,
            speed,
        }
    }

    pub fn config(&self) -> ExecutorConfig {
        self.config.clone()
    }
}

#[async_trait]
impl Executor for ScheduledExecutor {
    async fn run(
        &self,
        _requests: Arc<Mutex<dyn TextRequestGenerator + Send>>,
        responses_tx: UnboundedSender<TextGenerationAggregatedResponse>,
        stop_sender: broadcast::Sender<()>,
    ) {
        let start = tokio::time::Instant::now();
        let max_vus = self.config.max_vus.max(1);
        let (end_tx, mut end_rx): (Sender<bool>, Receiver<bool>) =
            tokio::sync::mpsc::channel(max_vus as usize);
        let mut stop_receiver = stop_sender.subscribe();
        let mut in_flight = 0u64;
        for chain in &self.chains {
            if let (Some(speed), Some(offset)) = (self.speed, chain[0].offset) {
                tokio::select! {
                    _ = tokio::time::sleep_until(start + offset.div_f64(speed)) => {},
                    _ = stop_receiver.recv() => return,
                }
            }
            while in_flight >= max_vus {
                tokio::select! {
                    _ = end_rx.recv() => in_flight -= 1,
                    _ = stop_receiver.recv() => return,
                }
            }
            start_vu(
                self.backend.clone(),
                chain.clone(),
                responses_tx.clone(),
                end_tx.clone(),
                stop_sender.clone(),
            )
            .await;
            in_flight += 1;
        }
        while in_flight > 0 {
            tokio::select! {
                _ = end_rx.recv() => in_flight -= 1,
                _ = stop_receiver.recv() => return,
            }
        }
        // every response is already queued: a VU forwards its response before it ends
        let _ = responses_tx.send(TextGenerationAggregatedResponse::new_as_ended());
    }

    fn expected_requests(&self) -> Option<u64> {
        Some(self.chains.iter().map(Vec::len).sum::<usize>() as u64)
    }
}
