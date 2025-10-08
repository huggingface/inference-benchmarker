use std::sync::atomic::AtomicI64;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use log::{info, trace, warn};
use serde::Serialize;
use tokio::sync::mpsc::{Receiver, Sender, UnboundedSender};
use tokio::sync::{broadcast, Mutex};
use tokio::task::JoinHandle;

use crate::requests::{
    TextGenerationAggregatedResponse, TextGenerationBackend, TextGenerationRequest,
    TextRequestGenerator,
};

#[serde_with::serde_as]
#[derive(Clone, Serialize)]
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

        let (timeout_tx, _) = tokio::sync::broadcast::channel::<()>(self.config.max_vus as usize);
        // sleep for duration and send timeout signal
        {
            let timeout_duration = self.config.duration;
            let timeout_tx = timeout_tx.clone();
            tokio::spawn(async move {
                tokio::time::sleep(timeout_duration).await;
                // send timeout signal to all VUs
                let _ = timeout_tx.send(());
            });
        }

        for _ in 0..self.config.max_vus {
            let mut requests_guard = requests.lock().await;
            let request = Arc::from(requests_guard.generate_request());
            drop(requests_guard);
            start_vu(
                self.backend.clone(),
                request,
                responses_tx.clone(),
                end_tx.clone(),
                stop_sender.clone(),
                Some(timeout_tx.subscribe()),
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
                        start_vu(self.backend.clone(), request, responses_tx.clone(), end_tx.clone(), stop_sender.clone(), Some(timeout_tx.subscribe())).await;
                    }
                }
            }=>{}
        }
    }
}

async fn start_vu(
    backend: Box<dyn TextGenerationBackend + Send + Sync>,
    request: Arc<TextGenerationRequest>,
    responses_tx: UnboundedSender<TextGenerationAggregatedResponse>,
    end_tx: Sender<bool>,
    stop_sender: broadcast::Sender<()>,
    mut timeout_stop_receiver: Option<broadcast::Receiver<()>>,
) -> JoinHandle<()> {
    let mut stop_receiver = stop_sender.subscribe();
    tokio::spawn(async move {
        tokio::select! {
            _ = stop_receiver.recv() => {
                let _ = end_tx.send(true).await;
            },
            _ = async {
                if let Some(ref mut receiver) = timeout_stop_receiver {
                    let _ = receiver.recv().await;
                } else {
                    std::future::pending::<()>().await; // wait forever if no timeout receiver is provided
                }
            } => {
                let _ = end_tx.send(true).await;
            },
            _ = async{
                let (tx, mut rx): (Sender<TextGenerationAggregatedResponse>, Receiver<TextGenerationAggregatedResponse>) = tokio::sync::mpsc::channel(1);
                trace!("VU started with request: {:?}", request);
                let req_thread = tokio::spawn(async move {
                    backend.generate(request.clone(), tx).await;
                });
                let send_thread = tokio::spawn(async move {
                    while let Some(response) = rx.recv().await {
                        // ignore errors, if the receiver is gone we want to finish the request
                        // to leave remote server in clean state
                        let _ = responses_tx.send(response);
                    }
                });
                req_thread.await.unwrap();
                send_thread.await.unwrap();
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
                                start_vu(backend.clone(), request.clone(), responses_tx.clone(), end_tx.clone(), stop_sender.clone(), None).await;
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
        vu_thread.await.unwrap();
    }
}
