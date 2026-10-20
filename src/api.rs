//! Library entry point: run a benchmark and get the report back.
//!
//! ```no_run
//! # async fn example() -> anyhow::Result<()> {
//! use inference_benchmarker::*;
//! use std::sync::Arc;
//!
//! let tokenizer = Arc::new(hub::load_tokenizer("gpt2", None, None).await?);
//! let config = BenchmarkConfig {
//!     benchmark_kind: BenchmarkKind::Throughput,
//!     max_vus: 8,
//!     duration: std::time::Duration::from_secs(30),
//!     prompt_options: Some(TokenizeOptions { num_tokens: Some(200), ..TokenizeOptions::new() }),
//!     ..Default::default()
//! };
//! let target = Target {
//!     url: "http://localhost:8000".parse()?,
//!     api_key: String::new(),
//!     model: "gpt2".into(),
//! };
//! let (stop, _) = tokio::sync::broadcast::channel(1);
//! let report = run_benchmark(config, target, Workload::Synthetic, tokenizer, RunOptions::default(), stop).await?;
//! println!("{}", serde_json::to_string_pretty(&report)?);
//! # Ok(()) }
//! ```
use crate::benchmark::{Benchmark, BenchmarkConfig, BenchmarkKind, Event};
use crate::record::RecordSink;
use crate::requests::{
    count_request_tokens, ConversationEntry, ConversationTextRequestGenerator,
    OpenAITextGenerationBackend, RequestListGenerator, SyntheticRequestGenerator,
    TextGenerationBackend, TextGenerationRequest, TextRequestGenerator, TokenizeOptions,
};
use crate::writers::BenchmarkReportWriter;
use reqwest::Url;
use std::fmt::{Display, Formatter};
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::broadcast::error::TryRecvError;
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::{broadcast, Mutex};

/// An OpenAI-compatible server. Requests go to `url` + `v1/chat/completions`.
#[derive(Clone, Debug)]
pub struct Target {
    pub url: Url,
    /// Sent as `Authorization: Bearer …`; empty for none.
    pub api_key: String,
    pub model: String,
}

/// What to send.
pub enum Workload {
    /// Random prompts sized by `prompt_options` (which needs `num_tokens`).
    Synthetic,
    /// Multi-turn conversations: each turn is sent with the model's own earlier
    /// answers. `Once` sends each conversation once; load kinds cycle through them.
    Conversations(Vec<ConversationEntry>),
    /// A fixed list of requests. `Once` and `Replay` send each exactly once
    /// (`Replay` at its `offset`); load kinds cycle through the list. A request
    /// with `num_prompt_tokens == 0` is counted with the tokenizer, and one
    /// without `num_decode_tokens` gets a length from `decode_options`.
    Requests(Vec<TextGenerationRequest>),
}

#[derive(Default)]
pub struct RunOptions {
    /// Progress and step events, for a live view.
    pub events: Option<UnboundedSender<Event>>,
    /// One trace record per request.
    pub records: Option<RecordSink>,
}

/// The run was stopped through the stop channel.
#[derive(Debug)]
pub struct Cancelled {
    /// The steps that finished before the stop, if any.
    pub partial: Option<Box<BenchmarkReportWriter>>,
}

impl Display for Cancelled {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "benchmark stopped")
    }
}

impl std::error::Error for Cancelled {}

/// Run a benchmark against `target` until it ends or `stop` fires.
///
/// Returns [`Cancelled`] (via `anyhow`) when stopped.
pub async fn run_benchmark(
    config: BenchmarkConfig,
    target: Target,
    workload: Workload,
    tokenizer: Arc<Tokenizer>,
    options: RunOptions,
    stop: broadcast::Sender<()>,
) -> anyhow::Result<BenchmarkReportWriter> {
    config.validate()?;
    let backend = OpenAITextGenerationBackend::try_new(
        target.api_key,
        target.url,
        target.model,
        tokenizer.clone(),
        config.request_timeout(),
    )?;
    let prompt_options = config.prompt_options.clone();
    let decode_options = config.decode_options.clone();
    let once = config.benchmark_kind == BenchmarkKind::Once;
    // tokenizing a dataset is CPU work: keep it off the async workers
    let (generator, schedule) = tokio::task::spawn_blocking(move || {
        build_workload(workload, tokenizer, prompt_options, decode_options, once)
    })
    .await??;
    execute(
        config,
        Box::new(backend),
        generator,
        schedule,
        options,
        stop,
    )
    .await
}

type Generator = Arc<Mutex<dyn TextRequestGenerator + Send>>;
/// What scheduled runs send: chains of one request, or a conversation's turns.
type Schedule = Vec<Vec<Arc<TextGenerationRequest>>>;

fn build_workload(
    workload: Workload,
    tokenizer: Arc<Tokenizer>,
    prompt_options: Option<TokenizeOptions>,
    decode_options: Option<TokenizeOptions>,
    once: bool,
) -> anyhow::Result<(Generator, Schedule)> {
    Ok(match workload {
        Workload::Synthetic => (
            Arc::new(Mutex::new(SyntheticRequestGenerator::new(
                tokenizer,
                prompt_options.unwrap_or_default(),
                decode_options,
            )?)),
            Vec::new(),
        ),
        Workload::Conversations(entries) => {
            let generator = ConversationTextRequestGenerator::from_entries(
                entries,
                tokenizer,
                prompt_options,
                decode_options,
            )?;
            let schedule = if once {
                let chains = generator.chains();
                chains
                    .into_iter()
                    .map(|chain| chain.into_iter().map(Arc::new).collect())
                    .collect()
            } else {
                Vec::new()
            };
            (Arc::new(Mutex::new(generator)), schedule)
        }
        Workload::Requests(mut requests) => {
            for request in &mut requests {
                if let Some(check) = &request.check {
                    check.validate()?;
                }
                if request.num_prompt_tokens == 0 {
                    request.num_prompt_tokens = count_request_tokens(&tokenizer, request);
                }
                if request.num_decode_tokens.is_none() {
                    request.num_decode_tokens =
                        decode_options.as_ref().and_then(TokenizeOptions::sample);
                }
            }
            let schedule = requests
                .iter()
                .cloned()
                .map(|r| vec![Arc::new(r)])
                .collect();
            (
                Arc::new(Mutex::new(RequestListGenerator::new(requests)?)),
                schedule,
            )
        }
    })
}

/// Run a prepared benchmark; shared by [`run_benchmark`] and the CLI.
pub(crate) async fn execute(
    config: BenchmarkConfig,
    backend: Box<dyn TextGenerationBackend + Send + Sync>,
    generator: Generator,
    schedule: Schedule,
    options: RunOptions,
    stop: broadcast::Sender<()>,
) -> anyhow::Result<BenchmarkReportWriter> {
    // events nobody listens to are dropped
    let events = options
        .events
        .unwrap_or_else(|| tokio::sync::mpsc::unbounded_channel().0);
    let mut benchmark = Benchmark::new(config.clone(), backend, generator, events, stop.clone())
        .with_schedule(schedule)
        .with_records(options.records);
    let mut stop_receiver = stop.subscribe();
    let stopped = tokio::select! {
        result = benchmark.run() => {
            // a stop also ends the running step early, so `run` may return first
            let stopped = !matches!(stop_receiver.try_recv(), Err(TryRecvError::Empty));
            if !stopped {
                result?;
            }
            stopped
        }
        _ = stop_receiver.recv() => true,
    };
    let mut report = benchmark.get_report();
    if stopped {
        report.end();
        let partial = BenchmarkReportWriter::try_new(config, report)
            .ok()
            .map(Box::new);
        return Err(Cancelled { partial }.into());
    }
    BenchmarkReportWriter::try_new(config, report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checks::Check;
    use crate::record::RequestRecord;
    use crate::requests::{
        chat_completions_url, Conversation, DummyTextGenerationBackend, ErrorKind,
        OpenAITextGenerationMessage,
    };
    use crate::test_util;
    use std::time::Duration;
    use tokio::sync::mpsc;

    fn request(prompt: &str, check: Option<Check>) -> TextGenerationRequest {
        TextGenerationRequest {
            prompt: prompt.to_string(),
            num_decode_tokens: Some(8),
            check,
            ..Default::default()
        }
    }

    fn once_config() -> BenchmarkConfig {
        BenchmarkConfig {
            benchmark_kind: BenchmarkKind::Once,
            max_vus: 2,
            ..Default::default()
        }
    }

    async fn collect(mut rx: mpsc::Receiver<RequestRecord>) -> Vec<RequestRecord> {
        let mut records = Vec::new();
        while let Some(r) = rx.recv().await {
            records.push(r);
        }
        records
    }

    fn sink(include_text: bool) -> (RecordSink, mpsc::Receiver<RequestRecord>) {
        let (tx, rx) = mpsc::channel(1024);
        (RecordSink { tx, include_text }, rx)
    }

    #[tokio::test]
    async fn once_scores_checks_and_uses_server_token_counts() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", "/proxy/v1/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_body(concat!(
                "data: {\"choices\":[{\"delta\":{\"content\":\"The answer is \"},\"finish_reason\":null}]}\n\n",
                "data: {\"choices\":[{\"delta\":{\"content\":\"(B).\"},\"finish_reason\":\"stop\"}]}\n\n",
                "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":5}}\n\n",
                "data: [DONE]\n\n",
            ))
            .expect(3)
            .create_async()
            .await;
        let target = Target {
            url: format!("{}/proxy", server.url()).parse().unwrap(),
            api_key: "key".into(),
            model: "m".into(),
        };
        let (records, rx) = sink(true);
        let workload = Workload::Requests(vec![
            request("q1", Some(Check::Choice { answer: "B".into() })),
            request("q2", Some(Check::Choice { answer: "C".into() })),
            request("q3", None),
        ]);
        let (stop, _) = broadcast::channel(1);
        let options = RunOptions {
            events: None,
            records: Some(records),
        };
        let report = run_benchmark(
            once_config(),
            target,
            workload,
            test_util::tokenizer(),
            options,
            stop,
        )
        .await
        .unwrap();
        let records = collect(rx).await;

        assert_eq!(report.results.len(), 1, "no warmup, one step");
        let step = &report.results[0];
        assert_eq!(step.id, "once");
        assert_eq!(step.total_requests, 3);
        assert_eq!(step.total_tokens, 15, "usage wins over re-tokenizing");
        assert_eq!(step.prompt_tokens_avg, 11.0);
        let accuracy = step.accuracy.as_ref().unwrap();
        assert_eq!((accuracy.checked, accuracy.passed), (2, 1));
        assert_eq!(records.len(), 3);
        assert!(records.iter().all(|r| r.ok && r.completion_tokens == 5));
        let q1 = records
            .iter()
            .find(|r| r.messages.as_ref().unwrap()[0].content == "q1")
            .unwrap();
        assert_eq!(q1.response.as_deref(), Some("The answer is (B)."));
        assert_eq!(q1.check.as_ref().unwrap().extracted.as_deref(), Some("B"));
        // the report round-trips through JSON
        let json = serde_json::to_string(&report).unwrap();
        let back: BenchmarkReportWriter = serde_json::from_str(&json).unwrap();
        assert_eq!(back.results, report.results);
    }

    #[tokio::test]
    async fn replay_sends_at_recorded_offsets_scaled_by_speed() {
        let offsets = [0u64, 400, 800];
        let requests: Vec<_> = offsets
            .iter()
            .rev() // the executor orders by offset
            .map(|ms| TextGenerationRequest {
                offset: Some(Duration::from_millis(*ms)),
                ..request(&format!("at {ms}"), None)
            })
            .collect();
        let schedule = requests
            .iter()
            .cloned()
            .map(|r| vec![Arc::new(r)])
            .collect();
        let config = BenchmarkConfig {
            benchmark_kind: BenchmarkKind::Replay,
            max_vus: 8,
            replay_speed: 2.0,
            ..Default::default()
        };
        let (records, rx) = sink(false);
        let (stop, _) = broadcast::channel(1);
        let report = execute(
            config,
            Box::new(DummyTextGenerationBackend::new(Duration::from_millis(10))),
            Arc::new(Mutex::new(RequestListGenerator::new(requests).unwrap())),
            schedule,
            RunOptions {
                events: None,
                records: Some(records),
            },
            stop,
        )
        .await
        .unwrap();
        assert_eq!(report.results[0].total_requests, 3);
        let mut starts: Vec<_> = collect(rx)
            .await
            .iter()
            .map(|r| r.started_at.unwrap())
            .collect();
        starts.sort();
        for (i, expected_ms) in [200i64, 400].iter().enumerate() {
            let gap = (starts[i + 1] - starts[0]).num_milliseconds();
            assert!(
                (gap - expected_ms).abs() < 60,
                "request {} sent {gap} ms after the first, expected {expected_ms}",
                i + 1
            );
        }
    }

    #[tokio::test]
    async fn conversation_turns_carry_the_models_own_answers() {
        let entries = vec![ConversationEntry {
            id: "c1".into(),
            conversations: vec![
                Conversation {
                    role: "system".into(),
                    content: "Be brief.".into(),
                    check: None,
                },
                Conversation {
                    role: "user".into(),
                    content: "My name is Asha.".into(),
                    check: None,
                },
                Conversation {
                    role: "user".into(),
                    content: "What is my name?".into(),
                    check: Some(Check::Contains {
                        value: "asha".into(),
                    }),
                },
            ],
        }];
        let generator = ConversationTextRequestGenerator::from_entries(
            entries,
            test_util::tokenizer(),
            None,
            None,
        )
        .unwrap();
        let config = BenchmarkConfig {
            benchmark_kind: BenchmarkKind::Throughput,
            max_vus: 1,
            duration: Duration::from_secs(1),
            ..Default::default()
        };
        let (records, rx) = sink(true);
        let (stop, _) = broadcast::channel(1);
        execute(
            config,
            Box::new(DummyTextGenerationBackend::new(Duration::from_millis(20))),
            Arc::new(Mutex::new(generator)),
            Vec::new(),
            RunOptions {
                events: None,
                records: Some(records),
            },
            stop,
        )
        .await
        .unwrap();
        let records = collect(rx).await;
        let second = records.iter().find(|r| r.turn == 1).expect("a second turn");
        assert_eq!(second.conversation_id.as_deref(), Some("c1"));
        assert_eq!(
            second.messages.as_ref().unwrap(),
            &vec![
                OpenAITextGenerationMessage::new("system", "Be brief."),
                OpenAITextGenerationMessage::new("user", "My name is Asha."),
                OpenAITextGenerationMessage::new("assistant", "echo: My name is Asha."),
                OpenAITextGenerationMessage::new("user", "What is my name?"),
            ]
        );
        // the dummy echoes the question, which does not contain the name
        assert!(!second.check.as_ref().unwrap().passed);
    }

    #[tokio::test]
    async fn once_sends_each_conversation_turn_by_turn() {
        let user = |content: &str, check: Option<Check>| Conversation {
            role: "user".into(),
            content: content.into(),
            check,
        };
        let entries = vec![
            ConversationEntry {
                id: "a".into(),
                conversations: vec![
                    user("I live in Pokhara.", None),
                    user(
                        "Where do I live?",
                        Some(Check::Contains {
                            value: "Pokhara".into(),
                        }),
                    ),
                ],
            },
            ConversationEntry {
                id: "b".into(),
                conversations: vec![user("Hello", None)],
            },
        ];
        let (generator, schedule) = build_workload(
            Workload::Conversations(entries),
            test_util::tokenizer(),
            None,
            None,
            true,
        )
        .unwrap();
        assert_eq!(
            schedule.iter().map(Vec::len).collect::<Vec<_>>(),
            vec![2, 1]
        );
        let (records, rx) = sink(true);
        let (stop, _) = broadcast::channel(1);
        let report = execute(
            once_config(),
            Box::new(DummyTextGenerationBackend::new(Duration::from_millis(10))),
            generator,
            schedule,
            RunOptions {
                events: None,
                records: Some(records),
            },
            stop,
        )
        .await
        .unwrap();
        assert_eq!(report.results[0].total_requests, 3, "each turn once");
        let records = collect(rx).await;
        let second = records
            .iter()
            .find(|r| r.conversation_id.as_deref() == Some("a") && r.turn == 1)
            .unwrap();
        let messages = second.messages.as_ref().unwrap();
        assert_eq!(messages[1].content, "echo: I live in Pokhara.");
        assert_eq!(messages[2].content, "Where do I live?");
        // the dummy echoes "Where do I live?": the fact is not in the answer
        let accuracy = report.results[0].accuracy.as_ref().unwrap();
        assert_eq!((accuracy.checked, accuracy.passed), (1, 0));
    }

    #[tokio::test]
    async fn http_errors_are_classified() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", "/v1/chat/completions")
            .with_status(503)
            .create_async()
            .await;
        let backend = OpenAITextGenerationBackend::try_new(
            String::new(),
            server.url().parse().unwrap(),
            "m".into(),
            test_util::tokenizer(),
            Duration::from_secs(5),
        )
        .unwrap();
        let (tx, mut rx) = mpsc::channel(1);
        backend.generate(Arc::new(request("hi", None)), tx).await;
        let response = rx.recv().await.unwrap();
        assert!(response.failed);
        let error = response.error.unwrap();
        assert_eq!(error.kind, ErrorKind::Http);
        assert_eq!(error.http_status, Some(503));
    }

    #[tokio::test]
    async fn stop_returns_cancelled_with_finished_steps() {
        let config = BenchmarkConfig {
            benchmark_kind: BenchmarkKind::Throughput,
            max_vus: 1,
            duration: Duration::from_secs(30),
            warmup_duration: Duration::from_secs(1),
            ..Default::default()
        };
        let (stop, _) = broadcast::channel(1);
        let stopper = stop.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(1500)).await;
            let _ = stopper.send(());
        });
        let err = execute(
            config,
            Box::new(DummyTextGenerationBackend::new(Duration::from_millis(50))),
            Arc::new(Mutex::new(crate::requests::DummyTextRequestGenerator::new())),
            Vec::new(),
            RunOptions::default(),
            stop,
        )
        .await
        .unwrap_err();
        let cancelled = err.downcast::<Cancelled>().unwrap();
        let partial = cancelled.partial.expect("the warmup finished");
        assert_eq!(partial.results[0].id, "warmup");
    }

    #[test]
    fn endpoint_keeps_the_path_prefix() {
        let url = |s: &str| {
            chat_completions_url(&s.parse().unwrap())
                .unwrap()
                .to_string()
        };
        assert_eq!(url("http://h:8000"), "http://h:8000/v1/chat/completions");
        assert_eq!(url("http://h/proxy"), "http://h/proxy/v1/chat/completions");
        assert_eq!(url("http://h/proxy/"), "http://h/proxy/v1/chat/completions");
    }

    #[test]
    fn config_rules() {
        assert!(once_config().validate().is_ok(), "once needs no duration");
        assert_eq!(once_config().request_timeout(), Duration::from_secs(600));
        let rate_without_rates = BenchmarkConfig {
            benchmark_kind: BenchmarkKind::Rate,
            max_vus: 1,
            duration: Duration::from_secs(1),
            ..Default::default()
        };
        assert!(rate_without_rates.validate().is_err());
        let negative_speed = BenchmarkConfig {
            replay_speed: -1.0,
            ..once_config()
        };
        assert!(negative_speed.validate().is_err());
    }
}
