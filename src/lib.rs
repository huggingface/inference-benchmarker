//! Benchmark OpenAI-compatible inference servers: load, latency, replayed
//! traffic, accuracy and multi-turn conversations.
//!
//! Use [`run_benchmark`] from Rust; the `cli` feature (on by default) adds the
//! `inference-benchmarker` command and its terminal UI.

pub mod checks;
pub mod cost;
pub mod hub;
pub mod record;
pub mod slo;

mod api;
mod benchmark;
mod executors;
mod requests;
mod results;
mod scheduler;
mod writers;

#[cfg(feature = "cli")]
mod app;
#[cfg(feature = "cli")]
mod cli;
#[cfg(feature = "cli")]
mod event;
#[cfg(feature = "cli")]
mod flux;
#[cfg(feature = "cli")]
mod profiles;
#[cfg(feature = "cli")]
mod table;

pub use api::{run_benchmark, Cancelled, RunOptions, Target, Workload};
pub use benchmark::{BenchmarkConfig, BenchmarkEvent, BenchmarkKind, Event, MessageEvent};
pub use executors::ExecutorConfig;
pub use requests::{
    count_request_tokens, Conversation, ConversationEntry, ErrorKind, OpenAITextGenerationMessage,
    RequestError, TextGenerationRequest, TokenizeOptions,
};
pub use results::{AccuracyResult, BenchmarkReport, BenchmarkResults};
pub use scheduler::ExecutorType;
/// The tokenizer type the benchmark takes, so callers build it with the same version.
pub use tokenizers::Tokenizer;
pub use writers::{BenchmarkReportWriter, BenchmarkResultsWriter, PercentilesWriter, SystemInfo};

#[cfg(feature = "cli")]
pub use app::run_console;
#[cfg(feature = "cli")]
pub use cli::{run, RunConfiguration};
#[cfg(feature = "cli")]
pub use profiles::apply_profile;

#[cfg(test)]
pub(crate) mod test_util {
    use std::sync::{Arc, OnceLock};
    use tokenizers::Tokenizer;

    /// The gpt2 tokenizer, downloaded once per test run (tests share it, so
    /// they never race on the same download).
    pub(crate) fn tokenizer() -> Arc<Tokenizer> {
        static TOKENIZER: OnceLock<Arc<Tokenizer>> = OnceLock::new();
        TOKENIZER
            .get_or_init(|| {
                Arc::new(Tokenizer::from_pretrained("gpt2", None).expect("gpt2 tokenizer"))
            })
            .clone()
    }
}
