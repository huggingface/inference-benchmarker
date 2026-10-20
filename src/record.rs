//! Per-request trace: one [`RequestRecord`] per request sent.
use crate::checks::CheckOutcome;
use crate::requests::{
    OpenAITextGenerationMessage, RequestError, TextGenerationAggregatedResponse,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::sync::mpsc;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RequestRecord {
    /// Benchmark step the request belongs to (`warmup`, `throughput`, …).
    pub step_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conversation_id: Option<String>,
    #[serde(default)]
    pub turn: u32,
    /// Wall-clock time the request was sent.
    #[serde(default)]
    pub started_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub ttft_ms: Option<f64>,
    /// Mean time between tokens.
    #[serde(default)]
    pub itl_ms: Option<f64>,
    #[serde(default)]
    pub e2e_ms: Option<f64>,
    #[serde(default)]
    pub prompt_tokens: u64,
    #[serde(default)]
    pub completion_tokens: u64,
    pub ok: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<RequestError>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub check: Option<CheckOutcome>,
    /// The messages sent, only when the sink asks for text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub messages: Option<Vec<OpenAITextGenerationMessage>>,
    /// The generated text, only when the sink asks for text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response: Option<String>,
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

impl RequestRecord {
    pub fn from_response(
        step_id: &str,
        response: &TextGenerationAggregatedResponse,
        include_text: bool,
    ) -> Self {
        let request = response.request.as_deref();
        Self {
            step_id: step_id.to_string(),
            conversation_id: request.and_then(|r| r.conversation_id.clone()),
            turn: request.map_or(0, |r| r.turn),
            started_at: response.started_at,
            ttft_ms: response.time_to_first_token().map(ms),
            itl_ms: response.inter_token_latency().map(ms),
            e2e_ms: response.e2e_latency().map(ms),
            prompt_tokens: response.prompt_tokens(),
            completion_tokens: response.num_generated_tokens,
            ok: !response.failed,
            error: response.error.clone(),
            check: response.check.clone(),
            messages: request.filter(|_| include_text).map(|r| r.messages()),
            response: response.response.clone().filter(|_| include_text),
        }
    }
}

/// Where the scheduler sends trace records. The channel is bounded: a slow
/// reader slows the bookkeeping, never the requests themselves.
#[derive(Clone)]
pub struct RecordSink {
    pub tx: mpsc::Sender<RequestRecord>,
    /// Also record the messages sent and the generated text.
    pub include_text: bool,
}
