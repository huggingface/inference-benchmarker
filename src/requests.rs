use crate::checks::{Check, CheckOutcome};
use async_trait::async_trait;
use futures_util::StreamExt;
use log::{debug, error, info, trace, warn};
use rand::Rng;
use rand_distr::Distribution;
use rayon::iter::split;
use rayon::prelude::*;
use reqwest::Url;
use reqwest_eventsource::{Error, Event, EventSource};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::fmt::Display;
use std::sync::{Arc, Mutex};
use std::time;
use tokenizers::Tokenizer;
use tokio::sync::mpsc::Sender;
use tokio::time::Instant;
use uuid::Uuid;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct TextGenerationRequest {
    /// Id of the next turn of the same conversation, if any.
    pub id: Option<Uuid>,
    /// The new user message.
    pub prompt: String,
    /// Prompt tokens, including `history`.
    pub num_prompt_tokens: u64,
    pub num_decode_tokens: Option<u64>,
    /// Messages sent before `prompt`: the system prompt and earlier turns.
    pub history: Vec<OpenAITextGenerationMessage>,
    pub conversation_id: Option<String>,
    /// 0-based user turn within the conversation.
    pub turn: u32,
    /// Scores the answer (accuracy and conversation benchmarks).
    pub check: Option<Check>,
    /// When to send it, measured from the start of a replay.
    pub offset: Option<time::Duration>,
}

impl TextGenerationRequest {
    /// The chat messages sent to the server.
    pub fn messages(&self) -> Vec<OpenAITextGenerationMessage> {
        let mut messages = self.history.clone();
        if !self.prompt.is_empty() {
            messages.push(OpenAITextGenerationMessage::new("user", &self.prompt));
        }
        messages
    }
}

#[async_trait]
pub trait TextGenerationBackend: TextGenerationBackendClone {
    async fn generate(
        &self,
        request: Arc<TextGenerationRequest>,
        sender: Sender<TextGenerationAggregatedResponse>,
    );
}

pub trait TextGenerationBackendClone {
    fn clone_box(&self) -> Box<dyn TextGenerationBackend + Send + Sync>;
}

impl<T> TextGenerationBackendClone for T
where
    T: 'static + TextGenerationBackend + Clone + Send + Sync,
{
    fn clone_box(&self) -> Box<dyn TextGenerationBackend + Send + Sync> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn TextGenerationBackend + Send + Sync> {
    fn clone(&self) -> Box<dyn TextGenerationBackend + Send + Sync> {
        self.clone_box()
    }
}

#[derive(Debug, Clone)]
pub struct OpenAITextGenerationBackend {
    pub api_key: String,
    /// The server URL + `v1/chat/completions`, keeping any path prefix.
    pub endpoint: Url,
    pub model_name: String,
    pub client: reqwest::Client,
    pub tokenizer: Arc<Tokenizer>,
    pub timeout: time::Duration,
}

#[derive(Deserialize, Serialize, Clone, Debug, Default, PartialEq)]
pub struct OpenAITextGenerationMessage {
    pub content: String,
    pub role: String,
}

impl OpenAITextGenerationMessage {
    pub fn new(role: &str, content: &str) -> Self {
        Self {
            role: role.to_string(),
            content: content.to_string(),
        }
    }
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct OpenAITextGenerationDelta {
    pub content: Option<String>,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct OpenAITextGenerationChoice {
    pub message: Option<OpenAITextGenerationMessage>,
    pub finish_reason: Option<String>,
    pub delta: Option<OpenAITextGenerationDelta>,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct OpenAIUsage {
    #[serde(default)]
    pub prompt_tokens: u64,
    #[serde(default)]
    pub completion_tokens: u64,
}

#[derive(Deserialize, Serialize, Clone)]
pub struct OpenAITextGenerationResponse {
    #[serde(default)]
    pub choices: Vec<OpenAITextGenerationChoice>,
    #[serde(default)]
    pub usage: Option<OpenAIUsage>,
    #[serde(default)]
    pub error: Option<serde_json::Value>,
}

#[derive(Deserialize, Serialize, Clone)]
pub struct OpenAIStreamOptions {
    pub include_usage: bool,
}

#[derive(Deserialize, Serialize, Clone)]
pub struct OpenAITextGenerationRequest {
    pub model: String,
    pub messages: Vec<OpenAITextGenerationMessage>,
    pub max_tokens: Option<u64>,
    pub stream: bool,
    pub stop: Option<String>,
    pub temperature: f64,
    /// Asks the server for exact token counts in the last chunk.
    pub stream_options: OpenAIStreamOptions,
}

/// `base` + `v1/chat/completions`, keeping any path prefix of `base`.
pub fn chat_completions_url(base: &Url) -> anyhow::Result<Url> {
    let mut base = base.clone();
    if !base.path().ends_with('/') {
        let path = format!("{}/", base.path());
        base.set_path(&path);
    }
    Ok(base.join("v1/chat/completions")?)
}

impl OpenAITextGenerationBackend {
    pub fn try_new(
        api_key: String,
        base_url: Url,
        model_name: String,
        tokenizer: Arc<Tokenizer>,
        timeout: time::Duration,
    ) -> anyhow::Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| anyhow::anyhow!("Error creating HTTP client: {e}"))?;
        Ok(Self {
            client,
            api_key,
            endpoint: chat_completions_url(&base_url)?,
            model_name,
            tokenizer,
            timeout,
        })
    }

    fn count_tokens(&self, text: &str) -> u64 {
        // a delta may hold several tokens (vLLM chunked prefill, speculative decoding)
        self.tokenizer
            .encode(text, false)
            .map(|e| e.len() as u64)
            .unwrap_or(1)
            .max(1)
    }
}

#[async_trait]
impl TextGenerationBackend for OpenAITextGenerationBackend {
    async fn generate(
        &self,
        request: Arc<TextGenerationRequest>,
        sender: Sender<TextGenerationAggregatedResponse>,
    ) {
        let mut aggregated_response = TextGenerationAggregatedResponse::new(request.clone());
        let body = OpenAITextGenerationRequest {
            model: self.model_name.clone(),
            messages: request.messages(),
            max_tokens: request.num_decode_tokens,
            stream: true,
            stop: None,
            temperature: 0.0,
            stream_options: OpenAIStreamOptions {
                include_usage: true,
            },
        };
        let req = self
            .client
            .post(self.endpoint.clone())
            .header(
                "Authorization",
                format!("Bearer {token}", token = self.api_key),
            )
            .json(&body)
            .timeout(self.timeout);
        // start timer
        aggregated_response.start();
        let mut es = match EventSource::new(req) {
            Ok(es) => es,
            Err(e) => {
                aggregated_response.fail(RequestError::new(ErrorKind::Connection, e.to_string()));
                let _ = sender.send(aggregated_response).await;
                return;
            }
        };
        let mut final_response = String::new();
        let mut usage_tokens = None;
        while let Some(event) = es.next().await {
            match event {
                Ok(Event::Open) => trace!("SSE connection opened"),
                Ok(Event::Message(message)) => {
                    if message.data == "\n" || message.data == "[DONE]" {
                        aggregated_response.stop();
                        continue;
                    }
                    let oai_response: OpenAITextGenerationResponse =
                        match serde_json::from_str(&message.data) {
                            Ok(response) => response,
                            Err(e) => {
                                error!("Error deserializing OpenAI API response: {e}");
                                aggregated_response
                                    .fail(RequestError::new(ErrorKind::Decode, e.to_string()));
                                es.close();
                                break;
                            }
                        };
                    if let Some(err) = oai_response.error {
                        error!("Error from OpenAI API: {err}");
                        aggregated_response
                            .fail(RequestError::new(ErrorKind::Api, err.to_string()));
                        es.close();
                        break;
                    }
                    if let Some(usage) = oai_response.usage {
                        usage_tokens = Some(usage);
                    }
                    // the usage chunk has no choices
                    let Some(choice) = oai_response.choices.first() else {
                        continue;
                    };
                    let content = choice
                        .delta
                        .as_ref()
                        .and_then(|d| d.content.clone())
                        .unwrap_or_default();
                    if !content.is_empty() {
                        aggregated_response.add_tokens(self.count_tokens(&content));
                        final_response += content.as_str();
                    }
                    if choice.finish_reason.is_some() {
                        aggregated_response.stop();
                        trace!("Generated text using OpenAI API | prompt: {prompt}, max tokens: {max_tokens:?}, response: {final_response}", prompt = request.prompt, max_tokens = request.num_decode_tokens);
                    }
                }
                Err(e) => {
                    match e {
                        Error::StreamEnded => {
                            if aggregated_response.num_generated_tokens == 0 {
                                aggregated_response.fail(RequestError::new(
                                    ErrorKind::Stream,
                                    "server sent no tokens".to_string(),
                                ));
                            } else if aggregated_response.end_time.is_none() {
                                // server closed the connection before we received the final response
                                warn!("Connection closed before completion. Received :: {num_tokens}/{max_tokens} tokens.", num_tokens = aggregated_response.num_generated_tokens, max_tokens = request.num_decode_tokens.unwrap_or(0));
                                aggregated_response.fail(RequestError::new(
                                    ErrorKind::Stream,
                                    "connection closed before completion".to_string(),
                                ));
                            }
                        }
                        other => aggregated_response.fail(RequestError::from_eventsource(other)),
                    }
                    es.close();
                }
            };
        }
        if let Some(usage) = usage_tokens {
            if usage.completion_tokens > 0 {
                aggregated_response.num_generated_tokens = usage.completion_tokens;
            }
            if usage.prompt_tokens > 0 {
                aggregated_response.num_prompt_tokens = Some(usage.prompt_tokens);
            }
        }
        aggregated_response.response = Some(final_response);
        // the receiver is gone only when the run was stopped
        let _ = sender.send(aggregated_response).await;
    }
}

/// A backend that answers `echo: <prompt>` after a fixed time (tests).
#[cfg(test)]
#[derive(Debug, Clone)]
pub struct DummyTextGenerationBackend {
    time_to_generate: time::Duration,
}

#[cfg(test)]
impl DummyTextGenerationBackend {
    pub fn new(time_to_generate: time::Duration) -> Self {
        Self { time_to_generate }
    }
}

#[cfg(test)]
impl Default for DummyTextGenerationBackend {
    fn default() -> Self {
        Self::new(time::Duration::from_secs(1))
    }
}

#[cfg(test)]
#[async_trait]
impl TextGenerationBackend for DummyTextGenerationBackend {
    async fn generate(
        &self,
        request: Arc<TextGenerationRequest>,
        sender: Sender<crate::requests::TextGenerationAggregatedResponse>,
    ) {
        let mut response = TextGenerationAggregatedResponse::new(request.clone());
        response.start();
        let num_tokens = request.num_decode_tokens.unwrap_or(10);
        let time_per_token = self
            .time_to_generate
            .checked_div(num_tokens as u32)
            .unwrap_or_default();
        for _ in 0..num_tokens {
            tokio::time::sleep(time_per_token).await;
            response.add_tokens(1);
        }
        response.stop();
        response.response = Some(format!("echo: {}", request.prompt));
        let _ = sender.send(response).await;
    }
}

pub trait TextRequestGenerator: Sync {
    fn generate_request(&mut self) -> TextGenerationRequest;
    /// callback can be used by generators to add new requests to the queue based on the response (e.g. for multi-turn conversation generation)
    fn callback(&mut self, request: Arc<TextGenerationRequest>, response: &str);
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct Conversation {
    pub role: String,
    pub content: String,
    /// On a user turn: scores the model's answer to that turn.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub check: Option<Check>,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct ConversationEntry {
    pub id: String,
    pub conversations: Vec<Conversation>,
}

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub struct TokenizeOptions {
    pub num_tokens: Option<u64>,
    pub min_tokens: u64,
    pub max_tokens: u64,
    pub variance: u64,
}

impl TokenizeOptions {
    pub fn new() -> Self {
        Self {
            num_tokens: None,
            min_tokens: 0,
            max_tokens: u64::MAX,
            variance: 0,
        }
    }

    /// Draw a token count, or `None` when no target is set.
    pub fn sample(&self) -> Option<u64> {
        self.num_tokens.map(|num_tokens| {
            sample_num_tokens(num_tokens, self.min_tokens, self.max_tokens, self.variance)
        })
    }
}

impl Default for TokenizeOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl Display for TokenizeOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "num_tokens={num_tokens:?},min_tokens={min_tokens},max_tokens={max_tokens},variance={variance}",
            num_tokens = self.num_tokens,
            min_tokens = self.min_tokens,
            max_tokens = self.max_tokens,
            variance = self.variance
        )
    }
}

#[derive(Clone)]
pub struct ConversationTurnRequest {
    #[allow(dead_code)] // tests rebuild conversations by it
    id: Uuid,
    priority: u64,
    tie: Instant,
    request: TextGenerationRequest,
}

impl Ord for ConversationTurnRequest {
    // order by increasing priority and decreasing tie-breaking
    // this way, we can pop the item with the highest priority and oldest tie-breaking
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority
            .cmp(&other.priority)
            .then_with(|| self.tie.cmp(&other.tie).reverse())
    }
}

impl PartialOrd for ConversationTurnRequest {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for ConversationTurnRequest {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for ConversationTurnRequest {}

#[derive(Clone)]
pub struct ConversationTextRequestGenerator {
    pub requests: HashMap<Uuid, ConversationTurnRequest>,
    pub queue: BinaryHeap<ConversationTurnRequest>,
    pub tokenizer: Arc<Tokenizer>,
}

impl ConversationTextRequestGenerator {
    #[cfg(feature = "cli")]
    /// Load a conversation dataset from a JSON file: an array of
    /// [`ConversationEntry`] (`{id, conversations: [{role, content, check?}]}`).
    pub fn load(
        filepath: std::path::PathBuf,
        tokenizer: Arc<Tokenizer>,
        prompt_tokenize_opts: Option<TokenizeOptions>,
        decode_tokenize_opts: Option<TokenizeOptions>,
    ) -> anyhow::Result<Self> {
        let input = std::fs::read_to_string(&filepath)?;
        let data: Vec<ConversationEntry> = serde_json::from_str(&input).map_err(|e| {
            anyhow::anyhow!("{}: not a list of conversations: {e}", filepath.display())
        })?;
        info!(
            "Generating requests from {filepath}",
            filepath = filepath.display()
        );
        Self::from_entries(data, tokenizer, prompt_tokenize_opts, decode_tokenize_opts)
    }

    /// Build the generator from conversations already in memory.
    ///
    /// Leading `system` messages become the history of the first turn. Every
    /// `user` message becomes one request; the dataset's `assistant` messages
    /// are dropped, because the model under test writes its own. The `id`
    /// field links each turn to the next one in the conversation.
    pub fn from_entries(
        data: Vec<ConversationEntry>,
        tokenizer: Arc<Tokenizer>,
        prompt_tokenize_opts: Option<TokenizeOptions>,
        decode_tokenize_opts: Option<TokenizeOptions>,
    ) -> anyhow::Result<Self> {
        let prompt_opts = prompt_tokenize_opts.unwrap_or_default();
        let requests: Arc<Mutex<HashMap<Uuid, ConversationTurnRequest>>> =
            Arc::from(Mutex::from(HashMap::new()));
        split(data, entry_splitter).for_each(|subrange| {
            for entry in subrange {
                let system: Vec<OpenAITextGenerationMessage> = entry
                    .conversations
                    .iter()
                    .take_while(|c| c.role == "system")
                    .map(|c| OpenAITextGenerationMessage::new("system", &c.content))
                    .collect();
                let system_tokens: u64 = system
                    .iter()
                    .map(|m| count_tokens(&tokenizer, &m.content))
                    .sum();
                let turns: Vec<&Conversation> = entry
                    .conversations
                    .iter()
                    .filter(|c| c.role == "user")
                    .collect();
                let ids: Vec<Uuid> = turns.iter().map(|_| Uuid::new_v4()).collect();
                for (turn_idx, c) in turns.iter().enumerate() {
                    debug!("Prompt: {prompt}", prompt = c.content);
                    let (prompt, num_tokens) =
                        match tokenize_prompt(c.content.clone(), tokenizer.clone(), &prompt_opts) {
                            Ok(prompt) => prompt,
                            Err(e) => {
                                // later turns cannot be reached without this one
                                debug!("Skipping rest of conversation {}: {e}", entry.id);
                                break;
                            }
                        };
                    let first = turn_idx == 0;
                    requests.lock().expect("requests lock").insert(
                        ids[turn_idx],
                        ConversationTurnRequest {
                            id: ids[turn_idx],
                            priority: turn_idx as u64,
                            tie: Instant::now(),
                            request: TextGenerationRequest {
                                id: ids.get(turn_idx + 1).copied(),
                                prompt,
                                num_prompt_tokens: num_tokens
                                    + if first { system_tokens } else { 0 },
                                num_decode_tokens: decode_tokenize_opts
                                    .as_ref()
                                    .and_then(TokenizeOptions::sample),
                                history: if first { system.clone() } else { Vec::new() },
                                conversation_id: Some(entry.id.clone()),
                                turn: turn_idx as u32,
                                check: c.check.clone(),
                                offset: None,
                            },
                        },
                    );
                }
            }
        });
        let requests = std::mem::take(&mut *requests.lock().expect("requests lock"));
        info!(
            "Generated {num_requests} requests",
            num_requests = requests.len()
        );
        // create the queue from the hashmap. Only queue first turns in the conversation
        let queue: BinaryHeap<ConversationTurnRequest> = requests
            .values()
            .filter(|item| item.priority == 0)
            .cloned()
            .collect();
        if queue.is_empty() {
            return Err(anyhow::anyhow!(
                "No usable prompts: every conversation was filtered out by the prompt options"
            ));
        }
        Ok(Self {
            requests,
            tokenizer,
            queue,
        })
    }
}

impl ConversationTextRequestGenerator {
    /// Every conversation as its turns in order, for runs that send each
    /// conversation once (each turn then carries the model's earlier answers).
    pub fn chains(&self) -> Vec<Vec<TextGenerationRequest>> {
        let mut firsts: Vec<&ConversationTurnRequest> =
            self.requests.values().filter(|t| t.priority == 0).collect();
        // dataset order is lost in the map: order by conversation id for a stable run
        firsts.sort_by(|a, b| a.request.conversation_id.cmp(&b.request.conversation_id));
        firsts
            .into_iter()
            .map(|first| {
                let mut chain = vec![first.request.clone()];
                while let Some(next) = chain
                    .last()
                    .and_then(|t| t.id)
                    .and_then(|id| self.requests.get(&id))
                {
                    chain.push(next.request.clone());
                }
                chain
            })
            .collect()
    }
}

fn count_tokens(tokenizer: &Tokenizer, text: &str) -> u64 {
    tokenizer
        .encode(text, false)
        .map(|e| e.len() as u64)
        .unwrap_or_default()
}

/// Tokens of every message of a request: used when the caller did not count them.
pub fn count_request_tokens(tokenizer: &Tokenizer, request: &TextGenerationRequest) -> u64 {
    request
        .messages()
        .iter()
        .map(|m| count_tokens(tokenizer, &m.content))
        .sum()
}

fn sample_num_tokens(num_tokens: u64, min_tokens: u64, max_tokens: u64, variance: u64) -> u64 {
    let sampled = rand_distr::Normal::new(num_tokens as f64, variance as f64)
        .map(|normal| normal.sample(&mut rand::rng()))
        .unwrap_or(num_tokens as f64);
    (sampled.max(0.0) as u64).clamp(min_tokens, max_tokens.max(min_tokens))
}

fn entry_splitter(
    gen: Vec<ConversationEntry>,
) -> (Vec<ConversationEntry>, Option<Vec<ConversationEntry>>) {
    if gen.len() <= 2 {
        return (gen, None);
    }
    let middle = gen.len() / 2;
    let (left, right) = gen.split_at(middle);
    let left = left.to_vec();
    let right = right.to_vec();
    (left, Some(right))
}

impl TextRequestGenerator for ConversationTextRequestGenerator {
    fn generate_request(&mut self) -> TextGenerationRequest {
        // never empty: first turns are pushed back, and `from_entries` rejects an empty queue
        let item = self.queue.pop().expect("conversation queue is never empty");
        // add the item back to the end of the queue if it is a first turn in the conversation
        if item.priority == 0 {
            let mut cloned_item = item.clone();
            cloned_item.tie = Instant::now(); // update the tie-breaking for intra-priority sorting
            self.queue.push(cloned_item);
        }
        item.request
    }

    /// Use callback to add the next chat turn to the queue.
    /// The next turn is sent with the whole conversation so far as history:
    /// earlier user turns and the model's own answers.
    /// Those turns must be scheduled as soon as possible so that we may benefit from
    /// KV cache hits. The `priority` field is used to move the turn to the front of the queue.
    fn callback(&mut self, request: Arc<TextGenerationRequest>, response: &str) {
        let Some(id) = request.id else {
            return;
        };
        let Some(next) = self.requests.get(&id) else {
            return;
        };
        let turn = ConversationTurnRequest {
            id,
            priority: 100,       // move to the front of the queue
            tie: Instant::now(), // use the current time as tie-breaking (older turns have higher priority)
            request: continue_conversation(
                &request,
                request.num_prompt_tokens,
                response,
                count_tokens(&self.tokenizer, response),
                &next.request,
            ),
        };
        self.queue.push(turn);
    }
}

/// The next turn of a conversation: `next` sent with the conversation so far
/// (`previous`'s messages and the model's answer) as history.
pub fn continue_conversation(
    previous: &TextGenerationRequest,
    previous_prompt_tokens: u64,
    response: &str,
    response_tokens: u64,
    next: &TextGenerationRequest,
) -> TextGenerationRequest {
    let mut history = previous.messages();
    history.push(OpenAITextGenerationMessage::new("assistant", response));
    TextGenerationRequest {
        history,
        num_prompt_tokens: previous_prompt_tokens + response_tokens + next.num_prompt_tokens,
        num_decode_tokens: previous.num_decode_tokens, // decode tokens do not change between turns
        ..next.clone()
    }
}

/// Cycles through a fixed list of requests: prompt lists, accuracy items, and
/// the warmup of replay runs.
pub struct RequestListGenerator {
    requests: Vec<TextGenerationRequest>,
    next: usize,
}

impl RequestListGenerator {
    pub fn new(requests: Vec<TextGenerationRequest>) -> anyhow::Result<Self> {
        if requests.is_empty() {
            return Err(anyhow::anyhow!("The request list is empty"));
        }
        Ok(Self { requests, next: 0 })
    }
}

impl TextRequestGenerator for RequestListGenerator {
    fn generate_request(&mut self) -> TextGenerationRequest {
        let request = self.requests[self.next].clone();
        self.next = (self.next + 1) % self.requests.len();
        request
    }
    fn callback(&mut self, _request: Arc<TextGenerationRequest>, _response: &str) {}
}

/// Random prompts of a set length, for runs without a dataset.
///
/// Prompts are random token ids decoded to text, so the server may count a
/// slightly different number of tokens; the server's count wins when it sends one.
pub struct SyntheticRequestGenerator {
    tokenizer: Arc<Tokenizer>,
    prompt_options: TokenizeOptions,
    decode_options: Option<TokenizeOptions>,
}

impl SyntheticRequestGenerator {
    pub fn new(
        tokenizer: Arc<Tokenizer>,
        prompt_options: TokenizeOptions,
        decode_options: Option<TokenizeOptions>,
    ) -> anyhow::Result<Self> {
        if prompt_options.num_tokens.unwrap_or_default() == 0 {
            return Err(anyhow::anyhow!(
                "Synthetic prompts need prompt options with num_tokens > 0"
            ));
        }
        Ok(Self {
            tokenizer,
            prompt_options,
            decode_options,
        })
    }
}

impl TextRequestGenerator for SyntheticRequestGenerator {
    fn generate_request(&mut self) -> TextGenerationRequest {
        let num_tokens = self.prompt_options.sample().unwrap_or(1);
        let vocab = self.tokenizer.get_vocab_size(false).max(1) as u32;
        let mut rng = rand::rng();
        let ids: Vec<u32> = (0..num_tokens)
            .map(|_| rng.random_range(0..vocab))
            .collect();
        TextGenerationRequest {
            prompt: self.tokenizer.decode(&ids, true).unwrap_or_default(),
            num_prompt_tokens: num_tokens,
            num_decode_tokens: self
                .decode_options
                .as_ref()
                .and_then(TokenizeOptions::sample),
            ..Default::default()
        }
    }
    fn callback(&mut self, _request: Arc<TextGenerationRequest>, _response: &str) {}
}

#[cfg(test)]
pub struct DummyTextRequestGenerator {}

#[cfg(test)]
impl DummyTextRequestGenerator {
    pub fn new() -> Self {
        Self {}
    }
}

#[cfg(test)]
impl Default for DummyTextRequestGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
impl TextRequestGenerator for DummyTextRequestGenerator {
    fn generate_request(&mut self) -> TextGenerationRequest {
        TextGenerationRequest {
            prompt: "Hello, world!".to_string(),
            num_prompt_tokens: 2,
            num_decode_tokens: Some(10),
            ..Default::default()
        }
    }
    fn callback(&mut self, _request: Arc<TextGenerationRequest>, _response: &str) {}
}

fn tokenize_prompt(
    prompt: String,
    tokenizer: Arc<Tokenizer>,
    options: &TokenizeOptions,
) -> anyhow::Result<(String, u64)> {
    let prompt_tokens = tokenizer
        .encode(prompt.clone(), false)
        .map_err(|_| anyhow::anyhow!("Error tokenizing prompt"))?;
    match options.num_tokens {
        None => {
            // check if we have a min/max number of tokens, skip prompts that are too short or too long
            if prompt_tokens.len() > options.max_tokens as usize
                || prompt_tokens.len() < options.min_tokens as usize
            {
                return Err(anyhow::anyhow!(format!(
                    "Prompt is too short or too long, skipping: {}<{}<{}",
                    options.min_tokens,
                    prompt_tokens.len(),
                    options.max_tokens
                )));
            }
            Ok((prompt, prompt_tokens.len() as u64))
        }
        Some(num_tokens) => {
            if prompt_tokens.len() < num_tokens as usize {
                return Err(anyhow::anyhow!(format!(
                    "Prompt is too short to tokenize: {}<{}",
                    prompt_tokens.len(),
                    num_tokens
                )));
            }
            let tokens = prompt_tokens
                .get_ids()
                .iter()
                .take(num_tokens as usize)
                .copied()
                .collect::<Vec<u32>>();
            let prompt = tokenizer
                .decode(&tokens, true)
                .map_err(|e| anyhow::anyhow!("Error decoding prompt: {e}"))?;
            Ok((prompt, num_tokens))
        }
    }
}

/// Why a request failed.
#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
    strum_macros::Display,
)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum ErrorKind {
    /// The server answered with a non-2xx status.
    Http,
    /// No complete answer within the request timeout.
    Timeout,
    /// Could not connect, or the connection broke.
    Connection,
    /// The stream ended early or sent no tokens.
    Stream,
    /// The server sent an error object in the stream.
    Api,
    /// The server sent something that is not a valid chunk.
    Decode,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RequestError {
    pub kind: ErrorKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub http_status: Option<u16>,
    pub message: String,
}

impl RequestError {
    pub fn new(kind: ErrorKind, message: String) -> Self {
        Self {
            kind,
            http_status: None,
            message,
        }
    }

    fn from_eventsource(e: Error) -> Self {
        match e {
            Error::InvalidStatusCode(status, _) => Self {
                kind: ErrorKind::Http,
                http_status: Some(status.as_u16()),
                message: status.to_string(),
            },
            Error::Transport(err) if err.is_timeout() => {
                Self::new(ErrorKind::Timeout, err.to_string())
            }
            Error::Transport(err) => Self::new(ErrorKind::Connection, err.to_string()),
            Error::InvalidContentType(..) | Error::Utf8(_) | Error::Parser(_) => {
                Self::new(ErrorKind::Decode, e.to_string())
            }
            other => Self::new(ErrorKind::Stream, other.to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TextGenerationAggregatedResponse {
    pub start_time: Option<tokio::time::Instant>,
    pub end_time: Option<tokio::time::Instant>,
    /// Wall-clock time the request was sent.
    pub started_at: Option<chrono::DateTime<chrono::Utc>>,
    pub num_generated_tokens: u64,
    /// Prompt tokens as counted by the server, when it reports usage.
    pub num_prompt_tokens: Option<u64>,
    pub times_to_tokens: Vec<time::Duration>,
    last_received_token_time: tokio::time::Instant,
    pub failed: bool,
    pub error: Option<RequestError>,
    pub ended: bool,
    pub request: Option<Arc<TextGenerationRequest>>,
    pub response: Option<String>,
    /// Result of the request's check, filled in by the scheduler.
    pub check: Option<CheckOutcome>,
}

impl TextGenerationAggregatedResponse {
    pub fn new(request: Arc<TextGenerationRequest>) -> Self {
        Self {
            request: Some(request),
            ..Self::empty()
        }
    }

    pub fn new_as_ended() -> Self {
        Self {
            ended: true,
            ..Self::empty()
        }
    }

    fn empty() -> Self {
        Self {
            start_time: None,
            end_time: None,
            started_at: None,
            num_generated_tokens: 0,
            num_prompt_tokens: None,
            times_to_tokens: Vec::new(),
            last_received_token_time: tokio::time::Instant::now(),
            failed: false,
            error: None,
            ended: false,
            request: None,
            response: None,
            check: None,
        }
    }

    fn start(&mut self) {
        self.start_time = Some(tokio::time::Instant::now());
        self.started_at = Some(chrono::Utc::now());
        self.last_received_token_time = tokio::time::Instant::now();
    }

    /// Mark the answer complete. The first call wins: a finish chunk is
    /// followed by a usage chunk and `[DONE]`, which must not move the end.
    fn stop(&mut self) {
        if self.end_time.is_none() {
            self.end_time = Some(tokio::time::Instant::now());
        }
    }

    pub(crate) fn fail(&mut self, error: RequestError) {
        self.end_time = Some(tokio::time::Instant::now());
        self.failed = true;
        self.error = Some(error);
    }

    fn add_tokens(&mut self, num_tokens: u64) {
        self.num_generated_tokens += num_tokens;
        let time_to_generate = self.last_received_token_time.elapsed();
        self.last_received_token_time = tokio::time::Instant::now();
        self.times_to_tokens.push(time_to_generate);
    }

    /// Prompt tokens: the server's count when reported, else the request's.
    pub fn prompt_tokens(&self) -> u64 {
        self.num_prompt_tokens
            .unwrap_or_else(|| self.request.as_ref().map_or(0, |r| r.num_prompt_tokens))
    }

    pub fn time_to_first_token(&self) -> Option<std::time::Duration> {
        match self.start_time {
            Some(_) => self.times_to_tokens.first().copied(),
            None => None,
        }
    }

    pub fn inter_token_latency(&self) -> Option<std::time::Duration> {
        match self.times_to_tokens.len() {
            0 => None,
            1 => Some(std::time::Duration::new(0, 0)),
            _ => {
                let mut total_time = std::time::Duration::new(0, 0);
                for i in 1..self.times_to_tokens.len() {
                    total_time += self.times_to_tokens[i];
                }
                Some(total_time / (self.num_generated_tokens.max(2) as u32 - 1))
            }
        }
    }
    pub fn e2e_latency(&self) -> Option<std::time::Duration> {
        match self.start_time {
            Some(start_time) => self.end_time.map(|end_time| end_time - start_time),
            None => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executors::ExecutorConfig;
    use crate::results::BenchmarkResults;
    use crate::scheduler::ExecutorType;
    use std::path::PathBuf;
    use std::sync::atomic::AtomicU64;
    use std::thread::sleep;
    use std::time::Duration;
    use tokio::sync::RwLock;

    pub(crate) fn test_tokenizer() -> Arc<Tokenizer> {
        crate::test_util::tokenizer()
    }

    #[tokio::test]
    async fn test_openai_token_count() {
        let mut s = mockito::Server::new_async().await;
        s.mock("POST", "/v1/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_chunked_body(|w| {
                w.write_all(b"data: {\"choices\": [{\"message\": null, \"finish_reason\": null, \"delta\": {\"content\": \"Hello, world!\"}}]}\n\n").unwrap();
                w.write_all(b"data: {\"choices\": [{\"message\": null, \"finish_reason\": null, \"delta\": {\"content\": \"Hello, world!\"}}]}\n\n").unwrap();
                w.write_all(b"data: {\"choices\": [{\"message\": null, \"finish_reason\": null, \"delta\": {\"content\": \"Hello, world!\"}}]}\n\n").unwrap();
                w.write_all(b"data: {\"choices\": [{\"message\": {\"content\": \"Hello, world!Hello, world!Hello, world!Hello, world!\", \"role\": \"user\"}, \"finish_reason\": \"stop\", \"delta\": {\"content\": \"Hello, world!\"}}]}\n\n").unwrap();
                w.write_all(b"data: [DONE]\n\n")
            })
            .create_async().await;
        let url = s.url().parse().unwrap();
        let tokenizer = test_tokenizer();
        let backend = OpenAITextGenerationBackend::try_new(
            "".to_string(),
            url,
            "gpt2".to_string(),
            tokenizer,
            time::Duration::from_secs(10),
        )
        .unwrap();
        let request = TextGenerationRequest {
            id: None,
            prompt: "Hello, world!".to_string(),
            num_prompt_tokens: 2,
            num_decode_tokens: Some(10),
            ..Default::default()
        };
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        let request = Arc::new(request);
        tokio::spawn(async move {
            backend.generate(request.clone(), tx).await;
        });
        let num_tokens = Arc::new(AtomicU64::new(0));
        let num_tokens_clone = num_tokens.clone();
        let t = tokio::spawn(async move {
            while let Some(item) = rx.recv().await {
                let response = item;
                assert_eq!(response.failed, false);
                num_tokens_clone.fetch_add(
                    response.num_generated_tokens,
                    std::sync::atomic::Ordering::SeqCst,
                );
            }
        });
        t.await.unwrap();
        assert_eq!(
            num_tokens.load(std::sync::atomic::Ordering::SeqCst),
            16 as u64
        );
    }

    /// Test that the timings are correct
    /// The tests may be flaky due to the nature of the SSE connection (it may depend on the testing environment)
    /// We need to account for the time it takes to establish the connection
    /// and the time it takes to receive the first message
    #[tokio::test]
    #[ignore] // flaky test, don't run it on CI
    async fn test_openai_timings() {
        let mut s = mockito::Server::new_async().await;
        s.mock("POST", "/v1/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_chunked_body(|w| {
                w.write_all(b"data: {\"choices\": [{\"message\": null, \"finish_reason\": null, \"delta\": {\"content\": \"Hello, world!\"}}]}\n\n").unwrap();
                // sleep for 500ms
                sleep(std::time::Duration::from_millis(500));
                w.write_all(b"data: {\"choices\": [{\"message\": null, \"finish_reason\": null, \"delta\": {\"content\": \"Hello, world!\"}}]}\n\n").unwrap();
                w.write_all(b"data: {\"choices\": [{\"message\": null, \"finish_reason\": null, \"delta\": {\"content\": \"Hello, world!\"}}]}\n\n").unwrap();
                w.write_all(b"data: {\"choices\": [{\"message\": {\"content\": \"Hello, world!Hello, world!Hello, world!Hello, world!\", \"role\": \"user\"}, \"finish_reason\": \"stop\", \"delta\": {\"content\": \"Hello, world!\"}}]}\n\n").unwrap();
                w.write_all(b"data: [DONE]\n\n")
            })
            .create_async().await;
        let url = s.url().parse().unwrap();
        let tokenizer = test_tokenizer();
        let backend = OpenAITextGenerationBackend::try_new(
            "".to_string(),
            url,
            "gpt2".to_string(),
            tokenizer,
            time::Duration::from_secs(10),
        )
        .unwrap();
        let request = TextGenerationRequest {
            id: None,
            prompt: "Hello, world!".to_string(),
            num_prompt_tokens: 2,
            num_decode_tokens: Some(16),
            ..Default::default()
        };
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        let request = Arc::new(request);
        tokio::spawn(async move {
            backend.generate(request.clone(), tx.clone()).await;
        });
        let results = BenchmarkResults::new(
            "test".to_string(),
            ExecutorType::ConstantArrivalRate,
            ExecutorConfig {
                max_vus: 1,
                duration: Duration::from_secs(10),
                rate: None,
            },
        );
        let results = Arc::new(RwLock::new(results));
        let results_clone = results.clone();
        let t = tokio::spawn(async move {
            let mut handle = results_clone.write().await;
            while let Some(item) = rx.recv().await {
                let response = item;
                handle.add_response(response);
            }
        });
        t.await.unwrap();
        let results = results.read().await;
        let e2e_latency_avg = results.e2e_latency_avg().unwrap();
        let inter_token_latency_avg = results.inter_token_latency_avg().unwrap();
        let ttft = results.time_to_first_token_avg().unwrap();

        let e2e_timing_overhead = Duration::from_millis(10);
        let expected_e2e_latency_avg = Duration::from_millis(500);
        let expected_inter_token_latency_avg = Duration::from_millis(33); // 16-1 tokens with a 500ms delay
        let inter_token_latency_overhead = Duration::from_millis(3);
        let expected_ttft = Duration::from_millis(3); // account for http overhead
        assert!(
            e2e_latency_avg > expected_e2e_latency_avg
                && e2e_latency_avg < expected_e2e_latency_avg + e2e_timing_overhead,
            "e2e_latency_avg: {:?} < {:?} < {:?}",
            expected_e2e_latency_avg,
            e2e_latency_avg,
            expected_e2e_latency_avg + e2e_timing_overhead
        );
        assert!(
            inter_token_latency_avg > expected_inter_token_latency_avg
                && inter_token_latency_avg
                    < expected_inter_token_latency_avg + inter_token_latency_overhead,
            "inter_token_latency_avg: {:?} < {:?} < {:?}",
            expected_inter_token_latency_avg,
            inter_token_latency_avg,
            expected_inter_token_latency_avg + inter_token_latency_overhead
        );
        assert!(
            ttft < expected_ttft,
            "TTFT: {:?} < {:?}",
            ttft,
            expected_ttft
        );
    }

    /// Test that server errors are handled correctly
    #[tokio::test]
    async fn test_openai_fails_on_error() {
        let mut s = mockito::Server::new_async().await;
        s.mock("POST", "/v1/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_chunked_body(|w| w.write_all(b"data: {\"error\": \"Internal server error\"}\n\n"))
            .create_async()
            .await;
        let url = s.url().parse().unwrap();
        let tokenizer = test_tokenizer();
        let backend = OpenAITextGenerationBackend::try_new(
            "".to_string(),
            url,
            "gpt2".to_string(),
            tokenizer,
            time::Duration::from_secs(10),
        )
        .unwrap();
        let request = TextGenerationRequest {
            id: None,
            prompt: "Hello, world!".to_string(),
            num_prompt_tokens: 2,
            num_decode_tokens: Some(16),
            ..Default::default()
        };
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        let request = Arc::new(request);
        tokio::spawn(async move {
            backend.generate(request.clone(), tx.clone()).await;
        });
        let responses = Arc::new(RwLock::new(Vec::new()));
        let responses_clone = responses.clone();
        let t = tokio::spawn(async move {
            let mut handle = responses_clone.write().await;
            while let Some(item) = rx.recv().await {
                let response = item;
                handle.push(response);
            }
        });
        t.await.unwrap();
        let responses = responses.read().await;
        assert_eq!(responses.len(), 1);
        assert_eq!(responses[0].failed, true);
    }

    /// Test that bad responses are handled correctly
    #[tokio::test]
    async fn test_openai_fails_on_bad_response() {
        let mut s = mockito::Server::new_async().await;
        s.mock("POST", "/v1/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_chunked_body(|w| w.write_all(b"this is wrong\n\n"))
            .create_async()
            .await;
        let url = s.url().parse().unwrap();
        let tokenizer = test_tokenizer();
        let backend = OpenAITextGenerationBackend::try_new(
            "".to_string(),
            url,
            "gpt2".to_string(),
            tokenizer,
            time::Duration::from_secs(10),
        )
        .unwrap();
        let request = TextGenerationRequest {
            id: None,
            prompt: "Hello, world!".to_string(),
            num_prompt_tokens: 2,
            num_decode_tokens: Some(16),
            ..Default::default()
        };
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        let request = Arc::new(request);
        tokio::spawn(async move {
            backend.generate(request.clone(), tx.clone()).await;
        });
        let responses = Arc::new(RwLock::new(Vec::new()));
        let responses_clone = responses.clone();
        let t = tokio::spawn(async move {
            let mut handle = responses_clone.write().await;
            while let Some(item) = rx.recv().await {
                let response = item;
                handle.push(response);
            }
        });
        t.await.unwrap();
        let responses = responses.read().await;
        assert_eq!(responses.len(), 1);
        assert_eq!(responses[0].failed, true);
    }

    /// Test that malformed JSON responses are handled correctly
    #[tokio::test]
    async fn test_openai_fails_on_malformed_json() {
        let mut s = mockito::Server::new_async().await;
        s.mock("POST", "/v1/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_chunked_body(|w| w.write_all(b"data: {\"foo\": \"bar\"}\n\n"))
            .create_async()
            .await;
        let url = s.url().parse().unwrap();
        let tokenizer = test_tokenizer();
        let backend = OpenAITextGenerationBackend::try_new(
            "".to_string(),
            url,
            "gpt2".to_string(),
            tokenizer,
            time::Duration::from_secs(10),
        )
        .unwrap();
        let request = TextGenerationRequest {
            id: None,
            prompt: "Hello, world!".to_string(),
            num_prompt_tokens: 2,
            num_decode_tokens: Some(16),
            ..Default::default()
        };
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        let request = Arc::new(request);
        tokio::spawn(async move {
            backend.generate(request.clone(), tx.clone()).await;
        });
        let responses = Arc::new(RwLock::new(Vec::new()));
        let responses_clone = responses.clone();
        let t = tokio::spawn(async move {
            let mut handle = responses_clone.write().await;
            while let Some(item) = rx.recv().await {
                let response = item;
                handle.push(response);
            }
        });
        t.await.unwrap();
        let responses = responses.read().await;
        assert_eq!(responses.len(), 1);
        assert_eq!(responses[0].failed, true);
    }

    /// Test that request timeout is handled correctly
    #[tokio::test]
    async fn test_timeout_should_fail_request() {
        let mut s = mockito::Server::new_async().await;
        s.mock("POST", "/v1/chat/completions")
            .with_status(200)
            .with_header("content-type", "text/event-stream")
            .with_chunked_body(|w| {
                w.write_all(b"data: {\"choices\": [{\"message\": null, \"finish_reason\": null, \"delta\": {\"content\": \"Hello, world!\"}}]}\n\n").unwrap();
                // sleep for 5s
                sleep(Duration::from_secs(5));
                w.write_all(b"data: [DONE]\n\n")
            })
            .create_async().await;
        let url = s.url().parse().unwrap();
        let tokenizer = test_tokenizer();
        let backend = OpenAITextGenerationBackend::try_new(
            "".to_string(),
            url,
            "gpt2".to_string(),
            tokenizer,
            Duration::from_secs(1),
        )
        .unwrap();
        let request = TextGenerationRequest {
            id: None,
            prompt: "Hello, world!".to_string(),
            num_prompt_tokens: 2,
            num_decode_tokens: Some(16),
            ..Default::default()
        };
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        let request = Arc::new(request);
        tokio::spawn(async move {
            backend.generate(request.clone(), tx.clone()).await;
        });
        let reponses = Arc::new(RwLock::new(Vec::new()));
        let responses_clone = reponses.clone();
        let t = tokio::spawn(async move {
            while let Some(item) = rx.recv().await {
                responses_clone.write().await.push(item);
            }
        });
        t.await.unwrap();
        let responses = reponses.read().await;
        assert_eq!(responses.len(), 1);
        assert_eq!(responses[0].failed, true);
    }

    /// Test that conversations are correctly loaded
    #[tokio::test]
    async fn test_load_conversations_from_file() {
        let filepath = PathBuf::from("test_data/conversations.json");
        let prompt_tokenize_opts = TokenizeOptions::default();
        let decode_tokenize_opts = TokenizeOptions::default();
        let generator = ConversationTextRequestGenerator::load(
            filepath,
            test_tokenizer(),
            Some(prompt_tokenize_opts),
            Some(decode_tokenize_opts),
        )
        .unwrap();
        // one request per user message (system messages become history)
        assert_eq!(generator.requests.len(), 17005);
    }

    /// Test that conversations are bounded by the min/max number of tokens
    #[tokio::test]
    async fn test_load_conversations_bounded() {
        let filepath = PathBuf::from("test_data/conversations.json");
        let prompt_tokenize_opts = TokenizeOptions {
            num_tokens: None,
            min_tokens: 4,
            max_tokens: 1024,
            variance: 0,
        };
        let decode_tokenize_opts = TokenizeOptions::default();
        let generator = ConversationTextRequestGenerator::load(
            filepath,
            test_tokenizer(),
            Some(prompt_tokenize_opts),
            Some(decode_tokenize_opts),
        )
        .unwrap();
        let min_tokens = generator
            .requests
            .iter()
            .map(|r| r.1.request.num_prompt_tokens)
            .min()
            .unwrap();
        let max_tokens = generator
            .requests
            .iter()
            .map(|r| r.1.request.num_prompt_tokens)
            .max()
            .unwrap();
        assert!(min_tokens >= 4, "Min tokens: {}", min_tokens);
        assert!(max_tokens <= 1024, "Max tokens: {}", max_tokens);
    }

    /// Test that conversations prompts have the correct number of tokens
    #[tokio::test]
    async fn test_load_conversations_fixed_tokens() {
        let filepath = PathBuf::from("test_data/conversations.json");
        let prompt_tokenize_opts = TokenizeOptions {
            num_tokens: Some(200),
            min_tokens: 200,
            max_tokens: 200,
            variance: 0,
        };
        let decode_tokenize_opts = TokenizeOptions::default();
        let generator = ConversationTextRequestGenerator::load(
            filepath,
            test_tokenizer(),
            Some(prompt_tokenize_opts),
            Some(decode_tokenize_opts),
        )
        .unwrap();
        for r in generator.requests.iter() {
            assert_eq!(r.1.request.num_prompt_tokens, 200);
        }
    }

    /// Test that multi-turn conversations are correctly loaded
    /// The test data contains 2 conversations: the first with 6 user turns, the second with a system prompt and 1 user turn
    #[tokio::test]
    async fn test_load_conversations_multi_turn() {
        let filepath = PathBuf::from("test_data/chat.json");
        let prompt_tokenize_opts = TokenizeOptions {
            num_tokens: None,
            min_tokens: 1,
            max_tokens: 200,
            variance: 0,
        };
        let decode_tokenize_opts = TokenizeOptions::default();
        let generator = ConversationTextRequestGenerator::load(
            filepath,
            test_tokenizer(),
            Some(prompt_tokenize_opts),
            Some(decode_tokenize_opts),
        )
        .unwrap();
        let turns = generator
            .requests
            .into_iter()
            .map(|r| r.1.clone())
            .collect::<Vec<ConversationTurnRequest>>();
        assert_eq!(turns.len(), 7);
        let first_turns = turns.clone().into_iter().filter(|t| t.priority == 0);
        // we expect to have 2 None values for the first turn in each conversation
        assert_eq!(first_turns.count(), 2);
        let first_conversation = turns.clone().into_iter().filter(|t| t.request.prompt == "Summarize the main ideas of Jeff Walker's Product Launch Formula into bullet points as it pertains to a growth marketing agency implementing these strategies and tactics for their clients...").collect::<Vec<ConversationTurnRequest>>();
        // rebuild the conversation from first turn
        let mut conversation = Vec::new();
        let mut current_turn = first_conversation[0].clone();
        loop {
            conversation.push(current_turn.clone());
            match turns
                .iter()
                .find(|t| t.id == current_turn.request.id.unwrap_or_default())
            {
                Some(t) => current_turn = t.clone(),
                None => break,
            }
        }
        assert_eq!(conversation.len(), 6);
        let got = conversation
            .iter()
            .map(|t| t.request.prompt.clone())
            .collect::<Vec<String>>()
            .join("\n");
        let expect = "Summarize the main ideas of Jeff Walker's Product Launch Formula into bullet points as it pertains to a growth marketing agency implementing these strategies and tactics for their clients...\nSummarize the main ideas of Brendon Burchard's Experts Academy into bullet points as it pertains to a growth marketing agency implementing these strategies and tactics for their clients...\nWhat are the mental triggers in Jeff Walker's Product Launch Formula and \"Launch\" book?\nWrite a summary of why scarcity and urgency are the strongest mental triggers and have been the driving force behind many of our best performing campaigns over the last 8 years.\nSummarize Russell Brunson's Perfect Webinar Script...\nSummarize the 6 human needs as Tony Robbins explains...";
        assert_eq!(expect, got);
    }

    /// Test that only first turns of multi-turn conversations are queued
    #[tokio::test]
    async fn test_conversation_queue() {
        let filepath = PathBuf::from("test_data/chat.json");
        let prompt_tokenize_opts = TokenizeOptions {
            num_tokens: None,
            min_tokens: 1,
            max_tokens: 200,
            variance: 0,
        };
        let decode_tokenize_opts = TokenizeOptions::default();
        let mut generator = ConversationTextRequestGenerator::load(
            filepath,
            test_tokenizer(),
            Some(prompt_tokenize_opts),
            Some(decode_tokenize_opts),
        )
        .unwrap();
        for i in 0..20 {
            let req = generator.generate_request();
            if i % 2 == 0 {
                // first turn of the first conversation
                assert_eq!(req.prompt, "Summarize the main ideas of Jeff Walker's Product Launch Formula into bullet points as it pertains to a growth marketing agency implementing these strategies and tactics for their clients...");
            }
            if i % 2 == 1 {
                // first turn of the second conversation, its system prompt as history
                assert!(req.prompt.starts_with("How to tell if a customer segment"));
                assert_eq!(
                    req.history,
                    vec![OpenAITextGenerationMessage::new(
                        "system",
                        "You are a helpful assistant."
                    )]
                );
            }
        }
    }

    /// Test that multi-turn conversations are correctly queued when we add responses
    #[tokio::test]
    async fn test_conversation_turns_queue() {
        let filepath = PathBuf::from("test_data/chat.json");
        let prompt_tokenize_opts = TokenizeOptions {
            num_tokens: None,
            min_tokens: 1,
            max_tokens: 200,
            variance: 0,
        };
        let decode_tokenize_opts = TokenizeOptions::default();
        let mut generator = ConversationTextRequestGenerator::load(
            filepath,
            test_tokenizer(),
            Some(prompt_tokenize_opts),
            Some(decode_tokenize_opts),
        )
        .unwrap();
        const FIRST: &str = "Summarize the main ideas of Jeff Walker's Product Launch Formula into bullet points as it pertains to a growth marketing agency implementing these strategies and tactics for their clients...";
        const SECOND: &str = "Summarize the main ideas of Brendon Burchard's Experts Academy into bullet points as it pertains to a growth marketing agency implementing these strategies and tactics for their clients...";
        // generate the first user turn
        let first = generator.generate_request();
        assert_eq!(first.prompt, FIRST);
        generator.callback(Arc::from(first.clone()), "This is my response");
        // the next turn carries the conversation so far, including the model's answer
        let req = generator.generate_request();
        assert_eq!(req.prompt, SECOND);
        assert_eq!(
            req.history,
            vec![
                OpenAITextGenerationMessage::new("user", FIRST),
                OpenAITextGenerationMessage::new("assistant", "This is my response"),
            ]
        );
        assert_eq!(req.turn, 1);
        let tokenizer = test_tokenizer();
        assert_eq!(
            req.num_prompt_tokens,
            first.num_prompt_tokens
                + count_tokens(&tokenizer, "This is my response")
                + count_tokens(&tokenizer, SECOND)
        );
        // now add a response to the second turn
        generator.callback(Arc::from(req), "This is my second response");
        let req = generator.generate_request();
        assert_eq!(
            req.prompt,
            "What are the mental triggers in Jeff Walker's Product Launch Formula and \"Launch\" book?"
        );
        assert_eq!(req.history.len(), 4);
        assert_eq!(req.history[3].content, "This is my second response");
        // check that next turn is a first turn
        let req = generator.generate_request();
        assert!(req.prompt.starts_with("How to tell if a customer segment"));
    }
}
