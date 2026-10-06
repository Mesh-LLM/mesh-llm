//! Native `/completion` projection. Counts come from the final native receipt;
//! content chunks are never substituted for generated tokens.
use super::{Decoder, TransportResult, post};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::time::Duration;

const BODY_LIMIT: usize = 1024 * 1024;
const LINE_LIMIT: usize = 64 * 1024;

#[derive(Debug, Serialize)]
pub(in crate::automation) struct Evidence {
    pub streaming: bool,
    pub elapsed_seconds: f64,
    /// Absent for serial JSON, which cannot observe a first-token arrival.
    pub ttft_seconds: Option<f64>,
    pub tokens_predicted: u64,
    pub tokens_evaluated: u64,
    /// Post-completion slot occupancy; not a prompt-cache-hit count.
    pub tokens_cached: Option<u64>,
    pub prompt_n: u64,
    pub cache_n: u64,
    pub prompt_ms: f64,
    pub predicted_ms: f64,
    pub content_events: usize,
    /// Native plain UTF-8 content digest; distinct from OpenAI's typed output digest.
    pub content_sha256: String,
    /// Native first nonempty UTF-8 content chunk digest. Chunk partition is observable.
    pub first_generated_sha256: Option<String>,
    pub model: Option<String>,
    pub stop_type: String,
}

#[derive(Deserialize)]
struct Event {
    error: Option<serde_json::Value>,
    stop: bool,
    #[serde(default)]
    content: String,
    tokens_predicted: Option<u64>,
    tokens_evaluated: Option<u64>,
    tokens_cached: Option<u64>,
    truncated: Option<bool>,
    model: Option<String>,
    stop_type: Option<String>,
    timings: Option<Timings>,
}

#[derive(Deserialize)]
struct Timings {
    prompt_n: u64,
    cache_n: u64,
    predicted_n: u64,
    prompt_ms: f64,
    predicted_ms: f64,
}

struct NativeDecoder {
    streaming: bool,
    output_tokens: u64,
    pending: Vec<u8>,
    bytes_seen: usize,
    final_event: Option<Event>,
    content: String,
    first: Option<Duration>,
    first_generated_sha256: Option<String>,
    content_events: usize,
}

impl NativeDecoder {
    fn new(streaming: bool, output_tokens: u64) -> Self {
        Self {
            streaming,
            output_tokens,
            pending: Vec::new(),
            bytes_seen: 0,
            final_event: None,
            content: String::new(),
            first: None,
            first_generated_sha256: None,
            content_events: 0,
        }
    }

    fn event(&mut self, bytes: &[u8], elapsed: Duration) -> Result<(), String> {
        let event: Event =
            serde_json::from_slice(bytes).map_err(|_| "invalid native completion JSON")?;
        if event.error.is_some() {
            return Err("native completion returned a server error".into());
        }
        if !event.content.is_empty() {
            if self.streaming {
                self.first.get_or_insert(elapsed);
                if self.first_generated_sha256.is_none() {
                    self.first_generated_sha256 =
                        Some(hex::encode(Sha256::digest(event.content.as_bytes())));
                }
            }
            self.content.push_str(&event.content);
            self.content_events += 1;
        }
        if event.stop {
            self.final_event = Some(event);
        } else if !self.streaming {
            return Err("serial native completion lacks terminal stop:true".into());
        }
        Ok(())
    }

    fn line(&mut self, bytes: &[u8], elapsed: Duration) -> Result<(), String> {
        let line = std::str::from_utf8(bytes)
            .map_err(|_| "invalid native SSE UTF-8")?
            .trim();
        // SSE comments/event names are framing, not completion events.
        let Some(payload) = line.strip_prefix("data:") else {
            return Ok(());
        };
        let payload = payload.trim_start();
        if payload == "[DONE]" {
            return Err("native completion ended without stop:true receipt".into());
        }
        self.event(payload.as_bytes(), elapsed)
    }

    fn observed(&mut self, elapsed: Duration) -> Result<Event, String> {
        if !self.pending.is_empty() && !self.terminal() {
            let pending = std::mem::take(&mut self.pending);
            if self.streaming {
                self.line(&pending, elapsed)?;
            } else {
                self.event(&pending, elapsed)?;
            }
        }
        self.final_event
            .take()
            .ok_or_else(|| "native completion ended without stop:true receipt".into())
    }
}

impl Decoder for NativeDecoder {
    type Evidence = Evidence;
    fn accepts(&self, status: hyper::StatusCode) -> bool {
        status == hyper::StatusCode::OK
    }
    fn terminal(&self) -> bool {
        self.final_event.is_some()
    }

    fn consume(&mut self, bytes: &[u8], elapsed: Duration) -> Result<(), String> {
        if bytes.len() > BODY_LIMIT.saturating_sub(self.bytes_seen) {
            return Err("native completion response exceeds 1 MiB".into());
        }
        self.bytes_seen += bytes.len();
        self.pending.extend_from_slice(bytes);
        if !self.streaming {
            return Ok(());
        }
        while let Some(end) = self.pending.iter().position(|byte| *byte == b'\n') {
            if end > LINE_LIMIT {
                return Err("native SSE line exceeds 64 KiB".into());
            }
            let line: Vec<_> = self.pending.drain(..=end).collect();
            self.line(&line, elapsed)?;
            if self.terminal() {
                break;
            }
        }
        if !self.terminal() && self.pending.len() > LINE_LIMIT {
            return Err("native SSE line exceeds 64 KiB".into());
        }
        Ok(())
    }

    fn finish(mut self, elapsed: Duration) -> Result<Evidence, String> {
        let event = self.observed(elapsed)?;
        let predicted = event
            .tokens_predicted
            .ok_or("native receipt lacks tokens_predicted")?;
        let evaluated = event
            .tokens_evaluated
            .ok_or("native receipt lacks tokens_evaluated")?;
        let timings = event.timings.ok_or("native receipt lacks timings")?;
        if predicted == 0
            || predicted > self.output_tokens
            || timings.predicted_n != predicted
            || evaluated == 0
            || timings.prompt_n > evaluated
            || timings.cache_n > evaluated
        {
            return Err("native receipt has inconsistent token counters".into());
        }
        if event.truncated != Some(false) {
            return Err("native receipt has absent/truncated prompt admission".into());
        }
        if !timings.prompt_ms.is_finite()
            || !timings.predicted_ms.is_finite()
            || timings.prompt_ms < 0.0
            || timings.predicted_ms < 0.0
        {
            return Err("native receipt has invalid timings".into());
        }
        if event
            .model
            .as_ref()
            .is_some_and(|model| model.len() > 4096 || model.chars().any(char::is_control))
        {
            return Err("native receipt has invalid bounded model label".into());
        }
        let stop_type = event.stop_type.ok_or("native receipt lacks stop_type")?;
        if !matches!(stop_type.as_str(), "limit" | "word" | "eos") {
            return Err("native receipt has unsupported stop_type".into());
        }
        if self.streaming && self.first.is_none() {
            return Err("native SSE completed without generated content".into());
        }
        Ok(Evidence {
            streaming: self.streaming,
            elapsed_seconds: elapsed.as_secs_f64(),
            ttft_seconds: self.first.map(|first| first.as_secs_f64()),
            tokens_predicted: predicted,
            tokens_evaluated: evaluated,
            tokens_cached: event.tokens_cached,
            prompt_n: timings.prompt_n,
            cache_n: timings.cache_n,
            prompt_ms: timings.prompt_ms,
            predicted_ms: timings.predicted_ms,
            content_events: self.content_events,
            content_sha256: hex::encode(Sha256::digest(self.content.as_bytes())),
            first_generated_sha256: self.first_generated_sha256,
            model: event.model,
            stop_type,
        })
    }
}

pub(in crate::automation) async fn request(
    base: &str,
    prompt: &str,
    output_tokens: u64,
    streaming: bool,
) -> TransportResult<Evidence> {
    if prompt.is_empty() || prompt.len() > 64 * 1024 || !(1..=4096).contains(&output_tokens) {
        return Err(
            "native completion requires bounded nonempty prompt and output tokens 1..4096".into(),
        );
    }
    let started = std::time::Instant::now();
    let uri = format!("{}/completion", base.trim_end_matches('/')).parse()?;
    let body = serde_json::json!({"prompt":prompt,"n_predict":output_tokens,
        "temperature":0,"top_k":1,"cache_prompt":true,"stream":streaming});
    post(
        uri,
        &body,
        started,
        false,
        NativeDecoder::new(streaming, output_tokens),
    )
    .await
}

#[cfg(test)]
#[path = "native_completion_tests.rs"]
mod tests;
