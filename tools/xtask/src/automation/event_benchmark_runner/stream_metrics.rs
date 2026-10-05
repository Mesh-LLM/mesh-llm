//! Minimal streaming benchmark metrics, independent from cache telemetry.
use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::time::Duration;

pub(super) const DECODE_EPSILON_SECONDS: f64 = 1e-6;
const MAX_BYTES: usize = 16 * 1024 * 1024;
const MAX_LINE: usize = 1024 * 1024;

#[derive(Default)]
pub(super) struct Stream {
    pending: Vec<u8>,
    seen: usize,
    done: bool,
    tokens: Option<u64>,
    first: Option<Duration>,
    server_error: bool,
}
#[derive(Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct Measurement {
    pub completion_tokens: Option<u64>,
    pub ttft_ms: Option<f64>,
    pub elapsed_ms: f64,
    pub decode_tok_s: Option<f64>,
    pub decode_only_tok_s: Option<f64>,
    pub malformed: bool,
}

pub(super) fn decode_rate(tokens: Option<u64>, elapsed_ms: Option<f64>) -> Option<f64> {
    let elapsed = elapsed_ms?;
    if !elapsed.is_finite() || elapsed <= 0.0 {
        return None;
    }
    let rate = tokens? as f64 / (elapsed / 1000.0);
    rate.is_finite().then_some(rate)
}
pub(super) fn decode_only(
    tokens: Option<u64>,
    elapsed_ms: Option<f64>,
    ttft_ms: Option<f64>,
) -> Option<f64> {
    let elapsed = elapsed_ms?;
    let ttft = ttft_ms?;
    if !elapsed.is_finite() || !ttft.is_finite() || ttft < 0.0 {
        return None;
    }
    let interval = (elapsed - ttft) / 1000.0;
    if interval <= 0.0 {
        return None;
    }
    let rate = tokens? as f64 / interval.max(DECODE_EPSILON_SECONDS);
    rate.is_finite().then_some(rate)
}

impl Stream {
    pub fn terminal(&self) -> bool {
        self.done || self.server_error
    }
    pub fn consume(&mut self, bytes: &[u8], elapsed: Duration) -> DynResult<()> {
        if self.terminal() {
            return Ok(());
        }
        self.seen = self
            .seen
            .checked_add(bytes.len())
            .ok_or("SSE byte count overflow")?;
        if self.seen > MAX_BYTES {
            return Err("benchmark SSE exceeds16MiB".into());
        }
        self.pending.extend_from_slice(bytes);
        while let Some(end) = self.pending.iter().position(|byte| *byte == b'\n') {
            if end > MAX_LINE {
                return Err("benchmark SSE line exceeds1MiB".into());
            }
            let line = self.pending.drain(..=end).collect::<Vec<_>>();
            self.line(&line, elapsed);
            if self.terminal() {
                break;
            }
        }
        if self.pending.len() > MAX_LINE {
            return Err("benchmark SSE line exceeds1MiB".into());
        }
        Ok(())
    }
    fn line(&mut self, bytes: &[u8], elapsed: Duration) {
        let Ok(line) = std::str::from_utf8(bytes) else {
            return;
        };
        let Some(payload) = line.trim().strip_prefix("data:") else {
            return;
        };
        let payload = payload.trim();
        if payload == "[DONE]" {
            self.done = true;
            return;
        }
        let Ok(value) = serde_json::from_str::<Value>(payload) else {
            return;
        };
        if value.get("error").is_some_and(|error| !error.is_null()) {
            self.server_error = true;
            return;
        }
        if self.first.is_none()
            && value
                .pointer("/choices/0/delta/content")
                .and_then(Value::as_str)
                .is_some_and(|content| !content.is_empty())
        {
            self.first = Some(elapsed);
        }
        if let Some(tokens) = value
            .pointer("/usage/completion_tokens")
            .and_then(Value::as_u64)
        {
            self.tokens = Some(tokens);
        }
    }
    pub fn finish(mut self, elapsed: Duration) -> Measurement {
        if !self.terminal() && !self.pending.is_empty() {
            let pending = std::mem::take(&mut self.pending);
            self.line(&pending, elapsed);
        }
        let malformed = self.server_error || self.tokens.is_none();
        let tokens = if malformed { None } else { self.tokens };
        let ttft = if malformed {
            None
        } else {
            self.first.map(|time| time.as_secs_f64() * 1000.0)
        };
        let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
        Measurement {
            completion_tokens: tokens,
            ttft_ms: ttft,
            elapsed_ms,
            decode_tok_s: decode_rate(tokens, Some(elapsed_ms)),
            decode_only_tok_s: decode_only(tokens, Some(elapsed_ms), ttft),
            malformed,
        }
    }
}

pub(super) fn chat_body(prompt: &str, tokens: u64, model: &str) -> DynResult<Value> {
    if prompt.is_empty() || model.trim().is_empty() || tokens == 0 {
        return Err(
            "streaming benchmark requests require prompt, resolved model and positive token count"
                .into(),
        );
    }
    Ok(
        json!({"model":model,"messages":[{"role":"user","content":prompt}],"max_tokens":tokens,"temperature":0.0,"stream":true,"stream_options":{"include_usage":true}}),
    )
}
pub(super) fn first_model(value: &Value) -> Option<&str> {
    value
        .get("data")?
        .as_array()?
        .first()?
        .get("id")?
        .as_str()
        .filter(|id| !id.is_empty())
}

#[cfg(test)]
#[path = "stream_metrics_tests.rs"]
mod tests;
