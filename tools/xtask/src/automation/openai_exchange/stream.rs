use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::time::Duration;

#[derive(Default)]
pub(in crate::automation) struct Stream {
    pending: Vec<u8>,
    done: bool,
    error: Option<String>,
    usage: Option<Usage>,
    first: Option<Duration>,
    first_generated_sha256: Option<String>,
    events: Vec<Duration>,
    content: String,
    reasoning: String,
    tools: BTreeMap<usize, Tool>,
    finish_reason: Option<String>,
    bytes_seen: usize,
}

#[derive(Deserialize)]
struct Event {
    error: Option<Value>,
    usage: Option<Usage>,
    #[serde(default)]
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Usage {
    prompt_tokens: u64,
    completion_tokens: u64,
    prompt_tokens_details: Details,
}

#[derive(Deserialize)]
struct Details {
    cached_tokens: u64,
}

#[derive(Deserialize)]
struct Choice {
    finish_reason: Option<String>,
    delta: Option<Delta>,
}

#[derive(Deserialize, Serialize)]
struct Delta {
    content: Option<String>,
    reasoning_content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolDelta>>,
}

#[derive(Deserialize, Serialize)]
struct ToolDelta {
    index: Option<usize>,
    #[serde(rename = "type")]
    kind: Option<String>,
    function: Option<FunctionDelta>,
}

#[derive(Deserialize, Serialize)]
struct FunctionDelta {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Default, Serialize)]
struct Tool {
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    kind: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    function: Option<Function>,
}

#[derive(Default, Serialize)]
struct Function {
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    arguments: Option<String>,
}

#[derive(Debug, Serialize)]
pub(in crate::automation) struct Evidence {
    pub ttft_seconds: f64,
    pub elapsed_seconds: f64,
    pub generation_seconds: f64,
    pub decode_inter_token_seconds: Vec<f64>,
    pub completion_tokens: u64,
    pub prompt_tokens: u64,
    pub cached_tokens: u64,
    pub cache_pct: f64,
    pub content_events: usize,
    pub finish_reason: Option<String>,
    pub content_sha256: String,
    /// SHA256 of the first nonempty typed delta, including null optional fields.
    pub first_generated_sha256: Option<String>,
}

impl Stream {
    pub(in crate::automation) fn terminal(&self) -> bool {
        self.done || self.error.is_some()
    }

    pub(in crate::automation) fn consume(
        &mut self,
        bytes: &[u8],
        elapsed: Duration,
    ) -> Result<(), String> {
        if self.done {
            return Ok(());
        }
        self.bytes_seen = self.bytes_seen.saturating_add(bytes.len());
        if self.bytes_seen > 64 * 1024 * 1024 {
            return Err("replay SSE response exceeds 64 MiB".into());
        }
        self.pending.extend_from_slice(bytes);
        while let Some(end) = self.pending.iter().position(|byte| *byte == b'\n') {
            if end > 16 * 1024 * 1024 {
                return Err("replay SSE line exceeds 16 MiB".into());
            }
            let line: Vec<_> = self.pending.drain(..=end).collect();
            self.line(&line, elapsed)?;
            if self.terminal() {
                break;
            }
        }
        if self.pending.len() > 16 * 1024 * 1024 {
            return Err("replay SSE line exceeds 16 MiB".into());
        }
        Ok(())
    }

    fn line(&mut self, line: &[u8], elapsed: Duration) -> Result<(), String> {
        let line = std::str::from_utf8(line)
            .map_err(|error| error.to_string())?
            .trim();
        let Some(payload) = line.strip_prefix("data: ") else {
            return Ok(());
        };
        if payload == "[DONE]" {
            self.done = true;
            return Ok(());
        }
        let event: Event = match serde_json::from_str(payload) {
            Ok(event) => event,
            Err(_) => return Ok(()),
        };
        if let Some(error) = event.error {
            let message = error
                .get("message")
                .and_then(Value::as_str)
                .map(str::to_owned)
                .unwrap_or_else(|| error.to_string());
            self.error = Some(format!("stream failed with server error: {message}"));
            return Ok(());
        }
        if let Some(usage) = event.usage {
            self.usage = Some(usage);
        }
        if let Some(choice) = event.choices.into_iter().next() {
            if choice.finish_reason.is_some() {
                self.finish_reason = choice.finish_reason;
            }
            if let Some(delta) = choice.delta {
                self.delta(delta, elapsed)?;
            }
        }
        Ok(())
    }

    fn delta(&mut self, delta: Delta, elapsed: Duration) -> Result<(), String> {
        let generated = delta.content.as_ref().is_some_and(|text| !text.is_empty())
            || delta
                .reasoning_content
                .as_ref()
                .is_some_and(|text| !text.is_empty())
            || delta
                .tool_calls
                .as_ref()
                .is_some_and(|calls| !calls.is_empty());
        if generated && self.first_generated_sha256.is_none() {
            let encoded = serde_json::to_vec(&delta).map_err(|error| error.to_string())?;
            self.first_generated_sha256 = Some(hex::encode(Sha256::digest(encoded)));
        }
        if generated {
            self.first.get_or_insert(elapsed);
            self.events.push(elapsed);
        }
        if let Some(content) = delta.content {
            self.content.push_str(&content);
        }
        if let Some(reasoning) = delta.reasoning_content {
            self.reasoning.push_str(&reasoning);
        }
        for (fallback, delta) in delta.tool_calls.unwrap_or_default().into_iter().enumerate() {
            let tool = self
                .tools
                .entry(delta.index.unwrap_or(fallback))
                .or_default();
            if delta.kind.is_some() {
                tool.kind = delta.kind;
            }
            if let Some(delta) = delta.function {
                let function = tool.function.get_or_insert_with(Function::default);
                if delta.name.is_some() {
                    function.name = delta.name;
                }
                if let Some(arguments) = delta.arguments {
                    function
                        .arguments
                        .get_or_insert_with(String::new)
                        .push_str(&arguments);
                }
            }
        }
        Ok(())
    }

    pub(in crate::automation) fn finish(
        mut self,
        elapsed: Duration,
        probe: bool,
    ) -> Result<Evidence, String> {
        if !self.pending.is_empty() && !self.done {
            let pending = std::mem::take(&mut self.pending);
            self.line(&pending, elapsed)?;
        }
        if let Some(error) = self.error {
            return Err(error);
        }
        if !self.done {
            return Err("stream ended without terminal [DONE] marker".into());
        }
        let first = match self.first {
            Some(first) => first,
            None if probe => elapsed,
            None => return Err("stream completed without generated content".into()),
        };
        let usage = self.usage.ok_or("missing or invalid prompt/cache usage")?;
        if usage.prompt_tokens == 0
            || usage.prompt_tokens_details.cached_tokens > usage.prompt_tokens
        {
            return Err("missing or invalid prompt/cache usage".into());
        }
        if usage.completion_tokens == 0 && !probe {
            return Err("stream completed without completion-token usage".into());
        }
        let identity = serde_json::json!({"content":self.content,"reasoning_content":self.reasoning,
            "tool_calls":self.tools.into_values().collect::<Vec<_>>()});
        let encoded = serde_json::to_vec(&identity).map_err(|error| error.to_string())?;
        Ok(Evidence {
            ttft_seconds: first.as_secs_f64(),
            elapsed_seconds: elapsed.as_secs_f64(),
            generation_seconds: elapsed.saturating_sub(first).as_secs_f64(),
            decode_inter_token_seconds: self
                .events
                .windows(2)
                .map(|pair| pair[1].saturating_sub(pair[0]).as_secs_f64())
                .collect(),
            completion_tokens: usage.completion_tokens,
            prompt_tokens: usage.prompt_tokens,
            cached_tokens: usage.prompt_tokens_details.cached_tokens,
            cache_pct: 100.0 * number(usage.prompt_tokens_details.cached_tokens)
                / number(usage.prompt_tokens),
            content_events: self.events.len(),
            finish_reason: self.finish_reason,
            content_sha256: hex::encode(Sha256::digest(encoded)),
            first_generated_sha256: self.first_generated_sha256,
        })
    }
}

pub(in crate::automation) fn number(value: u64) -> f64 {
    let upper = u32::try_from(value >> 32).unwrap_or(u32::MAX);
    let lower = u32::try_from(value & u64::from(u32::MAX)).unwrap_or(u32::MAX);
    f64::from(upper) * 4294967296.0 + f64::from(lower)
}

#[cfg(test)]
#[path = "stream_tests.rs"]
mod tests;
