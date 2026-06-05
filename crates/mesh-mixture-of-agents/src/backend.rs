//! Backend abstraction for calling models.
//!
//! The gateway doesn't care whether the model is local HTTP, remote QUIC,
//! or something else entirely — it talks to backends through the
//! [`ModelBackend`] trait. The default [`HttpBackend`] talks to any
//! OpenAI-compatible HTTP endpoint and is suitable for standalone/test
//! use. The mesh host-runtime provides mesh-native backends that dispatch
//! local models via direct HTTP and remote models via QUIC tunnel.

use crate::worker;
use serde_json::{Value, json};
use std::time::Duration;

// ─── Sampling params ─────────────────────────────────────────────────

/// Sampling hyperparameters sent to backend models.
/// Workers get higher temperature for diversity; reducer gets lower for precision.
///
/// `enable_thinking` propagates the caller's reasoning toggle
/// (`reasoning_effort: "none"`, `enable_thinking: false`, etc.) down to
/// each worker so reasoning models can skip their `<think>` phase when
/// MoA is mediating the call. `None` means "don't override the model's
/// default" — callers who haven't specified a preference get the same
/// behavior as before this field existed.
#[derive(Debug, Clone, Copy)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub enable_thinking: Option<bool>,
}

impl SamplingParams {
    /// High-diversity settings for MoA workers — encourages each model
    /// to explore different parts of the solution space.
    pub fn worker() -> Self {
        Self {
            temperature: 0.8,
            top_p: 0.95,
            enable_thinking: None,
        }
    }

    /// Low-variance settings for the reducer — precise synthesis.
    pub fn reducer() -> Self {
        Self {
            temperature: 0.3,
            top_p: 0.9,
            enable_thinking: None,
        }
    }

    /// Returns a copy with `enable_thinking` set. Convenience for the
    /// MoA gateway, which propagates the caller's `reasoning_effort` /
    /// `enable_thinking` knob to every worker (and the reducer).
    pub fn with_thinking(mut self, enable: Option<bool>) -> Self {
        self.enable_thinking = enable;
        self
    }
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self::reducer()
    }
}

// ─── Backend trait ───────────────────────────────────────────────────

/// Abstraction for calling a model.  The gateway doesn't care whether
/// the model is local HTTP, remote QUIC, or something else entirely.
#[async_trait::async_trait]
pub trait ModelBackend: Send + Sync + 'static {
    /// Call the model with the given messages (and optionally tools).
    /// Returns the full JSON response body from the model.
    async fn chat_completion(
        &self,
        model: &str,
        messages: &[Value],
        tools: Option<&Value>,
        max_tokens: u32,
        timeout: Duration,
        sampling: SamplingParams,
    ) -> Result<Value, String>;
}

/// Default HTTP backend — calls any OpenAI-compatible endpoint.
pub struct HttpBackend {
    pub base_url: String,
    http: reqwest::Client,
}

impl HttpBackend {
    pub fn new(base_url: String) -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .unwrap_or_default();
        Self { base_url, http }
    }
}

#[async_trait::async_trait]
impl ModelBackend for HttpBackend {
    async fn chat_completion(
        &self,
        model: &str,
        messages: &[Value],
        tools: Option<&Value>,
        max_tokens: u32,
        timeout: Duration,
        sampling: SamplingParams,
    ) -> Result<Value, String> {
        let url = format!("{}/chat/completions", self.base_url);
        let mut body = json!({
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "stream": false,
        });
        if let Some(tools) = tools {
            body.as_object_mut()
                .unwrap()
                .insert("tools".to_string(), tools.clone());
        }
        apply_enable_thinking(&mut body, sampling.enable_thinking);

        let resp = self
            .http
            .post(&url)
            .json(&body)
            .timeout(timeout)
            .send()
            .await
            .map_err(|e| format!("request failed: {e}"))?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(format!(
                "HTTP {status}: {}",
                crate::worker::truncate_chars(&text, 200)
            ));
        }

        resp.json::<Value>()
            .await
            .map_err(|e| format!("response parse: {e}"))
    }
}

// ─── Model entry ─────────────────────────────────────────────────────

/// A model available for MoA fan-out.
#[derive(Clone)]
pub struct ModelEntry {
    /// Model name (as used in the API).
    pub name: String,
    /// Index into the backends vec.  Multiple models can share a backend
    /// (e.g. all models behind the same proxy) or each have their own.
    pub backend_index: usize,
}

// ─── Backend call + text extraction ──────────────────────────────────

/// Call a backend and extract the assistant text from the response.
/// Retries once on HTTP 429 (rate limit) after the server's `retry-after`
/// delay (default 1s).
pub(crate) async fn call_backend(
    backend: &dyn ModelBackend,
    model: &str,
    messages: &[Value],
    tools: Option<&Value>,
    max_tokens: u32,
    timeout: Duration,
    sampling: SamplingParams,
) -> Result<String, String> {
    match backend
        .chat_completion(model, messages, tools, max_tokens, timeout, sampling)
        .await
    {
        Ok(resp) => extract_text_from_response(&resp),
        Err(e) if e.contains("429") => {
            // Parse retry-after from error message if present, default 1s
            let delay = parse_retry_after(&e).unwrap_or(1);
            tracing::info!("moa: 429 from {model}, retrying after {delay}s");
            tokio::time::sleep(Duration::from_secs(delay)).await;
            let resp = backend
                .chat_completion(model, messages, tools, max_tokens, timeout, sampling)
                .await?;
            extract_text_from_response(&resp)
        }
        Err(e) => Err(e),
    }
}

/// Inject `chat_template_kwargs.enable_thinking` and (for OpenAI-compat
/// servers) `reasoning_effort` into a chat-completion request body when
/// the caller has asked us to override the model's default thinking
/// behavior. Canonical form recognised by Qwen3 / DeepSeek-R1-Distill /
/// GLM chat templates inside llama.cpp.
///
/// `None` means "don't touch" — leaves the body unchanged so direct
/// callers (non-MoA paths) and tests don't see spurious fields.
pub fn apply_enable_thinking(body: &mut Value, enable: Option<bool>) {
    let Some(enable) = enable else {
        return;
    };
    let Some(obj) = body.as_object_mut() else {
        return;
    };

    // chat_template_kwargs.enable_thinking is the canonical knob the
    // llama.cpp templates read. Merge into any existing object. If a
    // caller passed a non-object for chat_template_kwargs (string,
    // array, etc.), normalize it to {} so the override still applies —
    // silently dropping the flag because the field was malformed is
    // worse than overwriting a bogus shape.
    let kwargs = obj
        .entry("chat_template_kwargs".to_string())
        .or_insert_with(|| json!({}));
    if !kwargs.is_object() {
        *kwargs = json!({});
    }
    if let Some(kwargs_obj) = kwargs.as_object_mut() {
        kwargs_obj.insert("enable_thinking".to_string(), json!(enable));
    }

    // reasoning_effort is the OpenAI-surface analogue. If the caller
    // wants thinking off, declare it both ways so backends that consume
    // either knob honour the request. If thinking is being explicitly
    // turned ON, leave reasoning_effort alone — we don't want to pick a
    // specific effort level on the caller's behalf.
    if !enable {
        obj.insert("reasoning_effort".to_string(), json!("none"));
    }
}

/// Extract retry-after seconds from an error message containing "retry-after: N".
///
/// The earlier shape called `err.to_lowercase()` (Unicode-aware) and then
/// sliced the *original* `err` using the byte offset from the lowercased
/// string. For non-ASCII inputs, `to_lowercase()` can change UTF-8 byte
/// length, so the offset could land mid-codepoint and either panic or
/// produce garbage. Use `to_ascii_lowercase()` which is a 1:1 byte mapping
/// so the offset stays valid in the original string.
fn parse_retry_after(err: &str) -> Option<u64> {
    let lower = err.to_ascii_lowercase();
    let i = lower.find("retry-after:")?;
    let tail = err.get(i + "retry-after:".len()..)?;
    tail.split_whitespace()
        .next()
        .and_then(|s| s.trim().parse::<u64>().ok())
}

/// Extract assistant text from a chat completion response body.
///
/// The response comes from an untrusted backend (peer node, local skippy,
/// or in tests a mock). Direct indexing like `resp["choices"][0]["message"]`
/// returns `Value::Null` on missing fields, which then silently produces
/// an empty string — or worse, panics in `.unwrap()` chains. Use
/// `.pointer()` so a malformed response surfaces as a structured `Err`
/// rather than a hidden empty answer.
fn extract_text_from_response(resp: &Value) -> Result<String, String> {
    let message = resp
        .pointer("/choices/0/message")
        .ok_or_else(|| "malformed response: missing choices[0].message".to_string())?;

    // Native tool_calls → KV format for normalizer
    let first_tool_call = message
        .get("tool_calls")
        .and_then(|tc| tc.as_array())
        .and_then(|tool_calls| tool_calls.first());
    if let Some(tc) = first_tool_call {
        let name = tc
            .pointer("/function/name")
            .and_then(|n| n.as_str())
            .unwrap_or("unknown");
        let args = tc
            .pointer("/function/arguments")
            .and_then(|a| a.as_str())
            .unwrap_or("{}");
        return Ok(format!(
            "kind: tool_proposal\ntool: {name}\narguments: {args}\nconfidence: 0.9\npayload: calling {name}",
        ));
    }

    let content = message
        .get("content")
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string();

    let stripped = worker::strip_thinking(&content);
    if !stripped.is_empty() {
        return Ok(stripped);
    }

    let thinking = worker::extract_thinking(&content);
    if !thinking.is_empty() {
        return Ok(thinking);
    }

    let reasoning = message
        .get("reasoning")
        .and_then(|r| r.as_str())
        .unwrap_or("");
    if !reasoning.is_empty() {
        return Ok(reasoning.to_string());
    }

    Err("empty response".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_retry_after_from_header() {
        let err = "HTTP 429: retry-after: 2\r\ncontent-type: application/json";
        assert_eq!(parse_retry_after(err), Some(2));
    }

    #[test]
    fn parse_retry_after_missing() {
        let err = "HTTP 429: Too Many Requests";
        assert_eq!(parse_retry_after(err), None);
    }

    #[test]
    fn parse_retry_after_case_insensitive() {
        let err = "Retry-After: 5";
        assert_eq!(parse_retry_after(err), Some(5));
    }

    #[test]
    fn parse_retry_after_handles_non_ascii_prefix_without_panic() {
        // Regression for PR #566 review: the prior shape used
        // `to_lowercase()` (Unicode-aware, can change byte length) and
        // sliced the original `err` with that offset. For inputs with
        // non-ASCII before the marker, the slice could land mid-codepoint
        // and panic. With `to_ascii_lowercase()` the offset is byte-stable.
        let err = "über-error: retry-after: 7";
        assert_eq!(parse_retry_after(err), Some(7));
    }

    #[test]
    fn extract_text_returns_err_on_missing_choices() {
        // Regression for PR #566 review: direct-indexing
        // `resp["choices"][0]["message"]` returns `Value::Null` on
        // malformed responses; downstream `.unwrap()` chains then panicked.
        // Now a structured error surfaces instead.
        let resp: Value = serde_json::json!({"id": "x"});
        let err = extract_text_from_response(&resp).unwrap_err();
        assert!(err.contains("missing choices"), "unexpected error: {err}");
    }

    #[test]
    fn extract_text_returns_err_on_empty_choices() {
        let resp: Value = serde_json::json!({"choices": []});
        let err = extract_text_from_response(&resp).unwrap_err();
        assert!(err.contains("missing choices"));
    }

    #[test]
    fn worker_sampling_high_diversity() {
        let s = SamplingParams::worker();
        assert!(s.temperature > 0.5, "workers need high temp for diversity");
        assert!(s.top_p > 0.9, "workers need high top_p for diversity");
    }

    #[test]
    fn reducer_sampling_low_variance() {
        let s = SamplingParams::reducer();
        assert!(s.temperature <= 0.4, "reducer needs low temp for precision");
        assert!(s.top_p <= 0.95, "reducer needs bounded top_p");
    }

    #[test]
    fn default_sampling_is_reducer() {
        let d = SamplingParams::default();
        let r = SamplingParams::reducer();
        assert!((d.temperature - r.temperature).abs() < f32::EPSILON);
        assert!((d.top_p - r.top_p).abs() < f32::EPSILON);
    }

    // ── enable_thinking propagation ────────────────────────────────────────────
    //
    // MoA mediates between the user's request and N worker models. When
    // the caller asks for no reasoning (`reasoning_effort: "none"`,
    // `enable_thinking: false`, etc.), MoA needs to forward that to every
    // worker (and the reducer) so reasoning models skip their `<think>`
    // phase. This is the wire-shape contract.

    #[test]
    fn sampling_default_thinking_is_none() {
        // Default (no override) keeps `enable_thinking = None` so callers
        // without a preference don't get any spurious `chat_template_kwargs`
        // baked into outbound requests.
        assert_eq!(SamplingParams::worker().enable_thinking, None);
        assert_eq!(SamplingParams::reducer().enable_thinking, None);
    }

    #[test]
    fn sampling_with_thinking_overrides_only_that_field() {
        let s = SamplingParams::worker().with_thinking(Some(false));
        assert_eq!(s.enable_thinking, Some(false));
        assert!((s.temperature - 0.8).abs() < f32::EPSILON);
        assert!((s.top_p - 0.95).abs() < f32::EPSILON);
    }

    #[test]
    fn apply_enable_thinking_none_leaves_body_untouched() {
        let mut body = json!({"model": "qwen", "messages": []});
        let before = body.clone();
        apply_enable_thinking(&mut body, None);
        assert_eq!(body, before, "None override must not modify body");
    }

    #[test]
    fn apply_enable_thinking_false_injects_kwargs_and_effort() {
        let mut body = json!({"model": "qwen", "messages": []});
        apply_enable_thinking(&mut body, Some(false));
        assert_eq!(
            body["chat_template_kwargs"]["enable_thinking"],
            json!(false)
        );
        assert_eq!(body["reasoning_effort"], json!("none"));
    }

    #[test]
    fn apply_enable_thinking_true_injects_kwargs_only() {
        let mut body = json!({"model": "qwen", "messages": []});
        apply_enable_thinking(&mut body, Some(true));
        assert_eq!(body["chat_template_kwargs"]["enable_thinking"], json!(true));
        // We don't pick a specific reasoning_effort on behalf of an
        // "on" caller — they should set their own.
        assert!(body.get("reasoning_effort").is_none());
    }

    #[test]
    fn apply_enable_thinking_merges_existing_kwargs() {
        // The caller may have already set chat_template_kwargs.foo — we
        // must merge, not clobber.
        let mut body = json!({
            "model": "qwen",
            "chat_template_kwargs": {"foo": "bar"}
        });
        apply_enable_thinking(&mut body, Some(false));
        assert_eq!(body["chat_template_kwargs"]["foo"], json!("bar"));
        assert_eq!(
            body["chat_template_kwargs"]["enable_thinking"],
            json!(false)
        );
    }

    #[test]
    fn apply_enable_thinking_normalizes_non_object_kwargs() {
        // If the caller passed a non-object for chat_template_kwargs
        // (here: a string), the override must still apply rather than
        // being silently dropped. We normalize the bogus shape to an
        // empty object before injecting `enable_thinking`.
        for bogus in [
            json!("some-string"),
            json!(123),
            json!([1, 2, 3]),
            json!(null),
        ] {
            let mut body = json!({
                "model": "qwen",
                "chat_template_kwargs": bogus.clone(),
            });
            apply_enable_thinking(&mut body, Some(false));
            assert_eq!(
                body["chat_template_kwargs"]["enable_thinking"],
                json!(false),
                "enable_thinking must be injected even when input kwargs was {bogus:?}",
            );
            assert_eq!(body["reasoning_effort"], json!("none"));
        }
    }

    #[test]
    fn apply_enable_thinking_no_op_on_non_object_body() {
        // Defensive: if someone hands us a non-object Value, we shouldn't
        // panic. Real bodies are always objects so this is a paranoia test.
        let mut body = json!(42);
        apply_enable_thinking(&mut body, Some(false));
        assert_eq!(body, json!(42));
    }
}
