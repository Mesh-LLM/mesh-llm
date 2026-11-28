//! Resuming a committed chat-completion stream on another target (#2998).
//!
//! Off unless `MESH_LLM_STREAM_RESUME=1`: a resumed answer is stitched from two
//! hosts, which a caller should opt into while #2998 settles the default.
//!
//! The state lives in a task-local the route loop scopes around its attempts,
//! as `private_scope` and `probe` do for theirs: the relay records what the
//! caller received and a later attempt runs as the continuation of that
//! stream, without a new parameter through every dispatch path.

use serde_json::Value;
use std::cell::RefCell;

tokio::task_local! {
    static STREAM_RESUME: RefCell<StreamResume>;
}

#[derive(Debug, Default)]
pub(in crate::network::openai) struct StreamResume {
    /// Set by the route loop before each attempt: another target is left.
    allow: bool,
    /// Filled by the relay when a committed stream broke and it held the
    /// caller's stream open instead of ending it.
    delivered: Option<DeliveredStream>,
    /// Set by the route loop before the attempt that continues the stream.
    continuation: Option<StreamContinuation>,
    /// Whether this request already resumed once.
    resumed: bool,
}

/// What the caller received before the stream broke.
#[derive(Debug, Clone, Default)]
pub(in crate::network::openai) struct DeliveredStream {
    pub(in crate::network::openai) content: String,
    pub(in crate::network::openai) content_chunks: u32,
    pub(in crate::network::openai) id: Option<Value>,
    pub(in crate::network::openai) created: Option<Value>,
    pub(in crate::network::openai) model: Option<Value>,
}

/// How a resumed attempt rewrites its stream into the caller's.
#[derive(Debug, Clone)]
pub(in crate::network::openai) struct StreamContinuation {
    pub(in crate::network::openai) id: Option<Value>,
    pub(in crate::network::openai) created: Option<Value>,
    pub(in crate::network::openai) model: Option<Value>,
    pub(in crate::network::openai) prior_completion_chunks: u32,
}

/// Run `future` with resume state available to the route loop and relay.
pub(in crate::network::openai) async fn with_stream_resume<F: std::future::Future>(
    future: F,
) -> F::Output {
    STREAM_RESUME
        .scope(RefCell::new(StreamResume::default()), future)
        .await
}

fn with<T>(f: impl FnOnce(&mut StreamResume) -> T) -> Option<T> {
    STREAM_RESUME
        .try_with(|cell| f(&mut cell.borrow_mut()))
        .ok()
}

const ENABLE_ENV: &str = "MESH_LLM_STREAM_RESUME";

/// Whether this node resumes broken streams at all (`MESH_LLM_STREAM_RESUME=1`).
fn enabled() -> bool {
    enabled_from(std::env::var(ENABLE_ENV).ok().as_deref())
}

fn enabled_from(value: Option<&str>) -> bool {
    matches!(
        value.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("1" | "true" | "yes" | "on")
    )
}

pub(in crate::network::openai) fn set_allowed(allowed: bool) {
    let allowed = allowed && enabled();
    with(|state| state.allow = allowed && !state.resumed);
}

pub(in crate::network::openai) fn allowed() -> bool {
    with(|state| state.allow).unwrap_or(false)
}

/// How long a committed stream that already produced output
/// may stay silent before it counts as lost and is resumed elsewhere.
const IDLE_LIMIT_ENV: &str = "MESH_LLM_STREAM_RESUME_IDLE_SECS";
const DEFAULT_IDLE_LIMIT: std::time::Duration = std::time::Duration::from_secs(10);

/// The idle limit, or `None` when it is turned off (`0`).
pub(in crate::network::openai) fn idle_limit() -> Option<std::time::Duration> {
    idle_limit_from(std::env::var(IDLE_LIMIT_ENV).ok().as_deref())
}

fn idle_limit_from(value: Option<&str>) -> Option<std::time::Duration> {
    match value.map(str::trim) {
        None | Some("") => Some(DEFAULT_IDLE_LIMIT),
        Some(text) => match text.parse::<u64>() {
            Ok(0) => None,
            Ok(secs) => Some(std::time::Duration::from_secs(secs)),
            Err(_) => Some(DEFAULT_IDLE_LIMIT),
        },
    }
}

pub(in crate::network::openai) fn record_delivered(delivered: DeliveredStream) {
    with(|state| state.delivered = Some(delivered));
}

pub(in crate::network::openai) fn take_delivered() -> Option<DeliveredStream> {
    with(|state| state.delivered.take()).flatten()
}

pub(in crate::network::openai) fn begin_continuation(continuation: StreamContinuation) {
    with(|state| {
        state.continuation = Some(continuation);
        state.resumed = true;
    });
}

/// The continuation the next relayed attempt must apply, if any. It stays
/// set until a continuation attempt commits, so a target that fails before
/// committing passes it on to the next one.
pub(in crate::network::openai) fn continuation() -> Option<StreamContinuation> {
    with(|state| state.continuation.clone()).flatten()
}

pub(in crate::network::openai) fn finish_continuation() {
    with(|state| state.continuation = None);
}

/// Rewrite one client-facing chunk of the continuing stream into the
/// caller's: drop the opening role frame, keep the first stream's identity,
/// and count the first attempt's output in `usage`. `None` drops the frame.
pub(in crate::network::openai) fn rewrite_continuation_chunk(
    data: &str,
    continuation: &StreamContinuation,
) -> Option<String> {
    let Ok(mut value) = serde_json::from_str::<Value>(data) else {
        return Some(data.to_string());
    };
    if let Some(choice) = value
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
    {
        let delta = choice.get("delta");
        let only_role = delta.is_some_and(|delta| {
            delta.get("role").is_some()
                && delta
                    .get("content")
                    .and_then(Value::as_str)
                    .is_none_or(str::is_empty)
                && delta.get("tool_calls").is_none()
                && delta.get("reasoning_content").is_none()
        }) && choice.get("finish_reason").is_none_or(Value::is_null);
        if only_role {
            return None;
        }
    }
    if let Some(object) = value.as_object_mut() {
        for (key, replacement) in [
            ("id", &continuation.id),
            ("created", &continuation.created),
            ("model", &continuation.model),
        ] {
            if let Some(replacement) = replacement {
                object.insert(key.to_string(), replacement.clone());
            }
        }
        if let Some(usage) = object.get_mut("usage").and_then(Value::as_object_mut) {
            let prior = u64::from(continuation.prior_completion_chunks);
            for key in ["completion_tokens", "total_tokens"] {
                if let Some(count) = usage.get(key).and_then(Value::as_u64) {
                    usage.insert(key.to_string(), Value::from(count + prior));
                }
            }
        }
    }
    Some(value.to_string())
}

/// The request to send to the next target: the original one with what the
/// caller received as `prefill_assistant`, and an output cap reduced by what
/// was already streamed. `None` when the request can't be rewritten.
pub(in crate::network::openai) fn resumed_request(
    raw: &[u8],
    delivered: &DeliveredStream,
) -> Option<Vec<u8>> {
    let header_end = raw.windows(4).position(|window| window == b"\r\n\r\n")? + 4;
    let head = std::str::from_utf8(&raw[..header_end]).ok()?;
    if head
        .to_ascii_lowercase()
        .contains("transfer-encoding: chunked")
    {
        return None;
    }
    let mut body: Value = serde_json::from_slice(&raw[header_end..]).ok()?;
    let object = body.as_object_mut()?;
    object.insert(
        "prefill_assistant".to_string(),
        Value::String(delivered.content.clone()),
    );
    for key in ["max_tokens", "max_completion_tokens"] {
        if let Some(cap) = object.get(key).and_then(Value::as_u64) {
            let left = cap
                .saturating_sub(u64::from(delivered.content_chunks))
                .max(1);
            object.insert(key.to_string(), Value::from(left));
        }
    }
    let body = serde_json::to_vec(&body).ok()?;
    let mut rewritten = String::new();
    for line in head.trim_end_matches("\r\n").split("\r\n") {
        if line.to_ascii_lowercase().starts_with("content-length:") {
            rewritten.push_str(&format!("Content-Length: {}\r\n", body.len()));
        } else {
            rewritten.push_str(line);
            rewritten.push_str("\r\n");
        }
    }
    rewritten.push_str("\r\n");
    let mut out = rewritten.into_bytes();
    out.extend_from_slice(&body);
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn continuation() -> StreamContinuation {
        StreamContinuation {
            id: Some(json!("chatcmpl-first")),
            created: Some(json!(100)),
            model: Some(json!("m")),
            prior_completion_chunks: 12,
        }
    }

    #[test]
    fn the_continuation_drops_its_role_frame_and_takes_the_first_identity() {
        let role = json!({"id":"x","created":2,"model":"m","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]});
        assert_eq!(
            rewrite_continuation_chunk(&role.to_string(), &continuation()),
            None
        );
        let text = json!({"id":"x","created":2,"model":"m","choices":[{"index":0,"delta":{"content":"11"},"finish_reason":null}]});
        let rewritten: Value = serde_json::from_str(
            &rewrite_continuation_chunk(&text.to_string(), &continuation()).unwrap(),
        )
        .unwrap();
        assert_eq!(rewritten["id"], "chatcmpl-first");
        assert_eq!(rewritten["created"], 100);
        assert_eq!(rewritten["choices"][0]["delta"]["content"], "11");
    }

    #[test]
    fn resuming_is_off_unless_the_node_opts_in() {
        assert!(!enabled_from(None));
        assert!(!enabled_from(Some("")));
        assert!(!enabled_from(Some("0")));
        assert!(!enabled_from(Some("off")));
        assert!(enabled_from(Some("1")));
        assert!(enabled_from(Some(" TRUE ")));
    }

    #[test]
    fn the_idle_limit_defaults_to_ten_seconds_and_zero_turns_it_off() {
        use std::time::Duration;
        assert_eq!(idle_limit_from(None), Some(Duration::from_secs(10)));
        assert_eq!(idle_limit_from(Some("0")), None);
        assert_eq!(idle_limit_from(Some(" 4 ")), Some(Duration::from_secs(4)));
        assert_eq!(idle_limit_from(Some("soon")), Some(Duration::from_secs(10)));
    }

    #[test]
    fn usage_counts_both_attempts() {
        let usage = json!({"id":"x","choices":[],"usage":{"prompt_tokens":70,"completion_tokens":30,"total_tokens":100}});
        let rewritten: Value = serde_json::from_str(
            &rewrite_continuation_chunk(&usage.to_string(), &continuation()).unwrap(),
        )
        .unwrap();
        assert_eq!(rewritten["usage"]["completion_tokens"], 42);
        assert_eq!(rewritten["usage"]["total_tokens"], 112);
    }

    #[test]
    fn the_resumed_request_carries_the_prefill_and_a_reduced_cap() {
        let body = json!({"model":"m","stream":true,"max_tokens":40,"messages":[{"role":"user","content":"Count."}]}).to_string();
        let raw = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        let delivered = DeliveredStream {
            content: "1\n2\n".into(),
            content_chunks: 4,
            ..DeliveredStream::default()
        };
        let resumed = resumed_request(raw.as_bytes(), &delivered).unwrap();
        let text = String::from_utf8(resumed).unwrap();
        let (head, body) = text.split_once("\r\n\r\n").unwrap();
        assert!(head.contains(&format!("Content-Length: {}", body.len())));
        let body: Value = serde_json::from_str(body).unwrap();
        assert_eq!(body["prefill_assistant"], "1\n2\n");
        assert_eq!(body["max_tokens"], 36);
    }
}
