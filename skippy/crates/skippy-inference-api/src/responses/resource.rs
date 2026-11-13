//! The Open Responses `ResponseResource` object, its output items, and usage.
//!
//! A response echoes the configuration it ran with. [`ResponseEnvelope`]
//! captures that echo from the client's request before it is translated onto
//! the shared chat backend, so JSON responses and every response-bearing
//! stream event report the same, schema-complete settings.

use serde_json::{Map, Value, json};

use crate::common::Usage;

/// Fields every `ResponseResource` must carry, with the value reported when
/// the request did not set them. Values the backend does not report (sampling
/// defaults) use the Open Responses defaults, which is what the request means
/// when it omits them.
type DefaultedField = (&'static str, fn() -> Value);

const DEFAULTED_FIELDS: &[DefaultedField] = &[
    ("background", || Value::Bool(false)),
    ("frequency_penalty", || json!(0.0)),
    ("instructions", || Value::Null),
    ("max_output_tokens", || Value::Null),
    ("max_tool_calls", || Value::Null),
    ("metadata", || Value::Object(Map::new())),
    ("parallel_tool_calls", || Value::Bool(true)),
    ("presence_penalty", || json!(0.0)),
    ("previous_response_id", || Value::Null),
    ("prompt_cache_key", || Value::Null),
    ("reasoning", || json!({"effort": null, "summary": null})),
    ("safety_identifier", || Value::Null),
    ("service_tier", || Value::String("default".into())),
    ("store", || Value::Bool(true)),
    ("temperature", || json!(1.0)),
    ("text", || json!({"format": {"type": "text"}})),
    ("tool_choice", || Value::String("auto".into())),
    ("tools", || Value::Array(Vec::new())),
    ("top_logprobs", || json!(0)),
    ("top_p", || json!(1.0)),
    ("truncation", || Value::String("disabled".into())),
];

/// The request configuration a response reports back, captured from the
/// client's Responses request before translation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResponseEnvelope {
    fields: Map<String, Value>,
}

impl Default for ResponseEnvelope {
    fn default() -> Self {
        Self::from_request(&Map::new())
    }
}

impl ResponseEnvelope {
    /// Capture the echoed configuration from a raw Responses request body.
    pub fn from_request(request: &Map<String, Value>) -> Self {
        let mut fields = Map::new();
        for (key, default) in DEFAULTED_FIELDS {
            let value = request
                .get(*key)
                .filter(|value| !value.is_null() || nullable_echo(key))
                .cloned()
                .unwrap_or_else(default);
            fields.insert((*key).to_string(), value);
        }
        // The service runs one tier; report it rather than the request's.
        fields.insert("service_tier".into(), Value::String("default".into()));
        // Instructions are only a string in the schema; drop other shapes.
        if !fields["instructions"].is_string() {
            fields.insert("instructions".into(), Value::Null);
        }
        if fields["metadata"].is_null() {
            fields.insert("metadata".into(), Value::Object(Map::new()));
        }
        if let Some(tools) = fields.get_mut("tools").and_then(Value::as_array_mut) {
            for tool in tools {
                if tool.get("type").and_then(Value::as_str) == Some("function")
                    && let Some(tool) = tool.as_object_mut()
                {
                    for field in ["description", "parameters", "strict"] {
                        tool.entry(field).or_insert(Value::Null);
                    }
                }
            }
        }
        // Request `allowed_tools.mode` is optional, while the resource form
        // requires the effective mode. Echo the specified default explicitly.
        if let Some(choice) = fields
            .get_mut("tool_choice")
            .and_then(Value::as_object_mut)
            .filter(|choice| choice.get("type").and_then(Value::as_str) == Some("allowed_tools"))
        {
            choice
                .entry("mode")
                .or_insert_with(|| Value::String("auto".into()));
        }
        Self { fields }
    }

    /// Whether the client asked for this response to be stored.
    pub fn store(&self) -> bool {
        self.fields
            .get("store")
            .and_then(Value::as_bool)
            .unwrap_or(true)
    }

    /// Record that the response was, or was not, actually stored.
    pub fn set_store(&mut self, stored: bool) {
        self.fields.insert("store".into(), Value::Bool(stored));
    }

    /// A complete `ResponseResource` for this envelope.
    pub fn response(&self, snapshot: ResponseSnapshot<'_>) -> Value {
        let mut response = Map::new();
        response.insert("id".into(), Value::String(snapshot.id.to_string()));
        response.insert("object".into(), Value::String("response".into()));
        response.insert("created_at".into(), Value::from(snapshot.created_at));
        response.insert(
            "completed_at".into(),
            match snapshot.status {
                ResponseStatus::Completed => Value::from(crate::responses::now_unix_secs_i64()),
                _ => Value::Null,
            },
        );
        response.insert(
            "status".into(),
            Value::String(snapshot.status.as_str().into()),
        );
        response.insert(
            "incomplete_details".into(),
            match snapshot.status {
                ResponseStatus::Incomplete(reason) => json!({"reason": reason}),
                _ => Value::Null,
            },
        );
        response.insert("model".into(), Value::String(snapshot.model.to_string()));
        let output = match snapshot.status {
            ResponseStatus::Failed => Vec::new(),
            _ => snapshot.output.to_vec(),
        };
        let output_text = output_text(&output);
        response.insert("output".into(), Value::Array(output));
        response.insert(
            "error".into(),
            snapshot
                .error
                .map(|error| json!({"code": error.code, "message": error.message}))
                .unwrap_or(Value::Null),
        );
        response.insert(
            "usage".into(),
            snapshot
                .usage
                .map(ResponsesUsage::to_value)
                .unwrap_or(Value::Null),
        );
        for (key, value) in &self.fields {
            response.insert(key.clone(), value.clone());
        }
        // Convenience text that OpenAI SDKs read; not part of the schema.
        response.insert("output_text".into(), Value::String(output_text));
        Value::Object(response)
    }
}

/// Request fields whose explicit `null` is itself the echoed value.
fn nullable_echo(key: &str) -> bool {
    matches!(
        key,
        "instructions"
            | "max_output_tokens"
            | "max_tool_calls"
            | "previous_response_id"
            | "prompt_cache_key"
            | "safety_identifier"
    )
}

/// Response lifecycle status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponseStatus {
    InProgress,
    Completed,
    /// Ended early; carries the `incomplete_details.reason`.
    Incomplete(&'static str),
    Failed,
}

impl ResponseStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::InProgress => "in_progress",
            Self::Completed => "completed",
            Self::Incomplete(_) => "incomplete",
            Self::Failed => "failed",
        }
    }

    /// Map an upstream chat `finish_reason` onto the response status.
    pub fn from_finish_reason(finish_reason: Option<&str>) -> Self {
        match finish_reason {
            Some("length") => Self::Incomplete("max_output_tokens"),
            Some("content_filter") => Self::Incomplete("content_filter"),
            _ => Self::Completed,
        }
    }

    /// The status of an item that ended when the response did.
    pub fn item_status(self) -> &'static str {
        match self {
            Self::Incomplete(_) => "incomplete",
            _ => "completed",
        }
    }
}

/// The `error` object of a failed response.
#[derive(Debug, Clone, Copy)]
pub struct ResponseError<'a> {
    pub code: &'a str,
    pub message: &'a str,
}

/// Everything about a response that is not its echoed configuration.
#[derive(Debug, Clone, Copy)]
pub struct ResponseSnapshot<'a> {
    pub id: &'a str,
    pub created_at: i64,
    pub model: &'a str,
    pub status: ResponseStatus,
    pub output: &'a [Value],
    pub usage: Option<ResponsesUsage>,
    pub error: Option<ResponseError<'a>>,
}

/// Token usage in the Open Responses shape. Every count is an integer; a
/// detail the backend did not report is zero.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ResponsesUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
    pub cached_tokens: u64,
    pub reasoning_tokens: u64,
}

impl ResponsesUsage {
    pub fn to_value(self) -> Value {
        json!({
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "input_tokens_details": {"cached_tokens": self.cached_tokens},
            "output_tokens_details": {"reasoning_tokens": self.reasoning_tokens},
        })
    }

    /// Read a Chat Completions `usage` object.
    pub fn from_chat_value(usage: &Value) -> Option<Self> {
        let count = |value: Option<&Value>| value.and_then(Value::as_u64);
        let input_tokens = count(usage.get("prompt_tokens"));
        let output_tokens = count(usage.get("completion_tokens"));
        if input_tokens.is_none() && output_tokens.is_none() {
            return None;
        }
        let input_tokens = input_tokens.unwrap_or(0);
        let output_tokens = output_tokens.unwrap_or(0);
        Some(Self {
            input_tokens,
            output_tokens,
            total_tokens: count(usage.get("total_tokens"))
                .unwrap_or(input_tokens.saturating_add(output_tokens)),
            cached_tokens: count(usage.pointer("/prompt_tokens_details/cached_tokens"))
                .unwrap_or(0),
            reasoning_tokens: count(usage.pointer("/completion_tokens_details/reasoning_tokens"))
                .unwrap_or(0),
        })
    }

    pub fn from_usage(usage: &Usage) -> Self {
        Self {
            input_tokens: usage.prompt_tokens.into(),
            output_tokens: usage.completion_tokens.into(),
            total_tokens: usage.total_tokens.into(),
            cached_tokens: usage
                .prompt_tokens_details
                .as_ref()
                .map(|details| details.cached_tokens.into())
                .unwrap_or(0),
            reasoning_tokens: 0,
        }
    }
}

/// Opaque, unique identifiers for responses and their items.
pub fn new_response_id() -> String {
    prefixed_id("resp")
}

pub fn new_item_id(prefix: &str) -> String {
    prefixed_id(prefix)
}

fn prefixed_id(prefix: &str) -> String {
    format!("{prefix}_{}", uuid::Uuid::new_v4().simple())
}

/// An assistant `output_text` content part.
pub fn output_text_part(text: &str, logprobs: Option<Value>) -> Value {
    let mut part = json!({"type": "output_text", "text": text, "annotations": []});
    if let Some(logprobs) = logprobs.filter(|logprobs| !logprobs.is_null()) {
        part["logprobs"] = responses_logprobs(logprobs);
    }
    part
}

/// Chat logprobs arrive as `{"content": [...]}`; Responses carries the list.
pub(super) fn responses_logprobs(logprobs: Value) -> Value {
    match logprobs {
        Value::Object(mut object) => object.remove("content").unwrap_or(Value::Object(object)),
        other => other,
    }
}

pub fn message_item(
    id: &str,
    status: &str,
    text: &str,
    logprobs: Option<Value>,
    phase: Option<&str>,
) -> Value {
    let mut item = json!({
        "id": id,
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": [output_text_part(text, logprobs)],
    });
    if let Some(phase @ ("commentary" | "final_answer")) = phase {
        item["phase"] = Value::String(phase.to_string());
    }
    item
}

pub fn reasoning_item(id: &str, text: &str) -> Value {
    json!({
        "id": id,
        "type": "reasoning",
        "summary": [],
        "content": [{"type": "reasoning_text", "text": text}],
    })
}

pub fn function_call_item(
    id: &str,
    status: &str,
    call_id: &str,
    name: &str,
    arguments: &str,
) -> Value {
    json!({
        "id": id,
        "type": "function_call",
        "status": status,
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
    })
}

/// The concatenated assistant text of an output list.
pub fn output_text(output: &[Value]) -> String {
    output
        .iter()
        .filter(|item| item.get("type").and_then(Value::as_str) == Some("message"))
        .filter_map(|item| item.get("content").and_then(Value::as_array))
        .flatten()
        .filter(|part| part.get("type").and_then(Value::as_str) == Some("output_text"))
        .filter_map(|part| part.get("text").and_then(Value::as_str))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot<'a>(output: &'a [Value], status: ResponseStatus) -> ResponseSnapshot<'a> {
        ResponseSnapshot {
            id: "resp_1",
            created_at: 7,
            model: "m",
            status,
            output,
            usage: Some(ResponsesUsage::default()),
            error: None,
        }
    }

    #[test]
    fn envelope_reports_every_required_field_with_schema_types() {
        let response =
            ResponseEnvelope::default().response(snapshot(&[], ResponseStatus::Completed));
        for key in [
            "id",
            "object",
            "created_at",
            "completed_at",
            "status",
            "incomplete_details",
            "model",
            "previous_response_id",
            "instructions",
            "output",
            "error",
            "tools",
            "tool_choice",
            "truncation",
            "parallel_tool_calls",
            "text",
            "top_p",
            "presence_penalty",
            "frequency_penalty",
            "top_logprobs",
            "temperature",
            "reasoning",
            "usage",
            "max_output_tokens",
            "max_tool_calls",
            "store",
            "background",
            "service_tier",
            "metadata",
            "safety_identifier",
            "prompt_cache_key",
        ] {
            assert!(response.get(key).is_some(), "missing {key}");
        }
        assert!(response["temperature"].is_number());
        assert!(response["top_p"].is_number());
        assert!(response["top_logprobs"].is_u64());
        assert!(response["service_tier"].is_string());
        assert!(response["completed_at"].is_i64());
        assert_eq!(
            response["usage"]["output_tokens_details"]["reasoning_tokens"],
            0
        );
    }

    #[test]
    fn envelope_echoes_request_configuration() {
        let request = json!({
            "instructions": "be brief",
            "temperature": 0.2,
            "max_output_tokens": 16,
            "store": false,
            "metadata": {"k": "v"},
            "tools": [{"type": "function", "name": "f"}],
            "previous_response_id": "resp_0",
            "service_tier": "priority",
        });
        let envelope = ResponseEnvelope::from_request(request.as_object().unwrap());
        let response = envelope.response(snapshot(&[], ResponseStatus::Completed));
        assert_eq!(response["instructions"], "be brief");
        assert_eq!(response["temperature"], 0.2);
        assert_eq!(response["max_output_tokens"], 16);
        assert_eq!(response["store"], false);
        assert_eq!(response["metadata"]["k"], "v");
        assert_eq!(response["tools"][0]["name"], "f");
        for field in ["description", "parameters", "strict"] {
            assert_eq!(response["tools"][0].get(field), Some(&Value::Null));
        }
        assert_eq!(response["previous_response_id"], "resp_0");
        assert_eq!(response["service_tier"], "default");
        assert!(!envelope.store());
    }

    #[test]
    fn incomplete_status_reports_its_reason() {
        let status = ResponseStatus::from_finish_reason(Some("length"));
        let response = ResponseEnvelope::default().response(snapshot(&[], status));
        assert_eq!(response["status"], "incomplete");
        assert_eq!(
            response["incomplete_details"]["reason"],
            "max_output_tokens"
        );
        assert_eq!(status.item_status(), "incomplete");
        assert!(response["completed_at"].is_null());
    }

    #[test]
    fn failed_response_carries_its_error_and_no_completion_time() {
        let partial = json!({
            "type": "message",
            "status": "incomplete",
            "content": [{"type": "output_text", "text": "discard me"}],
        });
        let output = [partial];
        let mut snapshot = snapshot(&output, ResponseStatus::Failed);
        snapshot.error = Some(ResponseError {
            code: "server_error",
            message: "boom",
        });
        let response = ResponseEnvelope::default().response(snapshot);
        assert_eq!(response["status"], "failed");
        assert_eq!(response["error"]["code"], "server_error");
        assert_eq!(response["output"], json!([]));
        assert_eq!(response["output_text"], "");
        assert!(response["completed_at"].is_null());
    }

    #[test]
    fn chat_usage_converts_with_integer_details() {
        let usage = ResponsesUsage::from_chat_value(&json!({
            "prompt_tokens": 5,
            "completion_tokens": 3,
            "prompt_tokens_details": {"cached_tokens": 2},
            "completion_tokens_details": {"reasoning_tokens": 1},
        }))
        .unwrap();
        assert_eq!(usage.total_tokens, 8);
        assert_eq!(usage.cached_tokens, 2);
        assert_eq!(usage.reasoning_tokens, 1);
        assert!(ResponsesUsage::from_chat_value(&json!({})).is_none());
    }

    #[test]
    fn ids_are_unique_and_prefixed() {
        let first = new_response_id();
        assert!(first.starts_with("resp_"));
        assert_ne!(first, new_response_id());
    }
}
