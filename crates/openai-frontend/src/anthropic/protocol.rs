//! Serde types for the Anthropic Messages wire protocol.
//!
//! Request shapes follow the public Anthropic Messages API; response and
//! streaming-event shapes follow the same document. Only the fields the
//! serving path can honor (or must reject explicitly) are modeled; unknown
//! fields are preserved per type via `#[serde(flatten)]` maps so strictness
//! never silently drops client intent.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Top-level `POST /v1/messages` request body.
#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicMessagesRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    pub max_tokens: u32,
    #[serde(default)]
    pub stream: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<AnthropicSystemPrompt>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<AnthropicToolDefinition>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<AnthropicToolChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<AnthropicMetadata>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Count requests have no generation budget. Keep their wire schema independent.
#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicCountTokensRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    #[serde(default)]
    pub system: Option<AnthropicSystemPrompt>,
    #[serde(default)]
    pub tools: Option<Vec<AnthropicToolDefinition>>,
    #[serde(default)]
    pub tool_choice: Option<AnthropicToolChoice>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl AnthropicCountTokensRequest {
    pub fn into_messages(self) -> AnthropicMessagesRequest {
        AnthropicMessagesRequest {
            model: self.model,
            messages: self.messages,
            system: self.system,
            tools: self.tools,
            tool_choice: self.tool_choice,
            extra: self.extra,
            max_tokens: 1,
            stream: false,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            metadata: None,
        }
    }
}

/// One conversation turn. `content` may be a plain string or a block array.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: AnthropicMessageContent,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum AnthropicMessageContent {
    Text(String),
    Blocks(Vec<AnthropicContentBlock>),
}

/// Top-level `system` prompt: a plain string or an array of text blocks.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum AnthropicSystemPrompt {
    Text(String),
    Blocks(Vec<AnthropicSystemBlock>),
}

#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicSystemBlock {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// A content block dispatched strictly by its `type` discriminator.
#[derive(Debug, Clone)]
pub enum AnthropicContentBlock {
    ToolUse(AnthropicToolUseContent),
    ToolResult(AnthropicToolResultContent),
    Text(AnthropicTextContent),
    Other(Value),
}

impl<'de> Deserialize<'de> for AnthropicContentBlock {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        match value.get("type").and_then(Value::as_str) {
            Some("text") => serde_json::from_value(value).map(Self::Text),
            Some("tool_use") => serde_json::from_value(value).map(Self::ToolUse),
            Some("tool_result") => serde_json::from_value(value).map(Self::ToolResult),
            _ => return Ok(Self::Other(value)),
        }
        .map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicTextContent {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicToolUseContent {
    #[serde(rename = "type")]
    pub kind: String,
    pub id: String,
    pub name: String,
    pub input: Value,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicToolResultContent {
    #[serde(rename = "type")]
    pub kind: String,
    pub tool_use_id: String,
    #[serde(default)]
    pub content: Option<Value>,
    #[serde(default)]
    pub is_error: bool,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicToolDefinition {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub input_schema: Option<Value>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicToolChoice {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct AnthropicMetadata {
    #[serde(default)]
    pub user_id: Option<String>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// `usage` object on responses and `message_delta` events.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u32>,
}

/// A block in a non-streaming response's `content` array.
#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "type")]
pub enum AnthropicResponseBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
}

/// Non-streaming `POST /v1/messages` response.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct AnthropicMessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub role: &'static str,
    pub model: String,
    pub content: Vec<AnthropicResponseBlock>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<&'static str>,
    pub stop_sequence: Option<Value>,
    pub usage: AnthropicUsage,
}

/// One SSE streaming event. Serialization adds the `type` discriminant.
#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "type")]
pub enum AnthropicMessagesStreamEvent {
    #[serde(rename = "message_start")]
    MessageStart { message: AnthropicMessageStart },
    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        index: usize,
        content_block: AnthropicResponseBlock,
    },
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta { index: usize, delta: AnthropicDelta },
    #[serde(rename = "content_block_stop")]
    ContentBlockStop { index: usize },
    #[serde(rename = "message_delta")]
    MessageDelta {
        delta: AnthropicMessageDelta,
        usage: AnthropicUsage,
    },
    #[serde(rename = "message_stop")]
    MessageStop {},
    #[serde(rename = "ping")]
    Ping {},
    #[serde(rename = "error")]
    Error { error: AnthropicErrorMessage },
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct AnthropicMessageStart {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub role: &'static str,
    pub model: String,
    pub content: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<&'static str>,
    pub stop_sequence: Option<Value>,
    pub usage: AnthropicUsage,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "type")]
pub enum AnthropicDelta {
    #[serde(rename = "text_delta")]
    Text { text: String },
    #[serde(rename = "input_json_delta")]
    InputJson { partial_json: String },
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct AnthropicMessageDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<&'static str>,
    pub stop_sequence: Option<Value>,
}

/// The `error` object inside error responses and error stream events.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct AnthropicErrorMessage {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub message: String,
}

/// Anthropic error envelope: `{"type":"error","error":{...}}`.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct AnthropicError {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub error: AnthropicErrorMessage,
}

/// The `tools` entry accepted by the Anthropic API. Kept separate from
/// [`AnthropicToolDefinition`] only for documentation; shapes are identical.
pub type AnthropicTool = AnthropicToolDefinition;

impl AnthropicError {
    pub fn new(kind: &'static str, message: impl Into<String>) -> Self {
        Self {
            kind: "error",
            error: AnthropicErrorMessage {
                kind,
                message: message.into(),
            },
        }
    }

    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::new("invalid_request_error", message)
    }

    pub fn api_error(message: impl Into<String>) -> Self {
        Self::new("api_error", message)
    }

    pub fn overloaded(message: impl Into<String>) -> Self {
        Self::new("overloaded_error", message)
    }

    pub fn body(&self) -> Self {
        self.clone()
    }
}

#[cfg(test)]
#[path = "protocol_tests.rs"]
mod tests;
