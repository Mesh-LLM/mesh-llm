use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    common::{
        FinishReason, PromptCacheRetention, ReasoningConfig, ReasoningEffort, StopSequence,
        StreamOptions, Usage, agent_session_metadata, agent_session_source_metadata,
    },
    errors::InferenceError,
};

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
pub struct ChatCompletionRequest {
    pub model: String,
    #[serde(default)]
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    pub max_tokens: Option<u32>,
    pub max_completion_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub n: Option<u32>,
    pub logprobs: Option<bool>,
    pub top_logprobs: Option<u32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub logit_bias: Option<BTreeMap<String, Value>>,
    pub response_format: Option<Value>,
    pub tools: Option<Value>,
    pub tool_choice: Option<Value>,
    pub parallel_tool_calls: Option<bool>,
    pub user: Option<String>,
    pub stop: Option<StopSequence>,
    pub seed: Option<u64>,
    pub reasoning: Option<ReasoningConfig>,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub prompt_cache_key: Option<String>,
    pub prompt_cache_retention: Option<PromptCacheRetention>,
    pub stream_options: Option<StreamOptions>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl ChatCompletionRequest {
    #[must_use]
    pub fn agent_session(&self) -> Option<&str> {
        agent_session_metadata(&self.extra)
    }

    #[must_use]
    pub fn agent_session_source(&self) -> Option<&str> {
        agent_session_source_metadata(&self.extra)
    }

    pub fn effective_max_tokens(&self) -> Option<u32> {
        self.max_completion_tokens.or(self.max_tokens)
    }

    pub fn include_usage(&self) -> bool {
        self.stream_options
            .as_ref()
            .map(StreamOptions::include_usage)
            .unwrap_or(false)
    }

    pub fn validate(&self) -> Result<(), InferenceError> {
        crate::generation::validation::RequestValidation {
            model: &self.model,
            messages: &self.messages,
            max_tokens: self.max_tokens,
            max_completion_tokens: self.max_completion_tokens,
            n: self.n,
            logprobs: self.logprobs,
            top_logprobs: self.top_logprobs,
            tools: self.tools.as_ref(),
            response_format: self.response_format.as_ref(),
        }
        .validate()
    }
}

pub use crate::generation::content::{
    GenerationMessage as ChatMessage, MessageContent, MessageContentPart, ensure_tool_call_ids,
    message_content_to_text, messages_to_plain_prompt,
};

/// `#[non_exhaustive]`: a downstream crate that builds this via struct
/// literal (rather than [`Self::new`]/[`Self::new_with_reason`]/
/// [`Self::from_parts`]) would otherwise break every time this crate adds a
/// field — as `capsule_marker` itself just did. New fields default through
/// the constructors instead of requiring every caller to be updated.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[non_exhaustive]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: Usage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timings: Option<BTreeMap<String, Value>>,
    /// Rung-ladder response-leg marker, minted by
    /// [`crate::hooks::InferenceHookPolicy::capsule_marker_for_response`]. Never
    /// serialized into the OpenAI-shaped JSON body — a real `X-Capsule-Id`
    /// rides as an HTTP response header, set from this field by the router's
    /// `frontend_lifecycle_middleware`, the same layer that already sets
    /// `x-request-id`.
    #[serde(skip)]
    pub capsule_marker: Option<CapsuleMarker>,
    /// The host-minted per-exchange id ([`crate::hooks::ChatExchangeRoute::exchange_id`]),
    /// the same value [`crate::hooks::InferenceHookPolicy::on_chat_completion_terminal`]
    /// received for this exchange. Never serialized into the OpenAI-shaped
    /// JSON body — it rides as an `X-Exchange-Id` HTTP response header, set
    /// from this field by the router's `frontend_lifecycle_middleware`,
    /// mirroring `capsule_marker`/`X-Capsule-Id` above. `None` for a response
    /// built by a caller that never dispatched through the exchange-tracked
    /// path (e.g. `Self::new`/`Self::new_with_reason`/`Self::from_parts`
    /// before a backend attaches it).
    #[serde(skip)]
    pub exchange_id: Option<String>,
}

pub use crate::generation::response::{CapsuleMarker, capsule_id_is_valid};

impl ChatCompletionResponse {
    pub fn new(model: impl Into<String>, content: impl Into<String>, usage: Usage) -> Self {
        Self::new_with_reason(model, content, usage, FinishReason::Stop)
    }

    pub fn new_with_reason(
        model: impl Into<String>,
        content: impl Into<String>,
        usage: Usage,
        finish_reason: FinishReason,
    ) -> Self {
        crate::generation::GenerationResponse::new_with_reason(model, content, usage, finish_reason)
            .into()
    }

    /// Construct a response with an explicit id/creation time/choice set —
    /// for callers (e.g. a backend reconstructing a parsed generation) that
    /// need full control over those fields, unlike [`Self::new`]/
    /// [`Self::new_with_reason`], which always mint a fresh id and a single
    /// choice. `capsule_marker` always starts `None`; it is attached later,
    /// by [`crate::hooks::HookedInferenceBackend`]. Exists so a downstream
    /// crate that previously built this via struct literal has a
    /// `#[non_exhaustive]`-safe replacement.
    pub fn from_parts(
        id: impl Into<String>,
        created: u64,
        model: impl Into<String>,
        choices: Vec<ChatCompletionChoice>,
        usage: Usage,
        timings: Option<BTreeMap<String, Value>>,
    ) -> Self {
        crate::generation::GenerationResponse::from_parts(
            id,
            created,
            model,
            choices.into_iter().map(Into::into).collect(),
            usage,
            timings,
        )
        .into()
    }

    pub fn with_timings(mut self, timings: Option<BTreeMap<String, Value>>) -> Self {
        self.timings = timings;
        self
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ChatCompletionChoice {
    pub index: u32,
    pub message: AssistantMessage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Value>,
    pub finish_reason: Option<FinishReason>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct AssistantMessage {
    pub role: &'static str,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Value>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timings: Option<BTreeMap<String, Value>>,
}

impl ChatCompletionChunk {
    pub fn role(model: impl Into<String>) -> Self {
        crate::generation::GenerationEvent::role(model).into()
    }

    pub fn delta(model: impl Into<String>, content: impl Into<String>) -> Self {
        crate::generation::GenerationEvent::delta(model, content).into()
    }

    pub fn done(model: impl Into<String>) -> Self {
        Self::done_with_reason(model, FinishReason::Stop)
    }

    pub fn done_with_reason(model: impl Into<String>, finish_reason: FinishReason) -> Self {
        crate::generation::GenerationEvent::done_with_reason(model, finish_reason).into()
    }

    pub fn usage(model: impl Into<String>, usage: Usage) -> Self {
        crate::generation::GenerationEvent::usage(model, usage).into()
    }

    pub fn usage_with_timings(
        model: impl Into<String>,
        usage: Usage,
        timings: Option<BTreeMap<String, Value>>,
    ) -> Self {
        Self {
            timings,
            ..Self::usage(model, usage)
        }
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ChatCompletionChunkChoice {
    pub index: u32,
    pub delta: ChatCompletionDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Value>,
    pub finish_reason: Option<FinishReason>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ChatCompletionDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Value>,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    #[test]
    fn chat_usage_chunk_serializes_optional_timings() {
        let chunk = ChatCompletionChunk::usage_with_timings(
            "model",
            Usage::new(3, 2),
            Some(BTreeMap::from([(
                "cache_restore_ms".to_string(),
                json!(2.5),
            )])),
        );

        let value = serde_json::to_value(chunk).unwrap();
        assert_eq!(value["usage"]["prompt_tokens"], json!(3));
        assert_eq!(value["timings"]["cache_restore_ms"], json!(2.5));
    }

    use super::*;

    #[test]
    fn assistant_message_serializes_reasoning_content_when_present() {
        let message = AssistantMessage {
            role: "assistant",
            content: Some("Final answer.".to_string()),
            reasoning_content: Some("Checked the facts first.".to_string()),
            tool_calls: None,
        };

        let value = serde_json::to_value(message).unwrap();

        assert_eq!(
            value,
            json!({
                "role": "assistant",
                "content": "Final answer.",
                "reasoning_content": "Checked the facts first."
            })
        );
    }

    #[test]
    fn chat_delta_serializes_reasoning_content_without_text_content() {
        let delta = ChatCompletionDelta {
            role: None,
            content: None,
            reasoning_content: Some("Still thinking.".to_string()),
            tool_calls: None,
        };

        let value = serde_json::to_value(delta).unwrap();

        assert_eq!(value, json!({ "reasoning_content": "Still thinking." }));
    }

    #[test]
    fn chat_message_round_trip_preserves_missing_and_null_content() {
        for expected in [
            json!({"role": "assistant", "tool_calls": []}),
            json!({"role": "assistant", "content": null, "tool_calls": []}),
        ] {
            let message: ChatMessage = serde_json::from_value(expected.clone()).unwrap();
            assert_eq!(serde_json::to_value(message).unwrap(), expected);
        }
    }
}

mod generation;
pub mod request_view;
