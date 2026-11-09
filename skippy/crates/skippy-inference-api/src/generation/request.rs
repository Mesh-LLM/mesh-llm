//! Backend generation input. HTTP deserialization belongs to each protocol
//! adapter; this type deliberately has no Serde wire representation.
//! Both token-limit aliases remain available so existing defaults and hooks
//! preserve their precedence and provenance during the boundary migration.

use super::{content::GenerationMessage, validation::RequestValidation};
use crate::{
    common::{
        AgentSessionIdentity, PromptCacheRetention, ReasoningConfig, ReasoningEffort, StopSequence,
        StreamOptions, agent_session_metadata, agent_session_source_metadata,
        set_agent_session_metadata,
    },
    errors::InferenceError,
};
use serde_json::Value;
use std::collections::BTreeMap;

/// Input owned by a generation backend, independent of frontend serialization.
///
/// Extension values remain opaque so adaptation preserves currently accepted
/// inputs, including fields interpreted later by runtime defaults or hooks.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GenerationRequest {
    pub model: String,

    pub messages: Vec<GenerationMessage>,

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
    /// Existing function definitions, including arbitrary JSON Schema and
    /// extension fields. Validation is intentionally unchanged.
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

    pub extra: BTreeMap<String, Value>,
}

impl GenerationRequest {
    pub(crate) fn set_agent_session(&mut self, identity: Option<AgentSessionIdentity>) {
        set_agent_session_metadata(&mut self.extra, identity);
    }

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
        RequestValidation {
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
