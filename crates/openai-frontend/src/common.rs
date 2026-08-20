use std::{
    collections::BTreeMap,
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::errors::OpenAiError;

const MAX_AGENT_SESSION_ID_BYTES: usize = 512;
const INTERNAL_AGENT_SESSION_ID: &str = "mesh_internal_agent_session_id";
const INTERNAL_AGENT_SESSION_SOURCE: &str = "mesh_internal_agent_session_source";
const INTERNAL_AGENT_SESSION_TRUSTED: &str = "mesh_internal_agent_session_trusted";

/// Where the OpenAI frontend obtained a stable, caller-authenticated session
/// identity. The identity is routing/lifecycle metadata, not a request ID. A
/// protocol-native conversation ID may also be used to derive a prompt-cache
/// key, but the two uses remain separate policy decisions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentSessionSource {
    /// A header explicitly configured as trusted by the endpoint operator.
    TrustedHeader(String),
    /// The protocol-native `conversation.id` from an OpenAI Responses request.
    ResponsesConversation,
}

/// Stable opaque identity supplied by the agent or its trusted upstream proxy.
/// Authentication remains the endpoint operator's responsibility.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentSessionIdentity {
    id: String,
    source: AgentSessionSource,
}

impl AgentSessionIdentity {
    pub fn new(id: impl Into<String>, source: AgentSessionSource) -> Result<Self, OpenAiError> {
        let id = id.into();
        let trimmed = id.trim();
        if trimmed.is_empty()
            || trimmed.len() > MAX_AGENT_SESSION_ID_BYTES
            || trimmed.chars().any(char::is_control)
        {
            return Err(OpenAiError::invalid_request(
                "agent session identity must be non-empty, printable, and at most 512 bytes",
            ));
        }
        Ok(Self {
            id: trimmed.to_owned(),
            source,
        })
    }

    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    #[must_use]
    pub fn source(&self) -> &AgentSessionSource {
        &self.source
    }
}

impl AgentSessionSource {
    #[must_use]
    pub(crate) fn label(&self) -> &str {
        match self {
            Self::TrustedHeader(header) => header,
            Self::ResponsesConversation => "responses.conversation",
        }
    }
}

pub(crate) fn set_agent_session_metadata(
    extra: &mut BTreeMap<String, Value>,
    identity: Option<AgentSessionIdentity>,
) {
    extra.remove(INTERNAL_AGENT_SESSION_ID);
    extra.remove(INTERNAL_AGENT_SESSION_SOURCE);
    extra.remove(INTERNAL_AGENT_SESSION_TRUSTED);
    if let Some(identity) = identity {
        extra.insert(
            INTERNAL_AGENT_SESSION_SOURCE.to_owned(),
            Value::String(identity.source().label().to_owned()),
        );
        extra.insert(
            INTERNAL_AGENT_SESSION_ID.to_owned(),
            Value::String(identity.id().to_owned()),
        );
    }
}

pub(crate) fn agent_session_metadata(extra: &BTreeMap<String, Value>) -> Option<&str> {
    extra.get(INTERNAL_AGENT_SESSION_ID).and_then(Value::as_str)
}

pub(crate) fn agent_session_source_metadata(extra: &BTreeMap<String, Value>) -> Option<&str> {
    extra
        .get(INTERNAL_AGENT_SESSION_SOURCE)
        .and_then(Value::as_str)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum StopSequence {
    One(String),
    Many(Vec<String>),
}

impl StopSequence {
    /// Build the wire representation used by both chat and completion
    /// requests from a stage/config default list.
    pub fn from_values(values: Vec<String>) -> Self {
        if values.len() == 1 {
            Self::One(values.into_iter().next().unwrap_or_default())
        } else {
            Self::Many(values)
        }
    }

    pub fn values(&self) -> Vec<&str> {
        match self {
            StopSequence::One(value) => vec![value.as_str()],
            StopSequence::Many(values) => values.iter().map(String::as_str).collect(),
        }
    }
}

/// Normalize the OpenAI/provider reasoning aliases that arrive in a raw JSON
/// request before typed request deserialization or backend dispatch.
///
/// This is intentionally the same compatibility policy used by
/// [`normalize_reasoning_template_options`].  It only injects the canonical
/// `chat_template_kwargs.enable_thinking` value; caller-supplied kwargs remain
/// authoritative and all source compatibility fields remain present.
pub(crate) fn normalize_reasoning_compat_fields(
    object: &mut Map<String, Value>,
) -> Result<bool, OpenAiError> {
    let mut enable_thinking = reasoning_object_override(object.get("reasoning"))?;

    if let Some(effort) = optional_string_field(object, "reasoning_effort")? {
        enable_thinking = Some(match effort {
            "none" => false,
            "minimal" | "low" | "medium" | "high" | "xhigh" | "max" => true,
            _ => {
                return Err(OpenAiError::invalid_request(
                    "reasoning_effort must be one of none, minimal, low, medium, high, xhigh, max",
                ));
            }
        });
    }

    for field in THINKING_BOOLEAN_ALIASES {
        if let Some(enabled) = optional_bool_field(object, field)? {
            enable_thinking = Some(enabled);
        }
    }
    if optional_u32_field(object, "thinking_budget")? == Some(0) {
        enable_thinking = Some(false);
    }

    if chat_template_kwargs_override(object)? {
        return Ok(false);
    }

    let Some(enabled) = enable_thinking else {
        return Ok(false);
    };
    let kwargs = ensure_chat_template_kwargs_object(object)?;
    kwargs.insert("enable_thinking".to_string(), Value::Bool(enabled));
    Ok(true)
}

fn reasoning_object_override(value: Option<&Value>) -> Result<Option<bool>, OpenAiError> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let object = value
        .as_object()
        .ok_or_else(|| OpenAiError::invalid_request("reasoning must be an object"))?;
    let enabled = optional_bool_field(object, "enabled")?;
    let effort = optional_string_field(object, "effort")?;
    let effort_is_valid = match effort {
        Some("none" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max") | None => true,
        Some(_) => false,
    };
    if !effort_is_valid {
        return Err(OpenAiError::invalid_request(
            "reasoning.effort must be one of none, minimal, low, medium, high, xhigh, max",
        ));
    }
    let max_tokens = optional_u32_field(object, "max_tokens")?;

    if enabled == Some(false) || effort == Some("none") || max_tokens == Some(0) {
        Ok(Some(false))
    } else if enabled == Some(true) || effort.is_some() || max_tokens.is_some() {
        Ok(Some(true))
    } else {
        Ok(None)
    }
}

fn chat_template_kwargs_override(object: &Map<String, Value>) -> Result<bool, OpenAiError> {
    let Some(value) = object.get("chat_template_kwargs") else {
        return Ok(false);
    };
    if value.is_null() {
        return Ok(false);
    }
    let kwargs = value
        .as_object()
        .ok_or_else(|| OpenAiError::invalid_request("chat_template_kwargs must be an object"))?;
    for field in THINKING_BOOLEAN_ALIASES {
        if optional_bool_field(kwargs, field)?.is_some() {
            return Ok(true);
        }
    }
    Ok(false)
}

fn ensure_chat_template_kwargs_object(
    object: &mut Map<String, Value>,
) -> Result<&mut Map<String, Value>, OpenAiError> {
    if !object.contains_key("chat_template_kwargs") {
        object.insert(
            "chat_template_kwargs".to_string(),
            Value::Object(Map::new()),
        );
    }
    object
        .get_mut("chat_template_kwargs")
        .and_then(Value::as_object_mut)
        .ok_or_else(|| OpenAiError::invalid_request("chat_template_kwargs must be an object"))
}

fn optional_bool_field(
    object: &Map<String, Value>,
    field: &str,
) -> Result<Option<bool>, OpenAiError> {
    object
        .get(field)
        .filter(|value| !value.is_null())
        .map(|value| {
            value
                .as_bool()
                .ok_or_else(|| OpenAiError::invalid_request(format!("{field} must be a boolean")))
        })
        .transpose()
}

fn optional_string_field<'a>(
    object: &'a Map<String, Value>,
    field: &str,
) -> Result<Option<&'a str>, OpenAiError> {
    object
        .get(field)
        .filter(|value| !value.is_null())
        .map(|value| {
            value
                .as_str()
                .ok_or_else(|| OpenAiError::invalid_request(format!("{field} must be a string")))
        })
        .transpose()
}

fn optional_u32_field(
    object: &Map<String, Value>,
    field: &str,
) -> Result<Option<u32>, OpenAiError> {
    object
        .get(field)
        .filter(|value| !value.is_null())
        .map(|value| {
            serde_json::from_value::<u32>(value.clone())
                .map_err(|_| OpenAiError::invalid_request(format!("{field} must be an integer")))
        })
        .transpose()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct StreamOptions {
    pub include_usage: Option<bool>,
    #[serde(flatten)]
    pub extra: std::collections::BTreeMap<String, Value>,
}

impl StreamOptions {
    pub fn include_usage(&self) -> bool {
        self.include_usage.unwrap_or(false)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct ReasoningConfig {
    pub enabled: Option<bool>,
    pub effort: Option<ReasoningEffort>,
    pub max_tokens: Option<u32>,
    pub exclude: Option<bool>,
    #[serde(flatten)]
    pub extra: std::collections::BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ReasoningTemplateOptions {
    pub enable_thinking: Option<bool>,
    pub chat_template_kwargs: BTreeMap<String, Value>,
}

pub const THINKING_BOOLEAN_ALIASES: &[&str] = &[
    "enable_thinking",
    "enableThinking",
    "enable_reasoning",
    "use_reasoning",
    "reasoning_enabled",
    "use_thinking",
    "thinking_enabled",
    "enable_think",
    "think_enabled",
];

pub fn normalize_reasoning_template_options(
    reasoning: Option<&ReasoningConfig>,
    reasoning_effort: Option<ReasoningEffort>,
    extra: &BTreeMap<String, Value>,
) -> Result<ReasoningTemplateOptions, OpenAiError> {
    let mut options = ReasoningTemplateOptions::default();

    if let Some(reasoning) = reasoning {
        if let Some(effort) = reasoning.effort {
            options
                .chat_template_kwargs
                .insert("reasoning_effort".to_string(), Value::from(effort.as_str()));
        }
        if reasoning.enabled == Some(false)
            || matches!(reasoning.effort, Some(ReasoningEffort::None))
            || reasoning.max_tokens == Some(0)
        {
            options.enable_thinking = Some(false);
        } else if reasoning.enabled == Some(true)
            || reasoning.effort.is_some()
            || reasoning.max_tokens.is_some()
        {
            options.enable_thinking = Some(true);
        }
    }

    if let Some(effort) = reasoning_effort {
        options
            .chat_template_kwargs
            .insert("reasoning_effort".to_string(), Value::from(effort.as_str()));
        options.enable_thinking = Some(!matches!(effort, ReasoningEffort::None));
    }

    for field in THINKING_BOOLEAN_ALIASES {
        if let Some(enable_thinking) = optional_bool_extra(extra, field)? {
            options.enable_thinking = Some(enable_thinking);
        }
    }
    if optional_u32_extra(extra, "thinking_budget")? == Some(0) {
        options.enable_thinking = Some(false);
    }

    if let Some(value) = extra.get("chat_template_kwargs") {
        let object = value.as_object().ok_or_else(|| {
            OpenAiError::invalid_request("chat_template_kwargs must be an object")
        })?;
        for (field, value) in object {
            options
                .chat_template_kwargs
                .insert(field.clone(), value.clone());
        }
        for field in THINKING_BOOLEAN_ALIASES {
            if let Some(value) = object.get(*field) {
                let enabled = value.as_bool().ok_or_else(|| {
                    OpenAiError::invalid_request(format!(
                        "chat_template_kwargs.{field} must be a boolean"
                    ))
                })?;
                options.enable_thinking = Some(enabled);
            }
        }
    }

    Ok(options)
}

impl ReasoningEffort {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Xhigh => "xhigh",
            Self::Max => "max",
        }
    }
}

fn optional_u32_extra(
    extra: &BTreeMap<String, Value>,
    field: &str,
) -> Result<Option<u32>, OpenAiError> {
    extra
        .get(field)
        .filter(|value| !value.is_null())
        .map(|value| {
            serde_json::from_value::<u32>(value.clone())
                .map_err(|_| OpenAiError::invalid_request(format!("{field} must be an integer")))
        })
        .transpose()
}

fn optional_bool_extra(
    extra: &BTreeMap<String, Value>,
    field: &str,
) -> Result<Option<bool>, OpenAiError> {
    extra
        .get(field)
        .filter(|value| !value.is_null())
        .map(|value| {
            value
                .as_bool()
                .ok_or_else(|| OpenAiError::invalid_request(format!("{field} must be a boolean")))
        })
        .transpose()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
}

impl Usage {
    pub fn new(prompt_tokens: u32, completion_tokens: u32) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens.saturating_add(completion_tokens),
            prompt_tokens_details: None,
        }
    }

    pub fn with_cached_tokens(mut self, cached_tokens: u32) -> Self {
        self.prompt_tokens_details = Some(PromptTokensDetails { cached_tokens });
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct PromptTokensDetails {
    pub cached_tokens: u32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PromptCacheRetention {
    InMemory,
    #[serde(rename = "24h")]
    TwentyFourHours,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
}

impl FinishReason {
    pub fn as_str(self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
            FinishReason::ToolCalls => "tool_calls",
            FinishReason::ContentFilter => "content_filter",
        }
    }
}

pub fn completion_id(prefix: &str) -> String {
    format!("{prefix}-{}", now_unix_millis())
}

pub fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn now_unix_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn normalize_reasoning_disables_thinking() {
        let reasoning = ReasoningConfig {
            enabled: Some(false),
            ..ReasoningConfig::default()
        };

        let options =
            normalize_reasoning_template_options(Some(&reasoning), None, &BTreeMap::new()).unwrap();

        assert_eq!(options.enable_thinking, Some(false));
    }

    #[test]
    fn normalize_reasoning_effort_none_disables_thinking() {
        let options = normalize_reasoning_template_options(
            None,
            Some(ReasoningEffort::None),
            &BTreeMap::new(),
        )
        .unwrap();

        assert_eq!(options.enable_thinking, Some(false));
    }

    #[test]
    fn normalize_provider_aliases_override_canonical_reasoning() {
        let reasoning = ReasoningConfig {
            enabled: Some(false),
            ..ReasoningConfig::default()
        };
        let mut extra = BTreeMap::new();
        extra.insert("enable_thinking".to_string(), json!(true));

        let options = normalize_reasoning_template_options(Some(&reasoning), None, &extra).unwrap();

        assert_eq!(options.enable_thinking, Some(true));
    }

    #[test]
    fn normalize_chat_template_kwargs_aliases() {
        let mut extra = BTreeMap::new();
        extra.insert(
            "chat_template_kwargs".to_string(),
            json!({"enable_thinking": false}),
        );

        let options = normalize_reasoning_template_options(None, None, &extra).unwrap();

        assert_eq!(options.enable_thinking, Some(false));
    }

    #[test]
    fn normalize_reasoning_effort_preserves_template_kwarg_and_explicit_kwargs_win() {
        let mut extra = BTreeMap::new();
        extra.insert(
            "chat_template_kwargs".to_string(),
            json!({"reasoning_effort": "custom", "temperature": 0.2}),
        );

        let options =
            normalize_reasoning_template_options(None, Some(ReasoningEffort::Max), &extra).unwrap();

        assert_eq!(options.enable_thinking, Some(true));
        assert_eq!(
            options.chat_template_kwargs.get("reasoning_effort"),
            Some(&json!("custom"))
        );
        assert_eq!(
            options.chat_template_kwargs.get("temperature"),
            Some(&json!(0.2))
        );
    }

    #[test]
    fn normalize_thinking_budget_zero_disables_thinking() {
        let reasoning = ReasoningConfig {
            enabled: Some(true),
            ..ReasoningConfig::default()
        };
        let mut extra = BTreeMap::new();
        extra.insert("thinking_budget".to_string(), json!(0));

        let options = normalize_reasoning_template_options(Some(&reasoning), None, &extra).unwrap();

        assert_eq!(options.enable_thinking, Some(false));
    }

    #[test]
    fn stop_sequence_from_values_preserves_single_and_many_wire_shapes() {
        assert_eq!(
            StopSequence::from_values(vec!["stop".to_string()]),
            StopSequence::One("stop".to_string())
        );
        assert_eq!(
            StopSequence::from_values(vec!["a".to_string(), "b".to_string()]),
            StopSequence::Many(vec!["a".to_string(), "b".to_string()])
        );
    }

    #[test]
    fn reasoning_compatibility_golden_table() {
        let cases = [
            (
                json!({"reasoning_effort": "none"}),
                json!({
                    "reasoning_effort": "none",
                    "chat_template_kwargs": {"enable_thinking": false}
                }),
                true,
            ),
            (
                json!({"reasoning": {"enabled": false, "effort": "low"}}),
                json!({
                    "reasoning": {"enabled": false, "effort": "low"},
                    "chat_template_kwargs": {"enable_thinking": false}
                }),
                true,
            ),
            (
                json!({
                    "reasoning_effort": "high",
                    "chat_template_kwargs": {"enable_thinking": false, "custom": 7}
                }),
                json!({
                    "reasoning_effort": "high",
                    "chat_template_kwargs": {"enable_thinking": false, "custom": 7}
                }),
                false,
            ),
        ];

        for (input, expected, expected_changed) in cases {
            let mut object = input.as_object().expect("object fixture").clone();
            let changed = normalize_reasoning_compat_fields(&mut object).unwrap();
            assert_eq!(Value::Object(object), expected);
            assert_eq!(changed, expected_changed);
        }
    }
}
