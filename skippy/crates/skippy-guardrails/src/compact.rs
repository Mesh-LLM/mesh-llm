use serde_json::Value;

pub const MESH_COMPACT_FIELD: &str = "mesh_compact";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactionOverride {
    Unset,
    Enabled,
    Disabled,
    InvalidType,
}

impl CompactionOverride {
    pub fn from_value(value: Option<&Value>) -> Self {
        match value {
            None => Self::Unset,
            Some(Value::Bool(true)) => Self::Enabled,
            Some(Value::Bool(false)) => Self::Disabled,
            Some(_) => Self::InvalidType,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompactionConfig {
    pub enabled: bool,
    pub context_limit_tokens: Option<usize>,
    pub trigger_ratio_percent: u8,
    pub target_ratio_percent: u8,
    pub allow_reasoning_drop: bool,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            context_limit_tokens: None,
            trigger_ratio_percent: 90,
            target_ratio_percent: 80,
            allow_reasoning_drop: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompactionRequest {
    pub messages: Vec<Value>,
    pub override_value: CompactionOverride,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactionDecision {
    Disabled,
    BelowThreshold,
    NoReduction,
    Compacted,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactionReport {
    pub decision: CompactionDecision,
    pub estimated_tokens_before: usize,
    pub estimated_tokens_after: usize,
    pub messages_before: usize,
    pub messages_after: usize,
    pub dropped_nudges: usize,
    pub dropped_tool_results: usize,
    pub dropped_reasoning: usize,
    pub warning_injected: bool,
}

pub fn compact_messages(
    request: CompactionRequest,
    config: CompactionConfig,
) -> (Vec<Value>, CompactionReport) {
    let before_tokens = estimate_messages_tokens(&request.messages);
    let mut report = CompactionReport {
        decision: CompactionDecision::Disabled,
        estimated_tokens_before: before_tokens,
        estimated_tokens_after: before_tokens,
        messages_before: request.messages.len(),
        messages_after: request.messages.len(),
        dropped_nudges: 0,
        dropped_tool_results: 0,
        dropped_reasoning: 0,
        warning_injected: false,
    };

    if !should_compact(&request, &config, before_tokens) {
        report.decision =
            if config.enabled || matches!(request.override_value, CompactionOverride::Enabled) {
                CompactionDecision::BelowThreshold
            } else {
                CompactionDecision::Disabled
            };
        return (request.messages, report);
    }

    let target_tokens = target_tokens(&config, before_tokens);
    let mut messages = request.messages;
    drop_messages_matching(
        &mut messages,
        is_retry_nudge_message,
        &mut report.dropped_nudges,
    );
    if estimate_messages_tokens(&messages) > target_tokens {
        drop_messages_matching(
            &mut messages,
            is_tool_result_message,
            &mut report.dropped_tool_results,
        );
    }
    if config.allow_reasoning_drop && estimate_messages_tokens(&messages) > target_tokens {
        report.dropped_reasoning = strip_reasoning_fields(&mut messages);
    }
    if report.dropped_nudges + report.dropped_tool_results + report.dropped_reasoning > 0 {
        inject_compaction_warning(&mut messages);
        report.warning_injected = true;
        report.decision = CompactionDecision::Compacted;
    } else {
        report.decision = CompactionDecision::NoReduction;
    }
    report.estimated_tokens_after = estimate_messages_tokens(&messages);
    report.messages_after = messages.len();
    (messages, report)
}

pub fn estimate_message_tokens(message: &Value) -> usize {
    // Estimate compactable text only. Encoded media bytes and image URLs are
    // not text tokens; the runtime owns the actual multimodal context limit.
    let chars = message.as_object().map_or_else(
        || estimate_value_chars(message),
        |object| {
            object
                .iter()
                .map(|(key, value)| {
                    if key == "content" {
                        estimate_content_chars(value)
                    } else {
                        estimate_value_chars(value)
                    }
                })
                .sum()
        },
    );
    chars / 4 + 1
}

fn estimate_content_chars(content: &Value) -> usize {
    let Value::Array(parts) = content else {
        return estimate_value_chars(content);
    };
    parts
        .iter()
        .map(|part| {
            if matches!(
                part.get("type").and_then(Value::as_str),
                Some("image_url" | "input_image" | "input_audio")
            ) {
                0
            } else {
                estimate_value_chars(part)
            }
        })
        .sum()
}

fn estimate_messages_tokens(messages: &[Value]) -> usize {
    messages.iter().map(estimate_message_tokens).sum()
}

fn should_compact(
    request: &CompactionRequest,
    config: &CompactionConfig,
    estimated_tokens: usize,
) -> bool {
    if matches!(request.override_value, CompactionOverride::Disabled) {
        return false;
    }
    if !config.enabled && !matches!(request.override_value, CompactionOverride::Enabled) {
        return false;
    }
    let Some(limit) = config.context_limit_tokens else {
        return matches!(request.override_value, CompactionOverride::Enabled);
    };
    estimated_tokens >= percent_of(limit, config.trigger_ratio_percent)
}

fn target_tokens(config: &CompactionConfig, fallback: usize) -> usize {
    config
        .context_limit_tokens
        .map(|limit| percent_of(limit, config.target_ratio_percent))
        .unwrap_or(fallback.saturating_mul(80) / 100)
}

fn percent_of(value: usize, percent: u8) -> usize {
    value.saturating_mul(percent as usize) / 100
}

fn estimate_value_chars(value: &Value) -> usize {
    match value {
        Value::String(value) => value.len(),
        Value::Array(values) => values.iter().map(estimate_value_chars).sum(),
        Value::Object(object) => object.values().map(estimate_value_chars).sum(),
        _ => value.to_string().len(),
    }
}

fn drop_messages_matching(
    messages: &mut Vec<Value>,
    predicate: fn(&Value) -> bool,
    dropped_count: &mut usize,
) {
    let before = messages.len();
    messages.retain(|message| !predicate(message));
    *dropped_count = before.saturating_sub(messages.len());
}

fn is_retry_nudge_message(message: &Value) -> bool {
    message
        .get("content")
        .and_then(Value::as_str)
        .is_some_and(|content| {
            content.contains("Your previous reply")
                && content.contains("valid")
                && content.contains("Do not add extra text")
        })
}

fn is_tool_result_message(message: &Value) -> bool {
    matches!(message.get("role").and_then(Value::as_str), Some("tool"))
}

fn strip_reasoning_fields(messages: &mut [Value]) -> usize {
    let mut dropped = 0;
    for message in messages {
        if let Some(object) = message.as_object_mut()
            && (object.remove("reasoning_content").is_some()
                || object.remove("reasoning").is_some()
                || object.remove("thinking").is_some())
        {
            dropped += 1;
        }
    }
    dropped
}

fn inject_compaction_warning(messages: &mut Vec<Value>) {
    messages.insert(
        0,
        serde_json::json!({
            "role": "system",
            "content": "Context was compacted before this turn: retry nudges, old tool results, or hidden reasoning may have been removed. Use the remaining messages as authoritative."
        }),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn media_shaped_tool_arguments_and_plain_text_still_count() {
        let payload = "A".repeat(64_000);
        for message in [
            serde_json::json!({"role":"assistant","tool_calls":[{"function":{"arguments":{"type":"image_url","image_url":{"url":payload}}}}]}),
            serde_json::json!({"role":"user","content":[{"type":"text","text":payload}]}),
        ] {
            assert!(estimate_message_tokens(&message) >= 16_000);
        }
    }

    #[test]
    fn encoded_media_does_not_trigger_text_compaction() {
        for part in [
            serde_json::json!({"type":"image_url","image_url":{"url":format!("data:image/png;base64,{}", "A".repeat(64_000))}}),
            serde_json::json!({"type":"input_audio","input_audio":{"data":"A".repeat(64_000),"format":"wav"}}),
        ] {
            let messages = vec![
                serde_json::json!({"role":"user","content":[{"type":"text","text":"Describe this."},part]}),
            ];
            let (actual, report) = compact_messages(
                CompactionRequest {
                    messages: messages.clone(),
                    override_value: CompactionOverride::Unset,
                },
                CompactionConfig {
                    enabled: true,
                    context_limit_tokens: Some(2048),
                    ..Default::default()
                },
            );
            assert_eq!(actual, messages);
            assert_eq!(report.decision, CompactionDecision::BelowThreshold);
            assert!(!report.warning_injected);
        }
    }

    #[test]
    fn unsuccessful_compaction_preserves_the_prompt_even_when_forced() {
        for override_value in [CompactionOverride::Unset, CompactionOverride::Enabled] {
            let messages = vec![serde_json::json!({"role":"user","content":"x".repeat(10_000)})];
            let (actual, report) = compact_messages(
                CompactionRequest {
                    messages: messages.clone(),
                    override_value,
                },
                CompactionConfig {
                    enabled: true,
                    context_limit_tokens: Some(2048),
                    ..Default::default()
                },
            );
            assert_eq!(actual, messages);
            assert_eq!(report.decision, CompactionDecision::NoReduction);
            assert!(!report.warning_injected);
        }
    }

    #[test]
    fn disabled_config_passes_through() {
        let messages = vec![serde_json::json!({"role":"user","content":"hi"})];
        let (compacted, report) = compact_messages(
            CompactionRequest {
                messages: messages.clone(),
                override_value: CompactionOverride::Unset,
            },
            CompactionConfig::default(),
        );
        assert_eq!(compacted, messages);
        assert_eq!(report.decision, CompactionDecision::Disabled);
    }

    #[test]
    fn forced_compaction_drops_tool_results_and_injects_warning() {
        let messages = vec![
            serde_json::json!({"role":"tool","content":"large result"}),
            serde_json::json!({"role":"user","content":"next"}),
        ];
        let (compacted, report) = compact_messages(
            CompactionRequest {
                messages,
                override_value: CompactionOverride::Enabled,
            },
            CompactionConfig::default(),
        );
        assert_eq!(report.decision, CompactionDecision::Compacted);
        assert_eq!(report.dropped_tool_results, 1);
        assert_eq!(compacted[0]["role"], "system");
    }
}
