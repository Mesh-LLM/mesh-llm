use crate::frontend::generation::GeneratedText;
use crate::frontend::has_requested_tools;
use crate::frontend::message_content_to_generation_text;
use crate::frontend::prefill::PrefillChunkPolicy;
use openai_frontend::ChatCompletionRequest;
use openai_frontend::ChatCompletionResponse;
use openai_frontend::ChatHookAction;
use openai_frontend::ChatHookOutcome;
use openai_frontend::FinishReason;
use openai_frontend::OpenAiError;
use openai_frontend::OpenAiResult;
use serde_json::Value;
use serde_json::json;
use skippy_runtime::ChatReasoningFormat;
use skippy_runtime::ChatTemplateOptions;
use skippy_runtime::GenerationSignalWindow;
use skippy_runtime::MediaInput;
use std::collections::BTreeMap;

pub(in crate::frontend) fn ensure_requested_model(
    advertised_model_id: &str,
    requested: &str,
) -> OpenAiResult<()> {
    if requested == advertised_model_id
        || strip_default_revision(requested) == strip_default_revision(advertised_model_id)
    {
        Ok(())
    } else {
        Err(OpenAiError::model_not_found(requested))
    }
}

/// Strip `@main` so `org/repo@main:Q4` and `org/repo:Q4` compare equal.
///
/// Only removes `@main` when it sits at a revision boundary — followed by
/// `:` (quant separator) or end-of-string.  This avoids corrupting repo
/// names that happen to contain `@main` as a prefix of a longer segment
/// (e.g. `@mainland`).
pub(in crate::frontend) fn strip_default_revision(id: &str) -> String {
    if let Some(pos) = id.find("@main") {
        let after = pos + "@main".len();
        if after == id.len() || id.as_bytes()[after] == b':' {
            let mut s = id[..pos].to_string();
            s.push_str(&id[after..]);
            return s;
        }
    }
    id.to_string()
}

pub(in crate::frontend) fn hook_injected_text(outcome: &ChatHookOutcome) -> Option<String> {
    let text = outcome
        .actions
        .iter()
        .filter_map(|action| match action {
            ChatHookAction::InjectText { text } if !text.is_empty() => Some(text.as_str()),
            ChatHookAction::InjectText { .. }
            | ChatHookAction::ConsumeMedia { .. }
            | ChatHookAction::None => None,
        })
        .collect::<Vec<_>>()
        .join("");
    (!text.is_empty()).then_some(text)
}

pub(in crate::frontend) fn mid_generation_window_should_fire(
    decoded_tokens: usize,
    last_hook_at: &Option<usize>,
    window: &GenerationSignalWindow,
) -> bool {
    const MIN_DECODED_TOKENS: usize = 12;
    const COOLDOWN_TOKENS: usize = 32;
    const REPETITION_TRIGGER_COUNT: u32 = 3;

    if decoded_tokens < MIN_DECODED_TOKENS || window.token_count == 0 {
        return false;
    }
    if last_hook_at.is_some_and(|last| decoded_tokens.saturating_sub(last) < COOLDOWN_TOKENS) {
        return false;
    }
    let sustained_entropy =
        window.high_entropy_count.saturating_mul(4) >= window.token_count.saturating_mul(3);
    sustained_entropy || window.repetition_count >= REPETITION_TRIGGER_COUNT
}

pub(in crate::frontend) fn attrs_insert_prefill_chunk_policy(
    attrs: &mut BTreeMap<String, Value>,
    policy: &PrefillChunkPolicy,
    min_chunk_size: usize,
    max_chunk_size: usize,
) {
    attrs.insert(
        "llama_stage.prefill_chunk_size".to_string(),
        json!(policy.fixed_chunk_size()),
    );
    attrs.insert(
        "llama_stage.prefill_chunk_policy".to_string(),
        json!(policy.policy_label()),
    );
    if let Some(schedule) = policy.schedule() {
        attrs.insert(
            "llama_stage.prefill_chunk_schedule".to_string(),
            json!(schedule.label()),
        );
    }
    if let Some((start, step, max)) = policy.adaptive_params() {
        attrs.insert(
            "llama_stage.prefill_adaptive_start".to_string(),
            json!(start),
        );
        attrs.insert("llama_stage.prefill_adaptive_step".to_string(), json!(step));
        attrs.insert("llama_stage.prefill_adaptive_max".to_string(), json!(max));
    }
    if min_chunk_size != usize::MAX {
        attrs.insert(
            "llama_stage.prefill_min_chunk_size".to_string(),
            json!(min_chunk_size),
        );
        attrs.insert(
            "llama_stage.prefill_max_chunk_size".to_string(),
            json!(max_chunk_size),
        );
    }
}

#[derive(Debug, Clone)]
pub(in crate::frontend) struct PreparedGenerationPrompt {
    pub(in crate::frontend) text: String,
    pub(in crate::frontend) media: Vec<MediaInput>,
    pub(in crate::frontend) chat_parse_metadata: Option<String>,
}

impl PreparedGenerationPrompt {
    pub(in crate::frontend) fn text(text: String) -> Self {
        Self {
            text,
            media: Vec::new(),
            chat_parse_metadata: None,
        }
    }

    pub(in crate::frontend) fn has_media(&self) -> bool {
        !self.media.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::frontend) struct ParsedToolCalls {
    pub(in crate::frontend) content: Option<String>,
    pub(in crate::frontend) tool_calls: Value,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::frontend) struct ParsedChatMessage {
    pub(in crate::frontend) content: Option<String>,
    pub(in crate::frontend) reasoning_content: Option<String>,
    pub(in crate::frontend) tool_calls: Option<Value>,
}

pub(in crate::frontend) fn tool_calls_requested(request: &ChatCompletionRequest) -> bool {
    request.tools.as_ref().is_some_and(has_requested_tools)
        && !request
            .tool_choice
            .as_ref()
            .is_some_and(|choice| matches!(choice.as_str(), Some("none")))
}

pub(in crate::frontend) fn chat_output_parser_required(
    request: &ChatCompletionRequest,
    template_options: &ChatTemplateOptions,
) -> bool {
    tool_calls_requested(request)
        || template_options
            .reasoning_format
            .is_some_and(ChatReasoningFormat::parses_reasoning)
}

pub(in crate::frontend) fn template_exposes_reasoning(
    template_options: &ChatTemplateOptions,
) -> bool {
    template_options
        .reasoning_format
        .is_some_and(ChatReasoningFormat::exposes_reasoning)
}

pub(in crate::frontend) fn apply_reasoning_visibility(
    parsed: Option<ParsedChatMessage>,
    template_options: &ChatTemplateOptions,
) -> Option<ParsedChatMessage> {
    let mut parsed = parsed?;
    if !template_exposes_reasoning(template_options) {
        parsed.reasoning_content = None;
    }
    Some(parsed)
}

pub(in crate::frontend) fn chat_response_from_generated_text(
    model: String,
    output: &GeneratedText,
    parsed_message: Option<ParsedChatMessage>,
) -> ChatCompletionResponse {
    if let Some(parsed) = parsed_message {
        let finish_reason = if parsed.tool_calls.is_some() {
            FinishReason::ToolCalls
        } else {
            output.finish_reason
        };
        return ChatCompletionResponse {
            id: openai_frontend::completion_id("chatcmpl"),
            object: "chat.completion",
            created: openai_frontend::now_unix_secs(),
            model,
            choices: vec![openai_frontend::ChatCompletionChoice {
                index: 0,
                message: openai_frontend::AssistantMessage {
                    role: "assistant",
                    content: parsed.content,
                    reasoning_content: parsed.reasoning_content,
                    tool_calls: parsed.tool_calls,
                },
                logprobs: None,
                finish_reason: Some(finish_reason),
            }],
            usage: output.usage(),
            timings: output.timings(),
        };
    }

    ChatCompletionResponse::new_with_reason(
        model,
        output.text.clone(),
        output.usage(),
        output.finish_reason,
    )
    .with_timings(output.timings())
}

pub(in crate::frontend) fn completion_response_from_generated_text(
    model: String,
    output: &GeneratedText,
) -> openai_frontend::CompletionResponse {
    openai_frontend::CompletionResponse::new_with_reason(
        model,
        output.text.clone(),
        output.usage(),
        output.finish_reason,
    )
    .with_timings(output.timings())
}

pub(in crate::frontend) fn parsed_chat_message_from_json(
    message_json: &str,
    request: &ChatCompletionRequest,
) -> Option<ParsedChatMessage> {
    let value = serde_json::from_str::<Value>(message_json).ok()?;
    let tool_calls =
        parsed_tool_calls_from_message_value(&value, request).map(|parsed| parsed.tool_calls);
    Some(ParsedChatMessage {
        content: string_field(&value, "content"),
        reasoning_content: string_field(&value, "reasoning_content"),
        tool_calls,
    })
}

#[cfg(test)]
pub(in crate::frontend) fn parsed_tool_calls_from_message_json(
    message_json: &str,
    request: &ChatCompletionRequest,
) -> Option<ParsedToolCalls> {
    let value = serde_json::from_str::<Value>(message_json).ok()?;
    parsed_tool_calls_from_message_value(&value, request)
}

pub(in crate::frontend) fn parsed_tool_calls_from_message_value(
    value: &Value,
    request: &ChatCompletionRequest,
) -> Option<ParsedToolCalls> {
    if !tool_calls_requested(request) {
        return None;
    }
    let allowed_names = request_allowed_tool_names(request);
    let mut tool_calls = value
        .get("tool_calls")
        .and_then(Value::as_array)?
        .iter()
        .filter(|call| tool_call_allowed(call, &allowed_names))
        .cloned()
        .collect::<Vec<_>>();
    if request.parallel_tool_calls == Some(false) {
        tool_calls.truncate(1);
    }
    if tool_calls.is_empty() {
        return None;
    }
    Some(ParsedToolCalls {
        content: string_field(value, "content"),
        tool_calls: Value::Array(tool_calls),
    })
}

pub(in crate::frontend) fn string_field(value: &Value, field: &str) -> Option<String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .filter(|content| !content.is_empty())
        .map(ToString::to_string)
}

pub(in crate::frontend) fn request_allowed_tool_names(
    request: &ChatCompletionRequest,
) -> Vec<String> {
    if let Some(choice_name) = request
        .tool_choice
        .as_ref()
        .and_then(tool_choice_function_name)
    {
        return vec![choice_name];
    }
    request_tool_names(request)
}

pub(in crate::frontend) fn tool_choice_function_name(value: &Value) -> Option<String> {
    value
        .as_object()
        .and_then(|object| {
            object
                .get("function")
                .and_then(|function| function.get("name"))
                .or_else(|| object.get("name"))
        })
        .and_then(Value::as_str)
        .or_else(|| {
            value
                .as_str()
                .filter(|choice| !matches!(*choice, "auto" | "none" | "required"))
        })
        .map(ToString::to_string)
}

pub(in crate::frontend) fn request_tool_names(request: &ChatCompletionRequest) -> Vec<String> {
    request
        .tools
        .as_ref()
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|tool| {
            tool.get("function")
                .and_then(|function| function.get("name"))
                .or_else(|| tool.get("name"))
                .and_then(Value::as_str)
                .map(ToString::to_string)
        })
        .collect()
}

pub(in crate::frontend) fn tool_call_allowed(value: &Value, allowed_names: &[String]) -> bool {
    let Some(object) = value.as_object() else {
        return false;
    };
    let function = object.get("function").and_then(Value::as_object);
    let Some(name) = function
        .and_then(|function| function.get("name"))
        .and_then(Value::as_str)
    else {
        return false;
    };
    allowed_names.is_empty() || allowed_names.iter().any(|allowed| allowed == name)
}

pub(in crate::frontend) fn tool_calls_stream_delta(tool_calls: Value) -> Value {
    match tool_calls {
        Value::Array(calls) => Value::Array(
            calls
                .into_iter()
                .enumerate()
                .map(|(index, call)| match call {
                    Value::Object(mut object) => {
                        object
                            .entry("index")
                            .or_insert_with(|| Value::from(index as u64));
                        Value::Object(object)
                    }
                    other => other,
                })
                .collect(),
        ),
        other => other,
    }
}

pub(in crate::frontend) fn chat_message_generation_value(
    message: &openai_frontend::ChatMessage,
    marker: &str,
    media: &mut Vec<MediaInput>,
) -> OpenAiResult<Value> {
    let mut value = serde_json::to_value(message)
        .map_err(|error| OpenAiError::invalid_request(format!("serialize message: {error}")))?;
    let content = message
        .content
        .as_ref()
        .map(|content| message_content_to_generation_text(content, marker, media))
        .transpose()?;
    if let Some(object) = value.as_object_mut() {
        match content {
            Some(content) => {
                object.insert("content".to_string(), Value::String(content));
            }
            None => {
                object.insert("content".to_string(), Value::Null);
            }
        }
    }
    Ok(value)
}
