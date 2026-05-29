use crate::{
    chat::{ChatCompletionRequest, ChatCompletionResponse},
    common::FinishReason,
    errors::OpenAiError,
    hooks::inject_text_into_chat_messages,
};

use super::{
    errors::validation_failed_error,
    policy::{GuardrailPolicy, RetryExhaustionMode},
    request_contract::RawToolChoice,
    rescue::{
        ClassifiedGuardrailResponse, GuardrailParserStage, GuardrailResponseCategory,
        strip_thinking_blocks,
    },
    state::{GuardrailRequestOutcome, PreparedGuardrailRequest},
    tools::is_reserved_tool_name,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GuardrailContractKind {
    Tool,
    Structured,
}

pub(crate) fn max_attempts(prepared: &PreparedGuardrailRequest, policy: &GuardrailPolicy) -> u8 {
    retry_budget(prepared, policy).saturating_add(1)
}

pub(crate) fn build_retry_request(
    prepared: &PreparedGuardrailRequest,
    attempt_index: u8,
    classified: &ClassifiedGuardrailResponse,
) -> ChatCompletionRequest {
    let mut retry_request = prepared_backend_request(prepared);
    inject_text_into_chat_messages(
        &mut retry_request.messages,
        retry_nudge(prepared, classified),
    );
    if attempt_index > 0 {
        retry_request.prompt_cache_key = None;
    }
    retry_request
}

pub(crate) fn sanitize_success_response(
    policy: &GuardrailPolicy,
    response: &ChatCompletionResponse,
    classified: &ClassifiedGuardrailResponse,
) -> Option<ChatCompletionResponse> {
    match classified.category {
        GuardrailResponseCategory::ValidToolCalls => {
            if classified.parser_stage == GuardrailParserStage::None {
                Some(response.clone())
            } else {
                Some(rewrite_response(
                    response,
                    None,
                    None,
                    classified.tool_calls.clone(),
                    Some(FinishReason::ToolCalls),
                ))
            }
        }
        GuardrailResponseCategory::ValidSyntheticRespond => Some(rewrite_response(
            response,
            None,
            classified.synthetic_text.clone(),
            None,
            Some(FinishReason::Stop),
        )),
        GuardrailResponseCategory::ValidSyntheticStructured => Some(rewrite_response(
            response,
            Some(policy),
            structured_payload_text(classified),
            None,
            Some(FinishReason::Stop),
        )),
        GuardrailResponseCategory::ValidText => Some(rewrite_response(
            response,
            Some(policy),
            classified.visible_content.clone(),
            None,
            classified.finish_reason,
        )),
        GuardrailResponseCategory::MalformedToolText
        | GuardrailResponseCategory::UnknownTool
        | GuardrailResponseCategory::InvalidToolArguments
        | GuardrailResponseCategory::InvalidStructuredPayload
        | GuardrailResponseCategory::MixedTerminalAndTool
        | GuardrailResponseCategory::ToolCallsNotAllowed
        | GuardrailResponseCategory::TooManyToolCalls
        | GuardrailResponseCategory::EmptyOutput => None,
    }
}

pub(crate) fn should_retry(classified: &ClassifiedGuardrailResponse) -> bool {
    matches!(
        classified.category,
        GuardrailResponseCategory::MalformedToolText
            | GuardrailResponseCategory::UnknownTool
            | GuardrailResponseCategory::InvalidToolArguments
            | GuardrailResponseCategory::InvalidStructuredPayload
            | GuardrailResponseCategory::MixedTerminalAndTool
            | GuardrailResponseCategory::ToolCallsNotAllowed
            | GuardrailResponseCategory::TooManyToolCalls
            | GuardrailResponseCategory::EmptyOutput
    )
}

pub(crate) fn exhaustion_result(
    policy: &GuardrailPolicy,
    response: ChatCompletionResponse,
    classified: &ClassifiedGuardrailResponse,
) -> Result<ChatCompletionResponse, OpenAiError> {
    match policy.retry_exhaustion_mode {
        RetryExhaustionMode::Error => Err(validation_failed_error()),
        RetryExhaustionMode::PassLastText => pass_last_text_response(policy, &response, classified)
            .ok_or_else(validation_failed_error),
    }
}

fn retry_budget(prepared: &PreparedGuardrailRequest, policy: &GuardrailPolicy) -> u8 {
    match request_kind(prepared) {
        GuardrailContractKind::Tool => policy.max_tool_retries,
        GuardrailContractKind::Structured => policy.max_structured_retries,
    }
}

fn request_kind(prepared: &PreparedGuardrailRequest) -> GuardrailContractKind {
    if prepared.state.request_contract.requests_structured_output() {
        GuardrailContractKind::Structured
    } else {
        GuardrailContractKind::Tool
    }
}

fn prepared_backend_request(prepared: &PreparedGuardrailRequest) -> ChatCompletionRequest {
    match &prepared.outcome {
        GuardrailRequestOutcome::Guarded { backend_request } => (**backend_request).clone(),
        _ => unreachable!("retry logic only applies to guarded requests"),
    }
}

fn retry_nudge(
    prepared: &PreparedGuardrailRequest,
    classified: &ClassifiedGuardrailResponse,
) -> String {
    let contract = if request_disables_tool_calls(prepared) {
        "Reply with normal assistant text and do not make a tool call."
    } else {
        match request_kind(prepared) {
            GuardrailContractKind::Tool => {
                "Reply with exactly one valid tool call using only the provided tools and valid JSON arguments."
            }
            GuardrailContractKind::Structured => {
                "Reply with exactly one valid structured-output tool call whose JSON arguments satisfy the requested schema."
            }
        }
    };
    let failure = match classified.category {
        GuardrailResponseCategory::MalformedToolText => {
            "Your previous reply was plain text instead of a valid guarded call."
        }
        GuardrailResponseCategory::UnknownTool => {
            "Your previous reply used a tool name that was not allowed."
        }
        GuardrailResponseCategory::InvalidToolArguments => {
            "Your previous reply used invalid JSON tool arguments."
        }
        GuardrailResponseCategory::InvalidStructuredPayload => {
            "Your previous reply used invalid structured JSON arguments."
        }
        GuardrailResponseCategory::MixedTerminalAndTool => {
            "Your previous reply mixed terminal text with guarded tool output."
        }
        GuardrailResponseCategory::ToolCallsNotAllowed => {
            "Your previous reply used a tool call even though tool calls were disabled for this turn."
        }
        GuardrailResponseCategory::TooManyToolCalls => {
            "Your previous reply used more tool calls than allowed for this turn."
        }
        GuardrailResponseCategory::EmptyOutput => {
            "Your previous reply was empty after hidden reasoning was stripped."
        }
        GuardrailResponseCategory::ValidText
        | GuardrailResponseCategory::ValidToolCalls
        | GuardrailResponseCategory::ValidSyntheticRespond
        | GuardrailResponseCategory::ValidSyntheticStructured => {
            "Your previous reply did not satisfy the guarded contract."
        }
    };
    format!("{failure} {contract} Do not add extra text.\n\n")
}

fn request_disables_tool_calls(prepared: &PreparedGuardrailRequest) -> bool {
    matches!(
        prepared.state.request_contract.tool_choice,
        RawToolChoice::None
    ) && !prepared.state.request_contract.requests_structured_output()
}

fn structured_payload_text(classified: &ClassifiedGuardrailResponse) -> Option<String> {
    classified
        .structured_payload
        .as_ref()
        .and_then(|payload| serde_json::to_string(payload).ok())
}

fn pass_last_text_response(
    policy: &GuardrailPolicy,
    response: &ChatCompletionResponse,
    classified: &ClassifiedGuardrailResponse,
) -> Option<ChatCompletionResponse> {
    let content = representable_last_text(policy, classified)?;
    Some(rewrite_response(
        response,
        Some(policy),
        Some(content),
        None,
        Some(FinishReason::Stop),
    ))
}

fn representable_last_text(
    policy: &GuardrailPolicy,
    classified: &ClassifiedGuardrailResponse,
) -> Option<String> {
    classified.synthetic_text.clone().or_else(|| {
        classified
            .visible_content
            .as_ref()
            .and_then(|content| sanitized_text(policy, content))
    })
}

fn sanitized_text(policy: &GuardrailPolicy, content: &str) -> Option<String> {
    let stripped = strip_thinking_blocks(content);
    let trimmed = stripped.trim();
    if trimmed.is_empty() || contains_synthetic_marker(policy, trimmed) {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn contains_synthetic_marker(policy: &GuardrailPolicy, content: &str) -> bool {
    content
        .split(|character: char| {
            !(character.is_ascii_alphanumeric() || character == '_' || character == '-')
        })
        .any(|token| {
            !token.is_empty() && is_reserved_tool_name(token, &policy.reserved_tool_prefix)
        })
}

fn rewrite_response(
    response: &ChatCompletionResponse,
    policy: Option<&GuardrailPolicy>,
    content: Option<String>,
    tool_calls: Option<serde_json::Value>,
    finish_reason: Option<FinishReason>,
) -> ChatCompletionResponse {
    let mut rewritten = response.clone();
    let Some(choice) = rewritten.choices.first_mut() else {
        return rewritten;
    };
    choice.message.content = content.or_else(|| {
        choice
            .message
            .content
            .as_ref()
            .and_then(|existing| policy.and_then(|policy| sanitized_text(policy, existing)))
    });
    choice.message.tool_calls = tool_calls;
    choice.finish_reason = finish_reason;
    rewritten
}
