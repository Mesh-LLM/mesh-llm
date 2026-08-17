use std::collections::BTreeSet;

pub(crate) use mesh_llm_guardrails::strip_thinking_blocks;
use serde_json::{Map, Value, json};

use crate::{chat::ChatCompletionResponse, common::FinishReason};

use super::{
    request_contract::{ParallelToolCalls, RawToolChoice},
    state::{GuardrailRequestOutcome, PreparedGuardrailRequest},
    tools::{MESH_EMIT_STRUCTURED_TOOL_NAME, MESH_RESPOND_TOOL_NAME},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GuardrailResponseCategory {
    ValidText,
    ValidToolCalls,
    ValidSyntheticRespond,
    ValidSyntheticStructured,
    MalformedToolText,
    UnknownTool,
    InvalidToolArguments,
    InvalidStructuredPayload,
    MixedTerminalAndTool,
    ToolCallsNotAllowed,
    TooManyToolCalls,
    EmptyOutput,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ClassifiedGuardrailResponse {
    pub category: GuardrailResponseCategory,
    pub visible_content: Option<String>,
    pub tool_calls: Option<Value>,
    pub synthetic_text: Option<String>,
    pub structured_payload: Option<Value>,
    pub finish_reason: Option<FinishReason>,
}

#[derive(Debug, Clone, PartialEq)]
struct ParsedToolCall {
    name: String,
    arguments: Map<String, Value>,
}

pub(crate) fn classify_response(
    prepared: &PreparedGuardrailRequest,
    response: &ChatCompletionResponse,
) -> ClassifiedGuardrailResponse {
    let Some(choice) = response.choices.first() else {
        return empty_output();
    };
    let content = choice.message.content.clone().unwrap_or_default();
    let stripped = strip_thinking_blocks(&content);
    let visible_content = normalized_visible_content(&stripped);
    let finish_reason = choice.finish_reason;

    if let Some(tool_calls) = &choice.message.tool_calls {
        if visible_content.is_some() {
            return ClassifiedGuardrailResponse {
                category: GuardrailResponseCategory::MixedTerminalAndTool,
                visible_content,
                tool_calls: Some(tool_calls.clone()),
                synthetic_text: None,
                structured_payload: None,
                finish_reason,
            };
        }
        return classify_tool_call_value(prepared, tool_calls, finish_reason);
    }

    if visible_content.is_none() {
        return empty_output();
    }

    if request_expects_guarded_contract(prepared) {
        return ClassifiedGuardrailResponse {
            category: GuardrailResponseCategory::MalformedToolText,
            visible_content: None,
            tool_calls: None,
            synthetic_text: None,
            structured_payload: None,
            finish_reason,
        };
    }

    ClassifiedGuardrailResponse {
        category: GuardrailResponseCategory::ValidText,
        visible_content,
        tool_calls: None,
        synthetic_text: None,
        structured_payload: None,
        finish_reason,
    }
}

fn classify_tool_call_value(
    prepared: &PreparedGuardrailRequest,
    value: &Value,
    finish_reason: Option<FinishReason>,
) -> ClassifiedGuardrailResponse {
    if let Some(classified) = classify_direct_structured_payload(prepared, value, finish_reason) {
        return classified;
    }

    let allowed_real_tools = allowed_real_tool_names(prepared);
    let allowed_backend_tools = allowed_backend_tool_names(prepared);
    let raw_tool_calls = match raw_tool_calls_from_value(value) {
        Some(tool_calls) if !tool_calls.is_empty() => tool_calls,
        _ => {
            return ClassifiedGuardrailResponse {
                category: GuardrailResponseCategory::MalformedToolText,
                visible_content: None,
                tool_calls: None,
                synthetic_text: None,
                structured_payload: None,
                finish_reason,
            };
        }
    };

    let mut parsed_calls = Vec::new();
    for tool_call in raw_tool_calls {
        match parse_tool_call(tool_call, &allowed_backend_tools) {
            ParsedToolCallStatus::Valid(tool_call) => parsed_calls.push(tool_call),
            ParsedToolCallStatus::UnknownTool => {
                return ClassifiedGuardrailResponse {
                    category: GuardrailResponseCategory::UnknownTool,
                    visible_content: None,
                    tool_calls: None,
                    synthetic_text: None,
                    structured_payload: None,
                    finish_reason,
                };
            }
            ParsedToolCallStatus::InvalidArguments { structured_payload } => {
                return ClassifiedGuardrailResponse {
                    category: if structured_payload {
                        GuardrailResponseCategory::InvalidStructuredPayload
                    } else {
                        GuardrailResponseCategory::InvalidToolArguments
                    },
                    visible_content: None,
                    tool_calls: None,
                    synthetic_text: None,
                    structured_payload: None,
                    finish_reason,
                };
            }
            ParsedToolCallStatus::Malformed => {
                return ClassifiedGuardrailResponse {
                    category: GuardrailResponseCategory::MalformedToolText,
                    visible_content: None,
                    tool_calls: None,
                    synthetic_text: None,
                    structured_payload: None,
                    finish_reason,
                };
            }
        }
    }

    let has_synthetic_respond = parsed_calls
        .iter()
        .any(|tool_call| tool_call.name == MESH_RESPOND_TOOL_NAME);
    let has_synthetic_structured = parsed_calls
        .iter()
        .any(|tool_call| tool_call.name == MESH_EMIT_STRUCTURED_TOOL_NAME);
    let has_real_tools = parsed_calls
        .iter()
        .any(|tool_call| allowed_real_tools.contains(tool_call.name.as_str()));

    if request_disables_tool_calls(prepared) {
        return ClassifiedGuardrailResponse {
            category: GuardrailResponseCategory::ToolCallsNotAllowed,
            visible_content: None,
            tool_calls: Some(normalized_tool_calls(&parsed_calls)),
            synthetic_text: None,
            structured_payload: None,
            finish_reason,
        };
    }

    if let Some(forced_name) = prepared.state.request_contract.forced_tool_name()
        && parsed_calls
            .iter()
            .any(|tool_call| tool_call.name != forced_name)
    {
        return ClassifiedGuardrailResponse {
            category: GuardrailResponseCategory::UnknownTool,
            visible_content: None,
            tool_calls: Some(normalized_tool_calls(&parsed_calls)),
            synthetic_text: None,
            structured_payload: None,
            finish_reason,
        };
    }

    if matches!(
        prepared.state.request_contract.parallel_tool_calls,
        ParallelToolCalls::Disabled
    ) && parsed_calls.len() > 1
    {
        return ClassifiedGuardrailResponse {
            category: GuardrailResponseCategory::TooManyToolCalls,
            visible_content: None,
            tool_calls: Some(normalized_tool_calls(&parsed_calls)),
            synthetic_text: None,
            structured_payload: None,
            finish_reason,
        };
    }

    if (has_synthetic_respond && (has_real_tools || has_synthetic_structured))
        || (has_synthetic_structured && has_real_tools)
    {
        return ClassifiedGuardrailResponse {
            category: GuardrailResponseCategory::MixedTerminalAndTool,
            visible_content: None,
            tool_calls: Some(normalized_tool_calls(&parsed_calls)),
            synthetic_text: None,
            structured_payload: None,
            finish_reason,
        };
    }

    if has_synthetic_respond {
        if parsed_calls.len() != 1 {
            return ClassifiedGuardrailResponse {
                category: GuardrailResponseCategory::MixedTerminalAndTool,
                visible_content: None,
                tool_calls: Some(normalized_tool_calls(&parsed_calls)),
                synthetic_text: None,
                structured_payload: None,
                finish_reason,
            };
        }
        let tool_call = &parsed_calls[0];
        let Some(message) = tool_call.arguments.get("message").and_then(Value::as_str) else {
            return ClassifiedGuardrailResponse {
                category: GuardrailResponseCategory::InvalidToolArguments,
                visible_content: None,
                tool_calls: None,
                synthetic_text: None,
                structured_payload: None,
                finish_reason,
            };
        };
        return ClassifiedGuardrailResponse {
            category: GuardrailResponseCategory::ValidSyntheticRespond,
            visible_content: None,
            tool_calls: Some(normalized_tool_calls(&parsed_calls)),
            synthetic_text: Some(message.to_string()),
            structured_payload: None,
            finish_reason: Some(FinishReason::ToolCalls),
        };
    }

    if has_synthetic_structured {
        if parsed_calls.len() != 1 {
            return ClassifiedGuardrailResponse {
                category: GuardrailResponseCategory::MixedTerminalAndTool,
                visible_content: None,
                tool_calls: Some(normalized_tool_calls(&parsed_calls)),
                synthetic_text: None,
                structured_payload: None,
                finish_reason,
            };
        }
        let structured_payload = Value::Object(parsed_calls[0].arguments.clone());
        let valid_payload = prepared
            .state
            .request_contract
            .structured_output_spec()
            .is_some_and(|spec| spec.validate_payload(&structured_payload).is_ok());
        return ClassifiedGuardrailResponse {
            category: if valid_payload {
                GuardrailResponseCategory::ValidSyntheticStructured
            } else {
                GuardrailResponseCategory::InvalidStructuredPayload
            },
            visible_content: None,
            tool_calls: if valid_payload {
                Some(normalized_tool_calls(&parsed_calls))
            } else {
                None
            },
            synthetic_text: None,
            structured_payload: if valid_payload {
                Some(structured_payload)
            } else {
                None
            },
            finish_reason: if valid_payload {
                Some(FinishReason::ToolCalls)
            } else {
                finish_reason
            },
        };
    }

    ClassifiedGuardrailResponse {
        category: GuardrailResponseCategory::ValidToolCalls,
        visible_content: None,
        tool_calls: Some(normalized_tool_calls(&parsed_calls)),
        synthetic_text: None,
        structured_payload: None,
        finish_reason: Some(FinishReason::ToolCalls),
    }
}

fn classify_direct_structured_payload(
    prepared: &PreparedGuardrailRequest,
    value: &Value,
    finish_reason: Option<FinishReason>,
) -> Option<ClassifiedGuardrailResponse> {
    if prepared.state.request_contract.has_real_tools() {
        return None;
    }
    let spec = prepared.state.request_contract.structured_output_spec()?;
    value.as_object()?;
    let valid_payload = spec.validate_payload(value).is_ok();
    Some(ClassifiedGuardrailResponse {
        category: if valid_payload {
            GuardrailResponseCategory::ValidSyntheticStructured
        } else {
            GuardrailResponseCategory::InvalidStructuredPayload
        },
        visible_content: None,
        tool_calls: None,
        synthetic_text: None,
        structured_payload: if valid_payload {
            Some(value.clone())
        } else {
            None
        },
        finish_reason,
    })
}

fn raw_tool_calls_from_value(value: &Value) -> Option<Vec<&Value>> {
    match value {
        Value::Array(entries) => Some(entries.iter().collect()),
        Value::Object(object) => object
            .get("tool_calls")
            .and_then(Value::as_array)
            .map(|entries| entries.iter().collect())
            .or_else(|| Some(vec![value])),
        _ => None,
    }
}

enum ParsedToolCallStatus {
    Valid(ParsedToolCall),
    UnknownTool,
    InvalidArguments { structured_payload: bool },
    Malformed,
}

fn parse_tool_call(
    value: &Value,
    allowed_backend_tools: &BTreeSet<String>,
) -> ParsedToolCallStatus {
    let Some((name, arguments_value)) = extract_tool_name_and_arguments(value) else {
        return ParsedToolCallStatus::Malformed;
    };
    if !allowed_backend_tools.contains(name) {
        return ParsedToolCallStatus::UnknownTool;
    }
    let Some(arguments) = normalize_arguments(arguments_value) else {
        return ParsedToolCallStatus::InvalidArguments {
            structured_payload: name == MESH_EMIT_STRUCTURED_TOOL_NAME,
        };
    };
    ParsedToolCallStatus::Valid(ParsedToolCall {
        name: name.to_string(),
        arguments,
    })
}

fn extract_tool_name_and_arguments(value: &Value) -> Option<(&str, &Value)> {
    let object = value.as_object()?;
    let nested_function = object.get("function").and_then(Value::as_object);
    let name = nested_function
        .and_then(|function| function.get("name"))
        .and_then(Value::as_str)
        .or_else(|| object.get("name").and_then(Value::as_str))
        .or_else(|| object.get("function").and_then(Value::as_str))?;
    let arguments = nested_function
        .and_then(|function| function.get("arguments"))
        .or_else(|| object.get("arguments"))?;
    Some((name, arguments))
}

fn normalize_arguments(arguments: &Value) -> Option<Map<String, Value>> {
    match arguments {
        Value::Object(arguments) => Some(arguments.clone()),
        Value::String(arguments) => serde_json::from_str::<Value>(arguments)
            .ok()?
            .as_object()
            .cloned(),
        _ => None,
    }
}

fn normalized_tool_calls(parsed_calls: &[ParsedToolCall]) -> Value {
    Value::Array(
        parsed_calls
            .iter()
            .enumerate()
            .map(|(index, tool_call)| {
                json!({
                    "id": format!("call_guardrail_{index}"),
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": serde_json::to_string(&Value::Object(tool_call.arguments.clone()))
                            .expect("tool arguments serialize to JSON")
                    }
                })
            })
            .collect(),
    )
}

fn allowed_real_tool_names(prepared: &PreparedGuardrailRequest) -> BTreeSet<String> {
    prepared
        .state
        .request_contract
        .tool_names()
        .map(ToString::to_string)
        .collect()
}

fn allowed_backend_tool_names(prepared: &PreparedGuardrailRequest) -> BTreeSet<String> {
    let mut allowed = allowed_real_tool_names(prepared);
    if let GuardrailRequestOutcome::Guarded { backend_request } = &prepared.outcome {
        let backend_contract = super::request_contract::from_request(backend_request);
        allowed.extend(backend_contract.tool_names().map(ToString::to_string));
    }
    allowed
}

fn request_expects_guarded_contract(prepared: &PreparedGuardrailRequest) -> bool {
    matches!(prepared.outcome, GuardrailRequestOutcome::Guarded { .. })
        && (prepared.state.request_contract.has_real_tools()
            || prepared.state.request_contract.requests_structured_output())
        && !request_disables_tool_calls(prepared)
        && !prepared.state.last_message_is_tool_result
}

fn request_disables_tool_calls(prepared: &PreparedGuardrailRequest) -> bool {
    matches!(
        prepared.state.request_contract.tool_choice,
        RawToolChoice::None
    ) && !prepared.state.request_contract.requests_structured_output()
}

fn normalized_visible_content(content: &str) -> Option<String> {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn empty_output() -> ClassifiedGuardrailResponse {
    ClassifiedGuardrailResponse {
        category: GuardrailResponseCategory::EmptyOutput,
        visible_content: None,
        tool_calls: None,
        synthetic_text: None,
        structured_payload: None,
        finish_reason: None,
    }
}
