use std::collections::BTreeSet;

use serde_json::{Map, Value, json};

use crate::{chat::ChatCompletionResponse, common::FinishReason};

use super::{
    request_contract::{ParallelToolCalls, RawToolChoice},
    state::{GuardrailRequestOutcome, PreparedGuardrailRequest},
    tools::{MESH_EMIT_STRUCTURED_TOOL_NAME, MESH_RESPOND_TOOL_NAME},
};

const MAX_RESCUE_INPUT_BYTES: usize = 64 * 1024;
const MAX_JSON_CANDIDATES: usize = 32;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GuardrailParserStage {
    None,
    JsonExact,
    JsonFenced,
    JsonSubstring,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ClassifiedGuardrailResponse {
    pub category: GuardrailResponseCategory,
    pub parser_stage: GuardrailParserStage,
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
                parser_stage: GuardrailParserStage::None,
                visible_content,
                tool_calls: Some(tool_calls.clone()),
                synthetic_text: None,
                structured_payload: None,
                finish_reason,
            };
        }
        return classify_tool_call_value(
            prepared,
            tool_calls,
            GuardrailParserStage::None,
            finish_reason,
        );
    }

    if visible_content.is_none() {
        return empty_output();
    }

    if let Some(classified) = rescue_from_text(prepared, &stripped, finish_reason) {
        return classified;
    }

    if request_expects_guarded_contract(prepared) {
        return ClassifiedGuardrailResponse {
            category: GuardrailResponseCategory::MalformedToolText,
            parser_stage: GuardrailParserStage::None,
            visible_content: None,
            tool_calls: None,
            synthetic_text: None,
            structured_payload: None,
            finish_reason,
        };
    }

    ClassifiedGuardrailResponse {
        category: GuardrailResponseCategory::ValidText,
        parser_stage: GuardrailParserStage::None,
        visible_content,
        tool_calls: None,
        synthetic_text: None,
        structured_payload: None,
        finish_reason,
    }
}

pub(crate) fn strip_thinking_blocks(content: &str) -> String {
    let stripped_html = strip_tag_pairs(content, "<think>", "</think>");
    let stripped_brackets = strip_tag_pairs(&stripped_html, "[THINK]", "[/THINK]");
    stripped_brackets.trim().to_string()
}

fn strip_tag_pairs(content: &str, start_tag: &str, end_tag: &str) -> String {
    let mut remainder = content;
    let mut result = String::new();
    while let Some(start_index) = remainder.find(start_tag) {
        result.push_str(&remainder[..start_index]);
        let after_start = &remainder[start_index + start_tag.len()..];
        if let Some(end_index) = after_start.find(end_tag) {
            remainder = &after_start[end_index + end_tag.len()..];
        } else {
            remainder = &remainder[..start_index];
            break;
        }
    }
    result.push_str(remainder);
    result
}

fn rescue_from_text(
    prepared: &PreparedGuardrailRequest,
    content: &str,
    finish_reason: Option<FinishReason>,
) -> Option<ClassifiedGuardrailResponse> {
    for json_candidate in openai_json_candidates(content) {
        if let Ok(value) = serde_json::from_str::<Value>(&json_candidate.content) {
            let classified =
                classify_tool_call_value(prepared, &value, json_candidate.stage, finish_reason);
            if classified.category != GuardrailResponseCategory::MalformedToolText {
                return Some(classified.without_visible_content());
            }
        }
    }

    if let Some(value) = parse_bracket_args_tool_syntax(content) {
        let classified = classify_tool_call_value(
            prepared,
            &value,
            GuardrailParserStage::JsonSubstring,
            finish_reason,
        );
        if classified.category != GuardrailResponseCategory::MalformedToolText {
            return Some(classified.without_visible_content());
        }
    }

    if let Some(value) = parse_qwen_xml_syntax(content) {
        let classified = classify_tool_call_value(
            prepared,
            &value,
            GuardrailParserStage::JsonSubstring,
            finish_reason,
        );
        if classified.category != GuardrailResponseCategory::MalformedToolText {
            return Some(classified.without_visible_content());
        }
    }

    if let Some(value) = parse_granite_tool_call_syntax(content) {
        let classified = classify_tool_call_value(
            prepared,
            &value,
            GuardrailParserStage::JsonSubstring,
            finish_reason,
        );
        if classified.category != GuardrailResponseCategory::MalformedToolText {
            return Some(classified.without_visible_content());
        }
    }

    None
}

struct JsonCandidate {
    content: String,
    stage: GuardrailParserStage,
}

fn openai_json_candidates(content: &str) -> Vec<JsonCandidate> {
    let content = bounded_prefix(content, MAX_RESCUE_INPUT_BYTES);
    let mut candidates = Vec::new();
    push_candidate(
        &mut candidates,
        content.trim(),
        GuardrailParserStage::JsonExact,
    );

    for fenced in fenced_code_blocks(content) {
        if candidates.len() >= MAX_JSON_CANDIDATES {
            break;
        }
        push_candidate(
            &mut candidates,
            fenced.trim(),
            GuardrailParserStage::JsonFenced,
        );
    }

    for balanced in balanced_json_substrings(content) {
        if candidates.len() >= MAX_JSON_CANDIDATES {
            break;
        }
        push_candidate(
            &mut candidates,
            balanced.trim(),
            GuardrailParserStage::JsonSubstring,
        );
    }

    candidates
}

fn bounded_prefix(content: &str, max_bytes: usize) -> &str {
    if content.len() <= max_bytes {
        return content;
    }

    let mut end = max_bytes;
    while end > 0 && !content.is_char_boundary(end) {
        end -= 1;
    }
    &content[..end]
}

fn push_candidate(
    candidates: &mut Vec<JsonCandidate>,
    candidate: &str,
    stage: GuardrailParserStage,
) {
    if candidate.is_empty() {
        return;
    }
    if !candidates
        .iter()
        .any(|existing| existing.content == candidate)
    {
        candidates.push(JsonCandidate {
            content: candidate.to_string(),
            stage,
        });
    }
}

fn fenced_code_blocks(content: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut remainder = content;
    while let Some(open_index) = remainder.find("```") {
        let after_open = &remainder[open_index + 3..];
        let Some(close_index) = after_open.find("```") else {
            break;
        };
        let block = &after_open[..close_index];
        let block = block
            .strip_prefix("json\n")
            .or_else(|| block.strip_prefix("JSON\n"))
            .unwrap_or(block);
        blocks.push(block.to_string());
        remainder = &after_open[close_index + 3..];
    }
    blocks
}

fn balanced_json_substrings(content: &str) -> Vec<String> {
    let bytes = content.as_bytes();
    let mut candidates = Vec::new();
    for (index, byte) in bytes.iter().enumerate() {
        if candidates.len() >= MAX_JSON_CANDIDATES {
            break;
        }
        let closing = match byte {
            b'{' => b'}',
            b'[' => b']',
            _ => continue,
        };
        if let Some(end) = balanced_substring_end(bytes, index, *byte, closing) {
            candidates.push(content[index..=end].to_string());
        }
    }
    candidates
}

fn balanced_substring_end(bytes: &[u8], start: usize, opening: u8, closing: u8) -> Option<usize> {
    let mut depth = 0_u32;
    let mut in_string = false;
    let mut escaped = false;
    for (index, byte) in bytes.iter().copied().enumerate().skip(start) {
        if in_string {
            if escaped {
                escaped = false;
                continue;
            }
            match byte {
                b'\\' => escaped = true,
                b'"' => in_string = false,
                _ => {}
            }
            continue;
        }
        match byte {
            b'"' => in_string = true,
            _ if byte == opening => depth += 1,
            _ if byte == closing => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return Some(index);
                }
            }
            _ => {}
        }
    }
    None
}

fn parse_bracket_args_tool_syntax(content: &str) -> Option<Value> {
    let marker = "[ARGS]";
    if let Some(marker_index) = content.find(marker) {
        let name = content[..marker_index]
            .trim()
            .rsplit(|character: char| {
                !character.is_ascii_alphanumeric() && character != '_' && character != '-'
            })
            .next()?
            .trim();
        if name.is_empty() {
            return None;
        }
        let after_marker = content[marker_index + marker.len()..].trim_start();
        let json_text = first_balanced_object(after_marker)?;
        let arguments = serde_json::from_str::<Value>(&json_text).ok()?;
        return Some(json!({
            "name": name,
            "arguments": arguments,
        }));
    }

    parse_parenthesized_tool_call(content)
}

fn parse_qwen_xml_syntax(content: &str) -> Option<Value> {
    let function_prefix = "<function=";
    let start_index = content.find(function_prefix)?;
    let after_prefix = &content[start_index + function_prefix.len()..];
    let name_end = after_prefix.find('>')?;
    let name = after_prefix[..name_end]
        .trim()
        .trim_matches('"')
        .trim_matches('\'');
    if name.is_empty() {
        return None;
    }
    let body = &after_prefix[name_end + 1..];
    let function_end = body.find("</function>")?;
    let parameters_body = &body[..function_end];
    let mut arguments = Map::new();
    let mut remainder = parameters_body;

    while let Some(parameter_start) = remainder.find("<parameter=") {
        let after_parameter = &remainder[parameter_start + "<parameter=".len()..];
        let name_end = after_parameter.find('>')?;
        let parameter_name = after_parameter[..name_end]
            .trim()
            .trim_matches('"')
            .trim_matches('\'');
        if parameter_name.is_empty() {
            return None;
        }
        let parameter_body = &after_parameter[name_end + 1..];
        let value_end = parameter_body.find("</parameter>")?;
        let value = parameter_body[..value_end].trim();
        let parsed_value = serde_json::from_str::<Value>(value)
            .unwrap_or_else(|_| Value::String(value.to_string()));
        arguments.insert(parameter_name.to_string(), parsed_value);
        remainder = &parameter_body[value_end + "</parameter>".len()..];
    }

    if arguments.is_empty() {
        return None;
    }

    Some(json!({
        "name": name,
        "arguments": Value::Object(arguments),
    }))
}

fn parse_granite_tool_call_syntax(content: &str) -> Option<Value> {
    let start_tag = "<tool_call>";
    let end_tag = "</tool_call>";
    let start_index = content.find(start_tag)?;
    let after_start = &content[start_index + start_tag.len()..];
    let end_index = after_start.find(end_tag)?;
    serde_json::from_str(after_start[..end_index].trim()).ok()
}

fn first_balanced_object(content: &str) -> Option<String> {
    let start = content.find('{')?;
    let end = balanced_substring_end(content.as_bytes(), start, b'{', b'}')?;
    Some(content[start..=end].to_string())
}

fn parse_parenthesized_tool_call(content: &str) -> Option<Value> {
    let open_paren = content.find('(')?;
    let name = content[..open_paren]
        .trim()
        .rsplit(|character: char| {
            !character.is_ascii_alphanumeric() && character != '_' && character != '-'
        })
        .next()?
        .trim();
    if name.is_empty() {
        return None;
    }
    let after_open = content[open_paren + 1..].trim_start();
    let json_text = first_balanced_object(after_open)?;
    let after_json = after_open[json_text.len()..].trim_start();
    if !after_json.starts_with(')') {
        return None;
    }
    let arguments = serde_json::from_str::<Value>(&json_text).ok()?;
    Some(json!({
        "name": name,
        "arguments": arguments,
    }))
}

fn classify_tool_call_value(
    prepared: &PreparedGuardrailRequest,
    value: &Value,
    parser_stage: GuardrailParserStage,
    finish_reason: Option<FinishReason>,
) -> ClassifiedGuardrailResponse {
    if let Some(classified) =
        classify_direct_structured_payload(prepared, value, parser_stage, finish_reason)
    {
        return classified;
    }

    let allowed_real_tools = allowed_real_tool_names(prepared);
    let allowed_backend_tools = allowed_backend_tool_names(prepared);
    let raw_tool_calls = match raw_tool_calls_from_value(value) {
        Some(tool_calls) if !tool_calls.is_empty() => tool_calls,
        _ => {
            return ClassifiedGuardrailResponse {
                category: GuardrailResponseCategory::MalformedToolText,
                parser_stage,
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
                    parser_stage,
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
                    parser_stage,
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
                    parser_stage,
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
            parser_stage,
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
            parser_stage,
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
            parser_stage,
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
            parser_stage,
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
                parser_stage,
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
                parser_stage,
                visible_content: None,
                tool_calls: None,
                synthetic_text: None,
                structured_payload: None,
                finish_reason,
            };
        };
        return ClassifiedGuardrailResponse {
            category: GuardrailResponseCategory::ValidSyntheticRespond,
            parser_stage,
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
                parser_stage,
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
            parser_stage,
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
        parser_stage,
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
    parser_stage: GuardrailParserStage,
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
        parser_stage,
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
        parser_stage: GuardrailParserStage::None,
        visible_content: None,
        tool_calls: None,
        synthetic_text: None,
        structured_payload: None,
        finish_reason: None,
    }
}

impl ClassifiedGuardrailResponse {
    fn without_visible_content(mut self) -> Self {
        self.visible_content = None;
        self
    }
}
