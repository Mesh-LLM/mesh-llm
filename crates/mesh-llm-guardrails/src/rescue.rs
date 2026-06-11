use std::collections::BTreeSet;

use serde_json::{Map, Value, json};

use crate::tools::{extract_tool_name_and_arguments, normalize_tool_arguments};

const MAX_RESCUE_INPUT_BYTES: usize = 64 * 1024;
const MAX_JSON_CANDIDATES: usize = 32;

#[derive(Debug, Clone, PartialEq)]
pub struct ParsedToolCall {
    pub name: String,
    pub arguments: Map<String, Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolCallParseError {
    Malformed,
    UnknownTool,
    InvalidArguments,
}

pub fn strip_thinking_blocks(content: &str) -> String {
    let stripped_html = strip_tag_pairs(content, "<think>", "</think>");
    let stripped_brackets = strip_tag_pairs(&stripped_html, "[THINK]", "[/THINK]");
    stripped_brackets.trim().to_string()
}

pub fn parse_tool_call_value(
    value: &Value,
    allowed_tools: &[String],
) -> Result<Vec<ParsedToolCall>, ToolCallParseError> {
    let raw_tool_calls = match raw_tool_calls_from_value(value) {
        Some(tool_calls) if !tool_calls.is_empty() => tool_calls,
        _ => return Err(ToolCallParseError::Malformed),
    };
    let allowed_tools = allowed_tools
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let mut parsed_calls = Vec::new();
    for tool_call in raw_tool_calls {
        parsed_calls.push(parse_one_tool_call(tool_call, &allowed_tools)?);
    }
    Ok(parsed_calls)
}

pub fn rescue_tool_call_from_text(
    content: &str,
    allowed_tools: &[String],
) -> Result<Vec<ParsedToolCall>, ToolCallParseError> {
    let content = strip_thinking_blocks(content);
    let mut last_error = ToolCallParseError::Malformed;
    for candidate in tool_call_candidates(&content) {
        match parse_tool_call_value(&candidate, allowed_tools) {
            Ok(parsed) => return Ok(parsed),
            Err(error) => last_error = more_specific_error(last_error, error),
        }
    }
    Err(last_error)
}

fn more_specific_error(
    current: ToolCallParseError,
    next: ToolCallParseError,
) -> ToolCallParseError {
    match (current, next) {
        (ToolCallParseError::InvalidArguments, _) | (_, ToolCallParseError::InvalidArguments) => {
            ToolCallParseError::InvalidArguments
        }
        (ToolCallParseError::UnknownTool, _) | (_, ToolCallParseError::UnknownTool) => {
            ToolCallParseError::UnknownTool
        }
        _ => ToolCallParseError::Malformed,
    }
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

fn tool_call_candidates(content: &str) -> Vec<Value> {
    let mut candidates = Vec::new();
    for json_candidate in json_candidates(content) {
        if let Ok(value) = serde_json::from_str::<Value>(&json_candidate) {
            candidates.push(value);
        }
    }
    if let Some(value) = parse_bracket_args_tool_syntax(content) {
        candidates.push(value);
    }
    if let Some(value) = parse_qwen_xml_syntax(content) {
        candidates.push(value);
    }
    if let Some(value) = parse_granite_tool_call_syntax(content) {
        candidates.push(value);
    }
    candidates
}

fn json_candidates(content: &str) -> Vec<String> {
    let content = bounded_prefix(content, MAX_RESCUE_INPUT_BYTES);
    let mut candidates = Vec::new();
    push_candidate(&mut candidates, content.trim());
    for fenced in fenced_code_blocks(content) {
        if candidates.len() >= MAX_JSON_CANDIDATES {
            break;
        }
        push_candidate(&mut candidates, fenced.trim());
    }
    for balanced in balanced_json_substrings(content) {
        if candidates.len() >= MAX_JSON_CANDIDATES {
            break;
        }
        push_candidate(&mut candidates, balanced.trim());
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

fn push_candidate(candidates: &mut Vec<String>, candidate: &str) {
    if !candidate.is_empty() && !candidates.iter().any(|existing| existing == candidate) {
        candidates.push(candidate.to_string());
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
        let name = trailing_tool_name(&content[..marker_index])?;
        let after_marker = content[marker_index + marker.len()..].trim_start();
        let json_text = first_balanced_object(after_marker)?;
        let arguments = serde_json::from_str::<Value>(&json_text).ok()?;
        return Some(json!({ "name": name, "arguments": arguments }));
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
    let mut arguments = Map::new();
    let mut remainder = &body[..function_end];
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
    Some(json!({ "name": name, "arguments": Value::Object(arguments) }))
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
    let name = trailing_tool_name(&content[..open_paren])?;
    let after_open = content[open_paren + 1..].trim_start();
    let json_text = first_balanced_object(after_open)?;
    let after_json = after_open[json_text.len()..].trim_start();
    if !after_json.starts_with(')') {
        return None;
    }
    let arguments = serde_json::from_str::<Value>(&json_text).ok()?;
    Some(json!({ "name": name, "arguments": arguments }))
}

fn trailing_tool_name(content: &str) -> Option<&str> {
    let name = content
        .trim()
        .rsplit(|character: char| {
            !character.is_ascii_alphanumeric() && character != '_' && character != '-'
        })
        .next()?
        .trim();
    (!name.is_empty()).then_some(name)
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

fn parse_one_tool_call(
    value: &Value,
    allowed_tools: &BTreeSet<&str>,
) -> Result<ParsedToolCall, ToolCallParseError> {
    let Some((name, arguments_value)) = extract_tool_name_and_arguments(value) else {
        return Err(ToolCallParseError::Malformed);
    };
    if !allowed_tools.is_empty() && !allowed_tools.contains(name) {
        return Err(ToolCallParseError::UnknownTool);
    }
    let Some(arguments) = normalize_tool_arguments(arguments_value) else {
        return Err(ToolCallParseError::InvalidArguments);
    };
    Ok(ParsedToolCall {
        name: name.to_string(),
        arguments,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rescues_qwen_xml_tool_call() {
        let calls = rescue_tool_call_from_text(
            r#"<function=read_file><parameter=path>README.md</parameter></function>"#,
            &[],
        )
        .unwrap();

        assert_eq!(calls[0].name, "read_file");
        assert_eq!(calls[0].arguments["path"], "README.md");
    }

    #[test]
    fn rescues_parenthesized_tool_call() {
        let calls = rescue_tool_call_from_text(r#"read_file({"path":"README.md"})"#, &[]).unwrap();

        assert_eq!(calls[0].name, "read_file");
        assert_eq!(calls[0].arguments["path"], "README.md");
    }

    #[test]
    fn rejects_unknown_tool_when_catalog_is_present() {
        let allowed_tools = vec!["read_file".to_string()];
        let error = rescue_tool_call_from_text(
            r#"{"name":"write_file","arguments":{"path":"README.md"}}"#,
            &allowed_tools,
        )
        .unwrap_err();

        assert_eq!(error, ToolCallParseError::UnknownTool);
    }
}
