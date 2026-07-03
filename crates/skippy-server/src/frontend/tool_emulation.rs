//! Serving-side tool-call emulation for models whose chat template does not
//! support native tool calling.
//!
//! Small / non-tool-trained models handle the OpenAI `tools` field poorly: the
//! chat template either ignores the schemas or the model loops re-issuing the
//! same call. The serving node is the right place to fix this because it is the
//! only party that knows the actual **chat template capability** of the loaded
//! model — clients only know a model name.
//!
//! This module is a port of the approach proven in goose's local-inference
//! provider (text-convention emulation + tolerant parser). When a request
//! carries `tools` for a template that is not tool-capable, we:
//!
//! 1. **Adapt the request**: strip `tools`/`tool_choice` from the template
//!    input, inject a compact instruction (tool name + description + compact
//!    parameter schema) describing the `TOOL_CALL {json}` convention, and
//!    rewrite conversation history so the template never sees tool roles.
//! 2. **Parse the response**: scan output for `TOOL_CALL` lines and convert
//!    them into real OpenAI `tool_calls` with `finish_reason: "tool_calls"`.
//!    Parsing is tolerant of surrounding prose and `<think>` blocks.
//!
//! Detection is gated on template capability, not model size: a lean
//! tool-trained model (e.g. Qwen3-0.6B) keeps native tool calling and sees
//! **zero behavior change**.

use std::collections::BTreeMap;

use openai_frontend::{ChatMessage, MessageContent};
use serde_json::{Map, Value};

/// Marker a model is instructed to emit, on its own line, to request a tool
/// call. The remainder of the line is a JSON object `{"name", "arguments"}`.
pub(super) const TOOL_CALL_MARKER: &str = "TOOL_CALL";

/// Returns true when the loaded model's chat template natively supports tool
/// calling.
///
/// This is the mesh-llm analogue of goose's
/// `template_result_supports_native_tool_calling`. goose reads llama.cpp's
/// `parse_tool_calls` + parser fields, but in mesh-llm's patched runtime
/// `parse_tool_calls` only reflects whether the request carried tools (it is
/// true for every tools request) and `chat_parser` is always a non-empty PEG
/// structure, so neither field distinguishes a tool-capable template.
///
/// The signal that does distinguish them is `grammar_triggers`: when the jinja
/// template natively describes tool calls, applying it with tools yields a
/// tool-call grammar trigger (e.g. `<tool_call>`). A template with no native
/// tool support (for example SmolLM2-135M) yields an empty `grammar_triggers`
/// list. We treat a non-empty `grammar_triggers` as native tool-call support.
pub(super) fn template_supports_native_tool_calls(metadata_json: &str) -> bool {
    let Ok(metadata) = serde_json::from_str::<Value>(metadata_json) else {
        return false;
    };
    metadata
        .get("grammar_triggers")
        .and_then(Value::as_array)
        .is_some_and(|triggers| !triggers.is_empty())
}

/// Extracts the function objects from an OpenAI `tools` array value.
fn tool_functions(tools: &Value) -> Vec<&Map<String, Value>> {
    tools
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|tool| {
            tool.get("function")
                .and_then(Value::as_object)
                .or_else(|| tool.as_object())
        })
        .collect()
}

/// Renders a compact parameter schema (property name + type) for the emulation
/// instruction. Full JSON schemas overwhelm tiny models; name+type is the
/// signal that keeps them from hallucinating argument names.
fn compact_parameter_schema(function: &Map<String, Value>) -> Option<String> {
    let parameters = function
        .get("parameters")
        .or_else(|| function.get("input_schema"))?;
    let properties = parameters.get("properties").and_then(Value::as_object)?;
    let required: Vec<&str> = parameters
        .get("required")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect();
    if properties.is_empty() {
        return Some("{}".to_string());
    }
    let mut rendered = Vec::new();
    for (name, schema) in properties {
        let ty = schema
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("string");
        let flag = if required.contains(&name.as_str()) {
            ""
        } else {
            "?"
        };
        rendered.push(format!("{name}{flag}: {ty}"));
    }
    Some(format!("{{{}}}", rendered.join(", ")))
}

/// Builds the emulation system instruction: the calling convention plus a
/// compact description of each available tool. Returns `None` when the tools
/// value carries no usable functions.
pub(super) fn build_emulation_instruction(tools: &Value) -> Option<String> {
    let functions = tool_functions(tools);
    if functions.is_empty() {
        return None;
    }
    let mut instruction = String::new();
    instruction.push_str(
        "# Tool calling\n\n\
         You can call tools. To call a tool, emit a single line beginning with ",
    );
    instruction.push_str(TOOL_CALL_MARKER);
    instruction.push_str(
        " followed by a JSON object with \"name\" and \"arguments\":\n\n\
         TOOL_CALL {\"name\": \"the_tool_name\", \"arguments\": {\"arg\": \"value\"}}\n\n\
         Emit the line exactly, on its own line, with valid JSON. Use the exact \
         argument names shown below. Call a tool only when it is needed; \
         otherwise answer normally.\n\n\
         ## Available tools\n\n",
    );
    for function in functions {
        let Some(name) = function.get("name").and_then(Value::as_str) else {
            continue;
        };
        let description = function
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim();
        instruction.push_str("- ");
        instruction.push_str(name);
        if !description.is_empty() {
            instruction.push_str(": ");
            instruction.push_str(description);
        }
        if let Some(schema) = compact_parameter_schema(function) {
            instruction.push_str("\n  arguments: ");
            instruction.push_str(&schema);
        }
        instruction.push('\n');
    }
    Some(instruction)
}

/// Flattens message content into plain text for history rewriting.
fn content_text(content: Option<&MessageContent>) -> String {
    content
        .and_then(openai_frontend::message_content_to_text)
        .unwrap_or_default()
}

/// Renders an assistant message's `tool_calls` (from history) back into the
/// `TOOL_CALL {json}` text convention so the template never sees tool roles.
fn assistant_tool_calls_as_text(extra: &BTreeMap<String, Value>) -> Option<String> {
    let calls = extra.get("tool_calls").and_then(Value::as_array)?;
    let mut lines = Vec::new();
    for call in calls {
        let function = call.get("function").and_then(Value::as_object);
        let name = function
            .and_then(|function| function.get("name"))
            .and_then(Value::as_str)
            .unwrap_or_default();
        if name.is_empty() {
            continue;
        }
        let arguments = function
            .and_then(|function| function.get("arguments"))
            .map(arguments_to_json_value)
            .unwrap_or(Value::Object(Map::new()));
        let payload = serde_json::json!({ "name": name, "arguments": arguments });
        lines.push(format!("{TOOL_CALL_MARKER} {payload}"));
    }
    if lines.is_empty() {
        None
    } else {
        Some(lines.join("\n"))
    }
}

/// Tool-call `arguments` on the wire may be a JSON string or an object. Return
/// a JSON value either way.
fn arguments_to_json_value(arguments: &Value) -> Value {
    match arguments {
        Value::String(text) => serde_json::from_str::<Value>(text).unwrap_or(Value::Object(
            std::iter::once(("_raw".to_string(), Value::String(text.clone()))).collect(),
        )),
        other => other.clone(),
    }
}

/// Rewrites conversation history so a non-tool-capable template never sees
/// `tool` roles or assistant `tool_calls`:
/// - assistant `tool_calls` become assistant text using the `TOOL_CALL`
///   convention (so the model sees its own prior calls in the same form),
/// - `role: "tool"` messages become `user` messages prefixed with
///   `Tool result:`.
///
/// The emulation instruction is prepended to (or merged into) the first system
/// message, or inserted as a new leading system message.
pub(super) fn rewrite_history_for_emulation(
    messages: &[ChatMessage],
    instruction: &str,
) -> Vec<ChatMessage> {
    let mut rewritten: Vec<ChatMessage> = Vec::with_capacity(messages.len() + 1);
    let mut instruction_placed = false;

    for message in messages {
        match message.role.as_str() {
            "system" if !instruction_placed => {
                instruction_placed = true;
                let existing = content_text(message.content.as_ref());
                let merged = if existing.trim().is_empty() {
                    instruction.to_string()
                } else {
                    format!("{existing}\n\n{instruction}")
                };
                rewritten.push(plain_message("system", merged));
            }
            "tool" => {
                let result = content_text(message.content.as_ref());
                rewritten.push(plain_message("user", format!("Tool result: {result}")));
            }
            "assistant" => {
                let mut text = content_text(message.content.as_ref());
                if let Some(tool_text) = assistant_tool_calls_as_text(&message.extra) {
                    if text.trim().is_empty() {
                        text = tool_text;
                    } else {
                        text = format!("{text}\n{tool_text}");
                    }
                }
                rewritten.push(plain_message("assistant", text));
            }
            role => {
                rewritten.push(plain_message(role, content_text(message.content.as_ref())));
            }
        }
    }

    if !instruction_placed {
        rewritten.insert(0, plain_message("system", instruction.to_string()));
    }

    rewritten
}

/// Builds a minimal `ChatMessage` with plain text content and no extra fields,
/// so the downstream template sees only `role` + `content`.
fn plain_message(role: &str, content: String) -> ChatMessage {
    ChatMessage {
        role: role.to_string(),
        content: Some(MessageContent::Text(content)),
        extra: BTreeMap::new(),
    }
}

/// Result of parsing an emulated response: any prose content plus the parsed
/// tool calls (already in OpenAI wire shape, arguments as a JSON string).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct EmulatedParse {
    pub content: Option<String>,
    pub tool_calls: Vec<Value>,
}

/// Scans generated text for `TOOL_CALL {json}` lines and converts them to
/// OpenAI tool-call objects, tolerant of surrounding prose and `<think>`
/// blocks. Text that is not part of a recognized tool call is preserved as
/// `content`. Only calls whose name is in `allowed_names` are kept (empty means
/// allow all).
pub(super) fn parse_emulated_tool_calls(text: &str, allowed_names: &[String]) -> EmulatedParse {
    let scannable = strip_think_blocks(text);
    let mut content_lines: Vec<String> = Vec::new();
    let mut tool_calls: Vec<Value> = Vec::new();

    for line in scannable.lines() {
        if let Some(call) = parse_tool_call_line(line, allowed_names) {
            tool_calls.push(call);
        } else {
            content_lines.push(line.to_string());
        }
    }

    let content = {
        let joined = content_lines.join("\n");
        let trimmed = joined.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    };

    EmulatedParse {
        content,
        tool_calls,
    }
}

/// Removes `<think>...</think>` spans so a reasoning model's scratchpad never
/// triggers or hides a tool call. An unterminated `<think>` drops the rest.
fn strip_think_blocks(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("<think>") {
        out.push_str(&rest[..start]);
        let after = &rest[start + "<think>".len()..];
        match after.find("</think>") {
            Some(end) => rest = &after[end + "</think>".len()..],
            None => return out,
        }
    }
    out.push_str(rest);
    out
}

/// Parses a single line as a tool call if it carries the marker and a JSON
/// object with a `name`. Returns the OpenAI tool-call object (without an id;
/// downstream assembly assigns `call_mesh_*` ids).
fn parse_tool_call_line(line: &str, allowed_names: &[String]) -> Option<Value> {
    let trimmed = line.trim();
    let after_marker = trimmed.strip_prefix(TOOL_CALL_MARKER)?;
    let json_part = after_marker.trim_start().trim_start_matches(':').trim();
    let payload = serde_json::from_str::<Value>(json_part).ok()?;
    let name = payload.get("name").and_then(Value::as_str)?.trim();
    if name.is_empty() {
        return None;
    }
    if !allowed_names.is_empty() && !allowed_names.iter().any(|allowed| allowed == name) {
        return None;
    }
    let arguments = payload
        .get("arguments")
        .cloned()
        .unwrap_or(Value::Object(Map::new()));
    let arguments_string = match &arguments {
        Value::String(text) => text.clone(),
        other => serde_json::to_string(other).unwrap_or_else(|_| "{}".to_string()),
    };
    Some(serde_json::json!({
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments_string,
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.to_string(),
            content: Some(MessageContent::Text(content.to_string())),
            extra: BTreeMap::new(),
        }
    }

    fn tools_value() -> Value {
        serde_json::json!([
            {
                "type": "function",
                "function": {
                    "name": "shell",
                    "description": "Run a shell command",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"},
                            "timeout": {"type": "integer"}
                        },
                        "required": ["command"]
                    }
                }
            }
        ])
    }

    #[test]
    fn native_detection_uses_grammar_triggers() {
        // Tool-capable template: applying it yields a tool-call grammar trigger.
        assert!(template_supports_native_tool_calls(
            r#"{"chat_format": 2, "grammar_triggers": [{"type": 1, "value": "<tool_call>"}]}"#
        ));
        // Non-tool-capable template (e.g. SmolLM2-135M): empty grammar triggers.
        assert!(!template_supports_native_tool_calls(
            r#"{"chat_format": 2, "grammar_triggers": []}"#
        ));
        // Missing field or non-array is treated as non-native.
        assert!(!template_supports_native_tool_calls(
            r#"{"chat_format": 2}"#
        ));
        assert!(!template_supports_native_tool_calls("not json"));
    }

    #[test]
    fn instruction_includes_compact_schema_with_required_flags() {
        let instruction = build_emulation_instruction(&tools_value()).unwrap();
        assert!(instruction.contains("TOOL_CALL"));
        assert!(instruction.contains("- shell: Run a shell command"));
        // Required arg has no marker; optional arg is flagged with `?`.
        assert!(instruction.contains("command: string"));
        assert!(instruction.contains("timeout?: integer"));
    }

    #[test]
    fn instruction_none_without_functions() {
        assert!(build_emulation_instruction(&serde_json::json!([])).is_none());
    }

    #[test]
    fn parse_single_tool_call() {
        let parse = parse_emulated_tool_calls(
            "Let me check.\nTOOL_CALL {\"name\": \"shell\", \"arguments\": {\"command\": \"ls\"}}",
            &[],
        );
        assert_eq!(parse.content.as_deref(), Some("Let me check."));
        assert_eq!(parse.tool_calls.len(), 1);
        let call = &parse.tool_calls[0];
        assert_eq!(call["function"]["name"], "shell");
        // arguments are serialized as a JSON string per OpenAI wire shape.
        let args: Value =
            serde_json::from_str(call["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["command"], "ls");
    }

    #[test]
    fn parse_ignores_marker_inside_think_block() {
        let parse = parse_emulated_tool_calls(
            "<think>\nTOOL_CALL {\"name\": \"shell\", \"arguments\": {}}\n</think>\nHello",
            &[],
        );
        assert!(parse.tool_calls.is_empty());
        assert_eq!(parse.content.as_deref(), Some("Hello"));
    }

    #[test]
    fn parse_tolerates_colon_after_marker() {
        let parse = parse_emulated_tool_calls(
            "TOOL_CALL: {\"name\": \"shell\", \"arguments\": {\"command\": \"pwd\"}}",
            &[],
        );
        assert_eq!(parse.tool_calls.len(), 1);
        assert_eq!(parse.tool_calls[0]["function"]["name"], "shell");
    }

    #[test]
    fn parse_filters_disallowed_names() {
        let parse = parse_emulated_tool_calls(
            "TOOL_CALL {\"name\": \"evil\", \"arguments\": {}}",
            &["shell".to_string()],
        );
        assert!(parse.tool_calls.is_empty());
        // Disallowed call falls back to content rather than vanishing.
        assert!(parse.content.is_some());
    }

    #[test]
    fn parse_plain_prose_has_no_tool_calls() {
        let parse = parse_emulated_tool_calls("Just a normal answer.", &[]);
        assert!(parse.tool_calls.is_empty());
        assert_eq!(parse.content.as_deref(), Some("Just a normal answer."));
    }

    #[test]
    fn parse_multiple_tool_calls() {
        let parse = parse_emulated_tool_calls(
            "TOOL_CALL {\"name\": \"shell\", \"arguments\": {\"command\": \"ls\"}}\n\
             TOOL_CALL {\"name\": \"shell\", \"arguments\": {\"command\": \"pwd\"}}",
            &[],
        );
        assert_eq!(parse.tool_calls.len(), 2);
        assert!(parse.content.is_none());
    }

    #[test]
    fn parse_malformed_json_is_treated_as_prose() {
        let parse = parse_emulated_tool_calls("TOOL_CALL {not valid json}", &[]);
        assert!(parse.tool_calls.is_empty());
        assert!(parse.content.is_some());
    }

    #[test]
    fn history_rewrite_merges_instruction_into_system() {
        let messages = vec![msg("system", "You are helpful."), msg("user", "hi")];
        let rewritten = rewrite_history_for_emulation(&messages, "INSTRUCTION");
        assert_eq!(rewritten.len(), 2);
        let system =
            openai_frontend::message_content_to_text(rewritten[0].content.as_ref().unwrap())
                .unwrap();
        assert!(system.contains("You are helpful."));
        assert!(system.contains("INSTRUCTION"));
    }

    #[test]
    fn history_rewrite_inserts_system_when_absent() {
        let messages = vec![msg("user", "hi")];
        let rewritten = rewrite_history_for_emulation(&messages, "INSTRUCTION");
        assert_eq!(rewritten.len(), 2);
        assert_eq!(rewritten[0].role, "system");
        assert_eq!(rewritten[1].role, "user");
    }

    #[test]
    fn history_rewrite_converts_tool_role_to_user() {
        let messages = vec![msg("tool", "exit 0")];
        let rewritten = rewrite_history_for_emulation(&messages, "I");
        // system instruction + rewritten tool message
        let tool_msg = rewritten.iter().find(|m| m.role == "user").unwrap();
        let text =
            openai_frontend::message_content_to_text(tool_msg.content.as_ref().unwrap()).unwrap();
        assert!(text.starts_with("Tool result: exit 0"));
    }

    #[test]
    fn history_rewrite_converts_assistant_tool_calls_to_text() {
        let mut extra = BTreeMap::new();
        extra.insert(
            "tool_calls".to_string(),
            serde_json::json!([{
                "type": "function",
                "function": {"name": "shell", "arguments": "{\"command\": \"ls\"}"}
            }]),
        );
        let assistant = ChatMessage {
            role: "assistant".to_string(),
            content: None,
            extra,
        };
        let rewritten = rewrite_history_for_emulation(&[assistant], "I");
        let asst = rewritten.iter().find(|m| m.role == "assistant").unwrap();
        let text =
            openai_frontend::message_content_to_text(asst.content.as_ref().unwrap()).unwrap();
        assert!(text.contains("TOOL_CALL"));
        assert!(text.contains("\"name\":\"shell\""));
        assert!(text.contains("\"command\":\"ls\""));
        // Rewritten message must not leak the raw tool_calls field to the template.
        assert!(!asst.extra.contains_key("tool_calls"));
    }
}
