//! Agent loops (Codex) resend the conversation as typed Responses items,
//! including the function calls the model made and their outputs. Chat
//! completions carry those as assistant `tool_calls` and `tool` messages.
use super::{translate_responses_input_message, translate_responses_message_content};
use crate::InferenceError;
use serde_json::Value;

fn responses_item_type(item: &Value) -> Option<&str> {
    item.get("type").and_then(Value::as_str)
}

pub(super) fn is_responses_tool_item(item: &Value) -> bool {
    matches!(
        responses_item_type(item),
        Some("function_call" | "function_call_output")
    )
}

/// Translate a typed Responses conversation into chat messages: consecutive
/// `function_call` items become one assistant message's `tool_calls`, each
/// `function_call_output` a `tool` message. Reasoning items carry no chat
/// equivalent and are dropped.
pub(super) fn translate_responses_conversation_items(
    items: &[Value],
) -> Result<Vec<Value>, InferenceError> {
    let mut messages: Vec<Value> = Vec::new();
    for item in items {
        match responses_item_type(item) {
            Some("function_call") => push_responses_function_call(&mut messages, item),
            Some("function_call_output") => {
                messages.push(responses_function_call_output_message(item)?);
            }
            Some("reasoning") => {}
            _ => messages.push(Value::Object(translate_responses_input_message(item)?)),
        }
    }
    Ok(messages)
}

fn push_responses_function_call(messages: &mut Vec<Value>, item: &Value) {
    // A call without arguments still sends a JSON object, which backends parse.
    let arguments = match item.get("arguments") {
        Some(Value::String(arguments)) if !arguments.is_empty() => arguments.clone(),
        Some(Value::String(_)) | Some(Value::Null) | None => "{}".to_string(),
        Some(other) => other.to_string(),
    };
    let call_id = item
        .get("call_id")
        .or_else(|| item.get("id"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    let tool_call = serde_json::json!({
        "id": call_id,
        "type": "function",
        "function": {
            "name": item.get("name").and_then(Value::as_str).unwrap_or("tool"),
            "arguments": arguments,
        },
    });
    // Parallel calls share the assistant turn that precedes their outputs.
    if let Some(last) = messages.last_mut().and_then(Value::as_object_mut)
        && last.get("role").and_then(Value::as_str) == Some("assistant")
    {
        let calls = last
            .entry("tool_calls")
            .or_insert_with(|| Value::Array(Vec::new()));
        if let Some(calls) = calls.as_array_mut() {
            calls.push(tool_call);
            return;
        }
    }
    messages.push(serde_json::json!({
        "role": "assistant",
        "content": Value::Null,
        "tool_calls": [tool_call],
    }));
}

fn responses_function_call_output_message(item: &Value) -> Result<Value, InferenceError> {
    let call_id = item.get("call_id").and_then(Value::as_str).ok_or_else(|| {
        InferenceError::invalid_request("function_call_output is missing call_id")
            .with_param("input")
    })?;
    let content = match item.get("output") {
        Some(Value::String(output)) => Value::String(output.clone()),
        Some(output @ (Value::Array(_) | Value::Object(_))) => {
            translate_responses_message_content(output)?
        }
        Some(other) => Value::String(other.to_string()),
        None => Value::String(String::new()),
    };
    Ok(serde_json::json!({
        "role": "tool",
        "tool_call_id": call_id,
        "content": content,
    }))
}

#[cfg(test)]
mod tests {
    use super::super::translate_responses_input_to_messages;

    #[test]
    fn responses_tool_loop_items_translate_to_chat_tool_messages() {
        let input = serde_json::json!([
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "cat it"}]},
            {"type": "reasoning", "summary": []},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Running."}]},
            {"type": "function_call", "call_id": "call_a", "name": "exec_command", "arguments": "{\"cmd\":\"cat a\"}"},
            {"type": "function_call", "call_id": "call_b", "name": "exec_command", "arguments": "{\"cmd\":\"cat b\"}"},
            {"type": "function_call_output", "call_id": "call_a", "output": "A"},
            {"type": "function_call_output", "call_id": "call_b", "output": [{"type": "input_text", "text": "B"}]},
        ]);
        let messages = translate_responses_input_to_messages(&input).unwrap();
        assert_eq!(messages.len(), 4);
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[1]["role"], "assistant");
        assert_eq!(messages[1]["content"], "Running.");
        assert_eq!(messages[1]["tool_calls"][0]["id"], "call_a");
        assert_eq!(
            messages[1]["tool_calls"][1]["function"]["name"],
            "exec_command"
        );
        assert_eq!(
            messages[2],
            serde_json::json!({"role": "tool", "tool_call_id": "call_a", "content": "A"})
        );
        assert_eq!(messages[3]["content"], "B");
    }

    /// A call the model made without arguments goes back to the backend as
    /// an empty JSON object, never as an empty string.
    #[test]
    fn responses_function_call_without_arguments_translates_to_an_empty_object() {
        let input = serde_json::json!([
            {"type": "message", "role": "user", "content": "what time is it"},
            {"type": "function_call", "call_id": "call_t", "name": "now", "arguments": ""},
            {"type": "function_call_output", "call_id": "call_t", "output": "12:00"},
            {"type": "function_call", "call_id": "call_u", "name": "uptime"},
            {"type": "function_call_output", "call_id": "call_u", "output": "3d"},
        ]);
        let messages = translate_responses_input_to_messages(&input).unwrap();
        assert_eq!(messages[1]["tool_calls"][0]["function"]["arguments"], "{}");
        assert_eq!(messages[3]["tool_calls"][0]["function"]["arguments"], "{}");
    }
}
