//! Bounded translation of one terminal Chat Completions choice into Responses
//! output items. Worker-owned output is treated as untrusted at this seam.

use serde_json::{Map, Value};

use super::{
    resource::{ResponseStatus, function_call_item, message_item, new_item_id, reasoning_item},
    tools::{MAX_TOOL_CALLS, ToolOutputViolation},
};

pub(super) fn chat_completion_output(
    chat: &Value,
) -> Result<(Vec<Value>, ResponseStatus), ToolOutputViolation> {
    let choice = chat
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first());
    if choice
        .and_then(|choice| serde_json::to_vec(choice).ok())
        .is_none_or(|bytes| bytes.len() > 8 * 1024 * 1024)
    {
        return Err(ToolOutputViolation::new(
            "response_output_too_large",
            "upstream response output exceeded the active-state limit",
        ));
    }
    let message = choice.and_then(|choice| choice.get("message"));
    let status = ResponseStatus::from_finish_reason(
        choice
            .and_then(|choice| choice.get("finish_reason"))
            .and_then(Value::as_str),
    );
    let mut output = Vec::new();
    let reasoning = message
        .and_then(|message| message.get("reasoning_content"))
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty());
    if let Some(reasoning) = reasoning {
        output.push(reasoning_item(&new_item_id("rs"), reasoning));
    }
    let text = message.map(chat_message_text).unwrap_or_default();
    let calls = terminal_tool_calls(message)?;
    if !text.is_empty() || calls.is_empty() {
        let logprobs = choice
            .and_then(|choice| choice.get("logprobs"))
            .filter(|logprobs| !logprobs.is_null())
            .cloned();
        let text_status = if calls.is_empty() {
            status.item_status()
        } else {
            "completed"
        };
        output.push(message_item(
            &new_item_id("msg"),
            text_status,
            &text,
            logprobs,
            message
                .and_then(|message| message.get("phase"))
                .and_then(Value::as_str),
        ));
    }
    for (position, call) in calls.iter().enumerate() {
        let call_status = if position + 1 == calls.len() {
            status.item_status()
        } else {
            "completed"
        };
        let (call_id, name, arguments) = terminal_call_fields(call)?;
        output.push(function_call_item(
            &new_item_id("fc"),
            call_status,
            call_id,
            name,
            arguments,
        ));
    }
    Ok((output, status))
}

fn terminal_tool_calls(message: Option<&Value>) -> Result<&[Value], ToolOutputViolation> {
    let Some(calls) = message.and_then(|message| message.get("tool_calls")) else {
        return Ok(&[]);
    };
    let calls = calls.as_array().ok_or_else(malformed_tool_call)?;
    if calls.len() > MAX_TOOL_CALLS {
        return Err(ToolOutputViolation::new(
            "tool_call_limit_exceeded",
            "upstream response exceeded the tool-call limit",
        ));
    }
    Ok(calls)
}

fn terminal_call_fields(call: &Value) -> Result<(&str, &str, &str), ToolOutputViolation> {
    let call = call.as_object().ok_or_else(malformed_tool_call)?;
    if call
        .get("type")
        .is_some_and(|kind| kind.as_str() != Some("function"))
    {
        return Err(malformed_tool_call());
    }
    let function = match (
        call.get("function"),
        call.get("name"),
        call.get("arguments"),
    ) {
        (Some(function), None, None) => function.as_object().ok_or_else(malformed_tool_call)?,
        (None, Some(_), Some(_)) => call,
        _ => return Err(malformed_tool_call()),
    };
    Ok((
        required_call_field(call, "id")?,
        required_call_field(function, "name")?,
        required_call_field(function, "arguments")?,
    ))
}

fn required_call_field<'a>(
    object: &'a Map<String, Value>,
    field: &str,
) -> Result<&'a str, ToolOutputViolation> {
    object
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty() || field == "arguments")
        .ok_or_else(malformed_tool_call)
}

pub(super) fn malformed_tool_call() -> ToolOutputViolation {
    ToolOutputViolation::new(
        "upstream_tool_call_invalid",
        "upstream response contained a malformed function call",
    )
}

fn chat_message_text(message: &Value) -> String {
    match message.get("content") {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .collect(),
        _ => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn output_orders_reasoning_text_and_calls() {
        let (output, status) = chat_completion_output(&json!({
            "choices": [{"message": {
                "reasoning_content": "think",
                "content": "say",
                "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": "{}"}}],
            }, "finish_reason": "tool_calls"}],
        }))
        .unwrap();
        assert_eq!(status, ResponseStatus::Completed);
        let kinds: Vec<_> = output.iter().map(|item| item["type"].clone()).collect();
        assert_eq!(kinds, ["reasoning", "message", "function_call"]);
        assert_eq!(output[0]["content"][0]["type"], "reasoning_text");
    }

    #[test]
    fn output_preserves_valid_assistant_phase() {
        let (output, _) = chat_completion_output(&json!({
            "choices": [{"message": {
                "content": "done",
                "phase": "final_answer"
            }, "finish_reason": "stop"}],
        }))
        .unwrap();
        assert_eq!(output[0]["phase"], "final_answer");
    }

    #[test]
    fn malformed_tool_calls_fail_closed() {
        for call in [
            json!({"type":"function","function":{"name":"safe","arguments":"{}"}}),
            json!({"id":"c","type":"function","function":{"arguments":"{}"}}),
            json!({"id":"c","type":"function","function":{"name":"safe","arguments":{}}}),
            json!({"id":"c","type":"function","name":"safe","arguments":"{}","function":{"name":"safe","arguments":"{}"}}),
        ] {
            let error = chat_completion_output(&json!({
                "choices":[{"message":{"tool_calls":[call]},"finish_reason":"tool_calls"}]
            }))
            .unwrap_err();
            assert_eq!(error.code, "upstream_tool_call_invalid");
        }
    }
}
