use serde_json::Value;

pub(super) fn validate_answer(content: &str, expected: &str) -> Result<String, String> {
    let mut visible = String::new();
    let mut remaining = content;
    while let Some(start) = remaining.find("<think>") {
        visible.push_str(&remaining[..start]);
        let hidden = &remaining[start + "<think>".len()..];
        let Some(end) = hidden.find("</think>") else {
            remaining = "";
            break;
        };
        remaining = &hidden[end + "</think>".len()..];
    }
    visible.push_str(remaining);
    if visible.contains(expected) {
        Ok(format!("final answer included {expected}"))
    } else {
        Err(format!("missing expected answer: {expected}"))
    }
}

pub(super) fn message(response: &Value) -> Result<&Value, String> {
    first_choice(response)?
        .get("message")
        .filter(|value| value.is_object())
        .ok_or_else(|| "first choice did not contain a message object".into())
}

pub(super) fn first_choice(response: &Value) -> Result<&Value, String> {
    response
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .filter(|choice| choice.is_object())
        .ok_or_else(|| "response had no valid choices".into())
}

pub(super) fn final_content(response: &Value) -> Result<&str, String> {
    let message = message(response)?;
    if message
        .get("tool_calls")
        .and_then(Value::as_array)
        .is_some_and(|calls| !calls.is_empty())
    {
        return Err("continuation returned another tool call".into());
    }
    message
        .get("content")
        .and_then(Value::as_str)
        .ok_or_else(|| "message content was not a string".into())
}

pub(super) fn stream_content(events: &[Value]) -> Result<String, String> {
    let mut content = String::new();
    let mut saw_choice = false;
    for event in events {
        let Some(choices) = event.get("choices").and_then(Value::as_array) else {
            continue;
        };
        for choice in choices {
            saw_choice |= choice.is_object();
            let Some(delta) = choice.get("delta") else {
                continue;
            };
            if delta
                .get("tool_calls")
                .and_then(Value::as_array)
                .is_some_and(|calls| !calls.is_empty())
            {
                return Err("continuation returned another streamed tool call".into());
            }
            if let Some(text) = delta.get("content").and_then(Value::as_str) {
                content.push_str(text);
            }
        }
    }
    if !saw_choice {
        return Err("stream returned no choices".into());
    }
    Ok(content)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn reasoning_only_values_do_not_qualify_an_answer() {
        assert!(validate_answer("<think>STABILITY_OK</think>wrong", "STABILITY_OK").is_err());
        assert!(validate_answer("<think>STREAM_OK", "STREAM_OK").is_err());
        assert!(validate_answer("<think>scratch</think>STABILITY_OK", "STABILITY_OK").is_ok());
        assert!(validate_answer("answer signal-7429", "signal-7429").is_ok());
    }

    #[test]
    fn continuation_rejects_more_tools_and_preserves_fragmented_content() {
        let response = json!({"choices":[{"message":{"content":"signal-7429","tool_calls":[{}]}}]});
        assert!(final_content(&response).is_err());
        let events = vec![
            json!({"choices":[{"delta":{"content":"signal-"}}]}),
            json!({"choices":[{"delta":{"content":"7429"}}]}),
            json!({"choices":[],"usage":{"completion_tokens":2}}),
        ];
        assert_eq!(stream_content(&events).unwrap(), "signal-7429");
        assert!(stream_content(&[json!({"choices":[{"delta":{"tool_calls":[{}]}}]})]).is_err());
        assert!(stream_content(&[json!({"choices":[]})]).is_err());
    }
}
