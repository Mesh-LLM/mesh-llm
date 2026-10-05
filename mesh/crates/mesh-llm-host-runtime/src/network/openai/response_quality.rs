#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResponseQualityFailure {
    EmptyAssistantOutput,
    LengthFinishReason,
    RepetitiveOutput,
}

impl ResponseQualityFailure {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::EmptyAssistantOutput => "empty_assistant_output",
            Self::LengthFinishReason => "length_finish_reason",
            Self::RepetitiveOutput => "repetitive_output",
        }
    }
}

pub(crate) fn failure_from_json_body(body: &[u8]) -> Option<ResponseQualityFailure> {
    let json = serde_json::from_slice::<serde_json::Value>(body).ok()?;
    if response_has_length_finish_reason(&json) {
        return Some(ResponseQualityFailure::LengthFinishReason);
    }
    if response_has_empty_assistant_output(&json) {
        return Some(ResponseQualityFailure::EmptyAssistantOutput);
    }
    if output_text_is_repetitive(&response_output_text(&json)) {
        return Some(ResponseQualityFailure::RepetitiveOutput);
    }
    None
}

fn response_has_length_finish_reason(json: &serde_json::Value) -> bool {
    chat_choices_have_length_finish_reason(json) || responses_incomplete_for_length(json)
}

fn chat_choices_have_length_finish_reason(json: &serde_json::Value) -> bool {
    json.get("choices")
        .and_then(serde_json::Value::as_array)
        .map(|choices| {
            choices.iter().any(|choice| {
                choice
                    .get("finish_reason")
                    .and_then(serde_json::Value::as_str)
                    .map(|reason| reason.eq_ignore_ascii_case("length"))
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false)
}

fn responses_incomplete_for_length(json: &serde_json::Value) -> bool {
    let incomplete_status = json
        .get("status")
        .and_then(serde_json::Value::as_str)
        .map(|status| status.eq_ignore_ascii_case("incomplete"))
        .unwrap_or(false);
    let length_reason = json
        .get("incomplete_details")
        .and_then(|details| details.get("reason"))
        .and_then(serde_json::Value::as_str)
        .map(|reason| {
            matches!(
                reason.to_ascii_lowercase().as_str(),
                "length" | "max_output_tokens" | "max_tokens"
            )
        })
        .unwrap_or(false);
    incomplete_status && length_reason
}

fn response_has_empty_assistant_output(json: &serde_json::Value) -> bool {
    chat_response_has_empty_assistant_output(json)
        || responses_body_has_empty_assistant_output(json)
}

fn chat_response_has_empty_assistant_output(json: &serde_json::Value) -> bool {
    let Some(choices) = json.get("choices").and_then(serde_json::Value::as_array) else {
        return false;
    };
    !choices.is_empty()
        && choices
            .iter()
            .all(|choice| !chat_choice_has_tool_call(choice) && chat_choice_text(choice).is_empty())
}

fn chat_choice_has_tool_call(choice: &serde_json::Value) -> bool {
    let message = choice.get("message");
    value_array_is_non_empty(message.and_then(|value| value.get("tool_calls")))
        || value_array_is_non_empty(choice.get("tool_calls"))
        || message
            .and_then(|value| value.get("function_call"))
            .map(|value| !value.is_null())
            .unwrap_or(false)
}

fn chat_choice_text(choice: &serde_json::Value) -> String {
    let mut text = String::new();
    if let Some(message) = choice.get("message") {
        append_openai_content_text(message.get("content"), &mut text);
    }
    append_openai_content_text(choice.get("text"), &mut text);
    text
}

fn responses_body_has_empty_assistant_output(json: &serde_json::Value) -> bool {
    let Some(output) = json.get("output").and_then(serde_json::Value::as_array) else {
        return false;
    };
    let mut saw_message = false;
    let mut saw_payload = false;
    for item in output {
        if responses_output_item_is_tool_call(item) {
            saw_payload = true;
        } else if responses_output_item_is_message(item) {
            saw_message = true;
            saw_payload |= !responses_output_item_text(item).is_empty();
        }
    }
    saw_message && !saw_payload
}

fn responses_output_item_is_tool_call(item: &serde_json::Value) -> bool {
    item.get("type")
        .and_then(serde_json::Value::as_str)
        .map(|kind| kind.contains("tool") || kind == "function_call")
        .unwrap_or(false)
}

fn responses_output_item_is_message(item: &serde_json::Value) -> bool {
    item.get("type")
        .and_then(serde_json::Value::as_str)
        .map(|kind| kind == "message")
        .unwrap_or(true)
}

fn responses_output_item_text(item: &serde_json::Value) -> String {
    let mut text = String::new();
    append_openai_content_text(item.get("content"), &mut text);
    append_openai_content_text(item.get("text"), &mut text);
    text
}

fn response_output_text(json: &serde_json::Value) -> String {
    let mut text = String::new();
    if let Some(choices) = json.get("choices").and_then(serde_json::Value::as_array) {
        for choice in choices {
            append_text(&mut text, &chat_choice_text(choice));
        }
    }
    if let Some(output) = json.get("output").and_then(serde_json::Value::as_array) {
        for item in output {
            append_text(&mut text, &responses_output_item_text(item));
        }
    }
    append_openai_content_text(json.get("output_text"), &mut text);
    text
}

fn append_openai_content_text(value: Option<&serde_json::Value>, output: &mut String) {
    match value {
        Some(serde_json::Value::String(text)) => append_text(output, text),
        Some(serde_json::Value::Array(items)) => {
            for item in items {
                append_openai_content_text(Some(item), output);
            }
        }
        Some(serde_json::Value::Object(map)) => {
            append_openai_content_text(map.get("text"), output);
            append_openai_content_text(map.get("content"), output);
        }
        _ => {}
    }
}

fn append_text(output: &mut String, text: &str) {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return;
    }
    if !output.is_empty() {
        output.push('\n');
    }
    output.push_str(trimmed);
}

fn value_array_is_non_empty(value: Option<&serde_json::Value>) -> bool {
    value
        .and_then(serde_json::Value::as_array)
        .map(|items| !items.is_empty())
        .unwrap_or(false)
}

fn output_text_is_repetitive(text: &str) -> bool {
    const MIN_WORDS: usize = 24;
    const MIN_REPEATS: usize = 4;
    const MAX_PATTERN_WORDS: usize = 8;

    let words = normalized_response_words(text);
    if words.len() < MIN_WORDS {
        return false;
    }
    (1..=MAX_PATTERN_WORDS)
        .any(|width| repeated_prefix_covers(&words, width, MIN_WORDS, MIN_REPEATS))
}

fn normalized_response_words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|word| {
            word.trim_matches(|ch: char| !ch.is_alphanumeric())
                .to_ascii_lowercase()
        })
        .filter(|word| !word.is_empty())
        .collect()
}

fn repeated_prefix_covers(
    words: &[String],
    width: usize,
    min_words: usize,
    min_repeats: usize,
) -> bool {
    if words.len() < width.saturating_mul(min_repeats) {
        return false;
    }
    let pattern = &words[..width];
    let mut matched_words = 0usize;
    for chunk in words.chunks(width) {
        if chunk.len() != width || chunk != pattern {
            break;
        }
        matched_words += width;
    }
    matched_words >= min_words
        && matched_words >= width.saturating_mul(min_repeats)
        && matched_words.saturating_mul(100) >= words.len().saturating_mul(80)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_empty_chat_output() {
        let body =
            br#"{"choices":[{"message":{"role":"assistant","content":""},"finish_reason":"stop"}]}"#;
        assert_eq!(
            failure_from_json_body(body),
            Some(ResponseQualityFailure::EmptyAssistantOutput)
        );
    }

    #[test]
    fn allows_tool_call_without_text() {
        let body = br#"{"choices":[{"message":{"role":"assistant","content":"","tool_calls":[{"type":"function","function":{"name":"lookup","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}"#;
        assert_eq!(failure_from_json_body(body), None);
    }

    #[test]
    fn detects_length_finish_reason() {
        let body = br#"{"choices":[{"message":{"role":"assistant","content":"partial"},"finish_reason":"length"}]}"#;
        assert_eq!(
            failure_from_json_body(body),
            Some(ResponseQualityFailure::LengthFinishReason)
        );
    }

    #[test]
    fn detects_responses_incomplete_for_max_tokens() {
        let body = br#"{"status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"output":[{"type":"message","content":[{"type":"output_text","text":"partial"}]}]}"#;
        assert_eq!(
            failure_from_json_body(body),
            Some(ResponseQualityFailure::LengthFinishReason)
        );
    }

    #[test]
    fn allows_responses_incomplete_for_non_length_reason() {
        let body = br#"{"status":"incomplete","incomplete_details":{"reason":"content_filter"},"output":[{"type":"message","content":[{"type":"output_text","text":"blocked"}]}]}"#;
        assert_eq!(failure_from_json_body(body), None);
    }

    #[test]
    fn allows_length_reason_without_incomplete_status() {
        let body = br#"{"status":"completed","incomplete_details":{"reason":"max_output_tokens"},"output":[{"type":"message","content":[{"type":"output_text","text":"complete"}]}]}"#;
        assert_eq!(failure_from_json_body(body), None);
    }

    #[test]
    fn detects_repetitive_output() {
        let repeated = "loop answer ".repeat(16);
        let body = serde_json::json!({
            "choices": [{
                "message": {"role": "assistant", "content": repeated},
                "finish_reason": "stop"
            }]
        })
        .to_string();

        assert_eq!(
            failure_from_json_body(body.as_bytes()),
            Some(ResponseQualityFailure::RepetitiveOutput)
        );
    }
}
