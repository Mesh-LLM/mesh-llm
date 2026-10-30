//! Reject unusable terminal tool responses before returning non-streaming success.
use super::{ParsedChatMessage, tool_calls_requested};
use serde_json::Value;
#[cfg(test)]
use skippy_inference_api::chat::ChatCompletionRequest;
use skippy_inference_api::{
    FinishReason, InferenceError, InferenceResult, generation::GenerationRequest,
};

pub(in crate::frontend) fn validate(
    request: &GenerationRequest,
    finish_reason: FinishReason,
    parsed: Option<&ParsedChatMessage>,
) -> InferenceResult<()> {
    if finish_reason != FinishReason::Stop || !tool_calls_requested(request) {
        return Ok(());
    }
    // Raw-output requests deliberately bypass chat parsing.
    let Some(parsed) = parsed else {
        return Ok(());
    };
    if let Some(calls) = parsed.tool_calls.as_ref().and_then(Value::as_array)
        && !calls.is_empty()
    {
        if calls.iter().all(|call| {
            call.pointer("/function/name")
                .and_then(Value::as_str)
                .is_some_and(|name| !name.trim().is_empty())
                && call
                    .pointer("/function/arguments")
                    .and_then(Value::as_str)
                    .and_then(|arguments| serde_json::from_str::<Value>(arguments).ok())
                    .is_some_and(|arguments| arguments.is_object())
        }) {
            return Ok(());
        }
        return Err(InferenceError::backend(
            "generation stopped with an invalid tool call",
        ));
    }
    if parsed
        .content
        .as_deref()
        .is_none_or(|content| content.trim().is_empty())
    {
        return Err(InferenceError::backend(
            "generation stopped without an answer or a complete tool call",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn request() -> GenerationRequest {
        serde_json::from_value::<ChatCompletionRequest>(
            json!({"model":"test","messages":[],"tools":[{
                "type":"function","function":{"name":"buzz__send","parameters":{"type":"object"}}
            }]}),
        )
        .unwrap()
        .into()
    }

    #[test]
    fn malformed_boolean_and_dropped_terminal_call_are_errors() {
        for arguments in [Some(r#"{"final":True}"#), Some(r#"{"final":"#), None] {
            let parsed = ParsedChatMessage {
                content: None,
                reasoning_content: Some("I will call the tool".into()),
                tool_calls: arguments.map(|arguments| {
                    json!([{"function":{
                        "name":"buzz__send","arguments":arguments
                    }}])
                }),
            };
            assert!(validate(&request(), FinishReason::Stop, Some(&parsed)).is_err());
            assert!(validate(&request(), FinishReason::Length, Some(&parsed)).is_ok());
        }
    }

    #[test]
    fn valid_tool_call_or_text_answer_and_raw_output_remain_allowed() {
        for parsed in [
            ParsedChatMessage {
                content: Some("No tool is needed".into()),
                reasoning_content: None,
                tool_calls: None,
            },
            ParsedChatMessage {
                content: None,
                reasoning_content: None,
                tool_calls: Some(json!([{
                    "function":{"name":"buzz__send","arguments":r#"{"final":true}"#}
                }])),
            },
        ] {
            assert!(validate(&request(), FinishReason::Stop, Some(&parsed)).is_ok());
        }
        assert!(validate(&request(), FinishReason::Stop, None).is_ok());
    }
}
