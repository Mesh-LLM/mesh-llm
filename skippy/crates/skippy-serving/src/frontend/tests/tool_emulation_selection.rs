use super::support::tool_request;
use crate::frontend::prompting::{
    PromptRenderSelection, RenderedChatPrompt, select_tool_prompt_render,
};
use serde_json::json;
use skippy_inference_api::{
    InferenceError, MessageContent,
    generation::{GenerationMessage, GenerationRequest},
};

fn tool_loop_request() -> GenerationRequest {
    let mut request = tool_request();
    request.messages.push(GenerationMessage {
        role: "assistant".to_string(),
        content: Some(MessageContent::Other(serde_json::Value::Null)),
        extra: std::collections::BTreeMap::from([(
            "tool_calls".to_string(),
            json!([{
                "id": "call_123",
                "type": "function",
                "function": {"name": "lookup", "arguments": "{\"city\":\"Sydney\"}"}
            }]),
        )]),
    });
    request.messages.push(GenerationMessage {
        role: "tool".to_string(),
        content: Some(MessageContent::Text("sunny".to_string())),
        extra: std::collections::BTreeMap::from([("tool_call_id".to_string(), json!("call_123"))]),
    });
    request
}

fn rendered_prompt(metadata_json: &str) -> RenderedChatPrompt {
    RenderedChatPrompt {
        prompt: "prompt".to_string(),
        media: Vec::new(),
        metadata_json: metadata_json.to_string(),
    }
}

#[test]
fn failed_tool_loop_render_recovers_when_probe_proves_no_native_tools() {
    let request = tool_loop_request();
    let mut render_count = 0;
    let selected = select_tool_prompt_render(&request, true, false, |messages| {
        render_count += 1;
        if render_count == 1 {
            assert_eq!(messages, request.messages.as_slice());
            Err(InferenceError::backend("native history failure"))
        } else {
            assert_eq!(messages.len(), 1);
            assert_eq!(messages[0].role, "user");
            assert_eq!(
                messages[0].content,
                Some(MessageContent::Text("Hello".to_string()))
            );
            assert!(messages[0].extra.is_empty());
            Ok(rendered_prompt(r#"{"grammar_triggers": []}"#))
        }
    })
    .expect("non-native probe should select emulation");

    assert!(matches!(selected, PromptRenderSelection::Emulated));
    assert_eq!(render_count, 2);
}

#[test]
fn failed_tool_loop_render_preserves_error_when_probe_supports_native_tools() {
    let request = tool_loop_request();
    let original = InferenceError::invalid_request("original native history failure");
    let mut render_count = 0;
    let error = select_tool_prompt_render(&request, true, false, |_| {
        render_count += 1;
        if render_count == 1 {
            Err(original.clone())
        } else {
            Ok(rendered_prompt(
                r#"{"grammar_triggers": [{"type": 1, "value": "<tool_call>"}]}"#,
            ))
        }
    })
    .expect_err("native-capable probe must preserve the original failure");

    assert_eq!(error.status(), original.status());
    assert_eq!(error.message(), original.message());
    assert_eq!(render_count, 2);
}

#[test]
fn failed_tool_loop_render_preserves_error_when_probe_fails() {
    let request = tool_loop_request();
    let original = InferenceError::backend("original native history failure");
    let mut render_count = 0;
    let error = select_tool_prompt_render(&request, true, false, |_| {
        render_count += 1;
        if render_count == 1 {
            Err(original.clone())
        } else {
            Err(InferenceError::backend("probe failure"))
        }
    })
    .expect_err("failed probe must preserve the original failure");

    assert_eq!(error.status(), original.status());
    assert_eq!(error.message(), original.message());
    assert_eq!(render_count, 2);
}

#[test]
fn ordinary_render_failure_does_not_probe() {
    let request = tool_request();
    let original = InferenceError::backend("ordinary native failure");
    let mut render_count = 0;
    let error = select_tool_prompt_render(&request, true, false, |_| {
        render_count += 1;
        Err(original.clone())
    })
    .expect_err("ordinary failure must not enter tool-loop recovery");

    assert_eq!(error.status(), original.status());
    assert_eq!(error.message(), original.message());
    assert_eq!(render_count, 1);
}

#[test]
fn forced_tool_loop_recovery_does_not_probe() {
    let request = tool_loop_request();
    let mut render_count = 0;
    let selected = select_tool_prompt_render(&request, true, true, |_| {
        render_count += 1;
        Err(InferenceError::backend("native history failure"))
    })
    .expect("force mode should select emulation");

    assert!(matches!(selected, PromptRenderSelection::Emulated));
    assert_eq!(render_count, 1);
}
