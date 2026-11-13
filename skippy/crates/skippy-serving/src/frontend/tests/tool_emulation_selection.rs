use super::support::tool_request;
use crate::frontend::prompting::{
    PromptRenderSelection, RenderedChatPrompt, ToolTemplateView, select_tool_prompt_render,
    template_tool_controls,
};
use axum::response::IntoResponse;
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

fn none_tool_loop_request() -> GenerationRequest {
    let mut request = tool_loop_request();
    request.tool_choice = Some(json!("none"));
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
fn none_history_probe_enables_tools_but_final_projection_omits_them() {
    let request = none_tool_loop_request();

    let (request_tools, request_choice) =
        template_tool_controls(&request, ToolTemplateView::Request).expect("request controls");
    assert!(request_tools.is_some());
    assert_eq!(request_choice.as_deref(), Some("\"none\""));

    let (probe_tools, probe_choice) =
        template_tool_controls(&request, ToolTemplateView::CapabilityProbe)
            .expect("probe controls");
    assert_eq!(probe_tools, request_tools);
    assert_eq!(probe_choice.as_deref(), Some("\"auto\""));

    let (projected_tools, projected_choice) =
        template_tool_controls(&request, ToolTemplateView::WithoutTools)
            .expect("projected controls");
    assert!(projected_tools.is_none());
    assert!(projected_choice.is_none());
}

#[test]
fn failed_tool_loop_render_recovers_when_probe_proves_no_native_tools() {
    let request = tool_loop_request();
    let mut render_count = 0;
    let selected = select_tool_prompt_render(&request, true, false, |messages, tool_view| {
        render_count += 1;
        if render_count == 1 {
            assert_eq!(tool_view, ToolTemplateView::Request);
            assert_eq!(messages, request.messages.as_slice());
            Err(InferenceError::backend("native history failure"))
        } else {
            assert_eq!(tool_view, ToolTemplateView::CapabilityProbe);
            assert_eq!(messages.len(), request.messages.len());
            assert!(messages.iter().all(|message| message.role != "tool"));
            assert!(messages.iter().all(|message| message.extra.is_empty()));
            Ok(rendered_prompt(r#"{"grammar_triggers": []}"#))
        }
    })
    .expect("non-native probe should select emulation");

    assert!(matches!(selected, PromptRenderSelection::Emulated));
    assert_eq!(render_count, 2);
}

#[test]
fn none_tool_loop_uses_history_projection_with_enabled_capability_probe() {
    let request = none_tool_loop_request();
    let mut render_count = 0;
    let selected = select_tool_prompt_render(&request, true, false, |messages, tool_view| {
        render_count += 1;
        if render_count == 1 {
            assert_eq!(tool_view, ToolTemplateView::Request);
            assert_eq!(messages, request.messages.as_slice());
            Err(InferenceError::backend("native history failure"))
        } else {
            assert!(messages.iter().all(|message| message.role != "tool"));
            assert!(messages.iter().all(|message| message.extra.is_empty()));
            assert!(messages.iter().any(|message| {
                message.content.as_ref().is_some_and(|content| {
                    skippy_inference_api::message_content_to_text(content)
                        .is_some_and(|text| text.contains("look this up"))
                })
            }));
            match tool_view {
                ToolTemplateView::CapabilityProbe => {
                    Ok(rendered_prompt(r#"{"grammar_triggers": []}"#))
                }
                ToolTemplateView::WithoutTools => Ok(RenderedChatPrompt {
                    prompt: "projected prompt".to_string(),
                    media: Vec::new(),
                    metadata_json: r#"{"grammar_triggers": []}"#.to_string(),
                }),
                ToolTemplateView::Request => unreachable!("request render already failed"),
            }
        }
    })
    .expect("non-native probe should select history-only projection");

    let PromptRenderSelection::ProjectedHistory(rendered) = selected else {
        panic!("history-only turn should use projected history")
    };
    assert_eq!(rendered.prompt, "projected prompt");
    assert_eq!(render_count, 3);
}

#[test]
fn failed_tool_loop_render_preserves_error_when_probe_supports_native_tools() {
    let mut request = none_tool_loop_request();
    request.tools = None;
    let original = InferenceError::invalid_request("original native history failure");
    let mut render_count = 0;
    let error = select_tool_prompt_render(&request, false, false, |_, tool_view| {
        render_count += 1;
        if render_count == 1 {
            assert_eq!(tool_view, ToolTemplateView::Request);
            Err(original.clone())
        } else {
            assert_eq!(tool_view, ToolTemplateView::CapabilityProbe);
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
    let request = none_tool_loop_request();
    let original = InferenceError::backend("original native history failure");
    let mut render_count = 0;
    let error = select_tool_prompt_render(&request, true, false, |_, tool_view| {
        render_count += 1;
        if render_count == 1 {
            assert_eq!(tool_view, ToolTemplateView::Request);
            Err(original.clone())
        } else {
            assert_eq!(tool_view, ToolTemplateView::CapabilityProbe);
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
    let error = select_tool_prompt_render(&request, true, false, |_, tool_view| {
        assert_eq!(tool_view, ToolTemplateView::Request);
        render_count += 1;
        Err(original.clone())
    })
    .expect_err("ordinary failure must not enter tool-loop recovery");

    assert_eq!(error.status(), original.status());
    assert_eq!(error.message(), original.message());
    assert_eq!(render_count, 1);
}

#[test]
fn successful_none_tool_history_keeps_native_render() {
    let request = none_tool_loop_request();
    let selected = select_tool_prompt_render(&request, true, false, |messages, tool_view| {
        assert_eq!(tool_view, ToolTemplateView::Request);
        assert_eq!(messages, request.messages.as_slice());
        Ok(rendered_prompt(r#"{"grammar_triggers": []}"#))
    })
    .expect("successful canonical render must remain native");

    assert!(matches!(selected, PromptRenderSelection::Native(_)));
}

#[test]
fn tool_history_without_current_tools_uses_historical_names_only_for_probe() {
    for current_tools in [None, Some(json!([]))] {
        let mut request = none_tool_loop_request();
        request.tools = current_tools;

        let (probe_tools, probe_choice) =
            template_tool_controls(&request, ToolTemplateView::CapabilityProbe)
                .expect("historical probe controls");
        let probe_tools: serde_json::Value =
            serde_json::from_str(probe_tools.as_deref().expect("probe tools")).unwrap();
        assert_eq!(probe_choice.as_deref(), Some("\"auto\""));
        assert_eq!(
            probe_tools,
            json!([{
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {"type": "object", "properties": {}}
                }
            }])
        );

        let mut render_count = 0;
        let selected = select_tool_prompt_render(&request, false, false, |messages, tool_view| {
            render_count += 1;
            match tool_view {
                ToolTemplateView::Request => {
                    assert_eq!(messages, request.messages.as_slice());
                    Err(InferenceError::backend("native history failure"))
                }
                ToolTemplateView::CapabilityProbe => {
                    assert!(messages.iter().all(|message| message.role != "tool"));
                    Ok(rendered_prompt(r#"{"grammar_triggers": []}"#))
                }
                ToolTemplateView::WithoutTools => Ok(RenderedChatPrompt {
                    prompt: "history-only prompt".to_string(),
                    media: Vec::new(),
                    metadata_json: r#"{"grammar_triggers": []}"#.to_string(),
                }),
            }
        })
        .expect("history-only turn should recover without current tools");

        let PromptRenderSelection::ProjectedHistory(rendered) = selected else {
            panic!("history-only turn should use projected history")
        };
        assert_eq!(rendered.prompt, "history-only prompt");
        assert_eq!(render_count, 3);
    }
}

#[test]
fn failed_final_history_projection_preserves_complete_original_error() {
    let mut request = none_tool_loop_request();
    request.tools = None;
    let original = InferenceError::invalid_request("original native history failure")
        .with_code("canonical_render_failed")
        .with_param("messages")
        .with_retry_after_secs(17);
    let expected_status = original.status();
    let expected_body = original.body();
    let mut render_count = 0;
    let error = select_tool_prompt_render(&request, false, false, |_, tool_view| {
        render_count += 1;
        match tool_view {
            ToolTemplateView::Request => Err(original.clone()),
            ToolTemplateView::CapabilityProbe => Ok(rendered_prompt(r#"{"grammar_triggers": []}"#)),
            ToolTemplateView::WithoutTools => {
                Err(InferenceError::backend("schema-free projection failed"))
            }
        }
    })
    .expect_err("failed final projection must preserve the canonical error");

    assert_eq!(error.status(), expected_status);
    assert_eq!(error.body(), expected_body);
    let response = error.into_response();
    assert_eq!(response.status(), expected_status);
    assert_eq!(response.headers()["retry-after"], "17");
    assert_eq!(render_count, 3);
}

#[test]
fn tool_result_without_a_historical_call_name_preserves_original_error() {
    let mut request = none_tool_loop_request();
    request.tools = None;
    request
        .messages
        .retain(|message| message.role != "assistant");
    let original = InferenceError::backend("canonical history failure");
    let mut render_count = 0;
    let error = select_tool_prompt_render(&request, false, false, |_, tool_view| {
        assert_eq!(tool_view, ToolTemplateView::Request);
        render_count += 1;
        Err(original.clone())
    })
    .expect_err("history without a probe name must preserve the canonical error");

    assert_eq!(error.status(), original.status());
    assert_eq!(error.body(), original.body());
    assert_eq!(render_count, 1);
}

#[test]
fn forced_tool_loop_recovery_does_not_probe() {
    let request = tool_loop_request();
    let mut render_count = 0;
    let selected = select_tool_prompt_render(&request, true, true, |_, tool_view| {
        assert_eq!(tool_view, ToolTemplateView::Request);
        render_count += 1;
        Err(InferenceError::backend("native history failure"))
    })
    .expect("force mode should select emulation");

    assert!(matches!(selected, PromptRenderSelection::Emulated));
    assert_eq!(render_count, 1);
}
