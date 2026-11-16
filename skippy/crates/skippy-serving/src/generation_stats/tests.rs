use super::*;
use skippy_inference_api::generation::{GenerationMessage, GenerationRequest, MessageContent};

fn message(role: &str, text: &str) -> GenerationMessage {
    GenerationMessage {
        role: role.to_string(),
        content: Some(MessageContent::Text(text.to_string())),
        extra: Default::default(),
    }
}

#[test]
fn an_agent_turn_is_tools_offered_with_tool_results() {
    let request = GenerationRequest {
        messages: vec![
            message("system", "You are a coding agent."),
            message("user", "Fix the build."),
            message("assistant", ""),
            message("tool", "error[E0425]: cannot find value"),
            message("user", "Try again."),
        ],
        ..Default::default()
    };

    let shape = RequestShape::of_chat(&request, true);

    assert_eq!(
        shape,
        RequestShape {
            chat: true,
            tools_offered: true,
            tool_results: true,
            system_prompt: true,
            user_turns: 2,
        }
    );
}

#[test]
fn a_plain_chat_has_no_tool_marks() {
    let request = GenerationRequest {
        messages: vec![message("user", "Hello")],
        ..Default::default()
    };

    let shape = RequestShape::of_chat(&request, false);

    assert!(shape.chat);
    assert!(!shape.tools_offered);
    assert!(!shape.tool_results);
    assert!(!shape.system_prompt);
    assert_eq!(shape.user_turns, 1);
}

#[test]
fn the_shape_never_holds_message_text() {
    let request = GenerationRequest {
        messages: vec![
            message("developer", "secret-system-prompt"),
            message("user", "secret-user-text"),
        ],
        ..Default::default()
    };

    let rendered = format!("{:?}", RequestShape::of_chat(&request, false));

    assert!(!rendered.contains("secret"), "{rendered}");
}

#[test]
fn a_completion_prompt_is_not_a_chat() {
    assert!(!RequestShape::default().chat);
}
