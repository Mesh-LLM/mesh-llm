use super::*;
use crate::chat::{ChatCompletionChunk, ChatCompletionChunkChoice, ChatCompletionDelta};
use serde_json::json;
use std::collections::BTreeMap;

fn chat_request_from(value: serde_json::Value) -> ChatCompletionRequest {
    messages_request_to_chat_request(serde_json::from_value(value).expect("request parses"))
        .expect("translates")
}

#[test]
fn minimal_request_maps_to_chat_shape() {
    let chat = chat_request_from(json!({
        "model": "m",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "hello"}]
    }));
    assert_eq!(chat.model, "m");
    assert_eq!(chat.messages.len(), 1);
    assert_eq!(chat.messages[0].role, "user");
    assert_eq!(
        chat.effective_max_tokens(),
        Some(32),
        "max_tokens maps to max_completion_tokens"
    );
    assert!(!chat.stream);
}

#[test]
fn claude_code_default_request_surface_translates() {
    let chat = chat_request_from(json!({
        "model": "test",
        "messages": [
            {"role": "user", "content": "Read marker.txt."},
            {"role": "system", "content": [
                {"type": "text", "text": "Today's date is 2026-09-26.",
                 "cache_control": {"type": "ephemeral"}}
            ]}
        ],
        "system": [
            {"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.273.1e4; cc_entrypoint=sdk-cli;"},
            {"type": "text", "text": "You are a Claude agent, built on Anthropic's Claude Agent SDK.",
             "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "Use Read to read marker.txt, then report its contents.",
             "cache_control": {"type": "ephemeral", "ttl": "1h"}}
        ],
        "tools": [{
            "name": "Read",
            "description": "Read a file",
            "input_schema": {
                "type": "object",
                "properties": {"file_path": {"type": "string"}},
                "required": ["file_path"]
            }
        }],
        "metadata": {"user_id": "{\"device_id\":\"device\",\"session_id\":\"session\"}"},
        "max_tokens": 32000,
        "thinking": {"type": "adaptive", "display": "omitted"},
        "context_management": {
            "edits": [{"type": "clear_thinking_20251015", "keep": "all"}]
        },
        "output_config": {"effort": "high"},
        "stream": true
    }));

    assert_eq!(chat.reasoning_effort, Some(ReasoningEffort::High));
    assert_eq!(
        chat.reasoning.as_ref().and_then(|value| value.enabled),
        Some(true)
    );
    assert_eq!(
        chat.prompt_cache_retention,
        Some(PromptCacheRetention::InMemory)
    );
    assert!(!chat.extra.contains_key("thinking"));
    assert!(!chat.extra.contains_key("context_management"));
    assert_eq!(
        chat.messages
            .iter()
            .map(|message| message.role.as_str())
            .collect::<Vec<_>>(),
        vec!["system", "user", "system"]
    );
}

#[test]
fn assistant_thinking_blocks_are_accepted_but_not_forwarded_as_prompt_text() {
    let chat = chat_request_from(json!({
        "model": "m",
        "max_tokens": 32,
        "messages": [
            {"role": "user", "content": "work"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "private", "signature": "signed"},
                {"type": "redacted_thinking", "data": "encrypted"},
                {"type": "text", "text": "visible"}
            ]},
            {"role": "user", "content": "continue"}
        ]
    }));
    let assistant = &chat.messages[1];
    assert_eq!(assistant.role, "assistant");
    assert_eq!(
        assistant
            .content
            .as_ref()
            .and_then(crate::message_content_to_text)
            .as_deref(),
        Some("visible")
    );
}

#[test]
fn tool_input_schema_is_required_and_must_be_an_object() {
    for schema in [Value::Null, json!("not-an-object")] {
        let request: AnthropicMessagesRequest = serde_json::from_value(json!({
            "model": "m",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "read"}],
            "tools": [{"name": "Read", "input_schema": schema}]
        }))
        .expect("request parses");
        let error = messages_request_to_chat_request(request).expect_err("schema rejected");
        assert!(error.to_string().contains("input_schema"), "{error}");
    }
}

#[test]
fn system_prompt_and_multi_turn_map_in_order() {
    let chat = chat_request_from(json!({
        "model": "m",
        "max_tokens": 32,
        "system": [{"type": "text", "text": "line one"}, {"type": "text", "text": "line two"}],
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "again"}
        ]
    }));
    let roles: Vec<&str> = chat.messages.iter().map(|m| m.role.as_str()).collect();
    assert_eq!(roles, vec!["system", "user", "assistant", "user"]);
    assert_eq!(
        match chat.messages[0].content.as_ref() {
            Some(content) => crate::message_content_to_text(content),
            None => None,
        }
        .as_deref(),
        Some("line one\nline two")
    );
}

#[test]
fn stop_sequences_and_metadata_translate() {
    let chat = chat_request_from(json!({
        "model": "m",
        "max_tokens": 8,
        "stop_sequences": ["END"],
        "metadata": {"user_id": "user-9"},
        "messages": [{"role": "user", "content": "x"}]
    }));
    assert_eq!(
        chat.stop,
        Some(crate::common::StopSequence::One("END".to_string()))
    );
    assert_eq!(chat.user.as_deref(), Some("user-9"));
}

#[test]
fn assistant_tool_use_block_becomes_openai_tool_call() {
    let chat = chat_request_from(json!({
        "model": "m",
        "max_tokens": 16,
        "messages": [
            {"role": "user", "content": "weather in oslo?"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather",
                 "input": {"city": "oslo"}}
            ]}
        ]
    }));
    let assistant = chat.messages.last().expect("assistant turn");
    assert_eq!(assistant.role, "assistant");
    let tool_calls = assistant
        .extra
        .get("tool_calls")
        .and_then(|value| value.as_array())
        .expect("tool_calls array");
    assert_eq!(tool_calls[0]["id"], json!("toolu_1"));
    assert_eq!(tool_calls[0]["function"]["name"], json!("get_weather"));
    assert_eq!(
        tool_calls[0]["function"]["arguments"],
        json!(r#"{"city":"oslo"}"#)
    );
}

#[test]
fn tool_result_user_turn_expands_to_tool_messages() {
    let chat = chat_request_from(json!({
        "model": "m",
        "max_tokens": 16,
        "messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {}}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1",
                 "cache_control": {"type": "ephemeral"},
                 "content": [{"type": "text", "text": "sunny, 21C"}]}
            ]}
        ]
    }));
    let roles: Vec<&str> = chat.messages.iter().map(|m| m.role.as_str()).collect();
    assert_eq!(roles, vec!["user", "assistant", "tool"]);
    let tool_message = chat.messages.last().expect("tool message");
    assert_eq!(
        tool_message.extra.get("tool_call_id"),
        Some(&json!("toolu_1"))
    );
    assert_eq!(
        match tool_message.content.as_ref() {
            Some(content) => crate::message_content_to_text(content),
            None => None,
        }
        .as_deref(),
        Some("sunny, 21C")
    );
}

#[test]
fn tools_and_tool_choice_translate_to_openai_shapes() {
    let chat = chat_request_from(json!({
        "model": "m",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "weather?"}],
        "tools": [
            {"name": "get_weather", "description": "d", "input_schema": {"type": "object"}}
        ],
        "tool_choice": {"type": "any"}
    }));
    let tools = chat.tools.expect("tools present");
    assert_eq!(tools[0]["type"], json!("function"));
    assert_eq!(tools[0]["function"]["name"], json!("get_weather"));
    assert_eq!(
        tools[0]["function"]["parameters"],
        json!({"type": "object"})
    );
    assert_eq!(chat.tool_choice, Some(json!("required")));
}

#[test]
fn tool_choice_tool_requires_name() {
    let result = messages_request_to_chat_request(
        serde_json::from_value(json!({
            "model": "m",
            "max_tokens": 4,
            "messages": [{"role": "user", "content": "x"}],
            "tool_choice": {"type": "tool"}
        }))
        .expect("parses"),
    );
    let error = result.expect_err("tool choice without name rejected");
    assert!(error.to_string().contains("tool_choice.name"));
}

#[test]
fn empty_messages_rejected() {
    let result = messages_request_to_chat_request(
        serde_json::from_value(json!({
            "model": "m",
            "max_tokens": 4,
            "messages": []
        }))
        .expect("parses"),
    );
    assert!(result.is_err());
}

#[test]
fn response_translates_to_content_blocks_and_stop_reason() {
    let mut assistant_extra = BTreeMap::new();
    assistant_extra.insert(
        "tool_calls".to_string(),
        json!([{
            "id": "toolu_9",
            "type": "function",
            "function": {"name": "get_weather", "arguments": "{\"city\":\"oslo\"}"}
        }]),
    );
    let response = ChatCompletionResponse::from_parts(
        "chatcmpl-x",
        1,
        "m",
        vec![crate::chat::ChatCompletionChoice {
            index: 0,
            message: crate::chat::AssistantMessage {
                role: "assistant",
                content: Some("checking".to_string()),
                reasoning_content: None,
                tool_calls: None,
            },
            logprobs: None,
            finish_reason: Some(FinishReason::Stop),
        }],
        Usage::new(11, 3),
        None,
    );
    let _ = assistant_extra;
    let message = messages_response_from_chat_response(&response).expect("translates");
    assert!(message.id.starts_with("msg_"));
    assert_eq!(message.kind, "message");
    assert_eq!(message.stop_reason, Some(STOP_REASON_END_TURN));
    assert_eq!(message.usage.input_tokens, 11);
    assert_eq!(message.usage.output_tokens, 3);
    match &message.content[0] {
        AnthropicResponseBlock::Text { text } => assert_eq!(text, "checking"),
        other => panic!("expected text block, got {other:?}"),
    }
}

#[test]
fn tool_call_response_becomes_tool_use_block() {
    let mut assistant_extra = BTreeMap::new();
    assistant_extra.insert(
        "tool_calls".to_string(),
        json!([{
            "id": "toolu_9",
            "type": "function",
            "function": {"name": "get_weather", "arguments": "{\"city\":\"oslo\"}"}
        }]),
    );
    let response = ChatCompletionResponse::from_parts(
        "chatcmpl-y",
        1,
        "m",
        vec![crate::chat::ChatCompletionChoice {
            index: 0,
            message: crate::chat::AssistantMessage {
                role: "assistant",
                content: None,
                reasoning_content: None,
                tool_calls: assistant_extra.remove("tool_calls"),
            },
            logprobs: None,
            finish_reason: Some(FinishReason::ToolCalls),
        }],
        Usage::new(11, 3),
        None,
    );
    let message = messages_response_from_chat_response(&response).expect("translates");
    assert_eq!(message.stop_reason, Some(STOP_REASON_TOOL_USE));
    match &message.content[0] {
        AnthropicResponseBlock::ToolUse { id, name, input } => {
            assert_eq!(id, "toolu_9");
            assert_eq!(name, "get_weather");
            assert_eq!(*input, json!({"city": "oslo"}));
        }
        other => panic!("expected tool_use block, got {other:?}"),
    }
}

fn delta_chunk(
    content: Option<&str>,
    tool_calls: Option<serde_json::Value>,
) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: "chatcmpl-1".to_string(),
        object: "chat.completion.chunk",
        created: 0,
        model: "m".to_string(),
        choices: vec![ChatCompletionChunkChoice {
            index: 0,
            delta: ChatCompletionDelta {
                role: None,
                content: content.map(str::to_string),
                reasoning_content: None,
                tool_calls,
            },
            logprobs: None,
            finish_reason: None,
        }],
        usage: None,
        timings: None,
    }
}

#[test]
fn assembler_streams_text_blocks_in_order() {
    let mut assembler = MessagesStreamAssembler::new();
    let mut events = assembler.absorb(&delta_chunk(Some("Hel"), None));
    events.extend(assembler.absorb(&delta_chunk(Some("lo"), None)));
    assert_eq!(events.len(), 3, "block start + two deltas");
    assert!(matches!(
        events[0],
        AnthropicMessagesStreamEvent::ContentBlockStart { index: 0, .. }
    ));
    assert!(matches!(
        &events[1],
        AnthropicMessagesStreamEvent::ContentBlockDelta { index: 0, delta: AnthropicDelta::Text { text } } if text == "Hel"
    ));
    events.extend(assembler.finish(Some(FinishReason::Stop)));
    assert!(matches!(
        &events[3],
        AnthropicMessagesStreamEvent::ContentBlockStop { index: 0 }
    ));
    assert!(matches!(
        &events[4],
        AnthropicMessagesStreamEvent::MessageDelta { delta, .. } if delta.stop_reason == Some(STOP_REASON_END_TURN)
    ));
    assert!(matches!(
        events[5],
        AnthropicMessagesStreamEvent::MessageStop {}
    ));
}

#[test]
fn assembler_streams_tool_json_and_reports_usage() {
    let mut assembler = MessagesStreamAssembler::new();
    let mut events = Vec::new();
    events.extend(assembler.absorb(&delta_chunk(
        None,
        Some(json!([
            {"index": 0, "id": "toolu_1", "type": "function",
             "function": {"name": "get_weather", "arguments": "{\"city\":"}}
        ])),
    )));
    events.extend(assembler.absorb(&delta_chunk(
        None,
        Some(json!([
            {"index": 0, "function": {"arguments": "\"oslo\"}"}}
        ])),
    )));
    events.extend(assembler.finish(Some(FinishReason::ToolCalls)));
    let starts = events
        .iter()
        .filter(|event| {
            matches!(
                event,
                AnthropicMessagesStreamEvent::ContentBlockStart { .. }
            )
        })
        .count();
    assert_eq!(starts, 1, "one tool_use block opened");
    let json_deltas: Vec<&AnthropicMessagesStreamEvent> = events
        .iter()
        .filter(|event| {
            matches!(
                event,
                AnthropicMessagesStreamEvent::ContentBlockDelta {
                    delta: AnthropicDelta::InputJson { .. },
                    ..
                }
            )
        })
        .collect();
    assert_eq!(json_deltas.len(), 2, "both JSON fragments stream");
    for (position, event) in json_deltas.iter().enumerate() {
        if let AnthropicMessagesStreamEvent::ContentBlockDelta {
            delta: AnthropicDelta::InputJson { partial_json },
            ..
        } = event
        {
            let expected = if position == 0 {
                "{\"city\":"
            } else {
                "\"oslo\"}"
            };
            assert!(
                partial_json.starts_with(expected),
                "fragment {position} should start with {expected}, got {partial_json}"
            );
        }
    }
}

#[test]
fn error_bodies_map_to_anthropic_kinds() {
    let event = translate_stream_error_body(&json!({
        "error": {"message": "boom", "type": "invalid_request_error", "param": null, "code": null}
    }));
    match &event {
        AnthropicMessagesStreamEvent::Error { error } => {
            assert_eq!(error.kind, "invalid_request_error");
            assert_eq!(error.message, "boom");
        }
        other => panic!("expected error event, got {other:?}"),
    }
    let unknown = translate_stream_error_body(&json!({"error": {"message": "x", "type": "weird"}}));
    assert!(matches!(
        &unknown,
        AnthropicMessagesStreamEvent::Error { error } if error.kind == "api_error"
    ));
}
