use super::*;
use serde_json::json;

#[test]
fn messages_request_parses_minimal_body() {
    let request: AnthropicMessagesRequest = serde_json::from_value(json!({
        "model": "Qwen/Qwen3-8B-GGUF:Q4_K_M",
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": "Hello there"}
        ]
    }))
    .expect("minimal request parses");
    assert_eq!(request.max_tokens, 64);
    assert!(!request.stream);
    assert!(request.system.is_none());
    assert!(request.tools.is_none());
}

#[test]
fn messages_request_parses_string_and_block_content() {
    let request: AnthropicMessagesRequest = serde_json::from_value(json!({
        "model": "m",
        "max_tokens": 16,
        "system": "Be terse.",
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "first"},
                {"type": "text", "text": "second"}
            ]},
            {"role": "assistant", "content": "noted"}
        ]
    }))
    .expect("mixed content parses");
    assert!(matches!(
        request.system,
        Some(AnthropicSystemPrompt::Text(_))
    ));
    assert_eq!(request.messages.len(), 2);
    assert!(matches!(
        &request.messages[0].content,
        AnthropicMessageContent::Blocks(blocks) if blocks.len() == 2
    ));
}

#[test]
fn tool_blocks_parse_with_schema_and_choice() {
    let request: AnthropicMessagesRequest = serde_json::from_value(json!({
        "model": "m",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "weather?"}],
        "tools": [
            {
                "name": "get_weather",
                "description": "Look up weather",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}
            }
        ],
        "tool_choice": {"type": "tool", "name": "get_weather"}
    }))
    .expect("tool request parses");
    let tools = request.tools.expect("tools present");
    assert_eq!(tools[0].name, "get_weather");
    let choice = request.tool_choice.expect("tool_choice present");
    assert_eq!(choice.kind, "tool");
    assert_eq!(choice.name.as_deref(), Some("get_weather"));
}

#[test]
fn response_serializes_anthropic_envelope() {
    let response = AnthropicMessagesResponse {
        id: "msg_1".to_string(),
        kind: "message",
        role: "assistant",
        model: "m".to_string(),
        content: vec![
            AnthropicResponseBlock::Text {
                text: "hi".to_string(),
            },
            AnthropicResponseBlock::ToolUse {
                id: "toolu_1".to_string(),
                name: "get_weather".to_string(),
                input: json!({"city": "oslo"}),
            },
        ],
        stop_reason: Some(STOP_TOOL_USE),
        stop_sequence: None,
        usage: AnthropicUsage {
            cache_read_input_tokens: None,
            input_tokens: 5,
            output_tokens: 7,
        },
    };
    let value = serde_json::to_value(&response).expect("serialize");
    assert_eq!(value["type"], json!("message"));
    assert_eq!(value["role"], json!("assistant"));
    assert_eq!(value["stop_reason"], json!("tool_use"));
    assert_eq!(value["content"][0]["type"], json!("text"));
    assert_eq!(value["content"][1]["type"], json!("tool_use"));
    assert_eq!(value["content"][1]["input"]["city"], json!("oslo"));
    assert_eq!(value["usage"]["input_tokens"], json!(5));
    assert_eq!(value["usage"]["output_tokens"], json!(7));
}

#[test]
fn error_envelope_serializes_type_and_message() {
    let error = AnthropicError::invalid_request("max_tokens is required");
    let value = serde_json::to_value(&error).expect("serialize");
    assert_eq!(value["type"], json!("error"));
    assert_eq!(value["error"]["type"], json!("invalid_request_error"));
    assert_eq!(value["error"]["message"], json!("max_tokens is required"));
}

#[test]
fn stream_events_serialize_with_event_discriminants() {
    let start = AnthropicMessagesStreamEvent::MessageStart {
        message: AnthropicMessageStart {
            id: "msg_1".to_string(),
            kind: "message",
            role: "assistant",
            model: "m".to_string(),
            content: vec![],
            stop_reason: None,
            stop_sequence: None,
            usage: AnthropicUsage {
                cache_read_input_tokens: None,
                input_tokens: 1,
                output_tokens: 0,
            },
        },
    };
    let value = serde_json::to_value(&start).expect("serialize");
    assert_eq!(value["type"], json!("message_start"));
    assert_eq!(value["message"]["usage"]["input_tokens"], json!(1));

    let delta = AnthropicMessagesStreamEvent::ContentBlockDelta {
        index: 0,
        delta: AnthropicDelta::Text {
            text: "he".to_string(),
        },
    };
    let value = serde_json::to_value(&delta).expect("serialize");
    assert_eq!(value["type"], json!("content_block_delta"));
    assert_eq!(value["delta"]["type"], json!("text_delta"));
    assert_eq!(value["delta"]["text"], json!("he"));

    let json_delta = AnthropicMessagesStreamEvent::ContentBlockDelta {
        index: 1,
        delta: AnthropicDelta::InputJson {
            partial_json: "{\"ci".to_string(),
        },
    };
    let value = serde_json::to_value(&json_delta).expect("serialize");
    assert_eq!(value["delta"]["type"], json!("input_json_delta"));

    let stop = AnthropicMessagesStreamEvent::MessageStop {};
    let value = serde_json::to_value(&stop).expect("serialize");
    assert_eq!(value["type"], json!("message_stop"));
}

const STOP_TOOL_USE: &str = "tool_use";
