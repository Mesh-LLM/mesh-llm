#[tokio::test]
async fn installed_plugin_can_override_non_streaming_mesh_chat() {
    let answer = json!({
        "id": "chatcmpl-plugin-override",
        "object": "chat.completion",
        "model": "fake-plugin-worker",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "plugin answer"}, "finish_reason": "stop"}],
    });
    let manager = start_standalone_virtual_model_plugin_manager("mesh", answer).await;
    let (proxy_addr, proxy_handle) =
        spawn_api_proxy_test_harness_with_plugin_manager(local_targets(&[]), manager).await;
    let body = json!({
        "model": "mesh",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": false,
    }).to_string();
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(), body
    );
    let response = send_request_and_read_response(proxy_addr, vec![request.into_bytes()]).await;
    assert!(response.starts_with("HTTP/1.1 200 OK"), "{response}");
    assert!(response.contains("plugin answer"), "{response}");
    assert!(response.contains("fake-plugin-worker"), "{response}");
    proxy_handle.abort();
}

#[tokio::test]
async fn test_streamed_virtual_model_responses_preserves_tool_calls() {
    let tool_call_response = json!({
        "id": "chatcmpl-tool-call",
        "object": "chat.completion",
        "model": "worker-a",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_lookup",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{\"q\":\"hi\"}"}
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}
    });
    let plugin_manager =
        start_standalone_virtual_model_plugin_manager("test-virtual", tool_call_response).await;
    let (proxy_addr, proxy_handle) =
        spawn_api_proxy_test_harness_with_plugin_manager(local_targets(&[]), plugin_manager).await;

    let body = json!({
        "model": "test-virtual",
        "input": "find the thing",
        "tools": [{"type": "function", "name": "lookup"}],
        "tool_choice": {"type": "allowed_tools", "tools": [{"type": "function", "name": "lookup"}]},
        "stream": true,
    })
    .to_string();
    let request = format!(
        "POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );
    let response = send_request_and_read_response(proxy_addr, vec![request.into_bytes()]).await;

    assert!(
        response.starts_with("HTTP/1.1 200 OK"),
        "unexpected virtual-model stream response: {response}"
    );
    let events = responses_sse_events(&response);
    assert!(
        events
            .iter()
            .any(|event| event["type"] == "response.function_call_arguments.done"),
        "a streamed Responses tool call must emit its function-call events: {response}"
    );
    let completed = events
        .iter()
        .find(|event| event["type"] == "response.completed")
        .unwrap_or_else(|| panic!("response.completed missing: {response}"));
    let call = completed["response"]["output"]
        .as_array()
        .and_then(|output| output.iter().find(|item| item["type"] == "function_call"))
        .unwrap_or_else(|| panic!("function_call output item missing: {response}"));
    assert_eq!(call["name"], "lookup");
    assert_eq!(call["call_id"], "call_lookup");
    assert_eq!(call["arguments"], "{\"q\":\"hi\"}");

    proxy_handle.abort();
}

#[tokio::test]
async fn virtual_model_responses_rejects_disallowed_function_calls() {
    let answer = json!({
        "model": "worker-a",
        "choices": [{"message": {"tool_calls": [{
            "id": "call_lookup", "type": "function",
            "function": {"name": "lookup", "arguments": "{}"}
        }]}, "finish_reason": "tool_calls"}]
    });
    let manager = start_standalone_virtual_model_plugin_manager("test-virtual", answer).await;
    let (addr, handle) =
        spawn_api_proxy_test_harness_with_plugin_manager(local_targets(&[]), manager).await;
    for streaming in [false, true] {
        let body = json!({
            "model": "test-virtual", "input": "use safe only", "stream": streaming,
            "tools": [{"type":"function","name":"safe"},{"type":"function","name":"lookup"}],
            "tool_choice": {"type":"allowed_tools","tools":[{"type":"function","name":"safe"}]}
        }).to_string();
        let request = format!(
            "POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(), body
        );
        let response = send_request_and_read_response(addr, vec![request.into_bytes()]).await;
        assert!(response.contains("tool_choice_violation"), "{response}");
        if streaming {
            let events = responses_sse_events(&response);
            assert!(events.iter().any(|event| event["type"] == "response.failed"));
            assert!(!events.iter().any(|event| event["type"] == "response.output_item.added"));
        } else {
            assert!(response.contains("\"status\":\"failed\""), "{response}");
        }
    }
    handle.abort();
}

#[tokio::test]
async fn anthropic_virtual_model_serves_messages_envelopes() {
    let answer = json!({
        "id": "chatcmpl-anthropic",
        "object": "chat.completion",
        "created": 1_700_000_000u64,
        "model": "test-anthropic",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "hello from the committee"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11}
    });
    let plugin_manager =
        start_standalone_virtual_model_plugin_manager("test-anthropic", answer).await;
    let (proxy_addr, proxy_handle) =
        spawn_api_proxy_test_harness_with_plugin_manager(local_targets(&[]), plugin_manager).await;

    // Non-streaming: an Anthropic Messages envelope, not a chat completion.
    let body = json!({
        "model": "test-anthropic",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "hello"}],
    })
    .to_string();
    let request = format!(
        "POST /v1/messages HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );
    let response = send_request_and_read_response(proxy_addr, vec![request.into_bytes()]).await;
    assert!(response.starts_with("HTTP/1.1 200 OK"), "{response}");
    let translated: serde_json::Value =
        serde_json::from_str(response.split_once("\r\n\r\n").unwrap().1).unwrap();
    assert_eq!(translated["type"], "message");
    assert_eq!(translated["content"][0]["type"], "text");
    assert_eq!(translated["content"][0]["text"], "hello from the committee");
    assert_eq!(translated["usage"]["input_tokens"], 7);

    // Streaming: Anthropic named events, never a chat-shaped chunk.
    let body = json!({
        "model": "test-anthropic",
        "max_tokens": 32,
        "stream": true,
        "messages": [{"role": "user", "content": "hello"}],
    })
    .to_string();
    let request = format!(
        "POST /v1/messages HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );
    let response = send_request_and_read_response(proxy_addr, vec![request.into_bytes()]).await;
    assert!(response.starts_with("HTTP/1.1 200 OK"), "{response}");
    assert!(response.contains("event: message_start"), "{response}");
    assert!(
        response.contains("event: content_block_delta"),
        "{response}"
    );
    assert!(response.contains("hello from the committee"), "{response}");
    assert!(response.contains("event: message_stop"), "{response}");
    assert!(
        !response.contains("chat.completion.chunk"),
        "an Anthropic client must never see a chat-shaped body: {response}"
    );

    proxy_handle.abort();
}

/// The completion id of the first chat chunk on the wire.
fn first_chat_chunk_id(text: &str) -> String {
    let marker = "\"id\":\"";
    let start = text
        .find(marker)
        .unwrap_or_else(|| panic!("no chunk id in {text}"))
        + marker.len();
    let rest = &text[start..];
    rest[..rest.find('"').expect("closing quote")].to_string()
}

#[tokio::test]
async fn streaming_virtual_model_drips_progress_while_the_turn_runs() {
    let answer = json!({
        "id": "chatcmpl-drip",
        "object": "chat.completion",
        "created": 1_700_000_000u64,
        "model": "test-drip",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "committee answer"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    });
    // The turn takes 1.5s; the drip ticks at 1s, so what reaches the wire
    // before the answer is exactly the progress phase.
    let plugin_manager = start_dripping_virtual_model_plugin_manager(
        "test-drip",
        answer,
        Duration::from_millis(1500),
    )
    .await;
    let (proxy_addr, proxy_handle) =
        spawn_api_proxy_test_harness_with_plugin_manager(local_targets(&[]), plugin_manager).await;

    let body = json!({
        "model": "test-drip",
        "stream": true,
        "messages": [{"role": "user", "content": "answer me"}],
    })
    .to_string();
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );
    let mut stream = TcpStream::connect(proxy_addr).await.unwrap();
    stream.write_all(request.as_bytes()).await.unwrap();
    stream.shutdown().await.unwrap();

    let progress =
        read_until_contains(&mut stream, DRIP_LINE.as_bytes(), Duration::from_secs(5)).await;
    let progress = String::from_utf8_lossy(&progress);
    assert!(
        progress.starts_with("HTTP/1.1 200 OK"),
        "the head precedes the drip: {progress}"
    );
    assert!(
        !progress.contains("committee answer"),
        "the drip must arrive before the answer it precedes: {progress}"
    );
    assert!(
        progress.contains(r#""reasoning_content":"Consulting peers…\n""#),
        "the declared line is dripped into the reasoning channel: {progress}"
    );
    let drip_id = first_chat_chunk_id(&progress);

    let full = read_until_contains(&mut stream, b"[DONE]", Duration::from_secs(5)).await;
    let full = String::from_utf8_lossy(&full);
    assert!(full.contains("committee answer"), "{full}");
    assert!(
        full.matches(&format!("\"id\":\"{drip_id}\"")).count() >= 2,
        "the drip and the answer must share one completion id: {full}"
    );

    proxy_handle.abort();
}
