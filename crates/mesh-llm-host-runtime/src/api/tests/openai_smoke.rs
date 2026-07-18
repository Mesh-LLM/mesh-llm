#[tokio::test]
async fn test_api_objects_routes_through_object_store_capability() {
    let (plugin_manager, blobstore_root) = build_blobstore_api_plugin_manager().await;
    let state = build_test_mesh_api_with_plugin_manager(3131, plugin_manager).await;
    let (addr, handle) = spawn_management_test_server(state).await;
    let body = json!({
        "request_id": "req-api-object",
        "mime_type": "text/plain",
        "file_name": "note.txt",
        "bytes_base64": "aGVsbG8=",
        "expires_in_secs": 60,
        "uses_remaining": 1,
    })
    .to_string();
    let request = format!(
        "POST /api/objects HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );
    let response = send_management_request(addr, request).await;

    assert!(response.starts_with("HTTP/1.1 201"));
    let payload = json_body(&response);
    assert_eq!(payload["request_id"], "req-api-object");
    assert_eq!(payload["mime_type"], "text/plain");
    assert!(
        payload["token"]
            .as_str()
            .unwrap_or_default()
            .starts_with("obj_")
    );

    handle.abort();
    let _ = std::fs::remove_dir_all(blobstore_root);
}

#[tokio::test]
async fn test_api_chat_smoke_for_image_request() {
    let (upstream_port, upstream_rx, upstream_handle) =
        spawn_capturing_upstream(r#"{"ok":true}"#).await;
    let state = build_test_mesh_api_with_api_port(upstream_port).await;
    state.update(true, true).await;
    let (addr, handle) = spawn_management_test_server(state).await;

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "describe this image"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}}
            ]
        }],
        "stream": false
    })
    .to_string();
    let request = format!(
        "POST /api/chat HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );

    let mut stream = TcpStream::connect(addr).await.unwrap();
    stream.write_all(request.as_bytes()).await.unwrap();
    stream.shutdown().await.unwrap();
    let mut response = Vec::new();
    stream.read_to_end(&mut response).await.unwrap();
    let response_text = String::from_utf8(response).unwrap();
    let raw = String::from_utf8(upstream_rx.await.unwrap()).unwrap();

    assert!(response_text.starts_with("HTTP/1.1 200 OK"));
    assert!(raw.starts_with("POST /v1/chat/completions HTTP/1.1"));
    assert!(raw.contains(r#""type":"image_url""#));
    assert!(raw.contains("data:image/png;base64,aGVsbG8="));

    handle.abort();
    let _ = upstream_handle.await;
}

#[tokio::test]
async fn test_api_chat_smoke_for_audio_request() {
    let (upstream_port, upstream_rx, upstream_handle) =
        spawn_capturing_upstream(r#"{"ok":true}"#).await;
    let state = build_test_mesh_api_with_api_port(upstream_port).await;
    state.update(true, true).await;
    let (addr, handle) = spawn_management_test_server(state).await;

    let body = serde_json::json!({
        "model": "test-model",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "transcribe this audio"},
                {"type": "input_audio", "input_audio": {
                    "data": "UklGRg==",
                    "format": "wav",
                    "mime_type": "audio/wav"
                }}
            ]
        }],
        "stream": false
    })
    .to_string();
    let request = format!(
        "POST /api/chat HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );

    let mut stream = TcpStream::connect(addr).await.unwrap();
    stream.write_all(request.as_bytes()).await.unwrap();
    stream.shutdown().await.unwrap();
    let mut response = Vec::new();
    stream.read_to_end(&mut response).await.unwrap();
    let response_text = String::from_utf8(response).unwrap();
    let raw = String::from_utf8(upstream_rx.await.unwrap()).unwrap();

    assert!(response_text.starts_with("HTTP/1.1 200 OK"));
    assert!(raw.starts_with("POST /v1/chat/completions HTTP/1.1"));
    assert!(raw.contains(r#""type":"input_audio""#));
    assert!(raw.contains(r#""data":"UklGRg==""#));
    assert!(raw.contains(r#""format":"wav""#));
    assert!(raw.contains(r#""mime_type":"audio/wav""#));

    handle.abort();
    let _ = upstream_handle.await;
}

#[tokio::test]
async fn test_api_responses_smoke_for_image_request() {
    let (upstream_port, upstream_rx, upstream_handle) =
            spawn_capturing_upstream(r#"{"id":"chatcmpl","object":"chat.completion","created":1,"model":"test-model","choices":[{"message":{"role":"assistant","content":"ok"}}]}"#).await;
    let state = build_test_mesh_api_with_api_port(upstream_port).await;
    state.update(true, true).await;
    let (addr, handle) = spawn_management_test_server(state).await;

    let body = serde_json::json!({
        "model": "test-model",
        "input": [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "describe this image"},
                {"type": "input_image", "image_url": "data:image/png;base64,aGVsbG8="}
            ]
        }],
        "stream": false
    })
    .to_string();
    let request = format!(
        "POST /api/responses HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );

    let mut stream = TcpStream::connect(addr).await.unwrap();
    stream.write_all(request.as_bytes()).await.unwrap();
    stream.shutdown().await.unwrap();
    let mut response = Vec::new();
    stream.read_to_end(&mut response).await.unwrap();
    let response_text = String::from_utf8(response).unwrap();
    let raw = String::from_utf8(upstream_rx.await.unwrap()).unwrap();

    assert!(response_text.starts_with("HTTP/1.1 200 OK"));
    assert!(raw.starts_with("POST /v1/chat/completions HTTP/1.1"));
    assert!(raw.contains(r#""type":"image_url""#));
    assert!(raw.contains("data:image/png;base64,aGVsbG8="));

    handle.abort();
    let _ = upstream_handle.await;
}

#[tokio::test]
async fn test_api_responses_smoke_for_file_request() {
    let (upstream_port, upstream_rx, upstream_handle) =
            spawn_capturing_upstream(r#"{"id":"chatcmpl","object":"chat.completion","created":1,"model":"test-model","choices":[{"message":{"role":"assistant","content":"ok"}}]}"#).await;
    let state = build_test_mesh_api_with_api_port(upstream_port).await;
    state.update(true, true).await;
    let (addr, handle) = spawn_management_test_server(state).await;

    let body = serde_json::json!({
        "model": "test-model",
        "input": [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "read this file"},
                {
                    "type": "input_file",
                    "input_file": {
                        "url": "data:text/plain;base64,aGVsbG8=",
                        "mime_type": "text/plain",
                        "file_name": "hello.txt"
                    }
                }
            ]
        }],
        "stream": false
    })
    .to_string();
    let request = format!(
        "POST /api/responses HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );

    let mut stream = TcpStream::connect(addr).await.unwrap();
    stream.write_all(request.as_bytes()).await.unwrap();
    stream.shutdown().await.unwrap();
    let mut response = Vec::new();
    stream.read_to_end(&mut response).await.unwrap();
    let response_text = String::from_utf8(response).unwrap();
    let raw = String::from_utf8(upstream_rx.await.unwrap()).unwrap();

    assert!(response_text.starts_with("HTTP/1.1 200 OK"));
    assert!(raw.starts_with("POST /v1/chat/completions HTTP/1.1"));
    assert!(raw.contains(r#""type":"input_file""#));
    assert!(raw.contains(r#""url":"data:text/plain;base64,aGVsbG8=""#));
    assert!(raw.contains(r#""mime_type":"text/plain""#));
    assert!(raw.contains(r#""file_name":"hello.txt""#));

    handle.abort();
    let _ = upstream_handle.await;
}

#[tokio::test]
async fn test_api_responses_stream_smoke() {
    let (upstream_port, upstream_rx, upstream_handle) = spawn_streaming_upstream(
        "text/event-stream",
        vec![(
            Duration::ZERO,
            br#"event: response.output_text.delta
data: {"type":"response.output_text.delta","delta":"hello"}

event: done
data: [DONE]

"#
            .to_vec(),
        )],
    )
    .await;
    let state = build_test_mesh_api_with_api_port(upstream_port).await;
    state.update(true, true).await;
    let (addr, handle) = spawn_management_test_server(state).await;

    let body = serde_json::json!({
        "model": "test-model",
        "input": "say hello",
        "stream": true
    })
    .to_string();
    let request = format!(
        "POST /api/responses HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );

    let mut stream = TcpStream::connect(addr).await.unwrap();
    stream.write_all(request.as_bytes()).await.unwrap();
    stream.shutdown().await.unwrap();
    let response = read_until_contains(
        &mut stream,
        br#"event: response.output_text.delta"#,
        Duration::from_secs(2),
    )
    .await;
    let response_text = String::from_utf8(response).unwrap();
    let raw = String::from_utf8(upstream_rx.await.unwrap()).unwrap();

    assert!(response_text.starts_with("HTTP/1.1 200 OK"));
    assert!(response_text.contains("event: response.output_text.delta"));
    assert!(raw.starts_with("POST /v1/chat/completions HTTP/1.1"));
    assert!(raw.contains(r#""stream":true"#));

    handle.abort();
    let _ = upstream_handle.await;
}
