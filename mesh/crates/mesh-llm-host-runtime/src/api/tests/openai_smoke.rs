#[tokio::test]
async fn document_ocr_passthrough_preserves_multipart_and_query() {
    let (upstream_port, upstream_rx, upstream_handle) =
        spawn_capturing_upstream(r#"{"pages":[]}"#).await;
    let state = build_test_mesh_api_with_api_port(upstream_port).await;
    let (addr, handle) = spawn_management_test_server(state).await;
    let body = "--ocr\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nindra\r\n--ocr\r\nContent-Disposition: form-data; name=\"file\"; filename=\"page.pdf\"\r\n\r\n%PDF-test\r\n--ocr--\r\n";
    let request = format!(
        "POST /mesh/v1/documents/ocr?trace=1 HTTP/1.1\r\nHost: localhost\r\nContent-Type: multipart/form-data; boundary=ocr\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    let response = send_management_request(addr, request).await;
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    let raw = tokio::time::timeout(Duration::from_secs(5), upstream_rx)
        .await
        .expect("console must forward OCR to inference")
        .unwrap();
    let raw = String::from_utf8(raw).unwrap();
    assert!(raw.starts_with("POST /mesh/v1/documents/ocr?trace=1 HTTP/1.1\r\n"));
    assert!(raw.contains("Content-Type: multipart/form-data; boundary=ocr\r\n"));
    assert_eq!(http_body_text(raw.as_bytes()), body);
    assert!(response.contains("Content-Security-Policy: sandbox\r\n"));
    handle.await.unwrap().unwrap();
    upstream_handle.await.unwrap();
}

#[tokio::test]
async fn console_task_uploads_preserve_binary_bytes() {
    for path in [
        "/mesh/v1/documents/ocr",
        "/mesh/v1/images/similarity",
        "/mesh/v1/images/depth",
        "/v1/audio/transcriptions",
    ] {
        let (upstream_port, upstream_rx, upstream_handle) =
            spawn_capturing_upstream(r#"{"ok":true}"#).await;
        let state = build_test_mesh_api_with_api_port(upstream_port).await;
        let (addr, handle) = spawn_management_test_server(state).await;
        let body = b"--upload\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nmodel\r\n--upload\r\nContent-Disposition: form-data; name=\"file\"; filename=\"input.bin\"\r\n\r\n\x00\xff\x80\r\n--upload--\r\n";
        let header = format!(
            "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: multipart/form-data; boundary=upload\r\nContent-Length: {}\r\n\r\n",
            body.len()
        );
        let mut stream = TcpStream::connect(addr).await.unwrap();
        stream.write_all(header.as_bytes()).await.unwrap();
        stream.write_all(body).await.unwrap();
        stream.shutdown().await.unwrap();
        let mut response = Vec::new();
        stream.read_to_end(&mut response).await.unwrap();
        assert!(response.starts_with(b"HTTP/1.1 200"), "{path}");
        let raw = tokio::time::timeout(Duration::from_secs(5), upstream_rx)
            .await
            .expect("task upload must reach inference")
            .unwrap();
        let body_start = raw.windows(4).position(|w| w == b"\r\n\r\n").unwrap() + 4;
        assert_eq!(&raw[body_start..], body, "{path}");
        handle.await.unwrap().unwrap();
        upstream_handle.await.unwrap();
    }
}

#[tokio::test]
async fn document_ocr_console_preflight_and_methods() {
    let state = build_test_mesh_api_with_api_port(0).await;
    for (method, status) in [("OPTIONS", "204"), ("GET", "405")] {
        let (addr, handle) = spawn_management_test_server(state.clone()).await;
        let response = send_management_request(
            addr,
            format!("{method} /mesh/v1/documents/ocr HTTP/1.1\r\nHost: localhost\r\n\r\n"),
        )
        .await;
        assert!(response.starts_with(&format!("HTTP/1.1 {status}")), "{response}");
        handle.await.unwrap().unwrap();
    }
}

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

async fn post_blobstore_tool(addr: std::net::SocketAddr) -> String {
    let body = json!({
        "request_id": "req-tools-gate",
        "mime_type": "text/plain",
        "file_name": "note.txt",
        "bytes_base64": "aGVsbG8=",
        "expires_in_secs": 60,
        "uses_remaining": 1,
    })
    .to_string();
    let request = format!(
        "POST /api/plugins/blobstore/tools/{} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        blobstore::PUT_REQUEST_OBJECT_TOOL,
        body.len(),
        body
    );
    send_management_request(addr, request).await
}

#[tokio::test]
async fn test_api_plugin_tools_rejects_unpublished_operation() {
    // Capability-only plugins (blobstore, bound providers) publish no
    // operations, so the generic tools route must not reach them: their
    // operations are only reachable through host bindings.
    let (plugin_manager, blobstore_root) = build_blobstore_api_plugin_manager().await;
    let state = build_test_mesh_api_with_plugin_manager(3131, plugin_manager).await;
    let (addr, handle) = spawn_management_test_server(state.clone()).await;

    let response = post_blobstore_tool(addr).await;
    assert!(response.starts_with("HTTP/1.1 404"), "{response}");
    assert!(response.contains("is not published"), "{response}");
    handle.abort();

    let (addr, handle) = spawn_management_test_server(state).await;

    let listing = send_management_request(
        addr,
        "GET /api/plugins/blobstore/tools HTTP/1.1\r\nHost: localhost\r\n\r\n".to_string(),
    )
    .await;
    assert!(listing.starts_with("HTTP/1.1 200"), "{listing}");
    assert_eq!(json_body(&listing), json!([]));

    handle.abort();
    let _ = std::fs::remove_dir_all(blobstore_root);
}

#[tokio::test]
async fn test_api_plugin_tools_allows_published_operation() {
    let (plugin_manager, blobstore_root) = build_blobstore_api_plugin_manager().await;
    plugin_manager
        .set_test_manifests(std::collections::BTreeMap::from([(
            "blobstore".to_string(),
            crate::plugin::proto::PluginManifest {
                operations: vec![crate::plugin::proto::OperationManifest {
                    name: blobstore::PUT_REQUEST_OBJECT_TOOL.into(),
                    input_schema_json: "{}".into(),
                    ..Default::default()
                }],
                ..Default::default()
            },
        )]))
        .await;
    let state = build_test_mesh_api_with_plugin_manager(3131, plugin_manager).await;
    let (addr, handle) = spawn_management_test_server(state).await;

    let response = post_blobstore_tool(addr).await;
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");

    handle.abort();
    let _ = std::fs::remove_dir_all(blobstore_root);
}

/// Upstream that reports whether the console's forwarded socket is registered
/// as a remote bridge while the connection is still alive, and whether the
/// relayed response reaches the client with the sandboxing headers.




#[tokio::test]
async fn console_relay_sandboxes_upstream_responses_and_rejects_arbitrary_get() {
    let (upstream_port, _upstream_rx, upstream_handle) =
        spawn_capturing_upstream(r#"{"ok":true}"#).await;
    let state = build_test_mesh_api_with_api_port(upstream_port).await;
    state.update(true, true).await;
    let (addr, handle) = spawn_management_test_server(state).await;
    let response = send_management_request(
        addr,
        "GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n".to_string(),
    )
    .await;
    assert!(response.starts_with("HTTP/1.1 200 OK"), "{response}");
    assert!(response.contains("X-Content-Type-Options: nosniff\r\n"), "{response}");
    assert!(response.contains("Content-Security-Policy: sandbox\r\n"), "{response}");
    handle.abort();
    upstream_handle.abort();

    // Arbitrary GET paths are no longer forwarded to the inference port.
    let state = build_test_mesh_api_with_api_port(1).await;
    state.update(true, true).await;
    let (addr, handle) = spawn_management_test_server(state).await;
    let response = send_management_request(
        addr,
        "GET /v1/browser-poc HTTP/1.1\r\nHost: localhost\r\n\r\n".to_string(),
    )
    .await;
    assert!(response.starts_with("HTTP/1.1 405"), "{response}");
    handle.abort();
}
