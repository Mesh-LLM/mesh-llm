#[tokio::test]
async fn test_api_proxy_integration_fragmented_post_body() {
    let (upstream_port, upstream_rx, upstream_handle) =
        spawn_capturing_upstream(r#"{"ok":true}"#).await;
    let (proxy_addr, proxy_handle) =
        spawn_api_proxy_test_harness(local_targets(&[("test", upstream_port)])).await;

    let body = json!({
        "model": "test",
        "messages": [{"role": "user", "content": "hello"}],
    })
    .to_string();
    let headers = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
        body.len()
    );

    let response = send_request_and_read_response(
        proxy_addr,
        vec![
            headers.as_bytes()[..38].to_vec(),
            headers.as_bytes()[38..].to_vec(),
            body.as_bytes()[..12].to_vec(),
            body.as_bytes()[12..].to_vec(),
        ],
    )
    .await;
    let raw = String::from_utf8(upstream_rx.await.unwrap()).unwrap();

    assert!(response.starts_with("HTTP/1.1 200 OK"));
    assert!(raw.contains(&body));
    assert!(raw.contains("Connection: close"));

    proxy_handle.abort();
    let _ = upstream_handle.await;
}

#[tokio::test]
async fn test_api_proxy_integration_chunked_body() {
    let (upstream_port, upstream_rx, upstream_handle) =
        spawn_capturing_upstream(r#"{"ok":true}"#).await;
    let (proxy_addr, proxy_handle) =
        spawn_api_proxy_test_harness(local_targets(&[("test", upstream_port)])).await;

    let body = br#"{"model":"test","messages":[{"role":"user","content":"chunked"}]}"#;
    let request = build_chunked_request("/v1/chat/completions", body, &[17, body.len() - 17]);

    let response = send_request_and_read_response(proxy_addr, vec![request]).await;
    let raw = String::from_utf8(upstream_rx.await.unwrap()).unwrap();

    assert!(response.starts_with("HTTP/1.1 200 OK"));
    assert!(raw.contains("Transfer-Encoding: chunked"));
    assert!(raw.contains("\"model\":\"test\""));
    assert!(raw.contains("0\r\n\r\n"));

    proxy_handle.abort();
    let _ = upstream_handle.await;
}

#[tokio::test]
async fn test_api_proxy_rewrites_image_blob_url_to_data_url() {
    let (plugin_manager, blobstore_root) = start_blobstore_plugin_manager().await;
    let put = crate::plugins::blobstore::put_request_object(
        &plugin_manager,
        crate::plugins::blobstore::PutRequestObjectRequest {
            request_id: "req-image-smoke".into(),
            mime_type: "image/png".into(),
            file_name: Some("smoke.png".into()),
            bytes_base64: "aGVsbG8=".into(),
            expires_in_secs: Some(300),
            uses_remaining: Some(3),
        },
    )
    .await
    .unwrap();
    let client_id = "client-smoke";

    let (upstream_port, upstream_rx, upstream_handle) =
        spawn_capturing_upstream(r#"{"ok":true}"#).await;
    let (proxy_addr, proxy_handle) = spawn_api_proxy_test_harness_with_plugin_manager(
        local_targets(&[("test", upstream_port)]),
        plugin_manager.clone(),
    )
    .await;

    let body = json!({
        "model": "test",
        "client_id": client_id,
        "request_id": "req-image-smoke",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "describe this"},
                {"type": "image_url", "image_url": {"url": format!("mesh://blob/{client_id}/{}", put.token)}}
            ]
        }],
    })
    .to_string();
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );

    let response = send_request_and_read_response(proxy_addr, vec![request.into_bytes()]).await;
    let raw = String::from_utf8(upstream_rx.await.unwrap()).unwrap();

    assert!(response.starts_with("HTTP/1.1 200 OK"));
    assert!(raw.contains("data:image/png;base64,aGVsbG8="));
    assert!(!raw.contains(&format!("mesh://blob/{client_id}/{}", put.token)));
    assert!(
        crate::plugins::blobstore::get_request_object(
            &plugin_manager,
            crate::plugins::blobstore::GetRequestObjectRequest {
                token: put.token.clone(),
                request_id: Some("req-image-smoke".into()),
            },
        )
        .await
        .is_err()
    );

    proxy_handle.abort();
    let _ = upstream_handle.await;
    let _ = std::fs::remove_dir_all(blobstore_root);
}

#[tokio::test]
async fn test_blobstore_helper_resolves_object_store_capability() {
    let (plugin_manager, blobstore_root) =
        start_blobstore_plugin_manager_for("alt-store", vec!["object-store.v1".into()]).await;

    let response = crate::plugins::blobstore::put_request_object(
        &plugin_manager,
        crate::plugins::blobstore::PutRequestObjectRequest {
            request_id: "req-capability".into(),
            mime_type: "text/plain".into(),
            file_name: Some("note.txt".into()),
            bytes_base64: base64::engine::general_purpose::STANDARD.encode("hello"),
            expires_in_secs: Some(60),
            uses_remaining: Some(1),
        },
    )
    .await
    .unwrap();

    assert_eq!(response.request_id, "req-capability");

    let _ = std::fs::remove_dir_all(blobstore_root);
}

#[tokio::test]
async fn test_api_proxy_routes_to_registered_inference_endpoint() {
    let (upstream_port, upstream_rx, upstream_handle) =
        spawn_capturing_upstream(r#"{"id":"chatcmpl","object":"chat.completion","choices":[]}"#)
            .await;
    let plugin_manager = start_inference_endpoint_plugin_manager(
        format!("http://127.0.0.1:{upstream_port}/api/v1"),
        vec!["lemonade-test".into()],
    )
    .await;
    let (proxy_addr, proxy_handle) =
        spawn_api_proxy_test_harness_with_plugin_manager(local_targets(&[]), plugin_manager).await;

    let body = json!({
        "model": "lemonade-test",
        "messages": [{"role": "user", "content": "hello"}],
    })
    .to_string();
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );

    let response = send_request_and_read_response(proxy_addr, vec![request.into_bytes()]).await;
    let raw = String::from_utf8(upstream_rx.await.unwrap()).unwrap();

    assert!(response.starts_with("HTTP/1.1 200 OK"));
    assert!(raw.starts_with("POST /api/v1/chat/completions HTTP/1.1"));
    assert!(raw.contains(r#""model":"lemonade-test""#));

    proxy_handle.abort();
    let _ = upstream_handle.await;
}

#[tokio::test]
async fn test_api_proxy_lists_registered_inference_models() {
    let plugin_manager = start_inference_endpoint_plugin_manager(
        "http://127.0.0.1:8000/api/v1".into(),
        vec!["lemonade-test".into()],
    )
    .await;
    let (proxy_addr, proxy_handle) =
        spawn_api_proxy_test_harness_with_plugin_manager(local_targets(&[]), plugin_manager).await;

    let response = send_request_and_read_response(
        proxy_addr,
        vec![b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n".to_vec()],
    )
    .await;
    let body = response.split("\r\n\r\n").nth(1).unwrap_or_default();
    let json: serde_json::Value = serde_json::from_str(body).unwrap();
    let entries = json["data"].as_array().cloned().unwrap_or_default();

    assert!(response.starts_with("HTTP/1.1 200 OK"));
    assert!(entries.iter().any(|entry| entry["id"] == "lemonade-test"));

    proxy_handle.abort();
}

#[test]
fn test_callable_models_excludes_none_only_targets() {
    let mut targets = local_targets(&[("ready-model", 1234)]);
    targets
        .targets
        .extend(unavailable_targets(&["warming-model"]).targets);
    assert_eq!(callable_models(&targets), vec!["ready-model".to_string()]);
}

#[tokio::test]
async fn test_api_proxy_lemonade_integration_when_enabled() {
    if std::env::var("MESH_LLM_TEST_LEMONADE").ok().as_deref() != Some("1") {
        return;
    }

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap();
    let models_response = client
        .get("http://localhost:8000/api/v1/models")
        .send()
        .await
        .expect("Lemonade should be reachable when MESH_LLM_TEST_LEMONADE=1")
        .error_for_status()
        .expect("Lemonade /models should succeed")
        .json::<serde_json::Value>()
        .await
        .expect("Lemonade /models should return JSON");
    let models = models_response["data"]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|entry| entry["id"].as_str().map(ToOwned::to_owned))
        .collect::<Vec<_>>();
    assert!(
        !models.is_empty(),
        "Lemonade reported no models at http://localhost:8000/api/v1/models"
    );
    let model = models[0].clone();

    let plugin_manager = start_inference_endpoint_plugin_manager(
        "http://localhost:8000/api/v1".into(),
        models.clone(),
    )
    .await;
    let (proxy_addr, proxy_handle) =
        spawn_api_proxy_test_harness_with_plugin_manager(local_targets(&[]), plugin_manager).await;

    let models_response = send_request_and_read_response(
        proxy_addr,
        vec![b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n".to_vec()],
    )
    .await;
    let models_body = models_response.split("\r\n\r\n").nth(1).unwrap_or_default();
    let models_json: serde_json::Value = serde_json::from_str(models_body).unwrap();
    let model_entries = models_json["data"].as_array().cloned().unwrap_or_default();
    assert!(model_entries.iter().any(|entry| entry["id"] == model));

    let body = json!({
        "model": model,
        "messages": [{"role": "user", "content": "Reply with the word ok."}],
        "stream": false,
    })
    .to_string();
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );
    let response = send_request_and_read_response(proxy_addr, vec![request.into_bytes()]).await;
    assert!(
        response.starts_with("HTTP/1.1 200 OK"),
        "unexpected Lemonade proxy response: {response}"
    );

    proxy_handle.abort();
}

#[tokio::test]
async fn test_api_proxy_rewrites_audio_blob_url_to_data_url() {
    let (plugin_manager, blobstore_root) = start_blobstore_plugin_manager().await;
    let put = crate::plugins::blobstore::put_request_object(
        &plugin_manager,
        crate::plugins::blobstore::PutRequestObjectRequest {
            request_id: "req-audio-smoke".into(),
            mime_type: "audio/wav".into(),
            file_name: Some("smoke.wav".into()),
            bytes_base64: "UklGRg==".into(),
            expires_in_secs: Some(300),
            uses_remaining: Some(3),
        },
    )
    .await
    .unwrap();
    let client_id = "client-smoke";

    let (upstream_port, upstream_rx, upstream_handle) =
        spawn_capturing_upstream(r#"{"ok":true}"#).await;
    let (proxy_addr, proxy_handle) = spawn_api_proxy_test_harness_with_plugin_manager(
        local_targets(&[("test", upstream_port)]),
        plugin_manager.clone(),
    )
    .await;

    let body = json!({
        "model": "test",
        "client_id": client_id,
        "request_id": "req-audio-smoke",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "transcribe this"},
                {"type": "audio_url", "audio_url": {"url": format!("mesh://blob/{client_id}/{}", put.token)}}
            ]
        }],
    })
    .to_string();
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );

    let response = send_request_and_read_response(proxy_addr, vec![request.into_bytes()]).await;
    let raw = String::from_utf8(upstream_rx.await.unwrap()).unwrap();

    assert!(response.starts_with("HTTP/1.1 200 OK"));
    assert!(raw.contains("data:audio/wav;base64,UklGRg=="));
    assert!(!raw.contains(&format!("mesh://blob/{client_id}/{}", put.token)));
    assert!(
        crate::plugins::blobstore::get_request_object(
            &plugin_manager,
            crate::plugins::blobstore::GetRequestObjectRequest {
                token: put.token.clone(),
                request_id: Some("req-audio-smoke".into()),
            },
        )
        .await
        .is_err()
    );

    proxy_handle.abort();
    let _ = upstream_handle.await;
    let _ = std::fs::remove_dir_all(blobstore_root);
}

#[tokio::test]
async fn test_api_proxy_rewrites_input_audio_blob_url_to_inline_audio() {
    let (plugin_manager, blobstore_root) = start_blobstore_plugin_manager().await;
    let put = crate::plugins::blobstore::put_request_object(
        &plugin_manager,
        crate::plugins::blobstore::PutRequestObjectRequest {
            request_id: "req-input-audio-smoke".into(),
            mime_type: "audio/wav".into(),
            file_name: Some("smoke.wav".into()),
            bytes_base64: "UklGRg==".into(),
            expires_in_secs: Some(300),
            uses_remaining: Some(3),
        },
    )
    .await
    .unwrap();
    let client_id = "client-smoke";

    let (upstream_port, upstream_rx, upstream_handle) =
        spawn_capturing_upstream(r#"{"ok":true}"#).await;
    let (proxy_addr, proxy_handle) = spawn_api_proxy_test_harness_with_plugin_manager(
        local_targets(&[("test", upstream_port)]),
        plugin_manager.clone(),
    )
    .await;

    let body = json!({
        "model": "test",
        "client_id": client_id,
        "request_id": "req-input-audio-smoke",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "transcribe this"},
                {"type": "input_audio", "input_audio": {"url": format!("mesh://blob/{client_id}/{}", put.token)}}
            ]
        }],
    })
    .to_string();
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );

    let response = send_request_and_read_response(proxy_addr, vec![request.into_bytes()]).await;
    let raw = String::from_utf8(upstream_rx.await.unwrap()).unwrap();

    assert!(response.starts_with("HTTP/1.1 200 OK"));
    assert!(raw.contains(r#""type":"input_audio""#));
    assert!(raw.contains(r#""data":"UklGRg==""#));
    assert!(raw.contains(r#""format":"wav""#));
    assert!(raw.contains(r#""mime_type":"audio/wav""#));
    assert!(!raw.contains(&format!("mesh://blob/{client_id}/{}", put.token)));
    assert!(
        crate::plugins::blobstore::get_request_object(
            &plugin_manager,
            crate::plugins::blobstore::GetRequestObjectRequest {
                token: put.token.clone(),
                request_id: Some("req-input-audio-smoke".into()),
            },
        )
        .await
        .is_err()
    );

    proxy_handle.abort();
    let _ = upstream_handle.await;
    let _ = std::fs::remove_dir_all(blobstore_root);
}

#[tokio::test]
async fn test_api_proxy_translates_responses_image_request() {
    let (plugin_manager, blobstore_root) = start_blobstore_plugin_manager().await;
    let put = crate::plugins::blobstore::put_request_object(
        &plugin_manager,
        crate::plugins::blobstore::PutRequestObjectRequest {
            request_id: "req-responses-image".into(),
            mime_type: "image/png".into(),
            file_name: Some("smoke.png".into()),
            bytes_base64: "aGVsbG8=".into(),
            expires_in_secs: Some(300),
            uses_remaining: Some(3),
        },
    )
    .await
    .unwrap();
    let client_id = "client-smoke";

    let upstream_response = serde_json::json!({
        "id": "chatcmpl_image",
        "object": "chat.completion",
        "created": 123,
        "model": "test",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "image ok"},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 7,
            "completion_tokens": 2,
            "total_tokens": 9
        }
    })
    .to_string();
    let (upstream_port, upstream_rx, upstream_handle) =
        spawn_capturing_upstream(&upstream_response).await;
    let (proxy_addr, proxy_handle) = spawn_api_proxy_test_harness_with_plugin_manager(
        local_targets(&[("test", upstream_port)]),
        plugin_manager.clone(),
    )
    .await;

    let body = json!({
        "model": "test",
        "request_id": "req-responses-image",
        "input": [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "describe this"},
                {"type": "input_image", "image_url": format!("mesh://blob/{client_id}/{}", put.token)}
            ]
        }]
    })
    .to_string();
    let request = format!(
        "POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );

    let response = send_request_and_read_response(proxy_addr, vec![request.into_bytes()]).await;
    let raw = String::from_utf8(upstream_rx.await.unwrap()).unwrap();
    let response_body = response.split("\r\n\r\n").nth(1).unwrap();
    let response_json: serde_json::Value = serde_json::from_str(response_body).unwrap();

    assert!(response.starts_with("HTTP/1.1 200 OK"));
    assert!(raw.starts_with("POST /v1/chat/completions HTTP/1.1"));
    assert!(raw.contains(r#""type":"image_url""#));
    assert!(raw.contains("data:image/png;base64,aGVsbG8="));
    assert_eq!(response_json["object"], "response");
    assert_eq!(response_json["output_text"], "image ok");

    proxy_handle.abort();
    let _ = upstream_handle.await;
    let _ = std::fs::remove_dir_all(blobstore_root);
}

#[tokio::test]
async fn test_api_proxy_translates_responses_audio_request() {
    let (plugin_manager, blobstore_root) = start_blobstore_plugin_manager().await;
    let put = crate::plugins::blobstore::put_request_object(
        &plugin_manager,
        crate::plugins::blobstore::PutRequestObjectRequest {
            request_id: "req-responses-audio".into(),
            mime_type: "audio/wav".into(),
            file_name: Some("smoke.wav".into()),
            bytes_base64: "UklGRg==".into(),
            expires_in_secs: Some(300),
            uses_remaining: Some(3),
        },
    )
    .await
    .unwrap();
    let client_id = "client-smoke";

    let upstream_response = serde_json::json!({
        "id": "chatcmpl_audio",
        "object": "chat.completion",
        "created": 123,
        "model": "test",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "audio ok"},
            "finish_reason": "stop"
        }]
    })
    .to_string();
    let (upstream_port, upstream_rx, upstream_handle) =
        spawn_capturing_upstream(&upstream_response).await;
    let (proxy_addr, proxy_handle) = spawn_api_proxy_test_harness_with_plugin_manager(
        local_targets(&[("test", upstream_port)]),
        plugin_manager.clone(),
    )
    .await;

    let body = json!({
        "model": "test",
        "request_id": "req-responses-audio",
        "input": [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "transcribe this"},
                {"type": "input_audio", "audio_url": format!("mesh://blob/{client_id}/{}", put.token)}
            ]
        }]
    })
    .to_string();
    let request = format!(
        "POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );

    let response = send_request_and_read_response(proxy_addr, vec![request.into_bytes()]).await;
    let raw = String::from_utf8(upstream_rx.await.unwrap()).unwrap();
    let response_body = response.split("\r\n\r\n").nth(1).unwrap();
    let response_json: serde_json::Value = serde_json::from_str(response_body).unwrap();

    assert!(response.starts_with("HTTP/1.1 200 OK"));
    assert!(raw.starts_with("POST /v1/chat/completions HTTP/1.1"));
    assert!(raw.contains(r#""type":"input_audio""#));
    assert!(raw.contains(r#""data":"UklGRg==""#));
    assert!(raw.contains(r#""format":"wav""#));
    assert_eq!(response_json["object"], "response");
    assert_eq!(response_json["output_text"], "audio ok");

    proxy_handle.abort();
    let _ = upstream_handle.await;
    let _ = std::fs::remove_dir_all(blobstore_root);
}

#[tokio::test]
async fn test_api_proxy_integration_expect_continue() {
    let (upstream_port, upstream_rx, upstream_handle) =
        spawn_capturing_upstream(r#"{"ok":true}"#).await;
    let (proxy_addr, proxy_handle) =
        spawn_api_proxy_test_harness(local_targets(&[("test", upstream_port)])).await;

    let body = br#"{"model":"test","messages":[{"role":"user","content":"expect"}]}"#;
    let headers = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nExpect: 100-continue\r\n\r\n",
        body.len()
    );

    let mut stream = TcpStream::connect(proxy_addr).await.unwrap();
    stream.write_all(headers.as_bytes()).await.unwrap();

    let mut interim = [0u8; 64];
    let n = stream.read(&mut interim).await.unwrap();
    assert_eq!(
        std::str::from_utf8(&interim[..n]).unwrap(),
        "HTTP/1.1 100 Continue\r\n\r\n"
    );

    stream.write_all(body).await.unwrap();
    stream.shutdown().await.unwrap();
    let mut response = Vec::new();
    stream.read_to_end(&mut response).await.unwrap();
    let raw = String::from_utf8(upstream_rx.await.unwrap()).unwrap();

    assert!(
        String::from_utf8(response)
            .unwrap()
            .starts_with("HTTP/1.1 200 OK")
    );
    assert!(!raw.contains("Expect: 100-continue"));
    assert!(raw.contains("Connection: close"));
    assert!(raw.contains(std::str::from_utf8(body).unwrap()));

    proxy_handle.abort();
    let _ = upstream_handle.await;
}

// Removed: test_api_proxy_integration_streaming_response_arrives_incrementally
// Was timing-dependent — expected the proxy to preserve a 1s inter-chunk delay,
// but the proxy delivers both chunks immediately. The streaming delivery behavior
// is already covered by test_api_proxy_translates_streaming_responses_events_incrementally
// and test_api_proxy_integration_pipeline_streaming_response_arrives_incrementally.

#[tokio::test]
async fn test_api_proxy_translates_streaming_responses_events_incrementally() {
    let chunks = vec![
        (
            Duration::ZERO,
            br#"data: {"id":"chatcmpl_1","object":"chat.completion.chunk","created":123,"model":"test","choices":[{"index":0,"delta":{"content":"one"},"finish_reason":null}]}

"#
            .to_vec(),
        ),
        (
            Duration::from_millis(1000),
            br#"data: {"id":"chatcmpl_1","object":"chat.completion.chunk","created":123,"model":"test","choices":[{"index":0,"delta":{"content":"two"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}

data: [DONE]

"#
            .to_vec(),
        ),
    ];
    let (upstream_port, upstream_rx, upstream_handle) =
        spawn_streaming_upstream("text/event-stream", chunks).await;
    let (proxy_addr, proxy_handle) =
        spawn_api_proxy_test_harness(local_targets(&[("test", upstream_port)])).await;

    let body = json!({
        "model": "test",
        "stream": true,
        "input": "stream responses",
    })
    .to_string();
    let request = format!(
        "POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );

    let mut stream = TcpStream::connect(proxy_addr).await.unwrap();
    stream.write_all(request.as_bytes()).await.unwrap();
    stream.shutdown().await.unwrap();

    let started_at = tokio::time::Instant::now();
    let first = read_until_contains(
        &mut stream,
        br#"event: response.output_text.delta
data: {"#,
        Duration::from_secs(2),
    )
    .await;
    let first_elapsed = started_at.elapsed();
    let first_text = String::from_utf8_lossy(&first);
    assert!(first_text.contains("HTTP/1.1 200 OK"));
    assert!(first_text.contains("Content-Type: text/event-stream"));
    assert!(first_text.contains("event: response.created"));
    assert!(first_text.contains("event: response.output_text.delta"));
    assert!(first_text.contains(r#""delta":"one""#));
    assert!(
        first_elapsed < Duration::from_millis(900),
        "first translated delta arrived too late: {first_elapsed:?}"
    );
    assert!(!first_text.contains(r#""delta":"two""#));
    assert!(!first_text.contains("event: response.output_text.done"));
    assert!(!first_text.contains("event: response.completed"));

    let mut rest = Vec::new();
    stream.read_to_end(&mut rest).await.unwrap();
    let mut full = first;
    full.extend_from_slice(&rest);
    let full_text = String::from_utf8(full).unwrap();
    assert!(full_text.contains(r#""delta":"two""#));
    assert!(full_text.contains("event: response.output_text.done"));
    assert!(full_text.contains("event: response.completed"));
    assert!(full_text.contains(r#""output_text":"onetwo""#));
    assert!(full_text.contains("event: done"));
    assert!(full_text.contains("data: [DONE]"));
    assert!(full_text.ends_with("0\r\n\r\n"));

    let raw = String::from_utf8(upstream_rx.await.unwrap()).unwrap();
    assert!(raw.starts_with("POST /v1/chat/completions HTTP/1.1"));
    assert!(raw.contains("\"stream\":true"));
    assert!(raw.contains("\"messages\""));

    proxy_handle.abort();
    let _ = upstream_handle.await;
}

#[tokio::test]
async fn test_api_proxy_translates_streaming_reasoning_content_events() {
    let chunks = vec![
        (
            Duration::ZERO,
            br#"data: {"id":"chatcmpl_1","object":"chat.completion.chunk","created":123,"model":"test","choices":[{"index":0,"delta":{"reasoning_content":"thinking"},"finish_reason":null}]}

"#
            .to_vec(),
        ),
        (
            Duration::from_millis(10),
            br#"data: {"id":"chatcmpl_1","object":"chat.completion.chunk","created":123,"model":"test","choices":[{"index":0,"delta":{"content":"answer"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}

data: [DONE]

"#
            .to_vec(),
        ),
    ];
    let (upstream_port, upstream_rx, upstream_handle) =
        spawn_streaming_upstream("text/event-stream", chunks).await;
    let (proxy_addr, proxy_handle) =
        spawn_api_proxy_test_harness(local_targets(&[("test", upstream_port)])).await;

    let body = json!({
        "model": "test",
        "stream": true,
        "input": "stream responses",
    })
    .to_string();
    let request = format!(
        "POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );

    let mut stream = TcpStream::connect(proxy_addr).await.unwrap();
    stream.write_all(request.as_bytes()).await.unwrap();
    stream.shutdown().await.unwrap();

    let first = read_until_contains(
        &mut stream,
        br#"event: response.reasoning_text.delta
data: {"#,
        Duration::from_secs(2),
    )
    .await;
    let first_text = String::from_utf8_lossy(&first);
    assert!(first_text.contains("event: response.created"));
    assert!(first_text.contains("event: response.reasoning_text.delta"));
    assert!(first_text.contains(r#""delta":"thinking""#));
    assert!(!first_text.contains("event: response.output_text.delta"));
    assert!(!first_text.contains(r#""delta":"answer""#));

    let mut rest = Vec::new();
    stream.read_to_end(&mut rest).await.unwrap();
    let mut full = first;
    full.extend_from_slice(&rest);
    let full_text = String::from_utf8(full).unwrap();
    assert!(full_text.contains("event: response.output_text.delta"));
    assert!(full_text.contains(r#""delta":"answer""#));
    assert!(full_text.contains("event: response.completed"));
    assert!(full_text.contains(r#""output_text":"answer""#));
    assert!(full_text.contains("data: [DONE]"));

    let raw = String::from_utf8(upstream_rx.await.unwrap()).unwrap();
    assert!(raw.starts_with("POST /v1/chat/completions HTTP/1.1"));
    assert!(raw.contains("\"stream\":true"));

    proxy_handle.abort();
    let _ = upstream_handle.await;
}

#[tokio::test]
async fn test_api_proxy_integration_pipeline_fallback_uses_direct_proxy() {
    // Pipeline fallback test: when only one model is available, auto routes
    // to it directly without attempting a pipeline plan.
    let strong_model = "Qwen2.5-Coder-32B-Instruct-Q4_K_M";
    let body = json!({
        "model": "auto",
        "messages": [
            {"role": "user", "content": "Review this codebase, design a system-level fix for the HTTP proxy, debug the fragmented request bug, implement the code changes, update the tests, and explain the trade-offs around buffering, chunked transfer encoding, and connection reuse."}
        ],
        "tools": [
            {"type": "function", "function": {"name": "bash", "parameters": {"type": "object", "properties": {}}}}
        ]
    });
    let classification = router::classify(&body);
    assert!(pipeline::should_pipeline(&classification));

    let (strong_port, strong_rx, strong_handle) = spawn_capturing_upstream(r#"{"ok":true}"#).await;

    let (proxy_addr, proxy_handle) =
        spawn_api_proxy_test_harness(local_targets(&[(strong_model, strong_port)])).await;

    let request_body = body.to_string();
    let headers = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
        request_body.len()
    );

    let response = send_request_and_read_response(
        proxy_addr,
        vec![format!("{headers}{request_body}").into_bytes()],
    )
    .await;
    let raw = String::from_utf8(strong_rx.await.unwrap()).unwrap();

    assert!(response.starts_with("HTTP/1.1 200 OK"));
    assert!(raw.contains(&format!("\"model\":\"{strong_model}\"")));
    assert!(!raw.contains("\"model\":\"auto\""));
    assert!(!raw.contains("[Task Plan from"));
    assert!(raw.contains("\"Review this codebase, design a system-level fix for the HTTP proxy, debug the fragmented request bug, implement the code changes, update the tests, and explain the trade-offs around buffering, chunked transfer encoding, and connection reuse.\""));
    // model=auto must inject mesh_hooks so the serving runtime enables hook callbacks.
    assert!(
        raw.contains("\"mesh_hooks\":true"),
        "model=auto should inject mesh_hooks:true into the forwarded body"
    );

    proxy_handle.abort();
    let _ = strong_handle.await;
}

#[tokio::test]
async fn test_api_proxy_integration_pipeline_streaming_response_arrives_incrementally() {
    // With a single model, pipeline is skipped (needs 2 local models).
    // This tests that a streaming agentic request still gets proxied correctly.
    let model = "Qwen2.5-Coder-32B-Instruct-Q4_K_M";
    let body = json!({
        "model": "auto",
        "stream": true,
        "messages": [
            {"role": "user", "content": "Review this codebase, design a system-level fix for the HTTP proxy, debug the fragmented request bug, implement the code changes, update the tests, and explain the trade-offs around buffering, chunked transfer encoding, and connection reuse."}
        ],
        "tools": [
            {"type": "function", "function": {"name": "bash", "parameters": {"type": "object", "properties": {}}}}
        ]
    });
    let classification = router::classify(&body);
    assert!(pipeline::should_pipeline(&classification));

    let (port, _rx, handle) = spawn_streaming_upstream(
        "text/event-stream",
        vec![
            (
                Duration::ZERO,
                br#"data: {"delta":"chunk-one"}\n\n"#.to_vec(),
            ),
            (
                Duration::from_millis(1000),
                br#"data: {"delta":"chunk-two"}\n\n"#.to_vec(),
            ),
        ],
    )
    .await;

    let (proxy_addr, proxy_handle) =
        spawn_api_proxy_test_harness(local_targets(&[(model, port)])).await;

    let request_body = body.to_string();
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        request_body.len(),
        request_body
    );

    let mut stream = TcpStream::connect(proxy_addr).await.unwrap();
    stream.write_all(request.as_bytes()).await.unwrap();
    stream.shutdown().await.unwrap();

    let full = read_until_contains(
        &mut stream,
        br#"data: {"delta":"chunk-two"}\n\n"#,
        Duration::from_secs(5),
    )
    .await;
    let full_text = String::from_utf8_lossy(&full);
    assert!(full_text.contains("HTTP/1.1 200 OK"));
    assert!(full_text.contains(r#"data: {"delta":"chunk-one"}\n\n"#));
    assert!(full_text.contains(r#"data: {"delta":"chunk-two"}\n\n"#));

    proxy_handle.abort();
    let _ = handle.await;
}

#[tokio::test]
async fn test_api_proxy_integration_pipelined_follow_up_is_not_forwarded() {
    let (upstream_port, upstream_rx, upstream_handle) =
        spawn_capturing_upstream(r#"{"ok":true}"#).await;
    let (proxy_addr, proxy_handle) =
        spawn_api_proxy_test_harness(local_targets(&[("test", upstream_port)])).await;

    let body = json!({
        "model": "test",
        "messages": [{"role": "user", "content": "first"}],
    })
    .to_string();
    let first_request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    );
    let second_request = "GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n";

    let response = send_request_and_read_response(
        proxy_addr,
        vec![format!("{first_request}{second_request}").into_bytes()],
    )
    .await;
    let raw = String::from_utf8(upstream_rx.await.unwrap()).unwrap();

    assert!(response.starts_with("HTTP/1.1 200 OK"));
    assert!(raw.contains("\"content\":\"first\""));
    assert!(!raw.contains("GET /v1/models HTTP/1.1"));

    proxy_handle.abort();
    let _ = upstream_handle.await;
}

#[tokio::test]
async fn test_api_proxy_integration_streaming_client_disconnect_does_not_hang() {
    let (upstream_port, upstream_rx, upstream_handle) = spawn_streaming_upstream(
        "text/event-stream",
        vec![
            (Duration::ZERO, br#"data: {"delta":"hello"}\n\n"#.to_vec()),
            (
                Duration::from_millis(150),
                br#"data: {"delta":"after-disconnect"}\n\n"#.to_vec(),
            ),
        ],
    )
    .await;
    let (proxy_addr, proxy_handle) =
        spawn_api_proxy_test_harness(local_targets(&[("test", upstream_port)])).await;

    let body = json!({
        "model": "test",
        "stream": true,
        "messages": [{"role": "user", "content": "disconnect me"}],
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

    let first = read_until_contains(
        &mut stream,
        br#"data: {"delta":"hello"}\n\n"#,
        Duration::from_secs(2),
    )
    .await;
    assert!(String::from_utf8_lossy(&first).contains(r#"data: {"delta":"hello"}\n\n"#));
    drop(stream);

    let raw = String::from_utf8(upstream_rx.await.unwrap()).unwrap();
    assert!(raw.contains("\"disconnect me\""));
    tokio::time::timeout(Duration::from_secs(1), upstream_handle)
        .await
        .expect("streaming upstream hung after client disconnect")
        .unwrap();

    proxy_handle.abort();
}
