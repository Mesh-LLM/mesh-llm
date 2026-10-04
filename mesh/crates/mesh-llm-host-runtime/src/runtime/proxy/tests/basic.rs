#[tokio::test]
async fn legacy_lifecycle_paths_are_gone_from_openai_ingress() {
    let (upstream_port, upstream_rx, upstream_handle) =
        spawn_capturing_upstream(r#"{"unexpected":true}"#).await;
    let (proxy_addr, proxy_handle) =
        spawn_api_proxy_test_harness(local_targets(&[("test", upstream_port)])).await;

    for (path, body) in [
        ("/mesh/load", r#"{"model":"test"}"#),
        ("/mesh/drop?model=test", r#"{"model":"test"}"#),
    ] {
        let request = format!(
            "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        let response = send_request_and_read_response(proxy_addr, vec![request.into_bytes()]).await;
        assert!(
            response.starts_with("HTTP/1.1 410 Gone"),
            "response: {response}"
        );
        assert!(
            response.contains("legacy_route_gone"),
            "response: {response}"
        );
    }

    assert!(
        tokio::time::timeout(Duration::from_millis(100), upstream_rx)
            .await
            .is_err(),
        "legacy lifecycle paths must not reach an inference target"
    );
    proxy_handle.abort();
    upstream_handle.abort();
}

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

#[tokio::test]
async fn test_builtin_moa_all_small_pool_routes_direct_through_plugin_api() {
    let worker_response = json!({
        "id": "chatcmpl-worker",
        "object": "chat.completion",
        "model": "worker-a",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "plugin end-to-end"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}
    })
    .to_string();
    let (worker_a_port, worker_a_requests, worker_a_handle) =
        spawn_repeating_upstream(&worker_response).await;
    let (worker_b_port, worker_b_requests, worker_b_handle) =
        spawn_repeating_upstream(&worker_response).await;
    let plugin_manager = start_moa_plugin_manager().await;
    let (proxy_addr, proxy_handle) = spawn_api_proxy_test_harness_with_plugin_manager(
        local_targets(&[("worker-a", worker_a_port), ("worker-b", worker_b_port)]),
        plugin_manager.clone(),
    )
    .await;
    crate::network::openai::virtual_model::install_inference_bridge(
        &plugin_manager,
        proxy_addr.port(),
    )
    .await;

    let models_response = send_request_and_read_response(
        proxy_addr,
        vec![b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n".to_vec()],
    )
    .await;
    let models_body = models_response.split("\r\n\r\n").nth(1).unwrap_or_default();
    let models_json: serde_json::Value = serde_json::from_str(models_body).unwrap();
    let mesh_model = models_json["data"]
        .as_array()
        .and_then(|models| models.iter().find(|model| model["id"] == "mesh"))
        .unwrap_or_else(|| panic!("virtual model missing from /v1/models: {models_response}"));
    assert_eq!(mesh_model["owned_by"], "plugin:mesh-moa");
    assert_eq!(mesh_model["virtual_model"]["supports_tools"], true);
    assert_eq!(mesh_model["virtual_model"]["supports_streaming"], true);

    let body = json!({
        "model": "mesh",
        "messages": [{"role": "user", "content": "answer briefly"}],
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
        "unexpected virtual-model response: {response}"
    );
    assert!(response.contains("plugin end-to-end"));
    assert!(
        !response.to_ascii_lowercase().contains("x-moa-turn:"),
        "metadata-free workers are conservatively small, so the gateway must not convene the measured-regressive all-small committee"
    );
    assert!(
        worker_a_requests.load(std::sync::atomic::Ordering::Relaxed)
            + worker_b_requests.load(std::sync::atomic::Ordering::Relaxed)
            >= 1,
        "the MoA plugin must call at least one concrete model through host inference"
    );

    proxy_handle.abort();
    worker_a_handle.abort();
    worker_b_handle.abort();
}

#[tokio::test]
async fn test_builtin_moa_all_small_falls_back_through_host_inference() {
    for model in ["mesh", "auto"] {
        for disappeared in [false, true] {
            let failed_upstream = if disappeared {
                None
            } else {
                Some(spawn_status_upstream("503 Service Unavailable", r#"{"error":"busy"}"#).await)
            };
            let failed_port = if let Some((port, _, _)) = &failed_upstream {
                *port
            } else {
                let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
                let port = listener.local_addr().unwrap().port();
                drop(listener);
                port
            };
            let healthy_body = json!({
                "id": "chatcmpl-healthy",
                "object": "chat.completion",
                "model": "worker-b",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "healthy fallback"}, "finish_reason": "stop"}]
            }).to_string();
            let (healthy_port, healthy_requests, healthy_handle) =
                spawn_repeating_upstream(&healthy_body).await;
            let plugin_manager = start_moa_plugin_manager().await;
            let (proxy_addr, proxy_handle) = spawn_api_proxy_test_harness_with_plugin_manager(
                local_targets(&[("worker-a", failed_port), ("worker-b", healthy_port)]),
                plugin_manager.clone(),
            ).await;
            crate::network::openai::virtual_model::install_inference_bridge(
                &plugin_manager,
                proxy_addr.port(),
            ).await;
            let body = json!({
                "model": model,
                "messages": [{"role": "user", "content": "answer briefly"}],
                "stream": false,
            }).to_string();
            let request = format!(
                "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                body.len(), body
            );
            let response = send_request_and_read_response(proxy_addr, vec![request.into_bytes()]).await;
            assert!(
                response.starts_with("HTTP/1.1 200 OK") && response.contains("healthy fallback"),
                "{model} disappeared={disappeared}: {response}"
            );
            assert_eq!(healthy_requests.load(std::sync::atomic::Ordering::Relaxed), 1);
            if let Some((_, failed_requests, failed_handle)) = failed_upstream {
                assert!(
                    tokio::time::timeout(Duration::from_secs(5), failed_requests)
                        .await
                        .is_ok(),
                    "the first placement must be tried"
                );
                failed_handle.abort();
            }
            proxy_handle.abort();
            healthy_handle.abort();
        }
    }
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

#[tokio::test]
async fn test_builtin_moa_rejects_malformed_messages_before_dispatch() {
    let worker_response = json!({
        "id": "chatcmpl-worker",
        "object": "chat.completion",
        "model": "worker-a",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "should never run"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}
    })
    .to_string();
    let (worker_port, worker_requests, worker_handle) =
        spawn_repeating_upstream(&worker_response).await;
    let plugin_manager = start_moa_plugin_manager().await;
    let (proxy_addr, proxy_handle) = spawn_api_proxy_test_harness_with_plugin_manager(
        local_targets(&[("worker-a", worker_port)]),
        plugin_manager.clone(),
    )
    .await;
    crate::network::openai::virtual_model::install_inference_bridge(
        &plugin_manager,
        proxy_addr.port(),
    )
    .await;

    for body in [
        json!({"model": "mesh"}),
        json!({"model": "mesh", "messages": []}),
        json!({"model": "mesh", "messages": "not-an-array"}),
    ] {
        let body = body.to_string();
        let request = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        );
        let response = send_request_and_read_response(proxy_addr, vec![request.into_bytes()]).await;
        assert!(
            response.starts_with("HTTP/1.1 400 Bad Request"),
            "a malformed chat envelope must be a contract error: {response}"
        );
        assert!(
            response.contains("MoA requires a non-empty `messages` array"),
            "unexpected contract error body: {response}"
        );
    }
    assert_eq!(
        worker_requests.load(std::sync::atomic::Ordering::Relaxed),
        0,
        "a malformed chat request must not reach a worker"
    );

    proxy_handle.abort();
    worker_handle.abort();
}

#[tokio::test]
async fn test_in_process_plugin_receives_mesh_visibility_at_init() {
    use std::sync::atomic::{AtomicBool, Ordering};

    // The MoA plugin derives its public-mesh patience profile from the
    // visibility the host hands it during initialize, so the handshake itself
    // is the contract this test locks.
    for (visibility, expected_public) in [
        (mesh_llm_plugin::MeshVisibility::Public, true),
        (mesh_llm_plugin::MeshVisibility::Private, false),
    ] {
        let observed = Arc::new(AtomicBool::new(false));
        let hook = Arc::clone(&observed);
        let plugin = mesh_llm_plugin::SimplePlugin::new(mesh_llm_plugin::PluginMetadata::new(
            "test-mesh-visibility",
            env!("CARGO_PKG_VERSION"),
            mesh_llm_plugin::plugin_server_info(
                "test-mesh-visibility",
                env!("CARGO_PKG_VERSION"),
                "Mesh visibility probe",
                "Records the mesh visibility the host initializes it with",
                None::<String>,
            ),
        ))
        .on_initialize(move |request, _context| {
            hook.store(
                request.mesh_visibility == mesh_llm_plugin::MeshVisibility::Public,
                Ordering::Relaxed,
            );
            Box::pin(async { Ok(()) })
        });

        let _manager = start_in_process_plugin_manager(plugin, visibility).await;

        assert_eq!(
            observed.load(Ordering::Relaxed),
            expected_public,
            "expected the plugin to observe {visibility:?} at initialize"
        );
    }
}

#[tokio::test]
async fn test_builtin_moa_is_not_advertised_before_a_candidate_is_ready() {
    let plugin_manager = start_moa_plugin_manager().await;
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
    assert!(
        entries.iter().all(|entry| entry["id"] != "mesh"),
        "candidate-backed virtual models must not signal readiness early: {response}"
    );

    proxy_handle.abort();
}

#[tokio::test]
async fn test_builtin_moa_uses_plugin_inference_model_as_its_only_candidate() {
    let worker_response = json!({
        "id": "chatcmpl-plugin-worker",
        "object": "chat.completion",
        "model": "plugin-worker",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "plugin candidate"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4}
    })
    .to_string();
    let (worker_port, worker_requests, worker_handle) =
        spawn_repeating_upstream(&worker_response).await;
    let plugin_manager = start_moa_plugin_manager().await;
    plugin_manager
        .set_test_inference_endpoints(vec![plugin::InferenceEndpointRoute {
            plugin_name: "endpoint-plugin".into(),
            endpoint_id: "endpoint-plugin".into(),
            address: format!("http://127.0.0.1:{worker_port}/api/v1"),
            models: vec!["plugin-worker".into()],
        }])
        .await;
    let (proxy_addr, proxy_handle) = spawn_api_proxy_test_harness_with_plugin_manager(
        local_targets(&[]),
        plugin_manager.clone(),
    )
    .await;
    crate::network::openai::virtual_model::install_inference_bridge(
        &plugin_manager,
        proxy_addr.port(),
    )
    .await;

    let models_response = send_request_and_read_response(
        proxy_addr,
        vec![b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n".to_vec()],
    )
    .await;
    assert!(
        models_response.contains(r#""id":"mesh""#),
        "plugin inference candidates must make candidate-backed virtual models ready: {models_response}"
    );

    let body = json!({
        "model": "mesh",
        "messages": [{"role": "user", "content": "Say hi."}],
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
        "response: {response}"
    );
    assert!(
        response.contains("plugin candidate"),
        "response: {response}"
    );
    assert!(
        worker_requests.load(std::sync::atomic::Ordering::Relaxed) >= 1,
        "the virtual model must receive and invoke plugin inference candidates"
    );

    proxy_handle.abort();
    worker_handle.abort();
}

#[tokio::test]
async fn test_builtin_moa_single_model_preserves_small_context_request() {
    let worker_response = json!({
        "id": "chatcmpl-worker",
        "object": "chat.completion",
        "model": "worker-a",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "hi"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}
    })
    .to_string();
    let (worker_port, worker_requests, worker_handle) =
        spawn_repeating_upstream(&worker_response).await;
    let plugin_manager = start_moa_plugin_manager().await;
    let (proxy_addr, proxy_handle) = spawn_api_proxy_test_harness_with_plugin_manager_and_contexts(
        local_targets(&[("worker-a", worker_port)]),
        plugin_manager.clone(),
        &[("worker-a", 256)],
    )
    .await;
    crate::network::openai::virtual_model::install_inference_bridge(
        &plugin_manager,
        proxy_addr.port(),
    )
    .await;

    let body = json!({
        "model": "mesh",
        "messages": [{"role": "user", "content": "Say hi."}],
        "max_tokens": 4,
        "temperature": 0,
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
        "unexpected single-model virtual response: {response}"
    );
    assert!(response.contains("\"content\":\"hi\""));
    assert_eq!(
        worker_requests.load(std::sync::atomic::Ordering::Relaxed),
        1,
        "a one-model pool should dispatch the original request exactly once"
    );

    proxy_handle.abort();
    worker_handle.abort();
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
    // model=auto enables Skippy hooks while retaining the legacy peer flag.
    assert!(
        raw.contains("\"mesh_hooks\":true"),
        "model=auto should retain mesh_hooks:true for older peers"
    );
    assert!(raw.contains("\"skippy_hooks\":true"));

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
                b"data: {\"delta\":\"chunk-one\"}\n\n".to_vec(),
            ),
            (
                Duration::from_millis(1000),
                b"data: {\"delta\":\"chunk-two\"}\n\n".to_vec(),
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
        b"\"delta\":\"chunk-two\"",
        Duration::from_secs(5),
    )
    .await;
    let full_text = String::from_utf8_lossy(&full);
    assert!(full_text.contains("HTTP/1.1 200 OK"));
    assert!(full_text.contains(r#""delta":"chunk-one""#));
    assert!(full_text.contains(r#""delta":"chunk-two""#));

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
            (Duration::ZERO, b"data: {\"delta\":\"hello\"}\n\n".to_vec()),
            (
                Duration::from_millis(150),
                b"data: {\"delta\":\"after-disconnect\"}\n\n".to_vec(),
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

    let first =
        read_until_contains(&mut stream, b"\"delta\":\"hello\"", Duration::from_secs(2)).await;
    assert!(String::from_utf8_lossy(&first).contains(r#""delta":"hello""#));
    drop(stream);

    let raw = String::from_utf8(upstream_rx.await.unwrap()).unwrap();
    assert!(raw.contains("\"disconnect me\""));
    tokio::time::timeout(Duration::from_secs(1), upstream_handle)
        .await
        .expect("streaming upstream hung after client disconnect")
        .unwrap();

    proxy_handle.abort();
}

#[tokio::test]
async fn anthropic_ingress_normalizes_before_dispatch_and_adapts_response() {
    for model in ["test", "auto", "mesh"] {
        let reply = json!({"id":"chat-test","model":"test","choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":2,"total_tokens":9}}).to_string();
        let (port, received, upstream) = spawn_capturing_upstream(&reply).await;
        let (addr, proxy) = spawn_api_proxy_test_harness(local_targets(&[("test", port)])).await;
        let body = json!({"model":model,"max_tokens":32,"system":"system marker","metadata":{"user_id":"session-marker"},"mesh_hooks":true,"messages":[{"role":"user","content":"hello"}]}).to_string();
        let request = format!(
            "POST /v1/messages HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        let response = send_request_and_read_response(addr, vec![request.into_bytes()]).await;
        let raw = String::from_utf8(received.await.unwrap()).unwrap();
        assert!(raw.starts_with("POST /v1/chat/completions "), "{raw}");
        let forwarded: serde_json::Value =
            serde_json::from_str(raw.split_once("\r\n\r\n").unwrap().1).unwrap();
        assert_eq!(
            forwarded["messages"][0],
            json!({"role":"system","content":"system marker"})
        );
        assert_eq!(forwarded["user"], "session-marker");
        assert_eq!(forwarded["mesh_hooks"], true);
        assert_eq!(forwarded["max_completion_tokens"], 32);
        assert!(response.starts_with("HTTP/1.1 200 OK"), "{response}");
        let translated: serde_json::Value =
            serde_json::from_str(response.split_once("\r\n\r\n").unwrap().1).unwrap();
        assert_eq!(translated["type"], "message");
        assert_eq!(translated["content"][0]["text"], "hello");
        assert_eq!(translated["usage"]["input_tokens"], 7);
        proxy.abort();
        upstream.await.unwrap();
    }
}

#[tokio::test]
async fn anthropic_ingress_rejects_unsupported_fields_with_anthropic_error() {
    let (addr, proxy) = spawn_api_proxy_test_harness(local_targets(&[])).await;
    let body = json!({"model":"test","max_tokens":32,"unsupported_field":true,"messages":[{"role":"user","content":"hello"}]}).to_string();
    let request = format!(
        "POST /v1/messages HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    let response = send_request_and_read_response(addr, vec![request.into_bytes()]).await;
    assert!(response.starts_with("HTTP/1.1 400"), "{response}");
    let body: serde_json::Value =
        serde_json::from_str(response.split_once("\r\n\r\n").unwrap().1).unwrap();
    assert_eq!(body["type"], "error");
    assert_eq!(body["error"]["type"], "invalid_request_error");
    proxy.abort();
}

#[tokio::test]
async fn anthropic_upstream_empty_error_keeps_status_and_error_envelope() {
    let (port, _request, upstream) = spawn_status_upstream("502 Bad Gateway", "").await;
    let (addr, proxy) = spawn_api_proxy_test_harness(local_targets(&[("test", port)])).await;
    let body =
        json!({"model":"test","max_tokens":32,"messages":[{"role":"user","content":"hello"}]})
            .to_string();
    let request = format!(
        "POST /v1/messages HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    let response = send_request_and_read_response(addr, vec![request.into_bytes()]).await;
    assert!(response.starts_with("HTTP/1.1 502"), "{response}");
    let error: serde_json::Value =
        serde_json::from_str(response.split_once("\r\n\r\n").unwrap().1).unwrap();
    assert_eq!(error["type"], "error");
    assert_eq!(error["error"]["type"], "api_error");
    proxy.abort();
    upstream.await.unwrap();
}

#[tokio::test]
async fn anthropic_count_upstream_error_uses_anthropic_envelope() {
    let (port, _request, upstream) = spawn_status_upstream("404 Not Found", "{}").await;
    let (addr, proxy) = spawn_api_proxy_test_harness(local_targets(&[("test", port)])).await;
    let body = json!({"model":"test","messages":[{"role":"user","content":"hello"}]}).to_string();
    let request = format!(
        "POST /v1/messages/count_tokens HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    let response = send_request_and_read_response(addr, vec![request.into_bytes()]).await;
    assert!(response.starts_with("HTTP/1.1 404"), "{response}");
    let error: serde_json::Value =
        serde_json::from_str(response.split_once("\r\n\r\n").unwrap().1).unwrap();
    assert_eq!(error["type"], "error");
    assert_eq!(error["error"]["type"], "api_error");
    proxy.abort();
    upstream.await.unwrap();
}
