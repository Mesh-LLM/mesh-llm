// Built-in Mixture-of-Agents plugin, started explicitly by these tests: the
// runtime registers it only when `[[plugins]] name = "mesh-moa"` is configured.

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
    assert_eq!(mesh_model["owned_by"], "mesh-llm");
    assert!(mesh_model.get("virtual_model").is_none());
    let catalog_response = send_request_and_read_response(
        proxy_addr,
        vec![b"GET /mesh/v1/models/lookup?id=mesh HTTP/1.1\r\nHost: localhost\r\n\r\n".to_vec()],
    )
    .await;
    let (_, catalog_body) = catalog_response.split_once("\r\n\r\n").unwrap();
    let catalog_model: serde_json::Value = serde_json::from_str(catalog_body).unwrap();
    assert_eq!(catalog_model["virtual_model"]["supports_tools"], true);
    assert_eq!(catalog_model["virtual_model"]["supports_streaming"], true);

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
            )
            .await;
            crate::network::openai::virtual_model::install_inference_bridge(
                &plugin_manager,
                proxy_addr.port(),
            )
            .await;
            let body = json!({
                "model": model,
                "messages": [{"role": "user", "content": "answer briefly"}],
                "stream": false,
            })
            .to_string();
            let request = format!(
                "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            );
            let response =
                send_request_and_read_response(proxy_addr, vec![request.into_bytes()]).await;
            assert!(
                response.starts_with("HTTP/1.1 200 OK") && response.contains("healthy fallback"),
                "{model} disappeared={disappeared}: {response}"
            );
            assert_eq!(
                healthy_requests.load(std::sync::atomic::Ordering::Relaxed),
                1
            );
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
            protocol: None,
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
