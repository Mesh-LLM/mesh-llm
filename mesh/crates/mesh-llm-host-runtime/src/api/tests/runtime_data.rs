#[tokio::test]
async fn runtime_data_api_routes_remain_payload_stable() {
    let plugin_manager = build_collector_backed_plugin_manager().await;
    let state = build_test_mesh_api_with_plugin_manager(3131, plugin_manager).await;
    seed_runtime_data_api_state(&state).await;

    let status_body = request_management_json(state.clone(), "/api/status").await;
    assert_runtime_status_payload(&status_body);

    let models_body = request_management_json(state.clone(), "/api/models").await;
    assert!(models_body["mesh_models"].is_array());

    let runtime_body = request_management_json(state.clone(), "/api/runtime").await;
    assert_eq!(runtime_body["models"][0]["name"], json!("collector-model"));
    assert_eq!(runtime_body["models"][0]["instance_id"], json!("runtime-1"));
    assert_eq!(
        runtime_body["models"][0]["backend"],
        json!("collector-backend")
    );
    assert_eq!(runtime_body["models"][0]["port"], json!(9337));

    let processes_body = request_management_json(state.clone(), "/api/runtime/processes").await;
    assert_eq!(
        processes_body["processes"][0]["name"],
        json!("collector-model")
    );
    assert_eq!(
        processes_body["processes"][0]["instance_id"],
        json!("runtime-1")
    );
    assert_eq!(
        processes_body["processes"][0]["backend"],
        json!("collector-backend")
    );
    assert_eq!(processes_body["processes"][0]["port"], json!(9337));
    assert_eq!(processes_body["processes"][0]["pid"], json!(777));

    let llama_body = request_management_json(state.clone(), "/api/runtime/llama").await;
    assert_runtime_llama_payload(&llama_body);

    let endpoints_body = request_management_json(state.clone(), "/api/runtime/endpoints").await;
    assert_eq!(
        endpoints_body["endpoints"].as_array().map(Vec::len),
        Some(1)
    );
    assert_eq!(
        endpoints_body["endpoints"][0]["plugin_name"],
        json!("collector-plugin")
    );
    assert_eq!(
        endpoints_body["endpoints"][0]["endpoint_id"],
        json!("chat-http")
    );
    let plugins_body = request_management_json(state, "/api/plugins").await;
    assert_eq!(plugins_body.as_array().map(Vec::len), Some(1));
    assert_eq!(plugins_body[0]["name"], json!("collector-plugin"));
    assert_eq!(plugins_body[0]["status"], json!("running"));
    assert_eq!(plugins_body[0]["capabilities"], json!(["chat"]));

    let state = build_test_mesh_api_with_plugin_manager(
        3131,
        build_collector_backed_plugin_manager().await,
    )
    .await;

    let plugin_endpoints_body =
        request_management_json(state.clone(), "/api/plugins/endpoints").await;
    assert_eq!(plugin_endpoints_body.as_array().map(Vec::len), Some(1));
    assert_eq!(
        plugin_endpoints_body[0]["plugin_name"],
        json!("collector-plugin")
    );
    assert_eq!(plugin_endpoints_body[0]["endpoint_id"], json!("chat-http"));

    let providers_body = request_management_json(state.clone(), "/api/plugins/providers").await;
    assert!(providers_body.as_array().is_some());
    assert!(
        providers_body
            .as_array()
            .unwrap()
            .iter()
            .any(|provider| provider["capability"] == json!("chat"))
    );

    let provider_body = request_management_json(state.clone(), "/api/plugins/providers/chat").await;
    assert_eq!(provider_body["capability"], json!("chat"));
    assert_eq!(provider_body["plugin_name"], json!("collector-plugin"));

    let manifest_body =
        request_management_json(state, "/api/plugins/collector-plugin/manifest").await;
    assert_eq!(manifest_body["capabilities"], json!(["chat"]));
    assert_eq!(manifest_body["endpoints"].as_array().map(Vec::len), Some(1));
}

#[tokio::test]
async fn status_includes_external_inference_endpoint_models() {
    let plugin_manager =
        build_inference_endpoint_plugin_manager(&["lemonade-small", "lemonade-large"]).await;
    let state = build_test_mesh_api_with_plugin_manager(3131, plugin_manager).await;

    let status_body = request_management_json(state, "/api/status").await;

    for field in ["models", "serving_models", "hosted_models"] {
        let models = status_body[field]
            .as_array()
            .unwrap_or_else(|| panic!("{field} should be an array"));
        assert!(
            models.iter().any(|model| model == "lemonade-small"),
            "{field} should include plugin endpoint model: {status_body}"
        );
        assert!(
            models.iter().any(|model| model == "lemonade-large"),
            "{field} should include plugin endpoint model: {status_body}"
        );
    }
}

#[tokio::test]
async fn status_reports_local_build_version_and_independent_latest_release() {
    let state = build_test_mesh_api().await;
    let latest_release = "9.9.9".to_string();
    {
        let mut inner = state.inner.lock().await;
        inner.latest_version = Some(latest_release.clone());
    }

    let status_body = request_management_json(state, "/api/status").await;

    assert_eq!(status_body["version"], json!(crate::BUILD_VERSION));
    assert_eq!(status_body["latest_version"], json!(latest_release));
}

#[tokio::test]
async fn management_mcp_endpoint_initializes_streamable_http_session() {
    let state = build_test_mesh_api().await;
    let (addr, handle) = spawn_management_test_server(state).await;
    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "mesh-api-test",
                "version": "0.1.0"
            }
        }
    })
    .to_string();
    let mut stream = TcpStream::connect(addr).await.unwrap();
    stream
        .write_all(
            format!(
                "POST /mcp HTTP/1.1\r\n\
                 Host: localhost\r\n\
                 Accept: application/json, text/event-stream\r\n\
                 Content-Type: application/json\r\n\
                 Content-Length: {}\r\n\r\n{}",
                body.len(),
                body
            )
            .as_bytes(),
        )
        .await
        .unwrap();
    let response =
        read_until_contains(&mut stream, b"\"serverInfo\"", Duration::from_secs(2)).await;
    let response = String::from_utf8(response).unwrap();

    assert!(
        response.starts_with("HTTP/1.1 200"),
        "unexpected MCP response: {response}"
    );
    assert_eq!(
        response_header(&response, "content-type"),
        Some("text/event-stream")
    );
    assert!(
        response_header(&response, "mcp-session-id").is_some(),
        "MCP initialize response should include a session id: {response}"
    );
    assert!(response.contains("\"serverInfo\""));
    handle.abort();
}

#[tokio::test]
async fn management_mcp_endpoint_rejects_cross_origin_browser_requests() {
    let state = build_test_mesh_api().await;
    let (addr, handle) = spawn_management_test_server(state).await;
    let response = send_management_request(
        addr,
        "POST /mcp HTTP/1.1\r\n\
         Host: localhost\r\n\
         Origin: https://attacker.example\r\n\
         Content-Type: application/json\r\n\
         Content-Length: 2\r\n\r\n{}"
            .to_string(),
    )
    .await;

    assert!(response.starts_with("HTTP/1.1 403"), "{response}");
    handle.await.unwrap().unwrap();
}

#[tokio::test]
async fn management_mcp_endpoint_rejects_dns_rebinding_host_headers() {
    let state = build_test_mesh_api().await;
    let (addr, handle) = spawn_management_test_server(state).await;
    let response = send_management_request(
        addr,
        "POST /mcp HTTP/1.1\r\n\
         Host: attacker.example\r\n\
         Content-Type: application/json\r\n\
         Content-Length: 2\r\n\r\n{}"
            .to_string(),
    )
    .await;

    assert!(response.starts_with("HTTP/1.1 403"), "{response}");
    handle.await.unwrap().unwrap();
}

#[tokio::test]
async fn runtime_data_sse_bridge_delivers_initial_and_incremental_updates() {
    let state = build_test_mesh_api().await;
    let (addr, handle) = spawn_management_test_server(state.clone()).await;

    let mut stream = TcpStream::connect(addr).await.unwrap();
    stream
        .write_all(b"GET /api/events HTTP/1.1\r\nHost: localhost\r\n\r\n")
        .await
        .unwrap();

    let initial = read_until_contains(&mut stream, b"data: {", Duration::from_secs(2)).await;
    let initial_text = String::from_utf8_lossy(&initial);
    assert!(initial_text.contains("HTTP/1.1 200 OK"));
    assert!(initial_text.contains("Content-Type: text/event-stream"));
    assert!(initial_text.contains("\"llama_ready\":false"));
    assert!(initial_text.contains("\"publication_state\":\"private\""));

    state.update(true, true).await;
    let runtime_update =
        read_until_contains(&mut stream, b"\"llama_ready\":true", Duration::from_secs(2)).await;
    let runtime_update_text = String::from_utf8_lossy(&runtime_update);
    assert!(runtime_update_text.contains("\"llama_ready\":true"));
    assert!(runtime_update_text.contains("\"is_host\":true"));

    state
        .set_publication_state(crate::api::PublicationState::PublishFailed)
        .await;
    let publication_update = read_until_contains(
        &mut stream,
        b"\"publication_state\":\"publish_failed\"",
        Duration::from_secs(2),
    )
    .await;
    let publication_update_text = String::from_utf8_lossy(&publication_update);
    assert!(publication_update_text.contains("\"publication_state\":\"publish_failed\""));

    drop(stream);
    handle.abort();
}

#[tokio::test]
async fn api_runtime_reads_from_collector_snapshot() {
    let state = build_test_mesh_api().await;

    {
        let mut inner = state.inner.lock().await;
        inner.primary_backend = Some("legacy-backend".into());
        inner.is_host = false;
        inner.llama_ready = false;
        inner.llama_port = Some(9999);
        inner.local_processes = vec![RuntimeProcessPayload {
            name: "legacy-model".into(),
            instance_id: None,
            backend: "legacy-backend".into(),
            status: "ready".into(),
            port: 9999,
            pid: 111,
            slots: 4,
            context_length: None,
            profile: String::new(),
        }];

        inner
            .runtime_data_producer
            .publish_runtime_status(|runtime_status| {
                runtime_status.primary_model = Some("collector-model".into());
                runtime_status.primary_backend = Some("collector-backend".into());
                runtime_status.is_host = true;
                runtime_status.llama_ready = true;
                runtime_status.llama_port = Some(9337);
                true
            });
        inner
            .runtime_data_producer
            .publish_local_processes(|local_processes| {
                local_processes.clear();
                local_processes.push(runtime_data::RuntimeProcessSnapshot {
                    model: "collector-model".into(),
                    instance_id: None,
                    profile: String::new(),
                    backend: "collector-backend".into(),
                    pid: 777,
                    port: 9337,
                    slots: 4,
                    context_length: Some(0),
                    command: Some("llama-server".into()),
                    state: "ready".into(),
                    start: Some(1_700_000_000),
                    health: Some("ready".into()),
                });
                true
            });
    }

    let runtime_status = state.runtime_status().await;
    assert_eq!(runtime_status.models.len(), 1);
    assert_eq!(runtime_status.models[0].name, "collector-model");
    assert_eq!(runtime_status.models[0].backend, "collector-backend");
    assert_eq!(runtime_status.models[0].status, "ready");
    assert_eq!(runtime_status.models[0].port, Some(9337));

    let runtime_processes = state.runtime_processes().await;
    assert_eq!(runtime_processes.processes.len(), 1);
    assert_eq!(runtime_processes.processes[0].name, "collector-model");
    assert_eq!(runtime_processes.processes[0].backend, "collector-backend");
    assert_eq!(runtime_processes.processes[0].status, "ready");
    assert_eq!(runtime_processes.processes[0].port, 9337);
    assert_eq!(runtime_processes.processes[0].pid, 777);
}
