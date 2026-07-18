#[tokio::test]
async fn control_plane_api_exposes_local_endpoint_only() {
    let state = build_test_mesh_api().await;
    state
        .set_control_bootstrap(crate::api::ControlBootstrapPayload {
            enabled: true,
            local_only: true,
            requires_explicit_remote_endpoint: true,
            endpoint: Some("http://127.0.0.1:7447".to_string()),
            disabled_reason: None,
            message: None,
            suggested_commands: None,
        })
        .await;
    let (addr, handle) = spawn_management_test_server(state).await;

    let response = send_management_request(
        addr,
        "GET /api/runtime/control-bootstrap HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
    )
    .await;
    let body = json_body(&response);

    assert_eq!(body["enabled"], serde_json::Value::Bool(true));
    assert_eq!(body["local_only"], serde_json::Value::Bool(true));
    assert_eq!(
        body["requires_explicit_remote_endpoint"],
        serde_json::Value::Bool(true)
    );
    assert_eq!(
        body["endpoint"],
        serde_json::Value::String("http://127.0.0.1:7447".into())
    );

    handle.await.unwrap().unwrap();
}

#[tokio::test]
async fn control_plane_api_explains_disabled_owner_control() {
    let state = build_test_mesh_api().await;
    let (addr, handle) = spawn_management_test_server(state).await;

    let response = send_management_request(
        addr,
        "GET /api/runtime/control-bootstrap HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
    )
    .await;
    let body = json_body(&response);

    assert_eq!(body["enabled"], serde_json::Value::Bool(false));
    assert_eq!(body["local_only"], serde_json::Value::Bool(true));
    assert_eq!(body["disabled_reason"], "missing_owner_identity");
    assert_eq!(
        body["message"],
        "Configuration saving requires a local owner identity."
    );
    assert_eq!(
        body["suggested_commands"],
        serde_json::json!([
            "mesh-llm auth status",
            "mesh-llm auth init --no-passphrase",
            "mesh-llm serve --owner-required"
        ])
    );
    assert!(body.get("endpoint").is_none());

    handle.await.unwrap().unwrap();
}

#[tokio::test]
async fn status_payload_control_plane_compat() {
    let state = build_test_mesh_api().await;
    state
        .set_control_bootstrap(crate::api::ControlBootstrapPayload {
            enabled: true,
            local_only: true,
            requires_explicit_remote_endpoint: true,
            endpoint: Some("control-endpoint-token".to_string()),
            disabled_reason: None,
            message: None,
            suggested_commands: None,
        })
        .await;

    let payload = serde_json::to_value(state.status().await).unwrap();
    assert!(payload.get("control_bootstrap").is_none());
    assert!(payload.get("control_endpoint").is_none());
    assert!(
        payload["peers"].as_array().unwrap().iter().all(|peer| {
            peer.get("control_endpoint").is_none() && peer.get("endpoint").is_none()
        })
    );
}

#[tokio::test]
async fn mesh_guardrails_runtime_mode_accepts_loopback_callers() {
    let state = build_test_mesh_api().await;
    let (control_tx, mut control_rx) = mpsc::unbounded_channel();
    state.set_runtime_control(control_tx).await;
    let (addr, handle) = spawn_management_test_server(state).await;
    let control_handle = tokio::spawn(async move {
        match control_rx.recv().await {
            Some(RuntimeControlRequest::SetOpenAiGuardrailMode { mode, resp }) => {
                assert_eq!(mode, openai_frontend::GuardrailMode::Enforce);
                let _ = resp.send(Ok(OpenAiGuardrailModeUpdateResponse {
                    mode: "enforce",
                    updated_models: 1,
                    status: None,
                }));
            }
            _ => panic!("expected SetOpenAiGuardrailMode request"),
        }
    });
    let body = r#"{"mode":"enforce"}"#;

    let response = send_management_request(
        addr,
        format!(
            "POST /api/runtime/mesh-guardrails HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        ),
    )
    .await;

    assert!(
        response.starts_with("HTTP/1.1 200 OK"),
        "response was {response:?}"
    );
    assert_eq!(
        json_body(&response)["mode"],
        serde_json::Value::String("enforce".to_string())
    );
    handle.await.unwrap().unwrap();
    control_handle.await.unwrap();
}

#[tokio::test]
async fn config_apply_does_not_emit_peer_churn() {
    let state = build_test_mesh_api().await;
    let (addr, handle) = spawn_management_test_server(state.clone()).await;

    let mut stream = TcpStream::connect(addr).await.unwrap();
    stream
        .write_all(b"GET /api/events HTTP/1.1\r\nHost: localhost\r\n\r\n")
        .await
        .unwrap();

    let initial = read_until_contains(&mut stream, b"data: {", Duration::from_secs(2)).await;
    let initial_text = String::from_utf8_lossy(&initial);
    assert!(initial_text.contains("\"peers\":"));
    assert!(!initial_text.contains("control-endpoint-token"));

    state
        .set_control_bootstrap(crate::api::ControlBootstrapPayload {
            enabled: true,
            local_only: true,
            requires_explicit_remote_endpoint: true,
            endpoint: Some("control-endpoint-token".to_string()),
            disabled_reason: None,
            message: None,
            suggested_commands: None,
        })
        .await;
    state.push_status().await;

    assert_no_stream_bytes_within(&mut stream, Duration::from_millis(250)).await;

    state.update(true, true).await;
    let updated =
        read_until_contains(&mut stream, b"\"llama_ready\":true", Duration::from_secs(2)).await;
    let updated_text = String::from_utf8_lossy(&updated);
    assert!(updated_text.contains("\"llama_ready\":true"));
    assert!(updated_text.contains("\"is_host\":true"));

    drop(stream);
    handle.abort();
}

#[tokio::test]
#[serial]
async fn control_plane_api_cli_requires_explicit_endpoint_and_runs_local_orchestration() {
    let temp = tempfile::tempdir().unwrap();
    // TODO: Audit that the environment access only happens in single-threaded code.
    unsafe { std::env::set_var("HOME", temp.path()) };
    let owner = OwnerKeypair::generate();
    let keystore_path = default_keystore_path().unwrap();
    save_keystore(&keystore_path, &owner, None, true).unwrap();

    let control_server = spawn_owner_control_test_server().await;
    let state = build_test_mesh_api().await;
    state.set_owner_key_path(Some(keystore_path)).await;
    let (addr, handle) = spawn_management_test_server(state.clone()).await;

    let missing_request_body = "{}";
    let missing = send_management_request(
        addr,
        format!(
            "POST /api/runtime/control/get-config HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            missing_request_body.len(),
            missing_request_body
        ),
    )
    .await;
    let missing_body = json_body(&missing);
    assert_eq!(missing_body["error"]["code"], "control_endpoint_required");
    handle.await.unwrap().unwrap();

    let (addr, handle) = spawn_management_test_server(state).await;
    let request_body = json!({ "endpoint": control_server.endpoint_token }).to_string();
    let response = send_management_request(
        addr,
        format!(
            "POST /api/runtime/control/get-config HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            request_body.len(),
            request_body
        ),
    )
    .await;
    let body = json_body(&response);
    assert_eq!(body["snapshot"]["revision"], 42, "response: {response}");
    assert_eq!(body["snapshot"]["hostname"], "control-target");
    assert_eq!(body["snapshot"]["config"]["version"], 1);

    handle.await.unwrap().unwrap();
    control_server.task.abort();
}

#[tokio::test]
#[serial]
async fn control_plane_api_apply_config_uses_full_mesh_config_contract() {
    let temp = tempfile::tempdir().unwrap();
    // TODO: Audit that the environment access only happens in single-threaded code.
    unsafe { std::env::set_var("HOME", temp.path()) };
    let owner = OwnerKeypair::generate();
    let keystore_path = default_keystore_path().unwrap();
    save_keystore(&keystore_path, &owner, None, true).unwrap();

    let get_server = spawn_owner_control_test_server().await;
    let OwnerControlApplyTestServer {
        endpoint_token: apply_endpoint_token,
        task: control_task,
        received_apply,
    } = spawn_owner_control_apply_test_server(OwnerControlApplyTestResponse::Success(
        OwnerControlApplyConfigResponse {
            success: true,
            current_revision: 43,
            config_hash: vec![0xab; 32],
            error: None,
            apply_mode: ConfigApplyMode::Staged as i32,
            diagnostics: Vec::new(),
        },
    ))
    .await;
    let state = build_test_mesh_api().await;
    state.set_owner_key_path(Some(keystore_path)).await;
    let (addr, handle) = spawn_management_test_server(state.clone()).await;

    let get_request_body = json!({ "endpoint": get_server.endpoint_token }).to_string();
    let get_response = send_management_request(
        addr,
        management_post_request("/api/runtime/control/get-config", &get_request_body),
    )
    .await;
    let get_body = json_body(&get_response);
    assert_eq!(
        get_body["snapshot"]["revision"], 42,
        "response: {get_response}"
    );
    let mut merged_config_json = get_body["snapshot"]["config"].clone();
    merge_json_object(
        &mut merged_config_json,
        serde_json::to_value(full_mesh_config_fixture()).unwrap(),
    );
    let expected_config: crate::plugin::MeshConfig =
        serde_json::from_value(merged_config_json).unwrap();
    handle.await.unwrap().unwrap();

    let (addr, handle) = spawn_management_test_server(state).await;

    let apply_request_body = json!({
        "endpoint": apply_endpoint_token,
        "expected_revision": get_body["snapshot"]["revision"],
        "config": expected_config.clone(),
    })
    .to_string();
    let apply_response = send_management_request(
        addr,
        management_post_request("/api/runtime/control/apply-config", &apply_request_body),
    )
    .await;
    let apply_body = json_body(&apply_response);
    assert!(
        apply_response.starts_with("HTTP/1.1 200"),
        "response: {apply_response}"
    );
    assert_eq!(apply_body["success"], true);
    assert_eq!(apply_body["current_revision"], 43);
    assert_eq!(apply_body["apply_mode"], "staged");
    assert_eq!(
        apply_body["config_hash"],
        "abababababababababababababababababababababababababababababababab"
    );

    let received_apply = received_apply
        .expect("apply-config flow should capture the forwarded full MeshConfig")
        .await
        .unwrap();
    assert_eq!(received_apply.expected_revision, 42);
    assert_eq!(
        received_apply.config,
        Some(crate::protocol::convert::mesh_config_to_proto(
            &expected_config
        ))
    );

    handle.await.unwrap().unwrap();
    get_server.task.abort();
    control_task.await.unwrap();
}

#[tokio::test]
#[serial]
async fn control_plane_api_apply_config_reports_revision_conflict() {
    let temp = tempfile::tempdir().unwrap();
    // TODO: Audit that the environment access only happens in single-threaded code.
    unsafe { std::env::set_var("HOME", temp.path()) };
    let owner = OwnerKeypair::generate();
    let keystore_path = default_keystore_path().unwrap();
    save_keystore(&keystore_path, &owner, None, true).unwrap();

    let OwnerControlApplyTestServer {
        endpoint_token,
        task: control_task,
        received_apply,
    } = spawn_owner_control_apply_test_server(OwnerControlApplyTestResponse::Error {
        code: OwnerControlErrorCode::RevisionConflict,
        message: "stale config revision".to_string(),
        current_revision: Some(7),
    })
    .await;
    let state = build_test_mesh_api().await;
    state.set_owner_key_path(Some(keystore_path)).await;
    let (addr, handle) = spawn_management_test_server(state).await;

    let request_body = json!({
        "endpoint": endpoint_token,
        "expected_revision": 6,
        "config": full_mesh_config_fixture(),
    })
    .to_string();
    let response = send_management_request(
        addr,
        management_post_request("/api/runtime/control/apply-config", &request_body),
    )
    .await;
    let body = json_body(&response);
    assert!(response.starts_with("HTTP/1.1 409"), "response: {response}");
    assert_eq!(body["error"]["code"], "revision_conflict");
    assert_eq!(body["error"]["message"], "stale config revision");
    assert_eq!(body["error"]["current_revision"], 7);

    let received_apply = received_apply
        .expect("revision conflict path should still capture apply requests")
        .await
        .unwrap();
    assert_eq!(received_apply.expected_revision, 6);

    handle.await.unwrap().unwrap();
    control_task.await.unwrap();
}

#[tokio::test]
async fn control_plane_api_apply_config_rejects_invalid_json() {
    let state = build_test_mesh_api().await;
    let (addr, handle) = spawn_management_test_server(state).await;

    let request_body = "{\"endpoint\":";
    let response = send_management_request(
        addr,
        management_post_request("/api/runtime/control/apply-config", request_body),
    )
    .await;
    let body = json_body(&response);
    assert!(response.starts_with("HTTP/1.1 400"), "response: {response}");
    assert_eq!(body["error"], "Invalid JSON body");

    handle.await.unwrap().unwrap();
}

#[tokio::test]
async fn control_route_rejects_non_loopback() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let client = tokio::spawn(async move {
        let mut stream = TcpStream::connect(addr).await.unwrap();
        let mut response = Vec::new();
        stream.read_to_end(&mut response).await.unwrap();
        String::from_utf8(response).unwrap()
    });

    let (mut server_stream, _) = listener.accept().await.unwrap();
    let allowed = crate::api::routes::runtime::ensure_loopback_control_caller_for_peer_addr(
        &mut server_stream,
        Ok(std::net::SocketAddr::from(([192, 0, 2, 10], 40123))),
    )
    .await
    .unwrap();
    assert!(!allowed);
    drop(server_stream);

    let response = client.await.unwrap();
    let body = json_body(&response);
    assert!(response.starts_with("HTTP/1.1 403"), "response: {response}");
    assert_eq!(
        body["error"],
        "runtime control endpoints only accept localhost connections"
    );
}

#[tokio::test]
#[serial]
async fn control_plane_api_reports_remote_endpoint_unreachable() {
    let temp = tempfile::tempdir().unwrap();
    // TODO: Audit that the environment access only happens in single-threaded code.
    unsafe { std::env::set_var("HOME", temp.path()) };
    let owner = OwnerKeypair::generate();
    let keystore_path = default_keystore_path().unwrap();
    save_keystore(&keystore_path, &owner, None, true).unwrap();

    let endpoint_token = unreachable_owner_control_endpoint_token().await;
    let state = build_test_mesh_api().await;
    state.set_owner_key_path(Some(keystore_path)).await;
    let (addr, handle) = spawn_management_test_server(state).await;

    let request_body = json!({ "endpoint": endpoint_token }).to_string();
    let response = send_management_request(
        addr,
        format!(
            "POST /api/runtime/control/get-config HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            request_body.len(),
            request_body
        ),
    )
    .await;
    let body = json_body(&response);
    let message = body["error"]["message"].as_str().unwrap_or_default();
    assert!(response.starts_with("HTTP/1.1 503"), "response: {response}");
    assert_eq!(body["error"]["code"], "control_unavailable");
    assert_eq!(body["error"]["legacy_retry_allowed"], false);
    assert!(
        message.contains("remote owner-control endpoint is unavailable or unreachable"),
        "message: {message}"
    );
    assert!(
        !message.contains("mesh-llm console"),
        "remote reachability failure should not be reported as a local console failure: {message}"
    );

    handle.await.unwrap().unwrap();
}

#[tokio::test]
#[serial]
async fn control_plane_api_cli_uses_custom_owner_key_path() {
    let temp = tempfile::tempdir().unwrap();
    // TODO: Audit that the environment access only happens in single-threaded code.
    unsafe { std::env::set_var("HOME", temp.path()) };
    let custom_owner_key = temp.path().join("custom-owner.json");
    save_keystore(&custom_owner_key, &OwnerKeypair::generate(), None, true).unwrap();

    let control_server = spawn_owner_control_test_server().await;
    let state = build_test_mesh_api().await;
    state.set_owner_key_path(Some(custom_owner_key)).await;
    let (addr, handle) = spawn_management_test_server(state).await;

    let request_body = json!({ "endpoint": control_server.endpoint_token }).to_string();
    let response = send_management_request(
        addr,
        format!(
            "POST /api/runtime/control/get-config HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            request_body.len(),
            request_body
        ),
    )
    .await;
    let body = json_body(&response);
    assert_eq!(body["snapshot"]["revision"], 42, "response: {response}");

    handle.await.unwrap().unwrap();
    control_server.task.abort();
}
