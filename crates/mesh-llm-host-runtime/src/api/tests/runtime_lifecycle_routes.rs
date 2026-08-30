async fn send_runtime_lifecycle_request(state: MeshApi, request: String) -> String {
    let (addr, handle) = spawn_management_test_server(state).await;
    let response = send_management_request(addr, request).await;
    handle.await.unwrap().unwrap();
    response
}

fn lifecycle_post(path: &str, body: &str) -> String {
    format!(
        "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
}

#[tokio::test]
async fn runtime_status_and_activity_routes_are_dispatched() {
    let state = build_test_mesh_api().await;

    let intents = send_runtime_lifecycle_request(
        state.clone(),
        "GET /api/runtime/intents HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
    )
    .await;
    assert!(
        intents.starts_with("HTTP/1.1 200 OK"),
        "unexpected intents response: {intents}"
    );
    assert!(json_body(&intents)["intents"].is_array());

    let activity = send_runtime_lifecycle_request(
        state,
        "GET /api/runtime/activity HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
    )
    .await;
    assert!(
        activity.starts_with("HTTP/1.1 200 OK"),
        "unexpected activity response: {activity}"
    );
    assert_eq!(json_body(&activity)["override_mode"], "auto");
}

#[tokio::test]
async fn local_runtime_model_routes_remain_the_lifecycle_api() {
    let state = build_test_mesh_api().await;
    let (control_tx, mut control_rx) = mpsc::unbounded_channel();
    state.set_runtime_control(control_tx).await;

    let load_task = tokio::spawn(async move {
        match control_rx.recv().await {
            Some(RuntimeControlRequest::Load {
                spec,
                config_model_id,
                profile,
                resp,
            }) => {
                assert_eq!(spec, "org/model:Q4_K_M");
                assert_eq!(config_model_id, None);
                assert_eq!(profile, "low-ctx");
                let _ = resp.send(Ok(RuntimeLoadResponse {
                    model_ref: spec.clone(),
                    model: spec,
                    instance_id: "runtime-load-1".to_string(),
                    profile,
                    backend: None,
                    context_length: None,
                }));
            }
            _ => panic!("expected local load request"),
        }
    });

    let loaded = send_runtime_lifecycle_request(
        state.clone(),
        lifecycle_post(
            "/api/runtime/models",
            r#"{"model":"org/model:Q4_K_M#low-ctx"}"#,
        ),
    )
    .await;
    load_task.await.unwrap();
    assert!(loaded.starts_with("HTTP/1.1 201 Created"), "response: {loaded}");
    assert_eq!(json_body(&loaded)["loaded"], "org/model:Q4_K_M");

    let (control_tx, mut control_rx) = mpsc::unbounded_channel();
    state.set_runtime_control(control_tx).await;
    let unload_task = tokio::spawn(async move {
        match control_rx.recv().await {
            Some(RuntimeControlRequest::Unload {
                target: mesh_llm_node::serving::UnloadTarget::Model(model),
                options: _,
                resp,
            }) => {
                assert_eq!(model, "org/model:Q4_K_M");
                let _ = resp.send(Ok(RuntimeUnloadResponse {
                    model,
                    instance_id: "runtime-load-1".to_string(),
                    unloaded: true,
                }));
            }
            _ => panic!("expected local unload request"),
        }
    });

    let unloaded = send_runtime_lifecycle_request(
        state,
        "DELETE /api/runtime/models/org%2Fmodel%3AQ4_K_M HTTP/1.1\r\nHost: localhost\r\n\r\n"
            .into(),
    )
    .await;
    unload_task.await.unwrap();
    assert!(unloaded.starts_with("HTTP/1.1 200 OK"), "response: {unloaded}");
    assert_eq!(json_body(&unloaded)["dropped"], "org/model:Q4_K_M");
}

#[tokio::test]
async fn runtime_activity_override_uses_put_and_delete() {
    let state = build_test_mesh_api().await;
    let body = r#""active""#;
    let put = send_runtime_lifecycle_request(
        state.clone(),
        format!(
            "PUT /api/runtime/activity/override HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        ),
    )
    .await;
    assert!(
        put.starts_with("HTTP/1.1 200 OK"),
        "unexpected PUT response: {put}"
    );
    assert_eq!(json_body(&put)["override_mode"], "active");

    let delete = send_runtime_lifecycle_request(
        state,
        "DELETE /api/runtime/activity/override HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
    )
    .await;
    assert!(
        delete.starts_with("HTTP/1.1 200 OK"),
        "unexpected DELETE response: {delete}"
    );
    assert_eq!(json_body(&delete)["override_mode"], "auto");
}

#[tokio::test]
async fn owner_lifecycle_routes_enforce_exact_target_shapes_before_connecting() {
    let state = build_test_mesh_api().await;
    let invalid = [
        (
            "/api/runtime/control/load-model",
            r#"{"endpoint":"unused","model":"model/test","instance_id":"runtime-2"}"#,
            "load-model requires a model reference",
        ),
        (
            "/api/runtime/control/ensure-model",
            r#"{"endpoint":"unused","instance_id":"runtime-2"}"#,
            "ensure-model requires a model reference",
        ),
        (
            "/api/runtime/control/unload-model",
            r#"{"endpoint":"unused","model":"model/test","instance_id":"runtime-2"}"#,
            "unload-model requires exactly one",
        ),
        (
            "/api/runtime/control/drain-model",
            r#"{"endpoint":"unused"}"#,
            "drain-model requires exactly one",
        ),
        (
            "/api/runtime/control/unload-model",
            r#"{"endpoint":"unused","model":"model/test","profile":"low-ctx"}"#,
            "unload-model requires exactly one",
        ),
        (
            "/api/runtime/control/load-model",
            r#"{"model":"model/test"}"#,
            "control_endpoint_required",
        ),
    ];

    for (path, body, expected) in invalid {
        let response =
            send_runtime_lifecycle_request(state.clone(), lifecycle_post(path, body)).await;
        assert!(
            response.starts_with("HTTP/1.1 400 Bad Request"),
            "unexpected response for {path}: {response}"
        );
        assert!(response.contains(expected), "unexpected response: {response}");
    }
}
