use super::*;

struct HomeEnvGuard {
    original_home: Option<std::ffi::OsString>,
}

impl HomeEnvGuard {
    fn set(home: &std::path::Path) -> Self {
        let original_home = std::env::var_os("HOME");
        unsafe { std::env::set_var("HOME", home) };
        Self { original_home }
    }
}

impl Drop for HomeEnvGuard {
    fn drop(&mut self) {
        match &self.original_home {
            Some(home) => unsafe { std::env::set_var("HOME", home) },
            None => unsafe { std::env::remove_var("HOME") },
        }
    }
}

#[tokio::test]
#[serial]
async fn control_plane_api_apply_config_serializes_structured_diagnostics() {
    let temp = tempfile::tempdir().unwrap();
    let _home_guard = HomeEnvGuard::set(temp.path());
    let owner = OwnerKeypair::generate();
    let keystore_path = default_keystore_path().unwrap();
    save_keystore(&keystore_path, &owner, None, true).unwrap();

    let OwnerControlApplyTestServer {
        endpoint_token,
        task: control_task,
        received_apply: _,
    } = spawn_owner_control_apply_test_server(OwnerControlApplyTestResponse::Success(
        OwnerControlApplyConfigResponse {
            success: false,
            current_revision: 7,
            config_hash: vec![0xcd; 32],
            error: Some(
                "models[0].request_defaults.reasoning_format must be one of: auto, none, deepseek, deepseek-legacy, hidden"
                    .to_string(),
            ),
            apply_mode: ConfigApplyMode::Unspecified as i32,
            diagnostics: vec![mesh_client::proto::node::ConfigDiagnostic {
                code: mesh_client::proto::node::ConfigDiagnosticCode::InvalidValue as i32,
                severity: mesh_client::proto::node::ConfigDiagnosticSeverity::Error as i32,
                source: mesh_client::proto::node::ConfigDiagnosticSource::Validation as i32,
                schema_source: Some(
                    mesh_client::proto::node::ConfigDiagnosticSchemaSource::BuiltIn as i32,
                ),
                path: Some("models[0].request_defaults.reasoning_format".to_string()),
                canonical_path: Some(
                    "models.<model-ref>.request_defaults.reasoning_format".to_string(),
                ),
                message:
                    "models[0].request_defaults.reasoning_format must be one of: auto, none, deepseek, deepseek-legacy, hidden"
                        .to_string(),
                help: Some("choose one of the supported reasoning formats".to_string()),
            }],
        },
    ))
    .await;
    let state = build_test_mesh_api().await;
    state.set_owner_key_path(Some(keystore_path)).await;
    let (addr, handle) = spawn_management_test_server(state).await;

    let apply_request_body = json!({
        "endpoint": endpoint_token,
        "expected_revision": 7,
        "config": full_mesh_config_fixture(),
    })
    .to_string();
    let apply_response = send_management_request(
        addr,
        management_post_request("/api/runtime/control/apply-config", &apply_request_body),
    )
    .await;
    let apply_body = json_body(&apply_response);

    assert_eq!(apply_body["success"], false, "response: {apply_response}");
    assert_eq!(apply_body["current_revision"], 7);
    assert_eq!(apply_body["apply_mode"], "unspecified");
    assert_eq!(
        apply_body["error"],
        "models[0].request_defaults.reasoning_format must be one of: auto, none, deepseek, deepseek-legacy, hidden"
    );
    assert_eq!(apply_body["diagnostics"][0]["code"], "invalid_value");
    assert_eq!(apply_body["diagnostics"][0]["severity"], "error");
    assert_eq!(apply_body["diagnostics"][0]["source"], "validation");
    assert_eq!(apply_body["diagnostics"][0]["schema_source"], "built_in");
    assert_eq!(
        apply_body["diagnostics"][0]["path"],
        "models[0].request_defaults.reasoning_format"
    );
    assert_eq!(
        apply_body["diagnostics"][0]["help"],
        "choose one of the supported reasoning formats"
    );

    handle.await.unwrap().unwrap();
    control_task.await.unwrap();
}

#[tokio::test]
#[serial]
async fn control_plane_api_apply_config_serializes_success_warning_diagnostics() {
    let temp = tempfile::tempdir().unwrap();
    let _home_guard = HomeEnvGuard::set(temp.path());
    let owner = OwnerKeypair::generate();
    let keystore_path = default_keystore_path().unwrap();
    save_keystore(&keystore_path, &owner, None, true).unwrap();

    let OwnerControlApplyTestServer {
        endpoint_token,
        task: control_task,
        received_apply: _,
    } = spawn_owner_control_apply_test_server(OwnerControlApplyTestResponse::Success(
        OwnerControlApplyConfigResponse {
            success: true,
            current_revision: 8,
            config_hash: vec![0xef; 32],
            error: None,
            apply_mode: ConfigApplyMode::Staged as i32,
            diagnostics: vec![mesh_client::proto::node::ConfigDiagnostic {
                code: mesh_client::proto::node::ConfigDiagnosticCode::LegacyUnvalidatedConfig
                    as i32,
                severity: mesh_client::proto::node::ConfigDiagnosticSeverity::Warning as i32,
                source: mesh_client::proto::node::ConfigDiagnosticSource::Plugin as i32,
                schema_source: Some(
                    mesh_client::proto::node::ConfigDiagnosticSchemaSource::Plugin as i32,
                ),
                path: Some("plugin.blackboard.settings".to_string()),
                canonical_path: Some("plugin.blackboard.settings".to_string()),
                message:
                    "plugin 'blackboard' allows legacy unvalidated config; custom settings are accepted without schema checks"
                        .to_string(),
                help: None,
            }],
        },
    ))
    .await;
    let state = build_test_mesh_api().await;
    state.set_owner_key_path(Some(keystore_path)).await;
    let (addr, handle) = spawn_management_test_server(state).await;

    let apply_request_body = json!({
        "endpoint": endpoint_token,
        "expected_revision": 7,
        "config": full_mesh_config_fixture(),
    })
    .to_string();
    let apply_response = send_management_request(
        addr,
        management_post_request("/api/runtime/control/apply-config", &apply_request_body),
    )
    .await;
    let apply_body = json_body(&apply_response);

    assert_eq!(apply_body["success"], true, "response: {apply_response}");
    assert_eq!(apply_body["current_revision"], 8);
    assert_eq!(apply_body["apply_mode"], "staged");
    assert_eq!(apply_body["error"], serde_json::Value::Null);
    assert_eq!(
        apply_body["diagnostics"][0]["code"],
        "legacy_unvalidated_config"
    );
    assert_eq!(apply_body["diagnostics"][0]["severity"], "warning");
    assert_eq!(apply_body["diagnostics"][0]["source"], "plugin");
    assert_eq!(apply_body["diagnostics"][0]["schema_source"], "plugin");
    assert_eq!(
        apply_body["diagnostics"][0]["canonical_path"],
        "plugin.blackboard.settings"
    );

    handle.await.unwrap().unwrap();
    control_task.await.unwrap();
}

#[tokio::test]
#[serial]
async fn control_plane_api_apply_config_preserves_misplaced_plugin_key_diagnostics() {
    let state = build_test_mesh_api().await;
    let (addr, handle) = spawn_management_test_server(state).await;

    let apply_request_body = json!({
        "endpoint": "control://ignored",
        "expected_revision": 7,
        "config": {
            "version": 1,
            "plugin": [
                {
                    "name": "blackboard",
                    "retention_days": 14,
                    "settings": {
                        "mode": "strict"
                    }
                }
            ]
        }
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
    assert_eq!(apply_body["success"], false);
    assert_eq!(apply_body["apply_mode"], "unspecified");
    assert!(
        apply_body["diagnostics"]
            .as_array()
            .expect("diagnostics should be an array")
            .iter()
            .any(|diagnostic| diagnostic["code"] == "misplaced_field")
    );

    handle.await.unwrap().unwrap();
}
