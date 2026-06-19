use super::*;

#[tokio::test]
async fn runtime_config_schema_api_exposes_control_metadata() {
    let state = build_test_mesh_api().await;
    let (addr, handle) = spawn_management_test_server(state).await;

    let response = send_management_request(
        addr,
        "GET /api/runtime/config-schema HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
    )
    .await;
    let body = json_body(&response);
    let settings = body["settings"]
        .as_array()
        .expect("config schema response should contain settings");
    let temperature = settings
        .iter()
        .find(|entry| entry["canonical_path"] == "defaults.request_defaults.temperature")
        .expect("temperature default should be exported");

    assert_eq!(temperature["owner"], "built_in");
    assert_eq!(temperature["source"]["kind"], "built_in");
    assert_eq!(temperature["support"], "supported");
    assert_eq!(temperature["value_schema"]["kind"], "float");
    assert_eq!(temperature["restart_scope"], "model_reload");
    assert_eq!(temperature["presentation"]["label"], "Temperature");
    assert_eq!(
        temperature["presentation"]["category_id"],
        "request-defaults"
    );
    let plugin_instances = body["plugin_instances"]
        .as_array()
        .expect("config schema response should contain plugin instances");
    assert!(
        plugin_instances
            .iter()
            .any(|instance| instance["name"] == crate::plugin::BLOBSTORE_PLUGIN_ID)
    );

    handle.await.unwrap().unwrap();
}

#[tokio::test]
async fn runtime_config_validate_api_reports_toml_diagnostics() {
    let state = build_test_mesh_api().await;
    let (addr, handle) = spawn_management_test_server(state).await;

    let valid_body = r#"{"toml":"version = 1\n","path":"x"}"#;
    let response = send_management_request(
        addr,
        format!(
            "POST /api/runtime/config/validate HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            valid_body.len(),
            valid_body
        ),
    )
    .await;
    let body = json_body(&response);

    assert_eq!(body["ok"], serde_json::Value::Bool(true));
    assert_eq!(body["path"], "x");
    assert!(body["diagnostics"].as_array().unwrap().is_empty());
    handle.await.unwrap().unwrap();

    let state = build_test_mesh_api().await;
    let (addr, handle) = spawn_management_test_server(state).await;
    let invalid_body = r#"{"toml":"not valid = ["}"#;
    let response = send_management_request(
        addr,
        format!(
            "POST /api/runtime/config/validate HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            invalid_body.len(),
            invalid_body
        ),
    )
    .await;
    let body = json_body(&response);

    assert_eq!(body["ok"], serde_json::Value::Bool(false));
    assert!(body["error"].as_str().unwrap().contains("TOML"));

    handle.await.unwrap().unwrap();
}
