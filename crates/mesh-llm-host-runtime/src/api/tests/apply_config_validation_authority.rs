use super::*;

#[tokio::test]
#[serial]
async fn control_plane_api_apply_config_rejects_gpu_assignment_conflict_before_owner_roundtrip() {
    let state = build_test_mesh_api().await;
    let (addr, handle) = spawn_management_test_server(state).await;

    let mut config =
        serde_json::to_value(full_mesh_config_fixture()).expect("fixture should serialize");
    config["models"][0]["hardware"] = json!({ "device": "metal:0" });
    let apply_request_body = json!({
        "endpoint": "control://ignored",
        "expected_revision": 7,
        "config": config,
    })
    .to_string();

    let apply_response = send_management_request(
        addr,
        management_post_request("/api/runtime/control/apply-config", &apply_request_body),
    )
    .await;
    let apply_body = json_body(&apply_response);
    let diagnostics = apply_body["diagnostics"]
        .as_array()
        .expect("diagnostics should be an array");
    let device_diagnostic = diagnostics
        .iter()
        .find(|diagnostic| diagnostic["path"] == "models[0].hardware.device")
        .expect("device diagnostic should be present");

    assert!(
        apply_response.starts_with("HTTP/1.1 200"),
        "response: {apply_response}"
    );
    assert_eq!(apply_body["success"], false, "response: {apply_response}");
    assert_eq!(apply_body["apply_mode"], "unspecified");
    assert_eq!(device_diagnostic["code"], "invalid_value");
    assert_eq!(device_diagnostic["severity"], "error");
    assert_eq!(device_diagnostic["source"], "validation");
    assert_eq!(device_diagnostic["schema_source"], "built_in");
    assert_eq!(
        device_diagnostic["canonical_path"],
        "models.<model-ref>.hardware.device"
    );
    assert!(
        device_diagnostic["message"]
            .as_str()
            .expect("message should be a string")
            .contains("must not be set when gpu.assignment = \"auto\"")
    );

    handle.await.unwrap().unwrap();
}
