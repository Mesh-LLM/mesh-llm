use super::*;

#[tokio::test]
#[serial]
async fn owner_control_scan_refresh_rejects_loopback_caller_without_owner_capability() {
    let state = build_test_mesh_api().await;
    let (addr, handle) = spawn_management_test_server(state).await;

    let request_body = json!({}).to_string();
    let response = send_management_request(
        addr,
        management_post_request("/api/runtime/control/scan-refresh", &request_body),
    )
    .await;

    assert!(
        response.starts_with("HTTP/1.1 401") || response.starts_with("HTTP/1.1 403"),
        "scan-refresh accepted a loopback caller that presented no owner capability: {response}"
    );

    handle.await.unwrap().unwrap();
}

#[tokio::test]
#[serial]
async fn owner_control_refresh_inventory_rejects_loopback_caller_without_owner_capability() {
    let state = build_test_mesh_api().await;
    let (addr, handle) = spawn_management_test_server(state).await;

    let request_body = json!({}).to_string();
    let response = send_management_request(
        addr,
        management_post_request("/api/runtime/control/refresh-inventory", &request_body),
    )
    .await;

    assert!(
        response.starts_with("HTTP/1.1 401") || response.starts_with("HTTP/1.1 403"),
        "refresh-inventory accepted a loopback caller that presented no owner capability: {response}"
    );

    handle.await.unwrap().unwrap();
}

#[tokio::test]
#[serial]
async fn owner_control_kv_cache_rejects_loopback_caller_without_owner_capability() {
    let state = build_test_mesh_api().await;
    let (addr, handle) = spawn_management_test_server(state).await;

    let request_body = json!({
        "endpoints": ["http://127.0.0.1:1"],
        "operation": "evict",
    })
    .to_string();
    let response = send_management_request(
        addr,
        management_post_request("/api/runtime/control/kv-cache", &request_body),
    )
    .await;

    assert!(
        response.starts_with("HTTP/1.1 401") || response.starts_with("HTTP/1.1 403"),
        "kv-cache accepted a loopback caller that presented no owner capability: {response}"
    );

    handle.await.unwrap().unwrap();
}

#[tokio::test]
#[serial]
async fn owner_control_load_model_rejects_loopback_caller_without_owner_capability() {
    let state = build_test_mesh_api().await;
    let (addr, handle) = spawn_management_test_server(state).await;

    let request_body = json!({}).to_string();
    let response = send_management_request(
        addr,
        management_post_request("/api/runtime/control/load-model", &request_body),
    )
    .await;

    assert!(
        response.starts_with("HTTP/1.1 401") || response.starts_with("HTTP/1.1 403"),
        "load-model accepted a loopback caller that presented no owner capability: {response}"
    );

    handle.await.unwrap().unwrap();
}

#[tokio::test]
#[serial]
async fn owner_control_unload_model_rejects_loopback_caller_without_owner_capability() {
    let state = build_test_mesh_api().await;
    let (addr, handle) = spawn_management_test_server(state).await;

    let request_body = json!({}).to_string();
    let response = send_management_request(
        addr,
        management_post_request("/api/runtime/control/unload-model", &request_body),
    )
    .await;

    assert!(
        response.starts_with("HTTP/1.1 401") || response.starts_with("HTTP/1.1 403"),
        "unload-model accepted a loopback caller that presented no owner capability: {response}"
    );

    handle.await.unwrap().unwrap();
}

#[tokio::test]
#[serial]
async fn owner_control_ensure_model_rejects_loopback_caller_without_owner_capability() {
    let state = build_test_mesh_api().await;
    let (addr, handle) = spawn_management_test_server(state).await;

    let request_body = json!({}).to_string();
    let response = send_management_request(
        addr,
        management_post_request("/api/runtime/control/ensure-model", &request_body),
    )
    .await;

    assert!(
        response.starts_with("HTTP/1.1 401") || response.starts_with("HTTP/1.1 403"),
        "ensure-model accepted a loopback caller that presented no owner capability: {response}"
    );

    handle.await.unwrap().unwrap();
}

#[tokio::test]
#[serial]
async fn owner_control_drain_model_rejects_loopback_caller_without_owner_capability() {
    let state = build_test_mesh_api().await;
    let (addr, handle) = spawn_management_test_server(state).await;

    let request_body = json!({}).to_string();
    let response = send_management_request(
        addr,
        management_post_request("/api/runtime/control/drain-model", &request_body),
    )
    .await;

    assert!(
        response.starts_with("HTTP/1.1 401") || response.starts_with("HTTP/1.1 403"),
        "drain-model accepted a loopback caller that presented no owner capability: {response}"
    );

    handle.await.unwrap().unwrap();
}
