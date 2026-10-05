#[tokio::test]
async fn wakeable_inventory_does_not_change_peer_count() {
    let state = build_test_mesh_api().await;
    replace_test_wakeable_inventory(
        &state,
        vec![make_test_wakeable_entry(
            "sleeping-node-1",
            "wakeable-only-model",
            48.0,
        )],
    )
    .await;
    let status = state.status().await;
    assert!(status.peers.is_empty());
    assert_eq!(status.wakeable_nodes.len(), 1);
    assert_eq!(status.wakeable_nodes[0].logical_id, "sleeping-node-1");
}

#[tokio::test]
async fn wakeable_inventory_does_not_change_mesh_vram_totals() {
    let state = build_test_mesh_api().await;
    let status_before = state.status().await;
    let (addr_before, handle_before) = spawn_management_test_server(state.clone()).await;
    let response_before = send_management_request(
        addr_before,
        "GET /api/status HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
    )
    .await;
    handle_before.await.unwrap().unwrap();
    let payload_before = json_body(&response_before);

    replace_test_wakeable_inventory(
        &state,
        vec![make_test_wakeable_entry(
            "sleeping-node-1",
            "wakeable-only-model",
            48.0,
        )],
    )
    .await;

    let status_after = state.status().await;
    let (addr_after, handle_after) = spawn_management_test_server(state.clone()).await;
    let response_after = send_management_request(
        addr_after,
        "GET /api/status HTTP/1.1\r\nHost: localhost\r\n\r\n".into(),
    )
    .await;
    handle_after.await.unwrap().unwrap();
    let payload_after = json_body(&response_after);

    assert_eq!(status_after.peers, status_before.peers);
    assert_eq!(status_after.my_vram_gb, status_before.my_vram_gb);
    assert_eq!(status_after.wakeable_nodes.len(), 1);
    assert!(response_before.starts_with("HTTP/1.1 200"));
    assert!(response_after.starts_with("HTTP/1.1 200"));
    assert_eq!(payload_after["peers"], payload_before["peers"]);
    assert_eq!(
        payload_after["my_vram_gb"],
        payload_before["my_vram_gb"]
    );
    assert_eq!(
        payload_after["wakeable_nodes"][0]["logical_id"],
        "sleeping-node-1"
    );

}

#[tokio::test]
async fn wakeable_inventory_is_not_routable_capacity() {
    let state = build_test_mesh_api().await;
    replace_test_wakeable_inventory(
        &state,
        vec![make_test_wakeable_entry(
            "sleeping-node-1",
            "wakeable-only-model",
            48.0,
        )],
    )
    .await;

    let node = { state.inner.lock().await.node.clone() };
    let status = state.status().await;
    let served_models = node.models_being_served().await;
    let hosts = node.hosts_for_model("wakeable-only-model").await;

    assert_eq!(status.wakeable_nodes.len(), 1);
    assert!(
        !served_models
            .iter()
            .any(|model| model == "wakeable-only-model")
    );
    assert!(hosts.is_empty());
}

#[tokio::test]
async fn wakeable_inventory_is_excluded_from_v1_models() {
    let state = build_test_mesh_api().await;
    replace_test_wakeable_inventory(
        &state,
        vec![make_test_wakeable_entry(
            "sleeping-node-1",
            "wakeable-only-model",
            48.0,
        )],
    )
    .await;

    let node = { state.inner.lock().await.node.clone() };
    let served_models = node.models_being_served().await;

    assert!(
        !served_models
            .iter()
            .any(|model| model == "wakeable-only-model")
    );
    assert!(served_models.is_empty());
}

#[tokio::test]
async fn wakeable_inventory_is_excluded_from_host_selection() {
    let state = build_test_mesh_api().await;
    replace_test_wakeable_inventory(
        &state,
        vec![make_test_wakeable_entry(
            "sleeping-node-1",
            "wakeable-only-model",
            48.0,
        )],
    )
    .await;

    let node = { state.inner.lock().await.node.clone() };
    let hosts = node.hosts_for_model("wakeable-only-model").await;

    assert!(hosts.is_empty());
}

#[test]
fn build_wakeable_node_preserves_typed_internal_state() {
    let sleeping = MeshApi::build_wakeable_node(WakeableInventoryEntry {
        logical_id: "sleeping-node".to_string(),
        models: vec!["test-model".to_string()],
        vram_gb: 24.0,
        provider: Some("test-provider".to_string()),
        state: WakeableState::Sleeping,
        wake_eta_secs: Some(45),
    });
    let waking = MeshApi::build_wakeable_node(WakeableInventoryEntry {
        logical_id: "waking-node".to_string(),
        models: vec!["test-model".to_string()],
        vram_gb: 24.0,
        provider: Some("test-provider".to_string()),
        state: WakeableState::Waking,
        wake_eta_secs: Some(10),
    });

    assert_eq!(sleeping.state, WakeableNodeState::Sleeping);
    assert_eq!(waking.state, WakeableNodeState::Waking);
}
