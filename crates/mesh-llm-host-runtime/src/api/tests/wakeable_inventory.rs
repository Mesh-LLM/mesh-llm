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
    let peers = vec![make_test_peer(
        0x51,
        mesh::NodeRole::Host { http_port: 9337 },
        vec!["wakeable-only-model"],
        vec!["wakeable-only-model"],
        true,
    )];
    let route_stats = http_route_stats("wakeable-only-model", &peers, &[], None, 0.0);

    assert_eq!(status.wakeable_nodes.len(), 1);
    assert_eq!(route_stats.node_count, 1);
    assert!(route_stats.mesh_vram_gb > 0.0);
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
