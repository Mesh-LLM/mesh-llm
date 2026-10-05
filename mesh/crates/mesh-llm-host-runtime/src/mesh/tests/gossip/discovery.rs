#[tokio::test]
pub(crate) async fn future_demand_timestamps_are_active_without_underflow() {
    let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();
    let now = now_secs();
    node.merge_remote_demand(&HashMap::from([(
        "future-demand".to_string(),
        ModelDemand {
            last_active: now + DEMAND_TTL_SECS + 60,
            request_count: 3,
        },
    )]));

    let active = node.active_demand().await;
    assert!(active.contains_key("future-demand"));

    node.gc_demand().await;
    assert!(node.get_demand().contains_key("future-demand"));
}

/// Regression test for the `--auto` startup wedge: when a transitive
/// gossip payload includes peers that would be rejected at ingest
/// (version-floor or idle-transitive-client), `maybe_connect_discovered_peer`
/// must skip the dial. Otherwise each unreachable ghost address triggers
/// a 30 s `connect_to_peer` timeout sequentially in the dial loop,
/// wedging the surrounding gossip exchange (and the `attempt_run_auto_join`
/// that initiated it) for tens of minutes.
///
/// The function returns without panicking and without dialing within a
/// generous time bound — a real dial to a fake address would block on
/// the 30 s `PEER_CONNECT_AND_GOSSIP_TIMEOUT`. We assert the result is
/// reached well under that bound and that no connection entry was created.
#[tokio::test]
pub(crate) async fn maybe_connect_discovered_peer_skips_filtered_announcements() {
    let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();
    let my_role = NodeRole::Worker;

    // Below-floor version — must be skipped without dialing.
    let old_addr = test_addr(0x57);
    let old_id = old_addr.id;
    let mut old_ann = test_announcement(None);
    old_ann.addr = old_addr.clone();
    old_ann.role = NodeRole::Client;
    old_ann.version = Some("0.57.0".to_string());

    // Idle transitive client (matching version, but no hostname / no
    // direct measurement / no model interests) — must also be skipped.
    let idle_addr = test_addr(0xC1);
    let idle_id = idle_addr.id;
    let mut idle_ann = test_announcement(None);
    idle_ann.addr = idle_addr.clone();
    idle_ann.role = NodeRole::Client;
    idle_ann.version = Some("0.65.1".to_string());

    // Both calls together must return well under the 30 s connect
    // timeout. If the dial-loop skip is missing, each call will block
    // on PEER_CONNECT_AND_GOSSIP_TIMEOUT (30 s) attempting to dial the
    // fake test address.
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        node.maybe_connect_discovered_peer(&my_role, old_addr, &old_ann, true, false)
            .await;
        node.maybe_connect_discovered_peer(&my_role, idle_addr, &idle_ann, true, false)
            .await;
    })
    .await
    .expect("filtered peers must be skipped quickly, not dialed");

    // No connection was attempted (no entry in state.connections), and
    // no peer was added (the filtered announcements never reach add_peer
    // or update_transitive_peer through this path).
    let state = node.state.lock().await;
    assert!(
        !state.connections.contains_key(&old_id),
        "below-floor peer must not be dialed"
    );
    assert!(
        !state.connections.contains_key(&idle_id),
        "idle transitive client must not be dialed"
    );
    assert!(
        !state.peers.contains_key(&old_id),
        "below-floor peer must not be added (this path is dial-only)"
    );
    assert!(
        !state.peers.contains_key(&idle_id),
        "idle transitive client must not be added (this path is dial-only)"
    );
}

#[tokio::test]
pub(crate) async fn client_auto_join_probe_returns_none_for_single_candidate() {
    let node = Node::new_for_tests(NodeRole::Client).await.unwrap();
    let token = encode_endpoint_addr_token(&test_addr(0x42));

    let selected = node
        .join_first_responsive_candidate(&[(token, Some("single".to_string()))])
        .await
        .unwrap();

    assert!(selected.is_none());
}

#[tokio::test]
pub(crate) async fn client_auto_join_probe_candidate_collection_filters_unusable_tokens() {
    let node = Node::new_for_tests(NodeRole::Client).await.unwrap();
    let valid_addr = test_addr(0x42);
    let dead_addr = test_addr(0x43);
    let self_token = encode_endpoint_addr_token(&node.endpoint_addr_for_advertisement());
    let dead_token = encode_endpoint_addr_token(&dead_addr);
    let valid_token = encode_endpoint_addr_token(&valid_addr);

    node.state
        .lock()
        .await
        .dead_peers
        .insert(dead_addr.id, std::time::Instant::now());

    let candidates = node
        .collect_join_probe_candidates(&[
            ("not-an-invite-token".to_string(), None),
            (self_token, None),
            (dead_token, None),
            (valid_token, Some("usable".to_string())),
        ])
        .await;

    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].addr.id, valid_addr.id);
    assert_eq!(candidates[0].mesh_name.as_deref(), Some("usable"));
}
