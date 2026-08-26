#[test]
pub(crate) fn test_merge_none_to_some() {
    let mut existing = test_peer(None);
    let ann = test_announcement(Some(100));

    apply_transitive_ann(
        &mut existing,
        &test_addr(0x33),
        &ann,
        test_endpoint_id(0xee),
    );

    assert_eq!(existing.first_joined_mesh_ts, Some(100));
}

#[test]
pub(crate) fn test_merge_some_to_none_keeps_existing() {
    let mut existing = test_peer(Some(100));
    let ann = test_announcement(None);

    apply_transitive_ann(
        &mut existing,
        &test_addr(0x33),
        &ann,
        test_endpoint_id(0xee),
    );

    assert_eq!(existing.first_joined_mesh_ts, Some(100));
}

#[test]
pub(crate) fn test_merge_earlier_incoming_wins() {
    let mut existing = test_peer(Some(200));
    let ann = test_announcement(Some(100));

    apply_transitive_ann(
        &mut existing,
        &test_addr(0x33),
        &ann,
        test_endpoint_id(0xee),
    );

    assert_eq!(existing.first_joined_mesh_ts, Some(100));
}

#[test]
pub(crate) fn test_merge_later_incoming_loses() {
    let mut existing = test_peer(Some(100));
    let ann = test_announcement(Some(200));

    apply_transitive_ann(
        &mut existing,
        &test_addr(0x33),
        &ann,
        test_endpoint_id(0xee),
    );

    assert_eq!(existing.first_joined_mesh_ts, Some(100));
}

#[test]
pub(crate) fn test_merge_equal_values_unchanged() {
    let mut existing = test_peer(Some(100));
    let ann = test_announcement(Some(100));

    apply_transitive_ann(
        &mut existing,
        &test_addr(0x33),
        &ann,
        test_endpoint_id(0xee),
    );

    assert_eq!(existing.first_joined_mesh_ts, Some(100));
}

#[test]
pub(crate) fn test_meaningfully_changed_first_joined_mesh_ts() {
    let old_peer = test_peer(Some(100));
    let new_peer = test_peer(Some(200));

    assert!(peer_meaningfully_changed(&old_peer, &new_peer));
}

#[test]
pub(crate) fn test_meaningfully_changed_explicit_model_interests() {
    let old_peer = test_peer(Some(100));
    let mut new_peer = test_peer(Some(100));
    new_peer.explicit_model_interests = vec!["Qwen/Qwen3-Coder-Next-GGUF@main:Q4_K_M".into()];

    assert!(peer_meaningfully_changed(&old_peer, &new_peer));
}

#[test]
pub(crate) fn test_meaningfully_changed_stage_status_list_support() {
    let old_peer = test_peer(Some(100));
    let mut new_peer = test_peer(Some(100));
    new_peer.stage_status_list_supported = !old_peer.stage_status_list_supported;

    assert!(peer_meaningfully_changed(&old_peer, &new_peer));
}

#[test]
pub(crate) fn test_meaningfully_changed_stage_protocol_generation_support() {
    let old_peer = test_peer(Some(100));
    let mut new_peer = test_peer(Some(100));
    new_peer.stage_protocol_generation_supported = !old_peer.stage_protocol_generation_supported;

    assert!(peer_meaningfully_changed(&old_peer, &new_peer));
}

#[test]
pub(crate) fn test_meaningfully_changed_ignores_cache_refresh_timestamp() {
    let mut old_peer = test_peer(Some(100));
    old_peer.cache_affinity = Some(
        mesh_llm_routing::cache_inventory::CacheAffinityAdvertisement {
            salt: [3; mesh_llm_routing::cache_inventory::CACHE_AFFINITY_SALT_BYTES],
            epoch: 7,
            generated_at_unix_ms: 1_000,
            ttl_ms: 120_000,
            entries: Vec::new(),
        },
    );
    let mut new_peer = old_peer.clone();
    new_peer
        .cache_affinity
        .as_mut()
        .expect("cache advertisement")
        .generated_at_unix_ms = 2_000;

    assert!(!peer_meaningfully_changed(&old_peer, &new_peer));
    new_peer
        .cache_affinity
        .as_mut()
        .expect("cache advertisement")
        .epoch = 8;
    assert!(peer_meaningfully_changed(&old_peer, &new_peer));
}

#[test]
pub(crate) fn test_transitive_merge_rejects_replayed_cache_advertisement() {
    let mut existing = test_peer(Some(100));
    existing.cache_affinity = Some(cache_advertisement(2_000, 8));
    let mut ann = test_announcement(Some(100));
    ann.cache_affinity = Some(cache_advertisement(1_000, 99));

    apply_transitive_ann(
        &mut existing,
        &test_addr(0x33),
        &ann,
        test_endpoint_id(0xee),
    );

    let retained = existing.cache_affinity.expect("cache advertisement");
    assert_eq!((retained.generated_at_unix_ms, retained.epoch), (2_000, 8));
}

#[tokio::test]
pub(crate) async fn test_direct_merge_accepts_only_newer_cache_advertisements() {
    let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();
    let peer_id = test_endpoint_id(0x47);
    let addr = test_addr(0x47);
    let now = crate::mesh::current_time_unix_ms();
    let mut ann = test_announcement(Some(100));
    ann.cache_affinity = Some(cache_advertisement(now, 7));
    node.add_peer(peer_id, addr.clone(), &ann, None).await;

    ann.cache_affinity = Some(cache_advertisement(now.saturating_sub(1), 99));
    node.add_peer(peer_id, addr.clone(), &ann, None).await;
    {
        let state = node.state.lock().await;
        let retained = state
            .peers
            .get(&peer_id)
            .and_then(|peer| peer.cache_affinity.as_ref())
            .expect("cache advertisement");
        assert_eq!((retained.generated_at_unix_ms, retained.epoch), (now, 7));
    }

    ann.cache_affinity = Some(cache_advertisement(now.saturating_add(1), 8));
    node.add_peer(peer_id, addr, &ann, None).await;
    let state = node.state.lock().await;
    let updated = state
        .peers
        .get(&peer_id)
        .and_then(|peer| peer.cache_affinity.as_ref())
        .expect("cache advertisement");
    assert_eq!((updated.generated_at_unix_ms, updated.epoch), (now + 1, 8));
}

#[test]
pub(crate) fn test_apply_transitive_ann_refreshes_explicit_model_interests() {
    let mut existing = test_peer(Some(100));
    let mut ann = test_announcement(Some(100));
    ann.explicit_model_interests = vec!["Qwen/Qwen3-Coder-Next-GGUF@main:Q4_K_M".into()];

    apply_transitive_ann(
        &mut existing,
        &test_addr(0x33),
        &ann,
        test_endpoint_id(0xee),
    );

    assert_eq!(
        existing.explicit_model_interests,
        vec!["Qwen/Qwen3-Coder-Next-GGUF@main:Q4_K_M".to_string()]
    );
}

#[test]
pub(crate) fn test_apply_transitive_ann_refreshes_stage_status_list_support() {
    let mut existing = test_peer(Some(100));
    existing.stage_status_list_supported = false;
    let mut ann = test_announcement(Some(100));
    ann.stage_status_list_supported = true;

    apply_transitive_ann(
        &mut existing,
        &test_addr(0x33),
        &ann,
        test_endpoint_id(0xee),
    );

    assert!(existing.stage_status_list_supported);
}

#[test]
pub(crate) fn test_apply_transitive_ann_refreshes_stage_protocol_generation_support() {
    let mut existing = test_peer(Some(100));
    existing.stage_protocol_generation_supported = false;
    let mut ann = test_announcement(Some(100));
    ann.stage_protocol_generation_supported = true;

    apply_transitive_ann(
        &mut existing,
        &test_addr(0x33),
        &ann,
        test_endpoint_id(0xee),
    );

    assert!(existing.stage_protocol_generation_supported);
}

#[test]
pub(crate) fn test_apply_transitive_ann_refreshes_advertised_model_throughput() {
    let mut existing = test_peer(Some(100));
    let mut ann = test_announcement(Some(100));
    ann.advertised_model_throughput = vec![crate::network::metrics::ModelThroughputHint {
        model_name: "qwen".to_string(),
        avg_tokens_per_second_milli: 35_000,
        throughput_samples: 4,
    }];

    apply_transitive_ann(
        &mut existing,
        &test_addr(0x33),
        &ann,
        test_endpoint_id(0xee),
    );

    assert_eq!(
        existing.advertised_model_throughput,
        ann.advertised_model_throughput
    );
}

#[tokio::test]
pub(crate) async fn test_add_peer_refreshes_stage_status_list_support() {
    let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();
    let peer_id = test_endpoint_id(0x44);
    let addr = test_addr(0x44);
    let mut ann = test_announcement(Some(100));
    ann.stage_status_list_supported = false;

    node.add_peer(peer_id, addr.clone(), &ann, None).await;
    ann.stage_status_list_supported = true;
    node.add_peer(peer_id, addr, &ann, None).await;

    let state = node.state.lock().await;
    let peer = state.peers.get(&peer_id).expect("peer should be tracked");
    assert!(peer.stage_status_list_supported);
}

#[tokio::test]
pub(crate) async fn test_add_peer_refreshes_stage_protocol_generation_support() {
    let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();
    let peer_id = test_endpoint_id(0x45);
    let addr = test_addr(0x45);
    let mut ann = test_announcement(Some(100));
    ann.stage_protocol_generation_supported = false;

    node.add_peer(peer_id, addr.clone(), &ann, None).await;
    ann.stage_protocol_generation_supported = true;
    node.add_peer(peer_id, addr, &ann, None).await;

    let state = node.state.lock().await;
    let peer = state.peers.get(&peer_id).expect("peer should be tracked");
    assert!(peer.stage_protocol_generation_supported);
}

#[tokio::test]
pub(crate) async fn test_add_peer_refreshes_advertised_model_throughput() {
    let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();
    let peer_id = test_endpoint_id(0x46);
    let addr = test_addr(0x46);
    let mut ann = test_announcement(Some(100));
    ann.advertised_model_throughput = vec![crate::network::metrics::ModelThroughputHint {
        model_name: "qwen".to_string(),
        avg_tokens_per_second_milli: 20_000,
        throughput_samples: 2,
    }];

    node.add_peer(peer_id, addr.clone(), &ann, None).await;
    ann.advertised_model_throughput[0].avg_tokens_per_second_milli = 48_000;
    ann.advertised_model_throughput[0].throughput_samples = 9;
    node.add_peer(peer_id, addr, &ann, None).await;

    let state = node.state.lock().await;
    let peer = state.peers.get(&peer_id).expect("peer should be tracked");
    assert_eq!(
        peer.advertised_model_throughput,
        ann.advertised_model_throughput
    );
}

#[tokio::test]
pub(crate) async fn test_collect_announcements_includes_self_explicit_model_interests() {
    let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();
    node.set_explicit_model_interests(vec![
        "Qwen/Qwen3-Coder-Next-GGUF@main:Q4_K_M".into(),
        "Qwen/Qwen3-Coder-Next-GGUF@main:Q4_K_M".into(),
    ])
    .await;

    let announcements = node.collect_announcements().await;
    let self_announcement = announcements
        .iter()
        .find(|announcement| announcement.addr.id == node.id())
        .expect("self announcement must be present");

    assert_eq!(
        self_announcement.explicit_model_interests,
        vec!["Qwen/Qwen3-Coder-Next-GGUF@main:Q4_K_M".to_string()]
    );
}
fn cache_advertisement(
    generated_at_unix_ms: u64,
    epoch: u64,
) -> mesh_llm_routing::cache_inventory::CacheAffinityAdvertisement {
    mesh_llm_routing::cache_inventory::CacheAffinityAdvertisement {
        salt: [3; mesh_llm_routing::cache_inventory::CACHE_AFFINITY_SALT_BYTES],
        epoch,
        generated_at_unix_ms,
        ttl_ms: 120_000,
        entries: Vec::new(),
    }
}
