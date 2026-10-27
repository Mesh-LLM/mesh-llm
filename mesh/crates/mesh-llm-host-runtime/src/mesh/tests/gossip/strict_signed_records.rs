async fn strict_node() -> Node {
    let mut node = Node::new_for_tests(NodeRole::Worker).await.unwrap();
    node.local_mesh_requirements.require_signed_gossip = true;
    node
}

#[tokio::test]
async fn unsigned_relay_cannot_replace_verified_models_or_cache() {
    let node = strict_node().await;
    receive_frame(
        &node,
        0x62,
        InboundGossip {
            announcements: Vec::new(),
            signed_records: vec![signed_worker_record(0x61, 1, "signed-model")],
            signed_cache_affinity: vec![signed_cache_affinity(0x61, 1, "signed-model", 1)],
        },
    )
    .await;
    let mut forged = unsigned_worker(0x61, "forged-model");
    forged.1.cache_affinity = unsigned_cache_affinity("forged-model", 999);
    let resolved = receive(&node, 0x63, vec![forged], Vec::new()).await;
    assert!(!ids(&resolved).contains(&test_endpoint_id(0x61)));
    assert_eq!(
        serving_models(&node, 0x61).await,
        Some(vec!["signed-model".into()])
    );
    assert_eq!(cache_epoch(&node, 0x61).await, Some(1));
}

#[tokio::test]
async fn invalid_signature_cannot_fall_back_to_an_unsigned_twin() {
    let node = strict_node().await;
    let mut tampered = signed_worker_record(0x61, 1, "signed-model");
    tampered.signature[0] ^= 1;
    receive(
        &node,
        0x62,
        vec![unsigned_worker(0x61, "forged-model")],
        vec![tampered],
    )
    .await;
    assert_eq!(serving_models(&node, 0x61).await, None);
    assert!(
        !node
            .state
            .lock()
            .await
            .node_records
            .contains(&test_endpoint_id(0x61))
    );
}

#[tokio::test]
async fn unsigned_sender_is_kept_but_cannot_speak_for_a_third_party() {
    let node = strict_node().await;
    let resolved = receive(
        &node,
        0x62,
        vec![unsigned_worker(0x61, "forged-model")],
        Vec::new(),
    )
    .await;
    assert_eq!(ids(&resolved), vec![test_endpoint_id(0x62)]);
    assert_eq!(
        serving_models(&node, 0x62).await,
        Some(vec!["relay-model".into()])
    );
}

#[tokio::test]
async fn missing_relay_signature_does_not_hide_an_origin_signed_record() {
    let node = strict_node().await;
    receive(
        &node,
        0x62,
        Vec::new(),
        vec![signed_worker_record(0x61, 1, "signed-model")],
    )
    .await;
    assert_eq!(
        serving_models(&node, 0x61).await,
        Some(vec!["signed-model".into()])
    );
    assert!(
        !node
            .state
            .lock()
            .await
            .node_records
            .contains(&test_endpoint_id(0x62))
    );
}

#[tokio::test]
async fn compact_gossip_keeps_origin_bytes_and_local_merged_demand() {
    let node = strict_node().await;
    let demand = HashMap::from([(
        "remote-demand".to_string(),
        ModelDemand {
            last_active: now_secs(),
            request_count: 3,
        },
    )]);
    node.merge_remote_demand(&demand);
    let record = signed_worker_record(0x61, 1, "signed-model");
    receive(&node, 0x62, Vec::new(), vec![record.clone()]).await;
    let full = node.collect_announcements().await;
    let own = full.iter().find(|a| a.addr.id == node.id()).unwrap();
    let compact = node.collect_outbound_gossip().await;
    assert_eq!(compact.announcements.len(), 1);
    assert_eq!(compact.announcements[0].addr.id, node.id());
    let propagated = &compact.announcements[0].model_demand["remote-demand"];
    assert_eq!(
        propagated.last_active,
        own.model_demand["remote-demand"].last_active
    );
    assert_eq!(propagated.last_active, demand["remote-demand"].last_active);
    assert_eq!(propagated.request_count, 3);
    assert!(
        compact
            .signed_records
            .iter()
            .any(|r| r.signed == record.signed && r.signature == record.signature)
    );
}

#[tokio::test]
async fn unsigned_cache_cannot_be_attached_to_a_signed_node() {
    let node = strict_node().await;
    let mut unsigned = unsigned_worker(0x61, "signed-model");
    unsigned.1.cache_affinity = unsigned_cache_affinity("signed-model", 99);
    receive(
        &node,
        0x62,
        vec![unsigned],
        vec![signed_worker_record(0x61, 1, "signed-model")],
    )
    .await;
    assert_eq!(cache_epoch(&node, 0x61).await, None);
}

#[tokio::test]
async fn verification_completes_while_membership_is_locked() {
    let node = strict_node().await;
    let _membership_guard = node.state.lock().await;
    let mut inbound = InboundGossip {
        announcements: Vec::new(),
        signed_records: vec![signed_worker_record(0x61, 1, "signed-model")],
        signed_cache_affinity: Vec::new(),
    };
    let prepared = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        node.prepare_gossip_records(test_endpoint_id(0x62), &mut inbound),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(prepared.nodes.len(), 1);
}

#[tokio::test]
async fn deferred_verification_does_not_erase_previously_verified_sender_records() {
    let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();
    let record = signed_worker_record(0x62, 1, "relay-model");
    receive_frame(
        &node,
        0x62,
        InboundGossip {
            announcements: Vec::new(),
            signed_records: vec![record.clone()],
            signed_cache_affinity: vec![signed_cache_affinity(0x62, 1, "relay-model", 1)],
        },
    )
    .await;
    receive(&node, 0x62, Vec::new(), vec![record; 4097]).await;
    let state = node.state.lock().await;
    assert!(state.node_records.contains(&test_endpoint_id(0x62)));
    assert!(
        state
            .cache_affinity_records
            .contains(&test_endpoint_id(0x62))
    );
}

#[tokio::test]
async fn exhausted_verification_budget_does_not_enable_unsigned_relay_fallback() {
    let node = strict_node().await;
    let resolved = receive(
        &node,
        0x62,
        vec![unsigned_worker(0x61, "forged-model")],
        vec![signed_worker_record(0x61, 1, "signed-model"); 4097],
    )
    .await;
    assert_eq!(ids(&resolved), vec![test_endpoint_id(0x62)]);
    assert_eq!(serving_models(&node, 0x61).await, None);
    assert!(node.state.lock().await.node_records.is_empty());
}

#[tokio::test]
async fn expired_relay_is_rejected_but_authenticated_restart_can_reset_sequence() {
    let node = strict_node().await;
    receive(
        &node,
        0x62,
        Vec::new(),
        vec![signed_worker_record(0x61, 100, "current")],
    )
    .await;
    let prototype = signed_worker_record(0x61, 1, "restarted");
    let verified =
        VerifiedRecord::verify(NODE_RECORD, &prototype.signed, &prototype.signature).unwrap();
    let expired = VerifiedRecord::sign(
        NODE_RECORD,
        &signing_key(0x61),
        101,
        current_time_unix_ms() - NODE_RECORD.max_age_ms - 1,
        verified.body(),
    );
    let wire = |record: &VerifiedRecord| crate::proto::node::SignedNodeRecord {
        signed: record.signed_bytes().to_vec(),
        signature: record.signature().to_vec(),
        hop: None,
    };
    receive(
        &node,
        0x62,
        vec![unsigned_worker(0x61, "forged")],
        vec![wire(&expired)],
    )
    .await;
    assert_eq!(
        serving_models(&node, 0x61).await,
        Some(vec!["current".into()])
    );
    let restarted = VerifiedRecord::sign(
        NODE_RECORD,
        &signing_key(0x61),
        1,
        expired.header().issued_at_unix_ms,
        verified.body(),
    );
    receive(&node, 0x61, Vec::new(), vec![wire(&restarted)]).await;
    assert_eq!(
        node.state
            .lock()
            .await
            .node_records
            .get(&test_endpoint_id(0x61))
            .unwrap()
            .header()
            .seq,
        1
    );
    let outbound = node.collect_outbound_gossip().await;
    assert!(
        !outbound
            .signed_records
            .iter()
            .any(|r| r.signed == restarted.signed_bytes())
    );
    receive(
        &node,
        0x62,
        Vec::new(),
        vec![signed_worker_record(0x61, 2, "restarted")],
    )
    .await;
    assert_eq!(
        serving_models(&node, 0x61).await,
        Some(vec!["restarted".into()])
    );
}

#[cfg(feature = "payments")]
#[tokio::test]
async fn unsigned_relay_cannot_change_the_price_seen_by_paid_routing() {
    let node = strict_node().await;
    let mut ann = unsigned_worker(0x61, "paid-model");
    let price = mesh_llm_payments_types::pricing::Pricing {
        input_msat_per_million: 500,
        output_msat_per_million: 1500,
        cache_discount_percent: 0,
    };
    ann.1
        .lightning_offers
        .insert("paid-model".into(), price.clone());
    let body = crate::protocol::node_record::node_record_from_wire_ann(
        &crate::protocol::local_ann_to_proto_ann(&ann.1),
        &ann.0,
    )
    .encode_to_vec();
    let record = VerifiedRecord::sign(
        NODE_RECORD,
        &signing_key(0x61),
        1,
        current_time_unix_ms(),
        &body,
    );
    receive(
        &node,
        0x62,
        Vec::new(),
        vec![crate::proto::node::SignedNodeRecord {
            signed: record.signed_bytes().to_vec(),
            signature: record.signature().to_vec(),
            hop: None,
        }],
    )
    .await;
    for forged_price in [1, 999_999] {
        ann.1
            .lightning_offers
            .get_mut("paid-model")
            .unwrap()
            .input_msat_per_million = forged_price;
        receive(&node, 0x62, vec![ann.clone()], Vec::new()).await;
        assert_eq!(
            node.peer_payment_offer(test_endpoint_id(0x61), "paid-model")
                .await,
            Some(price.clone())
        );
    }
}
