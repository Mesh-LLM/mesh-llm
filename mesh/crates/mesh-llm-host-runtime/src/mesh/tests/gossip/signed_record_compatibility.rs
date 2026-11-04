/// The pre-signature protobuf surface: decoding discards fields 4 and 5.
#[derive(Clone, PartialEq, prost::Message)]
struct LegacyGossip {
    #[prost(uint32, tag = "1")]
    generation: u32,
    #[prost(message, repeated, tag = "2")]
    peers: Vec<crate::proto::node::PeerAnnouncement>,
    #[prost(bytes = "vec", tag = "3")]
    sender: Vec<u8>,
}

#[tokio::test]
async fn invalid_self_signature_is_not_mistaken_for_a_legacy_downgrade() {
    let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();
    let mut record = signed_worker_record(0x62, 1, "relay-model");
    let mut cache = signed_cache_affinity(0x62, 1, "relay-model", 1);
    receive_frame(
        &node,
        0x62,
        InboundGossip {
            announcements: Vec::new(),
            signed_records: vec![record.clone()],
            signed_cache_affinity: vec![cache.clone()],
        },
    )
    .await;
    record.signature[0] ^= 1;
    cache.signature[0] ^= 1;
    receive_frame(
        &node,
        0x62,
        InboundGossip {
            announcements: Vec::new(),
            signed_records: vec![record],
            signed_cache_affinity: vec![cache],
        },
    )
    .await;
    let state = node.state.lock().await;
    assert!(state.node_records.contains(&test_endpoint_id(0x62)));
    assert!(
        state
            .cache_affinity_records
            .contains(&test_endpoint_id(0x62))
    );
}

#[tokio::test]
async fn new_old_new_path_preserves_default_mesh_discovery_and_cache_affinity() {
    let origin = Node::new_for_tests(NodeRole::Worker).await.unwrap();
    origin.set_serving_models(vec!["origin-model".into()]).await;
    let outbound = origin.collect_outbound_gossip().await;
    let mut frame = crate::protocol::build_gossip_frame(&outbound.announcements, origin.id());
    frame.signed_records = outbound.signed_records;
    frame.signed_cache_affinity = outbound.signed_cache_affinity;
    let mut legacy = LegacyGossip::decode(frame.encode_to_vec().as_slice()).unwrap();
    let old_peer = unsigned_worker(0x62, "old-model");
    legacy.sender = test_endpoint_id(0x62).as_bytes().to_vec();
    legacy
        .peers
        .push(crate::protocol::local_ann_to_proto_ann(&old_peer.1));
    let forwarded = crate::protocol::decode_gossip_frame(
        crate::protocol::ControlProtocol::ProtoV1,
        test_endpoint_id(0x62),
        &legacy.encode_to_vec(),
    )
    .unwrap();
    assert!(forwarded.signed_records.is_empty());
    assert!(forwarded.signed_cache_affinity.is_empty());
    let receiver = Node::new_for_tests(NodeRole::Worker).await.unwrap();
    let resolved = receiver
        .resolve_inbound_gossip(test_endpoint_id(0x62), forwarded)
        .await;
    let (_, announced_origin) = resolved
        .iter()
        .find(|(addr, _)| addr.id == origin.id())
        .unwrap();
    assert_eq!(announced_origin.serving_models, vec!["origin-model"]);
    assert!(announced_origin.cache_affinity.is_some());
    assert!(
        resolved
            .iter()
            .any(|(addr, _)| addr.id == test_endpoint_id(0x62))
    );
}

#[tokio::test]
async fn strict_compaction_removes_the_duplicate_description_bytes() {
    let mut node = Node::new_for_tests(NodeRole::Worker).await.unwrap();
    // Keep the name inside the remote-string bound; oversized names are
    // correctly removed at admission and cannot demonstrate compaction.
    let name = "large-model-name".repeat(30);
    receive(
        &node,
        0x62,
        Vec::new(),
        vec![signed_worker_record(0x61, 1, &name)],
    )
    .await;
    assert_eq!(serving_models(&node, 0x61).await, Some(vec![name.clone()]));
    let compatibility = node.collect_outbound_gossip().await;
    let compatibility_unsigned =
        crate::protocol::build_gossip_frame(&compatibility.announcements, node.id()).encoded_len();
    node.local_mesh_requirements.require_signed_gossip = true;
    let strict = node.collect_outbound_gossip().await;
    let strict_unsigned =
        crate::protocol::build_gossip_frame(&strict.announcements, node.id()).encoded_len();
    assert!(compatibility_unsigned - strict_unsigned > name.len());
    let origin_record = &compatibility
        .signed_records
        .iter()
        .find(|r| {
            VerifiedRecord::verify(NODE_RECORD, &r.signed, &r.signature)
                .unwrap()
                .endpoint_id()
                == test_endpoint_id(0x61)
        })
        .unwrap()
        .signed;
    assert!(
        strict
            .signed_records
            .iter()
            .any(|r| &r.signed == origin_record)
    );
}
