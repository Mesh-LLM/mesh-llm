use crate::protocol::InboundGossip;
use mesh_llm_membership::signed_record::{CACHE_AFFINITY_RECORD, NODE_RECORD, VerifiedRecord};
use prost::Message;

fn signing_key(seed: u8) -> ed25519_dalek::SigningKey {
    ed25519_dalek::SigningKey::from_bytes(&[seed; 32])
}

/// A worker's signed record, with a field this build does not know appended
/// to show relays keep bytes they cannot interpret.
fn signed_worker_record(seed: u8, seq: u64, model: &str) -> crate::proto::node::SignedNodeRecord {
    let mut body = crate::proto::node::NodeRecord {
        role: crate::proto::node::NodeRole::Worker as i32,
        version: crate::VERSION.to_string(),
        serving_models: vec![model.to_string()],
        hosted_models: vec![model.to_string()],
        ..Default::default()
    }
    .encode_to_vec();
    body.extend_from_slice(&[0x9a, 0x06, 0x03, b'n', b'e', b'w']);
    let record = VerifiedRecord::sign(
        NODE_RECORD,
        &signing_key(seed),
        seq,
        current_time_unix_ms(),
        &body,
    );
    crate::proto::node::SignedNodeRecord {
        signed: record.signed_bytes().to_vec(),
        signature: record.signature().to_vec(),
        hop: None,
    }
}

/// Cache-affinity evidence for `model`, tagged with `epoch` so tests can
/// tell which copy was applied.
fn cache_advertisement(model: &str, epoch: u64) -> crate::proto::node::CacheAffinityAdvertisement {
    crate::proto::node::CacheAffinityAdvertisement {
        salt: vec![3; 32],
        epoch,
        generated_at_unix_ms: current_time_unix_ms(),
        ttl_ms: 120_000,
        entries: vec![crate::proto::node::CacheAffinityEntry {
            model_name: model.to_string(),
            prefix_digest: vec![7; 16],
            matched_tokens: 64,
            suffix_prefill_tokens: 8,
            tier: crate::proto::node::CacheTier::L1 as i32,
            ..Default::default()
        }],
    }
}

fn signed_cache_affinity(
    seed: u8,
    seq: u64,
    model: &str,
    epoch: u64,
) -> crate::proto::node::SignedCacheAffinity {
    let record = VerifiedRecord::sign(
        CACHE_AFFINITY_RECORD,
        &signing_key(seed),
        seq,
        current_time_unix_ms(),
        &cache_advertisement(model, epoch).encode_to_vec(),
    );
    crate::proto::node::SignedCacheAffinity {
        signed: record.signed_bytes().to_vec(),
        signature: record.signature().to_vec(),
    }
}

fn unsigned_worker(seed: u8, model: &str) -> (EndpointAddr, PeerAnnouncement) {
    let mut ann = test_announcement(None);
    ann.addr = test_addr(seed);
    ann.version = Some(crate::VERSION.to_string());
    ann.serving_models = vec![model.to_string()];
    ann.hosted_models = Some(vec![model.to_string()]);
    (ann.addr.clone(), ann)
}

fn unsigned_cache_affinity(
    model: &str,
    epoch: u64,
) -> Option<mesh_llm_routing::cache_inventory::CacheAffinityAdvertisement> {
    crate::protocol::proto_cache_affinity_to_local(&cache_advertisement(model, epoch))
}

async fn receive_frame(
    node: &Node,
    relay_seed: u8,
    mut frame: InboundGossip,
) -> Vec<(EndpointAddr, PeerAnnouncement)> {
    let relay = test_endpoint_id(relay_seed);
    frame
        .announcements
        .push(unsigned_worker(relay_seed, "relay-model"));
    let resolved = node.resolve_inbound_gossip(relay, frame).await;
    node.apply_announced_peers(relay, &resolved, None, None, true)
        .await
        .expect("gossip applies");
    resolved
}

async fn receive(
    node: &Node,
    relay_seed: u8,
    announcements: Vec<(EndpointAddr, PeerAnnouncement)>,
    signed_records: Vec<crate::proto::node::SignedNodeRecord>,
) -> Vec<(EndpointAddr, PeerAnnouncement)> {
    receive_frame(
        node,
        relay_seed,
        InboundGossip {
            announcements,
            signed_records,
            signed_cache_affinity: Vec::new(),
        },
    )
    .await
}

fn ids(resolved: &[(EndpointAddr, PeerAnnouncement)]) -> Vec<EndpointId> {
    resolved.iter().map(|(addr, _)| addr.id).collect()
}

async fn serving_models(node: &Node, seed: u8) -> Option<Vec<String>> {
    node.state
        .lock()
        .await
        .peers
        .get(&test_endpoint_id(seed))
        .map(|peer| peer.serving_models.clone())
}

async fn cache_epoch(node: &Node, seed: u8) -> Option<u64> {
    node.state
        .lock()
        .await
        .peers
        .get(&test_endpoint_id(seed))
        .and_then(|peer| peer.cache_affinity.as_ref())
        .map(|advertisement| advertisement.epoch)
}

#[tokio::test]
async fn outbound_gossip_carries_signed_records_of_this_node() {
    let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();

    let first = node.collect_outbound_gossip().await;
    let second = node.collect_outbound_gossip().await;

    let [own] = first.signed_records.as_slice() else {
        panic!("expected only this node's record");
    };
    let verified = VerifiedRecord::verify(NODE_RECORD, &own.signed, &own.signature).unwrap();
    assert_eq!(verified.endpoint_id(), node.id());
    assert_eq!(
        crate::proto::node::NodeRecord::decode(verified.body())
            .unwrap()
            .version,
        crate::VERSION.to_string()
    );
    assert_eq!(
        second.signed_records, first.signed_records,
        "an unchanged record is not re-signed"
    );
    let [own_cache] = first.signed_cache_affinity.as_slice() else {
        panic!("expected only this node's cache affinity");
    };
    let verified = VerifiedRecord::verify(
        CACHE_AFFINITY_RECORD,
        &own_cache.signed,
        &own_cache.signature,
    )
    .unwrap();
    assert_eq!(verified.endpoint_id(), node.id());
    assert!(
        first
            .announcements
            .iter()
            .any(|ann| ann.addr.id == node.id() && ann.cache_affinity.is_some()),
        "older nodes still get the unsigned announcement and cache affinity"
    );
}

#[tokio::test]
async fn relayed_record_is_applied_and_forwarded_byte_for_byte() {
    let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();
    let record = signed_worker_record(0x61, 1, "signed-model");

    let resolved = receive(&node, 0x62, Vec::new(), vec![record.clone()]).await;

    assert!(ids(&resolved).contains(&test_endpoint_id(0x61)));
    assert_eq!(
        serving_models(&node, 0x61).await,
        Some(vec!["signed-model".to_string()])
    );
    let outbound = node.collect_outbound_gossip().await;
    let forwarded = outbound
        .signed_records
        .iter()
        .find(|wire| wire.signed == record.signed)
        .expect("record is forwarded with its original bytes");
    assert_eq!(forwarded.signature, record.signature);
    assert!(forwarded.hop.is_some(), "the relay adds its own hop view");
}

#[tokio::test]
async fn record_replaces_its_unsigned_twin_but_keeps_its_cache_affinity() {
    let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();
    let mut twin = unsigned_worker(0x61, "relay-view-model");
    twin.1.cache_affinity = Some(mesh_llm_routing::cache_inventory::CacheAffinityAdvertisement {
        salt: [3; mesh_llm_routing::cache_inventory::CACHE_AFFINITY_SALT_BYTES],
        epoch: 7,
        generated_at_unix_ms: current_time_unix_ms(),
        ttl_ms: 120_000,
        entries: Vec::new(),
    });

    receive(
        &node,
        0x62,
        vec![twin],
        vec![signed_worker_record(0x61, 1, "signed-model")],
    )
    .await;

    let state = node.state.lock().await;
    let peer = state.peers.get(&test_endpoint_id(0x61)).expect("peer added");
    assert_eq!(peer.serving_models, vec!["signed-model".to_string()]);
    assert_eq!(
        peer.cache_affinity.as_ref().map(|advertisement| advertisement.epoch),
        Some(7)
    );
}

#[tokio::test]
async fn unsigned_announcements_from_older_relays_still_apply() {
    let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();
    receive(
        &node,
        0x62,
        Vec::new(),
        vec![signed_worker_record(0x61, 1, "signed-model")],
    )
    .await;

    // A relay that predates signed records only sends unsigned entries.
    let resolved = receive(&node, 0x63, vec![unsigned_worker(0x61, "unsigned-model")], Vec::new()).await;

    assert!(ids(&resolved).contains(&test_endpoint_id(0x61)));
    assert_eq!(
        serving_models(&node, 0x61).await,
        Some(vec!["unsigned-model".to_string()])
    );
}

#[tokio::test]
async fn relays_can_only_move_a_record_forward() {
    let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();
    receive(
        &node,
        0x62,
        Vec::new(),
        vec![signed_worker_record(0x61, 5, "newer-model")],
    )
    .await;

    receive(
        &node,
        0x63,
        Vec::new(),
        vec![signed_worker_record(0x61, 4, "older-model")],
    )
    .await;

    assert_eq!(
        serving_models(&node, 0x61).await,
        Some(vec!["newer-model".to_string()])
    );
}

#[tokio::test]
async fn forged_record_is_dropped_and_its_unsigned_twin_applies() {
    let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();
    let mut forged = signed_worker_record(0x61, 1, "forged-model");
    let last = forged.signed.len() - 1;
    forged.signed[last] ^= 0x01;

    receive(
        &node,
        0x62,
        vec![unsigned_worker(0x61, "relay-view-model")],
        vec![forged],
    )
    .await;

    assert_eq!(
        serving_models(&node, 0x61).await,
        Some(vec!["relay-view-model".to_string()])
    );
    assert!(
        !node
            .state
            .lock()
            .await
            .node_records
            .contains(&test_endpoint_id(0x61)),
        "a forged record is never held or relayed"
    );
}

#[tokio::test]
async fn unsigned_announcements_still_work_for_nodes_without_records() {
    let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();

    let resolved = receive(&node, 0x62, vec![unsigned_worker(0x64, "legacy-model")], Vec::new()).await;

    assert!(ids(&resolved).contains(&test_endpoint_id(0x64)));
    assert_eq!(
        serving_models(&node, 0x64).await,
        Some(vec!["legacy-model".to_string()])
    );
}

#[tokio::test]
async fn signed_cache_affinity_replaces_the_unsigned_copy_and_is_forwarded() {
    let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();
    let mut twin = unsigned_worker(0x61, "signed-model");
    twin.1.cache_affinity = unsigned_cache_affinity("signed-model", 1);
    let signed = signed_cache_affinity(0x61, 1, "signed-model", 2);

    receive_frame(
        &node,
        0x62,
        InboundGossip {
            announcements: vec![twin],
            signed_records: vec![signed_worker_record(0x61, 1, "signed-model")],
            signed_cache_affinity: vec![signed.clone()],
        },
    )
    .await;

    assert_eq!(cache_epoch(&node, 0x61).await, Some(2));
    let outbound = node.collect_outbound_gossip().await;
    assert!(
        outbound
            .signed_cache_affinity
            .iter()
            .any(|wire| wire.signed == signed.signed && wire.signature == signed.signature),
        "cache affinity is forwarded with its original bytes"
    );
}

#[tokio::test]
async fn signed_cache_affinity_applies_to_peers_known_only_by_unsigned_entries() {
    let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();

    receive_frame(
        &node,
        0x62,
        InboundGossip {
            announcements: vec![unsigned_worker(0x61, "legacy-model")],
            signed_records: Vec::new(),
            signed_cache_affinity: vec![signed_cache_affinity(0x61, 1, "legacy-model", 5)],
        },
    )
    .await;

    assert_eq!(cache_epoch(&node, 0x61).await, Some(5));
}

#[tokio::test]
async fn forged_cache_affinity_falls_back_to_the_unsigned_copy() {
    let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();
    let mut twin = unsigned_worker(0x61, "legacy-model");
    twin.1.cache_affinity = unsigned_cache_affinity("legacy-model", 1);
    let mut forged = signed_cache_affinity(0x61, 1, "legacy-model", 9);
    let last = forged.signed.len() - 1;
    forged.signed[last] ^= 0x01;

    receive_frame(
        &node,
        0x62,
        InboundGossip {
            announcements: vec![twin],
            signed_records: Vec::new(),
            signed_cache_affinity: vec![forged],
        },
    )
    .await;

    assert_eq!(cache_epoch(&node, 0x61).await, Some(1));
    assert!(
        !node
            .state
            .lock()
            .await
            .cache_affinity_records
            .contains(&test_endpoint_id(0x61))
    );
}

#[tokio::test]
async fn cache_affinity_for_models_the_peer_does_not_route_is_dropped() {
    let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();

    receive_frame(
        &node,
        0x62,
        InboundGossip {
            announcements: vec![unsigned_worker(0x61, "served-model")],
            signed_records: Vec::new(),
            signed_cache_affinity: vec![signed_cache_affinity(0x61, 1, "other-model", 5)],
        },
    )
    .await;

    let state = node.state.lock().await;
    let advertisement = state
        .peers
        .get(&test_endpoint_id(0x61))
        .and_then(|peer| peer.cache_affinity.as_ref())
        .expect("advertisement kept");
    assert!(advertisement.entries.is_empty());
}
