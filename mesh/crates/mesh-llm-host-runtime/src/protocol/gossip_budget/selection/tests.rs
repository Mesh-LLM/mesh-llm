use super::*;
use crate::proto::node::{NodeRecord, NodeRole};
use mesh_llm_membership::signed_record::VerifiedRecord;
use std::collections::BTreeSet;

fn key(seed: u8) -> ed25519_dalek::SigningKey {
    ed25519_dalek::SigningKey::from_bytes(&[seed; 32])
}

fn endpoint(seed: u8) -> EndpointId {
    EndpointId::from_bytes(&key(seed).verifying_key().to_bytes()).unwrap()
}

fn signed_node(seed: u8, body: &[u8]) -> SignedNodeRecord {
    let record = VerifiedRecord::sign(
        NODE_RECORD,
        &key(seed),
        1,
        mesh_llm_membership::requirements::current_time_unix_ms(),
        body,
    );
    SignedNodeRecord {
        signed: record.signed_bytes().to_vec(),
        signature: record.signature().to_vec(),
        hop: None,
    }
}

fn sender_frame() -> GossipFrame {
    GossipFrame {
        r#gen: 1,
        sender_id: endpoint(250).as_bytes().to_vec(),
        peers: vec![PeerAnnouncement {
            endpoint_id: endpoint(250).as_bytes().to_vec(),
            hostname: Some("sender-with-admission-metadata".into()),
            ..Default::default()
        }],
        ..Default::default()
    }
}

fn large_frame(compatibility: bool) -> GossipFrame {
    let body = NodeRecord {
        role: NodeRole::Worker as i32,
        serving_models: (0..250)
            .map(|index| format!("{index:03}-{}", "model".repeat(80)))
            .collect(),
        ..Default::default()
    }
    .encode_to_vec();
    let mut frame = sender_frame();
    for seed in 1..=90 {
        let node = signed_node(seed, &body);
        VerifiedRecord::verify(NODE_RECORD, &node.signed, &node.signature).unwrap();
        frame.signed_records.push(node);
        let issued_at = mesh_llm_membership::requirements::current_time_unix_ms();
        let advertisement = crate::proto::node::CacheAffinityAdvertisement {
            salt: vec![3; 32],
            epoch: 1,
            generated_at_unix_ms: issued_at,
            ttl_ms: 120_000,
            entries: Vec::new(),
        };
        assert!(crate::protocol::proto_cache_affinity_to_local(&advertisement).is_some());
        let cache = VerifiedRecord::sign(
            CACHE_AFFINITY_RECORD,
            &key(seed),
            1,
            issued_at,
            &advertisement.encode_to_vec(),
        );
        frame.signed_cache_affinity.push(SignedCacheAffinity {
            signed: cache.signed_bytes().to_vec(),
            signature: cache.signature().to_vec(),
        });
        if compatibility {
            frame.peers.push(PeerAnnouncement {
                endpoint_id: endpoint(seed).as_bytes().to_vec(),
                serving_models: NodeRecord::decode(body.as_slice()).unwrap().serving_models,
                ..Default::default()
            });
        }
    }
    assert!(frame.encoded_len() > MAX_CONTROL_FRAME_BYTES);
    frame
}

fn assert_rotates_all_origins(compatibility: bool) {
    let encoder = GossipFrameEncoder::default();
    let frame = large_frame(compatibility);
    let mut seen = BTreeSet::new();
    for _ in 0..4 {
        let bytes = encoder.encode(frame.clone(), endpoint(251)).unwrap();
        assert!(bytes.len() <= MAX_CONTROL_FRAME_BYTES);
        let decoded = GossipFrame::decode(bytes.as_slice()).unwrap();
        assert_eq!(decoded.peers[0], frame.peers[0]);
        assert!(!decoded.signed_records.is_empty());
        assert_eq!(
            decoded.signed_records.len(),
            decoded.signed_cache_affinity.len()
        );
        for (node, cache) in decoded
            .signed_records
            .iter()
            .zip(&decoded.signed_cache_affinity)
        {
            let origin = RecordHeader::parse(&NODE_RECORD, &node.signed)
                .unwrap()
                .endpoint_id;
            assert_eq!(
                origin,
                RecordHeader::parse(&CACHE_AFFINITY_RECORD, &cache.signed)
                    .unwrap()
                    .endpoint_id
            );
            assert!(
                frame.signed_records.contains(node),
                "signed bytes and hop data stay intact"
            );
            assert!(frame.signed_cache_affinity.contains(cache));
            if compatibility {
                assert!(
                    decoded
                        .peers
                        .iter()
                        .any(|peer| peer.endpoint_id == origin.as_bytes())
                );
            }
            seen.insert(origin);
        }
    }
    assert_eq!(
        seen.len(),
        90,
        "later frames must carry the deferred origins"
    );
}

#[test]
fn strict_frames_rotate_valid_signed_records_and_matching_caches() {
    assert_rotates_all_origins(false);
}

#[test]
fn compatibility_frames_keep_unsigned_twins_with_each_signed_origin() {
    assert_rotates_all_origins(true);
}

#[test]
fn small_frames_preserve_all_bytes_and_order() {
    let mut frame = sender_frame();
    frame
        .signed_records
        .push(signed_node(1, &NodeRecord::default().encode_to_vec()));
    assert_eq!(
        GossipFrameEncoder::default()
            .encode(frame.clone(), endpoint(2))
            .unwrap(),
        frame.encode_to_vec()
    );
}

fn legacy_frame() -> GossipFrame {
    let mut frame = sender_frame();
    for seed in 1..=3 {
        frame.peers.push(PeerAnnouncement {
            endpoint_id: endpoint(seed).as_bytes().to_vec(),
            hostname: Some("legacy".repeat(40)),
            ..Default::default()
        });
    }
    frame
}

#[test]
fn destinations_rotate_independently_and_legacy_only_peers_make_progress() {
    let encoder = GossipFrameEncoder::default();
    let frame = legacy_frame();
    let limit =
        sender_frame().encoded_len() + prost::encoding::message::encoded_len(2, &frame.peers[1]);
    let first = encoder
        .encode_with_limit(frame.clone(), endpoint(248), limit)
        .unwrap();
    let next = encoder
        .encode_with_limit(frame.clone(), endpoint(248), limit)
        .unwrap();
    assert_ne!(first, next);
    let other = encoder
        .encode_with_limit(frame.clone(), endpoint(249), limit)
        .unwrap();
    assert_eq!(first, other);
    let last = encoder
        .encode_with_limit(frame.clone(), endpoint(248), limit)
        .unwrap();
    let seen: BTreeSet<_> = [first, next, last]
        .iter()
        .flat_map(|bytes| {
            let decoded = GossipFrame::decode(bytes.as_slice()).unwrap();
            assert_eq!(decoded.peers[0], frame.peers[0]);
            assert_eq!(decoded.peers.len(), 2);
            decoded
                .peers
                .into_iter()
                .skip(1)
                .map(|peer| peer.endpoint_id)
        })
        .collect();
    assert_eq!(seen.len(), 3);
}

#[test]
fn batches_also_respect_the_receivers_record_count_limit() {
    let encoder = GossipFrameEncoder::default();
    let mut frame = sender_frame();
    for seed in 1..=3 {
        frame.signed_records.extend(std::iter::repeat_n(
            signed_node(seed, &NodeRecord::default().encode_to_vec()),
            1800,
        ));
    }
    assert!(frame.encoded_len() < MAX_CONTROL_FRAME_BYTES);
    let mut seen = BTreeSet::new();
    for _ in 0..2 {
        let bytes = encoder.encode(frame.clone(), endpoint(249)).unwrap();
        let decoded = GossipFrame::decode(bytes.as_slice()).unwrap();
        assert!(signed_count(&decoded) <= MAX_GOSSIP_RECORDS);
        seen.extend(decoded.signed_records.iter().map(|node| {
            RecordHeader::parse(&NODE_RECORD, &node.signed)
                .unwrap()
                .endpoint_id
        }));
    }
    assert_eq!(seen.len(), 3);
}

#[test]
fn untransportable_sender_still_reports_local_deferral() {
    let encoder = GossipFrameEncoder::default();
    let frame = sender_frame();
    assert!(
        encoder
            .encode_with_limit(frame.clone(), endpoint(249), frame.encoded_len() - 1)
            .unwrap_err()
            .is::<GossipFrameBudgetExceeded>()
    );
}

#[test]
fn an_oversized_local_record_preserves_the_local_deferral_classification() {
    let mut frame = sender_frame();
    let own = signed_node(250, &vec![0; NODE_RECORD.max_signed_bytes]);
    assert!(matches!(
        RecordHeader::parse(&NODE_RECORD, &own.signed),
        Err(mesh_llm_membership::signed_record::RecordError::Oversize)
    ));
    frame.signed_records.push(own);
    let limit = frame.encoded_len() - 1;
    let error = GossipFrameEncoder::default()
        .encode_with_limit(frame, endpoint(249), limit)
        .unwrap_err();
    assert!(error.is::<GossipFrameBudgetExceeded>());
}

#[test]
fn an_untransportable_relay_does_not_block_smaller_origins() {
    let encoder = GossipFrameEncoder::default();
    let mut frame = legacy_frame();
    frame.peers[1].hostname = Some("oversized".repeat(1000));
    let limit =
        sender_frame().encoded_len() + prost::encoding::message::encoded_len(2, &frame.peers[2]);
    let mut seen = BTreeSet::new();
    for _ in 0..3 {
        let bytes = encoder
            .encode_with_limit(frame.clone(), endpoint(249), limit)
            .unwrap();
        let decoded = GossipFrame::decode(bytes.as_slice()).unwrap();
        assert_eq!(decoded.peers.len(), 2);
        seen.insert(decoded.peers[1].endpoint_id.clone());
    }
    assert_eq!(seen.len(), 2);
    assert!(!seen.contains(&frame.peers[1].endpoint_id));
}

#[test]
fn connection_churn_keeps_cursor_memory_bounded() {
    let encoder = GossipFrameEncoder::default();
    let frame = legacy_frame();
    let limit =
        sender_frame().encoded_len() + prost::encoding::message::encoded_len(2, &frame.peers[1]);
    for index in 0..=MAX_CURSOR_PEERS {
        let mut seed = [0; 32];
        seed[..8].copy_from_slice(&(index as u64).to_be_bytes());
        let remote = iroh::SecretKey::from_bytes(&seed).public();
        encoder
            .encode_with_limit(frame.clone(), remote, limit)
            .unwrap();
    }
    assert_eq!(encoder.cursors.lock().unwrap().len(), MAX_CURSOR_PEERS);
}
