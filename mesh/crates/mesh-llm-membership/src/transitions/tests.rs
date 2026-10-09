use super::*;
use crate::types::NodeRole;
use iroh::SecretKey;
use std::collections::HashMap;

pub(crate) fn test_endpoint_id(seed: u8) -> EndpointId {
    EndpointId::from(SecretKey::from_bytes(&[seed; 32]).public())
}

pub(crate) fn test_addr(seed: u8) -> EndpointAddr {
    EndpointAddr {
        id: test_endpoint_id(seed),
        addrs: Default::default(),
    }
}

pub(crate) fn test_announcement(ts: Option<u64>) -> PeerAnnouncement {
    PeerAnnouncement {
        addr: test_addr(0x11),
        role: NodeRole::Worker,
        first_joined_mesh_ts: ts,
        models: vec![],
        vram_bytes: 0,
        model_source: None,
        serving_models: vec![],
        hosted_models: None,
        available_models: vec![],
        requested_models: vec![],
        explicit_model_interests: vec![],
        version: None,
        model_demand: HashMap::new(),
        mesh_id: None,
        mesh_policy_hash: None,
        gpu_name: None,
        hostname: None,
        is_soc: None,
        gpu_vram: None,
        gpu_reserved_bytes: None,
        memory: None,
        gpu_mem_bandwidth_gbps: None,
        gpu_compute_tflops_fp32: None,
        gpu_compute_tflops_fp16: None,
        available_model_metadata: vec![],
        experts_summary: None,
        available_model_sizes: HashMap::new(),
        served_model_descriptors: vec![],
        served_model_runtime: vec![],
        owner_attestation: None,
        genesis_policy: None,
        release_attestation: None,
        direct_admission_proof: None,
        artifact_transfer_supported: true,
        stage_protocol_generation_supported: true,
        stage_status_list_supported: true,
        local_gguf_content_id_supported: true,
        decode_batch_policy_supported: true,
        #[cfg(feature = "payments")]
        lightning_offers: Default::default(),
        advertised_model_throughput: vec![],
        cache_affinity: None,
        latency_ms: None,
        latency_source: None,
        latency_age_ms: None,
        latency_observer_id: None,
        inference_admission_state: None,
        claimed_log_head: None,
    }
}

fn apply(state: &mut MembershipState, ann: &PeerAnnouncement) -> TransitivePeerUpdate {
    state.apply_accepted_transitive_peer(
        test_endpoint_id(1),
        ann.addr.id,
        &ann.addr,
        ann,
        test_endpoint_id(2),
        OwnershipSummary::default(),
    )
}

#[test]
fn bridge_cannot_admit_peer_or_refresh_direct_liveness() {
    let mut state = MembershipState::default();
    let ann = test_announcement(None);
    let TransitivePeerUpdate::Added(peer) = apply(&mut state, &ann) else {
        panic!("expected new peer")
    };
    assert!(!peer.is_admitted());
    assert!(!peer.local_gguf_content_id_supported);
    assert!(!peer.stage_protocol_generation_supported);
    assert!(peer.last_seen.elapsed() >= Duration::from_secs(PEER_STALE_SECS * 2));
    let last_seen = peer.last_seen;
    let TransitivePeerUpdate::Updated {
        peer,
        changed,
        admitted_count,
    } = apply(&mut state, &ann)
    else {
        panic!("expected existing peer")
    };
    assert_eq!(peer.last_seen, last_seen);
    assert!(!peer.is_admitted());
    assert!(!peer.local_gguf_content_id_supported);
    assert!(!peer.stage_protocol_generation_supported);
    assert!(!changed);
    assert_eq!(admitted_count, None);
    assert_eq!(state.admitted_peer_count(), 0);
}

#[test]
fn dead_peer_quarantine_expires_but_local_peer_is_always_ignored() {
    let mut state = MembershipState::default();
    let mut ann = test_announcement(None);
    state.dead_peers.insert(ann.addr.id, Instant::now());
    assert!(matches!(
        apply(&mut state, &ann),
        TransitivePeerUpdate::Ignored
    ));
    assert!(state.peers.is_empty());
    state
        .dead_peers
        .insert(ann.addr.id, Instant::now() - DEAD_PEER_TTL);
    assert!(matches!(
        apply(&mut state, &ann),
        TransitivePeerUpdate::Added(_)
    ));
    ann.addr = test_addr(1);
    assert!(matches!(
        apply(&mut state, &ann),
        TransitivePeerUpdate::Ignored
    ));
    assert_eq!(state.peers.len(), 1);
}

#[test]
fn direct_admission_promotes_transitive_peer_and_clears_policy_rejection() {
    let mut state = MembershipState::default();
    let ann = test_announcement(None);
    apply(&mut state, &ann);
    state
        .policy_rejected_peers
        .insert(ann.addr.id, Default::default());
    let now = Instant::now();
    let update = state
        .upsert_existing_direct_peer(
            ann.addr.id,
            ann.addr.clone(),
            &ann,
            OwnershipSummary::default(),
            now,
        )
        .expect("existing transitive peer");
    assert!(update.peer.is_admitted());
    assert!(update.peer.local_gguf_content_id_supported);
    assert_eq!(update.peer.last_seen, now);
    assert_eq!(update.admitted_count, 1);
    assert!(!state.policy_rejected_peers.contains_key(&ann.addr.id));
}

#[test]
fn transitive_serving_update_reports_count_without_renewing_direct_proof() {
    let mut state = MembershipState::default();
    let mut ann = test_announcement(None);
    let (direct, count) = state.insert_new_direct_peer(
        ann.addr.id,
        ann.addr.clone(),
        &ann,
        OwnershipSummary::default(),
    );
    assert_eq!(count, 1);
    ann.serving_models.push("model".into());
    let TransitivePeerUpdate::Updated {
        peer,
        changed,
        admitted_count,
    } = apply(&mut state, &ann)
    else {
        panic!("expected update")
    };
    assert!(changed);
    assert_eq!(admitted_count, Some(1));
    assert_eq!(peer.last_seen, direct.last_seen);
    assert!(peer.is_admitted());
}

#[test]
fn disallowed_removal_preserves_quarantine_and_rejection_records() {
    let mut state = MembershipState::default();
    let ann = test_announcement(None);
    state.insert_new_direct_peer(
        ann.addr.id,
        ann.addr.clone(),
        &ann,
        OwnershipSummary::default(),
    );
    state.dead_peers.insert(ann.addr.id, Instant::now());
    state.requirement_rejected_peers.insert(ann.addr.id);
    state
        .policy_rejected_peers
        .insert(ann.addr.id, Default::default());
    assert_eq!(state.remove_disallowed_peer(ann.addr.id), Some(0));
    assert_eq!(state.remove_disallowed_peer(ann.addr.id), None);
    assert!(state.dead_peers.contains_key(&ann.addr.id));
    assert!(state.requirement_rejected_peers.contains(&ann.addr.id));
    assert!(state.policy_rejected_peers.contains_key(&ann.addr.id));
}

#[test]
fn full_removal_clears_rejections_even_without_a_peer() {
    let mut state = MembershipState::default();
    let ann = test_announcement(None);
    let id = ann.addr.id;
    state.insert_new_direct_peer(id, ann.addr.clone(), &ann, OwnershipSummary::default());
    state.dead_peers.insert(id, Instant::now());
    state.requirement_rejected_peers.insert(id);
    state.policy_rejected_peers.insert(id, Default::default());
    let removed = state.remove_peer(id).expect("peer removal");
    assert_eq!(removed.peer.id, id);
    assert_eq!(removed.admitted_count, 0);
    assert_eq!(removed.remaining_count, 0);
    assert!(state.dead_peers.contains_key(&id));
    assert!(!state.requirement_rejected_peers.contains(&id));
    assert!(!state.policy_rejected_peers.contains_key(&id));
    state.requirement_rejected_peers.insert(id);
    state.policy_rejected_peers.insert(id, Default::default());
    assert!(state.remove_peer(id).is_none());
    assert!(!state.requirement_rejected_peers.contains(&id));
    assert!(!state.policy_rejected_peers.contains_key(&id));
}

#[test]
fn pruning_requires_both_direct_and_transitive_contact_to_be_stale() {
    let mut state = MembershipState::default();
    let ann = test_announcement(None);
    let id = ann.addr.id;
    state.insert_new_direct_peer(id, ann.addr.clone(), &ann, OwnershipSummary::default());
    let cutoff = Instant::now() - Duration::from_secs(10);
    let peer = state.peers.get_mut(&id).unwrap();
    peer.last_seen = cutoff - Duration::from_secs(1);
    assert!(state.stale_peers(cutoff).is_empty());
    state.peers.get_mut(&id).unwrap().last_mentioned = cutoff - Duration::from_secs(1);
    assert_eq!(state.stale_peers(cutoff), vec![id]);
    state.peers.get_mut(&id).unwrap().last_seen = Instant::now();
    assert!(state.stale_peers(cutoff).is_empty());
}

#[test]
fn heartbeat_gc_expires_each_cooldown_without_removing_peer_records() {
    let mut state = MembershipState::default();
    let ann = test_announcement(None);
    let expired_id = ann.addr.id;
    let live_id = test_endpoint_id(2);
    state.insert_new_direct_peer(
        expired_id,
        ann.addr.clone(),
        &ann,
        OwnershipSummary::default(),
    );
    let now = Instant::now();
    state.dead_peers.insert(expired_id, now - DEAD_PEER_TTL);
    state.dead_peers.insert(live_id, now);
    state.peer_down_rejections.insert(
        (live_id, expired_id),
        now - Duration::from_secs(PEER_DOWN_REPORTER_COOLDOWN_SECS),
    );
    state
        .peer_down_rejections
        .insert((expired_id, live_id), now);
    state
        .direct_path_request_last_at
        .insert(expired_id, now - Duration::from_secs(120));
    state.direct_path_request_last_at.insert(live_id, now);
    assert_eq!(
        state.retain_live_heartbeat_state(Duration::from_secs(120)),
        vec![expired_id]
    );
    assert_eq!(state.dead_peers.len(), 1);
    assert!(state.dead_peers.contains_key(&live_id));
    assert_eq!(state.peer_down_rejections.len(), 1);
    assert!(
        state
            .peer_down_rejections
            .contains_key(&(expired_id, live_id))
    );
    assert_eq!(state.direct_path_request_last_at.len(), 1);
    assert!(state.direct_path_request_last_at.contains_key(&live_id));
    assert!(state.peers.contains_key(&expired_id));
}

#[cfg(feature = "payments")]
#[test]
fn transitive_prices_propagate_until_direct_admission() {
    let mut state = MembershipState::default();
    let mut ann = test_announcement(None);
    apply(&mut state, &ann);
    ann.lightning_offers.insert(
        "model".into(),
        mesh_llm_payments_types::pricing::Pricing {
            input_msat_per_million: 500,
            output_msat_per_million: 1500,
        },
    );
    apply(&mut state, &ann);
    assert_eq!(
        state.peers[&ann.addr.id].lightning_offers,
        ann.lightning_offers
    );

    let mut direct = ann.clone();
    direct.lightning_offers.clear();
    state
        .upsert_existing_direct_peer(
            direct.addr.id,
            direct.addr.clone(),
            &direct,
            OwnershipSummary::default(),
            Instant::now(),
        )
        .unwrap();
    apply(&mut state, &ann);
    let peer = &state.peers[&ann.addr.id];
    assert!(peer.is_admitted());
    assert!(peer.lightning_offers.is_empty());
}
