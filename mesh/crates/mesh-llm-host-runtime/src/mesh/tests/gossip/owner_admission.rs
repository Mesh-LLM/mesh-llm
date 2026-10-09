use crate::crypto::{OwnerKeypair, TrustPolicy, TrustStore, sign_node_ownership};

fn owned_announcement(seed: u8, owner: &OwnerKeypair) -> Result<PeerAnnouncement> {
    let mut ann = test_announcement(None);
    ann.addr = test_addr(seed);
    ann.version = Some(crate::VERSION.to_string());
    ann.owner_attestation = Some(sign_node_ownership(
        owner,
        ann.addr.id.as_bytes(),
        current_time_unix_ms() + 60_000,
        None,
        None,
    )?);
    Ok(ann)
}

async fn owner_policy_node(policy: TrustPolicy, owner: &OwnerKeypair) -> Result<Node> {
    let mut node = Node::new_for_tests(NodeRole::Worker).await?;
    let mut trust_store = TrustStore::default();
    trust_store.add_trusted_owner(owner.owner_id(), None);
    *node.trust_store.lock().await = trust_store;
    node.trust_policy = policy;
    Ok(node)
}

async fn assert_rejected_relay_frame(
    policy: TrustPolicy,
    signed_untrusted_sender: bool,
    direct_first: bool,
    requirements_validated: bool,
) -> Result<()> {
    let owner = OwnerKeypair::from_bytes(&[0x31; 32], &[0x32; 32])?;
    let node = owner_policy_node(policy, &owner).await?;
    let mut victim = owned_announcement(0x95, &owner)?;
    victim.serving_models = vec!["trusted-model".to_string()];
    let victim_id = victim.addr.id;
    node.add_peer(
        victim_id,
        victim.addr.clone(),
        &victim,
        Some(NODE_PROTOCOL_GENERATION),
    )
    .await;
    assert!(node.state.lock().await.peers[&victim_id].is_admitted());

    // A public certificate authenticates the victim's endpoint, not these
    // mutable fields. The direct sender must be admitted before any replay.
    let mut forged = victim;
    forged.role = NodeRole::Host { http_port: 9337 };
    forged.serving_models = vec!["forged-model".to_string()];
    let mut sender = test_announcement(None);
    sender.addr = test_addr(0x96);
    if signed_untrusted_sender {
        let untrusted = OwnerKeypair::from_bytes(&[0x41; 32], &[0x42; 32])?;
        sender = owned_announcement(0x96, &untrusted)?;
    }
    let sender_id = sender.addr.id;
    let mut payload = vec![(forged.addr.clone(), forged), (sender.addr.clone(), sender)];
    if direct_first {
        payload.reverse();
    }
    let result = node
        .apply_announced_peers(
            sender_id,
            &payload,
            None,
            Some(NODE_PROTOCOL_GENERATION),
            requirements_validated,
        )
        .await;
    // Check metadata before the error so the unfixed code proves actual
    // poisoning, rather than merely a missing error return.
    let state = node.state.lock().await;
    assert_eq!(state.peers[&victim_id].role, NodeRole::Worker);
    assert_eq!(
        state.peers[&victim_id].serving_models,
        vec!["trusted-model"]
    );
    assert!(!state.peers.contains_key(&sender_id));
    assert!(state.policy_rejected_peers.contains_key(&sender_id));
    assert!(result.is_err(), "unauthorized sender must fail the frame");
    Ok(())
}

#[tokio::test]
pub(crate) async fn owner_rejection_precedes_relays_in_every_payload_order() -> Result<()> {
    for (policy, signed_untrusted_sender) in [
        (TrustPolicy::RequireOwned, false),
        (TrustPolicy::Allowlist, false),
        (TrustPolicy::Allowlist, true),
    ] {
        for direct_first in [false, true] {
            for requirements_validated in [false, true] {
                assert_rejected_relay_frame(
                    policy,
                    signed_untrusted_sender,
                    direct_first,
                    requirements_validated,
                )
                .await?;
            }
        }
    }
    Ok(())
}

#[tokio::test]
pub(crate) async fn authorized_senders_can_relay_before_their_direct_entry() -> Result<()> {
    let owner = OwnerKeypair::from_bytes(&[0x31; 32], &[0x32; 32])?;
    for policy in [
        TrustPolicy::Off,
        TrustPolicy::PreferOwned,
        TrustPolicy::RequireOwned,
        TrustPolicy::Allowlist,
    ] {
        let node = owner_policy_node(policy, &owner).await?;
        let relay = owned_announcement(0x95, &owner)?;
        let mut sender = owned_announcement(0x96, &owner)?;
        if matches!(policy, TrustPolicy::Off | TrustPolicy::PreferOwned) {
            sender.owner_attestation = None;
            sender.version = None; // Older frames may omit optional metadata.
        }
        node.apply_announced_peers(
            sender.addr.id,
            &[
                (relay.addr.clone(), relay.clone()),
                (sender.addr.clone(), sender.clone()),
            ],
            None,
            Some(NODE_PROTOCOL_GENERATION),
            false,
        )
        .await?;
        let state = node.state.lock().await;
        assert!(state.peers[&sender.addr.id].is_admitted());
        assert!(state.peers.contains_key(&relay.addr.id));
    }
    Ok(())
}

#[tokio::test]
pub(crate) async fn inbound_owner_rejection_preserves_liveness_state() -> Result<()> {
    let owner = OwnerKeypair::from_bytes(&[0x31; 32], &[0x32; 32])?;
    let node = owner_policy_node(TrustPolicy::RequireOwned, &owner).await?;
    let mut sender = test_announcement(None);
    sender.addr = test_addr(0x96);
    let remote = sender.addr.id;
    {
        let mut state = node.state.lock().await;
        state.dead_peers.insert(remote, std::time::Instant::now());
        state
            .departed_peers
            .insert(remote, std::time::Instant::now());
    }
    node.validate_and_capture_inbound_gossip(
        ControlProtocol::ProtoV1,
        &[(sender.addr.clone(), sender)],
        AnnouncedPeerContext::direct(remote, Some(NODE_PROTOCOL_GENERATION)),
    )
    .await
    .expect_err("owner rejection must precede liveness recovery");
    let state = node.state.lock().await;
    assert!(state.dead_peers.contains_key(&remote));
    assert!(state.departed_peers.contains_key(&remote));
    Ok(())
}
