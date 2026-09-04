use super::{
    Node, NodeRole as MeshNodeRole, PairingDecision, PairingSession, PairingSessionStatus,
};

async fn pairing_test_node() -> anyhow::Result<Node> {
    pairing_test_node_with_requirements(crate::MeshRequirements::unrestricted()).await
}

async fn pairing_test_node_with_requirements(
    requirements: crate::MeshRequirements,
) -> anyhow::Result<Node> {
    let mut node = Node::new_for_tests(MeshNodeRole::Worker).await?;
    if !requirements.is_unrestricted() {
        node.owner_keypair = Some(crate::crypto::OwnerKeypair::generate());
    }
    node.local_mesh_requirements = requirements;
    let accepting = node.clone();
    tokio::spawn(async move { accepting.accept_loop().await });
    node.start_accepting();
    Ok(node)
}

async fn initialize_requirement_mesh(node: &Node) -> anyhow::Result<()> {
    let owner = node
        .owner_keypair
        .as_ref()
        .expect("restricted test node should have an owner");
    let policy = crate::MeshGenesisPolicy::new(
        owner.owner_id(),
        crate::mesh::current_time_unix_ms(),
        node.local_mesh_requirements.clone(),
    )
    .map_err(|reason| anyhow::anyhow!("invalid test policy: {reason:?}"))?;
    let signed = crate::SignedMeshGenesisPolicy::sign(policy.clone(), owner)
        .map_err(|reason| anyhow::anyhow!("could not sign test policy: {reason:?}"))?;
    let mesh_id = policy
        .policy_derived_mesh_id()
        .map_err(|reason| anyhow::anyhow!("invalid test mesh id: {reason:?}"))?;
    let policy_hash = policy
        .canonical_hash_hex()
        .map_err(|reason| anyhow::anyhow!("invalid test policy hash: {reason:?}"))?;
    node.install_requirement_aware_mesh_state(mesh_id, policy_hash, policy, Some(signed), None)
        .await
}

async fn wait_for_pairing_status(
    node: &Node,
    session_id: &str,
    expected: PairingSessionStatus,
) -> PairingSession {
    let result = tokio::time::timeout(std::time::Duration::from_secs(30), async {
        loop {
            if let Some(session) = node
                .pairing_sessions()
                .await
                .into_iter()
                .find(|session| session.id == session_id)
                && session.status == expected
            {
                return session;
            }
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        }
    })
    .await;
    match result {
        Ok(session) => session,
        Err(_) => panic!(
            "pairing session should reach {expected:?}; latest sessions: {:?}",
            node.pairing_sessions().await
        ),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pairing_requires_matching_bilateral_approval_before_join() -> anyhow::Result<()> {
    let host = pairing_test_node().await?;
    let joiner = pairing_test_node().await?;
    host.set_display_name("Studio Mac".to_string()).await;
    joiner.set_display_name("Kitchen PC".to_string()).await;
    host.initialize_mesh_identity_as_originator(Some("home"), None)
        .await?;

    let offer = host.create_pairing_offer().await?;
    assert!(!offer.offer.contains("invite_token"));
    let outgoing = joiner.start_pairing(&offer.offer).await?;
    let outgoing_ready = wait_for_pairing_status(
        &joiner,
        &outgoing.id,
        PairingSessionStatus::AwaitingApproval,
    )
    .await;
    let incoming_ready =
        wait_for_pairing_status(&host, &outgoing.id, PairingSessionStatus::AwaitingApproval).await;
    assert_eq!(outgoing_ready.peer_name, "Studio Mac");
    assert_eq!(incoming_ready.peer_name, "Kitchen PC");
    assert_eq!(
        outgoing_ready.comparison_code,
        incoming_ready.comparison_code
    );

    joiner
        .decide_pairing(&outgoing.id, PairingDecision::Approve)
        .await?;
    host.decide_pairing(&outgoing.id, PairingDecision::Approve)
        .await?;

    wait_for_pairing_status(&host, &outgoing.id, PairingSessionStatus::Approved).await;
    wait_for_pairing_status(&joiner, &outgoing.id, PairingSessionStatus::Approved).await;
    tokio::time::timeout(std::time::Duration::from_secs(10), async {
        loop {
            if joiner.peers().await.iter().any(|peer| peer.id == host.id()) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        }
    })
    .await
    .expect("approved pairing should join the mesh");
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pairing_rejection_withholds_invite_and_prevents_join() -> anyhow::Result<()> {
    let host = pairing_test_node().await?;
    let joiner = pairing_test_node().await?;
    host.initialize_mesh_identity_as_originator(Some("home"), None)
        .await?;
    let offer = host.create_pairing_offer().await?;
    let outgoing = joiner.start_pairing(&offer.offer).await?;
    wait_for_pairing_status(
        &joiner,
        &outgoing.id,
        PairingSessionStatus::AwaitingApproval,
    )
    .await;
    wait_for_pairing_status(&host, &outgoing.id, PairingSessionStatus::AwaitingApproval).await;

    joiner
        .decide_pairing(&outgoing.id, PairingDecision::Approve)
        .await?;
    host.decide_pairing(&outgoing.id, PairingDecision::Reject)
        .await?;
    wait_for_pairing_status(&host, &outgoing.id, PairingSessionStatus::Rejected).await;
    wait_for_pairing_status(&joiner, &outgoing.id, PairingSessionStatus::Rejected).await;
    assert!(!joiner.peers().await.iter().any(|peer| peer.id == host.id()));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pairing_can_be_cancelled_after_local_approval() -> anyhow::Result<()> {
    let host = pairing_test_node().await?;
    let joiner = pairing_test_node().await?;
    host.initialize_mesh_identity_as_originator(Some("home"), None)
        .await?;
    let offer = host.create_pairing_offer().await?;
    let outgoing = joiner.start_pairing(&offer.offer).await?;
    wait_for_pairing_status(
        &joiner,
        &outgoing.id,
        PairingSessionStatus::AwaitingApproval,
    )
    .await;
    wait_for_pairing_status(&host, &outgoing.id, PairingSessionStatus::AwaitingApproval).await;

    joiner
        .decide_pairing(&outgoing.id, PairingDecision::Approve)
        .await?;
    joiner
        .decide_pairing(&outgoing.id, PairingDecision::Cancel)
        .await?;

    wait_for_pairing_status(&joiner, &outgoing.id, PairingSessionStatus::Cancelled).await;
    wait_for_pairing_status(&host, &outgoing.id, PairingSessionStatus::Cancelled).await;
    assert!(!joiner.peers().await.iter().any(|peer| peer.id == host.id()));
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pairing_offer_is_single_use() -> anyhow::Result<()> {
    let host = pairing_test_node().await?;
    let first = pairing_test_node().await?;
    let replay = pairing_test_node().await?;
    let offer = host.create_pairing_offer().await?;
    let first_session = first.start_pairing(&offer.offer).await?;
    wait_for_pairing_status(
        &host,
        &first_session.id,
        PairingSessionStatus::AwaitingApproval,
    )
    .await;

    let replay_session = replay.start_pairing(&offer.offer).await?;
    let failed =
        wait_for_pairing_status(&replay, &replay_session.id, PairingSessionStatus::Failed).await;
    assert!(
        failed.error.is_some(),
        "replay failure should report an error: {failed:?}"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pairing_approval_does_not_bypass_mesh_admission_policy() -> anyhow::Result<()> {
    let requirements = crate::MeshRequirements {
        protocol_generation: crate::ProtocolGenerationBounds {
            min: Some(u32::MAX),
            max: None,
        },
        ..crate::MeshRequirements::unrestricted()
    };
    let host = pairing_test_node_with_requirements(requirements).await?;
    let joiner = pairing_test_node().await?;
    initialize_requirement_mesh(&host).await?;

    let offer = host.create_pairing_offer().await?;
    let session = joiner.start_pairing(&offer.offer).await?;
    wait_for_pairing_status(&joiner, &session.id, PairingSessionStatus::AwaitingApproval).await;
    wait_for_pairing_status(&host, &session.id, PairingSessionStatus::AwaitingApproval).await;

    joiner
        .decide_pairing(&session.id, PairingDecision::Approve)
        .await?;
    host.decide_pairing(&session.id, PairingDecision::Approve)
        .await?;

    let joiner_failed =
        wait_for_pairing_status(&joiner, &session.id, PairingSessionStatus::Failed).await;
    let host_failed =
        wait_for_pairing_status(&host, &session.id, PairingSessionStatus::Failed).await;
    assert!(joiner_failed.error.is_some());
    assert!(host_failed.error.is_some());
    assert!(!joiner.peers().await.iter().any(|peer| peer.id == host.id()));
    assert!(!host.peers().await.iter().any(|peer| peer.id == joiner.id()));
    Ok(())
}
