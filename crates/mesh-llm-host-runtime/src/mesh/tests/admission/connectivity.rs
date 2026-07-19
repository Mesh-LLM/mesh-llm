use super::*;
use tokio::sync::watch;

/// RTT re-election: when a peer's RTT drops from above the 80ms split
/// threshold to below it (e.g. relay → direct), update_peer_rtt must
/// trigger a peer_change event so the election loop re-runs and can
/// now include the peer in split mode.
#[tokio::test]
async fn test_rtt_drop_triggers_reelection() -> Result<()> {
    let node = make_test_node(super::super::NodeRole::Worker).await?;
    let peer_key = SecretKey::generate();
    let peer_id = EndpointId::from(peer_key.public());

    // Add a fake peer with high relay RTT
    {
        let mut state = node.state.lock().await;
        state
            .peers
            .insert(peer_id, make_test_peer(peer_id, Some(2600), 16));
    }

    let rx = node.peer_change_rx.clone();

    // Update RTT to still-high value — should NOT trigger
    node.update_peer_rtt(peer_id, 500).await;
    assert!(
        !rx.has_changed()
            .expect("peer_change_rx closed unexpectedly"),
        "RTT 2600→500 (both above threshold) should not trigger re-election"
    );

    // Update RTT to below threshold — SHOULD trigger
    node.update_peer_rtt(peer_id, 15).await;
    assert!(
        rx.has_changed()
            .expect("peer_change_rx closed unexpectedly"),
        "RTT 500→15 (crossing threshold) must trigger re-election"
    );

    Ok(())
}

/// RTT re-election should NOT trigger when RTT was already below threshold.
#[tokio::test]
async fn test_rtt_below_threshold_no_reelection() -> Result<()> {
    let node = make_test_node(super::super::NodeRole::Worker).await?;
    let peer_key = SecretKey::generate();
    let peer_id = EndpointId::from(peer_key.public());

    {
        let mut state = node.state.lock().await;
        state
            .peers
            .insert(peer_id, make_test_peer(peer_id, Some(20), 16));
    }

    let rx = node.peer_change_rx.clone();

    // Update RTT to another low value — should NOT trigger
    node.update_peer_rtt(peer_id, 15).await;
    assert!(
        !rx.has_changed()
            .expect("peer_change_rx closed unexpectedly"),
        "RTT 20→15 (both below threshold) should not trigger re-election"
    );

    Ok(())
}

/// RTT re-election should NOT trigger for unknown peers.
#[tokio::test]
async fn test_rtt_update_unknown_peer_no_panic() -> Result<()> {
    let node = make_test_node(super::super::NodeRole::Worker).await?;
    let peer_key = SecretKey::generate();
    let peer_id = EndpointId::from(peer_key.public());

    let rx = node.peer_change_rx.clone();

    // Update RTT for a peer that doesn't exist — should not panic or trigger
    node.update_peer_rtt(peer_id, 15).await;
    assert!(
        !rx.has_changed()
            .expect("peer_change_rx closed unexpectedly"),
        "RTT update for unknown peer should not trigger re-election"
    );

    Ok(())
}

/// RTT should never increase — relay gossip RTT must not overwrite
/// a known-good direct path measurement.
#[tokio::test]
async fn test_rtt_cannot_regress() -> Result<()> {
    let node = make_test_node(super::super::NodeRole::Worker).await?;
    let peer_key = SecretKey::generate();
    let peer_id = EndpointId::from(peer_key.public());

    {
        let mut state = node.state.lock().await;
        state
            .peers
            .insert(peer_id, make_test_peer(peer_id, Some(20), 16));
    }

    // Try to raise RTT — should be rejected
    node.update_peer_rtt(peer_id, 2600).await;
    {
        let state = node.state.lock().await;
        let rtt = state.peers.get(&peer_id).unwrap().rtt_ms;
        assert_eq!(rtt, Some(20), "RTT must not increase from 20 to 2600");
    }

    // Lower RTT — should be accepted
    node.update_peer_rtt(peer_id, 10).await;
    {
        let state = node.state.lock().await;
        let rtt = state.peers.get(&peer_id).unwrap().rtt_ms;
        assert_eq!(rtt, Some(10), "RTT must decrease from 20 to 10");
    }

    Ok(())
}

/// Discovered peers must still be dialed directly before admission.
#[tokio::test]
async fn test_connect_to_peer_attempts_direct_verification_for_known_unadmitted_peer() -> Result<()>
{
    let node = make_test_node(super::super::NodeRole::Client).await?;
    let peer_key = SecretKey::generate();
    let peer_id = EndpointId::from(peer_key.public());

    // Simulate a transitive peer: tracked as a hint but not yet admitted.
    {
        let mut state = node.state.lock().await;
        let mut peer = make_test_peer(peer_id, Some(50), 8);
        peer.admitted = false;
        state.peers.insert(peer_id, peer);
        assert!(
            !state.connections.contains_key(&peer_id),
            "setup: peer must not have a connection"
        );
    }

    // connect_to_peer must attempt direct verification instead of treating the
    // hint as already admitted.
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(1),
        node.connect_to_peer(super::super::EndpointAddr {
            id: peer_id,
            addrs: Default::default(),
        }),
    )
    .await;

    assert!(
        result.is_ok(),
        "connect_to_peer should complete quickly for a discovered-only peer"
    );
    assert!(
        result.unwrap().is_err(),
        "connect_to_peer must try direct verification instead of silently accepting a hint"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_on_demand_transitive_peer_connection_completes_gossip() -> Result<()> {
    let host = make_test_node(super::super::NodeRole::Host { http_port: 9337 }).await?;
    let bridge = make_test_node(super::super::NodeRole::Worker).await?;
    let client = make_test_node(super::super::NodeRole::Client).await?;

    host.set_hosted_models(vec!["remote-coding-model".to_string()])
        .await;
    host.start_accepting();
    bridge.start_accepting();
    client.start_accepting();

    bridge.sync_from_peer_for_tests(&host).await;
    assert!(bridge.peers().await.iter().any(|peer| peer.id == host.id()));

    client.sync_from_peer_for_tests(&bridge).await;
    assert!(
        client
            .peers()
            .await
            .iter()
            .any(|peer| peer.id == bridge.id())
    );

    {
        let state = client.state.lock().await;
        assert!(
            !state.connections.contains_key(&host.id()),
            "setup: host should be known transitively but not directly connected"
        );
    }
    assert!(
        !client
            .hosts_for_model("remote-coding-model")
            .await
            .contains(&host.id()),
        "setup: client must not route to the transitive host before direct verification"
    );

    let _conn = client.connection_to_peer(host.id()).await?;

    wait_for_peer(&client, host.id()).await;
    {
        let state = client.state.lock().await;
        assert!(
            state.connections.contains_key(&host.id()),
            "on-demand connection should be retained after gossip succeeds"
        );
    }
    assert!(
        client
            .hosts_for_model("remote-coding-model")
            .await
            .contains(&host.id()),
        "the host should become routable after direct gossip succeeds"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cached_unadmitted_connection_completes_gossip_before_reuse() -> Result<()> {
    let requester = make_test_node(super::super::NodeRole::Worker).await?;
    let remote = make_test_node(super::super::NodeRole::Worker).await?;
    let trusted_signer = test_release_signer_key_id(9);
    let policy = requirement_policy(&trusted_signer);

    configure_requirement_node(&requester, &policy, Some(&trusted_signer)).await?;
    configure_requirement_node(&remote, &policy, None).await?;
    requester.start_accepting();
    remote.start_accepting();

    let cached_conn = connect_mesh(
        &requester.endpoint,
        remote.endpoint_addr_for_advertisement(),
    )
    .await?;
    {
        let mut state = requester.state.lock().await;
        let mut peer = make_test_peer(remote.id(), Some(20), 8);
        peer.addr = remote.endpoint_addr_for_advertisement();
        peer.admitted = false;
        state.peers.insert(remote.id(), peer);
        state.connections.insert(remote.id(), cached_conn);
    }

    let error = requester
        .connection_to_peer(remote.id())
        .await
        .expect_err("cached unadmitted connection must complete gossip before reuse");

    assert!(
        error.to_string().contains("Failed to complete gossip"),
        "unexpected error: {error:#}"
    );
    assert!(
        !requester
            .state
            .lock()
            .await
            .connections
            .contains_key(&remote.id()),
        "failed cached admission must remove the unusable connection"
    );
    assert!(
        !requester
            .state
            .lock()
            .await
            .pending_connections
            .contains_key(&remote.id()),
        "failed cached admission must clear the pending handshake"
    );

    Ok(())
}

#[tokio::test]
async fn pending_connection_handshake_single_flights_waiters() -> Result<()> {
    let node = make_test_node(super::super::NodeRole::Worker).await?;
    let peer_id = make_test_endpoint_id(0xa1);

    let owner = match node.reserve_pending_connection(peer_id).await {
        PendingConnectionReservation::Owner(owner) => owner,
        PendingConnectionReservation::Waiter(_) => {
            anyhow::bail!("first pending reservation should own the handshake")
        }
    };
    assert!(
        node.discovered_peer_already_known(peer_id, true).await,
        "connection-based discovery must treat pending handshakes as known"
    );
    assert!(
        node.discovered_peer_already_known(peer_id, false).await,
        "peer-based discovery must also avoid redialing pending handshakes"
    );

    let waiter = match node.reserve_pending_connection(peer_id).await {
        PendingConnectionReservation::Owner(_) => {
            anyhow::bail!("second pending reservation must wait on the owner")
        }
        PendingConnectionReservation::Waiter(waiter) => waiter,
    };
    let waiter_node = node.clone();
    let waiter_task =
        tokio::spawn(async move { waiter_node.await_pending_connection(waiter).await });

    node.finish_pending_connection(owner, PendingConnectionOutcome::Admitted)
        .await;
    waiter_task.await??;
    assert!(
        !node
            .state
            .lock()
            .await
            .pending_connections
            .contains_key(&peer_id),
        "completed handshakes must leave no pending slot behind"
    );

    Ok(())
}

#[tokio::test]
async fn pending_connection_failure_cleanup_is_owner_scoped() -> Result<()> {
    let node = make_test_node(super::super::NodeRole::Worker).await?;
    let peer_id = make_test_endpoint_id(0xa2);

    let stale_owner = match node.reserve_pending_connection(peer_id).await {
        PendingConnectionReservation::Owner(owner) => owner,
        PendingConnectionReservation::Waiter(_) => {
            anyhow::bail!("first pending reservation should own the handshake")
        }
    };
    let current_attempt_id = PendingConnectionAttemptId(stale_owner.attempt_id.0 + 1);
    let (current_tx, current_rx) = watch::channel(None);
    {
        let mut state = node.state.lock().await;
        state.pending_connections.insert(
            peer_id,
            PendingConnectionHandshake {
                attempt_id: current_attempt_id,
                outcome_rx: current_rx,
            },
        );
    }

    node.finish_pending_connection(
        stale_owner,
        PendingConnectionOutcome::Failed("stale attempt failed".to_string()),
    )
    .await;
    assert_eq!(
        node.state
            .lock()
            .await
            .pending_connections
            .get(&peer_id)
            .map(|pending| pending.attempt_id),
        Some(current_attempt_id),
        "stale owners must not remove a newer pending handshake"
    );

    node.finish_pending_connection(
        PendingConnectionAttemptOwner {
            peer_id,
            attempt_id: current_attempt_id,
            outcome_tx: current_tx,
        },
        PendingConnectionOutcome::Failed("current attempt failed".to_string()),
    )
    .await;
    assert!(
        !node
            .state
            .lock()
            .await
            .pending_connections
            .contains_key(&peer_id),
        "the current owner must remove its own failed pending handshake"
    );

    Ok(())
}

#[tokio::test]
async fn recovered_connection_waiter_failure_removes_stale_peer() -> Result<()> {
    // Given: recovery lost the pending-connection race for a tracked peer.
    let node = make_test_node(super::super::NodeRole::Worker).await?;
    let peer_id = make_test_endpoint_id(0xa4);
    node.state
        .lock()
        .await
        .peers
        .insert(peer_id, make_test_peer(peer_id, Some(20), 8));
    let owner = match node.reserve_pending_connection(peer_id).await {
        PendingConnectionReservation::Owner(owner) => owner,
        PendingConnectionReservation::Waiter(_) => {
            anyhow::bail!("first pending reservation should own the handshake")
        }
    };
    let waiter = match node.reserve_pending_connection(peer_id).await {
        PendingConnectionReservation::Owner(_) => {
            anyhow::bail!("recovered connection should wait on the active handshake")
        }
        PendingConnectionReservation::Waiter(waiter) => waiter,
    };
    node.finish_pending_connection(
        owner,
        PendingConnectionOutcome::Failed("owner gossip failed".to_string()),
    )
    .await;

    // When: the recovered-connection waiter observes the failed owner.
    node.complete_recovered_connection_waiter(waiter).await;

    // Then: the peer whose dispatcher already exited is not left routable.
    assert!(!node.state.lock().await.peers.contains_key(&peer_id));
    Ok(())
}

#[tokio::test]
async fn dropped_pending_connection_owner_is_pruned() -> Result<()> {
    let node = make_test_node(super::super::NodeRole::Worker).await?;
    let peer_id = make_test_endpoint_id(0xa3);

    let owner = match node.reserve_pending_connection(peer_id).await {
        PendingConnectionReservation::Owner(owner) => owner,
        PendingConnectionReservation::Waiter(_) => {
            anyhow::bail!("first pending reservation should own the handshake")
        }
    };
    let waiter = match node.reserve_pending_connection(peer_id).await {
        PendingConnectionReservation::Owner(_) => {
            anyhow::bail!("second pending reservation must wait on the owner")
        }
        PendingConnectionReservation::Waiter(waiter) => waiter,
    };
    drop(owner);

    let error = node
        .await_pending_connection(waiter)
        .await
        .expect_err("dropped owners must fail waiters instead of hanging");
    assert!(
        error
            .to_string()
            .contains("ended without a terminal result"),
        "unexpected error: {error:#}"
    );
    assert!(
        !node.discovered_peer_already_known(peer_id, true).await,
        "stale pending slots must not suppress rediscovery"
    );

    let owner = match node.reserve_pending_connection(peer_id).await {
        PendingConnectionReservation::Owner(owner) => owner,
        PendingConnectionReservation::Waiter(_) => {
            anyhow::bail!("stale pending slot should be pruned before the next reservation")
        }
    };
    node.finish_pending_connection(
        owner,
        PendingConnectionOutcome::Failed("test cleanup".to_string()),
    )
    .await;

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn connect_to_peer_direct_announcement_during_gossip_does_not_redial() -> Result<()> {
    let requester = make_test_node(super::super::NodeRole::Worker).await?;
    let remote = make_test_node(super::super::NodeRole::Worker).await?;
    let trusted_signer = test_release_signer_key_id(9);
    let policy = requirement_policy(&trusted_signer);

    configure_requirement_node(&requester, &policy, Some(&trusted_signer)).await?;
    configure_requirement_node(&remote, &policy, Some(&trusted_signer)).await?;
    requester.start_accepting();
    remote.start_accepting();

    tokio::time::timeout(
        std::time::Duration::from_secs(3),
        requester.connect_to_peer(remote.endpoint_addr_for_advertisement()),
    )
    .await
    .expect("connect_to_peer must not recurse while processing direct gossip announcement")?;

    let state = requester.state.lock().await;
    assert!(
        state.connections.contains_key(&remote.id()),
        "successful gossip must leave the direct connection cached"
    );
    assert!(
        state
            .peers
            .get(&remote.id())
            .is_some_and(super::super::PeerInfo::is_admitted),
        "successful direct gossip must admit the remote peer"
    );
    assert!(
        !state.pending_connections.contains_key(&remote.id()),
        "successful direct gossip must clear the pending handshake"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn recovered_connection_gossip_failure_does_not_publish_unadmitted_connection() -> Result<()>
{
    let requester = make_test_node(super::super::NodeRole::Worker).await?;
    let remote = make_test_node(super::super::NodeRole::Worker).await?;
    let trusted_signer = test_release_signer_key_id(9);
    let policy = requirement_policy(&trusted_signer);

    configure_requirement_node(&requester, &policy, Some(&trusted_signer)).await?;
    configure_requirement_node(&remote, &policy, None).await?;
    remote.start_accepting();
    {
        let mut peer = make_test_peer(remote.id(), Some(20), 8);
        peer.addr = remote.endpoint_addr_for_advertisement();
        requester.state.lock().await.peers.insert(remote.id(), peer);
    }

    let recovered = connect_mesh(
        &requester.endpoint,
        remote.endpoint_addr_for_advertisement(),
    )
    .await?;
    requester
        .complete_recovered_connection(remote.id(), recovered.clone())
        .await;

    let state = requester.state.lock().await;
    assert!(
        !state.connections.contains_key(&remote.id()),
        "failed recovered gossip must not publish a reusable connection"
    );
    assert!(
        !state.pending_connections.contains_key(&remote.id()),
        "failed recovered gossip must clear its pending handshake"
    );
    assert!(
        !state.peers.contains_key(&remote.id()),
        "failed recovered gossip must remove the zombie peer"
    );
    drop(state);

    let closed = tokio::time::timeout(std::time::Duration::from_secs(2), recovered.closed()).await;
    assert!(
        closed.is_ok(),
        "failed recovered gossip must close its own QUIC connection"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn connection_failure_cleanup_removes_only_matching_connection() -> Result<()> {
    let requester = make_test_node(super::super::NodeRole::Worker).await?;
    let remote = make_test_node(super::super::NodeRole::Worker).await?;
    remote.start_accepting();

    let retained = connect_mesh(
        &requester.endpoint,
        remote.endpoint_addr_for_advertisement(),
    )
    .await?;
    let retained_id = retained.stable_id();
    let stale = connect_mesh(
        &requester.endpoint,
        remote.endpoint_addr_for_advertisement(),
    )
    .await?;

    {
        let mut state = requester.state.lock().await;
        state.connections.insert(remote.id(), retained);
    }

    let removed = requester
        .remove_connection_if_stable_id(remote.id(), &stale)
        .await;
    stale.close(0u32.into(), b"test-stale-cleanup");
    assert!(
        removed.is_none(),
        "cleanup must not remove a connection with a different stable id"
    );
    assert_eq!(
        requester
            .state
            .lock()
            .await
            .connections
            .get(&remote.id())
            .map(|conn| conn.stable_id()),
        Some(retained_id),
        "a raced replacement connection must remain cached"
    );

    let retained = requester
        .state
        .lock()
        .await
        .connections
        .get(&remote.id())
        .cloned()
        .expect("retained connection should still be cached");
    let removed = requester
        .remove_connection_if_stable_id(remote.id(), &retained)
        .await;
    assert!(
        removed.is_some(),
        "cleanup must remove the matching failed connection"
    );

    Ok(())
}

#[test]
fn legacy_config_stream_ids_are_reserved_and_require_admission() {
    assert!(
        !stream_allowed_before_admission(STREAM_CONFIG_SUBSCRIBE, TrustPolicy::Off),
        "reserved STREAM_CONFIG_SUBSCRIBE (0x0b) must not bypass admission"
    );
    assert!(
        !stream_allowed_before_admission(STREAM_CONFIG_PUSH, TrustPolicy::Off),
        "reserved STREAM_CONFIG_PUSH (0x0c) must not bypass admission"
    );
}
