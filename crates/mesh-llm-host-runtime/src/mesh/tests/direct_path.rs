use super::super::direct_path::{
    DIRECT_PATH_REPAIR_COOLDOWN_SECS, DIRECT_PATH_REPAIR_GRACE_SECS,
    DirectPathMaintenanceController, DirectPathObservation, DirectPathRepairReason,
    endpoint_addr_with_previously_advertised_direct_candidates,
};
use super::super::heartbeat::{RelayPathSnapshot, SelectedPathKind};
use super::{
    PendingConnectionOutcome, PendingConnectionReservation, configure_requirement_node,
    connect_mesh, make_test_endpoint_id, make_test_node, requirement_policy,
    test_release_signer_key_id,
};
use iroh::{EndpointAddr, TransportAddr};

#[test]
fn direct_path_maintenance_requires_candidate_and_grace_period() {
    let now = std::time::Instant::now();
    let peer = make_test_endpoint_id(31);
    let mut controller = DirectPathMaintenanceController::default();
    let relay_observation = DirectPathObservation {
        peer_id: peer,
        snapshot: RelayPathSnapshot {
            kind: SelectedPathKind::Relay,
            rtt_ms: Some(200),
        },
        has_direct_candidate: true,
    };

    assert_eq!(
        controller.plan_request([relay_observation], now, 0),
        None,
        "first non-direct observation starts the grace timer"
    );
    assert_eq!(
        controller.plan_request(
            [relay_observation],
            now + std::time::Duration::from_secs(DIRECT_PATH_REPAIR_GRACE_SECS + 2),
            0,
        ),
        Some((peer, DirectPathRepairReason::RelaySelected))
    );

    let mut no_candidate_controller = DirectPathMaintenanceController::default();
    assert_eq!(
        no_candidate_controller.plan_request(
            [DirectPathObservation {
                has_direct_candidate: false,
                ..relay_observation
            }],
            now + std::time::Duration::from_secs(DIRECT_PATH_REPAIR_GRACE_SECS + 1),
            0,
        ),
        None,
        "without a direct candidate there is nothing useful to request"
    );
}

#[test]
fn direct_path_maintenance_cooldown_and_inflight_suppress_requests() {
    let now = std::time::Instant::now();
    let peer = make_test_endpoint_id(32);
    let mut controller = DirectPathMaintenanceController::default();
    let observation = DirectPathObservation {
        peer_id: peer,
        snapshot: RelayPathSnapshot {
            kind: SelectedPathKind::Unknown,
            rtt_ms: None,
        },
        has_direct_candidate: true,
    };

    assert_eq!(controller.plan_request([observation], now, 1), None);
    assert!(
        controller
            .peer_health(peer)
            .and_then(|health| health.non_direct_since)
            .is_some(),
        "active requests suppress repair but still record path state"
    );

    let ready_at = now + std::time::Duration::from_secs(DIRECT_PATH_REPAIR_GRACE_SECS + 1);
    assert_eq!(
        controller.plan_request([observation], ready_at, 0),
        Some((peer, DirectPathRepairReason::UnknownSelected))
    );
    controller.record_request_attempt(peer, ready_at);
    assert_eq!(
        controller.plan_request(
            [observation],
            ready_at + std::time::Duration::from_secs(DIRECT_PATH_REPAIR_COOLDOWN_SECS - 1),
            0,
        ),
        None,
        "cooldown prevents repeated reverse-dial requests"
    );
}

#[test]
fn direct_path_request_keeps_only_previously_advertised_direct_candidates() {
    let peer_id = make_test_endpoint_id(33);
    let advertised_direct = TransportAddr::Ip("10.0.0.7:47916".parse().unwrap());
    let unadvertised_direct = TransportAddr::Ip("10.0.0.99:47916".parse().unwrap());
    let advertised_relay = TransportAddr::Relay("https://relay.example.com".parse().unwrap());

    let mut advertised = EndpointAddr {
        id: peer_id,
        addrs: Default::default(),
    };
    advertised.addrs.insert(advertised_direct.clone());
    advertised.addrs.insert(advertised_relay.clone());

    let mut requested = EndpointAddr {
        id: peer_id,
        addrs: Default::default(),
    };
    requested.addrs.insert(advertised_direct.clone());
    requested.addrs.insert(unadvertised_direct.clone());
    requested.addrs.insert(advertised_relay.clone());

    let filtered =
        endpoint_addr_with_previously_advertised_direct_candidates(requested, &advertised)
            .expect("the previously advertised direct candidate should be kept");
    assert!(filtered.addrs.contains(&advertised_direct));
    assert!(!filtered.addrs.contains(&unadvertised_direct));
    assert!(!filtered.addrs.contains(&advertised_relay));

    let mut unknown_only = EndpointAddr {
        id: peer_id,
        addrs: Default::default(),
    };
    unknown_only.addrs.insert(unadvertised_direct);
    assert!(
        endpoint_addr_with_previously_advertised_direct_candidates(unknown_only, &advertised)
            .is_none(),
        "requests with only unknown direct candidates must not trigger reverse dials"
    );
}

#[test]
fn direct_path_reverse_dial_keeps_existing_connection_when_gossip_fails() -> anyhow::Result<()> {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?
        .block_on(async {
            let node = make_test_node(super::super::NodeRole::Worker).await?;
            let remote = make_test_node(super::super::NodeRole::Worker).await?;
            remote.start_accepting();

            let existing = connect_mesh(&node.endpoint, remote.endpoint_addr_for_advertisement()).await?;
            let existing_id = existing.stable_id();
            {
                let mut state = node.state.lock().await;
                state.connections.insert(remote.id(), existing);
                state.peers.insert(
                    remote.id(),
                    super::make_test_peer_info(remote.id()),
                );
            }

            let trusted_signer = test_release_signer_key_id(9);
            let policy = requirement_policy(&trusted_signer);
            configure_requirement_node(&node, &policy, Some(&trusted_signer)).await?;
            configure_requirement_node(&remote, &policy, None).await?;
            let replacement = connect_mesh(&node.endpoint, remote.endpoint_addr_for_advertisement()).await?;

            node.install_direct_path_request_connection(remote.id(), replacement)
                .await;

            let retained_id = node
                .state
                .lock()
                .await
                .connections
                .get(&remote.id())
                .expect("failed reverse dial gossip must retain the old connection")
                .stable_id();
            assert_eq!(
                retained_id, existing_id,
                "direct-path replacement must not overwrite the old connection unless gossip succeeds"
            );

            Ok(())
        })
}

#[test]
fn direct_path_reverse_dial_does_not_publish_during_pending_handshake() -> anyhow::Result<()> {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?
        .block_on(async {
            let node = make_test_node(super::super::NodeRole::Worker).await?;
            let remote = make_test_node(super::super::NodeRole::Worker).await?;
            remote.start_accepting();

            let owner = match node.reserve_pending_connection(remote.id()).await {
                PendingConnectionReservation::Owner(owner) => owner,
                PendingConnectionReservation::Waiter(_) => {
                    anyhow::bail!("first pending reservation should own the handshake")
                }
            };
            let replacement =
                connect_mesh(&node.endpoint, remote.endpoint_addr_for_advertisement()).await?;

            node.install_direct_path_request_connection(remote.id(), replacement.clone())
                .await;

            let state = node.state.lock().await;
            assert!(
                !state.connections.contains_key(&remote.id()),
                "reverse dial must not publish a connection while another handshake is pending"
            );
            assert!(
                state.pending_connections.contains_key(&remote.id()),
                "reverse dial must not clean up another attempt's pending handshake"
            );
            drop(state);

            let closed =
                tokio::time::timeout(std::time::Duration::from_secs(2), replacement.closed()).await;
            assert!(
                closed.is_ok(),
                "reverse dial raced against pending admission must close its own connection"
            );
            node.finish_pending_connection(
                owner,
                PendingConnectionOutcome::Failed("test cleanup".to_string()),
            )
            .await;

            Ok(())
        })
}
