use super::super::direct_path::endpoint_addr_with_previously_advertised_direct_candidates;
use super::super::heartbeat::{direct_only_addr, selected_connection_is_direct};
use super::{
    PendingConnectionOutcome, PendingConnectionReservation, configure_requirement_node,
    connect_mesh, make_test_endpoint_id, make_test_node, requirement_policy,
    test_release_signer_key_id,
};
use iroh::{EndpointAddr, TransportAddr};

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
fn direct_rescue_address_keeps_only_ip_candidates() {
    let peer_id = make_test_endpoint_id(34);
    let direct = TransportAddr::Ip("10.0.0.7:47916".parse().unwrap());
    let relay = TransportAddr::Relay("https://relay.example.com".parse().unwrap());
    let mut mixed = EndpointAddr::from_parts(peer_id, [direct.clone(), relay.clone()]);

    let filtered = direct_only_addr(mixed.clone()).expect("mixed address has a direct candidate");
    assert_eq!(filtered.id, peer_id);
    assert_eq!(filtered.addrs.len(), 1);
    assert!(filtered.addrs.contains(&direct));
    assert!(!filtered.addrs.contains(&relay));

    mixed
        .addrs
        .retain(|candidate| matches!(candidate, TransportAddr::Relay(_)));
    assert!(direct_only_addr(mixed).is_none());
}

#[test]
fn direct_rescue_uses_same_identity_and_stays_alive_with_installed_connection() -> anyhow::Result<()>
{
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?
        .block_on(async {
            let node = make_test_node(super::super::NodeRole::Worker).await?;
            let remote = make_test_node(super::super::NodeRole::Worker).await?;
            remote.start_accepting();

            let existing =
                connect_mesh(&node.endpoint, remote.endpoint_addr_for_advertisement()).await?;
            let existing_id = existing.stable_id();
            {
                let mut state = node.state.lock().await;
                state.connections.insert(remote.id(), existing.clone());
                state
                    .peers
                    .insert(remote.id(), super::make_test_peer_info(remote.id()));
            }

            let (rescue_endpoint, replacement) = node
                .dial_direct_rescue_connection(
                    remote.id(),
                    remote.endpoint_addr_for_advertisement(),
                )
                .await
                .expect("loopback direct rescue should connect");
            assert_eq!(rescue_endpoint.id(), node.id());
            assert!(selected_connection_is_direct(&replacement));
            assert!(
                node.refreshed_connection_completed_gossip(remote.id(), &replacement)
                    .await
            );
            let replacement_id = replacement.stable_id();
            assert!(
                node.install_refreshed_peer_connection(
                    remote.id(),
                    existing_id,
                    replacement.clone(),
                    Some(rescue_endpoint),
                )
                .await
            );

            {
                let retained_endpoints = node.direct_rescue_endpoints.lock().await;
                let retained = retained_endpoints
                    .get(&remote.id())
                    .expect("installed rescue endpoint must be retained");
                assert_eq!(retained.connection_stable_id, replacement_id);
                assert!(!retained.endpoint.is_closed());
            }

            // The stale dispatcher for the replaced connection must not release
            // the endpoint that owns the newer direct connection.
            node.release_direct_rescue_endpoint(remote.id(), existing_id)
                .await;
            assert_eq!(
                node.direct_rescue_endpoints
                    .lock()
                    .await
                    .get(&remote.id())
                    .map(|retained| retained.connection_stable_id),
                Some(replacement_id)
            );

            existing.close(0u32.into(), b"test old relay retirement");
            assert!(replacement.close_reason().is_none());
            replacement.close(0u32.into(), b"test complete");
            node.release_direct_rescue_endpoint(remote.id(), replacement_id)
                .await;
            assert!(node.direct_rescue_endpoints.lock().await.is_empty());

            node.close_endpoint().await;
            remote.close_endpoint().await;
            Ok(())
        })
}

#[test]
fn direct_rescue_failure_keeps_existing_connection_tracked() -> anyhow::Result<()> {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?
        .block_on(async {
            let node = make_test_node(super::super::NodeRole::Worker).await?;
            let remote = make_test_node(super::super::NodeRole::Worker).await?;
            remote.start_accepting();
            let existing =
                connect_mesh(&node.endpoint, remote.endpoint_addr_for_advertisement()).await?;
            let existing_id = existing.stable_id();
            node.state
                .lock()
                .await
                .connections
                .insert(remote.id(), existing);

            let unusable = EndpointAddr::from_parts(
                remote.id(),
                [TransportAddr::Ip("127.0.0.1:9".parse().unwrap())],
            );
            assert!(
                node.dial_direct_rescue_connection(remote.id(), unusable)
                    .await
                    .is_none()
            );
            assert_eq!(
                node.state
                    .lock()
                    .await
                    .connections
                    .get(&remote.id())
                    .map(|conn| conn.stable_id()),
                Some(existing_id)
            );
            assert!(node.direct_rescue_endpoints.lock().await.is_empty());

            node.close_endpoint().await;
            remote.close_endpoint().await;
            Ok(())
        })
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
fn direct_path_reverse_dial_keeps_replaced_connection_open_for_inflight_streams()
-> anyhow::Result<()> {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?
        .block_on(async {
            let node = make_test_node(super::super::NodeRole::Worker).await?;
            let remote = make_test_node(super::super::NodeRole::Worker).await?;
            remote.start_accepting();

            let existing =
                connect_mesh(&node.endpoint, remote.endpoint_addr_for_advertisement()).await?;
            let existing_id = existing.stable_id();
            {
                let mut state = node.state.lock().await;
                state.connections.insert(remote.id(), existing.clone());
                state
                    .peers
                    .insert(remote.id(), super::make_test_peer_info(remote.id()));
            }

            let replacement =
                connect_mesh(&node.endpoint, remote.endpoint_addr_for_advertisement()).await?;
            let replacement_id = replacement.stable_id();
            node.install_direct_path_request_connection(remote.id(), replacement)
                .await;

            let tracked_id = node
                .state
                .lock()
                .await
                .connections
                .get(&remote.id())
                .expect("replacement connection should be tracked")
                .stable_id();
            assert_eq!(tracked_id, replacement_id);
            assert_ne!(tracked_id, existing_id);
            assert!(
                tokio::time::timeout(std::time::Duration::from_millis(100), existing.closed())
                    .await
                    .is_err(),
                "replaced connection must remain open so existing streams can drain"
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
