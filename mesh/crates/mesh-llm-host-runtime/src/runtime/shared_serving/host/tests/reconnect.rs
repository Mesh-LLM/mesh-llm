use super::*;

#[test]
/// A silent secondary peer must not block seed admission or the other mesh's inference.
fn reconnect_does_not_wait_for_an_unresponsive_transitive_peer() {
    std::thread::Builder::new()
        .name("reconnect-test".to_owned())
        .stack_size(8 * 1024 * 1024)
        .spawn(|| {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("build reconnect test runtime")
                .block_on(reconnect_with_pending_secondary_exchange());
        })
        .expect("spawn reconnect test thread")
        .join()
        .expect("reconnect test thread panicked");
}

async fn reconnect_with_pending_secondary_exchange() {
    // Advertise the secondary before alpha's first admission; a prior connection
    // can admit an inbound dial-back and bypass outbound discovery on reconnect.
    let fixture = Fixture::detached().await;
    fixture.host.reconnect("beta").await.unwrap();
    // A previously healthy host can disappear without a departure message.
    // Accept its gossip request but hold the response until admission completes.
    let ghost = Node::new_for_tests(NodeRole::Host { http_port: 1 })
        .await
        .unwrap();
    ghost.set_mesh_id_force("shared-test-alpha".into()).await;
    let mut announcement =
        ghost.build_local_announcement(ghost.snapshot_local_announcement_data().await);
    let port = ghost
        .endpoint
        .addr()
        .ip_addrs()
        .find(|addr| addr.is_ipv4())
        .unwrap()
        .port();
    let address = iroh::EndpointAddr::new(ghost.id())
        .with_ip_addr(std::net::SocketAddr::from(([127, 0, 0, 1], port)));
    announcement.addr = address.clone();
    let mut peer = crate::mesh::PeerInfo::from_announcement(
        ghost.id(),
        address,
        &announcement,
        Default::default(),
    );
    peer.admitted = true;
    fixture.seeds[0].insert_test_peer(peer).await;
    tokio::time::timeout(Duration::from_secs(120), async {
        let accept_gossip = async {
            let connection = ghost.endpoint.accept().await.unwrap().await.unwrap();
            let (send, mut recv) = connection.accept_bi().await.unwrap();
            let mut stream_type = [0];
            recv.read_exact(&mut stream_type).await.unwrap();
            assert_eq!(stream_type[0], crate::protocol::STREAM_GOSSIP);
            (connection, send, recv)
        };
        let (reconnected, (connection, send, recv)) =
            tokio::join!(fixture.host.reconnect("alpha"), accept_gossip);
        reconnected.expect("admitting the seed must not await secondary peer discovery");
        // A discovery timeout must not masquerade as admission independence.
        assert!(connection.close_reason().is_none());
        assert!(futures_util::poll!(std::pin::pin!(send.stopped())).is_pending());
        drop((send, recv));
        connection.close(0u32.into(), b"test exchange released");
    })
    .await
    .expect("reconnect and secondary gossip contact must complete within the hang guard");
    let beta = fixture.request(1, "beta").await;
    ghost.close_endpoint().await;
    fixture.close().await;
    assert!(beta.starts_with("HTTP/1.1 200"));
}
