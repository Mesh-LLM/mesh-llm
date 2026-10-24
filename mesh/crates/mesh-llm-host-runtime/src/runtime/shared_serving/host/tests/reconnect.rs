use super::*;

#[tokio::test]
async fn reconnect_does_not_wait_for_an_unresponsive_transitive_peer() {
    let fixture = Fixture::new().await;
    fixture.host.leave("alpha").await.unwrap();
    // A previously healthy host can disappear without a departure message.
    // Keep its endpoint bound, but never accept gossip, so a secondary dial
    // cannot finish within the assertion's deadline.
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
    let reconnected =
        tokio::time::timeout(Duration::from_secs(5), fixture.host.reconnect("alpha")).await;
    let beta = fixture.request(1, "beta").await;
    ghost.close_endpoint().await;
    fixture.close().await;
    reconnected
        .expect("admitting the seed must not await secondary peer discovery")
        .unwrap();
    assert!(beta.starts_with("HTTP/1.1 200"));
}
