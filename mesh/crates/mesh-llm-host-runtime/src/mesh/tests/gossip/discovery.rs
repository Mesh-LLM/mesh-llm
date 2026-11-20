#[tokio::test]
pub(crate) async fn future_demand_timestamps_are_active_without_underflow() {
    let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();
    let now = now_secs();
    node.merge_remote_demand(&HashMap::from([(
        "future-demand".to_string(),
        ModelDemand {
            last_active: now + DEMAND_TTL_SECS + 60,
            request_count: 3,
        },
    )]));

    let active = node.active_demand().await;
    assert!(active.contains_key("future-demand"));

    node.gc_demand().await;
    assert!(node.get_demand().contains_key("future-demand"));
}

#[tokio::test]
async fn join_returns_while_discovered_peer_is_still_handshaking() {
    let joiner = Node::new_for_tests(NodeRole::Worker).await.unwrap();
    let seed = Node::new_for_tests(NodeRole::Worker).await.unwrap();
    let secondary_role = NodeRole::Host { http_port: 9337 };
    let secondary = Node::new_for_tests(secondary_role.clone()).await.unwrap();
    let model = "secondary-model";
    secondary.set_serving_models(vec![model.into()]).await;
    secondary.set_hosted_models(vec![model.into()]).await;
    let secondary_addr = secondary.endpoint_addr_for_advertisement();
    let mut announcement = test_announcement(None);
    announcement.addr = secondary_addr.clone();
    announcement.role = secondary_role;
    announcement.serving_models = vec![model.into()];
    seed.update_transitive_peer(secondary.id(), &secondary_addr, &announcement, seed.id())
        .await;

    let accepting = seed.clone();
    let listener = tokio::spawn(async move { accepting.accept_loop().await });
    seed.start_accepting();
    let token = encode_endpoint_addr_token(&seed.endpoint_addr_for_advertisement());

    // Accept the follow-on dial and keep its gossip stream open without replying.
    let pending_handshake = async {
        let conn = secondary.endpoint.accept().await.unwrap().await.unwrap();
        let (send, mut recv) = conn.accept_bi().await.unwrap();
        let mut stream_type = [0];
        recv.read_exact(&mut stream_type).await.unwrap();
        assert_eq!(stream_type, [STREAM_GOSSIP]);
        (conn, send, recv)
    };
    let (result, (conn, send, recv)) =
        tokio::time::timeout(std::time::Duration::from_secs(20), async {
            tokio::join!(joiner.join(&token), pending_handshake)
        })
        .await
        .expect("bootstrap join must finish while secondary gossip is pending");
    result.unwrap();

    {
        let state = joiner.state.lock().await;
        assert!(state.connections.contains_key(&seed.id()));
        assert!(state.peers.get(&seed.id()).unwrap().is_admitted());
        assert!(state.peers.contains_key(&secondary.id()));
        assert!(!state.connections.contains_key(&secondary.id()));
    }
    assert!(joiner.hosts_for_model(model).await.is_empty());

    let mesh_llm_membership::connection_reservation::PendingConnectionReservation::Waiter(waiter) =
        joiner.reserve_pending_connection(secondary.id()).await
    else {
        panic!("secondary admission must still be in progress");
    };
    tokio::time::timeout(std::time::Duration::from_secs(20), async {
        secondary
            .handle_gossip_stream(
                joiner.id(),
                crate::protocol::connection_protocol(&conn),
                send,
                recv,
            )
            .await
            .unwrap();
        joiner.await_pending_connection(waiter).await.unwrap();
    })
    .await
    .expect("secondary must finish admission after its gossip reply");
    assert_eq!(joiner.hosts_for_model(model).await, vec![secondary.id()]);

    joiner.close_endpoint().await;
    seed.close_endpoint().await;
    secondary.close_endpoint().await;
    listener.await.unwrap();
}

#[tokio::test]
pub(crate) async fn client_auto_join_probe_returns_none_for_single_candidate() {
    let node = Node::new_for_tests(NodeRole::Client).await.unwrap();
    let token = encode_endpoint_addr_token(&test_addr(0x42));

    let selected = node
        .join_first_responsive_candidate(&[(token, Some("single".to_string()))])
        .await
        .unwrap();

    assert!(selected.is_none());
}

#[tokio::test]
pub(crate) async fn client_auto_join_probe_candidate_collection_filters_unusable_tokens() {
    let node = Node::new_for_tests(NodeRole::Client).await.unwrap();
    let valid_addr = test_addr(0x42);
    let dead_addr = test_addr(0x43);
    let self_token = encode_endpoint_addr_token(&node.endpoint_addr_for_advertisement());
    let dead_token = encode_endpoint_addr_token(&dead_addr);
    let valid_token = encode_endpoint_addr_token(&valid_addr);

    node.state
        .lock()
        .await
        .dead_peers
        .insert(dead_addr.id, std::time::Instant::now());

    let candidates = node
        .collect_join_probe_candidates(&[
            ("not-an-invite-token".to_string(), None),
            (self_token, None),
            (dead_token, None),
            (valid_token, Some("usable".to_string())),
        ])
        .await;

    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].addr.id, valid_addr.id);
    assert_eq!(candidates[0].mesh_name.as_deref(), Some("usable"));
}
