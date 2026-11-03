use super::*;

const ORIGIN_COUNT: u8 = 90;

#[tokio::test]
async fn oversized_snapshots_complete_real_quic_exchanges_in_both_directions() {
    let sender = Node::new_for_tests(NodeRole::Worker).await.unwrap();
    let receiver = Node::new_for_tests(NodeRole::Worker).await.unwrap();
    let body = crate::proto::node::NodeRecord {
        role: crate::proto::node::NodeRole::Worker as i32,
        version: crate::VERSION.to_string(),
        serving_models: (0..250)
            .map(|index| format!("{index:03}-{}", "model".repeat(80)))
            .collect(),
        ..Default::default()
    }
    .encode_to_vec();
    let records = (1..=ORIGIN_COUNT)
        .map(|seed| {
            let record = VerifiedRecord::sign(
                NODE_RECORD,
                &signing_key(seed),
                1,
                current_time_unix_ms(),
                &body,
            );
            crate::proto::node::SignedNodeRecord {
                signed: record.signed_bytes().to_vec(),
                signature: record.signature().to_vec(),
                hop: None,
            }
        })
        .collect();
    receive(&sender, 0xf0, Vec::new(), records).await;
    let outbound = sender.collect_outbound_gossip().await;
    assert!(
        outbound
            .signed_records
            .iter()
            .map(prost::Message::encoded_len)
            .sum::<usize>()
            > 8 * 1024 * 1024
    );
    // Unit-test nodes do not start background accept loops automatically.
    let mut listeners = Vec::new();
    for node in [&sender, &receiver] {
        let accepting = node.clone();
        listeners.push(tokio::spawn(async move { accepting.accept_loop().await }));
        node.start_accepting();
    }
    // Test endpoints bind loopback explicitly; use that socket rather than
    // waiting for iroh's asynchronously discovered advertisement addresses.
    let socket = receiver
        .endpoint
        .bound_sockets()
        .into_iter()
        .find(|addr| addr.is_ipv4())
        .unwrap();
    assert!(socket.ip().is_loopback());
    let address = EndpointAddr::new(receiver.id()).with_ip_addr(socket);
    let connection = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        sender.endpoint.connect(address, crate::protocol::ALPN_V1),
    )
    .await
    .unwrap()
    .unwrap();
    // Compatibility announcements expand model descriptions, so the frame's
    // capacity cannot be inferred from the signed bytes alone. Require every
    // exchange to make progress; at least one new origin per exchange bounds
    // the round count without assuming a particular encoded record size.
    let mut progress = Vec::new();
    for _ in 0..ORIGIN_COUNT {
        tokio::time::timeout(
            std::time::Duration::from_secs(10),
            sender.initiate_gossip_inner(connection.clone(), receiver.id(), false),
        )
        .await
        .unwrap()
        .unwrap();
        let state = receiver.state.lock().await;
        let received = (1..=ORIGIN_COUNT)
            .filter(|seed| state.node_records.contains(&test_endpoint_id(*seed)))
            .count();
        assert!(
            received > progress.last().copied().unwrap_or(0),
            "every exchange must advance delivery; previous counts: {progress:?}, received: {received}"
        );
        progress.push(received);
        if received == usize::from(ORIGIN_COUNT) {
            break;
        }
    }
    let state = receiver.state.lock().await;
    for seed in 1..=ORIGIN_COUNT {
        assert!(
            state.node_records.contains(&test_endpoint_id(seed)),
            "origin {seed} must eventually arrive; received origin counts: {progress:?}"
        );
    }
    assert!(
        state.peers.contains_key(&sender.id()),
        "inbound handler must apply the direct sender"
    );
    drop(state);
    assert!(sender.state.lock().await.peers.contains_key(&receiver.id()));
    sender.close_endpoint().await;
    receiver.close_endpoint().await;
    for listener in listeners {
        tokio::time::timeout(std::time::Duration::from_secs(2), listener)
            .await
            .unwrap()
            .unwrap();
    }
}
