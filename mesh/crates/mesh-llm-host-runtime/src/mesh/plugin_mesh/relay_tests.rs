use super::*;

async fn connect_recipient(node: &Node, supported: bool) -> (Node, Connection) {
    let remote = Node::new_for_tests(NodeRole::Client).await.unwrap();
    let accept = remote.endpoint.clone();
    let accepted = tokio::spawn(async move { accept.accept().await.unwrap().await.unwrap() });
    let connection = node
        .endpoint
        .connect(remote.endpoint.addr(), ALPN)
        .await
        .unwrap();
    let accepted = accepted.await.unwrap();
    let announcement =
        remote.build_local_announcement(remote.snapshot_local_announcement_data().await);
    let mut peer = PeerInfo::from_announcement(
        remote.id(),
        remote.endpoint.addr(),
        &announcement,
        Default::default(),
    );
    peer.admitted = true;
    peer.plugin_origin_signature_supported = supported;
    let mut state = node.state.lock().await;
    state.peers.insert(remote.id(), peer);
    state.connections.insert(remote.id(), connection);
    (remote, accepted)
}

async fn receive_frame(connection: Connection) -> (u8, Vec<u8>) {
    let (_send, mut recv) = connection.accept_bi().await.unwrap();
    let mut kind = [0];
    recv.read_exact(&mut kind).await.unwrap();
    let mut length = [0; 4];
    recv.read_exact(&mut length).await.unwrap();
    let mut bytes = vec![0; u32::from_le_bytes(length) as usize];
    recv.read_exact(&mut bytes).await.unwrap();
    (kind[0], bytes)
}

#[tokio::test]
async fn signed_broadcasts_select_capable_relays_and_deliver_direct_targets() {
    let (node, secret) = Node::new_for_tests_with_secret(NodeRole::Client)
        .await
        .unwrap();
    let (capable, capable_connection) = connect_recipient(&node, true).await;
    let (legacy, legacy_connection) = connect_recipient(&node, false).await;
    let recipients = node.plugin_frame_connections("", true, None).await;
    assert_eq!(recipients.len(), 2);
    assert!(recipients.iter().any(|(id, _)| *id == legacy.id()));
    assert!(
        node.plugin_frame_connections("", true, Some(capable.id()))
            .await
            .is_empty()
    );
    assert_eq!(
        node.plugin_frame_connections("", false, None).await.len(),
        2
    );

    let mut channel = crate::plugin::proto::MeshChannelFrame {
        plugin_id: "test".into(),
        message_id: "signed-channel".into(),
        message: Some(crate::plugin::proto::ChannelMessage {
            source_peer_id: endpoint_id_hex(node.id()),
            ..Default::default()
        }),
        ..Default::default()
    };
    sign_channel_frame(&secret, &mut channel);
    let received = tokio::spawn(receive_frame(capable_connection.clone()));
    let legacy_received = tokio::spawn(receive_frame(legacy_connection.clone()));
    node.broadcast_plugin_channel_frame(&channel, None)
        .await
        .unwrap();
    let (kind, bytes) = tokio::time::timeout(std::time::Duration::from_secs(5), received)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(kind, STREAM_PLUGIN_CHANNEL);
    assert_eq!(bytes, channel.encode_to_vec());
    let (kind, bytes) = tokio::time::timeout(std::time::Duration::from_secs(5), legacy_received)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(kind, STREAM_PLUGIN_CHANNEL);
    assert_eq!(bytes, channel.encode_to_vec());
    // A legacy relay loses unknown signature fields on decode/re-encode.
    // Its claimed origin is not its authenticated immediate-peer identity.
    let mut stripped = channel.clone();
    stripped.origin_signature.clear();
    stripped.expires_at_unix_ms = 0;
    assert!(authenticate_plugin_frame(&stripped, legacy.id()).is_none());
    assert!(matches!(
        authenticate_plugin_frame(&stripped, node.id()),
        Some(PluginFrameAuthentication::ImmediatePeer)
    ));

    let mut bulk = crate::plugin::proto::MeshBulkFrame {
        plugin_id: "test".into(),
        message_id: "signed-bulk".into(),
        message: Some(crate::plugin::proto::BulkTransferMessage {
            source_peer_id: endpoint_id_hex(node.id()),
            target_peer_id: endpoint_id_hex(legacy.id()),
            ..Default::default()
        }),
        ..Default::default()
    };
    sign_bulk_frame(&secret, &mut bulk);
    let received = tokio::spawn(receive_frame(legacy_connection.clone()));
    node.broadcast_plugin_bulk_frame(&bulk, Some(capable.id()))
        .await
        .unwrap();
    let (kind, bytes) = tokio::time::timeout(std::time::Duration::from_secs(5), received)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(kind, STREAM_PLUGIN_BULK_TRANSFER);
    assert_eq!(bytes, bulk.encode_to_vec());
    assert!(matches!(
        authenticate_plugin_frame(&bulk, node.id()),
        Some(PluginFrameAuthentication::OriginSignature)
    ));

    let received = tokio::spawn(receive_frame(legacy_connection));
    node.forward_targeted_plugin_frame(
        &endpoint_id_hex(legacy.id()),
        STREAM_PLUGIN_BULK_TRANSFER,
        bulk.encode_to_vec(),
    )
    .await;
    let (_, bytes) = tokio::time::timeout(std::time::Duration::from_secs(5), received)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(bytes, bulk.encode_to_vec());
}
