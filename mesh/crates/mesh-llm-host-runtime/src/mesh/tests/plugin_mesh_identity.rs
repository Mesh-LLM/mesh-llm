// Inbound plugin-frame sender identity and signed relay coverage (#2214).
// These tests drive real nodes over QUIC and verify that unsigned source claims
// stay bound to the immediate peer while valid origin signatures survive relays.

use crate::plugin::proto::{BulkTransferMessage, ChannelMessage, MeshBulkFrame, MeshChannelFrame};

/// Poll `predicate` until it holds, failing with `context` once the bound is
/// exhausted. No fixed sleep stands in for the observed event.
async fn wait_until<F, Fut>(context: &str, mut predicate: F)
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = bool>,
{
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while std::time::Instant::now() < deadline {
        if predicate().await {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
    panic!("timed out waiting for: {context}");
}

/// Start a real mesh node on loopback with no relays.
async fn start_plugin_identity_node() -> (Node, TunnelChannels) {
    let relay_urls = Vec::new();
    let relay_auths = HashMap::new();
    Node::start(
        super::NodeRole::Client,
        RelayConfig {
            urls: &relay_urls,
            auths: &relay_auths,
            policy: RelayPolicy::Disabled,
        },
        QuicBindSelection {
            ip: Some(std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST)),
            port: Some(0),
        },
        Some(0.0),
        false,
        None,
        None,
        crate::MeshRequirements::unrestricted(),
    )
    .await
    .expect("start plugin identity test node")
}

/// Connect two real nodes so the receiver's `remote` is a genuinely
/// authenticated QUIC sender rather than a value the test hands the handler.
async fn connect_plugin_identity_pair() -> (Node, Node) {
    let (sender, _sender_channels) = start_plugin_identity_node().await;
    let (receiver, _receiver_channels) = start_plugin_identity_node().await;
    sender.start_accepting();
    receiver.start_accepting();
    sender
        .connect_to_peer(receiver.endpoint_addr_for_advertisement())
        .await
        .expect("connect plugin identity test nodes");
    let receiver_id = receiver.id();
    wait_until(
        "the sender holds an admitted connection to the receiver",
        || async { sender.connection_to_peer(receiver_id).await.is_ok() },
    )
    .await;
    (sender, receiver)
}

/// Connect a relay to a forwarder and a forwarder to a final recipient.
async fn connect_plugin_identity_chain() -> (Node, Node, Node) {
    let (relay, _relay_channels) = start_plugin_identity_node().await;
    let (forwarder, _forwarder_channels) = start_plugin_identity_node().await;
    let (receiver, _receiver_channels) = start_plugin_identity_node().await;
    relay.start_accepting();
    forwarder.start_accepting();
    receiver.start_accepting();

    relay
        .connect_to_peer(forwarder.endpoint_addr_for_advertisement())
        .await
        .expect("connect relay to plugin forwarder");
    forwarder
        .connect_to_peer(receiver.endpoint_addr_for_advertisement())
        .await
        .expect("connect plugin forwarder to receiver");
    wait_until(
        "the relay and forwarder hold an admitted connection",
        || async { relay.connection_to_peer(forwarder.id()).await.is_ok() },
    )
    .await;
    wait_until(
        "the forwarder and receiver hold an admitted connection",
        || async { forwarder.connection_to_peer(receiver.id()).await.is_ok() },
    )
    .await;
    (relay, forwarder, receiver)
}

fn peer_hex(node: &Node) -> String {
    hex::encode(node.id().as_bytes())
}

/// Write one plugin frame to `receiver_id` on `stream_type` exactly as a remote
/// peer puts it on the wire: stream-type byte, little-endian length, protobuf.
async fn write_plugin_frame(sender: &Node, receiver_id: EndpointId, stream_type: u8, frame: &[u8]) {
    let conn = sender
        .connection_to_peer(receiver_id)
        .await
        .expect("authenticated connection to receiver");
    let (mut send, mut recv) = conn.open_bi().await.expect("open plugin stream");
    tokio::io::AsyncWriteExt::write_all(&mut send, &[stream_type])
        .await
        .expect("write stream type");
    tokio::io::AsyncWriteExt::write_all(&mut send, &(frame.len() as u32).to_le_bytes())
        .await
        .expect("write frame length");
    tokio::io::AsyncWriteExt::write_all(&mut send, frame)
        .await
        .expect("write frame body");
    let _ = send.finish();
    let mut drain = [0u8; 1];
    let _ = tokio::io::AsyncReadExt::read(&mut recv, &mut drain).await;
}

/// A channel frame whose claimed source is a peer other than the one that sent
/// it must never be accepted for delivery to a local plugin.
#[tokio::test]
async fn forged_plugin_channel_source_is_not_dispatched() {
    let (attacker, victim) = connect_plugin_identity_pair().await;
    let victim_id = victim.id();
    // An identity neither node owns: the peer the attacker is impersonating.
    let claimed = EndpointId::from(SecretKey::from_bytes(&[0x22; 32]).public());
    let claimed_hex = hex::encode(claimed.as_bytes());
    let message_id = "forged-channel";

    let frame = MeshChannelFrame {
        plugin_id: "trusted-plugin".to_string(),
        message_id: message_id.to_string(),
        message: Some(ChannelMessage {
            channel: "authority".to_string(),
            source_peer_id: claimed_hex.clone(),
            target_peer_id: String::new(),
            content_type: "application/json".to_string(),
            body: br#"{"grant":"admin"}"#.to_vec(),
            ..Default::default()
        }),
        ..Default::default()
    }
    .encode_to_vec();

    write_plugin_frame(&attacker, victim_id, STREAM_PLUGIN_CHANNEL, &frame).await;

    // The frame really arrived and really was recognised as a forged claim, so
    // the only question left is what the receiver did next.
    wait_until(
        "the receiver counts the channel source mismatch",
        || async { victim.plugin_frame_source_mismatch_total() >= 1 },
    )
    .await;

    assert!(
        !victim.has_plugin_message(message_id).await,
        "#2214: a plugin channel frame claiming source_peer_id {claimed_hex} while sent by {} was still accepted for local plugin dispatch; the claim is unauthenticated and must be rejected before delivery",
        peer_hex(&attacker),
    );
}

/// The bulk-transfer receive path applies the same binding.
#[tokio::test]
async fn forged_plugin_bulk_source_is_not_dispatched() {
    let (attacker, victim) = connect_plugin_identity_pair().await;
    let victim_id = victim.id();
    let claimed = EndpointId::from(SecretKey::from_bytes(&[0x33; 32]).public());
    let claimed_hex = hex::encode(claimed.as_bytes());
    let message_id = "forged-bulk";

    let frame = MeshBulkFrame {
        plugin_id: "trusted-plugin".to_string(),
        message_id: message_id.to_string(),
        message: Some(BulkTransferMessage {
            kind: crate::plugin::proto::bulk_transfer_message::Kind::Offer as i32,
            transfer_id: "transfer-1".to_string(),
            channel: "authority".to_string(),
            source_peer_id: claimed_hex.clone(),
            target_peer_id: String::new(),
            content_type: "application/octet-stream".to_string(),
            total_bytes: 4,
            body: vec![1, 2, 3, 4],
            ..Default::default()
        }),
        ..Default::default()
    }
    .encode_to_vec();

    write_plugin_frame(&attacker, victim_id, STREAM_PLUGIN_BULK_TRANSFER, &frame).await;

    wait_until("the receiver counts the bulk source mismatch", || async {
        victim.plugin_frame_source_mismatch_total() >= 1
    })
    .await;

    assert!(
        !victim.has_plugin_message(message_id).await,
        "#2214: a plugin bulk frame claiming source_peer_id {claimed_hex} while sent by {} was still accepted for local bulk dispatch",
        peer_hex(&attacker),
    );
}

/// A frame claiming the identity of the peer that actually sent it is still
/// accepted for delivery and is not counted as a mismatch.
#[tokio::test]
async fn matching_plugin_channel_source_is_dispatched() {
    let (sender, receiver) = connect_plugin_identity_pair().await;
    let receiver_id = receiver.id();
    let message_id = "honest-channel";

    let frame = MeshChannelFrame {
        plugin_id: "trusted-plugin".to_string(),
        message_id: message_id.to_string(),
        message: Some(ChannelMessage {
            channel: "authority".to_string(),
            source_peer_id: peer_hex(&sender),
            target_peer_id: String::new(),
            content_type: "application/json".to_string(),
            body: b"{}".to_vec(),
            ..Default::default()
        }),
        ..Default::default()
    }
    .encode_to_vec();

    write_plugin_frame(&sender, receiver_id, STREAM_PLUGIN_CHANNEL, &frame).await;

    wait_until("the honest frame is accepted for delivery", || async {
        receiver.has_plugin_message(message_id).await
    })
    .await;

    assert_eq!(
        receiver.plugin_frame_source_mismatch_total(),
        0,
        "a frame claiming the identity of the peer that actually sent it must not be treated as a mismatch"
    );
}

/// Empty-source legacy frames can cross a relay, with each hop supplying its
/// own authenticated identity when it dispatches the message.
#[tokio::test]
async fn empty_source_plugin_frames_continue_through_a_relay() {
    let (relay, forwarder, receiver) = connect_plugin_identity_chain().await;
    let target_peer_id = peer_hex(&receiver);
    let channel_message_id = "empty-source-relayed-channel";
    let channel_frame = MeshChannelFrame {
        plugin_id: "trusted-plugin".to_string(),
        message_id: channel_message_id.to_string(),
        message: Some(ChannelMessage {
            channel: "authority".to_string(),
            source_peer_id: String::new(),
            target_peer_id: target_peer_id.clone(),
            content_type: "application/json".to_string(),
            body: b"{}".to_vec(),
            ..Default::default()
        }),
        ..Default::default()
    }
    .encode_to_vec();

    write_plugin_frame(
        &relay,
        forwarder.id(),
        STREAM_PLUGIN_CHANNEL,
        &channel_frame,
    )
    .await;
    wait_until(
        "the empty-source channel frame reaches its target",
        || async { receiver.has_plugin_message(channel_message_id).await },
    )
    .await;
    assert!(forwarder.has_plugin_message(channel_message_id).await);

    let bulk_message_id = "empty-source-relayed-bulk";
    let bulk_frame = MeshBulkFrame {
        plugin_id: "trusted-plugin".to_string(),
        message_id: bulk_message_id.to_string(),
        message: Some(BulkTransferMessage {
            kind: crate::plugin::proto::bulk_transfer_message::Kind::Offer as i32,
            transfer_id: "transfer-empty-source".to_string(),
            channel: "authority".to_string(),
            source_peer_id: String::new(),
            target_peer_id,
            content_type: "application/octet-stream".to_string(),
            total_bytes: 4,
            body: vec![1, 2, 3, 4],
            ..Default::default()
        }),
        ..Default::default()
    }
    .encode_to_vec();

    write_plugin_frame(
        &relay,
        forwarder.id(),
        STREAM_PLUGIN_BULK_TRANSFER,
        &bulk_frame,
    )
    .await;
    wait_until("the empty-source bulk frame reaches its target", || async {
        receiver.has_plugin_message(bulk_message_id).await
    })
    .await;
    assert!(forwarder.has_plugin_message(bulk_message_id).await);
    assert_eq!(forwarder.plugin_frame_source_mismatch_total(), 0);
    assert_eq!(receiver.plugin_frame_source_mismatch_total(), 0);
}

/// A relay cannot change the authenticated origin, but it may route a frame
/// whose signature covers the complete message envelope.
#[tokio::test]
async fn valid_origin_signed_channel_frame_is_accepted_through_a_relay() {
    let (relay, forwarder, receiver) = connect_plugin_identity_chain().await;
    let origin = SecretKey::generate();
    let message_id = "signed-relayed-channel";
    let mut frame = MeshChannelFrame {
        plugin_id: "trusted-plugin".to_string(),
        message_id: message_id.to_string(),
        message: Some(ChannelMessage {
            channel: "authority".to_string(),
            source_peer_id: hex::encode(origin.public().as_bytes()),
            target_peer_id: peer_hex(&receiver),
            content_type: "application/json".to_string(),
            body: b"{\"grant\":\"read\"}".to_vec(),
            ..Default::default()
        }),
        ..Default::default()
    };
    super::plugin_mesh::sign_channel_frame_with_expiry(
        &origin,
        &mut frame,
        crate::mesh::current_time_unix_ms() + 30_000,
    );

    write_plugin_frame(
        &relay,
        forwarder.id(),
        STREAM_PLUGIN_CHANNEL,
        &frame.encode_to_vec(),
    )
    .await;

    wait_until("the forwarder accepts the signed origin frame", || async {
        forwarder.has_plugin_message(message_id).await
    })
    .await;
    wait_until(
        "the signed frame reaches its target through the relay",
        || async { receiver.has_plugin_message(message_id).await },
    )
    .await;
    assert_eq!(forwarder.plugin_frame_source_mismatch_total(), 0);
    assert_eq!(receiver.plugin_frame_source_mismatch_total(), 0);
}

/// Changing any signed field invalidates a relayed frame before deduplication.
#[tokio::test]
async fn altered_origin_signed_channel_frame_is_rejected() {
    let (relay, receiver) = connect_plugin_identity_pair().await;
    let receiver_id = receiver.id();
    let origin = SecretKey::generate();
    let message_id = "altered-signed-channel";
    let mut frame = MeshChannelFrame {
        plugin_id: "trusted-plugin".to_string(),
        message_id: message_id.to_string(),
        message: Some(ChannelMessage {
            channel: "authority".to_string(),
            source_peer_id: hex::encode(origin.public().as_bytes()),
            target_peer_id: String::new(),
            content_type: "application/json".to_string(),
            body: b"{\"grant\":\"read\"}".to_vec(),
            ..Default::default()
        }),
        ..Default::default()
    };
    super::plugin_mesh::sign_channel_frame_with_expiry(
        &origin,
        &mut frame,
        crate::mesh::current_time_unix_ms() + 30_000,
    );
    frame.message.as_mut().expect("channel message").body = b"{\"grant\":\"admin\"}".to_vec();

    write_plugin_frame(
        &relay,
        receiver_id,
        STREAM_PLUGIN_CHANNEL,
        &frame.encode_to_vec(),
    )
    .await;

    wait_until("the altered origin claim is rejected", || async {
        receiver.plugin_frame_source_mismatch_total() >= 1
    })
    .await;
    assert!(!receiver.has_plugin_message(message_id).await);
}

/// Expired signatures are rejected even when their cryptographic bytes match.
#[tokio::test]
async fn expired_origin_signed_bulk_frame_is_rejected() {
    let (relay, receiver) = connect_plugin_identity_pair().await;
    let receiver_id = receiver.id();
    let origin = SecretKey::generate();
    let message_id = "expired-signed-bulk";
    let mut frame = MeshBulkFrame {
        plugin_id: "trusted-plugin".to_string(),
        message_id: message_id.to_string(),
        message: Some(BulkTransferMessage {
            kind: crate::plugin::proto::bulk_transfer_message::Kind::Offer as i32,
            transfer_id: "transfer-expired".to_string(),
            channel: "authority".to_string(),
            source_peer_id: hex::encode(origin.public().as_bytes()),
            target_peer_id: String::new(),
            content_type: "application/octet-stream".to_string(),
            total_bytes: 4,
            body: vec![1, 2, 3, 4],
            ..Default::default()
        }),
        ..Default::default()
    };
    super::plugin_mesh::sign_bulk_frame_with_expiry(
        &origin,
        &mut frame,
        crate::mesh::current_time_unix_ms().saturating_sub(1),
    );

    write_plugin_frame(
        &relay,
        receiver_id,
        STREAM_PLUGIN_BULK_TRANSFER,
        &frame.encode_to_vec(),
    )
    .await;

    wait_until("the expired origin signature is rejected", || async {
        receiver.plugin_frame_source_mismatch_total() >= 1
    })
    .await;
    assert!(!receiver.has_plugin_message(message_id).await);
}

/// Bulk-transfer frames use the same signed origin identity as channel frames.
#[tokio::test]
async fn valid_origin_signed_bulk_frame_is_accepted_through_a_relay() {
    let (relay, forwarder, receiver) = connect_plugin_identity_chain().await;
    let origin = SecretKey::generate();
    let message_id = "signed-relayed-bulk";
    let mut frame = MeshBulkFrame {
        plugin_id: "trusted-plugin".to_string(),
        message_id: message_id.to_string(),
        message: Some(BulkTransferMessage {
            kind: crate::plugin::proto::bulk_transfer_message::Kind::Offer as i32,
            transfer_id: "transfer-signed".to_string(),
            channel: "authority".to_string(),
            source_peer_id: hex::encode(origin.public().as_bytes()),
            target_peer_id: peer_hex(&receiver),
            content_type: "application/octet-stream".to_string(),
            total_bytes: 4,
            body: vec![1, 2, 3, 4],
            ..Default::default()
        }),
        ..Default::default()
    };
    super::plugin_mesh::sign_bulk_frame_with_expiry(
        &origin,
        &mut frame,
        crate::mesh::current_time_unix_ms() + 30_000,
    );

    write_plugin_frame(
        &relay,
        forwarder.id(),
        STREAM_PLUGIN_BULK_TRANSFER,
        &frame.encode_to_vec(),
    )
    .await;

    wait_until("the forwarder accepts the signed bulk origin", || async {
        forwarder.has_plugin_message(message_id).await
    })
    .await;
    wait_until(
        "the signed bulk frame reaches its target through the relay",
        || async { receiver.has_plugin_message(message_id).await },
    )
    .await;
    assert_eq!(forwarder.plugin_frame_source_mismatch_total(), 0);
    assert_eq!(receiver.plugin_frame_source_mismatch_total(), 0);
}
