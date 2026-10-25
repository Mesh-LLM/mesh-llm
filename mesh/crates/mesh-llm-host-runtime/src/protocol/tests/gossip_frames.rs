use super::*;

/// When the encoded gossip frame exceeds the control-frame ceiling, the
/// writer prunes application descriptors from relayed entries only: the
/// sender's own entry keeps them (SDK consumers discover the local node's
/// protocols from it) and built-in subprotocols survive everywhere because
/// capability flags derive from them.
#[test]
fn prune_relayed_application_subprotocols_keeps_own_entry_and_built_ins() {
    let sender_id = EndpointId::from(SecretKey::from_bytes(&[0x5E; 32]).public());
    let relayed_id = EndpointId::from(SecretKey::from_bytes(&[0x5F; 32]).public());
    let descriptor = |name: &str, major: u32| crate::proto::node::MeshSubprotocol {
        name: name.into(),
        major,
        features: vec![],
    };
    let peer = |id: EndpointId| crate::proto::node::PeerAnnouncement {
        endpoint_id: id.as_bytes().to_vec(),
        subprotocols: vec![
            descriptor(
                skippy_protocol::STAGE_SUBPROTOCOL_NAME,
                skippy_protocol::STAGE_SUBPROTOCOL_MAJOR,
            ),
            descriptor("echo", 1),
        ],
        ..Default::default()
    };
    let mut frame = crate::proto::node::GossipFrame {
        r#gen: NODE_PROTOCOL_GENERATION,
        sender_id: sender_id.as_bytes().to_vec(),
        peers: vec![peer(sender_id), peer(relayed_id)],
        ..Default::default()
    };

    prune_relayed_application_subprotocols(&mut frame);

    assert_eq!(
        frame.peers[0].subprotocols.len(),
        2,
        "the sender's own entry keeps its application descriptors"
    );
    assert!(
        frame.peers[0]
            .subprotocols
            .iter()
            .any(|subprotocol| subprotocol.name == "echo")
    );
    assert_eq!(
        frame.peers[1].subprotocols.len(),
        1,
        "relayed entries keep only built-in subprotocols"
    );
    assert_eq!(
        frame.peers[1].subprotocols[0].name,
        skippy_protocol::STAGE_SUBPROTOCOL_NAME
    );
}
