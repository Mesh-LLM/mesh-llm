use super::*;

#[test]
fn protocol_from_alpn_defaults_to_v1() {
    assert_eq!(protocol_from_alpn(ALPN_V1), ControlProtocol::ProtoV1);
    assert_eq!(
        protocol_from_alpn(b"mesh-llm/999"),
        ControlProtocol::ProtoV1
    );
}
#[test]
fn control_frame_roundtrip() {
    let frame = make_valid_gossip_frame();
    let encoded = encode_control_frame(STREAM_GOSSIP, &frame);
    let decoded: GossipFrame = decode_control_frame(STREAM_GOSSIP, &encoded)
        .expect("valid gossip frame must decode successfully");
    assert_eq!(decoded.r#gen, NODE_PROTOCOL_GENERATION);
    assert_eq!(decoded.peers.len(), 1);
    assert_eq!(decoded.peers[0].endpoint_id, vec![0u8; 32]);
    assert_eq!(decoded.peers[0].role, NodeRole::Worker as i32);
}

#[test]
fn mesh_subprotocol_open_roundtrips_and_validates() {
    let open = MeshSubprotocolOpen {
        r#gen: NODE_PROTOCOL_GENERATION,
        name: skippy_protocol::STAGE_SUBPROTOCOL_NAME.to_string(),
        major: skippy_protocol::STAGE_SUBPROTOCOL_MAJOR,
    };
    let encoded = encode_control_frame(STREAM_SUBPROTOCOL, &open);
    let decoded: MeshSubprotocolOpen = decode_control_frame(STREAM_SUBPROTOCOL, &encoded).unwrap();
    assert_eq!(decoded.name, skippy_protocol::STAGE_SUBPROTOCOL_NAME);
    assert_eq!(decoded.major, skippy_protocol::STAGE_SUBPROTOCOL_MAJOR);

    let bad = MeshSubprotocolOpen {
        r#gen: NODE_PROTOCOL_GENERATION,
        name: String::new(),
        major: skippy_protocol::STAGE_SUBPROTOCOL_MAJOR,
    };
    let encoded = encode_control_frame(STREAM_SUBPROTOCOL, &bad);
    let err = decode_control_frame::<MeshSubprotocolOpen>(STREAM_SUBPROTOCOL, &encoded)
        .expect_err("empty subprotocol names must be rejected");
    assert!(matches!(err, ControlFrameError::InvalidSubprotocol));
}

#[test]
fn proto_v1_route_table_rejects_bad_generation_or_legacy_payload() {
    use crate::proto::node::RouteTable;

    let zero_gen_req = RouteTableRequest {
        requester_id: vec![0u8; 32],
        r#gen: 0,
    };
    let encoded = encode_control_frame(STREAM_ROUTE_REQUEST, &zero_gen_req);
    let err = decode_control_frame::<RouteTableRequest>(STREAM_ROUTE_REQUEST, &encoded)
        .expect_err("request gen=0 must be rejected");
    assert!(
        matches!(err, ControlFrameError::BadGeneration { got: 0 }),
        "expected BadGeneration{{got:0}}, got {:?}",
        err
    );

    let wrong_gen_req = RouteTableRequest {
        requester_id: vec![0u8; 32],
        r#gen: 99,
    };
    let encoded = encode_control_frame(STREAM_ROUTE_REQUEST, &wrong_gen_req);
    let err = decode_control_frame::<RouteTableRequest>(STREAM_ROUTE_REQUEST, &encoded)
        .expect_err("request gen=99 must be rejected");
    assert!(
        matches!(err, ControlFrameError::BadGeneration { got: 99 }),
        "expected BadGeneration{{got:99}}, got {:?}",
        err
    );

    let bad_gen_response = RouteTable {
        entries: vec![],
        mesh_id: None,
        r#gen: 0,
    };
    let encoded = encode_control_frame(STREAM_ROUTE_REQUEST, &bad_gen_response);
    let err = decode_control_frame::<RouteTable>(STREAM_ROUTE_REQUEST, &encoded)
        .expect_err("response gen=0 must be rejected");
    assert!(
        matches!(err, ControlFrameError::BadGeneration { got: 0 }),
        "expected BadGeneration{{got:0}} for response, got {:?}",
        err
    );

    let wrong_gen_response = RouteTable {
        entries: vec![],
        mesh_id: None,
        r#gen: 42,
    };
    let encoded = encode_control_frame(STREAM_ROUTE_REQUEST, &wrong_gen_response);
    let err = decode_control_frame::<RouteTable>(STREAM_ROUTE_REQUEST, &encoded)
        .expect_err("response gen=42 must be rejected");
    assert!(
        matches!(err, ControlFrameError::BadGeneration { got: 42 }),
        "expected BadGeneration{{got:42}} for response, got {:?}",
        err
    );

    let legacy_json = b"{\"hosts\":[],\"mesh_id\":null}";
    let mut fake_frame = vec![STREAM_ROUTE_REQUEST];
    fake_frame.extend_from_slice(&(legacy_json.len() as u32).to_le_bytes());
    fake_frame.extend_from_slice(legacy_json);
    let err = decode_control_frame::<RouteTableRequest>(STREAM_ROUTE_REQUEST, &fake_frame)
        .expect_err("legacy JSON payload must be rejected");
    assert!(
        matches!(err, ControlFrameError::DecodeError(_)),
        "expected DecodeError for JSON payload, got {:?}",
        err
    );
}

#[test]
fn peer_lifecycle_messages_roundtrip() {
    use crate::proto::node::{PeerDown, PeerLeaving};

    let leaving_id = EndpointId::from(SecretKey::from_bytes(&[0x55; 32]).public());

    let mut peers: HashMap<EndpointId, PeerInfo> = HashMap::new();
    peers.insert(leaving_id, make_test_peer_info(leaving_id));
    let mut connection_ids: HashSet<EndpointId> = HashSet::new();
    connection_ids.insert(leaving_id);

    let leaving_msg = PeerLeaving {
        peer_id: leaving_id.as_bytes().to_vec(),
        r#gen: NODE_PROTOCOL_GENERATION,
    };
    let encoded = encode_control_frame(STREAM_PEER_LEAVING, &leaving_msg);
    let decoded_leaving: PeerLeaving =
        decode_control_frame(STREAM_PEER_LEAVING, &encoded).expect("valid PeerLeaving must decode");

    let accepted_id = resolve_peer_leaving(leaving_id, &decoded_leaving)
        .expect("PeerLeaving from sender itself must be accepted");

    peers.remove(&accepted_id);
    connection_ids.remove(&accepted_id);

    assert!(
        !peers.contains_key(&leaving_id),
        "leaving peer must be removed from peers after accepted PeerLeaving"
    );
    assert!(
        !connection_ids.contains(&leaving_id),
        "leaving peer must be removed from connections after accepted PeerLeaving"
    );

    let self_id = EndpointId::from(SecretKey::from_bytes(&[0xAA; 32]).public());
    let dead_id = EndpointId::from(SecretKey::from_bytes(&[0xBB; 32]).public());

    let mut peers: HashMap<EndpointId, PeerInfo> = HashMap::new();
    peers.insert(dead_id, make_test_peer_info(dead_id));
    let mut connection_ids: HashSet<EndpointId> = HashSet::new();
    connection_ids.insert(dead_id);

    let down_msg = PeerDown {
        peer_id: dead_id.as_bytes().to_vec(),
        r#gen: NODE_PROTOCOL_GENERATION,
    };
    let encoded = encode_control_frame(STREAM_PEER_DOWN, &down_msg);
    let decoded_down: PeerDown =
        decode_control_frame(STREAM_PEER_DOWN, &encoded).expect("valid PeerDown must decode");

    let result = resolve_peer_down(self_id, dead_id, true);
    assert_eq!(
        result,
        Some(dead_id),
        "confirmed-unreachable peer must be returned for removal"
    );

    if let Some(id) = result {
        peers.remove(&id);
        connection_ids.remove(&id);
    }

    assert!(
        !peers.contains_key(&dead_id),
        "dead peer must be removed from peers when confirmed unreachable"
    );
    assert!(
        !connection_ids.contains(&dead_id),
        "dead peer must be removed from connections when confirmed unreachable"
    );

    assert_eq!(decoded_down.r#gen, NODE_PROTOCOL_GENERATION);
}

#[test]
fn peer_lifecycle_rejects_forged_sender_or_unverified_down() {
    use crate::proto::node::{PeerDown, PeerLeaving};

    let valid_peer_bytes = EndpointId::from(SecretKey::from_bytes(&[0x77; 32]).public())
        .as_bytes()
        .to_vec();

    let bad_gen_down = PeerDown {
        peer_id: valid_peer_bytes.clone(),
        r#gen: 0,
    };
    let encoded = encode_control_frame(STREAM_PEER_DOWN, &bad_gen_down);
    let err = decode_control_frame::<PeerDown>(STREAM_PEER_DOWN, &encoded)
        .expect_err("PeerDown gen=0 must be rejected");
    assert!(
        matches!(err, ControlFrameError::BadGeneration { got: 0 }),
        "expected BadGeneration{{got:0}} for PeerDown, got {:?}",
        err
    );

    let bad_gen_leaving = PeerLeaving {
        peer_id: valid_peer_bytes.clone(),
        r#gen: 0,
    };
    let encoded = encode_control_frame(STREAM_PEER_LEAVING, &bad_gen_leaving);
    let err = decode_control_frame::<PeerLeaving>(STREAM_PEER_LEAVING, &encoded)
        .expect_err("PeerLeaving gen=0 must be rejected");
    assert!(
        matches!(err, ControlFrameError::BadGeneration { got: 0 }),
        "expected BadGeneration{{got:0}} for PeerLeaving, got {:?}",
        err
    );

    let remote_id = EndpointId::from(SecretKey::from_bytes(&[0x11; 32]).public());
    let victim_id = EndpointId::from(SecretKey::from_bytes(&[0x22; 32]).public());

    let mut peers: HashMap<EndpointId, PeerInfo> = HashMap::new();
    peers.insert(victim_id, make_test_peer_info(victim_id));

    let forged = PeerLeaving {
        peer_id: victim_id.as_bytes().to_vec(),
        r#gen: NODE_PROTOCOL_GENERATION,
    };
    let encoded = encode_control_frame(STREAM_PEER_LEAVING, &forged);
    let decoded: PeerLeaving = decode_control_frame(STREAM_PEER_LEAVING, &encoded)
        .expect("structurally valid PeerLeaving must decode");

    let err = resolve_peer_leaving(remote_id, &decoded)
        .expect_err("forged PeerLeaving (peer_id != remote) must be rejected");
    assert!(
        matches!(err, crate::protocol::ControlFrameError::ForgedSender),
        "expected ForgedSender, got {:?}",
        err
    );

    assert!(
        peers.contains_key(&victim_id),
        "victim peer must NOT be removed when PeerLeaving is forged"
    );

    let self_id = EndpointId::from(SecretKey::from_bytes(&[0x33; 32]).public());
    let still_alive_id = EndpointId::from(SecretKey::from_bytes(&[0x44; 32]).public());

    let mut peers: HashMap<EndpointId, PeerInfo> = HashMap::new();
    peers.insert(still_alive_id, make_test_peer_info(still_alive_id));

    let result = resolve_peer_down(self_id, still_alive_id, false);
    assert!(
        result.is_none(),
        "PeerDown must not trigger removal when peer is still reachable"
    );

    assert!(
        peers.contains_key(&still_alive_id),
        "reachable peer must NOT be removed after PeerDown with should_remove=false"
    );
}

#[test]
fn proto_v1_control_frames_reject_legacy_json_and_wrong_gen() {
    use crate::proto::node::{PeerDown, PeerLeaving};

    // JSON bytes that look plausible for the old wire format on each stream
    let json_gossip = b"[{\"addr\":{\"id\":\"aabbcc\",\"addrs\":[]}}]";
    let json_tunnel_map = b"{\"owner\":\"aabbcc\",\"entries\":[]}";
    let json_route = b"{\"hosts\":[],\"mesh_id\":null}";
    let json_peer_down = b"\"aabbccdd\"";
    let json_peer_leaving = b"\"aabbccdd\"";

    // All migrated streams must reject legacy JSON with DecodeError
    for (stream_type, json_bytes) in [
        (STREAM_GOSSIP, json_gossip.as_slice()),
        (STREAM_TUNNEL_MAP, json_tunnel_map.as_slice()),
        (STREAM_ROUTE_REQUEST, json_route.as_slice()),
        (STREAM_PEER_DOWN, json_peer_down.as_slice()),
        (STREAM_PEER_LEAVING, json_peer_leaving.as_slice()),
    ] {
        let mut frame = vec![stream_type];
        frame.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
        frame.extend_from_slice(json_bytes);
        let err = match stream_type {
            STREAM_GOSSIP => decode_control_frame::<GossipFrame>(stream_type, &frame).unwrap_err(),
            STREAM_TUNNEL_MAP => {
                decode_control_frame::<crate::proto::node::TunnelMap>(stream_type, &frame)
                    .unwrap_err()
            }
            STREAM_ROUTE_REQUEST => {
                decode_control_frame::<RouteTableRequest>(stream_type, &frame).unwrap_err()
            }
            STREAM_PEER_DOWN => decode_control_frame::<PeerDown>(stream_type, &frame).unwrap_err(),
            STREAM_PEER_LEAVING => {
                decode_control_frame::<PeerLeaving>(stream_type, &frame).unwrap_err()
            }
            _ => unreachable!("all stream types in this table are migrated control frames"),
        };
        assert!(
            matches!(err, ControlFrameError::DecodeError(_)),
            "stream {:#04x}: expected DecodeError for JSON, got {:?}",
            stream_type,
            err
        );
    }

    // All migrated streams must also reject gen=0 and gen=99 where gen is checked
    let bad_gen_gossip = GossipFrame {
        r#gen: 0,
        sender_id: vec![],
        peers: vec![PeerAnnouncement {
            endpoint_id: vec![0u8; 32],
            role: NodeRole::Worker as i32,
            ..Default::default()
        }],
    };
    let encoded = encode_control_frame(STREAM_GOSSIP, &bad_gen_gossip);
    let err = decode_control_frame::<GossipFrame>(STREAM_GOSSIP, &encoded)
        .expect_err("GossipFrame gen=0 must be rejected");
    assert!(matches!(err, ControlFrameError::BadGeneration { got: 0 }));

    let bad_gen_req = RouteTableRequest {
        requester_id: vec![0u8; 32],
        r#gen: 0,
    };
    let encoded = encode_control_frame(STREAM_ROUTE_REQUEST, &bad_gen_req);
    let err = decode_control_frame::<RouteTableRequest>(STREAM_ROUTE_REQUEST, &encoded)
        .expect_err("RouteTableRequest gen=0 must be rejected");
    assert!(matches!(err, ControlFrameError::BadGeneration { got: 0 }));

    let bad_gen_down = PeerDown {
        peer_id: vec![0u8; 32],
        r#gen: 0,
    };
    let encoded = encode_control_frame(STREAM_PEER_DOWN, &bad_gen_down);
    let err = decode_control_frame::<PeerDown>(STREAM_PEER_DOWN, &encoded)
        .expect_err("PeerDown gen=0 must be rejected");
    assert!(matches!(err, ControlFrameError::BadGeneration { got: 0 }));

    let bad_gen_leaving = PeerLeaving {
        peer_id: vec![0u8; 32],
        r#gen: 0,
    };
    let encoded = encode_control_frame(STREAM_PEER_LEAVING, &bad_gen_leaving);
    let err = decode_control_frame::<PeerLeaving>(STREAM_PEER_LEAVING, &encoded)
        .expect_err("PeerLeaving gen=0 must be rejected");
    assert!(matches!(err, ControlFrameError::BadGeneration { got: 0 }));

    // Wrong gen (e.g. 2) also rejected
    let wrong_gen_gossip = GossipFrame {
        r#gen: 2,
        sender_id: vec![0u8; 32],
        peers: vec![PeerAnnouncement {
            endpoint_id: vec![0u8; 32],
            role: NodeRole::Worker as i32,
            ..Default::default()
        }],
    };
    let encoded = encode_control_frame(STREAM_GOSSIP, &wrong_gen_gossip);
    let err = decode_control_frame::<GossipFrame>(STREAM_GOSSIP, &encoded)
        .expect_err("GossipFrame gen=2 (future version) must be rejected");
    assert!(matches!(err, ControlFrameError::BadGeneration { got: 2 }));
}
