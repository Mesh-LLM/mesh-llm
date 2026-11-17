use super::*;
use prost::Message;

fn worker_announcement(seed: u8) -> PeerAnnouncement {
    let peer_id = EndpointId::from(SecretKey::from_bytes(&[seed; 32]).public());
    PeerAnnouncement {
        endpoint_id: peer_id.as_bytes().to_vec(),
        role: NodeRole::Worker as i32,
        ..Default::default()
    }
}

/// The two leading `PeerAnnouncement` fields, as a node that predates
/// `refuses_peer_model_downloads` decodes them.
#[derive(Clone, PartialEq, prost::Message)]
struct LegacyPeerAnnouncement {
    #[prost(bytes = "vec", tag = "1")]
    endpoint_id: Vec<u8>,
    #[prost(enumeration = "NodeRole", tag = "2")]
    role: i32,
}

#[test]
fn announcement_from_an_older_peer_accepts_peer_model_downloads() {
    let (_, ann) = proto_ann_to_local(&worker_announcement(0xA1)).expect("legacy announcement");

    assert!(!ann.refuses_peer_model_downloads);
}

#[test]
fn peer_model_download_refusal_roundtrips_through_proto_announcement() {
    let mut proto = worker_announcement(0xA2);
    proto.refuses_peer_model_downloads = Some(true);

    let (_, ann) = proto_ann_to_local(&proto).expect("announcement should decode");
    assert!(ann.refuses_peer_model_downloads);
    assert_eq!(
        local_ann_to_proto_ann(&ann).refuses_peer_model_downloads,
        Some(true)
    );

    proto.refuses_peer_model_downloads = Some(false);
    let (_, ann) = proto_ann_to_local(&proto).expect("announcement should decode");
    assert!(!ann.refuses_peer_model_downloads);
    assert_eq!(
        local_ann_to_proto_ann(&ann).refuses_peer_model_downloads,
        None
    );
}

#[test]
fn older_peer_decodes_an_announcement_that_refuses_peer_model_downloads() {
    let mut proto = worker_announcement(0xA3);
    proto.refuses_peer_model_downloads = Some(true);

    let legacy = LegacyPeerAnnouncement::decode(proto.encode_to_vec().as_slice())
        .expect("an older decoder must skip the unknown field");

    assert_eq!(legacy.endpoint_id, proto.endpoint_id);
    assert_eq!(legacy.role, NodeRole::Worker as i32);
}
