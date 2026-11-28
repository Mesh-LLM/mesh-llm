use super::{ControlFrameError, validate_node_record};
use crate::proto::node::{NodeRecord, NodeRole};
use prost::Message;

fn record() -> NodeRecord {
    NodeRecord {
        role: NodeRole::Client as i32,
        ..Default::default()
    }
}

#[test]
fn opaque_evidence_survives_wire_roundtrip_at_supported_bounds() {
    let mut record = record();
    for index in 0..8 {
        record
            .peer_evidence
            .insert(format!("{index:064}"), vec![index; 1024]);
    }
    validate_node_record(&record).unwrap();
    let decoded = NodeRecord::decode(record.encode_to_vec().as_slice()).unwrap();
    assert_eq!(decoded.peer_evidence, record.peer_evidence);
    validate_node_record(&decoded).unwrap();
}

#[test]
fn empty_evidence_remains_valid_and_oversized_references_are_rejected() {
    let mut record = record();
    validate_node_record(&record).unwrap();
    for (namespace, bytes) in [
        (String::new(), vec![1]),
        ("n".repeat(65), vec![1]),
        ("namespace".into(), vec![1; 1025]),
    ] {
        record.peer_evidence = [(namespace, bytes)].into();
        assert!(matches!(
            validate_node_record(&record),
            Err(ControlFrameError::InvalidPeerEvidence)
        ));
    }
    record.peer_evidence = (0..9).map(|i| (i.to_string(), vec![1])).collect();
    assert!(matches!(
        validate_node_record(&record),
        Err(ControlFrameError::InvalidPeerEvidence)
    ));
}
