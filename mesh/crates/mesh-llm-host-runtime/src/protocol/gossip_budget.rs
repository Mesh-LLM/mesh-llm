//! Local gossip encoding limits are not evidence that a remote peer failed.

use super::MAX_CONTROL_FRAME_BYTES;
use prost::Message;

mod selection;
pub(crate) use selection::{GossipFrameEncoder, MAX_GOSSIP_RECORDS};

#[derive(Debug)]
pub(crate) struct GossipFrameBudgetExceeded {
    size: usize,
}

impl std::fmt::Display for GossipFrameBudgetExceeded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "local gossip frame is {} bytes (limit {MAX_CONTROL_FRAME_BYTES})",
            self.size
        )
    }
}

impl std::error::Error for GossipFrameBudgetExceeded {}

pub(super) fn encode_gossip(frame: &crate::proto::node::GossipFrame) -> anyhow::Result<Vec<u8>> {
    let size = frame.encoded_len();
    if size > MAX_CONTROL_FRAME_BYTES {
        return Err(GossipFrameBudgetExceeded { size }.into());
    }
    Ok(frame.encode_to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn actual_over_limit_frame_is_a_local_budget_error() {
        use mesh_llm_membership::signed_record::{NODE_RECORD, VerifiedRecord};
        let body = crate::proto::node::NodeRecord {
            role: crate::proto::node::NodeRole::Worker as i32,
            serving_models: (0..250)
                .map(|index| format!("{index:03}-{}", "model".repeat(80)))
                .collect(),
            ..Default::default()
        }
        .encode_to_vec();
        // Individually valid, bounded records can still exceed the aggregate
        // frame limit. Exercise real signed envelopes, not an oversized field
        // that would itself be rejected by the record verifier.
        let issued_at = mesh_llm_membership::requirements::current_time_unix_ms();
        let records = (1..=90)
            .map(|seed| {
                let key = ed25519_dalek::SigningKey::from_bytes(&[seed; 32]);
                let record = VerifiedRecord::sign(NODE_RECORD, &key, 1, issued_at, &body);
                VerifiedRecord::verify(NODE_RECORD, record.signed_bytes(), record.signature())
                    .unwrap();
                let decoded = crate::proto::node::NodeRecord::decode(body.as_slice()).unwrap();
                let (_, announcement) = super::super::node_record::node_record_to_local(
                    record.endpoint_id(),
                    &decoded,
                    None,
                )
                .unwrap();
                assert_eq!(announcement.serving_models.len(), 250);
                crate::proto::node::SignedNodeRecord {
                    signed: record.signed_bytes().to_vec(),
                    signature: record.signature().to_vec(),
                    hop: None,
                }
            })
            .collect();
        let frame = crate::proto::node::GossipFrame {
            signed_records: records,
            ..Default::default()
        };
        let error = encode_gossip(&frame).unwrap_err();
        assert!(error.downcast_ref::<GossipFrameBudgetExceeded>().is_some());
        assert!(frame.encoded_len() > MAX_CONTROL_FRAME_BYTES);
        assert_eq!(
            crate::mesh::heartbeat::GossipHealth::from_exchange(Ok(Err(error))),
            crate::mesh::heartbeat::GossipHealth::Deferred
        );
    }

    #[test]
    fn frame_at_the_limit_is_accepted_and_one_extra_byte_is_rejected() {
        // Field 3 has one tag byte and a four-byte length at this size.
        let mut frame = crate::proto::node::GossipFrame {
            sender_id: vec![0; MAX_CONTROL_FRAME_BYTES - 5],
            ..Default::default()
        };
        assert_eq!(frame.encoded_len(), MAX_CONTROL_FRAME_BYTES);
        assert_eq!(
            encode_gossip(&frame).unwrap().len(),
            MAX_CONTROL_FRAME_BYTES
        );
        frame.sender_id.push(0);
        assert!(
            encode_gossip(&frame)
                .unwrap_err()
                .is::<GossipFrameBudgetExceeded>()
        );
    }
}
