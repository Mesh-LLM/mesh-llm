use super::*;
use mesh_llm_protocol::proto::node as proto_node;
use prost::Message;

const NOW: u64 = 10 * NODE_RECORD.max_age_ms;

fn key(seed: u8) -> SigningKey {
    SigningKey::from_bytes(&[seed; 32])
}

fn id(seed: u8) -> EndpointId {
    EndpointId::from_bytes(key(seed).verifying_key().as_bytes()).unwrap()
}

fn body(version: &str) -> Vec<u8> {
    proto_node::NodeRecord {
        version: version.to_string(),
        vram_bytes: 24_000_000_000,
        serving_models: vec!["Qwen3-8B-Q4_K_M".to_string()],
        ..Default::default()
    }
    .encode_to_vec()
}

fn sign(seq: u64, issued_at: u64, version: &str) -> VerifiedRecord {
    VerifiedRecord::sign(NODE_RECORD, &key(1), seq, issued_at, &body(version))
}

#[test]
fn signed_record_verifies_and_exposes_original_bytes() {
    let signed = sign(7, 1_000, "0.78.0");

    let verified =
        VerifiedRecord::verify(NODE_RECORD, signed.signed_bytes(), signed.signature()).unwrap();

    assert_eq!(verified, signed);
    assert_eq!(verified.header().seq, 7);
    assert_eq!(verified.header().issued_at_unix_ms, 1_000);
    assert_eq!(verified.endpoint_id(), id(1));
    assert_eq!(verified.body(), body("0.78.0").as_slice());
}

#[test]
fn any_changed_byte_breaks_the_signature() {
    let signed = sign(7, 1_000, "0.78.0");
    let bytes = signed.signed_bytes();
    for index in [
        ENDPOINT_ID_LEN,
        ENDPOINT_ID_LEN + 8,
        HEADER_LEN,
        bytes.len() - 1,
    ] {
        let mut tampered = bytes.to_vec();
        tampered[index] ^= 0x01;
        assert_eq!(
            VerifiedRecord::verify(NODE_RECORD, &tampered, signed.signature()),
            Err(RecordError::BadSignature),
            "byte {index} was not covered"
        );
    }
}

#[test]
fn record_cannot_be_reattributed_to_another_node() {
    let signed = sign(7, 1_000, "0.78.0");
    let mut reattributed = signed.signed_bytes().to_vec();
    reattributed[..ENDPOINT_ID_LEN].copy_from_slice(id(2).as_bytes());

    assert_eq!(
        VerifiedRecord::verify(NODE_RECORD, &reattributed, signed.signature()),
        Err(RecordError::BadSignature)
    );
}

#[test]
fn signature_without_the_domain_tag_is_rejected() {
    let signed = sign(7, 1_000, "0.78.0");
    let untagged = key(1).sign(signed.signed_bytes()).to_bytes();

    assert_eq!(
        VerifiedRecord::verify(NODE_RECORD, signed.signed_bytes(), &untagged),
        Err(RecordError::BadSignature)
    );
}

#[test]
fn a_record_of_one_kind_does_not_verify_as_another() {
    let node_record = sign(7, 1_000, "0.78.0");

    assert_eq!(
        VerifiedRecord::verify(
            CACHE_AFFINITY_RECORD,
            node_record.signed_bytes(),
            node_record.signature()
        ),
        Err(RecordError::BadSignature)
    );
}

#[test]
fn malformed_envelopes_are_rejected_before_verification() {
    let signed = sign(7, 1_000, "0.78.0");

    assert_eq!(
        VerifiedRecord::verify(
            NODE_RECORD,
            &signed.signed_bytes()[..HEADER_LEN - 1],
            signed.signature()
        ),
        Err(RecordError::Truncated)
    );
    assert_eq!(
        VerifiedRecord::verify(
            NODE_RECORD,
            signed.signed_bytes(),
            &signed.signature()[..63]
        ),
        Err(RecordError::InvalidSignatureLength)
    );
    let oversize = vec![0u8; CACHE_AFFINITY_RECORD.max_signed_bytes + 1];
    assert_eq!(
        RecordHeader::parse(&CACHE_AFFINITY_RECORD, &oversize),
        Err(RecordError::Oversize)
    );
}

#[test]
fn fields_unknown_to_this_build_survive_verification() {
    let mut unknown = body("0.99.0");
    // Field 99, wire type 2 (length-delimited), 3 bytes.
    unknown.extend_from_slice(&[0x9a, 0x06, 0x03, b'n', b'e', b'w']);
    let signed = VerifiedRecord::sign(NODE_RECORD, &key(1), 7, 1_000, &unknown);

    let verified =
        VerifiedRecord::verify(NODE_RECORD, signed.signed_bytes(), signed.signature()).unwrap();

    assert_eq!(verified.body(), unknown.as_slice());
    assert_eq!(
        proto_node::NodeRecord::decode(verified.body())
            .unwrap()
            .version,
        "0.99.0"
    );
}

#[test]
fn ordering_prefers_higher_sequence_and_skips_duplicates() {
    let held = sign(7, 1_000, "0.78.0");
    let order = |incoming: &VerifiedRecord, held: Option<&VerifiedRecord>| {
        order_against_held(held, &incoming.header(), incoming.signed_bytes())
    };

    assert_eq!(order(&held, None), RecordOrdering::New);
    assert_eq!(order(&held, Some(&held)), RecordOrdering::Duplicate);
    assert_eq!(
        order(&sign(8, 2_000, "0.78.1"), Some(&held)),
        RecordOrdering::Newer
    );
    assert_eq!(
        order(&sign(6, 500, "0.77.0"), Some(&held)),
        RecordOrdering::Superseded
    );
    assert_eq!(
        order(&sign(7, 1_000, "0.78.9"), Some(&held)),
        RecordOrdering::Superseded,
        "the same sequence with different bytes does not replace the held record"
    );
}

#[test]
fn freshness_follows_each_kinds_bounds() {
    let header = |issued_at_unix_ms| RecordHeader {
        endpoint_id: id(1),
        seq: 1,
        issued_at_unix_ms,
    };
    for kind in [NODE_RECORD, CACHE_AFFINITY_RECORD] {
        assert!(header(NOW).is_fresh_at(&kind, NOW));
        assert!(header(NOW - kind.max_age_ms).is_fresh_at(&kind, NOW));
        assert!(!header(NOW - kind.max_age_ms - 1).is_fresh_at(&kind, NOW));
        assert!(header(NOW + kind.max_future_skew_ms).is_fresh_at(&kind, NOW));
        assert!(!header(NOW + kind.max_future_skew_ms + 1).is_fresh_at(&kind, NOW));
    }
}

#[test]
fn local_sequence_strictly_increases_even_if_the_clock_stalls() {
    assert_eq!(next_record_seq(None, 1_000), 1_000);
    assert_eq!(next_record_seq(Some(1_000), 1_000), 1_001);
    assert_eq!(next_record_seq(Some(5_000), 1_000), 5_001);
    assert_eq!(next_record_seq(Some(1_000), 9_000), 9_000);
}

mod held_records {
    use super::*;

    #[derive(Clone, Copy)]
    enum Via {
        Originator,
        Relay,
    }

    fn admit(held: &mut HeldRecords, record: &VerifiedRecord, via: Via) -> Option<RecordSkip> {
        let sender = match via {
            Via::Originator => record.endpoint_id(),
            Via::Relay => id(2),
        };
        let admission = held
            .admit(
                id(9),
                sender,
                record.signed_bytes(),
                record.signature(),
                NOW,
                |body| Ok(body.len()),
            )
            .expect("header parses");
        match admission {
            RecordAdmission::Accepted { .. } => None,
            RecordAdmission::Skipped { reason, .. } => Some(reason),
        }
    }

    fn held_seq(held: &HeldRecords) -> Option<u64> {
        held.get(&id(1)).map(|record| record.header().seq)
    }

    #[test]
    fn relay_records_replace_only_older_ones() {
        let mut held = HeldRecords::new(NODE_RECORD);

        assert_eq!(admit(&mut held, &sign(7, NOW, "a"), Via::Relay), None);
        assert_eq!(admit(&mut held, &sign(8, NOW, "b"), Via::Relay), None);
        assert_eq!(
            admit(&mut held, &sign(7, NOW, "a"), Via::Relay),
            Some(RecordSkip::Superseded)
        );
        assert_eq!(held_seq(&held), Some(8));
    }

    #[test]
    fn duplicates_are_accepted_without_verifying_again() {
        let mut held = HeldRecords::new(NODE_RECORD);
        let record = sign(7, NOW, "a");
        admit(&mut held, &record, Via::Relay);

        let admission = held
            .admit(
                id(9),
                id(2),
                record.signed_bytes(),
                &[0u8; 64],
                NOW,
                |body| Ok(body.len()),
            )
            .unwrap();

        assert!(matches!(admission, RecordAdmission::Accepted { .. }));
    }

    #[test]
    fn originator_may_roll_its_sequence_back() {
        let mut held = HeldRecords::new(NODE_RECORD);
        admit(&mut held, &sign(8, NOW, "b"), Via::Relay);

        assert_eq!(admit(&mut held, &sign(7, NOW, "a"), Via::Originator), None);
        assert_eq!(held_seq(&held), Some(7));
    }

    #[test]
    fn relays_drop_expired_records_and_nobody_keeps_future_ones() {
        let mut held = HeldRecords::new(NODE_RECORD);
        let expired = sign(7, NOW - NODE_RECORD.max_age_ms - 1, "a");
        let future = sign(7, NOW + NODE_RECORD.max_future_skew_ms + 1, "a");

        assert_eq!(
            admit(&mut held, &expired, Via::Relay),
            Some(RecordSkip::Expired)
        );
        assert_eq!(
            admit(&mut held, &future, Via::Relay),
            Some(RecordSkip::Expired)
        );
        assert!(held.is_empty());
        assert_eq!(
            admit(&mut held, &future, Via::Originator),
            Some(RecordSkip::Expired),
            "a node's own record from too far in the future is not held either"
        );
        assert!(held.is_empty());
        assert_eq!(admit(&mut held, &expired, Via::Originator), None);
    }

    #[test]
    fn forged_records_are_not_held() {
        let mut held = HeldRecords::new(NODE_RECORD);
        let record = sign(7, NOW, "a");
        let mut forged = record.signed_bytes().to_vec();
        forged[HEADER_LEN] ^= 0x01;

        let admission = held
            .admit(id(9), id(2), &forged, record.signature(), NOW, |body| {
                Ok(body.len())
            })
            .unwrap();

        assert!(matches!(
            admission,
            RecordAdmission::Skipped {
                reason: RecordSkip::Invalid(RecordError::BadSignature),
                ..
            }
        ));
        assert!(held.is_empty());
    }

    #[test]
    fn bodies_the_caller_rejects_are_not_held() {
        let mut held = HeldRecords::new(NODE_RECORD);
        let record = sign(7, NOW, "a");

        let admission = held
            .admit(
                id(9),
                id(2),
                record.signed_bytes(),
                record.signature(),
                NOW,
                |_| Err::<(), _>(RecordError::Decode),
            )
            .unwrap();

        assert!(matches!(
            admission,
            RecordAdmission::Skipped {
                reason: RecordSkip::Invalid(RecordError::Decode),
                ..
            }
        ));
        assert!(held.is_empty());
    }

    #[test]
    fn records_are_held_per_kind() {
        let mut held = HeldRecords::new(CACHE_AFFINITY_RECORD);

        assert_eq!(
            admit(&mut held, &sign(7, NOW, "a"), Via::Relay),
            Some(RecordSkip::Invalid(RecordError::BadSignature)),
            "a node record cannot be admitted as cache affinity"
        );
    }

    #[test]
    fn own_record_echoed_back_is_skipped() {
        let mut held = HeldRecords::new(NODE_RECORD);
        let own = VerifiedRecord::sign(NODE_RECORD, &key(9), 7, NOW, &body("a"));

        assert_eq!(
            admit(&mut held, &own, Via::Relay),
            Some(RecordSkip::OwnRecord)
        );
    }

    #[test]
    fn local_record_is_resigned_only_when_changed_or_due() {
        let mut held = HeldRecords::new(NODE_RECORD);

        let first = held.refresh_local(&key(9), &body("a"), NOW).clone();
        let unchanged = held.refresh_local(&key(9), &body("a"), NOW + 1).clone();
        let changed = held.refresh_local(&key(9), &body("b"), NOW + 2).clone();
        let due_at = NOW + 2 + NODE_RECORD.refresh_ms;
        let due = held.refresh_local(&key(9), &body("b"), due_at).clone();

        assert_eq!(unchanged, first);
        assert!(changed.header().seq > first.header().seq);
        assert!(due.header().seq > changed.header().seq);
        assert_eq!(due.header().issued_at_unix_ms, due_at);
        assert_eq!(due.endpoint_id(), id(9));
    }
}
