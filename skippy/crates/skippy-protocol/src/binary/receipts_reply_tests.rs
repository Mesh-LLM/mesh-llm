//! The receipts messages of the stage connection: a request with a JSON
//! body, and a reply that says what kind of answer it is and is read only
//! up to what its request can produce.

use super::*;
use std::io::Cursor;

/// A receipts request carries its JSON body, and its reply is read
/// only as a receipts reply.
#[test]
fn receipts_messages_round_trip() {
    let message = StageWireMessage::receipts(br#"{"commit":{}}"#.to_vec());
    let mut bytes = Vec::new();
    write_stage_message(&mut bytes, &message).unwrap();
    let decoded = read_stage_message(Cursor::new(bytes), 2048).unwrap();
    assert_eq!(decoded.kind, WireMessageKind::Receipts);
    assert_eq!(decoded.raw_bytes, br#"{"commit":{}}"#.to_vec());

    let mut reply = Vec::new();
    let answer = ReceiptsReply::Answer(b"[1]".to_vec());
    send_receipts_reply(&mut reply, &answer).unwrap();
    assert_eq!(
        recv_receipts_reply(Cursor::new(reply.clone()), 3).unwrap(),
        answer
    );
    assert!(recv_reply(Cursor::new(reply.clone())).is_err());
    reply.truncate(reply.len() - 1);
    assert!(recv_receipts_reply(Cursor::new(reply), 3).is_err());

    for other in [
        ReceiptsReply::Refused("no turn".into()),
        ReceiptsReply::NotFound("no receipt r1".into()),
        ReceiptsReply::CannotOpen(br#"{"reason":"expired"}"#.to_vec()),
        ReceiptsReply::Unavailable("receipts are off".into()),
    ] {
        let mut bytes = Vec::new();
        send_receipts_reply(&mut bytes, &other).unwrap();
        // what is not an answer is not held to the answer's bound
        assert_eq!(recv_receipts_reply(Cursor::new(bytes), 0).unwrap(), other);
    }
    let mut ack = Vec::new();
    send_reply_ack(&mut ack).unwrap();
    assert!(recv_receipts_reply(Cursor::new(ack), 3).is_err());
}

/// A reply longer than its request can produce is refused from its
/// declared length: the stage that asked reads none of it.
#[test]
fn a_receipts_reply_is_bounded_by_its_request() {
    let mut reply = Vec::new();
    send_receipts_reply(&mut reply, &ReceiptsReply::Answer(vec![b'x'; 1000])).unwrap();
    assert!(recv_receipts_reply(Cursor::new(reply.clone()), 1000).is_ok());
    let error = recv_receipts_reply(Cursor::new(reply), 999).unwrap_err();
    assert!(error.to_string().contains("exceeds"), "{error}");

    // a declared length alone is refused, with no body behind it
    let mut declared = Vec::new();
    declared.extend_from_slice(&(WireReplyKind::Receipts as i32).to_le_bytes());
    declared.extend_from_slice(&0u32.to_le_bytes());
    declared.extend_from_slice(&u32::MAX.to_le_bytes());
    let error = recv_receipts_reply(Cursor::new(declared), usize::MAX).unwrap_err();
    assert!(error.to_string().contains("exceeds"), "{error}");

    let long = ReceiptsReply::Refused("x".repeat(MAX_STAGE_RECEIPTS_REFUSAL_BYTES + 1));
    assert!(send_receipts_reply(Vec::new(), &long).is_err());
}
