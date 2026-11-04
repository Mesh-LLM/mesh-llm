use std::sync::Arc;

use reqwest::StatusCode;
use reqwest::blocking::{Client, Response};
use skippy_receipts::attention::opening::BlockChallenge;
use skippy_receipts::recorder::MemoryOp;
use skippy_receipts::turn;
use skippy_receipts::wire::{CannotOpen, CannotOpenReason, Challenge, Receipt};

use super::{client_routes, router};
use crate::receipts::downstream::CommitRequest;
use crate::receipts::kept_turns::KeptTurns;
use crate::receipts::tests::{
    POSITIONS, SEQ, SESSION, api_state, bound_turn, chain, challenge, serve_routes, session_turn,
    stage, turn as turn_tokens,
};
use crate::receipts::{StageReceipts, lock};

fn post(url: &str, route: &str, body: &impl serde::Serialize) -> Response {
    Client::new()
        .post(format!("{url}{route}"))
        .json(body)
        .send()
        .expect("reply")
}

/// A single stage that issued the bound receipt `r1`, served on its own
/// API and on the client routes, with the challenge its client draws.
fn served_bound_turn() -> (Arc<StageReceipts>, String, Receipt, Challenge) {
    let receipts = Arc::new(stage(0, 0..4, None));
    let (receipt, challenge) = bound_turn(&receipts, "r1", &[1]);
    let routes = router(api_state(&receipts)).merge(client_routes(api_state(&receipts)));
    (receipts.clone(), serve_routes(routes), receipt, challenge)
}

#[test]
fn downstream_rejects_an_unknown_session() {
    let downstream = chain().downstream;
    let reply = post(
        &downstream,
        "/receipts/commit",
        &CommitRequest {
            receipt_id: "r1".into(),
            session: "other".into(),
            position_start: POSITIONS.start,
            position_end: POSITIONS.end,
            client_key: None,
        },
    );
    assert_eq!(reply.status(), StatusCode::BAD_REQUEST);
    assert!(reply.text().expect("body").contains("no native sequence"));
}

// The OpenAI surface opens only a bound receipt, for its client's challenge.
#[test]
fn the_client_route_opens_only_bound_receipts() {
    let unbound = Arc::new(stage(0, 0..4, None));
    unbound
        .finish_turn("unbound", SESSION, session_turn(), turn_tokens(), None)
        .expect("receipt");
    let unbound_url = serve_routes(client_routes(api_state(&unbound)));
    let (_receipts, url, receipt, bound) = served_bound_turn();
    let open_at = |url: &str, challenge: &Challenge| {
        let mut body = serde_json::to_value(challenge).unwrap();
        body["model"] = "model".into();
        post(url, "/v1/receipts/open", &body)
    };

    let reply = open_at(&unbound_url, &challenge("unbound", &[1]));
    assert_eq!(reply.status(), StatusCode::FORBIDDEN);
    assert_eq!(
        open_at(&url, &challenge("missing", &[1])).status(),
        StatusCode::NOT_FOUND
    );

    let mut unproven = bound.clone();
    unproven.proof = None;
    assert_eq!(open_at(&url, &unproven).status(), StatusCode::BAD_REQUEST);
    let reply = open_at(&url, &bound);
    assert!(reply.status().is_success(), "{}", reply.status());
    let signature = reply
        .headers()
        .get(turn::OPENING_SIGNATURE_HEADER)
        .expect("signed")
        .to_str()
        .unwrap()
        .to_owned();
    let bytes = reply.bytes().expect("body");
    let verified = turn::verify(&receipt, None).unwrap();
    turn::verify_opening(&verified, &bound, &bytes, &signature).expect("signature");
}

// A receipt id the stage keeps no record of is not found on every route,
// whatever the recorder still holds: nothing opens it without the proof
// its record asked for.
#[test]
fn an_evicted_receipt_is_not_found_on_every_route() {
    let (receipts, url, _, challenge) = served_bound_turn();
    *lock(&receipts.kept) = KeptTurns::default();
    assert!(receipts.reports.recorder().holds_turn("r1"));
    let mut unproven = challenge.clone();
    unproven.proof = None;
    let rows = BlockChallenge {
        receipt_id: "r1".into(),
        picks: Vec::new(),
        proof: None,
    };
    for challenge in [&challenge, &unproven] {
        for route in ["/receipts/open", "/v1/receipts/open"] {
            let reply = post(&url, route, challenge);
            assert_eq!(reply.status(), StatusCode::NOT_FOUND, "{route}");
            assert!(
                reply
                    .headers()
                    .get(turn::OPENING_SIGNATURE_HEADER)
                    .is_none()
            );
        }
    }
    for route in ["/receipts/rows", "/v1/receipts/rows"] {
        assert_eq!(post(&url, route, &rows).status(), StatusCode::NOT_FOUND);
    }
    let reply = reqwest::blocking::get(format!("{url}/receipts/r1")).expect("reply");
    assert_eq!(reply.status(), StatusCode::NOT_FOUND);
}

/// The node's signed "cannot open" answer to `challenge`, checked against
/// the receipt.
fn signed_cannot_open(reply: Response, receipt: &Receipt, challenge: &Challenge) -> CannotOpen {
    let signature = reply
        .headers()
        .get(turn::OPENING_SIGNATURE_HEADER)
        .expect("the node signs that it cannot open")
        .to_str()
        .unwrap()
        .to_owned();
    let bytes = reply.bytes().expect("body");
    let bound = turn::verify(receipt, None).unwrap();
    turn::verify_cannot_open(&bound, &challenge.digest(), &bytes, &signature).expect("signature");
    // it is no opening
    assert!(turn::verify_opening(&bound, challenge, &bytes, &signature).is_err());
    let body: serde_json::Value = serde_json::from_slice(&bytes).expect("json");
    assert!(body["error"].as_str().is_some_and(|e| !e.is_empty()));
    serde_json::from_value(body["cannot_open"].clone()).expect("a cannot-open answer")
}

// A client tells "the node can no longer open my turn" from a request the
// node refuses: by the status, the body and the node's signature.
#[test]
fn a_turn_the_node_cannot_open_is_answered_as_such() {
    let (receipts, url, receipt, challenge) = served_bound_turn();
    let recorder = receipts.reports.recorder();
    // the session ended, then a ubatch the recorder could not follow
    recorder.memory_op(MemoryOp::SeqRm, SEQ, -1, -1, -1);
    recorder.unreceiptable("two lanes in one recurrent ubatch");
    let reply = post(&url, "/v1/receipts/open", &challenge);
    assert_eq!(reply.status(), StatusCode::CONFLICT);
    let cannot = signed_cannot_open(reply, &receipt, &challenge);
    assert_eq!(cannot.reason, CannotOpenReason::CellsUnknown);
    assert_eq!((cannot.receipt_id.as_str(), cannot.stage_index), ("r1", 0));

    // a malformed request of the same receipt is still a plain refusal
    let mut malformed = challenge.clone();
    malformed.picks[0].positions = vec![9, 4];
    let reply = post(&url, "/v1/receipts/open", &malformed);
    assert_eq!(reply.status(), StatusCode::BAD_REQUEST);
    assert!(
        reply
            .headers()
            .get(turn::OPENING_SIGNATURE_HEADER)
            .is_none()
    );

    // the turn left the stage's opening window
    let (receipts, url, receipt, challenge) = served_bound_turn();
    receipts.reports.recorder().discard("r1");
    for route in ["/receipts/open", "/v1/receipts/open"] {
        let reply = post(&url, route, &challenge);
        assert_eq!(reply.status(), StatusCode::GONE, "{route}");
        let cannot = signed_cannot_open(reply, &receipt, &challenge);
        assert_eq!(cannot.reason, CannotOpenReason::Expired);
    }
}
