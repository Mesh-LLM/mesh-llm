use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, mpsc};
use std::time::Duration;

use reqwest::StatusCode;
use skippy_protocol::binary::ReceiptsReply;
use skippy_receipts::wire::{CannotOpen, CannotOpenReason, StageCommitment};

use super::{ChainRequest, CommitRequest, Downstream, ForwardedOpen, StageConnection, http_reply};
use crate::receipts::refusal::{self, NotFound, Unavailable};
use crate::receipts::reply_limits::ReplyLimits;
use crate::receipts::tests::{
    GENEROUS, POSITIONS, SESSION, challenge, fake_stage, session_turn, sessions, stage,
    stage_config, turn,
};

fn commit_request() -> CommitRequest {
    CommitRequest {
        receipt_id: "r1".into(),
        session: SESSION.into(),
        position_start: POSITIONS.start,
        position_end: POSITIONS.end,
        client_key: None,
    }
}

/// What an honest stage 1 commits for the fixture's turn.
fn honest_commitments() -> Vec<StageCommitment> {
    stage(1, 2..4, None)
        .commit_session(sessions().as_ref(), &commit_request())
        .expect("commit")
}

fn answer(value: &impl serde::Serialize) -> Option<ReceiptsReply> {
    Some(ReceiptsReply::Answer(serde_json::to_vec(value).unwrap()))
}

// Stage 0 signs what the later stages commit, so it keeps a commitment
// only when it continues the plan: the next stage, the next layers, the
// leaves of the turn's positions.
#[test]
fn a_later_stages_commitment_must_continue_the_plan() {
    type Change = fn(&mut StageCommitment);
    let forgeries: [Change; 4] = [
        |commitment| commitment.stage_index = 5,
        |commitment| commitment.layer_start = 0,
        |commitment| commitment.leaf_count *= 2,
        |commitment| commitment.root = "not a root".into(),
    ];
    for (i, forge) in forgeries.into_iter().enumerate() {
        let mut forged = honest_commitments();
        forge(&mut forged[0]);
        let (endpoint, _) = fake_stage(move |_| answer(&forged));
        let next = StageConnection::new(&stage_config(&endpoint), GENEROUS, GENEROUS);
        let stage0 = stage(0, 0..2, Some(Box::new(next)));
        let result = stage0.finish_turn("r1", SESSION, session_turn(), turn(), None);
        assert!(result.is_err(), "forgery {i} was signed");
        assert!(!stage0.reports.recorder().holds_turn("r1"));
    }
    let honest = honest_commitments();
    let (endpoint, _) = fake_stage(move |_| answer(&honest));
    let next = StageConnection::new(&stage_config(&endpoint), GENEROUS, GENEROUS);
    let stage0 = stage(0, 0..2, Some(Box::new(next)));
    let receipt = stage0
        .finish_turn("r1", SESSION, session_turn(), turn(), None)
        .expect("receipt");
    assert_eq!(receipt.stages.len(), 2);
}

// A stage reads no more of a later stage's reply than its request can
// produce: a commit's few commitments, an opening's picked leaves.
#[test]
fn a_reply_longer_than_its_request_can_produce_is_refused() {
    let honest = honest_commitments();
    let (endpoint, _) = fake_stage(move |request| match request {
        ChainRequest::Commit(request) if request.receipt_id == "padded" => {
            Some(ReceiptsReply::Answer(vec![b' '; 4 << 20]))
        }
        ChainRequest::Commit(_) => answer(&honest),
        _ => Some(ReceiptsReply::Answer(vec![b' '; 64 << 20])),
    });
    let config = stage_config(&endpoint);
    let padded = StageConnection::new(&config, GENEROUS, GENEROUS);
    let stage0 = stage(0, 0..2, Some(Box::new(padded)));
    let error = stage0
        .finish_turn("padded", SESSION, session_turn(), turn(), None)
        .unwrap_err();
    assert!(format!("{error:#}").contains("exceeds"), "{error:#}");

    let next = StageConnection::new(&config, GENEROUS, GENEROUS);
    let stage0 = stage(0, 0..2, Some(Box::new(next)));
    stage0
        .finish_turn("r1", SESSION, session_turn(), turn(), None)
        .expect("receipt");
    // the challenge picks nothing on the later stage: boundary leaves only
    let error = stage0.open(&challenge("r1", &[1]), None).unwrap_err();
    assert!(format!("{error:#}").contains("exceeds"), "{error:#}");
}

// A commit ends every turn's response, so it has a connection of its own:
// it does not wait for an opening that replays another client's turn.
#[test]
fn a_commit_does_not_wait_behind_an_opening() {
    let (opening, opened) = mpsc::channel();
    let (finish, finished) = mpsc::channel::<()>();
    let (opening, finished) = (Mutex::new(opening), Mutex::new(finished));
    let (endpoint, accepted) = fake_stage(move |request| match request {
        ChainRequest::Commit(_) => Some(ReceiptsReply::Refused("committed".into())),
        _ => {
            // an opening that replays for as long as the test wants
            opening.lock().unwrap().send(()).unwrap();
            let _ = finished.lock().unwrap().recv_timeout(GENEROUS);
            Some(ReceiptsReply::Refused("opened".into()))
        }
    });
    let next = Arc::new(StageConnection::new(
        &stage_config(&endpoint),
        GENEROUS,
        GENEROUS,
    ));
    let opener = next.clone();
    let open = std::thread::spawn(move || {
        let request = ForwardedOpen {
            challenge: challenge("r1", &[3]),
            turn_root: None,
        };
        opener.open(&request, 1 << 20)
    });
    opened.recv_timeout(GENEROUS).expect("the opening started");
    let error = next
        .commit(&commit_request(), ReplyLimits::commit(6))
        .unwrap_err();
    assert!(format!("{error:#}").contains("committed"), "{error:#}");
    finish.send(()).unwrap();
    let error = open.join().unwrap().unwrap_err();
    assert!(format!("{error:#}").contains("opened"), "{error:#}");
    assert_eq!(accepted.load(Ordering::SeqCst), 2);
}

// A next stage that does not answer a commit costs one short wait: the turn
// has no receipt, and the stage is not asked again for a while.
#[test]
fn a_next_stage_that_does_not_answer_is_not_asked_again() {
    let (release, released) = mpsc::channel::<()>();
    let released = Mutex::new(released);
    let (endpoint, accepted) = fake_stage(move |_| {
        // holds the request until the test ends
        let _ = released.lock().unwrap().recv_timeout(GENEROUS);
        None
    });
    let short = Duration::from_millis(200);
    let next = StageConnection::new(&stage_config(&endpoint), short, GENEROUS);
    let limit = ReplyLimits::commit(6);
    let error = next.commit(&commit_request(), limit).unwrap_err();
    assert!(format!("{error:#}").contains("did not answer"), "{error:#}");
    let error = next.commit(&commit_request(), limit).unwrap_err();
    assert!(format!("{error:#}").contains("not asked"), "{error:#}");
    assert_eq!(accepted.load(Ordering::SeqCst), 1);
    drop(release);
}

// The receipts API of a next stage says by its status what the stage
// connection says by its reply kind.
#[test]
fn an_http_status_stands_for_a_reply_kind() {
    let cannot = CannotOpen {
        receipt_id: "r".into(),
        stage_index: 1,
        reason: CannotOpenReason::Expired,
        detail: "gone".into(),
    };
    let body = |value: serde_json::Value| serde_json::to_vec(&value).unwrap();
    let error = |status, json| refusal::answer(http_reply(status, body(json))).unwrap_err();
    let gone = serde_json::json!({ "error": "gone", "cannot_open": cannot });
    let answered = error(StatusCode::GONE, gone.clone());
    assert_eq!(answered.downcast_ref::<CannotOpen>(), Some(&cannot));
    let answered = error(StatusCode::CONFLICT, gone);
    assert_eq!(answered.downcast_ref::<CannotOpen>(), Some(&cannot));
    let missing = serde_json::json!({ "error": "no receipt r" });
    let answered = error(StatusCode::NOT_FOUND, missing.clone());
    assert!(answered.downcast_ref::<NotFound>().is_some());
    let answered = error(StatusCode::SERVICE_UNAVAILABLE, missing.clone());
    assert!(answered.downcast_ref::<Unavailable>().is_some());
    let answered = error(StatusCode::BAD_REQUEST, missing);
    assert!(format!("{answered:#}").contains("no receipt r"));
    assert_eq!(
        refusal::answer(http_reply(StatusCode::OK, b"[]".to_vec())).unwrap(),
        b"[]"
    );
}
