use std::io::Write;
use std::net::{Ipv4Addr, SocketAddr, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::Duration;

use skippy_protocol::binary::{
    ReceiptsReply, StageWireMessage, read_stage_message, recv_ready, recv_receipts_reply,
    send_ready, write_stage_message,
};
use skippy_receipts::binding::verify_paths;
use skippy_receipts::wire::CannotOpenReason;

use super::{RECEIPTS_CONNECTIONS, serve_connection_until_idle};
use crate::receipts::downstream::{
    ChainRequest, CommitRequest, Downstream, ForwardedOpen, StageConnection,
};
use crate::receipts::kept_turns::KeptTurns;
use crate::receipts::refusal::{NotFound, Unavailable};
use crate::receipts::reply_limits::ReplyLimits;
use crate::receipts::tests::{
    GENEROUS, POSITIONS, SESSION, cannot_open, challenge, fake_stage, record, session_turn,
    sessions, stage, stage_config, turn,
};
use crate::receipts::{StageReceipts, lock};

/// A stage's binary listener, as far as receipts go.
struct Served {
    endpoint: String,
    /// Connections accepted so far.
    accepted: Arc<AtomicUsize>,
    /// One message per connection the stage stopped serving.
    closed: mpsc::Receiver<()>,
}

/// Serves `receipts` the way a stage's binary listener does, on an
/// ephemeral loopback port. A connection that sends nothing for `idle` is
/// closed.
fn serve_binary(receipts: Option<Arc<StageReceipts>>, idle: Duration) -> Served {
    let listener =
        std::net::TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0))).expect("bind");
    let endpoint = format!("tcp://{}", listener.local_addr().expect("address"));
    let accepted = Arc::new(AtomicUsize::new(0));
    let count = accepted.clone();
    let (closing, closed) = mpsc::channel();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { return };
            count.fetch_add(1, Ordering::SeqCst);
            let (receipts, closing) = (receipts.clone(), closing.clone());
            std::thread::spawn(move || {
                send_ready(&mut stream).expect("ready");
                if let Ok(first) = read_stage_message(&mut stream, 0) {
                    let sessions = sessions();
                    let _ = serve_connection_until_idle(
                        receipts,
                        sessions.as_ref(),
                        stream,
                        first,
                        idle,
                    );
                }
                let _ = closing.send(());
            });
        }
    });
    Served {
        endpoint,
        accepted,
        closed,
    }
}

/// Opens a receipts connection to `endpoint` with `request`, and returns it
/// with the stage's reply.
fn connect_with(endpoint: &str, request: &ChainRequest) -> (TcpStream, ReceiptsReply) {
    let address = endpoint.strip_prefix("tcp://").expect("a tcp endpoint");
    let mut stream = TcpStream::connect(address).expect("connect");
    stream.set_read_timeout(Some(GENEROUS)).expect("timeout");
    recv_ready(&mut stream).expect("ready");
    let body = serde_json::to_vec(request).expect("json");
    write_stage_message(&mut stream, &StageWireMessage::receipts(body)).expect("request");
    stream.flush().expect("flush");
    let reply = recv_receipts_reply(&mut stream, 1 << 20).expect("reply");
    (stream, reply)
}

/// The tokens of a turn the fixture's session computes after its first:
/// its prompt holds the first turn's positions.
fn next_turn() -> crate::receipts::TurnTokens {
    crate::receipts::TurnTokens {
        generation_start: POSITIONS.end + 1,
        sampled: 2,
        ..turn()
    }
}

fn commit_request(session: &str) -> CommitRequest {
    CommitRequest {
        receipt_id: "r1".into(),
        session: session.into(),
        position_start: POSITIONS.start,
        position_end: POSITIONS.end,
        client_key: None,
    }
}

fn open_request(receipt_id: &str) -> ForwardedOpen {
    ForwardedOpen {
        challenge: challenge(receipt_id, &[3]),
        turn_root: None,
    }
}

fn next_stage(endpoint: &str) -> StageConnection {
    StageConnection::new(&stage_config(endpoint), GENEROUS, GENEROUS)
}

#[test]
fn the_stage_connection_commits_and_opens_the_next_stage() {
    let served = serve_binary(Some(Arc::new(stage(1, 2..4, None))), GENEROUS);
    let stage0 = stage(0, 0..2, Some(Box::new(next_stage(&served.endpoint))));
    let receipt = stage0
        .finish_turn("r1", SESSION, session_turn(), turn(), None)
        .expect("receipt");
    let challenge = challenge("r1", &[1, 3]);
    for _ in 0..2 {
        let opening = stage0.open(&challenge, None).expect("opening");
        let stages: Vec<u32> = opening.stages.iter().map(|s| s.stage_index).collect();
        assert_eq!(stages, [0, 1]);
        verify_paths(&receipt, &challenge, &opening).expect("paths verify");
    }
    // commits have a connection of their own, and both are kept
    assert_eq!(served.accepted.load(Ordering::SeqCst), 2);
}

#[test]
fn the_stage_connection_carries_the_next_stage_error() {
    let served = serve_binary(Some(Arc::new(stage(1, 2..4, None))), GENEROUS);
    let next = next_stage(&served.endpoint);
    let limit = ReplyLimits::commit(6);
    let error = next
        .commit(&commit_request("other"), limit)
        .expect_err("unknown session");
    assert!(
        format!("{error:#}").contains("no native sequence"),
        "{error:#}"
    );
    // the connection survives a refused request, and a refusal does not
    // stop the next commit from being asked
    next.commit(&commit_request(SESSION), limit)
        .expect("commit");
    assert_eq!(served.accepted.load(Ordering::SeqCst), 1);
}

// A receipt id the stage keeps no record of is not found on the stage
// connection either, whatever its recorder still holds, and a turn it can
// no longer open is answered as such.
#[test]
fn an_unknown_or_evicted_receipt_is_not_opened_on_the_stage_connection() {
    let stage1 = Arc::new(stage(1, 2..4, None));
    let served = serve_binary(Some(stage1.clone()), GENEROUS);
    let next = next_stage(&served.endpoint);
    let error = next.open(&open_request("missing"), 1 << 20).unwrap_err();
    assert!(error.downcast_ref::<NotFound>().is_some(), "{error:#}");

    next.commit(&commit_request(SESSION), ReplyLimits::commit(6))
        .expect("commit");
    next.open(&open_request("r1"), 1 << 20).expect("opening");
    // the turn left the recorder's window: the stage cannot open another
    // challenge of it
    stage1.reports.recorder().discard("r1");
    let mut other = open_request("r1");
    other.challenge.picks[0].layer = 2;
    let error = next.open(&other, 1 << 20).unwrap_err();
    let cannot = cannot_open(&error);
    assert_eq!(cannot.reason, CannotOpenReason::Expired);
    assert_eq!(cannot.stage_index, 1);

    // the record is dropped, as when its window has passed
    *lock(&stage1.kept) = KeptTurns::default();
    let error = next.open(&open_request("r1"), 1 << 20).unwrap_err();
    assert!(error.downcast_ref::<NotFound>().is_some(), "{error:#}");
}

// A stage with receipts off answers a receipts message once, that it has
// none, and hangs up. The stage before then asks it no more: a next stage
// without receipts costs a turn nothing after the first.
#[test]
fn a_stage_without_receipts_refuses_receipts_messages() {
    let served = serve_binary(None, GENEROUS);
    let request = ChainRequest::Commit(commit_request(SESSION));
    let (_stream, reply) = connect_with(&served.endpoint, &request);
    assert!(matches!(reply, ReceiptsReply::Unavailable(_)), "{reply:?}");
    served.closed.recv_timeout(GENEROUS).expect("hung up");

    let next = next_stage(&served.endpoint);
    let stage0 = stage(0, 0..2, Some(Box::new(next)));
    let error = stage0
        .finish_turn("r1", SESSION, session_turn(), turn(), None)
        .unwrap_err();
    assert!(error.downcast_ref::<Unavailable>().is_some(), "{error:#}");
    // the turn has no receipt, and says why
    assert!(stage0.receipt("r1").expect("kept").is_err());
    let asked = served.accepted.load(Ordering::SeqCst);
    record(&stage0, POSITIONS.end..POSITIONS.end + 2);
    let error = stage0
        .finish_turn("r2", SESSION, session_turn(), next_turn(), None)
        .unwrap_err();
    assert!(format!("{error:#}").contains("not asked"), "{error:#}");
    assert_eq!(served.accepted.load(Ordering::SeqCst), asked);
}

// A stage that predates receipts closes the connection on the message kind
// it does not know. The turn then has no receipt, and that is all: the
// commit fails without failing anything else, and the stage is asked no
// more.
#[test]
fn a_stage_that_predates_receipts_yields_no_receipt() {
    let (endpoint, accepted) = fake_stage(|_| None);
    let stage0 = stage(0, 0..2, Some(Box::new(next_stage(&endpoint))));
    let error = stage0
        .finish_turn("r1", SESSION, session_turn(), turn(), None)
        .unwrap_err();
    assert!(error.downcast_ref::<Unavailable>().is_some(), "{error:#}");
    let reason = stage0.receipt("r1").expect("kept").expect_err("no receipt");
    assert!(reason.contains("no receipts request"), "{reason}");
    assert!(!stage0.reports.recorder().holds_turn("r1"));
    record(&stage0, POSITIONS.end..POSITIONS.end + 2);
    let error = stage0
        .finish_turn("r2", SESSION, session_turn(), next_turn(), None)
        .unwrap_err();
    assert!(format!("{error:#}").contains("not asked"), "{error:#}");
    assert_eq!(accepted.load(Ordering::SeqCst), 1);
}

// A receipts connection holds a slot only while it is used: one that sends
// nothing is closed, and a stage serves only so many at once. When they are
// all held, a new connection closes the one that waited longest, so a peer
// that only holds connections cannot keep out the stage before.
#[test]
fn receipts_connections_are_counted_and_closed_when_idle() {
    let stage1 = Arc::new(stage(1, 2..4, None));
    let served = serve_binary(Some(stage1.clone()), Duration::from_millis(100));
    let request = ChainRequest::Open(open_request("missing"));
    let (_idle, reply) = connect_with(&served.endpoint, &request);
    assert!(matches!(reply, ReceiptsReply::NotFound(_)), "{reply:?}");
    // the connection sends nothing more: the stage closes it
    served
        .closed
        .recv_timeout(GENEROUS)
        .expect("closed when idle");
    assert_eq!(stage1.connections.held(), 0);

    let served = serve_binary(Some(stage1.clone()), GENEROUS);
    let mut held: Vec<TcpStream> = (0..RECEIPTS_CONNECTIONS)
        .map(|_| {
            let (stream, reply) = connect_with(&served.endpoint, &request);
            assert!(matches!(reply, ReceiptsReply::NotFound(_)), "{reply:?}");
            stream
        })
        .collect();
    assert_eq!(stage1.connections.held(), RECEIPTS_CONNECTIONS);
    let (_newest, reply) = connect_with(&served.endpoint, &request);
    assert!(matches!(reply, ReceiptsReply::NotFound(_)), "{reply:?}");
    served
        .closed
        .recv_timeout(GENEROUS)
        .expect("the oldest idle closed");
    assert_eq!(stage1.connections.held(), RECEIPTS_CONNECTIONS);
    // the first connection was the one closed: the stage hung up on it
    let mut oldest = held.remove(0);
    let body = serde_json::to_vec(&request).expect("json");
    let _ = write_stage_message(&mut oldest, &StageWireMessage::receipts(body));
    assert!(recv_receipts_reply(&mut oldest, 1 << 20).is_err());
    // a connection that closes frees its slot
    drop(held.pop());
    served.closed.recv_timeout(GENEROUS).expect("closed");
    assert_eq!(stage1.connections.held(), RECEIPTS_CONNECTIONS - 1);
}

// A connection that lost its slot ends its wait by itself. On some
// platforms a shutdown from another thread does not end a blocked read,
// so the wait must not need one: here nothing touches the socket at all.
#[test]
fn a_connection_that_lost_its_slot_stops_waiting_without_a_shutdown() {
    let listener =
        std::net::TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0))).expect("bind");
    let _silent_peer =
        TcpStream::connect(listener.local_addr().expect("address")).expect("connect");
    let (upstream, _) = listener.accept().expect("accept");
    let slots = Arc::new(super::ConnectionSlots::default());
    let slot = slots.acquire(&upstream).expect("clone").expect("a slot");
    // a newer connection took the slot
    slots.lock().held.clear();
    let started = std::time::Instant::now();
    let waiting = super::wait_for_request(&upstream, &slot, GENEROUS).expect("wait");
    assert!(!waiting, "the connection ended");
    assert!(started.elapsed() < GENEROUS / 2, "{:?}", started.elapsed());
}
