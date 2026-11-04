//! Which turn of which native sequence each session computes, while the
//! stage issues inference receipts ([`crate::receipts`]).
//!
//! A native sequence serves one session after another, and a session id can
//! come back for a later request. So a session that takes a sequence starts
//! a new turn there, with an identity of its own, which the receipt
//! recorder is told before the session's first ubatch. A commit names the
//! session; the stage resolves it to that turn, and the recorder then takes
//! that turn's log and no other's, also when the sequence has moved on.
//!
//! A session's turn stays resolvable for a few sessions after it ends, for
//! a commit that arrives late; the recorder keeps such a turn's log about
//! as long.

use std::collections::{BTreeMap, VecDeque};

use skippy_receipts::recorder::TurnId;

use super::*;

/// Turns of ended sessions that a late commit can still name.
const RELEASED_TURNS: usize = 64;

/// The turn a session computes: its native sequence, and the identity the
/// receipt recorder knows the turn by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SessionTurn {
    pub seq: i32,
    pub turn: TurnId,
}

#[derive(Default)]
pub(super) struct ReceiptTurns {
    /// The last identity given out; identities start at 1, and the
    /// recorder's default names what no session armed.
    last: u64,
    /// By session that holds a sequence: its turn.
    bound: BTreeMap<String, SessionTurn>,
    /// Turns of sessions that ended, oldest first.
    released: VecDeque<(String, SessionTurn)>,
}

impl ReceiptTurns {
    /// `session` took `seq`: it starts a new turn there.
    fn bind(&mut self, session: &str, seq: i32) -> SessionTurn {
        self.release(session);
        self.last += 1;
        let turn = SessionTurn {
            seq,
            turn: TurnId(self.last),
        };
        self.bound.insert(session.to_owned(), turn);
        turn
    }

    /// `session` ended: its turn stays resolvable for a late commit, until
    /// enough sessions have ended after it.
    fn release(&mut self, session: &str) {
        let Some((session, turn)) = self.bound.remove_entry(session) else {
            return;
        };
        self.released.push_back((session, turn));
        while self.released.len() > RELEASED_TURNS {
            self.released.pop_front();
        }
    }

    /// The turn `session` computes, or computed last.
    fn turn(&self, session: &str) -> Option<SessionTurn> {
        self.bound.get(session).copied().or_else(|| {
            self.released
                .iter()
                .rev()
                .find(|(released, _)| released == session)
                .map(|(_, turn)| *turn)
        })
    }
}

impl RuntimeState {
    /// A session took a native sequence: while receipts are on, it starts
    /// a turn there, which the recorder learns before the session's first
    /// ubatch.
    pub(super) fn note_receipt_sequence(&mut self, session_id: &str) {
        let Some(receipts) = self.receipts.clone() else {
            return;
        };
        let Some(seq) = self
            .sessions
            .get(session_id)
            .and_then(|lane| lane.session.native_sequence_id().ok())
        else {
            return;
        };
        self.release_ended_receipt_turns();
        let turn = self.receipt_turns.bind(session_id, seq);
        receipts.begin_turn(turn);
    }

    /// Releases the turns of sessions that hold no sequence any more, on
    /// whatever path they lost it. A session parked for its exports is one
    /// of them; its turn stays resolvable like any released one.
    fn release_ended_receipt_turns(&mut self) {
        let ended: Vec<String> = self
            .receipt_turns
            .bound
            .keys()
            .filter(|session| !self.sessions.contains_key(*session))
            .cloned()
            .collect();
        for session in ended {
            self.receipt_turns.release(&session);
        }
    }

    /// A session gave up its sequence.
    pub(super) fn release_receipt_turn(&mut self, session_id: &str) {
        self.receipt_turns.release(session_id);
    }

    /// Turns receipts on: later session bindings start turns the recorder
    /// knows.
    pub fn set_receipts(&mut self, receipts: Arc<crate::receipts::StageReceipts>) {
        self.receipts = Some(receipts);
    }

    pub fn receipts(&self) -> Option<&Arc<crate::receipts::StageReceipts>> {
        self.receipts.as_ref()
    }

    /// The turn a session computes, or computed last.
    pub fn receipt_turn(&self, session_id: &str) -> Option<SessionTurn> {
        self.receipt_turns.turn(session_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_session_that_takes_a_sequence_starts_a_new_turn() {
        let mut turns = ReceiptTurns::default();
        let first = turns.bind("a", 3);
        assert_eq!(turns.turn("a"), Some(first));
        // another request's session takes the same sequence
        turns.release("a");
        let second = turns.bind("b", 3);
        assert_eq!((first.seq, second.seq), (3, 3));
        assert_ne!(first.turn, second.turn);
        // a late commit of the first session still names its own turn
        assert_eq!(turns.turn("a"), Some(first));
        // a session id that comes back names its new turn
        turns.release("b");
        let again = turns.bind("a", 5);
        assert_eq!(turns.turn("a"), Some(again));
        assert_ne!(again.turn, first.turn);
        assert_eq!(turns.turn("never"), None);
    }

    // One session per request must not leave one entry per request.
    #[test]
    fn ended_sessions_are_forgotten() {
        let mut turns = ReceiptTurns::default();
        for request in 0..10 * RELEASED_TURNS {
            let session = format!("session-{request}");
            turns.bind(&session, 0);
            turns.release(&session);
        }
        assert!(turns.bound.is_empty());
        assert_eq!(turns.released.len(), RELEASED_TURNS);
        assert_eq!(turns.turn("session-0"), None);
        let last = format!("session-{}", 10 * RELEASED_TURNS - 1);
        assert!(turns.turn(&last).is_some());
    }
}
