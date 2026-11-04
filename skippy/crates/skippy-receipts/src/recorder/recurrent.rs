//! The recurrent state each turn of a hybrid model starts from.
//!
//! A recurrent layer (the gated delta net, Mamba) keeps one state per
//! sequence that every ubatch overwrites, so a replay cannot rebuild the
//! state a turn started from out of the stage's cache, as it does an
//! attention layer's KV cells. The native recorder exports the state before
//! a turn's first ubatch. The stage commits its digest with the turn and
//! keeps the bytes while a client can challenge the turn, and a replay
//! starts from them. A state is tens of megabytes for a hybrid model of a
//! few billion parameters, so a turn no client can challenge drops it at
//! once.

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;

use super::{Recorder, RecorderState};
use crate::commitment::{Hash, recurrent_state_digest};

/// The state a turn started from: its digest, and its bytes while the
/// turn can still be opened.
#[derive(Clone, Debug)]
pub(super) struct TurnState {
    pub digest: Hash,
    pub bytes: Option<Arc<[u8]>>,
}

impl TurnState {
    fn new(state: &[u8]) -> Self {
        Self {
            digest: recurrent_state_digest(state),
            bytes: Some(Arc::from(state)),
        }
    }

    pub(super) fn bytes_held(&self) -> usize {
        self.bytes.as_ref().map_or(0, |bytes| bytes.len())
    }
}

impl Recorder {
    /// Marks the stage as one with hybrid memory: each turn must start from
    /// a reported recurrent state ([`Self::record_state`]).
    pub fn set_recurrent(&self, recurrent: bool) {
        self.recurrent.store(recurrent, Ordering::Relaxed);
    }

    /// Whether the stage keeps recurrent state ([`Self::set_recurrent`]).
    /// A turn that continues a cached prefix of such a stage gets no
    /// receipt: the state it started from is not recorded.
    pub fn is_recurrent(&self) -> bool {
        self.recurrent.load(Ordering::Relaxed)
    }

    /// The recurrent state the next turn of `seq` starts from, or `None`
    /// when the stage could not read it. A report for a turn already under
    /// way is ignored: the turn keeps the state it started from.
    pub fn record_state(&self, seq: i32, state: Option<&[u8]>) {
        let state = state.map(TurnState::new);
        let mut guard = self.lock();
        let log = &mut guard.seqs.entry(seq).or_default().live;
        if log.prefix.is_some() {
            return;
        }
        log.state = Some(state.ok_or_else(|| "the stage could not read it".to_string()));
    }
}

impl RecorderState {
    /// Drops the state bytes of committed turns no client can still
    /// challenge.
    pub(super) fn drop_unpinned_states(&mut self, now: Instant) {
        for turn in &mut self.committed {
            if turn.pin.holds(now) {
                continue;
            }
            if let Some(state) = turn.state.as_mut() {
                let held = state.bytes_held();
                state.bytes = None;
                turn.bytes -= held;
                self.committed_bytes -= held;
            }
        }
    }
}
