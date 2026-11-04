//! Which KV cells a stage keeps out of later requests' way, so a turn a
//! client may still challenge can be opened.
//!
//! A pinned cell is skipped while a request fits elsewhere, so the cache's
//! attention window grows by the pinned cells and every later request
//! reads them as masked rows. On back-to-back Qwen3-4B requests, pinning
//! every turn for 30 seconds slowed decode by about 5%. So a stage pins a
//! turn only while a client can still challenge it, and once challenged,
//! until the client has the rows of its attention blocks.

use std::collections::BTreeSet;
use std::time::{Duration, Instant};

use super::{KvCell, PrefixCell, Recorder, attention_opening::written_cells, turn_mut};
use crate::attention::opening::ROWS_PIN;

/// How long after its commit an unchallenged turn stays pinned: the
/// client's 10-second challenge window, and 2 seconds for the response to
/// reach it and the challenge to come back.
pub const CHALLENGE_PIN: Duration = Duration::from_secs(12);

/// Whether a committed turn's cells are pinned.
#[derive(Clone, Copy, Debug)]
pub(super) struct Pin {
    committed_at: Instant,
    opened_at: Option<Instant>,
    released: bool,
}

impl Pin {
    pub(super) fn new(committed_at: Instant) -> Self {
        Self {
            committed_at,
            opened_at: None,
            released: false,
        }
    }

    pub(super) fn holds(&self, now: Instant) -> bool {
        if self.released {
            return false;
        }
        match self.opened_at {
            Some(at) => now.saturating_duration_since(at) < ROWS_PIN,
            None => now.saturating_duration_since(self.committed_at) < CHALLENGE_PIN,
        }
    }

    /// The turn was opened: with attention partials, it stays pinned until
    /// their rows are opened; without, nothing will ask for its rows.
    pub(super) fn opened(&mut self, now: Instant, partials: bool) {
        if partials {
            self.opened_at = Some(now);
        } else {
            self.released = true;
        }
    }

    pub(super) fn release(&mut self) {
        self.released = true;
    }

    /// When the pin stops holding by itself, while it holds.
    fn expiry(&self) -> Option<Instant> {
        if self.released {
            return None;
        }
        Some(match self.opened_at {
            Some(at) => at + ROWS_PIN,
            None => self.committed_at + CHALLENGE_PIN,
        })
    }
}

impl Recorder {
    /// The cells that a pinned turn's opening would read and that still
    /// hold what the turn read or wrote. The stage pins them, so a later
    /// request writes other cells while it fits there.
    pub fn pinned_cells(&self, now: Instant) -> Vec<KvCell> {
        let state = self.lock();
        let mut pinned = BTreeSet::new();
        for turn in state.committed.iter().filter(|t| t.pin.holds(now)) {
            let written = written_cells(turn);
            let own = turn
                .cells
                .iter()
                .zip(turn.position_start..)
                .filter_map(|(c, pos)| {
                    Some(PrefixCell {
                        pos: i32::try_from(pos).ok()?,
                        stream: state.cells.streams().of(turn.seq),
                        cell: *written.get(&pos)?,
                        digest: c.digest,
                    })
                });
            pinned.extend(
                turn.prefix
                    .cells
                    .iter()
                    .copied()
                    .chain(own)
                    .filter(|c| state.cells.holds(c))
                    .map(|c| KvCell {
                        stream: c.stream,
                        cell: c.cell,
                    }),
            );
        }
        pinned.into_iter().collect()
    }

    /// When the next pinned turn's pin runs out, so the stage can pin anew
    /// then and not hold the turn's cells until its next commit. `None`
    /// when no turn is pinned.
    pub fn next_pin_expiry(&self, now: Instant) -> Option<Instant> {
        self.lock()
            .committed
            .iter()
            .filter(|turn| turn.pin.holds(now))
            .filter_map(|turn| turn.pin.expiry())
            .min()
    }

    /// Stops pinning a turn's cells: no client can challenge it, or the
    /// client has its rows.
    pub fn release_pin(&self, receipt_id: &str) {
        let mut state = self.lock();
        if let Some(turn) = turn_mut(&mut state.committed, receipt_id) {
            turn.pin.release();
        }
        state.drop_unpinned_states(Instant::now());
    }
}
