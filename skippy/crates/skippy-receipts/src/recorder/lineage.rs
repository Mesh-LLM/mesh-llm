//! The ubatches that wrote the cells a stage's KV cache holds, and stamps
//! that tell whether the cells below a ubatch are still the ones it ran on.
//!
//! A client may challenge a position of a turn's cached prefix. The stage
//! opens it by replaying the one ubatch that wrote the position, seeded
//! with the cells the sequence holds below that ubatch. That reproduces the
//! ubatch only when those cells are the ones the ubatch read. A sequence's
//! cells change in few ways, so a number per sequence is enough to tell:
//!
//! - an empty sequence has the stamp [`EMPTY`];
//! - a ubatch that appends to a sequence notes the stamp it found
//!   ([`Run::before`]) and leaves a stamp of that one and of itself
//!   ([`Run::after`]);
//! - a sequence cut back to a position has the stamp its cells had when the
//!   ubatch of that position ended, or a stamp of that one and the position
//!   when the cut falls inside the ubatch ([`Runs::stamp_through`]);
//! - any other change leaves the sequence without a stamp
//!   ([`Shape::UNREPLAYABLE`]) until it is emptied.
//!
//! Two sequences with the same stamp hold cells written by the same
//! ubatches in the same order. So a prefix position can be replayed when
//! the stamp its ubatch found is the stamp of the cells below it now.
//!
//! A stage after the first keeps the input rows of every recorded ubatch,
//! one residual row per token. The cells hold those ubatches alive, so the
//! rows are kept within a byte budget: past it, the oldest ubatch is
//! dropped, and its cells can no longer be opened as a prefix.

use std::collections::{HashMap, VecDeque};
use std::ops::Range;
use std::sync::Arc;

use super::UbatchRecord;

/// Identifies the cells a sequence holds, in order.
pub(super) type Stamp = u64;

/// The stamp of a sequence that holds no cells.
pub(super) const EMPTY: Stamp = 0;

const STAMP_DOMAIN: &[u8] = b"skippy-receipt-cell-stamp-v1";
const APPENDED: u8 = 0;
const CUT: u8 = 1;
const UNFOLLOWED: u8 = 2;

fn stamp(kind: u8, a: u64, b: u64) -> Stamp {
    let mut h = blake3::Hasher::new();
    h.update(STAMP_DOMAIN);
    h.update(&[kind]);
    h.update(&a.to_le_bytes());
    h.update(&b.to_le_bytes());
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&h.finalize().as_bytes()[..8]);
    Stamp::from_le_bytes(bytes)
}

/// What the mirror knows of the cells a sequence holds, as a whole.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Shape {
    pub stamp: Stamp,
    /// The sequence holds positions from 0 without a gap, each appended to
    /// the ones below it, so `stamp` identifies them.
    pub replayable: bool,
}

impl Shape {
    /// A sequence whose cells a replay of one ubatch cannot start from.
    pub const UNREPLAYABLE: Self = Self {
        stamp: EMPTY,
        replayable: false,
    };
}

/// The positions one ubatch computed for one sequence.
pub(super) struct Run {
    /// The ubatch as recorded, until the input budget drops it.
    pub record: Option<Arc<UbatchRecord>>,
    /// The positions of the sequence the ubatch computed, in column order.
    pub positions: Range<i32>,
    /// The stamp of the sequence's cells below the run when it was
    /// computed, or `None` when the run did not append to them.
    pub before: Option<Stamp>,
    /// The stamp of the sequence's cells through the run's last position.
    pub after: Stamp,
    /// Cells that still hold a position of the run.
    cells: usize,
    /// Bytes of input rows the run counts against the budget.
    input_bytes: usize,
}

/// The runs that wrote the cells of a mirror, by an id that grows with
/// time.
pub(super) struct Runs {
    runs: HashMap<u64, Run>,
    next: u64,
    /// Runs that hold input rows, oldest first. An id may name a run that
    /// is gone.
    with_input: VecDeque<u64>,
    input_bytes: usize,
    input_budget: usize,
}

impl Default for Runs {
    fn default() -> Self {
        Self {
            runs: HashMap::new(),
            next: 0,
            with_input: VecDeque::new(),
            input_bytes: 0,
            input_budget: super::DEFAULT_LINEAGE_INPUT_BYTES,
        }
    }
}

impl Runs {
    pub fn set_input_budget(&mut self, bytes: usize) {
        self.input_budget = bytes;
        self.enforce_budget();
    }

    pub fn get(&self, run: u64) -> Option<&Run> {
        self.runs.get(&run)
    }

    /// Forgets every run. Ids are not used again.
    pub fn clear(&mut self) {
        self.runs.clear();
        self.with_input.clear();
        self.input_bytes = 0;
    }

    /// A ubatch computed `positions` of a sequence in `columns` of its
    /// columns, on cells stamped `before`, or on cells it did not append to.
    /// Returns the run's id and the stamp it leaves.
    pub fn begin(
        &mut self,
        record: &Arc<UbatchRecord>,
        columns: usize,
        positions: Range<i32>,
        before: Option<Stamp>,
    ) -> (u64, Stamp) {
        let id = self.next;
        self.next += 1;
        let after = match before {
            Some(before) => stamp(APPENDED, before, id),
            None => stamp(UNFOLLOWED, id, 0),
        };
        let input_bytes = record
            .input
            .as_ref()
            .map_or(0, |input| columns * input.row_bytes);
        if input_bytes > 0 {
            self.input_bytes += input_bytes;
            self.with_input.push_back(id);
        }
        self.runs.insert(
            id,
            Run {
                record: Some(Arc::clone(record)),
                positions,
                before,
                after,
                cells: 0,
                input_bytes,
            },
        );
        (id, after)
    }

    /// One more cell holds a position of `run`.
    pub fn hold(&mut self, run: u64) {
        if let Some(run) = self.runs.get_mut(&run) {
            run.cells += 1;
        }
    }

    /// A cell that held a position of `run` holds it no longer. A run no
    /// cell holds is forgotten.
    pub fn release(&mut self, run: u64) {
        let Some(entry) = self.runs.get_mut(&run) else {
            return;
        };
        entry.cells = entry.cells.saturating_sub(1);
        if entry.cells == 0 {
            if entry.record.is_some() {
                self.input_bytes -= entry.input_bytes;
            }
            self.runs.remove(&run);
        }
    }

    /// The stamp of a sequence that holds the cells `run` ran on and the
    /// run's positions through `position`.
    pub fn stamp_through(&self, run: u64, position: i32) -> Option<Stamp> {
        let run = self.runs.get(&run)?;
        Some(if position + 1 == run.positions.end {
            run.after
        } else {
            stamp(CUT, run.after, u64::from(position.unsigned_abs()))
        })
    }

    /// Drops the recorded ubatch of the oldest runs while the input rows
    /// the runs hold are past the budget.
    pub fn enforce_budget(&mut self) {
        while self.input_bytes > self.input_budget {
            let Some(id) = self.with_input.pop_front() else {
                break;
            };
            if let Some(run) = self.runs.get_mut(&id)
                && run.record.take().is_some()
            {
                self.input_bytes -= run.input_bytes;
            }
        }
        // ids of runs that are gone would otherwise pile up
        if self.with_input.len() > 2 * self.runs.len() + 64 {
            self.with_input.retain(|id| {
                self.runs
                    .get(id)
                    .is_some_and(|run| run.record.is_some() && run.input_bytes > 0)
            });
        }
    }

    /// Bytes of input rows the runs hold.
    #[cfg(test)]
    pub fn input_bytes(&self) -> usize {
        self.input_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recorder::StageInput;

    fn record(columns: usize, row_bytes: usize) -> Arc<UbatchRecord> {
        Arc::new(UbatchRecord {
            token: vec![0; columns],
            seq: vec![1; columns],
            pos: (0..columns as i32).collect(),
            output: vec![0; columns],
            cell: (0..columns as u32).collect(),
            n_kv: 64,
            input: (row_bytes > 0).then(|| StageInput {
                rows: vec![0; columns * row_bytes],
                row_bytes,
                ggml_type: 0,
            }),
        })
    }

    #[test]
    fn a_stamp_names_the_runs_below_and_where_they_were_cut() {
        let mut runs = Runs::default();
        let (a, after_a) = runs.begin(&record(3, 0), 3, 0..3, Some(EMPTY));
        let (b, after_b) = runs.begin(&record(2, 0), 2, 3..5, Some(after_a));
        assert_eq!(runs.stamp_through(a, 2), Some(after_a));
        assert_eq!(runs.stamp_through(b, 4), Some(after_b));
        // a cut inside a run is a state of its own, one per position
        let cut = runs.stamp_through(a, 1).unwrap();
        assert_ne!(cut, after_a);
        assert_ne!(cut, runs.stamp_through(a, 0).unwrap());
        // the same ubatch on other cells leaves another stamp
        let (_, on_cut) = runs.begin(&record(2, 0), 2, 2..4, Some(cut));
        assert_ne!(on_cut, after_b);
        // a run that did not append has a stamp nothing else leaves
        let (c, alone) = runs.begin(&record(1, 0), 1, 9..10, None);
        assert_eq!(runs.get(c).unwrap().before, None);
        assert_ne!(alone, EMPTY);
        assert_eq!(runs.stamp_through(99, 0), None);
    }

    #[test]
    fn a_run_lives_while_a_cell_holds_it() {
        let mut runs = Runs::default();
        let (a, _) = runs.begin(&record(2, 8), 2, 0..2, Some(EMPTY));
        runs.hold(a);
        runs.hold(a);
        assert_eq!(runs.input_bytes(), 16);
        runs.release(a);
        assert!(runs.get(a).is_some());
        runs.release(a);
        assert!(runs.get(a).is_none());
        assert_eq!(runs.input_bytes(), 0);
    }

    #[test]
    fn the_oldest_input_rows_go_first_past_the_budget() {
        let mut runs = Runs::default();
        runs.set_input_budget(40);
        let ids: Vec<u64> = (0..3)
            .map(|i| {
                let (id, _) = runs.begin(&record(2, 8), 2, 2 * i..2 * i + 2, None);
                runs.hold(id);
                runs.enforce_budget();
                id
            })
            .collect();
        assert_eq!(runs.input_bytes(), 32);
        assert!(runs.get(ids[0]).unwrap().record.is_none());
        assert!(runs.get(ids[1]).unwrap().record.is_some());
        // a run without input rows costs nothing and is never dropped
        let (free, _) = runs.begin(&record(64, 0), 64, 6..70, None);
        runs.hold(free);
        runs.enforce_budget();
        assert!(runs.get(free).unwrap().record.is_some());
        // a dropped run frees nothing twice
        runs.release(ids[0]);
        assert_eq!(runs.input_bytes(), 32);
    }
}
