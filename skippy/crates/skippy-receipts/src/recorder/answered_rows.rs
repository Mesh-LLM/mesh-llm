//! The rows requests a stage answered for a turn.
//!
//! A rows request reads stored rows from the live KV cache under the
//! runtime's lock, so a client that repeats one could hold that lock for
//! as long as it likes. A stage keeps each answer with its turn and gives
//! it again for the same picks, and reads rows for at most
//! [`MAX_ROWS_REQUESTS_PER_RECEIPT`] requests of one receipt. A kept answer
//! also lets a client whose answer was lost ask again after the turn's pin
//! ended and the cells were written.

use anyhow::{Result, bail};

use super::{Recorder, turn_mut};
use crate::attention::opening::{BlockChallenge, BlockPick, RowProof, StageRows};

/// Rows requests of one receipt a stage reads rows for. A client asks
/// once per challenge; the rest covers a retry that failed before an
/// answer was kept.
pub const MAX_ROWS_REQUESTS_PER_RECEIPT: u32 = 4;

/// The rows requests of one committed turn.
#[derive(Default)]
pub(super) struct RowsRequests {
    /// Requests the stage read rows for.
    read: u32,
    answered: Vec<(Vec<BlockPick>, StageRows)>,
}

impl Recorder {
    /// The kept answer to `challenge`, if the stage answered it before.
    pub(super) fn answered_rows(&self, challenge: &BlockChallenge) -> Result<Option<StageRows>> {
        let state = self.lock();
        let turn = self.opening_turn(&state.committed, &challenge.receipt_id)?;
        Ok(turn
            .rows
            .answered
            .iter()
            .find(|(picks, _)| picks == &challenge.picks)
            .map(|(_, rows)| rows.clone()))
    }

    /// Counts a read of rows for a request of `receipt_id`, or refuses it
    /// past [`MAX_ROWS_REQUESTS_PER_RECEIPT`].
    pub(super) fn count_rows_read(&self, receipt_id: &str) -> Result<()> {
        let mut state = self.lock();
        let turn = self.opening_turn_mut(&mut state.committed, receipt_id)?;
        if turn.rows.read >= MAX_ROWS_REQUESTS_PER_RECEIPT {
            bail!(
                "the stage already read rows for {MAX_ROWS_REQUESTS_PER_RECEIPT} requests of \
                 receipt {receipt_id}"
            );
        }
        turn.rows.read += 1;
        Ok(())
    }

    /// Keeps the answer to `challenge` with its turn, counted in the
    /// turn's bytes.
    pub(super) fn keep_rows(&self, challenge: &BlockChallenge, rows: &StageRows) {
        let mut guard = self.lock();
        let state = &mut *guard;
        let Some(turn) = turn_mut(&mut state.committed, &challenge.receipt_id) else {
            return;
        };
        let bytes = rows_bytes(rows);
        turn.rows
            .answered
            .push((challenge.picks.clone(), rows.clone()));
        turn.bytes += bytes;
        state.committed_bytes += bytes;
    }
}

fn rows_bytes(rows: &StageRows) -> usize {
    let paths = |path: &[String]| path.iter().map(String::len).sum::<usize>();
    rows.rows
        .iter()
        .map(|row| {
            let stored: usize = row
                .stored
                .iter()
                .map(|t| t.values.as_ref().map_or(0, String::len))
                .sum();
            let proof = match &row.proof {
                RowProof::Turn { rest, path } => rest.len() + paths(path),
                RowProof::Prefix {
                    cell_path,
                    leaves,
                    prefix_path,
                    ..
                } => paths(cell_path) + leaves.len() + paths(prefix_path),
            };
            stored + proof
        })
        .sum()
}
