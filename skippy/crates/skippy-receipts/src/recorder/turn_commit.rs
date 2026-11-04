//! Commits a recorded turn: takes the turn's log, by the turn's identity,
//! and seals its leaves under a Merkle root.

use std::ops::Range;
use std::time::Instant;

use anyhow::{Context, Result, bail};

use super::cells::TurnPrefix;
use super::event_log::{Event, ubatch_starts};
use super::{CommittedTurn, Recorder, SeqLog, SeqState, TurnCell, TurnId, cells, pins, recurrent};
use crate::commitment::{Hash, MerkleTree, prefix_tokens_digest};
use crate::wire::{PrefixCommitment, StageCommitment};

impl Recorder {
    /// Commits the leaves that turn `turn` of `seq` recorded in `positions`
    /// as receipt `receipt_id`. Fails, taking nothing, unless a log of that
    /// very turn covers the positions: the sequence's live log when it
    /// still records the turn, else the log kept when the turn ended. A
    /// receipt id committed before names the new turn from here on.
    pub fn commit(
        &self,
        receipt_id: &str,
        seq: i32,
        turn: TurnId,
        positions: Range<u32>,
    ) -> Result<StageCommitment> {
        if positions.is_empty() {
            bail!("receipt {receipt_id}: empty position range");
        }
        let taken = self.take_covering_turn(receipt_id, seq, turn, &positions)?;
        // hashing the turn does not hold the lock the recording waits on
        let (commitment, committed) = self.seal(receipt_id, seq, positions, taken)?;
        let now = Instant::now();
        let mut state = self.lock();
        // a reused id names a new turn; the old one and its openings are gone
        state.discard(receipt_id);
        state.committed_bytes += committed.bytes;
        state.committed.push_back(committed);
        state.drop_unpinned_states(now);
        state.drop_stale_logs(now);
        state.enforce_window(self.window_bytes);
        Ok(commitment)
    }

    /// Forgets the committed turn of `receipt_id`: its chain did not
    /// commit, so no receipt names it and nothing will open it.
    pub fn discard(&self, receipt_id: &str) {
        self.lock().discard(receipt_id);
    }

    /// Takes the recorded turn `turn` of `seq` that covers `positions`.
    fn take_covering_turn(
        &self,
        receipt_id: &str,
        seq: i32,
        turn: TurnId,
        positions: &Range<u32>,
    ) -> Result<TakenTurn> {
        let slots = self.slots();
        let recurrent = self.is_recurrent();
        let mut guard = self.lock();
        let state = &mut *guard;
        let last_error = state
            .last_error
            .as_ref()
            .map(|e| format!(" ({e})"))
            .unwrap_or_default();
        let entry = state.seqs.get_mut(&seq).with_context(|| {
            format!(
                "receipt {receipt_id}: stage {} recorded nothing for sequence {seq}{last_error}",
                self.stage_index,
            )
        })?;
        let taken = if entry.live.covers(turn, positions) {
            take_turn(&mut entry.live, positions, slots, recurrent)
        } else if let Some(index) = entry
            .retired
            .iter()
            .rposition(|retired| retired.log.covers(turn, positions))
        {
            // the turn ended before its commit: its log is used up here
            let mut retired = entry.retired.remove(index).expect("an index of the queue");
            state.retired_bytes -= retired.bytes;
            take_turn(&mut retired.log, positions, slots, recurrent)
        } else {
            bail!(
                "receipt {receipt_id}: stage {}: {}{last_error}",
                self.stage_index,
                uncovered(entry, seq, turn, positions)
            );
        };
        // a client may challenge a position of the prefix as soon as it
        // holds the receipt
        let taken = taken.and_then(|turn| {
            state.cells.check_lineage(&turn.prefix)?;
            Ok(turn)
        });
        taken.map_err(|error| {
            anyhow::anyhow!(
                "receipt {receipt_id}: stage {}: {error:#}{last_error}",
                self.stage_index
            )
        })
    }

    /// The commitment of a taken turn, and the turn as the stage keeps it.
    fn seal(
        &self,
        receipt_id: &str,
        seq: i32,
        positions: Range<u32>,
        turn: TakenTurn,
    ) -> Result<(StageCommitment, CommittedTurn)> {
        let tree = MerkleTree::new(turn.leaves)?;
        // the stage that holds layer 0 knows the tokens of the prefix
        let tokens = (self.layers.start == 0)
            .then(|| hex::encode(prefix_tokens_digest(&turn.prefix.tokens)));
        let prefix = cells::prefix_root(&turn.prefix)?.map(|root| PrefixCommitment {
            root: hex::encode(root),
            positions: u32::try_from(turn.prefix.len()).unwrap_or(u32::MAX),
            tokens,
        });
        let commitment = StageCommitment {
            stage_index: self.stage_index,
            layer_start: self.layers.start,
            layer_end: self.layers.end,
            root: hex::encode(tree.root()),
            leaf_count: tree.leaf_count(),
            backend: self.backend.clone(),
            cache_types: turn.cache_types,
            prefix,
            ubatches: ubatch_starts(&turn.events, seq),
            recurrent_state: turn.state.as_ref().map(|s| hex::encode(s.digest)),
        };
        let bytes = tree.bytes()
            + turn
                .cells
                .iter()
                .map(|c| (c.rests.len() + 1) * size_of::<Hash>())
                .sum::<usize>()
            + turn.prefix.bytes()
            + turn.events.iter().map(Event::bytes).sum::<usize>()
            + turn
                .state
                .as_ref()
                .map_or(0, recurrent::TurnState::bytes_held);
        let committed = CommittedTurn {
            receipt_id: receipt_id.to_owned(),
            seq,
            position_start: positions.start,
            position_end: positions.end,
            prefix: turn.prefix,
            events: turn.events,
            tree,
            cells: turn.cells,
            attention: Vec::new(),
            bytes,
            replays: 0,
            pin: pins::Pin::new(Instant::now()),
            state: turn.state,
        };
        Ok((commitment, committed))
    }
}

/// Why no log of `turn` covers `positions` of `seq`.
fn uncovered(entry: &SeqState, seq: i32, turn: TurnId, positions: &Range<u32>) -> String {
    let known = entry.live.turn == turn || entry.retired.iter().any(|r| r.log.turn == turn);
    if known {
        format!("turn {turn} of sequence {seq} recorded no leaves covering positions {positions:?}")
    } else {
        format!(
            "sequence {seq} holds no log of turn {turn}: it records turn {} now, and the log of \
             a turn that ended uncommitted is kept only for a while",
            entry.live.turn
        )
    }
}

/// What a commit takes from a sequence's log.
struct TakenTurn {
    leaves: Vec<Hash>,
    cells: Vec<TurnCell>,
    prefix: TurnPrefix,
    events: Vec<Event>,
    cache_types: Vec<String>,
    state: Option<recurrent::TurnState>,
}

/// Takes the pending turn out of `log`, which then waits for the next turn.
/// A turn that cannot be committed is dropped as well.
fn take_turn(
    log: &mut SeqLog,
    positions: &Range<u32>,
    slots: u64,
    recurrent: bool,
) -> Result<TakenTurn> {
    let prefix = log.prefix.take();
    let state = log.state.take();
    let events = std::mem::take(&mut log.events);
    log.pending_start = None;
    if let Some(reason) = log.tainted.take() {
        bail!("the turn cannot be replayed: {reason}");
    }
    if let Some(&quantized) = log
        .cache_types
        .iter()
        .find(|&&stored| crate::cache_types::audited(stored).is_none())
    {
        // an honest stage would fail the loose attention check
        // ([`crate::cache_types`])
        bail!(
            "the stage stores KV rows as {}, and no audit covers a quantized KV cache",
            crate::cache_types::name(quantized)
        );
    }
    let prefix = match prefix {
        Some(Ok(prefix)) => prefix,
        Some(Err(reason)) => bail!("the turn's cached prefix is unknown: {reason}"),
        None => bail!("no turn is pending"),
    };
    let state = match (recurrent, state) {
        (false, _) => None,
        (true, Some(Ok(state))) => Some(state),
        (true, Some(Err(reason))) => {
            bail!("the recurrent state the turn started from is unknown: {reason}")
        }
        (true, None) => bail!("the stage reported no recurrent state for the turn's start"),
    };
    if recurrent && !prefix.is_empty() {
        // a client may challenge a prefix position, which the stage opens by
        // replaying the ubatch that wrote it; the recurrent state that
        // ubatch started from is gone
        bail!("a stage with recurrent state cannot open the cached prefix of a turn");
    }
    let mut leaves = Vec::with_capacity(positions.len() * usize::try_from(slots)?);
    let mut cells = Vec::with_capacity(positions.len());
    for position in positions.clone() {
        let committed = log
            .leaves
            .get(&position)
            .with_context(|| format!("computed nothing at position {position}"))?;
        leaves.extend_from_slice(&committed.leaves);
        cells.push(TurnCell {
            digest: committed.cell,
            rests: committed.rests.clone(),
        });
    }
    let cache_types = log
        .cache_types
        .iter()
        .map(|&t| crate::ggml_types::type_name(t as u32))
        .collect();
    Ok(TakenTurn {
        leaves,
        cells,
        prefix,
        events,
        cache_types,
        state,
    })
}
