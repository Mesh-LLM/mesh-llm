//! Node side: records each ubatch's committed leaves and a replay log per
//! native sequence, commits a turn to a Merkle root, and opens a challenge
//! by replaying the turn in a scratch context ("replay, not storage" in
//! RECEIPTS.md).
//!
//! Per position, a stage keeps only leaf hashes. Per ubatch, it keeps what
//! replay needs to reproduce it bit for bit: tokens, sequences, positions,
//! output rows, KV cells, n_kv, and the stage input rows of a stage after
//! the first. A turn's replay starts from the KV cells the turn attended to
//! below its first position, copied from the stage's cache, so it replays
//! only the turn's own ubatches, also after a shared cached prefix
//! ([`cells`]). A KV-cache change the replay cannot reproduce (a position
//! shift, a state load, compaction) makes the turn unreceiptable.
//!
//! A native sequence serves one request after another, so a log belongs to
//! a turn, not to a sequence: the session that computes a turn arms its
//! [`TurnId`] before its first ubatch ([`Recorder::begin_turn`]), and a
//! commit names that id. A commit never takes the log of another turn,
//! also when the sequence was reset and reused before the commit arrived.
//!
//! A replay costs about what serving the turn did, so a receipt is replayed
//! for at most [`MAX_REPLAYS_PER_RECEIPT`] distinct challenges, and a
//! repeated challenge is answered from the openings already made.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::ops::Range;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::commitment::{Hash, MerkleTree, slots_per_position};
use crate::layer_tensors::{TensorKind, UbatchColumns};
use crate::wire::{CannotOpen, CannotOpenReason};

/// Committed turns, and logs of turns that ended without a commit, are
/// dropped, oldest first, past this many bytes.
pub(crate) const DEFAULT_WINDOW_BYTES: usize = 1 << 30;
/// Input rows of recorded ubatches a stage after the first keeps for the
/// cells of its KV cache, so a turn that reads them as its cached prefix can
/// open one. Past this many bytes the oldest are dropped, and a turn that
/// reads their cells gets no receipt.
pub(crate) const DEFAULT_LINEAGE_INPUT_BYTES: usize = 1 << 30;
/// Distinct challenges a committed turn is replayed for, at most.
pub const MAX_REPLAYS_PER_RECEIPT: u32 = 4;
/// Logs of turns that ended before they were committed, kept per sequence.
const RETIRED_LOGS: usize = 4;
/// How long the log of a turn that ended without a commit waits for one. A
/// commit follows its turn at once, so a log this old has none coming.
const RETIRED_LOG_TTL: Duration = Duration::from_secs(60);
/// Token id of the dummy columns that stand in for other sequences in a replay.
const DUMMY_TOKEN: i32 = 0;

mod answered_rows;
mod attention_opening;
mod cells;
mod challenge_shape;
mod event_log;
mod leaves;
mod lineage;
mod openings;
mod pins;
mod prefix_opening;
mod recurrent;
mod recurrent_opening;
mod replay;
mod stage_opening;
mod turn_commit;
mod turn_seed;

use cells::{CellMirror, ColumnLeaves, TurnPrefix};
pub use cells::{KvCell, KvStreams, MemoryOp, PrefixCell};
use event_log::Event;
use leaves::PositionLeaves;
use openings::OpeningCache;
pub use pins::{CHALLENGE_PIN, OPENING_PIN};
pub use replay::{
    AnchorValue, ReplayAnchor, ReplayScope, ReplaySeed, ReplayStep, ReplayedLayer, ReplayedMatmul,
    ReplayedMixing, ReplayedPosition, Replayer, StoredRows,
};

/// Names one turn of a native sequence. The session that computes the turn
/// arms it before the turn's first ubatch ([`Recorder::begin_turn`]) and
/// carries it to the commit, which takes that turn's log and no other's.
/// The default names what a sequence records before any session armed one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TurnId(pub u64);

impl std::fmt::Display for TurnId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// Stage input rows of a ubatch, for a stage after the first.
#[derive(Clone, Debug, PartialEq)]
pub struct StageInput {
    pub rows: Vec<u8>,
    pub row_bytes: usize,
    /// ggml type of the rows.
    pub ggml_type: i32,
}

/// What replay needs to reproduce one computed ubatch, column per token.
#[derive(Clone, Debug, PartialEq)]
pub struct UbatchRecord {
    pub token: Vec<i32>,
    pub seq: Vec<i32>,
    pub pos: Vec<i32>,
    pub output: Vec<i8>,
    pub cell: Vec<u32>,
    pub n_kv: u32,
    pub input: Option<StageInput>,
}

impl UbatchRecord {
    fn bytes(&self) -> usize {
        self.token.len() * 21 + self.input.as_ref().map_or(0, |input| input.rows.len())
    }
}

#[derive(Default)]
struct SeqLog {
    /// The turn the log records.
    turn: TurnId,
    events: Vec<Event>,
    /// Bytes of [`Self::events`], kept as they arrive so the window counts
    /// a log before its turn commits ([`RecorderState::shed_live`]).
    event_bytes: usize,
    /// The log outgrew the window and its turn gets no receipt: it records
    /// no more ubatches until the sequence's next turn.
    shed: bool,
    /// What the stage committed per position.
    leaves: BTreeMap<u32, PositionLeaves>,
    /// First position recorded since the last commit.
    pending_start: Option<u32>,
    /// Why the pending turn cannot be replayed.
    tainted: Option<String>,
    /// A ubatch the recorder could not record ran while the log was live.
    /// The native report does not say which sequences it computed, so the
    /// turn commits only if its positions show it computed nothing the log
    /// lacks: they start where the turn did, and every one was recorded.
    missed: bool,
    /// The cells the pending turn attended to below its first position, as
    /// they were when its first ubatch was recorded, or why they are unknown.
    prefix: Option<Result<TurnPrefix, String>>,
    /// ggml types of the cache rows the stage stored.
    cache_types: BTreeSet<i32>,
    /// On a stage with hybrid memory, the recurrent state the pending turn
    /// started from, or why it is unknown.
    state: Option<Result<recurrent::TurnState, String>>,
}

impl SeqLog {
    fn of(turn: TurnId) -> Self {
        Self {
            turn,
            ..Self::default()
        }
    }

    fn push_event(&mut self, event: Event) {
        if self.shed {
            return;
        }
        self.event_bytes += event.bytes();
        self.events.push(event);
    }

    /// Takes the events, for a commit or a new turn.
    fn take_events(&mut self) -> Vec<Event> {
        self.event_bytes = 0;
        self.shed = false;
        std::mem::take(&mut self.events)
    }

    /// Bytes a live log holds that a commit moves into its committed turn:
    /// its ubatches, with their input rows, and its recurrent state.
    fn live_bytes(&self) -> usize {
        let state = match &self.state {
            Some(Ok(state)) => state.bytes_held(),
            _ => 0,
        };
        self.event_bytes + state
    }

    fn pending(&self) -> Option<Range<u32>> {
        let start = self.pending_start?;
        let (&last, _) = self.leaves.range(start..).next_back()?;
        Some(start..last + 1)
    }

    /// Whether the log holds a pending turn of `turn` that recorded every
    /// one of `positions`, and, after a ubatch it could not record, starts
    /// where they do ([`Self::missed`]). A commit is checked against it
    /// before it takes the log, so one that names positions the log cannot
    /// commit leaves the log for the commit that can.
    fn covers(&self, turn: TurnId, positions: &Range<u32>) -> bool {
        self.turn == turn
            && self.pending_start.is_some_and(|start| {
                start <= positions.start && (!self.missed || start == positions.start)
            })
            && positions
                .clone()
                .all(|position| self.leaves.contains_key(&position))
    }

    /// Bytes the log holds.
    fn bytes(&self) -> usize {
        let events: usize = self.events.iter().map(Event::bytes).sum();
        let leaves: usize = self.leaves.values().map(PositionLeaves::bytes).sum();
        let state = match &self.state {
            Some(Ok(state)) => state.bytes_held(),
            _ => 0,
        };
        events + leaves + state
    }
}

/// The log of a turn that ended before it was committed.
struct RetiredLog {
    log: SeqLog,
    retired_at: Instant,
    bytes: usize,
}

#[derive(Default)]
struct SeqState {
    live: SeqLog,
    /// Logs of turns that ended before they were committed, newest last.
    /// Each stays addressable by its turn until it is committed, expires or
    /// is crowded out.
    retired: VecDeque<RetiredLog>,
}

impl SeqState {
    fn taint(&mut self, reason: &str) {
        self.live.tainted.get_or_insert_with(|| reason.to_string());
    }
}

struct CommittedTurn {
    receipt_id: String,
    seq: i32,
    position_start: u32,
    position_end: u32,
    /// The cells the replay starts from, with what the stage committed at
    /// each of their positions.
    prefix: TurnPrefix,
    /// The runs that wrote the prefix cells, whose ubatches a replay of a
    /// prefix position needs ([`CellMirror::lineage_runs`]).
    lineage_runs: Vec<u64>,
    events: Vec<Event>,
    tree: MerkleTree,
    /// By position from `position_start`, the cell the turn wrote it to.
    cells: Vec<TurnCell>,
    /// The attention picks the stage opened partials for.
    attention: Vec<attention_opening::OpenedAttention>,
    /// The rows requests answered, so a repeated request reads no rows
    /// again and a lost answer can be asked for after the pin ends.
    rows: answered_rows::RowsRequests,
    bytes: usize,
    /// Replays started for this turn.
    replays: u32,
    pin: pins::Pin,
    /// On a stage with hybrid memory, the recurrent state the turn started
    /// from.
    state: Option<recurrent::TurnState>,
}

/// A turn position's cell: its digest, and per layer the half of the
/// mixing leaf that opens the stored rows ([`crate::commitment::mixing_rest`]).
struct TurnCell {
    digest: Hash,
    rests: Vec<Hash>,
}

#[derive(Default)]
struct RecorderState {
    seqs: HashMap<i32, SeqState>,
    cells: CellMirror,
    committed: VecDeque<CommittedTurn>,
    committed_bytes: usize,
    /// Bytes of every retired log.
    retired_bytes: usize,
    openings: OpeningCache,
    /// The cells the stage was last told to pin ([`Recorder::pinned_cells`]).
    pins: BTreeSet<KvCell>,
    /// Cells in a stream of the stage's KV cache; 0 when unknown
    /// ([`Recorder::set_pin_budget`]).
    pin_budget: u32,
    last_error: Option<String>,
    /// Layers whose token mixing reads a recurrent state.
    state_layers: BTreeSet<u32>,
}

impl RecorderState {
    /// Ends the live log of `seq`, whose next log is of `turn` (of the same
    /// turn when `None`). A turn the log holds that was not committed is
    /// kept, addressable by its turn, for the commit that may still come.
    fn retire(&mut self, seq: i32, turn: Option<TurnId>, now: Instant) {
        let entry = self.seqs.entry(seq).or_default();
        let turn = turn.unwrap_or(entry.live.turn);
        let old = std::mem::replace(&mut entry.live, SeqLog::of(turn));
        if old.pending().is_none() {
            return;
        }
        let bytes = old.bytes();
        self.retired_bytes += bytes;
        entry.retired.push_back(RetiredLog {
            log: old,
            retired_at: now,
            bytes,
        });
        while entry.retired.len() > RETIRED_LOGS {
            if let Some(dropped) = entry.retired.pop_front() {
                self.retired_bytes -= dropped.bytes;
            }
        }
    }

    /// Drops the retired logs no commit is still coming for.
    fn drop_stale_logs(&mut self, now: Instant) {
        for entry in self.seqs.values_mut() {
            while entry
                .retired
                .front()
                .is_some_and(|log| now.saturating_duration_since(log.retired_at) >= RETIRED_LOG_TTL)
            {
                if let Some(dropped) = entry.retired.pop_front() {
                    self.retired_bytes -= dropped.bytes;
                }
            }
        }
    }

    /// Drops the oldest retired log of any sequence; `false` when there is
    /// none.
    fn drop_oldest_retired(&mut self) -> bool {
        let oldest = self
            .seqs
            .iter()
            .filter_map(|(&seq, entry)| Some((entry.retired.front()?.retired_at, seq)))
            .min();
        let Some((_, seq)) = oldest else {
            return false;
        };
        if let Some(dropped) = self
            .seqs
            .get_mut(&seq)
            .and_then(|entry| entry.retired.pop_front())
        {
            self.retired_bytes -= dropped.bytes;
        }
        true
    }

    /// Bytes of every live log ([`SeqLog::live_bytes`]).
    fn live_bytes(&self) -> usize {
        self.seqs
            .values()
            .map(|entry| entry.live.live_bytes())
            .sum()
    }

    fn held_bytes(&self) -> usize {
        self.committed_bytes + self.retired_bytes + self.live_bytes()
    }

    /// Keeps what the recorder holds within `window` bytes as far as it may:
    /// logs nobody committed go first, then committed turns no client can
    /// still challenge, oldest first. A turn inside its challenge window
    /// ([`pins::Pin::holds`]) is never dropped for room; a new turn that
    /// does not fit beside them gets no receipt ([`Self::make_room`]), and
    /// a live log that does not fit is shed ([`Self::shed_live`]).
    fn enforce_window(&mut self, window: usize, now: Instant) {
        while self.held_bytes() > window {
            if self.drop_oldest_retired() {
                continue;
            }
            let Some(index) = self.committed.iter().position(|t| !t.pin.holds(now)) else {
                break;
            };
            if let Some(old) = self.committed.remove(index) {
                self.committed_bytes -= old.bytes;
                self.openings.forget(&old.receipt_id);
            }
        }
    }

    /// Sheds live logs, largest first, while what the recorder holds is
    /// past `window` after [`Self::enforce_window`]: a long or wide prefill
    /// must not hold more than the window before its turn commits. A shed
    /// log's turn gets no receipt; the sequence's next turn records again.
    fn shed_live(&mut self, window: usize) {
        while self.held_bytes() > window {
            let largest = self
                .seqs
                .values_mut()
                .map(|entry| &mut entry.live)
                .filter(|log| log.live_bytes() > 0)
                .max_by_key(|log| log.live_bytes());
            let Some(log) = largest else {
                return;
            };
            log.tainted.get_or_insert_with(|| {
                format!("its replay log outgrew the recorder's window of {window} bytes")
            });
            log.take_events();
            log.shed = true;
            log.state = None;
        }
    }

    /// Makes room for a turn of `bytes` within `window`, or fails, dropping
    /// nothing a client can still challenge.
    fn make_room(&mut self, window: usize, bytes: usize, now: Instant) -> anyhow::Result<()> {
        self.enforce_window(window.saturating_sub(bytes), now);
        let held = self.held_bytes();
        if held + bytes > window {
            anyhow::bail!(
                "the stage holds {held} bytes of turns a client can still challenge, of \
                 {window} at most; this turn of {bytes} bytes gets no receipt"
            );
        }
        Ok(())
    }

    /// The runs whose ubatches the lineage budget keeps: those a turn a
    /// client can still challenge replays a prefix position from.
    fn lineage_in_window(&self, now: Instant) -> BTreeSet<u64> {
        self.committed
            .iter()
            .filter(|turn| turn.pin.holds(now))
            .flat_map(|turn| turn.lineage_runs.iter().copied())
            .collect()
    }

    /// Forgets the committed turns under `receipt_id` and their openings.
    fn discard(&mut self, receipt_id: &str) {
        let mut freed = 0;
        self.committed.retain(|turn| {
            let keep = turn.receipt_id != receipt_id;
            if !keep {
                freed += turn.bytes;
            }
            keep
        });
        self.committed_bytes -= freed;
        self.openings.forget(receipt_id);
    }
}

/// The committed turn under `receipt_id`.
fn turn<'a>(committed: &'a VecDeque<CommittedTurn>, receipt_id: &str) -> Option<&'a CommittedTurn> {
    committed.iter().rev().find(|t| t.receipt_id == receipt_id)
}

fn turn_mut<'a>(
    committed: &'a mut VecDeque<CommittedTurn>,
    receipt_id: &str,
) -> Option<&'a mut CommittedTurn> {
    committed
        .iter_mut()
        .rev()
        .find(|t| t.receipt_id == receipt_id)
}

pub struct Recorder {
    stage_index: u32,
    layers: Range<u32>,
    backend: String,
    /// The stage holds the model's final layer, so it opens the residual
    /// the output head reads.
    holds_output: bool,
    window_bytes: usize,
    /// The stage has hybrid memory ([`Self::set_recurrent`]).
    recurrent: AtomicBool,
    state: Mutex<RecorderState>,
}

impl Recorder {
    pub fn new(stage_index: u32, layers: Range<u32>, backend: impl Into<String>) -> Self {
        Self {
            stage_index,
            layers,
            backend: backend.into(),
            holds_output: false,
            window_bytes: DEFAULT_WINDOW_BYTES,
            recurrent: AtomicBool::new(false),
            state: Mutex::new(RecorderState::default()),
        }
    }

    /// Marks the stage as the one holding the model's final layer.
    pub fn holding_output(mut self, holds_output: bool) -> Self {
        self.holds_output = holds_output;
        self
    }

    /// Sets the bytes of committed turns and uncommitted logs the recorder
    /// keeps, in place of [`DEFAULT_WINDOW_BYTES`].
    pub fn with_window_bytes(mut self, window_bytes: usize) -> Self {
        self.window_bytes = window_bytes;
        self
    }

    /// Sets the bytes of input rows the recorder keeps for the cells of the
    /// stage's KV cache, in place of [`DEFAULT_LINEAGE_INPUT_BYTES`].
    pub fn with_lineage_input_bytes(self, bytes: usize) -> Self {
        self.lock().cells.set_input_budget(bytes);
        self
    }

    pub fn stage_index(&self) -> u32 {
        self.stage_index
    }

    pub fn layers(&self) -> Range<u32> {
        self.layers.clone()
    }

    fn slots(&self) -> u64 {
        slots_per_position(self.layers.start, self.layers.end)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, RecorderState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The error of a challenge this stage can no longer open.
    fn cannot_open(
        &self,
        receipt_id: &str,
        reason: CannotOpenReason,
        detail: impl Into<String>,
    ) -> anyhow::Error {
        anyhow::Error::new(CannotOpen {
            receipt_id: receipt_id.to_owned(),
            stage_index: self.stage_index,
            reason,
            detail: detail.into(),
        })
    }

    /// The committed turn a challenge of `receipt_id` opens.
    fn opening_turn<'a>(
        &self,
        committed: &'a VecDeque<CommittedTurn>,
        receipt_id: &str,
    ) -> anyhow::Result<&'a CommittedTurn> {
        turn(committed, receipt_id).ok_or_else(|| self.expired(receipt_id))
    }

    fn opening_turn_mut<'a>(
        &self,
        committed: &'a mut VecDeque<CommittedTurn>,
        receipt_id: &str,
    ) -> anyhow::Result<&'a mut CommittedTurn> {
        turn_mut(committed, receipt_id).ok_or_else(|| self.expired(receipt_id))
    }

    fn expired(&self, receipt_id: &str) -> anyhow::Error {
        self.cannot_open(
            receipt_id,
            CannotOpenReason::Expired,
            format!("receipt {receipt_id} is not in the opening window"),
        )
    }

    /// Whether the stage still holds the committed turn of `receipt_id`.
    pub fn holds_turn(&self, receipt_id: &str) -> bool {
        turn(&self.lock().committed, receipt_id).is_some()
    }

    /// A session starts computing turn `turn` on `seq`. Call it in report
    /// order, after every ubatch computed before the session took the
    /// sequence and before the session's first: the ubatches that follow
    /// are the turn's. A turn the sequence still holds uncommitted is kept
    /// for its own commit.
    pub fn begin_turn(&self, seq: i32, turn: TurnId) {
        let now = Instant::now();
        let mut state = self.lock();
        if state
            .seqs
            .get(&seq)
            .is_some_and(|entry| entry.live.turn == turn)
        {
            return;
        }
        state.retire(seq, Some(turn), now);
        state.drop_stale_logs(now);
        state.enforce_window(self.window_bytes, now);
    }

    /// Records one computed ubatch: the leaves of every column, and the
    /// ubatch in the replay log of every sequence it touched.
    pub fn record_ubatch(&self, record: UbatchRecord, columns: &dyn UbatchColumns) {
        let leaves =
            leaves::ubatch_leaves(self.layers.clone(), &record.pos, &record.token, columns);
        let cache_types = self.cache_types(columns);
        let state_layers: Vec<u32> = self
            .layers
            .clone()
            .filter(|&layer| columns.count(layer, TensorKind::MixingState) > 0)
            .collect();
        let record = Arc::new(record);
        let now = Instant::now();
        let mut guard = self.lock();
        let state = &mut *guard;
        // the lineage of a turn a client can still challenge outlasts the
        // input budget
        let protected = state.lineage_in_window(now);
        state.cells.protect_runs(protected);
        state.state_layers.extend(state_layers);
        let mut first: BTreeMap<i32, i32> = BTreeMap::new();
        for (&seq, &pos) in record.seq.iter().zip(&record.pos) {
            let p = first.entry(seq).or_insert(pos);
            *p = (*p).min(pos);
        }
        for (&seq, &pos) in &first {
            let log = &mut state.seqs.entry(seq).or_default().live;
            if log.prefix.is_none() {
                // a new turn: it replays from the cells it attends to now
                log.take_events();
                log.tainted = None;
                log.missed = false;
                let end = u32::try_from(pos).unwrap_or(0);
                log.prefix = Some(state.cells.prefix(seq, end).map_err(|e| format!("{e:#}")));
            }
            log.push_event(Event::Ubatch(record.clone()));
            log.cache_types.extend(&cache_types);
        }
        let mut committed = Vec::with_capacity(record.seq.len());
        for (col, leaf) in leaves.into_iter().enumerate() {
            let seq = state.seqs.entry(record.seq[col]).or_default();
            committed.push(match leaf {
                Ok((position, hashes)) => {
                    // the cell keeps the leaves, for a turn that reads it
                    // as its cached prefix
                    let column = ColumnLeaves {
                        cell: hashes.cell,
                        leaves: Arc::clone(&hashes.leaves),
                        position: hashes.position,
                    };
                    let log = &mut seq.live;
                    log.leaves.insert(position, hashes);
                    log.pending_start =
                        Some(log.pending_start.map_or(position, |p| p.min(position)));
                    Some(column)
                }
                Err(error) => {
                    seq.taint(&format!("{error:#}"));
                    None
                }
            });
        }
        state.cells.write_ubatch(&record, committed);
        state.enforce_window(self.window_bytes, now);
        state.shed_live(self.window_bytes);
    }

    fn cache_types(&self, columns: &dyn UbatchColumns) -> BTreeSet<i32> {
        self.layers
            .clone()
            .flat_map(|layer| {
                (0..columns.count(layer, TensorKind::MixingStored)).filter_map(move |index| {
                    columns
                        .column(layer, TensorKind::MixingStored, index, 0)
                        .map(|c| c.ggml_type)
                })
            })
            .collect()
    }

    /// The recorder lost track of the stage: a report it could not apply,
    /// or one it does not know. Every live sequence's pending turn is
    /// tainted, and every cell something unseen may have written is
    /// forgotten or exposed ([`CellMirror::unseen_write`]).
    pub fn unreceiptable(&self, reason: &str) {
        let mut state = self.lock();
        for seq in state.seqs.values_mut() {
            seq.taint(reason);
        }
        let RecorderState { cells, pins, .. } = &mut *state;
        cells.unseen_write(pins);
        state.last_error = Some(reason.to_string());
    }

    /// The stage computed one ubatch the native recorder could not receipt,
    /// such as one of image embeddings. The report names neither its
    /// sequences nor its cells, so the damage is bounded by what such a
    /// ubatch can touch:
    ///
    /// - The positions it computed are missing from the log of each
    ///   sequence it computed, so that sequence's turn has a gap, or its
    ///   prefix lacks cells, and gets no receipt. A turn of another
    ///   sequence attends only to its own cells and keeps its receipt.
    ///   Every live log is marked, so a turn that started before the ubatch
    ///   commits only from its first position ([`SeqLog::missed`]).
    /// - It wrote cells no sequence held: a free cell that is not pinned is
    ///   forgotten, and a pinned one, which the cache takes only when
    ///   nothing else fits, is checked before an opening reads it
    ///   ([`CellMirror::unseen_write`]).
    pub fn unrecorded_ubatch(&self, reason: &str) {
        let mut state = self.lock();
        for seq in state.seqs.values_mut() {
            seq.live.missed = true;
        }
        let RecorderState { cells, pins, .. } = &mut *state;
        cells.unseen_write(pins);
        state.last_error = Some(reason.to_string());
    }

    /// Sets how the stage's KV cache lays out its cells. Call it before the
    /// first ubatch: the mirror forgets the cells it knew.
    pub fn set_kv_streams(&self, streams: KvStreams) {
        self.lock().cells.set_streams(streams);
    }

    /// Mirrors a KV-cache operation (`skippy_receipt_memory_op`).
    pub fn memory_op(&self, op: MemoryOp, seq: i32, seq_dst: i32, p0: i32, p1: i32) {
        let now = Instant::now();
        let mut state = self.lock();
        let RecorderState { cells, pins, .. } = &mut *state;
        cells.memory_op(op, (seq, seq_dst), (p0, p1), pins);
        let all: Vec<i32> = state.seqs.keys().copied().collect();
        let targets = if seq < 0 { all.clone() } else { vec![seq] };
        let taint = |state: &mut RecorderState, seqs: &[i32], reason: &str| {
            for &s in seqs {
                state.seqs.entry(s).or_default().taint(reason);
            }
        };
        match op {
            MemoryOp::Clear => all.iter().for_each(|&s| state.retire(s, None, now)),
            MemoryOp::SeqRm if event_log::removes_all(p0, p1) => {
                targets.iter().for_each(|&s| state.retire(s, None, now));
            }
            MemoryOp::SeqRm => {
                for s in targets {
                    let log = &mut state.seqs.entry(s).or_default().live;
                    log.push_event(Event::SeqRm { p0, p1 });
                    log.leaves
                        .retain(|&position, _| !event_log::removes(p0, p1, position));
                }
            }
            // seq_cp: a turn that has not started takes the shared cells as
            // its prefix; one in progress cannot replay the change
            MemoryOp::SeqCp if seq != seq_dst => {
                if let Some(entry) = state.seqs.get_mut(&seq_dst)
                    && entry.live.prefix.is_some()
                {
                    entry.taint("the sequence took KV cells from another sequence mid-turn");
                }
            }
            MemoryOp::SeqKeep => all
                .iter()
                .filter(|&&s| s != seq)
                .for_each(|&s| state.retire(s, None, now)),
            MemoryOp::SeqShift => taint(
                &mut state,
                &targets,
                "the sequence's positions were shifted",
            ),
            MemoryOp::StateRead => taint(
                &mut state,
                &targets,
                "the sequence's KV cells were loaded from saved state",
            ),
            MemoryOp::Compact => taint(&mut state, &all, "the KV cache was compacted"),
            MemoryOp::SeqCp => {}
        }
        state.enforce_window(self.window_bytes, now);
    }

    /// Why the last unreceiptable ubatch could not be receipted.
    pub fn last_error(&self) -> Option<String> {
        self.lock().last_error.clone()
    }

    /// Positions turn `turn` of `seq` recorded since its last commit.
    pub fn recorded_range(&self, seq: i32, turn: TurnId) -> Option<Range<u32>> {
        let state = self.lock();
        let entry = state.seqs.get(&seq)?;
        let live = (entry.live.turn == turn)
            .then(|| entry.live.pending())
            .flatten();
        live.or_else(|| {
            entry
                .retired
                .iter()
                .rev()
                .filter(|retired| retired.log.turn == turn)
                .find_map(|retired| retired.log.pending())
        })
    }
}

#[cfg(test)]
#[path = "recorder/tests.rs"]
mod tests;
