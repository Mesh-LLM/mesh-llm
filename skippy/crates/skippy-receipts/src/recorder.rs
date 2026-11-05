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
//! A replay costs about what serving the turn did, so a receipt is replayed
//! for at most [`MAX_REPLAYS_PER_RECEIPT`] distinct challenges, and a
//! repeated challenge is answered from the openings already made.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::ops::Range;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail};

use crate::attention::AttentionDescriptor;
use crate::commitment::{Hash, MerkleTree, slots_per_position};
use crate::layer_tensors::{FLAG_ROUTER, GGML_F32, MIXING_KINDS, TensorKind, UbatchColumns};
use crate::wire::{
    MatrixOpening, MixingKind, PrefixCommitment, SelectionOpening, StageCommitment, TensorOpening,
    encode_bytes,
};

/// Committed turns are dropped, oldest first, past this many bytes.
pub const DEFAULT_WINDOW_BYTES: usize = 1 << 30;
/// Distinct challenges a committed turn is replayed for, at most.
pub const MAX_REPLAYS_PER_RECEIPT: u32 = 4;
/// Logs of a sequence reset before their turn was committed, kept per sequence.
const RETIRED_LOGS: usize = 4;
/// Token id of the dummy columns that stand in for other sequences in a replay.
const DUMMY_TOKEN: i32 = 0;

mod attention_opening;
mod pins;
pub use pins::{CHALLENGE_PIN, ROWS_PIN};
mod cells;
mod leaves;
mod openings;
mod recurrent;
mod recurrent_opening;
mod stage_opening;
mod turn_seed;

use cells::CellMirror;
pub(crate) use cells::prefix_leaf;
pub use cells::{KvCell, KvStreams, MemoryOp, PrefixCell};
use leaves::PositionLeaves;
use openings::OpeningCache;

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

    /// The column that computed `seq` at `pos`.
    fn column(&self, seq: i32, pos: i32) -> Option<usize> {
        (0..self.seq.len()).find(|&col| self.seq[col] == seq && self.pos[col] == pos)
    }
}

/// One step of a replay, columns ready for the native side. Other
/// sequences' columns are dummies at their recorded cells.
#[derive(Clone, Debug, PartialEq)]
pub enum ReplayStep {
    Decode {
        record: Arc<UbatchRecord>,
        /// Positions of the replayed sequence to capture at this step.
        capture: Vec<u32>,
        /// Positions of the replayed sequence whose token-mixing values
        /// alone to capture at this step, besides `capture`.
        mixing: Vec<u32>,
    },
    /// Removes positions `[p0, p1)` of the replayed sequence; `p1 < 0` has no end.
    SeqRm { p0: i32, p1: i32 },
}

/// One position of the challenged layer, as the replay recomputed it.
#[derive(Clone, Debug, PartialEq)]
pub struct ReplayedLayer {
    /// Little-endian f32 bytes.
    pub residual_in: Vec<u8>,
    pub residual_out: Option<Vec<u8>>,
    /// The logits a final stage's last layer leads to, little-endian f32,
    /// where the replay reported them.
    pub logits: Option<Vec<u8>>,
    /// The mixing leaf's values, in leaf order.
    pub mixing: Vec<ReplayedMixing>,
    pub matmuls: Vec<ReplayedMatmul>,
    pub selections: Vec<SelectionOpening>,
    /// The layer's attention ops, as the replay's graph describes them.
    pub attention: Vec<AttentionDescriptor>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReplayedMixing {
    pub kind: MixingKind,
    pub index: u32,
    /// ggml type and bytes; `None` where the layer kept no row.
    pub value: Option<(i32, Vec<u8>)>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReplayedMatmul {
    pub index: u32,
    pub weight: String,
    pub router: bool,
    pub experts: Option<Vec<i32>>,
    /// Little-endian f32 bytes.
    pub input: Vec<u8>,
    pub output: Vec<u8>,
}

impl ReplayedLayer {
    /// The token-mixing values alone of column `col` of a captured replay
    /// ubatch for `layer`.
    pub fn from_mixing_columns(layer: u32, columns: &dyn UbatchColumns, col: usize) -> Self {
        Self {
            residual_in: Vec::new(),
            residual_out: None,
            logits: None,
            mixing: mixing_columns(layer, columns, col),
            matmuls: Vec::new(),
            selections: Vec::new(),
            attention: Vec::new(),
        }
    }

    /// Collects column `col` of a captured replay ubatch for `layer`.
    pub fn from_columns(layer: u32, columns: &dyn UbatchColumns, col: usize) -> Result<Self> {
        let f32_column = |l: u32, kind: TensorKind, index: u32| -> Result<Option<Vec<u8>>> {
            match columns.column(l, kind, index, col) {
                Some(c) if c.ggml_type != GGML_F32 => {
                    bail!(
                        "replay reported {} {index} of layer {l} as type {}",
                        kind.name(),
                        c.ggml_type
                    )
                }
                c => Ok(c.map(|c| c.bytes.to_vec())),
            }
        };
        let residual_in = f32_column(layer, TensorKind::Residual, 0)?
            .with_context(|| format!("replay did not capture the residual of layer {layer}"))?;
        let residual_out = f32_column(layer + 1, TensorKind::Residual, 0)?;
        let logits = f32_column(layer + 1, TensorKind::Logits, 0)?;
        let mixing = mixing_columns(layer, columns, col);
        let mut matmuls = Vec::new();
        for index in 0..columns.count(layer, TensorKind::MatmulInput) {
            let (Some(input), Some(output)) = (
                columns.column(layer, TensorKind::MatmulInput, index, col),
                f32_column(layer, TensorKind::MatmulOutput, index)?,
            ) else {
                // the layer dropped this position's row before the matmul
                continue;
            };
            if input.ggml_type != GGML_F32 {
                bail!(
                    "replay reported matmul input {index} of layer {layer} as type {}",
                    input.ggml_type
                );
            }
            let experts = columns
                .column(layer, TensorKind::ExpertIds, index, col)
                .map(|c| c.i32_values().context("expert ids are not i32"))
                .transpose()?;
            matmuls.push(ReplayedMatmul {
                index,
                weight: input.name.to_owned(),
                router: input.flags & FLAG_ROUTER != 0,
                experts,
                input: input.bytes.to_vec(),
                output,
            });
        }
        let selections = (0..columns.count(layer, TensorKind::Selection))
            .filter_map(|index| {
                let column = columns.column(layer, TensorKind::Selection, index, col)?;
                Some(
                    column
                        .i32_values()
                        .context("a selection is not i32")
                        .map(|ids| SelectionOpening { index, ids }),
                )
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            residual_in,
            residual_out,
            logits,
            mixing,
            matmuls,
            selections,
            attention: Vec::new(),
        })
    }

    fn mixing_openings(&self) -> Vec<TensorOpening> {
        self.mixing
            .iter()
            .map(|m| {
                TensorOpening::new(
                    m.kind,
                    m.index,
                    m.value.as_ref().map(|(t, bytes)| (*t, bytes.as_slice())),
                )
            })
            .collect()
    }

    fn matrices(&self) -> Vec<MatrixOpening> {
        self.matmuls
            .iter()
            .map(|m| MatrixOpening {
                index: m.index,
                weight: m.weight.clone(),
                router: m.router,
                experts: m.experts.clone(),
                input: encode_bytes(&m.input),
                output: encode_bytes(&m.output),
            })
            .collect()
    }
}

/// The mixing leaf's values of `layer` in column `col`, in leaf order.
fn mixing_columns(layer: u32, columns: &dyn UbatchColumns, col: usize) -> Vec<ReplayedMixing> {
    MIXING_KINDS
        .iter()
        .flat_map(|&kind| (0..columns.count(layer, kind)).map(move |index| (kind, index)))
        .map(|(kind, index)| ReplayedMixing {
            kind: MixingKind::from_tensor_kind(kind).expect("a mixing kind"),
            index,
            value: columns
                .column(layer, kind, index, col)
                .map(|c| (c.ggml_type, c.bytes.to_vec())),
        })
        .collect()
}

/// One position of one layer, as a replay recomputed it.
pub type ReplayedPosition = (u32, u32, ReplayedLayer);

/// The cells of the stage's KV cache a replay starts from, copied into the
/// replay context before its steps: the prefix the turn attended to, and
/// the turn's own cells when the replay computes only some of its ubatches
/// (`turn_seed`).
#[derive(Clone, Debug, PartialEq)]
pub struct ReplaySeed {
    pub cells: Vec<PrefixCell>,
    /// KV cells the replay context needs: the seeded ones and every one
    /// the steps use.
    pub n_cells: u32,
    /// On a stage with hybrid memory, the recurrent state the turn started
    /// from, which the replay imports after the cells.
    pub state: Option<Arc<[u8]>>,
}

impl ReplaySeed {
    pub fn indices(&self) -> Vec<u32> {
        self.cells.iter().map(|c| c.cell).collect()
    }

    pub fn positions(&self) -> Vec<i32> {
        self.cells.iter().map(|c| c.pos).collect()
    }
}

/// Which tensors a replay reports, and from which values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ReplayScope<'a> {
    /// The committed values and the matmuls and selections an opening
    /// shows besides them.
    Full,
    /// The committed values alone, as the live recorder marks them. Marking
    /// more can change which kernels a backend fuses (CUDA's fused MoE
    /// routing and gated matmuls, for two), so a full replay may not
    /// reproduce the committed values bit for bit while this one does.
    Committed,
    /// As [`Self::Full`], with each anchor's committed value written into
    /// its token's column as soon as the replay computes it, so the matmuls
    /// and selections of the anchored tokens follow from the committed
    /// values.
    Anchored(&'a [ReplayAnchor]),
}

/// A committed value an anchored replay writes into one token's column.
#[derive(Clone, Debug, PartialEq)]
pub struct ReplayAnchor {
    /// The decode step, as an index into the replay's steps.
    pub step: usize,
    pub seq: i32,
    pub pos: i32,
    pub layer: u32,
    pub value: AnchorValue,
    pub values: Vec<f32>,
}

/// What an anchor replaces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnchorValue {
    /// The residual entering the layer.
    Residual,
    /// A token-mixing output, by index.
    MixingOutput(u32),
}

/// Replays a sequence's recorded ubatches; implemented over the native runtime.
pub trait Replayer {
    /// Runs `steps` for `seq`, from the cells of `seed` when there is one,
    /// and returns, for every captured position and every one of `layers`,
    /// the position, the layer and its values. A seed is copied with the
    /// stage idle, after [`Recorder::check_seed`] confirms its cells still
    /// hold what the turn read or wrote. `scope` says which tensors it
    /// reports.
    fn replay(
        &self,
        seq: i32,
        layers: &[u32],
        scope: ReplayScope<'_>,
        seed: Option<&ReplaySeed>,
        steps: &[ReplayStep],
    ) -> Result<Vec<ReplayedPosition>>;

    /// The rows the stage stored for `layer` in `cells`, read from its KV
    /// cache with the stage idle, after [`Recorder::check_cells`] confirms
    /// they still hold what the turn read or wrote. One [`StoredRows`] per
    /// stored row of the layer, by index, each with a row per cell.
    fn read_rows(&self, layer: u32, cells: &[PrefixCell]) -> Result<Vec<StoredRows>>;

    /// Pins `cells` of the stage's KV cache in place of those pinned before
    /// ([`Recorder::pinned_cells`]): while one is empty, the stage writes
    /// other cells as long as a ubatch fits there. It does not wait for the
    /// stage to be idle.
    fn pin_cells(&self, cells: &[KvCell]) -> Result<()>;
}

/// One stored row of a layer in several KV cells, in the cache's type.
#[derive(Clone, Debug, PartialEq)]
pub struct StoredRows {
    pub ggml_type: i32,
    pub row_bytes: usize,
    /// Row `i` at `values[i * row_bytes..]`.
    pub values: Vec<u8>,
}

impl StoredRows {
    pub fn row(&self, i: usize) -> &[u8] {
        self.values
            .get(i * self.row_bytes..(i + 1) * self.row_bytes)
            .unwrap_or(&[])
    }
}

#[derive(Clone)]
enum Event {
    Ubatch(Arc<UbatchRecord>),
    SeqRm { p0: i32, p1: i32 },
}

impl Event {
    fn ubatch(&self) -> Option<&UbatchRecord> {
        match self {
            Self::Ubatch(record) => Some(record),
            Self::SeqRm { .. } => None,
        }
    }
}

#[derive(Default)]
struct SeqLog {
    events: Vec<Event>,
    /// What the stage committed per position.
    leaves: BTreeMap<u32, PositionLeaves>,
    /// First position recorded since the last commit.
    pending_start: Option<u32>,
    /// Why the pending turn cannot be replayed.
    tainted: Option<String>,
    /// The cells the pending turn attended to below its first position, as
    /// they were when its first ubatch was recorded, or why they are unknown.
    prefix: Option<Result<Vec<PrefixCell>, String>>,
    /// ggml types of the cache rows the stage stored.
    cache_types: BTreeSet<i32>,
    /// On a stage with hybrid memory, the recurrent state the pending turn
    /// started from, or why it is unknown.
    state: Option<Result<recurrent::TurnState, String>>,
}

impl SeqLog {
    fn pending(&self) -> Option<Range<u32>> {
        let start = self.pending_start?;
        let (&last, _) = self.leaves.range(start..).next_back()?;
        Some(start..last + 1)
    }
}

#[derive(Default)]
struct SeqState {
    live: SeqLog,
    /// Logs reset before their turn was committed, newest last.
    retired: VecDeque<SeqLog>,
}

impl SeqState {
    fn reset(&mut self) {
        let old = std::mem::take(&mut self.live);
        if old.pending().is_some() {
            self.retired.push_back(old);
            while self.retired.len() > RETIRED_LOGS {
                self.retired.pop_front();
            }
        }
    }

    fn taint(&mut self, reason: &str) {
        self.live.tainted.get_or_insert_with(|| reason.to_string());
    }
}

struct CommittedTurn {
    receipt_id: String,
    seq: i32,
    position_start: u32,
    position_end: u32,
    /// The cells the replay starts from.
    prefix: Vec<PrefixCell>,
    events: Vec<Event>,
    tree: MerkleTree,
    /// By position from `position_start`, the cell the turn wrote it to.
    cells: Vec<TurnCell>,
    /// The attention picks the stage opened partials for.
    attention: Vec<attention_opening::OpenedAttention>,
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
    openings: OpeningCache,
    last_error: Option<String>,
    /// Layers whose token mixing reads a recurrent state.
    state_layers: BTreeSet<u32>,
}

/// The committed turn under `receipt_id`, the latest if the id repeats.
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

/// The committed turn a challenge of `receipt_id` opens.
fn opening_turn<'a>(
    committed: &'a VecDeque<CommittedTurn>,
    receipt_id: &str,
) -> Result<&'a CommittedTurn> {
    turn(committed, receipt_id)
        .with_context(|| format!("receipt {receipt_id} is not in the opening window"))
}

fn opening_turn_mut<'a>(
    committed: &'a mut VecDeque<CommittedTurn>,
    receipt_id: &str,
) -> Result<&'a mut CommittedTurn> {
    turn_mut(committed, receipt_id)
        .with_context(|| format!("receipt {receipt_id} is not in the opening window"))
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
        let mut guard = self.lock();
        let state = &mut *guard;
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
                log.events.clear();
                log.tainted = None;
                let end = u32::try_from(pos).unwrap_or(0);
                log.prefix = Some(state.cells.prefix(seq, end).map_err(|e| format!("{e:#}")));
            }
            log.events.push(Event::Ubatch(record.clone()));
            log.cache_types.extend(&cache_types);
        }
        for (col, leaf) in leaves.into_iter().enumerate() {
            let seq = state.seqs.entry(record.seq[col]).or_default();
            let digest = match leaf {
                Ok((position, hashes)) => {
                    let digest = hashes.cell;
                    let log = &mut seq.live;
                    log.leaves.insert(position, hashes);
                    log.pending_start =
                        Some(log.pending_start.map_or(position, |p| p.min(position)));
                    Some(digest)
                }
                Err(error) => {
                    seq.taint(&format!("{error:#}"));
                    None
                }
            };
            state
                .cells
                .write(record.cell[col], record.pos[col], record.seq[col], digest);
        }
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

    /// A ubatch could not be receipted; every live sequence is tainted, and
    /// the cells it wrote are unknown.
    pub fn unreceiptable(&self, reason: &str) {
        let mut state = self.lock();
        for seq in state.seqs.values_mut() {
            seq.taint(reason);
        }
        state.cells.clear();
        state.last_error = Some(reason.to_string());
    }

    /// Sets how the stage's KV cache lays out its cells. Call it before the
    /// first ubatch: the mirror forgets the cells it knew.
    pub fn set_kv_streams(&self, streams: KvStreams) {
        self.lock().cells.set_streams(streams);
    }

    /// Mirrors a KV-cache operation (`skippy_receipt_memory_op`).
    pub fn memory_op(&self, op: MemoryOp, seq: i32, seq_dst: i32, p0: i32, p1: i32) {
        let mut state = self.lock();
        state.cells.memory_op(op, seq, seq_dst, p0, p1);
        let all: Vec<i32> = state.seqs.keys().copied().collect();
        let targets = if seq < 0 { all } else { vec![seq] };
        let taint = |state: &mut RecorderState, seqs: &[i32], reason: &str| {
            for &s in seqs {
                state.seqs.entry(s).or_default().taint(reason);
            }
        };
        match op {
            MemoryOp::Clear => state.seqs.values_mut().for_each(SeqState::reset),
            MemoryOp::SeqRm => {
                for s in targets {
                    let entry = state.seqs.entry(s).or_default();
                    if p0 <= 0 && p1 < 0 {
                        entry.reset();
                    } else {
                        let log = &mut entry.live;
                        log.events.push(Event::SeqRm { p0, p1 });
                        let start = u32::try_from(p0.max(0)).unwrap_or(0);
                        let end = u32::try_from(p1).unwrap_or(u32::MAX);
                        log.leaves
                            .retain(|position, _| !(start..end).contains(position));
                    }
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
            MemoryOp::SeqKeep => {
                for (s, entry) in &mut state.seqs {
                    if *s != seq {
                        entry.reset();
                    }
                }
            }
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
            MemoryOp::Compact => state
                .seqs
                .values_mut()
                .for_each(|s| s.taint("the KV cache was compacted")),
            MemoryOp::SeqCp => {}
        }
    }

    /// Why the last unreceiptable ubatch could not be receipted.
    pub fn last_error(&self) -> Option<String> {
        self.lock().last_error.clone()
    }

    /// Positions recorded for a sequence since its last commit.
    pub fn recorded_range(&self, seq: i32) -> Option<Range<u32>> {
        let state = self.lock();
        let entry = state.seqs.get(&seq)?;
        entry
            .live
            .pending()
            .or_else(|| entry.retired.iter().rev().find_map(SeqLog::pending))
    }

    /// Commits the sequence's leaves in `positions` as receipt `receipt_id`.
    pub fn commit(
        &self,
        receipt_id: &str,
        seq: i32,
        positions: Range<u32>,
    ) -> Result<StageCommitment> {
        if positions.is_empty() {
            bail!("receipt {receipt_id}: empty position range");
        }
        let turn = self.take_covering_turn(receipt_id, seq, &positions)?;
        // hashing the turn does not hold the lock the recording waits on
        let (commitment, committed) = self.seal(receipt_id, seq, positions, turn)?;
        let mut state = self.lock();
        state.committed_bytes += committed.bytes;
        // a reused id names a new turn; its old openings no longer apply
        state.openings.forget(receipt_id);
        state.committed.push_back(committed);
        state.drop_unpinned_states(std::time::Instant::now());
        while state.committed_bytes > self.window_bytes && state.committed.len() > 1 {
            if let Some(old) = state.committed.pop_front() {
                state.committed_bytes -= old.bytes;
                state.openings.forget(&old.receipt_id);
            }
        }
        Ok(commitment)
    }

    /// Takes the recorded turn of `seq` that covers `positions` from its log.
    fn take_covering_turn(
        &self,
        receipt_id: &str,
        seq: i32,
        positions: &Range<u32>,
    ) -> Result<TakenTurn> {
        let slots = self.slots();
        let mut state = self.lock();
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
        let covers = |log: &SeqLog| {
            log.pending_start
                .is_some_and(|start| start <= positions.start)
                && log.leaves.contains_key(&(positions.end - 1))
        };
        let log = if covers(&entry.live) {
            &mut entry.live
        } else {
            let index = entry.retired.iter().rposition(covers).with_context(|| {
                format!(
                    "receipt {receipt_id}: no recorded turn covers positions {positions:?}{last_error}"
                )
            })?;
            &mut entry.retired[index]
        };
        take_turn(log, positions, slots, self.is_recurrent()).map_err(|error| {
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
        let prefix = cells::prefix_root(&turn.prefix)?.map(|root| PrefixCommitment {
            root: hex::encode(root),
            positions: u32::try_from(turn.prefix.len()).unwrap_or(u32::MAX),
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
            + turn.prefix.len() * size_of::<PrefixCell>()
            + turn
                .events
                .iter()
                .map(|event| match event {
                    Event::Ubatch(record) => record.bytes(),
                    Event::SeqRm { .. } => 8,
                })
                .sum::<usize>()
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
            pin: pins::Pin::new(std::time::Instant::now()),
            state: turn.state,
        };
        Ok((commitment, committed))
    }
}

/// The first position `seq` has in each ubatch of `events` that computed it.
fn ubatch_starts(events: &[Event], seq: i32) -> Vec<u32> {
    events
        .iter()
        .filter_map(Event::ubatch)
        .filter_map(|record| {
            record
                .seq
                .iter()
                .zip(&record.pos)
                .filter(|&(&s, _)| s == seq)
                .filter_map(|(_, &p)| u32::try_from(p).ok())
                .min()
        })
        .collect()
}

/// What a commit takes from a sequence's log.
struct TakenTurn {
    leaves: Vec<Hash>,
    cells: Vec<TurnCell>,
    prefix: Vec<PrefixCell>,
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
        .map(|&t| crate::weights::type_name(t as u32))
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

#[cfg(test)]
#[path = "recorder/tests.rs"]
mod tests;
