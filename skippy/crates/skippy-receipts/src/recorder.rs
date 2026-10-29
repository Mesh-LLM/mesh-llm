//! Node side: records each ubatch's committed leaves and a replay log per
//! native sequence, commits a turn to a Merkle root, and opens a challenge
//! by replaying the turn in a scratch context ("replay, not storage" in
//! RECEIPTS.md).
//!
//! Per position, a stage keeps only leaf hashes. Per ubatch, it keeps what
//! replay needs to reproduce it bit for bit: tokens, sequences, positions,
//! output rows, KV cells, n_kv, and the stage input rows of a stage after
//! the first. A KV-cache change the replay cannot reproduce (a shared
//! prefix, a position shift, a state load, compaction) makes the sequence
//! unreceiptable until it is cleared.
//!
//! A replay costs about what serving the turn did, so a receipt is replayed
//! for at most [`MAX_REPLAYS_PER_RECEIPT`] distinct challenges, and a
//! repeated challenge is answered from the openings already made.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::ops::Range;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail};

use crate::commitment::{
    Hash, MerkleTree, attention_leaf, attention_slot, encode_path, leaf_index, residual_leaf,
    residual_slot, slots_per_position,
};
use crate::wire::{Challenge, LeafOpening, MatrixOpening, Opening, StageCommitment, encode_f32};

/// Committed turns are dropped, oldest first, past this many bytes.
pub const DEFAULT_WINDOW_BYTES: usize = 1 << 30;
/// Distinct challenges a committed turn is replayed for, at most.
pub const MAX_REPLAYS_PER_RECEIPT: u32 = 4;
/// Logs of a sequence reset before their turn was committed, kept per sequence.
const RETIRED_LOGS: usize = 4;
/// Token id of the dummy columns that stand in for other sequences in a replay.
const DUMMY_TOKEN: i32 = 0;

mod leaves;
mod openings;

use openings::OpeningCache;

/// A per-layer tensor the native recorder reports (`skippy_receipt_tensor_kind`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TensorKind {
    Residual,
    AttnQ,
    AttnK,
    AttnV,
    AttnOut,
    AttnIn,
    AttnProj,
    FfnIn,
    FfnGate,
    FfnUp,
    FfnDownIn,
    FfnDown,
}

impl TensorKind {
    pub fn from_code(code: i32) -> Option<Self> {
        Some(match code {
            0 => Self::Residual,
            1 => Self::AttnQ,
            2 => Self::AttnK,
            3 => Self::AttnV,
            4 => Self::AttnOut,
            5 => Self::AttnIn,
            6 => Self::AttnProj,
            7 => Self::FfnIn,
            8 => Self::FfnGate,
            9 => Self::FfnUp,
            10 => Self::FfnDownIn,
            11 => Self::FfnDown,
            _ => return None,
        })
    }
}

/// The committed tensors of a ubatch, by column.
pub trait UbatchColumns: Sync {
    /// Column `col` of tensor `kind` at `layer`; `None` when the layer kept
    /// no row for that column or the tensor was not reported.
    fn column(&self, layer: u32, kind: TensorKind, col: usize) -> Option<&[f32]>;
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

/// One step of a replay, columns ready for the native side. Other
/// sequences' columns are dummies at their recorded cells.
#[derive(Clone, Debug, PartialEq)]
pub enum ReplayStep {
    Decode {
        record: UbatchRecord,
        /// Positions of the replayed sequence to capture at this step.
        capture: Vec<u32>,
    },
    /// Removes positions `[p0, p1)` of the replayed sequence; `p1 < 0` has no end.
    SeqRm { p0: i32, p1: i32 },
}

/// One position of the challenged layer, as the replay recomputed it.
#[derive(Clone, Debug, PartialEq)]
pub struct ReplayedLayer {
    pub residual_in: Vec<f32>,
    pub residual_out: Option<Vec<f32>>,
    pub attn_in: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub attn_proj: Vec<f32>,
    pub ffn: Option<ReplayedFfn>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReplayedFfn {
    pub ffn_in: Vec<f32>,
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub down_in: Vec<f32>,
    pub down: Vec<f32>,
}

impl ReplayedLayer {
    /// Collects column `col` of a captured replay ubatch for `layer`.
    pub fn from_columns(layer: u32, columns: &dyn UbatchColumns, col: usize) -> Result<Self> {
        let get = |l: u32, kind: TensorKind| columns.column(l, kind, col).map(<[f32]>::to_vec);
        let need = |kind: TensorKind| {
            get(layer, kind)
                .with_context(|| format!("replay did not capture {kind:?} of layer {layer}"))
        };
        let residual_out = get(layer + 1, TensorKind::Residual);
        let ffn = match residual_out {
            Some(_) => Some(ReplayedFfn {
                ffn_in: need(TensorKind::FfnIn)?,
                gate: need(TensorKind::FfnGate)?,
                up: need(TensorKind::FfnUp)?,
                down_in: need(TensorKind::FfnDownIn)?,
                down: need(TensorKind::FfnDown)?,
            }),
            None => None,
        };
        Ok(Self {
            residual_in: need(TensorKind::Residual)?,
            residual_out,
            attn_in: need(TensorKind::AttnIn)?,
            q: need(TensorKind::AttnQ)?,
            k: need(TensorKind::AttnK)?,
            v: need(TensorKind::AttnV)?,
            attn_out: need(TensorKind::AttnOut)?,
            attn_proj: need(TensorKind::AttnProj)?,
            ffn,
        })
    }

    fn matrices(&self, layer: u32) -> Vec<MatrixOpening> {
        let name = |role: &str| format!("blk.{layer}.{role}.weight");
        let mut matrices = vec![
            MatrixOpening::new(&name("attn_q"), &self.attn_in, &self.q),
            MatrixOpening::new(&name("attn_k"), &self.attn_in, &self.k),
            MatrixOpening::new(&name("attn_v"), &self.attn_in, &self.v),
            MatrixOpening::new(&name("attn_output"), &self.attn_out, &self.attn_proj),
        ];
        if let Some(ffn) = &self.ffn {
            matrices.extend([
                MatrixOpening::new(&name("ffn_gate"), &ffn.ffn_in, &ffn.gate),
                MatrixOpening::new(&name("ffn_up"), &ffn.ffn_in, &ffn.up),
                MatrixOpening::new(&name("ffn_down"), &ffn.down_in, &ffn.down),
            ]);
        }
        matrices
    }
}

/// Replays a sequence's recorded ubatches; implemented over the native runtime.
pub trait Replayer {
    /// Runs `steps` for `seq` and returns the captured positions of `layer`.
    fn replay(
        &self,
        seq: i32,
        layer: u32,
        steps: &[ReplayStep],
    ) -> Result<Vec<(u32, ReplayedLayer)>>;
}

#[derive(Clone)]
enum Event {
    Ubatch(Arc<UbatchRecord>),
    SeqRm { p0: i32, p1: i32 },
}

#[derive(Default)]
struct SeqLog {
    events: Vec<Event>,
    /// Leaf hashes per position, in slot order.
    leaves: BTreeMap<u32, Vec<Hash>>,
    /// First position recorded since the last commit.
    pending_start: Option<u32>,
    /// Why the sequence cannot be replayed, until it is cleared.
    tainted: Option<String>,
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
    events: Vec<Event>,
    tree: MerkleTree,
    bytes: usize,
    /// Replays started for this turn.
    replays: u32,
}

#[derive(Default)]
struct RecorderState {
    seqs: HashMap<i32, SeqState>,
    committed: VecDeque<CommittedTurn>,
    committed_bytes: usize,
    openings: OpeningCache,
    last_error: Option<String>,
}

pub struct Recorder {
    stage_index: u32,
    layers: Range<u32>,
    backend: String,
    window_bytes: usize,
    state: Mutex<RecorderState>,
}

impl Recorder {
    pub fn new(stage_index: u32, layers: Range<u32>, backend: impl Into<String>) -> Self {
        Self {
            stage_index,
            layers,
            backend: backend.into(),
            window_bytes: DEFAULT_WINDOW_BYTES,
            state: Mutex::new(RecorderState::default()),
        }
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
        let leaves = leaves::ubatch_leaves(self.layers.clone(), &record.pos, columns);
        let record = Arc::new(record);
        let mut state = self.lock();
        let touched: BTreeSet<i32> = record.seq.iter().copied().collect();
        for seq in &touched {
            let log = &mut state.seqs.entry(*seq).or_default().live;
            log.events.push(Event::Ubatch(record.clone()));
        }
        for (col, leaf) in leaves.into_iter().enumerate() {
            let seq = state.seqs.entry(record.seq[col]).or_default();
            match leaf {
                Ok((position, hashes)) => {
                    let log = &mut seq.live;
                    log.leaves.insert(position, hashes);
                    log.pending_start =
                        Some(log.pending_start.map_or(position, |p| p.min(position)));
                }
                Err(error) => seq.taint(&format!("{error:#}")),
            }
        }
    }

    /// A ubatch could not be receipted; every live sequence is tainted.
    pub fn unreceiptable(&self, reason: &str) {
        let mut state = self.lock();
        for seq in state.seqs.values_mut() {
            seq.taint(reason);
        }
        state.last_error = Some(reason.to_string());
    }

    /// Mirrors a KV-cache operation (`skippy_receipt_memory_op`).
    pub fn memory_op(&self, op: i32, seq: i32, seq_dst: i32, p0: i32, p1: i32) {
        let mut state = self.lock();
        let all: Vec<i32> = state.seqs.keys().copied().collect();
        let targets = if seq < 0 { all.clone() } else { vec![seq] };
        match op {
            // clear
            0 => state.seqs.values_mut().for_each(SeqState::reset),
            // seq_rm
            1 => {
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
            // seq_cp
            2 if seq != seq_dst => state
                .seqs
                .entry(seq_dst)
                .or_default()
                .taint("the sequence shares KV cells with another sequence"),
            // seq_keep
            3 => {
                for (s, entry) in &mut state.seqs {
                    if *s != seq {
                        entry.reset();
                    }
                }
            }
            // seq_add, seq_div
            4 => {
                for s in targets {
                    state
                        .seqs
                        .entry(s)
                        .or_default()
                        .taint("the sequence's positions were shifted");
                }
            }
            // state_read
            5 => {
                for s in targets {
                    state
                        .seqs
                        .entry(s)
                        .or_default()
                        .taint("the sequence's KV cells were loaded from saved state");
                }
            }
            // compact
            6 => state
                .seqs
                .values_mut()
                .for_each(|s| s.taint("the KV cache was compacted")),
            _ => {}
        }
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
        let slots = self.slots();
        let mut state = self.lock();
        let last_error = state.last_error.clone();
        let entry = state.seqs.get_mut(&seq).with_context(|| {
            format!(
                "receipt {receipt_id}: stage {} recorded nothing for sequence {seq}{}",
                self.stage_index,
                last_error.map(|e| format!(" ({e})")).unwrap_or_default()
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
                format!("receipt {receipt_id}: no recorded turn covers positions {positions:?}")
            })?;
            &mut entry.retired[index]
        };
        if let Some(reason) = &log.tainted {
            bail!("receipt {receipt_id}: the turn cannot be replayed: {reason}");
        }
        let mut leaves = Vec::with_capacity(positions.len() * usize::try_from(slots)?);
        for position in positions.clone() {
            let hashes = log.leaves.get(&position).with_context(|| {
                format!(
                    "receipt {receipt_id}: stage {} computed nothing at position {position}",
                    self.stage_index
                )
            })?;
            leaves.extend_from_slice(hashes);
        }
        log.pending_start = None;
        let events = log.events.clone();
        let tree = MerkleTree::new(leaves)?;
        let commitment = StageCommitment {
            stage_index: self.stage_index,
            layer_start: self.layers.start,
            layer_end: self.layers.end,
            root: hex::encode(tree.root()),
            leaf_count: tree.leaf_count(),
            backend: self.backend.clone(),
        };
        let bytes = tree.bytes()
            + events
                .iter()
                .map(|event| match event {
                    Event::Ubatch(record) => record.bytes(),
                    Event::SeqRm { .. } => 8,
                })
                .sum::<usize>();
        state.committed_bytes += bytes;
        // a reused id names a new turn; its old openings no longer apply
        state.openings.forget(receipt_id);
        state.committed.push_back(CommittedTurn {
            receipt_id: receipt_id.to_owned(),
            seq,
            position_start: positions.start,
            position_end: positions.end,
            events,
            tree,
            bytes,
            replays: 0,
        });
        while state.committed_bytes > self.window_bytes && state.committed.len() > 1 {
            if let Some(old) = state.committed.pop_front() {
                state.committed_bytes -= old.bytes;
                state.openings.forget(&old.receipt_id);
            }
        }
        Ok(commitment)
    }

    /// Opens the challenged leaves of a committed turn, by replaying it
    /// unless the same challenge was opened before.
    pub fn open(&self, challenge: &Challenge, replayer: &dyn Replayer) -> Result<Opening> {
        if !self.layers.contains(&challenge.layer) {
            bail!(
                "stage {} does not hold layer {}",
                self.stage_index,
                challenge.layer
            );
        }
        let cached = self.lock().openings.get(challenge);
        if let Some(opening) = cached {
            return Ok(Opening::clone(&opening));
        }
        let opening = Arc::new(self.replay_opening(challenge, replayer)?);
        let mut state = self.lock();
        if state
            .committed
            .iter()
            .any(|turn| turn.receipt_id == challenge.receipt_id)
        {
            state.openings.insert(challenge.clone(), opening.clone());
        }
        drop(state);
        Ok(Arc::unwrap_or_clone(opening))
    }

    fn replay_opening(&self, challenge: &Challenge, replayer: &dyn Replayer) -> Result<Opening> {
        let layer = challenge.layer;
        let plan = self.plan_replay(challenge)?;
        let replayed: BTreeMap<u32, ReplayedLayer> = replayer
            .replay(plan.seq, layer, &plan.steps)?
            .into_iter()
            .collect();
        let mut leaves = Vec::with_capacity(challenge.positions.len());
        for (&position, expected) in challenge.positions.iter().zip(plan.expected) {
            let values = replayed
                .get(&position)
                .with_context(|| format!("replay did not reach position {position}"))?;
            let recomputed = [
                residual_leaf(position, layer, Some(&values.residual_in)),
                attention_leaf(
                    position,
                    layer,
                    &values.q,
                    &values.k,
                    &values.v,
                    &values.attn_out,
                ),
                residual_leaf(position, layer + 1, values.residual_out.as_deref()),
            ];
            for ((name, leaf), committed) in ["residual-in", "attention", "residual-out"]
                .iter()
                .zip(recomputed)
                .zip(expected.hashes)
            {
                if leaf != committed {
                    bail!(
                        "replay diverged from the committed turn at position {position}, layer {layer}: \
                         the {name} leaf differs"
                    );
                }
            }
            let [residual_in_path, attention_path, residual_out_path] = expected.paths;
            leaves.push(LeafOpening {
                position,
                residual_in: encode_f32(&values.residual_in),
                residual_out: values.residual_out.as_deref().map(encode_f32),
                residual_in_path,
                attention_path,
                residual_out_path,
                matrices: values.matrices(layer),
            });
        }
        Ok(Opening {
            receipt_id: challenge.receipt_id.clone(),
            stage_index: self.stage_index,
            layer,
            leaves,
        })
    }

    fn plan_replay(&self, challenge: &Challenge) -> Result<ReplayPlan> {
        let layer = challenge.layer;
        let slots = self.slots();
        let mut state = self.lock();
        let turn = state
            .committed
            .iter_mut()
            .rev()
            .find(|t| t.receipt_id == challenge.receipt_id)
            .with_context(|| {
                format!(
                    "receipt {} is not in the opening window",
                    challenge.receipt_id
                )
            })?;
        if turn.replays >= MAX_REPLAYS_PER_RECEIPT {
            bail!(
                "receipt {} was already replayed for {MAX_REPLAYS_PER_RECEIPT} challenges",
                challenge.receipt_id
            );
        }
        turn.replays += 1;
        let turn = &*turn;
        let mut expected = Vec::with_capacity(challenge.positions.len());
        let mut capture: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
        for &position in &challenge.positions {
            if !(turn.position_start..turn.position_end).contains(&position) {
                bail!("position {position} is outside the receipt");
            }
            let wanted = i32::try_from(position)?;
            let step = turn
                .events
                .iter()
                .rposition(|event| match event {
                    Event::Ubatch(record) => record
                        .seq
                        .iter()
                        .zip(&record.pos)
                        .any(|(&s, &p)| s == turn.seq && p == wanted),
                    Event::SeqRm { .. } => false,
                })
                .with_context(|| format!("no recorded ubatch computed position {position}"))?;
            capture.entry(step).or_default().push(position);
            let slot_indices = [
                residual_slot(layer, self.layers.start),
                attention_slot(layer, self.layers.start),
                residual_slot(layer + 1, self.layers.start),
            ];
            let mut hashes = [[0u8; 32]; 3];
            let mut paths: [Vec<String>; 3] = Default::default();
            for (i, slot) in slot_indices.into_iter().enumerate() {
                let index = leaf_index(position, slot, turn.position_start, slots);
                hashes[i] = turn
                    .tree
                    .leaf(index)
                    .context("leaf outside the committed tree")?;
                paths[i] = encode_path(&turn.tree.path(index)?);
            }
            expected.push(ExpectedLeaves { hashes, paths });
        }
        let last = capture
            .keys()
            .next_back()
            .copied()
            .context("the challenge has no positions")?;
        let steps = turn.events[..=last]
            .iter()
            .enumerate()
            .map(|(index, event)| match event {
                Event::Ubatch(record) => ReplayStep::Decode {
                    record: replay_columns(record, turn.seq),
                    capture: capture.get(&index).cloned().unwrap_or_default(),
                },
                Event::SeqRm { p0, p1 } => ReplayStep::SeqRm { p0: *p0, p1: *p1 },
            })
            .collect();
        Ok(ReplayPlan {
            seq: turn.seq,
            steps,
            expected,
        })
    }
}

/// The replay steps of a challenge, and per position the committed leaves.
struct ReplayPlan {
    seq: i32,
    steps: Vec<ReplayStep>,
    expected: Vec<ExpectedLeaves>,
}

/// The committed residual-in, attention and residual-out leaves of one
/// position, with their Merkle paths.
struct ExpectedLeaves {
    hashes: [Hash; 3],
    paths: [Vec<String>; 3],
}

/// A recorded ubatch as `seq` replays it: other sequences' columns become
/// dummy tokens at positions 0.. of their own sequence, with zero inputs.
fn replay_columns(record: &UbatchRecord, seq: i32) -> UbatchRecord {
    let mut out = record.clone();
    let mut next_pos: BTreeMap<i32, i32> = BTreeMap::new();
    for col in 0..record.seq.len() {
        if record.seq[col] == seq {
            continue;
        }
        out.token[col] = DUMMY_TOKEN;
        let pos = next_pos.entry(record.seq[col]).or_insert(0);
        out.pos[col] = *pos;
        *pos += 1;
        if let Some(input) = &mut out.input {
            let row = input.row_bytes;
            input.rows[col * row..(col + 1) * row].fill(0);
        }
    }
    out
}

#[cfg(test)]
#[path = "recorder/tests.rs"]
mod tests;
