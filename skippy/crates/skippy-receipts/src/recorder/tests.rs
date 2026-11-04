use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use anyhow::Result;

use super::*;
use crate::binding::{verify_paths, verify_shape};
use crate::layer_tensors::{
    FLAG_ROUTER, GGML_F32, GGML_I32, TensorColumn, TensorKind, UbatchColumns, f32_bytes, i32_bytes,
};
use crate::wire::{
    CannotOpen, CannotOpenReason, Challenge, MixingKind, Opening, Pick, RECEIPT_VERSION, Receipt,
    StageCommitment, TensorOpening, encode_f32,
};

/// The model has four layers; stage 0 holds 0..2 and stage 1 holds 2..4.
const MODEL_LAYERS: u32 = 4;
const STAGES: [Range<u32>; 2] = [0..2, 2..4];
/// ggml type of the stored rows: f16.
const CACHE_TYPE: i32 = 1;

/// A deterministic stand-in for the values a layer computes for a token.
fn value(layer: u32, kind: TensorKind, index: u32, token: i32, position: i32) -> Vec<f32> {
    let base = layer as f32 * 100.0
        + kind.code() as f32
        + index as f32 * 10.0
        + token as f32 * 0.25
        + position as f32 * 0.5;
    vec![base, base + 1.0, -base]
}

type Key = (u32, TensorKind, u32, usize);

/// A stage of the test model. `entry_offset` shifts the residual it
/// receives, as a stage that does not use the previous stage's output would.
/// The token mixing of `state_layers` reads a recurrent state, which each
/// ubatch reports at the first position of every sequence.
#[derive(Clone)]
struct Stage {
    layers: Range<u32>,
    entry_offset: f32,
    state_layers: Vec<u32>,
}

impl Stage {
    fn new(index: usize) -> Self {
        Self {
            layers: STAGES[index].clone(),
            entry_offset: 0.0,
            state_layers: Vec::new(),
        }
    }
}

/// Columns of a ubatch computed by `value`: per layer a mixing input, a
/// stored row and a mixing output; in opened layers also four attention
/// matmuls and two FFN matmuls (one with experts) with a selection. The
/// model's final layer keeps the FFN and its residual out only for output
/// rows.
struct Columns(HashMap<Key, (i32, i32, String, Vec<u8>)>);

impl Columns {
    fn new(record: &UbatchRecord, stage: &Stage, open: &[u32]) -> Self {
        let mut columns = Self(HashMap::new());
        for col in 0..record.token.len() {
            let (token, pos) = (record.token[col], record.pos[col]);
            let f32_at = |layer, kind, index| f32_bytes(&value(layer, kind, index, token, pos));
            let output = record.output[col] != 0;
            for layer in stage.layers.clone() {
                for kind in [
                    TensorKind::Residual,
                    TensorKind::MixingInput,
                    TensorKind::MixingOutput,
                ] {
                    let mut values = value(layer, kind, 0, token, pos);
                    if kind == TensorKind::Residual && layer == stage.layers.start {
                        values.iter_mut().for_each(|v| *v += stage.entry_offset);
                    }
                    columns.put((layer, kind, 0, col), GGML_F32, 0, "", f32_bytes(&values));
                }
                let stored = stored_row(layer, token, pos);
                columns.put(
                    (layer, TensorKind::MixingStored, 0, col),
                    CACHE_TYPE,
                    0,
                    "",
                    stored,
                );
                if open.contains(&layer) {
                    let full = output || layer + 1 < MODEL_LAYERS;
                    columns.add_matmuls(layer, col, if full { 6 } else { 4 }, &f32_at);
                }
            }
            let end = stage.layers.end;
            if end < MODEL_LAYERS || output {
                let key = (end, TensorKind::Residual, 0, col);
                columns.put(key, GGML_F32, 0, "", f32_at(end, TensorKind::Residual, 0));
            }
        }
        columns.add_states(record, stage);
        columns
    }

    /// The state each sequence's token mixing read before its first
    /// position of the ubatch, the one its previous position left.
    fn add_states(&mut self, record: &UbatchRecord, stage: &Stage) {
        let mut first: HashMap<i32, usize> = HashMap::new();
        for col in 0..record.pos.len() {
            let entry = first.entry(record.seq[col]).or_insert(col);
            if record.pos[col] < record.pos[*entry] {
                *entry = col;
            }
        }
        for &col in first.values() {
            for &layer in &stage.state_layers {
                let state = value(layer, TensorKind::MixingState, 0, 0, record.pos[col]);
                let key = (layer, TensorKind::MixingState, 0, col);
                self.put(key, GGML_F32, 0, "", f32_bytes(&state));
            }
        }
    }

    fn put(&mut self, key: Key, ggml_type: i32, flags: i32, name: &str, bytes: Vec<u8>) {
        self.0
            .insert(key, (ggml_type, flags, name.to_owned(), bytes));
    }

    fn add_matmuls(
        &mut self,
        layer: u32,
        col: usize,
        count: u32,
        f32_at: &dyn Fn(u32, TensorKind, u32) -> Vec<u8>,
    ) {
        for index in 0..count {
            let name = format!("blk.{layer}.w{index}.weight");
            let flags = if index == 4 { FLAG_ROUTER } else { 0 };
            for kind in [TensorKind::MatmulInput, TensorKind::MatmulOutput] {
                self.put(
                    (layer, kind, index, col),
                    GGML_F32,
                    flags,
                    &name,
                    f32_at(layer, kind, index),
                );
            }
        }
        if count > 4 {
            let ids = i32_bytes(&[3, 1]);
            self.put(
                (layer, TensorKind::ExpertIds, 5, col),
                GGML_I32,
                0,
                "",
                ids.clone(),
            );
            self.put((layer, TensorKind::Selection, 0, col), GGML_I32, 0, "", ids);
        }
    }
}

impl UbatchColumns for Columns {
    fn count(&self, layer: u32, kind: TensorKind) -> u32 {
        self.0
            .keys()
            .filter(|k| k.0 == layer && k.1 == kind)
            .map(|k| k.2 + 1)
            .max()
            .unwrap_or(0)
    }

    fn column(
        &self,
        layer: u32,
        kind: TensorKind,
        index: u32,
        col: usize,
    ) -> Option<TensorColumn<'_>> {
        self.0
            .get(&(layer, kind, index, col))
            .map(|(ggml_type, flags, name, bytes)| TensorColumn {
                ggml_type: *ggml_type,
                flags: *flags,
                name,
                bytes,
                digest: None,
            })
    }
}

fn ubatch(columns: &[(i32, i32)], cells: &[u32], output_last: bool) -> UbatchRecord {
    let n = columns.len();
    let mut output = vec![0; n];
    if output_last {
        output[n - 1] = 1;
    }
    UbatchRecord {
        token: columns.iter().map(|(_, p)| 1000 + p).collect(),
        seq: columns.iter().map(|(s, _)| *s).collect(),
        pos: columns.iter().map(|(_, p)| *p).collect(),
        output,
        cell: cells.to_vec(),
        n_kv: 256,
        input: Some(StageInput {
            rows: vec![7; n * 4],
            row_bytes: 4,
            ggml_type: 0,
        }),
    }
}

fn record_on(recorder: &Recorder, stage: &Stage, record: UbatchRecord) {
    let columns = Columns::new(&record, stage, &[]);
    recorder.record_ubatch(record, &columns);
}

/// Records on a lone recorder for stage 1.
fn record(recorder: &Recorder, record: UbatchRecord) {
    record_on(recorder, &Stage::new(1), record);
}

/// A replay a [`FakeReplayer`] ran: the cells it copied and what each of
/// its steps captured.
type Run = (Vec<u32>, Vec<Vec<u32>>);

/// Replays by recomputing `value`, and keeps the steps it was given.
struct FakeReplayer {
    stage: Stage,
    steps: RefCell<Vec<ReplayStep>>,
    seed: RefCell<Option<ReplaySeed>>,
    /// Every replay: its seed's cells and its steps' captures.
    runs: RefCell<Vec<Run>>,
    layers: RefCell<Vec<u32>>,
    replays: Cell<u32>,
    /// The scope of every replay.
    scopes: RefCell<Vec<&'static str>>,
    /// The anchors of every anchored replay.
    anchors: RefCell<Vec<ReplayAnchor>>,
    corrupt: bool,
    /// Corrupts the committed values of full replays, anchored or not, as a
    /// backend that fuses kernels differently when more tensors are marked
    /// does.
    fused: bool,
    /// The attention layout every replayed layer reports.
    attention: Option<crate::attention::AttentionLayout>,
}

impl FakeReplayer {
    fn new(stage: Stage) -> Self {
        Self {
            stage,
            steps: RefCell::default(),
            seed: RefCell::default(),
            runs: RefCell::default(),
            layers: RefCell::default(),
            replays: Cell::new(0),
            scopes: RefCell::default(),
            anchors: RefCell::default(),
            corrupt: false,
            fused: false,
            attention: None,
        }
    }

    fn for_stage(index: usize) -> Self {
        Self::new(Stage::new(index))
    }
}

impl Replayer for FakeReplayer {
    fn replay(
        &self,
        seq: i32,
        layers: &[u32],
        scope: ReplayScope<'_>,
        seed: Option<&ReplaySeed>,
        steps: &[ReplayStep],
    ) -> Result<Vec<ReplayedPosition>> {
        *self.steps.borrow_mut() = steps.to_vec();
        *self.seed.borrow_mut() = seed.cloned();
        self.runs.borrow_mut().push((
            seed.map(ReplaySeed::indices).unwrap_or_default(),
            steps.iter().map(step_capture).collect(),
        ));
        *self.layers.borrow_mut() = layers.to_vec();
        self.replays.set(self.replays.get() + 1);
        let name = match scope {
            ReplayScope::Full => "full",
            ReplayScope::Committed => "committed",
            ReplayScope::Anchored(anchors) => {
                self.anchors.borrow_mut().extend_from_slice(anchors);
                "anchored"
            }
        };
        self.scopes.borrow_mut().push(name);
        let corrupt = self.corrupt || (self.fused && scope != ReplayScope::Committed);
        let mut out = Vec::new();
        for step in steps {
            let ReplayStep::Decode {
                record,
                capture,
                mixing,
            } = step
            else {
                continue;
            };
            let columns = Columns::new(record, &self.stage, layers);
            for col in 0..record.pos.len() {
                let position = u32::try_from(record.pos[col])?;
                if record.seq[col] != seq {
                    continue;
                }
                if mixing.contains(&position) {
                    for &layer in layers {
                        let values = ReplayedLayer::from_mixing_columns(layer, &columns, col);
                        out.push((position, layer, values));
                    }
                }
                if !capture.contains(&position) {
                    continue;
                }
                for &layer in layers {
                    let mut values = ReplayedLayer::from_columns(layer, &columns, col)?;
                    values.attention.extend(self.attention.clone().map(Ok));
                    if corrupt && let Some((_, bytes)) = &mut values.mixing[0].value {
                        bytes[0] ^= 1;
                    }
                    out.push((position, layer, values));
                }
            }
        }
        Ok(out)
    }

    /// The stored row the test model writes at each cell's position, for a
    /// token [`ubatch`] puts there.
    fn read_rows(&self, layer: u32, cells: &[PrefixCell]) -> Result<Vec<StoredRows>> {
        let values = cells
            .iter()
            .flat_map(|c| stored_row(layer, 1000 + c.pos, c.pos))
            .collect();
        Ok(vec![StoredRows {
            ggml_type: CACHE_TYPE,
            row_bytes: 6,
            values,
        }])
    }

    fn pin_cells(&self, _cells: &[KvCell]) -> Result<()> {
        Ok(())
    }
}

fn step_capture(step: &ReplayStep) -> Vec<u32> {
    match step {
        ReplayStep::Decode { capture, .. } => capture.clone(),
        ReplayStep::SeqRm { .. } => Vec::new(),
    }
}

fn stored_row(layer: u32, token: i32, position: i32) -> Vec<u8> {
    value(layer, TensorKind::MixingStored, 0, token, position)
        .iter()
        .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
        .collect()
}

/// A two-stage node: both stages record the same ubatches.
struct Node {
    stages: [Recorder; 2],
    replayers: [FakeReplayer; 2],
}

impl Node {
    fn new() -> Self {
        Self::with_stages([Stage::new(0), Stage::new(1)])
    }

    fn with_stages(stages: [Stage; 2]) -> Self {
        Self {
            stages: [0, 1].map(|i| {
                Recorder::new(i as u32, stages[i].layers.clone(), "cpu").holding_output(i == 1)
            }),
            replayers: stages.map(FakeReplayer::new),
        }
    }

    fn record(&self, record: UbatchRecord) {
        for (recorder, replayer) in self.stages.iter().zip(&self.replayers) {
            record_on(recorder, &replayer.stage, record.clone());
        }
    }

    /// A prompt of three tokens for sequence 7, then a decode step batched
    /// with sequence 9.
    fn two_lane_turn() -> Self {
        let node = Self::new();
        node.record(ubatch(&[(7, 0), (7, 1), (7, 2)], &[0, 1, 2], true));
        node.record(ubatch(&[(9, 40), (7, 3)], &[8, 3], true));
        node
    }

    /// Commits sequence `seq` at `positions` on both stages, generation
    /// starting at `generation_start`.
    fn commit(&self, seq: i32, positions: Range<u32>, generation_start: u32) -> Receipt {
        let stages = self
            .stages
            .iter()
            .map(|s| {
                s.commit("r1", seq, TurnId::default(), positions.clone())
                    .unwrap()
            })
            .collect();
        Receipt {
            version: RECEIPT_VERSION,
            receipt_id: "r1".into(),
            model: "m".into(),
            position_start: positions.start,
            position_end: positions.end,
            tokens: self.stages[0].turn_tokens("r1").unwrap(),
            generation_start,
            final_token: Some(2000),
            stages,
            binding: None,
        }
    }

    fn open(&self, challenge: &Challenge) -> Result<Opening> {
        let stages = self
            .stages
            .iter()
            .zip(&self.replayers)
            .map(|(recorder, replayer)| recorder.open(challenge, replayer))
            .collect::<Result<_>>()?;
        Ok(Opening {
            receipt_id: challenge.receipt_id.clone(),
            stages,
        })
    }
}

fn challenge(picks: &[(u32, &[u32])]) -> Challenge {
    Challenge {
        receipt_id: "r1".into(),
        picks: picks
            .iter()
            .map(|(layer, positions)| Pick {
                layer: *layer,
                positions: positions.to_vec(),
            })
            .collect(),
        proof: None,
        head: Vec::new(),
        attention: Vec::new(),
    }
}

/// Position 1 is not an output row: on the model's final layer its
/// residual out, FFN matmuls and selection are absent. Position 3 has them
/// all.
fn check_final_layer(opening: &Opening) {
    let layer = &opening.stages[1].layers[0];
    assert_eq!(layer.layer, 3);
    let (early, output) = (&layer.leaves[0], &layer.leaves[1]);
    assert!(early.residual_out.is_none());
    assert_eq!(early.matrices.len(), 4);
    assert!(early.selections.is_empty());
    assert_eq!(output.matrices.len(), 6);
    assert_eq!(output.mixing.len(), 3);
    assert_eq!(output.mixing[1].kind, crate::wire::MixingKind::Stored);
    assert_eq!(output.mixing[1].ggml_type, CACHE_TYPE);
    assert!(output.matrices[4].router);
    assert_eq!(output.matrices[5].experts, Some(vec![3, 1]));
    assert_eq!(output.selections[0].ids, vec![3, 1]);
    assert_eq!(output.matrices[2].weight, "blk.3.w2.weight");
}

#[test]
fn a_replayed_opening_verifies_against_the_roots() {
    let node = Node::two_lane_turn();
    assert_eq!(
        node.stages[0].recorded_range(7, TurnId::default()),
        Some(0..4)
    );
    let receipt = node.commit(7, 0..4, 3);
    assert_eq!(receipt.stages[1].leaf_count, 4 * 5);
    assert_eq!(receipt.tokens, vec![1000, 1001, 1002, 1003]);
    verify_shape(&receipt, MODEL_LAYERS).unwrap();
    let challenge = challenge(&[(1, &[0, 2]), (3, &[1, 3])]);
    let opening = node.open(&challenge).unwrap();
    assert_eq!(receipt.stages[1].cache_types, vec!["f16".to_string()]);
    verify_paths(&receipt, &challenge, &opening).unwrap();
    check_final_layer(&opening);
    // the input of layer 0 and the boundary at every picked position
    let positions = [0, 1, 2, 3];
    let anchors: Vec<u32> = opening.stages[0]
        .anchors
        .iter()
        .map(|a| a.position)
        .collect();
    assert_eq!(anchors, positions);
    assert_eq!(opening.stages[0].anchors[3].token, 1003);
    assert_eq!(opening.stages[1].entry.len(), positions.len());
    assert!(opening.stages[0].entry.is_empty());

    // a receipt that does not name the stored rows' type is refused
    let mut unnamed = receipt;
    unnamed.stages[1].cache_types = vec!["q8_0".into()];
    assert!(verify_paths(&unnamed, &challenge, &opening).is_err());

    // stage 0 replays layer 1 and layer 0 for the anchors, in one pass
    assert_eq!(*node.replayers[0].layers.borrow(), vec![0, 1]);
    let steps = node.replayers[1].steps.borrow();
    let ReplayStep::Decode {
        record, capture, ..
    } = &steps[1]
    else {
        panic!("decode step");
    };
    // The other lane's column is a dummy at its recorded cell.
    assert_eq!(record.seq, vec![9, 7]);
    assert_eq!(record.token, vec![DUMMY_TOKEN, 1003]);
    assert_eq!(record.pos, vec![0, 3]);
    assert_eq!(record.cell, vec![8, 3]);
    assert_eq!(
        record.input.as_ref().unwrap().rows,
        vec![0, 0, 0, 0, 7, 7, 7, 7]
    );
    assert_eq!(capture, &vec![3]);
}

#[test]
fn the_input_of_layer_0_binds_the_token() {
    let node = Node::two_lane_turn();
    let receipt = node.commit(7, 0..4, 3);
    let challenge = challenge(&[(2, &[1, 3])]);
    let opening = node.open(&challenge).unwrap();
    verify_paths(&receipt, &challenge, &opening).unwrap();
    // a receipt over other tokens no longer matches the committed leaves
    let mut other = receipt.clone();
    other.tokens[3] = 1500;
    assert!(verify_paths(&other, &challenge, &opening).is_err());
    // and an anchor that names those tokens no longer hashes to its leaf
    let mut forged = opening.clone();
    forged.stages[0].anchors[1].token = 1500;
    assert!(verify_paths(&other, &challenge, &forged).is_err());
    // an anchor must be opened at every picked position
    let mut missing = opening;
    missing.stages[0].anchors.remove(0);
    assert!(verify_paths(&receipt, &challenge, &missing).is_err());
}

#[test]
fn a_stage_must_receive_what_the_previous_stage_sent() {
    let mut shifted = Stage::new(1);
    shifted.entry_offset = 0.5;
    let node = Node::with_stages([Stage::new(0), shifted]);
    node.record(ubatch(&[(7, 0), (7, 1)], &[0, 1], true));
    let receipt = node.commit(7, 0..2, 1);
    let challenge = challenge(&[(2, &[0, 1])]);
    let opening = node.open(&challenge).unwrap();
    let error = verify_paths(&receipt, &challenge, &opening).unwrap_err();
    assert!(
        error.to_string().contains("different residual"),
        "{error:#}"
    );
}

#[test]
fn a_stage_without_picks_opens_its_boundaries_without_a_replay() {
    let node = Node::two_lane_turn();
    let receipt = node.commit(7, 0..4, 3);
    let challenge = challenge(&[(1, &[2, 3])]);
    let opening = node.open(&challenge).unwrap();
    verify_paths(&receipt, &challenge, &opening).unwrap();
    assert_eq!(node.replayers[1].replays.get(), 0);
    assert!(opening.stages[1].layers.is_empty());
    assert_eq!(opening.stages[1].entry.len(), 2);
}

#[test]
fn a_replay_that_diverges_is_refused() {
    let node = Node::two_lane_turn();
    node.commit(7, 0..4, 3);
    let mut replayer = FakeReplayer::for_stage(1);
    replayer.corrupt = true;
    let error = node.stages[1]
        .open(&challenge(&[(2, &[2])]), &replayer)
        .unwrap_err();
    assert!(
        format!("{error:#}").contains("replay diverged"),
        "{error:#}"
    );
}

#[test]
fn a_full_replay_that_diverges_is_anchored_to_the_committed_values() {
    let node = Node::two_lane_turn();
    node.commit(7, 0..4, 3);
    let mut replayer = FakeReplayer::for_stage(1);
    replayer.fused = true;
    let challenge = challenge(&[(2, &[2])]);
    let opened = node.stages[1].open(&challenge, &replayer).unwrap();
    assert_eq!(*replayer.scopes.borrow(), ["full", "committed", "anchored"]);
    let anchors = replayer.anchors.borrow();
    assert!(!anchors.is_empty());
    assert!(anchors.iter().all(|a| a.pos == 2 && a.seq == 7));
    // a fresh node: openings are cached
    let fresh = Node::two_lane_turn();
    fresh.commit(7, 0..4, 3);
    let honest = FakeReplayer::for_stage(1);
    let expected = fresh.stages[1].open(&challenge, &honest).unwrap();
    assert_eq!(opened, expected);
    assert_eq!(*honest.scopes.borrow(), ["full"]);
}

#[test]
fn the_final_stage_opens_the_output_residual() {
    let node = Node::two_lane_turn();
    let receipt = node.commit(7, 0..4, 3);
    assert_eq!(receipt.output_positions(), 2..4);
    let mut challenge = challenge(&[(1, &[0])]);
    challenge.head = vec![2, 3];
    let opening = node.open(&challenge).unwrap();
    verify_paths(&receipt, &challenge, &opening).unwrap();
    assert!(opening.stages[0].head.is_empty());
    let head = &opening.stages[1].head;
    assert_eq!(head.iter().map(|h| h.position).collect::<Vec<_>>(), [2, 3]);
    // the final stage replays its last layer for the head alone
    assert_eq!(*node.replayers[1].layers.borrow(), vec![3]);
    assert_eq!(
        head[1].residual_values().unwrap(),
        value(4, TensorKind::Residual, 0, 1003, 3)
    );

    let mut forged = opening.clone();
    forged.stages[1].head[0].residual = encode_f32(&[1.0, 2.0, 3.0]);
    assert!(verify_paths(&receipt, &challenge, &forged).is_err());
    let mut missing = opening;
    missing.stages[1].head.pop();
    assert!(verify_paths(&receipt, &challenge, &missing).is_err());

    // position 1 is not an output row: no token was sampled from it
    let mut early = challenge.clone();
    early.head = vec![1];
    let error = node.stages[1].open(&early, &node.replayers[1]).unwrap_err();
    assert!(
        format!("{error:#}").contains("not an output row"),
        "{error:#}"
    );
}

#[test]
fn a_forged_opening_fails_its_path() {
    let node = Node::two_lane_turn();
    let receipt = node.commit(7, 0..4, 3);
    let challenge = challenge(&[(2, &[0])]);
    let mut opening = node.open(&challenge).unwrap();
    opening.stages[1].layers[0].leaves[0].residual_in = encode_f32(&[1.0, 2.0, 3.0]);
    assert!(verify_paths(&receipt, &challenge, &opening).is_err());
}

#[test]
fn removed_positions_are_replayed_as_removals() {
    let recorder = Recorder::new(1, STAGES[1].clone(), "cpu");
    record(&recorder, ubatch(&[(1, 0), (1, 1)], &[0, 1], true));
    // A rejected draft: position 3 is computed, removed, and computed again.
    record(&recorder, ubatch(&[(1, 2), (1, 3)], &[2, 3], true));
    recorder.memory_op(MemoryOp::SeqRm, 1, -1, 3, -1);
    record(&recorder, ubatch(&[(1, 3)], &[5], true));
    recorder.commit("r1", 1, TurnId::default(), 0..4).unwrap();
    // serving drops the turn's cells from position 2, and another request
    // writes cell 2, so the replay starts at the ubatch that wrote position
    // 2, before the removal
    recorder.memory_op(MemoryOp::SeqRm, 1, -1, 2, -1);
    record(&recorder, ubatch(&[(2, 0)], &[2], true));
    let replayer = FakeReplayer::for_stage(1);
    recorder.open(&challenge(&[(2, &[3])]), &replayer).unwrap();
    let steps = replayer.steps.borrow();
    assert_eq!(steps.len(), 3);
    assert_eq!(steps[1], ReplayStep::SeqRm { p0: 3, p1: -1 });
    let ReplayStep::Decode {
        record, capture, ..
    } = &steps[2]
    else {
        panic!("decode step");
    };
    assert_eq!((record.cell.clone(), capture.clone()), (vec![5], vec![3]));
    let seed = replayer.seed.borrow().clone().expect("a seeded replay");
    assert_eq!(seed.positions(), vec![0, 1]);
}

#[test]
fn a_cleared_turn_can_still_be_committed() {
    let recorder = Recorder::new(1, STAGES[1].clone(), "cpu");
    record(&recorder, ubatch(&[(1, 0), (1, 1)], &[0, 1], true));
    recorder.memory_op(MemoryOp::SeqRm, 1, -1, -1, -1);
    assert_eq!(recorder.recorded_range(1, TurnId::default()), Some(0..2));
    recorder.commit("r1", 1, TurnId::default(), 0..2).unwrap();
    assert_eq!(recorder.recorded_range(1, TurnId::default()), None);
}

#[test]
fn a_shared_prefix_seeds_the_replay() {
    let recorder = Recorder::new(1, STAGES[1].clone(), "cpu");
    // a cached prompt in sequence 3, shared with sequence 1, which continues it
    record(
        &recorder,
        ubatch(&[(3, 0), (3, 1), (3, 2)], &[0, 1, 2], true),
    );
    recorder.memory_op(MemoryOp::SeqCp, 3, 1, -1, -1);
    record(&recorder, ubatch(&[(1, 3), (1, 4)], &[3, 4], true));
    let commitment = recorder.commit("r1", 1, TurnId::default(), 3..5).unwrap();
    let prefix = commitment.prefix.expect("a prefix commitment");
    assert_eq!(prefix.positions, 3);

    let replayer = FakeReplayer::for_stage(1);
    recorder.open(&challenge(&[(2, &[4])]), &replayer).unwrap();
    // only the turn's own ubatch is replayed, on top of the shared cells
    assert_eq!(replayer.steps.borrow().len(), 1);
    let seed = replayer.seed.borrow().clone().expect("a seeded replay");
    assert_eq!(seed.indices(), vec![0, 1, 2]);
    assert_eq!(seed.positions(), vec![0, 1, 2]);
    assert_eq!(seed.n_cells, 256);
    // a later stage commits the prefix's cells and leaves, not its tokens
    assert_eq!(prefix.root.len(), 64);
    assert_eq!(prefix.tokens, None);
    recorder.check_seed(&seed).unwrap();

    // once another sequence reuses a cell, the prefix is gone
    recorder.memory_op(MemoryOp::SeqRm, 3, -1, -1, -1);
    recorder.memory_op(MemoryOp::SeqRm, 1, -1, -1, -1);
    record(&recorder, ubatch(&[(5, 0)], &[1], true));
    let error = recorder.check_seed(&seed).unwrap_err();
    assert!(format!("{error:#}").contains("reused"), "{error:#}");
}

#[test]
fn a_continued_sequence_replays_only_its_new_turn() {
    let recorder = Recorder::new(1, STAGES[1].clone(), "cpu");
    record(
        &recorder,
        ubatch(&[(1, 0), (1, 1), (1, 2)], &[0, 1, 2], true),
    );
    assert!(
        recorder
            .commit("r1", 1, TurnId::default(), 0..3)
            .unwrap()
            .prefix
            .is_none()
    );
    record(&recorder, ubatch(&[(1, 3)], &[3], true));
    record(&recorder, ubatch(&[(1, 4)], &[4], true));
    assert_eq!(recorder.recorded_range(1, TurnId::default()), Some(3..5));
    let commitment = recorder.commit("r2", 1, TurnId::default(), 3..5).unwrap();
    assert_eq!(commitment.ubatches, vec![3, 4]);
    assert_eq!(commitment.prefix.unwrap().positions, 3);
    let replayer = FakeReplayer::for_stage(1);
    let mut challenge = challenge(&[(3, &[3])]);
    challenge.receipt_id = "r2".into();
    recorder.open(&challenge, &replayer).unwrap();
    assert_eq!(replayer.steps.borrow().len(), 1);
    assert_eq!(
        replayer.seed.borrow().as_ref().unwrap().indices(),
        vec![0, 1, 2]
    );
}

#[test]
fn cells_taken_mid_turn_make_the_turn_unreceiptable() {
    let recorder = Recorder::new(1, STAGES[1].clone(), "cpu");
    record(&recorder, ubatch(&[(3, 0), (3, 1)], &[0, 1], true));
    record(&recorder, ubatch(&[(1, 0), (1, 1)], &[4, 5], true));
    recorder.memory_op(MemoryOp::SeqCp, 3, 1, 0, 2);
    let error = recorder
        .commit("r1", 1, TurnId::default(), 0..2)
        .unwrap_err();
    assert!(format!("{error:#}").contains("mid-turn"), "{error:#}");
    // the next turn starts afresh
    recorder.memory_op(MemoryOp::SeqRm, 1, -1, -1, -1);
    record(&recorder, ubatch(&[(1, 0)], &[4], true));
    recorder.commit("r2", 1, TurnId::default(), 0..1).unwrap();
}

#[test]
fn a_turn_after_a_shift_has_no_known_prefix() {
    let recorder = Recorder::new(1, STAGES[1].clone(), "cpu");
    record(&recorder, ubatch(&[(1, 0), (1, 1)], &[0, 1], true));
    recorder.commit("r1", 1, TurnId::default(), 0..2).unwrap();
    recorder.memory_op(MemoryOp::SeqShift, 1, -1, 0, -1);
    record(&recorder, ubatch(&[(1, 2)], &[2], true));
    let error = recorder
        .commit("r2", 1, TurnId::default(), 2..3)
        .unwrap_err();
    assert!(
        format!("{error:#}").contains("prefix is unknown"),
        "{error:#}"
    );
}

#[test]
fn a_turn_on_a_quantized_kv_cache_is_unreceiptable() {
    // An honest stage with a q8_0 cache fails the loose attention check,
    // so its turns get no receipt, whatever else is in order.
    for quantized in [crate::ggml_types::GGML_Q8_0, crate::ggml_types::GGML_Q4_0] {
        let recorder = Recorder::new(1, STAGES[1].clone(), "cpu");
        let turn = ubatch(&[(1, 0), (1, 1)], &[0, 1], true);
        let mut columns = Columns::new(&turn, &Stage::new(1), &[]);
        for (key, column) in &mut columns.0 {
            if key.1 == TensorKind::MixingStored {
                column.0 = quantized as i32;
            }
        }
        recorder.record_ubatch(turn, &columns);
        let error = recorder
            .commit("r", 1, TurnId::default(), 0..2)
            .unwrap_err();
        let message = format!("{error:#}");
        assert!(message.contains("quantized KV cache"), "{message}");
        assert!(
            message.contains(&crate::ggml_types::type_name(quantized)),
            "{message}"
        );
    }
    // the same turn on an f16 cache commits
    let recorder = Recorder::new(1, STAGES[1].clone(), "cpu");
    record(&recorder, ubatch(&[(1, 0), (1, 1)], &[0, 1], true));
    recorder.commit("r", 1, TurnId::default(), 0..2).unwrap();
}

#[test]
fn a_gap_in_the_turn_is_refused() {
    let recorder = Recorder::new(1, STAGES[1].clone(), "cpu");
    record(&recorder, ubatch(&[(1, 0)], &[0], true));
    record(&recorder, ubatch(&[(1, 2)], &[2], true));
    assert!(recorder.commit("r", 1, TurnId::default(), 0..3).is_err());
}

#[test]
fn a_repeated_challenge_is_answered_without_a_replay() {
    let node = Node::two_lane_turn();
    let receipt = node.commit(7, 0..4, 3);
    let challenge = challenge(&[(3, &[1, 3])]);
    let first = node.open(&challenge).unwrap();
    let again = node.open(&challenge).unwrap();
    assert_eq!(node.replayers[1].replays.get(), 1);
    assert_eq!(first, again);
    verify_paths(&receipt, &challenge, &again).unwrap();
}

#[test]
fn a_new_turn_under_the_same_id_drops_its_old_openings() {
    let node = Node::two_lane_turn();
    node.commit(7, 0..4, 3);
    let challenge = challenge(&[(2, &[3])]);
    let first = node.open(&challenge).unwrap();
    node.record(ubatch(&[(7, 4)], &[4], true));
    node.commit(7, 4..5, 4);
    // position 3 is a position of the new turn's prefix: the stage opens
    // it anew, and not with what it opened for the old turn
    let again = node.open(&challenge).unwrap();
    assert_ne!(first, again);
    assert_eq!(again.stages[1].prefix.len(), 1);
    assert_eq!(node.replayers[1].replays.get(), 2);
    // a position past the new turn is outside it
    let mut past = challenge.clone();
    past.picks[0].positions = vec![5];
    let error = node.open(&past).unwrap_err();
    assert!(
        format!("{error:#}").contains("outside the receipt"),
        "{error:#}"
    );
}

#[test]
fn a_receipt_is_replayed_for_a_bounded_number_of_challenges() {
    let node = Node::two_lane_turn();
    node.commit(7, 0..4, 3);
    let recorder = &node.stages[1];
    let replayer = FakeReplayer::for_stage(1);
    let positions: [&[u32]; 5] = [&[0], &[1], &[2], &[3], &[0, 1]];
    for (i, positions) in positions.into_iter().enumerate() {
        let opened = recorder.open(&challenge(&[(2, positions)]), &replayer);
        if i < MAX_REPLAYS_PER_RECEIPT as usize {
            opened.unwrap();
        } else {
            let error = opened.unwrap_err();
            assert!(
                format!("{error:#}").contains("already replayed"),
                "{error:#}"
            );
        }
    }
    // an opening already made is still answered
    recorder.open(&challenge(&[(2, &[0])]), &replayer).unwrap();
    assert_eq!(replayer.replays.get(), MAX_REPLAYS_PER_RECEIPT);
}

#[test]
fn a_receipt_must_have_the_shape_it_names() {
    let node = Node::two_lane_turn();
    let receipt = node.commit(7, 0..4, 3);
    verify_shape(&receipt, MODEL_LAYERS).unwrap();

    let mut other = receipt.clone();
    other.version += 1;
    assert!(verify_shape(&other, MODEL_LAYERS).is_err());

    let mut grown = receipt.clone();
    grown.stages[1].leaf_count += 1;
    assert!(verify_shape(&grown, MODEL_LAYERS).is_err());

    // the stages must cover every layer of the model, from 0
    assert!(verify_shape(&receipt, MODEL_LAYERS + 1).is_err());
    let mut truncated = receipt.clone();
    truncated.stages.pop();
    assert!(verify_shape(&truncated, MODEL_LAYERS).is_err());
    let mut reordered = receipt.clone();
    reordered.stages.swap(0, 1);
    assert!(verify_shape(&reordered, MODEL_LAYERS).is_err());

    let mut short = receipt.clone();
    short.tokens.pop();
    assert!(verify_shape(&short, MODEL_LAYERS).is_err());

    // only the model's final layer may keep no row for a position
    let challenge = challenge(&[(2, &[0, 1])]);
    let mut dropped = node.open(&challenge).unwrap();
    dropped.stages[1].layers[0].leaves[1].residual_out = None;
    let error = verify_paths(&receipt, &challenge, &dropped).unwrap_err();
    assert!(
        error.to_string().contains("opened no residual out"),
        "{error:#}"
    );
}

#[path = "attention_tests.rs"]
mod attention_tests;

#[path = "recurrent_tests.rs"]
mod recurrent_tests;

#[path = "stream_tests.rs"]
mod stream_tests;

#[path = "turn_seed_tests.rs"]
mod turn_seed_tests;

#[path = "turn_identity_tests.rs"]
mod turn_identity_tests;

#[path = "cannot_open_tests.rs"]
mod cannot_open_tests;

#[path = "prefix_tests.rs"]
mod prefix_tests;
