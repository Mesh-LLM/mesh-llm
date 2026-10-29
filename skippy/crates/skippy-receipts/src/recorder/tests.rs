use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use super::*;
use crate::audit::verify_paths;
use crate::layer_tensors::{GGML_I32, TensorColumn, f32_bytes, i32_bytes};
use crate::wire::{RECEIPT_VERSION, Receipt, encode_f32};

const LAYERS: Range<u32> = 2..4;
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

/// Columns of a ubatch computed by `value`: per layer a mixing input, a
/// stored row and a mixing output; in the opened layer also four attention
/// matmuls, and for output rows two FFN matmuls (one with experts) and a
/// selection. The final residual of the last layer exists only for output
/// rows, as on a final stage.
struct Columns(HashMap<Key, (i32, i32, String, Vec<u8>)>);

impl Columns {
    fn new(record: &UbatchRecord, open_layer: Option<u32>) -> Self {
        let mut columns = Self(HashMap::new());
        for col in 0..record.token.len() {
            let (token, pos) = (record.token[col], record.pos[col]);
            let f32_at = |layer, kind, index| f32_bytes(&value(layer, kind, index, token, pos));
            for layer in LAYERS {
                for kind in [
                    TensorKind::Residual,
                    TensorKind::MixingInput,
                    TensorKind::MixingOutput,
                ] {
                    columns.put(
                        (layer, kind, 0, col),
                        GGML_F32,
                        0,
                        "",
                        f32_at(layer, kind, 0),
                    );
                }
                let stored = value(layer, TensorKind::MixingStored, 0, token, pos)
                    .iter()
                    .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
                    .collect();
                columns.put(
                    (layer, TensorKind::MixingStored, 0, col),
                    CACHE_TYPE,
                    0,
                    "",
                    stored,
                );
                if open_layer == Some(layer) {
                    let matmuls = if record.output[col] != 0 { 6 } else { 4 };
                    columns.add_matmuls(layer, col, matmuls, &f32_at);
                }
            }
            if record.output[col] != 0 {
                let key = (LAYERS.end, TensorKind::Residual, 0, col);
                columns.put(
                    key,
                    GGML_F32,
                    0,
                    "",
                    f32_at(LAYERS.end, TensorKind::Residual, 0),
                );
            }
        }
        columns
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

fn record(recorder: &Recorder, record: UbatchRecord) {
    let columns = Columns::new(&record, None);
    recorder.record_ubatch(record, &columns);
}

/// Replays by recomputing `value`, and keeps the steps it was given.
#[derive(Default)]
struct FakeReplayer {
    steps: RefCell<Vec<ReplayStep>>,
    replays: Cell<u32>,
    corrupt: bool,
}

impl Replayer for FakeReplayer {
    fn replay(
        &self,
        seq: i32,
        layer: u32,
        steps: &[ReplayStep],
    ) -> Result<Vec<(u32, ReplayedLayer)>> {
        *self.steps.borrow_mut() = steps.to_vec();
        self.replays.set(self.replays.get() + 1);
        let mut out = Vec::new();
        for step in steps {
            let ReplayStep::Decode { record, capture } = step else {
                continue;
            };
            let columns = Columns::new(record, Some(layer));
            for col in 0..record.pos.len() {
                let position = u32::try_from(record.pos[col])?;
                if record.seq[col] == seq && capture.contains(&position) {
                    let mut layer_values = ReplayedLayer::from_columns(layer, &columns, col)?;
                    if self.corrupt
                        && let Some((_, bytes)) = &mut layer_values.mixing[0].value
                    {
                        bytes[0] ^= 1;
                    }
                    out.push((position, layer_values));
                }
            }
        }
        Ok(out)
    }
}

fn receipt(stage: StageCommitment, positions: Range<u32>) -> Receipt {
    Receipt {
        version: RECEIPT_VERSION,
        receipt_id: "r1".into(),
        model: "m".into(),
        position_start: positions.start,
        position_end: positions.end,
        stages: vec![stage],
    }
}

/// A prompt of three tokens for sequence 7, then a decode step batched with
/// sequence 9.
fn two_lane_turn() -> Recorder {
    let recorder = Recorder::new(1, LAYERS, "cpu");
    record(
        &recorder,
        ubatch(&[(7, 0), (7, 1), (7, 2)], &[0, 1, 2], true),
    );
    record(&recorder, ubatch(&[(9, 40), (7, 3)], &[8, 3], true));
    recorder
}

/// Position 1 is not an output row: its final residual, FFN matmuls and
/// selection are absent. Position 3 has them all.
fn check_opened_layer(opening: &Opening) {
    let (early, output) = (&opening.leaves[0], &opening.leaves[1]);
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
fn a_replayed_opening_verifies_against_the_root() {
    let recorder = two_lane_turn();
    assert_eq!(recorder.recorded_range(7), Some(0..4));
    let stage = recorder.commit("r1", 7, 0..4).unwrap();
    assert_eq!(stage.leaf_count, 4 * 5);
    let challenge = Challenge {
        receipt_id: "r1".into(),
        layer: 3,
        positions: vec![1, 3],
    };
    let replayer = FakeReplayer::default();
    let opening = recorder.open(&challenge, &replayer).unwrap();
    assert_eq!(stage.cache_types, vec!["f16".to_string()]);
    verify_paths(&receipt(stage.clone(), 0..4), &challenge, &opening).unwrap();
    check_opened_layer(&opening);

    // a receipt that does not name the stored rows' type is refused
    let mut unnamed = stage;
    unnamed.cache_types = vec!["q8_0".into()];
    assert!(verify_paths(&receipt(unnamed, 0..4), &challenge, &opening).is_err());

    let steps = replayer.steps.borrow();
    let ReplayStep::Decode { record, capture } = &steps[1] else {
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
fn a_replay_that_diverges_is_refused() {
    let recorder = two_lane_turn();
    recorder.commit("r1", 7, 0..4).unwrap();
    let challenge = Challenge {
        receipt_id: "r1".into(),
        layer: 2,
        positions: vec![2],
    };
    let replayer = FakeReplayer {
        corrupt: true,
        ..Default::default()
    };
    let error = recorder.open(&challenge, &replayer).unwrap_err();
    assert!(
        format!("{error:#}").contains("replay diverged"),
        "{error:#}"
    );
}

#[test]
fn a_forged_opening_fails_its_path() {
    let recorder = two_lane_turn();
    let stage = recorder.commit("r1", 7, 0..4).unwrap();
    let challenge = Challenge {
        receipt_id: "r1".into(),
        layer: 2,
        positions: vec![0],
    };
    let mut opening = recorder.open(&challenge, &FakeReplayer::default()).unwrap();
    opening.leaves[0].residual_in = encode_f32(&[1.0, 2.0, 3.0]);
    assert!(verify_paths(&receipt(stage, 0..4), &challenge, &opening).is_err());
}

#[test]
fn removed_positions_are_replayed_as_removals() {
    let recorder = Recorder::new(0, LAYERS, "cpu");
    record(&recorder, ubatch(&[(1, 0), (1, 1)], &[0, 1], true));
    // A rejected draft: position 2 is computed, removed, and computed again.
    record(&recorder, ubatch(&[(1, 2)], &[2], true));
    recorder.memory_op(1, 1, -1, 2, -1);
    record(&recorder, ubatch(&[(1, 2)], &[5], true));
    recorder.commit("r1", 1, 0..3).unwrap();
    let replayer = FakeReplayer::default();
    let challenge = Challenge {
        receipt_id: "r1".into(),
        layer: 2,
        positions: vec![2],
    };
    recorder.open(&challenge, &replayer).unwrap();
    let steps = replayer.steps.borrow();
    assert_eq!(steps.len(), 4);
    assert_eq!(steps[2], ReplayStep::SeqRm { p0: 2, p1: -1 });
    let ReplayStep::Decode { record, capture } = &steps[3] else {
        panic!("decode step");
    };
    assert_eq!((record.cell.clone(), capture.clone()), (vec![5], vec![2]));
}

#[test]
fn a_cleared_turn_can_still_be_committed() {
    let recorder = Recorder::new(0, LAYERS, "cpu");
    record(&recorder, ubatch(&[(1, 0), (1, 1)], &[0, 1], true));
    recorder.memory_op(1, 1, -1, -1, -1);
    assert_eq!(recorder.recorded_range(1), Some(0..2));
    recorder.commit("r1", 1, 0..2).unwrap();
    assert_eq!(recorder.recorded_range(1), None);
}

#[test]
fn shared_prefixes_make_a_sequence_unreceiptable() {
    let recorder = Recorder::new(0, LAYERS, "cpu");
    record(&recorder, ubatch(&[(1, 0), (1, 1)], &[0, 1], true));
    recorder.memory_op(2, 3, 1, 0, 2);
    let error = recorder.commit("r1", 1, 0..2).unwrap_err();
    assert!(
        format!("{error:#}").contains("shares KV cells"),
        "{error:#}"
    );
}

#[test]
fn a_gap_in_the_turn_is_refused() {
    let recorder = Recorder::new(0, LAYERS, "cpu");
    record(&recorder, ubatch(&[(1, 0)], &[0], true));
    record(&recorder, ubatch(&[(1, 2)], &[2], true));
    assert!(recorder.commit("r", 1, 0..3).is_err());
}

#[test]
fn a_repeated_challenge_is_answered_without_a_replay() {
    let recorder = two_lane_turn();
    let stage = recorder.commit("r1", 7, 0..4).unwrap();
    let challenge = Challenge {
        receipt_id: "r1".into(),
        layer: 3,
        positions: vec![1, 3],
    };
    let replayer = FakeReplayer::default();
    let first = recorder.open(&challenge, &replayer).unwrap();
    let again = recorder.open(&challenge, &replayer).unwrap();
    assert_eq!(replayer.replays.get(), 1);
    assert_eq!(first, again);
    verify_paths(&receipt(stage, 0..4), &challenge, &again).unwrap();
}

#[test]
fn a_new_turn_under_the_same_id_drops_its_old_openings() {
    let recorder = two_lane_turn();
    recorder.commit("r1", 7, 0..4).unwrap();
    let challenge = Challenge {
        receipt_id: "r1".into(),
        layer: 2,
        positions: vec![3],
    };
    let replayer = FakeReplayer::default();
    recorder.open(&challenge, &replayer).unwrap();
    record(&recorder, ubatch(&[(7, 4)], &[4], true));
    recorder.commit("r1", 7, 4..5).unwrap();
    let error = recorder.open(&challenge, &replayer).unwrap_err();
    assert!(
        format!("{error:#}").contains("outside the receipt"),
        "{error:#}"
    );
    assert_eq!(replayer.replays.get(), 1);
}

#[test]
fn a_receipt_is_replayed_for_a_bounded_number_of_challenges() {
    let recorder = two_lane_turn();
    recorder.commit("r1", 7, 0..4).unwrap();
    let replayer = FakeReplayer::default();
    let positions = [vec![0], vec![1], vec![2], vec![3], vec![0, 1]];
    for (i, positions) in positions.into_iter().enumerate() {
        let challenge = Challenge {
            receipt_id: "r1".into(),
            layer: 2,
            positions,
        };
        let opened = recorder.open(&challenge, &replayer);
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
    let challenge = Challenge {
        receipt_id: "r1".into(),
        layer: 2,
        positions: vec![0],
    };
    recorder.open(&challenge, &replayer).unwrap();
    assert_eq!(replayer.replays.get(), MAX_REPLAYS_PER_RECEIPT);
}
