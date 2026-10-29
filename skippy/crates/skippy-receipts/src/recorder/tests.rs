use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use super::*;
use crate::audit::verify_paths;
use crate::wire::{RECEIPT_VERSION, Receipt};

const LAYERS: Range<u32> = 2..4;

/// A deterministic stand-in for the values a layer computes for a token.
fn value(layer: u32, kind: TensorKind, token: i32, position: i32) -> Vec<f32> {
    let base =
        layer as f32 * 100.0 + kind as u8 as f32 + token as f32 * 0.25 + position as f32 * 0.5;
    vec![base, base + 1.0, -base]
}

/// Columns of a ubatch computed by `value`. The final residual of the last
/// layer exists only for output rows, as on a final stage.
struct Columns(HashMap<(u32, TensorKind, usize), Vec<f32>>);

impl Columns {
    fn new(record: &UbatchRecord, open_layer: Option<u32>) -> Self {
        let mut map = HashMap::new();
        for col in 0..record.token.len() {
            let (token, pos) = (record.token[col], record.pos[col]);
            for layer in LAYERS {
                for kind in [
                    TensorKind::Residual,
                    TensorKind::AttnQ,
                    TensorKind::AttnK,
                    TensorKind::AttnV,
                    TensorKind::AttnOut,
                ] {
                    map.insert((layer, kind, col), value(layer, kind, token, pos));
                }
                if open_layer == Some(layer) {
                    for kind in [
                        TensorKind::AttnIn,
                        TensorKind::AttnProj,
                        TensorKind::FfnIn,
                        TensorKind::FfnGate,
                        TensorKind::FfnUp,
                        TensorKind::FfnDownIn,
                        TensorKind::FfnDown,
                    ] {
                        map.insert((layer, kind, col), value(layer, kind, token, pos));
                    }
                }
            }
            if record.output[col] != 0 {
                map.insert(
                    (LAYERS.end, TensorKind::Residual, col),
                    value(LAYERS.end, TensorKind::Residual, token, pos),
                );
            }
        }
        Self(map)
    }
}

impl UbatchColumns for Columns {
    fn column(&self, layer: u32, kind: TensorKind, col: usize) -> Option<&[f32]> {
        self.0.get(&(layer, kind, col)).map(Vec::as_slice)
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
                    if self.corrupt {
                        layer_values.q[0] += 1.0;
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
    verify_paths(&receipt(stage, 0..4), &challenge, &opening).unwrap();
    // Position 1 is not an output row: its final residual and FFN are absent.
    assert!(opening.leaves[0].residual_out.is_none());
    assert_eq!(opening.leaves[0].matrices.len(), 4);
    assert_eq!(opening.leaves[1].matrices.len(), 7);

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
