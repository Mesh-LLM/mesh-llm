use super::*;
use crate::glue::{GlueOutput, GlueTensor, StoredRowError};
use crate::wire::{TensorOpening, encode_f32};

/// A toy recurrence: per position, output = state + input, and the state
/// becomes the output.
struct ToyRegion;

impl GlueEvaluator for ToyRegion {
    fn evaluate(&self, _: &GlueRequest) -> Result<GlueOutput> {
        bail!("the toy region has no glue")
    }

    fn evaluate_region(&self, request: &GlueRequest) -> Result<GlueOutput> {
        let get = |kind| {
            request
                .inputs
                .iter()
                .find(|i| i.kind == kind && i.index == 0)
                .map(|i| match &i.values {
                    GlueValues::F32(v) => v.clone(),
                    GlueValues::I32(_) => panic!("f32 expected"),
                })
                .unwrap()
        };
        let inputs = get(TensorKind::MixingInput);
        let mut state = get(TensorKind::MixingState);
        let mut outputs = Vec::new();
        for x in inputs.chunks(state.len()) {
            for (s, x) in state.iter_mut().zip(x) {
                *s += x;
            }
            outputs.extend(&state);
        }
        let tensor = |kind, values: Vec<f32>| GlueTensor {
            layer: request.layer,
            kind,
            index: 0,
            flags: 0,
            name: String::new(),
            len: 2,
            values: GlueValues::F32(values),
        };
        Ok(GlueOutput {
            tensors: vec![
                tensor(TensorKind::MixingOutput, outputs),
                tensor(TensorKind::MixingStateOut, state),
            ],
            unchecked_weights: Vec::new(),
            attention: Vec::new(),
        })
    }

    fn entry(&self, _: &[i32], _: &[i32]) -> Result<Vec<f32>> {
        bail!("no entry")
    }

    fn stored_row_error(&self, _: i32, _: &[u8], _: &[f32]) -> Result<StoredRowError> {
        bail!("no stored rows")
    }
}

fn value(kind: MixingKind, values: Option<[f32; 2]>) -> TensorOpening {
    TensorOpening {
        kind,
        index: 0,
        ggml_type: crate::layer_tensors::GGML_F32,
        values: values.map(|v| encode_f32(&v)),
    }
}

/// The honest region `start..start + inputs.len()` from `state`, with the
/// leaf after it.
fn region(start: u32, state: [f32; 2], inputs: &[[f32; 2]]) -> RecurrentOpening {
    let mut s = state;
    let mut leaves = Vec::new();
    for (i, x) in inputs.iter().enumerate() {
        let read = (i == 0).then_some(s);
        s = [s[0] + x[0], s[1] + x[1]];
        leaves.push(MixingLeafOpening {
            position: start + i as u32,
            mixing: vec![
                value(MixingKind::Input, Some(*x)),
                value(MixingKind::Output, Some(s)),
                value(MixingKind::State, read),
            ],
            path: Vec::new(),
        });
    }
    let end = start + inputs.len() as u32;
    leaves.push(MixingLeafOpening {
        position: end,
        mixing: vec![
            value(MixingKind::Input, Some([0.0, 0.0])),
            value(MixingKind::Output, Some(s)),
            value(MixingKind::State, Some(s)),
        ],
        path: Vec::new(),
    });
    RecurrentOpening {
        position: start,
        layer: 3,
        start,
        end,
        leaves,
    }
}

fn verdict<'a>(report: &'a RegionReport, check: &str) -> &'a GlueVerdict {
    report.verdicts.iter().find(|v| v.check == check).unwrap()
}

#[test]
fn an_honest_region_passes() {
    let opening = region(4, [1.0, -1.0], &[[0.5, 0.5], [2.0, 0.0], [0.0, 1.0]]);
    let report = check_region(&ToyRegion, &opening, 8).unwrap();
    assert!(report.pass, "{report:?}");
    assert_eq!(verdict(&report, "recurrent outputs").tokens, 3);
    assert_eq!(verdict(&report, "states left").tokens, 1);
}

#[test]
fn the_request_holds_the_inputs_and_the_state_read() {
    let opening = region(4, [1.0, -1.0], &[[0.5, 0.5], [2.0, 0.0]]);
    let request = region_request(&opening, 8).unwrap();
    assert_eq!(request.positions, vec![4, 5]);
    assert_eq!(request.residual, vec![0.0; 16]);
    let state = request
        .inputs
        .iter()
        .find(|i| i.kind == TensorKind::MixingState)
        .unwrap();
    assert_eq!(state.values, GlueValues::F32(vec![1.0, -1.0]));
}

#[test]
fn a_wrong_output_fails() {
    let mut opening = region(4, [1.0, -1.0], &[[0.5, 0.5], [2.0, 0.0]]);
    opening.leaves[1].mixing[1] = value(MixingKind::Output, Some([9.0, 9.0]));
    let report = check_region(&ToyRegion, &opening, 8).unwrap();
    assert!(!report.pass);
    assert!(!verdict(&report, "recurrent outputs").pass);
}

#[test]
fn a_state_the_ubatch_did_not_leave_fails() {
    let mut opening = region(4, [1.0, -1.0], &[[0.5, 0.5]]);
    opening.leaves[1].mixing[2] = value(MixingKind::State, Some([0.0, 0.0]));
    let report = check_region(&ToyRegion, &opening, 8).unwrap();
    assert!(!verdict(&report, "states left").pass);
}

#[test]
fn a_sequence_starts_from_zero() {
    let opening = region(0, [1.0, 0.0], &[[0.5, 0.5]]);
    let report = check_region(&ToyRegion, &opening, 8).unwrap();
    assert!(!verdict(&report, "first state").pass);
    let opening = region(0, [0.0, 0.0], &[[0.5, 0.5]]);
    assert!(check_region(&ToyRegion, &opening, 8).unwrap().pass);
}

#[test]
fn the_most_recent_of_the_snapshots_is_compared() {
    assert_eq!(most_recent(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2), &[1.0, 2.0]);
    assert_eq!(most_recent(&[1.0, 2.0, 3.0], 2), &[1.0, 2.0, 3.0]);
}
