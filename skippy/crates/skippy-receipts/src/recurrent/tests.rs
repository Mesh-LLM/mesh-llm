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
fn a_region_outside_the_turn_is_refused_before_it_sizes_anything() {
    use crate::node_fault::NodeFault;
    // the toy turn computes positions 0..5; the node names the ubatch
    let receipt = crate::toy_turn::Turn::new().receipt;
    let pick = |position| AttentionPick {
        position,
        layer: 1,
        head_draw: 0,
    };
    let named = |position, start, end| RecurrentOpening {
        position,
        layer: 1,
        start,
        end,
        leaves: Vec::new(),
    };
    for (position, start, end) in [
        // four billion positions, which the check used to list
        (2, 0, u32::MAX),
        (2, 2, 6),
        (2, 3, 3),
        (2, 4, 1),
        (u32::MAX - 1, u32::MAX - 2, u32::MAX),
    ] {
        let error =
            verify_region(&receipt, &pick(position), &named(position, start, end)).unwrap_err();
        assert!(NodeFault::find(&error).is_some(), "{error:#}");
        assert!(
            format!("{error:#}").contains("not positions of the turn"),
            "{start}..{end}: {error:#}"
        );
    }
    // inside the turn, the leaves are counted against the ubatch
    let error = verify_region(&receipt, &pick(2), &named(2, 0, 3)).unwrap_err();
    assert!(format!("{error:#}").contains("opens 0 leaves"), "{error:#}");
    // and a region of another pick is not this pick's
    assert!(verify_region(&receipt, &pick(1), &named(2, 0, 3)).is_err());
}

#[test]
fn values_that_are_not_numbers_fail_the_region() {
    for bad in [f32::NAN, f32::INFINITY] {
        let mut output = region(4, [1.0, -1.0], &[[0.5, 0.5], [2.0, 0.0]]);
        output.leaves[1].mixing[1] = value(MixingKind::Output, Some([bad, 2.0]));
        let report = check_region(&ToyRegion, &output, 8).unwrap();
        assert!(!verdict(&report, "recurrent outputs").pass, "{bad}");
        let mut state = region(4, [1.0, -1.0], &[[0.5, 0.5]]);
        state.leaves[1].mixing[2] = value(MixingKind::State, Some([bad, 0.0]));
        let report = check_region(&ToyRegion, &state, 8).unwrap();
        assert!(!verdict(&report, "states left").pass, "{bad}");
    }
}

#[test]
fn a_region_that_makes_no_request_is_the_nodes_fault() {
    use crate::node_fault::NodeFault;
    // an input the ubatch needs is absent, or changes its length
    let mut absent = region(4, [1.0, -1.0], &[[0.5, 0.5], [2.0, 0.0]]);
    absent.leaves[1].mixing[0] = value(MixingKind::Input, None);
    let mut ragged = region(4, [1.0, -1.0], &[[0.5, 0.5], [2.0, 0.0]]);
    ragged.leaves[1].mixing[0].values = Some(encode_f32(&[1.0]));
    let mut short = region(4, [1.0, -1.0], &[[0.5, 0.5], [2.0, 0.0]]);
    short.leaves.truncate(1);
    for opening in [absent, ragged, short] {
        let error = check_region(&ToyRegion, &opening, 8).unwrap_err();
        assert!(NodeFault::find(&error).is_some(), "{error:#}");
    }
}

#[test]
fn the_most_recent_of_the_snapshots_is_compared() {
    assert_eq!(most_recent(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2), &[1.0, 2.0]);
    assert_eq!(most_recent(&[1.0, 2.0, 3.0], 2), &[1.0, 2.0, 3.0]);
}
