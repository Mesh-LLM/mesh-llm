use anyhow::bail;

use super::*;
use crate::layer_tensors::{FLAG_ROUTER, GGML_F32};
use crate::wire::{AnchorOpening, MatrixOpening, SelectionOpening, TensorOpening, encode_f32};

/// A toy layer the evaluator computes on opened values:
///
/// ```text
/// matmul 0 (blk.0.a.weight):      input = 2 · residual
/// matmul 1 (blk.0.router.weight): input = residual, a router
/// selection 0:                    scores = matmul 1 output
/// mixing input 0, stored row 0:   residual
/// residual out = residual + matmul 0 output + mixing output 0
/// entry of layer 0 = [token, position]
/// ```
struct ToyLayer;

impl ToyLayer {
    /// An opened column; like the native evaluator, an unopened value is zeros.
    fn column(
        request: &GlueRequest,
        kind: TensorKind,
        index: u32,
        col: usize,
        len: usize,
    ) -> Vec<f32> {
        let Some(input) = request
            .inputs
            .iter()
            .find(|i| i.kind == kind && i.index == index)
        else {
            return vec![0.0; len];
        };
        match &input.values {
            GlueValues::F32(v) => v[col * input.len..(col + 1) * input.len].to_vec(),
            GlueValues::I32(_) => panic!("f32 expected"),
        }
    }
}

fn tensor(
    kind: TensorKind,
    index: u32,
    flags: i32,
    name: &str,
    len: usize,
    values: Vec<f32>,
) -> GlueTensor {
    GlueTensor {
        layer: if kind == TensorKind::Residual { 1 } else { 0 },
        kind,
        index,
        flags,
        name: name.into(),
        len,
        values: GlueValues::F32(values),
    }
}

impl GlueEvaluator for ToyLayer {
    fn evaluate(&self, request: &GlueRequest) -> Result<GlueOutput> {
        let n = request.positions.len();
        let len = request.residual.len() / n;
        let mut doubled = Vec::new();
        let mut scores = Vec::new();
        let mut out = Vec::new();
        for col in 0..n {
            let r = &request.residual[col * len..(col + 1) * len];
            doubled.extend(r.iter().map(|v| v * 2.0));
            scores.extend(Self::column(request, TensorKind::MatmulOutput, 1, col, 3));
            let m = Self::column(request, TensorKind::MatmulOutput, 0, col, len);
            let a = Self::column(request, TensorKind::MixingOutput, 0, col, len);
            out.extend((0..len).map(|i| r[i] + m[i] + a[i]));
        }
        let experts = scores.len() / n;
        Ok(GlueOutput {
            tensors: vec![
                tensor(
                    TensorKind::MatmulInput,
                    0,
                    0,
                    "blk.0.a.weight",
                    len,
                    doubled,
                ),
                tensor(
                    TensorKind::MatmulInput,
                    1,
                    FLAG_ROUTER,
                    "blk.0.router.weight",
                    len,
                    request.residual.clone(),
                ),
                tensor(
                    TensorKind::MixingInput,
                    0,
                    0,
                    "q",
                    len,
                    request.residual.clone(),
                ),
                tensor(
                    TensorKind::MixingStored,
                    0,
                    0,
                    "k",
                    len,
                    request.residual.clone(),
                ),
                tensor(TensorKind::SelectionScores, 0, 0, "probs", experts, scores),
                tensor(TensorKind::Residual, 0, 0, "l_out", len, out),
            ],
            unchecked_weights: vec!["blk.0.sinks".into()],
            attention: Vec::new(),
        })
    }

    fn entry(&self, tokens: &[i32], positions: &[i32]) -> Result<Vec<f32>> {
        Ok(tokens
            .iter()
            .zip(positions)
            .flat_map(|(&t, &p)| [t as f32, p as f32])
            .collect())
    }

    fn evaluate_region(&self, _: &GlueRequest) -> Result<GlueOutput> {
        bail!("the toy layer keeps no recurrent state")
    }

    fn stored_row_error(
        &self,
        ggml_type: i32,
        stored: &[u8],
        computed: &[f32],
    ) -> Result<StoredRowError> {
        // the check never hands over a row of a type it does not know, or
        // of fewer bytes than the type packs the computed row into
        assert_eq!(ggml_type, GGML_F32);
        assert_eq!(stored.len(), computed.len() * 4);
        let stored = crate::layer_tensors::decode_f32_bytes(stored);
        let l2 = |v: &mut dyn Iterator<Item = f64>| v.map(|x| x * x).sum::<f64>().sqrt();
        Ok(StoredRowError {
            stored: l2(&mut stored.iter().zip(computed).map(|(a, b)| f64::from(a - b))),
            rounding: 0.0,
            norm: l2(&mut computed.iter().map(|&v| f64::from(v))),
        })
    }
}

fn honest_leaf(position: u32, r: [f32; 2]) -> LeafOpening {
    let m = [0.5, -0.25];
    let a = [0.125, 1.0];
    let scores = [0.1f32, 0.9, 0.4];
    let mixing = |kind, values: &[f32]| {
        TensorOpening::new(
            kind,
            0,
            Some((GGML_F32, &crate::layer_tensors::f32_bytes(values))),
        )
    };
    LeafOpening {
        position,
        residual_in: encode_f32(&r),
        residual_out: Some(encode_f32(&[r[0] + m[0] + a[0], r[1] + m[1] + a[1]])),
        residual_in_path: Vec::new(),
        mixing_path: Vec::new(),
        residual_out_path: Vec::new(),
        mixing: vec![
            mixing(MixingKind::Input, &r),
            mixing(MixingKind::Stored, &r),
            mixing(MixingKind::Output, &a),
        ],
        matrices: vec![
            MatrixOpening {
                index: 0,
                weight: "blk.0.a.weight".into(),
                router: false,
                experts: None,
                input: encode_f32(&[r[0] * 2.0, r[1] * 2.0]),
                output: encode_f32(&m),
            },
            MatrixOpening {
                index: 1,
                weight: "blk.0.router.weight".into(),
                router: true,
                experts: None,
                input: encode_f32(&r),
                output: encode_f32(&scores),
            },
        ],
        selections: vec![SelectionOpening {
            index: 0,
            ids: vec![1, 2],
        }],
    }
}

fn opening(leaves: Vec<LeafOpening>) -> LayerOpening {
    LayerOpening { layer: 0, leaves }
}

fn honest() -> LayerOpening {
    opening(vec![
        honest_leaf(3, [1.0, 2.0]),
        honest_leaf(4, [-1.0, 0.5]),
    ])
}

fn failing(report: &GlueReport) -> Vec<&str> {
    report
        .verdicts
        .iter()
        .filter(|v| !v.pass)
        .map(|v| v.check.as_str())
        .collect()
}

#[test]
fn an_honest_opening_passes() {
    let report = check_glue(&ToyLayer, &honest()).unwrap();
    assert!(report.pass, "{:?}", report.verdicts);
    assert_eq!(report.unchecked_weights, vec!["blk.0.sinks".to_string()]);
    let checks: Vec<&str> = report.verdicts.iter().map(|v| v.check.as_str()).collect();
    for check in [
        "matmul inputs",
        "mixing inputs",
        "stored rows",
        "selections",
        "residual out",
    ] {
        assert!(checks.contains(&check), "{check} missing from {checks:?}");
    }
}

#[test]
fn a_changed_matmul_input_fails() {
    let mut opening = honest();
    opening.leaves[1].matrices[0].input = encode_f32(&[-2.0, 1.01]);
    assert_eq!(
        failing(&check_glue(&ToyLayer, &opening).unwrap()),
        ["matmul inputs"]
    );
}

#[test]
fn a_changed_stored_row_fails() {
    let mut opening = honest();
    let row = crate::layer_tensors::f32_bytes(&[1.0, 2.001]);
    opening.leaves[0].mixing[1] = TensorOpening::new(MixingKind::Stored, 0, Some((GGML_F32, &row)));
    assert_eq!(
        failing(&check_glue(&ToyLayer, &opening).unwrap()),
        ["stored rows"]
    );
}

#[test]
fn a_steered_selection_fails() {
    let mut opening = honest();
    opening.leaves[0].selections[0].ids = vec![1, 0];
    assert_eq!(
        failing(&check_glue(&ToyLayer, &opening).unwrap()),
        ["selections"]
    );
}

#[test]
fn a_mislabeled_matmul_fails_the_plan() {
    let mut opening = honest();
    opening.leaves[0].matrices[0].weight = "blk.0.b.weight".into();
    assert!(failing(&check_glue(&ToyLayer, &opening).unwrap()).contains(&"layer plan"));
    let mut opening = honest();
    opening.leaves[0].matrices[1].router = false;
    assert!(failing(&check_glue(&ToyLayer, &opening).unwrap()).contains(&"layer plan"));
}

#[test]
fn a_missing_mixing_value_fails_the_plan() {
    let mut opening = honest();
    opening.leaves[1].mixing.remove(1);
    assert!(failing(&check_glue(&ToyLayer, &opening).unwrap()).contains(&"layer plan"));
}

#[test]
fn a_matmul_opened_twice_fails_the_plan() {
    // copies of an honest column would pad the row check's allowance
    let mut opening = honest();
    let copy = opening.leaves[0].matrices[0].clone();
    opening.leaves[0].matrices.push(copy);
    let report = check_glue(&ToyLayer, &opening).unwrap();
    assert_eq!(failing(&report), ["layer plan"], "{:?}", report.verdicts);
}

#[test]
fn a_selection_the_layer_does_not_have_fails_the_plan() {
    let mut opening = honest();
    opening.leaves[0].selections.push(SelectionOpening {
        index: 1,
        ids: vec![0],
    });
    assert!(failing(&check_glue(&ToyLayer, &opening).unwrap()).contains(&"layer plan"));
    // nor may the one it has go by another index
    let mut opening = honest();
    opening.leaves[0].selections[0].index = 1;
    assert!(failing(&check_glue(&ToyLayer, &opening).unwrap()).contains(&"layer plan"));
}

#[test]
fn a_value_absent_on_a_kept_row_fails_the_plan() {
    // an absent value is not compared, so a kept row may not leave one out
    for index in 0..3 {
        let mut opening = honest();
        let value = &mut opening.leaves[1].mixing[index];
        *value = TensorOpening::new(value.kind, 0, None);
        let report = check_glue(&ToyLayer, &opening).unwrap();
        assert!(
            failing(&report).contains(&"layer plan"),
            "mixing value {index}: {:?}",
            report.verdicts
        );
    }
}

#[test]
fn the_plan_is_the_auditors_and_names_the_routers() {
    // the node clears every router flag; the report still says which are
    let mut opening = honest();
    for leaf in &mut opening.leaves {
        leaf.matrices[1].router = false;
    }
    let report = check_glue(&ToyLayer, &opening).unwrap();
    assert_eq!(failing(&report), ["layer plan"]);
    let plan: Vec<(u32, &str, bool)> = report
        .matmuls
        .iter()
        .map(|(index, m)| (*index, m.weight.as_str(), m.router))
        .collect();
    assert_eq!(
        plan,
        [
            (0, "blk.0.a.weight", false),
            (1, "blk.0.router.weight", true)
        ]
    );
}

#[test]
fn a_value_that_is_not_a_number_fails_its_comparison() {
    type Change = fn(&mut LeafOpening);
    let nan = f32::NAN;
    let changes: [(&str, Change); 5] = [
        ("matmul inputs", |leaf| {
            leaf.matrices[0].input = encode_f32(&[f32::NAN, 4.0]);
        }),
        ("matmul inputs", |leaf| {
            leaf.matrices[0].input = encode_f32(&[f32::INFINITY, 4.0]);
        }),
        ("mixing inputs", |leaf| {
            let bytes = crate::layer_tensors::f32_bytes(&[1.0, f32::NAN]);
            leaf.mixing[0] = TensorOpening::new(MixingKind::Input, 0, Some((GGML_F32, &bytes)));
        }),
        ("stored rows", |leaf| {
            let bytes = crate::layer_tensors::f32_bytes(&[f32::NAN, 2.0]);
            leaf.mixing[1] = TensorOpening::new(MixingKind::Stored, 0, Some((GGML_F32, &bytes)));
        }),
        ("residual out", |leaf| {
            leaf.residual_out = Some(encode_f32(&[f32::NAN, f32::NAN]));
        }),
    ];
    for (check, change) in changes {
        let mut opening = honest();
        change(&mut opening.leaves[0]);
        let report = check_glue(&ToyLayer, &opening).unwrap();
        assert!(!report.pass, "{check}");
        assert!(failing(&report).contains(&check), "{check}: {report:?}");
    }
    // the comparisons themselves, which `f64::max` used to let through
    assert_eq!(relative_error(&[nan, 1.0], &[1.0, 1.0]), f64::INFINITY);
    assert_eq!(relative_error(&[1.0, 1.0], &[nan, 1.0]), f64::INFINITY);
    assert_eq!(selection_error(&[0], &[nan, 1.0]), f64::INFINITY);
    let distance = |stored, rounding, norm| StoredRowError {
        stored,
        rounding,
        norm,
    };
    for error in [
        distance(f64::NAN, 0.01, 10.0),
        distance(0.01, f64::NAN, 10.0),
        distance(0.01, 0.01, f64::NAN),
    ] {
        assert_eq!(stored_excess(&error), f64::INFINITY, "{error:?}");
    }
    let mut verdicts = Verdicts::default();
    verdicts.add("check", "a value", f64::NAN);
    assert!(!verdicts.0[0].pass);
    assert_eq!(verdicts.0[0].max_error, f64::INFINITY);
}

#[test]
fn a_stored_row_of_another_type_or_length_never_reaches_the_evaluator() {
    // the toy evaluator panics on either, as the native one may read past
    // the row
    let short = crate::layer_tensors::f32_bytes(&[1.0]);
    let whole = crate::layer_tensors::f32_bytes(&[1.0, 2.0]);
    for (ggml_type, bytes) in [(GGML_F32, &short), (39, &whole), (-1, &whole), (3, &whole)] {
        let mut opening = honest();
        opening.leaves[0].mixing[1] =
            TensorOpening::new(MixingKind::Stored, 0, Some((ggml_type, bytes)));
        let report = check_glue(&ToyLayer, &opening).unwrap();
        assert_eq!(failing(&report), ["stored rows"], "type {ggml_type}");
    }
}

#[test]
fn an_opening_that_makes_no_request_is_the_nodes_fault() {
    use crate::node_fault::NodeFault;
    let empty = check_glue(&ToyLayer, &opening(Vec::new())).unwrap_err();
    assert!(NodeFault::find(&empty).is_some(), "{empty:#}");
    // a matmul output that changes length between positions
    let mut ragged = honest();
    ragged.leaves[1].matrices[0].output = encode_f32(&[0.5]);
    let error = check_glue(&ToyLayer, &ragged).unwrap_err();
    assert!(NodeFault::find(&error).is_some(), "{error:#}");
    let mut undecodable = honest();
    undecodable.leaves[0].residual_in = "not base64".into();
    let error = check_glue(&ToyLayer, &undecodable).unwrap_err();
    assert!(NodeFault::find(&error).is_some(), "{error:#}");
}

/// An evaluator that cannot compute the layer, as one whose claimed GGUF
/// is out of reach.
struct Unavailable;

impl GlueEvaluator for Unavailable {
    fn evaluate(&self, _: &GlueRequest) -> Result<GlueOutput> {
        bail!("the layer did not load")
    }

    fn evaluate_region(&self, _: &GlueRequest) -> Result<GlueOutput> {
        bail!("the layer did not load")
    }

    fn entry(&self, _: &[i32], _: &[i32]) -> Result<Vec<f32>> {
        bail!("the layer did not load")
    }

    fn stored_row_error(&self, _: i32, _: &[u8], _: &[f32]) -> Result<StoredRowError> {
        bail!("the layer did not load")
    }
}

#[test]
fn an_evaluator_that_cannot_run_blames_nobody() {
    use crate::node_fault::NodeFault;
    let error = check_glue(&Unavailable, &honest()).unwrap_err();
    assert!(NodeFault::find(&error).is_none(), "{error:#}");
    let anchors = [anchor(3, 7, [7.0, 3.0])];
    let error = check_anchors(&Unavailable, &anchors).unwrap_err();
    assert!(NodeFault::find(&error).is_none(), "{error:#}");
}

#[test]
fn a_changed_residual_out_fails() {
    let mut opening = honest();
    opening.leaves[1].residual_out = Some(encode_f32(&[-0.375, 1.3]));
    assert_eq!(
        failing(&check_glue(&ToyLayer, &opening).unwrap()),
        ["residual out"]
    );
}

#[test]
fn absent_rows_are_zero_and_not_compared() {
    let mut opening = honest();
    opening.leaves[1].residual_out = None;
    let request = glue_request(&opening).unwrap();
    assert_eq!(request.positions, vec![3, 4]);
    assert_eq!(request.residual, vec![1.0, 2.0, -1.0, 0.5]);
    let selection = request
        .inputs
        .iter()
        .find(|i| i.kind == TensorKind::Selection)
        .unwrap();
    assert_eq!(selection.values, GlueValues::I32(vec![1, 2, 1, 2]));
    assert!(check_glue(&ToyLayer, &opening).unwrap().pass);
}

#[test]
fn a_dropped_row_opens_no_selection() {
    let mut opening = honest();
    let leaf = &mut opening.leaves[1];
    leaf.residual_out = None;
    leaf.matrices.retain(|m| !m.router);
    leaf.selections.clear();
    let report = check_glue(&ToyLayer, &opening).unwrap();
    assert!(report.pass, "{:?}", report.verdicts);
    // a row that is kept must open its selections
    let mut opening = honest();
    opening.leaves[1].selections.clear();
    assert!(failing(&check_glue(&ToyLayer, &opening).unwrap()).contains(&"layer plan"));
}

#[test]
fn selections_accept_real_ties_only() {
    let scores = [0.5, 0.2, 0.5 - 1e-6, 0.1];
    assert_eq!(selection_error(&[0, 2], &scores), 0.0);
    assert!(selection_error(&[2, 0], &scores) <= GLUE_TOLERANCE);
    assert!(selection_error(&[0, 1], &scores) > GLUE_TOLERANCE);
    // a full argsort, in either direction
    assert_eq!(selection_error(&[0, 2, 1, 3], &scores), 0.0);
    assert_eq!(selection_error(&[3, 1, 2, 0], &scores), 0.0);
    assert!(selection_error(&[0, 1, 2, 3], &scores) > GLUE_TOLERANCE);
    // repeated or out-of-range ids
    assert!(selection_error(&[0, 0], &scores).is_infinite());
    assert!(selection_error(&[4], &scores).is_infinite());
}

#[test]
fn stored_rows_may_round_like_the_auditor() {
    let error = |stored, rounding| StoredRowError {
        stored,
        rounding,
        norm: 10.0,
    };
    assert_eq!(stored_excess(&error(0.015, 0.01)), 0.0);
    assert!(stored_excess(&error(0.0155, 0.01)) > 0.0);
    assert!(stored_excess(&error(0.0155, 0.01)) <= GLUE_TOLERANCE);
    assert!(stored_excess(&error(0.05, 0.01)) > GLUE_TOLERANCE);
}

#[test]
fn rotated_values_get_more_tolerance_further_out() {
    // Q at position 2319 off by its f32 angle rounding, as measured on CUDA
    let rotated =
        |position: u32| GLUE_TOLERANCE + ROTATION_TOLERANCE_PER_POSITION * f64::from(position);
    let mut verdicts = Verdicts::default();
    verdicts.add_within("mixing inputs", "position 2319", 1.8e-4, rotated(2319));
    assert!(verdicts.0[0].pass);
    verdicts.add_within("mixing inputs", "position 10", 1.8e-4, rotated(10));
    assert!(!verdicts.0[0].pass);
    assert_eq!(verdicts.0[0].failing_tokens, 1);
    // still tight next to the values: 0.8% at 32k
    assert!(rotated(32_768) < 0.01);
}

#[test]
fn an_unopened_matmul_fails_the_plan() {
    // The node skips matmul 0 and commits the residual out it gives as
    // zeros; the auditor's own computation agrees, so only the plan catches it.
    let mut opening = honest();
    for leaf in &mut opening.leaves {
        let r = leaf.residual_in_values().unwrap();
        leaf.matrices.retain(|m| m.index != 0);
        leaf.residual_out = Some(encode_f32(&[r[0] + 0.125, r[1] + 1.0]));
    }
    let report = check_glue(&ToyLayer, &opening).unwrap();
    assert_eq!(failing(&report), ["layer plan"], "{:?}", report.verdicts);
}

fn anchor(position: u32, token: i32, residual: [f32; 2]) -> AnchorOpening {
    AnchorOpening {
        position,
        token,
        residual: encode_f32(&residual),
        path: Vec::new(),
    }
}

#[test]
fn anchors_must_be_the_input_their_tokens_make() {
    let honest = [anchor(3, 7, [7.0, 3.0]), anchor(5, 9, [9.0, 5.0])];
    assert!(check_anchors(&ToyLayer, &honest).unwrap().pass);
    // the input of another token under this token's name
    let swapped = [anchor(3, 7, [7.0, 3.0]), anchor(5, 9, [8.0, 5.0])];
    let verdict = check_anchors(&ToyLayer, &swapped).unwrap();
    assert!(!verdict.pass);
    assert_eq!(verdict.failing_tokens, 1);
    assert!(!check_anchors(&ToyLayer, &[]).unwrap().pass);
}
