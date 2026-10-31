use super::*;
use crate::layer_tensors::GGML_F32;
use crate::wire::{MatrixOpening, SelectionOpening, TensorOpening, encode_f32};

/// A toy layer the evaluator computes on opened values:
///
/// ```text
/// matmul 0 (blk.0.a.weight):      input = 2 · residual
/// matmul 1 (blk.0.router.weight): input = residual, a router
/// selection 0:                    scores = matmul 1 output
/// mixing input 0, stored row 0:   residual
/// residual out = residual + matmul 0 output + mixing output 0
/// ```
struct ToyLayer;

impl ToyLayer {
    fn column(request: &GlueRequest, kind: TensorKind, index: u32, col: usize) -> &[f32] {
        let input = request
            .inputs
            .iter()
            .find(|i| i.kind == kind && i.index == index)
            .expect("an opened value");
        match &input.values {
            GlueValues::F32(v) => &v[col * input.len..(col + 1) * input.len],
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
            scores.extend_from_slice(Self::column(request, TensorKind::MatmulOutput, 1, col));
            let m = Self::column(request, TensorKind::MatmulOutput, 0, col);
            let a = Self::column(request, TensorKind::MixingOutput, 0, col);
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
        })
    }

    fn stored_row_error(
        &self,
        ggml_type: i32,
        stored: &[u8],
        computed: &[f32],
    ) -> Result<StoredRowError> {
        assert_eq!(ggml_type, GGML_F32);
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

fn opening(leaves: Vec<LeafOpening>) -> Opening {
    Opening {
        receipt_id: "r".into(),
        stage_index: 0,
        layer: 0,
        leaves,
    }
}

fn honest() -> Opening {
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
