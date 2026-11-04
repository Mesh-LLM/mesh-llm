use super::*;

fn verdict_of(weight: &str, tokens: usize, failing: usize) -> MatrixVerdict {
    MatrixVerdict {
        weight: weight.into(),
        weight_type: "q8_0".into(),
        router: weight.contains("gate_inp"),
        exact: weight.contains("gate_inp"),
        tokens,
        failing_tokens: failing,
        past_cap: 0,
        repaired_tokens: 0,
        max_error: 0.0,
        best_profile: "raw",
        pass: failing <= allowed_failing_columns(tokens),
    }
}

fn report(layer: u32, matrices: Vec<MatrixVerdict>) -> AuditReport {
    AuditReport {
        layer,
        tokens: 4,
        pass: matrices.iter().all(|m| m.pass),
        matrices,
    }
}

#[test]
fn a_role_is_judged_on_every_opened_layer() {
    // cheating on half the positions: two of four columns fail in each
    // layer, which each layer alone would let pass
    let reports: Vec<AuditReport> = (0..4)
        .map(|l| {
            report(
                l,
                vec![
                    verdict_of(&format!("blk.{l}.attn_q.weight"), 4, 2),
                    verdict_of(&format!("blk.{l}.ffn_up.weight"), 4, 0),
                ],
            )
        })
        .collect();
    assert!(reports.iter().all(|r| r.pass));
    let roles = pool_roles(&reports);
    assert_eq!(roles.len(), 2);
    assert_eq!(roles[0].role, "attn_q.weight");
    assert_eq!((roles[0].tokens, roles[0].failing_tokens), (16, 8));
    assert!(!roles[0].pass);
    assert_eq!(roles[0].layers, vec![0, 1, 2, 3]);
    assert!(roles[1].pass);
}

#[test]
fn honest_outliers_spread_over_layers_still_pass() {
    let reports = [
        report(3, vec![verdict_of("blk.3.attn_k.weight", 4, 1)]),
        report(9, vec![verdict_of("blk.9.attn_k.weight", 4, 1)]),
        report(20, vec![verdict_of("blk.20.ffn_gate_inp.weight", 4, 1)]),
    ];
    let roles = pool_roles(&reports);
    assert!(roles[0].pass, "{roles:?}");
    // a router may not fail a single column
    assert!(roles[1].router && !roles[1].pass);
}

#[test]
fn roles_drop_only_a_layer_prefix() {
    assert_eq!(
        weight_role("blk.12.ffn_down_exps.weight"),
        "ffn_down_exps.weight"
    );
    assert_eq!(weight_role("output.weight"), "output.weight");
    assert_eq!(weight_role("blk.x.attn_q.weight"), "blk.x.attn_q.weight");
}

#[test]
fn a_matrix_fails_on_a_few_bad_columns() {
    // decode tokens are often a small share of a turn's positions; a
    // node that cheats on four of 16 must fail
    assert!(4 > allowed_failing_columns(16));
    assert_eq!(allowed_failing_columns(16), 2);
    assert_eq!(allowed_failing_columns(128), 6);
}

#[test]
fn exact_columns_have_zero_error() {
    let (a, b) = ([1.0, 2.0], [-1.0, 0.5]);
    let rows = vec![(0, &a[..]), (1, &b[..])];
    let x = [0.5f32, 0.25];
    let y = [1.0f32, -0.375];
    assert_eq!(column_error(&rows, &Profile::Raw.apply(&x), &y), 0.0);
}

#[test]
fn a_wrong_output_shows_up() {
    let a = [1.0, 2.0];
    let rows = vec![(0, &a[..])];
    let y = [1.01f32, 5.0];
    assert!(column_error(&rows, &Profile::Raw.apply(&[0.5, 0.25]), &y) > TOLERANCE);
}

#[test]
fn a_huge_value_in_an_undrawn_row_hides_no_other_weights() {
    // The dead-neuron cheat: a node serves gate and up from other weights
    // (1.2% off, as Q6_K is from Q8_0) and opens gate = -1e6 and up = 1e6
    // at one row r. silu(-1e6) is 0, so the input of ffn_down is still
    // what its glue check computes, and r is drawn only 16 times in 256.
    // The error used to be relative to every opened output, r's among
    // them, so 1e6 hid the 1.2% from every drawn row.
    let (k, m) = (32, 256);
    let w: Vec<Vec<f64>> = (0..m)
        .map(|r| {
            (0..k)
                .map(|i| ((r * 31 + i * 17) % 23) as f64 / 50.0 - 0.22)
                .collect()
        })
        .collect();
    let x: Vec<f32> = (0..k).map(|i| (i * 7 % 13) as f32 / 10.0 - 0.6).collect();
    let xs = Profile::Raw.apply(&x);
    for (weight, huge) in [
        ("blk.3.ffn_gate.weight", -1e6f32),
        ("blk.3.ffn_up.weight", 1e6),
    ] {
        let picked = RowCache::default().picked(weight, m, false, b"rows seed");
        let rows: Vec<(usize, &[f64])> = picked.iter().map(|&r| (r, w[r].as_slice())).collect();
        let mut y: Vec<f32> = w
            .iter()
            .enumerate()
            .map(|(r, row)| {
                let other = if r % 2 == 0 { 1.012 } else { 0.988 };
                (row.iter().zip(&xs).map(|(a, b)| a * b).sum::<f64>() * other) as f32
            })
            .collect();
        assert!(column_error(&rows, &xs, &y) > TOLERANCE, "{weight}");
        let r = (0..m)
            .find(|r| !picked.contains(r))
            .expect("an undrawn row");
        y[r] = huge;
        let error = column_error(&rows, &xs, &y);
        assert!(error > TOLERANCE, "{weight}: hidden at {error:e}");
        // the honest column at that row still passes
        let honest: Vec<f32> = w
            .iter()
            .map(|row| row.iter().zip(&xs).map(|(a, b)| a * b).sum::<f64>() as f32)
            .collect();
        assert!(column_error(&rows, &xs, &honest) < 1e-6);
    }
}

fn tensor(dims: Vec<u64>) -> TensorInfo {
    TensorInfo {
        name: "w".into(),
        dims,
        ggml_type: 0,
        offset: 0,
    }
}

fn matrix(experts: Option<Vec<i32>>) -> MatrixOpening {
    MatrixOpening {
        index: 0,
        weight: "w".into(),
        router: false,
        experts,
        input: String::new(),
        output: String::new(),
    }
}

#[test]
fn expert_slices_start_at_their_expert() {
    let w = tensor(vec![4, 3, 8]);
    let s = matmul_slices(&w, &matrix(Some(vec![5, 2])), 4, 6).unwrap();
    assert_eq!(s.slices, vec![(15, 0), (6, 0)]);
    let s = matmul_slices(&w, &matrix(Some(vec![5, 2])), 8, 6).unwrap();
    assert_eq!(s.slices, vec![(15, 0), (6, 1)]);
    assert!(matmul_slices(&w, &matrix(Some(vec![9, 2])), 4, 6).is_err());
    assert!(matmul_slices(&w, &matrix(Some(vec![1])), 4, 6).is_err());
}

#[test]
fn a_shared_weight_meets_every_batch() {
    let s = matmul_slices(&tensor(vec![4, 3]), &matrix(None), 8, 6).unwrap();
    assert_eq!(s.slices, vec![(0, 0), (0, 1)]);
    let s = matmul_slices(&tensor(vec![4, 3, 2]), &matrix(None), 8, 6).unwrap();
    assert_eq!(s.slices, vec![(0, 0), (3, 1)]);
    assert!(matmul_slices(&tensor(vec![4, 3]), &matrix(None), 5, 3).is_err());
    // an empty column would be compared with nothing, and pass
    assert!(matmul_slices(&tensor(vec![4, 3]), &matrix(None), 0, 0).is_err());
}

#[test]
fn an_error_that_is_not_a_number_fails_its_column() {
    let weights = crate::toy_turn::weights();
    let check = |error| ColumnCheck {
        error,
        profile: "raw",
        flips: 0,
    };
    // `error > TOLERANCE` is false for NaN, so these columns used to pass
    let checks = [check(0.0), check(f64::NAN), check(f64::INFINITY)];
    let router = verdict(&weights, "blk.0.router.weight".into(), true, &checks).unwrap();
    assert_eq!(router.failing_tokens, 2);
    assert_eq!(router.max_error, f64::INFINITY);
    assert!(!router.pass);
    // three bad columns of three is past what a matrix may fail
    let bad = [check(f64::NAN); 3];
    let dense = verdict(&weights, "blk.0.a.weight".into(), false, &bad).unwrap();
    assert_eq!((dense.failing_tokens, dense.pass), (3, false));
    let fine = verdict(&weights, "blk.0.a.weight".into(), false, &[check(1e-9)]).unwrap();
    assert!(fine.pass);
}

#[path = "row_check_tests.rs"]
mod row_check_tests;

#[test]
fn a_column_far_past_the_tolerance_fails_its_role_whatever_the_allowance() {
    // two columns of 16 may fail, but not by any amount: other weights on
    // two positions passed every audit
    let weights = crate::toy_turn::weights();
    let check = |error| ColumnCheck {
        error,
        profile: "raw",
        flips: 0,
    };
    let mut checks = vec![check(1e-7); 16];
    checks[3] = check(2.5e-3);
    checks[9] = check(3e-3);
    let honest = verdict(&weights, "blk.0.a.weight".into(), false, &checks).unwrap();
    assert!(honest.pass && honest.past_cap == 0, "{honest:?}");
    checks[9] = check(0.5);
    let cheat = verdict(&weights, "blk.0.a.weight".into(), false, &checks).unwrap();
    assert_eq!(
        (cheat.failing_tokens, cheat.past_cap, cheat.pass),
        (2, 1, false)
    );
    // the role pools it with the other layers' columns and still fails
    let roles = pool_roles(&[report(0, vec![cheat]), report(5, vec![honest])]);
    assert_eq!((roles[0].tokens, roles[0].past_cap), (32, 1));
    assert!(!roles[0].pass);
}

#[test]
fn only_a_router_of_float_weights_may_fail_no_column() {
    use crate::ggml_types::{GGML_BF16, GGML_F16, GGML_F32, GGML_Q4_K, GGML_Q8_0};
    for float in [GGML_F32, GGML_F16, GGML_BF16] {
        assert!(exact_router(true, float));
        assert!(!exact_router(false, float));
    }
    // a quantized router rounds as other quantized weights do: at the
    // measured 1 in 8,350 honest column failures, no allowance failed an
    // honest node about 1 in 400 audits
    for quantized in [GGML_Q8_0, GGML_Q4_K] {
        assert!(!exact_router(true, quantized));
    }
    let mut quantized = verdict_of("blk.4.ffn_gate_inp.weight", 16, 1);
    quantized.exact = false;
    let roles = pool_roles(&[report(4, vec![quantized])]);
    assert!(roles[0].router && roles[0].pass, "{roles:?}");
}
