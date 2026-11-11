use super::*;

fn verdict_of(weight: &str, tokens: usize, failing: usize) -> MatrixVerdict {
    MatrixVerdict {
        weight: weight.into(),
        weight_type: "q8_0".into(),
        router: weight.contains("gate_inp"),
        tokens,
        failing_tokens: failing,
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
fn each_backend_names_its_own_profiles() {
    use crate::ggml_types::{GGML_F16, GGML_F32, GGML_Q8_0};
    assert_eq!(BackendFamily::of("CPU"), Some(BackendFamily::Cpu));
    assert_eq!(BackendFamily::of("CUDA0"), Some(BackendFamily::Cuda));
    assert_eq!(BackendFamily::of("MTL0"), Some(BackendFamily::Metal));
    // a backend without measured profiles has no family: Vulkan's kernels
    // round in ways no profile models
    for unprofiled in ["Vulkan0", "Vulkan1", "Metal", "ROCm0", ""] {
        assert_eq!(BackendFamily::of(unprofiled), None, "{unprofiled}");
        assert!(BackendFamily::checked(unprofiled).is_err());
    }
    assert_eq!(
        profiles_for(BackendFamily::Cpu, GGML_Q8_0),
        Some(&[Profile::CpuQ8][..])
    );
    // CUDA's q8 kernels, never the CPU one
    let cuda = profiles_for(BackendFamily::Cuda, GGML_Q8_0).unwrap();
    assert!(!cuda.contains(&Profile::CpuQ8) && !cuda.contains(&Profile::Raw));
    assert_eq!(
        profiles_for(BackendFamily::Cpu, GGML_F32),
        Some(&[Profile::Raw][..])
    );
    assert_eq!(
        profiles_for(BackendFamily::Cpu, GGML_F16),
        Some(&[Profile::F16][..])
    );
    // the CPU quantizes K-quant activations per 256, CUDA per 32 or 64
    assert_eq!(
        profiles_for(BackendFamily::Cpu, crate::ggml_types::GGML_Q4_K),
        Some(&[Profile::CpuQ8K][..])
    );
    assert!(
        profiles_for(BackendFamily::Cuda, crate::ggml_types::GGML_Q2_K)
            .unwrap()
            .contains(&Profile::Q8Mul64F32)
    );
    // Metal: f32 activations in decode, f16 ones rounded either way in
    // prefill; no bf16 profile
    assert_eq!(
        profiles_for(BackendFamily::Metal, GGML_Q8_0),
        Some(&[Profile::Raw, Profile::F16Weights, Profile::F16WeightsRtz][..])
    );
    assert_eq!(
        profiles_for(BackendFamily::Metal, crate::ggml_types::GGML_Q4_K),
        None
    );
    assert_eq!(
        profiles_for(BackendFamily::Metal, crate::ggml_types::GGML_BF16),
        None
    );
    // types without a reference dequantization or a known profile have none
    assert_eq!(profiles_for(BackendFamily::Cpu, 19), None);
    assert_eq!(
        profiles_for(BackendFamily::Cuda, crate::ggml_types::GGML_Q4_1),
        None
    );
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
