use super::*;

fn op() -> AttentionOp {
    AttentionOp {
        n_head: 4,
        n_head_kv: 2,
        head_dim_k: 3,
        head_dim_v: 2,
        scale: 0.5,
        softcap: 0.0,
        sinks: None,
        window: Window::Causal,
        output_rotation: None,
        q_rounding: None,
    }
}

/// A deterministic value in [-1, 1).
fn value(seed: u64) -> f32 {
    let x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(17);
    (x % 2000) as f32 / 1000.0 - 1.0
}

fn context(op: &AttentionOp, n: u32) -> Vec<ContextRow> {
    (0..n)
        .map(|position| ContextRow {
            position,
            key: (0..op.n_head_kv * op.head_dim_k)
                .map(|i| f64::from(value(u64::from(position) * 1000 + i as u64)))
                .collect(),
            value: (0..op.n_head_kv * op.head_dim_v)
                .map(|i| f64::from(value(u64::from(position) * 1000 + 500 + i as u64)))
                .collect(),
        })
        .collect()
}

fn query(op: &AttentionOp) -> Vec<f32> {
    (0..op.n_head * op.head_dim_k)
        .map(|i| value(77 + i as u64) * 3.0)
        .collect()
}

/// Plain softmax attention for one head over every row.
fn exact(op: &AttentionOp, head: usize, q: &[f32], rows: &[ContextRow]) -> Vec<f64> {
    let g = op.kv_head(head);
    let q = op.rounded_q(q, head).unwrap();
    let logits: Vec<f64> = rows
        .iter()
        .map(|r| op.logit(&q, head_slice(&r.key, g, op.head_dim_k).unwrap()))
        .collect();
    let sink = op
        .sinks
        .as_ref()
        .map_or(f64::NEG_INFINITY, |s| f64::from(s[head]));
    let max = logits.iter().copied().fold(sink, f64::max);
    let weights: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f64 = weights.iter().sum::<f64>() + (sink - max).exp();
    (0..op.head_dim_v)
        .map(|d| {
            rows.iter()
                .zip(&weights)
                .map(|(r, w)| w * head_slice(&r.value, g, op.head_dim_v).unwrap()[d])
                .sum::<f64>()
                / sum
        })
        .collect()
}

#[test]
fn partials_combine_to_the_exact_attention() {
    for (sinks, softcap) in [(None, 0.0), (Some(vec![0.3, -1.0, 2.0, 0.0]), 1.5)] {
        let op = AttentionOp {
            sinks,
            softcap,
            ..op()
        };
        let rows = context(&op, 150);
        let q = query(&op);
        for head in 0..op.n_head {
            let partials = head_partials(&op, head, &q, &rows).unwrap();
            assert_eq!(partials.len(), 3 * op.partial_len());
            let combined = combine(&op, head, &partials).unwrap();
            let exact: Vec<f32> = exact(&op, head, &q, &rows)
                .iter()
                .map(|&x| x as f32)
                .collect();
            assert!(combine_error(&combined, &exact) < 1e-6, "head {head}");
        }
    }
}

#[test]
fn a_dropped_block_is_seen_in_its_partial_or_the_combination() {
    let op = op();
    let rows = context(&op, 192);
    let q = query(&op);
    let honest = head_partials(&op, 1, &q, &rows).unwrap();
    // the node attended to the last two blocks only
    let cut: Vec<f32> = exact(&op, 1, &q, &rows[64..])
        .iter()
        .map(|&x| x as f32)
        .collect();
    let honest_out = combine(&op, 1, &honest).unwrap();
    assert!(combine_error(&honest_out, &cut) > COMBINE_TOLERANCE);
    // partials that combine to the cut output give the first block no weight
    let mut forged = honest.clone();
    forged[1] = 0.0;
    forged[2..op.partial_len()].fill(0.0);
    assert!(combine_error(&combine(&op, 1, &forged).unwrap(), &cut) < 1e-6);
    let recomputed = block_partial(&op, 1, &q, &rows[..64]).unwrap();
    assert!(partial_error(&forged[..op.partial_len()], &recomputed) > 0.5);
    assert!(partial_error(&honest[..op.partial_len()], &recomputed) < PARTIAL_TOLERANCE);
}

#[test]
fn a_head_reads_its_own_kv_head() {
    let op = op();
    assert_eq!(
        (0..4).map(|h| op.kv_head(h)).collect::<Vec<_>>(),
        vec![0, 0, 1, 1]
    );
}

#[test]
fn windows_bound_the_attended_positions() {
    assert!(Window::Causal.attends(5, 0));
    assert!(!Window::Causal.attends(5, 6));
    assert!(Window::Sliding(4).attends(5, 2));
    assert!(!Window::Sliding(4).attends(5, 1));
    assert!(Window::Chunked(4).attends(5, 4));
    assert!(!Window::Chunked(4).attends(5, 3));
}

fn f32_opening(kind: MixingKind, index: u32, values: &[f32]) -> TensorOpening {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    TensorOpening::new(kind, index, Some((crate::layer_tensors::GGML_F32, &bytes)))
}

/// Checks the partials of head 1 at the last of 70 positions against an
/// output that `op`'s rotation and then a gate transformed, read as
/// `layout` says.
fn check_transformed(
    op: &AttentionOp,
    read_as: impl Fn(AttentionLayout) -> AttentionLayout,
) -> bool {
    use crate::attention::opening::{AttentionPartials, AttentionPick};
    use crate::attention::verify::check_partials;
    use crate::wire::LeafOpening;

    let rows = context(op, 70);
    let q = query(op);
    let head = 1;
    let pick = AttentionPick {
        position: 69,
        layer: 0,
        head_draw: 1,
    };
    let opened = AttentionPartials::new(&pick, 1, &head_partials(op, head, &q, &rows).unwrap());
    let gate: Vec<f32> = (0..op.n_head * op.head_dim_v)
        .map(|i| 0.25 + value(900 + i as u64).abs() / 2.0)
        .collect();
    let mut exact = exact(op, head, &q, &rows);
    if let Some(rotation) = &op.output_rotation {
        rotation.apply(&mut exact).unwrap();
    }
    let mut output = vec![0.0; op.n_head * op.head_dim_v];
    for (d, x) in exact.into_iter().enumerate() {
        let i = head * op.head_dim_v + d;
        output[i] = (x * f64::from(gate[i])) as f32;
    }
    let leaf = LeafOpening {
        position: 69,
        residual_in: String::new(),
        residual_out: None,
        residual_in_path: Vec::new(),
        mixing_path: Vec::new(),
        residual_out_path: Vec::new(),
        mixing: vec![
            f32_opening(MixingKind::Input, 0, &q),
            f32_opening(MixingKind::Input, 1, &gate),
            f32_opening(MixingKind::Output, 0, &output),
        ],
        matrices: Vec::new(),
        selections: Vec::new(),
    };
    let layout = read_as(AttentionLayout {
        layer: 0,
        op: op.clone(),
        q_input: 0,
        k_store: 0,
        v_store: 1,
        output: 0,
        output_gate: Some(1),
    });
    check_partials(&layout, &pick, &leaf, &opened)
        .unwrap()
        .passed()
}

#[test]
fn a_gated_output_is_checked_after_its_gate() {
    assert!(check_transformed(&op(), |layout| layout));
    // read as ungated, the committed output is not the attention's
    assert!(!check_transformed(&op(), |layout| AttentionLayout {
        output_gate: None,
        ..layout
    }));
}

#[test]
fn a_rotated_output_is_checked_after_its_rotation() {
    let (s, c) = 0.6f32.sin_cos();
    let rotated = AttentionOp {
        output_rotation: Some(OutputRotation {
            n: 2,
            matrix: vec![c, -s, s, c],
        }),
        ..op()
    };
    assert!(check_transformed(&rotated, |layout| layout));
    let unrotated = |layout: AttentionLayout| AttentionLayout {
        op: AttentionOp {
            output_rotation: None,
            ..layout.op.clone()
        },
        ..layout
    };
    assert!(!check_transformed(&rotated, unrotated));
}

#[test]
fn a_rounded_q_is_checked_as_the_kernel_rounds_it() {
    let rounded = AttentionOp {
        head_dim_k: 32,
        q_rounding: Some(q_rounding(8).unwrap()),
        ..op()
    };
    rounded.check_shape().unwrap();
    let raw = AttentionOp {
        q_rounding: None,
        ..rounded.clone()
    };
    let rows = context(&rounded, 70);
    let q = query(&rounded);
    let as_kernel: Vec<f32> = Profile::CpuQ8.apply(&q).iter().map(|&x| x as f32).collect();
    let partials = |op: &AttentionOp, q: &[f32]| head_partials(op, 1, q, &rows).unwrap();
    assert_eq!(partials(&rounded, &q), partials(&raw, &as_kernel));
    assert_ne!(partials(&rounded, &q), partials(&raw, &q));
    assert!(check_transformed(&rounded, |layout| layout));
    assert!(q_rounding(2).is_err());
    let short = AttentionOp {
        q_rounding: Some(Profile::CpuQ8),
        ..op()
    };
    assert!(short.check_shape().is_err());
}
