use super::*;
use crate::activation_rounding::Profile;
use crate::attention::test_op::{context, exact, op, query, value};
use crate::attention::{AttentionOp, OutputRotation, head_partials, q_rounding};
use crate::wire::{MixingKind, TensorOpening};

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
    assert_eq!(rounded.q_rounding, Some(Profile::CpuQ8));
    assert!(check_transformed(&rounded, |layout| layout));
}

#[test]
fn values_that_are_not_numbers_fail_the_attention_comparisons() {
    let op = op();
    let rows = context(&op, 150);
    let q = query(&op);
    let honest = head_partials(&op, 1, &q, &rows).unwrap();
    let recomputed = block_partial(&op, 1, &q, &rows[..64]).unwrap();
    let len = op.partial_len();
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for at in 0..len {
            let mut forged = honest.clone();
            forged[at] = bad;
            assert!(combine(&op, 1, &forged).is_err(), "value {at} as {bad}");
            assert_eq!(partial_error(&forged[..len], &recomputed), f64::INFINITY);
        }
    }
    // a recomputation that is not a number is no match either
    let mut broken = recomputed.clone();
    broken[2] = f64::NAN;
    assert_eq!(partial_error(&honest[..len], &broken), f64::INFINITY);
    let combined = combine(&op, 1, &honest).unwrap();
    let output: Vec<f32> = combined.iter().map(|&x| x as f32).collect();
    assert!(combine_error(&combined, &output) < 1e-6);
    for bad in [f32::NAN, f32::INFINITY] {
        let mut forged = output.clone();
        forged[1] = bad;
        assert_eq!(combine_error(&combined, &forged), f64::INFINITY, "{bad}");
    }
    assert_eq!(combine_error(&[f64::NAN, 0.0], &[0.0, 0.0]), f64::INFINITY);
}

#[test]
fn the_combine_tolerance_grows_with_the_context() {
    // a short context keeps the floor
    assert_eq!(combine_tolerance(1), COMBINE_TOLERANCE);
    assert_eq!(combine_tolerance(400), COMBINE_TOLERANCE);
    // the error honest f16 prefill was measured at passes at its context,
    // and would fail at a short one
    for (context, honest) in [(7_321, 0.0194), (18_876, 0.0285), (29_962, 0.0390)] {
        assert!(honest < combine_tolerance(context), "{context}");
        assert!(honest > COMBINE_TOLERANCE * 0.9);
    }
    assert!(0.0390 > combine_tolerance(1_000));
    // it never grows faster than the square root
    assert!(combine_tolerance(32_768) < 0.18);
    assert!(combine_tolerance(4_096) < combine_tolerance(16_384));
}

mod asked_rows {
    use crate::attention::opening::{
        AttentionPick, BlockChallenge, BlockPick, RowOpening, RowProof, RowsOpening, StageRows,
    };
    use crate::attention_check::{OpenedHead, verify_answer};
    use crate::node_fault::NodeFault;
    use crate::toy_turn::Turn;

    /// One head at position 99 of layer 1, of which block 1 (positions
    /// 64..100) is asked, and the rows an honest node opens for it.
    fn asked() -> (Turn, BlockChallenge, Vec<OpenedHead>, RowsOpening) {
        let turn = Turn::new();
        let pick = AttentionPick {
            position: 99,
            layer: 1,
            head_draw: 0,
        };
        let head = OpenedHead {
            pick,
            head: 0,
            partials: Vec::new(),
            context: (0..100).collect(),
            combine_error: 0.0,
        };
        let blocks = BlockChallenge {
            receipt_id: turn.receipt.receipt_id.clone(),
            picks: vec![BlockPick {
                position: 99,
                layer: 1,
                blocks: vec![1],
            }],
            proof: None,
        };
        let row = |position| RowOpening {
            position,
            layer: 1,
            stored: Vec::new(),
            proof: RowProof::Turn {
                rest: String::new(),
                path: Vec::new(),
            },
        };
        let stage = |stage_index, rows| StageRows { stage_index, rows };
        let rows = RowsOpening {
            receipt_id: turn.receipt.receipt_id.clone(),
            stages: vec![stage(0, Vec::new()), stage(1, (64..100).map(row).collect())],
        };
        (turn, blocks, vec![head], rows)
    }

    #[test]
    fn the_rows_asked_for_are_accepted() {
        let (turn, blocks, heads, rows) = asked();
        verify_answer(&turn.receipt, &blocks, &heads, &rows).unwrap();
    }

    #[test]
    fn a_row_left_out_twice_or_unasked_is_the_nodes_fault() {
        let changes: [fn(&mut RowsOpening); 7] = [
            |rows| {
                rows.stages[1].rows.pop();
            },
            |rows| {
                let copy = rows.stages[1].rows[0].clone();
                rows.stages[1].rows.push(copy);
            },
            // a row of a block that was not asked
            |rows| rows.stages[1].rows[0].position = 3,
            |rows| rows.stages[1].rows[0].layer = 0,
            // a decoy stage in front, a stage twice, and none
            |rows| {
                let mut decoy = rows.stages[1].clone();
                decoy.stage_index = 9;
                rows.stages.insert(0, decoy);
            },
            |rows| rows.stages.push(rows.stages[1].clone()),
            |rows| rows.stages.clear(),
        ];
        for (i, change) in changes.iter().enumerate() {
            let (turn, blocks, heads, mut rows) = asked();
            change(&mut rows);
            let error = verify_answer(&turn.receipt, &blocks, &heads, &rows).unwrap_err();
            assert!(NodeFault::find(&error).is_some(), "change {i}: {error:#}");
        }
        let (turn, blocks, heads, mut rows) = asked();
        rows.receipt_id = "another".into();
        assert!(verify_answer(&turn.receipt, &blocks, &heads, &rows).is_err());
    }
}
