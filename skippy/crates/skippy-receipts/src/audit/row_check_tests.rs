//! The row check on layers of an honest toy turn, and on what a node could
//! make of them.

use crate::audit::{MatmulPlan, Routers, allowed_failing_columns, check_rows};
use crate::glue::check_glue;
use crate::node_fault::NodeFault;
use crate::toy_turn::{ToyModel, Turn, weights, weights_out_of_reach};
use crate::wire::{LayerOpening, encode_f32};

const SEED: &[u8] = b"rows seed";

/// Layer 1 of an honest turn at four positions, and the auditor's plan of
/// it.
fn layer() -> (LayerOpening, MatmulPlan) {
    let turn = Turn::new();
    let challenge = turn.challenge(&[(1, &[0, 1, 3, 4])], &[]);
    let layer = turn.open(&challenge).stages[1].layers.remove(0);
    let plan = check_glue(&ToyModel, &layer).unwrap().matmuls;
    (layer, plan)
}

#[test]
fn an_honest_layer_passes_its_row_checks() {
    let (layer, plan) = layer();
    let report = check_rows(&weights(), &layer, Routers::Planned(&plan), SEED, "CPU").unwrap();
    assert!(report.pass, "{:?}", report.matrices);
    let names: Vec<&str> = report.matrices.iter().map(|m| m.weight.as_str()).collect();
    assert_eq!(
        names,
        ["blk.1.a.weight", "blk.1.exps.weight", "blk.1.router.weight"]
    );
    // one column per position, and one per expert used
    let columns: Vec<usize> = report.matrices.iter().map(|m| m.tokens).collect();
    assert_eq!(columns, [4, 8, 4]);
    assert!(report.matrices.iter().all(|m| m.failing_tokens == 0));
}

#[test]
fn copies_of_a_matmul_do_not_pad_the_fail_rule() {
    // The attack: every position cheats on `a`, which fails four columns
    // of four. Each position then opens 31 more copies of an honest column
    // of it, so 4 failures of 128 are within the 2 + n/32 the rule allows
    // a matrix.
    let (mut layer, plan) = layer();
    for leaf in &mut layer.leaves {
        let honest = leaf.matrices[0].clone();
        leaf.matrices[0].output = encode_f32(&[9.0, 9.0, 9.0, 9.0]);
        leaf.matrices.extend(vec![honest; 31]);
    }
    assert!(4 <= allowed_failing_columns(4 * 32));
    let error = check_rows(&weights(), &layer, Routers::Planned(&plan), SEED, "CPU").unwrap_err();
    assert!(NodeFault::find(&error).is_some(), "{error:#}");
    assert!(format!("{error:#}").contains("more than once"), "{error:#}");
    // without the copies the cheated matmul fails
    let (mut layer, plan) = self::layer();
    for leaf in &mut layer.leaves {
        leaf.matrices[0].output = encode_f32(&[9.0, 9.0, 9.0, 9.0]);
    }
    let report = check_rows(&weights(), &layer, Routers::Planned(&plan), SEED, "CPU").unwrap();
    let a = &report.matrices[0];
    assert_eq!((a.tokens, a.failing_tokens, a.pass), (4, 4, false));
}

#[test]
fn the_router_flag_comes_from_the_plan_and_not_the_node() {
    // A node clears the flag of its router to have 16 rows checked instead
    // of all, or sets it everywhere to make the client read every row of
    // every weight. Neither is read.
    let (mut layer, plan) = layer();
    for leaf in &mut layer.leaves {
        for m in &mut leaf.matrices {
            m.router = !m.router;
        }
    }
    let report = check_rows(&weights(), &layer, Routers::Planned(&plan), SEED, "CPU").unwrap();
    let routers: Vec<(&str, bool)> = report
        .matrices
        .iter()
        .map(|m| (m.weight.as_str(), m.router))
        .collect();
    assert_eq!(
        routers,
        [
            ("blk.1.a.weight", false),
            ("blk.1.exps.weight", false),
            ("blk.1.router.weight", true)
        ]
    );
}

#[test]
fn a_matmul_the_layer_does_not_have_is_the_nodes_fault() {
    let (mut layer, plan) = layer();
    layer.leaves[0].matrices[2].index = 7;
    let error = check_rows(&weights(), &layer, Routers::Planned(&plan), SEED, "CPU").unwrap_err();
    assert!(NodeFault::find(&error).is_some(), "{error:#}");
    // a weight of the same shape under the matmul's index
    let (mut layer, plan) = self::layer();
    layer.leaves[0].matrices[0].weight = "blk.0.a.weight".into();
    let error = check_rows(&weights(), &layer, Routers::Planned(&plan), SEED, "CPU").unwrap_err();
    assert!(NodeFault::find(&error).is_some(), "{error:#}");
    // every matmul left out
    let (mut layer, plan) = self::layer();
    layer.leaves.iter_mut().for_each(|l| l.matrices.clear());
    let error = check_rows(&weights(), &layer, Routers::Planned(&plan), SEED, "CPU").unwrap_err();
    assert!(NodeFault::find(&error).is_some(), "{error:#}");
}

#[test]
fn a_column_that_is_not_a_number_or_does_not_fit_is_the_nodes_fault() {
    for output in [
        encode_f32(&[f32::NAN, 0.0, 0.0, 0.0]),
        encode_f32(&[f32::INFINITY, 0.0, 0.0, 0.0]),
        encode_f32(&[1.0, 2.0, 3.0]),
        "not base64".to_owned(),
    ] {
        let (mut layer, plan) = layer();
        layer.leaves[1].matrices[0].output = output.clone();
        let error =
            check_rows(&weights(), &layer, Routers::Planned(&plan), SEED, "CPU").unwrap_err();
        assert!(NodeFault::find(&error).is_some(), "{output}: {error:#}");
    }
}

#[test]
fn a_weight_the_client_cannot_read_blames_nobody() {
    let (layer, plan) = layer();
    let error = check_rows(
        &weights_out_of_reach(),
        &layer,
        Routers::Planned(&plan),
        SEED,
        "CPU",
    )
    .unwrap_err();
    assert!(NodeFault::find(&error).is_none(), "{error:#}");
    // nor does a backend without rounding profiles
    let error =
        check_rows(&weights(), &layer, Routers::Planned(&plan), SEED, "Vulkan0").unwrap_err();
    assert!(NodeFault::find(&error).is_none(), "{error:#}");
}

#[test]
fn without_a_plan_no_matmul_is_taken_for_a_router_and_a_bad_column_still_fails() {
    // The auditor could not evaluate the layer, so it has no plan. The
    // opening then does not get to say which weights are read whole.
    let (mut layer, _) = layer();
    for leaf in &mut layer.leaves {
        for m in &mut leaf.matrices {
            m.router = true;
        }
    }
    let report = check_rows(&weights(), &layer, Routers::Unknown, SEED, "CPU").unwrap();
    assert!(report.pass, "{:?}", report.matrices);
    assert!(report.matrices.iter().all(|m| !m.router));
    // a column that is not the weight's still fails
    for leaf in &mut layer.leaves {
        leaf.matrices[0].output = encode_f32(&[9.0, 9.0, 9.0, 9.0]);
    }
    let report = check_rows(&weights(), &layer, Routers::Unknown, SEED, "CPU").unwrap();
    assert_eq!(report.matrices[0].failing_tokens, 4);
    assert!(!report.pass);
    // and a copy of a matmul, or a weight the GGUF does not have, is the
    // node's fault with or without a plan
    let copy = layer.leaves[0].matrices[1].clone();
    layer.leaves[0].matrices.push(copy);
    let error = check_rows(&weights(), &layer, Routers::Unknown, SEED, "CPU").unwrap_err();
    assert!(NodeFault::find(&error).is_some(), "{error:#}");
    let (mut layer, _) = self::layer();
    layer.leaves[0].matrices[0].weight = "blk.1.none.weight".into();
    let error = check_rows(&weights(), &layer, Routers::Unknown, SEED, "CPU").unwrap_err();
    assert!(NodeFault::find(&error).is_some(), "{error:#}");
}
