use super::*;
use crate::node_fault::NodeFault;
use crate::toy_turn::{LAYERS, Turn};
use crate::wire::{LeafOpening, MixingKind, TensorOpening};

/// What a node changes in an honest opening.
type Change<'a> = Box<dyn Fn(&mut Opening) + 'a>;

/// Layer 0 at positions 1 and 3, layer 1 at positions 0 and 3, and the
/// head at two output positions.
fn challenged(turn: &Turn) -> Challenge {
    turn.challenge(&[(0, &[1, 3]), (1, &[0, 3])], &[2, 4])
}

/// The node's fault `change` makes of an honest opening.
fn fault(change: impl FnOnce(&mut Opening)) -> String {
    let turn = Turn::new();
    let challenge = challenged(&turn);
    let mut opening = turn.open(&challenge);
    change(&mut opening);
    let error = verify_paths(&turn.receipt, &challenge, &opening).unwrap_err();
    assert!(
        NodeFault::find(&error).is_some(),
        "not blamed on the node: {error:#}"
    );
    format!("{error:#}")
}

#[test]
fn an_honest_opening_is_verified() {
    let turn = Turn::new();
    verify_shape(&turn.receipt, LAYERS).unwrap();
    let challenge = challenged(&turn);
    let opening = turn.open(&challenge);
    let verified = verify_paths(&turn.receipt, &challenge, &opening).unwrap();
    // the view finds what the challenge asked, and nothing else
    let (layer, backend) = verified.layer(1).unwrap();
    assert_eq!((layer.layer, backend), (1, "CPU"));
    assert_eq!(verified.leaf(1, 3).unwrap().position, 3);
    assert!(verified.leaf(1, 1).is_none());
    assert!(verified.layer(2).is_none());
    let anchors: Vec<u32> = verified.anchors().iter().map(|a| a.position).collect();
    assert_eq!(anchors, [0, 1, 3]);
    let (_, head) = verified.head().unwrap();
    assert_eq!(head.iter().map(|h| h.position).collect::<Vec<_>>(), [2, 4]);
    assert!(verified.partials(&challenge_pick(1, 3)).is_none());
    assert!(verified.region(&turn.receipt.stages[1]).is_none());
}

fn challenge_pick(layer: u32, position: u32) -> crate::attention::opening::AttentionPick {
    crate::attention::opening::AttentionPick {
        position,
        layer,
        head_draw: 0,
    }
}

#[test]
fn a_decoy_stage_in_front_is_refused() {
    // The attack: a stage with an index the receipt does not have, put
    // first, with a layer of the node's own values under a picked layer's
    // number. The Merkle check looked stages up by index and skipped it;
    // the value checks took the first layer that matched and read it.
    let message = fault(|opening| {
        let mut decoy = opening.stages[1].clone();
        decoy.stage_index = 99;
        decoy.layers[0].leaves[0].residual_in = crate::wire::encode_f32(&[9.0; 4]);
        opening.stages.insert(0, decoy);
    });
    assert!(message.contains("stages [99, 0, 1]"), "{message}");
}

#[test]
fn a_stage_twice_missing_or_out_of_order_is_refused() {
    fault(|opening| opening.stages.push(opening.stages[1].clone()));
    fault(|opening| opening.stages.insert(1, opening.stages[0].clone()));
    fault(|opening| opening.stages.swap(0, 1));
    fault(|opening| {
        opening.stages.pop();
    });
    fault(|opening| opening.stages.clear());
}

#[test]
fn a_layer_the_challenge_did_not_pick_is_refused() {
    // an extra layer of the stage's own, a second copy, one from another
    // stage, and a picked layer left out
    let message = fault(|opening| {
        let mut extra = opening.stages[1].layers[0].clone();
        extra.layer = 0;
        opening.stages[1].layers.push(extra);
    });
    assert!(message.contains("opens layers [1, 0]"), "{message}");
    fault(|opening| {
        let copy = opening.stages[0].layers[0].clone();
        opening.stages[0].layers.insert(0, copy);
    });
    fault(|opening| {
        let moved = opening.stages[0].layers.remove(0);
        opening.stages[1].layers.push(moved);
    });
    fault(|opening| opening.stages[1].layers.clear());
}

#[test]
fn a_position_twice_missing_or_unasked_is_refused() {
    fn leaves(opening: &mut Opening) -> &mut Vec<LeafOpening> {
        &mut opening.stages[1].layers[0].leaves
    }
    let message = fault(|opening| {
        let copy = leaves(opening)[1].clone();
        leaves(opening).push(copy);
    });
    assert!(message.contains("opens positions [0, 3, 3]"), "{message}");
    fault(|opening| leaves(opening).swap(0, 1));
    fault(|opening| {
        leaves(opening).pop();
    });
    // a decoy leaf under a picked position's number, in front
    fault(|opening| {
        let mut decoy = leaves(opening)[1].clone();
        decoy.position = 0;
        leaves(opening).insert(0, decoy);
    });
}

#[test]
fn a_matmul_selection_or_mixing_value_twice_is_refused() {
    fn leaf(opening: &mut Opening) -> &mut LeafOpening {
        &mut opening.stages[0].layers[0].leaves[0]
    }
    // 27 copies of an honest column around one that cheats would have
    // raised the row check's allowance past it
    let message = fault(|opening| {
        let copy = leaf(opening).matrices[0].clone();
        leaf(opening).matrices.insert(0, copy);
    });
    assert!(message.contains("each matmul once"), "{message}");
    fault(|opening| leaf(opening).matrices.swap(0, 2));
    fault(|opening| {
        let copy = leaf(opening).selections[0].clone();
        leaf(opening).selections.push(copy);
    });
    // a second stored row under index 0, and values out of leaf order
    fault(|opening| {
        let copy = leaf(opening).mixing[1].clone();
        leaf(opening).mixing.insert(1, copy);
    });
    fault(|opening| leaf(opening).mixing.swap(0, 2));
    fault(|opening| leaf(opening).mixing[2].index = 1);
}

#[test]
fn mixing_values_are_in_leaf_order_from_index_zero() {
    let value = |kind, index| TensorOpening::new(kind, index, None);
    let (input, stored, output, state) = (
        MixingKind::Input,
        MixingKind::Stored,
        MixingKind::Output,
        MixingKind::State,
    );
    let honest = [
        value(input, 0),
        value(input, 1),
        value(stored, 0),
        value(output, 0),
        value(state, 0),
    ];
    canonical::check_mixing(&honest).unwrap();
    canonical::check_mixing(&[]).unwrap();
    // a kind may be absent
    canonical::check_mixing(&[value(stored, 0), value(stored, 1)]).unwrap();
    for bad in [
        vec![value(input, 1)],
        vec![value(input, 0), value(input, 0)],
        vec![value(input, 0), value(input, 2)],
        vec![value(stored, 0), value(input, 0)],
        vec![value(output, 0), value(state, 1)],
    ] {
        assert!(canonical::check_mixing(&bad).is_err(), "{bad:?}");
    }
}

#[test]
fn a_list_in_the_wrong_stage_or_with_extras_is_refused() {
    // anchors belong to the stage holding layer 0, the head to the last,
    // and an entry leaf to every stage but the first
    let message = fault(|opening| {
        let anchors = opening.stages[0].anchors.clone();
        opening.stages[1].anchors = anchors;
    });
    assert!(message.contains("the input of layer 0"), "{message}");
    fault(|opening| {
        let head = opening.stages[1].head.clone();
        opening.stages[0].head = head;
    });
    fault(|opening| {
        let entry = opening.stages[1].entry.clone();
        opening.stages[0].entry = entry;
    });
    // twice, missing, or at a position the challenge did not pick
    fault(|opening| {
        let copy = opening.stages[0].anchors[0].clone();
        opening.stages[0].anchors.insert(0, copy);
    });
    fault(|opening| {
        opening.stages[0].anchors.pop();
    });
    fault(|opening| {
        let copy = opening.stages[1].head[0].clone();
        opening.stages[1].head.push(copy);
    });
    fault(|opening| opening.stages[1].head.clear());
    fault(|opening| {
        let copy = opening.stages[1].exit[0].clone();
        opening.stages[1].exit.push(copy);
    });
    fault(|opening| {
        opening.stages[0].exit.remove(0);
    });
    fault(|opening| opening.stages[1].entry.swap(0, 1));
}

#[test]
fn partials_and_regions_are_opened_only_for_attention_picks() {
    use crate::attention::opening::AttentionPartials;
    let turn = Turn::new();
    let mut challenge = challenged(&turn);
    challenge.attention = vec![challenge_pick(1, 3)];
    let honest = turn.open(&challenge);
    let partials = |layer, position| {
        AttentionPartials::new(&challenge_pick(layer, position), 0, &[0.0, 1.0, 2.0])
    };
    let verify = |change: &dyn Fn(&mut Opening)| {
        let mut opening = honest.clone();
        change(&mut opening);
        verify_paths(&turn.receipt, &challenge, &opening).map(|_| ())
    };
    // partials are no leaf: the pick's own are accepted here, and the
    // attention check judges them
    verify(&|opening| opening.stages[1].attention = vec![partials(1, 3)]).unwrap();
    let refused: [&dyn Fn(&mut Opening); 5] = [
        // not an attention pick
        &|opening| opening.stages[1].attention = vec![partials(1, 0)],
        // the pick's, on a stage that does not hold its layer
        &|opening| opening.stages[0].attention = vec![partials(1, 3)],
        // twice
        &|opening| opening.stages[1].attention = vec![partials(1, 3), partials(1, 3)],
        // a region for a position that is no attention pick
        &|opening| opening.stages[1].recurrent = vec![region(1, 0)],
        // two regions
        &|opening| opening.stages[1].recurrent = vec![region(1, 3), region(1, 3)],
    ];
    for (i, change) in refused.iter().enumerate() {
        let error = verify(change).unwrap_err();
        assert!(NodeFault::find(&error).is_some(), "change {i}: {error:#}");
    }
    // the pick's region is checked against the tree like any leaf
    let error = verify(&|opening| opening.stages[1].recurrent = vec![region(1, 3)]).unwrap_err();
    assert!(NodeFault::find(&error).is_some(), "{error:#}");
}

fn region(layer: u32, position: u32) -> crate::wire::RecurrentOpening {
    crate::wire::RecurrentOpening {
        position,
        layer,
        start: position,
        end: position + 1,
        leaves: Vec::new(),
    }
}

#[test]
fn every_opened_leaf_is_checked_against_its_root() {
    let other = crate::wire::encode_f32(&[1.0, 2.0, 3.0, 4.0]);
    let changes: Vec<Change<'_>> = vec![
        Box::new(|o| o.stages[0].layers[0].leaves[1].residual_in = other.clone()),
        Box::new(|o| o.stages[1].layers[0].leaves[0].residual_out = Some(other.clone())),
        Box::new(|o| {
            let bytes = [0u8; 8];
            o.stages[1].layers[0].leaves[0].mixing[1] =
                TensorOpening::new(MixingKind::Stored, 0, Some((1, &bytes)));
        }),
        Box::new(|o| o.stages[0].anchors[2].residual = other.clone()),
        Box::new(|o| o.stages[1].head[1].residual = other.clone()),
        // the leaf leaving the final stage, which no later stage vouches for
        Box::new(|o| o.stages[1].exit[0].leaf = hex::encode([3u8; 32])),
        Box::new(|o| {
            let leaf = hex::encode([3u8; 32]);
            o.stages[0].exit[1].leaf = leaf.clone();
            o.stages[1].entry[1].leaf = leaf;
        }),
        Box::new(|o| o.stages[0].layers[0].leaves[0].mixing_path.clear()),
    ];
    for (i, change) in changes.into_iter().enumerate() {
        let message = fault(change);
        assert!(!message.is_empty(), "change {i}");
    }
}

#[test]
fn a_stored_row_of_a_type_the_receipt_does_not_name_is_refused() {
    let message = fault(|opening| {
        let row = &mut opening.stages[0].layers[0].leaves[0].mixing[1];
        row.ggml_type = 39;
    });
    assert!(message.contains("type39"), "{message}");
}

#[test]
fn a_challenge_for_another_receipt_blames_nobody() {
    let turn = Turn::new();
    let mut challenge = challenged(&turn);
    let opening = turn.open(&challenge);
    challenge.receipt_id = "another".into();
    let error = verify_paths(&turn.receipt, &challenge, &opening).unwrap_err();
    assert!(NodeFault::find(&error).is_none(), "{error:#}");
}

#[test]
fn a_cached_prefix_is_committed_exactly_when_the_turn_continues_one() {
    use crate::wire::PrefixCommitment;
    let fresh = Turn::new().receipt;
    verify_shape(&fresh, LAYERS).unwrap();
    let continued = Turn::starting_at(40).receipt;
    verify_shape(&continued, LAYERS).unwrap();
    let prefix = |positions| {
        Some(PrefixCommitment {
            root: hex::encode([7u8; 32]),
            positions,
            tokens: None,
        })
    };
    let mut claimed = fresh.clone();
    claimed.stages[1].prefix = prefix(40);
    let mut uncommitted = continued.clone();
    uncommitted.stages[0].prefix = None;
    let mut shorter = continued.clone();
    shorter.stages[1].prefix = prefix(39);
    // the stage that holds layer 0 commits the prefix's tokens, alone
    let mut no_tokens = continued.clone();
    no_tokens.stages[0].prefix.as_mut().unwrap().tokens = None;
    let mut extra_tokens = continued.clone();
    extra_tokens.stages[1].prefix.as_mut().unwrap().tokens = Some(hex::encode([1u8; 32]));
    for receipt in [claimed, uncommitted, shorter, no_tokens, extra_tokens] {
        let error = verify_shape(&receipt, LAYERS).unwrap_err();
        assert!(NodeFault::find(&error).is_some(), "{error:#}");
        assert!(format!("{error:#}").contains("cached prefix"), "{error:#}");
    }
}

#[test]
fn a_receipt_past_any_sequence_is_refused() {
    // positions are i32 in the runtime: one past that would stop every
    // check with an error instead of a verdict
    let mut receipt = Turn::new().receipt;
    let span = receipt.position_end - receipt.position_start;
    receipt.position_start = u32::MAX - span;
    receipt.position_end = u32::MAX;
    receipt.generation_start = receipt.position_start;
    let error = verify_shape(&receipt, LAYERS).unwrap_err();
    assert!(NodeFault::find(&error).is_some(), "{error:#}");
    // another version is not a defect: this verifier cannot read it
    let mut other = Turn::new().receipt;
    other.version += 1;
    let error = verify_shape(&other, LAYERS).unwrap_err();
    assert!(NodeFault::find(&error).is_none(), "{error:#}");
}
