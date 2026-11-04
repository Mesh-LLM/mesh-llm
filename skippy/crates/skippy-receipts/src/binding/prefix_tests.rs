use super::super::{verify_paths, verify_shape};
use crate::node_fault::NodeFault;
use crate::toy_turn::{LAYERS, Turn};
use crate::wire::{Challenge, Opening, encode_f32};

/// A turn that continued a prefix of 40 positions.
const PREFIX: u32 = 40;

/// Positions 7 and 31 of the prefix and two of the turn, in both layers.
fn challenged(turn: &Turn) -> Challenge {
    turn.challenge(&[(0, &[7, 31, 41, 43]), (1, &[7, 31, 40, 43])], &[42, 44])
}

/// The node's fault `change` makes of an honest opening of a cached turn.
fn fault(change: impl FnOnce(&mut Opening)) -> String {
    let turn = Turn::starting_at(PREFIX);
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
fn an_honest_opening_of_a_cached_turn_is_verified() {
    let turn = Turn::starting_at(PREFIX);
    verify_shape(&turn.receipt, LAYERS).unwrap();
    let challenge = challenged(&turn);
    assert_eq!(challenge.prefix_positions(PREFIX), [7, 31]);
    let opening = turn.open(&challenge);
    let verified = verify_paths(&turn.receipt, &challenge, &opening).unwrap();
    // a position of the prefix is opened like one of the turn
    assert_eq!(verified.leaf(1, 31).unwrap().position, 31);
    let anchors: Vec<u32> = verified.anchors().iter().map(|a| a.position).collect();
    assert_eq!(anchors, [7, 31, 40, 41, 43]);
    assert_eq!(verified.prefix_tokens().len(), PREFIX as usize);
    // a turn from position 0 opens nothing of a prefix
    let fresh = Turn::new();
    let challenge = fresh.challenge(&[(0, &[1])], &[]);
    let opening = fresh.open(&challenge);
    assert!(opening.stages.iter().all(|s| s.prefix.is_empty()));
    let verified = verify_paths(&fresh.receipt, &challenge, &opening).unwrap();
    assert!(verified.prefix_tokens().is_empty());
}

#[test]
fn a_value_the_prefix_did_not_commit_is_refused() {
    // a residual, a mixing value and the input of layer 0
    let message = fault(|o| o.stages[1].layers[0].leaves[0].residual_in = encode_f32(&[9.0; 4]));
    assert!(message.contains("position 7"), "{message}");
    assert!(message.contains("cached prefix"), "{message}");
    fault(|o| o.stages[0].layers[0].leaves[1].mixing[2].values = Some("AAAAAA==".into()));
    fault(|o| o.stages[0].anchors[0].residual = encode_f32(&[9.0; 4]));
    // the leaves at the stage boundary must be the ones both stages commit
    fault(|o| o.stages[0].exit[0].leaf = hex::encode([3u8; 32]));
    // a leaf of the prefix is in no tree: a path is not an answer
    let message = fault(|o| {
        let path = o.stages[0].layers[0].leaves[2].residual_in_path.clone();
        o.stages[0].layers[0].leaves[0].residual_in_path = path;
    });
    assert!(message.contains("with a path"), "{message}");
}

#[test]
fn the_opened_leaves_of_a_prefix_position_are_bound_to_the_prefix_root() {
    // an altered leaf, another cell, another position's opening
    let message = fault(|o| o.stages[0].prefix[0].leaves[1] = hex::encode([3u8; 32]));
    assert!(message.contains("prefix root"), "{message}");
    fault(|o| o.stages[0].prefix[0].cell += 1);
    fault(|o| {
        let other = o.stages[1].prefix[1].clone();
        o.stages[1].prefix[0].leaves = other.leaves;
    });
    fault(|o| {
        o.stages[1].prefix[0]
            .path
            .pop()
            .map(drop)
            .unwrap_or_default()
    });
    // the rows the cell stores are the rows the mixing leaf holds
    let message = fault(|o| o.stages[0].prefix[0].stored[0] = hex::encode([3u8; 32]));
    assert!(message.contains("prefix root"), "{message}");
    // a list of another length
    fault(|o| o.stages[0].prefix[0].leaves.push(hex::encode([3u8; 32])));
    fault(|o| {
        o.stages[0].prefix[0].leaves.pop();
    });
    fault(|o| o.stages[1].prefix[1].stored.clear());
}

#[test]
fn a_prefix_position_is_opened_exactly_when_it_is_picked() {
    // missing, twice, out of order, or one the challenge did not pick
    let message = fault(|o| {
        o.stages[0].prefix.pop();
    });
    assert!(message.contains("of the prefix"), "{message}");
    fault(|o| o.stages[1].prefix.clear());
    fault(|o| o.stages[0].prefix.swap(0, 1));
    fault(|o| {
        let twice = o.stages[0].prefix[0].clone();
        o.stages[0].prefix.push(twice);
    });
    fault(|o| o.stages[0].prefix[1].position = 32);
    // a position of the turn is not one of the prefix
    fault(|o| o.stages[0].prefix[1].position = 41);
}

#[test]
fn the_tokens_of_the_prefix_are_bound_to_the_receipt() {
    // a wrong token: the list no longer hashes to what stage 0 committed
    let message = fault(|o| o.stages[0].prefix_tokens[5] += 1);
    assert!(message.contains("tokens of the prefix"), "{message}");
    // a short list, a long one, and one on a stage that holds no layer 0
    let message = fault(|o| {
        o.stages[0].prefix_tokens.pop();
    });
    assert!(message.contains("it must open 40"), "{message}");
    fault(|o| o.stages[0].prefix_tokens.push(1));
    fault(|o| o.stages[0].prefix_tokens.clear());
    fault(|o| o.stages[1].prefix_tokens = o.stages[0].prefix_tokens.clone());
    // the token an anchor names is the prefix's
    let message = fault(|o| o.stages[0].anchors[1].token += 1);
    assert!(message.contains("anchor holds token"), "{message}");
}

#[test]
fn a_node_that_computed_a_prefix_position_from_another_token_fails() {
    // The node commits the tokens the client sent, but computed position 7
    // of the prefix from another one. The layer-0 leaf it committed binds
    // the token it used, so no opening of the position matches it.
    let turn = Turn::starting_at(PREFIX);
    let challenge = challenged(&turn);
    let honest = turn.open(&challenge);
    let mut receipt = turn.receipt.clone();
    let mut tokens = honest.stages[0].prefix_tokens.clone();
    tokens[7] += 1;
    receipt.stages[0].prefix.as_mut().unwrap().tokens = Some(hex::encode(
        crate::commitment::prefix_tokens_digest(&tokens),
    ));
    let mut opening = honest;
    opening.stages[0].prefix_tokens = tokens;
    opening.stages[0].anchors[0].token += 1;
    let error = verify_paths(&receipt, &challenge, &opening).unwrap_err();
    assert!(NodeFault::find(&error).is_some(), "{error:#}");
    assert!(format!("{error:#}").contains("position 7"), "{error:#}");
}

#[test]
fn a_prefix_opening_of_a_turn_without_a_prefix_is_refused() {
    let cached = Turn::starting_at(PREFIX);
    let stolen = cached.open(&challenged(&cached)).stages[0].prefix[0].clone();
    let turn = Turn::new();
    let challenge = turn.challenge(&[(0, &[1])], &[]);
    let mut opening = turn.open(&challenge);
    opening.stages[0].prefix.push(stolen);
    let error = verify_paths(&turn.receipt, &challenge, &opening).unwrap_err();
    assert!(NodeFault::find(&error).is_some(), "{error:#}");
}
