//! The one form an opening of a challenge has.
//!
//! A check that looks a value up in an opening finds the first entry that
//! fits. Were an opening free to hold more than the challenge asks, a node
//! could put a decoy first: an extra stage, a second copy of a layer or a
//! position, a list in the wrong stage. The Merkle check would run on one
//! entry and the value checks on another. So an opening must hold exactly
//! what the challenge asks, in the order a stage opens it:
//!
//! - the receipt's stages, in order, by index;
//! - in each stage, the picked layers it holds, once each, in the
//!   challenge's order, and in each layer the pick's positions;
//! - in each position, the mixing values in leaf order, and each matmul and
//!   selection index once, in increasing order;
//! - the input of layer 0 at every picked position, on the stage that
//!   holds layer 0 and on no other;
//! - the leaf entering every stage but the first, and the leaf leaving
//!   every stage, at every picked position;
//! - the output residual at the head positions, on the final stage alone;
//! - attention partials and a recurrent region only for attention picks of
//!   the stage's layers, once each;
//! - what the stage committed at every picked position of the cached
//!   prefix, in position order, and the tokens of the whole prefix on the
//!   stage that holds layer 0 and on no other.
//!
//! Every failure here is the node's fault.

use std::collections::BTreeSet;

use anyhow::Result;

use crate::node_fault::ensure_node;
use crate::wire::{
    Challenge, LeafOpening, MixingKind, Opening, Receipt, StageCommitment, StageOpening,
    TensorOpening,
};

pub(super) fn check(receipt: &Receipt, challenge: &Challenge, opening: &Opening) -> Result<()> {
    ensure_node!(
        opening.receipt_id == receipt.receipt_id,
        "the opening names another receipt"
    );
    let stages: Vec<u32> = receipt.stages.iter().map(|s| s.stage_index).collect();
    let opened: Vec<u32> = opening.stages.iter().map(|s| s.stage_index).collect();
    ensure_node!(
        !stages.is_empty() && opened == stages,
        "the opening holds stages {opened:?}; the receipt's stages are {stages:?}"
    );
    let positions = challenge.positions();
    let prefix = challenge.prefix_positions(receipt.position_start);
    let last = receipt.stages.len() - 1;
    for (i, (stage, opened)) in receipt.stages.iter().zip(&opening.stages).enumerate() {
        check_layers(stage, challenge, opened)?;
        check_prefix(receipt, stage, opened, &prefix)?;
        let none: &[u32] = &[];
        let lists = [
            (
                "the input of layer 0",
                positions_of(opened.anchors.iter().map(|a| a.position)),
                if stage.layer_start == 0 {
                    &positions
                } else {
                    none
                },
            ),
            (
                "the residual entering the stage",
                positions_of(opened.entry.iter().map(|b| b.position)),
                if i > 0 { &positions } else { none },
            ),
            (
                "the residual leaving the stage",
                positions_of(opened.exit.iter().map(|b| b.position)),
                &positions,
            ),
            (
                "the output residual",
                positions_of(opened.head.iter().map(|h| h.position)),
                if i == last { &challenge.head } else { none },
            ),
        ];
        for (what, opened, asked) in lists {
            ensure_node!(
                opened == asked,
                "stage {} opens {what} at {opened:?}, the challenge asks {asked:?}",
                stage.stage_index
            );
        }
        check_attention(stage, challenge, opened)?;
    }
    Ok(())
}

fn positions_of(positions: impl Iterator<Item = u32>) -> Vec<u32> {
    positions.collect()
}

/// The stage opens what it committed at the picked positions of the
/// prefix, and the tokens of the prefix when it holds layer 0. A list of
/// another length would let a node send what no check reads.
fn check_prefix(
    receipt: &Receipt,
    stage: &StageCommitment,
    opened: &StageOpening,
    asked: &[u32],
) -> Result<()> {
    let positions = positions_of(opened.prefix.iter().map(|p| p.position));
    ensure_node!(
        positions == asked,
        "stage {} opens positions {positions:?} of the prefix, the challenge asks {asked:?}",
        stage.stage_index
    );
    let tokens = if stage.layer_start == 0 {
        receipt.position_start as usize
    } else {
        0
    };
    ensure_node!(
        opened.prefix_tokens.len() == tokens,
        "stage {} opens {} tokens of the prefix; it must open {tokens}",
        stage.stage_index,
        opened.prefix_tokens.len()
    );
    Ok(())
}

/// The stage opens the picked layers it holds, in the challenge's order,
/// each at its pick's positions.
fn check_layers(
    stage: &StageCommitment,
    challenge: &Challenge,
    opened: &StageOpening,
) -> Result<()> {
    let held = stage.layer_start..stage.layer_end;
    let picks: Vec<_> = challenge
        .picks
        .iter()
        .filter(|p| held.contains(&p.layer))
        .collect();
    let asked: Vec<u32> = picks.iter().map(|p| p.layer).collect();
    let layers: Vec<u32> = opened.layers.iter().map(|l| l.layer).collect();
    ensure_node!(
        layers == asked,
        "stage {} opens layers {layers:?}, the challenge picks {asked:?} of its layers",
        stage.stage_index
    );
    for (pick, layer) in picks.iter().zip(&opened.layers) {
        let positions = positions_of(layer.leaves.iter().map(|l| l.position));
        ensure_node!(
            positions == pick.positions,
            "layer {} opens positions {positions:?}, the challenge asked {:?}",
            pick.layer,
            pick.positions
        );
        for leaf in &layer.leaves {
            check_leaf(pick.layer, leaf)?;
        }
    }
    Ok(())
}

fn check_leaf(layer: u32, leaf: &LeafOpening) -> Result<()> {
    let position = leaf.position;
    check_mixing(&leaf.mixing).map_err(|error| error.context(format!("position {position}")))?;
    let increasing = |indices: Vec<u32>| indices.windows(2).all(|pair| pair[0] < pair[1]);
    ensure_node!(
        increasing(leaf.matrices.iter().map(|m| m.index).collect()),
        "position {position}: layer {layer} does not open each matmul once, in order"
    );
    ensure_node!(
        increasing(leaf.selections.iter().map(|s| s.index).collect()),
        "position {position}: layer {layer} does not open each selection once, in order"
    );
    Ok(())
}

/// A mixing leaf lists its values by kind, in the order the leaf hashes
/// them, and within a kind by index from 0 without a gap.
pub(crate) fn check_mixing(mixing: &[TensorOpening]) -> Result<()> {
    let mut previous: Option<(MixingKind, u32)> = None;
    for value in mixing {
        let expected = match previous {
            Some((kind, index)) if kind == value.kind => index.checked_add(1),
            Some((kind, _)) if kind > value.kind => None,
            _ => Some(0),
        };
        ensure_node!(
            expected == Some(value.index),
            "the mixing values are not in leaf order, each once: {:?} {} follows {previous:?}",
            value.kind,
            value.index
        );
        previous = Some((value.kind, value.index));
    }
    Ok(())
}

/// Partials and a recurrent region are opened only for attention picks of
/// the stage's layers: partials once per pick, and at most one region.
fn check_attention(
    stage: &StageCommitment,
    challenge: &Challenge,
    opened: &StageOpening,
) -> Result<()> {
    let held = stage.layer_start..stage.layer_end;
    let picked = |position: u32, layer: u32| {
        held.contains(&layer)
            && challenge
                .attention
                .iter()
                .any(|p| p.position == position && p.layer == layer)
    };
    let mut seen = BTreeSet::new();
    for partials in &opened.attention {
        ensure_node!(
            picked(partials.position, partials.layer)
                && seen.insert((partials.layer, partials.position)),
            "stage {} opens attention partials at position {} of layer {}, which is not one \
             attention pick of its layers, once",
            stage.stage_index,
            partials.position,
            partials.layer
        );
    }
    ensure_node!(
        opened.recurrent.len() <= 1,
        "stage {} opens {} recurrent regions; a stage opens one at most",
        stage.stage_index,
        opened.recurrent.len()
    );
    for region in &opened.recurrent {
        ensure_node!(
            picked(region.position, region.layer),
            "stage {} opens a recurrent region at position {} of layer {}, which is not an \
             attention pick of its layers",
            stage.stage_index,
            region.position,
            region.layer
        );
    }
    Ok(())
}
