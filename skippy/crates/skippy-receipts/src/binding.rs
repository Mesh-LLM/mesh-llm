//! Client-side checks that bind a receipt to the model and the request,
//! and an opening to the receipt.
//!
//! - **Shape:** the receipt's stages cover every layer of the claimed
//!   model, from 0, without gaps; each stage names the leaf count its
//!   positions and layers make; it holds a token for every position; and
//!   every stage commits a cached prefix exactly when the turn starts
//!   after position 0.
//! - **Form:** an opening has one form for a challenge ([`canonical`]):
//!   the receipt's stages, the picked layers and positions, and nothing
//!   else. So nothing in it escapes the checks below.
//! - **Paths:** every opened leaf hashes to its stage root along a path of
//!   the shape that tree has. This covers each picked layer, the input of
//!   layer 0 with its token (the anchor), the residual leaves at stage
//!   boundaries, which must be equal on both sides, the residual the
//!   output head reads at every head position, and the mixing leaves of a
//!   recurrent region.
//! - **Prefix:** a picked position of the turn's cached prefix is in no
//!   stage tree. The stage opens every leaf it committed when it computed
//!   the position, by hash. Their digest, with the cell and the digest of
//!   the rows the cell stores, must be a leaf of the prefix tree the
//!   receipt commits, and every opened leaf of the position must be one of
//!   them ([`prefix`]). The tokens of the prefix must hash to the digest
//!   the stage that holds layer 0 committed.
//!
//! [`verify_paths`] returns the opening as a [`VerifiedOpening`]. The
//! checks of the opened values take that, so they cannot run on values
//! that were not checked against a root.

use anyhow::{Context, Result, bail};

use crate::commitment::{
    hex_array, leaf_index, mixing_leaf, mixing_slot, residual_leaf, residual_slot, root_from_path,
    slots_per_position,
};
use crate::layer_tensors::f32_bytes;
use crate::node_fault::{BlameNode, ensure_node};
use crate::wire::{
    AnchorOpening, BoundaryLeaf, Challenge, HeadOpening, LayerOpening, MixingKind, Opening,
    RECEIPT_VERSION, Receipt, StageCommitment, StageOpening, TensorOpening, mixing_bytes,
    mixing_values,
};

mod canonical;
mod prefix;
mod verified;

use prefix::PrefixLeaves;

pub(crate) use canonical::check_mixing as check_mixing_order;
pub use verified::VerifiedOpening;

/// Checks the receipt against the claimed model: version, stages that
/// cover layers `0..layer_count` in order, each stage's leaf count, a
/// token for every position, and a prefix commitment per stage exactly
/// when the turn continues a cached prefix. `layer_count` comes from the
/// client's own planner for the claimed GGUF, never from the receipt.
///
/// But for the version, a receipt that fails this is the node's fault: it
/// signed a receipt that does not describe a turn of the claimed model.
pub fn verify_shape(receipt: &Receipt, layer_count: u32) -> Result<()> {
    if receipt.version != RECEIPT_VERSION {
        bail!(
            "receipt version {} is not {RECEIPT_VERSION}",
            receipt.version
        );
    }
    let span = receipt
        .position_end
        .checked_sub(receipt.position_start)
        .filter(|&span| span > 0)
        .context("the receipt covers no positions")
        .or_node_fault()?;
    // a sequence position is an i32 in the runtime
    ensure_node!(
        i32::try_from(receipt.position_end).is_ok(),
        "the receipt ends at position {}, past any sequence",
        receipt.position_end
    );
    ensure_node!(
        receipt.tokens.len() == span as usize,
        "the receipt holds {} tokens for {span} positions",
        receipt.tokens.len()
    );
    ensure_node!(
        (receipt.position_start..=receipt.position_end).contains(&receipt.generation_start),
        "generation starts at {}, outside positions {}..{}",
        receipt.generation_start,
        receipt.position_start,
        receipt.position_end
    );
    let mut next = 0;
    for (i, stage) in receipt.stages.iter().enumerate() {
        ensure_node!(
            stage.stage_index as usize == i && stage.layer_start == next && stage.layer_end > next,
            "stage {} covers layers {}..{}; the stages must cover 0..{layer_count} in order",
            stage.stage_index,
            stage.layer_start,
            stage.layer_end
        );
        let leaves = u64::from(span) * slots_per_position(stage.layer_start, stage.layer_end);
        ensure_node!(
            stage.leaf_count == leaves,
            "stage {} names {} leaves; its positions and layers make {leaves}",
            stage.stage_index,
            stage.leaf_count
        );
        verify_prefix(receipt, stage)?;
        next = stage.layer_end;
    }
    ensure_node!(
        next == layer_count,
        "the receipt covers layers 0..{next}; the claimed model has {layer_count}"
    );
    Ok(())
}

/// A stage commits the cells of a cached prefix exactly when the turn
/// starts after position 0, one per position below the turn. The stage
/// that holds layer 0 commits the prefix's tokens with them, and no other
/// stage does.
fn verify_prefix(receipt: &Receipt, stage: &StageCommitment) -> Result<()> {
    let committed = stage.prefix.as_ref().map(|prefix| prefix.positions);
    let expected = (receipt.position_start > 0).then_some(receipt.position_start);
    ensure_node!(
        committed == expected,
        "stage {} commits a cached prefix of {} positions; the turn starts at position {}",
        stage.stage_index,
        committed.map_or("no".to_owned(), |n| n.to_string()),
        receipt.position_start
    );
    if let Some(prefix) = &stage.prefix {
        ensure_node!(
            prefix.tokens.is_some() == (stage.layer_start == 0),
            "stage {} {} the tokens of the cached prefix; the stage that holds layer 0 commits them",
            stage.stage_index,
            if prefix.tokens.is_some() {
                "commits"
            } else {
                "does not commit"
            }
        );
    }
    Ok(())
}

/// Checks that `opening` has the one form an answer to `challenge` has,
/// and every opened leaf against its stage root. Run [`verify_shape`]
/// first: this relies on the stages and leaf counts it checks.
///
/// An error is the node's fault ([`crate::node_fault::NodeFault`]), unless
/// the challenge names another receipt.
pub fn verify_paths<'a>(
    receipt: &'a Receipt,
    challenge: &'a Challenge,
    opening: &'a Opening,
) -> Result<VerifiedOpening<'a>> {
    if challenge.receipt_id != receipt.receipt_id {
        bail!("the challenge names another receipt");
    }
    canonical::check(receipt, challenge, opening)?;
    verify_leaves(receipt, challenge, opening).or_node_fault()?;
    Ok(VerifiedOpening::new(receipt, challenge, opening))
}

/// Every leaf of a canonical opening against its stage root.
fn verify_leaves(receipt: &Receipt, challenge: &Challenge, opening: &Opening) -> Result<()> {
    let positions = challenge.positions();
    let mut trees = Vec::with_capacity(receipt.stages.len());
    for (stage, opened) in receipt.stages.iter().zip(&opening.stages) {
        prefix::verify_tokens(receipt, stage, opened)?;
        let tree = Tree::new(receipt, stage, opened)?;
        for layer in &opened.layers {
            verify_layer(&tree, opened, layer)?;
        }
        verify_anchors(&tree, opened)?;
        verify_head(&tree, &opened.head)?;
        trees.push(tree);
        for region in &opened.recurrent {
            let pick = challenge
                .attention
                .iter()
                .find(|p| p.position == region.position && p.layer == region.layer)
                .context("a canonical opening holds regions of attention picks alone")?;
            crate::recurrent::verify_region(receipt, pick, region)?;
        }
    }
    verify_boundaries(&trees, opening, &positions)
}

/// The residual leaving the final layer at every head position, which
/// must be one the turn sampled a token from. A position of the cached
/// prefix is never one: the turn sampled nothing there.
fn verify_head(tree: &Tree<'_>, head: &[HeadOpening]) -> Result<()> {
    if let Some(h) = head
        .iter()
        .find(|h| !tree.receipt.output_positions().contains(&h.position))
    {
        bail!("head position {} sampled no token", h.position);
    }
    let layer = tree.stage.layer_end;
    for h in head {
        let values = f32_bytes(&h.residual_values()?);
        tree.check(
            h.position,
            residual_slot(layer, tree.stage.layer_start),
            residual_leaf(h.position, layer, None, Some(&values)),
            &h.path,
            "output residual",
        )?;
    }
    Ok(())
}

/// Where a leaf sits in what a stage committed: in its tree of the turn,
/// or among the leaves of a picked position of the cached prefix.
struct Tree<'a> {
    receipt: &'a Receipt,
    stage: &'a StageCommitment,
    root: [u8; 32],
    /// By picked position of the prefix, the leaves the stage opened for
    /// it, checked against the prefix tree.
    prefix: std::collections::BTreeMap<u32, PrefixLeaves>,
}

impl<'a> Tree<'a> {
    /// The stage's commitment, with the positions of the prefix `opened`
    /// opens checked against it.
    fn new(
        receipt: &'a Receipt,
        stage: &'a StageCommitment,
        opened: &StageOpening,
    ) -> Result<Self> {
        Ok(Self {
            prefix: prefix::verify_positions(stage, &opened.prefix)?,
            ..Self::of_turn(receipt, stage)?
        })
    }

    /// The stage's tree of the turn alone.
    fn of_turn(receipt: &'a Receipt, stage: &'a StageCommitment) -> Result<Self> {
        let root = hex_array(
            &stage.root,
            &format!("the root of stage {}", stage.stage_index),
        )?;
        Ok(Self {
            receipt,
            stage,
            root,
            prefix: std::collections::BTreeMap::new(),
        })
    }

    /// The token at `position`: the receipt's for a position of the turn,
    /// else the one `opened`, the opening of the stage that holds layer 0,
    /// names for the prefix.
    fn token(&self, opened: &StageOpening, position: u32) -> Result<i32> {
        if position < self.receipt.position_start {
            return opened
                .prefix_tokens
                .get(position as usize)
                .copied()
                .context("no token for the position of the prefix");
        }
        self.receipt
            .token_at(position)
            .context("no token for the position")
    }

    fn check(
        &self,
        position: u32,
        slot: u64,
        leaf: [u8; 32],
        path: &[String],
        what: &str,
    ) -> Result<()> {
        if let Some(committed) = self.prefix.get(&position) {
            return committed.check(position, slot, leaf, path, what);
        }
        if position < self.receipt.position_start || position >= self.receipt.position_end {
            bail!("position {position} is outside the receipt");
        }
        let slots = slots_per_position(self.stage.layer_start, self.stage.layer_end);
        let index = leaf_index(position, slot, self.receipt.position_start, slots);
        if root_from_path(leaf, index, self.stage.leaf_count, path)? != self.root {
            bail!(
                "position {position}: the opened {what} does not match stage {}'s root",
                self.stage.stage_index
            );
        }
        Ok(())
    }
}

/// Checks one position's opened mixing values against the root of the
/// stage holding `layer`.
pub(crate) fn verify_mixing_leaf(
    receipt: &Receipt,
    layer: u32,
    position: u32,
    mixing: &[TensorOpening],
    path: &[String],
) -> Result<()> {
    let stage = receipt
        .stage_for_layer(layer)
        .with_context(|| format!("the receipt has no stage for layer {layer}"))?;
    let decoded = mixing_bytes(mixing)?;
    Tree::of_turn(receipt, stage)?.check(
        position,
        mixing_slot(layer, stage.layer_start),
        mixing_leaf(position, layer, &mixing_values(&decoded)),
        path,
        "mixing values",
    )
}

fn verify_layer(tree: &Tree<'_>, opened: &StageOpening, opening: &LayerOpening) -> Result<()> {
    let (receipt, stage) = (tree.receipt, tree.stage);
    let layer = opening.layer;
    // only the model's final layer keeps no row for positions it does not
    // output; any other layer that drops a row would skip its checks
    let final_layer = layer + 1 == receipt.layer_count();
    for leaf in &opening.leaves {
        let position = leaf.position;
        if leaf.residual_out.is_none() && !final_layer {
            bail!("position {position}: layer {layer} opened no residual out");
        }
        let stored = leaf
            .mixing
            .iter()
            .filter(|t| t.kind == MixingKind::Stored && t.values.is_some());
        for stored in stored {
            let name = crate::cache_types::name(stored.ggml_type);
            if !stage.cache_types.contains(&name) {
                bail!(
                    "position {position}: a stored row is {name}, which the receipt does not name"
                );
            }
        }
        let token = if layer == 0 {
            Some(tree.token(opened, position)?)
        } else {
            None
        };
        if let Some(committed) = tree.prefix.get(&position) {
            // the rows an attention check reads from this cell are the
            // ones this position's mixing leaf holds
            committed.check_stored(position, layer - stage.layer_start, &leaf.mixing)?;
        }
        let residual_in = f32_bytes(&leaf.residual_in_values()?);
        let residual_out = leaf.residual_out_values()?.map(|v| f32_bytes(&v));
        let decoded = mixing_bytes(&leaf.mixing)?;
        let start = stage.layer_start;
        tree.check(
            position,
            residual_slot(layer, start),
            residual_leaf(position, layer, token, Some(&residual_in)),
            &leaf.residual_in_path,
            "residual-in",
        )?;
        tree.check(
            position,
            mixing_slot(layer, start),
            mixing_leaf(position, layer, &mixing_values(&decoded)),
            &leaf.mixing_path,
            "mixing values",
        )?;
        tree.check(
            position,
            residual_slot(layer + 1, start),
            residual_leaf(position, layer + 1, None, residual_out.as_deref()),
            &leaf.residual_out_path,
            "residual-out",
        )?;
    }
    Ok(())
}

/// The input of layer 0 at every picked position, with the token the
/// receipt names for it. A stage that does not hold layer 0 opens none.
fn verify_anchors(tree: &Tree<'_>, opened: &StageOpening) -> Result<()> {
    let anchors: &[AnchorOpening] = &opened.anchors;
    for anchor in anchors {
        let token = tree.token(opened, anchor.position)?;
        if anchor.token != token {
            bail!(
                "position {}: the anchor holds token {}, the receipt {token}",
                anchor.position,
                anchor.token
            );
        }
        let values = f32_bytes(&anchor.residual_values()?);
        tree.check(
            anchor.position,
            residual_slot(0, 0),
            residual_leaf(anchor.position, 0, Some(token), Some(&values)),
            &anchor.path,
            "input of layer 0",
        )?;
    }
    Ok(())
}

/// Where stage `k` hands over to stage `k + 1`, both stages commit the same
/// residual: the leaf must be the same and in both trees. The leaf leaving
/// the final stage must be in its tree.
fn verify_boundaries(trees: &[Tree<'_>], opening: &Opening, positions: &[u32]) -> Result<()> {
    let stages: Vec<(&Tree<'_>, &StageOpening)> = trees.iter().zip(&opening.stages).collect();
    for pair in stages.windows(2) {
        let ((below_tree, sent), (above_tree, received)) = (pair[0], pair[1]);
        let (below, above) = (below_tree.stage, above_tree.stage);
        let layer = below.layer_end;
        for &position in positions {
            let find =
                |leaves: &[BoundaryLeaf]| leaves.iter().find(|b| b.position == position).cloned();
            let (Some(exit), Some(entry)) = (find(&sent.exit), find(&received.entry)) else {
                bail!("position {position}: the boundary at layer {layer} was not opened");
            };
            if exit.leaf != entry.leaf {
                bail!(
                    "position {position}: stage {} received a different residual at layer {layer} \
                     than stage {} sent",
                    above.stage_index,
                    below.stage_index
                );
            }
            let leaf = hex_array(&exit.leaf, "a boundary leaf")?;
            below_tree.check(
                position,
                residual_slot(layer, below.layer_start),
                leaf,
                &exit.path,
                "residual leaving the stage",
            )?;
            above_tree.check(
                position,
                residual_slot(layer, above.layer_start),
                leaf,
                &entry.path,
                "residual entering the stage",
            )?;
        }
    }
    if let Some((tree, opened)) = stages.last() {
        let stage = tree.stage;
        for exit in &opened.exit {
            tree.check(
                exit.position,
                residual_slot(stage.layer_end, stage.layer_start),
                hex_array(&exit.leaf, "a boundary leaf")?,
                &exit.path,
                "residual leaving the stage",
            )?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
