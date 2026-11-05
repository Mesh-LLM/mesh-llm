//! Client-side checks that bind a receipt to the model and the request,
//! and an opening to the receipt.
//!
//! - **Shape:** the receipt's stages cover every layer of the claimed
//!   model, from 0, without gaps; each stage names the leaf count its
//!   positions and layers make; it holds a token for every position.
//! - **Paths:** every opened leaf hashes to its stage root along a path of
//!   the shape that tree has. This covers each picked layer, the input of
//!   layer 0 with its token (the anchor), and the residual leaves at stage
//!   boundaries, which must be equal on both sides, and the residual the
//!   output head reads at every head position.

use anyhow::{Context, Result, bail};

use crate::commitment::{
    hex_array, leaf_index, mixing_leaf, mixing_slot, residual_leaf, residual_slot, root_from_path,
    slots_per_position,
};
use crate::layer_tensors::f32_bytes;
use crate::wire::{
    BoundaryLeaf, Challenge, LayerOpening, MixingKind, Opening, RECEIPT_VERSION, Receipt,
    StageCommitment, TensorOpening, mixing_bytes, mixing_values,
};

/// Checks the receipt against the claimed model: version, stages that
/// cover layers `0..layer_count` in order, each stage's leaf count, and a
/// token for every position. `layer_count` comes from the client's own
/// planner for the claimed GGUF, never from the receipt.
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
        .context("the receipt covers no positions")?;
    if receipt.tokens.len() != span as usize {
        bail!(
            "the receipt holds {} tokens for {span} positions",
            receipt.tokens.len()
        );
    }
    if !(receipt.position_start..=receipt.position_end).contains(&receipt.generation_start) {
        bail!(
            "generation starts at {}, outside positions {}..{}",
            receipt.generation_start,
            receipt.position_start,
            receipt.position_end
        );
    }
    let mut next = 0;
    for (i, stage) in receipt.stages.iter().enumerate() {
        if stage.stage_index as usize != i || stage.layer_start != next || stage.layer_end <= next {
            bail!(
                "stage {} covers layers {}..{}; the stages must cover 0..{layer_count} in order",
                stage.stage_index,
                stage.layer_start,
                stage.layer_end
            );
        }
        let leaves = u64::from(span) * slots_per_position(stage.layer_start, stage.layer_end);
        if stage.leaf_count != leaves {
            bail!(
                "stage {} names {} leaves; its positions and layers make {leaves}",
                stage.stage_index,
                stage.leaf_count
            );
        }
        next = stage.layer_end;
    }
    if next != layer_count {
        bail!("the receipt covers layers 0..{next}; the claimed model has {layer_count}");
    }
    Ok(())
}

/// Checks every opened leaf against its stage root. Run [`verify_shape`]
/// first: this relies on the stages and leaf counts it checks.
pub fn verify_paths(receipt: &Receipt, challenge: &Challenge, opening: &Opening) -> Result<()> {
    if opening.receipt_id != receipt.receipt_id || challenge.receipt_id != receipt.receipt_id {
        bail!("the opening or challenge names another receipt");
    }
    let positions = challenge.positions();
    for pick in &challenge.picks {
        let stage = receipt
            .stage_for_layer(pick.layer)
            .with_context(|| format!("the receipt has no stage for layer {}", pick.layer))?;
        let layer = opening
            .stage(stage.stage_index)
            .and_then(|s| s.layers.iter().find(|l| l.layer == pick.layer))
            .with_context(|| format!("layer {} was not opened", pick.layer))?;
        let opened: Vec<u32> = layer.leaves.iter().map(|l| l.position).collect();
        if opened != pick.positions {
            bail!(
                "layer {} opens positions {opened:?}, the challenge asked {:?}",
                pick.layer,
                pick.positions
            );
        }
        verify_layer(receipt, stage, layer)?;
    }
    verify_anchors(receipt, opening, &positions)?;
    verify_boundaries(receipt, opening, &positions)?;
    verify_head(receipt, challenge, opening)
}

/// The final stage opens the residual leaving the final layer at every
/// head position, which must be one the turn sampled a token from.
fn verify_head(receipt: &Receipt, challenge: &Challenge, opening: &Opening) -> Result<()> {
    if challenge.head.is_empty() {
        return Ok(());
    }
    if let Some(position) = challenge
        .head
        .iter()
        .find(|p| !receipt.output_positions().contains(p))
    {
        bail!("head position {position} sampled no token");
    }
    let stage = receipt.stages.last().context("the receipt has no stages")?;
    let head = &opening
        .stage(stage.stage_index)
        .context("the final stage opened nothing")?
        .head;
    let opened: Vec<u32> = head.iter().map(|h| h.position).collect();
    if opened != challenge.head {
        bail!(
            "the output residual is opened at {opened:?}, the challenge asked {:?}",
            challenge.head
        );
    }
    let tree = Tree::new(receipt, stage)?;
    let layer = stage.layer_end;
    for h in head {
        let values = f32_bytes(&h.residual_values()?);
        tree.check(
            h.position,
            residual_slot(layer, stage.layer_start),
            residual_leaf(h.position, layer, None, Some(&values)),
            &h.path,
            "output residual",
        )?;
    }
    Ok(())
}

/// Where a leaf sits in its stage tree.
struct Tree<'a> {
    receipt: &'a Receipt,
    stage: &'a StageCommitment,
    root: [u8; 32],
}

impl<'a> Tree<'a> {
    fn new(receipt: &'a Receipt, stage: &'a StageCommitment) -> Result<Self> {
        let root = hex_array(
            &stage.root,
            &format!("the root of stage {}", stage.stage_index),
        )?;
        Ok(Self {
            receipt,
            stage,
            root,
        })
    }

    fn check(
        &self,
        position: u32,
        slot: u64,
        leaf: [u8; 32],
        path: &[String],
        what: &str,
    ) -> Result<()> {
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
    Tree::new(receipt, stage)?.check(
        position,
        mixing_slot(layer, stage.layer_start),
        mixing_leaf(position, layer, &mixing_values(&decoded)),
        path,
        "mixing values",
    )
}

fn verify_layer(receipt: &Receipt, stage: &StageCommitment, opening: &LayerOpening) -> Result<()> {
    let layer = opening.layer;
    let tree = Tree::new(receipt, stage)?;
    // only the model's final layer keeps no row for positions it does not
    // output; any other layer that drops a row would skip its checks
    let final_layer = layer + 1 == receipt.layer_count();
    for leaf in &opening.leaves {
        let position = leaf.position;
        if leaf.residual_out.is_none() && !final_layer {
            bail!("position {position}: layer {layer} opened no residual out");
        }
        for stored in leaf.mixing.iter().filter(|t| t.kind == MixingKind::Stored) {
            let name = crate::weights::type_name(stored.ggml_type as u32);
            if !stage.cache_types.contains(&name) {
                bail!(
                    "position {position}: a stored row is {name}, which the receipt does not name"
                );
            }
        }
        let token = if layer == 0 {
            Some(
                receipt
                    .token_at(position)
                    .context("no token for the position")?,
            )
        } else {
            None
        };
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

/// The input of layer 0 is opened at every picked position, with the token
/// the receipt names for it.
fn verify_anchors(receipt: &Receipt, opening: &Opening, positions: &[u32]) -> Result<()> {
    let stage = receipt
        .stages
        .first()
        .context("the receipt has no stages")?;
    let anchors = &opening
        .stage(stage.stage_index)
        .context("the stage holding layer 0 opened nothing")?
        .anchors;
    let opened: Vec<u32> = anchors.iter().map(|a| a.position).collect();
    if opened != positions {
        bail!("the input of layer 0 is opened at {opened:?}, the challenge picked {positions:?}");
    }
    let tree = Tree::new(receipt, stage)?;
    for anchor in anchors {
        let token = receipt
            .token_at(anchor.position)
            .context("no token for the position")?;
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
/// residual: the leaf must be the same and in both trees.
fn verify_boundaries(receipt: &Receipt, opening: &Opening, positions: &[u32]) -> Result<()> {
    for pair in receipt.stages.windows(2) {
        let (below, above) = (&pair[0], &pair[1]);
        let exit = &opening
            .stage(below.stage_index)
            .with_context(|| format!("stage {} opened nothing", below.stage_index))?
            .exit;
        let entry = &opening
            .stage(above.stage_index)
            .with_context(|| format!("stage {} opened nothing", above.stage_index))?
            .entry;
        let (below_tree, above_tree) = (Tree::new(receipt, below)?, Tree::new(receipt, above)?);
        let layer = below.layer_end;
        for &position in positions {
            let find =
                |leaves: &[BoundaryLeaf]| leaves.iter().find(|b| b.position == position).cloned();
            let (Some(exit), Some(entry)) = (find(exit), find(entry)) else {
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
    Ok(())
}
