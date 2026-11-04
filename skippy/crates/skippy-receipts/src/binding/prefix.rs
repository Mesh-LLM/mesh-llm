//! What a stage opens of a turn's cached prefix, against the receipt.
//!
//! A turn that continued a cached prefix computed nothing below its first
//! position, so its stage trees hold no leaf of the prefix. Each stage
//! commits the prefix apart: a tree with one leaf per position, which binds
//! the position's KV cell, the digest of the rows the cell stores, and the
//! digest of every leaf the stage committed when it computed the position
//! ([`crate::commitment::prefix_leaf`]).
//!
//! For a picked position of the prefix a stage opens those leaves, by hash,
//! and the leaves of the rows each of its layers stored in the cell
//! ([`PrefixOpening`]). [`verify_positions`] checks them against the
//! prefix root. A value the node opens at that position is then checked
//! against the leaf it must hash to, as a value of the turn is checked
//! against the stage tree. The stored rows a mixing leaf holds must be the
//! ones the cell's digest covers, because an attention check of a later
//! position reads the cell's rows through that digest.
//!
//! The stage that holds layer 0 opens the tokens of the whole prefix, which
//! must hash to the digest it committed ([`verify_tokens`]). The layer-0
//! leaf of a picked position binds its token, so a token the node did not
//! compute the position with fails there.
//!
//! Every failure here is the node's fault.

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};

use crate::attention::opening::stored_digests;
use crate::commitment::{
    Hash, cell_digest, hex_array, position_digest, prefix_leaf, prefix_tokens_digest,
    root_from_path, slots_per_position, stored_leaf,
};
use crate::wire::{
    MixingKind, PrefixOpening, Receipt, StageCommitment, StageOpening, TensorOpening,
};

/// The leaves a stage committed at one picked position of the prefix,
/// checked against the prefix root.
pub(super) struct PrefixLeaves {
    stage_index: u32,
    /// The first layer the stage holds.
    first_layer: u32,
    /// The position's leaves, in slot order.
    leaves: Vec<Hash>,
    /// The leaf of the rows each layer of the stage stored in the cell.
    stored: Vec<Hash>,
}

impl PrefixLeaves {
    /// Checks a leaf the node opened at the position against the one the
    /// stage committed in `slot`. Such a leaf has no path: it is in no
    /// tree.
    pub(super) fn check(
        &self,
        position: u32,
        slot: u64,
        leaf: Hash,
        path: &[String],
        what: &str,
    ) -> Result<()> {
        if !path.is_empty() {
            bail!("position {position}: the {what} of a prefix position was opened with a path");
        }
        let committed = usize::try_from(slot)
            .ok()
            .and_then(|slot| self.leaves.get(slot));
        if committed != Some(&leaf) {
            bail!(
                "position {position}: the opened {what} does not match what stage {} committed \
                 for the cached prefix",
                self.stage_index
            );
        }
        Ok(())
    }

    /// Checks that the stored rows of an opened mixing leaf of the stage's
    /// layer `index` are the rows the cell's digest covers.
    pub(super) fn check_stored(
        &self,
        position: u32,
        index: u32,
        mixing: &[TensorOpening],
    ) -> Result<()> {
        let layer = self.first_layer + index;
        let stored: Vec<TensorOpening> = mixing
            .iter()
            .filter(|value| value.kind == MixingKind::Stored)
            .cloned()
            .collect();
        let leaf = stored_leaf(position, layer, &stored_digests(&stored)?);
        if self.stored.get(index as usize) != Some(&leaf) {
            bail!(
                "position {position}: the rows layer {layer} stored are not the rows of the \
                 cell the prefix commits"
            );
        }
        Ok(())
    }
}

/// Checks every position of the prefix `opened` holds against the prefix
/// tree `stage` commits, and returns their leaves by position.
pub(super) fn verify_positions(
    stage: &StageCommitment,
    opened: &[PrefixOpening],
) -> Result<BTreeMap<u32, PrefixLeaves>> {
    let Some(prefix) = &stage.prefix else {
        if !opened.is_empty() {
            bail!(
                "stage {} opens positions of a prefix it did not commit",
                stage.stage_index
            );
        }
        return Ok(BTreeMap::new());
    };
    let root: Hash = hex_array(&prefix.root, "the prefix root")?;
    let slots = slots_per_position(stage.layer_start, stage.layer_end);
    let layers = stage.layer_end - stage.layer_start;
    let mut verified = BTreeMap::new();
    for position in opened {
        let at = position.position;
        let hashes = |list: &[String], what: &str| -> Result<Vec<Hash>> {
            list.iter().map(|hash| hex_array(hash, what)).collect()
        };
        if position.leaves.len() as u64 != slots || position.stored.len() != layers as usize {
            bail!(
                "position {at}: stage {} opens {} leaves and {} stored leaves; its layers make \
                 {slots} and {layers}",
                stage.stage_index,
                position.leaves.len(),
                position.stored.len()
            );
        }
        let leaves = hashes(&position.leaves, "a leaf of a prefix position")?;
        let stored = hashes(&position.stored, "a stored leaf of a prefix cell")?;
        let leaf = prefix_leaf(
            i32::try_from(at).context("a prefix position past any sequence")?,
            position.cell,
            &cell_digest(stored.clone())?,
            &position_digest(&leaves),
        );
        let committed = root_from_path(
            leaf,
            u64::from(at),
            u64::from(prefix.positions),
            &position.path,
        )
        .with_context(|| format!("position {at} of the prefix"))?;
        if committed != root {
            bail!(
                "position {at}: the opened leaves do not match stage {}'s prefix root",
                stage.stage_index
            );
        }
        verified.insert(
            at,
            PrefixLeaves {
                stage_index: stage.stage_index,
                first_layer: stage.layer_start,
                leaves,
                stored,
            },
        );
    }
    Ok(verified)
}

/// Checks the tokens of the prefix a stage opened: the stage that holds
/// layer 0 opens one per position of the prefix, which hash to the digest
/// it committed, and no other stage opens any.
pub(super) fn verify_tokens(
    receipt: &Receipt,
    stage: &StageCommitment,
    opened: &StageOpening,
) -> Result<()> {
    let committed = stage
        .prefix
        .as_ref()
        .and_then(|prefix| prefix.tokens.as_deref());
    let Some(committed) = committed else {
        if !opened.prefix_tokens.is_empty() {
            bail!(
                "stage {} opens tokens of a prefix it committed none for",
                stage.stage_index
            );
        }
        return Ok(());
    };
    if opened.prefix_tokens.len() != receipt.position_start as usize {
        bail!(
            "stage {} opens {} tokens of the prefix; the turn starts at position {}",
            stage.stage_index,
            opened.prefix_tokens.len(),
            receipt.position_start
        );
    }
    if hex::encode(prefix_tokens_digest(&opened.prefix_tokens)) != committed {
        bail!(
            "the opened tokens of the prefix do not match what stage {} committed",
            stage.stage_index
        );
    }
    Ok(())
}

#[cfg(test)]
#[path = "prefix_tests.rs"]
mod tests;
