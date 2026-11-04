//! Bounds the work a challenge asks of a stage, before the stage replays
//! anything.
//!
//! The stage that issued a receipt checks a challenge against the receipt
//! ([`crate::challenge::check_shape`]). A later stage holds only its own
//! commitment of the turn, so it checks what it can see: that the challenge
//! asks for no more layers, positions, head positions and attention picks
//! than a draw makes, all at positions of the turn it committed or of its
//! cached prefix. A stage then replays at most [`LAYERS_PER_CHALLENGE`]
//! layers at [`crate::challenge::POSITIONS_PER_LAYER`] positions of the turn
//! each, and the ubatches of at most [`crate::challenge::PREFIX_POSITIONS`]
//! positions of the prefix, whoever sends the challenge.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Result, bail};

use super::Recorder;
use crate::challenge::{
    ATTENTION_POSITIONS_PER_LAYER, HEAD_POSITIONS, LAYERS_PER_CHALLENGE, check_picked_positions,
};
use crate::wire::Challenge;

impl Recorder {
    /// Refuses a challenge that asks this stage for more than a draw does,
    /// or for positions outside the turn it committed. A turn the stage no
    /// longer holds is a [`crate::wire::CannotOpen`].
    pub fn check_challenge(&self, challenge: &Challenge) -> Result<()> {
        let turn = {
            let state = self.lock();
            let turn = self.opening_turn(&state.committed, &challenge.receipt_id)?;
            turn.position_start..turn.position_end
        };
        if challenge.picks.len() > LAYERS_PER_CHALLENGE {
            bail!(
                "the challenge picks {} layers; a draw picks at most {LAYERS_PER_CHALLENGE}",
                challenge.picks.len()
            );
        }
        let increasing = |values: &[u32]| values.windows(2).all(|pair| pair[0] < pair[1]);
        let in_turn = |positions: &[u32]| positions.iter().all(|p| turn.contains(p));
        let layers: Vec<u32> = challenge.picks.iter().map(|p| p.layer).collect();
        if !increasing(&layers) {
            bail!("the challenge's layers are not in increasing order");
        }
        if self.holds_output
            && let Some(layer) = layers.iter().find(|&&layer| layer >= self.layers.end)
        {
            bail!("no stage holds layer {layer}");
        }
        check_picked_positions(&challenge.picks, &turn)?;
        if challenge.head.len() > HEAD_POSITIONS
            || !increasing(&challenge.head)
            || !in_turn(&challenge.head)
        {
            bail!(
                "the head positions are not at most {HEAD_POSITIONS} increasing positions of the turn"
            );
        }
        let mut seen = BTreeSet::new();
        let mut per_layer: BTreeMap<u32, usize> = BTreeMap::new();
        for pick in &challenge.attention {
            let picked = challenge
                .picks
                .iter()
                .any(|p| p.layer == pick.layer && p.positions.contains(&pick.position));
            if !picked || !seen.insert((pick.layer, pick.position)) {
                bail!(
                    "attention pick at position {} of layer {} is not one picked position, once",
                    pick.position,
                    pick.layer
                );
            }
            let count = per_layer.entry(pick.layer).or_default();
            *count += 1;
            if *count > ATTENTION_POSITIONS_PER_LAYER {
                bail!(
                    "layer {} has more than {ATTENTION_POSITIONS_PER_LAYER} attention picks",
                    pick.layer
                );
            }
        }
        Ok(())
    }
}
