//! An opening whose form and Merkle paths were checked.

use crate::attention::opening::{AttentionPartials, AttentionPick};
use crate::wire::{
    AnchorOpening, Challenge, HeadOpening, LayerOpening, LeafOpening, Opening, Receipt,
    RecurrentOpening, StageCommitment, StageOpening,
};

/// An opening that [`super::verify_paths`] accepted for a receipt and a
/// challenge. Only that function makes one.
///
/// The opening is canonical, so it holds nothing but what the challenge
/// asks, and every leaf in it hashes to its stage root. A lookup here
/// therefore finds the one entry the checks ran on. What the node
/// recomputed by replay and no leaf commits (the matmuls, selections,
/// logits and attention partials) is here too; the other checks judge
/// those against the committed values.
#[derive(Clone, Copy, Debug)]
pub struct VerifiedOpening<'a> {
    receipt: &'a Receipt,
    challenge: &'a Challenge,
    opening: &'a Opening,
}

impl<'a> VerifiedOpening<'a> {
    pub(super) fn new(
        receipt: &'a Receipt,
        challenge: &'a Challenge,
        opening: &'a Opening,
    ) -> Self {
        Self {
            receipt,
            challenge,
            opening,
        }
    }

    pub fn receipt(&self) -> &'a Receipt {
        self.receipt
    }

    pub fn challenge(&self) -> &'a Challenge {
        self.challenge
    }

    /// The whole opening.
    pub fn opening(&self) -> &'a Opening {
        self.opening
    }

    /// Each stage of the receipt with what it opened, in order.
    pub fn stages(
        &self,
    ) -> impl Iterator<Item = (&'a StageCommitment, &'a StageOpening)> + use<'a> {
        self.receipt.stages.iter().zip(&self.opening.stages)
    }

    /// The stage holding `layer`, with what it opened.
    fn stage_of(&self, layer: u32) -> Option<(&'a StageCommitment, &'a StageOpening)> {
        self.stages()
            .find(|(stage, _)| (stage.layer_start..stage.layer_end).contains(&layer))
    }

    /// The opening of a picked layer, with the backend of its stage.
    pub fn layer(&self, layer: u32) -> Option<(&'a LayerOpening, &'a str)> {
        let (stage, opened) = self.stage_of(layer)?;
        let layer = opened.layers.iter().find(|l| l.layer == layer)?;
        Some((layer, stage.backend.as_str()))
    }

    /// One picked position of a picked layer.
    pub fn leaf(&self, layer: u32, position: u32) -> Option<&'a LeafOpening> {
        let (layer, _) = self.layer(layer)?;
        layer.leaves.iter().find(|leaf| leaf.position == position)
    }

    /// The tokens of the turn's cached prefix, from position 0, as the
    /// stage that holds layer 0 opened them: they hash to the digest the
    /// receipt commits. Empty for a turn that starts at position 0.
    pub fn prefix_tokens(&self) -> &'a [i32] {
        self.opening
            .stages
            .first()
            .map_or(&[], |stage| stage.prefix_tokens.as_slice())
    }

    /// The input of layer 0 at every picked position.
    pub fn anchors(&self) -> &'a [AnchorOpening] {
        self.opening
            .stages
            .first()
            .map_or(&[], |stage| stage.anchors.as_slice())
    }

    /// The final stage's backend and the residual the output head reads at
    /// every head position of the challenge.
    pub fn head(&self) -> Option<(&'a str, &'a [HeadOpening])> {
        let (stage, opened) = self.stages().last()?;
        Some((stage.backend.as_str(), opened.head.as_slice()))
    }

    /// The partials the node opened for an attention pick, if it opened
    /// any: a stage opens none in a layer without attention.
    pub fn partials(&self, pick: &AttentionPick) -> Option<&'a AttentionPartials> {
        let (_, opened) = self.stage_of(pick.layer)?;
        opened
            .attention
            .iter()
            .find(|a| a.position == pick.position && a.layer == pick.layer)
    }

    /// The recurrent region `stage` opened, if it opened one.
    pub fn region(&self, stage: &StageCommitment) -> Option<&'a RecurrentOpening> {
        let (_, opened) = self
            .stages()
            .find(|(s, _)| s.stage_index == stage.stage_index)?;
        opened.recurrent.first()
    }
}
