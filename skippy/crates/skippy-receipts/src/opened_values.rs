//! Every opened value against the shapes of the claimed model, before any
//! check reads it.
//!
//! An opening is a node's free text until it is checked. A value that is
//! not a number passes a comparison written with `>`; a column of the wrong
//! length stops a check with an error instead of a verdict; an id out of
//! range indexes past a tensor. So nothing is compared, read from the
//! claimed GGUF or given to the native evaluator before this has passed:
//!
//! - every f32 array is whole and holds only numbers: residuals, matmul
//!   inputs and outputs, mixing values, logits and attention partials;
//! - a residual has the model's width, a matmul's input and output fit its
//!   weight's columns and rows, and logits count the vocabulary;
//! - a matmul names a tensor of the claimed GGUF, and an expert or selected
//!   id is one its weights hold;
//! - a stored row is of a type the auditor checks, in whole blocks.
//!
//! The shapes come from the GGUF's header alone, so this reads no weights
//! and needs no evaluator. Every failure is the node's fault
//! ([`crate::node_fault::NodeFault`]). What is left to fail afterwards is
//! the client's own: a weight read, a layer that does not load.

use std::ops::Deref;

use anyhow::{Context, Result, bail};

use crate::audit::check_matmul_shape;
use crate::binding::VerifiedOpening;
use crate::claimed_reads::TOKEN_EMBEDDING;
use crate::error_bound::all_finite;
use crate::head::head_weight;
use crate::node_fault::{BlameNode, ensure_node, node_fault};
use crate::weights::{MetaValue, WeightFile};
use crate::wire::{
    AnchorOpening, HeadOpening, LeafOpening, MatrixOpening, MixingKind, Receipt, StageOpening,
    TensorOpening,
};

/// The shapes of the claimed model an opening must fit, from its GGUF's
/// header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelShapes {
    /// Values of the residual stream at one position.
    pub hidden: usize,
    /// Logits the output head gives: one per token.
    pub vocabulary: u64,
    /// Tokens the model embeds, when it has a token embedding.
    pub tokens: Option<u64>,
}

impl ModelShapes {
    /// Fails when the header does not name them, which says nothing about
    /// a node.
    pub fn of(weights: &WeightFile) -> Result<Self> {
        let architecture = weights.metadata_str("general.architecture")?;
        let key = format!("{architecture}.embedding_length");
        let hidden = match weights.metadata.get(&key) {
            Some(MetaValue::Int(hidden)) if *hidden > 0 => usize::try_from(*hidden)?,
            _ => bail!("the claimed GGUF names no {key}"),
        };
        Ok(Self {
            hidden,
            vocabulary: weights.tensor(head_weight(weights))?.rows(),
            tokens: weights.tensors.get(TOKEN_EMBEDDING).map(|t| t.rows()),
        })
    }
}

/// Checks that every token a receipt names is one the claimed model
/// embeds. A receipt is signed, so a token that is not is the node's fault.
pub fn check_tokens(receipt: &Receipt, shapes: &ModelShapes) -> Result<()> {
    let Some(tokens) = shapes.tokens else {
        return Ok(());
    };
    let held = |token: i32| u64::try_from(token).is_ok_and(|t| t < tokens);
    if let Some(token) = receipt
        .tokens
        .iter()
        .chain(&receipt.final_token)
        .find(|&&token| !held(token))
    {
        node_fault!("the receipt names token {token}; the claimed model embeds {tokens}");
    }
    Ok(())
}

/// A verified opening whose every value fits the claimed model
/// ([`validate`]). The checks of an audit take this.
#[derive(Clone, Copy, Debug)]
pub struct ValidatedOpening<'a>(VerifiedOpening<'a>);

impl<'a> Deref for ValidatedOpening<'a> {
    type Target = VerifiedOpening<'a>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Checks every value of `opening` against the claimed model's shapes.
///
/// An error that blames the node is a defect of the opening. Any other is
/// a header that does not name the model's shapes.
pub fn validate<'a>(
    opening: VerifiedOpening<'a>,
    weights: &WeightFile,
) -> Result<ValidatedOpening<'a>> {
    let shapes = ModelShapes::of(weights)?;
    for (stage, opened) in opening.stages() {
        check_stage(opened, weights, &shapes)
            .with_context(|| format!("stage {}", stage.stage_index))
            .or_node_fault()?;
    }
    Ok(ValidatedOpening(opening))
}

fn check_stage(opened: &StageOpening, weights: &WeightFile, shapes: &ModelShapes) -> Result<()> {
    // the tokens of a cached prefix are the node's, like the receipt's
    if let Some(tokens) = shapes.tokens
        && let Some(token) = opened
            .prefix_tokens
            .iter()
            .find(|&&token| !u64::try_from(token).is_ok_and(|t| t < tokens))
    {
        node_fault!("the prefix names token {token}; the claimed model embeds {tokens}");
    }
    for layer in &opened.layers {
        for leaf in &layer.leaves {
            check_leaf(leaf, weights, shapes)
                .with_context(|| format!("layer {}, position {}", layer.layer, leaf.position))?;
        }
    }
    for anchor in &opened.anchors {
        check_anchor(anchor, shapes).with_context(|| format!("position {}", anchor.position))?;
    }
    for head in &opened.head {
        check_head(head, shapes).with_context(|| format!("position {}", head.position))?;
    }
    for partials in &opened.attention {
        let values = partials.values()?;
        ensure_node!(
            all_finite(&values),
            "position {}, layer {}: an attention partial is not a number",
            partials.position,
            partials.layer
        );
    }
    for region in &opened.recurrent {
        for leaf in &region.leaves {
            check_mixing(&leaf.mixing).with_context(|| {
                format!(
                    "recurrent region of layer {}, position {}",
                    region.layer, leaf.position
                )
            })?;
        }
    }
    Ok(())
}

/// A residual of the model's width, of numbers alone.
fn check_residual(what: &str, values: &[f32], shapes: &ModelShapes) -> Result<()> {
    ensure_node!(
        values.len() == shapes.hidden,
        "{what} has {} values; the model's residual stream has {}",
        values.len(),
        shapes.hidden
    );
    ensure_node!(
        all_finite(values),
        "{what} holds a value that is not a number"
    );
    Ok(())
}

fn check_leaf(leaf: &LeafOpening, weights: &WeightFile, shapes: &ModelShapes) -> Result<()> {
    check_residual("the residual in", &leaf.residual_in_values()?, shapes)?;
    if let Some(out) = leaf.residual_out_values()? {
        check_residual("the residual out", &out, shapes)?;
    }
    check_mixing(&leaf.mixing)?;
    // the experts a selection can pick are the ones the layer's expert
    // weights hold
    let mut experts: Option<u64> = None;
    for m in &leaf.matrices {
        let held = check_matrix(m, weights)?;
        if m.experts.is_some() {
            experts = Some(experts.map_or(held, |least| least.min(held)));
        }
    }
    for selection in &leaf.selections {
        let in_range =
            |id: i32| u64::try_from(id).is_ok_and(|id| experts.is_none_or(|held| id < held));
        if let Some(id) = selection.ids.iter().find(|&&id| !in_range(id)) {
            node_fault!(
                "selection {} picks id {id}, outside the layer's {} experts",
                selection.index,
                experts.map_or("non-negative".to_owned(), |held| held.to_string())
            );
        }
    }
    Ok(())
}

/// A mixing leaf's values: a stored row in a type the auditor checks, in
/// whole blocks; every other value f32. All of numbers alone.
fn check_mixing(mixing: &[TensorOpening]) -> Result<()> {
    for value in mixing {
        let name = value.kind.tensor_kind().name();
        let Some(bytes) = value.bytes()? else {
            continue;
        };
        if value.kind == MixingKind::Stored {
            crate::cache_types::decode_row(value.ggml_type, &bytes)
                .with_context(|| format!("{name} {}", value.index))?;
            continue;
        }
        ensure_node!(
            value.ggml_type == crate::layer_tensors::GGML_F32 && bytes.len().is_multiple_of(4),
            "{name} {} is not whole f32 values",
            value.index
        );
        ensure_node!(
            all_finite(&crate::layer_tensors::decode_f32_bytes(&bytes)),
            "{name} {} holds a value that is not a number",
            value.index
        );
    }
    Ok(())
}

/// An opened matmul against the weight it names. Returns the matrices the
/// weight holds: its experts, or 1.
fn check_matrix(m: &MatrixOpening, weights: &WeightFile) -> Result<u64> {
    let Some(tensor) = weights.tensors.get(&m.weight) else {
        node_fault!(
            "matmul {} names {:?}, a tensor the claimed GGUF does not have",
            m.index,
            m.weight
        );
    };
    let (input, output) = (m.input_values()?, m.output_values()?);
    check_matmul_shape(tensor, m, input.len(), output.len())?;
    ensure_node!(
        all_finite(&input) && all_finite(&output),
        "{} is opened with a value that is not a number",
        m.weight
    );
    Ok(tensor.dims.get(2).copied().unwrap_or(1))
}

fn check_anchor(anchor: &AnchorOpening, shapes: &ModelShapes) -> Result<()> {
    check_residual("the input of layer 0", &anchor.residual_values()?, shapes)
}

fn check_head(head: &HeadOpening, shapes: &ModelShapes) -> Result<()> {
    check_residual("the output residual", &head.residual_values()?, shapes)?;
    let Some(logits) = head.logit_values()? else {
        return Ok(());
    };
    ensure_node!(
        logits.len() as u64 == shapes.vocabulary,
        "{} logits are opened for {} tokens",
        logits.len(),
        shapes.vocabulary
    );
    ensure_node!(all_finite(&logits), "a logit is not a number");
    Ok(())
}

#[cfg(test)]
mod tests;
