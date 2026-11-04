//! Client side of the recurrent check, for layers whose token mixing reads
//! a recurrent state (the gated delta net, Mamba's scan, a short
//! convolution). The stage's side is in `recorder::recurrent_opening`.
//!
//! The glue check takes such a layer's mixing outputs as opened, as it does
//! an attention layer's. Here the auditor recomputes them: for the first
//! attention pick of a stage on such a layer, the stage opens the mixing
//! leaves of the ubatch that computed the pick, and of the position after
//! it. The auditor runs the layer's own token-mixing ops, with its
//! llama.cpp, on the opened inputs and the state the ubatch read, and
//! compares the outputs at every position and the state the ops leave with
//! the one the next ubatch read. The comparisons chain: each ubatch's state
//! is the one the ubatch before it left, and the first one is zero.
//!
//! Weights inside the token mixing (a short convolution's kernel, a decay)
//! come from the auditor's GGUF, so a node that ran other weights there
//! fails too. The state a turn started from, when an earlier turn left it,
//! is only bound by the receipt's digest of the exported state.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};

use crate::attention::opening::AttentionPick;
use crate::binding::verify_mixing_leaf;
use crate::glue::{
    Computed, GlueEvaluator, GlueInput, GlueRequest, GlueValues, GlueVerdict, Verdicts,
    blame_refused_values, relative_error,
};
use crate::layer_tensors::TensorKind;
use crate::node_fault::BlameNode;
use crate::wire::{Challenge, MixingKind, MixingLeafOpening, Receipt, RecurrentOpening};

/// A recurrent comparison fails above this error, relative to the largest
/// value of the auditor's vector. The ops sum over every token of the
/// ubatch, and a backend may order the sums differently (in chunks), so
/// this is looser than the glue's.
pub(crate) const RECURRENT_TOLERANCE: f64 = 1e-3;

/// The attention pick of `challenge` a stage holding `layers` opens the
/// region of: its first on a layer in `state_layers`.
pub fn region_pick<'a>(
    challenge: &'a Challenge,
    layers: &Range<u32>,
    state_layers: &BTreeSet<u32>,
) -> Option<&'a AttentionPick> {
    challenge
        .attention
        .iter()
        .find(|p| layers.contains(&p.layer) && state_layers.contains(&p.layer))
}

/// Checks that `opening` is the region of `pick`, and every leaf against
/// the receipt: the ubatch lies in the turn and holds the pick, the leaves
/// run from its first position to the one after it (unless the turn ends
/// there), and a state is opened at both ends and nowhere between.
///
/// An error is the node's fault ([`crate::node_fault::NodeFault`]).
pub(crate) fn verify_region(
    receipt: &Receipt,
    pick: &AttentionPick,
    opening: &RecurrentOpening,
) -> Result<()> {
    region_leaves(receipt, pick, opening).or_node_fault()
}

fn region_leaves(
    receipt: &Receipt,
    pick: &AttentionPick,
    opening: &RecurrentOpening,
) -> Result<()> {
    let RecurrentOpening {
        position,
        layer,
        start,
        end,
        ..
    } = *opening;
    ensure!(
        position == pick.position && layer == pick.layer,
        "the region opens position {position}, layer {layer}, not the pick at position {}, \
         layer {}",
        pick.position,
        pick.layer
    );
    // the node names the ubatch: it must lie in the turn before anything
    // is sized by it
    ensure!(
        receipt.position_start <= start && start < end && end <= receipt.position_end,
        "the region {start}..{end} is not positions of the turn {}..{}",
        receipt.position_start,
        receipt.position_end
    );
    ensure!(
        (start..end).contains(&position),
        "the region {start}..{end} does not hold position {position}"
    );
    let continues = end < receipt.position_end;
    let wanted = start..end + u32::from(continues);
    ensure!(
        opening.leaves.len() == wanted.len()
            && opening.leaves.iter().map(|l| l.position).eq(wanted.clone()),
        "the region opens {} leaves, not one at each position of {wanted:?}",
        opening.leaves.len()
    );
    for leaf in &opening.leaves {
        crate::binding::check_mixing_order(&leaf.mixing)
            .with_context(|| format!("position {}", leaf.position))?;
        verify_mixing_leaf(receipt, layer, leaf.position, &leaf.mixing, &leaf.path)?;
    }
    let keys = |leaf: &MixingLeafOpening| -> Vec<(MixingKind, u32)> {
        leaf.mixing.iter().map(|t| (t.kind, t.index)).collect()
    };
    let first = &opening.leaves[0];
    if let Some(leaf) = opening.leaves.iter().find(|l| keys(l) != keys(first)) {
        bail!(
            "position {}: the mixing leaf holds other values than at position {start}",
            leaf.position
        );
    }
    ensure!(
        first.mixing.iter().any(|t| t.kind == MixingKind::State),
        "the region opens no recurrent state"
    );
    for leaf in &opening.leaves {
        // a ubatch reads its state before its first position
        let boundary = leaf.position == start || leaf.position == end;
        for state in leaf.mixing.iter().filter(|t| t.kind == MixingKind::State) {
            ensure!(
                state.values.is_some() == boundary,
                "position {}: state {} is {} in the region {start}..{end}",
                leaf.position,
                state.index,
                if boundary { "not opened" } else { "opened" }
            );
        }
    }
    Ok(())
}

/// The evaluation of the region: the opened inputs at each position of the
/// ubatch, the states it read, and a zero residual, which the token mixing
/// does not read. `residual_len` is the width of the residual stream.
///
/// An error is the node's fault: the request is made of the opening alone.
pub(crate) fn region_request(
    opening: &RecurrentOpening,
    residual_len: usize,
) -> Result<GlueRequest> {
    opened_region(opening, residual_len).or_node_fault()
}

fn opened_region(opening: &RecurrentOpening, residual_len: usize) -> Result<GlueRequest> {
    let n = usize::try_from(opening.end.saturating_sub(opening.start))?;
    let leaves = opening
        .leaves
        .get(..n)
        .context("the region opens too few positions")?;
    let positions = leaves
        .iter()
        .map(|l| i32::try_from(l.position))
        .collect::<Result<Vec<_>, _>>()?;
    let mut columns: BTreeMap<u32, Vec<f32>> = BTreeMap::new();
    let mut len: BTreeMap<u32, usize> = BTreeMap::new();
    for leaf in leaves {
        for input in leaf.mixing.iter().filter(|t| t.kind == MixingKind::Input) {
            let values = input.f32_values()?.with_context(|| {
                format!(
                    "position {}: mixing input {} is not opened",
                    leaf.position, input.index
                )
            })?;
            let width = *len.entry(input.index).or_insert(values.len());
            ensure!(
                width == values.len(),
                "mixing input {} changes length",
                input.index
            );
            columns.entry(input.index).or_default().extend(values);
        }
    }
    let mut inputs: Vec<GlueInput> = columns
        .into_iter()
        .map(|(index, values)| GlueInput {
            kind: TensorKind::MixingInput,
            index,
            len: len[&index],
            values: GlueValues::F32(values),
        })
        .collect();
    for state in leaves[0]
        .mixing
        .iter()
        .filter(|t| t.kind == MixingKind::State)
    {
        let values = state
            .f32_values()?
            .context("the start state is not opened")?;
        inputs.push(GlueInput {
            kind: TensorKind::MixingState,
            index: state.index,
            len: values.len(),
            values: GlueValues::F32(values),
        });
    }
    Ok(GlueRequest {
        layer: opening.layer,
        residual: vec![0.0; residual_len * n],
        positions,
        inputs,
    })
}

#[derive(Clone, Debug, PartialEq)]
pub struct RegionReport {
    pub layer: u32,
    pub start: u32,
    pub end: u32,
    pub verdicts: Vec<GlueVerdict>,
    /// Wall time of the auditor's evaluation.
    pub eval_seconds: f64,
    pub pass: bool,
}

/// Recomputes the region and compares the outputs at every position, the
/// state left with the one the next ubatch read, and, at the sequence's
/// start, the state read with zero.
pub fn check_region(
    evaluator: &dyn GlueEvaluator,
    opening: &RecurrentOpening,
    residual_len: usize,
) -> Result<RegionReport> {
    let request = region_request(opening, residual_len)?;
    let started = Instant::now();
    let output = evaluator
        .evaluate_region(&request)
        .map_err(|error| blame_refused_values(error, "recurrent region"))?;
    let eval_seconds = started.elapsed().as_secs_f64();
    let computed = Computed::new(&output, opening.layer);
    let mut verdicts = Verdicts::default();
    let n = request.positions.len();
    for (col, leaf) in opening.leaves[..n].iter().enumerate() {
        for out in leaf.mixing.iter().filter(|t| t.kind == MixingKind::Output) {
            let what = format!("position {}: mixing output {}", leaf.position, out.index);
            let reference = computed
                .get(TensorKind::MixingOutput, out.index)
                .and_then(|t| t.f32_column(col));
            let error = match (out.f32_values().or_node_fault()?, reference) {
                (Some(opened), Some(reference)) => relative_error(&opened, reference),
                _ => f64::INFINITY,
            };
            verdicts.add_within("recurrent outputs", &what, error, RECURRENT_TOLERANCE);
        }
    }
    let states = |leaf: &MixingLeafOpening| -> Result<Vec<(u32, Vec<f32>)>> {
        leaf.mixing
            .iter()
            .filter(|t| t.kind == MixingKind::State)
            .map(|t| {
                let values = t.f32_values().or_node_fault()?;
                Ok((t.index, values.unwrap_or_default()))
            })
            .collect()
    };
    if let Some(next) = opening.leaves.get(n) {
        for (index, opened) in states(next)? {
            let what = format!("position {}: state {index}", next.position);
            let error = computed
                .get(TensorKind::MixingStateOut, index)
                .and_then(|t| t.f32_column(0))
                .map_or(f64::INFINITY, |left| {
                    relative_error(&opened, most_recent(left, opened.len()))
                });
            verdicts.add_within("states left", &what, error, RECURRENT_TOLERANCE);
        }
    }
    if opening.start == 0 {
        for (index, read) in states(&opening.leaves[0])? {
            let zero = read.iter().all(|&v| v == 0.0);
            let what = format!("position 0: state {index}");
            verdicts.add("first state", &what, if zero { 0.0 } else { f64::INFINITY });
        }
    }
    if verdicts.0.is_empty() {
        verdicts.inconsistent("the region compared nothing".into());
    }
    let pass = verdicts.0.iter().all(|v| v.pass);
    Ok(RegionReport {
        layer: opening.layer,
        start: opening.start,
        end: opening.end,
        verdicts: verdicts.0,
        eval_seconds,
        pass,
    })
}

/// The most recent state of `left`: an op that keeps rollback snapshots of
/// earlier states for speculative decoding writes them after it.
fn most_recent(left: &[f32], len: usize) -> &[f32] {
    if len > 0 && left.len() > len && left.len().is_multiple_of(len) {
        &left[..len]
    } else {
        left
    }
}

#[cfg(test)]
#[path = "recurrent/tests.rs"]
mod tests;
