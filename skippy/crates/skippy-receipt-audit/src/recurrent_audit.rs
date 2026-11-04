//! The recurrent check of an audited turn ([`skippy_receipts::recurrent`]):
//! for each stage, the first attention pick on a layer whose token mixing
//! reads a recurrent state, as the auditor's own graph has it, must come
//! with the ubatch that computed it. The auditor recomputes the ubatch's
//! token mixing from the opened inputs and state.

use std::collections::BTreeSet;
use std::io::Write;

use serde::Serialize;
use skippy_receipts::attention::opening::AttentionPick;
use skippy_receipts::node_fault::NodeFault;
use skippy_receipts::opened_values::ValidatedOpening;
use skippy_receipts::recurrent::{check_region, region_pick};
use skippy_receipts::wire::{RecurrentOpening, StageCommitment};

use crate::claimed_model::{CacheTypes, ClaimedModel};
use crate::node_text::shown;
use crate::output::GlueOutput;

#[derive(Serialize)]
pub struct RecurrentOutput {
    pub position: u32,
    pub layer: u32,
    /// The ubatch's positions, as opened.
    pub start: Option<u32>,
    pub end: Option<u32>,
    pub verdicts: Vec<GlueOutput>,
    pub eval_seconds: Option<f64>,
    /// Why the check could not run here; says nothing about the node.
    pub unchecked: Option<String>,
    /// Why the region failed.
    pub failure: Option<String>,
    pub pass: bool,
}

impl RecurrentOutput {
    fn new(position: u32, layer: u32) -> Self {
        Self {
            position,
            layer,
            start: None,
            end: None,
            verdicts: Vec::new(),
            eval_seconds: None,
            unchecked: None,
            failure: None,
            pass: false,
        }
    }
}

/// The layers of an audit whose glue ran, so the auditor knows from its own
/// graph whether each keeps a recurrent state.
pub(crate) struct KnownLayers<'a> {
    /// The picked layers whose glue ran.
    pub(crate) evaluated: &'a BTreeSet<u32>,
    /// Those of them that keep a recurrent state.
    pub(crate) state: &'a BTreeSet<u32>,
}

/// Checks the region of every stage that holds a picked layer the
/// auditor's graph gives a recurrent state, and that no other stage opened
/// one.
pub(crate) fn check_recurrent(
    model: &dyn ClaimedModel,
    opening: &ValidatedOpening<'_>,
    layers: &KnownLayers<'_>,
) -> Vec<RecurrentOutput> {
    opening
        .receipt()
        .stages
        .iter()
        .filter_map(|stage| check_stage(model, opening, stage, layers))
        .collect()
}

fn check_stage(
    model: &dyn ClaimedModel,
    opening: &ValidatedOpening<'_>,
    stage: &StageCommitment,
    layers: &KnownLayers<'_>,
) -> Option<RecurrentOutput> {
    let held = stage.layer_start..stage.layer_end;
    let pick = region_pick(opening.challenge(), &held, layers.state);
    let region = opening.region(stage);
    let (Some(pick), Some(region)) = (pick, region) else {
        return unpaired(pick, region, layers);
    };
    let mut output = RecurrentOutput::new(pick.position, pick.layer);
    if (region.position, region.layer) != (pick.position, pick.layer) {
        output.failure = Some(format!(
            "the stage opened the region of position {}, layer {}, not of its first pick on a \
             layer that keeps a state",
            region.position, region.layer
        ));
        return Some(output);
    }
    check_one(model, opening, pick, region, &mut output);
    Some(output)
}

/// A stage that owes a region and opened none, or opened one it does not
/// owe. The second is a failure only where the auditor's own graph says
/// the layer keeps no state.
fn unpaired(
    pick: Option<&AttentionPick>,
    region: Option<&RecurrentOpening>,
    layers: &KnownLayers<'_>,
) -> Option<RecurrentOutput> {
    match (pick, region) {
        (Some(pick), _) => {
            let mut output = RecurrentOutput::new(pick.position, pick.layer);
            output.failure = Some("the stage opened no recurrent region".into());
            Some(output)
        }
        (None, Some(region)) => {
            let mut output = RecurrentOutput::new(region.position, region.layer);
            output.start = Some(region.start);
            output.end = Some(region.end);
            if layers.evaluated.contains(&region.layer) {
                output.failure = Some(
                    "the stage opened a recurrent region on a layer that keeps no state".into(),
                );
            } else {
                output.unchecked = Some("the layer's glue did not run".into());
            }
            Some(output)
        }
        (None, None) => None,
    }
}

fn check_one(
    model: &dyn ClaimedModel,
    opening: &ValidatedOpening<'_>,
    pick: &AttentionPick,
    region: &RecurrentOpening,
    output: &mut RecurrentOutput,
) {
    output.start = Some(region.start);
    output.end = Some(region.end);
    // the width of the residual stream, from the picked layer's opening
    let residual_len = opening
        .leaf(pick.layer, pick.position)
        .and_then(|leaf| leaf.residual_in_values().ok())
        .map(|residual| residual.len());
    let Some(residual_len) = residual_len else {
        output.unchecked = Some("the attention position is not a picked one".into());
        return;
    };
    let mut report = None;
    let checked = model.with_layer(pick.layer, CacheTypes::default(), &mut |evaluator| {
        report = Some(check_region(evaluator, region, residual_len)?);
        Ok(())
    });
    match (checked, report) {
        (Ok(()), Some(report)) => {
            output.verdicts = report.verdicts.iter().map(GlueOutput::from).collect();
            output.eval_seconds = Some(report.eval_seconds);
            output.pass = report.pass;
            if !report.pass {
                output.failure = Some("the recomputed region differs".into());
            }
        }
        // a region that makes no request is the node's; a layer that does
        // not load or compute is the client's
        (Err(error), _) if NodeFault::find(&error).is_some() => {
            output.failure = Some(format!("{error:#}"));
        }
        (Err(error), _) => output.unchecked = Some(format!("{error:#}")),
        (Ok(()), None) => output.unchecked = Some("the region was not checked".into()),
    }
}

pub(crate) fn failures(recurrent: &[RecurrentOutput]) -> Vec<String> {
    recurrent
        .iter()
        .filter_map(|r| {
            Some(format!(
                "recurrent region at position {}, layer {}: {}",
                r.position,
                r.layer,
                r.failure.as_ref()?
            ))
        })
        .collect()
}

pub(crate) fn unchecked(recurrent: &[RecurrentOutput]) -> Vec<String> {
    recurrent
        .iter()
        .filter_map(|r| {
            Some(format!(
                "recurrent region of layer {}: {}",
                r.layer,
                r.unchecked.as_ref()?
            ))
        })
        .collect()
}

pub(crate) fn render(out: &mut dyn Write, recurrent: &[RecurrentOutput]) -> std::io::Result<()> {
    if recurrent.is_empty() {
        return Ok(());
    }
    writeln!(out, "recurrent token mixing, per stage:")?;
    for r in recurrent {
        let span = match (r.start, r.end) {
            (Some(start), Some(end)) => format!("positions {start}..{end}"),
            _ => "-".into(),
        };
        let what = format!("position {} layer {}", r.position, r.layer);
        if let Some(reason) = &r.unchecked {
            writeln!(out, "  {what:<26} cannot check ({})", shown(reason))?;
            continue;
        }
        writeln!(
            out,
            "  {what:<26} {span}  {}{}{}",
            if r.pass { "ok" } else { "FAIL" },
            r.eval_seconds
                .map(|s| format!("  ({s:.2} s)"))
                .unwrap_or_default(),
            r.failure
                .as_ref()
                .map(|f| format!(" ({})", shown(f)))
                .unwrap_or_default()
        )?;
        for v in &r.verdicts {
            writeln!(
                out,
                "    {:<18} {:>4}/{:<4} pass  max err {:.1e}{}",
                shown(&v.check),
                v.tokens - v.failing_tokens,
                v.tokens,
                v.max_error,
                if v.pass {
                    String::new()
                } else {
                    format!("  worst: {}", shown(&v.worst))
                }
            )?;
        }
    }
    Ok(())
}
