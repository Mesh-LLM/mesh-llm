//! The recurrent check of an audited turn ([`skippy_receipts::recurrent`]):
//! for each stage, the first attention pick on a layer whose token mixing
//! reads a recurrent state, as the auditor's own graph has it, must come
//! with the ubatch that computed it. The auditor recomputes the ubatch's
//! token mixing from the opened inputs and state.

use std::collections::BTreeSet;
use std::io::Write;

use anyhow::{Context, Result};
use serde::Serialize;
use skippy_receipts::attention::opening::AttentionPick;
use skippy_receipts::recurrent::{check_region, region_pick, verify_region};
use skippy_receipts::wire::{Challenge, Opening, Receipt, RecurrentOpening};

use super::output::GlueOutput;
use super::{CacheTypes, ClaimedModel};

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
    fn new(pick: &AttentionPick) -> Self {
        Self {
            position: pick.position,
            layer: pick.layer,
            start: None,
            end: None,
            verdicts: Vec::new(),
            eval_seconds: None,
            unchecked: None,
            failure: None,
            pass: false,
        }
    }

    /// Passed, or could not run here.
    pub fn passes(&self) -> bool {
        self.pass || (self.unchecked.is_some() && self.failure.is_none())
    }
}

/// Checks the region of every stage that holds a picked layer the
/// auditor's graph gives a recurrent state (`state_layers`).
pub(super) fn check_recurrent(
    model: &dyn ClaimedModel,
    receipt: &Receipt,
    challenge: &Challenge,
    opening: &Opening,
    state_layers: &BTreeSet<u32>,
) -> Vec<RecurrentOutput> {
    receipt
        .stages
        .iter()
        .filter_map(|stage| {
            let layers = stage.layer_start..stage.layer_end;
            let pick = region_pick(challenge, &layers, state_layers)?;
            let mut output = RecurrentOutput::new(pick);
            let region = opening
                .stage(stage.stage_index)
                .and_then(|s| s.recurrent.iter().find(|r| r.layer == pick.layer));
            match region {
                None => output.failure = Some("the stage opened no recurrent region".into()),
                Some(region) => check_one(model, receipt, opening, pick, region, &mut output),
            }
            Some(output)
        })
        .collect()
}

fn check_one(
    model: &dyn ClaimedModel,
    receipt: &Receipt,
    opening: &Opening,
    pick: &AttentionPick,
    region: &RecurrentOpening,
    output: &mut RecurrentOutput,
) {
    output.start = Some(region.start);
    output.end = Some(region.end);
    if let Err(error) = verify_region(receipt, pick, region) {
        output.failure = Some(format!("{error:#}"));
        return;
    }
    let residual_len = match residual_len(opening, pick) {
        Ok(len) => len,
        Err(error) => {
            output.failure = Some(format!("{error:#}"));
            return;
        }
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
        (Err(error), _) => output.unchecked = Some(format!("{error:#}")),
        (Ok(()), None) => output.unchecked = Some("the region was not checked".into()),
    }
}

/// The width of the residual stream, from the picked layer's opening.
fn residual_len(opening: &Opening, pick: &AttentionPick) -> Result<usize> {
    let leaf = opening
        .stages
        .iter()
        .flat_map(|s| &s.layers)
        .filter(|l| l.layer == pick.layer)
        .flat_map(|l| &l.leaves)
        .find(|l| l.position == pick.position)
        .context("an attention pick outside the opened positions")?;
    Ok(leaf.residual_in_values()?.len())
}

pub(super) fn failures(recurrent: &[RecurrentOutput]) -> Vec<String> {
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

pub(super) fn unchecked(recurrent: &[RecurrentOutput]) -> Vec<String> {
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

pub(super) fn render(out: &mut dyn Write, recurrent: &[RecurrentOutput]) -> std::io::Result<()> {
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
            writeln!(out, "  {what:<26} cannot check ({reason})")?;
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
                .map(|f| format!(" ({f})"))
                .unwrap_or_default()
        )?;
        for v in &r.verdicts {
            writeln!(
                out,
                "    {:<18} {:>4}/{:<4} pass  max err {:.1e}{}",
                v.check,
                v.tokens - v.failing_tokens,
                v.tokens,
                v.max_error,
                if v.pass {
                    String::new()
                } else {
                    format!("  worst: {}", v.worst)
                }
            )?;
        }
    }
    Ok(())
}
