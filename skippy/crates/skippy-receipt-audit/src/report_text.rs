//! The text report of an audit, as `skippy audit-receipt` and
//! `skippy recheck-receipt` print it. Its last line is the outcome:
//! `AUDIT PASS`, `AUDIT FAIL`, `NO AUDIT` or `NO ANSWER`.
//!
//! Much of what a report shows is text the node chose. Every such text
//! goes through [`shown`], so a report has the lines this module writes
//! and no others.

use std::io::Write;

use skippy_receipts::wire::Receipt;

use crate::node_text::{short, shown};
use crate::outcome::{Check, Outcome};
use crate::output::{
    AuditOutput, HeadOutput, LayerOutput, NoAuditOutput, TokenOutput, UnansweredOutput,
};
use crate::recurrent_audit;

fn ok(pass: bool) -> &'static str {
    if pass { "ok" } else { "FAIL" }
}

pub fn render_no_audit(out: &mut dyn Write, output: &NoAuditOutput) -> std::io::Result<()> {
    writeln!(
        out,
        "receipt {}: {}",
        shown(&output.receipt_id),
        shown(&output.reason)
    )?;
    writeln!(out, "{}", Outcome::NoAudit.line())
}

pub fn render_unanswered(out: &mut dyn Write, output: &UnansweredOutput) -> std::io::Result<()> {
    writeln!(
        out,
        "receipt {}: the node did not answer /receipts/{}: {}",
        shown(&output.receipt_id),
        output.request,
        shown(&output.reason)
    )?;
    writeln!(out, "{}", Outcome::Unanswered.line())
}

pub fn render(out: &mut dyn Write, output: &AuditOutput) -> std::io::Result<()> {
    if output.tokens.is_some() {
        writeln!(out, "response: {}", shown(output.response.trim()))?;
        writeln!(out)?;
    }
    render_receipt(out, &output.receipt)?;
    for pick in &output.challenge.picks {
        let stage = output
            .receipt
            .stage_for_layer(pick.layer)
            .map_or(0, |s| s.stage_index);
        writeln!(
            out,
            "challenge: layer {} (stage {stage}), positions {:?}",
            pick.layer, pick.positions
        )?;
    }
    if let Some(seconds) = output.open_seconds {
        writeln!(out, "opening: {seconds:.2} s")?;
    }
    if output.hand_picked {
        writeln!(
            out,
            "challenge picked by hand (--layer), not drawn from the VRF: not evidence"
        )?;
    }
    render_binding(out, output)?;
    for layer in &output.layers {
        render_layer(out, layer)?;
    }
    if !output.roles.is_empty() {
        writeln!(out, "weight rows, each role across the opened layers:")?;
    }
    for r in &output.roles {
        writeln!(
            out,
            "  {:<24} {:>3}/{:<3} pass  {}{}",
            shown(&r.role),
            r.tokens - r.failing_tokens,
            r.tokens,
            ok(r.pass),
            if r.router {
                "  (router, no failure allowed)"
            } else {
                ""
            }
        )?;
    }
    if output.opened() {
        render_checks(out, output)?;
    }
    for note in &output.notes {
        writeln!(out, "note: {}", shown(note))?;
    }
    for failure in &output.failures {
        writeln!(
            out,
            "failed: {}{}",
            shown(&failure.check),
            if failure.provable {
                ""
            } else {
                "  (seen by this client alone)"
            }
        )?;
    }
    for unchecked in &output.unchecked {
        writeln!(out, "could not check: {}", shown(unchecked))?;
    }
    writeln!(out, "{}", output.outcome.line())
}

fn render_binding(out: &mut dyn Write, output: &AuditOutput) -> std::io::Result<()> {
    for error in &output.request_errors {
        writeln!(out, "receipt against the request: FAIL ({})", shown(error))?;
    }
    if let Some(error) = &output.binding_error {
        return writeln!(
            out,
            "receipt binding and signature: FAIL ({})",
            shown(error)
        );
    }
    if let Some(error) = &output.receipt_error {
        return writeln!(
            out,
            "receipt shape against the claimed model: FAIL ({})",
            shown(error)
        );
    }
    match &output.opening_error {
        Some(error) => writeln!(
            out,
            "opening form, merkle paths and values: FAIL ({})",
            shown(error)
        ),
        None => writeln!(
            out,
            "receipt binding and signature, shape, opening form, merkle paths, stage boundaries \
             and values: ok"
        ),
    }
}

/// The checks of the opened values, which ran because the receipt and the
/// opening were accepted.
fn render_checks(out: &mut dyn Write, output: &AuditOutput) -> std::io::Result<()> {
    match &output.anchors {
        Check::Verdict(a) => writeln!(
            out,
            "input of layer 0 from the receipt's tokens: {}/{} pass  max err {:.1e}  {}",
            a.tokens - a.failing_tokens,
            a.tokens,
            a.max_error,
            ok(a.pass)
        )?,
        Check::NodeFault(fault) => writeln!(out, "input of layer 0: FAIL ({})", shown(fault))?,
        Check::Unavailable(reason) => {
            writeln!(out, "input of layer 0: cannot check ({})", shown(reason))?;
        }
    }
    render_head(out, output.head.as_ref())?;
    if let Some(attention) = &output.attention {
        crate::attention_audit::render(out, attention)?;
    }
    recurrent_audit::render(out, &output.recurrent)?;
    match &output.tokens {
        Some(tokens) => render_tokens(out, tokens, &output.receipt),
        None => writeln!(
            out,
            "response: not part of the evidence (the node does not sign it)"
        ),
    }
}

fn render_receipt(out: &mut dyn Write, receipt: &Receipt) -> std::io::Result<()> {
    writeln!(
        out,
        "receipt {}  model {}  positions {}..{}  generation from {}",
        shown(&receipt.receipt_id),
        shown(&receipt.model),
        receipt.position_start,
        receipt.position_end,
        receipt.generation_start
    )?;
    for stage in &receipt.stages {
        writeln!(
            out,
            "  stage {}  layers {:>2}..{:<2}  root {}…  leaves {}  cache {}  ({})",
            stage.stage_index,
            stage.layer_start,
            stage.layer_end,
            short(&stage.root, 16),
            stage.leaf_count,
            shown(&stage.cache_types.join(",")),
            shown(&stage.backend)
        )?;
    }
    if let Some(binding) = &receipt.binding {
        writeln!(
            out,
            "  turn root {}…  node key {}…  client key {}…  {}  sampler {}",
            short(&binding.turn_root, 16),
            short(&binding.node_key, 16),
            short(&binding.client_key, 16),
            shown(&binding.runtime),
            shown(&binding.sampler)
        )?;
    }
    Ok(())
}

fn render_layer(out: &mut dyn Write, layer: &LayerOutput) -> std::io::Result<()> {
    writeln!(out, "layer {}:", layer.layer)?;
    for m in &layer.matrices {
        writeln!(
            out,
            "  {:<30} {:<5} {:>3}/{:<3} pass  max err {:.1e}  {}{}{}",
            shown(&m.weight),
            shown(&m.weight_type),
            m.tokens - m.failing_tokens,
            m.tokens,
            m.max_error,
            m.best_profile,
            match m.repaired_tokens {
                0 => String::new(),
                n => format!("  ({n} repaired near-ties)"),
            },
            if m.router { "  (router, all rows)" } else { "" }
        )?;
    }
    if let Some(fault) = &layer.rows_fault {
        writeln!(out, "  weight rows: FAIL ({})", shown(fault))?;
    }
    if let Some(error) = &layer.rows_error {
        writeln!(out, "  weight rows: cannot check ({})", shown(error))?;
    }
    if let Some(fault) = &layer.glue_fault {
        writeln!(out, "  glue: FAIL ({})", shown(fault))?;
    }
    if let Some(error) = &layer.glue_error {
        writeln!(out, "  glue: cannot check ({})", shown(error))?;
    }
    for g in &layer.glue {
        writeln!(
            out,
            "  glue {:<14} {:>3}/{:<3} pass  max err {:.1e}  {}{}",
            shown(&g.check),
            g.tokens - g.failing_tokens,
            g.tokens,
            g.max_error,
            ok(g.pass),
            if g.pass {
                String::new()
            } else {
                format!("  worst: {}", shown(&g.worst))
            }
        )?;
        for failure in &g.failures {
            writeln!(out, "      {}", shown(failure))?;
        }
    }
    if !layer.unchecked_weights.is_empty() {
        writeln!(
            out,
            "  unchecked weights (inside token mixing): {}",
            shown(&layer.unchecked_weights.join(", "))
        )?;
    }
    let cost = &layer.cost;
    writeln!(
        out,
        "  client cost: layer load {:.2} s, glue {:.2} ms per position ({} positions, first run {:.0} ms), row checks {:.0} ms",
        cost.layer_load_seconds,
        cost.eval_ms_per_position(),
        cost.positions,
        cost.glue_eval_seconds.first().copied().unwrap_or(0.0) * 1000.0,
        cost.row_check_seconds * 1000.0
    )
}

fn render_head(out: &mut dyn Write, head: Option<&Check<HeadOutput>>) -> std::io::Result<()> {
    let h = match head {
        None => return Ok(()),
        Some(Check::NodeFault(fault)) => {
            return writeln!(out, "output head: FAIL ({})", shown(fault));
        }
        Some(Check::Unavailable(reason)) => {
            return writeln!(out, "output head: cannot check ({})", shown(reason));
        }
        Some(Check::Verdict(h)) => h,
    };
    let positions: Vec<u32> = h.positions.iter().map(|p| p.position).collect();
    writeln!(
        out,
        "sampled tokens at {positions:?} are {}: {}/{} pass  ties {}{}  {}  ({:.2} s{})",
        shown(&h.rule),
        h.positions.iter().filter(|p| p.pass).count(),
        h.positions.len(),
        h.ties,
        if h.flagged { " (flagged)" } else { "" },
        ok(h.pass),
        h.seconds,
        if h.keyed {
            ", node's logits checked with the head key"
        } else if h.every_row {
            ", node's logits checked on every row of the output matrix"
        } else {
            ""
        }
    )?;
    if let Some(seconds) = h.key_build_seconds {
        writeln!(out, "  built the head key in {seconds:.2} s")?;
    }
    for p in h.positions.iter().filter(|p| !p.pass) {
        let (what, error) = match (p.key_error, p.every_row_error) {
            (Some(error), _) => ("key error", error),
            (None, Some(error)) => ("error over every row", error),
            (None, None) => continue,
        };
        writeln!(
            out,
            "  position {}: {what} {error:.3e}, row error {:.3e}",
            p.position,
            p.row_error.unwrap_or_default()
        )?;
    }
    for p in h
        .positions
        .iter()
        .filter(|p| p.rounding_flips.unwrap_or(0) > 0)
    {
        writeln!(
            out,
            "  position {}: {} activation rounding(s) on a boundary, taken the other way",
            p.position,
            p.rounding_flips.unwrap_or_default()
        )?;
    }
    for p in h
        .positions
        .iter()
        .filter(|p| p.forced.is_none() && (!p.pass || p.within_band))
    {
        writeln!(
            out,
            "  position {}: sampled {}, best {}, margin {:.3e} (band {:.3e})",
            p.position, p.sampled, p.best, p.margin, p.band
        )?;
    }
    render_reasoning(out, h)?;
    render_draws(out, h)
}

/// The reasoning budget of the turn as the node states it, how many of
/// the checked tokens its sampler forced, and each forced position that
/// holds another token.
fn render_reasoning(out: &mut dyn Write, h: &HeadOutput) -> std::io::Result<()> {
    let Some(reasoning) = &h.reasoning else {
        return Ok(());
    };
    let ends: Vec<String> = reasoning.ends.iter().map(|end| shown(end)).collect();
    writeln!(
        out,
        "  reasoning budget of {} tokens, from {} to {} as the node states them: forced {} of the checked tokens",
        reasoning.budget,
        shown(&reasoning.start),
        ends.join(" or "),
        reasoning.forced
    )?;
    for p in &h.positions {
        let Some(forced) = p.forced.filter(|forced| *forced != p.sampled) else {
            continue;
        };
        writeln!(
            out,
            "  position {}: the reasoning budget forces token {forced}; the turn sampled {}",
            p.position, p.sampled
        )?;
    }
    Ok(())
}

/// The replayed draws of a sampled turn: how many pick the sampled token,
/// and each one that does not, or does only within the band.
fn render_draws(out: &mut dyn Write, h: &HeadOutput) -> std::io::Result<()> {
    let draws: Vec<_> = h
        .positions
        .iter()
        .filter_map(|p| p.draw.as_ref().map(|draw| (p, draw)))
        .collect();
    if draws.is_empty() {
        return Ok(());
    }
    writeln!(
        out,
        "  draws replayed: {}/{} pick the sampled token, {} only within the band",
        draws.iter().filter(|(_, d)| d.pass).count(),
        draws.len(),
        h.band_draws
    )?;
    for (p, draw) in draws.iter().filter(|(_, d)| !d.exact) {
        writeln!(
            out,
            "  position {}: token {} {}: the draw's uniform is {:.6}, and it picks the token from {:.6} up to {:.6}",
            p.position,
            p.sampled,
            if draw.failed() {
                "is not the one the draw picks"
            } else {
                "is picked only within the band"
            },
            draw.uniform,
            draw.lower,
            draw.upper
        )?;
    }
    Ok(())
}

fn render_tokens(
    out: &mut dyn Write,
    tokens: &Check<TokenOutput>,
    receipt: &Receipt,
) -> std::io::Result<()> {
    let t = match tokens {
        Check::NodeFault(fault) => return writeln!(out, "tokens: FAIL ({})", shown(fault)),
        Check::Unavailable(reason) => {
            return writeln!(out, "tokens: cannot check ({})", shown(reason));
        }
        Check::Verdict(t) => t,
    };
    writeln!(
        out,
        "response is the text of the sampled tokens: {}",
        ok(t.response_ok)
    )?;
    if receipt.position_start > 0 {
        writeln!(
            out,
            "prompt is in the tokens of the cached prefix ({} positions) and the receipt: {}",
            receipt.position_start,
            ok(t.prompt_ok)
        )?;
    } else {
        writeln!(
            out,
            "prompt is in the receipt's tokens: {}",
            ok(t.prompt_ok)
        )?;
    }
    if !t.response_ok {
        writeln!(out, "  sampled text: {}", shown(t.generated.trim()))?;
    }
    Ok(())
}
