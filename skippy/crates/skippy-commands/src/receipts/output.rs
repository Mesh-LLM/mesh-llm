//! What `skippy audit-receipt` reports, as JSON or text.

use std::io::Write;

use serde::Serialize;
use skippy_receipts::audit::{MatrixVerdict, RoleVerdict};
use skippy_receipts::glue::GlueVerdict;
use skippy_receipts::head::{HeadVerdict, PositionVerdict};
use skippy_receipts::wire::{Challenge, Receipt};

use super::attention_audit::AttentionOutput;
use super::recurrent_audit::{self, RecurrentOutput};

#[derive(Serialize)]
pub struct AuditOutput {
    pub response: String,
    pub receipt: Receipt,
    pub challenge: Challenge,
    /// Why the receipt does not fit the claimed model, or an opened leaf is
    /// not in its tree.
    pub binding_error: Option<String>,
    pub layers: Vec<LayerOutput>,
    /// The row fail rule, per weight role across the opened layers.
    pub roles: Vec<RoleOutput>,
    /// The input of layer 0 against the receipt's tokens.
    pub anchors: Result<GlueOutput, String>,
    /// The sampled tokens against the output head. Absent when the turn
    /// sampled no token from a position it computed.
    pub head: Option<Result<HeadOutput, String>>,
    /// The response against the sampled tokens. Absent when rechecking
    /// evidence: the node does not sign the response.
    pub tokens: Option<Result<TokenOutput, String>>,
    /// Hex VRF proof of the rows checked, from the opening as received.
    pub rows_proof: Option<String>,
    /// The attention of the picked heads. Absent when it did not run.
    pub attention: Option<AttentionOutput>,
    /// The recurrent token mixing of each stage's first attention pick on
    /// a layer that keeps a recurrent state.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub recurrent: Vec<RecurrentOutput>,
    /// The challenge was picked by hand (`--layer`), not drawn from the VRF.
    pub hand_picked: bool,
    /// Seconds from sending the challenge to receiving the whole opening.
    /// The node is meant to answer within the 10-second challenge window.
    pub open_seconds: Option<f64>,
    pub pass: bool,
}

impl AuditOutput {
    pub(super) fn new(
        response: String,
        receipt: Receipt,
        challenge: Challenge,
        binding: &anyhow::Result<()>,
    ) -> Self {
        Self {
            response,
            receipt,
            challenge,
            binding_error: binding.as_ref().err().map(|e| format!("{e:#}")),
            layers: Vec::new(),
            roles: Vec::new(),
            anchors: Err("not checked".to_owned()),
            head: None,
            tokens: Some(Err("not checked".to_owned())),
            rows_proof: None,
            attention: None,
            recurrent: Vec::new(),
            hand_picked: false,
            open_seconds: None,
            pass: false,
        }
    }

    pub(super) fn finish(&mut self) {
        self.pass = self.binding_error.is_none()
            && self.layers.len() == self.challenge.picks.len()
            && !self.roles.is_empty()
            && self.roles.iter().all(|r| r.pass)
            && self.layers.iter().all(|l| l.glue_pass)
            && self.anchors.as_ref().is_ok_and(|a| a.pass)
            && self.head_pass()
            && self.attention.as_ref().is_none_or(AttentionOutput::pass)
            && self.recurrent.iter().all(RecurrentOutput::passes)
            && self
                .tokens
                .as_ref()
                .is_none_or(|t| t.as_ref().is_ok_and(|t| t.response_ok && t.prompt_ok));
    }
}

impl AuditOutput {
    /// The checks that ran and failed, in a few words each. A check that
    /// could not run is in [`Self::unchecked`] instead.
    pub fn failures(&self) -> Vec<String> {
        let mut out = Vec::new();
        if let Some(error) = &self.binding_error {
            out.push(format!("binding: {error}"));
        }
        out.extend(self.roles.iter().filter(|r| !r.pass).map(|r| {
            format!(
                "weight rows of {}: {}/{} fail",
                r.role, r.failing_tokens, r.tokens
            )
        }));
        out.extend(
            self.layers
                .iter()
                .filter(|l| !l.glue_pass && l.glue_error.is_none())
                .map(|l| format!("glue of layer {}", l.layer)),
        );
        if self.binding_error.is_none() && self.anchors.as_ref().is_ok_and(|a| !a.pass) {
            out.push("input of layer 0".into());
        }
        if let Some(Ok(head)) = &self.head
            && !head.pass
        {
            out.push("output head".into());
        }
        if let Some(attention) = &self.attention {
            out.extend(attention.failures());
        }
        out.extend(recurrent_audit::failures(&self.recurrent));
        if let Some(Ok(tokens)) = &self.tokens {
            if !tokens.response_ok {
                out.push("response against the sampled tokens".into());
            }
            if !tokens.prompt_ok {
                out.push("prompt against the receipt's tokens".into());
            }
        }
        out
    }

    /// The checks that could not run here, and why: they say nothing about
    /// the node.
    pub fn unchecked(&self) -> Vec<String> {
        if self.binding_error.is_some() {
            return Vec::new();
        }
        let mut out: Vec<String> = self
            .layers
            .iter()
            .filter_map(|l| {
                Some(format!(
                    "glue of layer {}: {}",
                    l.layer,
                    l.glue_error.as_ref()?
                ))
            })
            .collect();
        if let Err(error) = &self.anchors {
            out.push(format!("input of layer 0: {error}"));
        }
        if let Some(Err(error)) = &self.head {
            out.push(format!("output head: {error}"));
        }
        if let Some(attention) = &self.attention {
            out.extend(attention.unchecked());
        }
        out.extend(recurrent_audit::unchecked(&self.recurrent));
        if self.head.is_none() && !self.challenge.head.is_empty() {
            out.push("output head: not opened".into());
        }
        if let Some(Err(error)) = &self.tokens {
            out.push(format!("tokens: {error}"));
        }
        out
    }

    /// The report `skippy audit-receipt` prints.
    pub fn render_text(&self) -> String {
        let mut out = Vec::new();
        let _ = render(&mut out, self);
        String::from_utf8_lossy(&out).into_owned()
    }
}

/// A receipt this verifier cannot audit, and why. It is no audit, not a
/// failed one ([`skippy_receipts::coverage`]).
#[derive(Serialize)]
pub struct NoAuditOutput {
    pub outcome: &'static str,
    pub receipt_id: String,
    pub reason: String,
}

impl NoAuditOutput {
    pub(super) fn new(receipt_id: String, reason: String) -> Self {
        Self {
            outcome: "no_audit",
            receipt_id,
            reason,
        }
    }
}

pub(super) fn render_no_audit(out: &mut dyn Write, output: &NoAuditOutput) -> std::io::Result<()> {
    writeln!(out, "receipt {}: {}", output.receipt_id, output.reason)?;
    writeln!(out, "NO AUDIT")
}

impl AuditOutput {
    /// A challenge that named head positions needs their check to pass.
    fn head_pass(&self) -> bool {
        match &self.head {
            Some(head) => head.as_ref().is_ok_and(|h| h.pass),
            None => self.challenge.head.is_empty(),
        }
    }
}

#[derive(Serialize)]
pub struct HeadOutput {
    pub positions: Vec<HeadPositionOutput>,
    /// Tokens accepted only within the tie band.
    pub ties: usize,
    pub flagged: bool,
    pub pass: bool,
    /// Checking the node's logits with the client's head key, or, without
    /// them, reading the output matrix and computing every logit.
    pub seconds: f64,
    /// The node's logits were checked with the client's head key.
    pub keyed: bool,
    /// Building the head key, when this audit built it.
    pub key_build_seconds: Option<f64>,
    /// How the tokens were judged: the greedy choice, kept by the turn's
    /// sampler, or not judged and why.
    pub rule: String,
}

#[derive(Serialize)]
pub struct HeadPositionOutput {
    pub position: u32,
    pub sampled: i32,
    pub best: i32,
    pub margin: f64,
    pub band: f64,
    pub within_band: bool,
    /// The sampler let the check judge the token.
    pub judged: bool,
    /// The key check's error relative to the logits' norm, and the row
    /// check's, when the node's logits were checked with the head key.
    pub key_error: Option<f64>,
    pub row_error: Option<f64>,
    /// Roundings of the activation the node took the other way, which
    /// the key check allowed for.
    pub rounding_flips: Option<usize>,
    pub pass: bool,
}

impl HeadOutput {
    pub(super) fn new(verdict: &HeadVerdict, seconds: f64) -> Self {
        let position = |p: &PositionVerdict| HeadPositionOutput {
            position: p.position,
            sampled: p.sampled,
            best: p.best,
            margin: p.margin,
            band: p.band,
            within_band: p.within_band,
            judged: p.judged,
            key_error: p.logits.map(|l| l.key_error),
            row_error: p.logits.map(|l| l.row_error),
            rounding_flips: p.logits.map(|l| l.rounding_flips),
            pass: p.pass,
        };
        Self {
            positions: verdict.positions.iter().map(position).collect(),
            ties: verdict.ties,
            flagged: verdict.flagged,
            pass: verdict.pass,
            seconds,
            keyed: verdict.positions.iter().all(|p| p.logits.is_some()),
            key_build_seconds: None,
            rule: String::new(),
        }
    }
}

#[derive(Serialize)]
pub struct LayerOutput {
    pub layer: u32,
    pub matrices: Vec<MatrixOutput>,
    pub glue: Vec<GlueOutput>,
    pub glue_pass: bool,
    /// Why the glue could not be checked.
    pub glue_error: Option<String>,
    /// Weights read inside token-mixing ops, which no check covers.
    pub unchecked_weights: Vec<String>,
    pub cost: CostOutput,
}

#[derive(Serialize)]
pub struct RoleOutput {
    pub role: String,
    pub layers: Vec<u32>,
    pub router: bool,
    pub tokens: usize,
    pub failing_tokens: usize,
    pub pass: bool,
}

impl From<&RoleVerdict> for RoleOutput {
    fn from(r: &RoleVerdict) -> Self {
        Self {
            role: r.role.clone(),
            layers: r.layers.clone(),
            router: r.router,
            tokens: r.tokens,
            failing_tokens: r.failing_tokens,
            pass: r.pass,
        }
    }
}

#[derive(Serialize)]
pub struct TokenOutput {
    /// The text of the sampled tokens.
    pub generated: String,
    pub response_ok: bool,
    pub prompt_ok: bool,
}

#[derive(Serialize)]
pub struct GlueOutput {
    pub check: String,
    pub tokens: usize,
    pub failing_tokens: usize,
    pub max_error: f64,
    pub worst: String,
    pub failures: Vec<String>,
    pub pass: bool,
}

impl From<&GlueVerdict> for GlueOutput {
    fn from(g: &GlueVerdict) -> Self {
        Self {
            check: g.check.clone(),
            tokens: g.tokens,
            failing_tokens: g.failing_tokens,
            max_error: g.max_error,
            worst: g.worst.clone(),
            failures: g.failures.clone(),
            pass: g.pass,
        }
    }
}

/// What checking one layer cost this machine, measured.
#[derive(Clone, Serialize)]
pub struct CostOutput {
    pub positions: usize,
    /// Loading the layer alone on the CPU.
    pub layer_load_seconds: f64,
    /// Each evaluation of the layer at every opened position; the first
    /// includes creating the evaluation context.
    pub glue_eval_seconds: Vec<f64>,
    /// The whole glue check: evaluation and comparisons.
    pub glue_check_seconds: f64,
    /// Reading weight rows and checking the matmul columns.
    pub row_check_seconds: f64,
}

impl CostOutput {
    /// The fastest evaluation, per opened position.
    fn eval_ms_per_position(&self) -> f64 {
        let best = self
            .glue_eval_seconds
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        best * 1000.0 / self.positions.max(1) as f64
    }
}

#[derive(Serialize)]
pub struct MatrixOutput {
    pub weight: String,
    pub weight_type: String,
    pub router: bool,
    pub tokens: usize,
    pub failing_tokens: usize,
    pub max_error: f64,
    pub best_profile: String,
}

impl From<&MatrixVerdict> for MatrixOutput {
    fn from(m: &MatrixVerdict) -> Self {
        Self {
            weight: m.weight.clone(),
            weight_type: m.weight_type.clone(),
            router: m.router,
            tokens: m.tokens,
            failing_tokens: m.failing_tokens,
            max_error: m.max_error,
            best_profile: m.best_profile.to_string(),
        }
    }
}

fn ok(pass: bool) -> &'static str {
    if pass { "ok" } else { "FAIL" }
}

pub(super) fn render(out: &mut dyn Write, output: &AuditOutput) -> std::io::Result<()> {
    if output.tokens.is_some() {
        writeln!(out, "response: {}", output.response.trim())?;
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
    match &output.binding_error {
        Some(error) => writeln!(
            out,
            "receipt binding, shape and merkle paths: FAIL ({error})"
        )?,
        None => writeln!(
            out,
            "receipt binding and signature, shape, merkle paths and stage boundaries: ok"
        )?,
    }
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
            r.role,
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
    if output.binding_error.is_none() {
        match &output.anchors {
            Ok(a) => writeln!(
                out,
                "input of layer 0 from the receipt's tokens: {}/{} pass  max err {:.1e}  {}",
                a.tokens - a.failing_tokens,
                a.tokens,
                a.max_error,
                ok(a.pass)
            )?,
            Err(error) => writeln!(out, "input of layer 0: cannot check ({error})")?,
        }
        render_head(out, output.head.as_ref())?;
        if let Some(attention) = &output.attention {
            super::attention_audit::render(out, attention)?;
        }
        recurrent_audit::render(out, &output.recurrent)?;
        match &output.tokens {
            Some(tokens) => render_tokens(out, tokens)?,
            None => writeln!(
                out,
                "response: not part of the evidence (the node does not sign it)"
            )?,
        }
    }
    writeln!(
        out,
        "{}",
        if output.pass {
            "AUDIT PASS"
        } else {
            "AUDIT FAIL"
        }
    )
}

fn render_receipt(out: &mut dyn Write, receipt: &Receipt) -> std::io::Result<()> {
    writeln!(
        out,
        "receipt {}  model {}  positions {}..{}  generation from {}",
        receipt.receipt_id,
        receipt.model,
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
            &stage.root[..16.min(stage.root.len())],
            stage.leaf_count,
            stage.cache_types.join(","),
            stage.backend
        )?;
    }
    if let Some(binding) = &receipt.binding {
        let short = |hex: &str| hex[..16.min(hex.len())].to_owned();
        writeln!(
            out,
            "  turn root {}…  node key {}…  client key {}…  {}  sampler {}",
            short(&binding.turn_root),
            short(&binding.node_key),
            short(&binding.client_key),
            binding.runtime,
            binding.sampler
        )?;
    }
    Ok(())
}

fn render_layer(out: &mut dyn Write, layer: &LayerOutput) -> std::io::Result<()> {
    writeln!(out, "layer {}:", layer.layer)?;
    for m in &layer.matrices {
        writeln!(
            out,
            "  {:<30} {:<5} {:>3}/{:<3} pass  max err {:.1e}  {}{}",
            m.weight,
            m.weight_type,
            m.tokens - m.failing_tokens,
            m.tokens,
            m.max_error,
            m.best_profile,
            if m.router { "  (router, all rows)" } else { "" }
        )?;
    }
    if let Some(error) = &layer.glue_error {
        writeln!(out, "  glue: cannot check ({error})")?;
    }
    for g in &layer.glue {
        writeln!(
            out,
            "  glue {:<14} {:>3}/{:<3} pass  max err {:.1e}  {}{}",
            g.check,
            g.tokens - g.failing_tokens,
            g.tokens,
            g.max_error,
            ok(g.pass),
            if g.pass {
                String::new()
            } else {
                format!("  worst: {}", g.worst)
            }
        )?;
        for failure in &g.failures {
            writeln!(out, "      {failure}")?;
        }
    }
    if !layer.unchecked_weights.is_empty() {
        writeln!(
            out,
            "  unchecked weights (inside token mixing): {}",
            layer.unchecked_weights.join(", ")
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

fn render_head(
    out: &mut dyn Write,
    head: Option<&Result<HeadOutput, String>>,
) -> std::io::Result<()> {
    match head {
        None => Ok(()),
        Some(Err(error)) => writeln!(out, "output head: cannot check ({error})"),
        Some(Ok(h)) => {
            let positions: Vec<u32> = h.positions.iter().map(|p| p.position).collect();
            writeln!(
                out,
                "sampled tokens at {positions:?} are {}: {}/{} pass  ties {}{}  {}  ({:.2} s{})",
                h.rule,
                h.positions.iter().filter(|p| p.pass).count(),
                h.positions.len(),
                h.ties,
                if h.flagged { " (flagged)" } else { "" },
                ok(h.pass),
                h.seconds,
                if h.keyed {
                    ", node's logits checked with the head key"
                } else {
                    ""
                }
            )?;
            if let Some(seconds) = h.key_build_seconds {
                writeln!(out, "  built the head key in {seconds:.2} s")?;
            }
            for p in h
                .positions
                .iter()
                .filter(|p| p.key_error.is_some() && !p.pass)
            {
                writeln!(
                    out,
                    "  position {}: key error {:.3e}, row error {:.3e}",
                    p.position,
                    p.key_error.unwrap_or_default(),
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
            for p in h.positions.iter().filter(|p| !p.pass || p.within_band) {
                writeln!(
                    out,
                    "  position {}: sampled {}, best {}, margin {:.3e} (band {:.3e})",
                    p.position, p.sampled, p.best, p.margin, p.band
                )?;
            }
            Ok(())
        }
    }
}

fn render_tokens(out: &mut dyn Write, tokens: &Result<TokenOutput, String>) -> std::io::Result<()> {
    match tokens {
        Ok(t) => {
            writeln!(
                out,
                "response is the text of the sampled tokens: {}",
                ok(t.response_ok)
            )?;
            writeln!(
                out,
                "prompt is in the receipt's tokens: {}",
                ok(t.prompt_ok)
            )?;
            if !t.response_ok {
                writeln!(out, "  sampled text: {}", t.generated.trim())?;
            }
            Ok(())
        }
        Err(error) => writeln!(out, "tokens: cannot check ({error})"),
    }
}
