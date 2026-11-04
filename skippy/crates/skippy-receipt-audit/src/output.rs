//! What an audit of a turn reports: every check's result, and what they
//! add up to. `skippy audit-receipt` prints it as JSON, or as text
//! ([`crate::report_text`]).

use serde::Serialize;
use skippy_receipts::audit::{MatrixVerdict, RoleVerdict};
use skippy_receipts::glue::GlueVerdict;
use skippy_receipts::head::reasoning_budget::Forcing;
use skippy_receipts::head::{HeadVerdict, PositionVerdict};
use skippy_receipts::wire::{Challenge, Receipt};

use crate::attention_audit::AttentionOutput;
use crate::outcome::{Check, Failure, Outcome};
use crate::recurrent_audit::{self, RecurrentOutput};

#[derive(Serialize)]
pub struct AuditOutput {
    /// What the checks add up to: `pass`, `fail`, `no_audit` when a check
    /// could not run here, or `unanswered` when the node did not open the
    /// attention rows.
    pub outcome: Outcome,
    /// Whether the outcome is `pass`.
    pub pass: bool,
    /// The checks that failed, and whether the evidence shows each.
    pub failures: Vec<Failure>,
    /// The checks that could not run here, and why: they say nothing about
    /// the node.
    pub unchecked: Vec<String>,
    /// What the audit does not cover for this turn: neither a pass nor a
    /// failure.
    pub notes: Vec<String>,
    pub response: String,
    pub receipt: Receipt,
    pub challenge: Challenge,
    /// Why the receipt is not one this client can hold the node to: it is
    /// not bound to the client's key and signed by the node. A failure
    /// only this client can see.
    pub binding_error: Option<String>,
    /// Why the receipt the node signed does not describe a turn of the
    /// claimed model: its stages, positions, tokens or cached prefix. The
    /// signed receipt shows this failure to anyone. An audit that refuses
    /// a receipt sends no challenge, so it keeps no evidence file.
    pub receipt_error: Option<String>,
    /// Why the opening the node signed is defective: it is not the answer
    /// to the challenge, a leaf is not in its tree, or a value does not fit
    /// the claimed model. A failure the evidence shows.
    pub opening_error: Option<String>,
    /// How the receipt contradicts the request the client sent: its id or
    /// its sampler statement. A failure no evidence shows, since the
    /// request is not part of a receipt.
    pub request_errors: Vec<String>,
    pub layers: Vec<LayerOutput>,
    /// The row fail rule, per weight role across the opened layers.
    pub roles: Vec<RoleOutput>,
    /// The input of layer 0 against the receipt's tokens.
    pub anchors: Check<GlueOutput>,
    /// The sampled tokens against the output head. Absent when the turn
    /// sampled no token from a position it computed.
    pub head: Option<Check<HeadOutput>>,
    /// The response against the sampled tokens. Absent when rechecking
    /// evidence: the node does not sign the response.
    pub tokens: Option<Check<TokenOutput>>,
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
}

impl AuditOutput {
    pub(crate) fn new(response: String, receipt: Receipt, challenge: Challenge) -> Self {
        Self {
            outcome: Outcome::NoAudit,
            pass: false,
            failures: Vec::new(),
            unchecked: Vec::new(),
            notes: Vec::new(),
            response,
            receipt,
            challenge,
            binding_error: None,
            receipt_error: None,
            opening_error: None,
            request_errors: Vec::new(),
            layers: Vec::new(),
            roles: Vec::new(),
            anchors: Check::Unavailable("not checked".to_owned()),
            head: None,
            tokens: None,
            rows_proof: None,
            attention: None,
            recurrent: Vec::new(),
            hand_picked: false,
            open_seconds: None,
        }
    }

    /// Whether the checks of the opened values ran: they do not when the
    /// receipt or the opening is refused.
    pub(crate) fn opened(&self) -> bool {
        self.binding_error.is_none() && self.receipt_error.is_none() && self.opening_error.is_none()
    }

    /// Adds the checks up. A failure decides the outcome whatever else
    /// happened; a node that then did not open the attention rows left the
    /// audit unanswered; a check that could not run makes it no audit.
    pub(crate) fn finish(&mut self) {
        self.failures = self.collect_failures();
        self.unchecked = self.collect_unchecked();
        let unanswered = self
            .attention
            .as_ref()
            .is_some_and(|a| a.rows_unanswered.is_some());
        self.outcome = if !self.failures.is_empty() {
            Outcome::Fail
        } else if unanswered {
            Outcome::Unanswered
        } else if !self.unchecked.is_empty() {
            Outcome::NoAudit
        } else {
            Outcome::Pass
        };
        self.pass = self.outcome == Outcome::Pass;
    }

    /// The checks that failed, in a few words each: what
    /// [`Self::failures`] holds once the audit is finished.
    pub fn failures(&self) -> Vec<String> {
        self.failures.iter().map(|f| f.check.clone()).collect()
    }

    /// The checks that could not run here, and why.
    pub fn unchecked(&self) -> Vec<String> {
        self.unchecked.clone()
    }

    /// Whether the evidence shows a failure: `skippy recheck-receipt`
    /// fails too.
    pub fn provable(&self) -> bool {
        self.failures.iter().any(|f| f.provable)
    }

    fn collect_failures(&self) -> Vec<Failure> {
        let mut out = Vec::new();
        if let Some(error) = &self.binding_error {
            out.push(Failure::unprovable(format!("binding: {error}")));
        }
        if let Some(error) = &self.receipt_error {
            out.push(Failure::provable(format!("receipt: {error}")));
        }
        out.extend(
            self.request_errors
                .iter()
                .map(|error| Failure::unprovable(format!("request: {error}"))),
        );
        if let Some(error) = &self.opening_error {
            out.push(Failure::provable(format!("opening: {error}")));
        }
        if !self.opened() {
            return out;
        }
        out.extend(self.roles.iter().filter(|r| !r.pass).map(|r| {
            Failure::provable(format!(
                "weight rows of {}: {}/{} fail",
                r.role, r.failing_tokens, r.tokens
            ))
        }));
        for layer in &self.layers {
            out.extend(layer.failures().into_iter().map(Failure::provable));
        }
        match &self.anchors {
            Check::Verdict(anchors) if !anchors.pass => {
                out.push(Failure::provable("input of layer 0"));
            }
            Check::NodeFault(fault) => {
                out.push(Failure::provable(format!("input of layer 0: {fault}")));
            }
            _ => {}
        }
        out.extend(self.head_failure());
        if let Some(attention) = &self.attention {
            out.extend(attention.failures().into_iter().map(Failure::provable));
        }
        out.extend(
            recurrent_audit::failures(&self.recurrent)
                .into_iter()
                .map(Failure::provable),
        );
        out.extend(self.token_failures());
        out
    }

    /// The evidence shows a token the node's own logits do not give, and
    /// logits that a check of every row of the output matrix refused,
    /// which anyone repeats. Logits that only the client's key check
    /// refused convince nobody who lacks the key. A recheck of the
    /// evidence checks every row, and shows them when they are that far
    /// off.
    fn head_failure(&self) -> Option<Failure> {
        match self.head.as_ref()? {
            Check::Verdict(head) if !head.pass => Some(Failure {
                check: "output head".to_owned(),
                provable: head
                    .positions
                    .iter()
                    .any(|p| !p.pass && !p.refused_by_the_key_alone()),
            }),
            // a recheck reads the same opening and finds the same defect
            Check::NodeFault(fault) => Some(Failure::provable(format!("output head: {fault}"))),
            _ => None,
        }
    }

    /// The node does not sign the response, and the request is not part of
    /// a receipt: these convince only the client that saw both.
    fn token_failures(&self) -> Vec<Failure> {
        let mut out = Vec::new();
        match &self.tokens {
            Some(Check::Verdict(tokens)) => {
                if !tokens.response_ok {
                    out.push(Failure::unprovable("response against the sampled tokens"));
                }
                if !tokens.prompt_ok {
                    out.push(Failure::unprovable("prompt against the receipt's tokens"));
                }
            }
            Some(Check::NodeFault(fault)) => {
                out.push(Failure::unprovable(format!("tokens: {fault}")));
            }
            _ => {}
        }
        out
    }

    fn collect_unchecked(&self) -> Vec<String> {
        if !self.opened() {
            return Vec::new();
        }
        let mut out: Vec<String> = self
            .layers
            .iter()
            .flat_map(LayerOutput::unchecked)
            .collect();
        if self.layers.len() != self.challenge.picks.len() {
            out.push("picked layers: not every one was checked".to_owned());
        }
        if let Some(reason) = self.anchors.unavailable() {
            out.push(format!("input of layer 0: {reason}"));
        }
        match &self.head {
            Some(head) => out.extend(
                head.unavailable()
                    .map(|reason| format!("output head: {reason}")),
            ),
            None if !self.challenge.head.is_empty() => {
                out.push("output head: not checked".to_owned());
            }
            None => {}
        }
        match &self.attention {
            Some(attention) => out.extend(attention.unchecked()),
            None if !self.challenge.attention.is_empty() => {
                out.push("attention: not checked".to_owned());
            }
            None => {}
        }
        out.extend(recurrent_audit::unchecked(&self.recurrent));
        if let Some(reason) = self.tokens.as_ref().and_then(Check::unavailable) {
            out.push(format!("tokens: {reason}"));
        }
        out
    }

    /// The report `skippy audit-receipt` prints.
    pub fn render_text(&self) -> String {
        let mut out = Vec::new();
        let _ = crate::report_text::render(&mut out, self);
        String::from_utf8_lossy(&out).into_owned()
    }
}

/// A receipt this verifier cannot audit, and why. It is no audit, not a
/// failed one ([`skippy_receipts::coverage`]).
#[derive(Serialize)]
pub struct NoAuditOutput {
    pub outcome: Outcome,
    pub pass: bool,
    pub receipt_id: String,
    pub reason: String,
}

impl NoAuditOutput {
    pub fn new(receipt_id: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            outcome: Outcome::NoAudit,
            pass: false,
            receipt_id: receipt_id.into(),
            reason: reason.into(),
        }
    }
}

/// A challenge the node gave no usable signed answer to: nothing was
/// checked, and that is the node's doing. Neither a pass, a failure nor no
/// audit.
#[derive(Serialize)]
pub struct UnansweredOutput {
    pub outcome: Outcome,
    pub pass: bool,
    pub receipt_id: String,
    /// What the node did not answer: `open` for the challenge, `rows` for
    /// the attention rows.
    pub request: &'static str,
    pub reason: String,
}

impl UnansweredOutput {
    pub(crate) fn new(
        receipt_id: impl Into<String>,
        request: &'static str,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            outcome: Outcome::Unanswered,
            pass: false,
            receipt_id: receipt_id.into(),
            request,
            reason: reason.into(),
        }
    }
}

#[derive(Serialize)]
pub struct HeadOutput {
    pub positions: Vec<HeadPositionOutput>,
    /// Tokens accepted only within the tie band.
    pub ties: usize,
    /// Draws that pick the sampled token only within the band.
    pub band_draws: usize,
    pub flagged: bool,
    pub pass: bool,
    /// Checking the node's logits with the client's head key, or, without
    /// one, reading the output matrix and checking them on every row.
    pub seconds: f64,
    /// The node's logits were checked with the client's head key.
    pub keyed: bool,
    /// The node's logits were checked against every row of the output
    /// matrix, with no key.
    pub every_row: bool,
    /// Building the head key, when this audit built it.
    pub key_build_seconds: Option<f64>,
    /// How the tokens were judged: the greedy choice, kept by the turn's
    /// sampler, or not judged and why.
    pub rule: String,
    /// The reasoning budget of the turn, when the node states the sampler
    /// that enforces it and the check replayed it.
    pub reasoning: Option<ReasoningOutput>,
}

/// The reasoning-budget sampler of a turn, as the node states it, and what
/// its replay found. The tags are the node's words: a client that knows
/// the model's chat template can compare them with it.
#[derive(Serialize)]
pub struct ReasoningOutput {
    /// The tokens of reasoning the sampler allows.
    pub budget: i32,
    /// The tag that starts the reasoning.
    pub start: String,
    /// The tags that end it. The sampler forces the first when the budget
    /// runs out.
    pub ends: Vec<String>,
    /// The text the chat template placed after the prompt.
    pub prefill: String,
    /// The head positions whose token the sampler forced.
    pub forced: usize,
}

impl ReasoningOutput {
    /// The replayed sampler of `forcing`, with the forced tokens among the
    /// positions of `verdict`.
    pub(crate) fn of(forcing: &Forcing, verdict: &HeadVerdict) -> Option<Self> {
        let Forcing::Replayed(replayed) = forcing else {
            return None;
        };
        let statement = &replayed.statement;
        Some(Self {
            budget: statement.budget,
            start: statement.start.clone(),
            ends: statement.ends.clone(),
            prefill: statement.prefill.clone(),
            forced: verdict
                .positions
                .iter()
                .filter(|p| p.forced.is_some())
                .count(),
        })
    }
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
    /// The key check's error relative to the logits' norm, when the node's
    /// logits were checked with the head key.
    pub key_error: Option<f64>,
    /// The error of the node's logits over every row of the output matrix,
    /// relative to their norm, when they were checked without a key.
    pub every_row_error: Option<f64>,
    /// The row check's error, when the node's logits were checked.
    pub row_error: Option<f64>,
    /// Whether the node's logits passed their check, when one ran.
    pub logits_ok: Option<bool>,
    /// Roundings of the activation the node took the other way, which
    /// the check of its logits allowed for.
    pub rounding_flips: Option<usize>,
    /// The replay of the draw that picked the token, when the turn's
    /// sampler statement lets the check replay it.
    pub draw: Option<DrawOutput>,
    /// The token the turn's reasoning budget forced here, when it forced
    /// one: the sampled token must be it.
    pub forced: Option<i32>,
    pub pass: bool,
}

/// One replayed draw: the uniform of the seed and the position, and the
/// uniforms that pick the sampled token at the widest the band allows.
#[derive(Serialize)]
pub struct DrawOutput {
    pub uniform: f64,
    pub lower: f64,
    pub upper: f64,
    /// The uniform picks the token without the band's help.
    pub exact: bool,
    pub pass: bool,
}

impl DrawOutput {
    /// The draw does not pick the token the node sampled.
    pub(crate) fn failed(&self) -> bool {
        !self.pass
    }
}

impl HeadPositionOutput {
    /// The check of the node's logits refused them.
    pub(crate) fn logits_failed(&self) -> bool {
        self.logits_ok == Some(false)
    }

    /// Only the client's key refused the node's logits: no check of every
    /// row did.
    fn refused_by_the_key_alone(&self) -> bool {
        self.logits_failed() && self.key_error.is_some()
    }
}

/// Whether the logits of every position were checked: against every row
/// of the output matrix when `every_row`, else with a head key.
fn checked(verdict: &HeadVerdict, every_row: bool) -> bool {
    verdict
        .positions
        .iter()
        .all(|p| p.logits.is_some_and(|l| l.every_row == every_row))
}

impl HeadOutput {
    pub(crate) fn new(verdict: &HeadVerdict, seconds: f64) -> Self {
        let position = |p: &PositionVerdict| HeadPositionOutput {
            position: p.position,
            sampled: p.sampled,
            best: p.best,
            margin: p.margin,
            band: p.band,
            within_band: p.within_band,
            judged: p.judged,
            key_error: p.logits.filter(|l| !l.every_row).map(|l| l.error),
            every_row_error: p.logits.filter(|l| l.every_row).map(|l| l.error),
            row_error: p.logits.map(|l| l.row_error),
            logits_ok: p.logits.map(|l| l.pass),
            rounding_flips: p.logits.map(|l| l.rounding_flips),
            draw: p.draw.map(|d| DrawOutput {
                uniform: d.uniform,
                lower: d.lower,
                upper: d.upper,
                exact: d.exact,
                pass: d.pass,
            }),
            forced: p.forced,
            pass: p.pass,
        };
        Self {
            positions: verdict.positions.iter().map(position).collect(),
            ties: verdict.ties,
            band_draws: verdict.band_draws,
            flagged: verdict.flagged,
            pass: verdict.pass,
            seconds,
            keyed: checked(verdict, false),
            every_row: checked(verdict, true),
            key_build_seconds: None,
            rule: String::new(),
            reasoning: None,
        }
    }
}

#[derive(Serialize)]
pub struct LayerOutput {
    pub layer: u32,
    pub matrices: Vec<MatrixOutput>,
    /// Why the row check found the opening defective: a failure.
    pub rows_fault: Option<String>,
    /// Why the rows could not be checked here.
    pub rows_error: Option<String>,
    pub glue: Vec<GlueOutput>,
    pub glue_pass: bool,
    /// Why the glue check found the opening defective: a failure.
    pub glue_fault: Option<String>,
    /// Why the glue could not be checked here.
    pub glue_error: Option<String>,
    /// Weights read inside token-mixing ops, which no check covers.
    pub unchecked_weights: Vec<String>,
    pub cost: CostOutput,
}

impl LayerOutput {
    /// The layer's failures: a defect of its opening, or a glue comparison
    /// that does not hold. Row failures are judged per role, over every
    /// opened layer.
    fn failures(&self) -> Vec<String> {
        let mut out = Vec::new();
        if let Some(fault) = &self.rows_fault {
            out.push(format!("weight rows of layer {}: {fault}", self.layer));
        }
        match &self.glue_fault {
            Some(fault) => out.push(format!("glue of layer {}: {fault}", self.layer)),
            None if self.glue_error.is_none() && !self.glue_pass => {
                out.push(format!("glue of layer {}", self.layer));
            }
            None => {}
        }
        out
    }

    fn unchecked(&self) -> Vec<String> {
        let mut out = Vec::new();
        if let Some(reason) = &self.glue_error {
            out.push(format!("glue of layer {}: {reason}", self.layer));
        }
        if let Some(reason) = &self.rows_error {
            out.push(format!("weight rows of layer {}: {reason}", self.layer));
        }
        out
    }
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
    /// Whether the prompt is in the text of the tokens before the sampled
    /// ones: the receipt's prompt tokens, after the tokens of the cached
    /// prefix when the turn continued one.
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
#[derive(Clone, Default, Serialize)]
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
    pub(crate) fn eval_ms_per_position(&self) -> f64 {
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
    pub repaired_tokens: usize,
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
            repaired_tokens: m.repaired_tokens,
            max_error: m.max_error,
            best_profile: m.best_profile.to_string(),
        }
    }
}

#[cfg(test)]
mod tests;
