//! Client-side checks that bind an opening's matmuls to the committed
//! residual stream and token-mixing values.
//!
//! A replay recomputes the challenged layer, so its matmul columns pass the
//! row checks no matter which weights served the turn. The glue check ties
//! them to the committed values instead. The auditor computes the layer
//! itself, with its own llama.cpp and the claimed GGUF, on the opened
//! values: every weight matmul output, token-mixing output and selection is
//! the opened value, so no large weight is read. Everything between them
//! (norms, RoPE, activations, routing, residual adds) is the auditor's own
//! computation, which it compares with the opened matmul inputs, expert
//! ids, mixing inputs, stored KV rows and the residual leaving the layer
//! (see `mesh/docs/design/RECEIPTS_GLUE.md`). No code here names an
//! architecture or a tensor.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use anyhow::{Context, Result};

use crate::attention::AttentionDescriptor;
use crate::cache_types;
use crate::error_bound::{exceeds, worst, worst_of};
use crate::layer_tensors::TensorKind;
use crate::node_fault::{BlameNode, NodeFault, ensure_node};
use crate::wire::{AnchorOpening, LayerOpening, LeafOpening, MixingKind, TensorOpening};

mod plan;

pub use plan::PlannedMatmul;

/// A glue comparison fails above this error, relative to the largest value
/// of the auditor's vector.
pub(crate) const GLUE_TOLERANCE: f64 = 1e-4;
/// Rotary embeddings turn a value by an angle of up to its position in
/// radians, and backends compute the angle in f32. The turned value is then
/// off by up to about the angle's rounding, relative to its magnitude, so
/// the token-mixing values a layer can rotate (mixing inputs and stored rows,
/// such as Q and K; the check does not know which are rotated) get this much
/// more tolerance per position: 0.008 at position 32,768.
pub(crate) const ROTATION_TOLERANCE_PER_POSITION: f64 = 2.0 * f32::EPSILON as f64;
/// A stored row passes within this multiple of the auditor's own rounding
/// error into the cache type (plus [`GLUE_TOLERANCE`] of the row's norm).
pub(crate) const STORED_ROUNDING_FACTOR: f64 = 1.5;

/// Values of a tensor, column after column.
#[derive(Clone, Debug, PartialEq)]
pub enum GlueValues {
    F32(Vec<f32>),
    I32(Vec<i32>),
}

impl GlueValues {
    fn len(&self) -> usize {
        match self {
            Self::F32(v) => v.len(),
            Self::I32(v) => v.len(),
        }
    }
}

/// An opened value the evaluation uses instead of computing it.
#[derive(Clone, Debug, PartialEq)]
pub struct GlueInput {
    /// `MatmulOutput`, `MixingOutput` or `Selection`.
    pub kind: TensorKind,
    pub index: u32,
    /// Values per column.
    pub len: usize,
    pub values: GlueValues,
}

/// One layer at the opened positions, one column per position.
#[derive(Clone, Debug, PartialEq)]
pub struct GlueRequest {
    pub layer: u32,
    pub positions: Vec<i32>,
    /// Residual stream entering the layer, column after column.
    pub residual: Vec<f32>,
    pub inputs: Vec<GlueInput>,
}

/// A tensor the evaluation computed, one column per position.
#[derive(Clone, Debug, PartialEq)]
pub struct GlueTensor {
    pub layer: u32,
    pub kind: TensorKind,
    pub index: u32,
    pub flags: i32,
    /// The weight's name for a matmul input, else the graph node's name.
    pub name: String,
    /// Values per column.
    pub len: usize,
    pub values: GlueValues,
}

impl GlueTensor {
    pub(crate) fn f32_column(&self, col: usize) -> Option<&[f32]> {
        match &self.values {
            GlueValues::F32(v) => v.get(col * self.len..(col + 1) * self.len),
            GlueValues::I32(_) => None,
        }
    }

    fn i32_column(&self, col: usize) -> Option<&[i32]> {
        match &self.values {
            GlueValues::I32(v) => v.get(col * self.len..(col + 1) * self.len),
            GlueValues::F32(_) => None,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct GlueOutput {
    pub tensors: Vec<GlueTensor>,
    /// Weights read inside token-mixing regions, which no check covers.
    pub unchecked_weights: Vec<String>,
    /// The layer's attention ops, as the auditor's graph describes them.
    pub attention: Vec<AttentionDescriptor>,
}

/// How far a stored KV row is from the row the auditor computed (L2).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StoredRowError {
    pub stored: f64,
    /// The computed row's distance from its own round trip through the cache type.
    pub rounding: f64,
    pub norm: f64,
}

/// Computes one layer on opened values; implemented over libllama.
pub trait GlueEvaluator {
    fn evaluate(&self, request: &GlueRequest) -> Result<GlueOutput>;
    /// Computes the recurrent token mixing of one layer from the opened
    /// inputs at consecutive positions of one sequence and the opened
    /// states before the first (`MixingState`, one column). Reports the
    /// `MixingOutput` values and the `MixingStateOut` states left.
    fn evaluate_region(&self, request: &GlueRequest) -> Result<GlueOutput>;
    /// The residual entering layer 0 for each token at its position, column
    /// after column, computed from the tokens alone. Only an evaluator of
    /// layer 0 can compute it.
    fn entry(&self, tokens: &[i32], positions: &[i32]) -> Result<Vec<f32>>;
    /// Compares a stored row in `ggml_type` with a computed f32 row.
    fn stored_row_error(
        &self,
        ggml_type: i32,
        stored: &[u8],
        computed: &[f32],
    ) -> Result<StoredRowError>;
}

#[derive(Clone, Debug, PartialEq)]
pub struct GlueVerdict {
    pub check: String,
    /// Comparisons made, one per position (and tensor).
    pub tokens: usize,
    pub failing_tokens: usize,
    pub max_error: f64,
    /// The tensor with the largest error, or what was inconsistent.
    pub worst: String,
    /// The first failing comparisons, with their errors.
    pub failures: Vec<String>,
    pub pass: bool,
}

/// Failing comparisons a verdict lists, at most.
const LISTED_FAILURES: usize = 8;

#[derive(Clone, Debug, PartialEq)]
pub struct GlueReport {
    pub verdicts: Vec<GlueVerdict>,
    pub unchecked_weights: Vec<String>,
    /// The layer's attention ops, as the auditor's graph describes them.
    pub attention: Vec<AttentionDescriptor>,
    /// The layer's weight matmuls by index, as the auditor's graph has
    /// them: which weight each reads, and whether it is a router. The row
    /// check takes these, never the node's word.
    pub matmuls: BTreeMap<u32, PlannedMatmul>,
    /// Recurrent states the layer's token mixing reads, as the auditor's
    /// graph has them: a layer with any is checked by its region.
    pub states: usize,
    /// Wall time of the auditor's layer evaluation.
    pub eval_seconds: f64,
    pub pass: bool,
}

/// Largest difference, relative to the largest reference value. Infinite
/// when a value on either side is not a number.
pub(crate) fn relative_error(actual: &[f32], reference: &[f32]) -> f64 {
    if actual.len() != reference.len() {
        return f64::INFINITY;
    }
    let scale = worst_of(reference.iter().map(|&v| f64::from(v).abs()));
    let diff = worst_of(
        actual
            .iter()
            .zip(reference)
            .map(|(&a, &r)| (f64::from(a) - f64::from(r)).abs()),
    );
    if !diff.is_finite() || !scale.is_finite() {
        return f64::INFINITY;
    }
    if scale == 0.0 {
        return if diff == 0.0 { 0.0 } else { f64::INFINITY };
    }
    diff / scale
}

/// How far a selection is from a valid pick on the auditor's scores,
/// relative to the largest score: 0 when valid.
///
/// A full permutation (`argsort`) must be ordered by score, in either
/// direction. A shorter pick (top-k) must hold the highest scores. Scores
/// within the tolerance of each other count as ties, so either choice of a
/// real tie passes.
pub(crate) fn selection_error(ids: &[i32], scores: &[f32]) -> f64 {
    let n = scores.len();
    let mut seen = BTreeSet::new();
    let mut picked = Vec::with_capacity(ids.len());
    for &id in ids {
        match usize::try_from(id) {
            Ok(i) if i < n && seen.insert(i) => picked.push(f64::from(scores[i])),
            _ => return f64::INFINITY,
        }
    }
    let scale = worst_of(scores.iter().map(|&v| f64::from(v).abs()));
    if scale == 0.0 || !scale.is_finite() {
        return if scale == 0.0 { 0.0 } else { f64::INFINITY };
    }
    let violation = if picked.len() == n {
        let descending = picked.windows(2).map(|w| w[1] - w[0]).fold(0.0, f64::max);
        let ascending = picked.windows(2).map(|w| w[0] - w[1]).fold(0.0, f64::max);
        descending.min(ascending)
    } else {
        let lowest_picked = picked.iter().copied().fold(f64::INFINITY, f64::min);
        let highest_left = (0..n)
            .filter(|i| !seen.contains(i))
            .map(|i| f64::from(scores[i]))
            .fold(f64::NEG_INFINITY, f64::max);
        (highest_left - lowest_picked).max(0.0)
    };
    violation / scale
}

/// The stored-row rule: the opened row may be no further from the
/// auditor's value than the auditor's own rounding into the cache type
/// allows. Returns the excess over that bound, relative to the row's norm.
pub(crate) fn stored_excess(error: &StoredRowError) -> f64 {
    let bound = STORED_ROUNDING_FACTOR * error.rounding;
    // `(NaN).max(0.0)` is 0: a distance that is not a number must not pass
    if !(error.stored.is_finite() && error.rounding.is_finite() && error.norm.is_finite()) {
        return f64::INFINITY;
    }
    if error.norm == 0.0 {
        return if error.stored <= bound {
            0.0
        } else {
            f64::INFINITY
        };
    }
    (error.stored - bound).max(0.0) / error.norm
}

/// Builds the evaluation request: the opened residual, and every opened
/// matmul output, mixing output and selection. A value the layer kept no
/// row for at some position is zero there; nothing is compared at it.
///
/// An error is the node's fault: the request is made of the opening alone.
pub fn glue_request(opening: &LayerOpening) -> Result<GlueRequest> {
    opened_request(opening).or_node_fault()
}

fn opened_request(opening: &LayerOpening) -> Result<GlueRequest> {
    let leaves = &opening.leaves;
    let positions = leaves
        .iter()
        .map(|l| i32::try_from(l.position))
        .collect::<Result<Vec<_>, _>>()?;
    let mut residual = Vec::new();
    for leaf in leaves {
        residual.extend(leaf.residual_in_values()?);
    }
    let mut columns: BTreeMap<(TensorKind, u32), Vec<Option<GlueValues>>> = BTreeMap::new();
    for (col, leaf) in leaves.iter().enumerate() {
        for (key, values) in opened_inputs(leaf)? {
            columns
                .entry(key)
                .or_insert_with(|| vec![None; leaves.len()])[col] = values;
        }
    }
    let inputs = columns
        .into_iter()
        .map(|((kind, index), cols)| {
            let len = cols.iter().flatten().map(GlueValues::len).next();
            join_columns(kind, index, len.unwrap_or(0), cols)
        })
        .collect::<Result<_>>()?;
    Ok(GlueRequest {
        layer: opening.layer,
        positions,
        residual,
        inputs,
    })
}

type OpenedInputs = Vec<((TensorKind, u32), Option<GlueValues>)>;

fn opened_inputs(leaf: &LeafOpening) -> Result<OpenedInputs> {
    let mut out = Vec::new();
    for m in &leaf.matrices {
        let values = GlueValues::F32(m.output_values()?);
        out.push(((TensorKind::MatmulOutput, m.index), Some(values)));
    }
    for t in leaf.mixing.iter().filter(|t| t.kind == MixingKind::Output) {
        let values = t.f32_values()?.map(GlueValues::F32);
        out.push(((TensorKind::MixingOutput, t.index), values));
    }
    for s in &leaf.selections {
        let values = GlueValues::I32(s.ids.clone());
        out.push(((TensorKind::Selection, s.index), Some(values)));
    }
    Ok(out)
}

fn join_columns(
    kind: TensorKind,
    index: u32,
    len: usize,
    cols: Vec<Option<GlueValues>>,
) -> Result<GlueInput> {
    let mismatch = || anyhow::anyhow!("opened {} {index} changes length", kind.name());
    let values = if kind == TensorKind::Selection {
        let mut all = Vec::with_capacity(len * cols.len());
        for col in cols {
            match col {
                Some(GlueValues::I32(v)) if v.len() == len => all.extend(v),
                None => all.extend(std::iter::repeat_n(0, len)),
                _ => return Err(mismatch()),
            }
        }
        GlueValues::I32(all)
    } else {
        let mut all = Vec::with_capacity(len * cols.len());
        for col in cols {
            match col {
                Some(GlueValues::F32(v)) if v.len() == len => all.extend(v),
                None => all.extend(std::iter::repeat_n(0.0, len)),
                _ => return Err(mismatch()),
            }
        }
        GlueValues::F32(all)
    };
    Ok(GlueInput {
        kind,
        index,
        len,
        values,
    })
}

/// Collects comparisons into verdicts by check name.
#[derive(Default)]
pub(crate) struct Verdicts(pub(crate) Vec<GlueVerdict>);

impl Verdicts {
    pub(crate) fn add(&mut self, check: &str, what: &str, error: f64) {
        self.add_within(check, what, error, GLUE_TOLERANCE);
    }

    /// [`Self::add`] with its own tolerance.
    pub(crate) fn add_within(&mut self, check: &str, what: &str, error: f64, tolerance: f64) {
        let index = match self.0.iter().position(|v| v.check == check) {
            Some(index) => index,
            None => {
                self.0.push(GlueVerdict {
                    check: check.to_owned(),
                    tokens: 0,
                    failing_tokens: 0,
                    max_error: 0.0,
                    worst: String::new(),
                    failures: Vec::new(),
                    pass: true,
                });
                self.0.len() - 1
            }
        };
        let verdict = &mut self.0[index];
        verdict.tokens += 1;
        // an error that is not a number is the worst there is
        let error = worst(error, 0.0);
        if error > verdict.max_error || verdict.worst.is_empty() {
            verdict.max_error = worst(verdict.max_error, error);
            verdict.worst = what.to_owned();
        }
        if exceeds(error, tolerance) {
            verdict.failing_tokens += 1;
            verdict.pass = false;
            if verdict.failures.len() < LISTED_FAILURES {
                verdict.failures.push(format!("{what} ({error:.1e})"));
            }
        }
    }

    pub(crate) fn inconsistent(&mut self, what: String) {
        self.add("layer plan", &what, f64::INFINITY);
    }
}

/// The auditor's tensors by kind and index.
pub(crate) struct Computed<'a>(BTreeMap<(TensorKind, u32), &'a GlueTensor>);

impl<'a> Computed<'a> {
    pub(crate) fn new(output: &'a GlueOutput, layer: u32) -> Self {
        Self(
            output
                .tensors
                .iter()
                .filter(|t| {
                    // the residual reported is the one leaving the layer
                    if t.kind == TensorKind::Residual {
                        t.layer == layer + 1
                    } else {
                        t.layer == layer
                    }
                })
                .map(|t| ((t.kind, t.index), t))
                .collect(),
        )
    }

    pub(crate) fn get(&self, kind: TensorKind, index: u32) -> Option<&'a GlueTensor> {
        self.0.get(&(kind, index)).copied()
    }

    pub(crate) fn count(&self, kind: TensorKind) -> usize {
        self.indices(kind).count()
    }

    fn indices(&self, kind: TensorKind) -> impl Iterator<Item = u32> + '_ {
        self.0
            .keys()
            .filter(move |(k, _)| *k == kind)
            .map(|(_, index)| *index)
    }
}

/// Phrases of the native evaluator's refusals of a request's values: a
/// value no node of the layer takes, a mixing output left out, a value of
/// another length or type, a selected id out of range, a position the
/// model has no row for. The values come from the opening, so a refusal
/// that names one is the node's fault. Any other error stays the
/// auditor's (see [`crate::node_fault`]).
const REFUSED_VALUE: [&str; 7] = [
    "matches nothing in the layer",
    "no opened value for",
    "the opened value for",
    "the opened selection for",
    "has a kind, index, length or type the evaluation does not take",
    "is not a position of the model",
    "a region's positions must be consecutive",
];

/// `error` of an evaluation of opened values, blamed on the node when the
/// evaluator refused one of them, else on the auditor (`what` it could not
/// compute).
pub(crate) fn blame_refused_values(error: anyhow::Error, what: &str) -> anyhow::Error {
    let message = format!("{error:#}");
    if REFUSED_VALUE.iter().any(|phrase| message.contains(phrase)) {
        return NodeFault::error(format!(
            "the auditor's {what} refuses an opened value: {message}"
        ));
    }
    error.context(format!("the auditor cannot compute the {what}"))
}

/// The request with the values the node committed to alone: the residual
/// and the token-mixing outputs. Matmul outputs and selections are in no
/// leaf, so the node chooses them after it sees the challenge; the
/// evaluator computes them as zeros here.
fn committed_request(request: &GlueRequest) -> GlueRequest {
    GlueRequest {
        inputs: request
            .inputs
            .iter()
            .filter(|input| input.kind == TensorKind::MixingOutput)
            .cloned()
            .collect(),
        ..request.clone()
    }
}

/// Evaluates the challenged layer on the opening and compares.
///
/// The layer is evaluated twice. First on the committed values alone,
/// which gives the auditor's plan of the layer before any value the node
/// chose after the challenge reaches the evaluator; the opening must fit
/// that plan. Then on every opened value. Once the committed values
/// evaluate, an evaluation the opening's other values stop fails the
/// layer: the same layer took everything else. So a node cannot turn a
/// failing layer into one the auditor could not compute. Both failures
/// keep the plan, so the row check still reads it.
///
/// An error that blames the node ([`crate::node_fault::NodeFault`]) is a
/// defect of the opening. Any other error is the auditor's: it could not
/// compute the layer.
pub fn check_glue(evaluator: &dyn GlueEvaluator, opening: &LayerOpening) -> Result<GlueReport> {
    ensure_node!(
        !opening.leaves.is_empty(),
        "the opening has no positions to check"
    );
    let request = glue_request(opening)?;
    let planned = evaluator
        .evaluate(&committed_request(&request))
        .map_err(|error| blame_refused_values(error, "challenged layer"))?;
    let mut verdicts = Verdicts::default();
    plan::check(
        opening,
        &Computed::new(&planned, opening.layer),
        &mut verdicts,
    );
    if verdicts.0.iter().any(|v| !v.pass) {
        // the evaluation would take values the layer does not have
        return Ok(report(verdicts, planned, opening.layer, 0.0));
    }
    let started = Instant::now();
    let output = match evaluator.evaluate(&request) {
        Ok(output) => output,
        Err(error) => {
            verdicts.inconsistent(format!(
                "the layer computes the committed values and refuses the matmul outputs or \
                 selections opened with them: {error:#}"
            ));
            return Ok(report(verdicts, planned, opening.layer, 0.0));
        }
    };
    let eval_seconds = started.elapsed().as_secs_f64();
    let computed = Computed::new(&output, opening.layer);
    for (col, leaf) in opening.leaves.iter().enumerate() {
        compare_position(evaluator, leaf, col, &computed, &mut verdicts)?;
    }
    Ok(report(verdicts, output, opening.layer, eval_seconds))
}

fn report(verdicts: Verdicts, output: GlueOutput, layer: u32, eval_seconds: f64) -> GlueReport {
    let computed = Computed::new(&output, layer);
    let states = computed.count(TensorKind::MixingState);
    let matmuls = plan::matmuls(&computed);
    let pass = verdicts.0.iter().all(|v| v.pass);
    GlueReport {
        verdicts: verdicts.0,
        unchecked_weights: output.unchecked_weights,
        attention: output.attention,
        matmuls,
        states,
        eval_seconds,
        pass,
    }
}

/// The auditor's own row for a stored value, compared with the opened one.
fn stored_error(
    evaluator: &dyn GlueEvaluator,
    opened: &TensorOpening,
    reference: &[f32],
) -> Result<f64> {
    let Ok(Some(stored)) = opened.bytes() else {
        return Ok(f64::INFINITY);
    };
    // the evaluator reads as many bytes as the type packs the auditor's row
    // into: a shorter row must never reach it
    let whole = cache_types::decoded(opened.ggml_type)
        .and_then(|ggml_type| cache_types::row_bytes(ggml_type, reference.len()));
    if whole != Some(stored.len()) {
        return Ok(f64::INFINITY);
    }
    let error = evaluator.stored_row_error(opened.ggml_type, &stored, reference)?;
    Ok(stored_excess(&error))
}

fn compare_position(
    evaluator: &dyn GlueEvaluator,
    leaf: &LeafOpening,
    col: usize,
    computed: &Computed<'_>,
    verdicts: &mut Verdicts,
) -> Result<()> {
    let at = |what: &str| format!("position {}: {what}", leaf.position);
    let rotated = GLUE_TOLERANCE + ROTATION_TOLERANCE_PER_POSITION * f64::from(leaf.position);
    for m in &leaf.matrices {
        if let Some(t) = computed.get(TensorKind::MatmulInput, m.index) {
            let error = t.f32_column(col).map_or(f64::INFINITY, |reference| {
                m.input_values()
                    .map_or(f64::INFINITY, |opened| relative_error(&opened, reference))
            });
            verdicts.add("matmul inputs", &at(&m.weight), error);
        }
        let ids = computed
            .get(TensorKind::ExpertIds, m.index)
            .and_then(|t| t.i32_column(col));
        if ids.is_some() || m.experts.is_some() {
            let same = ids.is_some() && ids == m.experts.as_deref();
            let error = if same { 0.0 } else { f64::INFINITY };
            verdicts.add("expert ids", &at(&m.weight), error);
        }
    }
    for t in &leaf.mixing {
        let kind = t.kind.tensor_kind();
        let what = at(&format!("{} {}", kind.name(), t.index));
        let reference = computed.get(kind, t.index).and_then(|c| c.f32_column(col));
        match (t.kind, reference) {
            // the glue takes the outputs, and states belong to the region
            (MixingKind::Output | MixingKind::State, _) => {}
            (_, None) => verdicts.add(&format!("{}s", kind.name()), &what, f64::INFINITY),
            // a value the layer kept no row for is absent; the plan check
            // fails one that is absent on a kept row
            (_, Some(_)) if t.values.is_none() => {}
            (MixingKind::Input, Some(reference)) => {
                let error = match t.f32_values() {
                    Ok(Some(opened)) => relative_error(&opened, reference),
                    _ => f64::INFINITY,
                };
                verdicts.add_within("mixing inputs", &what, error, rotated);
            }
            (MixingKind::Stored, Some(reference)) => {
                let error = stored_error(evaluator, t, reference)?;
                verdicts.add_within("stored rows", &what, error, rotated);
            }
        }
    }
    for s in &leaf.selections {
        let scores = computed
            .get(TensorKind::SelectionScores, s.index)
            .and_then(|t| t.f32_column(col));
        let error = scores.map_or(f64::INFINITY, |scores| selection_error(&s.ids, scores));
        verdicts.add("selections", &at(&format!("selection {}", s.index)), error);
    }
    if leaf.residual_out.is_some() {
        let reference = computed
            .get(TensorKind::Residual, 0)
            .and_then(|t| t.f32_column(col));
        let error = match (leaf.residual_out_values(), reference) {
            (Ok(Some(opened)), Some(reference)) => relative_error(&opened, reference),
            _ => f64::INFINITY,
        };
        verdicts.add("residual out", &at("residual leaving the layer"), error);
    }
    Ok(())
}

/// Recomputes the residual entering layer 0 from each anchor's token and
/// compares it with the opened one. This ties the committed residual
/// stream to the tokens of the request.
pub fn check_anchors(
    evaluator: &dyn GlueEvaluator,
    anchors: &[AnchorOpening],
) -> Result<GlueVerdict> {
    let mut verdicts = Verdicts::default();
    if anchors.is_empty() {
        verdicts.add("anchors", "no anchor was opened", f64::INFINITY);
        return Ok(verdicts.0.remove(0));
    }
    let tokens: Vec<i32> = anchors.iter().map(|a| a.token).collect();
    let positions = anchors
        .iter()
        .map(|a| i32::try_from(a.position))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| NodeFault::error("an anchor's position is past any sequence"))?;
    let computed = evaluator
        .entry(&tokens, &positions)
        .context("the auditor cannot compute the input of layer 0")?;
    let len = computed.len() / anchors.len();
    for (col, anchor) in anchors.iter().enumerate() {
        let what = format!("position {}: token {}", anchor.position, anchor.token);
        let error = anchor.residual_values().map_or(f64::INFINITY, |opened| {
            relative_error(&opened, &computed[col * len..(col + 1) * len])
        });
        verdicts.add("anchors", &what, error);
    }
    Ok(verdicts.0.remove(0))
}

#[cfg(test)]
#[path = "glue/tests.rs"]
mod tests;
