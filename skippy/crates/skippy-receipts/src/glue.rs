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

use anyhow::{Context, Result, bail};

use crate::layer_tensors::{FLAG_ROUTER, TensorKind};
use crate::wire::{LeafOpening, MixingKind, Opening};

/// A glue comparison fails above this error, relative to the largest value
/// of the auditor's vector.
pub const GLUE_TOLERANCE: f64 = 1e-4;
/// A stored row passes within this multiple of the auditor's own rounding
/// error into the cache type (plus [`GLUE_TOLERANCE`] of the row's norm).
pub const STORED_ROUNDING_FACTOR: f64 = 1.5;

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
    fn f32_column(&self, col: usize) -> Option<&[f32]> {
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
    /// Wall time of the auditor's layer evaluation.
    pub eval_seconds: f64,
    pub pass: bool,
}

/// Largest difference, relative to the largest reference value.
pub fn relative_error(actual: &[f32], reference: &[f32]) -> f64 {
    if actual.len() != reference.len() {
        return f64::INFINITY;
    }
    let scale = reference
        .iter()
        .fold(0.0f64, |m, &v| m.max(f64::from(v).abs()));
    let diff = actual.iter().zip(reference).fold(0.0f64, |m, (&a, &r)| {
        m.max((f64::from(a) - f64::from(r)).abs())
    });
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
pub fn selection_error(ids: &[i32], scores: &[f32]) -> f64 {
    let n = scores.len();
    let mut seen = BTreeSet::new();
    let mut picked = Vec::with_capacity(ids.len());
    for &id in ids {
        match usize::try_from(id) {
            Ok(i) if i < n && seen.insert(i) => picked.push(f64::from(scores[i])),
            _ => return f64::INFINITY,
        }
    }
    let scale = scores
        .iter()
        .fold(0.0f64, |m, &v| m.max(f64::from(v).abs()));
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
pub fn stored_excess(error: &StoredRowError) -> f64 {
    let bound = STORED_ROUNDING_FACTOR * error.rounding;
    if !error.stored.is_finite() {
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
pub fn glue_request(opening: &Opening) -> Result<GlueRequest> {
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
struct Verdicts(Vec<GlueVerdict>);

impl Verdicts {
    fn add(&mut self, check: &str, what: &str, error: f64) {
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
        if error > verdict.max_error || verdict.worst.is_empty() {
            verdict.max_error = verdict.max_error.max(error);
            verdict.worst = what.to_owned();
        }
        if error.is_nan() || error > GLUE_TOLERANCE {
            verdict.failing_tokens += 1;
            verdict.pass = false;
            if verdict.failures.len() < LISTED_FAILURES {
                verdict.failures.push(format!("{what} ({error:.1e})"));
            }
        }
    }

    fn inconsistent(&mut self, what: String) {
        self.add("layer plan", &what, f64::INFINITY);
    }
}

/// The auditor's tensors by kind and index.
struct Computed<'a>(BTreeMap<(TensorKind, u32), &'a GlueTensor>);

impl<'a> Computed<'a> {
    fn new(output: &'a GlueOutput, layer: u32) -> Self {
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

    fn get(&self, kind: TensorKind, index: u32) -> Option<&'a GlueTensor> {
        self.0.get(&(kind, index)).copied()
    }

    fn count(&self, kind: TensorKind) -> usize {
        self.0.keys().filter(|(k, _)| *k == kind).count()
    }
}

/// Evaluates the challenged layer on the opening and compares.
pub fn check_glue(evaluator: &dyn GlueEvaluator, opening: &Opening) -> Result<GlueReport> {
    if opening.leaves.is_empty() {
        bail!("the opening has no positions to check");
    }
    let request = glue_request(opening)?;
    let started = Instant::now();
    let output = evaluator
        .evaluate(&request)
        .context("the auditor cannot compute the challenged layer")?;
    let eval_seconds = started.elapsed().as_secs_f64();
    let computed = Computed::new(&output, opening.layer);
    let mut verdicts = Verdicts::default();
    check_plan(opening, &computed, &mut verdicts);
    for (col, leaf) in opening.leaves.iter().enumerate() {
        compare_position(evaluator, leaf, col, &computed, &mut verdicts)?;
    }
    let pass = verdicts.0.iter().all(|v| v.pass);
    Ok(GlueReport {
        verdicts: verdicts.0,
        unchecked_weights: output.unchecked_weights,
        eval_seconds,
        pass,
    })
}

/// The opening must describe the layer the auditor built: the same weight
/// matmuls by index and weight, the same routers, and the same number of
/// mixing values and selections.
fn check_plan(opening: &Opening, computed: &Computed<'_>, verdicts: &mut Verdicts) {
    for leaf in &opening.leaves {
        let position = leaf.position;
        for m in &leaf.matrices {
            match computed.get(TensorKind::MatmulInput, m.index) {
                None => verdicts.inconsistent(format!(
                    "position {position}: the layer has no matmul {} ({})",
                    m.index, m.weight
                )),
                Some(t) if t.name != m.weight => verdicts.inconsistent(format!(
                    "position {position}: matmul {} reads {}, opened as {}",
                    m.index, t.name, m.weight
                )),
                Some(t) if (t.flags & FLAG_ROUTER != 0) != m.router => {
                    verdicts.inconsistent(format!(
                        "position {position}: {} is opened with the wrong router flag",
                        m.weight
                    ))
                }
                Some(_) => {}
            }
        }
        for (kind, mixing) in [
            (TensorKind::MixingInput, MixingKind::Input),
            (TensorKind::MixingStored, MixingKind::Stored),
        ] {
            let opened = leaf.mixing.iter().filter(|t| t.kind == mixing).count();
            if opened != computed.count(kind) {
                verdicts.inconsistent(format!(
                    "position {position}: {opened} opened {}s, the layer has {}",
                    kind.name(),
                    computed.count(kind)
                ));
            }
        }
        // a final layer drops the rows it does not output, with their
        // selections; the committed residual out records which rows those are
        let dropped = leaf.residual_out.is_none() && leaf.selections.is_empty();
        if leaf.selections.len() != computed.count(TensorKind::SelectionScores) && !dropped {
            verdicts.inconsistent(format!(
                "position {position}: {} opened selections, the layer has {}",
                leaf.selections.len(),
                computed.count(TensorKind::SelectionScores)
            ));
        }
    }
}

fn compare_position(
    evaluator: &dyn GlueEvaluator,
    leaf: &LeafOpening,
    col: usize,
    computed: &Computed<'_>,
    verdicts: &mut Verdicts,
) -> Result<()> {
    let at = |what: &str| format!("position {}: {what}", leaf.position);
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
            (MixingKind::Output, _) => {}
            (_, None) => verdicts.add(&format!("{}s", kind.name()), &what, f64::INFINITY),
            (MixingKind::Input, Some(reference)) => {
                if let Some(opened) = t.f32_values()? {
                    verdicts.add("mixing inputs", &what, relative_error(&opened, reference));
                }
            }
            (MixingKind::Stored, Some(reference)) => {
                if let Some(stored) = t.bytes()? {
                    let error = evaluator.stored_row_error(t.ggml_type, &stored, reference)?;
                    verdicts.add("stored rows", &what, stored_excess(&error));
                }
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
    if let Some(opened) = leaf.residual_out_values()? {
        let reference = computed
            .get(TensorKind::Residual, 0)
            .and_then(|t| t.f32_column(col));
        let error = reference.map_or(f64::INFINITY, |r| relative_error(&opened, r));
        verdicts.add("residual out", &at("residual leaving the layer"), error);
    }
    Ok(())
}

#[cfg(test)]
#[path = "glue/tests.rs"]
mod tests;
