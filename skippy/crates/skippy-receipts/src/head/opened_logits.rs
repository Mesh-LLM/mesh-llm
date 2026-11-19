//! The logits a node opened at a head position, read for a check against
//! the claimed output matrix.
//!
//! Two checks read them: one with the client's secret head key
//! ([`super::key`]) and one over every row of the matrix, which needs no
//! secret ([`super::every_row`]). Both take the logits as the node sampled
//! from them, undo the scale and the bias a model applies after the
//! matrix, and compare a few drawn rows like any matmul. What they share
//! is here.

use anyhow::Result;

use super::OutputHead;
use crate::draw::DrawStream;
use crate::error_bound::all_finite;
use crate::node_fault::{BlameNode, ensure_node, node_fault};
use crate::weights::{MetaValue, WeightFile};
use crate::wire::HeadOpening;

/// Random rows checked per position, besides the sampled token's.
pub(crate) const HEAD_ROWS: usize = 16;

/// How far the scale the rows show may be from a candidate.
const SCALE_TOLERANCE: f64 = 1e-3;

/// The check of one head position's opened logits against the claimed
/// output matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogitsCheck {
    /// How far the logits are from the claimed head's, relative to their
    /// norm, for the best profile. With a head key it is the largest
    /// Freivalds difference over the key's vectors. Over every row it is
    /// the norm of the difference of all of them.
    pub error: f64,
    /// The row check's error over the drawn rows and the sampled token's,
    /// for the best profile.
    pub row_error: f64,
    pub profile: &'static str,
    /// Roundings of the activation the node took the other way: values
    /// within a few ulps of a rounding boundary
    /// ([`crate::audit::rounding_flips`]).
    pub rounding_flips: usize,
    /// The check read every row of the output matrix, and no head key.
    /// Anyone with the claimed GGUF repeats it.
    pub every_row: bool,
    pub pass: bool,
}

/// One head position's logits and the activation the output matrix met.
pub(super) struct Opened {
    /// The logits as the node opened them, one per token.
    pub(super) logits: Vec<f32>,
    /// The opened residual after the output norm.
    pub(super) activation: Vec<f32>,
}

/// Reads the logits and the residual of `opening` for a matrix of `rows`
/// rows. A position without logits is the node's fault: a node may not
/// leave them out to be judged on logits the client computes with other
/// rounding. So is a residual or a logit vector of the wrong length, or
/// one that holds a value that is not a number.
pub(super) fn read(head: &OutputHead, rows: u64, opening: &HeadOpening) -> Result<Opened> {
    let Some(logits) = opening.logit_values().or_node_fault()? else {
        node_fault!("position {} opened no logits", opening.position);
    };
    ensure_node!(
        logits.len() as u64 == rows,
        "position {} opened {} logits for {} tokens",
        opening.position,
        logits.len(),
        rows
    );
    ensure_node!(
        all_finite(&logits),
        "position {} opened a logit that is not a number",
        opening.position
    );
    let activation = head
        .norm
        .apply(&opening.residual_values().or_node_fault()?)?;
    Ok(Opened {
        logits,
        activation: activation.iter().map(|&v| v as f32).collect(),
    })
}

/// The uniform scale a model may apply to its logits after the matrix
/// (`logit_scale`), which leaves the greedy choice as it is. Some
/// architectures multiply by it and some divide, so a check accepts
/// either, or none, whichever the checked rows show.
pub(super) fn scale_candidates(weights: &WeightFile) -> Vec<f64> {
    let scale = weights
        .metadata
        .iter()
        .find(|(k, _)| k.ends_with(".logit_scale"))
        .and_then(|(_, v)| match v {
            MetaValue::Float(s) if *s > 0.0 => Some(*s),
            _ => None,
        });
    let mut out = vec![1.0];
    if let Some(s) = scale {
        out.extend([s, 1.0 / s]);
    }
    out
}

/// The candidate scale that the opened logits `y` show against the
/// reference logits of the checked rows. When none fits, 1: the logits are
/// then checked as they are, and the errors show how far off they are.
pub(super) fn shown_scale(reference: &[(usize, f64)], y: &[f32], scales: &[f64]) -> f64 {
    let (dot, sq) = reference.iter().fold((0.0, 0.0), |(d, s), &(r, v)| {
        (d + f64::from(y[r]) * v, s + v * v)
    });
    if sq == 0.0 {
        return 1.0;
    }
    let shown = dot / sq;
    scales
        .iter()
        .copied()
        .find(|&c| (shown / c - 1.0).abs() <= SCALE_TOLERANCE)
        .unwrap_or(1.0)
}

/// The logits the matrix alone gives: `y` without the scale and the bias.
pub(super) fn unbiased(head: &OutputHead, y: &[f32], scale: f64) -> Vec<f32> {
    y.iter()
        .enumerate()
        .map(|(i, &v)| (f64::from(v) / scale - head.bias_of(i)) as f32)
        .collect()
}

/// The norm of `values`, and never 0, so an error relative to it is a
/// number.
pub(super) fn norm(values: &[f32]) -> f64 {
    values
        .iter()
        .map(|&v| f64::from(v).powi(2))
        .sum::<f64>()
        .sqrt()
        .max(f64::MIN_POSITIVE)
}

pub(super) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

/// `count` distinct rows of `rows`, from the stream `domain` of `seed`.
pub(super) fn drawn_rows(seed: &[u8], domain: &str, count: usize, rows: u64) -> Vec<u64> {
    let mut stream = DrawStream::new(seed, domain);
    let mut picked = std::collections::BTreeSet::new();
    while picked.len() < count.min(usize::try_from(rows).unwrap_or(usize::MAX)) {
        picked.insert(stream.below(rows));
    }
    picked.into_iter().collect()
}

/// [`HEAD_ROWS`] distinct rows of `rows`, from `seed`: the rows of the row
/// check.
pub(super) fn picked_rows(seed: &[u8], rows: u64) -> Vec<u64> {
    drawn_rows(seed, "rows/output-head", HEAD_ROWS, rows)
}

/// The rows of the row check at one position: the drawn ones and the
/// sampled token's.
pub(super) fn checked_rows(picked: &[u64], sampled: i32, rows: u64) -> Vec<u64> {
    let mut checked = picked.to_vec();
    checked.extend(u64::try_from(sampled).ok().filter(|&s| s < rows));
    checked
}
