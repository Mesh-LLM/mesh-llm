//! The output head check with a head key: the client checks the logits the
//! node opened instead of computing the whole output matrix.
//!
//! The key holds [`KEY_VECTORS`] secret ±1 vectors `r` over the vocabulary
//! and, for each, `rᵀW` over the output matrix `W`: a few vectors of the
//! model width, built in one pass over the matrix. For opened logits `y`
//! at a residual whose normed input is `x`, an honest node has
//! `r · y = (rᵀW) · x` up to rounding (Freivalds). A node that changes any
//! logit by more than the rounding shows up, since it does not know `r`.
//! The check also compares a few rows of `W` with `y` like any matmul,
//! always including the sampled token's, and judges the greedy choice on
//! `y` itself.
//!
//! The key must stay secret, so it lives only on the client. Someone who
//! rechecks evidence without it compares the logits with every row of the
//! output matrix instead ([`super::every_row`]), which accepts the logits
//! this check accepts.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use super::opened_logits::{
    self, LogitsCheck, checked_rows, dot, picked_rows, scale_candidates, shown_scale, unbiased,
};
use super::reasoning_budget::Forcing;
use super::sampling::TokenRule;
use super::{HeadVerdict, OutputHead, judge_position, verdict};
use crate::activation_rounding::Profile;
use crate::audit;
use crate::audit::rounding_flips::{self, Flip};
use crate::commitment::hex_array;
use crate::error_bound::within;
use crate::rounding_profiles::BackendFamily;
use crate::weights::WeightFile;
use crate::wire::{HeadOpening, Receipt};

/// Secret vectors per key. A change of a logit that another change
/// cancels in every vector needs two rows whose signs agree in all of
/// them: 1 in 65,536 for a pair of rows the node picks blind.
pub(crate) const KEY_VECTORS: usize = 16;
/// The layout of a key file. A file of another layout, such as the first
/// one, which had 4 vectors and no version, is built again: it is never
/// read as this one.
pub(crate) const KEY_VERSION: u32 = 2;
/// The key check fails above this error, relative to the logits' norm.
/// Honest Qwen3-4B Q8_0 logits on CUDA were within 1.5e-7. A change that
/// steers the greedy choice must move a logit past the tie band, about
/// 1e-2 of the logits' RMS, or some 3e-5 of their norm over a vocabulary
/// of 150k, so it shows up above this.
pub(crate) const KEY_TOLERANCE: f64 = 1e-5;

/// A client's secret head key for one claimed model.
#[derive(Clone, Serialize, Deserialize, PartialEq)]
pub struct HeadKey {
    /// [`KEY_VERSION`]. A file without it does not parse.
    version: u32,
    /// The output matrix the key is for, and its shape and type.
    pub weight: String,
    pub dims: Vec<u64>,
    pub ggml_type: u32,
    /// Hex seed of the secret signs.
    seed: String,
    /// Per vector, `rᵀW` as base64 little-endian f64.
    vectors: Vec<String>,
}

/// The key without its secrets: a log line or a panic message must not
/// hold the seed, or the vectors, from which the signs follow.
impl std::fmt::Debug for HeadKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HeadKey")
            .field("weight", &self.weight)
            .field("dims", &self.dims)
            .field("ggml_type", &self.ggml_type)
            .finish_non_exhaustive()
    }
}

/// The key in a usable form.
struct Key {
    /// Per row, the sign bits of the vectors.
    signs: Vec<u16>,
    vectors: Vec<Vec<f64>>,
}

impl HeadKey {
    /// Builds a key from `seed`, which must be secret and fresh, in one pass
    /// over the output matrix. The pass reads the matrix once: a sparse
    /// copy does not keep it.
    pub fn build(weights: &WeightFile, seed: [u8; 32]) -> Result<Self> {
        let weight = super::head_weight(weights);
        let tensor = weights.tensor(weight)?;
        let (rows, width) = (tensor.rows(), usize::try_from(tensor.row_len())?);
        let signs = signs(&seed, rows);
        let threads = std::thread::available_parallelism()
            .map_or(1, std::num::NonZero::get)
            .min(16) as u64;
        let span = rows.div_ceil(threads).max(1);
        let parts = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..rows)
                .step_by(span as usize)
                .map(|first| {
                    let signs = &signs;
                    scope.spawn(move || -> Result<Vec<Vec<f64>>> {
                        let mut sums = vec![vec![0.0f64; width]; KEY_VECTORS];
                        let range = first..(first + span).min(rows);
                        weights.stream_rows(weight, range, &mut |row, values| {
                            for (k, sum) in sums.iter_mut().enumerate() {
                                let r = sign(signs, row, k);
                                for (s, w) in sum.iter_mut().zip(values) {
                                    *s += r * w;
                                }
                            }
                        })?;
                        Ok(sums)
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .unwrap_or_else(|_| bail!("a head key thread panicked"))
                })
                .collect::<Result<Vec<_>>>()
        })?;
        let mut vectors = vec![vec![0.0f64; width]; KEY_VECTORS];
        for part in parts {
            for (all, sums) in vectors.iter_mut().zip(part) {
                for (a, s) in all.iter_mut().zip(sums) {
                    *a += s;
                }
            }
        }
        Ok(Self {
            version: KEY_VERSION,
            weight: weight.to_owned(),
            dims: tensor.dims.clone(),
            ggml_type: tensor.ggml_type,
            seed: hex::encode(seed),
            vectors: vectors.iter().map(|v| encode_f64(v)).collect(),
        })
    }

    /// Whether the key is for the output matrix of `weights` and holds a
    /// seed and vectors a check can use. A key file that parses but fails
    /// this is built again; kept, it would leave the head unchecked for
    /// good.
    pub fn fits(&self, weights: &WeightFile) -> bool {
        let weight = super::head_weight(weights);
        weights.tensor(weight).is_ok_and(|t| {
            self.version == KEY_VERSION
                && self.weight == weight
                && t.dims == self.dims
                && t.ggml_type == self.ggml_type
                && usize::try_from(t.row_len())
                    .is_ok_and(|width| self.usable(t.rows(), width).is_ok())
        })
    }

    fn usable(&self, rows: u64, width: usize) -> Result<Key> {
        let seed = hex_array(&self.seed, "the head key's seed")?;
        let vectors = self
            .vectors
            .iter()
            .map(|v| decode_f64(v))
            .collect::<Result<Vec<_>>>()?;
        let finite = |v: &Vec<f64>| v.iter().all(|x| x.is_finite());
        if vectors.len() != KEY_VECTORS || vectors.iter().any(|v| v.len() != width || !finite(v)) {
            bail!("the head key does not fit the output matrix");
        }
        Ok(Key {
            signs: signs(&seed, rows),
            vectors,
        })
    }
}

/// Checks the logits the final stage opened at each head position with
/// `key`, and the token sampled from each by `rule`, for a stage on `backend`. The
/// checked rows come from `rows_seed`.
///
/// A position without logits is the node's fault: a node may not leave
/// them out to be judged on logits the client computes with other
/// rounding. So is a
/// residual or a logit vector of the wrong length, or one that holds a
/// value that is not a number ([`crate::node_fault::NodeFault`]). Any
/// other error is the auditor's.
pub fn check_head_with_key(
    weights: &WeightFile,
    key: &HeadKey,
    receipt: &Receipt,
    openings: &[HeadOpening],
    backend: &str,
    rows_seed: &[u8],
    (rule, forcing): (&TokenRule, &Forcing),
) -> Result<HeadVerdict> {
    if !key.fits(weights) {
        bail!("the head key is for another output matrix");
    }
    let head = OutputHead::from_weights(weights)?;
    let tensor = weights.tensor(&head.weight)?;
    let family = BackendFamily::checked(backend)?;
    let profiles = audit::checked_profiles(family, tensor.ggml_type)?;
    let usable = key.usable(tensor.rows(), usize::try_from(tensor.row_len())?)?;
    let picked = picked_rows(rows_seed, tensor.rows());
    let scales = scale_candidates(weights);
    let checker = LogitsChecker {
        weights,
        head: &head,
        key: &usable,
        profiles,
        scales: &scales,
    };
    let mut positions = Vec::with_capacity(openings.len());
    for opening in openings {
        let sampled = receipt
            .sampled_after(opening.position)
            .with_context(|| format!("no token was sampled at position {}", opening.position))?;
        let opened = opened_logits::read(&head, tensor.rows(), opening)?;
        let rows = checked_rows(&picked, sampled, tensor.rows());
        let check = check_logits(&checker, &opened.activation, &opened.logits, &rows)?;
        let as_f64: Vec<f64> = opened.logits.iter().map(|&v| f64::from(v)).collect();
        // the opened logits are the ones the node sampled from, scale and all
        let mut judged = judge_position(rule, forcing, opening.position, &as_f64, sampled)?;
        judged.pass &= check.pass;
        judged.logits = Some(check);
        positions.push(judged);
    }
    Ok(verdict(positions))
}

/// What checking one position's logits needs besides them.
struct LogitsChecker<'a> {
    weights: &'a WeightFile,
    head: &'a OutputHead,
    key: &'a Key,
    profiles: &'a [Profile],
    scales: &'a [f64],
}

fn check_logits(c: &LogitsChecker<'_>, x: &[f32], y: &[f32], rows: &[u64]) -> Result<LogitsCheck> {
    let mut ranges: Vec<(u64, u64)> = Vec::with_capacity(rows.len());
    for &row in rows {
        ranges.push(c.weights.row_range(&c.head.weight, row)?);
    }
    c.weights.prefetch(&ranges)?;
    let weight_rows = rows
        .iter()
        .map(|&row| Ok((row as usize, c.weights.row(&c.head.weight, row)?)))
        .collect::<Result<Vec<_>>>()?;
    let mut best: Option<LogitsCheck> = None;
    for &profile in c.profiles {
        let check = check_profile(c, profile, x, y, &weight_rows);
        if best.is_none_or(|b| score(&check) < score(&b)) {
            best = Some(check);
        }
    }
    best.context("no rounding profile to check the logits with")
}

/// [`check_logits`] for one profile.
fn check_profile(
    c: &LogitsChecker<'_>,
    profile: Profile,
    x: &[f32],
    y: &[f32],
    weight_rows: &[(usize, Vec<f64>)],
) -> LogitsCheck {
    let mut xp = profile.apply(x);
    let reference: Vec<(usize, f64)> = weight_rows
        .iter()
        .map(|(r, w)| (*r, dot(w, &xp) + c.head.bias_of(*r)))
        .collect();
    let scale = shown_scale(&reference, y, c.scales);
    // the logits the matrix alone gives, before any scale or bias
    let unbiased = unbiased(c.head, y, scale);
    let norm = opened_logits::norm(&unbiased);
    let residuals: Vec<f64> = c
        .key
        .vectors
        .iter()
        .enumerate()
        .map(|(k, v)| signed_sum(c.key, k, &unbiased) - dot(v, &xp))
        .collect();
    let flips = rounding_flips::flips(profile, x, &xp);
    let (taken, key_error) = explained(c.key, &residuals, &flips);
    for flip in taken.iter().map(|&f| &flips[f]) {
        for &(i, change) in &flip.changes {
            xp[i] += change;
        }
    }
    let checked: Vec<(usize, &[f64])> = weight_rows
        .iter()
        .map(|(r, w)| (*r, w.as_slice()))
        .collect();
    let key_error = key_error / norm;
    let row_error = audit::column_error(&checked, &xp, &unbiased);
    LogitsCheck {
        error: key_error,
        row_error,
        profile: profile.name(),
        rounding_flips: taken.len(),
        every_row: false,
        pass: within(key_error, KEY_TOLERANCE) && within(row_error, audit::TOLERANCE),
    }
}

/// The flips that best explain the key's `residuals`, at most
/// two of them, and the largest residual they leave.
fn explained(key: &Key, residuals: &[f64], flips: &[Flip]) -> (Vec<usize>, f64) {
    let effects: Vec<Vec<f64>> = flips
        .iter()
        .map(|flip| {
            key.vectors
                .iter()
                .map(|v| flip.changes.iter().map(|&(i, d)| v[i] * d).sum())
                .collect()
        })
        .collect();
    rounding_flips::best_explanation(residuals, &effects)
}

/// `r · y` for the key's `k`-th vector `r`.
fn signed_sum(key: &Key, k: usize, y: &[f32]) -> f64 {
    y.iter()
        .enumerate()
        .map(|(i, &u)| sign(&key.signs, i as u64, k) * f64::from(u))
        .sum()
}

/// How far a key check's `error` was under [`KEY_TOLERANCE`], in whole
/// powers of ten from -3 to 3; below 0 it was over. A report shows this and
/// never the error: a node that read the errors of its probes in shared
/// reports could learn which rows share the key's signs.
pub fn key_margin(error: f64) -> i8 {
    if error.is_nan() {
        return -3;
    }
    if error <= 0.0 {
        return 3;
    }
    (KEY_TOLERANCE / error).log10().floor().clamp(-3.0, 3.0) as i8
}

/// Orders checks by how far they are over their tolerances.
fn score(check: &LogitsCheck) -> f64 {
    (check.error / KEY_TOLERANCE).max(check.row_error / audit::TOLERANCE)
}

/// Per row, [`KEY_VECTORS`] sign bits from `seed`.
fn signs(seed: &[u8; 32], rows: u64) -> Vec<u16> {
    let mut hasher = blake3::Hasher::new_keyed(seed);
    hasher.update(b"mesh-llm receipts head key v2");
    let mut bytes = vec![0u8; rows as usize * 2];
    hasher.finalize_xof().fill(&mut bytes);
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| u16::from_le_bytes(*b))
        .collect()
}

fn sign(signs: &[u16], row: u64, k: usize) -> f64 {
    if signs[row as usize] >> k & 1 == 1 {
        1.0
    } else {
        -1.0
    }
}

fn encode_f64(values: &[f64]) -> String {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    crate::wire::encode_bytes(&bytes)
}

fn decode_f64(text: &str) -> Result<Vec<f64>> {
    let bytes = crate::wire::decode_bytes(text).context("decode a head key vector")?;
    Ok(bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| f64::from_le_bytes(*c))
        .collect())
}

#[cfg(test)]
mod tests;
