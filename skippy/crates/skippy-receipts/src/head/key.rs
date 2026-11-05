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
//! rechecks evidence without it computes the whole head instead.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use super::{HeadVerdict, OutputHead, judge_with, token_rule, verdict};
use crate::audit::rounding_flips::{self, Flip};
use crate::audit::{self, BackendFamily, Profile};
use crate::commitment::hex_array;
use crate::draw::DrawStream;
use crate::weights::WeightFile;
use crate::wire::{HeadOpening, Receipt};

/// Secret vectors per key.
pub const KEY_VECTORS: usize = 4;
/// The key check fails above this error, relative to the logits' norm.
/// Honest Qwen3-4B Q8_0 logits on CUDA were within 1.5e-7. A change that
/// steers the greedy choice must move a logit past the tie band, about
/// 1e-2 of the logits' RMS, or some 3e-5 of their norm over a vocabulary
/// of 150k, so it shows up above this.
pub const KEY_TOLERANCE: f64 = 1e-5;
/// Random rows checked per position, besides the sampled token's.
pub const HEAD_ROWS: usize = 16;

/// A client's secret head key for one claimed model.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct HeadKey {
    /// The output matrix the key is for, and its shape and type.
    pub weight: String,
    pub dims: Vec<u64>,
    pub ggml_type: u32,
    /// Hex seed of the secret signs.
    seed: String,
    /// Per vector, `rᵀW` as base64 little-endian f64.
    vectors: Vec<String>,
}

/// The key in a usable form.
struct Key {
    /// Per row, the sign bits of the vectors.
    signs: Vec<u8>,
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
            weight: weight.to_owned(),
            dims: tensor.dims.clone(),
            ggml_type: tensor.ggml_type,
            seed: hex::encode(seed),
            vectors: vectors.iter().map(|v| encode_f64(v)).collect(),
        })
    }

    /// Whether the key is for the output matrix of `weights`.
    pub fn fits(&self, weights: &WeightFile) -> bool {
        let weight = super::head_weight(weights);
        weights.tensor(weight).is_ok_and(|t| {
            self.weight == weight && t.dims == self.dims && t.ggml_type == self.ggml_type
        })
    }

    fn usable(&self, rows: u64, width: usize) -> Result<Key> {
        let seed = hex_array(&self.seed, "the head key's seed")?;
        let vectors = self
            .vectors
            .iter()
            .map(|v| decode_f64(v))
            .collect::<Result<Vec<_>>>()?;
        if vectors.len() != KEY_VECTORS || vectors.iter().any(|v| v.len() != width) {
            bail!("the head key does not fit the output matrix");
        }
        Ok(Key {
            signs: signs(&seed, rows),
            vectors,
        })
    }
}

/// The key check of one head position.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogitsCheck {
    /// The largest Freivalds difference over the key's vectors, relative to
    /// the logits' norm, for the best profile.
    pub key_error: f64,
    /// The row check's error over the checked rows, for the best profile.
    pub row_error: f64,
    pub profile: &'static str,
    /// Roundings of the activation the node took the other way, by the
    /// key: values within a few ulps of a rounding boundary
    /// ([`rounding_flips`]).
    pub rounding_flips: usize,
    pub pass: bool,
}

/// Checks the logits the final stage opened at each head position with
/// `key`, and the token sampled from each, for a stage on `backend`. The
/// checked rows come from `rows_seed`.
pub fn check_head_with_key(
    weights: &WeightFile,
    key: &HeadKey,
    receipt: &Receipt,
    openings: &[HeadOpening],
    backend: &str,
    rows_seed: &[u8],
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
    // the opened logits are the ones the node sampled from, scale and all
    let rule = token_rule(receipt);
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
        let logits = opening
            .logit_values()?
            .with_context(|| format!("position {} opened no logits", opening.position))?;
        if logits.len() as u64 != tensor.rows() {
            bail!(
                "position {} opened {} logits for {} tokens",
                opening.position,
                logits.len(),
                tensor.rows()
            );
        }
        let x = head.norm.apply(&opening.residual_values()?)?;
        let x: Vec<f32> = x.iter().map(|&v| v as f32).collect();
        let mut rows = picked.clone();
        rows.extend(u64::try_from(sampled).ok().filter(|&s| s < tensor.rows()));
        let check = check_logits(&checker, &x, &logits, &rows)?;
        let as_f64: Vec<f64> = logits.iter().map(|&v| f64::from(v)).collect();
        let mut judged = judge_with(&rule, opening.position, &as_f64, sampled)?;
        judged.pass &= check.pass;
        judged.logits = Some(check);
        positions.push(judged);
    }
    Ok(verdict(positions))
}

/// The uniform scale a model may apply to its logits after the matrix
/// (`logit_scale`), which leaves the greedy choice as it is. Some
/// architectures multiply by it and some divide, so the check accepts
/// either, or none, whichever the checked rows show.
fn scale_candidates(weights: &WeightFile) -> Vec<f64> {
    let scale = weights
        .metadata
        .iter()
        .find(|(k, _)| k.ends_with(".logit_scale"))
        .and_then(|(_, v)| match v {
            crate::weights::MetaValue::Float(s) if *s > 0.0 => Some(*s),
            _ => None,
        });
    let mut out = vec![1.0];
    if let Some(s) = scale {
        out.extend([s, 1.0 / s]);
    }
    out
}

/// How far the scale the rows show may be from a candidate.
const SCALE_TOLERANCE: f64 = 1e-3;

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
    let bias = |i: usize| c.head.bias.as_ref().map_or(0.0, |b| b[i]);
    let mut xp = profile.apply(x);
    let reference: Vec<(usize, f64)> = weight_rows
        .iter()
        .map(|(r, w)| (*r, dot(w, &xp) + bias(*r)))
        .collect();
    let scale = shown_scale(&reference, y, c.scales);
    // the logits the matrix and bias give, before any scale
    let unbiased: Vec<f32> = y
        .iter()
        .enumerate()
        .map(|(i, &v)| (f64::from(v) / scale - bias(i)) as f32)
        .collect();
    let norm = unbiased
        .iter()
        .map(|&v| f64::from(v).powi(2))
        .sum::<f64>()
        .sqrt()
        .max(f64::MIN_POSITIVE);
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
        key_error,
        row_error,
        profile: profile.name(),
        rounding_flips: taken.len(),
        pass: key_error <= KEY_TOLERANCE && row_error <= audit::TOLERANCE,
    }
}

/// Flips past which only single ones are tried.
const PAIRED_FLIPS: usize = 32;

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
    let left = |taken: &[usize]| {
        residuals
            .iter()
            .enumerate()
            .map(|(k, &e)| (e - taken.iter().map(|&f| effects[f][k]).sum::<f64>()).abs())
            .fold(0.0f64, f64::max)
    };
    let mut best = (Vec::new(), left(&[]));
    let mut consider = |taken: Vec<usize>| {
        let error = left(&taken);
        if error < best.1 {
            best = (taken, error);
        }
    };
    for a in 0..flips.len() {
        consider(vec![a]);
        if flips.len() <= PAIRED_FLIPS {
            for b in a + 1..flips.len() {
                consider(vec![a, b]);
            }
        }
    }
    best
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

/// `r · y` for the key's `k`-th vector `r`.
fn signed_sum(key: &Key, k: usize, y: &[f32]) -> f64 {
    y.iter()
        .enumerate()
        .map(|(i, &u)| sign(&key.signs, i as u64, k) * f64::from(u))
        .sum()
}

/// The candidate scale that the opened logits `y` show against the
/// reference logits of the checked rows. When none fits, 1: the logits are
/// then checked as they are, and the key and row errors show how far off
/// they are.
fn shown_scale(reference: &[(usize, f64)], y: &[f32], scales: &[f64]) -> f64 {
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

/// Orders checks by how far they are over their tolerances.
fn score(check: &LogitsCheck) -> f64 {
    (check.key_error / KEY_TOLERANCE).max(check.row_error / audit::TOLERANCE)
}

/// [`HEAD_ROWS`] distinct rows of `rows`, from `seed`.
fn picked_rows(seed: &[u8], rows: u64) -> Vec<u64> {
    let mut stream = DrawStream::new(seed, "rows/output-head");
    let mut picked = std::collections::BTreeSet::new();
    while picked.len() < HEAD_ROWS.min(rows as usize) {
        picked.insert(stream.below(rows));
    }
    picked.into_iter().collect()
}

/// Per row, [`KEY_VECTORS`] sign bits from `seed`.
fn signs(seed: &[u8; 32], rows: u64) -> Vec<u8> {
    let mut hasher = blake3::Hasher::new_keyed(seed);
    hasher.update(b"mesh-llm receipts head key v1");
    let mut out = vec![0u8; rows as usize];
    hasher.finalize_xof().fill(&mut out);
    out
}

fn sign(signs: &[u8], row: u64, k: usize) -> f64 {
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
