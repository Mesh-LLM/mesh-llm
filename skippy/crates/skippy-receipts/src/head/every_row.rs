//! The output head check without a head key: the logits the node opened,
//! against every row of the claimed output matrix.
//!
//! A recheck of evidence has no head key, since the key is one client's
//! secret. It reads the whole output matrix instead, which shows the same
//! thing with no secret: for opened logits `y` at a residual whose normed
//! input is `x`, an honest node has `y = W · x` on every row, up to the
//! rounding of its backend's kernel ([`crate::activation_rounding`]). Then
//! the check judges the token on `y`, as the check with a key does
//! ([`super::key`]).
//!
//! The two checks are meant to give one verdict:
//!
//! - Both judge the token on the node's own logits. Neither judges it on
//!   logits it computed with other rounding.
//! - Both run the same row check on the same drawn rows and the sampled
//!   token's.
//! - A key's vectors hold a sign per row, so one changed logit moves the
//!   key's error by exactly its size. The error here is the norm of the
//!   difference over all rows, which is at least that size. So a logit
//!   moved past [`EVERY_ROW_TOLERANCE`] of the logits' norm fails both,
//!   and this check also fails changes spread over many rows that a key
//!   catches only with some probability.
//!
//! A value of the activation within a few ulps of a rounding boundary may
//! round the other way on the node ([`rounding_flips`]). The key check
//! finds those flips from its key. This check finds them from a few rows
//! ([`GUIDE_ROWS`]) and then confirms them on every row. When that fails,
//! it looks once more, at the rows that were furthest off, since a flip
//! shows on every row its weight column is not zero in. The choice of
//! flips gives a node nothing: the logits must still be within the
//! tolerance of `W` times an activation at most two flips from the
//! profile's, on every row.

use anyhow::{Context, Result};

use super::key::KEY_TOLERANCE;
use super::opened_logits::{
    self, LogitsCheck, Opened, checked_rows, dot, drawn_rows, picked_rows, scale_candidates,
    shown_scale, unbiased,
};
use super::reasoning_budget::Forcing;
use super::{DrawnFrom, HeadVerdict, OutputHead, judge_position, token_rule_for, verdict};
use crate::activation_rounding::Profile;
use crate::audit::rounding_flips;
use crate::audit::{self, BackendFamily};
use crate::error_bound::within;
use crate::weights::WeightFile;
use crate::wire::{HeadOpening, Receipt};

/// The check fails above this error, relative to the logits' norm: the
/// norm of the difference between the opened logits and the claimed
/// head's, over every row. It is the head key's tolerance, so a logit the
/// key check refuses is refused here.
///
/// Honest Qwen3-4B Q8_0 logits were within 4.3e-7 on CPU (132 positions)
/// and within 1.6e-7 on CUDA (36 positions).
pub(crate) const EVERY_ROW_TOLERANCE: f64 = KEY_TOLERANCE;

/// Rows that show which rounding flips the node took, besides the rows of
/// the row check. One flip moves every row by its weight in one column, so
/// a few rows tell the flips apart.
const GUIDE_ROWS: usize = 48;

/// One profile's reading of a position: the activation the node's kernel
/// met by it, and the logits the matrix alone gave.
struct Reading {
    profile: Profile,
    /// The profile's activation, with the rounding flips the guide rows
    /// show.
    activation: Vec<f64>,
    flips: usize,
    /// The opened logits without the scale and the bias.
    unbiased: Vec<f32>,
    /// The worst difference left on the guide rows.
    guide_error: f64,
}

/// What reading one position's logits needs besides them.
struct Reader<'a> {
    weights: &'a WeightFile,
    head: &'a OutputHead,
    profiles: &'a [Profile],
    scales: &'a [f64],
}

impl Reader<'_> {
    /// The reading of `opened` that fits the guide rows best.
    fn read(&self, opened: &Opened, guide: &[u64]) -> Result<Reading> {
        let rows = weight_rows(self.weights, &self.head.weight, guide)?;
        self.profiles
            .iter()
            .map(|&profile| self.read_as(profile, opened, &rows))
            .reduce(|best, next| {
                if next.guide_error < best.guide_error {
                    next
                } else {
                    best
                }
            })
            .context("no rounding profile to check the logits with")
    }

    /// The reading of `opened` under one profile.
    fn read_as(&self, profile: Profile, opened: &Opened, rows: &[(usize, Vec<f64>)]) -> Reading {
        let mut activation = profile.apply(&opened.activation);
        let reference: Vec<(usize, f64)> = rows
            .iter()
            .map(|(r, w)| (*r, dot(w, &activation) + self.head.bias_of(*r)))
            .collect();
        let scale = shown_scale(&reference, &opened.logits, self.scales);
        let unbiased = unbiased(self.head, &opened.logits, scale);
        let residuals: Vec<f64> = rows
            .iter()
            .map(|(r, w)| f64::from(unbiased[*r]) - dot(w, &activation))
            .collect();
        let flips = rounding_flips::flips(profile, &opened.activation, &activation);
        let effects: Vec<Vec<f64>> = flips
            .iter()
            .map(|flip| {
                rows.iter()
                    .map(|(_, w)| flip.changes.iter().map(|&(i, d)| w[i] * d).sum())
                    .collect()
            })
            .collect();
        let (taken, guide_error) = rounding_flips::best_explanation(&residuals, &effects);
        for flip in taken.iter().map(|&f| &flips[f]) {
            for &(i, change) in &flip.changes {
                activation[i] += change;
            }
        }
        Reading {
            profile,
            activation,
            flips: taken.len(),
            unbiased,
            guide_error,
        }
    }
}

/// Rows of `weight`, by index.
fn weight_rows(weights: &WeightFile, weight: &str, rows: &[u64]) -> Result<Vec<(usize, Vec<f64>)>> {
    rows.iter()
        .map(|&row| Ok((row as usize, weights.row(weight, row)?)))
        .collect()
}

/// The check of one reading against the matrix's `products` with its
/// activation, one per row, and against the `checked` rows of the row
/// check.
fn check(reading: &Reading, products: &[f64], checked: &[(usize, Vec<f64>)]) -> LogitsCheck {
    let difference: f64 = reading
        .unbiased
        .iter()
        .zip(products)
        .map(|(&y, reference)| (f64::from(y) - reference).powi(2))
        .sum();
    let error = difference.sqrt() / opened_logits::norm(&reading.unbiased);
    let rows: Vec<(usize, &[f64])> = checked.iter().map(|(r, w)| (*r, w.as_slice())).collect();
    let row_error = audit::column_error(&rows, &reading.activation, &reading.unbiased);
    LogitsCheck {
        error,
        row_error,
        profile: reading.profile.name(),
        rounding_flips: reading.flips,
        every_row: true,
        pass: within(error, EVERY_ROW_TOLERANCE) && within(row_error, audit::TOLERANCE),
    }
}

/// One head position and what its checks read.
struct Position {
    position: u32,
    sampled: i32,
    opened: Opened,
    /// The rows of the row check, with their weights.
    checked: Vec<(usize, Vec<f64>)>,
}

/// The `count` rows where the logits of `reading` are furthest from the
/// matrix's `products` with its activation.
fn furthest_rows(reading: &Reading, products: &[f64], count: usize) -> Vec<u64> {
    let mut rows: Vec<(f64, u64)> = reading
        .unbiased
        .iter()
        .zip(products)
        .enumerate()
        .map(|(row, (&y, reference))| ((f64::from(y) - reference).abs(), row as u64))
        .collect();
    rows.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    rows.truncate(count);
    rows.into_iter().map(|(_, row)| row).collect()
}

/// The checks of `readings`, one per position, in one pass over the
/// matrix, with the matrix's products with each reading's activation.
fn check_all(
    reader: &Reader<'_>,
    positions: &[&Position],
    readings: &[Reading],
) -> Result<(Vec<LogitsCheck>, Vec<Vec<f64>>)> {
    let activations: Vec<Vec<f64>> = readings.iter().map(|r| r.activation.clone()).collect();
    let products = reader.head.products(reader.weights, &activations)?;
    let checks = readings
        .iter()
        .zip(&products)
        .zip(positions)
        .map(|((reading, products), p)| check(reading, products, &p.checked))
        .collect();
    Ok((checks, products))
}

/// Reads the positions that `checks` refused once more, from the rows
/// where their logits were furthest from the `products` of the first pass,
/// and keeps a check that comes out less far off.
fn check_refused_again(
    reader: &Reader<'_>,
    positions: &[Position],
    readings: &[Reading],
    products: &[Vec<f64>],
    checks: &mut [LogitsCheck],
) -> Result<()> {
    let refused: Vec<usize> = (0..checks.len()).filter(|&i| !checks[i].pass).collect();
    if refused.is_empty() {
        return Ok(());
    }
    let mut again = Vec::with_capacity(refused.len());
    for &i in &refused {
        let guide = furthest_rows(&readings[i], &products[i], GUIDE_ROWS);
        again.push(reader.read(&positions[i].opened, &guide)?);
    }
    let of: Vec<&Position> = refused.iter().map(|&i| &positions[i]).collect();
    let (rechecked, _) = check_all(reader, &of, &again)?;
    for (&i, check) in refused.iter().zip(rechecked) {
        if check.pass || within(check.error, checks[i].error) {
            checks[i] = check;
        }
    }
    Ok(())
}

/// Checks the logits the final stage on `backend` opened at each head
/// position against every row of the claimed output matrix, and the token
/// sampled from each on those logits. The rows of the row check come from
/// `rows_seed`, as an audit draws them.
///
/// A position without logits is the node's fault, as it is for an audit:
/// a node may not leave them out to be judged on other logits. So is a
/// residual or a logit vector of the wrong length, or one that holds a
/// value that is not a number ([`crate::node_fault::NodeFault`]). Any
/// other error is the auditor's.
pub fn check_head_on_every_row(
    weights: &WeightFile,
    receipt: &Receipt,
    openings: &[HeadOpening],
    backend: &str,
    rows_seed: &[u8],
    forcing: &Forcing,
) -> Result<HeadVerdict> {
    let head = OutputHead::from_weights(weights)?;
    let tensor = weights.tensor(&head.weight)?;
    let rows = tensor.rows();
    let family = BackendFamily::checked(backend)?;
    let reader = Reader {
        weights,
        head: &head,
        profiles: audit::checked_profiles(family, tensor.ggml_type)?,
        scales: &scale_candidates(weights),
    };
    OutputHead::prefetch(weights)?;
    let picked = picked_rows(rows_seed, rows);
    let extra = drawn_rows(rows_seed, "rows/output-head-flips", GUIDE_ROWS, rows);
    let mut positions = Vec::with_capacity(openings.len());
    let mut readings = Vec::with_capacity(openings.len());
    for opening in openings {
        let sampled = receipt
            .sampled_after(opening.position)
            .with_context(|| format!("no token was sampled at position {}", opening.position))?;
        let opened = opened_logits::read(&head, rows, opening)?;
        let checked = checked_rows(&picked, sampled, rows);
        let guide: Vec<u64> = checked.iter().chain(&extra).copied().collect();
        readings.push(reader.read(&opened, &guide)?);
        positions.push(Position {
            position: opening.position,
            sampled,
            opened,
            checked: weight_rows(weights, &head.weight, &checked)?,
        });
    }
    let all: Vec<&Position> = positions.iter().collect();
    let (mut checks, products) = check_all(&reader, &all, &readings)?;
    check_refused_again(&reader, &positions, &readings, &products, &mut checks)?;
    // the opened logits are the ones the node sampled from, scale and all
    let rule = token_rule_for(weights, receipt, true);
    let mut verdicts = Vec::with_capacity(positions.len());
    for (p, checked) in positions.iter().zip(checks) {
        let logits: Vec<f64> = p.opened.logits.iter().map(|&v| f64::from(v)).collect();
        let drawn_from = DrawnFrom::Opened(&logits);
        let mut judged =
            judge_position(&rule, forcing, p.position, &logits, drawn_from, p.sampled)?;
        judged.pass &= checked.pass;
        judged.logits = Some(checked);
        verdicts.push(judged);
    }
    Ok(verdict(verdicts))
}

#[cfg(test)]
mod tests;
