//! One column check: the opened output of a matmul at one position
//! against the claimed weight rows the client drew, under one rounding
//! profile, repaired when an activation near-tie fails it
//! ([`super::near_ties`]).

use super::rounding_flips::{self, Flip};
use super::{TOLERANCE, activation_scale, block_sums, near_ties, reference_error};
use crate::activation_rounding::Profile;
use crate::dequant::MinBlock;

/// One column check: its error, the profile that gave it, and the
/// near-ties a repair rounded the other way ([`near_ties`]).
#[derive(Clone, Copy, Debug)]
pub(super) struct ColumnCheck {
    pub(super) error: f64,
    pub(super) profile: &'static str,
    pub(super) flips: usize,
}

/// The checked weight rows of one output slice, as each profile meets the
/// activation.
pub(super) struct CheckedRows<'a> {
    /// Per checked row: its index and its dequantized weights.
    pub(super) rows: Vec<(usize, &'a [f64])>,
    /// The same rows as sub-blocks, for the profiles with block sums.
    pub(super) blocks: Vec<&'a [MinBlock]>,
}

impl CheckedRows<'_> {
    /// The reference outputs of the checked rows for the activation `x`,
    /// which `p` rounds to `xq`.
    fn references(&self, p: Profile, x: &[f32], xq: &[f64]) -> Vec<(usize, f64)> {
        match p.block_sums() {
            Some(rounded) => self
                .rows
                .iter()
                .zip(&self.blocks)
                .map(|(&(r, _), blocks)| (r, block_sums::reference(blocks, x, xq, rounded)))
                .collect(),
            None => self
                .rows
                .iter()
                .map(|&(r, w)| (r, w.iter().zip(xq).map(|(w, a)| w * a).sum()))
                .collect(),
        }
    }

    /// The weight that value `i` of the rounded activation meets in the
    /// `k`-th checked row.
    fn weight(&self, p: Profile, k: usize, i: usize) -> f64 {
        match p.block_sums() {
            Some(rounded) => block_sums::activation_weight(self.blocks[k], i, rounded),
            None => self.rows[k].1[i],
        }
    }

    /// The column check of `x` against the opened output `y` under `p`,
    /// repaired when a near-tie fails it.
    pub(super) fn check(&self, p: Profile, x: &[f32], y: &[f32]) -> ColumnCheck {
        let xq = p.apply(x);
        let references = self.references(p, x, &xq);
        let floor = activation_scale(&self.rows, &xq);
        let error = reference_error(&references, y, floor);
        let unrepaired = ColumnCheck {
            error,
            profile: p.name(),
            flips: 0,
        };
        if error <= TOLERANCE {
            return unrepaired;
        }
        let flips = rounding_flips::flips(p, x, &xq);
        let effect = |k: usize, flip: &Flip| {
            flip.changes
                .iter()
                .map(|&(i, change)| self.weight(p, k, i) * change)
                .sum()
        };
        near_ties::repair(&references, y, floor, &flips, effect).map_or(unrepaired, |repair| {
            ColumnCheck {
                error: repair.error,
                flips: repair.flips,
                ..unrepaired
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_column_a_near_tie_failed_is_repaired() {
        // d = 0.01: 0.255 lies on the boundary between steps 25 and 26
        let mut x = vec![0.05f32; 32];
        x[0] = 1.27;
        x[5] = 0.255;
        let p = Profile::Q8DivHalf;
        let xq = p.apply(&x);
        let tie = rounding_flips::flips(p, &x, &xq)
            .into_iter()
            .find(|f| f.changes.iter().any(|&(i, _)| i == 5))
            .expect("value 5 is a near-tie");
        let mut node = xq.clone();
        for &(i, change) in &tie.changes {
            node[i] += change;
        }
        let w: Vec<Vec<f64>> = (0..4)
            .map(|r| {
                (0..32)
                    .map(|i| f64::from((r * 7 + i * 3) % 11) * 0.1 - 0.5)
                    .collect()
            })
            .collect();
        let y: Vec<f32> = w
            .iter()
            .map(|row| row.iter().zip(&node).map(|(a, b)| a * b).sum::<f64>() as f32)
            .collect();
        let rows = CheckedRows {
            rows: w
                .iter()
                .enumerate()
                .map(|(r, row)| (r, row.as_slice()))
                .collect(),
            blocks: Vec::new(),
        };
        assert!(reference_error(&rows.references(p, &x, &xq), &y, 0.0) > TOLERANCE);
        let check = rows.check(p, &x, &y);
        assert_eq!(check.flips, 1, "{check:?}");
        assert!(check.error <= near_ties::REPAIRED_TOLERANCE, "{check:?}");

        // other weights are not repaired
        let other: Vec<f32> = y.iter().map(|v| v * 1.05).collect();
        let check = rows.check(p, &x, &other);
        assert_eq!(check.flips, 0);
        assert!(check.error > TOLERANCE);
    }
}
