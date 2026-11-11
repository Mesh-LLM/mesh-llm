//! One column check: the opened output of a matmul at one position
//! against the claimed weight rows the client drew, under one rounding
//! profile, repaired when an activation near-tie fails it
//! ([`super::near_ties`]).

use super::rounding_flips::{self, Flip};
use super::{TOLERANCE, block_sums, near_ties, reference_error};
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
    /// The same rows rounded to f16, for the profiles that round weights.
    pub(super) f16_rows: Vec<Vec<f64>>,
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
            None => (0..self.rows.len())
                .map(|k| {
                    let w = self.weights(p, k);
                    (self.rows[k].0, w.iter().zip(xq).map(|(w, a)| w * a).sum())
                })
                .collect(),
        }
    }

    /// The `k`-th checked row as `p` meets the activation.
    fn weights(&self, p: Profile, k: usize) -> &[f64] {
        match p.rounds_weights() {
            true => &self.f16_rows[k],
            false => self.rows[k].1,
        }
    }

    /// The weight that value `i` of the rounded activation meets in the
    /// `k`-th checked row.
    fn weight(&self, p: Profile, k: usize, i: usize) -> f64 {
        match p.block_sums() {
            Some(rounded) => block_sums::activation_weight(self.blocks[k], i, rounded),
            None => self.weights(p, k)[i],
        }
    }

    /// The column check of `x` against the opened output `y` under `p`,
    /// repaired when a near-tie fails it.
    pub(super) fn check(&self, p: Profile, x: &[f32], y: &[f32]) -> ColumnCheck {
        let xq = p.apply(x);
        let references = self.references(p, x, &xq);
        let error = reference_error(&references, y);
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
        near_ties::repair(&references, y, &flips, effect).map_or(unrepaired, |repair| ColumnCheck {
            error: repair.error,
            flips: repair.flips,
            ..unrepaired
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
            f16_rows: Vec::new(),
        };
        assert!(reference_error(&rows.references(p, &x, &xq), &y) > TOLERANCE);
        let check = rows.check(p, &x, &y);
        assert_eq!(check.flips, 1, "{check:?}");
        assert!(check.error <= near_ties::REPAIRED_TOLERANCE, "{check:?}");

        // other weights are not repaired
        let other: Vec<f32> = y.iter().map(|v| v * 1.05).collect();
        let check = rows.check(p, &x, &other);
        assert_eq!(check.flips, 0);
        assert!(check.error > TOLERANCE);
    }

    #[test]
    fn metal_columns_use_independent_binary_fraction_references() {
        // Q8 scale 1 + 2^-10, quant 3; nearest-half weight is 3 + 2^-8.
        let weights = [3.0029296875, -3.0029296875];
        let rows = CheckedRows {
            rows: vec![(0, &weights)],
            blocks: Vec::new(),
            f16_rows: vec![vec![3.00390625, -3.00390625]],
        };
        let x = [1.0009, -0.50045];
        for (profile, expected) in [
            (
                Profile::Raw,
                3.0029296875 * (f64::from(x[0]) - f64::from(x[1])),
            ),
            (Profile::F16Weights, 3.00390625 * 1.50146484375),
            (Profile::F16WeightsRtz, 3.00390625 * 1.5),
        ] {
            let honest = [expected as f32];
            assert!(rows.check(profile, &x, &honest).error <= TOLERANCE);
            let corrupt = [honest[0] + 0.25];
            assert!(rows.check(profile, &x, &corrupt).error > TOLERANCE);
        }
        // One activation crosses a half boundary within normalization slack.
        for (profile, boundary) in [
            (Profile::F16Weights, 1.0 + 1.0 / 2048.0),
            (Profile::F16WeightsRtz, 1.0 + 1.0 / 1024.0 - f32::EPSILON),
        ] {
            let x = [boundary, 1.0];
            let honest = [(3.00390625 / 1024.0) as f32];
            let check = rows.check(profile, &x, &honest);
            assert_eq!(check.flips, 1, "{profile:?}: {check:?}");
            assert!(check.error <= near_ties::REPAIRED_TOLERANCE);
            assert!(rows.check(profile, &x, &[honest[0] + 0.25]).error > TOLERANCE);
        }
    }

    #[test]
    fn f16_weight_profiles_meet_the_rounded_rows() {
        use crate::activation_rounding::round_to_f16;
        let w: Vec<f64> = (0..64).map(|i| 0.0123 * f64::from(i - 31) + 1e-4).collect();
        let x: Vec<f32> = (0..64).map(|i| 0.37 - 0.011 * i as f32 + 1e-5).collect();
        let rounded = round_to_f16(&w);
        // a kernel that truncates the activation to f16 against f16 weights
        let kernel = rounded
            .iter()
            .zip(Profile::F16WeightsRtz.apply(&x))
            .map(|(a, b)| a * b)
            .sum::<f64>() as f32;
        let rows = CheckedRows {
            rows: vec![(0, w.as_slice())],
            blocks: Vec::new(),
            f16_rows: vec![rounded],
        };
        let rtz = rows.check(Profile::F16WeightsRtz, &x, &[kernel]).error;
        let rte = rows.check(Profile::F16Weights, &x, &[kernel]).error;
        let raw = rows.check(Profile::Raw, &x, &[kernel]).error;
        assert!(rtz < 1e-7, "{rtz}");
        assert!(rte > 1e-5, "{rte}");
        assert!(raw > 1e-5, "{raw}");
    }
}
