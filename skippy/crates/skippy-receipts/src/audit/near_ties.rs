//! Repair of a column check that an activation near-tie failed.
//!
//! A backend rounds the activation to its profile's grid before the
//! weights meet it. When the value it rounded differs from the opened one
//! by a few ulps (a fused kernel computes it again, or divides with a fast
//! reciprocal), a value that lies on a rounding boundary rounds the other
//! way, and the output moves by one step of that value times its weight
//! column. That is far above [`super::TOLERANCE`] for some columns, though
//! the node computed honestly, and it was the main source of honest column
//! failures (RECEIPTS.md, "Fail rule").
//!
//! The repair rounds at most two of the activation's near-ties
//! ([`rounding_flips`]) the other way and keeps the result only when it
//! explains the difference on every checked row, within
//! [`REPAIRED_TOLERANCE`]. That gives the node no freedom: a flip is a fixed
//! change of the reference, known only once the client has drawn the rows,
//! and the values that can flip lie in the opened activation, which the
//! glue check binds to the layer's computation.

use super::rounding_flips::{Flip, PAIRED_FLIPS};
use super::{TOLERANCE, reference_error};

/// The error a repaired column must reach: well under the tolerance, so a
/// repair explains the difference and does not merely bring it under the
/// bar.
pub(crate) const REPAIRED_TOLERANCE: f64 = TOLERANCE / 4.0;

/// The best repair of a column: its error, and how many near-ties it
/// rounded the other way.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Repair {
    pub error: f64,
    pub flips: usize,
}

/// Repairs a column whose checked rows have the reference outputs
/// `references`, against the opened output `y`, with the scale floor of
/// [`super::reference_error`]. `effect(k, flip)` is the
/// change `flip` makes to the reference of the `k`-th checked row. Returns
/// `None` when no repair of at most two flips reaches
/// [`REPAIRED_TOLERANCE`].
pub fn repair(
    references: &[(usize, f64)],
    y: &[f32],
    floor: f64,
    flips: &[Flip],
    effect: impl Fn(usize, &Flip) -> f64,
) -> Option<Repair> {
    let effects: Vec<Vec<f64>> = flips
        .iter()
        .map(|flip| (0..references.len()).map(|k| effect(k, flip)).collect())
        .collect();
    let error = |taken: &[usize]| {
        let moved: Vec<(usize, f64)> = references
            .iter()
            .enumerate()
            .map(|(k, &(row, reference))| {
                (
                    row,
                    reference + taken.iter().map(|&f| effects[f][k]).sum::<f64>(),
                )
            })
            .collect();
        reference_error(&moved, y, floor)
    };
    let mut best: Option<Repair> = None;
    let mut consider = |taken: &[usize]| {
        let e = error(taken);
        if best.is_none_or(|b| e < b.error) {
            best = Some(Repair {
                error: e,
                flips: taken.len(),
            });
        }
    };
    for a in 0..flips.len() {
        consider(&[a]);
        if flips.len() <= PAIRED_FLIPS {
            for b in a + 1..flips.len() {
                consider(&[a, b]);
            }
        }
    }
    best.filter(|b| b.error <= REPAIRED_TOLERANCE)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flip(i: usize, change: f64) -> Flip {
        Flip {
            changes: vec![(i, change)],
        }
    }

    /// A 4-row matrix, the client's activation, and the node's output when
    /// value `i` of the activation rounded by `step` the other way.
    fn column(tied: Option<(usize, f64)>) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f32>) {
        let w: Vec<Vec<f64>> = (0..4)
            .map(|r| (0..8).map(|i| f64::from(r * 8 + i) * 0.01 - 0.15).collect())
            .collect();
        let x: Vec<f64> = (0..8).map(|i| 0.1 * f64::from(i) - 0.3).collect();
        let mut node = x.clone();
        if let Some((i, step)) = tied {
            node[i] += step;
        }
        let y = w.iter().map(|row| dot(row, &node) as f32).collect();
        (w, x, y)
    }

    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(a, b)| a * b).sum()
    }

    fn references(w: &[Vec<f64>], x: &[f64]) -> Vec<(usize, f64)> {
        w.iter()
            .enumerate()
            .map(|(r, row)| (r, dot(row, x)))
            .collect()
    }

    #[test]
    fn a_near_tie_explains_its_column() {
        let (w, x, y) = column(Some((3, 0.5)));
        let refs = references(&w, &x);
        assert!(reference_error(&refs, &y, 0.0) > TOLERANCE);
        let flips = [flip(1, 0.5), flip(3, 0.5), flip(5, -0.5)];
        let repair = repair(&refs, &y, 0.0, &flips, |k, f| {
            f.changes.iter().map(|&(i, d)| w[k][i] * d).sum()
        })
        .expect("the tie at value 3 explains the column");
        assert_eq!(repair.flips, 1);
        assert!(repair.error < 1e-6, "{repair:?}");
    }

    #[test]
    fn a_flip_cannot_explain_other_weights() {
        // the node used other weights: every row is off by its own amount
        let (w, x, mut y) = column(None);
        for (r, v) in y.iter_mut().enumerate() {
            *v += 0.02 * (r as f32 + 1.0) * if r % 2 == 0 { 1.0 } else { -1.0 };
        }
        let refs = references(&w, &x);
        assert!(reference_error(&refs, &y, 0.0) > TOLERANCE);
        let flips: Vec<Flip> = (0..8).map(|i| flip(i, 0.5)).collect();
        let effect = |k: usize, f: &Flip| f.changes.iter().map(|&(i, d)| w[k][i] * d).sum();
        assert_eq!(repair(&refs, &y, 0.0, &flips, effect), None);
    }

    #[test]
    fn two_near_ties_are_repaired_together() {
        let (w, x, _) = column(None);
        let mut moved = x.clone();
        moved[2] += 0.25;
        moved[6] -= 0.25;
        let y: Vec<f32> = w.iter().map(|row| dot(row, &moved) as f32).collect();
        let refs = references(&w, &x);
        let flips = [flip(2, 0.25), flip(4, 0.25), flip(6, -0.25)];
        let effect = |k: usize, f: &Flip| f.changes.iter().map(|&(i, d)| w[k][i] * d).sum();
        let repair = repair(&refs, &y, 0.0, &flips, effect).expect("two ties explain it");
        assert_eq!(repair.flips, 2);
    }

    #[test]
    fn no_flips_no_repair() {
        let (w, x, y) = column(Some((3, 0.5)));
        let refs = references(&w, &x);
        assert_eq!(repair(&refs, &y, 0.0, &[], |_, _| 0.0), None);
    }
}
