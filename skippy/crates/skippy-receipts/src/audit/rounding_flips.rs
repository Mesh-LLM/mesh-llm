//! The roundings of an activation that a backend may take the other way.
//!
//! The client normalizes a residual in f64 and the node in f32, so their
//! activations differ by a few ulps. Where a value lies that close to a
//! rounding boundary of a profile (a q8 step, or an f16 midpoint), the two
//! sides round it apart, and the node's matmul output differs from the
//! client's reference by one step of that value times its weight column.
//! Over a whole output matrix that is far above the head key's tolerance,
//! though the node computed honestly. Few values lie that close: with q8,
//! under one per 2,560-wide activation on average.

use std::collections::HashSet;

use crate::activation_rounding::Profile;
use crate::error_bound::worst_of;

/// How far, relative to its magnitude, a value may sit from the client's
/// and still be the node's: 8 f32 ulps.
const SLACK: f32 = 8.0 * f32::EPSILON;

/// Flips of one activation past which a repair tries only single ones:
/// pairs cost the square of the count. The row check's repair and the
/// output head's key check share this bound.
pub(crate) const PAIRED_FLIPS: usize = 32;

/// An activation the profile gives when one value moves by up to
/// [`SLACK`]: the changed outputs, as (index, change).
#[derive(Clone, Debug, PartialEq)]
pub struct Flip {
    pub changes: Vec<(usize, f64)>,
}

/// The values of a block of a profile's activation that share a scale.
fn block_len(profile: Profile) -> usize {
    match profile {
        Profile::Raw
        | Profile::F16
        | Profile::F16Weights
        | Profile::F16WeightsRtz
        | Profile::Bf16 => 1,
        Profile::Q8Mul64F32 => 64,
        Profile::CpuQ8K => 256,
        Profile::Q8DivHalf
        | Profile::Q8MulHalf
        | Profile::Q8MulHalfSums
        | Profile::Q8MulHalfSumsDp4a
        | Profile::Q8MulF32
        | Profile::CpuQ8 => 32,
    }
}

/// The flips of `profile` at `x`, whose activation is `applied`. A value's
/// move changes its block's scale when it is the block's largest, so a flip
/// may change the whole block. Changes as small as the move itself are not
/// flips.
pub fn flips(profile: Profile, x: &[f32], applied: &[f64]) -> Vec<Flip> {
    let len = block_len(profile);
    let mut out = Vec::new();
    // the flips found so far, by the bits of their changes: an activation
    // can have thousands, and comparing each new one with all of them
    // would cost their square
    let mut seen: HashSet<Vec<(usize, u64)>> = HashSet::new();
    for (b, block) in x.chunks(len).enumerate() {
        let first = b * len;
        let base = &applied[first..first + block.len()];
        let mut moved = block.to_vec();
        for i in 0..block.len() {
            for factor in [1.0 + SLACK, 1.0 - SLACK] {
                moved[i] = block[i] * factor;
                let changes: Vec<(usize, f64)> = profile
                    .apply(&moved)
                    .iter()
                    .zip(base)
                    .enumerate()
                    .filter(|&(_, (a, b))| (a - b).abs() > 4.0 * f64::from(SLACK) * b.abs())
                    .map(|(j, (a, b))| (first + j, a - b))
                    .collect();
                if changes.is_empty() {
                    continue;
                }
                let bits = changes.iter().map(|&(j, c)| (j, c.to_bits())).collect();
                if seen.insert(bits) {
                    out.push(Flip { changes });
                }
            }
            moved[i] = block[i];
        }
    }
    out
}

/// The flips that best explain `residuals`, at most two of them, and the
/// largest residual they leave. `effects[f][k]` is what flip `f` adds to the
/// reference of residual `k`. No flip is taken unless it leaves less than
/// none does, and pairs are tried only up to [`PAIRED_FLIPS`] flips.
pub(crate) fn best_explanation(residuals: &[f64], effects: &[Vec<f64>]) -> (Vec<usize>, f64) {
    let left = |taken: &[usize]| {
        worst_of(
            residuals
                .iter()
                .enumerate()
                .map(|(k, &e)| (e - taken.iter().map(|&f| effects[f][k]).sum::<f64>()).abs()),
        )
    };
    let mut best = (Vec::new(), left(&[]));
    let mut consider = |taken: Vec<usize>| {
        let error = left(&taken);
        if error < best.1 {
            best = (taken, error);
        }
    };
    for a in 0..effects.len() {
        consider(vec![a]);
        if effects.len() <= PAIRED_FLIPS {
            for b in a + 1..effects.len() {
                consider(vec![a, b]);
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_value_on_a_q8_step_boundary_flips_alone() {
        let mut x = vec![0.25f32; 32];
        x[0] = 1.27;
        // d = 0.01: 0.255 lies on the boundary between steps 25 and 26
        x[5] = 0.255;
        let applied = Profile::Q8DivHalf.apply(&x);
        let flips = flips(Profile::Q8DivHalf, &x, &applied);
        assert!(
            flips
                .iter()
                .any(|f| f.changes.len() == 1 && f.changes[0].0 == 5)
        );
        assert!(flips.iter().all(|f| f.changes.iter().all(|&(j, _)| j == 5)));
    }

    #[test]
    fn few_values_of_an_activation_flip() {
        let mut stream = crate::draw::DrawStream::new(b"flips", "x");
        let x: Vec<f32> = (0..2560)
            .map(|_| (stream.below(1 << 20) as f32 / (1 << 19) as f32) - 1.0)
            .collect();
        assert!(flips(Profile::Raw, &x, &Profile::Raw.apply(&x)).is_empty());
        for profile in [Profile::Q8DivHalf, Profile::Q8MulHalf, Profile::Q8MulF32] {
            let found = flips(profile, &x, &profile.apply(&x));
            assert!(found.len() < 8, "{profile:?}: {}", found.len());
        }
    }

    #[test]
    fn many_flips_are_each_found_once_in_order() {
        // an activation a node could open to make every value a near-tie:
        // each one halfway between two f16 values
        let x: Vec<f32> = (0..4096)
            .map(|i| 1.0 + 2f32.powi(-11) + i as f32 * 2f32.powi(-10) * 2.0)
            .map(|v| v.min(1.9))
            .collect();
        let found = flips(Profile::F16, &x, &Profile::F16.apply(&x));
        assert!(found.len() > 400, "{}", found.len());
        let mut seen = HashSet::new();
        for flip in &found {
            let bits: Vec<(usize, u64)> = flip
                .changes
                .iter()
                .map(|&(j, c)| (j, c.to_bits()))
                .collect();
            assert!(seen.insert(bits), "a flip was found twice: {flip:?}");
        }
        let first: Vec<usize> = found.iter().map(|f| f.changes[0].0).collect();
        assert!(first.windows(2).all(|pair| pair[0] <= pair[1]));
    }

    #[test]
    fn an_f16_midpoint_flips() {
        // halfway between the f16 values 1.0 and 1.0 + 2^-10
        let x = [1.0f32 + 2f32.powi(-11), 0.5];
        let applied = Profile::F16.apply(&x);
        let found = flips(Profile::F16, &x, &applied);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].changes.len(), 1);
        assert_eq!(found[0].changes[0].0, 0);
    }
}
