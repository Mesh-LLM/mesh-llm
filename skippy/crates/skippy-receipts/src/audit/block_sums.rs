//! CUDA MMQ for weights with a minimum per 32 values (Q4_K, Q5_K).
//!
//! The kernel does not subtract the minimum from each weight. Per 32
//! values it computes `A · d8 · Σ q·q8 + M · S`, with `q` the weight's
//! quants, `q8` and `d8` the activation quantized as `quantize_mmq_q8_1`
//! does, `S` the sum of the unquantized activation stored as f16, and
//! `A = d·scale` and `M = −dmin·min`. The MMA kernels (Turing and later)
//! round `A` and `M` to f16 (`load_tiles_q4_K`, `load_tiles_q5_K`); the
//! dp4a kernels keep them exact. The sum of the unquantized activation is
//! why an activation profile alone cannot reproduce these matmuls.

use half::f16;

use super::Profile;
use crate::dequant::MinBlock;

/// The kernel's output for one weight row, with the activation `x` and its
/// MMQ quantization `xq`.
pub(super) fn reference(blocks: &[MinBlock], x: &[f32], xq: &[f64], rounded: bool) -> f64 {
    let round = |v: f64| {
        if rounded {
            f16::from_f64(v).to_f64()
        } else {
            v
        }
    };
    blocks
        .iter()
        .zip(x.chunks(32).zip(xq.chunks(32)))
        .map(|(block, (x, xq))| {
            let dot: f64 = block
                .quants
                .iter()
                .zip(xq)
                .map(|(&q, &a)| f64::from(q) * a)
                .sum();
            let sum = f16::from_f32(x.iter().sum::<f32>()).to_f64();
            round(block.scale) * dot - round(block.min) * sum
        })
        .sum()
}

/// The activation quantization of the MMQ kernels with block sums.
pub(super) fn quantized(x: &[f32]) -> Vec<f64> {
    Profile::Q8MulHalf.apply(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_minimum_meets_the_sum_of_the_unquantized_activation() {
        let block = MinBlock {
            scale: 0.5,
            min: 0.25,
            quants: [2; 32],
        };
        let x = [0.3f32; 32];
        let xq = quantized(&x);
        let sum = f64::from(f16::from_f32(0.3 * 32.0).to_f32());
        let expected = 0.5 * 2.0 * xq.iter().sum::<f64>() - 0.25 * sum;
        let got = reference(std::slice::from_ref(&block), &x, &xq, true);
        assert!((got - expected).abs() < 1e-12, "{got} {expected}");
        // the per-value form differs: the activation's rounding meets the minimum
        let per_value: f64 = xq.iter().map(|a| (0.5 * 2.0 - 0.25) * a).sum();
        assert_ne!(got, per_value);
    }

    #[test]
    fn mma_kernels_round_the_scales_to_f16() {
        let block = MinBlock {
            scale: 1.0 + 1.0 / 4096.0,
            min: 0.0,
            quants: [1; 32],
        };
        let x = [1.0f32; 32];
        let xq = quantized(&x);
        let unscaled: f64 = xq.iter().sum();
        assert_eq!(
            reference(std::slice::from_ref(&block), &x, &xq, true),
            unscaled
        );
        assert!(reference(std::slice::from_ref(&block), &x, &xq, false) > unscaled);
    }
}
