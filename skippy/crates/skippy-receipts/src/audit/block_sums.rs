//! CUDA MMQ for weights with a minimum per 32 values (Q4_K, Q5_K).
//!
//! The kernel does not subtract the minimum from each weight. Per 32
//! values it computes `A · d8 · Σ q·q8 + M · S`, with `q` the weight's
//! quants, `q8` and `d8` the activation quantized as `quantize_mmq_q8_1`
//! does, `S` the block sum that function stores as f16, and `A = d·scale`
//! and `M = −dmin·min`. The MMA kernels (Turing and later) round `A` and
//! `M` to f16 (`load_tiles_q4_K`, `load_tiles_q5_K`); the dp4a kernels keep
//! them exact.
//!
//! `S` is the block's f32 scale times the sum of its quants
//! (`d * float(sum_q)`), not the scale as f16 that the products use, so an
//! activation profile alone does not reproduce these matmuls. Before
//! llama.cpp #31247 it was the sum of the unquantized activation; a
//! receipt names its runtime, and a verifier of another one does not audit
//! it, so only the current rule is kept.

use half::f16;

use crate::dequant::MinBlock;

/// The kernel's output for one weight row, with the activation `x` and its
/// MMQ quantization `xq`.
pub(super) fn reference(blocks: &[MinBlock], x: &[f32], xq: &[f64], rounded: bool) -> f64 {
    let round = |v: f64| kernel_scale(v, rounded);
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
            round(block.scale) * dot - round(block.min) * stored_sum(x)
        })
        .sum()
}

/// The sum of one block of the activation as `quantize_mmq_q8_1` stores
/// it: the block's scale times the sum of its quants, computed in f32 and
/// kept as f16.
fn stored_sum(x: &[f32]) -> f64 {
    let amax = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    if amax == 0.0 {
        return 0.0;
    }
    let d_inv = 127.0 / amax;
    let quants: i32 = x.iter().map(|&v| (v * d_inv).round() as i32).sum();
    let d = 1.0 / d_inv;
    f16::from_f32(d * quants as f32).to_f64()
}

/// The weight that value `i` of the quantized activation meets: its
/// block's scale times its quant, less the block's minimum, because the
/// stored sum moves with each quant of the activation too.
pub(super) fn activation_weight(blocks: &[MinBlock], i: usize, rounded: bool) -> f64 {
    let block = &blocks[i / 32];
    kernel_scale(block.scale, rounded) * f64::from(block.quants[i % 32])
        - kernel_scale(block.min, rounded)
}

/// A weight scale or minimum as the kernel holds it.
fn kernel_scale(v: f64, rounded: bool) -> f64 {
    if rounded {
        f16::from_f64(v).to_f64()
    } else {
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation_rounding::Profile;

    /// The activation quantization of the MMQ kernels with block sums.
    fn quantized(x: &[f32]) -> Vec<f64> {
        Profile::Q8MulHalfSums.apply(x)
    }

    #[test]
    fn the_minimum_meets_the_stored_sum_of_the_quantized_activation() {
        let block = MinBlock {
            scale: 0.5,
            min: 0.25,
            quants: [2; 32],
        };
        // one large value sets the scale; the others round to 38 of 127
        let mut x = [0.3f32; 32];
        x[0] = 1.0;
        let xq = quantized(&x);
        let d_inv = 127.0f32;
        let quants = 127 + 31 * (0.3f32 * d_inv).round() as i32;
        let sum = f64::from(f16::from_f32((1.0 / d_inv) * quants as f32).to_f32());
        assert_eq!(stored_sum(&x), sum);
        let expected = 0.5 * 2.0 * xq.iter().sum::<f64>() - 0.25 * sum;
        let got = reference(std::slice::from_ref(&block), &x, &xq, true);
        assert!((got - expected).abs() < 1e-12, "{got} {expected}");
        // it is not the sum of the unquantized activation
        let unquantized = f64::from(f16::from_f32(x.iter().sum::<f32>()).to_f32());
        assert_ne!(sum, unquantized);
        // a block of zeros has a zero sum, not a quotient by zero
        assert_eq!(stored_sum(&[0.0; 32]), 0.0);
    }

    #[test]
    fn a_quant_of_the_activation_meets_the_scale_and_the_minimum() {
        let block = MinBlock {
            scale: 0.5,
            min: 0.25,
            quants: [2; 32],
        };
        let blocks = std::slice::from_ref(&block);
        assert_eq!(activation_weight(blocks, 3, true), 0.5 * 2.0 - 0.25);
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
