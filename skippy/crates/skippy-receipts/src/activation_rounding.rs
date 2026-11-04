//! How a backend rounds f32 values before a kernel multiplies them: the
//! rounding profiles of the matmul activations, by kernel.
//!
//! The client applies a profile to an opened activation before it compares
//! a matmul with the weights of the claimed GGUF ([`crate::audit`]). A
//! node and a client both apply one to q when they compute attention
//! partials for a kernel that rounds q ([`crate::attention`]), so the
//! rounding is the same arithmetic on both sides.

/// How a backend rounds the f32 activation before it meets the weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Profile {
    Raw,
    F16,
    /// Rounded to bf16, as CPU bf16 dot products and cuBLAS bf16 do.
    Bf16,
    /// CUDA `quantize_q8_1` (MMVQ): d = amax/127, q = round(x/d), d stored as f16.
    Q8DivHalf,
    /// CUDA `quantize_mmq_q8_1` (MMQ): q = round(x·127/amax), d stored as f16
    /// (the layout with sums, for Q4_0).
    Q8MulHalf,
    /// MMQ for Q4_K and Q5_K: [`Self::Q8MulHalf`], with the minimum applied
    /// to the block sum the kernel stores per 32 values (the block's scale
    /// times the sum of its quants, as f16), and the weight's scale and
    /// minimum rounded to f16 (the MMA kernels).
    Q8MulHalfSums,
    /// [`Self::Q8MulHalfSums`] with exact weight scales (the dp4a kernels).
    Q8MulHalfSumsDp4a,
    /// MMQ with d stored as f32 (Q5_0, Q3_K, Q6_K, IQ4_NL, IQ4_XS).
    Q8MulF32,
    /// MMQ with one f32 d per 64 values (Q2_K).
    Q8Mul64F32,
    /// CPU `quantize_row_q8_0`: d = amax/127, q = round(x·(1/d)), d stored as f16.
    CpuQ8,
    /// CPU `quantize_row_q8_K` for the K-quants: per 256 values, with `max`
    /// the value of largest magnitude, q = min(127, round_even(x·-127/max)),
    /// d = -max/127 stored as f32.
    CpuQ8K,
}

impl Profile {
    pub fn name(self) -> &'static str {
        match self {
            Self::Raw => "raw",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
            Self::Q8DivHalf => "q8_32_div_dh",
            Self::Q8MulHalf => "q8_32_mul_dh",
            Self::Q8MulHalfSums => "q8_32_mul_dh_sums",
            Self::Q8MulHalfSumsDp4a => "q8_32_mul_dh_sums_dp4a",
            Self::Q8MulF32 => "q8_32_mul_d",
            Self::Q8Mul64F32 => "q8_64_mul_d",
            Self::CpuQ8 => "cpu_q8_0",
            Self::CpuQ8K => "cpu_q8_K",
        }
    }

    /// For the profiles that apply a weight's minimum to block sums of the
    /// activation: whether the weight's scales are rounded to f16.
    pub(crate) fn block_sums(self) -> Option<bool> {
        match self {
            Self::Q8MulHalfSums => Some(true),
            Self::Q8MulHalfSumsDp4a => Some(false),
            _ => None,
        }
    }

    pub fn apply(self, x: &[f32]) -> Vec<f64> {
        match self {
            Self::Raw => x.iter().map(|&v| f64::from(v)).collect(),
            Self::F16 => x
                .iter()
                .map(|&v| f64::from(half::f16::from_f32(v).to_f32()))
                .collect(),
            Self::Bf16 => x
                .iter()
                .map(|&v| f64::from(half::bf16::from_f32(v).to_f32()))
                .collect(),
            Self::Q8DivHalf | Self::Q8MulHalf | Self::CpuQ8 => quantize_q8_blocks(self, x),
            // the MMQ kernels with block sums quantize the activation as
            // the ones without
            Self::Q8MulHalfSums | Self::Q8MulHalfSumsDp4a => Self::Q8MulHalf.apply(x),
            Self::Q8MulF32 => quantize_mul_blocks(x, 32),
            Self::Q8Mul64F32 => quantize_mul_blocks(x, 64),
            Self::CpuQ8K => quantize_q8_k(x),
        }
    }
}

fn quantize_q8_blocks(profile: Profile, x: &[f32]) -> Vec<f64> {
    let mut out = Vec::with_capacity(x.len());
    for block in x.chunks(32) {
        let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        if amax == 0.0 {
            out.extend(std::iter::repeat_n(0.0, block.len()));
            continue;
        }
        let (d, quant): (f32, Box<dyn Fn(f32) -> f32>) = match profile {
            Profile::Q8DivHalf => {
                let d = amax / 127.0;
                (d, Box::new(move |v| (v / d).round()))
            }
            Profile::Q8MulHalf => {
                let d_inv = 127.0 / amax;
                (1.0 / d_inv, Box::new(move |v| (v * d_inv).round()))
            }
            _ => {
                let d = amax / 127.0;
                let id = 1.0 / d;
                (d, Box::new(move |v| (v * id).round()))
            }
        };
        let dd = f64::from(half::f16::from_f32(d).to_f32());
        out.extend(block.iter().map(|&v| dd * f64::from(quant(v))));
    }
    out
}

/// CUDA MMQ blocks of `block` values with an f32 scale.
fn quantize_mul_blocks(x: &[f32], block: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(x.len());
    for values in x.chunks(block) {
        let amax = values.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        if amax == 0.0 {
            out.extend(std::iter::repeat_n(0.0, values.len()));
            continue;
        }
        let d_inv = 127.0 / amax;
        let d = f64::from(1.0 / d_inv);
        out.extend(values.iter().map(|&v| d * f64::from((v * d_inv).round())));
    }
    out
}

/// CPU `quantize_row_q8_K_ref`, blocks of 256.
fn quantize_q8_k(x: &[f32]) -> Vec<f64> {
    let mut out = Vec::with_capacity(x.len());
    for values in x.chunks(256) {
        let max = values
            .iter()
            .fold(0.0f32, |m, &v| if v.abs() > m.abs() { v } else { m });
        if max == 0.0 {
            out.extend(std::iter::repeat_n(0.0, values.len()));
            continue;
        }
        let iscale = -127.0 / max;
        let d = f64::from(1.0 / iscale);
        out.extend(
            values
                .iter()
                .map(|&v| d * f64::from((iscale * v).round_ties_even().min(127.0))),
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cpu_k_quant_profile_scales_by_the_largest_value() {
        let mut x = vec![0.5f32; 256];
        x[7] = -2.0;
        let y = Profile::CpuQ8K.apply(&x);
        // d = 2/127; the largest value is exact and 0.5 is 31.75 steps
        assert!((y[7] + 2.0).abs() < 1e-6);
        assert!((y[0] - 32.0 * 2.0 / 127.0).abs() < 1e-6);
        assert_eq!(Profile::CpuQ8K.apply(&[0.0; 256]), vec![0.0; 256]);
        // MMQ blocks of 64 share one scale
        let mut x = vec![1.0f32; 128];
        x[0] = 4.0;
        let y = Profile::Q8Mul64F32.apply(&x);
        assert!((y[63] - 32.0 * 4.0 / 127.0).abs() < 1e-6);
        assert!((y[64] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn q8_profiles_round_to_the_block_grid() {
        let x: Vec<f32> = (0..32).map(|i| i as f32 / 31.0).collect();
        for p in [Profile::Q8DivHalf, Profile::Q8MulHalf, Profile::CpuQ8] {
            let q = p.apply(&x);
            assert!((q[31] - 1.0).abs() < 1e-3, "{:?}", p);
            assert!(
                q.iter()
                    .zip(&x)
                    .all(|(a, b)| (a - f64::from(*b)).abs() < 5e-3)
            );
        }
    }
}
