//! Which rounding profiles a backend may use for a weight type: the table
//! an audit picks its profiles from ([`crate::audit`]), and the one a
//! stage reads at start to know whether an audit can cover the model it
//! serves. A pair without an entry cannot be checked.

use anyhow::{Context, Result};

use crate::activation_rounding::Profile;

/// The backends with measured rounding profiles. Metal has none: through
/// `skippy serve`, its prefill outputs differ from every profile by up to
/// 8/10,000, which would fail honest nodes (RECEIPTS_MEASUREMENTS.md,
/// "Detection").
///
/// Vulkan has none either. Its integer-dot kernels round K-quant weights
/// in a way no profile models, so an honest node with Q4_K_M weights on
/// an integrated GPU passed 2 of 16 audits, and Q8_0 weights there passed
/// with 1% of margin (RECEIPTS_MEASUREMENTS.md, "Vulkan on the final
/// build"). A stage on a backend without a family issues no receipts, and
/// a receipt that names one is not audited (`coverage`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendFamily {
    Cpu,
    Cuda,
}

impl BackendFamily {
    /// The family of a receipt's backend name, such as `CPU` or `CUDA0`.
    pub fn of(backend: &str) -> Option<Self> {
        if backend.eq_ignore_ascii_case("cpu") {
            Some(Self::Cpu)
        } else if backend.starts_with("CUDA") {
            Some(Self::Cuda)
        } else {
            None
        }
    }

    /// [`Self::of`], failing for a backend without rounding profiles.
    pub fn checked(backend: &str) -> Result<Self> {
        Self::of(backend).with_context(|| format!("no rounding profiles for backend {backend} yet"))
    }
}

/// The profiles a backend may use for a weight type. A pair without an
/// entry cannot be checked, which makes the audit unusable rather than
/// letting the most lenient profile decide.
///
/// The CPU multiplies f16 weights with the f32 activation as it is
/// (`vec_dot_type` of f16 is f32, in `ggml_vec_dot_f16_f32` and the
/// llamafile kernels), and rounds the activation to bf16 for bf16 weights.
///
/// CUDA picks its kernel by batch size: MMVQ (`Q8DivHalf`) for decode,
/// MMQ (`Q8MulHalf`) for prefill, and a dequantize-and-cuBLAS path that
/// rounds the activation to f16.
///
/// f16 and bf16 weights on CUDA have none. For f16, decode sums the
/// products in half precision (`mmvf.cu`) and prefill runs cuBLAS with a
/// 16-bit compute type, which no profile of the activation models: an
/// honest node failed every audit, with 278 of 1,120 columns past
/// [`crate::audit`]'s tolerance. bf16 was never measured on CUDA.
///
/// Q4_1 and Q5_1 have none: their kernels round the product of the weight
/// and activation scales, and the activation block sum, which a profile of
/// the activation alone cannot reproduce (RECEIPTS_MEASUREMENTS.md, "Detection").
pub fn profiles_for(family: BackendFamily, weight_type: u32) -> Option<&'static [Profile]> {
    use crate::ggml_types::{
        GGML_BF16, GGML_F16, GGML_F32, GGML_IQ4_NL, GGML_IQ4_XS, GGML_Q2_K, GGML_Q3_K, GGML_Q4_0,
        GGML_Q4_K, GGML_Q5_0, GGML_Q5_K, GGML_Q6_K, GGML_Q8_0,
    };
    use Profile::{
        Bf16, CpuQ8, CpuQ8K, F16, Q8DivHalf, Q8Mul64F32, Q8MulF32, Q8MulHalf, Q8MulHalfSums,
        Q8MulHalfSumsDp4a, Raw,
    };
    let profiles: &'static [Profile] = match (family, weight_type) {
        (_, GGML_F32) | (BackendFamily::Cpu, GGML_F16) => &[Raw],
        (BackendFamily::Cpu, GGML_BF16) => &[Bf16],
        (BackendFamily::Cpu, GGML_Q8_0 | GGML_Q4_0 | GGML_Q5_0 | GGML_IQ4_NL) => &[CpuQ8],
        (
            BackendFamily::Cpu,
            GGML_Q2_K | GGML_Q3_K | GGML_Q4_K | GGML_Q5_K | GGML_Q6_K | GGML_IQ4_XS,
        ) => &[CpuQ8K],
        (BackendFamily::Cuda, GGML_Q8_0) => &[Q8DivHalf, Q8MulHalf, F16],
        (BackendFamily::Cuda, GGML_Q4_0) => &[Q8DivHalf, Q8MulHalf, F16],
        (BackendFamily::Cuda, GGML_Q4_K | GGML_Q5_K) => {
            &[Q8DivHalf, Q8MulHalfSums, Q8MulHalfSumsDp4a, F16]
        }
        (BackendFamily::Cuda, GGML_Q5_0 | GGML_Q3_K | GGML_Q6_K | GGML_IQ4_NL | GGML_IQ4_XS) => {
            &[Q8DivHalf, Q8MulF32, F16]
        }
        (BackendFamily::Cuda, GGML_Q2_K) => &[Q8DivHalf, Q8Mul64F32, F16],
        _ => return None,
    };
    Some(profiles)
}

/// Why no audit can check a stage on `family` that multiplies `tensors`
/// (name and GGML type), if one of them has a type without profiles.
pub fn unprofiled<'a>(
    family: BackendFamily,
    tensors: impl IntoIterator<Item = (&'a str, u32)>,
) -> Option<String> {
    let (name, ggml_type) = tensors
        .into_iter()
        .find(|(_, ggml_type)| profiles_for(family, *ggml_type).is_none())?;
    Some(format!(
        "no rounding profile for {} weights ({name}) on {family:?}",
        crate::ggml_types::type_name(ggml_type)
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ggml_types::{
        GGML_BF16, GGML_F16, GGML_F32, GGML_Q2_K, GGML_Q4_1, GGML_Q4_K, GGML_Q8_0,
    };

    #[test]
    fn each_backend_names_its_own_profiles() {
        assert_eq!(BackendFamily::of("CPU"), Some(BackendFamily::Cpu));
        assert_eq!(BackendFamily::of("CUDA0"), Some(BackendFamily::Cuda));
        // a backend without measured profiles has no family: Vulkan's kernels
        // round in ways no profile models, and Metal's prefill fits none
        for unprofiled in ["Vulkan0", "Vulkan1", "MTL0", "Metal", "ROCm0", ""] {
            assert_eq!(BackendFamily::of(unprofiled), None, "{unprofiled}");
            assert!(BackendFamily::checked(unprofiled).is_err());
        }
        assert_eq!(
            profiles_for(BackendFamily::Cpu, GGML_Q8_0),
            Some(&[Profile::CpuQ8][..])
        );
        // CUDA's q8 kernels, never the CPU one
        let cuda = profiles_for(BackendFamily::Cuda, GGML_Q8_0).unwrap();
        assert!(!cuda.contains(&Profile::CpuQ8) && !cuda.contains(&Profile::Raw));
        assert_eq!(
            profiles_for(BackendFamily::Cpu, GGML_F32),
            Some(&[Profile::Raw][..])
        );
        // the CPU kernels take the f32 activation of f16 weights as it is
        assert_eq!(
            profiles_for(BackendFamily::Cpu, GGML_F16),
            Some(&[Profile::Raw][..])
        );
        assert_eq!(
            profiles_for(BackendFamily::Cpu, GGML_BF16),
            Some(&[Profile::Bf16][..])
        );
        // the CPU quantizes K-quant activations per 256, CUDA per 32 or 64
        assert_eq!(
            profiles_for(BackendFamily::Cpu, GGML_Q4_K),
            Some(&[Profile::CpuQ8K][..])
        );
        assert!(
            profiles_for(BackendFamily::Cuda, GGML_Q2_K)
                .unwrap()
                .contains(&Profile::Q8Mul64F32)
        );
        // types without a reference dequantization or a known profile have none
        assert_eq!(profiles_for(BackendFamily::Cpu, 19), None);
        assert_eq!(profiles_for(BackendFamily::Cuda, GGML_Q4_1), None);
    }

    /// CUDA sums f16 products in half precision and runs cuBLAS with a
    /// 16-bit compute type, which no profile models: an honest node failed
    /// every audit. bf16 was never measured there.
    #[test]
    fn cuda_has_no_profile_for_16_bit_float_weights() {
        for weight_type in [GGML_F16, GGML_BF16] {
            assert_eq!(profiles_for(BackendFamily::Cuda, weight_type), None);
        }
    }

    #[test]
    fn the_first_unprofiled_tensor_is_named() {
        let tensors = [
            ("blk.0.attn_q.weight", GGML_Q8_0),
            ("blk.0.attn_k.weight", GGML_F16),
            ("blk.0.attn_v.weight", GGML_Q4_1),
        ];
        assert_eq!(unprofiled(BackendFamily::Cpu, tensors[..2].to_vec()), None);
        let cpu = unprofiled(BackendFamily::Cpu, tensors.to_vec()).unwrap();
        assert!(cpu.contains("q4_1 weights (blk.0.attn_v.weight)"), "{cpu}");
        let cuda = unprofiled(BackendFamily::Cuda, tensors.to_vec()).unwrap();
        assert!(
            cuda.contains("f16 weights (blk.0.attn_k.weight) on Cuda"),
            "{cuda}"
        );
    }
}
