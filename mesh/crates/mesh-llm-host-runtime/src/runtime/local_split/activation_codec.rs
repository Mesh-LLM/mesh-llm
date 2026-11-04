//! The split's activation codec: the operator's `skippy.activation_codec`
//! choice.

use skippy_protocol::{StageActivationCodec as Codec, StageActivationCodecPolicy as Policy};

/// The operator's `skippy.activation_codec` choice for split activations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct SplitActivationCodec {
    pub(super) codec: Codec,
    pub(super) policy: Policy,
}

impl SplitActivationCodec {
    /// The default: f16 on every hop. Against single-node decode it keeps
    /// top-1 agreement and greedy output (mean log-prob drift 0.0006 on
    /// Qwen3.5-4B) at half of lossless's bytes, which shortens prefill and
    /// decode on a slow link and costs nothing on a fast one. Every
    /// stage-protocol peer decodes f16, so it needs no capability check.
    pub(super) const DEFAULT: Self = Self::fixed(Codec::F16RneV1);

    /// `lossless`: exact values. RawF32, compacted to BF16 or F16 only when
    /// the whole frame (sidebands included) round-trips bit-for-bit, which
    /// real activations rarely do.
    pub(super) const LOSSLESS: Self = Self {
        codec: Codec::RawF32V1,
        policy: Policy::AutoLosslessV1,
    };

    const fn fixed(codec: Codec) -> Self {
        Self {
            codec,
            policy: Policy::Fixed,
        }
    }

    /// Parse a `skippy.activation_codec` value, ignoring ASCII case as config
    /// validation does. Every value but `lossless` fixes one codec for every
    /// frame.
    pub(super) fn from_setting(value: &str) -> Option<Self> {
        let codec = match value.to_ascii_lowercase().as_str() {
            "lossless" => return Some(Self::LOSSLESS),
            "f32" => Codec::RawF32V1,
            "f16" => Codec::F16RneV1,
            "bf16" => Codec::Bf16RneV1,
            "s8-row" => Codec::S8RowF32RneV1,
            "fp8-e4m3-row" => Codec::Fp8E4m3RowF32RneV1,
            "fp8-e4m3-block32" => Codec::Fp8E4m3Block32E8m0RneV1,
            "fp8-e5m2-row" => Codec::Fp8E5m2RowF32RneV1,
            _ => return None,
        };
        Some(Self::fixed(codec))
    }
}
