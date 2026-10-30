//! The split's activation codec: the operator's `skippy.activation_codec`
//! choice, and the fallback when a stage cannot decode it.

use super::{RuntimeSliceStagePlan, SplitParticipant};
use skippy_protocol::{StageActivationCodec as Codec, StageActivationCodecPolicy as Policy};

/// The operator's `skippy.activation_codec` choice for split activations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct SplitActivationCodec {
    pub(super) codec: Codec,
    pub(super) policy: Policy,
}

impl SplitActivationCodec {
    /// The default. Frames carry exact values: RawF32, compacted to BF16 or
    /// F16 only when the whole frame (sidebands included) round-trips
    /// bit-for-bit, which real activations rarely do.
    pub(super) const LOSSLESS: Self = Self {
        codec: Codec::RawF32V1,
        policy: Policy::AutoLosslessV1,
    };

    /// What an FP8 choice becomes when a stage cannot decode FP8: the
    /// closest-quality codec every stage-protocol peer decodes.
    pub(super) const FP8_FALLBACK: Self = Self::fixed(Codec::F16RneV1);

    const fn fixed(codec: Codec) -> Self {
        Self {
            codec,
            policy: Policy::Fixed,
        }
    }

    /// Parse a `skippy.activation_codec` value. Every value but `lossless`
    /// fixes one codec for every frame.
    pub(super) fn from_setting(value: &str) -> Option<Self> {
        let codec = match value {
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

    /// This choice if every stage can decode it, else [`Self::FP8_FALLBACK`].
    ///
    /// Peers that predate the FP8 codecs reject a stage load naming one, so
    /// an FP8 choice needs every stage's node to advertise
    /// `activation-codec-fp8-v1`. A stage whose node is not among the
    /// participants counts as unable. Checked per generation, because a
    /// replan can move a stage onto an older peer.
    pub(super) fn supported_by(
        self,
        stages: &[RuntimeSliceStagePlan],
        participants: &[SplitParticipant],
    ) -> Self {
        if !self.codec.is_fp8() {
            return self;
        }
        let unsupported = stages.iter().find(|stage| {
            !participants.iter().any(|participant| {
                participant.node_id == stage.node_id && participant.activation_codec_fp8_supported
            })
        });
        let Some(stage) = unsupported else {
            return self;
        };
        tracing::warn!(
            activation_codec = self.codec.identity(),
            fallback = Self::FP8_FALLBACK.codec.identity(),
            stage_id = stage.stage_id,
            node_id = %stage.node_id.fmt_short(),
            feature = skippy_protocol::STAGE_SUBPROTOCOL_FEATURE_ACTIVATION_CODEC_FP8_V1,
            "a split stage's peer cannot decode FP8 activations; this generation \
             sends f16 instead; upgrade that peer to use the configured codec"
        );
        Self::FP8_FALLBACK
    }
}
