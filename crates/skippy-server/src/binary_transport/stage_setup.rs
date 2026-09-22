use anyhow::{Context, Result, bail};
use skippy_protocol::StageConfig;
use skippy_protocol::binary::{
    ActivationAgreement, ActivationDimension, ActivationPartProfile, ActivationProfile,
    MAX_STAGE_DECODED_ACTIVATION_BYTES,
};
pub(crate) use skippy_protocol::binary::{ConnectionRole, StageStream, client_setup, server_setup};
use skippy_runtime::ActivationBoundaryDesc;

/// Derived once from the realized native boundary. Non-token dimensions retain
/// their own bounded dynamic fields; they are not inferred from token count.
pub(crate) fn output_profile(
    config: &StageConfig,
    boundary: ActivationBoundaryDesc,
) -> Result<ActivationProfile> {
    let parts = boundary
        .parts()?
        .iter()
        .map(|part| {
            let token_axis =
                u32::try_from(part.token_axis).context("negative boundary token axis")?;
            let dimensions = std::array::from_fn(|axis| {
                if axis == token_axis as usize {
                    ActivationDimension::Tokens
                } else if axis >= part.rank as usize {
                    ActivationDimension::Fixed(1)
                } else if part.dimensions[axis] > 0 {
                    ActivationDimension::Fixed(part.dimensions[axis] as u64)
                } else {
                    ActivationDimension::Dynamic {
                        min: 1,
                        max: MAX_STAGE_DECODED_ACTIVATION_BYTES as u64,
                    }
                }
            });
            Ok(ActivationPartProfile {
                identity: part.identity,
                ggml_type: part.ggml_type,
                rank: part.rank,
                token_axis,
                optional: part.flags & skippy_protocol::binary::STAGE_ACTIVATION_PART_OPTIONAL != 0,
                dimensions,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let profile = ActivationProfile {
        id: 1,
        producer_stage_index: config.stage_index as i32,
        layer_start: config.layer_start as i32,
        layer_end: config.layer_end as i32,
        frontier_identity: boundary.frontier_identity,
        // Wire admission covers configured context/batch capacity; native KV and
        // sampler guards still validate each local execution independently.
        max_tokens: config
            .ctx_size
            .max(config.n_batch.unwrap_or(0))
            .saturating_mul(config.lane_count.max(1))
            .min(i32::MAX as u32),
        max_sequences: config
            .lane_count
            .max(1)
            .saturating_mul(if config.native_mtp_enabled { 2 } else { 1 }),
        parts,
    };
    ActivationAgreement {
        generation: [1; 16],
        profiles: vec![profile.clone()],
    }
    .validate()?;
    Ok(profile)
}

pub(crate) fn include_forwarded_optional(
    output: &mut ActivationProfile,
    upstream: &ActivationAgreement,
) -> Result<()> {
    for source in upstream
        .profiles
        .iter()
        .flat_map(|p| &p.parts)
        .filter(|p| p.optional)
    {
        if let Some(existing) = output.parts.iter().find(|p| p.identity == source.identity) {
            if existing.ggml_type != source.ggml_type
                || existing.rank != source.rank
                || existing.token_axis != source.token_axis
            {
                bail!("forwarded optional activation conflicts with output boundary");
            }
        } else {
            output.parts.push(source.clone());
        }
    }
    ActivationAgreement {
        generation: [1; 16],
        profiles: vec![output.clone()],
    }
    .validate()?;
    Ok(())
}

#[cfg(test)]
pub(crate) fn test_profile(config: &StageConfig) -> ActivationProfile {
    ActivationProfile {
        id: 1,
        producer_stage_index: config.stage_index as i32,
        layer_start: config.layer_start as i32,
        layer_end: config.layer_end as i32,
        frontier_identity: [1; 32],
        max_tokens: 4096,
        max_sequences: 4096,
        parts: vec![ActivationPartProfile {
            identity: [2; 32],
            ggml_type: 0,
            rank: 2,
            token_axis: 1,
            optional: false,
            dimensions: [
                ActivationDimension::Fixed(2),
                ActivationDimension::Tokens,
                ActivationDimension::Fixed(1),
                ActivationDimension::Fixed(1),
            ],
        }],
    }
}
