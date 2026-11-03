use anyhow::{Context, Result, bail};
use skippy_protocol::{
    StageActivationCodec,
    binary::{
        STAGE_ACTIVATION_FRAME_VERSION, StageActivationDesc, StageActivationPartDesc,
        StageWireMessage, activation_frame_wire_bytes,
    },
};
use skippy_runtime::ActivationBoundaryDesc;

/// Largest prefix whose complete wire message fits the stage frame budget.
pub(in crate::frontend) fn bounded_chunk_size(
    requested: usize,
    boundary: &ActivationBoundaryDesc,
    codec: StageActivationCodec,
    message: &StageWireMessage,
) -> Result<usize> {
    let overhead = message
        .estimated_wire_bytes()
        .saturating_sub(message.tokens.len() * std::mem::size_of::<i32>())
        .saturating_sub(message.activation.len());
    let mut low = 0;
    let mut high = requested;
    while low < high {
        let middle = low + (high - low).div_ceil(2);
        let activation = activation_bytes(boundary, codec, middle)?;
        let total = overhead
            .saturating_add(middle.saturating_mul(4))
            .saturating_add(activation);
        if total <= skippy_protocol::MAX_STAGE_FRAME_BYTES {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    if low == 0 {
        bail!("one prefill token exceeds the stage frame budget");
    }
    Ok(low)
}

fn activation_bytes(
    boundary: &ActivationBoundaryDesc,
    codec: StageActivationCodec,
    tokens: usize,
) -> Result<usize> {
    let mut payload_bytes = 0_u64;
    let mut parts = Vec::new();
    for part in boundary.parts()? {
        let mut dimensions = part.dimensions;
        let token_axis = usize::try_from(part.token_axis).context("negative token axis")?;
        let rank = usize::try_from(part.rank).context("activation rank overflow")?;
        if token_axis >= rank || rank > dimensions.len() {
            bail!("invalid activation token axis");
        }
        dimensions[token_axis] = i64::try_from(tokens).context("prefill token count overflow")?;
        let mut strides = [0; 4];
        let mut bytes = match part.ggml_type {
            0 | 26 => 4_u64,
            1 | 30 => 2_u64,
            other => bail!("unsupported activation dtype {other}"),
        };
        for axis in 0..rank {
            strides[axis] = bytes;
            bytes = bytes
                .checked_mul(
                    u64::try_from(dimensions[axis]).context("unresolved activation dimension")?,
                )
                .context("activation size overflow")?;
        }
        parts.push(StageActivationPartDesc {
            identity: part.identity,
            ggml_type: part.ggml_type,
            rank: part.rank,
            token_axis: part.token_axis,
            flags: part.flags,
            dimensions,
            byte_strides: strides,
            payload_offset: payload_bytes,
            payload_bytes: bytes,
        });
        payload_bytes = payload_bytes
            .checked_add(bytes)
            .context("activation size overflow")?;
    }
    Ok(activation_frame_wire_bytes(
        codec,
        &StageActivationDesc {
            version: STAGE_ACTIVATION_FRAME_VERSION,
            producer_stage_index: 0,
            layer_start: 0,
            layer_end: 1,
            token_count: u32::try_from(tokens).context("prefill token count overflow")?,
            sequence_count: 1,
            payload_bytes,
            frontier_identity: boundary.frontier_identity,
            parts,
        },
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefill_chunks_include_headers_sidebands_and_actual_sampling() {
        let mut boundary = ActivationBoundaryDesc {
            version: skippy_runtime::ACTIVATION_BOUNDARY_DESC_VERSION,
            part_count: 1,
            ..Default::default()
        };
        boundary.parts[0] = skippy_runtime::ActivationPartDesc {
            ggml_type: 0,
            rank: 2,
            token_axis: 1,
            dimensions: [4096, 1, 1, 1],
            ..Default::default()
        };
        let message =
            StageWireMessage::configure_generation(1, 2, 1024, Some(Default::default()), None);

        let chunk =
            bounded_chunk_size(1024, &boundary, StageActivationCodec::RawF32V1, &message).unwrap();

        let size = |tokens| {
            message.estimated_wire_bytes()
                + tokens * 4
                + activation_bytes(&boundary, StageActivationCodec::RawF32V1, tokens).unwrap()
        };
        assert!(size(chunk) <= skippy_protocol::MAX_STAGE_FRAME_BYTES);
        assert!(size(chunk + 1) > skippy_protocol::MAX_STAGE_FRAME_BYTES);
    }
}
