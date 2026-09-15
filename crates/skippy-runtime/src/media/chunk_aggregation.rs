use anyhow::{Context, Result, anyhow};

use crate::{ActivationFrame, MediaPrefillChunkFrame};

pub(super) fn aggregate_media_chunk_outputs(
    chunks: &[MediaPrefillChunkFrame],
) -> Result<ActivationFrame> {
    let first = chunks
        .first()
        .ok_or_else(|| anyhow!("multimodal prefill produced no activation output"))?;
    let mut desc = first.output.desc;
    let mut token_count = 0usize;
    let mut common_flags = desc.flags;

    for (index, chunk) in chunks.iter().enumerate() {
        let frame = &chunk.output;
        if desc.version != frame.desc.version
            || desc.dtype != frame.desc.dtype
            || desc.layout != frame.desc.layout
            || desc.producer_stage_index != frame.desc.producer_stage_index
            || desc.layer_start != frame.desc.layer_start
            || desc.layer_end != frame.desc.layer_end
            || desc.sequence_count != frame.desc.sequence_count
        {
            return Err(anyhow!(
                "multimodal chunk {index} produced incompatible activation descriptor"
            ));
        }
        token_count = token_count
            .checked_add(chunk.token_count)
            .context("multimodal activation token count overflow")?;
        common_flags &= frame.desc.flags;
    }

    let differing_flags = chunks
        .iter()
        .fold(0_u64, |flags, chunk| flags | chunk.output.desc.flags)
        & !common_flags;
    if differing_flags & !skippy_ffi::ACTIVATION_FLAG_INKLING_MTP_EMBD != 0 {
        return Err(anyhow!(
            "multimodal chunks produced incompatible activation sideband flags {differing_flags:#x}"
        ));
    }

    let mut payload = Vec::new();
    if differing_flags == skippy_ffi::ACTIVATION_FLAG_INKLING_MTP_EMBD {
        let base_chunk = chunks
            .iter()
            .find(|chunk| {
                chunk.output.desc.flags & skippy_ffi::ACTIVATION_FLAG_INKLING_MTP_EMBD == 0
            })
            .ok_or_else(|| anyhow!("multimodal Inkling output has no base activation chunk"))?;
        let bytes_per_token = base_chunk
            .output
            .payload
            .len()
            .checked_div(base_chunk.token_count)
            .filter(|_| base_chunk.output.payload.len() % base_chunk.token_count == 0)
            .ok_or_else(|| anyhow!("multimodal base activation payload is not token-aligned"))?;
        for (index, chunk) in chunks.iter().enumerate() {
            let hidden_bytes = bytes_per_token
                .checked_mul(chunk.token_count)
                .context("multimodal base activation byte count overflow")?;
            let hidden = chunk.output.payload.get(..hidden_bytes).ok_or_else(|| {
                anyhow!("multimodal chunk {index} is smaller than its base activation payload")
            })?;
            payload.extend_from_slice(hidden);
        }
        desc.flags = common_flags;
    } else if common_flags & skippy_ffi::ACTIVATION_FLAG_INKLING_MTP_EMBD != 0 {
        let mut hidden_planes = Vec::new();
        let mut mtp_planes = Vec::new();
        for (index, chunk) in chunks.iter().enumerate() {
            if chunk.output.payload.len() % 2 != 0 {
                return Err(anyhow!(
                    "multimodal Inkling chunk {index} sideband payload is not evenly split"
                ));
            }
            let plane_bytes = chunk.output.payload.len() / 2;
            hidden_planes.extend_from_slice(&chunk.output.payload[..plane_bytes]);
            mtp_planes.extend_from_slice(&chunk.output.payload[plane_bytes..]);
        }
        payload = hidden_planes;
        payload.extend_from_slice(&mtp_planes);
    } else {
        for chunk in chunks {
            payload.extend_from_slice(&chunk.output.payload);
        }
    }

    desc.token_count = u32::try_from(token_count).context("multimodal token count exceeds u32")?;
    desc.payload_bytes =
        u64::try_from(payload.len()).context("multimodal activation payload length exceeds u64")?;
    Ok(ActivationFrame { desc, payload })
}

#[cfg(test)]
mod tests {
    use super::aggregate_media_chunk_outputs;
    use crate::{
        ActivationDesc, ActivationFrame, MediaPrefillChunkFrame, RuntimeActivationDType,
        RuntimeActivationLayout,
    };

    fn chunk(token_count: usize, flags: u64, payload: Vec<u8>) -> MediaPrefillChunkFrame {
        MediaPrefillChunkFrame {
            token_count,
            tokens: Vec::new(),
            positions: Vec::new(),
            output: ActivationFrame {
                desc: ActivationDesc {
                    version: 1,
                    dtype: RuntimeActivationDType::F32,
                    layout: RuntimeActivationLayout::TokenMajor,
                    producer_stage_index: 0,
                    layer_start: 0,
                    layer_end: 1,
                    token_count: token_count as u32,
                    sequence_count: 1,
                    payload_bytes: payload.len() as u64,
                    flags,
                },
                payload,
            },
        }
    }

    #[test]
    fn aggregation_rejects_empty_output() {
        assert!(aggregate_media_chunk_outputs(&[]).is_err());
    }

    #[test]
    fn aggregation_rejects_different_stage_descriptors() {
        let first = chunk(1, 0, vec![1, 2, 3, 4]);
        let mut second = chunk(1, 0, vec![5, 6, 7, 8]);
        second.output.desc.producer_stage_index = 1;
        let error = aggregate_media_chunk_outputs(&[first, second]).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("incompatible activation descriptor")
        );
    }

    #[test]
    fn only_inkling_sidebands_may_differ_between_chunks() {
        for flag in [
            skippy_ffi::ACTIVATION_FLAG_GLM_DSA_TOP_K,
            skippy_ffi::ACTIVATION_FLAG_KIMI_K3_RESIDUAL,
        ] {
            let chunks = [
                chunk(1, 0, vec![1, 2, 3, 4]),
                chunk(1, flag, vec![5, 6, 7, 8]),
            ];
            let error = aggregate_media_chunk_outputs(&chunks).unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("incompatible activation sideband flags")
            );
        }
    }

    #[test]
    fn mixed_inkling_chunks_aggregate_the_common_hidden_plane() -> anyhow::Result<()> {
        let chunks = vec![
            chunk(1, 0, vec![1, 2, 3, 4]),
            chunk(
                2,
                skippy_ffi::ACTIVATION_FLAG_INKLING_MTP_EMBD,
                vec![5, 6, 7, 8, 9, 10, 11, 12, 21, 22, 23, 24, 25, 26, 27, 28],
            ),
        ];

        let output = aggregate_media_chunk_outputs(&chunks)?;

        assert_eq!(output.desc.token_count, 3);
        assert_eq!(output.desc.flags, 0);
        assert_eq!(output.desc.payload_bytes, 12);
        assert_eq!(output.payload, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        Ok(())
    }

    #[test]
    fn uniform_inkling_chunks_aggregate_each_plane_in_token_order() -> anyhow::Result<()> {
        let flag = skippy_ffi::ACTIVATION_FLAG_INKLING_MTP_EMBD;
        let chunks = vec![
            chunk(1, flag, vec![1, 2, 3, 4, 11, 12, 13, 14]),
            chunk(1, flag, vec![5, 6, 7, 8, 15, 16, 17, 18]),
        ];

        let output = aggregate_media_chunk_outputs(&chunks)?;

        assert_eq!(output.desc.token_count, 2);
        assert_eq!(output.desc.flags, flag);
        assert_eq!(output.desc.payload_bytes, 16);
        assert_eq!(
            output.payload,
            vec![1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18]
        );
        Ok(())
    }
}
