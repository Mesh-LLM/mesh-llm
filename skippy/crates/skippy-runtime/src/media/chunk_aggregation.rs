use anyhow::{Context, Result, anyhow};

use crate::{ActivationFrame, MediaPrefillChunkFrame};

/// Join native media chunks by typed frontier identity, retaining only shared optional parts.
pub(super) fn aggregate_media_chunk_outputs(
    chunks: &[MediaPrefillChunkFrame],
) -> Result<ActivationFrame> {
    let first = chunks
        .first()
        .ok_or_else(|| anyhow!("multimodal prefill produced no activation output"))?;
    let mut desc = first.output.desc;
    let mut token_count = 0usize;

    for (index, chunk) in chunks.iter().enumerate() {
        let frame = &chunk.output;
        if desc.version != frame.desc.version
            || desc.producer_stage_index != frame.desc.producer_stage_index
            || desc.layer_start != frame.desc.layer_start
            || desc.layer_end != frame.desc.layer_end
            || desc.sequence_count != frame.desc.sequence_count
            || desc.frontier_identity != frame.desc.frontier_identity
        {
            return Err(anyhow!(
                "multimodal chunk {index} produced incompatible activation descriptor"
            ));
        }
        if chunk.token_count != frame.desc.token_count as usize {
            return Err(anyhow!(
                "multimodal chunk {index} token count does not match its activation descriptor"
            ));
        }
        token_count = token_count
            .checked_add(chunk.token_count)
            .context("multimodal activation token count overflow")?;
    }

    for (chunk_index, chunk) in chunks.iter().enumerate() {
        for part in chunk.output.desc.parts()? {
            if !part.is_optional()
                && chunks.iter().any(|candidate| {
                    candidate.output.desc.parts().map_or(true, |parts| {
                        !parts.iter().any(|item| item.identity == part.identity)
                    })
                })
            {
                return Err(anyhow!(
                    "multimodal chunk {chunk_index} has a required activation part missing from another chunk"
                ));
            }
        }
    }

    let mut payload = Vec::new();
    let mut output_parts = [crate::ActivationPartDesc::default(); skippy_ffi::ACTIVATION_MAX_PARTS];
    let first_parts = first.output.desc.parts()?;
    let mut output_part_count = 0usize;
    for first_part in first_parts {
        let matching = chunks
            .iter()
            .map(|chunk| {
                chunk
                    .output
                    .desc
                    .parts()?
                    .iter()
                    .find(|part| part.identity == first_part.identity)
                    .copied()
                    .ok_or_else(|| anyhow!("optional activation part is not common to every chunk"))
            })
            .collect::<Result<Vec<_>>>();
        let matching = match matching {
            Ok(parts) => parts,
            Err(_) if first_part.is_optional() => continue,
            Err(error) => return Err(error),
        };

        for (chunk_index, part) in matching.iter().enumerate() {
            let token_axis = usize::try_from(part.token_axis)
                .context("activation part has a negative token axis")?;
            let rank = usize::try_from(part.rank).context("activation part rank exceeds usize")?;
            if rank == 0 || rank > part.dimensions.len() || token_axis >= rank {
                return Err(anyhow!(
                    "multimodal chunk {chunk_index} has an invalid activation part shape"
                ));
            }
            if part.ggml_type != first_part.ggml_type
                || part.rank != first_part.rank
                || part.token_axis != first_part.token_axis
                || part.flags != first_part.flags
                || part.dimensions[..rank]
                    .iter()
                    .enumerate()
                    .any(|(axis, dimension)| {
                        axis != token_axis && *dimension != first_part.dimensions[axis]
                    })
                || part.byte_strides[..=token_axis] != first_part.byte_strides[..=token_axis]
            {
                return Err(anyhow!(
                    "multimodal chunk {chunk_index} produced incompatible metadata for activation part"
                ));
            }
            if part.dimensions[token_axis] != chunks[chunk_index].token_count as i64 {
                return Err(anyhow!(
                    "multimodal chunk {chunk_index} activation part token dimension does not match its chunk"
                ));
            }
        }

        let token_axis = first_part.token_axis as usize;
        let inner_bytes = usize::try_from(first_part.byte_strides[token_axis])
            .context("activation part inner stride exceeds usize")?;
        if inner_bytes == 0 {
            return Err(anyhow!("activation part has a zero token-axis stride"));
        }
        let mut slab_bytes = Vec::with_capacity(chunks.len());
        let mut outer_count = None;
        for (chunk_index, (chunk, part)) in chunks.iter().zip(&matching).enumerate() {
            let slab = inner_bytes
                .checked_mul(chunk.token_count)
                .context("multimodal activation part slab size overflow")?;
            let part_bytes = usize::try_from(part.payload_bytes)
                .context("activation part payload size exceeds usize")?;
            if slab == 0 || part_bytes % slab != 0 {
                return Err(anyhow!(
                    "multimodal chunk {chunk_index} activation part is not token-aligned"
                ));
            }
            let current_outer_count = part_bytes / slab;
            if outer_count
                .replace(current_outer_count)
                .is_some_and(|value| value != current_outer_count)
            {
                return Err(anyhow!(
                    "multimodal activation parts have incompatible outer dimensions"
                ));
            }
            slab_bytes.push(slab);
        }

        let output_offset = payload.len();
        for outer_index in 0..outer_count.unwrap_or(0) {
            for (chunk_index, (chunk, part)) in chunks.iter().zip(&matching).enumerate() {
                let part_offset = usize::try_from(part.payload_offset)
                    .context("activation part offset exceeds usize")?;
                let start = part_offset
                    .checked_add(
                        outer_index
                            .checked_mul(slab_bytes[chunk_index])
                            .context("activation part offset overflow")?,
                    )
                    .context("activation part offset overflow")?;
                let end = start
                    .checked_add(slab_bytes[chunk_index])
                    .context("activation part range overflow")?;
                let bytes = chunk.output.payload.get(start..end).ok_or_else(|| {
                    anyhow!("multimodal chunk {chunk_index} activation part exceeds its payload")
                })?;
                payload.extend_from_slice(bytes);
            }
        }

        let mut output_part = *first_part;
        output_part.dimensions[token_axis] =
            i64::try_from(token_count).context("multimodal activation token count exceeds i64")?;
        for axis in token_axis + 1..output_part.rank as usize {
            let previous_dimension = u64::try_from(output_part.dimensions[axis - 1])
                .context("activation part has an unresolved output dimension")?;
            output_part.byte_strides[axis] = output_part.byte_strides[axis - 1]
                .checked_mul(previous_dimension)
                .context("activation part output stride overflow")?;
        }
        output_part.payload_offset = u64::try_from(output_offset)
            .context("multimodal activation payload offset exceeds u64")?;
        output_part.payload_bytes = u64::try_from(payload.len() - output_offset)
            .context("multimodal activation part size exceeds u64")?;
        output_parts[output_part_count] = output_part;
        output_part_count += 1;
    }

    desc.token_count = u32::try_from(token_count).context("multimodal token count exceeds u32")?;
    desc.part_count =
        u32::try_from(output_part_count).context("activation part count exceeds u32")?;
    desc.payload_bytes =
        u64::try_from(payload.len()).context("multimodal activation payload length exceeds u64")?;
    desc.parts = output_parts;
    Ok(ActivationFrame { desc, payload })
}

#[cfg(test)]
mod tests {
    use super::aggregate_media_chunk_outputs;
    use crate::{
        ACTIVATION_FRAME_VERSION, ACTIVATION_IDENTITY_BYTES, ACTIVATION_MAX_PARTS,
        ACTIVATION_PART_OPTIONAL, ActivationDesc, ActivationFrame, ActivationPartDesc,
        GGML_TYPE_F32, MediaPrefillChunkFrame,
    };

    /// Build identity-tagged activation planes with independently controllable descriptors.
    fn chunk(token_count: usize, parts: &[(u8, u32, Vec<u8>)]) -> MediaPrefillChunkFrame {
        let mut payload = Vec::new();
        let mut descriptors = [ActivationPartDesc::default(); ACTIVATION_MAX_PARTS];
        for (index, (identity, flags, bytes)) in parts.iter().enumerate() {
            descriptors[index] = ActivationPartDesc {
                identity: [*identity; ACTIVATION_IDENTITY_BYTES],
                ggml_type: GGML_TYPE_F32,
                rank: 2,
                token_axis: 1,
                flags: *flags,
                dimensions: [1, token_count as i64, 0, 0],
                byte_strides: [4, 4, 0, 0],
                payload_offset: payload.len() as u64,
                payload_bytes: bytes.len() as u64,
            };
            payload.extend_from_slice(bytes);
        }
        MediaPrefillChunkFrame {
            token_count,
            tokens: Vec::new(),
            positions: Vec::new(),
            output: ActivationFrame {
                desc: ActivationDesc {
                    version: ACTIVATION_FRAME_VERSION,
                    producer_stage_index: 0,
                    layer_start: 0,
                    layer_end: 1,
                    token_count: token_count as u32,
                    sequence_count: 1,
                    part_count: parts.len() as u32,
                    payload_bytes: payload.len() as u64,
                    frontier_identity: [9; ACTIVATION_IDENTITY_BYTES],
                    parts: descriptors,
                },
                payload,
            },
        }
    }

    /// An absent native output must not become an apparently valid activation frame.
    #[test]
    fn aggregation_rejects_empty_output() {
        assert!(aggregate_media_chunk_outputs(&[]).is_err());
    }

    /// Chunks from different producer stages cannot form one downstream activation.
    #[test]
    fn aggregation_rejects_different_stage_descriptors() {
        let first = chunk(1, &[(1, 0, vec![1, 2, 3, 4])]);
        let mut second = first.clone();
        second.output.desc.producer_stage_index = 1;
        let error = aggregate_media_chunk_outputs(&[first, second]).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("incompatible activation descriptor")
        );
    }

    /// Matching dimensions do not make distinct model frontiers interchangeable.
    #[test]
    fn aggregation_rejects_different_frontier_identities() {
        let first = chunk(1, &[(1, 0, vec![1, 2, 3, 4])]);
        let mut second = first.clone();
        second.output.desc.frontier_identity[0] ^= 1;
        let error = aggregate_media_chunk_outputs(&[first, second]).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("incompatible activation descriptor")
        );
    }

    /// Required planes must cover every chunk regardless of which chunk introduces them.
    #[test]
    fn required_parts_cannot_disappear_between_chunks() {
        let first = chunk(1, &[(1, 0, vec![1, 2, 3, 4])]);
        let second = chunk(1, &[(1, 0, vec![5, 6, 7, 8]), (2, 0, vec![9, 10, 11, 12])]);
        for chunks in [[first.clone(), second.clone()], [second, first]] {
            let error = aggregate_media_chunk_outputs(&chunks).unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("required activation part missing")
            );
        }
    }

    /// The wrapper token count must agree with its native activation descriptor.
    #[test]
    fn aggregation_rejects_inconsistent_chunk_token_count() {
        let mut invalid = chunk(1, &[(1, 0, vec![1, 2, 3, 4])]);
        invalid.token_count = 2;
        let error = aggregate_media_chunk_outputs(&[invalid]).unwrap_err();
        assert!(error.to_string().contains("token count does not match"));
    }

    /// Invalid plane offsets are rejected before copying activation bytes.
    #[test]
    fn aggregation_rejects_parts_outside_the_payload() {
        let mut invalid = chunk(1, &[(1, 0, vec![1, 2, 3, 4])]);
        invalid.output.desc.parts[0].payload_offset = 4;
        let error = aggregate_media_chunk_outputs(&[invalid]).unwrap_err();
        assert!(error.to_string().contains("exceeds its payload"));
    }

    /// Omit optional planes with incomplete token coverage instead of padding invented bytes.
    #[test]
    fn optional_part_in_first_chunk_is_dropped_when_absent_later() -> anyhow::Result<()> {
        let chunks = [
            chunk(
                1,
                &[
                    (1, 0, vec![1, 2, 3, 4]),
                    (2, ACTIVATION_PART_OPTIONAL, vec![9, 10, 11, 12]),
                ],
            ),
            chunk(1, &[(1, 0, vec![5, 6, 7, 8])]),
        ];
        let output = aggregate_media_chunk_outputs(&chunks)?;
        assert_eq!(output.desc.part_count, 1);
        assert_eq!(output.desc.parts[0].dimensions[1], 2);
        assert_eq!(output.payload, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        Ok(())
    }

    /// Plane identity, not descriptor position, determines cross-chunk concatenation.
    #[test]
    fn part_order_does_not_change_identity_based_aggregation() -> anyhow::Result<()> {
        let chunks = [
            chunk(1, &[(1, 0, vec![1, 2, 3, 4]), (2, 0, vec![11, 12, 13, 14])]),
            chunk(1, &[(2, 0, vec![15, 16, 17, 18]), (1, 0, vec![5, 6, 7, 8])]),
        ];
        let output = aggregate_media_chunk_outputs(&chunks)?;
        assert_eq!(output.desc.part_count, 2);
        assert_eq!(output.desc.parts[1].payload_offset, 8);
        assert_eq!(
            output.payload,
            vec![1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18]
        );
        Ok(())
    }

    /// Mixed text/media output retains only the shared hidden plane in original token order.
    #[test]
    fn mixed_inkling_chunks_aggregate_the_common_hidden_plane() -> anyhow::Result<()> {
        let chunks = vec![
            chunk(1, &[(1, 0, vec![1, 2, 3, 4])]),
            chunk(
                2,
                &[
                    (1, 0, vec![5, 6, 7, 8, 9, 10, 11, 12]),
                    (
                        2,
                        ACTIVATION_PART_OPTIONAL,
                        vec![21, 22, 23, 24, 25, 26, 27, 28],
                    ),
                ],
            ),
        ];

        let output = aggregate_media_chunk_outputs(&chunks)?;

        assert_eq!(output.desc.token_count, 3);
        assert_eq!(output.desc.part_count, 1);
        assert_eq!(output.desc.payload_bytes, 12);
        assert_eq!(output.payload, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        Ok(())
    }

    /// Uniform multimodal chunks retain every fully covered plane and its payload offsets.
    #[test]
    fn uniform_inkling_chunks_aggregate_each_plane_in_token_order() -> anyhow::Result<()> {
        let chunks = vec![
            chunk(
                1,
                &[
                    (1, 0, vec![1, 2, 3, 4]),
                    (2, ACTIVATION_PART_OPTIONAL, vec![11, 12, 13, 14]),
                ],
            ),
            chunk(
                1,
                &[
                    (1, 0, vec![5, 6, 7, 8]),
                    (2, ACTIVATION_PART_OPTIONAL, vec![15, 16, 17, 18]),
                ],
            ),
        ];

        let output = aggregate_media_chunk_outputs(&chunks)?;

        assert_eq!(output.desc.token_count, 2);
        assert_eq!(output.desc.part_count, 2);
        assert_eq!(output.desc.payload_bytes, 16);
        assert_eq!(
            output.payload,
            vec![1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18]
        );
        Ok(())
    }
}
