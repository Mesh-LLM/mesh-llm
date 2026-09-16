use anyhow::{Context, Result, bail};
use skippy_protocol::binary::{
    StageActivationDesc, StageActivationFrame, StageActivationPartDesc, WireMessageKind,
};
use skippy_runtime::{ActivationDesc, ActivationFrame, ActivationPartDesc};

pub(in crate::binary_transport) fn stage_activation_desc(
    desc: &ActivationDesc,
) -> Result<StageActivationDesc> {
    Ok(StageActivationDesc {
        version: desc.version,
        producer_stage_index: desc.producer_stage_index,
        layer_start: desc.layer_start,
        layer_end: desc.layer_end,
        token_count: desc.token_count,
        sequence_count: desc.sequence_count,
        payload_bytes: desc.payload_bytes,
        frontier_identity: desc.frontier_identity,
        parts: desc
            .parts()?
            .iter()
            .map(|part| StageActivationPartDesc {
                identity: part.identity,
                ggml_type: part.ggml_type,
                rank: part.rank,
                token_axis: part.token_axis,
                flags: part.flags,
                dimensions: part.dimensions,
                byte_strides: part.byte_strides,
                payload_offset: part.payload_offset,
                payload_bytes: part.payload_bytes,
            })
            .collect(),
    })
}

pub(in crate::binary_transport) fn runtime_activation_frame(
    frame: StageActivationFrame,
) -> Result<ActivationFrame> {
    if frame.desc.parts.len() > skippy_runtime::ACTIVATION_MAX_PARTS {
        bail!("wire activation part count exceeds runtime maximum");
    }
    let mut parts = [ActivationPartDesc::default(); skippy_runtime::ACTIVATION_MAX_PARTS];
    for (output, part) in parts.iter_mut().zip(&frame.desc.parts) {
        *output = ActivationPartDesc {
            identity: part.identity,
            ggml_type: part.ggml_type,
            rank: part.rank,
            token_axis: part.token_axis,
            flags: part.flags,
            dimensions: part.dimensions,
            byte_strides: part.byte_strides,
            payload_offset: part.payload_offset,
            payload_bytes: part.payload_bytes,
        };
    }
    Ok(ActivationFrame {
        desc: ActivationDesc {
            version: frame.desc.version,
            producer_stage_index: frame.desc.producer_stage_index,
            layer_start: frame.desc.layer_start,
            layer_end: frame.desc.layer_end,
            token_count: frame.desc.token_count,
            sequence_count: frame.desc.sequence_count,
            part_count: u32::try_from(frame.desc.parts.len())
                .context("wire activation part count exceeds u32")?,
            payload_bytes: frame.desc.payload_bytes,
            frontier_identity: frame.desc.frontier_identity,
            parts,
        },
        payload: frame.payload,
    })
}

pub(in crate::binary_transport) fn executable_prefill_start(
    kind: WireMessageKind,
    restored_tokens: usize,
    token_count: usize,
    layer_start: u32,
    has_downstream: bool,
) -> usize {
    let partial_restore = restored_tokens > 0 && restored_tokens < token_count;
    if kind.is_prefill() && partial_restore && (layer_start == 0 || !has_downstream) {
        restored_tokens
    } else {
        0
    }
}

pub(in crate::binary_transport) fn suffix_activation_frame(
    input: Option<ActivationFrame>,
    token_start: usize,
) -> Result<Option<ActivationFrame>> {
    let Some(frame) = input else {
        return Ok(None);
    };
    if token_start == 0 {
        return Ok(Some(frame));
    }
    let token_count =
        usize::try_from(frame.desc.token_count).context("activation token count overflow")?;
    if token_start >= token_count {
        bail!("suffix activation start {token_start} exceeds frame token count {token_count}");
    }
    let suffix_tokens = token_count - token_start;
    let mut payload = Vec::new();
    let mut parts = [ActivationPartDesc::default(); skippy_runtime::ACTIVATION_MAX_PARTS];
    for (part_index, source) in frame.desc.parts()?.iter().enumerate() {
        let rank = usize::try_from(source.rank).context("activation part rank exceeds usize")?;
        let token_axis =
            usize::try_from(source.token_axis).context("activation part token axis is negative")?;
        if rank == 0 || rank > source.dimensions.len() || token_axis >= rank {
            bail!("activation part has an invalid token axis");
        }
        let inner_bytes = usize::try_from(source.byte_strides[token_axis])
            .context("activation part token stride exceeds usize")?;
        let plane_bytes = inner_bytes
            .checked_mul(token_count)
            .context("activation part plane size overflow")?;
        let source_bytes = usize::try_from(source.payload_bytes)
            .context("activation part payload size exceeds usize")?;
        if plane_bytes == 0 || !source_bytes.is_multiple_of(plane_bytes) {
            bail!("activation part payload is not aligned to its token axis");
        }
        let outer_count = source_bytes / plane_bytes;
        let source_offset = usize::try_from(source.payload_offset)
            .context("activation part payload offset exceeds usize")?;
        let output_offset = payload.len();
        let suffix_offset = token_start
            .checked_mul(inner_bytes)
            .context("activation part suffix offset overflow")?;
        for outer_index in 0..outer_count {
            let plane_start = source_offset
                .checked_add(
                    outer_index
                        .checked_mul(plane_bytes)
                        .context("activation part plane offset overflow")?,
                )
                .context("activation part plane offset overflow")?;
            let start = plane_start
                .checked_add(suffix_offset)
                .context("activation part suffix range overflow")?;
            let end = plane_start
                .checked_add(plane_bytes)
                .context("activation part suffix range overflow")?;
            payload.extend_from_slice(
                frame
                    .payload
                    .get(start..end)
                    .ok_or_else(|| anyhow::anyhow!("activation part exceeds frame payload"))?,
            );
        }
        let mut output = *source;
        output.dimensions[token_axis] =
            i64::try_from(suffix_tokens).context("activation suffix token count exceeds i64")?;
        for axis in token_axis + 1..rank {
            output.byte_strides[axis] = output.byte_strides[axis - 1]
                .checked_mul(
                    u64::try_from(output.dimensions[axis - 1])
                        .context("activation part dimension is unresolved")?,
                )
                .context("activation part output stride overflow")?;
        }
        output.payload_offset =
            u64::try_from(output_offset).context("activation part output offset exceeds u64")?;
        output.payload_bytes = u64::try_from(payload.len() - output_offset)
            .context("activation part output size exceeds u64")?;
        parts[part_index] = output;
    }
    let mut desc = frame.desc;
    desc.token_count = u32::try_from(suffix_tokens).context("suffix token count overflow")?;
    desc.sequence_count = if suffix_tokens > 0 { 1 } else { 0 };
    desc.payload_bytes = payload.len() as u64;
    desc.parts = parts;
    Ok(Some(ActivationFrame { desc, payload }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn final_non_first_stage_executes_only_suffix_after_partial_restore() {
        assert_eq!(
            executable_prefill_start(WireMessageKind::PrefillEmbd, 3, 5, 8, false),
            3
        );
    }

    #[test]
    fn intermediate_non_first_stage_preserves_full_activation_range() {
        assert_eq!(
            executable_prefill_start(WireMessageKind::PrefillEmbd, 3, 5, 8, true),
            0
        );
    }

    fn frame(token_count: u32, row_bytes: usize) -> ActivationFrame {
        let mut payload = Vec::with_capacity(row_bytes * token_count as usize);
        for row_idx in 0..token_count {
            // Each row is filled with a distinct byte value (the row index)
            payload.extend(vec![row_idx as u8; row_bytes]);
        }
        crate::test_activation::frame(
            token_count,
            vec![crate::test_activation::PartBytes {
                identity: 1,
                ggml_type: skippy_runtime::GGML_TYPE_F32,
                flags: 0,
                bytes: payload,
            }],
        )
    }

    #[test]
    fn suffix_frame_slices_payload_rows_and_rebuilds_descriptor() {
        let sliced = suffix_activation_frame(Some(frame(5, 8)), 3)
            .unwrap()
            .unwrap();
        assert_eq!(sliced.desc.token_count, 2);
        assert_eq!(sliced.desc.sequence_count, 1);
        assert_eq!(sliced.payload.len(), 16);
        assert_eq!(sliced.desc.payload_bytes, 16);
        // Verify the payload contains rows 3 and 4 from the original frame
        // Row 3: 8 bytes of 0x03, Row 4: 8 bytes of 0x04
        let mut expected = vec![3_u8; 8];
        expected.extend(vec![4_u8; 8]);
        assert_eq!(&sliced.payload, &expected);
    }

    #[test]
    fn suffix_frame_slices_each_part_by_token() {
        let mut parts = Vec::new();
        for (part_index, plane) in [0_u8, 10].into_iter().enumerate() {
            let mut bytes = Vec::new();
            for row_idx in 0..5 {
                bytes.extend(vec![plane + row_idx; 4]);
            }
            parts.push(crate::test_activation::PartBytes {
                identity: part_index as u8 + 1,
                ggml_type: skippy_runtime::GGML_TYPE_F32,
                flags: 0,
                bytes,
            });
        }
        let input = crate::test_activation::frame(5, parts);

        let sliced = suffix_activation_frame(Some(input), 3).unwrap().unwrap();
        let mut expected = vec![3_u8; 4];
        expected.extend(vec![4_u8; 4]);
        expected.extend(vec![13_u8; 4]);
        expected.extend(vec![14_u8; 4]);
        assert_eq!(sliced.payload, expected);
        assert_eq!(sliced.desc.token_count, 2);
        assert_eq!(sliced.desc.payload_bytes, 16);
        assert_eq!(sliced.desc.part_count, 2);
    }

    #[test]
    fn suffix_frame_start_zero_is_identity() {
        let original = frame(5, 8);
        let sliced = suffix_activation_frame(Some(original.clone()), 0)
            .unwrap()
            .unwrap();
        assert_eq!(sliced.desc.token_count, original.desc.token_count);
        assert_eq!(sliced.payload.len(), original.payload.len());
    }

    #[test]
    fn suffix_frame_none_passes_through() {
        assert!(suffix_activation_frame(None, 3).unwrap().is_none());
    }

    #[test]
    fn suffix_frame_rejects_start_beyond_token_count() {
        assert!(suffix_activation_frame(Some(frame(5, 8)), 5).is_err());
        assert!(suffix_activation_frame(Some(frame(5, 8)), 6).is_err());
    }
}
