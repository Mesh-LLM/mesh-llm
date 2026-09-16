use std::ffi::{CStr, CString};
use std::ptr;

use anyhow::{Context, Result, anyhow};
use skippy_ffi::Model as RawModel;

use crate::error::{ensure_ok, free_error};
use crate::native::StageModel;
use crate::path_cstring::path_to_cstring;
use crate::session::StageSession;
use crate::{
    ActivationFrame, MediaInput, MediaPrefill, MediaPrefillChunkFrame, MediaPrefillFrame,
    SamplingConfig,
};

pub(crate) struct MediaProjector {
    pub(crate) raw: *mut skippy_ffi::MtmdContext,
    marker: String,
}

type MediaFrameEval = (
    usize,
    u64,
    Vec<i32>,
    ActivationFrame,
    Vec<MediaPrefillChunkFrame>,
);

fn aggregate_media_chunk_outputs(chunks: &[MediaPrefillChunkFrame]) -> Result<ActivationFrame> {
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

// The experimental C ABI owns synchronization internally for model/session use.
// Rust stage-server access is additionally serialized behind a Mutex.
unsafe impl Send for MediaProjector {}

impl MediaProjector {
    pub(crate) fn open(
        path: &str,
        model: *mut RawModel,
        config: &crate::RuntimeConfig,
    ) -> Result<Self> {
        let path = path_to_cstring(std::path::Path::new(path), "projector path")?;
        let raw_model = unsafe { skippy_ffi::skippy_model_llama_model(model) };
        if raw_model.is_null() {
            return Err(anyhow!("model did not expose a llama_model handle"));
        }
        let mut params = unsafe { skippy_ffi::mtmd_context_params_default() };
        if let Some(use_gpu) = config.projector_use_gpu {
            params.use_gpu = use_gpu;
        }
        let marker = config
            .media_marker
            .as_deref()
            .map(CString::new)
            .transpose()
            .context("media_marker contains an interior NUL byte")?;
        if let Some(marker) = marker.as_ref() {
            params.media_marker = marker.as_ptr();
        }
        if let Some(value) = config.image_min_tokens {
            params.image_min_tokens =
                i32::try_from(value).context("image_min_tokens exceeds i32")?;
        }
        if let Some(value) = config.image_max_tokens {
            params.image_max_tokens =
                i32::try_from(value).context("image_max_tokens exceeds i32")?;
        }
        if let Some(value) = config.batch_max_tokens {
            params.batch_max_tokens =
                i32::try_from(value).context("batch_max_tokens exceeds i32")?;
        }
        let raw = unsafe { skippy_ffi::mtmd_init_from_file(path.as_ptr(), raw_model, params) };
        if raw.is_null() {
            return Err(anyhow!("failed to load multimodal projector {path:?}"));
        }
        Ok(Self {
            raw,
            marker: config.media_marker.clone().unwrap_or_else(Self::marker),
        })
    }

    fn marker() -> String {
        let marker = unsafe { skippy_ffi::mtmd_default_marker() };
        if marker.is_null() {
            "<__media__>".to_string()
        } else {
            unsafe { CStr::from_ptr(marker) }
                .to_string_lossy()
                .into_owned()
        }
    }
}

impl Drop for MediaProjector {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                skippy_ffi::mtmd_free(self.raw);
            }
        }
    }
}

impl StageModel {
    pub fn media_marker(&self) -> String {
        self.media
            .as_ref()
            .map(|projector| projector.marker.clone())
            .unwrap_or_else(MediaProjector::marker)
    }

    pub fn has_media_projector(&self) -> bool {
        self.media.is_some()
    }

    fn eval_media(
        &self,
        session: &mut StageSession,
        prompt: &str,
        media: &[MediaInput],
    ) -> Result<(usize, u64)> {
        let projector = self
            .media
            .as_ref()
            .ok_or_else(|| anyhow!("model was not loaded with a multimodal projector"))?;
        if media.is_empty() {
            return Err(anyhow!("media prefill requires at least one media item"));
        }
        if prompt.is_empty() {
            return Err(anyhow!("media prompt must not be empty"));
        }

        struct Bitmap {
            raw: *mut skippy_ffi::MtmdBitmap,
            video: *mut skippy_ffi::MtmdHelperVideo,
        }
        impl Drop for Bitmap {
            fn drop(&mut self) {
                if !self.raw.is_null() {
                    unsafe {
                        skippy_ffi::mtmd_bitmap_free(self.raw);
                    }
                }
                if !self.video.is_null() {
                    unsafe {
                        skippy_ffi::mtmd_helper_video_free(self.video);
                    }
                }
            }
        }
        struct Chunks {
            raw: *mut skippy_ffi::MtmdInputChunks,
        }
        impl Drop for Chunks {
            fn drop(&mut self) {
                if !self.raw.is_null() {
                    unsafe {
                        skippy_ffi::mtmd_input_chunks_free(self.raw);
                    }
                }
            }
        }

        let mut bitmaps = Vec::with_capacity(media.len());
        for item in media {
            if item.bytes.is_empty() {
                return Err(anyhow!("media item must not be empty"));
            }
            let wrapper = unsafe {
                skippy_ffi::mtmd_helper_bitmap_init_from_buf(
                    projector.raw,
                    item.bytes.as_ptr(),
                    item.bytes.len(),
                    false,
                    skippy_ffi::mtmd_helper_init_opt_default(),
                )
            };
            // Take ownership before the null check: on a partial failure the
            // wrapper can still carry a video context that has to be freed.
            let bitmap = Bitmap {
                raw: wrapper.bitmap,
                video: wrapper.video_ctx,
            };
            if bitmap.raw.is_null() {
                return Err(anyhow!("failed to decode media item for projector"));
            }
            bitmaps.push(bitmap);
        }

        let chunks = Chunks {
            raw: unsafe { skippy_ffi::mtmd_input_chunks_init() },
        };
        if chunks.raw.is_null() {
            return Err(anyhow!("failed to allocate multimodal input chunks"));
        }
        let prompt = CString::new(prompt.as_bytes())
            .context("multimodal prompt contains an interior NUL byte")?;
        let input_text = skippy_ffi::MtmdInputText {
            text: prompt.as_ptr(),
            text_len: prompt.as_bytes().len(),
            add_special: true,
            parse_special: true,
        };
        let bitmap_ptrs = bitmaps
            .iter()
            .map(|bitmap| bitmap.raw.cast_const())
            .collect::<Vec<_>>();
        let tokenize_status = unsafe {
            skippy_ffi::mtmd_tokenize(
                projector.raw,
                chunks.raw,
                &input_text,
                bitmap_ptrs.as_ptr(),
                bitmap_ptrs.len(),
            )
        };
        if tokenize_status != 0 {
            return Err(anyhow!(
                "multimodal tokenization failed with status {tokenize_status}"
            ));
        }

        let token_count = unsafe { skippy_ffi::mtmd_helper_get_n_tokens(chunks.raw) };
        if token_count == 0 {
            return Err(anyhow!("multimodal prompt produced no tokens"));
        }
        let n_past = unsafe { skippy_ffi::skippy_session_position(session.raw) };
        if n_past < 0 {
            return Err(anyhow!("skippy session is not initialized"));
        }
        let seq_id = session.native_sequence_id()?;
        let n_batch = unsafe { skippy_ffi::skippy_session_batch_size(session.raw) };
        if n_batch <= 0 {
            return Err(anyhow!("skippy session has no valid batch size"));
        }
        let lctx = unsafe { skippy_ffi::skippy_session_llama_context(session.raw) };
        if lctx.is_null() {
            return Err(anyhow!(
                "skippy session did not expose a llama_context handle"
            ));
        }
        let mut guard_error = ptr::null_mut();
        let guard_status = unsafe {
            skippy_ffi::skippy_session_begin_external_decode(session.raw, &mut guard_error)
        };
        ensure_ok(guard_status, guard_error)?;

        struct ExternalDecodeGuard(*mut skippy_ffi::Session);

        impl Drop for ExternalDecodeGuard {
            fn drop(&mut self) {
                let mut error = ptr::null_mut();
                unsafe {
                    let _ = skippy_ffi::skippy_session_end_external_decode(self.0, &mut error);
                }
                free_error(error);
            }
        }

        let _external_decode_guard = ExternalDecodeGuard(session.raw);

        let mut new_n_past = 0_i32;
        let eval_status = unsafe {
            skippy_ffi::mtmd_helper_eval_chunks(
                projector.raw,
                lctx,
                chunks.raw,
                n_past,
                seq_id,
                n_batch,
                true,
                &mut new_n_past,
            )
        };
        if eval_status != 0 {
            return Err(anyhow!(
                "multimodal prompt evaluation failed with status {eval_status}"
            ));
        }

        let mut error = ptr::null_mut();
        let status =
            unsafe { skippy_ffi::skippy_session_set_position(session.raw, new_n_past, &mut error) };
        ensure_ok(status, error)?;
        session.token_count =
            u64::try_from(new_n_past).context("multimodal position is negative")?;

        Ok((token_count, session.token_count))
    }

    fn eval_media_frame(
        &self,
        session: &mut StageSession,
        prompt: &str,
        media: &[MediaInput],
    ) -> Result<MediaFrameEval> {
        let projector = self
            .media
            .as_ref()
            .ok_or_else(|| anyhow!("model was not loaded with a multimodal projector"))?;
        if media.is_empty() {
            return Err(anyhow!("media prefill requires at least one media item"));
        }
        if prompt.is_empty() {
            return Err(anyhow!("media prompt must not be empty"));
        }

        struct Bitmap {
            raw: *mut skippy_ffi::MtmdBitmap,
            video: *mut skippy_ffi::MtmdHelperVideo,
        }
        impl Drop for Bitmap {
            fn drop(&mut self) {
                if !self.raw.is_null() {
                    unsafe {
                        skippy_ffi::mtmd_bitmap_free(self.raw);
                    }
                }
                if !self.video.is_null() {
                    unsafe {
                        skippy_ffi::mtmd_helper_video_free(self.video);
                    }
                }
            }
        }
        struct Chunks {
            raw: *mut skippy_ffi::MtmdInputChunks,
        }
        impl Drop for Chunks {
            fn drop(&mut self) {
                if !self.raw.is_null() {
                    unsafe {
                        skippy_ffi::mtmd_input_chunks_free(self.raw);
                    }
                }
            }
        }
        struct ExternalDecodeGuard(*mut skippy_ffi::Session);
        impl Drop for ExternalDecodeGuard {
            fn drop(&mut self) {
                let mut error = ptr::null_mut();
                unsafe {
                    let _ = skippy_ffi::skippy_session_end_external_decode(self.0, &mut error);
                }
                free_error(error);
            }
        }

        let mut bitmaps = Vec::with_capacity(media.len());
        for item in media {
            if item.bytes.is_empty() {
                return Err(anyhow!("media item must not be empty"));
            }
            let wrapper = unsafe {
                skippy_ffi::mtmd_helper_bitmap_init_from_buf(
                    projector.raw,
                    item.bytes.as_ptr(),
                    item.bytes.len(),
                    false,
                    skippy_ffi::mtmd_helper_init_opt_default(),
                )
            };
            // Take ownership before the null check: on a partial failure the
            // wrapper can still carry a video context that has to be freed.
            let bitmap = Bitmap {
                raw: wrapper.bitmap,
                video: wrapper.video_ctx,
            };
            if bitmap.raw.is_null() {
                return Err(anyhow!("failed to decode media item for projector"));
            }
            bitmaps.push(bitmap);
        }

        let chunks = Chunks {
            raw: unsafe { skippy_ffi::mtmd_input_chunks_init() },
        };
        if chunks.raw.is_null() {
            return Err(anyhow!("failed to allocate multimodal input chunks"));
        }
        let prompt = CString::new(prompt.as_bytes())
            .context("multimodal prompt contains an interior NUL byte")?;
        let input_text = skippy_ffi::MtmdInputText {
            text: prompt.as_ptr(),
            text_len: prompt.as_bytes().len(),
            add_special: true,
            parse_special: true,
        };
        let bitmap_ptrs = bitmaps
            .iter()
            .map(|bitmap| bitmap.raw.cast_const())
            .collect::<Vec<_>>();
        let tokenize_status = unsafe {
            skippy_ffi::mtmd_tokenize(
                projector.raw,
                chunks.raw,
                &input_text,
                bitmap_ptrs.as_ptr(),
                bitmap_ptrs.len(),
            )
        };
        if tokenize_status != 0 {
            return Err(anyhow!(
                "multimodal tokenization failed with status {tokenize_status}"
            ));
        }

        let token_count = unsafe { skippy_ffi::mtmd_helper_get_n_tokens(chunks.raw) };
        if token_count == 0 {
            return Err(anyhow!("multimodal prompt produced no tokens"));
        }
        let mut n_past = unsafe { skippy_ffi::skippy_session_position(session.raw) };
        if n_past < 0 {
            return Err(anyhow!("skippy session is not initialized"));
        }
        let seq_id = session.native_sequence_id()?;
        let n_batch = unsafe { skippy_ffi::skippy_session_batch_size(session.raw) };
        if n_batch <= 0 {
            return Err(anyhow!("skippy session has no valid batch size"));
        }
        let lctx = unsafe { skippy_ffi::skippy_session_llama_context(session.raw) };
        if lctx.is_null() {
            return Err(anyhow!(
                "skippy session did not expose a llama_context handle"
            ));
        }

        let mut guard_error = ptr::null_mut();
        let guard_status = unsafe {
            skippy_ffi::skippy_session_begin_external_decode(session.raw, &mut guard_error)
        };
        ensure_ok(guard_status, guard_error)?;
        let _external_decode_guard = ExternalDecodeGuard(session.raw);

        let chunk_count = unsafe { skippy_ffi::mtmd_input_chunks_size(chunks.raw) };
        let use_mrope = unsafe { skippy_ffi::mtmd_decode_use_mrope(projector.raw) };
        let mut token_positions = Vec::<[i32; 4]>::new();
        let mut chunk_frames = Vec::new();
        let mut copied_tokens = 0usize;
        for index in 0..chunk_count {
            let chunk = unsafe { skippy_ffi::mtmd_input_chunks_get(chunks.raw, index) };
            if chunk.is_null() {
                return Err(anyhow!("multimodal chunk {index} is null"));
            }
            let chunk_type = unsafe { skippy_ffi::mtmd_input_chunk_get_type(chunk) };
            let chunk_tokens = unsafe { skippy_ffi::mtmd_input_chunk_get_n_tokens(chunk) };
            if chunk_tokens == 0 {
                continue;
            }
            if chunk_tokens > n_batch as usize {
                return Err(anyhow!(
                    "multimodal chunk {index} has {chunk_tokens} tokens, exceeding n_batch {n_batch}; increase n_batch for staged media prefill"
                ));
            }
            let chunk_token_ids = if chunk_type == skippy_ffi::MtmdInputChunkType::Text {
                let mut text_token_count = 0usize;
                let text_tokens = unsafe {
                    skippy_ffi::mtmd_input_chunk_get_tokens_text(chunk, &mut text_token_count)
                };
                if text_tokens.is_null() || text_token_count != chunk_tokens {
                    return Err(anyhow!(
                        "multimodal text chunk {index} token view did not match its declared token count"
                    ));
                }
                unsafe { std::slice::from_raw_parts(text_tokens, text_token_count) }.to_vec()
            } else {
                Vec::new()
            };
            let chunk_positions = if use_mrope {
                let chunk_positions = match chunk_type {
                    skippy_ffi::MtmdInputChunkType::Image => {
                        let image_tokens =
                            unsafe { skippy_ffi::mtmd_input_chunk_get_tokens_image(chunk) };
                        if image_tokens.is_null() {
                            return Err(anyhow!(
                                "multimodal image chunk {index} has no image tokens"
                            ));
                        }
                        let mut positions = vec![
                            skippy_ffi::MtmdDecoderPos {
                                t: 0,
                                x: 0,
                                y: 0,
                                z: 0,
                            };
                            chunk_tokens
                        ];
                        unsafe {
                            skippy_ffi::mtmd_helper_image_get_decoder_pos(
                                image_tokens,
                                n_past,
                                positions.as_mut_ptr(),
                            );
                        }
                        positions
                            .into_iter()
                            .map(|position| {
                                [
                                    i32::try_from(position.t).unwrap_or(i32::MAX),
                                    i32::try_from(position.y).unwrap_or(i32::MAX),
                                    i32::try_from(position.x).unwrap_or(i32::MAX),
                                    i32::try_from(position.z).unwrap_or(i32::MAX),
                                ]
                            })
                            .collect::<Vec<_>>()
                    }
                    _ => (0..chunk_tokens)
                        .map(|offset| {
                            let position = n_past.saturating_add(offset as i32);
                            [position, position, position, 0]
                        })
                        .collect::<Vec<_>>(),
                };
                token_positions.extend(chunk_positions.iter().copied());
                let mut flattened = Vec::with_capacity(chunk_tokens * 4);
                for dim in 0..4 {
                    flattened.extend(chunk_positions.iter().map(|position| position[dim]));
                }
                flattened
            } else {
                Vec::new()
            };
            let mut new_n_past = n_past;
            let eval_status = unsafe {
                skippy_ffi::mtmd_helper_eval_chunk_single(
                    projector.raw,
                    lctx,
                    chunk,
                    n_past,
                    seq_id,
                    n_batch,
                    false,
                    &mut new_n_past,
                )
            };
            if eval_status != 0 {
                return Err(anyhow!(
                    "multimodal chunk {index} evaluation failed with status {eval_status}"
                ));
            }
            let frame = session.copy_output_activation_frame(chunk_tokens, 0)?;
            copied_tokens = copied_tokens
                .checked_add(chunk_tokens)
                .context("multimodal activation token count overflow")?;
            chunk_frames.push(MediaPrefillChunkFrame {
                token_count: chunk_tokens,
                tokens: chunk_token_ids,
                positions: chunk_positions,
                output: frame,
            });
            n_past = new_n_past;
        }

        let mut error = ptr::null_mut();
        let status =
            unsafe { skippy_ffi::skippy_session_set_position(session.raw, n_past, &mut error) };
        ensure_ok(status, error)?;
        session.token_count = u64::try_from(n_past).context("multimodal position is negative")?;

        if copied_tokens != token_count {
            return Err(anyhow!(
                "multimodal activation tokens copied {copied_tokens} did not match prompt tokens {token_count}"
            ));
        }
        let output = aggregate_media_chunk_outputs(&chunk_frames)?;
        let positions = if use_mrope {
            let mut positions = Vec::with_capacity(copied_tokens * 4);
            for dim in 0..4 {
                positions.extend(token_positions.iter().map(|position| position[dim]));
            }
            positions
        } else {
            Vec::new()
        };
        Ok((
            token_count,
            session.token_count,
            positions,
            output,
            chunk_frames,
        ))
    }

    pub fn prefill_media(
        &self,
        session: &mut StageSession,
        prompt: &str,
        media: &[MediaInput],
        sampling: Option<&SamplingConfig>,
    ) -> Result<MediaPrefill> {
        let (token_count, position) = self.eval_media(session, prompt, media)?;

        let first_token = session.sample_current(sampling)?;

        Ok(MediaPrefill {
            token_count,
            position,
            first_token,
        })
    }

    pub fn prefill_media_frame(
        &self,
        session: &mut StageSession,
        prompt: &str,
        media: &[MediaInput],
    ) -> Result<MediaPrefillFrame> {
        let (token_count, position, positions, output, chunks) =
            self.eval_media_frame(session, prompt, media)?;
        Ok(MediaPrefillFrame {
            token_count,
            position,
            positions,
            output,
            chunks,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::aggregate_media_chunk_outputs;
    use crate::{
        ACTIVATION_FRAME_VERSION, ACTIVATION_IDENTITY_BYTES, ACTIVATION_MAX_PARTS,
        ACTIVATION_PART_OPTIONAL, ActivationDesc, ActivationFrame, ActivationPartDesc,
        GGML_TYPE_F32, MediaPrefillChunkFrame,
    };

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
