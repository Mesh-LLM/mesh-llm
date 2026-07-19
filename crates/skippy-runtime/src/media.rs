use std::ffi::{CStr, CString};
use std::ptr;

use anyhow::{Context, Result, anyhow};
use skippy_ffi::Model as RawModel;

use crate::error::{ensure_ok, free_error};
use crate::native::StageModel;
use crate::path_cstring::path_to_cstring;
use crate::session::StageSession;
use crate::{
    ActivationDesc, ActivationFrame, MediaInput, MediaPrefill, MediaPrefillChunkFrame,
    MediaPrefillFrame, SamplingConfig,
};

pub(crate) struct MediaProjector {
    pub(crate) raw: *mut skippy_ffi::MtmdContext,
}

type MediaFrameEval = (
    usize,
    u64,
    Vec<i32>,
    ActivationFrame,
    Vec<MediaPrefillChunkFrame>,
);

// The experimental C ABI owns synchronization internally for model/session use.
// Rust stage-server access is additionally serialized behind a Mutex.
unsafe impl Send for MediaProjector {}

impl MediaProjector {
    pub(crate) fn open(path: &str, model: *mut RawModel) -> Result<Self> {
        let path = path_to_cstring(std::path::Path::new(path), "projector path")?;
        let raw_model = unsafe { skippy_ffi::skippy_model_llama_model(model) };
        if raw_model.is_null() {
            return Err(anyhow!("model did not expose a llama_model handle"));
        }
        let mut params = unsafe { skippy_ffi::mtmd_context_params_default() };
        params.use_gpu = true;
        let raw = unsafe { skippy_ffi::mtmd_init_from_file(path.as_ptr(), raw_model, params) };
        if raw.is_null() {
            return Err(anyhow!("failed to load multimodal projector {path:?}"));
        }
        Ok(Self { raw })
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
        MediaProjector::marker()
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
        }
        impl Drop for Bitmap {
            fn drop(&mut self) {
                if !self.raw.is_null() {
                    unsafe {
                        skippy_ffi::mtmd_bitmap_free(self.raw);
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
            let raw = unsafe {
                skippy_ffi::mtmd_helper_bitmap_init_from_buf(
                    projector.raw,
                    item.bytes.as_ptr(),
                    item.bytes.len(),
                )
            };
            if raw.is_null() {
                return Err(anyhow!("failed to decode media item for projector"));
            }
            bitmaps.push(Bitmap { raw });
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
                0,
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
        }
        impl Drop for Bitmap {
            fn drop(&mut self) {
                if !self.raw.is_null() {
                    unsafe {
                        skippy_ffi::mtmd_bitmap_free(self.raw);
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
            let raw = unsafe {
                skippy_ffi::mtmd_helper_bitmap_init_from_buf(
                    projector.raw,
                    item.bytes.as_ptr(),
                    item.bytes.len(),
                )
            };
            if raw.is_null() {
                return Err(anyhow!("failed to decode media item for projector"));
            }
            bitmaps.push(Bitmap { raw });
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
        let mut output_desc: Option<ActivationDesc> = None;
        let mut output_payload = Vec::new();
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
                    0,
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
            if let Some(desc) = output_desc.as_ref() {
                if desc.version != frame.desc.version
                    || desc.dtype != frame.desc.dtype
                    || desc.layout != frame.desc.layout
                    || desc.producer_stage_index != frame.desc.producer_stage_index
                    || desc.layer_start != frame.desc.layer_start
                    || desc.layer_end != frame.desc.layer_end
                    || desc.sequence_count != frame.desc.sequence_count
                    || desc.flags != frame.desc.flags
                {
                    return Err(anyhow!(
                        "multimodal chunk {index} produced incompatible activation descriptor"
                    ));
                }
            } else {
                output_desc = Some(frame.desc);
            }
            copied_tokens = copied_tokens
                .checked_add(chunk_tokens)
                .context("multimodal activation token count overflow")?;
            output_payload.extend_from_slice(&frame.payload);
            chunk_frames.push(MediaPrefillChunkFrame {
                token_count: chunk_tokens,
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
        let mut desc = output_desc
            .ok_or_else(|| anyhow!("multimodal prefill produced no activation output"))?;
        desc.token_count =
            u32::try_from(copied_tokens).context("multimodal token count exceeds u32")?;
        desc.payload_bytes = u64::try_from(output_payload.len())
            .context("multimodal activation payload length exceeds u64")?;
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
            ActivationFrame {
                desc,
                payload: output_payload,
            },
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
