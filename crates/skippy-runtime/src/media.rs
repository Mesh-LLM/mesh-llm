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

/// Audio encoding returned by the native speech generator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeechOutputFormat {
    Wav,
    PcmS16Le,
}

/// Full-model speech inputs and deterministic sampling controls.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeechSynthesisConfig {
    pub prompt: String,
    pub language: Option<String>,
    pub top_k: i32,
    pub top_p: f32,
    pub seed: u32,
    pub output_format: SpeechOutputFormat,
    pub max_frames: usize,
}

/// Complete generated audio and its native frame count. Reaching the configured
/// frame limit returns an error instead of this response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeechAudio {
    pub bytes: Vec<u8>,
    pub sample_rate: u32,
    pub sample_count: u64,
    pub generated_frames: usize,
}

/// Restore the session's generation mode on every speech exit, including
/// failures before the external-decode guard can be acquired.
struct SpeechEmbeddingsGuard(*mut skippy_ffi::Opaque);

impl Drop for SpeechEmbeddingsGuard {
    fn drop(&mut self) {
        // SAFETY: the borrowed StageSession outlives this guard and owns the
        // context; speech generation holds exclusive access to the session.
        unsafe { skippy_ffi::llama_set_embeddings(self.0, false) };
    }
}

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

    pub fn supports_speech_synthesis(&self) -> bool {
        self.media.as_ref().is_some_and(|projector| {
            let info = unsafe { skippy_ffi::mtmd_gen_audio_get_info(projector.raw) };
            info.audio_type != skippy_ffi::MtmdGenAudioType::None
        })
    }

    /// Generate bounded audio while restoring the session's normal decode mode
    /// on success, cancellation, and native failure. Sessions remain exclusive.
    pub fn synthesize_speech(
        &self,
        session: &mut StageSession,
        config: &SpeechSynthesisConfig,
        cancellation_requested: impl Fn() -> bool,
    ) -> Result<SpeechAudio> {
        let projector = self
            .media
            .as_ref()
            .ok_or_else(|| anyhow!("speech synthesis requires a configured projector"))?;
        let info = unsafe { skippy_ffi::mtmd_gen_audio_get_info(projector.raw) };
        if info.audio_type == skippy_ffi::MtmdGenAudioType::None {
            return Err(anyhow!(
                "configured projector does not support speech synthesis"
            ));
        }
        if config.prompt.is_empty() || config.max_frames == 0 {
            return Err(anyhow!(
                "speech prompt and max_frames must not be empty or zero"
            ));
        }
        let prompt = CString::new(config.prompt.as_bytes())
            .context("speech prompt contains an interior NUL byte")?;
        let language = config
            .language
            .as_deref()
            .map(CString::new)
            .transpose()
            .context("speech language contains an interior NUL byte")?;
        let lctx = unsafe { skippy_ffi::skippy_session_llama_context(session.raw) };
        if lctx.is_null() {
            return Err(anyhow!("speech session did not expose a llama context"));
        }

        struct AudioGenerator(*mut skippy_ffi::MtmdHelperGenAudio);
        impl Drop for AudioGenerator {
            fn drop(&mut self) {
                if !self.0.is_null() {
                    unsafe { skippy_ffi::mtmd_helper_gen_audio_free(self.0) };
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
        session.reset()?;
        unsafe { skippy_ffi::llama_set_embeddings(lctx, true) };
        let _embeddings_mode = SpeechEmbeddingsGuard(lctx);
        let mut guard_error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_session_begin_external_decode(session.raw, &mut guard_error)
        };
        ensure_ok(status, guard_error)?;
        let _external_decode = ExternalDecodeGuard(session.raw);
        let generator =
            AudioGenerator(unsafe { skippy_ffi::mtmd_helper_gen_audio_init(lctx, projector.raw) });
        if generator.0.is_null() {
            return Err(anyhow!("failed to initialize speech synthesis pipeline"));
        }
        let output_type = match config.output_format {
            SpeechOutputFormat::Wav => skippy_ffi::MtmdHelperGenAudioOutputType::Wav,
            SpeechOutputFormat::PcmS16Le => skippy_ffi::MtmdHelperGenAudioOutputType::Pcm,
        };
        let input = skippy_ffi::MtmdHelperGenAudioInput {
            seq_id: session.native_sequence_id()?,
            prompt: prompt.as_ptr(),
            prompt_len: config.prompt.len(),
            speaker_ref: ptr::null_mut(),
            lang: language
                .as_ref()
                .map_or(ptr::null(), |value| value.as_ptr()),
            top_k: config.top_k,
            top_p: config.top_p,
            seed: config.seed,
            out_type: output_type,
        };
        if unsafe { skippy_ffi::mtmd_helper_gen_audio_set_input(generator.0, &input) } != 0 {
            return Err(anyhow!("speech synthesis rejected the input"));
        }
        let batch_size =
            i32::try_from(session.batch_size()?).context("speech batch size exceeds i32")?;
        loop {
            if cancellation_requested() {
                return Err(anyhow!("speech synthesis cancelled"));
            }
            let remaining =
                unsafe { skippy_ffi::mtmd_helper_gen_audio_step_prompt(generator.0, batch_size) };
            if remaining < 0 {
                return Err(anyhow!("speech prompt evaluation failed"));
            }
            if remaining == 0 {
                break;
            }
        }

        let sampling = SamplingConfig {
            enabled: true,
            seed: config.seed,
            top_k: config.top_k,
            top_p: config.top_p,
            ..SamplingConfig::default()
        };
        let mut sampled = session.sample_current(Some(&sampling))?;
        let mut hidden_state =
            unsafe { skippy_ffi::llama_get_embeddings_ith(lctx, -1) }.cast_const();
        if hidden_state.is_null() {
            return Err(anyhow!("speech backbone did not produce a hidden state"));
        }
        let mut generated_frames = 0usize;
        let mut stopped = false;
        while generated_frames < config.max_frames {
            if cancellation_requested() {
                return Err(anyhow!("speech synthesis cancelled"));
            }
            let mut next_hidden_state = ptr::null();
            let mut stop = false;
            let step = unsafe {
                skippy_ffi::mtmd_helper_gen_audio_step_gen(
                    generator.0,
                    sampled,
                    hidden_state,
                    &mut next_hidden_state,
                    &mut stop,
                )
            };
            if step != 0 {
                return Err(anyhow!(
                    "speech synthesis failed at frame {generated_frames}"
                ));
            }
            if stop || next_hidden_state.is_null() {
                stopped = true;
                break;
            }
            generated_frames += 1;
            hidden_state = next_hidden_state;
            sampled = session.sample_current(Some(&sampling))?;
        }
        if !stopped {
            return Err(anyhow!(
                "speech synthesis exceeded the configured {} frame limit",
                config.max_frames
            ));
        }

        let mut sample_rate = 0_i32;
        let mut data = ptr::null();
        let mut data_len = 0usize;
        let mut sample_count = 0_i64;
        let output_status = unsafe {
            skippy_ffi::mtmd_helper_gen_audio_get_output(
                generator.0,
                &mut sample_rate,
                &mut data,
                &mut data_len,
                &mut sample_count,
            )
        };
        if output_status != 0 || data.is_null() || data_len == 0 {
            return Err(anyhow!("speech synthesis produced no audio"));
        }
        let native_bytes = unsafe { std::slice::from_raw_parts(data.cast::<u8>(), data_len) };
        let bytes = match config.output_format {
            SpeechOutputFormat::Wav => native_bytes.to_vec(),
            SpeechOutputFormat::PcmS16Le => pcm_f32_to_s16le(native_bytes)?,
        };
        Ok(SpeechAudio {
            bytes,
            sample_rate: u32::try_from(sample_rate).context("invalid speech sample rate")?,
            sample_count: u64::try_from(sample_count).context("invalid speech sample count")?,
            generated_frames,
        })
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

fn pcm_f32_to_s16le(bytes: &[u8]) -> Result<Vec<u8>> {
    if !bytes.len().is_multiple_of(std::mem::size_of::<f32>()) {
        return Err(anyhow!("native PCM payload is not aligned to f32 samples"));
    }
    let mut output = Vec::with_capacity(bytes.len() / 2);
    let (samples, remainder) = bytes.as_chunks::<4>();
    debug_assert!(remainder.is_empty());
    for sample in samples {
        let sample = f32::from_ne_bytes(*sample);
        let quantized = (sample.clamp(-1.0, 1.0) * f32::from(i16::MAX)).round() as i16;
        output.extend_from_slice(&quantized.to_le_bytes());
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::pcm_f32_to_s16le;

    #[test]
    fn pcm_conversion_clamps_and_quantizes_native_float_samples() {
        let samples = [-2.0_f32, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
        let bytes = samples
            .iter()
            .flat_map(|sample| sample.to_ne_bytes())
            .collect::<Vec<_>>();

        let converted = pcm_f32_to_s16le(&bytes).expect("aligned native PCM");
        let (converted_samples, remainder) = converted.as_chunks::<2>();
        assert!(remainder.is_empty());
        let actual = converted_samples
            .iter()
            .map(|sample| i16::from_le_bytes(*sample))
            .collect::<Vec<_>>();

        assert_eq!(
            actual,
            vec![-32_767, -32_767, -16_384, 0, 16_384, 32_767, 32_767]
        );
    }

    #[test]
    fn pcm_conversion_rejects_misaligned_native_payload() {
        assert!(pcm_f32_to_s16le(&[0, 1, 2]).is_err());
    }
}

#[cfg(test)]
#[path = "media/speech_session_tests.rs"]
mod speech_session_tests;
