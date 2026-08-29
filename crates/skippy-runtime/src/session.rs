use std::ffi::CString;
use std::ptr;

use anyhow::{Context, Result, anyhow};
use skippy_ffi::{
    GenerationSignalWindow as RawGenerationSignalWindow, NativeMtpDraft as RawNativeMtpDraft,
    SamplingConfig as RawSamplingConfig, Session as RawSession, TokenSignal as RawTokenSignal,
};

use crate::error::ensure_ok;
use crate::{GenerationSignalWindow, NativeMtpDraft, SamplingConfig, TokenSignal};

const EMPTY_CHAT_GRAMMAR_METADATA: &str = r#"{"grammar":""}"#;

fn chat_sampling_metadata_for_native(metadata_json: &str) -> &str {
    // Template metadata can contain a large serialized PEG response parser,
    // but native sampling only consults grammar fields. The renderer emits
    // compact JSON, so an empty top-level grammar can use the minimal
    // equivalent payload and avoid reparsing the unrelated PEG graph before
    // first-token decode. Escaped JSON nested inside `chat_parser` cannot
    // contain this unescaped byte sequence.
    if metadata_json.contains(r#""grammar":"""#) {
        EMPTY_CHAT_GRAMMAR_METADATA
    } else {
        metadata_json
    }
}

pub struct StageSession {
    pub(crate) raw: *mut RawSession,
    pub(crate) token_count: u64,
    pub(crate) include_output: bool,
}

pub struct DecodeBatchRequest<'a> {
    pub session: &'a mut StageSession,
    pub token_id: i32,
    pub sampling: Option<&'a SamplingConfig>,
}

fn native_position_to_u64(native_position: i32) -> Result<u64> {
    u64::try_from(native_position)
        .map_err(|_| anyhow!("skippy session has no valid native position"))
}

// The experimental C ABI owns synchronization internally for model/session use.
// Rust stage-server access is additionally serialized behind a Mutex.
unsafe impl Send for StageSession {}

impl StageSession {
    pub fn token_count(&self) -> u64 {
        self.token_count
    }

    /// Returns the position reported by the native runtime.
    pub fn native_position(&self) -> Result<u64> {
        let native_position = unsafe { skippy_ffi::skippy_session_position(self.raw) };
        native_position_to_u64(native_position)
    }

    pub fn batch_size(&self) -> Result<usize> {
        let n_batch = unsafe { skippy_ffi::skippy_session_batch_size(self.raw) };
        if n_batch <= 0 {
            return Err(anyhow!("skippy session has no valid batch size"));
        }
        usize::try_from(n_batch).context("session batch size exceeds usize")
    }

    pub fn reset(&mut self) -> Result<()> {
        let mut error = ptr::null_mut();
        let status = unsafe { skippy_ffi::skippy_session_reset(self.raw, &mut error) };
        ensure_ok(status, error)?;
        self.token_count = 0;
        Ok(())
    }

    pub fn configure_chat_sampling(
        &mut self,
        metadata_json: &str,
        prompt_token_count: u64,
        sampling: Option<&SamplingConfig>,
    ) -> Result<()> {
        let metadata_json = CString::new(chat_sampling_metadata_for_native(metadata_json))
            .context("chat sampling metadata contains an interior NUL byte")?;
        let raw_sampling = sampling.map(SamplingConfig::as_raw);
        let sampling_ptr = raw_sampling
            .as_ref()
            .map_or(ptr::null(), |sampling| sampling as *const RawSamplingConfig);
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_session_configure_chat_sampling(
                self.raw,
                sampling_ptr,
                metadata_json.as_ptr(),
                prompt_token_count,
                &mut error,
            )
        };
        ensure_ok(status, error)
    }

    /// Retires the exact recovery checkpoint for a fully accepted verify window.
    pub fn retire_verify_checkpoint(&mut self, token_start: u64, token_count: u64) -> Result<()> {
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_retire_verify_checkpoint(
                self.raw,
                token_start,
                token_count,
                &mut error,
            )
        };
        ensure_ok(status, error)
    }

    pub fn trim_session(&mut self, token_count: u64) -> Result<()> {
        let mut error = ptr::null_mut();
        let status = unsafe { skippy_ffi::skippy_trim_session(self.raw, token_count, &mut error) };
        ensure_ok(status, error)?;
        self.token_count = token_count;
        Ok(())
    }

    pub fn set_position(&mut self, token_count: u64) -> Result<()> {
        let n_past = i32::try_from(token_count).context("session position exceeds i32")?;
        let mut error = ptr::null_mut();
        let status =
            unsafe { skippy_ffi::skippy_session_set_position(self.raw, n_past, &mut error) };
        ensure_ok(status, error)?;
        self.token_count = token_count;
        Ok(())
    }

    pub fn save_prefix(&mut self, cache_seq_id: i32, token_count: u64) -> Result<()> {
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_session_save_prefix(self.raw, cache_seq_id, token_count, &mut error)
        };
        ensure_ok(status, error)
    }

    pub fn restore_prefix(&mut self, cache_seq_id: i32, token_ids: &[i32]) -> Result<()> {
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_session_restore_prefix(
                self.raw,
                cache_seq_id,
                token_ids.as_ptr(),
                token_ids.len(),
                &mut error,
            )
        };
        ensure_ok(status, error)?;
        self.token_count = u64::try_from(token_ids.len()).context("token count exceeds u64")?;
        Ok(())
    }

    pub fn drop_sequence(&mut self, seq_id: i32) -> Result<()> {
        let mut error = ptr::null_mut();
        let status =
            unsafe { skippy_ffi::skippy_session_drop_sequence(self.raw, seq_id, &mut error) };
        ensure_ok(status, error)
    }

    pub fn prefill_chunk(&mut self, token_ids: &[i32]) -> Result<()> {
        let mut output_bytes = 0usize;
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_prefill_chunk(
                self.raw,
                token_ids.as_ptr(),
                token_ids.len(),
                ptr::null(),
                0,
                ptr::null_mut(),
                0,
                &mut output_bytes,
                &mut error,
            )
        };
        ensure_ok(status, error)?;
        self.token_count = self
            .token_count
            .checked_add(u64::try_from(token_ids.len()).context("token count exceeds u64")?)
            .context("session token count overflow")?;
        Ok(())
    }

    pub fn prefill_chunked(&mut self, token_ids: &[i32]) -> Result<()> {
        if token_ids.is_empty() {
            return Ok(());
        }
        let batch_size = self.batch_size()?.max(1);
        for chunk in token_ids.chunks(batch_size) {
            self.prefill_chunk(chunk)?;
        }
        Ok(())
    }

    pub fn decode_step(&mut self, token_id: i32) -> Result<i32> {
        self.decode_step_sampled(token_id, None)
    }

    pub fn decode_step_sampled(
        &mut self,
        token_id: i32,
        sampling: Option<&SamplingConfig>,
    ) -> Result<i32> {
        let mut output_bytes = 0usize;
        let mut predicted_token = 0_i32;
        let mut error = ptr::null_mut();
        let raw_sampling = sampling.map(SamplingConfig::as_raw);
        let sampling_ptr = raw_sampling
            .as_ref()
            .map_or(ptr::null(), |sampling| sampling as *const RawSamplingConfig);
        let status = unsafe {
            skippy_ffi::skippy_decode_step_sampled(
                self.raw,
                token_id,
                sampling_ptr,
                ptr::null(),
                0,
                ptr::null_mut(),
                0,
                &mut output_bytes,
                &mut predicted_token,
                &mut error,
            )
        };
        ensure_ok(status, error)?;
        self.token_count = self
            .token_count
            .checked_add(1)
            .context("session token count overflow")?;
        Ok(predicted_token)
    }

    pub fn decode_step_sampled_mtp(
        &mut self,
        token_id: i32,
        sampling: Option<&SamplingConfig>,
        max_draft_tokens: usize,
    ) -> Result<(i32, Option<NativeMtpDraft>)> {
        if skippy_ffi::skippy_decode_step_sampled_mtp_fn().is_none() {
            let (predicted, draft, _) =
                self.decode_step_frame_sampled_mtp(token_id, sampling, None, 0, max_draft_tokens)?;
            return Ok((predicted, draft));
        }

        let mut predicted_token = 0_i32;
        let mut mtp_draft = RawNativeMtpDraft::default();
        let mut error = ptr::null_mut();
        let raw_sampling = sampling.map(SamplingConfig::as_raw);
        let sampling_ptr = raw_sampling
            .as_ref()
            .map_or(ptr::null(), |sampling| sampling as *const RawSamplingConfig);
        let status = unsafe {
            skippy_ffi::skippy_decode_step_sampled_mtp(
                self.raw,
                token_id,
                sampling_ptr,
                &mut predicted_token,
                max_draft_tokens.min(skippy_ffi::NATIVE_MTP_MAX_DRAFT_TOKENS),
                &mut mtp_draft,
                &mut error,
            )
        };
        ensure_ok(status, error)?;
        self.token_count = self
            .token_count
            .checked_add(1)
            .context("session token count overflow")?;
        Ok((predicted_token, NativeMtpDraft::from_raw(mtp_draft)))
    }

    pub fn decode_batch_sampled(requests: &mut [DecodeBatchRequest<'_>]) -> Result<Vec<i32>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }

        let sessions = requests
            .iter_mut()
            .map(|request| request.session.raw)
            .collect::<Vec<_>>();
        let token_ids = requests
            .iter()
            .map(|request| request.token_id)
            .collect::<Vec<_>>();
        let raw_sampling = requests
            .iter()
            .map(|request| request.sampling.map(SamplingConfig::as_raw))
            .collect::<Vec<_>>();
        let sampling = raw_sampling
            .iter()
            .map(|sampling| {
                sampling
                    .as_ref()
                    .map_or(ptr::null(), |sampling| sampling as *const RawSamplingConfig)
            })
            .collect::<Vec<_>>();
        let mut predicted_tokens = vec![0_i32; requests.len()];
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_decode_batch_sampled(
                sessions.as_ptr(),
                token_ids.as_ptr(),
                sampling.as_ptr(),
                requests.len(),
                predicted_tokens.as_mut_ptr(),
                predicted_tokens.len(),
                &mut error,
            )
        };
        ensure_ok(status, error)?;
        for request in requests {
            request.session.token_count = request
                .session
                .token_count
                .checked_add(1)
                .context("session token count overflow")?;
        }
        Ok(predicted_tokens)
    }

    pub fn last_token_signal(&mut self) -> Result<TokenSignal> {
        let mut signal = RawTokenSignal::default();
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_session_last_token_signal(self.raw, &mut signal, &mut error)
        };
        ensure_ok(status, error)?;
        Ok(signal.into())
    }

    pub fn signal_window(&mut self, window_tokens: u32) -> Result<GenerationSignalWindow> {
        let mut window = RawGenerationSignalWindow::default();
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_session_signal_window(
                self.raw,
                window_tokens,
                &mut window,
                &mut error,
            )
        };
        ensure_ok(status, error)?;
        Ok(window.into())
    }

    /// Verifies a speculative window and advances the session.
    ///
    /// Call [`Self::retire_verify_checkpoint`] when the complete window is accepted,
    /// or [`Self::trim_session`] when any suffix is rejected.
    pub fn verify_tokens(&mut self, token_ids: &[i32]) -> Result<Vec<i32>> {
        if token_ids.is_empty() {
            return Ok(Vec::new());
        }
        let mut predicted = vec![0_i32; token_ids.len()];
        let mut output_count = 0usize;
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_verify_tokens(
                self.raw,
                token_ids.as_ptr(),
                token_ids.len(),
                predicted.as_mut_ptr(),
                predicted.len(),
                &mut output_count,
                &mut error,
            )
        };
        ensure_ok(status, error)?;
        self.token_count = self
            .token_count
            .checked_add(u64::try_from(token_ids.len()).context("token count exceeds u64")?)
            .context("session token count overflow")?;
        predicted.truncate(output_count);
        Ok(predicted)
    }

    pub fn verify_tokens_sampled(
        &mut self,
        token_ids: &[i32],
        sampling: Option<&SamplingConfig>,
    ) -> Result<Vec<i32>> {
        self.verify_tokens_sampled_without_mtp(token_ids, sampling)
    }

    /// Runs batched verification and trims the speculative suffix.
    pub fn verify_tokens_rewound(&mut self, token_ids: &[i32]) -> Result<Vec<i32>> {
        if token_ids.is_empty() {
            return Ok(Vec::new());
        }
        let token_count = self.token_count;
        match self.verify_tokens(token_ids) {
            Ok(predicted) => {
                self.trim_session(token_count)?;
                Ok(predicted)
            }
            Err(error) => {
                let _ = self.trim_session(token_count);
                Err(error)
            }
        }
    }
}

impl Drop for StageSession {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                let _ = skippy_ffi::skippy_session_free(self.raw, ptr::null_mut());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        EMPTY_CHAT_GRAMMAR_METADATA, chat_sampling_metadata_for_native, native_position_to_u64,
    };

    #[test]
    fn native_position_conversion_accepts_non_negative_positions() {
        assert_eq!(native_position_to_u64(0).unwrap(), 0);
        assert_eq!(native_position_to_u64(i32::MAX).unwrap(), i32::MAX as u64);
    }

    #[test]
    fn native_position_conversion_rejects_negative_positions() {
        assert!(native_position_to_u64(-1).is_err());
        assert!(native_position_to_u64(i32::MIN).is_err());
    }

    #[test]
    fn empty_chat_grammar_uses_minimal_native_metadata() {
        let metadata = r#"{"chat_parser":"large","grammar":"","grammar_triggers":[]}"#;
        assert_eq!(
            chat_sampling_metadata_for_native(metadata),
            EMPTY_CHAT_GRAMMAR_METADATA
        );
        let grammar = r#"{"grammar":"root ::= \"ok\""}"#;
        assert_eq!(chat_sampling_metadata_for_native(grammar), grammar);
    }
}
