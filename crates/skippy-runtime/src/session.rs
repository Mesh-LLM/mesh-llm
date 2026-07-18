use std::ffi::CString;
use std::ptr;

use anyhow::{Context, Result, anyhow};
use skippy_ffi::{
    GenerationSignalWindow as RawGenerationSignalWindow, NativeMtpDraft as RawNativeMtpDraft,
    SamplingConfig as RawSamplingConfig, Session as RawSession, TokenSignal as RawTokenSignal,
};

use crate::error::ensure_ok;
use crate::{GenerationSignalWindow, NativeMtpDraft, SamplingConfig, TokenSignal};

pub struct StageSession {
    pub(crate) raw: *mut RawSession,
    pub(crate) token_count: u64,
}

pub struct DecodeBatchRequest<'a> {
    pub session: &'a mut StageSession,
    pub token_id: i32,
    pub sampling: Option<&'a SamplingConfig>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageSessionCheckpoint {
    token_count: u64,
}

// The experimental C ABI owns synchronization internally for model/session use.
// Rust stage-server access is additionally serialized behind a Mutex.
unsafe impl Send for StageSession {}

impl StageSession {
    pub fn token_count(&self) -> u64 {
        self.token_count
    }

    pub fn batch_size(&self) -> Result<usize> {
        let n_batch = unsafe { skippy_ffi::skippy_session_batch_size(self.raw) };
        if n_batch <= 0 {
            return Err(anyhow!("skippy session has no valid batch size"));
        }
        usize::try_from(n_batch).context("session batch size exceeds usize")
    }

    /// Captures the current position and asks the native runtime to keep an
    /// in-session recurrent checkpoint. Attention KV is restored by trimming
    /// the speculative suffix back to this position.
    pub fn checkpoint(&mut self) -> Result<StageSessionCheckpoint> {
        let mut token_count = 0u64;
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_checkpoint_session(self.raw, &mut token_count, &mut error)
        };
        ensure_ok(status, error)?;
        self.token_count = token_count;
        Ok(StageSessionCheckpoint { token_count })
    }

    pub fn restore_checkpoint(&mut self, checkpoint: &StageSessionCheckpoint) -> Result<()> {
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_restore_session_checkpoint(
                self.raw,
                checkpoint.token_count,
                &mut error,
            )
        };
        ensure_ok(status, error)?;
        self.token_count = checkpoint.token_count;
        Ok(())
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
        let metadata_json = CString::new(metadata_json)
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

    /// Runs batched verification and restores the prior checkpoint.
    pub fn verify_tokens_rewound(&mut self, token_ids: &[i32]) -> Result<Vec<i32>> {
        if token_ids.is_empty() {
            return Ok(Vec::new());
        }
        let checkpoint = self.checkpoint()?;
        match self.verify_tokens(token_ids) {
            Ok(predicted) => {
                self.restore_checkpoint(&checkpoint)?;
                Ok(predicted)
            }
            Err(error) => {
                let _ = self.restore_checkpoint(&checkpoint);
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
