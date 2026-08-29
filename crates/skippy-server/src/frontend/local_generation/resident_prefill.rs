use crate::frontend::generation::StageOpenAiBackend;
use openai_frontend::{OpenAiError, OpenAiResult};
use skippy_runtime::{IterationBatchPhase, SamplingConfig};

const MAX_NATIVE_ITERATION_TOKENS: usize = 2_048;

pub(in crate::frontend) struct ResidentSuffixPrefillOutcome {
    pub(in crate::frontend) predicted: i32,
    pub(in crate::frontend) chunk_count: usize,
    pub(in crate::frontend) max_batch_size: usize,
    pub(in crate::frontend) runtime_lock_wait_ms: f64,
    pub(in crate::frontend) runtime_lock_hold_ms: f64,
}

impl StageOpenAiBackend {
    /// Submit resident-KV suffix chunks as first-class native iterations.
    /// Concurrent requests can therefore share a mixed batch instead of each
    /// restore closure monopolizing the runtime through its whole suffix.
    pub(super) fn prefill_resident_suffix(
        &self,
        session_id: &str,
        suffix: &[i32],
        sampling: Option<&SamplingConfig>,
    ) -> OpenAiResult<ResidentSuffixPrefillOutcome> {
        if suffix.is_empty() {
            return Err(OpenAiError::backend(
                "resident suffix prefill requires at least one token",
            ));
        }
        let chunk_tokens = usize::try_from(self.config.n_ubatch.unwrap_or(256))
            .unwrap_or(MAX_NATIVE_ITERATION_TOKENS)
            .clamp(1, MAX_NATIVE_ITERATION_TOKENS);
        let channel = self.iteration_scheduler.direct_iteration_channel();
        let mut predicted = None;
        let mut chunk_count = 0usize;
        let mut max_batch_size = 0usize;
        let mut runtime_lock_wait_ms = 0.0;
        let mut runtime_lock_hold_ms = 0.0;
        for (index, chunk) in suffix.chunks(chunk_tokens).enumerate() {
            let sample_last = index == suffix.len().div_ceil(chunk_tokens).saturating_sub(1);
            let outcome = self.iteration_scheduler.execute_iteration_on(
                &channel,
                session_id,
                chunk,
                &[],
                sampling,
                sample_last,
                IterationBatchPhase::Prefill,
            )?;
            chunk_count = chunk_count.saturating_add(1);
            max_batch_size = max_batch_size.max(outcome.batch_size);
            runtime_lock_wait_ms += outcome.runtime_lock_wait_ms;
            runtime_lock_hold_ms += outcome.runtime_lock_hold_ms;
            if sample_last {
                predicted = Some(outcome.predicted);
            }
        }
        Ok(ResidentSuffixPrefillOutcome {
            predicted: predicted
                .ok_or_else(|| OpenAiError::backend("resident suffix prefill was not sampled"))?,
            chunk_count,
            max_batch_size,
            runtime_lock_wait_ms,
            runtime_lock_hold_ms,
        })
    }
}
