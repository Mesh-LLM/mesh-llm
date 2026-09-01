use crate::frontend::generation::StageOpenAiBackend;
use openai_frontend::{OpenAiError, OpenAiResult};
use skippy_runtime::{IterationBatchPhase, SamplingConfig};

const MAX_NATIVE_ITERATION_TOKENS: usize = 2_048;

fn suffix_chunk_sample_flags(
    suffix_len: usize,
    chunk_tokens: usize,
    sample_last: bool,
) -> Vec<bool> {
    let chunk_count = suffix_len.div_ceil(chunk_tokens);
    (0..chunk_count)
        .map(|index| sample_last && index == chunk_count.saturating_sub(1))
        .collect()
}

pub(in crate::frontend) struct SuffixPrefillOutcome {
    pub(in crate::frontend) predicted: Option<i32>,
    pub(in crate::frontend) chunk_count: usize,
    pub(in crate::frontend) max_batch_size: usize,
    pub(in crate::frontend) runtime_lock_wait_ms: f64,
    pub(in crate::frontend) runtime_lock_hold_ms: f64,
}

impl StageOpenAiBackend {
    /// Submit deferred suffix chunks as first-class native iterations.
    /// Concurrent requests can therefore share a mixed batch instead of each
    /// cache-aware restore closure monopolizing the runtime through its whole
    /// suffix. Sampling is deliberately restricted to the final chunk.
    pub(super) fn prefill_suffix(
        &self,
        session_id: &str,
        suffix: &[i32],
        sampling: Option<&SamplingConfig>,
        sample_last: bool,
    ) -> OpenAiResult<SuffixPrefillOutcome> {
        if suffix.is_empty() {
            return Err(OpenAiError::backend(
                "deferred suffix prefill requires at least one token",
            ));
        }
        let chunk_tokens = usize::try_from(self.config.n_ubatch.unwrap_or(256))
            .unwrap_or(MAX_NATIVE_ITERATION_TOKENS)
            .clamp(1, MAX_NATIVE_ITERATION_TOKENS);
        let sample_flags = suffix_chunk_sample_flags(suffix.len(), chunk_tokens, sample_last);
        let channel = self.iteration_scheduler.direct_iteration_channel();
        let mut predicted = None;
        let mut chunk_count = 0usize;
        let mut max_batch_size = 0usize;
        let mut runtime_lock_wait_ms = 0.0;
        let mut runtime_lock_hold_ms = 0.0;
        for (chunk, should_sample) in suffix.chunks(chunk_tokens).zip(sample_flags) {
            let outcome = self.iteration_scheduler.execute_iteration_on(
                &channel,
                session_id,
                chunk,
                &[],
                sampling,
                should_sample,
                IterationBatchPhase::Prefill,
            )?;
            chunk_count = chunk_count.saturating_add(1);
            max_batch_size = max_batch_size.max(outcome.batch_size);
            runtime_lock_wait_ms += outcome.runtime_lock_wait_ms;
            runtime_lock_hold_ms += outcome.runtime_lock_hold_ms;
            if should_sample {
                predicted = Some(outcome.predicted);
            }
        }
        Ok(SuffixPrefillOutcome {
            predicted,
            chunk_count,
            max_batch_size,
            runtime_lock_wait_ms,
            runtime_lock_hold_ms,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::suffix_chunk_sample_flags;

    #[test]
    fn deferred_suffix_samples_only_the_final_chunk() {
        assert_eq!(
            suffix_chunk_sample_flags(7, 3, true),
            vec![false, false, true]
        );
        assert_eq!(
            suffix_chunk_sample_flags(7, 3, false),
            vec![false, false, false]
        );
    }

    #[test]
    fn resident_suffix_keeps_final_chunk_sampling() {
        assert_eq!(suffix_chunk_sample_flags(3, 3, true), vec![true]);
        assert_eq!(suffix_chunk_sample_flags(4, 3, true), vec![false, true]);
    }
}
