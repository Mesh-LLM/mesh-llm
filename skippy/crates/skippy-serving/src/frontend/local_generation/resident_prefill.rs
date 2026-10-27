use crate::frontend::generation::StageOpenAiBackend;
use crate::frontend::iteration_scheduler::PrefillTurn;
use skippy_inference_api::{InferenceError, InferenceResult};
use skippy_runtime::{IterationBatchPhase, SamplingConfig};
use std::time::Instant;

const MAX_NATIVE_ITERATION_TOKENS: usize = 2_048;

fn suffix_chunk_tokens(
    lane_count: u32,
    exports_activations: bool,
    n_batch: Option<u32>,
    n_ubatch: Option<u32>,
    admitted_prefills: usize,
) -> usize {
    let batch = usize::try_from(n_batch.unwrap_or(2048))
        .unwrap_or(MAX_NATIVE_ITERATION_TOKENS)
        .clamp(1, MAX_NATIVE_ITERATION_TOKENS);
    if lane_count == 1 && !exports_activations {
        // Native decode performs its own microbatch splitting. With no other
        // lane to schedule and no activation export to materialize, submitting
        // one microbatch at a time adds an unnecessary synchronization boundary.
        return batch;
    }
    let per_prefill_budget = if exports_activations {
        batch
    } else {
        batch / admitted_prefills.max(1)
    };
    usize::try_from(n_ubatch.unwrap_or(256))
        .unwrap_or(batch)
        .clamp(1, batch)
        .min(per_prefill_budget.max(1))
}

pub(in crate::frontend) struct SuffixPrefillOutcome {
    pub(in crate::frontend) predicted: Option<i32>,
    pub(in crate::frontend) chunk_count: usize,
    pub(in crate::frontend) max_batch_size: usize,
    pub(in crate::frontend) runtime_lock_wait_ms: f64,
    pub(in crate::frontend) runtime_lock_hold_ms: f64,
    pub(in crate::frontend) admission_wait_ms: f64,
    pub(in crate::frontend) admission_yields: usize,
}

pub(super) struct SuffixPrefillRequest<'a> {
    pub(super) session_id: &'a str,
    pub(super) suffix: &'a [i32],
    pub(super) sampling: Option<&'a SamplingConfig>,
    pub(super) sample_last: bool,
    pub(super) deadline: Instant,
    pub(super) cancellation: Option<&'a skippy_inference_api::CancellationToken>,
}

impl StageOpenAiBackend {
    /// Submit deferred suffix chunks as first-class native iterations.
    /// Waiting prompts rotate through the shared allocator between chunks;
    /// sampling is restricted to the final chunk.
    pub(super) fn prefill_suffix(
        &self,
        request: SuffixPrefillRequest<'_>,
        prefill_turn: &mut PrefillTurn<'_>,
    ) -> InferenceResult<SuffixPrefillOutcome> {
        if request.suffix.is_empty() {
            return Err(InferenceError::backend(
                "deferred suffix prefill requires at least one token",
            ));
        }
        let channel = self.iteration_scheduler.direct_iteration_channel();
        let mut predicted = None;
        let mut chunk_count = 0usize;
        let mut max_batch_size = 0usize;
        let mut runtime_lock_wait_ms = 0.0;
        let mut runtime_lock_hold_ms = 0.0;
        let mut admission_wait_ms = 0.0;
        let mut admission_yields = 0usize;
        let mut offset = 0;
        while offset < request.suffix.len() {
            ensure_suffix_prefill_active(request.deadline, request.cancellation)?;
            if let Some(waited) =
                prefill_turn.yield_to_waiters(request.deadline, request.cancellation)?
            {
                admission_wait_ms += waited.as_secs_f64() * 1000.0;
                admission_yields = admission_yields.saturating_add(1);
            }
            let chunk_tokens = suffix_chunk_tokens(
                self.config.lane_count,
                self.config.emits_activation_frame(),
                self.config.n_batch,
                self.config.n_ubatch,
                prefill_turn.admitted_count(),
            );
            let end = offset
                .saturating_add(chunk_tokens)
                .min(request.suffix.len());
            let should_sample = request.sample_last && end == request.suffix.len();
            let outcome = self.iteration_scheduler.execute_iteration_on(
                &channel,
                request.session_id,
                &request.suffix[offset..end],
                &[],
                request.sampling,
                should_sample,
                IterationBatchPhase::Prefill,
                Some(request.deadline),
                request.cancellation,
            )?;
            chunk_count = chunk_count.saturating_add(1);
            max_batch_size = max_batch_size.max(outcome.batch_size);
            runtime_lock_wait_ms += outcome.runtime_lock_wait_ms;
            runtime_lock_hold_ms += outcome.runtime_lock_hold_ms;
            if should_sample {
                predicted = Some(outcome.predicted);
            }
            offset = end;
        }
        Ok(SuffixPrefillOutcome {
            predicted,
            chunk_count,
            max_batch_size,
            runtime_lock_wait_ms,
            runtime_lock_hold_ms,
            admission_wait_ms,
            admission_yields,
        })
    }
}

fn ensure_suffix_prefill_active(
    deadline: Instant,
    cancellation: Option<&skippy_inference_api::CancellationToken>,
) -> InferenceResult<()> {
    if cancellation.is_some_and(skippy_inference_api::CancellationToken::is_cancelled) {
        return Err(InferenceError::cancelled(
            "request cancelled during deferred suffix prefill",
        ));
    }
    if Instant::now() >= deadline {
        return Err(InferenceError::timeout(
            "cache operation deadline exceeded during deferred suffix prefill",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::suffix_chunk_tokens;

    #[test]
    fn single_lane_prefill_batches_preserve_microbatch_export_and_fairness_limits() {
        assert_eq!(
            suffix_chunk_tokens(1, false, Some(2048), Some(512), 1),
            2048
        );
        assert_eq!(suffix_chunk_tokens(1, true, Some(2048), Some(512), 1), 512);
        assert_eq!(suffix_chunk_tokens(2, false, Some(2048), Some(512), 1), 512);
        assert_eq!(
            suffix_chunk_tokens(1, false, Some(4096), Some(512), 1),
            2048
        );
        assert_eq!(suffix_chunk_tokens(2, false, Some(128), Some(512), 1), 128);
        assert_eq!(suffix_chunk_tokens(1, false, Some(0), Some(0), 1), 1);
    }

    #[test]
    fn concurrent_prefills_share_one_large_native_microbatch() {
        assert_eq!(
            suffix_chunk_tokens(4, false, Some(2048), Some(2048), 1),
            2048
        );
        assert_eq!(
            suffix_chunk_tokens(4, false, Some(2048), Some(2048), 2),
            1024
        );
        assert_eq!(suffix_chunk_tokens(4, false, Some(2048), Some(512), 2), 512);
        assert_eq!(
            suffix_chunk_tokens(4, true, Some(2048), Some(2048), 2),
            2048
        );
    }
}
