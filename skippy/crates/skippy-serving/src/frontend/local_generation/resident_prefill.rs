use crate::frontend::generation::StageOpenAiBackend;
use crate::frontend::iteration_scheduler::PrefillTurn;
use skippy_inference_api::{InferenceError, InferenceResult};
use skippy_runtime::{IterationBatchPhase, SamplingConfig};
use std::time::Instant;

const MAX_NATIVE_ITERATION_TOKENS: usize = 2_048;

fn suffix_chunk_tokens(
    lane_count: u32,
    exports_activations: bool,
    resident_recurrent: bool,
    n_batch: Option<u32>,
    n_ubatch: Option<u32>,
) -> usize {
    let batch = usize::try_from(n_batch.unwrap_or(2048))
        .unwrap_or(MAX_NATIVE_ITERATION_TOKENS)
        .clamp(1, MAX_NATIVE_ITERATION_TOKENS);
    if lane_count == 1 && !exports_activations && !resident_recurrent {
        // Native decode performs its own microbatch splitting. With no other
        // lane to schedule and no activation export to materialize, submitting
        // one microbatch at a time adds an unnecessary synchronization boundary.
        return batch;
    }
    // A resident recurrent checkpoint has to observe the same state whether
    // the configured batch contains one or several native microbatches.
    // Submitting the boundaries explicitly keeps hybrid prefill and sampling
    // independent of the native backend's internal batch split.
    usize::try_from(n_ubatch.unwrap_or(256))
        .unwrap_or(batch)
        .clamp(1, batch)
}

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
        let chunk_tokens = suffix_chunk_tokens(
            self.config.lane_count,
            self.config.emits_activation_frame(),
            self.kv
                .as_ref()
                .is_some_and(|kv| kv.resident_recurrent_snapshots),
            self.config.n_batch,
            self.config.n_ubatch,
        );
        let sample_flags =
            suffix_chunk_sample_flags(request.suffix.len(), chunk_tokens, request.sample_last);
        let channel = self.iteration_scheduler.direct_iteration_channel();
        let mut predicted = None;
        let mut chunk_count = 0usize;
        let mut max_batch_size = 0usize;
        let mut runtime_lock_wait_ms = 0.0;
        let mut runtime_lock_hold_ms = 0.0;
        let mut admission_wait_ms = 0.0;
        let mut admission_yields = 0usize;
        for (chunk, should_sample) in request.suffix.chunks(chunk_tokens).zip(sample_flags) {
            ensure_suffix_prefill_active(request.deadline, request.cancellation)?;
            if let Some(waited) =
                prefill_turn.yield_to_waiters(request.deadline, request.cancellation)?
            {
                admission_wait_ms += waited.as_secs_f64() * 1000.0;
                admission_yields = admission_yields.saturating_add(1);
            }
            let outcome = self.iteration_scheduler.execute_iteration_on(
                &channel,
                request.session_id,
                chunk,
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
    use super::{suffix_chunk_sample_flags, suffix_chunk_tokens};

    #[test]
    fn single_lane_prefill_batches_preserve_microbatch_export_and_fairness_limits() {
        assert_eq!(
            suffix_chunk_tokens(1, false, false, Some(2048), Some(512)),
            2048
        );
        assert_eq!(
            suffix_chunk_tokens(1, true, false, Some(2048), Some(512)),
            512
        );
        assert_eq!(
            suffix_chunk_tokens(2, false, false, Some(2048), Some(512)),
            512
        );
        assert_eq!(
            suffix_chunk_tokens(1, false, false, Some(4096), Some(512)),
            2048
        );
        assert_eq!(
            suffix_chunk_tokens(1, false, true, Some(2048), Some(512)),
            512
        );
        assert_eq!(
            suffix_chunk_tokens(2, false, true, Some(2048), Some(512)),
            512
        );
        assert_eq!(
            suffix_chunk_tokens(2, false, false, Some(128), Some(512)),
            128
        );
        assert_eq!(suffix_chunk_tokens(1, false, false, Some(0), Some(0)), 1);
    }

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
