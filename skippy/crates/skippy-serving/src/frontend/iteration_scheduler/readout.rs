//! Unsampled causal iterations share the normal direct-iteration queue.

use super::*;
use skippy_inference_api::CancellationToken;

/// One prompt chunk or teacher-forced candidate continuation. Only its final
/// row is read; an empty selection computes KV/frontier state without logits.
pub(crate) struct ReadoutIteration<'a> {
    pub session_id: &'a str,
    pub position: u64,
    pub tokens: &'a [i32],
    pub selected_tokens: &'a [i32],
    pub input: Option<ActivationFrame>,
    pub cancellation: Option<CancellationToken>,
}

impl IterationScheduler {
    /// Return the prefill chunk bound used by readout scheduling.
    pub(crate) fn readout_chunk_limit(&self) -> usize {
        self.shared.max_direct_iteration_tokens.max(1)
    }
    /// Queue a complete independent wave before waiting, so questions from one
    /// HTTP request can coalesce exactly like concurrent generation requests.
    pub(crate) fn execute_readout_wave(
        &self,
        wave: Vec<ReadoutIteration<'_>>,
    ) -> InferenceResult<Vec<SchedulerIterationOutcome>> {
        for request in &wave {
            validate_direct_iteration(
                request.tokens,
                &[],
                self.shared.max_direct_iteration_tokens,
            )?;
            if request.selected_tokens.len() > 256
                || request.selected_tokens.iter().any(|token| *token < 0)
            {
                return Err(InferenceError::invalid_request(
                    "invalid selected-token readout",
                ));
            }
        }
        let mut receivers = Vec::with_capacity(wave.len());
        let mut enqueue_error = None;
        for request in wave {
            let (sender, receiver) = std_mpsc::sync_channel(1);
            let result = self.enqueue_command(SchedulerCommand::ExecuteIteration(Box::new(
                DirectIteration {
                    session_id: request.session_id.to_owned(),
                    target_token_count: Some(request.position),
                    token_ids: request.tokens.to_vec(),
                    positions: Vec::new(),
                    sampling: None,
                    input: request.input,
                    sample_last: false,
                    selected_tokens: Some(request.selected_tokens.to_vec()),
                    phase: IterationBatchPhase::Prefill,
                    deadline: None,
                    cancellation: request.cancellation,
                    enqueued_at: Instant::now(),
                    reply: sender.into(),
                },
            )));
            if let Err(error) = result {
                enqueue_error = Some(error);
                break;
            }
            receivers.push(receiver);
        }
        // Drain every admitted operation on any failure before the caller
        // releases the sessions. Otherwise cleanup could race a queued read.
        let mut outputs = Vec::with_capacity(receivers.len());
        for receiver in receivers {
            match receiver
                .recv()
                .map_err(|error| {
                    InferenceError::backend(format!("readout scheduler stopped: {error}"))
                })
                .and_then(std::convert::identity)
            {
                Ok(output) => outputs.push(output),
                Err(error) => {
                    enqueue_error.get_or_insert(error);
                }
            }
        }
        if let Some(error) = enqueue_error {
            return Err(error);
        }
        Ok(outputs)
    }
}
