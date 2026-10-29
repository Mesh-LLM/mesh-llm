use super::*;

impl StageOpenAiBackend {
    /// Drives feature-specific proposal/checkpoint state while the iteration
    /// scheduler worker remains the sole owner of target runtime execution.
    /// This is not a fallback serving scheduler: native MTP, linear proposals,
    /// hooks, and cache operations submit their target work to that worker.
    pub(super) fn run_scheduler_feature_loop(
        &self,
        request: &mut LocalGeneration<'_>,
        session_id: &str,
        prompt_prefill_sample: Option<i32>,
        cache_stats: &mut GenerationCacheStats,
        receipt_cancelled: &mut bool,
        emit_token: &mut impl FnMut(i32) -> InferenceResult<TokenControl>,
    ) -> InferenceResult<Duration> {
        let decode_timer = PhaseTimer::start();
        let mut state =
            self.prepare_decode_state(request, session_id, prompt_prefill_sample, emit_token)?;
        let recurrent_checkpoint_required = self
            .kv
            .as_ref()
            .is_some_and(|kv| kv.records_lane_owned_recurrent());
        let recurrent_state_cache = self
            .kv
            .as_ref()
            .is_some_and(|kv| kv.records_lane_owned_recurrent() || kv.resident_recurrent_snapshots);
        // The direct iteration channel still batches decode across lanes.
        // Handing a recurrent-state continuation to the planned scheduler
        // changes c=4 output after the first token on CUDA.
        let resume_scheduler_decode = self.linear_proposal_ingress.is_none()
            && self.draft.is_none()
            && !request.native_mtp_enabled
            && !state.generation_hooks_active
            && !recurrent_state_cache;
        if scheduler_resume_ready(
            resume_scheduler_decode,
            !recurrent_checkpoint_required,
            state.stopped,
            state.decoded_tokens,
            request.max_tokens,
        ) {
            return self.resume_prefilled_decode(
                request,
                session_id,
                &mut state,
                cache_stats,
                decode_timer,
                emit_token,
            );
        }
        // A recurrent request needs one serial decode before proposals can
        // advance the native session beyond the prompt boundary. The serial
        // attempt is the gate: a short prompt may be below the policy's
        // minimum checkpoint size, in which case recording is correctly
        // skipped but proposals must not remain disabled for the rest of the
        // request.
        let mut first_post_decode_checkpoint_attempted = initial_recurrent_checkpoint_attempted(
            recurrent_checkpoint_required,
            prompt_prefill_sample,
        );
        while !state.stopped && state.decoded_tokens < request.max_tokens as usize {
            if request
                .cancellation
                .is_some_and(skippy_inference_api::CancellationToken::is_cancelled)
            {
                *receipt_cancelled = true;
                break;
            }
            let linear_progress = if linear_proposal_allowed(
                recurrent_checkpoint_required,
                first_post_decode_checkpoint_attempted,
            ) {
                self.try_execute_linear_proposal(request, session_id, &mut state, emit_token)?
            } else {
                // A proposal can consume the final prompt token and commit
                // multiple target tokens in one call. Give recurrent caches
                // one serial decode first so the exact full-prompt boundary
                // is published before that proposal advances the session.
                LinearProposalProgress::NotUsed
            };
            match linear_progress {
                LinearProposalProgress::Continue => continue,
                LinearProposalProgress::Stop => break,
                LinearProposalProgress::NotUsed => {}
            }
            // A batched MTP span commits multiple tokens from one forward. It
            // shares the recurrent-checkpoint gate with linear proposals for
            // the same reason: it can advance the session past the prompt
            // boundary before that boundary has been published.
            if linear_proposal_allowed(
                recurrent_checkpoint_required,
                first_post_decode_checkpoint_attempted,
            ) {
                match self
                    .try_execute_native_mtp_span(request, session_id, &mut state, emit_token)?
                {
                    NativeMtpSpanProgress::Continue => continue,
                    NativeMtpSpanProgress::Stop => break,
                    NativeMtpSpanProgress::NotUsed => {}
                }
            }
            let control = self.decode_one_token(request, session_id, &mut state, emit_token)?;
            if !first_post_decode_checkpoint_attempted && state.generated_token_ids.len() == 1 {
                first_post_decode_checkpoint_attempted = true;
                self.record_post_decode_exact_state(request, session_id, &state);
            }
            if state.linear_proposal_max_tokens > 0 {
                state.pending_linear_proposal_tokens.push(state.current);
            }
            if control == TokenControl::Stop {
                break;
            }
            if scheduler_resume_ready(
                resume_scheduler_decode,
                first_post_decode_checkpoint_attempted,
                state.stopped,
                state.decoded_tokens,
                request.max_tokens,
            ) {
                return self.resume_prefilled_decode(
                    request,
                    session_id,
                    &mut state,
                    cache_stats,
                    decode_timer,
                    emit_token,
                );
            }
        }
        let model_generation_elapsed =
            self.emit_decode_summary(request, &mut state, cache_stats, decode_timer)?;
        // The first-token checkpoint names the prompt boundary. Avoid
        // exporting the same recurrent state twice for a one-token response;
        // longer responses still get their final exact boundary recorded.
        if state.generated_token_ids.len() > 1 {
            let _ = self.record_post_decode_exact_state(request, session_id, &state);
        }
        Ok(model_generation_elapsed)
    }
    fn resume_prefilled_decode(
        &self,
        request: &LocalGeneration<'_>,
        session_id: &str,
        state: &mut DecodeState,
        cache_stats: &mut GenerationCacheStats,
        decode_timer: PhaseTimer,
        emit_token: &mut impl FnMut(i32) -> InferenceResult<TokenControl>,
    ) -> InferenceResult<Duration> {
        let initial_generated_tokens = state.generated_token_ids.clone();
        let stats = self.iteration_scheduler.resume_generation(
            ScheduledResumeRequest {
                id: session_id,
                prompt_tokens: request.prompt_token_ids,
                generated_tokens: &initial_generated_tokens,
                max_tokens: request.max_tokens,
                sampling: request.sampling.enabled.then_some(request.sampling),
                cancellation: request.cancellation,
            },
            |token| {
                state.current = token;
                state.decoded_tokens = state.decoded_tokens.saturating_add(1);
                state.generated_token_ids.push(token);
                emit_token(token)
            },
        )?;
        cache_stats.predicted_ms = stats.predicted_ms;
        let model_generation_elapsed =
            self.emit_decode_summary(request, state, cache_stats, decode_timer)?;
        if state.generated_token_ids.len() > 1 {
            let _ = self.record_post_decode_exact_state(request, session_id, state);
        }
        Ok(model_generation_elapsed)
    }
}

/// A cached dense prefill leaves its last prompt token for serial decode.
/// Once that produces a token, it can join the same decode batches as a
/// request sampled during prefill. Never resume an empty generated history.
fn scheduler_resume_ready(
    eligible: bool,
    checkpoint_attempted: bool,
    stopped: bool,
    decoded_tokens: usize,
    max_tokens: u32,
) -> bool {
    eligible
        && checkpoint_attempted
        && !stopped
        && decoded_tokens > 0
        && decoded_tokens < max_tokens as usize
}

#[cfg(test)]
mod tests {
    use super::scheduler_resume_ready;

    #[test]
    fn cached_dense_decode_hands_off_after_its_first_token() {
        assert!(!scheduler_resume_ready(true, true, false, 0, 8));
        assert!(scheduler_resume_ready(true, true, false, 1, 8));
    }

    #[test]
    fn recurrent_decode_waits_for_checkpoint_attempt() {
        assert!(!scheduler_resume_ready(true, false, false, 1, 8));
        assert!(scheduler_resume_ready(true, true, false, 1, 8));
    }

    #[test]
    fn handoff_preserves_feature_and_completion_gates() {
        assert!(!scheduler_resume_ready(false, true, false, 1, 8));
        assert!(!scheduler_resume_ready(true, true, true, 1, 8));
        assert!(!scheduler_resume_ready(true, true, false, 1, 1));
        assert!(!scheduler_resume_ready(true, true, false, 0, 0));
    }
}
