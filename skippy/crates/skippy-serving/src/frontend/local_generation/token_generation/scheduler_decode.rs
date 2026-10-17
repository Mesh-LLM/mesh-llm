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
            .is_some_and(|kv| kv.payload == StagePrefixCachePayload::KvRecurrent);
        let resume_scheduler_decode = prompt_prefill_sample.is_some()
            && !state.stopped
            && state.decoded_tokens < request.max_tokens as usize
            && self.linear_proposal_ingress.is_none()
            && self.draft.is_none()
            && !request.native_mtp_enabled
            && !state.generation_hooks_active
            // Recurrent handoff is qualified for single-slot execution. Use
            // the hard limit because adaptive concurrency can grow mid-request.
            && (!recurrent_checkpoint_required || self.generation_limit.hard_limit() == 1);
        if resume_scheduler_decode && !recurrent_checkpoint_required {
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
            if resume_scheduler_decode
                && first_post_decode_checkpoint_attempted
                && state.decoded_tokens < request.max_tokens as usize
            {
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
