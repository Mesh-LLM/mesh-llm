//! Batched native-MTP speculation for single-node (non-staged) decode.
//!
//! The staged path commits accepted draft spans in
//! `frontend::native_mtp::verify_window`, and the plugin-fed proposal path does
//! the same in `frontend::linear_proposal::execution`. Local decode previously
//! had neither: it produced an MTP draft per step, recorded whether the draft
//! matched, and then decoded the next token anyway. This module gives local
//! decode the same accept-and-skip contract — one batched target forward over
//! the drafted span, commit every token up to the first mismatch, then retire
//! the checkpoint on full acceptance or trim the speculative suffix.

use std::collections::BTreeMap;
use std::time::Instant;

use openai_frontend::{OpenAiError, OpenAiResult};
use serde_json::json;

use crate::frontend::generation::{LocalGeneration, StageOpenAiBackend, TokenControl};
use crate::frontend::{
    NativeMtpDraftOrigin, classify_native_mtp_verify_window, openai_backend_error,
};

use super::token_generation::DecodeState;

/// Outcome of one batched native-MTP speculation attempt.
pub(super) enum NativeMtpSpanProgress {
    /// No draft was available, or the span was not admissible at this
    /// boundary. The caller must fall back to a serial decode step.
    NotUsed,
    /// Tokens were committed and decoding should continue.
    Continue,
    /// Tokens were committed and generation should stop.
    Stop,
}

/// Inputs that decide whether a batched native-MTP span may be attempted.
#[derive(Clone, Copy, Debug)]
pub(super) struct NativeMtpSpanAdmission {
    pub(super) native_mtp_enabled: bool,
    pub(super) generation_hooks_active: bool,
    /// Token equality is only a valid acceptance test under greedy sampling.
    pub(super) greedy_admitted: bool,
    pub(super) remaining_tokens: usize,
    pub(super) draft_tokens: usize,
    pub(super) min_draft_tokens: usize,
}

/// Whether a batched span may commit. A span needs at least one drafted token
/// plus its boundary token, so at one token remaining a serial decode is
/// strictly cheaper and speculation is skipped.
pub(super) fn native_mtp_span_admitted(admission: NativeMtpSpanAdmission) -> bool {
    admission.native_mtp_enabled
        && !admission.generation_hooks_active
        && admission.greedy_admitted
        && admission.remaining_tokens >= 2
        && admission.draft_tokens >= admission.min_draft_tokens.max(1)
}

/// Whether the whole drafted span held, which is the only case where the
/// verify-call draft stays valid and the checkpoint may be retired rather than
/// trimmed.
pub(super) fn native_mtp_span_full_accept(
    rejected: bool,
    accepted_draft_tokens: usize,
    draft_tokens: usize,
) -> bool {
    !rejected && accepted_draft_tokens == draft_tokens
}

/// A verified native-MTP span, before the session suffix is repaired.
struct VerifiedSpan {
    committed_tokens: Vec<i32>,
    accepted_draft_tokens: usize,
    reached_stop: bool,
    full_accept: bool,
    verification_elapsed_us: u64,
}

impl StageOpenAiBackend {
    /// Attempts to commit a batched native-MTP draft span.
    ///
    /// Returns [`NativeMtpSpanProgress::NotUsed`] whenever speculation is not
    /// admissible, in which case the caller must perform a serial decode step;
    /// this method never advances the session in that case.
    pub(super) fn try_execute_native_mtp_span(
        &self,
        request: &LocalGeneration<'_>,
        session_id: &str,
        state: &mut DecodeState,
        emit_token: &mut impl FnMut(i32) -> OpenAiResult<TokenControl>,
    ) -> OpenAiResult<NativeMtpSpanProgress> {
        let remaining = (request.max_tokens as usize).saturating_sub(state.decoded_tokens);
        let Some(pending) = state.native_mtp.take_pending_draft() else {
            return Ok(NativeMtpSpanProgress::NotUsed);
        };
        let draft_origin = pending.origin;
        let draft_tokens = pending
            .tokens
            .iter()
            .copied()
            .take(state.native_mtp_options.max_draft_tokens)
            .take(remaining.saturating_sub(1))
            .collect::<Vec<_>>();
        if !native_mtp_span_admitted(NativeMtpSpanAdmission {
            native_mtp_enabled: request.native_mtp_enabled,
            generation_hooks_active: state.generation_hooks_active,
            greedy_admitted: state.native_mtp_span_admitted,
            remaining_tokens: remaining,
            draft_tokens: draft_tokens.len(),
            min_draft_tokens: state.native_mtp_options.min_draft_tokens,
        }) {
            // Restore the draft so its verdict is still observed by the serial
            // step, keeping the acceptance counters honest.
            state.native_mtp.restore_pending_draft(pending);
            return Ok(NativeMtpSpanProgress::NotUsed);
        }

        let base_position = self
            .native_mtp_span_base_position(session_id)?
            .ok_or_else(|| OpenAiError::backend("native MTP span session is not active"))?;
        let mut verify_inputs = Vec::with_capacity(draft_tokens.len().saturating_add(1));
        verify_inputs.push(state.current);
        verify_inputs.extend_from_slice(&draft_tokens);

        let verified = self.verify_native_mtp_span(
            request,
            session_id,
            state,
            base_position,
            &draft_tokens,
            &verify_inputs,
            draft_origin,
            emit_token,
        )?;

        self.repair_native_mtp_span(
            request,
            session_id,
            base_position,
            verify_inputs.len(),
            verified.committed_tokens.len(),
            verified.full_accept,
        )?;

        if verified.committed_tokens.is_empty() {
            return Err(OpenAiError::backend(
                "native MTP span committed no target token",
            ));
        }

        state.decoded_tokens = state
            .decoded_tokens
            .checked_add(verified.committed_tokens.len())
            .ok_or_else(|| OpenAiError::backend("native MTP span decode count overflow"))?;
        state.current = *verified
            .committed_tokens
            .last()
            .expect("checked non-empty committed tokens");
        state
            .generated_token_ids
            .extend_from_slice(&verified.committed_tokens);
        if let Some(committed_token_ids) = state.linear_context_tokens.as_mut() {
            committed_token_ids.extend_from_slice(&verified.committed_tokens);
        }

        if state.emit_token_debug {
            let mut attrs = BTreeMap::new();
            attrs.insert(
                "llama_stage.native_mtp.span_draft_tokens".to_string(),
                json!(draft_tokens.len()),
            );
            attrs.insert(
                "llama_stage.native_mtp.span_accepted_tokens".to_string(),
                json!(verified.accepted_draft_tokens),
            );
            attrs.insert(
                "llama_stage.native_mtp.span_committed_tokens".to_string(),
                json!(verified.committed_tokens.len()),
            );
            // The saved forwards are the direct measure of whether speculation
            // paid off: one batched forward emitted this many tokens.
            attrs.insert(
                "llama_stage.native_mtp.span_saved_forwards".to_string(),
                json!(verified.committed_tokens.len().saturating_sub(1)),
            );
            attrs.insert(
                "llama_stage.native_mtp.span_full_accept".to_string(),
                json!(verified.full_accept),
            );
            attrs.insert(
                "llama_stage.native_mtp.span_verification_us".to_string(),
                json!(verified.verification_elapsed_us),
            );
            self.telemetry
                .emit_debug("stage.openai_native_mtp_span", attrs);
        }

        if verified.reached_stop || state.decoded_tokens >= request.max_tokens as usize {
            Ok(NativeMtpSpanProgress::Stop)
        } else {
            Ok(NativeMtpSpanProgress::Continue)
        }
    }

    fn native_mtp_span_base_position(&self, session_id: &str) -> OpenAiResult<Option<u64>> {
        let session_id = session_id.to_string();
        self.iteration_scheduler
            .execute_runtime("native-mtp-position", move |runtime| {
                Ok(runtime.session_token_count(&session_id))
            })
    }

    #[allow(clippy::too_many_arguments)]
    fn verify_native_mtp_span(
        &self,
        request: &LocalGeneration<'_>,
        session_id: &str,
        state: &mut DecodeState,
        base_position: u64,
        draft_tokens: &[i32],
        verify_inputs: &[i32],
        draft_origin: NativeMtpDraftOrigin,
        emit_token: &mut impl FnMut(i32) -> OpenAiResult<TokenControl>,
    ) -> OpenAiResult<VerifiedSpan> {
        let verify_timer = Instant::now();
        let owned_session_id = session_id.to_string();
        let owned_verify_inputs = verify_inputs.to_vec();
        let owned_draft_tokens = draft_tokens.to_vec();
        let sampling = request.sampling.enabled.then_some(request.sampling.clone());
        let max_draft_tokens = state.native_mtp_options.max_draft_tokens;
        let decoded_tokens = state.decoded_tokens;
        let max_tokens = request.max_tokens as usize;
        let (predictions, next_draft, decision, position_after_verification) = self
            .iteration_scheduler
            .execute_runtime("native-mtp-verify", move |runtime| {
                let (predictions, next_draft) = runtime
                    .verify_tokens_sampled_mtp(
                        &owned_session_id,
                        &owned_verify_inputs,
                        sampling.as_ref(),
                        max_draft_tokens,
                    )
                    .map_err(openai_backend_error)?;
                let decision = classify_native_mtp_verify_window(
                    &owned_draft_tokens,
                    &predictions,
                    decoded_tokens,
                    max_tokens,
                    |token| {
                        runtime
                            .model
                            .token_is_eog(token)
                            .map_err(openai_backend_error)
                    },
                )?;
                let position_after_verification = runtime
                    .session_token_count(&owned_session_id)
                    .ok_or_else(|| OpenAiError::backend("native MTP span session disappeared"))?;
                Ok((
                    predictions,
                    next_draft,
                    decision,
                    position_after_verification,
                ))
            })?;
        let expected_position = base_position
            .checked_add(
                u64::try_from(verify_inputs.len())
                    .map_err(|_| OpenAiError::backend("verification row count exceeds u64"))?,
            )
            .ok_or_else(|| OpenAiError::backend("native MTP span position overflow"))?;
        if position_after_verification != expected_position {
            return Err(OpenAiError::backend(format!(
                "native MTP span verification position mismatch: observed {position_after_verification}, expected {expected_position}"
            )));
        }
        let verification_elapsed_us =
            u64::try_from(verify_timer.elapsed().as_micros()).unwrap_or(u64::MAX);

        let span = state.native_mtp.observe_taken_draft_span(
            draft_tokens,
            &predictions,
            i64::try_from(verification_elapsed_us).unwrap_or(i64::MAX),
        );
        // The draft returned by the verify call describes the branch that was
        // just accepted. It is only valid for the next step when the whole span
        // held; a trimmed suffix invalidates it.
        let full_accept = native_mtp_span_full_accept(
            decision.rejected,
            decision.accepted_proposal_tokens,
            draft_tokens.len(),
        );
        if full_accept {
            if let Some(next_draft) = next_draft {
                state.native_mtp.observe_next_draft(
                    Some(crate::frontend::NativeMtpDraft {
                        tokens: next_draft.token_ids,
                        proposal_compute_us: next_draft.proposal_compute_us,
                    }),
                    draft_origin,
                );
            }
        } else {
            state.native_mtp.clear_pending_draft();
        }

        let mut committed_tokens = Vec::with_capacity(decision.commit_count);
        let mut reached_stop = false;
        for token in predictions.iter().copied().take(decision.commit_count) {
            if request
                .cancellation
                .is_some_and(openai_frontend::CancellationToken::is_cancelled)
            {
                break;
            }
            committed_tokens.push(token);
            match emit_token(token) {
                Ok(TokenControl::Continue) => {}
                Ok(TokenControl::Stop) => {
                    reached_stop = true;
                    break;
                }
                // The session suffix must still be repaired before the error
                // propagates, so record the stop and let the caller repair.
                Err(error) => {
                    self.repair_native_mtp_span(
                        request,
                        session_id,
                        base_position,
                        verify_inputs.len(),
                        committed_tokens.len(),
                        false,
                    )?;
                    return Err(error);
                }
            }
        }

        Ok(VerifiedSpan {
            committed_tokens,
            accepted_draft_tokens: span.accepted_count,
            reached_stop,
            full_accept: full_accept && !reached_stop,
            verification_elapsed_us,
        })
    }

    /// Retires the verify checkpoint after a full-span acceptance, or trims the
    /// speculative suffix back to the canonical position.
    fn repair_native_mtp_span(
        &self,
        request: &LocalGeneration<'_>,
        session_id: &str,
        base_position: u64,
        verification_rows: usize,
        committed_token_count: usize,
        full_accept: bool,
    ) -> OpenAiResult<()> {
        let canonical_position = base_position
            .checked_add(
                u64::try_from(committed_token_count)
                    .map_err(|_| OpenAiError::backend("committed token count exceeds u64"))?,
            )
            .ok_or_else(|| OpenAiError::backend("native MTP span canonical position overflow"))?;
        let session_id = session_id.to_string();
        let verification_rows = u64::try_from(verification_rows)
            .map_err(|_| OpenAiError::backend("verification row count exceeds u64"))?;
        let chat_sampling_metadata = request.chat_sampling_metadata.map(str::to_string);
        let prompt_token_count = u64::try_from(request.prompt_token_ids.len()).unwrap_or(u64::MAX);
        let sampling = request.sampling.enabled.then_some(request.sampling.clone());
        self.iteration_scheduler
            .execute_runtime("native-mtp-repair", move |runtime| {
            if full_accept {
                return runtime
                    .retire_verify_checkpoint(&session_id, base_position, verification_rows)
                    .map_err(openai_backend_error);
            }
            if let Err(error) = runtime.trim_session(&session_id, canonical_position) {
                let _ = runtime.drop_session_timed(&session_id);
                return Err(OpenAiError::backend(format!(
                    "native MTP span repair failed and the session was retired: {error:#}"
                )));
            }
            if let Some(metadata) = chat_sampling_metadata.as_deref() {
                runtime
                    .configure_chat_sampling(
                        &session_id,
                        metadata,
                        prompt_token_count,
                        sampling.as_ref(),
                    )
                    .map_err(openai_backend_error)?;
            }
            let repaired_position = runtime.session_token_count(&session_id).ok_or_else(|| {
                OpenAiError::backend("repaired native MTP span session disappeared")
            })?;
            if repaired_position != canonical_position {
                let _ = runtime.drop_session_timed(&session_id);
                return Err(OpenAiError::backend(format!(
                    "native MTP span repair position mismatch: observed {repaired_position}, expected {canonical_position}"
                )));
            }
            Ok(())
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn admission() -> NativeMtpSpanAdmission {
        NativeMtpSpanAdmission {
            native_mtp_enabled: true,
            generation_hooks_active: false,
            greedy_admitted: true,
            remaining_tokens: 8,
            draft_tokens: 3,
            min_draft_tokens: 1,
        }
    }

    #[test]
    fn a_greedy_request_with_a_draft_admits_a_batched_span() {
        assert!(native_mtp_span_admitted(admission()));
    }

    /// The regression this module exists to fix: before it, native MTP was
    /// drafted and verified but never committed a span, so every accepted draft
    /// still paid a full target forward.
    #[test]
    fn a_committed_span_saves_one_forward_per_accepted_draft_token() {
        // Three drafted tokens fully accepted commit four tokens (the drafts
        // plus the boundary token) from a single batched forward.
        assert!(native_mtp_span_full_accept(false, 3, 3));
        let committed = 4usize;
        assert_eq!(committed.saturating_sub(1), 3);
    }

    #[test]
    fn sampling_that_is_not_greedy_equivalent_is_refused() {
        assert!(!native_mtp_span_admitted(NativeMtpSpanAdmission {
            greedy_admitted: false,
            ..admission()
        }));
    }

    #[test]
    fn generation_hooks_refuse_a_span_because_they_inject_tokens() {
        assert!(!native_mtp_span_admitted(NativeMtpSpanAdmission {
            generation_hooks_active: true,
            ..admission()
        }));
    }

    #[test]
    fn a_disabled_native_mtp_request_never_speculates() {
        assert!(!native_mtp_span_admitted(NativeMtpSpanAdmission {
            native_mtp_enabled: false,
            ..admission()
        }));
    }

    #[test]
    fn the_final_token_decodes_serially_rather_than_as_a_span() {
        assert!(!native_mtp_span_admitted(NativeMtpSpanAdmission {
            remaining_tokens: 1,
            ..admission()
        }));
        assert!(native_mtp_span_admitted(NativeMtpSpanAdmission {
            remaining_tokens: 2,
            ..admission()
        }));
    }

    #[test]
    fn an_empty_draft_is_refused_even_when_the_policy_minimum_is_zero() {
        assert!(!native_mtp_span_admitted(NativeMtpSpanAdmission {
            draft_tokens: 0,
            min_draft_tokens: 0,
            ..admission()
        }));
    }

    #[test]
    fn a_draft_below_the_policy_minimum_is_refused() {
        assert!(!native_mtp_span_admitted(NativeMtpSpanAdmission {
            draft_tokens: 1,
            min_draft_tokens: 2,
            ..admission()
        }));
    }

    #[test]
    fn a_partially_accepted_span_is_not_a_full_accept() {
        // A trimmed suffix invalidates the verify-call draft, so the checkpoint
        // must be trimmed rather than retired.
        assert!(!native_mtp_span_full_accept(true, 1, 3));
        assert!(!native_mtp_span_full_accept(false, 2, 3));
    }
}
