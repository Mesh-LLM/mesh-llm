//! DFlash block speculation for single-node decode.
//!
//! The DFlash draft attached to the target model drafts a whole block from the
//! target hidden states captured on every target decode. The block then runs
//! through the linear-proposal verifier: one batched target forward, commit
//! through the first mismatch, then trim or retire the speculative suffix.

use std::time::Instant;

use skippy_inference_api::{InferenceError, InferenceResult};

use crate::frontend::generation::{LocalGeneration, StageOpenAiBackend, TokenControl};
use crate::frontend::linear_proposal::{LinearProposalExecution, LinearProposalExecutionParams};
use crate::frontend::speculative::InferenceSpeculativeStats;
use crate::frontend::util::openai_backend_error;

use super::token_generation::DecodeState;

/// Outcome of one DFlash block attempt.
pub(super) enum DFlashSpanProgress {
    /// No block was drafted or verified; the caller decodes serially.
    NotUsed,
    /// Tokens were committed and decoding should continue.
    Continue,
    /// Tokens were committed and generation should stop.
    Stop,
}

/// Consecutive windows that accept no draft before a request backs off.
const MISS_LIMIT: u32 = 2;
/// Serial tokens decoded after the first back-off; each further miss doubles
/// it up to [`MAX_BACKOFF_TOKENS`].
const FIRST_BACKOFF_TOKENS: usize = 4;
const MAX_BACKOFF_TOKENS: usize = 64;

/// Request-local DFlash bounds and counters.
pub(super) struct DFlashDecodeState {
    max_draft_tokens: usize,
    backoff: DFlashBackoff,
    pub(super) stats: InferenceSpeculativeStats,
}

/// Per-request back-off from windows that accept nothing.
///
/// A window that accepts no draft pays a full block verify for the one token
/// a serial step would have produced. Without this, a request whose output the
/// draft cannot predict pays that on every token for the whole request; the
/// model-wide speculation gate only reacts after many requests.
#[derive(Debug)]
struct DFlashBackoff {
    misses: u32,
    next_backoff_tokens: usize,
    serial_tokens_remaining: usize,
}

impl Default for DFlashBackoff {
    fn default() -> Self {
        Self {
            misses: 0,
            next_backoff_tokens: FIRST_BACKOFF_TOKENS,
            serial_tokens_remaining: 0,
        }
    }
}

impl DFlashBackoff {
    /// Whether the next token should decode serially instead of drafting.
    fn take_serial_token(&mut self) -> bool {
        let serial = self.serial_tokens_remaining > 0;
        self.serial_tokens_remaining = self.serial_tokens_remaining.saturating_sub(1);
        serial
    }

    fn record_window(&mut self, accepted: usize) {
        if accepted > 0 {
            *self = Self::default();
            return;
        }
        self.misses += 1;
        if self.misses < MISS_LIMIT {
            return;
        }
        self.serial_tokens_remaining = self.next_backoff_tokens;
        self.next_backoff_tokens = (self.next_backoff_tokens * 2).min(MAX_BACKOFF_TOKENS);
        // A window that misses right after a back-off backs off again, longer.
        self.misses = MISS_LIMIT - 1;
    }
}

impl DFlashDecodeState {
    /// DFlash commits by token equality, which is only a valid acceptance
    /// test under greedy sampling, and generation hooks need serial steps.
    fn for_request(
        request: &LocalGeneration<'_>,
        greedy_admitted: bool,
        generation_hooks_active: bool,
    ) -> Option<Self> {
        let dflash = request.speculative.dflash.as_ref()?;
        (greedy_admitted && !generation_hooks_active).then(|| Self {
            max_draft_tokens: dflash.max_draft_tokens.unwrap_or(usize::MAX),
            backoff: DFlashBackoff::default(),
            stats: InferenceSpeculativeStats::default(),
        })
    }

    /// The target verifies the anchor plus every draft in one batch, so a
    /// block holds at most `batch_size - 1` drafts. Returns `None` when no
    /// draft fits, leaving the request to decode serially.
    fn capped_to_batch(mut self, batch_size: usize) -> Option<Self> {
        self.max_draft_tokens = self.max_draft_tokens.min(batch_size.saturating_sub(1));
        (self.max_draft_tokens > 0).then_some(self)
    }
}

/// Draft tokens worth requesting: a verified block commits one token beyond
/// its drafts, so the final remaining token is always a serial step.
pub(super) fn dflash_block_limit(max_draft_tokens: usize, remaining_tokens: usize) -> usize {
    max_draft_tokens.min(remaining_tokens.saturating_sub(1))
}

fn record_window(
    stats: &mut InferenceSpeculativeStats,
    draft_tokens: usize,
    verify_rows: usize,
    execution: &LinearProposalExecution,
) -> usize {
    let accepted = execution
        .decision
        .accepted_proposal_tokens
        .min(draft_tokens);
    stats.windows += 1;
    stats.draft_tokens += draft_tokens;
    stats.accepted_tokens += accepted;
    stats.rejected_tokens += draft_tokens - accepted;
    stats.primary_verify_requests += 1;
    stats.primary_verify_tokens += verify_rows;
    stats.primary_verify_elapsed_ms += execution.verification_elapsed_us as f64 / 1_000.0;
    stats.primary_verify_runtime_lock_wait_ms += execution.runtime_lock_wait_us as f64 / 1_000.0;
    stats.primary_verify_runtime_lock_hold_ms += execution.runtime_lock_hold_us as f64 / 1_000.0;
    if execution.decision.rejected {
        stats.rejected_windows += 1;
        stats.first_reject_position_sum += accepted;
    } else if accepted == draft_tokens {
        stats.full_accept_windows += 1;
    } else if execution.reached_stop {
        stats.accepted_stop_windows += 1;
    }
    accepted
}

impl StageOpenAiBackend {
    /// Admits DFlash for a request, bounded by the session's batch size.
    pub(super) fn admit_dflash(
        &self,
        request: &LocalGeneration<'_>,
        session_id: &str,
        greedy_admitted: bool,
        generation_hooks_active: bool,
    ) -> InferenceResult<Option<DFlashDecodeState>> {
        let Some(dflash) =
            DFlashDecodeState::for_request(request, greedy_admitted, generation_hooks_active)
        else {
            return Ok(None);
        };
        let scheduler_session_id = session_id.to_string();
        let batch_size =
            self.iteration_scheduler
                .execute_runtime("dflash-admission", move |runtime| {
                    runtime
                        .admit_session_batch_size(&scheduler_session_id)
                        .map_err(openai_backend_error)
                })?;
        Ok(dflash.capped_to_batch(batch_size))
    }

    /// Drafts one DFlash block and commits its verified prefix.
    ///
    /// Returns [`DFlashSpanProgress::NotUsed`] without advancing the session
    /// when no block is drafted, in which case the caller decodes serially.
    pub(super) fn try_execute_dflash_span(
        &self,
        request: &LocalGeneration<'_>,
        session_id: &str,
        state: &mut DecodeState,
        emit_token: &mut impl FnMut(i32) -> InferenceResult<TokenControl>,
    ) -> InferenceResult<DFlashSpanProgress> {
        let Some(dflash) = state.dflash.as_mut() else {
            return Ok(DFlashSpanProgress::NotUsed);
        };
        if dflash.backoff.take_serial_token() {
            return Ok(DFlashSpanProgress::NotUsed);
        }
        let max_draft_tokens = dflash.max_draft_tokens;
        let remaining = (request.max_tokens as usize).saturating_sub(state.decoded_tokens);
        let limit = dflash_block_limit(max_draft_tokens, remaining);
        if limit == 0 {
            return Ok(DFlashSpanProgress::NotUsed);
        }
        let propose_started = Instant::now();
        let owned_session_id = session_id.to_string();
        let anchor = state.current;
        let proposal =
            self.iteration_scheduler
                .execute_runtime("dflash-propose", move |runtime| {
                    runtime
                        .dflash_propose(&owned_session_id, anchor, limit)
                        .map_err(openai_backend_error)
                })?;
        if let Some(dflash) = state.dflash.as_mut() {
            dflash.stats.draft_propose_ms += propose_started.elapsed().as_secs_f64() * 1_000.0;
        }
        let Some(proposal) = proposal.filter(|proposal| !proposal.tokens.is_empty()) else {
            return Ok(DFlashSpanProgress::NotUsed);
        };

        // Prefill leaves the final prompt token undecoded; see the linear
        // proposal path for the shared position contract.
        let base_position = request
            .prompt_token_ids
            .len()
            .saturating_sub(1)
            .checked_add(state.decoded_tokens)
            .and_then(|position| u64::try_from(position).ok())
            .ok_or_else(|| InferenceError::backend("DFlash base position overflow"))?;
        let mut verify_inputs = Vec::with_capacity(proposal.tokens.len() + 1);
        verify_inputs.push(state.current);
        verify_inputs.extend_from_slice(&proposal.tokens);
        let Some(execution) = self.execute_local_linear_proposal_inner(
            LinearProposalExecutionParams {
                request_id: request.ids.request_id,
                request_session_id: request.ids.session_id,
                session_id,
                current: state.current,
                base_position,
                generated_len: state.decoded_tokens,
                max_new_tokens: request.max_tokens as usize,
                sampling: request.sampling,
                chat_sampling_metadata: request.chat_sampling_metadata,
                prompt_token_count: request.prompt_token_ids.len(),
            },
            &proposal.tokens,
            &verify_inputs,
            request.cancellation,
            emit_token,
        )?
        else {
            return Ok(DFlashSpanProgress::NotUsed);
        };
        if let Some(dflash) = state.dflash.as_mut() {
            let accepted = record_window(
                &mut dflash.stats,
                proposal.tokens.len(),
                verify_inputs.len(),
                &execution,
            );
            dflash.backoff.record_window(accepted);
        }
        apply_committed_tokens(state, &execution)?;
        if execution.reached_stop || state.decoded_tokens >= request.max_tokens as usize {
            Ok(DFlashSpanProgress::Stop)
        } else {
            Ok(DFlashSpanProgress::Continue)
        }
    }
}

fn apply_committed_tokens(
    state: &mut DecodeState,
    execution: &LinearProposalExecution,
) -> InferenceResult<()> {
    let committed = &execution.committed_tokens;
    state.decoded_tokens = state
        .decoded_tokens
        .checked_add(committed.len())
        .ok_or_else(|| InferenceError::backend("DFlash decode count overflow"))?;
    state.current = *committed
        .last()
        .ok_or_else(|| InferenceError::backend("DFlash block committed no tokens"))?;
    state.generated_token_ids.extend_from_slice(committed);
    if let Some(context) = state.linear_context_tokens.as_mut() {
        context.extend_from_slice(committed);
    }
    let lock_wait_ms = execution.runtime_lock_wait_us as f64 / 1_000.0;
    let lock_hold_ms = execution.runtime_lock_hold_us as f64 / 1_000.0;
    state.runtime_lock_wait_ms += lock_wait_ms;
    state.runtime_lock_wait_max_ms = state.runtime_lock_wait_max_ms.max(lock_wait_ms);
    state.runtime_lock_hold_ms += lock_hold_ms;
    state.runtime_lock_hold_max_ms = state.runtime_lock_hold_max_ms.max(lock_hold_ms);
    state.runtime_lock_acquires = state
        .runtime_lock_acquires
        .saturating_add(execution.runtime_lock_acquires);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frontend::NativeMtpVerifyWindowDecision;

    #[test]
    fn block_limit_leaves_the_last_token_to_a_serial_step() {
        assert_eq!(dflash_block_limit(15, 100), 15);
        assert_eq!(dflash_block_limit(15, 4), 3);
        assert_eq!(dflash_block_limit(15, 1), 0);
        assert_eq!(dflash_block_limit(15, 0), 0);
    }

    #[test]
    fn blocks_fit_the_target_batch_with_the_anchor() {
        let state = |max_draft_tokens| DFlashDecodeState {
            max_draft_tokens,
            backoff: DFlashBackoff::default(),
            stats: InferenceSpeculativeStats::default(),
        };

        let capped = state(15).capped_to_batch(8).expect("seven drafts fit");
        assert_eq!(capped.max_draft_tokens, 7);
        let uncapped = state(15).capped_to_batch(2048).expect("block fits");
        assert_eq!(uncapped.max_draft_tokens, 15);
        assert!(state(15).capped_to_batch(1).is_none());
    }

    fn serial_run(backoff: &mut DFlashBackoff) -> usize {
        std::iter::from_fn(|| backoff.take_serial_token().then_some(())).count()
    }

    #[test]
    fn missed_windows_back_off_longer_until_a_draft_lands() {
        let mut backoff = DFlashBackoff::default();
        backoff.record_window(0);
        assert_eq!(serial_run(&mut backoff), 0, "one miss is tolerated");

        backoff.record_window(0);
        assert_eq!(serial_run(&mut backoff), 4);
        backoff.record_window(0);
        assert_eq!(serial_run(&mut backoff), 8);
        for _ in 0..10 {
            backoff.record_window(0);
        }
        assert_eq!(serial_run(&mut backoff), MAX_BACKOFF_TOKENS);

        backoff.record_window(3);
        backoff.record_window(0);
        assert_eq!(
            serial_run(&mut backoff),
            0,
            "acceptance resets the back-off"
        );
        backoff.record_window(0);
        assert_eq!(serial_run(&mut backoff), 4);
    }

    fn execution(accepted: usize, rejected: bool, reached_stop: bool) -> LinearProposalExecution {
        LinearProposalExecution {
            decision: NativeMtpVerifyWindowDecision {
                accepted_proposal_tokens: accepted,
                commit_count: accepted + 1,
                rejected,
            },
            predictions: Vec::new(),
            committed_tokens: Vec::new(),
            reached_stop,
            position_after_verification: 0,
            canonical_position: 0,
            verification_elapsed_us: 2_000,
            repair_elapsed_us: 0,
            runtime_lock_wait_us: 0,
            runtime_lock_hold_us: 0,
            runtime_lock_acquires: 1,
        }
    }

    #[test]
    fn windows_record_acceptance_and_rejection_shape() {
        let mut stats = InferenceSpeculativeStats::default();

        record_window(&mut stats, 8, 9, &execution(8, false, false));
        record_window(&mut stats, 8, 9, &execution(3, true, false));

        assert_eq!(stats.windows, 2);
        assert_eq!(stats.draft_tokens, 16);
        assert_eq!(stats.accepted_tokens, 11);
        assert_eq!(stats.rejected_tokens, 5);
        assert_eq!(stats.full_accept_windows, 1);
        assert_eq!(stats.rejected_windows, 1);
        assert_eq!(stats.first_reject_position_sum, 3);
        assert_eq!(stats.primary_verify_tokens, 18);
        assert!((stats.primary_verify_elapsed_ms - 4.0).abs() < f64::EPSILON);
    }
}
