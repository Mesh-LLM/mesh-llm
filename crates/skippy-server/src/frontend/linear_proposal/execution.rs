use std::time::Instant;

use openai_frontend::{OpenAiError, OpenAiResult};
use skippy_runtime::SamplingConfig;

use crate::frontend::{
    LinearProposalDisposition, NativeMtpVerifyWindowDecision, StageOpenAiBackend, TokenControl,
    classify_native_mtp_verify_window, openai_backend_error,
};

use super::{LinearProposalReceipt, QueriedLinearProposal};

struct LinearProposalExecution {
    decision: NativeMtpVerifyWindowDecision,
    predictions: Vec<i32>,
    committed_tokens: Vec<i32>,
    reached_stop: bool,
    position_after_verification: u64,
    canonical_position: u64,
    verification_elapsed_us: u64,
    repair_elapsed_us: u64,
    runtime_lock_wait_us: u64,
    runtime_lock_hold_us: u64,
    runtime_lock_acquires: usize,
}

#[derive(Clone, Copy)]
pub(crate) struct LinearProposalExecutionParams<'a> {
    pub(crate) request_id: u64,
    pub(crate) request_session_id: u64,
    pub(crate) session_id: &'a str,
    pub(crate) current: i32,
    pub(crate) base_position: u64,
    pub(crate) generated_len: usize,
    pub(crate) max_new_tokens: usize,
    pub(crate) sampling: &'a SamplingConfig,
    pub(crate) chat_sampling_metadata: Option<&'a str>,
    pub(crate) prompt_token_count: usize,
}

#[derive(Default)]
struct LinearProposalRepairTiming {
    elapsed_us: u64,
    runtime_lock_wait_us: u64,
    runtime_lock_hold_us: u64,
    runtime_lock_acquires: usize,
}

struct LinearProposalRepairParams<'a> {
    session_id: &'a str,
    checkpoint_start: u64,
    checkpoint_count: usize,
    canonical_position: u64,
    position_after_verification: u64,
    sampling: &'a SamplingConfig,
    chat_sampling_metadata: Option<&'a str>,
    prompt_token_count: usize,
}

impl StageOpenAiBackend {
    pub(crate) fn execute_local_linear_proposal(
        &self,
        params: LinearProposalExecutionParams<'_>,
        queried: QueriedLinearProposal,
        cancellation: Option<&openai_frontend::CancellationToken>,
        on_token: &mut (impl FnMut(i32) -> OpenAiResult<TokenControl> + ?Sized),
    ) -> OpenAiResult<Option<LinearProposalReceipt>> {
        ensure_request_active(cancellation)?;
        let proposal_token_count = queried.proposal.token_ids.len();
        let mut verify_inputs = Vec::with_capacity(proposal_token_count.saturating_add(1));
        verify_inputs.push(params.current);
        verify_inputs.extend_from_slice(&queried.proposal.token_ids);

        let Some(execution) = self.execute_local_linear_proposal_inner(
            params,
            &queried.proposal.token_ids,
            &verify_inputs,
            cancellation,
            on_token,
        )?
        else {
            return Ok(None);
        };
        let accepted_proposal_tokens = execution
            .decision
            .accepted_proposal_tokens
            .min(execution.committed_tokens.len());
        let disposition = linear_proposal_disposition(
            execution.decision,
            proposal_token_count,
            execution.committed_tokens.len(),
            execution.reached_stop,
        );
        if execution.committed_tokens.is_empty() {
            return Err(OpenAiError::backend(
                "linear proposal committed no target token",
            ));
        }
        let correction_or_boundary_token = (disposition != LinearProposalDisposition::Stopped)
            .then(|| {
                execution
                    .committed_tokens
                    .last()
                    .copied()
                    .expect("checked non-empty committed tokens")
            });
        let generated_token_count = params
            .generated_len
            .checked_add(execution.committed_tokens.len())
            .ok_or_else(|| {
                OpenAiError::backend("linear proposal generated-token count overflow")
            })?;
        let total_elapsed_us = elapsed_us(queried.operation_started);
        Ok(Some(LinearProposalReceipt {
            request_id: params.request_id,
            session_id: params.request_session_id,
            decision_id: queried.proposal.decision_id,
            disposition,
            proposal_token_count,
            verification_rows: verify_inputs.len(),
            accepted_proposal_tokens,
            generated_token_count,
            canonical_prediction_count: execution.committed_tokens.len(),
            committed_tokens: execution.committed_tokens.into_boxed_slice(),
            verification_row_predictions: execution.predictions.into_boxed_slice(),
            correction_or_boundary_token,
            base_position: params.base_position,
            position_after_verification: execution.position_after_verification,
            canonical_position: execution.canonical_position,
            trimmed_rows: usize::try_from(
                execution
                    .position_after_verification
                    .saturating_sub(execution.canonical_position),
            )
            .map_err(|_| OpenAiError::backend("trimmed row count exceeds usize"))?,
            proposal_elapsed_us: queried.proposal_elapsed_us,
            verification_elapsed_us: execution.verification_elapsed_us,
            repair_elapsed_us: execution.repair_elapsed_us,
            total_elapsed_us,
            runtime_lock_wait_us: execution.runtime_lock_wait_us,
            runtime_lock_hold_us: execution.runtime_lock_hold_us,
            runtime_lock_acquires: execution.runtime_lock_acquires,
        }))
    }

    fn execute_local_linear_proposal_inner(
        &self,
        params: LinearProposalExecutionParams<'_>,
        proposal_tokens: &[i32],
        verify_inputs: &[i32],
        cancellation: Option<&openai_frontend::CancellationToken>,
        on_token: &mut (impl FnMut(i32) -> OpenAiResult<TokenControl> + ?Sized),
    ) -> OpenAiResult<Option<LinearProposalExecution>> {
        ensure_request_active(cancellation)?;
        let verify_timer = Instant::now();
        let session_id = params.session_id.to_string();
        let owned_verify_inputs = verify_inputs.to_vec();
        let owned_proposal_tokens = proposal_tokens.to_vec();
        let sampling = params.sampling.enabled.then_some(params.sampling.clone());
        let base_position = params.base_position;
        let generated_len = params.generated_len;
        let max_new_tokens = params.max_new_tokens;
        let verification = self.iteration_scheduler.execute_runtime_timed(
            "linear-proposal-verify",
            move |runtime| {
                let observed_position = runtime
                    .session_token_count(&session_id)
                    .ok_or_else(|| OpenAiError::backend("linear proposal session is not active"))?;
                if observed_position != base_position {
                    return Ok(None);
                }
                let predictions = runtime
                    .verify_tokens_sampled(&session_id, &owned_verify_inputs, sampling.as_ref())
                    .map_err(openai_backend_error)?;
                let decision = classify_native_mtp_verify_window(
                    &owned_proposal_tokens,
                    &predictions,
                    generated_len,
                    max_new_tokens,
                    |token| {
                        runtime
                            .model
                            .token_is_eog(token)
                            .map_err(openai_backend_error)
                    },
                )?;
                let position_after_verification = runtime
                    .session_token_count(&session_id)
                    .ok_or_else(|| OpenAiError::backend("linear proposal session disappeared"))?;
                Ok(Some((predictions, decision, position_after_verification)))
            },
        )?;
        let verification_lock_wait_us = millis_to_micros(verification.runtime_lock_wait_ms);
        let verification_lock_hold_us = millis_to_micros(verification.runtime_lock_hold_ms);
        let execution = verification.value;
        let Some((predictions, decision, position_after_verification)) = execution else {
            return Ok(None);
        };
        let expected_position_after_verification = params
            .base_position
            .checked_add(
                u64::try_from(verify_inputs.len())
                    .map_err(|_| OpenAiError::backend("verification row count exceeds u64"))?,
            )
            .ok_or_else(|| OpenAiError::backend("linear proposal position overflow"))?;
        if position_after_verification != expected_position_after_verification {
            return Err(OpenAiError::backend(format!(
                "linear proposal verification position mismatch: observed {position_after_verification}, expected {expected_position_after_verification}"
            )));
        }
        let verification_elapsed_us = elapsed_us(verify_timer);

        let mut committed_tokens = Vec::with_capacity(decision.commit_count);
        let mut reached_stop = false;
        let mut callback_error = None;
        for token in predictions.iter().copied().take(decision.commit_count) {
            if cancellation.is_some_and(openai_frontend::CancellationToken::is_cancelled) {
                callback_error = Some(OpenAiError::backend("request cancelled"));
                break;
            }
            committed_tokens.push(token);
            match on_token(token) {
                Ok(TokenControl::Continue) => {}
                Ok(TokenControl::Stop) => {
                    reached_stop = true;
                    break;
                }
                Err(error) => {
                    callback_error = Some(error);
                    break;
                }
            }
        }
        let canonical_position = params
            .base_position
            .checked_add(
                u64::try_from(committed_tokens.len())
                    .map_err(|_| OpenAiError::backend("committed token count exceeds u64"))?,
            )
            .ok_or_else(|| OpenAiError::backend("linear proposal canonical position overflow"))?;
        let repair = finish_linear_proposal_after_repair(callback_error, || {
            self.trim_branch_suffix_or_retire(LinearProposalRepairParams {
                session_id: params.session_id,
                checkpoint_start: params.base_position,
                checkpoint_count: verify_inputs.len(),
                canonical_position,
                position_after_verification,
                sampling: params.sampling,
                chat_sampling_metadata: params.chat_sampling_metadata,
                prompt_token_count: params.prompt_token_count,
            })
        })?;

        if committed_tokens.is_empty() {
            return Err(OpenAiError::backend(
                "linear proposal classifier committed no target prediction",
            ));
        }

        Ok(Some(LinearProposalExecution {
            decision,
            predictions,
            committed_tokens,
            reached_stop,
            position_after_verification,
            canonical_position,
            verification_elapsed_us,
            repair_elapsed_us: repair.elapsed_us,
            runtime_lock_wait_us: verification_lock_wait_us
                .saturating_add(repair.runtime_lock_wait_us),
            runtime_lock_hold_us: verification_lock_hold_us
                .saturating_add(repair.runtime_lock_hold_us),
            runtime_lock_acquires: 1usize.saturating_add(repair.runtime_lock_acquires),
        }))
    }

    fn trim_branch_suffix_or_retire(
        &self,
        params: LinearProposalRepairParams<'_>,
    ) -> OpenAiResult<LinearProposalRepairTiming> {
        let LinearProposalRepairParams {
            session_id,
            checkpoint_start,
            checkpoint_count,
            canonical_position,
            position_after_verification,
            sampling,
            chat_sampling_metadata,
            prompt_token_count,
        } = params;
        let session_id = session_id.to_string();
        let sampling = sampling.clone();
        let chat_sampling_metadata = chat_sampling_metadata.map(str::to_string);
        if canonical_position >= position_after_verification {
            let retire_timer = Instant::now();
            let outcome = self.iteration_scheduler.execute_runtime_timed(
                "linear-proposal-retire",
                move |runtime| {
                    runtime
                        .retire_verify_checkpoint(
                            &session_id,
                            checkpoint_start,
                            checkpoint_count as u64,
                        )
                        .map_err(openai_backend_error)
                },
            )?;
            let elapsed_us = elapsed_us(retire_timer);
            return Ok(LinearProposalRepairTiming {
                elapsed_us,
                runtime_lock_wait_us: millis_to_micros(outcome.runtime_lock_wait_ms),
                runtime_lock_hold_us: millis_to_micros(outcome.runtime_lock_hold_ms),
                runtime_lock_acquires: 1,
            });
        }

        let repair_timer = Instant::now();
        let outcome = self.iteration_scheduler.execute_runtime_timed(
            "linear-proposal-repair",
            move |runtime| {
                if let Err(error) = runtime.trim_session(&session_id, canonical_position) {
                    let _ = runtime.drop_session_timed(&session_id);
                    return Err(OpenAiError::backend(format!(
                        "linear proposal repair failed and the session was retired: {error:#}"
                    )));
                }
                if let Some(metadata) = chat_sampling_metadata.as_deref() {
                    runtime
                        .configure_chat_sampling(
                            &session_id,
                            metadata,
                            u64::try_from(prompt_token_count).unwrap_or(u64::MAX),
                            sampling.enabled.then_some(&sampling),
                        )
                        .map_err(openai_backend_error)?;
                }
                let repaired_position =
                    runtime.session_token_count(&session_id).ok_or_else(|| {
                        OpenAiError::backend("repaired linear proposal session disappeared")
                    })?;
                if repaired_position != canonical_position {
                    let _ = runtime.drop_session_timed(&session_id);
                    return Err(OpenAiError::backend(format!(
                        "linear proposal repair position mismatch: observed {repaired_position}, expected {canonical_position}"
                    )));
                }
                Ok(())
            },
        )?;
        let repair_elapsed_us = elapsed_us(repair_timer);
        Ok(LinearProposalRepairTiming {
            elapsed_us: repair_elapsed_us,
            runtime_lock_wait_us: millis_to_micros(outcome.runtime_lock_wait_ms),
            runtime_lock_hold_us: millis_to_micros(outcome.runtime_lock_hold_ms),
            runtime_lock_acquires: 1,
        })
    }
}

fn ensure_request_active(
    cancellation: Option<&openai_frontend::CancellationToken>,
) -> OpenAiResult<()> {
    if cancellation.is_some_and(openai_frontend::CancellationToken::is_cancelled) {
        Err(OpenAiError::backend("request cancelled"))
    } else {
        Ok(())
    }
}

pub(crate) fn elapsed_us(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX)
}

fn millis_to_micros(value: f64) -> u64 {
    if value.is_finite() && value > 0.0 {
        (value * 1_000.0).round() as u64
    } else {
        0
    }
}

fn linear_proposal_disposition(
    decision: NativeMtpVerifyWindowDecision,
    proposal_token_count: usize,
    committed_token_count: usize,
    reached_stop: bool,
) -> LinearProposalDisposition {
    if reached_stop
        || (!decision.rejected
            && (decision.accepted_proposal_tokens != proposal_token_count
                || committed_token_count != proposal_token_count.saturating_add(1)))
    {
        LinearProposalDisposition::Stopped
    } else if decision.rejected {
        LinearProposalDisposition::FirstMismatch
    } else {
        LinearProposalDisposition::FullAccept
    }
}

fn finish_linear_proposal_after_repair<T>(
    callback_error: Option<OpenAiError>,
    repair: impl FnOnce() -> OpenAiResult<T>,
) -> OpenAiResult<T> {
    let repaired = repair()?;
    callback_error.map_or(Ok(repaired), Err)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    fn decision(proposal: &[i32], predictions: &[i32]) -> NativeMtpVerifyWindowDecision {
        classify_native_mtp_verify_window(proposal, predictions, 0, 64, |_| Ok(false)).unwrap()
    }

    #[test]
    fn disposition_distinguishes_full_mismatch_and_early_stop() {
        let full = decision(&[11, 12], &[11, 12, 13]);
        assert_eq!(
            linear_proposal_disposition(full, 2, 3, false),
            LinearProposalDisposition::FullAccept
        );

        let mismatch = decision(&[11, 12], &[11, 99, 13]);
        assert_eq!(
            linear_proposal_disposition(mismatch, 2, 2, false),
            LinearProposalDisposition::FirstMismatch
        );

        assert_eq!(
            linear_proposal_disposition(full, 2, 1, true),
            LinearProposalDisposition::Stopped
        );
        assert_eq!(
            linear_proposal_disposition(full, 2, 1, false),
            LinearProposalDisposition::Stopped
        );
    }

    #[test]
    fn callback_error_is_returned_only_after_repair_runs() {
        let repair_ran = Cell::new(false);
        let result = finish_linear_proposal_after_repair(
            Some(OpenAiError::backend("synthetic callback failure")),
            || {
                repair_ran.set(true);
                Ok(())
            },
        );

        assert!(repair_ran.get());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("synthetic callback failure")
        );
    }

    #[test]
    fn cancellation_error_is_returned_only_after_repair_runs() {
        let repair_ran = Cell::new(false);
        let result = finish_linear_proposal_after_repair(
            Some(OpenAiError::backend("request cancelled")),
            || {
                repair_ran.set(true);
                Ok(())
            },
        );

        assert!(repair_ran.get());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("request cancelled")
        );
    }
}
