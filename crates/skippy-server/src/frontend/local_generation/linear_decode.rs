use std::collections::BTreeMap;
use std::time::Duration;

use openai_frontend::{OpenAiError, OpenAiResult};
use serde_json::json;

use crate::frontend::generation::{LocalGeneration, StageOpenAiBackend, TokenControl};
use crate::frontend::linear_proposal::{
    LinearProposalDiscardReason, LinearProposalExecutionParams, LinearProposalQueryOutcome,
    LinearProposalQueryParams, execute_linear_proposal_with_terminal_discard,
    query_linear_proposal, report_linear_proposal_receipt,
};
use crate::frontend::{LinearProposalDisposition, LinearProposalSourceTelemetry};

use super::token_generation::{DecodeState, LinearProposalProgress};

impl StageOpenAiBackend {
    pub(super) fn try_execute_linear_proposal(
        &self,
        request: &LocalGeneration<'_>,
        session_id: &str,
        state: &mut DecodeState,
        emit_token: &mut impl FnMut(i32) -> OpenAiResult<TokenControl>,
    ) -> OpenAiResult<LinearProposalProgress> {
        let Some(config) = self.linear_proposal_ingress.as_ref() else {
            return Ok(LinearProposalProgress::NotUsed);
        };
        let Some(committed_token_ids) = state.linear_context_tokens.as_mut() else {
            return Ok(LinearProposalProgress::NotUsed);
        };
        if request
            .cancellation
            .is_some_and(openai_frontend::CancellationToken::is_cancelled)
        {
            return Err(OpenAiError::backend("request cancelled"));
        }
        let remaining_new_tokens =
            (request.max_tokens as usize).saturating_sub(state.decoded_tokens);
        // Prefill leaves the final prompt token undecoded. When whole-prompt
        // prefill also samples the first target token, decoded_tokens is one;
        // otherwise it is zero. Those two modes therefore share this position.
        let base_position = u64::try_from(
            request
                .prompt_token_ids
                .len()
                .saturating_sub(1)
                .checked_add(state.decoded_tokens)
                .ok_or_else(|| {
                    OpenAiError::backend("linear proposal base position exceeds usize")
                })?,
        )
        .map_err(|_| OpenAiError::backend("linear proposal base position exceeds u64"))?;
        let queried = match query_linear_proposal(
            config,
            LinearProposalQueryParams {
                request_id: request.ids.request_id,
                session_id: request.ids.session_id,
                prompt_token_count: request.prompt_token_ids.len(),
                decode_step: state.decoded_tokens,
                committed_token_count: request
                    .prompt_token_ids
                    .len()
                    .saturating_add(state.decoded_tokens),
                remaining_new_tokens,
                runtime_max_proposal_tokens: state.linear_proposal_max_tokens,
                pending_token_ids: state
                    .pending_linear_proposal_tokens
                    .clone()
                    .into_boxed_slice(),
            },
        ) {
            Ok(LinearProposalQueryOutcome::Skipped) => {
                return Ok(LinearProposalProgress::NotUsed);
            }
            Ok(outcome) => {
                state.pending_linear_proposal_tokens.clear();
                match outcome {
                    LinearProposalQueryOutcome::Skipped => unreachable!("handled above"),
                    LinearProposalQueryOutcome::NoProposal { source_telemetry } => {
                        self.emit_linear_proposal_source_telemetry(
                            source_telemetry,
                            state.emit_token_debug,
                        );
                        None
                    }
                    LinearProposalQueryOutcome::DeadlineExceeded {
                        proposal_elapsed_us,
                        source_telemetry,
                    } => {
                        self.emit_linear_proposal_source_telemetry(
                            source_telemetry,
                            state.emit_token_debug,
                        );
                        let mut attrs = BTreeMap::new();
                        attrs.insert(
                            "llama_stage.linear_proposal.discard_reason".to_string(),
                            json!("deadline_exceeded"),
                        );
                        attrs.insert(
                            "llama_stage.linear_proposal.proposal_us".to_string(),
                            json!(proposal_elapsed_us),
                        );
                        self.telemetry
                            .emit("stage.openai_linear_proposal_late", attrs);
                        None
                    }
                    LinearProposalQueryOutcome::Ready(queried) => {
                        self.emit_linear_proposal_source_telemetry(
                            queried.source_telemetry,
                            state.emit_token_debug,
                        );
                        Some(queried)
                    }
                }
            }
            Err(error) => {
                state.linear_context_tokens = None;
                state.linear_proposal_max_tokens = 0;
                self.telemetry.emit(
                    "stage.openai_linear_proposal_disabled",
                    BTreeMap::from([(
                        "llama_stage.linear_proposal.error".to_string(),
                        json!(error.to_string()),
                    )]),
                );
                return Ok(LinearProposalProgress::NotUsed);
            }
        };
        if request
            .cancellation
            .is_some_and(openai_frontend::CancellationToken::is_cancelled)
        {
            return Err(OpenAiError::backend("request cancelled"));
        }
        let Some(queried) = queried else {
            return Ok(LinearProposalProgress::NotUsed);
        };
        let decision_id = queried.proposal.decision_id.clone();
        let receipt = execute_linear_proposal_with_terminal_discard(config, &decision_id, || {
            self.execute_local_linear_proposal(
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
                queried,
                request.cancellation,
                emit_token,
            )
        })?;
        if receipt.is_none() {
            let discard_failed = config
                .source()
                .discard(&decision_id, LinearProposalDiscardReason::PositionMismatch)
                .is_err();
            if discard_failed {
                self.telemetry.emit(
                    "stage.openai_linear_proposal_discard_failed",
                    BTreeMap::from([(
                        "llama_stage.linear_proposal.discard_reason".to_string(),
                        json!("position_mismatch"),
                    )]),
                );
            }
        }
        let Some(receipt) = receipt else {
            return Ok(LinearProposalProgress::NotUsed);
        };
        if report_linear_proposal_receipt(config, &receipt).is_some() {
            let mut attrs = BTreeMap::new();
            receipt.insert_telemetry_attrs(&mut attrs);
            attrs.insert(
                "llama_stage.linear_proposal.report_outcome".to_string(),
                json!("failed"),
            );
            self.telemetry
                .emit("stage.openai_linear_proposal_report_failed", attrs);
        }
        let proposal_runtime_lock_wait_ms =
            Duration::from_micros(receipt.runtime_lock_wait_us).as_secs_f64() * 1_000.0;
        let proposal_runtime_lock_hold_ms =
            Duration::from_micros(receipt.runtime_lock_hold_us).as_secs_f64() * 1_000.0;
        state.runtime_lock_wait_ms += proposal_runtime_lock_wait_ms;
        state.runtime_lock_wait_max_ms = state
            .runtime_lock_wait_max_ms
            .max(proposal_runtime_lock_wait_ms);
        state.runtime_lock_hold_ms += proposal_runtime_lock_hold_ms;
        state.runtime_lock_hold_max_ms = state
            .runtime_lock_hold_max_ms
            .max(proposal_runtime_lock_hold_ms);
        state.runtime_lock_acquires = state
            .runtime_lock_acquires
            .saturating_add(receipt.runtime_lock_acquires);
        state.decoded_tokens = state
            .decoded_tokens
            .checked_add(receipt.committed_tokens.len())
            .ok_or_else(|| OpenAiError::backend("linear proposal decode count overflow"))?;
        state.current = *receipt
            .committed_tokens
            .last()
            .ok_or_else(|| OpenAiError::backend("linear proposal receipt committed no tokens"))?;
        committed_token_ids.extend_from_slice(&receipt.committed_tokens);
        let stopped_by_proposal = receipt.disposition == LinearProposalDisposition::Stopped;
        if state.emit_token_debug {
            let mut proposal_attrs = BTreeMap::new();
            receipt.insert_telemetry_attrs(&mut proposal_attrs);
            self.telemetry
                .emit_debug("stage.openai_linear_proposal", proposal_attrs);
        }
        if stopped_by_proposal || state.decoded_tokens >= request.max_tokens as usize {
            Ok(LinearProposalProgress::Stop)
        } else {
            Ok(LinearProposalProgress::Continue)
        }
    }

    fn emit_linear_proposal_source_telemetry(
        &self,
        source_telemetry: Option<LinearProposalSourceTelemetry>,
        emit_token_debug: bool,
    ) {
        let Some(source_telemetry) = source_telemetry else {
            return;
        };
        let mut attrs = BTreeMap::new();
        source_telemetry.insert_telemetry_attrs(&mut attrs);
        match source_telemetry.outcome {
            crate::frontend::LinearProposalSourceOutcome::Ready
            | crate::frontend::LinearProposalSourceOutcome::Abstained
                if emit_token_debug =>
            {
                self.telemetry
                    .emit_debug("stage.openai_linear_proposal_source", attrs);
            }
            crate::frontend::LinearProposalSourceOutcome::Ready
            | crate::frontend::LinearProposalSourceOutcome::Abstained => {}
            _ => self
                .telemetry
                .emit("stage.openai_linear_proposal_source", attrs),
        }
    }
}
