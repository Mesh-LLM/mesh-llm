use std::{
    collections::BTreeMap,
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::{Result, bail};
use openai_frontend::{OpenAiError, OpenAiResult};
use serde_json::json;
use skippy_runtime::SamplingConfig;

use crate::frontend::{
    NativeMtpVerifyWindowDecision, StageOpenAiBackend, TokenControl,
    classify_native_mtp_verify_window, openai_backend_error,
};

const MAX_OPAQUE_DECISION_ID_BYTES: usize = 64;
const MAX_LINEAR_PROPOSAL_TOKENS: usize = 256;

/// Source-owned identity that Skippy carries without interpreting.
///
/// Rich proposal provenance remains with the proposal source. Skippy uses this
/// bounded value only to join an authoritative verification receipt back to
/// the exact proposal decision that produced it.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct OpaqueProposalDecisionId(Box<[u8]>);

impl OpaqueProposalDecisionId {
    /// Validates and stores a source-defined decision identifier.
    pub fn new(bytes: impl Into<Vec<u8>>) -> Result<Self> {
        let bytes = bytes.into();
        if bytes.is_empty() {
            bail!("linear proposal decision ID must not be empty");
        }
        if bytes.len() > MAX_OPAQUE_DECISION_ID_BYTES {
            bail!(
                "linear proposal decision ID has {} bytes; maximum is {MAX_OPAQUE_DECISION_ID_BYTES}",
                bytes.len()
            );
        }
        Ok(Self(bytes.into_boxed_slice()))
    }

    /// Returns the opaque identifier bytes exactly as supplied by the source.
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

/// One source-selected linear proposal. The API is width one by construction.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct LinearProposal {
    /// Source identity used to correlate the eventual receipt or discard.
    pub decision_id: OpaqueProposalDecisionId,
    /// Width-one continuation tokens selected by the source.
    pub token_ids: Box<[i32]>,
}

impl LinearProposal {
    /// Creates a proposal for one source decision.
    pub fn new(decision_id: OpaqueProposalDecisionId, token_ids: impl Into<Vec<i32>>) -> Self {
        Self {
            decision_id,
            token_ids: token_ids.into().into_boxed_slice(),
        }
    }
}

/// Causal, committed-only state supplied to a proposal source.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct LinearProposalQuery<'a> {
    /// OpenAI request identity.
    pub request_id: u64,
    /// OpenAI session identity.
    pub session_id: u64,
    /// Number of target tokens already generated.
    pub decode_step: usize,
    /// Prompt and target tokens committed before this query.
    pub committed_token_ids: &'a [i32],
    /// Maximum proposal width Skippy will accept for this query.
    pub max_proposal_tokens: usize,
    /// Advisory deadline the synchronous source must honor.
    pub deadline: Instant,
}

/// Why Skippy rejected a source decision without producing a receipt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum LinearProposalDiscardReason {
    /// The source returned after the advisory deadline.
    DeadlineExceeded,
    /// The proposal was empty or exceeded the per-query token bound.
    InvalidTokenCount,
    /// The proposal contained an invalid negative token identifier.
    InvalidTokenId,
    /// The runtime session moved before verification could begin.
    PositionMismatch,
    /// Verification or canonical-state repair failed.
    ExecutionFailed,
}

/// How verification committed a linear proposal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum LinearProposalDisposition {
    /// Every proposal token matched and the boundary token was committed.
    FullAccept,
    /// Verification committed the target correction at the first mismatch.
    FirstMismatch,
    /// Generation stopped before the ordinary proposal boundary.
    Stopped,
}

/// Skippy-owned outcome for one verified linear proposal.
///
/// `committed_tokens` is the target-authoritative stream prefix. Predictions
/// after `canonical_prediction_count` are branch-conditioned observations and
/// must not be interpreted as future canonical target tokens after a mismatch.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct LinearProposalReceipt {
    /// Source identity copied from the proposal.
    pub decision_id: OpaqueProposalDecisionId,
    /// Target-authoritative verification outcome.
    pub disposition: LinearProposalDisposition,
    /// Number of tokens supplied by the proposal source.
    pub proposal_token_count: usize,
    /// Number of target verification rows executed.
    pub verification_rows: usize,
    /// Number of source tokens accepted by target verification.
    pub accepted_proposal_tokens: usize,
    /// Target tokens committed to the response stream.
    pub committed_tokens: Box<[i32]>,
    /// Prediction observed for every verification row.
    pub verification_row_predictions: Box<[i32]>,
    /// Prefix length of row predictions that remained canonical.
    pub canonical_prediction_count: usize,
    /// Correction token on mismatch or boundary token on full acceptance.
    pub correction_or_boundary_token: Option<i32>,
    /// Runtime session position before verification.
    pub base_position: u64,
    /// Runtime session position immediately after speculative verification.
    pub position_after_verification: u64,
    /// Runtime session position after canonical repair.
    pub canonical_position: u64,
    /// Non-canonical verification rows trimmed during repair.
    pub trimmed_rows: usize,
    /// Time spent waiting for the proposal source.
    pub proposal_elapsed_us: u64,
    /// Time spent verifying the proposal.
    pub verification_elapsed_us: u64,
    /// Time spent repairing the runtime session.
    pub repair_elapsed_us: u64,
    /// End-to-end proposal decision time.
    pub total_elapsed_us: u64,
    /// Aggregate runtime mutex wait time.
    pub runtime_lock_wait_us: u64,
    /// Aggregate runtime mutex hold time.
    pub runtime_lock_hold_us: u64,
    /// Number of runtime mutex acquisitions.
    pub runtime_lock_acquires: usize,
}

impl LinearProposalReceipt {
    pub(crate) fn insert_telemetry_attrs(&self, attrs: &mut BTreeMap<String, serde_json::Value>) {
        attrs.insert(
            "llama_stage.linear_proposal.disposition".to_string(),
            json!(match self.disposition {
                LinearProposalDisposition::FullAccept => "full_accept",
                LinearProposalDisposition::FirstMismatch => "first_mismatch",
                LinearProposalDisposition::Stopped => "stopped",
            }),
        );
        attrs.insert(
            "llama_stage.linear_proposal.proposed".to_string(),
            json!(self.proposal_token_count),
        );
        attrs.insert(
            "llama_stage.linear_proposal.verify_rows".to_string(),
            json!(self.verification_rows),
        );
        attrs.insert(
            "llama_stage.linear_proposal.accepted".to_string(),
            json!(self.accepted_proposal_tokens),
        );
        attrs.insert(
            "llama_stage.linear_proposal.committed".to_string(),
            json!(self.committed_tokens.len()),
        );
        attrs.insert(
            "llama_stage.linear_proposal.canonical_predictions".to_string(),
            json!(self.canonical_prediction_count),
        );
        attrs.insert(
            "llama_stage.linear_proposal.base_position".to_string(),
            json!(self.base_position),
        );
        attrs.insert(
            "llama_stage.linear_proposal.position_after_verification".to_string(),
            json!(self.position_after_verification),
        );
        attrs.insert(
            "llama_stage.linear_proposal.canonical_position".to_string(),
            json!(self.canonical_position),
        );
        attrs.insert(
            "llama_stage.linear_proposal.trimmed_rows".to_string(),
            json!(self.trimmed_rows),
        );
        attrs.insert(
            "llama_stage.linear_proposal.proposal_us".to_string(),
            json!(self.proposal_elapsed_us),
        );
        attrs.insert(
            "llama_stage.linear_proposal.verify_us".to_string(),
            json!(self.verification_elapsed_us),
        );
        attrs.insert(
            "llama_stage.linear_proposal.repair_us".to_string(),
            json!(self.repair_elapsed_us),
        );
        attrs.insert(
            "llama_stage.linear_proposal.total_us".to_string(),
            json!(self.total_elapsed_us),
        );
        attrs.insert(
            "llama_stage.linear_proposal.runtime_lock_wait_us".to_string(),
            json!(self.runtime_lock_wait_us),
        );
        attrs.insert(
            "llama_stage.linear_proposal.runtime_lock_hold_us".to_string(),
            json!(self.runtime_lock_hold_us),
        );
        attrs.insert(
            "llama_stage.linear_proposal.runtime_lock_acquires".to_string(),
            json!(self.runtime_lock_acquires),
        );
    }
}

/// In-process, source-neutral width-one proposal boundary.
///
/// Implementations must honor `query.deadline`. Skippy independently rejects a
/// proposal that arrives after it and calls `discard` so the source can resolve
/// any pending decision without treating it as verified.
pub trait LinearProposalIngress: Send + Sync {
    /// Returns an optional bounded proposal for the committed query state.
    fn propose(&self, query: LinearProposalQuery<'_>) -> Result<Option<LinearProposal>>;

    /// Receives the target-authoritative outcome for a verified proposal.
    fn report(&self, receipt: &LinearProposalReceipt) -> Result<()>;

    /// Resolves a source decision that Skippy could not verify.
    fn discard(
        &self,
        _decision_id: &OpaqueProposalDecisionId,
        _reason: LinearProposalDiscardReason,
    ) -> Result<()> {
        Ok(())
    }
}

#[derive(Clone)]
pub struct LinearProposalIngressConfig {
    source: Arc<dyn LinearProposalIngress>,
    deadline: Duration,
    max_proposal_tokens: usize,
}

impl LinearProposalIngressConfig {
    /// Creates a bounded proposal ingress.
    ///
    /// The deadline is advisory because `propose` executes synchronously on
    /// the decode thread. Implementations must observe `query.deadline` and
    /// return promptly; Skippy discards a proposal returned after the deadline
    /// but cannot preempt a blocked source.
    pub fn new(
        source: Arc<dyn LinearProposalIngress>,
        deadline: Duration,
        max_proposal_tokens: usize,
    ) -> Result<Self> {
        if deadline.is_zero() {
            bail!("linear proposal deadline must be greater than zero");
        }
        if max_proposal_tokens == 0 {
            bail!("linear proposal maximum token count must be greater than zero");
        }
        if max_proposal_tokens > MAX_LINEAR_PROPOSAL_TOKENS {
            bail!(
                "linear proposal maximum token count is {max_proposal_tokens}; hard limit is {MAX_LINEAR_PROPOSAL_TOKENS}"
            );
        }
        Ok(Self {
            source,
            deadline,
            max_proposal_tokens,
        })
    }

    /// Returns the advisory source deadline for each proposal query.
    pub fn deadline(&self) -> Duration {
        self.deadline
    }

    /// Returns the configured proposal-width bound.
    pub fn max_proposal_tokens(&self) -> usize {
        self.max_proposal_tokens
    }

    /// Returns the configured proposal source.
    pub fn source(&self) -> &Arc<dyn LinearProposalIngress> {
        &self.source
    }
}

pub(crate) struct QueriedLinearProposal {
    pub(crate) proposal: LinearProposal,
    pub(crate) proposal_elapsed_us: u64,
    pub(crate) operation_started: Instant,
}

pub(crate) enum LinearProposalQueryOutcome {
    NoProposal,
    DeadlineExceeded { proposal_elapsed_us: u64 },
    Ready(QueriedLinearProposal),
}

pub(crate) fn query_linear_proposal(
    config: &LinearProposalIngressConfig,
    request_id: u64,
    session_id: u64,
    decode_step: usize,
    committed_token_ids: &[i32],
    remaining_new_tokens: usize,
    runtime_max_proposal_tokens: usize,
) -> OpenAiResult<LinearProposalQueryOutcome> {
    let max_proposal_tokens = remaining_new_tokens
        .saturating_sub(1)
        .min(runtime_max_proposal_tokens)
        .min(config.max_proposal_tokens());
    if max_proposal_tokens == 0 {
        return Ok(LinearProposalQueryOutcome::NoProposal);
    }
    let operation_started = Instant::now();
    let deadline = operation_started
        .checked_add(config.deadline())
        .ok_or_else(|| OpenAiError::backend("linear proposal deadline overflow"))?;
    let proposal_started = Instant::now();
    let proposal = config
        .source()
        .propose(LinearProposalQuery {
            request_id,
            session_id,
            decode_step,
            committed_token_ids,
            max_proposal_tokens,
            deadline,
        })
        .map_err(openai_backend_error)?;
    let proposal_elapsed_us = elapsed_us(proposal_started);
    let Some(proposal) = proposal else {
        return Ok(LinearProposalQueryOutcome::NoProposal);
    };
    if Instant::now() > deadline {
        config
            .source()
            .discard(
                &proposal.decision_id,
                LinearProposalDiscardReason::DeadlineExceeded,
            )
            .map_err(openai_backend_error)?;
        return Ok(LinearProposalQueryOutcome::DeadlineExceeded {
            proposal_elapsed_us,
        });
    }
    if proposal.token_ids.is_empty() || proposal.token_ids.len() > max_proposal_tokens {
        config
            .source()
            .discard(
                &proposal.decision_id,
                LinearProposalDiscardReason::InvalidTokenCount,
            )
            .map_err(openai_backend_error)?;
        return Ok(LinearProposalQueryOutcome::NoProposal);
    }
    if proposal.token_ids.iter().any(|token| *token < 0) {
        config
            .source()
            .discard(
                &proposal.decision_id,
                LinearProposalDiscardReason::InvalidTokenId,
            )
            .map_err(openai_backend_error)?;
        return Ok(LinearProposalQueryOutcome::NoProposal);
    }
    Ok(LinearProposalQueryOutcome::Ready(QueriedLinearProposal {
        proposal,
        proposal_elapsed_us,
        operation_started,
    }))
}

pub(crate) fn execute_linear_proposal_with_terminal_discard<T>(
    config: &LinearProposalIngressConfig,
    decision_id: &OpaqueProposalDecisionId,
    execute: impl FnOnce() -> OpenAiResult<T>,
) -> OpenAiResult<T> {
    match execute() {
        Ok(value) => Ok(value),
        Err(primary_error) => {
            if config
                .source()
                .discard(decision_id, LinearProposalDiscardReason::ExecutionFailed)
                .is_err()
            {
                eprintln!(
                    "linear proposal terminal discard failed; preserving the primary execution error"
                );
            }
            Err(primary_error)
        }
    }
}

pub(crate) fn report_linear_proposal_receipt(
    config: &LinearProposalIngressConfig,
    receipt: &LinearProposalReceipt,
) -> Option<anyhow::Error> {
    config.source().report(receipt).err()
}

pub(crate) fn greedy_linear_proposal_admitted(
    sampling: &SamplingConfig,
    chat_sampling_metadata: Option<&str>,
) -> bool {
    let greedy_equivalent = !sampling.enabled
        || (sampling.temperature <= 0.0
            && sampling.presence_penalty == 0.0
            && sampling.frequency_penalty == 0.0
            && sampling.repeat_penalty == 1.0
            && sampling.logit_bias.is_empty());
    if !greedy_equivalent {
        return false;
    }
    match chat_sampling_metadata {
        None => true,
        Some(metadata) => serde_json::from_str::<serde_json::Value>(metadata).is_ok(),
    }
}

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

impl StageOpenAiBackend {
    pub(crate) fn execute_local_linear_proposal(
        &self,
        params: LinearProposalExecutionParams<'_>,
        queried: QueriedLinearProposal,
        on_token: &mut impl FnMut(i32) -> OpenAiResult<TokenControl>,
    ) -> OpenAiResult<Option<LinearProposalReceipt>> {
        let proposal_token_count = queried.proposal.token_ids.len();
        let mut verify_inputs = Vec::with_capacity(proposal_token_count.saturating_add(1));
        verify_inputs.push(params.current);
        verify_inputs.extend_from_slice(&queried.proposal.token_ids);

        let Some(execution) = self.execute_local_linear_proposal_inner(
            params,
            &queried.proposal.token_ids,
            &verify_inputs,
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
        let total_elapsed_us = elapsed_us(queried.operation_started);
        Ok(Some(LinearProposalReceipt {
            decision_id: queried.proposal.decision_id,
            disposition,
            proposal_token_count,
            verification_rows: verify_inputs.len(),
            accepted_proposal_tokens,
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
        on_token: &mut impl FnMut(i32) -> OpenAiResult<TokenControl>,
    ) -> OpenAiResult<Option<LinearProposalExecution>> {
        let verify_timer = Instant::now();
        let verify_lock_timer = Instant::now();
        let mut runtime = self
            .runtime
            .lock()
            .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
        let verify_lock_wait_us = elapsed_us(verify_lock_timer);
        let verify_hold_timer = Instant::now();
        let observed_position = runtime
            .session_token_count(params.session_id)
            .ok_or_else(|| OpenAiError::backend("linear proposal session is not active"))?;
        if observed_position != params.base_position {
            return Ok(None);
        }
        let predictions = runtime
            .verify_tokens_sampled(
                params.session_id,
                verify_inputs,
                params.sampling.enabled.then_some(params.sampling),
            )
            .map_err(openai_backend_error)?;
        let decision = classify_native_mtp_verify_window(
            proposal_tokens,
            &predictions,
            params.generated_len,
            params.max_new_tokens,
            |token| {
                runtime
                    .model
                    .token_is_eog(token)
                    .map_err(openai_backend_error)
            },
        )?;
        let position_after_verification = runtime
            .session_token_count(params.session_id)
            .ok_or_else(|| OpenAiError::backend("linear proposal session disappeared"))?;
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
        let verify_lock_hold_us = elapsed_us(verify_hold_timer);
        drop(runtime);
        let verification_elapsed_us = elapsed_us(verify_timer);

        let mut committed_tokens = Vec::with_capacity(decision.commit_count);
        let mut reached_stop = false;
        let mut callback_error = None;
        for token in predictions.iter().copied().take(decision.commit_count) {
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
        if committed_tokens.is_empty() {
            return Err(OpenAiError::backend(
                "linear proposal classifier committed no target prediction",
            ));
        }

        let canonical_position = params
            .base_position
            .checked_add(
                u64::try_from(committed_tokens.len())
                    .map_err(|_| OpenAiError::backend("committed token count exceeds u64"))?,
            )
            .ok_or_else(|| OpenAiError::backend("linear proposal canonical position overflow"))?;
        let repair = finish_linear_proposal_after_repair(callback_error, || {
            self.trim_branch_suffix_or_retire(
                params.session_id,
                params.base_position,
                verify_inputs.len(),
                canonical_position,
                position_after_verification,
                params.sampling,
                params.chat_sampling_metadata,
                params.prompt_token_count,
            )
        })?;

        Ok(Some(LinearProposalExecution {
            decision,
            predictions,
            committed_tokens,
            reached_stop,
            position_after_verification,
            canonical_position,
            verification_elapsed_us,
            repair_elapsed_us: repair.elapsed_us,
            runtime_lock_wait_us: verify_lock_wait_us.saturating_add(repair.runtime_lock_wait_us),
            runtime_lock_hold_us: verify_lock_hold_us.saturating_add(repair.runtime_lock_hold_us),
            runtime_lock_acquires: 1usize.saturating_add(repair.runtime_lock_acquires),
        }))
    }

    fn trim_branch_suffix_or_retire(
        &self,
        session_id: &str,
        checkpoint_start: u64,
        checkpoint_count: usize,
        canonical_position: u64,
        position_after_verification: u64,
        sampling: &SamplingConfig,
        chat_sampling_metadata: Option<&str>,
        prompt_token_count: usize,
    ) -> OpenAiResult<LinearProposalRepairTiming> {
        if canonical_position >= position_after_verification {
            let mut runtime = self.runtime.lock().map_err(|_| {
                OpenAiError::backend("runtime lock poisoned during verify retirement")
            })?;
            runtime
                .retire_verify_checkpoint(session_id, checkpoint_start, checkpoint_count as u64)
                .map_err(openai_backend_error)?;
            return Ok(LinearProposalRepairTiming::default());
        }

        let repair_timer = Instant::now();
        let repair_lock_timer = Instant::now();
        let mut runtime = self
            .runtime
            .lock()
            .map_err(|_| OpenAiError::backend("runtime lock poisoned during proposal repair"))?;
        let runtime_lock_wait_us = elapsed_us(repair_lock_timer);
        let repair_hold_timer = Instant::now();
        let trim_result = runtime.trim_session(session_id, canonical_position);
        let runtime_lock_hold_us = elapsed_us(repair_hold_timer);
        if let Err(error) = trim_result {
            let _ = runtime.drop_session_timed(session_id);
            return Err(OpenAiError::backend(format!(
                "linear proposal repair failed and the session was retired: {error:#}"
            )));
        }
        if let Some(metadata) = chat_sampling_metadata {
            runtime
                .configure_chat_sampling(
                    session_id,
                    metadata,
                    u64::try_from(prompt_token_count).unwrap_or(u64::MAX),
                    sampling.enabled.then_some(sampling),
                )
                .map_err(openai_backend_error)?;
        }
        let repaired_position = runtime
            .session_token_count(session_id)
            .ok_or_else(|| OpenAiError::backend("repaired linear proposal session disappeared"))?;
        if repaired_position != canonical_position {
            let _ = runtime.drop_session_timed(session_id);
            return Err(OpenAiError::backend(format!(
                "linear proposal repair position mismatch: observed {repaired_position}, expected {canonical_position}"
            )));
        }
        Ok(LinearProposalRepairTiming {
            elapsed_us: elapsed_us(repair_timer),
            runtime_lock_wait_us,
            runtime_lock_hold_us,
            runtime_lock_acquires: 1,
        })
    }
}

fn elapsed_us(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX)
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
    use std::{cell::Cell, sync::Mutex, thread};

    use super::*;

    #[derive(Debug, PartialEq, Eq)]
    struct RecordedQuery {
        request_id: u64,
        session_id: u64,
        decode_step: usize,
        committed_token_ids: Vec<i32>,
        max_proposal_tokens: usize,
    }

    #[derive(Default)]
    struct FakeIngress {
        proposal: Mutex<Option<LinearProposal>>,
        delay: Mutex<Duration>,
        discard_fails: Mutex<bool>,
        report_fails: Mutex<bool>,
        queries: Mutex<Vec<RecordedQuery>>,
        receipts: Mutex<Vec<LinearProposalReceipt>>,
        discards: Mutex<Vec<(OpaqueProposalDecisionId, LinearProposalDiscardReason)>>,
    }

    impl LinearProposalIngress for FakeIngress {
        fn propose(&self, query: LinearProposalQuery<'_>) -> Result<Option<LinearProposal>> {
            self.queries.lock().unwrap().push(RecordedQuery {
                request_id: query.request_id,
                session_id: query.session_id,
                decode_step: query.decode_step,
                committed_token_ids: query.committed_token_ids.to_vec(),
                max_proposal_tokens: query.max_proposal_tokens,
            });
            thread::sleep(*self.delay.lock().unwrap());
            Ok(self.proposal.lock().unwrap().take())
        }

        fn report(&self, receipt: &LinearProposalReceipt) -> Result<()> {
            self.receipts.lock().unwrap().push(receipt.clone());
            if *self.report_fails.lock().unwrap() {
                bail!("synthetic report failure");
            }
            Ok(())
        }

        fn discard(
            &self,
            decision_id: &OpaqueProposalDecisionId,
            reason: LinearProposalDiscardReason,
        ) -> Result<()> {
            self.discards
                .lock()
                .unwrap()
                .push((decision_id.clone(), reason));
            if *self.discard_fails.lock().unwrap() {
                bail!("synthetic terminal discard failure");
            }
            Ok(())
        }
    }

    fn decision(proposal: &[i32], predictions: &[i32]) -> NativeMtpVerifyWindowDecision {
        classify_native_mtp_verify_window(proposal, predictions, 0, 64, |_| Ok(false)).unwrap()
    }

    #[test]
    fn opaque_decision_ids_are_nonempty_and_bounded() {
        assert!(OpaqueProposalDecisionId::new(Vec::new()).is_err());
        assert!(OpaqueProposalDecisionId::new(vec![1; 64]).is_ok());
        assert!(OpaqueProposalDecisionId::new(vec![1; 65]).is_err());
    }

    #[test]
    fn ingress_config_requires_positive_bounds() {
        let source = Arc::new(FakeIngress::default());
        assert!(LinearProposalIngressConfig::new(source.clone(), Duration::ZERO, 8).is_err());
        assert!(
            LinearProposalIngressConfig::new(source.clone(), Duration::from_millis(1), 0).is_err()
        );
        assert!(
            LinearProposalIngressConfig::new(
                source.clone(),
                Duration::from_millis(1),
                MAX_LINEAR_PROPOSAL_TOKENS + 1,
            )
            .is_err()
        );
        assert!(LinearProposalIngressConfig::new(source, Duration::from_millis(1), 8).is_ok());
    }

    #[test]
    fn native_classifier_is_the_only_acceptance_authority() {
        let full = decision(&[11, 12, 13], &[11, 12, 13, 14]);
        assert_eq!(full.accepted_proposal_tokens, 3);
        assert_eq!(full.commit_count, 4);
        assert!(!full.rejected);

        for accepted in [0, 1, 2] {
            let proposal = [11, 12, 13];
            let mut predictions = [11, 12, 13, 14];
            predictions[accepted] = 99;
            let mismatch = decision(&proposal, &predictions);
            assert_eq!(mismatch.accepted_proposal_tokens, accepted);
            assert_eq!(mismatch.commit_count, accepted + 1);
            assert!(mismatch.rejected);
        }
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
    fn execution_error_discards_exactly_once_without_masking_primary_error() {
        let source = Arc::new(FakeIngress::default());
        let config =
            LinearProposalIngressConfig::new(source.clone(), Duration::from_secs(1), 4).unwrap();
        let id = OpaqueProposalDecisionId::new(vec![91]).unwrap();

        let result = execute_linear_proposal_with_terminal_discard(&config, &id, || {
            Err::<(), _>(OpenAiError::backend("synthetic execution failure"))
        });

        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("synthetic execution failure")
        );
        assert_eq!(
            source.discards.lock().unwrap().as_slice(),
            &[(id.clone(), LinearProposalDiscardReason::ExecutionFailed)]
        );

        *source.discard_fails.lock().unwrap() = true;
        let result = execute_linear_proposal_with_terminal_discard(&config, &id, || {
            Err::<(), _>(OpenAiError::backend("primary error survives"))
        });
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("primary error survives")
        );
        assert_eq!(
            source.discards.lock().unwrap().as_slice(),
            &[
                (id.clone(), LinearProposalDiscardReason::ExecutionFailed),
                (id, LinearProposalDiscardReason::ExecutionFailed),
            ]
        );
    }

    #[test]
    fn report_failure_is_observed_without_becoming_an_execution_error() {
        let source = Arc::new(FakeIngress::default());
        *source.report_fails.lock().unwrap() = true;
        let config =
            LinearProposalIngressConfig::new(source.clone(), Duration::from_secs(1), 4).unwrap();
        let receipt = LinearProposalReceipt {
            decision_id: OpaqueProposalDecisionId::new(vec![90]).unwrap(),
            disposition: LinearProposalDisposition::FullAccept,
            proposal_token_count: 1,
            verification_rows: 2,
            accepted_proposal_tokens: 1,
            committed_tokens: vec![11, 12].into_boxed_slice(),
            verification_row_predictions: vec![11, 12].into_boxed_slice(),
            canonical_prediction_count: 2,
            correction_or_boundary_token: Some(12),
            base_position: 10,
            position_after_verification: 12,
            canonical_position: 12,
            trimmed_rows: 0,
            proposal_elapsed_us: 1,
            verification_elapsed_us: 2,
            repair_elapsed_us: 0,
            total_elapsed_us: 3,
            runtime_lock_wait_us: 0,
            runtime_lock_hold_us: 2,
            runtime_lock_acquires: 1,
        };

        let error = report_linear_proposal_receipt(&config, &receipt)
            .expect("report failure should be available for logging");

        assert!(error.to_string().contains("synthetic report failure"));
        assert_eq!(source.receipts.lock().unwrap().as_slice(), &[receipt]);
        assert!(source.discards.lock().unwrap().is_empty());
    }

    #[test]
    fn greedy_admission_rejects_stochastic_sampling_but_accepts_valid_grammar_metadata() {
        let disabled = SamplingConfig::default();
        let temperature_zero = SamplingConfig {
            enabled: true,
            temperature: 0.0,
            top_p: 0.95,
            top_k: 40,
            min_p: 0.05,
            ..SamplingConfig::default()
        };
        let stochastic = SamplingConfig {
            enabled: true,
            temperature: 0.8,
            ..SamplingConfig::default()
        };
        let biased_greedy = SamplingConfig {
            enabled: true,
            temperature: 0.0,
            logit_bias: vec![skippy_runtime::LogitBias {
                token_id: 7,
                bias: 1.0,
            }],
            ..SamplingConfig::default()
        };

        assert!(greedy_linear_proposal_admitted(&disabled, None));
        assert!(greedy_linear_proposal_admitted(&disabled, Some("{}")));
        assert!(greedy_linear_proposal_admitted(
            &disabled,
            Some(r#"{"grammar":""}"#)
        ));
        assert!(greedy_linear_proposal_admitted(&temperature_zero, None));
        assert!(!greedy_linear_proposal_admitted(&stochastic, None));
        assert!(!greedy_linear_proposal_admitted(&biased_greedy, None));
        assert!(greedy_linear_proposal_admitted(
            &disabled,
            Some(r#"{"grammar":"root ::= value"}"#)
        ));
        assert!(!greedy_linear_proposal_admitted(&disabled, Some("{")));
    }

    #[test]
    fn query_passes_exact_causal_context_and_accepts_a_bounded_proposal() {
        let source = Arc::new(FakeIngress::default());
        let id = OpaqueProposalDecisionId::new(vec![1, 2, 3]).unwrap();
        *source.proposal.lock().unwrap() = Some(LinearProposal::new(id.clone(), vec![31, 32, 33]));
        let config =
            LinearProposalIngressConfig::new(source.clone(), Duration::from_secs(1), 4).unwrap();

        let LinearProposalQueryOutcome::Ready(queried) =
            query_linear_proposal(&config, 7, 8, 9, &[21, 22, 23], 5, 4).unwrap()
        else {
            panic!("bounded proposal should be ready");
        };

        assert_eq!(queried.proposal.decision_id, id);
        assert_eq!(queried.proposal.token_ids.as_ref(), &[31, 32, 33]);
        assert_eq!(
            source.queries.lock().unwrap().as_slice(),
            &[RecordedQuery {
                request_id: 7,
                session_id: 8,
                decode_step: 9,
                committed_token_ids: vec![21, 22, 23],
                max_proposal_tokens: 4,
            }]
        );
        assert!(source.discards.lock().unwrap().is_empty());
    }

    #[test]
    fn query_discards_invalid_and_late_proposals_without_verification() {
        let invalid_source = Arc::new(FakeIngress::default());
        let invalid_id = OpaqueProposalDecisionId::new(vec![4]).unwrap();
        *invalid_source.proposal.lock().unwrap() =
            Some(LinearProposal::new(invalid_id.clone(), Vec::new()));
        let invalid_config =
            LinearProposalIngressConfig::new(invalid_source.clone(), Duration::from_secs(1), 4)
                .unwrap();
        assert!(matches!(
            query_linear_proposal(&invalid_config, 1, 2, 3, &[4], 5, 4).unwrap(),
            LinearProposalQueryOutcome::NoProposal
        ));
        assert_eq!(
            invalid_source.discards.lock().unwrap().as_slice(),
            &[(invalid_id, LinearProposalDiscardReason::InvalidTokenCount)]
        );

        let late_source = Arc::new(FakeIngress::default());
        let late_id = OpaqueProposalDecisionId::new(vec![5]).unwrap();
        *late_source.proposal.lock().unwrap() =
            Some(LinearProposal::new(late_id.clone(), vec![41]));
        *late_source.delay.lock().unwrap() = Duration::from_millis(5);
        let late_config =
            LinearProposalIngressConfig::new(late_source.clone(), Duration::from_millis(1), 4)
                .unwrap();
        let LinearProposalQueryOutcome::DeadlineExceeded {
            proposal_elapsed_us,
        } = query_linear_proposal(&late_config, 1, 2, 3, &[4], 5, 4).unwrap()
        else {
            panic!("late proposal should produce deadline telemetry");
        };
        assert!(proposal_elapsed_us >= 1_000);
        assert_eq!(
            late_source.discards.lock().unwrap().as_slice(),
            &[(late_id, LinearProposalDiscardReason::DeadlineExceeded)]
        );

        let invalid_token_source = Arc::new(FakeIngress::default());
        let invalid_token_id = OpaqueProposalDecisionId::new(vec![6]).unwrap();
        *invalid_token_source.proposal.lock().unwrap() =
            Some(LinearProposal::new(invalid_token_id.clone(), vec![41, -1]));
        let invalid_token_config = LinearProposalIngressConfig::new(
            invalid_token_source.clone(),
            Duration::from_secs(1),
            4,
        )
        .unwrap();
        assert!(matches!(
            query_linear_proposal(&invalid_token_config, 1, 2, 3, &[4], 5, 4).unwrap(),
            LinearProposalQueryOutcome::NoProposal
        ));
        assert_eq!(
            invalid_token_source.discards.lock().unwrap().as_slice(),
            &[(
                invalid_token_id,
                LinearProposalDiscardReason::InvalidTokenId
            )]
        );
    }

    #[test]
    fn fake_source_preserves_exact_receipt_and_discard_identity() {
        let source = FakeIngress::default();
        let id = OpaqueProposalDecisionId::new(vec![7, 8, 9]).unwrap();
        let receipt = LinearProposalReceipt {
            decision_id: id.clone(),
            disposition: LinearProposalDisposition::FirstMismatch,
            proposal_token_count: 4,
            verification_rows: 5,
            accepted_proposal_tokens: 1,
            committed_tokens: vec![11, 42].into_boxed_slice(),
            verification_row_predictions: vec![11, 42, 43, 44, 45].into_boxed_slice(),
            canonical_prediction_count: 2,
            correction_or_boundary_token: Some(42),
            base_position: 100,
            position_after_verification: 105,
            canonical_position: 102,
            trimmed_rows: 3,
            proposal_elapsed_us: 5,
            verification_elapsed_us: 10,
            repair_elapsed_us: 2,
            total_elapsed_us: 17,
            runtime_lock_wait_us: 1,
            runtime_lock_hold_us: 9,
            runtime_lock_acquires: 2,
        };
        source.report(&receipt).unwrap();
        source
            .discard(&id, LinearProposalDiscardReason::DeadlineExceeded)
            .unwrap();

        assert_eq!(source.receipts.lock().unwrap().as_slice(), &[receipt]);
        assert_eq!(
            source.discards.lock().unwrap().as_slice(),
            &[(id, LinearProposalDiscardReason::DeadlineExceeded)]
        );
    }

    #[test]
    fn query_caps_proposals_to_the_runtime_batch_window() {
        let source = Arc::new(FakeIngress::default());
        let config =
            LinearProposalIngressConfig::new(source.clone(), Duration::from_secs(1), 32).unwrap();

        assert!(matches!(
            query_linear_proposal(&config, 1, 2, 3, &[4], 64, 7).unwrap(),
            LinearProposalQueryOutcome::NoProposal
        ));
        assert_eq!(source.queries.lock().unwrap()[0].max_proposal_tokens, 7);
    }

    #[test]
    fn receipt_telemetry_excludes_source_ids_tokens_and_error_text() {
        let secret = "private-decision-/Users/nick/prompt.txt";
        let receipt = LinearProposalReceipt {
            decision_id: OpaqueProposalDecisionId::new(secret.as_bytes().to_vec()).unwrap(),
            disposition: LinearProposalDisposition::FullAccept,
            proposal_token_count: 1,
            verification_rows: 2,
            accepted_proposal_tokens: 1,
            committed_tokens: vec![12_345, 67_890].into_boxed_slice(),
            verification_row_predictions: vec![12_345, 67_890].into_boxed_slice(),
            canonical_prediction_count: 2,
            correction_or_boundary_token: Some(67_890),
            base_position: 3,
            position_after_verification: 5,
            canonical_position: 5,
            trimmed_rows: 0,
            proposal_elapsed_us: 1,
            verification_elapsed_us: 2,
            repair_elapsed_us: 3,
            total_elapsed_us: 6,
            runtime_lock_wait_us: 1,
            runtime_lock_hold_us: 2,
            runtime_lock_acquires: 1,
        };
        let mut attrs = BTreeMap::new();

        receipt.insert_telemetry_attrs(&mut attrs);
        let encoded = serde_json::to_string(&attrs).unwrap();

        assert!(!encoded.contains(secret));
        assert!(!encoded.contains("12345"));
        assert!(!encoded.contains("67890"));
        assert!(!attrs.keys().any(|key| key.contains("decision_id")));
        assert!(!attrs.keys().any(|key| key.contains("error")));
    }
}
