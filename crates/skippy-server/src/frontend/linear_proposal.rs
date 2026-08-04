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

/// Bounded, target-authoritative state supplied to a proposal source.
///
/// The proposal path deliberately does not pass a full context buffer. A
/// lifecycle observer receives canonical committed token deltas separately;
/// the request path carries only O(1) identity and position data so a source
/// can honor its absolute deadline without copying or hashing an arbitrarily
/// large prompt.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct LinearProposalQuery {
    /// OpenAI request identity.
    pub request_id: u64,
    /// OpenAI session identity.
    pub session_id: u64,
    /// Number of leading committed tokens supplied by the original prompt.
    ///
    /// This is target-authoritative lifecycle information. Proposal sources
    /// use it to separate the immutable request prompt from generated tokens
    /// that may already have been committed during prefill.
    pub prompt_token_count: usize,
    /// Total canonical tokens committed at this query boundary, including the
    /// prompt and every target token previously delivered to the lifecycle
    /// observer.
    pub committed_token_count: usize,
    /// Number of target tokens already generated.
    pub decode_step: usize,
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
    fn propose(&self, query: LinearProposalQuery) -> Result<Option<LinearProposal>>;

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

#[derive(Clone, Copy)]
pub(crate) struct LinearProposalQueryParams {
    pub(crate) request_id: u64,
    pub(crate) session_id: u64,
    pub(crate) prompt_token_count: usize,
    pub(crate) decode_step: usize,
    pub(crate) committed_token_count: usize,
    pub(crate) remaining_new_tokens: usize,
    pub(crate) runtime_max_proposal_tokens: usize,
}

pub(crate) fn query_linear_proposal(
    config: &LinearProposalIngressConfig,
    params: LinearProposalQueryParams,
) -> OpenAiResult<LinearProposalQueryOutcome> {
    if params.prompt_token_count == 0
        || params.committed_token_count
            != params.prompt_token_count.saturating_add(params.decode_step)
    {
        return Err(OpenAiError::backend(
            "linear proposal query does not match the authoritative prompt/decode boundary",
        ));
    }
    let max_proposal_tokens = params
        .remaining_new_tokens
        .saturating_sub(1)
        .min(params.runtime_max_proposal_tokens)
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
            request_id: params.request_id,
            session_id: params.session_id,
            prompt_token_count: params.prompt_token_count,
            committed_token_count: params.committed_token_count,
            decode_step: params.decode_step,
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
        canonical_position: u64,
        position_after_verification: u64,
        sampling: &SamplingConfig,
        chat_sampling_metadata: Option<&str>,
        prompt_token_count: usize,
    ) -> OpenAiResult<LinearProposalRepairTiming> {
        if canonical_position >= position_after_verification {
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
#[path = "linear_proposal_tests.rs"]
mod tests;
