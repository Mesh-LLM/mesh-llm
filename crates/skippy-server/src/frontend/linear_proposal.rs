mod execution;

pub(crate) use execution::{LinearProposalExecutionParams, elapsed_us};

use std::{
    collections::BTreeMap,
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::{Result, bail};
use openai_frontend::{OpenAiError, OpenAiResult};
use serde_json::json;
use skippy_runtime::SamplingConfig;

use crate::frontend::openai_backend_error;

const MAX_OPAQUE_DECISION_ID_BYTES: usize = 64;

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
/// The proposal path deliberately does not pass a full context buffer. The
/// bounded canonical token delta since the previous proposal boundary travels
/// with this query, so a source can update its state synchronously without a
/// separate lifecycle event or copying or hashing an arbitrarily large prompt.
#[derive(Clone, Debug)]
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
    /// prompt and every target token previously delivered to the proposal
    /// source.
    pub committed_token_count: usize,
    /// Number of target tokens already generated.
    pub decode_step: usize,
    /// Maximum proposal width Skippy will accept for this query.
    pub max_proposal_tokens: usize,
    /// Canonical target tokens emitted since the previous proposal boundary.
    ///
    /// Proposal sources apply this delta before producing the proposal. Keeping
    /// the delta on the synchronous query avoids a separate lifecycle event on
    /// the local-generation hot path.
    pub pending_token_ids: Box<[i32]>,
    /// Advisory deadline the synchronous source must honor.
    pub deadline: Instant,
}

impl LinearProposalQuery {
    /// Creates one target-authoritative proposal query.
    #[must_use]
    pub fn new(
        request_id: u64,
        session_id: u64,
        prompt_token_count: usize,
        committed_token_count: usize,
        decode_step: usize,
        max_proposal_tokens: usize,
        deadline: Instant,
    ) -> Self {
        Self {
            request_id,
            session_id,
            prompt_token_count,
            committed_token_count,
            decode_step,
            max_proposal_tokens,
            pending_token_ids: Vec::new().into_boxed_slice(),
            deadline,
        }
    }

    #[must_use]
    pub fn with_pending_token_ids(mut self, pending_token_ids: Box<[i32]>) -> Self {
        self.pending_token_ids = pending_token_ids;
        self
    }
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
    /// OpenAI request identity for the proposal's generation.
    pub request_id: u64,
    /// OpenAI session identity for the proposal's generation.
    pub session_id: u64,
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
    /// Total target tokens generated after applying this proposal outcome.
    ///
    /// Serving integrations must use this target-authoritative position to
    /// verify that ordered lifecycle commits reached the proposal boundary
    /// before reporting the outcome. It is deliberately separate from the
    /// committed token slice: the slice describes this outcome, while this
    /// count describes the complete generation prefix at the boundary.
    pub generated_token_count: usize,
    /// Target tokens committed to the response stream.
    pub committed_tokens: Box<[i32]>,
    /// Authoritative prediction prefix through the full-accept boundary or
    /// first mismatch. Rejected branch-conditioned suffixes are not sampled.
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
    #[cfg(feature = "test-support")]
    #[doc(hidden)]
    pub fn test_fixture(decision_id: OpaqueProposalDecisionId) -> Self {
        Self::test_fixture_with_generated_token_count(decision_id, 1)
    }

    #[cfg(feature = "test-support")]
    #[doc(hidden)]
    pub fn test_fixture_with_generated_token_count(
        decision_id: OpaqueProposalDecisionId,
        generated_token_count: usize,
    ) -> Self {
        Self {
            request_id: 1,
            session_id: 2,
            decision_id,
            disposition: LinearProposalDisposition::FullAccept,
            proposal_token_count: 1,
            verification_rows: 1,
            accepted_proposal_tokens: 1,
            generated_token_count,
            committed_tokens: vec![1].into_boxed_slice(),
            verification_row_predictions: vec![1].into_boxed_slice(),
            canonical_prediction_count: 1,
            correction_or_boundary_token: Some(1),
            base_position: 0,
            position_after_verification: 1,
            canonical_position: 1,
            trimmed_rows: 0,
            proposal_elapsed_us: 0,
            verification_elapsed_us: 0,
            repair_elapsed_us: 0,
            total_elapsed_us: 0,
            runtime_lock_wait_us: 0,
            runtime_lock_hold_us: 0,
            runtime_lock_acquires: 0,
        }
    }

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

/// Per-request result from an in-process linear proposal source.
///
/// The telemetry travels with the corresponding request result so a shared
/// source cannot accidentally attribute timings from one decode to another.
pub struct LinearProposalSourceResponse {
    proposal: Option<LinearProposal>,
    telemetry: Option<LinearProposalSourceTelemetry>,
}

impl LinearProposalSourceResponse {
    /// Creates a result without source-specific timing data.
    #[must_use]
    pub fn new(proposal: Option<LinearProposal>) -> Self {
        Self {
            proposal,
            telemetry: None,
        }
    }

    /// Creates a result with telemetry for this exact proposal request.
    #[must_use]
    pub fn with_telemetry(
        proposal: Option<LinearProposal>,
        telemetry: LinearProposalSourceTelemetry,
    ) -> Self {
        Self {
            proposal,
            telemetry: Some(telemetry),
        }
    }

    fn into_parts(
        self,
    ) -> (
        Option<LinearProposal>,
        Option<LinearProposalSourceTelemetry>,
    ) {
        (self.proposal, self.telemetry)
    }
}

/// In-process, source-neutral width-one proposal boundary.
///
/// Implementations must honor `query.deadline`. Skippy independently rejects a
/// proposal that arrives after it and calls `discard` so the source can resolve
/// any pending decision without treating it as verified.
pub trait LinearProposalIngress: Send + Sync {
    /// Returns an optional bounded proposal and telemetry for this exact
    /// committed query state.
    fn propose(&self, query: LinearProposalQuery) -> Result<LinearProposalSourceResponse>;

    /// Receives the target-authoritative outcome for a verified proposal.
    fn report(&self, receipt: &LinearProposalReceipt) -> Result<()>;

    /// Returns asynchronous failures observed after a report was accepted.
    fn report_delivery_failures(&self) -> u64 {
        0
    }

    /// Resolves a source decision that Skippy could not verify.
    fn discard(
        &self,
        _decision_id: &OpaqueProposalDecisionId,
        _reason: LinearProposalDiscardReason,
    ) -> Result<()> {
        Ok(())
    }
}

/// Bounded source-side outcome for one proposal query.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum LinearProposalSourceOutcome {
    /// The source returned before the configured deadline.
    Ready,
    /// The source deliberately returned no proposal before the deadline.
    Abstained,
    /// The host stopped waiting at the configured wall-clock deadline.
    HostDeadlineExceeded,
    /// The host queue was at capacity, so the source was never submitted.
    QueueFull,
    /// The worker established that the deadline had elapsed before dispatch.
    DeadlineExceededBeforeDispatch,
    /// A plugin callback exhausted the deadline without producing a candidate.
    DeadlineExceededInPlugin,
    /// A plugin callback returned a candidate after the deadline.
    CandidateReturnedTooLate,
    /// A plugin callback failed and was treated as a fail-open abstention.
    SourceError,
}

impl LinearProposalSourceOutcome {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::Abstained => "abstained",
            Self::HostDeadlineExceeded => "host_deadline_exceeded",
            Self::QueueFull => "queue_full",
            Self::DeadlineExceededBeforeDispatch => "deadline_exceeded_before_dispatch",
            Self::DeadlineExceededInPlugin => "deadline_exceeded_in_plugin",
            Self::CandidateReturnedTooLate => "candidate_returned_too_late",
            Self::SourceError => "source_error",
        }
    }
}

/// Privacy-safe source timing for one proposal query.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LinearProposalSourceTelemetry {
    /// Time from source submission until the worker began dispatch.
    pub queue_wait_us: u64,
    /// Time spent crossing the source's callback boundary.
    pub callback_elapsed_us: u64,
    /// Bounded completion or abstention outcome.
    pub outcome: LinearProposalSourceOutcome,
}

impl LinearProposalSourceTelemetry {
    pub(crate) fn insert_telemetry_attrs(self, attrs: &mut BTreeMap<String, serde_json::Value>) {
        attrs.insert(
            "llama_stage.linear_proposal.source_queue_wait_us".to_string(),
            json!(self.queue_wait_us),
        );
        attrs.insert(
            "llama_stage.linear_proposal.source_callback_us".to_string(),
            json!(self.callback_elapsed_us),
        );
        attrs.insert(
            "llama_stage.linear_proposal.source_outcome".to_string(),
            json!(self.outcome.as_str()),
        );
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

    /// Returns report callback failures observed after enqueue-time success.
    pub fn report_delivery_failures(&self) -> u64 {
        self.source.report_delivery_failures()
    }
}

pub(crate) struct QueriedLinearProposal {
    pub(crate) proposal: LinearProposal,
    pub(crate) proposal_elapsed_us: u64,
    pub(crate) operation_started: Instant,
    pub(crate) source_telemetry: Option<LinearProposalSourceTelemetry>,
}

pub(crate) enum LinearProposalQueryOutcome {
    /// No proposal command was submitted because the bounded query was not
    /// admissible at this boundary.
    Skipped,
    NoProposal {
        source_telemetry: Option<LinearProposalSourceTelemetry>,
    },
    DeadlineExceeded {
        proposal_elapsed_us: u64,
        source_telemetry: Option<LinearProposalSourceTelemetry>,
    },
    Ready(QueriedLinearProposal),
}

#[derive(Clone)]
pub(crate) struct LinearProposalQueryParams {
    pub(crate) request_id: u64,
    pub(crate) session_id: u64,
    pub(crate) prompt_token_count: usize,
    pub(crate) decode_step: usize,
    pub(crate) committed_token_count: usize,
    pub(crate) remaining_new_tokens: usize,
    pub(crate) runtime_max_proposal_tokens: usize,
    pub(crate) pending_token_ids: Box<[i32]>,
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
        return Ok(LinearProposalQueryOutcome::Skipped);
    }
    let operation_started = Instant::now();
    let deadline = operation_started
        .checked_add(config.deadline())
        .ok_or_else(|| OpenAiError::backend("linear proposal deadline overflow"))?;
    let proposal_started = Instant::now();
    let response = config
        .source()
        .propose(
            LinearProposalQuery::new(
                params.request_id,
                params.session_id,
                params.prompt_token_count,
                params.committed_token_count,
                params.decode_step,
                max_proposal_tokens,
                deadline,
            )
            .with_pending_token_ids(params.pending_token_ids),
        )
        .map_err(openai_backend_error)?;
    let proposal_elapsed_us = elapsed_us(proposal_started);
    let (proposal, source_telemetry) = response.into_parts();
    let Some(proposal) = proposal else {
        return Ok(LinearProposalQueryOutcome::NoProposal { source_telemetry });
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
            source_telemetry,
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
        return Ok(LinearProposalQueryOutcome::NoProposal { source_telemetry });
    }
    if proposal.token_ids.iter().any(|token| *token < 0) {
        config
            .source()
            .discard(
                &proposal.decision_id,
                LinearProposalDiscardReason::InvalidTokenId,
            )
            .map_err(openai_backend_error)?;
        return Ok(LinearProposalQueryOutcome::NoProposal { source_telemetry });
    }
    Ok(LinearProposalQueryOutcome::Ready(QueriedLinearProposal {
        proposal,
        proposal_elapsed_us,
        operation_started,
        source_telemetry,
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

#[cfg(test)]
mod tests {
    use std::{sync::Mutex, thread};

    use super::*;
    use crate::frontend::{NativeMtpVerifyWindowDecision, classify_native_mtp_verify_window};

    #[derive(Debug, PartialEq, Eq)]
    struct RecordedQuery {
        request_id: u64,
        session_id: u64,
        prompt_token_count: usize,
        committed_token_count: usize,
        decode_step: usize,
        max_proposal_tokens: usize,
        pending_token_ids: Box<[i32]>,
    }

    fn query_params(
        request_id: u64,
        session_id: u64,
        prompt_token_count: usize,
        decode_step: usize,
        committed_token_count: usize,
        remaining_new_tokens: usize,
        runtime_max_proposal_tokens: usize,
    ) -> LinearProposalQueryParams {
        LinearProposalQueryParams {
            request_id,
            session_id,
            prompt_token_count,
            decode_step,
            committed_token_count,
            remaining_new_tokens,
            runtime_max_proposal_tokens,
            pending_token_ids: Vec::new().into_boxed_slice(),
        }
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
        fn propose(&self, query: LinearProposalQuery) -> Result<LinearProposalSourceResponse> {
            self.queries.lock().unwrap().push(RecordedQuery {
                request_id: query.request_id,
                session_id: query.session_id,
                prompt_token_count: query.prompt_token_count,
                committed_token_count: query.committed_token_count,
                decode_step: query.decode_step,
                max_proposal_tokens: query.max_proposal_tokens,
                pending_token_ids: query.pending_token_ids,
            });
            thread::sleep(*self.delay.lock().unwrap());
            Ok(LinearProposalSourceResponse::new(
                self.proposal.lock().unwrap().take(),
            ))
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
            request_id: 1,
            session_id: 2,
            decision_id: OpaqueProposalDecisionId::new(vec![90]).unwrap(),
            disposition: LinearProposalDisposition::FullAccept,
            proposal_token_count: 1,
            verification_rows: 2,
            accepted_proposal_tokens: 1,
            generated_token_count: 2,
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
    fn query_passes_bounded_committed_position_and_accepts_a_bounded_proposal() {
        let source = Arc::new(FakeIngress::default());
        let id = OpaqueProposalDecisionId::new(vec![1, 2, 3]).unwrap();
        *source.proposal.lock().unwrap() = Some(LinearProposal::new(id.clone(), vec![31, 32, 33]));
        let config =
            LinearProposalIngressConfig::new(source.clone(), Duration::from_secs(1), 4).unwrap();

        let LinearProposalQueryOutcome::Ready(queried) =
            query_linear_proposal(&config, query_params(7, 8, 2, 1, 3, 5, 4)).unwrap()
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
                prompt_token_count: 2,
                decode_step: 1,
                committed_token_count: 3,
                max_proposal_tokens: 4,
                pending_token_ids: Vec::new().into_boxed_slice(),
            }]
        );
        assert!(source.discards.lock().unwrap().is_empty());
    }

    #[test]
    fn query_forwards_pending_tokens_to_the_proposal_source() {
        let source = Arc::new(FakeIngress::default());
        let id = OpaqueProposalDecisionId::new(vec![7]).unwrap();
        *source.proposal.lock().unwrap() = Some(LinearProposal::new(id, vec![41]));
        let config =
            LinearProposalIngressConfig::new(source.clone(), Duration::from_secs(1), 4).unwrap();
        let mut params = query_params(7, 8, 2, 1, 3, 5, 4);
        params.pending_token_ids = vec![31, 32].into_boxed_slice();

        query_linear_proposal(&config, params).expect("proposal query should succeed");

        assert_eq!(
            source.queries.lock().unwrap().as_slice(),
            &[RecordedQuery {
                request_id: 7,
                session_id: 8,
                prompt_token_count: 2,
                committed_token_count: 3,
                decode_step: 1,
                max_proposal_tokens: 4,
                pending_token_ids: vec![31, 32].into_boxed_slice(),
            }]
        );
    }

    #[test]
    fn query_rejects_an_inconsistent_prompt_decode_boundary_before_ingress() {
        let source = Arc::new(FakeIngress::default());
        let config =
            LinearProposalIngressConfig::new(source.clone(), Duration::from_secs(1), 4).unwrap();

        for (prompt_token_count, decode_step) in [(0, 3), (4, 0), (2, 0), (1, 3)] {
            assert!(
                query_linear_proposal(
                    &config,
                    query_params(7, 8, prompt_token_count, decode_step, 3, 5, 4),
                )
                .is_err()
            );
        }
        assert!(source.queries.lock().unwrap().is_empty());
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
            query_linear_proposal(&invalid_config, query_params(1, 2, 1, 0, 1, 5, 4)).unwrap(),
            LinearProposalQueryOutcome::NoProposal { .. }
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
            ..
        } = query_linear_proposal(&late_config, query_params(1, 2, 1, 0, 1, 5, 4)).unwrap()
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
            query_linear_proposal(&invalid_token_config, query_params(1, 2, 1, 0, 1, 5, 4),)
                .unwrap(),
            LinearProposalQueryOutcome::NoProposal { .. }
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
            request_id: 1,
            session_id: 2,
            decision_id: id.clone(),
            disposition: LinearProposalDisposition::FirstMismatch,
            proposal_token_count: 4,
            verification_rows: 5,
            accepted_proposal_tokens: 1,
            generated_token_count: 2,
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
            query_linear_proposal(&config, query_params(1, 2, 1, 0, 1, 64, 7)).unwrap(),
            LinearProposalQueryOutcome::NoProposal { .. }
        ));
        assert_eq!(source.queries.lock().unwrap()[0].max_proposal_tokens, 7);
    }

    #[test]
    fn receipt_telemetry_excludes_source_ids_tokens_and_error_text() {
        let secret = "private-decision-/Users/nick/prompt.txt";
        let receipt = LinearProposalReceipt {
            request_id: 1,
            session_id: 2,
            decision_id: OpaqueProposalDecisionId::new(secret.as_bytes().to_vec()).unwrap(),
            disposition: LinearProposalDisposition::FullAccept,
            proposal_token_count: 1,
            verification_rows: 2,
            accepted_proposal_tokens: 1,
            generated_token_count: 2,
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
        LinearProposalSourceTelemetry {
            queue_wait_us: 7,
            callback_elapsed_us: 11,
            outcome: LinearProposalSourceOutcome::DeadlineExceededBeforeDispatch,
        }
        .insert_telemetry_attrs(&mut attrs);
        let encoded = serde_json::to_string(&attrs).unwrap();

        assert!(!encoded.contains(secret));
        assert!(!encoded.contains("12345"));
        assert!(!encoded.contains("67890"));
        assert!(!attrs.keys().any(|key| key.contains("decision_id")));
        assert!(!attrs.keys().any(|key| key.contains("error")));
        assert_eq!(
            attrs["llama_stage.linear_proposal.source_outcome"],
            json!("deadline_exceeded_before_dispatch")
        );
    }
}
