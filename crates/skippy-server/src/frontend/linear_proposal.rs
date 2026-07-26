use std::{
    collections::BTreeMap,
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::{Result, bail};
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use openai_frontend::{OpenAiError, OpenAiResult};
use serde_json::json;

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

    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

/// One source-selected linear proposal. The API is width one by construction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinearProposal {
    pub decision_id: OpaqueProposalDecisionId,
    pub token_ids: Box<[i32]>,
}

impl LinearProposal {
    pub fn new(decision_id: OpaqueProposalDecisionId, token_ids: impl Into<Vec<i32>>) -> Self {
        Self {
            decision_id,
            token_ids: token_ids.into().into_boxed_slice(),
        }
    }
}

/// Causal, committed-only state supplied to a proposal source.
#[derive(Clone, Copy, Debug)]
pub struct LinearProposalQuery<'a> {
    pub request_id: u64,
    pub session_id: u64,
    pub decode_step: usize,
    pub committed_token_ids: &'a [i32],
    pub max_proposal_tokens: usize,
    pub deadline: Instant,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinearProposalDiscardReason {
    DeadlineExceeded,
    InvalidTokenCount,
    InvalidTokenId,
    ExecutionFailed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinearProposalDisposition {
    FullAccept,
    FirstMismatch,
    Stopped,
}

/// Skippy-owned outcome for one verified linear proposal.
///
/// `committed_tokens` is the target-authoritative stream prefix. Predictions
/// after `canonical_prediction_count` are branch-conditioned observations and
/// must not be interpreted as future canonical target tokens after a mismatch.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearProposalReceipt {
    pub decision_id: OpaqueProposalDecisionId,
    pub disposition: LinearProposalDisposition,
    pub proposal_token_count: usize,
    pub verification_rows: usize,
    pub accepted_proposal_tokens: usize,
    pub committed_tokens: Box<[i32]>,
    pub verification_row_predictions: Box<[i32]>,
    pub canonical_prediction_count: usize,
    pub correction_or_boundary_token: Option<i32>,
    pub base_position: u64,
    pub position_after_verification: u64,
    pub canonical_position: u64,
    pub trimmed_rows: usize,
    pub proposal_elapsed_us: u64,
    pub verification_elapsed_us: u64,
    pub repair_elapsed_us: u64,
    pub total_elapsed_us: u64,
    pub runtime_lock_wait_us: u64,
    pub runtime_lock_hold_us: u64,
    pub runtime_lock_acquires: usize,
}

impl LinearProposalReceipt {
    pub(crate) fn insert_telemetry_attrs(&self, attrs: &mut BTreeMap<String, serde_json::Value>) {
        attrs.insert(
            "llama_stage.linear_proposal.decision_id".to_string(),
            json!(URL_SAFE_NO_PAD.encode(self.decision_id.as_bytes())),
        );
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
    fn propose(&self, query: LinearProposalQuery<'_>) -> Result<Option<LinearProposal>>;

    fn report(&self, receipt: &LinearProposalReceipt) -> Result<()>;

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

    pub fn deadline(&self) -> Duration {
        self.deadline
    }

    pub fn max_proposal_tokens(&self) -> usize {
        self.max_proposal_tokens
    }

    pub fn source(&self) -> &Arc<dyn LinearProposalIngress> {
        &self.source
    }
}

pub(crate) struct QueriedLinearProposal {
    pub(crate) proposal: LinearProposal,
    pub(crate) proposal_elapsed_us: u64,
    pub(crate) operation_started: Instant,
}

pub(crate) fn query_linear_proposal(
    config: &LinearProposalIngressConfig,
    request_id: u64,
    session_id: u64,
    decode_step: usize,
    committed_token_ids: &[i32],
    remaining_new_tokens: usize,
) -> OpenAiResult<Option<QueriedLinearProposal>> {
    let max_proposal_tokens = remaining_new_tokens
        .saturating_sub(1)
        .min(config.max_proposal_tokens());
    if max_proposal_tokens == 0 {
        return Ok(None);
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
        return Ok(None);
    };
    if Instant::now() > deadline {
        config
            .source()
            .discard(
                &proposal.decision_id,
                LinearProposalDiscardReason::DeadlineExceeded,
            )
            .map_err(openai_backend_error)?;
        return Ok(None);
    }
    if proposal.token_ids.is_empty() || proposal.token_ids.len() > max_proposal_tokens {
        config
            .source()
            .discard(
                &proposal.decision_id,
                LinearProposalDiscardReason::InvalidTokenCount,
            )
            .map_err(openai_backend_error)?;
        return Ok(None);
    }
    if proposal.token_ids.iter().any(|token| *token < 0) {
        config
            .source()
            .discard(
                &proposal.decision_id,
                LinearProposalDiscardReason::InvalidTokenId,
            )
            .map_err(openai_backend_error)?;
        return Ok(None);
    }
    Ok(Some(QueriedLinearProposal {
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

pub(crate) fn greedy_linear_proposal_admitted(
    sampling_enabled: bool,
    chat_sampling_metadata: Option<&str>,
) -> bool {
    if sampling_enabled {
        return false;
    }
    match chat_sampling_metadata {
        None => true,
        Some(metadata) => serde_json::from_str::<serde_json::Value>(metadata)
            .ok()
            .is_some_and(|value| {
                value
                    .get("grammar")
                    .and_then(serde_json::Value::as_str)
                    .is_none_or(str::is_empty)
            }),
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

impl StageOpenAiBackend {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_local_linear_proposal(
        &self,
        session_id: &str,
        current: i32,
        base_position: u64,
        generated_len: usize,
        max_new_tokens: usize,
        queried: QueriedLinearProposal,
        on_token: &mut impl FnMut(i32) -> OpenAiResult<TokenControl>,
    ) -> OpenAiResult<LinearProposalReceipt> {
        let proposal_token_count = queried.proposal.token_ids.len();
        let mut verify_inputs = Vec::with_capacity(proposal_token_count.saturating_add(1));
        verify_inputs.push(current);
        verify_inputs.extend_from_slice(&queried.proposal.token_ids);

        let execution = self.execute_local_linear_proposal_inner(
            session_id,
            base_position,
            generated_len,
            max_new_tokens,
            &queried.proposal.token_ids,
            &verify_inputs,
            on_token,
        )?;
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
        Ok(LinearProposalReceipt {
            decision_id: queried.proposal.decision_id,
            disposition,
            proposal_token_count,
            verification_rows: verify_inputs.len(),
            accepted_proposal_tokens,
            canonical_prediction_count: execution.committed_tokens.len(),
            committed_tokens: execution.committed_tokens.into_boxed_slice(),
            verification_row_predictions: execution.predictions.into_boxed_slice(),
            correction_or_boundary_token,
            base_position,
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
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_local_linear_proposal_inner(
        &self,
        session_id: &str,
        base_position: u64,
        generated_len: usize,
        max_new_tokens: usize,
        proposal_tokens: &[i32],
        verify_inputs: &[i32],
        on_token: &mut impl FnMut(i32) -> OpenAiResult<TokenControl>,
    ) -> OpenAiResult<LinearProposalExecution> {
        let verify_timer = Instant::now();
        let verify_lock_timer = Instant::now();
        let mut runtime = self
            .runtime
            .lock()
            .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
        let verify_lock_wait_us = elapsed_us(verify_lock_timer);
        let verify_hold_timer = Instant::now();
        let observed_position = runtime
            .session_token_count(session_id)
            .ok_or_else(|| OpenAiError::backend("linear proposal session is not active"))?;
        if observed_position != base_position {
            return Err(OpenAiError::backend(format!(
                "linear proposal session position mismatch: observed {observed_position}, expected {base_position}"
            )));
        }
        let predictions = runtime
            .verify_tokens(session_id, verify_inputs)
            .map_err(openai_backend_error)?;
        let decision = classify_native_mtp_verify_window(
            proposal_tokens,
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
            .session_token_count(session_id)
            .ok_or_else(|| OpenAiError::backend("linear proposal session disappeared"))?;
        let expected_position_after_verification = base_position
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

        let canonical_position = base_position
            .checked_add(
                u64::try_from(committed_tokens.len())
                    .map_err(|_| OpenAiError::backend("committed token count exceeds u64"))?,
            )
            .ok_or_else(|| OpenAiError::backend("linear proposal canonical position overflow"))?;
        let mut repair_elapsed_us = 0;
        let mut repair_lock_wait_us = 0;
        let mut repair_lock_hold_us = 0;
        let mut runtime_lock_acquires = 1;
        finish_linear_proposal_after_repair(callback_error, || {
            if canonical_position < position_after_verification {
                let repair_timer = Instant::now();
                let repair_lock_timer = Instant::now();
                let mut runtime = self.runtime.lock().map_err(|_| {
                    OpenAiError::backend("runtime lock poisoned during proposal repair")
                })?;
                repair_lock_wait_us = elapsed_us(repair_lock_timer);
                let repair_hold_timer = Instant::now();
                let trim_result = runtime.trim_session(session_id, canonical_position);
                repair_lock_hold_us = elapsed_us(repair_hold_timer);
                runtime_lock_acquires += 1;
                if let Err(error) = trim_result {
                    let _ = runtime.drop_session_timed(session_id);
                    return Err(OpenAiError::backend(format!(
                        "linear proposal repair failed and the session was retired: {error:#}"
                    )));
                }
                let repaired_position =
                    runtime.session_token_count(session_id).ok_or_else(|| {
                        OpenAiError::backend("repaired linear proposal session disappeared")
                    })?;
                if repaired_position != canonical_position {
                    let _ = runtime.drop_session_timed(session_id);
                    return Err(OpenAiError::backend(format!(
                        "linear proposal repair position mismatch: observed {repaired_position}, expected {canonical_position}"
                    )));
                }
                repair_elapsed_us = elapsed_us(repair_timer);
            }
            Ok(())
        })?;

        Ok(LinearProposalExecution {
            decision,
            predictions,
            committed_tokens,
            reached_stop,
            position_after_verification,
            canonical_position,
            verification_elapsed_us,
            repair_elapsed_us,
            runtime_lock_wait_us: verify_lock_wait_us.saturating_add(repair_lock_wait_us),
            runtime_lock_hold_us: verify_lock_hold_us.saturating_add(repair_lock_hold_us),
            runtime_lock_acquires,
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

fn finish_linear_proposal_after_repair(
    callback_error: Option<OpenAiError>,
    repair: impl FnOnce() -> OpenAiResult<()>,
) -> OpenAiResult<()> {
    repair()?;
    callback_error.map_or(Ok(()), Err)
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
    fn greedy_admission_rejects_sampling_and_grammar() {
        assert!(greedy_linear_proposal_admitted(false, None));
        assert!(greedy_linear_proposal_admitted(false, Some("{}")));
        assert!(greedy_linear_proposal_admitted(
            false,
            Some(r#"{"grammar":""}"#)
        ));
        assert!(!greedy_linear_proposal_admitted(true, None));
        assert!(!greedy_linear_proposal_admitted(
            false,
            Some(r#"{"grammar":"root ::= value"}"#)
        ));
        assert!(!greedy_linear_proposal_admitted(false, Some("{")));
    }

    #[test]
    fn query_passes_exact_causal_context_and_accepts_a_bounded_proposal() {
        let source = Arc::new(FakeIngress::default());
        let id = OpaqueProposalDecisionId::new(vec![1, 2, 3]).unwrap();
        *source.proposal.lock().unwrap() = Some(LinearProposal::new(id.clone(), vec![31, 32, 33]));
        let config =
            LinearProposalIngressConfig::new(source.clone(), Duration::from_secs(1), 4).unwrap();

        let queried = query_linear_proposal(&config, 7, 8, 9, &[21, 22, 23], 5)
            .unwrap()
            .unwrap();

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
        assert!(
            query_linear_proposal(&invalid_config, 1, 2, 3, &[4], 5)
                .unwrap()
                .is_none()
        );
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
        assert!(
            query_linear_proposal(&late_config, 1, 2, 3, &[4], 5)
                .unwrap()
                .is_none()
        );
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
        assert!(
            query_linear_proposal(&invalid_token_config, 1, 2, 3, &[4], 5)
                .unwrap()
                .is_none()
        );
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
}
