//! One admitted decision sequence and its downstream connection lease.

use super::*;
use crate::binary_transport::{forwarded_stage_message, write_stage_message_conditioned};
use crate::frontend::generation::{
    InferenceBackendMode, PersistentStageLane, PersistentStageLanePool,
};
use crate::frontend::iteration_scheduler::readout::ReadoutIteration;
use crate::frontend::util::openai_io_error;
use crate::frontend::wire_messages::{OpenAiPrefillChunk, embedded_prefill_message};
use skippy_protocol::binary::{
    StageStateHeader, StageWireMessage, WireMessageKind, WireReplyKind, recv_reply,
};
use std::net::Shutdown;

pub(super) struct DecisionSequence<'a> {
    pub backend: &'a StageOpenAiBackend,
    pub ids: &'a InferenceGenerationIds,
    pub cancellation: &'a CancellationToken,
    pub session: String,
    pub lane: Option<(Arc<PersistentStageLanePool>, PersistentStageLane)>,
    pub position: usize,
    pub evaluated: usize,
    pub cached: usize,
    prompt_tokens: usize,
    closed: bool,
}

/// Execute one readout sequence and release its local and downstream sessions.
pub(super) fn execute(
    backend: &StageOpenAiBackend,
    job: &ReadoutJob,
    ids: &InferenceGenerationIds,
    cancellation: &CancellationToken,
) -> Result<ReadoutResult, ReadoutFailure> {
    let mut sequence = DecisionSequence::new(backend, ids, cancellation)?;
    let result = sequence.score(job).and_then(|scores| {
        sequence.close()?;
        Ok(scores)
    });
    result
        .map(|scores| ReadoutResult {
            scores,
            evaluated_tokens: sequence.evaluated,
            cached_tokens: sequence.cached,
        })
        .map_err(|error| ReadoutFailure {
            error: Box::new(error),
            evaluated_tokens: sequence.evaluated,
            cached_tokens: sequence.cached,
        })
}

impl<'a> DecisionSequence<'a> {
    /// Create an independent sequence and acquire its downstream lane when split.
    fn new(
        backend: &'a StageOpenAiBackend,
        ids: &'a InferenceGenerationIds,
        cancellation: &'a CancellationToken,
    ) -> InferenceResult<Self> {
        let lane = match &backend.mode {
            InferenceBackendMode::EmbeddedStageZero {
                config, lane_pool, ..
            } if config.downstream.is_some() => {
                let pool = lane_pool.as_ref().ok_or_else(|| {
                    InferenceError::unsupported("decision readout requires a connected stage chain")
                })?;
                Some((pool.clone(), pool.checkout(ids)?))
            }
            _ => None,
        };
        Ok(Self {
            backend,
            ids,
            cancellation,
            session: ids.session_id_string(),
            lane,
            position: 0,
            evaluated: 0,
            cached: 0,
            prompt_tokens: 0,
            closed: false,
        })
    }

    /// Score selected tokens or teacher-force a candidate without sampling an answer.
    fn score(&mut self, job: &ReadoutJob) -> InferenceResult<Vec<f64>> {
        if job.prompt.is_empty() {
            return Err(InferenceError::invalid_request("empty decision prompt"));
        }
        self.prompt_tokens = job.prompt.len();
        // Always evaluate the last row: a KV hit does not imply cached logits.
        self.restore_prefix(&job.prompt[..job.prompt.len() - 1])?;
        let first = job
            .candidates
            .iter()
            .map(|candidate| candidate[0])
            .collect::<Vec<_>>();
        let mut scores = Vec::new();
        let chunk_limit = self.backend.iteration_scheduler.readout_chunk_limit();
        while self.position < job.prompt.len() {
            let end = (self.position + chunk_limit).min(job.prompt.len());
            let selection = if end == job.prompt.len() {
                first.as_slice()
            } else {
                &[]
            };
            scores = self.step(&job.prompt[self.position..end], selection)?;
        }
        self.backend
            .record_embedded_stage0_full_prefill(&self.session, self.ids, &job.prompt)?;
        if job.method == Some(ReadoutMethod::CandidateLogLikelihood) {
            let candidate = &job.candidates[0];
            for tokens in candidate.windows(2) {
                scores[0] += self.step(&tokens[..1], &tokens[1..])?[0];
            }
        }
        Ok(scores)
    }

    /// Run one bounded prefill/readout step and account for evaluated tokens.
    fn step(&mut self, tokens: &[i32], selected: &[i32]) -> InferenceResult<Vec<f64>> {
        if self.cancellation.is_cancelled() {
            return Err(InferenceError::cancelled("decision request cancelled"));
        }
        let mut outputs =
            self.backend
                .iteration_scheduler
                .execute_readout_wave(vec![ReadoutIteration {
                    session_id: &self.session,
                    position: self.position as u64,
                    tokens,
                    selected_tokens: selected,
                    input: None,
                    cancellation: Some(self.cancellation.clone()),
                }])?;
        let output = outputs
            .pop()
            .ok_or_else(|| InferenceError::backend("missing decision iteration"))?;
        // The local evaluation completed even if forwarding or validation fails.
        self.evaluated += tokens.len();
        let scores = if self.lane.is_some() {
            self.forward(tokens, selected, &output.output)?
        } else {
            output.log_probabilities
        };
        if scores.len() != selected.len()
            || scores
                .iter()
                .any(|score| !score.is_finite() || *score > 0.0)
        {
            return Err(InferenceError::backend("invalid decision score vector"));
        }
        self.position += tokens.len();
        Ok(scores.into_iter().map(f64::from).collect())
    }

    /// Forward a readout step and validate the terminal stage's selected-token scores.
    fn forward(
        &mut self,
        tokens: &[i32],
        selected: &[i32],
        activation: &skippy_runtime::ActivationFrame,
    ) -> InferenceResult<Vec<f32>> {
        let InferenceBackendMode::EmbeddedStageZero {
            config,
            activation_width,
            downstream_wire_condition,
            ..
        } = &self.backend.mode
        else {
            return Err(InferenceError::backend(
                "missing decision stage configuration",
            ));
        };
        let mut message = embedded_prefill_message(OpenAiPrefillChunk {
            seq_id: 0,
            pos_start: self.position,
            prefill_token_count: self.position + tokens.len(),
            tokens,
            request_id: self.ids.request_id,
            session_id: self.ids.session_id,
        })?;
        message.kind = WireMessageKind::PrefillReadout;
        message.state = StageStateHeader::new(message.kind);
        message.state.prompt_token_count = i32::try_from(self.prompt_tokens)
            .map_err(|_| InferenceError::invalid_request("decision prompt exceeds wire bounds"))?;
        message.readout_tokens = selected.to_vec();
        let forwarded = forwarded_stage_message(config, &message, activation, *activation_width)
            .map_err(openai_backend_error)?;
        let (_, lane) = self
            .lane
            .as_mut()
            .ok_or_else(|| InferenceError::backend("missing decision stage lane"))?;
        write_stage_message_conditioned(&mut lane.stream, &forwarded, *downstream_wire_condition)
            .map_err(openai_io_error)?;
        let reply = recv_reply(&mut lane.stream).map_err(openai_io_error)?;
        if reply.kind != WireReplyKind::TokenScores {
            return Err(InferenceError::backend(
                "downstream did not return decision scores",
            ));
        }
        Ok(reply.log_probabilities)
    }

    /// Close the sequence's downstream connection and remove local session state.
    fn close(&mut self) -> InferenceResult<()> {
        if let Some((pool, mut lane)) = self.lane.take() {
            let stop =
                StageWireMessage::stop_with_identity(self.ids.request_id, self.ids.session_id);
            skippy_protocol::binary::write_stage_message(&mut lane.stream, &stop)
                .map_err(openai_io_error)?;
            let reply = recv_reply(&mut lane.stream).map_err(openai_io_error)?;
            if reply.kind != WireReplyKind::Ack {
                return Err(InferenceError::backend("expected decision cleanup ACK"));
            }
            pool.return_lane(lane);
        }
        self.drop_local()?;
        self.closed = true;
        Ok(())
    }

    /// Remove the local sequence session through the runtime scheduler.
    fn drop_local(&self) -> InferenceResult<()> {
        let session = self.session.clone();
        self.backend
            .iteration_scheduler
            .execute_runtime("decision-session-drop", move |runtime| {
                runtime
                    .drop_session_timed(&session)
                    .map(|_| ())
                    .map_err(openai_backend_error)
            })
    }
}

impl ReadoutJob {
    /// Reject malformed internal jobs before indexing candidates or acquiring a lane.
    pub(super) fn validate(&self) -> InferenceResult<()> {
        let valid = !self.prompt.is_empty()
            && match self.method {
                None => self.candidates.is_empty(),
                Some(ReadoutMethod::SelectedTokenLogits) => {
                    !self.candidates.is_empty()
                        && self.candidates.iter().all(|tokens| tokens.len() == 1)
                }
                Some(ReadoutMethod::CandidateLogLikelihood) => {
                    self.candidates.len() == 1 && !self.candidates[0].is_empty()
                }
            };
        if !valid {
            return Err(InferenceError::backend(
                "decision job does not match its scoring method",
            ));
        }
        Ok(())
    }
}

impl Drop for DecisionSequence<'_> {
    /// Best-effort cleanup releases a sequence abandoned before explicit close.
    fn drop(&mut self) {
        if self.closed {
            return;
        }
        // An incomplete exchange must never be returned to the reusable pool.
        if let Some((_, lane)) = self.lane.take() {
            let _ = lane.stream.shutdown(Shutdown::Both);
        }
        if let Err(error) = self.drop_local() {
            tracing::warn!("decision session cleanup failed: {error}");
        }
    }
}
