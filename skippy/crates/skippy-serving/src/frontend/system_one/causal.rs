//! Causal questions use independent, admitted scheduler sequences.

use futures_util::{StreamExt, stream};
use skippy_inference_api::system_one::{DecisionReadout, DecisionReadoutUsage};
use skippy_inference_api::{
    CancellationToken, InferenceError, InferenceResult, SystemOneRequest, SystemOneResponse,
};
use skippy_runtime::DecisionOutput;
use skippy_runtime::decision::causal::{DecisionPlan, ReadoutMethod, normalize_candidate_scores};
use std::sync::Arc;

use crate::frontend::StageOpenAiBackend;
use crate::frontend::generation::{
    GenerationAdmissionScheduling, GenerationAdmissionWork, InferenceCacheHints,
    InferenceGenerationIds,
};
use crate::frontend::util::openai_backend_error;

mod cache;
mod preparation;
mod sequence;

#[derive(Clone, Debug)]
struct ReadoutJob {
    question: usize,
    candidate: Option<usize>,
    prompt: Arc<Vec<i32>>,
    candidates: Vec<Vec<i32>>,
    /// None denotes cache preparation; scoring never infers its rule from cardinality.
    method: Option<ReadoutMethod>,
}

#[derive(Debug)]
struct ReadoutResult {
    scores: Vec<f64>,
    evaluated_tokens: usize,
    cached_tokens: usize,
}

#[derive(Debug)]
struct ReadoutFailure {
    error: Box<InferenceError>,
    /// Completed local steps remain work even when a later exchange fails.
    evaluated_tokens: usize,
    cached_tokens: usize,
}

impl From<InferenceError> for ReadoutFailure {
    /// Failures before execution have no completed work to report.
    fn from(error: InferenceError) -> Self {
        Self {
            error: Box::new(error),
            evaluated_tokens: 0,
            cached_tokens: 0,
        }
    }
}

struct CancelOnDrop(CancellationToken);
impl Drop for CancelOnDrop {
    /// Cancel outstanding readout work when its frontend owner is dropped.
    fn drop(&mut self) {
        self.0.cancel();
    }
}

impl StageOpenAiBackend {
    /// Plan and serve independent decisions without tokenizing on the scheduler worker.
    pub(in crate::frontend) async fn serve_system_one(
        &self,
        request: SystemOneRequest,
    ) -> InferenceResult<SystemOneResponse> {
        super::validate_request_fields(&request, &self.model_id)?;
        let decision = super::decision_request(&request)?;
        let scheduler = self.iteration_scheduler.clone();
        let plan = tokio::task::spawn_blocking(move || {
            let reader = scheduler.execute_runtime("decision-plan", |runtime| {
                Ok(runtime
                    .model
                    .supports_causal_readout()
                    .then(|| runtime.model.reader()))
            })?;
            let Some(reader) = reader else {
                return Ok(None);
            };
            // Vocabulary reads must not occupy the scheduler worker or runtime
            // lock: a request may tokenize up to 64 prompts and 1,664 labels.
            DecisionPlan::for_reader(&reader, &decision)
                .map(Some)
                .map_err(|error| InferenceError::invalid_request(error.to_string()))
        })
        .await
        .map_err(|error| openai_backend_error(error.into()))??;
        let Some(plan) = plan else {
            let backend = self.clone();
            return tokio::task::spawn_blocking(move || backend.run_system_one(request))
                .await
                .map_err(|error| openai_backend_error(error.into()))?;
        };
        validate_context(&plan, self.ctx_size)?;
        let guard = CancelOnDrop(CancellationToken::new());
        let preparation = self.warm_decision_prefixes(&plan, &guard.0).await?;
        let jobs = decision_jobs(&plan);
        let mut executor = self.clone();
        if !preparation.reuse_cache {
            // Retry in fresh sequences without touching a cache that just failed.
            // This affects only this request, not the shared backend or other users.
            executor.kv = None;
        }
        let results = executor.run_readout_jobs(jobs, &guard.0).await?;
        decision_response(request, &plan, results, preparation.usage)
    }

    /// Run a bounded set of readout jobs and drain admitted work after cancellation.
    async fn run_readout_jobs(
        &self,
        jobs: Vec<ReadoutJob>,
        cancellation: &CancellationToken,
    ) -> InferenceResult<Vec<(ReadoutJob, ReadoutResult)>> {
        let width = self.generation_limit.current_limit().clamp(1, 64);
        let mut running = stream::iter(jobs.into_iter().map(|job| async move {
            let result = self.run_readout_job(job.clone(), cancellation).await;
            (job, result)
        }))
        .buffer_unordered(width);
        let mut completed = Vec::new();
        let mut failure = None;
        while let Some((job, result)) = running.next().await {
            match result {
                Ok(result) => completed.push((job, result)),
                Err(error) => {
                    failure.get_or_insert(*error.error);
                    cancellation.cancel();
                }
            }
        }
        failure.map_or(Ok(completed), Err)
    }

    /// Acquire normal generation admission before executing one independent readout job.
    async fn run_readout_job(
        &self,
        job: ReadoutJob,
        cancellation: &CancellationToken,
    ) -> Result<ReadoutResult, ReadoutFailure> {
        job.validate()?;
        let ids = InferenceGenerationIds::new_with_trust(
            InferenceCacheHints::default(),
            None,
            false,
            None,
        );
        let continuation = job.candidates.iter().map(Vec::len).max().unwrap_or(0);
        let work =
            GenerationAdmissionWork::new(job.prompt.len() + continuation.saturating_sub(1), 0);
        let permit = self
            .acquire_generation_admission(
                &ids,
                cancellation,
                work,
                GenerationAdmissionScheduling::default(),
            )
            .await?;
        let backend = self.clone();
        let cancellation = cancellation.clone();
        tokio::task::spawn_blocking(move || {
            let _permit = permit;
            sequence::execute(&backend, &job, &ids, &cancellation)
        })
        .await
        .map_err(|error| ReadoutFailure::from(openai_backend_error(error.into())))?
    }
}

/// Construct a prefill-only job that records a reusable prefix without scoring.
fn warmup_job(tokens: Vec<i32>) -> ReadoutJob {
    ReadoutJob {
        question: 0,
        candidate: None,
        prompt: Arc::new(tokens),
        candidates: Vec::new(),
        method: None,
    }
}

/// Reject questions whose prompt and teacher-forced continuation exceed context.
fn validate_context(plan: &DecisionPlan, context: usize) -> InferenceResult<()> {
    for question in &plan.questions {
        let continuation = question.candidates.iter().map(Vec::len).max().unwrap_or(0);
        if question.prompt.len() + continuation.saturating_sub(1) > context {
            return Err(InferenceError::invalid_request(format!(
                "question {:?} and its longest candidate exceed the loaded context ({context} tokens)",
                question.key,
            )));
        }
    }
    Ok(())
}

/// Expand question plans into independent selected-token or candidate-likelihood jobs.
fn decision_jobs(plan: &DecisionPlan) -> Vec<ReadoutJob> {
    plan.questions
        .iter()
        .enumerate()
        .flat_map(|(question, plan)| {
            let prompt = Arc::new(plan.prompt.clone());
            if plan.method == ReadoutMethod::SelectedTokenLogits {
                vec![ReadoutJob {
                    question,
                    candidate: None,
                    prompt,
                    candidates: plan.candidates.clone(),
                    method: Some(plan.method),
                }]
            } else {
                plan.candidates
                    .iter()
                    .enumerate()
                    .map(|(index, candidate)| ReadoutJob {
                        question,
                        candidate: Some(index),
                        prompt: prompt.clone(),
                        candidates: vec![candidate.clone()],
                        method: Some(plan.method),
                    })
                    .collect()
            }
        })
        .collect()
}

/// Reassemble unordered job completions into ordered answers and execution evidence.
fn decision_response(
    request: SystemOneRequest,
    plan: &DecisionPlan,
    results: Vec<(ReadoutJob, ReadoutResult)>,
    mut usage: DecisionReadoutUsage,
) -> InferenceResult<SystemOneResponse> {
    let mut evidence = plan
        .questions
        .iter()
        .map(|question| DecisionReadout {
            method: question.method.as_str(),
            normalization: "sum_log_probabilities",
            probability_semantics: "relative_candidate_probability_not_calibrated_truth",
            candidate_log_probabilities: vec![f64::NAN; question.candidates.len()],
            candidate_token_counts: question.candidates.iter().map(Vec::len).collect(),
            evaluated_tokens: 0,
            cached_tokens: 0,
        })
        .collect::<Vec<_>>();
    for (job, result) in results {
        usage.evaluated_tokens += result.evaluated_tokens;
        usage.cached_tokens += result.cached_tokens;
        record_result(&mut evidence, &job, result)?;
    }
    let probabilities = evidence
        .iter()
        .map(|entry| {
            normalize_candidate_scores(&entry.candidate_log_probabilities)
                .map_err(openai_backend_error)
        })
        .collect::<InferenceResult<Vec<_>>>()?;
    // Usage describes executed inputs, including independent candidate branches
    // and preparation, so cached inputs are a subset of input_tokens.
    let input_tokens = usage.evaluated_tokens.saturating_add(usage.cached_tokens);
    let mut response = super::response(
        request,
        DecisionOutput {
            probabilities,
            input_tokens,
        },
    )?;
    response.usage.cached_tokens = u32::try_from(usage.cached_tokens).unwrap_or(u32::MAX);
    response.readouts = plan
        .questions
        .iter()
        .zip(evidence)
        .map(|(q, evidence)| (q.key.clone(), evidence))
        .collect();
    response.readout_usage = Some(usage);
    Ok(response)
}

/// Validate and record one job result without accepting duplicate candidate evidence.
fn record_result(
    evidence: &mut [DecisionReadout],
    job: &ReadoutJob,
    result: ReadoutResult,
) -> InferenceResult<()> {
    let entry = evidence
        .get_mut(job.question)
        .ok_or_else(|| InferenceError::backend("unknown decision question result"))?;
    let slots = match job.candidate {
        Some(candidate) => entry
            .candidate_log_probabilities
            .get_mut(candidate..candidate.saturating_add(1)),
        None => Some(entry.candidate_log_probabilities.as_mut_slice()),
    }
    .ok_or_else(|| InferenceError::backend("unknown decision candidate result"))?;
    if slots.len() != result.scores.len()
        || slots.iter().any(|score| !score.is_nan())
        || result
            .scores
            .iter()
            .any(|score| !score.is_finite() || *score > 0.0)
    {
        return Err(InferenceError::backend(
            "invalid or duplicate decision score vector",
        ));
    }
    slots.copy_from_slice(&result.scores);
    entry.evaluated_tokens += result.evaluated_tokens;
    entry.cached_tokens += result.cached_tokens;
    Ok(())
}

#[cfg(test)]
mod tests;
