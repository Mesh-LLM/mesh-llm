//! Execute admitted direct iterations under one native runtime lease.

use super::*;
use skippy_runtime::IterationBatchOutput;

type Runnable = (DirectIteration, Option<RuntimeSessionAlignStats>);

struct BatchTiming {
    wait: Vec<f64>,
    lock_wait_ms: f64,
    lock_hold_ms: f64,
}

impl SchedulerWorker {
    /// Align and execute a direct batch, then deliver each request's own result.
    pub(super) fn run_direct_iteration_batch(&mut self) {
        let batch = take_direct_iteration_batch(
            &mut self.direct_iterations,
            self.direct_group_batch_size,
            self.max_direct_iteration_tokens,
        );
        debug_assert!(!batch.is_empty(), "validated direct queue must yield work");
        let active = retain_active(batch);
        if active.is_empty() {
            return;
        }
        let lock_started = Instant::now();
        let mut runtime = lock_runtime(&self.runtime);
        let lock_wait_ms = lock_started.elapsed().as_secs_f64() * 1_000.0;
        let hold_started = Instant::now();
        let runnable = align_requests(&mut runtime, active);
        if runnable.is_empty() {
            return;
        }
        let requests = runnable
            .iter()
            .map(|(request, _)| RuntimeIterationBatchRequest {
                session_id: &request.session_id,
                token_ids: &request.token_ids,
                positions: &request.positions,
                sampling: request.sampling.as_ref(),
                input: request.input.as_ref(),
                sample_last: request.sample_last,
                phase: request.phase,
            })
            .collect::<Vec<_>>();
        #[cfg(test)]
        if requests
            .iter()
            .any(|request| request.session_id == tests::PANICKING_DIRECT_SESSION)
        {
            panic!("direct iteration batch panicked");
        }
        let selections = runnable
            .iter()
            .map(|(request, _)| request.selected_tokens.as_deref().unwrap_or_default())
            .collect::<Vec<_>>();
        let wait = runnable
            .iter()
            .map(|(request, _)| request.enqueued_at.elapsed().as_secs_f64() * 1_000.0)
            .collect();
        let result = runtime
            .iteration_batch_scored(&requests, &selections)
            .map_err(openai_backend_error);
        let hold = hold_started.elapsed();
        drop(runtime);
        self.record_compute(hold);
        let timing = BatchTiming {
            wait,
            lock_wait_ms,
            lock_hold_ms: hold.as_secs_f64() * 1_000.0,
        };
        self.record_direct_batch(&runnable, &timing);
        match result {
            Ok((output, scores)) => send_outputs(runnable, output, scores, timing),
            Err(error) => fail_requests(runnable, error),
        }
    }

    /// Record execution and timing evidence for one native direct batch.
    fn record_direct_batch(&self, requests: &[Runnable], timing: &BatchTiming) {
        let decode_steps = requests
            .iter()
            .filter(|(request, _)| request.phase == IterationBatchPhase::Decode)
            .count();
        self.compute_meter.record_decode_tokens(decode_steps as u64);
        if let Some(telemetry) = self.telemetry.as_ref() {
            telemetry.emit_debug(
                "stage.scheduler_feature_iteration",
                BTreeMap::from([
                    (
                        "skippy.scheduler.batch_size".to_string(),
                        json!(requests.len()),
                    ),
                    (
                        "skippy.scheduler.token_count".to_string(),
                        json!(
                            requests
                                .iter()
                                .map(|(r, _)| r.token_ids.len())
                                .sum::<usize>()
                        ),
                    ),
                    (
                        "skippy.scheduler.batch_wait_max_ms".to_string(),
                        json!(timing.wait.iter().copied().fold(0.0_f64, f64::max)),
                    ),
                    (
                        "skippy.scheduler.runtime_lock_wait_ms".to_string(),
                        json!(timing.lock_wait_ms),
                    ),
                    (
                        "skippy.scheduler.runtime_lock_hold_ms".to_string(),
                        json!(timing.lock_hold_ms),
                    ),
                ]),
            );
        }
    }
}

/// Remove cancelled requests before a direct batch enters native execution.
fn retain_active(batch: Vec<DirectIteration>) -> Vec<DirectIteration> {
    batch
        .into_iter()
        .filter_map(|request| match request.ensure_active() {
            Ok(()) => Some(request),
            Err(error) => {
                request.reply.send(Err(error));
                None
            }
        })
        .collect()
}

/// Align every retained request with its session's current token position.
fn align_requests(runtime: &mut RuntimeState, batch: Vec<DirectIteration>) -> Vec<Runnable> {
    batch
        .into_iter()
        .filter_map(|request| {
            let alignment =
                request
                    .ensure_active()
                    .and_then(|()| match request.target_token_count {
                        Some(target) => align_request(runtime, &request, target),
                        None => Ok(None),
                    });
            match alignment {
                Ok(alignment) => Some((request, alignment)),
                Err(error) => {
                    request.reply.send(Err(error));
                    None
                }
            }
        })
        .collect()
}

/// Align one request or report its error without corrupting the remaining batch.
fn align_request(
    runtime: &mut RuntimeState,
    request: &DirectIteration,
    target: u64,
) -> InferenceResult<Option<RuntimeSessionAlignStats>> {
    if request.selected_tokens.is_some() && !runtime.model.supports_causal_readout() {
        return Err(InferenceError::unsupported(
            "loaded stage does not support causal readouts",
        ));
    }
    let alignment = runtime
        .align_session_to_token_count_if_ahead(&request.session_id, target)
        .map_err(openai_backend_error)?;
    if request.selected_tokens.is_some()
        && runtime
            .session_token_count(&request.session_id)
            .unwrap_or(0)
            != target
    {
        return Err(InferenceError::invalid_request(
            "decision readout position does not match session prefix",
        ));
    }
    Ok(alignment)
}

/// Deliver the same terminal failure to every affected request.
fn fail_requests(requests: Vec<Runnable>, error: InferenceError) {
    for (request, _) in requests {
        request.reply.send(Err(error.clone()));
    }
}

/// Map sparse native predictions back to their original request slots.
fn prediction_slots(
    output: &IterationBatchOutput,
    count: usize,
    score_count: usize,
) -> InferenceResult<Vec<Option<i32>>> {
    if output.request_outputs.len() != count || score_count != count {
        return Err(InferenceError::backend(format!(
            "scheduler returned {} outputs and {score_count} readouts for {count} requests",
            output.request_outputs.len()
        )));
    }
    let mut predicted = vec![None; count];
    for sample in &output.samples {
        let Some(slot) = predicted.get_mut(sample.request_index) else {
            return Err(InferenceError::backend(format!(
                "scheduler sample references request {}, but only {count} requests ran",
                sample.request_index
            )));
        };
        if slot.replace(sample.predicted_token).is_some() {
            return Err(InferenceError::backend(format!(
                "scheduler returned duplicate sample for request {}",
                sample.request_index
            )));
        }
    }
    Ok(predicted)
}

/// Deliver frames, predictions, scores and timing to their matching request owners.
fn send_outputs(
    requests: Vec<Runnable>,
    output: IterationBatchOutput,
    scores: Vec<Vec<f32>>,
    timing: BatchTiming,
) {
    let predicted = match prediction_slots(&output, requests.len(), scores.len()) {
        Ok(predicted) => predicted,
        Err(error) => {
            fail_requests(requests, error);
            return;
        }
    };
    let batch_size = requests.len();
    for (((((request, session_alignment), output), predicted), batch_wait_ms), log_probabilities) in
        requests
            .into_iter()
            .zip(output.request_outputs)
            .zip(predicted)
            .zip(timing.wait)
            .zip(scores)
    {
        request.reply.send(Ok(SchedulerIterationOutcome {
            predicted: predicted.unwrap_or(-1),
            log_probabilities,
            output,
            batch_size,
            batch_wait_ms,
            runtime_lock_wait_ms: timing.lock_wait_ms,
            runtime_lock_hold_ms: timing.lock_hold_ms,
            session_alignment,
        }));
    }
}
