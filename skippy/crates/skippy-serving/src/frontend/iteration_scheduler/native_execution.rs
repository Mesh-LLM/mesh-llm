use super::*;

impl SchedulerWorker {
    pub(super) fn run_direct_iteration_batch(&mut self) {
        let batch = take_direct_iteration_batch_with_policy(
            &mut self.direct_iterations,
            self.direct_group_batch_size,
            self.max_direct_iteration_tokens,
            self.direct_batch_policy,
        );
        debug_assert!(!batch.is_empty(), "validated direct queue must yield work");

        let mut active = Vec::with_capacity(batch.len());
        for request in batch {
            match request.ensure_active() {
                Ok(()) => active.push(request),
                Err(error) => {
                    request.reply.send(Err(error));
                }
            }
        }
        if active.is_empty() {
            return;
        }

        let lock_started = Instant::now();
        let mut runtime = lock_runtime(&self.runtime);
        let runtime_lock_wait_ms = lock_started.elapsed().as_secs_f64() * 1_000.0;
        let hold_started = Instant::now();
        let mut runnable = Vec::with_capacity(active.len());
        for request in active {
            if let Err(error) = request.ensure_active() {
                request.reply.send(Err(error));
                continue;
            }
            if request.selected_tokens.is_some() && !runtime.model.supports_causal_readout() {
                request.reply.send(Err(InferenceError::unsupported(
                    "loaded stage does not support causal readouts",
                )));
                continue;
            }
            let alignment = match request.target_token_count {
                Some(target_token_count) => runtime
                    .align_session_to_token_count_if_ahead(&request.session_id, target_token_count),
                None => Ok(None),
            };
            match alignment {
                Ok(_)
                    if request.selected_tokens.is_some()
                        && runtime
                            .session_token_count(&request.session_id)
                            .unwrap_or(0)
                            != request.target_token_count.unwrap_or(0) =>
                {
                    request.reply.send(Err(InferenceError::invalid_request(
                        "decision readout position does not match session prefix",
                    )));
                }
                Ok(alignment) => runnable.push((request, alignment)),
                Err(error) => {
                    request.reply.send(Err(openai_backend_error(error)));
                }
            }
        }
        if runnable.is_empty() {
            return;
        }
        let batch_size = runnable.len();
        let decode_steps = runnable
            .iter()
            .filter(|(request, _)| request.phase == IterationBatchPhase::Decode)
            .count();
        let token_count = runnable
            .iter()
            .map(|(request, _)| request.token_ids.len())
            .sum::<usize>();
        let batch_wait_ms = runnable
            .iter()
            .map(|(request, _)| request.enqueued_at.elapsed().as_secs_f64() * 1_000.0)
            .collect::<Vec<_>>();
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
        let result = runtime
            .iteration_batch_scored(&requests, &selections)
            .map_err(openai_backend_error);
        let hold = hold_started.elapsed();
        let runtime_lock_hold_ms = hold.as_secs_f64() * 1_000.0;
        let completed = runtime.take_completed_execution();
        drop(runtime);
        self.record_compute(hold);
        if result.is_ok() {
            if let Some(completed) = completed {
                self.service_estimator.observe_execution(
                    GenerationAdmissionWork {
                        prompt_tokens: completed.prompt_tokens,
                        decode_tokens: completed.decode_tokens,
                    },
                    completed.elapsed,
                );
            }
            for (request, _) in &runnable {
                let prefill = request.phase == IterationBatchPhase::Prefill;
                self.service_estimator.advance(
                    &request.session_id,
                    GenerationAdmissionWork {
                        prompt_tokens: if prefill {
                            request.token_ids.len() as u64
                        } else {
                            0
                        },
                        decode_tokens: if prefill {
                            u64::from(request.sample_last)
                        } else {
                            1
                        },
                    },
                );
            }
        }

        self.compute_meter.record_decode_tokens(decode_steps as u64);
        if let Some(telemetry) = self.telemetry.as_ref() {
            telemetry.emit_debug(
                "stage.scheduler_feature_iteration",
                BTreeMap::from([
                    ("skippy.scheduler.batch_size".to_string(), json!(batch_size)),
                    (
                        "skippy.scheduler.token_count".to_string(),
                        json!(token_count),
                    ),
                    (
                        "skippy.scheduler.batch_wait_max_ms".to_string(),
                        json!(batch_wait_ms.iter().copied().fold(0.0_f64, f64::max)),
                    ),
                    (
                        "skippy.scheduler.runtime_lock_wait_ms".to_string(),
                        json!(runtime_lock_wait_ms),
                    ),
                    (
                        "skippy.scheduler.runtime_lock_hold_ms".to_string(),
                        json!(runtime_lock_hold_ms),
                    ),
                ]),
            );
        }

        match result {
            Ok((outputs, scores)) => {
                if outputs.request_outputs.len() != runnable.len() {
                    let error = InferenceError::backend(format!(
                        "scheduler iteration returned {} outputs for {} requests",
                        outputs.request_outputs.len(),
                        runnable.len()
                    ));
                    for (request, _) in runnable {
                        request.reply.send(Err(error.clone()));
                    }
                    return;
                }
                if scores.len() != runnable.len() {
                    let error = InferenceError::backend(format!(
                        "scheduler iteration returned {} score sets for {} requests",
                        scores.len(),
                        runnable.len()
                    ));
                    for (request, _) in runnable {
                        request.reply.send(Err(error.clone()));
                    }
                    return;
                }
                let mut predicted = vec![None; runnable.len()];
                for sample in outputs.samples {
                    let Some(slot) = predicted.get_mut(sample.request_index) else {
                        let error = InferenceError::backend(format!(
                            "scheduler iteration sample references request {}, but only {} requests ran",
                            sample.request_index,
                            runnable.len()
                        ));
                        for (request, _) in runnable {
                            request.reply.send(Err(error.clone()));
                        }
                        return;
                    };
                    if slot.replace(sample.predicted_token).is_some() {
                        let error = InferenceError::backend(format!(
                            "scheduler iteration returned duplicate sample for request {}",
                            sample.request_index
                        ));
                        for (request, _) in runnable {
                            request.reply.send(Err(error.clone()));
                        }
                        return;
                    }
                }
                for (
                    ((((request, session_alignment), output), predicted), batch_wait_ms),
                    scores,
                ) in runnable
                    .into_iter()
                    .zip(outputs.request_outputs)
                    .zip(predicted)
                    .zip(batch_wait_ms)
                    .zip(scores)
                {
                    request.reply.send(Ok(SchedulerIterationOutcome {
                        predicted: predicted.unwrap_or(-1),
                        log_probabilities: scores,
                        output,
                        batch_size,
                        batch_wait_ms,
                        runtime_lock_wait_ms,
                        runtime_lock_hold_ms,
                        session_alignment,
                    }));
                }
            }
            Err(error) => {
                for (request, _) in runnable {
                    request.reply.send(Err(error.clone()));
                }
            }
        }
    }

    pub(super) fn execute_plan(
        &self,
        plan: &skippy_scheduler::IterationPlan,
    ) -> InferenceResult<Vec<IterationPrediction>> {
        let mut runtime = lock_runtime(&self.runtime);
        let hold_started = Instant::now();
        let requests = plan
            .work
            .iter()
            .map(|work| RuntimeIterationBatchRequest {
                session_id: &work.sequence_id,
                token_ids: &work.tokens,
                // Text-only serving leaves positions to the native runtime so
                // model-specific widths (for example Qwen mRoPE's four
                // position dimensions) are expanded correctly.
                positions: &[],
                sampling: work.sampling.as_ref(),
                input: None,
                sample_last: work.sample_last,
                phase: match work.phase {
                    IterationPhase::Decode => IterationBatchPhase::Decode,
                    IterationPhase::Prefill | IterationPhase::Recompute => {
                        IterationBatchPhase::Prefill
                    }
                },
            })
            .collect::<Vec<_>>();
        let result = runtime.iteration_batch_sampled(&requests);
        let completed = runtime.take_completed_execution();
        drop(runtime);
        let elapsed = hold_started.elapsed();
        self.record_compute(elapsed);
        if result.is_ok() {
            for work in &plan.work {
                let decode = work.phase == IterationPhase::Decode;
                let progress = GenerationAdmissionWork {
                    prompt_tokens: if decode { 0 } else { work.tokens.len() as u64 },
                    decode_tokens: if decode {
                        1
                    } else {
                        u64::from(work.sample_last)
                    },
                };
                self.service_estimator.advance(&work.sequence_id, progress);
            }
            if let Some(completed) = completed {
                self.service_estimator.observe_execution(
                    GenerationAdmissionWork {
                        prompt_tokens: completed.prompt_tokens,
                        decode_tokens: completed.decode_tokens,
                    },
                    completed.elapsed,
                );
            }
        }

        let decode_steps = plan
            .work
            .iter()
            .filter(|work| work.phase == IterationPhase::Decode)
            .count();
        self.compute_meter.record_decode_tokens(decode_steps as u64);
        result
            .map(|outputs| {
                outputs
                    .samples
                    .into_iter()
                    .map(|sample| IterationPrediction {
                        work_index: sample.request_index,
                        token: sample.predicted_token,
                    })
                    .collect()
            })
            .map_err(openai_backend_error)
    }
}
