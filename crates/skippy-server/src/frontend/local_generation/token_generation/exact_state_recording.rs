use super::*;

/// Owns one unit of the test-only outstanding-capture counter for a single
/// detached capture task. Construction increments the counter, so the unit is
/// accounted for from before the task is submitted; dropping the guard — on
/// completion, skip, error, panic, or when a rejected/never-run operation is
/// dropped by the scheduler — releases it exactly once.
#[cfg(test)]
pub(crate) struct CaptureTaskOutstandingGuard(std::sync::Arc<std::sync::atomic::AtomicUsize>);

#[cfg(test)]
impl CaptureTaskOutstandingGuard {
    pub(crate) fn new(counter: std::sync::Arc<std::sync::atomic::AtomicUsize>) -> Self {
        counter.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        Self(counter)
    }
}

#[cfg(test)]
impl Drop for CaptureTaskOutstandingGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
    }
}

impl StageOpenAiBackend {
    pub(super) fn record_post_decode_exact_state(
        &self,
        request: &LocalGeneration<'_>,
        session_id: &str,
        state: &DecodeState,
    ) -> bool {
        #[cfg(test)]
        {
            super::capture_trace::POST_DECODE_ENTERED
                .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        }
        let Some(kv) = self.kv.as_ref() else {
            return false;
        };
        if !kv.payload_is_exact_state() {
            return false;
        }
        let Some(checkpoint_tokens) =
            post_decode_checkpoint_tokens(request.prompt_token_ids, &state.generated_token_ids)
        else {
            return false;
        };
        #[cfg(test)]
        {
            super::capture_trace::POST_DECODE_GATES_PASSED
                .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        }
        let scheduled = self.enqueue_exact_state_record_at_tokens(
            session_id,
            request.ids,
            checkpoint_tokens,
            "post_decode_checkpoint",
        );
        #[cfg(test)]
        if !scheduled {
            super::capture_trace::POST_DECODE_SCHEDULER_REJECTED
                .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        }
        scheduled
    }

    pub(in crate::frontend) fn enqueue_exact_state_record_at_tokens(
        &self,
        session_id: &str,
        ids: &OpenAiGenerationIds,
        checkpoint_tokens: Vec<i32>,
        decision_prefix: &'static str,
    ) -> bool {
        let scheduler_backend = self.clone();
        let scheduler_session_id = session_id.to_string();
        let scheduler_ids = ids.clone();
        // Test-only completion boundary: the guard is constructed BEFORE
        // submission (construction increments the outstanding counter), then
        // moved into the detached task and held for its full execution. Every
        // terminal path — the task running to completion, being skipped, or
        // the accepted operation being dropped unexecuted on scheduler
        // rejection or shutdown — releases the same owned guard exactly once,
        // so no manual error-path accounting exists to get wrong.
        #[cfg(test)]
        let capture_outstanding_guard = self.kv.as_ref().map(|kv| {
            CaptureTaskOutstandingGuard::new(std::sync::Arc::clone(
                &kv.exact_state_captures_outstanding,
            ))
        });
        #[cfg(test)]
        let checkpoint_token_count = checkpoint_tokens.len() as u64;
        let enqueue = self.iteration_scheduler.execute_runtime_detached(
            "feature-exact-state-checkpoint",
            move |runtime| {
                #[cfg(test)]
                let _capture_outstanding_guard = capture_outstanding_guard;
                #[cfg(test)]
                if decision_prefix == "post_decode_checkpoint" {
                    super::capture_trace::POST_DECODE_TASK_EXECUTED
                        .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
                }
                scheduler_backend.record_exact_state_at_tokens(
                    runtime,
                    &scheduler_session_id,
                    &scheduler_ids,
                    &checkpoint_tokens,
                    decision_prefix,
                );
            },
        );
        if let Err(error) = enqueue {
            #[cfg(test)]
            super::capture_trace::log_capture_decision(
                decision_prefix,
                "scheduler_rejected",
                checkpoint_token_count,
                None,
                None,
            );
            let mut attrs = self.openai_attrs(ids);
            attrs.insert(
                "skippy.kv.decision".to_string(),
                json!(format!("{decision_prefix}_scheduler_error")),
            );
            attrs.insert("skippy.kv.error".to_string(), json!(error.to_string()));
            self.telemetry
                .emit("stage.openai_kv_record_decision", attrs);
            return false;
        }
        true
    }

    /// Record a recurrent state only when the native session is at the exact
    /// token boundary named by `checkpoint_tokens`.
    ///
    /// The caller must hold the runtime lock. This check is intentionally
    /// canonical-position based: token text or a caller-supplied count cannot
    /// authorize exporting a state at a different native position.
    pub(in crate::frontend) fn record_exact_state_at_tokens(
        &self,
        runtime: &mut RuntimeState,
        session_id: &str,
        ids: &OpenAiGenerationIds,
        checkpoint_tokens: &[i32],
        decision_prefix: &str,
    ) -> bool {
        let Some(kv) = self.kv.as_ref() else {
            return false;
        };
        if !kv.payload_is_exact_state() {
            return false;
        }
        let Ok(checkpoint_token_count) = u64::try_from(checkpoint_tokens.len()) else {
            return false;
        };
        let runtime_token_count = match runtime.canonical_session_position(session_id) {
            Ok(position) => position,
            Err(error) => {
                #[cfg(test)]
                {
                    if decision_prefix == "post_decode_checkpoint" {
                        super::capture_trace::POST_DECODE_SKIPPED_POSITION_ERROR
                            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
                    }
                    super::capture_trace::log_capture_decision(
                        decision_prefix,
                        "position_error",
                        checkpoint_token_count,
                        None,
                        None,
                    );
                }
                let mut attrs = self.openai_attrs(ids);
                attrs.insert(
                    "skippy.kv.decision".to_string(),
                    json!(format!("{decision_prefix}_skipped")),
                );
                attrs.insert("skippy.kv.error".to_string(), json!(error.to_string()));
                self.telemetry
                    .emit("stage.openai_kv_record_decision", attrs);
                return false;
            }
        };
        if runtime_token_count != checkpoint_token_count {
            #[cfg(test)]
            {
                if decision_prefix == "post_decode_checkpoint" {
                    super::capture_trace::POST_DECODE_SKIPPED_POSITION_MISMATCH
                        .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
                    super::capture_trace::LAST_RUNTIME_POSITION.store(
                        runtime_token_count as i64,
                        std::sync::atomic::Ordering::Release,
                    );
                    super::capture_trace::LAST_CHECKPOINT_COUNT.store(
                        checkpoint_token_count as i64,
                        std::sync::atomic::Ordering::Release,
                    );
                }
                super::capture_trace::log_capture_decision(
                    decision_prefix,
                    "position_mismatch",
                    checkpoint_token_count,
                    Some(runtime_token_count),
                    None,
                );
            }
            let mut attrs = self.openai_attrs(ids);
            attrs.insert(
                "skippy.kv.decision".to_string(),
                json!(format!("{decision_prefix}_skipped")),
            );
            attrs.insert(
                "skippy.kv.checkpoint_token_count".to_string(),
                json!(checkpoint_token_count),
            );
            attrs.insert(
                "skippy.kv.runtime_token_count".to_string(),
                json!(runtime_token_count),
            );
            self.telemetry
                .emit("stage.openai_kv_record_decision", attrs);
            return false;
        }

        let base = self.local_kv_message_base(session_id, ids);
        let identity = kv.prefill_identity(&self.config, &base, 0, checkpoint_tokens);
        #[cfg(test)]
        let decision_namespace = identity.namespace.clone();
        let admission = if decision_prefix == "post_decode_checkpoint" {
            crate::kv_integration::CaptureAdmission::Continuation
        } else {
            crate::kv_integration::CaptureAdmission::BestEffort
        };
        match kv.record_exact_state(runtime, session_id, &identity, admission) {
            Ok(Some(record)) => {
                let mut attrs = self.openai_attrs(ids);
                attrs.insert(
                    "skippy.kv.decision".to_string(),
                    json!(format!("{decision_prefix}_recorded")),
                );
                attrs.insert(
                    "skippy.exact_cache.recorded_page_id".to_string(),
                    json!(record.page_id),
                );
                attrs.insert(
                    "skippy.exact_cache.payload_kind".to_string(),
                    json!(record.payload_kind.to_string()),
                );
                attrs.insert(
                    "skippy.exact_cache.recorded_tokens".to_string(),
                    json!(record.token_count),
                );
                attrs.insert("skippy.exact_cache.queued".to_string(), json!(true));
                self.telemetry
                    .emit("stage.openai_kv_record_decision", attrs);
                #[cfg(test)]
                {
                    if decision_prefix == "post_decode_checkpoint" {
                        super::capture_trace::POST_DECODE_RECORDED
                            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
                    }
                    super::capture_trace::log_capture_decision(
                        decision_prefix,
                        "recorded_enqueued",
                        checkpoint_token_count,
                        Some(runtime_token_count),
                        Some(&decision_namespace),
                    );
                }
                true
            }
            Ok(None) => {
                #[cfg(test)]
                {
                    if decision_prefix == "post_decode_checkpoint" {
                        super::capture_trace::POST_DECODE_RECORD_NONE
                            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
                    }
                    super::capture_trace::log_capture_decision(
                        decision_prefix,
                        "record_none",
                        checkpoint_token_count,
                        Some(runtime_token_count),
                        Some(&decision_namespace),
                    );
                }
                false
            }
            Err(error) => {
                #[cfg(test)]
                {
                    if decision_prefix == "post_decode_checkpoint" {
                        super::capture_trace::POST_DECODE_RECORD_ERROR
                            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
                    }
                    super::capture_trace::log_capture_decision(
                        decision_prefix,
                        "record_error",
                        checkpoint_token_count,
                        Some(runtime_token_count),
                        Some(&decision_namespace),
                    );
                }
                let mut attrs = self.openai_attrs(ids);
                attrs.insert(
                    "skippy.kv.decision".to_string(),
                    json!(format!("{decision_prefix}_error")),
                );
                attrs.insert("skippy.kv.error".to_string(), json!(error.to_string()));
                self.telemetry
                    .emit("stage.openai_kv_record_decision", attrs);
                false
            }
        }
    }
}

#[cfg(test)]
mod capture_guard_tests {
    use super::CaptureTaskOutstandingGuard;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    #[test]
    fn capture_guard_counts_construction_and_drop() {
        let counter = Arc::new(AtomicUsize::new(0));
        {
            let _guard = CaptureTaskOutstandingGuard::new(Arc::clone(&counter));
            assert_eq!(
                counter.load(Ordering::Acquire),
                1,
                "construction increments"
            );
        }
        assert_eq!(counter.load(Ordering::Acquire), 0, "drop releases");
    }

    #[test]
    fn capture_guard_releases_on_panic() {
        let counter = Arc::new(AtomicUsize::new(0));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = CaptureTaskOutstandingGuard::new(Arc::clone(&counter));
            assert_eq!(counter.load(Ordering::Acquire), 1);
            panic!("capture task panicked");
        }));
        assert!(result.is_err(), "panic should propagate past the guard");
        assert_eq!(
            counter.load(Ordering::Acquire),
            0,
            "guard releases during unwind"
        );
    }
}
