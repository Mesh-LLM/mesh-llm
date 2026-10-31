//! Checkpoint-aware prefill yields native iterations between exact boundaries.
use super::*;

pub(super) fn deferred_suffix_prefill_enabled(
    scheduling: Option<CacheSchedulingPolicy>,
    max_tokens: u32,
) -> bool {
    max_tokens > 0 && scheduling.is_some_and(CacheSchedulingPolicy::cooperative)
}

fn deferred_suffix_split(
    restored_prefill_tokens: usize,
    prompt_token_count: usize,
    checkpoint_token_count: Option<usize>,
) -> (usize, Option<usize>) {
    let suffix_start = restored_prefill_tokens.min(prompt_token_count);
    let checkpoint = checkpoint_token_count
        .filter(|&boundary| boundary > suffix_start && boundary < prompt_token_count);
    (suffix_start, checkpoint)
}

/// Pick the deferred-prefill checkpoint boundary. A valid EXPLICIT chat
/// boundary wins: it is the reusable message-history prefix a follow-up
/// request shares token-for-token, so it is the checkpoint most likely to be
/// on a future request's path. The shared-grid checkpoint is the fallback for
/// absent, invalid, or already-restored chat boundaries.
fn select_deferred_checkpoint_boundary(
    shared_checkpoint: Option<usize>,
    chat_checkpoint: Option<usize>,
    restored_prefill_tokens: usize,
    prompt_token_count: usize,
) -> Option<usize> {
    let valid =
        |boundary: &usize| *boundary > restored_prefill_tokens && *boundary < prompt_token_count;
    chat_checkpoint
        .filter(valid)
        .or_else(|| shared_checkpoint.filter(valid))
}

impl StageOpenAiBackend {
    pub(super) fn restore_or_record_kv(
        &self,
        request: &LocalGeneration<'_>,
        session_id: &str,
        cache_stats: &mut GenerationCacheStats,
    ) -> InferenceResult<Option<i32>> {
        let prefill_timer = PhaseTimer::start();
        let prefill_tokens =
            Arc::<[i32]>::from(&request.prompt_token_ids[..request.prompt_token_ids.len() - 1]);
        let recurrent_cache_prefix_token_ids = request
            .recurrent_cache_prefix_token_ids
            .map(<[i32]>::to_vec);
        let max_tokens = request.max_tokens;
        let identity_timer = PhaseTimer::start();
        let (lookup_identities, refresh_cache_affinity) = match self.kv.as_ref() {
            Some(kv) => {
                let base = self.local_kv_message_base(session_id, request.ids);
                let identities = Arc::<[PrefillKvIdentity]>::from(kv.lookup_identities(
                    &self.config,
                    &base,
                    0,
                    prefill_tokens.as_ref(),
                ));
                let kv = kv.clone();
                let config = self.config.clone();
                let refresh_identities = Arc::clone(&identities);
                (
                    identities,
                    Box::new(move || kv.peek_cache_affinity(&config, &refresh_identities))
                        as Box<dyn Fn() -> skippy_scheduler::CacheAffinity + Send>,
                )
            }
            None => (
                Arc::<[PrefillKvIdentity]>::from([]),
                Box::new(skippy_scheduler::CacheAffinity::default)
                    as Box<dyn Fn() -> skippy_scheduler::CacheAffinity + Send>,
            ),
        };
        let kv_identity_ms = identity_timer.elapsed_ms();
        let cache_scheduling = self
            .kv
            .as_ref()
            .map_or(CacheSchedulingPolicy::Serialized, |kv| {
                kv.scheduling_policy()
            });
        let scheduler_backend = self.clone();
        let scheduler_session_id = session_id.to_string();
        let scheduler_ids = request.ids.clone();
        let scheduler_lookup_identities = Arc::clone(&lookup_identities);
        let scheduler_prefill_tokens = Arc::clone(&prefill_tokens);
        let record_prefill_tokens = Arc::clone(&prefill_tokens);
        let mut scheduler_cache_stats = std::mem::take(cache_stats);
        let cache_operation_deadline = cache_operation_deadline(
            self.generation_admission_timeout,
            request.prompt_token_ids.len(),
        );
        let outcome = self.iteration_scheduler.execute_cache_aware_runtime_timed(
            "feature-kv-restore-prefill-record",
            crate::frontend::iteration_scheduler::cache_runtime::CacheAwareRuntimeRequest {
                operation_id: request.ids.request_id_string(),
                deadline: cache_operation_deadline,
                cancellation: request.cancellation,
                prompt_tokens: Arc::clone(&prefill_tokens),
                priority: 0,
                scheduling: cache_scheduling,
                refresh_affinity: refresh_cache_affinity,
            },
            move |runtime, cache_operation| {
                let outcome = scheduler_backend.restore_or_record_kv_on_runtime(
                    runtime,
                    cache_operation,
                    &scheduler_ids,
                    &scheduler_session_id,
                    scheduler_prefill_tokens.as_ref(),
                    scheduler_lookup_identities.as_ref(),
                    kv_identity_ms,
                    recurrent_cache_prefix_token_ids.as_deref(),
                    max_tokens,
                    &mut scheduler_cache_stats,
                )?;
                Ok((outcome, scheduler_cache_stats))
            },
        )?;
        let mut runtime_lock_wait_ms = outcome.runtime_lock_wait_ms;
        let mut runtime_lock_hold_ms = outcome.runtime_lock_hold_ms;
        let mut runtime_lock_acquires = 1usize;
        let (outcome, updated_cache_stats) = outcome.value;
        *cache_stats = updated_cache_stats;
        let KvRestoreOutcome {
            runtime_sessions_before,
            mut runtime_sessions_after,
            restored_prefill,
            restored_prefill_tokens,
            capacity,
            mut record,
            mut prompt_prefill_sample,
        } = outcome;
        let mut prefill_chunk_count = 1usize;
        let mut scheduler_batch_size_max = 1usize;
        let deferred_suffix_prefill = deferred_suffix_prefill_enabled(
            self.kv.as_ref().map(|kv| kv.scheduling_policy()),
            max_tokens,
        );
        if deferred_suffix_prefill && capacity.admitted {
            let exact_state = self
                .kv
                .as_ref()
                .is_some_and(|kv| kv.retains_recurrent_checkpoints());
            let checkpoint = exact_state
                .then(|| {
                    self.deferred_exact_checkpoint_identity(
                        request,
                        session_id,
                        &prefill_tokens,
                        restored_prefill_tokens,
                    )
                })
                .flatten();
            let (mut suffix_start, checkpoint_boundary) = deferred_suffix_split(
                restored_prefill_tokens,
                request.prompt_token_ids.len(),
                checkpoint
                    .as_ref()
                    .and_then(|identity| usize::try_from(identity.identity.token_count).ok()),
            );
            if exact_state && let Some(boundary) = checkpoint_boundary {
                let checkpoint_prefill = self.prefill_suffix(
                    session_id,
                    &request.prompt_token_ids[suffix_start..boundary],
                    None,
                    false,
                    cache_operation_deadline,
                    request.cancellation,
                )?;
                prefill_chunk_count = prefill_chunk_count
                    .saturating_add(checkpoint_prefill.chunk_count)
                    .saturating_sub(1);
                scheduler_batch_size_max =
                    scheduler_batch_size_max.max(checkpoint_prefill.max_batch_size);
                runtime_lock_wait_ms += checkpoint_prefill.runtime_lock_wait_ms;
                runtime_lock_hold_ms += checkpoint_prefill.runtime_lock_hold_ms;
                runtime_lock_acquires =
                    runtime_lock_acquires.saturating_add(checkpoint_prefill.chunk_count);

                if self.enqueue_exact_state_record_at_tokens(
                    session_id,
                    request.ids,
                    request.prompt_token_ids[..boundary].to_vec(),
                    "shared_prefill_checkpoint",
                    true,
                ) {
                    record.resident_enqueued_checkpoints =
                        record.resident_enqueued_checkpoints.saturating_add(1);
                }
                suffix_start = boundary;
            }

            let suffix_outcome = self.prefill_suffix(
                session_id,
                &request.prompt_token_ids[suffix_start..],
                request.sampling.enabled.then_some(request.sampling),
                true,
                cache_operation_deadline,
                request.cancellation,
            )?;
            prompt_prefill_sample = Some(suffix_outcome.predicted.ok_or_else(|| {
                InferenceError::backend("deferred suffix prefill was not sampled")
            })?);
            prefill_chunk_count = if prefill_chunk_count == 1 {
                suffix_outcome.chunk_count
            } else {
                prefill_chunk_count.saturating_add(suffix_outcome.chunk_count)
            };
            scheduler_batch_size_max = scheduler_batch_size_max.max(suffix_outcome.max_batch_size);
            runtime_lock_wait_ms += suffix_outcome.runtime_lock_wait_ms;
            runtime_lock_hold_ms += suffix_outcome.runtime_lock_hold_ms;
            runtime_lock_acquires =
                runtime_lock_acquires.saturating_add(suffix_outcome.chunk_count);

            if exact_state {
                // The final chunk samples the prompt's next token but leaves
                // the native session at the full-prompt boundary. Exporting
                // that state is useful for growing prompts, but is strictly
                // best-effort after the preferred shared checkpoint above.
                // Keep off-checkpoint states resident so every stage reopens
                // from the same durable boundary after a restart.
                if self.enqueue_exact_state_record_at_tokens(
                    session_id,
                    request.ids,
                    request.prompt_token_ids.to_vec(),
                    "final_prefill_state",
                    self.kv.as_ref().is_some_and(|kv| {
                        kv.full_exact_state_writes_through_l3(request.prompt_token_ids.len())
                    }),
                ) {
                    record.resident_enqueued_checkpoints =
                        record.resident_enqueued_checkpoints.saturating_add(1);
                }
            } else {
                let scheduler_backend = self.clone();
                let scheduler_session_id = session_id.to_string();
                let scheduler_ids = request.ids.clone();
                let emit_debug = self.telemetry.is_debug_enabled();
                let record_outcome = self.iteration_scheduler.execute_runtime_timed_bounded(
                    "feature-kv-record",
                    request.ids.request_id_string(),
                    cache_operation_deadline,
                    request.cancellation,
                    move |runtime| {
                        let record = scheduler_backend.record_and_evict_kv(
                            runtime,
                            &scheduler_session_id,
                            &scheduler_ids,
                            record_prefill_tokens.as_ref(),
                            restored_prefill,
                            restored_prefill_tokens < record_prefill_tokens.len(),
                        );
                        Ok((record, emit_debug.then(|| runtime.session_stats())))
                    },
                )?;
                runtime_lock_wait_ms += record_outcome.runtime_lock_wait_ms;
                runtime_lock_hold_ms += record_outcome.runtime_lock_hold_ms;
                runtime_lock_acquires = runtime_lock_acquires.saturating_add(1);
                (record, runtime_sessions_after) = record_outcome.value;
            }
        }
        if prompt_prefill_sample.is_some() {
            cache_stats.suffix_prefill_tokens = saturating_u32(
                request
                    .prompt_token_ids
                    .len()
                    .saturating_sub(restored_prefill_tokens),
            );
        }
        let prompt_tail_tokens = usize::from(prompt_prefill_sample.is_some());
        cache_stats.prompt_ms = prefill_timer.elapsed_ms();
        if self.telemetry.is_debug_enabled() {
            let mut attrs = self.openai_attrs(request.ids);
            attrs.insert(
                "llama_stage.prefill_token_count".to_string(),
                json!(
                    request
                        .prompt_token_ids
                        .len()
                        .saturating_sub(1)
                        .saturating_add(prompt_tail_tokens)
                ),
            );
            attrs.insert(
                "llama_stage.prefill_chunk_count".to_string(),
                json!(prefill_chunk_count),
            );
            attrs.insert(
                "skippy.scheduler.prefill_batch_size_max".to_string(),
                json!(scheduler_batch_size_max),
            );
            attrs.insert(
                "skippy.kv.restored_prefill".to_string(),
                json!(restored_prefill),
            );
            attrs.insert(
                "skippy.kv.restored_prefill_tokens".to_string(),
                json!(restored_prefill_tokens),
            );
            attrs.insert(
                "skippy.kv.prefill_suffix_tokens".to_string(),
                json!(
                    request
                        .prompt_token_ids
                        .len()
                        .saturating_sub(1)
                        .saturating_sub(restored_prefill_tokens)
                        .saturating_add(prompt_tail_tokens)
                ),
            );
            attrs.insert(
                "skippy.kv.recorded_pages".to_string(),
                json!(record.resident_recorded_pages),
            );
            attrs.insert(
                "skippy.kv.enqueued_checkpoint_pages".to_string(),
                json!(record.resident_enqueued_checkpoints),
            );
            insert_resident_capacity_attrs(&mut attrs, &capacity);
            attrs.insert(
                "llama_stage.runtime_lock_wait_ms".to_string(),
                json!(runtime_lock_wait_ms),
            );
            attrs.insert(
                "llama_stage.runtime_lock_hold_ms".to_string(),
                json!(runtime_lock_hold_ms),
            );
            attrs.insert(
                "llama_stage.runtime_lock_acquires".to_string(),
                json!(runtime_lock_acquires),
            );
            Self::insert_runtime_session_stats(
                &mut attrs,
                "llama_stage.runtime_sessions_before",
                runtime_sessions_before
                    .as_ref()
                    .expect("debug session snapshot collected"),
            );
            Self::insert_runtime_session_stats(
                &mut attrs,
                "llama_stage.runtime_sessions_after",
                runtime_sessions_after
                    .as_ref()
                    .expect("debug session snapshot collected"),
            );
            self.emit_openai_phase("stage.openai_prefill", prefill_timer, attrs);
        }
        self.telemetry.emit(
            "stage.openai_kv_record_decision",
            proactive_eviction_attrs(
                record.proactive_eviction_status,
                record.proactive_eviction_error_kind,
                record.proactive_eviction_target_tokens,
                record.proactive_evicted_entries,
                record.proactive_evicted_tokens,
            ),
        );
        let mut capacity_attrs = self.openai_attrs(request.ids);
        insert_resident_capacity_attrs(&mut capacity_attrs, &capacity);
        self.telemetry
            .emit("stage.openai_kv_capacity_decision", capacity_attrs);
        if !capacity.admitted {
            return Err(resident_capacity_admission_error(&capacity));
        }
        if let Some(error) = record.proactive_eviction_error {
            return Err(openai_backend_error(error));
        }
        Ok(prompt_prefill_sample)
    }

    fn deferred_exact_checkpoint_identity(
        &self,
        request: &LocalGeneration<'_>,
        session_id: &str,
        prefill_tokens: &[i32],
        restored_prefill_tokens: usize,
    ) -> Option<PrefillKvIdentity> {
        let kv = self
            .kv
            .as_ref()
            .filter(|kv| kv.retains_recurrent_checkpoints())?;
        let base = self.local_kv_message_base(session_id, request.ids);
        let shared_checkpoint =
            kv.exact_shared_checkpoint_identity(&self.config, &base, 0, prefill_tokens);
        let chat_checkpoint = request
            .recurrent_cache_prefix_token_ids
            .filter(|checkpoint_tokens| {
                !checkpoint_tokens.is_empty()
                    && checkpoint_tokens.len() <= prefill_tokens.len()
                    && prefill_tokens.starts_with(checkpoint_tokens)
            })
            .map(|checkpoint_tokens| {
                kv.prefill_identity(&self.config, &base, 0, checkpoint_tokens)
            });
        let shared_boundary = shared_checkpoint
            .as_ref()
            .and_then(|identity| usize::try_from(identity.identity.token_count).ok());
        let chat_boundary = chat_checkpoint
            .as_ref()
            .and_then(|identity| usize::try_from(identity.identity.token_count).ok());
        let selected_boundary = select_deferred_checkpoint_boundary(
            shared_boundary,
            chat_boundary,
            restored_prefill_tokens,
            prefill_tokens.len().saturating_add(1),
        );
        if selected_boundary == shared_boundary {
            shared_checkpoint
        } else {
            chat_checkpoint
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn restore_or_record_kv_on_runtime(
        &self,
        runtime: &mut RuntimeState,
        cache_operation: &CacheRuntimeContext,
        ids: &InferenceGenerationIds,
        session_id: &str,
        prefill_tokens: &[i32],
        lookup_identities: &[PrefillKvIdentity],
        kv_identity_ms: f64,
        recurrent_cache_prefix_token_ids: Option<&[i32]>,
        max_tokens: u32,
        cache_stats: &mut GenerationCacheStats,
    ) -> InferenceResult<KvRestoreOutcome> {
        ensure_cache_operation_active(runtime, session_id, cache_operation)?;
        let emit_debug = self.telemetry.is_debug_enabled();
        let runtime_sessions_before = emit_debug.then(|| runtime.session_stats());
        let (restored_prefill, restored_prefill_tokens, protected_resident_seq_id) =
            if let Some(kv) = self.kv.as_ref() {
                cache_stats.status = "miss";
                self.lookup_and_restore_kv(
                    kv,
                    runtime,
                    session_id,
                    ids,
                    prefill_tokens,
                    lookup_identities,
                    kv_identity_ms,
                    cache_stats,
                )
            } else {
                (false, 0, None)
            };
        ensure_cache_operation_active(runtime, session_id, cache_operation)?;
        let capacity = if let Some(kv) = self.kv.as_ref() {
            let decode_batch_tokens = u64::from(self.config.n_batch.unwrap_or(2048));
            let target_free_tokens =
                decode_batch_tokens.saturating_add(u64::from(max_tokens).min(decode_batch_tokens));
            let request_tokens = if kv.payload == StagePrefixCachePayload::ResidentKv {
                prefill_tokens
                    .len()
                    .saturating_sub(restored_prefill_tokens)
                    .saturating_add(usize::from(max_tokens > 0)) as u64
            } else {
                prefill_tokens.len() as u64
            };
            kv.admit_resident_capacity(
                runtime,
                session_id,
                request_tokens,
                decode_batch_tokens,
                target_free_tokens,
                protected_resident_seq_id,
            )
            .map_err(openai_backend_error)?
        } else {
            crate::kv_integration::ResidentCapacityDecision {
                admitted: true,
                ..crate::kv_integration::ResidentCapacityDecision::default()
            }
        };
        ensure_cache_operation_active(runtime, session_id, cache_operation)?;
        if !capacity.admitted {
            return Ok(KvRestoreOutcome {
                runtime_sessions_before,
                runtime_sessions_after: emit_debug.then(|| runtime.session_stats()),
                restored_prefill: false,
                restored_prefill_tokens: 0,
                capacity,
                record: KvRecordResult::default(),
                prompt_prefill_sample: None,
            });
        }
        let deferred_suffix_prefill = deferred_suffix_prefill_enabled(
            self.kv.as_ref().map(|kv| kv.scheduling_policy()),
            max_tokens,
        );
        if deferred_suffix_prefill {
            runtime
                .ensure_session_active(session_id)
                .map_err(openai_backend_error)?;
            cache_stats.matched_prefix_tokens = saturating_u32(restored_prefill_tokens);
            cache_stats.suffix_prefill_tokens =
                saturating_u32(prefill_tokens.len().saturating_sub(restored_prefill_tokens));
            return Ok(KvRestoreOutcome {
                runtime_sessions_before,
                runtime_sessions_after: emit_debug.then(|| runtime.session_stats()),
                restored_prefill,
                restored_prefill_tokens,
                capacity,
                record: KvRecordResult::default(),
                prompt_prefill_sample: None,
            });
        }
        let prompt_prefill_sample = None;
        let mut decoded_prefill_suffix = false;
        let cache_prefill_chunk_tokens = usize::try_from(self.config.n_ubatch.unwrap_or(256))
            .unwrap_or(2_048)
            .clamp(1, 2_048);
        if restored_prefill_tokens < prefill_tokens.len() {
            decoded_prefill_suffix = true;
            if let Some(checkpoint_tokens) =
                recurrent_cache_prefix_token_ids.filter(|checkpoint_tokens| {
                    !checkpoint_tokens.is_empty()
                        && checkpoint_tokens.len() <= prefill_tokens.len()
                        && prefill_tokens.starts_with(checkpoint_tokens)
                        && restored_prefill_tokens < checkpoint_tokens.len()
                })
            {
                prefill_cache_chunks(
                    runtime,
                    session_id,
                    &checkpoint_tokens[restored_prefill_tokens..],
                    cache_prefill_chunk_tokens,
                    cache_operation,
                )?;
                ensure_cache_operation_active(runtime, session_id, cache_operation)?;
                // This branch is the max_tokens=0 prefill-only fallback. The
                // chat-prefix state must be exported before this same runtime
                // operation advances through the remaining suffix. A detached
                // command cannot run until this operation releases the scheduler,
                // by which point the canonical-position guard would reject the
                // earlier checkpoint. Generation requests use the split, detached
                // FIFO checkpoint path in `restore_or_record_kv` instead.
                let _ = self.record_exact_state_at_tokens(
                    runtime,
                    session_id,
                    ids,
                    checkpoint_tokens,
                    "chat_prefix_checkpoint",
                    true,
                );
                prefill_cache_chunks(
                    runtime,
                    session_id,
                    &prefill_tokens[checkpoint_tokens.len()..],
                    cache_prefill_chunk_tokens,
                    cache_operation,
                )?;
            } else if let Some(kv) = self.kv.as_ref().filter(|kv| kv.payload_is_exact_state()) {
                // Recurrent and full-state payloads cannot reconstruct a shorter
                // shared prefix from the state at the end of the request. Stop
                // at the near-tail grid boundary while prefilling and snapshot
                // native state there; the final exact state is recorded by
                // `record_and_evict_kv` below. One checkpoint bounds both the
                // extra prefill split and the very large recurrent-state export.
                let base = self.local_kv_message_base(session_id, ids);
                let checkpoint =
                    kv.exact_shared_checkpoint_identity(&self.config, &base, 0, prefill_tokens);
                if let Some(identity) = checkpoint.filter(|identity| {
                    identity.identity.token_count as usize > restored_prefill_tokens
                }) {
                    let boundary = identity.identity.token_count as usize;
                    prefill_cache_chunks(
                        runtime,
                        session_id,
                        &prefill_tokens[restored_prefill_tokens..boundary],
                        cache_prefill_chunk_tokens,
                        cache_operation,
                    )?;
                    ensure_cache_operation_active(runtime, session_id, cache_operation)?;
                    let cold_prefill_cost = self
                        .generation_service_estimator
                        .estimated_prefill_ms(boundary);
                    let l3_cost = kv.l3_benefit_cost(cold_prefill_cost);
                    kv.record_exact_state_with_cost(
                        runtime,
                        session_id,
                        &identity,
                        crate::kv_integration::CaptureAdmission::BestEffort,
                        l3_cost,
                    )
                    .map_err(openai_backend_error)?;
                    prefill_cache_chunks(
                        runtime,
                        session_id,
                        &prefill_tokens[boundary..],
                        cache_prefill_chunk_tokens,
                        cache_operation,
                    )?;
                } else {
                    prefill_cache_chunks(
                        runtime,
                        session_id,
                        &prefill_tokens[restored_prefill_tokens..],
                        cache_prefill_chunk_tokens,
                        cache_operation,
                    )?;
                }
            } else {
                prefill_cache_chunks(
                    runtime,
                    session_id,
                    &prefill_tokens[restored_prefill_tokens..],
                    cache_prefill_chunk_tokens,
                    cache_operation,
                )?;
            }
        }
        ensure_cache_operation_active(runtime, session_id, cache_operation)?;
        cache_stats.matched_prefix_tokens = saturating_u32(restored_prefill_tokens);
        cache_stats.suffix_prefill_tokens =
            saturating_u32(prefill_tokens.len().saturating_sub(restored_prefill_tokens));
        let record = self.record_and_evict_kv(
            runtime,
            session_id,
            ids,
            prefill_tokens,
            restored_prefill,
            decoded_prefill_suffix,
        );
        let runtime_sessions_after = emit_debug.then(|| runtime.session_stats());
        Ok(KvRestoreOutcome {
            runtime_sessions_before,
            runtime_sessions_after,
            restored_prefill,
            restored_prefill_tokens,
            capacity,
            record,
            prompt_prefill_sample,
        })
    }
}

#[cfg(test)]
#[test]
fn resident_kv_suffix_prefill_stays_inside_the_cache_operation() {
    assert!(!deferred_suffix_prefill_enabled(
        Some(CacheSchedulingPolicy::Serialized),
        1,
    ));
}

#[cfg(test)]
#[test]
fn recurrent_suffix_prefill_requires_a_nonzero_generation_budget() {
    assert!(deferred_suffix_prefill_enabled(
        Some(CacheSchedulingPolicy::Cooperative),
        1,
    ));
    assert!(deferred_suffix_prefill_enabled(
        Some(CacheSchedulingPolicy::Cooperative),
        1,
    ));
    assert!(!deferred_suffix_prefill_enabled(
        Some(CacheSchedulingPolicy::Cooperative),
        0,
    ));
    assert!(!deferred_suffix_prefill_enabled(None, 1));
}

#[cfg(test)]
#[test]
fn deferred_suffix_split_respects_restore_and_prompt_boundaries() {
    assert_eq!(deferred_suffix_split(0, 10, Some(4)), (0, Some(4)));
    assert_eq!(deferred_suffix_split(4, 10, Some(4)), (4, None));
    assert_eq!(deferred_suffix_split(8, 10, Some(4)), (8, None));
    assert_eq!(deferred_suffix_split(4, 10, Some(10)), (4, None));
    assert_eq!(deferred_suffix_split(99, 10, Some(4)), (10, None));
}

#[cfg(test)]
#[test]
fn deferred_checkpoint_prefers_chat_boundary_with_shared_grid_fallback() {
    let prompt = 10;
    // Both valid: the explicit chat boundary wins over the shared grid.
    assert_eq!(
        select_deferred_checkpoint_boundary(Some(8), Some(6), 0, prompt),
        Some(6)
    );
    // Chat absent or None: shared grid is the fallback.
    assert_eq!(
        select_deferred_checkpoint_boundary(Some(8), None, 0, prompt),
        Some(8)
    );
    // Chat already restored (equal to the restored prefix): shared fallback.
    assert_eq!(
        select_deferred_checkpoint_boundary(Some(8), Some(4), 4, prompt),
        Some(8)
    );
    // Chat at the prompt end: invalid, shared fallback.
    assert_eq!(
        select_deferred_checkpoint_boundary(Some(8), Some(10), 0, prompt),
        Some(8)
    );
    // Chat beyond the prompt: invalid, shared fallback.
    assert_eq!(
        select_deferred_checkpoint_boundary(Some(8), Some(99), 0, prompt),
        Some(8)
    );
    // Shared already restored but chat valid: chat wins.
    assert_eq!(
        select_deferred_checkpoint_boundary(Some(4), Some(6), 4, prompt),
        Some(6)
    );
    // Both invalid: nothing selected.
    assert_eq!(
        select_deferred_checkpoint_boundary(Some(4), Some(10), 4, prompt),
        None
    );
    // Neither candidate exists: nothing selected.
    assert_eq!(
        select_deferred_checkpoint_boundary(None, None, 0, prompt),
        None
    );
}

#[cfg(test)]
#[test]
fn deferred_exact_prefill_satisfies_the_recurrent_checkpoint_gate() {
    assert!(initial_recurrent_checkpoint_attempted(true, Some(42)));
    assert!(!initial_recurrent_checkpoint_attempted(true, None));
    assert!(initial_recurrent_checkpoint_attempted(false, None));
}
