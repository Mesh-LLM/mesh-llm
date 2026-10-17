use super::*;

impl StageOpenAiBackend {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn restore_or_record_kv_on_runtime(
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
        hybrid_sampling: Option<(String, Option<SamplingConfig>)>,
        cache_stats: &mut GenerationCacheStats,
    ) -> InferenceResult<KvRestoreOutcome> {
        ensure_cache_operation_active(runtime, session_id, cache_operation)?;
        let emit_debug = self.telemetry.is_debug_enabled();
        let runtime_sessions_before = emit_debug.then(|| runtime.session_stats());
        if self
            .kv
            .as_ref()
            .is_some_and(|kv| kv.payload == StagePrefixCachePayload::KvRecurrent)
        {
            runtime.enable_resident_recurrent_request(session_id);
        }
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
        if let Some((metadata, sampling)) = hybrid_sampling {
            runtime
                .configure_chat_sampling(
                    session_id,
                    &metadata,
                    (prefill_tokens.len() + 1) as u64,
                    sampling.as_ref(),
                )
                .map_err(openai_backend_error)?;
        }
        let deferred_suffix_prefill =
            deferred_suffix_prefill_enabled(self.kv.as_ref().map(|kv| kv.payload), max_tokens);
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
