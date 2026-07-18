use crate::frontend::NativeMtpDecodeOptions;
use crate::frontend::NativeMtpDraft;
use crate::frontend::NativeMtpDraftOrigin;
use crate::frontend::NativeMtpVerifier;
use crate::frontend::generation::GenerationCacheStats;
use crate::frontend::generation::LocalGeneration;
use crate::frontend::generation::PhaseTimer;
use crate::frontend::generation::StageOpenAiBackend;
use crate::frontend::generation::TokenControl;
use crate::frontend::generation::decode_token_phase;
use crate::frontend::util::openai_backend_error;
use crate::frontend::util::saturating_u32;
use crate::kv_integration::proactive_eviction_attrs;
use crate::kv_integration::proactive_eviction_error_kind;
use openai_frontend::OpenAiError;
use openai_frontend::OpenAiResult;
use serde_json::json;
use std::collections::BTreeMap;

impl StageOpenAiBackend {
    pub(super) fn generate_local_tokens(
        &self,
        request: LocalGeneration<'_>,
        mut on_token: impl FnMut(i32) -> OpenAiResult<TokenControl>,
    ) -> OpenAiResult<GenerationCacheStats> {
        let session_id = request.ids.session_label.clone();
        let mut cache_stats = GenerationCacheStats::default();
        let result = (|| {
            let mut prompt_prefill_sample = None;
            let mut chat_sampling_configured = false;
            let can_sample_whole_prompt_in_prefill = if request.max_tokens > 0
                && request.prompt_token_ids.len() > 1
                && self.kv.is_none()
            {
                let mut runtime = self
                    .runtime
                    .lock()
                    .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
                runtime
                    .ensure_session_active(&session_id)
                    .map_err(openai_backend_error)?;
                let batch_size = runtime
                    .session_batch_size(&session_id)
                    .map_err(openai_backend_error)?;
                prompt_fits_single_prefill_sample(request.prompt_token_ids.len(), batch_size)
            } else {
                false
            };
            if can_sample_whole_prompt_in_prefill {
                if let Some(metadata) = request.chat_sampling_metadata {
                    let mut runtime = self
                        .runtime
                        .lock()
                        .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
                    runtime
                        .configure_chat_sampling(
                            &session_id,
                            metadata,
                            request.prompt_token_ids.len() as u64,
                            request.sampling.enabled.then_some(request.sampling),
                        )
                        .map_err(openai_backend_error)?;
                    chat_sampling_configured = true;
                }
                let prefill_timer = PhaseTimer::start();
                let lock_timer = PhaseTimer::start();
                let mut runtime = self
                    .runtime
                    .lock()
                    .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
                let runtime_lock_wait_ms = lock_timer.elapsed_ms();
                let runtime_lock_hold_timer = PhaseTimer::start();
                let runtime_sessions_before = runtime.session_stats();
                let (predicted, _) = runtime
                    .prefill_final_frame_sampled(
                        &session_id,
                        request.prompt_token_ids,
                        &[],
                        request.sampling.enabled.then_some(request.sampling),
                        None,
                    )
                    .map_err(openai_backend_error)?;
                prompt_prefill_sample = Some(predicted);
                cache_stats.suffix_prefill_tokens = saturating_u32(request.prompt_token_ids.len());
                let runtime_sessions_after = runtime.session_stats();
                let runtime_lock_hold_ms = runtime_lock_hold_timer.elapsed_ms();
                let mut attrs = self.openai_attrs(request.ids);
                attrs.insert(
                    "llama_stage.prefill_token_count".to_string(),
                    json!(request.prompt_token_ids.len()),
                );
                attrs.insert("llama_stage.prefill_chunk_count".to_string(), json!(1));
                attrs.insert("skippy.kv.restored_prefill".to_string(), json!(false));
                attrs.insert("skippy.kv.restored_prefill_tokens".to_string(), json!(0));
                attrs.insert(
                    "skippy.kv.prefill_suffix_tokens".to_string(),
                    json!(request.prompt_token_ids.len()),
                );
                attrs.insert("skippy.kv.recorded_pages".to_string(), json!(0));
                attrs.insert(
                    "llama_stage.runtime_lock_wait_ms".to_string(),
                    json!(runtime_lock_wait_ms),
                );
                attrs.insert(
                    "llama_stage.runtime_lock_hold_ms".to_string(),
                    json!(runtime_lock_hold_ms),
                );
                attrs.insert("llama_stage.runtime_lock_acquires".to_string(), json!(1));
                Self::insert_runtime_session_stats(
                    &mut attrs,
                    "llama_stage.runtime_sessions_before",
                    &runtime_sessions_before,
                );
                Self::insert_runtime_session_stats(
                    &mut attrs,
                    "llama_stage.runtime_sessions_after",
                    &runtime_sessions_after,
                );
                cache_stats.prompt_ms = prefill_timer.elapsed_ms();
                self.emit_openai_phase("stage.openai_prefill", prefill_timer, attrs);
            } else if request.prompt_token_ids.len() > 1 {
                let prefill_timer = PhaseTimer::start();
                let prefill_tokens =
                    &request.prompt_token_ids[..request.prompt_token_ids.len() - 1];
                let mut restored_prefill = false;
                let mut restored_prefill_tokens = 0usize;
                let mut resident_recorded_pages = 0usize;
                let lock_timer = PhaseTimer::start();
                let mut runtime = self
                    .runtime
                    .lock()
                    .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
                let runtime_lock_wait_ms = lock_timer.elapsed_ms();
                let runtime_lock_hold_timer = PhaseTimer::start();
                let runtime_sessions_before = runtime.session_stats();
                if let Some(kv) = self.kv.as_ref() {
                    cache_stats.status = "miss";
                    let base = self.local_kv_message_base(&session_id, request.ids);
                    let kv_identity_timer = PhaseTimer::start();
                    let identities = kv.lookup_identities(&self.config, &base, 0, prefill_tokens);
                    let kv_identity_ms = kv_identity_timer.elapsed_ms();
                    let kv_restore_timer = PhaseTimer::start();
                    match kv.restore_exact_state(&mut runtime, &session_id, &identities) {
                        Ok(Some(restored)) => {
                            restored_prefill = true;
                            cache_stats.status = "hit";
                            cache_stats.hit_kind = Some("exact_prefix");
                            let mut attrs = self.openai_attrs(request.ids);
                            attrs.insert("skippy.kv.decision".to_string(), json!("exact_hit"));
                            attrs.insert(
                                "skippy.exact_cache.hit_page_id".to_string(),
                                json!(restored.page_id),
                            );
                            attrs.insert(
                                "skippy.exact_cache.payload_kind".to_string(),
                                json!(restored.payload_kind.to_string()),
                            );
                            attrs.insert(
                                "skippy.exact_cache.restored_tokens".to_string(),
                                json!(restored.token_count),
                            );
                            attrs.insert(
                                "skippy.kv.matched_prefix_tokens".to_string(),
                                json!(restored.token_count),
                            );
                            attrs.insert(
                                "skippy.kv.suffix_prefill_tokens".to_string(),
                                json!(prefill_tokens.len().saturating_sub(restored.token_count)),
                            );
                            restored_prefill_tokens = restored.token_count;
                            cache_stats.cached_prompt_tokens =
                                saturating_u32(restored_prefill_tokens);
                            attrs.insert(
                                "skippy.exact_cache.logical_bytes".to_string(),
                                json!(restored.logical_bytes),
                            );
                            attrs.insert(
                                "skippy.exact_cache.entries".to_string(),
                                json!(restored.entries),
                            );
                            attrs.insert(
                                "skippy.exact_cache.reconstruct_ms".to_string(),
                                json!(restored.reconstruct_ms),
                            );
                            attrs.insert(
                                "skippy.exact_cache.reconstruct_bytes".to_string(),
                                json!(restored.reconstruct_bytes),
                            );
                            attrs.insert(
                                "skippy.exact_cache.reconstruct_blocks".to_string(),
                                json!(restored.reconstruct_blocks),
                            );
                            self.telemetry
                                .emit("stage.openai_kv_lookup_decision", attrs);
                        }
                        Ok(None) => match kv.restore_resident_prefix(
                            &mut runtime,
                            &session_id,
                            &identities,
                            prefill_tokens,
                        ) {
                            Ok(Some(restored)) => {
                                restored_prefill = true;
                                cache_stats.status = "hit";
                                cache_stats.hit_kind = Some("resident_prefix");
                                let mut attrs = self.openai_attrs(request.ids);
                                attrs.insert(
                                    "skippy.kv.decision".to_string(),
                                    json!("resident_hit"),
                                );
                                attrs.insert(
                                    "skippy.kv.hit_page_id".to_string(),
                                    json!(restored.page_id),
                                );
                                attrs.insert(
                                    "skippy.kv.restored_tokens".to_string(),
                                    json!(restored.token_count),
                                );
                                attrs.insert(
                                    "skippy.kv.matched_prefix_tokens".to_string(),
                                    json!(restored.token_count),
                                );
                                attrs.insert(
                                    "skippy.kv.suffix_prefill_tokens".to_string(),
                                    json!(
                                        prefill_tokens.len().saturating_sub(restored.token_count)
                                    ),
                                );
                                restored_prefill_tokens = restored.token_count;
                                cache_stats.cached_prompt_tokens =
                                    saturating_u32(restored_prefill_tokens);
                                attrs.insert(
                                    "skippy.kv.resident_seq_id".to_string(),
                                    json!(restored.seq_id),
                                );
                                attrs.insert(
                                    "skippy.kv.resident_lane_hit".to_string(),
                                    json!(restored.borrowed),
                                );
                                self.telemetry
                                    .emit("stage.openai_kv_lookup_decision", attrs);
                            }
                            Ok(None) => {
                                self.telemetry.emit(
                                    "stage.openai_kv_lookup_decision",
                                    BTreeMap::from([
                                        ("skippy.kv.decision".to_string(), json!("miss")),
                                        (
                                            "llama_stage.request_id".to_string(),
                                            json!(request.ids.request_id_string()),
                                        ),
                                    ]),
                                );
                            }
                            Err(error) => {
                                let mut attrs = self.openai_attrs(request.ids);
                                attrs.insert(
                                    "skippy.kv.decision".to_string(),
                                    json!("resident_error"),
                                );
                                attrs.insert(
                                    "skippy.kv.error".to_string(),
                                    json!(error.to_string()),
                                );
                                self.telemetry
                                    .emit("stage.openai_kv_lookup_decision", attrs);
                            }
                        },
                        Err(error) => {
                            let mut attrs = self.openai_attrs(request.ids);
                            attrs.insert("skippy.kv.decision".to_string(), json!("exact_error"));
                            attrs.insert("skippy.kv.error".to_string(), json!(error.to_string()));
                            self.telemetry
                                .emit("stage.openai_kv_lookup_decision", attrs);
                        }
                    }
                    let mut attrs = self.openai_attrs(request.ids);
                    attrs.insert("skippy.kv.identity_ms".to_string(), json!(kv_identity_ms));
                    attrs.insert(
                        "skippy.kv.restore_ms".to_string(),
                        json!(kv_restore_timer.elapsed_ms()),
                    );
                    attrs.insert(
                        "skippy.kv.identity_count".to_string(),
                        json!(identities.len()),
                    );
                    self.telemetry.emit_debug("stage.openai_kv_timing", attrs);
                }
                let mut decoded_prefill_suffix = false;
                if restored_prefill_tokens < prefill_tokens.len() {
                    decoded_prefill_suffix = true;
                    runtime
                        .prefill(&session_id, &prefill_tokens[restored_prefill_tokens..])
                        .map_err(openai_backend_error)?;
                }
                cache_stats.matched_prefix_tokens = saturating_u32(restored_prefill_tokens);
                cache_stats.suffix_prefill_tokens =
                    saturating_u32(prefill_tokens.len().saturating_sub(restored_prefill_tokens));
                if let (true, Some(kv)) = (
                    !restored_prefill || decoded_prefill_suffix,
                    self.kv.as_ref(),
                ) {
                    let base = self.local_kv_message_base(&session_id, request.ids);
                    let exact_identity =
                        kv.prefill_identity(&self.config, &base, 0, prefill_tokens);
                    if let Ok(Some(record)) =
                        kv.record_exact_state(&mut runtime, &session_id, &exact_identity)
                    {
                        resident_recorded_pages = resident_recorded_pages.saturating_add(1);
                        let mut attrs = self.openai_attrs(request.ids);
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
                        attrs.insert(
                            "skippy.exact_cache.stored".to_string(),
                            json!(record.stored),
                        );
                        attrs.insert(
                            "skippy.exact_cache.logical_bytes".to_string(),
                            json!(record.logical_bytes),
                        );
                        attrs.insert(
                            "skippy.exact_cache.physical_bytes".to_string(),
                            json!(record.physical_bytes),
                        );
                        attrs.insert(
                            "skippy.exact_cache.entries".to_string(),
                            json!(record.entries),
                        );
                        attrs.insert(
                            "skippy.exact_cache.evicted_entries".to_string(),
                            json!(record.evicted_entries),
                        );
                        attrs.insert(
                            "skippy.exact_cache.evicted_logical_bytes".to_string(),
                            json!(record.evicted_logical_bytes),
                        );
                        attrs.insert(
                            "skippy.exact_cache.dedupe_hash_ms".to_string(),
                            json!(record.dedupe.hash_ms),
                        );
                        attrs.insert(
                            "skippy.exact_cache.dedupe_block_count".to_string(),
                            json!(record.dedupe.block_count),
                        );
                        attrs.insert(
                            "skippy.exact_cache.dedupe_new_block_count".to_string(),
                            json!(record.dedupe.new_block_count),
                        );
                        attrs.insert(
                            "skippy.exact_cache.dedupe_reused_block_count".to_string(),
                            json!(record.dedupe.reused_block_count),
                        );
                        self.telemetry
                            .emit("stage.openai_kv_record_decision", attrs);
                    }
                    for identity in kv.record_identities(&self.config, &base, 0, prefill_tokens) {
                        if let Ok(Some(record)) = kv.record_resident_prefix(
                            &mut runtime,
                            &session_id,
                            &identity,
                            prefill_tokens,
                        ) {
                            resident_recorded_pages = resident_recorded_pages.saturating_add(1);
                            let mut attrs = self.openai_attrs(request.ids);
                            attrs.insert(
                                "skippy.kv.recorded_page_id".to_string(),
                                json!(record.page_id),
                            );
                            attrs.insert(
                                "skippy.kv.recorded_tokens".to_string(),
                                json!(record.token_count),
                            );
                            attrs.insert(
                                "skippy.kv.resident_seq_id".to_string(),
                                json!(record.seq_id),
                            );
                            attrs.insert(
                                "skippy.kv.resident_entries".to_string(),
                                json!(record.entries),
                            );
                            attrs.insert(
                                "skippy.kv.evicted_entries".to_string(),
                                json!(record.evicted_entries),
                            );
                            self.telemetry
                                .emit("stage.openai_kv_record_decision", attrs);
                        }
                    }
                }
                // Proactive eviction: after prefill recording, evict enough
                // LRU resident-prefix entries to free one native decode batch
                // for grammar-triggered retries during the decode loop.
                let mut proactive_eviction_status = "disabled";
                let mut proactive_eviction_error_kind_attr = None;
                let mut proactive_eviction_target_tokens = 0_u64;
                let mut proactive_evicted_entries = 0_usize;
                let mut proactive_evicted_tokens = 0_u64;
                let mut proactive_eviction_error = None;
                if let Some(kv) = self.kv.as_ref() {
                    match kv.evict_resident_prefix_for_decode_batch(&mut runtime, &session_id) {
                        Ok(eviction) => {
                            proactive_eviction_status = if eviction.evicted_entries > 0 {
                                "evicted"
                            } else {
                                "noop"
                            };
                            proactive_eviction_target_tokens = eviction.target_tokens;
                            proactive_evicted_entries = eviction.evicted_entries;
                            proactive_evicted_tokens = eviction.evicted_tokens;
                        }
                        Err(error) => {
                            proactive_eviction_status = "error";
                            proactive_eviction_error_kind_attr =
                                Some(proactive_eviction_error_kind(&error));
                            proactive_eviction_error = Some(
                                error
                                    .context("evict resident-prefix KV before local OpenAI decode"),
                            );
                        }
                    }
                }
                let runtime_sessions_after = runtime.session_stats();
                let runtime_lock_hold_ms = runtime_lock_hold_timer.elapsed_ms();
                let mut attrs = self.openai_attrs(request.ids);
                attrs.insert(
                    "llama_stage.prefill_token_count".to_string(),
                    json!(prefill_tokens.len()),
                );
                attrs.insert("llama_stage.prefill_chunk_count".to_string(), json!(1));
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
                    json!(prefill_tokens.len().saturating_sub(restored_prefill_tokens)),
                );
                attrs.insert(
                    "skippy.kv.recorded_pages".to_string(),
                    json!(resident_recorded_pages),
                );
                attrs.insert(
                    "llama_stage.runtime_lock_wait_ms".to_string(),
                    json!(runtime_lock_wait_ms),
                );
                attrs.insert(
                    "llama_stage.runtime_lock_hold_ms".to_string(),
                    json!(runtime_lock_hold_ms),
                );
                attrs.insert("llama_stage.runtime_lock_acquires".to_string(), json!(1));
                Self::insert_runtime_session_stats(
                    &mut attrs,
                    "llama_stage.runtime_sessions_before",
                    &runtime_sessions_before,
                );
                Self::insert_runtime_session_stats(
                    &mut attrs,
                    "llama_stage.runtime_sessions_after",
                    &runtime_sessions_after,
                );
                cache_stats.prompt_ms = prefill_timer.elapsed_ms();
                self.emit_openai_phase("stage.openai_prefill", prefill_timer, attrs);
                self.telemetry.emit(
                    "stage.openai_kv_record_decision",
                    proactive_eviction_attrs(
                        proactive_eviction_status,
                        proactive_eviction_error_kind_attr,
                        proactive_eviction_target_tokens,
                        proactive_evicted_entries,
                        proactive_evicted_tokens,
                    ),
                );
                if let Some(error) = proactive_eviction_error {
                    return Err(openai_backend_error(error));
                }
            }
            let chat_sampling_metadata = (!chat_sampling_configured)
                .then_some(request.chat_sampling_metadata)
                .flatten();
            if let Some(metadata) = chat_sampling_metadata {
                let mut runtime = self
                    .runtime
                    .lock()
                    .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
                runtime
                    .configure_chat_sampling(
                        &session_id,
                        metadata,
                        request.prompt_token_ids.len() as u64,
                        request.sampling.enabled.then_some(request.sampling),
                    )
                    .map_err(openai_backend_error)?;
            }
            let decode_timer = PhaseTimer::start();
            let mut decoded_tokens = 0usize;
            let mut runtime_lock_wait_ms = 0.0;
            let mut runtime_lock_wait_max_ms = 0.0_f64;
            let mut runtime_lock_hold_ms = 0.0;
            let mut runtime_lock_hold_max_ms = 0.0_f64;
            let mut runtime_lock_acquires = 0usize;
            let mut runtime_sessions_before = None;
            let mut runtime_sessions_after = None;
            let mut current = *request
                .prompt_token_ids
                .last()
                .expect("checked non-empty prompt");
            let mut stopped = false;
            if let Some(predicted) = prompt_prefill_sample {
                current = predicted;
                decoded_tokens += 1;
                stopped = on_token(current)? == TokenControl::Stop;
            }
            let mut hook_request = request.hook_request;
            let hook_runtime = request.hook_runtime;
            let generation_hooks_active =
                self.generation_hooks_active(&hook_request, hook_runtime.as_ref());
            let emit_token_debug = self.telemetry.is_debug_enabled();
            let native_mtp_options = NativeMtpDecodeOptions::from_config(request.speculative);
            let mut native_mtp = NativeMtpVerifier::default();
            let mut post_prefill_hook_checked = false;
            let mut last_mid_generation_hook_at = None;
            while !stopped && decoded_tokens < request.max_tokens as usize {
                if request
                    .cancellation
                    .is_some_and(openai_frontend::CancellationToken::is_cancelled)
                {
                    break;
                }
                let decode_step = decoded_tokens;
                let token_timer = PhaseTimer::start();
                let token_signal_ms;
                let token_signal;
                let signal_window;
                let decode_call_timer = PhaseTimer::start();
                let mut native_mtp_draft;
                let token_batch_size;
                let token_batch_wait_ms;
                let token_runtime_lock_wait_ms;
                let token_runtime_lock_hold_ms;
                if request.native_mtp_enabled {
                    let lock_timer = PhaseTimer::start();
                    let mut runtime = self
                        .runtime
                        .lock()
                        .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
                    token_runtime_lock_wait_ms = lock_timer.elapsed_ms();
                    let hold_timer = PhaseTimer::start();
                    let (predicted, draft) = runtime
                        .decode_sampled_mtp(
                            &session_id,
                            current,
                            request.sampling.enabled.then_some(request.sampling),
                            native_mtp_options.max_draft_tokens,
                        )
                        .map_err(openai_backend_error)?;
                    current = predicted;
                    native_mtp_draft = draft.map(|draft| NativeMtpDraft {
                        tokens: draft.token_ids,
                        proposal_compute_us: draft.proposal_compute_us,
                    });
                    token_batch_size = 1;
                    token_batch_wait_ms = 0.0;
                    token_runtime_lock_hold_ms = hold_timer.elapsed_ms();
                } else {
                    let outcome = self.decode_batcher.decode(
                        &session_id,
                        current,
                        request.sampling.enabled.then_some(request.sampling),
                    )?;
                    current = outcome.predicted;
                    native_mtp_draft = None;
                    token_batch_size = outcome.batch_size;
                    token_batch_wait_ms = outcome.batch_wait_ms;
                    token_runtime_lock_wait_ms = outcome.runtime_lock_wait_ms;
                    token_runtime_lock_hold_ms = outcome.runtime_lock_hold_ms;
                }
                if native_mtp_draft
                    .as_ref()
                    .is_some_and(|draft: &NativeMtpDraft| {
                        draft.tokens.len() < native_mtp_options.min_draft_tokens
                    })
                {
                    native_mtp_draft = None;
                }
                let is_first_draft = decoded_tokens == 0;
                let draft_origin = if is_first_draft {
                    NativeMtpDraftOrigin::InitialSerial
                } else {
                    NativeMtpDraftOrigin::SerialAfterGap
                };
                let native_mtp_decision = request.native_mtp_enabled.then(|| {
                    native_mtp.observe_target_token(current, 0, native_mtp_draft, draft_origin)
                });
                runtime_lock_wait_ms += token_runtime_lock_wait_ms;
                runtime_lock_wait_max_ms = runtime_lock_wait_max_ms.max(token_runtime_lock_wait_ms);
                runtime_lock_hold_ms += token_runtime_lock_hold_ms;
                runtime_lock_hold_max_ms = runtime_lock_hold_max_ms.max(token_runtime_lock_hold_ms);
                runtime_lock_acquires += 1;
                let token_decode_ms = if emit_token_debug {
                    decode_call_timer.elapsed_ms()
                } else {
                    0.0
                };
                if generation_hooks_active {
                    let signal_timer = PhaseTimer::start();
                    let mut runtime = self
                        .runtime
                        .lock()
                        .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
                    runtime_sessions_before.get_or_insert_with(|| runtime.session_stats());
                    token_signal = runtime.last_token_signal(&session_id).ok();
                    signal_window = runtime.signal_window(&session_id, 16).ok();
                    runtime_sessions_after = Some(runtime.session_stats());
                    token_signal_ms = signal_timer.elapsed_ms();
                } else {
                    token_signal = None;
                    signal_window = None;
                    token_signal_ms = 0.0;
                }
                let injected_current = if generation_hooks_active {
                    self.maybe_run_generation_hooks(
                        &session_id,
                        &mut hook_request,
                        hook_runtime.as_ref(),
                        decoded_tokens,
                        &mut post_prefill_hook_checked,
                        &mut last_mid_generation_hook_at,
                        token_signal,
                        signal_window,
                    )?
                } else {
                    None
                };
                if let Some(injected_current) = injected_current {
                    current = injected_current;
                    continue;
                }
                decoded_tokens += 1;
                if emit_token_debug {
                    let mut token_attrs = self.openai_attrs(request.ids);
                    token_attrs.insert("llama_stage.decode_step".to_string(), json!(decode_step));
                    token_attrs.insert(
                        "llama_stage.decode_token_phase".to_string(),
                        json!(decode_token_phase(
                            u32::try_from(decode_step).unwrap_or(u32::MAX)
                        )),
                    );
                    token_attrs.insert(
                        "llama_stage.stage0_compute_ms".to_string(),
                        json!(token_timer.elapsed_ms()),
                    );
                    token_attrs.insert(
                        "llama_stage.decode_call_ms".to_string(),
                        json!(token_decode_ms),
                    );
                    token_attrs.insert(
                        "llama_stage.decode_batch_size".to_string(),
                        json!(token_batch_size),
                    );
                    token_attrs.insert(
                        "llama_stage.decode_batch_wait_ms".to_string(),
                        json!(token_batch_wait_ms),
                    );
                    token_attrs.insert("llama_stage.signal_ms".to_string(), json!(token_signal_ms));
                    token_attrs.insert(
                        "llama_stage.runtime_lock_wait_ms".to_string(),
                        json!(token_runtime_lock_wait_ms),
                    );
                    token_attrs.insert(
                        "llama_stage.runtime_lock_hold_ms".to_string(),
                        json!(token_runtime_lock_hold_ms),
                    );
                    token_attrs.insert("llama_stage.predicted_token".to_string(), json!(current));
                    if let Some(native_mtp_decision) = native_mtp_decision {
                        token_attrs.insert(
                            "llama_stage.native_mtp.verification".to_string(),
                            json!(native_mtp_decision.label()),
                        );
                    }
                    token_attrs
                        .insert("llama_stage.message_kind".to_string(), json!("DecodeToken"));
                    self.emit_openai_phase("stage.openai_decode_token", token_timer, token_attrs);
                }
                if on_token(current)? == TokenControl::Stop {
                    break;
                }
            }
            let mut attrs = self.openai_attrs(request.ids);
            attrs.insert(
                "llama_stage.decode_token_count".to_string(),
                json!(decoded_tokens),
            );
            attrs.insert(
                "llama_stage.runtime_lock_wait_ms".to_string(),
                json!(runtime_lock_wait_ms),
            );
            attrs.insert(
                "llama_stage.runtime_lock_wait_max_ms".to_string(),
                json!(runtime_lock_wait_max_ms),
            );
            attrs.insert(
                "llama_stage.runtime_lock_hold_ms".to_string(),
                json!(runtime_lock_hold_ms),
            );
            attrs.insert(
                "llama_stage.runtime_lock_hold_max_ms".to_string(),
                json!(runtime_lock_hold_max_ms),
            );
            attrs.insert(
                "llama_stage.runtime_lock_acquires".to_string(),
                json!(runtime_lock_acquires),
            );
            if let Some(stats) = runtime_sessions_before.as_ref() {
                Self::insert_runtime_session_stats(
                    &mut attrs,
                    "llama_stage.runtime_sessions_before",
                    stats,
                );
            }
            if let Some(stats) = runtime_sessions_after.as_ref() {
                Self::insert_runtime_session_stats(
                    &mut attrs,
                    "llama_stage.runtime_sessions_after",
                    stats,
                );
            }
            request.speculative.insert_telemetry_attrs(&mut attrs);
            let native_mtp_stats = native_mtp.stats();
            cache_stats.native_mtp_stats = native_mtp_stats;
            cache_stats.predicted_ms = decode_timer.elapsed_ms();
            native_mtp_stats.insert_attrs(&mut attrs);
            self.emit_openai_summary("stage.openai_decode", decode_timer, attrs);
            Ok(())
        })();
        let lock_timer = PhaseTimer::start();
        if let Ok(mut runtime) = self.runtime.lock() {
            let runtime_lock_wait_ms = lock_timer.elapsed_ms();
            if let Ok(drop_stats) = runtime.drop_session_timed(&session_id) {
                let mut attrs = self.openai_attrs(request.ids);
                attrs.insert(
                    "llama_stage.runtime_lock_wait_ms".to_string(),
                    json!(runtime_lock_wait_ms),
                );
                attrs.insert(
                    "llama_stage.session_reset_ms".to_string(),
                    json!(drop_stats.reset_ms),
                );
                attrs.insert(
                    "llama_stage.session_reset".to_string(),
                    json!(drop_stats.reset_session),
                );
                attrs.insert(
                    "llama_stage.lane_discarded".to_string(),
                    json!(drop_stats.lane_discarded),
                );
                if let Some(reason) = drop_stats.lane_discard_reason.as_deref() {
                    attrs.insert("llama_stage.lane_discard_reason".to_string(), json!(reason));
                }
                Self::insert_runtime_session_stats(
                    &mut attrs,
                    "llama_stage.runtime_sessions_after",
                    &drop_stats.stats_after,
                );
                self.telemetry
                    .emit_debug("stage.openai_session_stop", attrs);
            }
        }
        result?;
        Ok(cache_stats)
    }
}

fn prompt_fits_single_prefill_sample(prompt_token_count: usize, session_batch_size: usize) -> bool {
    prompt_token_count > 1 && prompt_token_count <= session_batch_size
}

#[cfg(test)]
mod tests {
    use super::prompt_fits_single_prefill_sample;

    #[test]
    fn single_prefill_sample_requires_prompt_to_fit_session_batch() {
        assert!(!prompt_fits_single_prefill_sample(0, 2048));
        assert!(!prompt_fits_single_prefill_sample(1, 2048));
        assert!(prompt_fits_single_prefill_sample(2048, 2048));
        assert!(!prompt_fits_single_prefill_sample(2049, 2048));
    }

    #[test]
    fn local_native_mtp_decode_uses_non_frame_runtime_api() {
        let source = include_str!("local_generation.rs");
        assert!(source.contains(concat!(".decode", "_sampled_mtp(")));
        assert!(!source.contains(concat!(".decode_frame", "_sampled_mtp(")));
    }
}
