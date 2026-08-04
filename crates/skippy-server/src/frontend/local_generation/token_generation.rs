use crate::frontend::LinearProposalDisposition;
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
use crate::frontend::generation_receipt::{
    GenerationReceiptObservation, complete_generation_before_cleanup,
};
use crate::frontend::linear_proposal::{
    LinearProposalDiscardReason, LinearProposalExecutionParams, LinearProposalQueryOutcome,
    execute_linear_proposal_with_terminal_discard, greedy_linear_proposal_admitted,
    query_linear_proposal, report_linear_proposal_receipt,
};
use crate::frontend::util::openai_backend_error;
use crate::frontend::util::saturating_u32;
use crate::kv_integration::proactive_eviction_attrs;
use crate::kv_integration::proactive_eviction_error_kind;
use openai_frontend::OpenAiError;
use openai_frontend::OpenAiResult;
use serde_json::json;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::time::Duration;

use super::{LocalGenerationReceiptFinalization, prompt_fits_single_prefill_sample};

impl StageOpenAiBackend {
    pub(in crate::frontend) fn generate_local_tokens(
        &self,
        request: LocalGeneration<'_>,
        mut on_token: impl FnMut(i32) -> OpenAiResult<TokenControl>,
    ) -> OpenAiResult<GenerationCacheStats> {
        let session_id = request.ids.session_label.clone();
        let receipt_request_id = request.ids.request_id;
        let receipt_session_id = request.ids.session_id;
        let receipt_prompt_token_ids = request.prompt_token_ids;
        let receipt_observation = self.generation_receipt.as_ref().map(|_| {
            RefCell::new(Some(GenerationReceiptObservation::new(
                usize::try_from(request.max_tokens)
                    .expect("supported targets represent u32 token budgets as usize"),
            )))
        });
        let mut receipt_cancelled = false;
        let mut receipt_model_generation_elapsed = None;
        let mut cache_stats = GenerationCacheStats::default();
        let mut emit_token = |token_id| {
            if let Some(observation) = receipt_observation.as_ref()
                && let Some(observation) = observation.borrow_mut().as_mut()
            {
                observation.record_token(token_id)?;
            }
            let control = on_token(token_id)?;
            if control == TokenControl::Stop
                && let Some(observation) = receipt_observation.as_ref()
                && let Some(observation) = observation.borrow_mut().as_mut()
            {
                observation.mark_callback_stop();
            }
            Ok(control)
        };
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
                stopped = emit_token(current)? == TokenControl::Stop;
            }
            let mut hook_request = request.hook_request;
            let hook_runtime = request.hook_runtime;
            let generation_hooks_active =
                self.generation_hooks_active(&hook_request, hook_runtime.as_ref());
            let linear_proposals_enabled = self.linear_proposal_ingress.is_some()
                && !request.native_mtp_enabled
                && !generation_hooks_active
                && greedy_linear_proposal_admitted(
                    request.sampling,
                    request.chat_sampling_metadata,
                );
            let linear_proposal_max_tokens = if linear_proposals_enabled {
                let mut runtime = self
                    .runtime
                    .lock()
                    .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
                runtime
                    .session_batch_size(&session_id)
                    .map_err(openai_backend_error)?
                    .saturating_sub(1)
            } else {
                0
            };
            let mut linear_context_tokens = (linear_proposal_max_tokens > 0).then(|| {
                let mut tokens = request.prompt_token_ids.to_vec();
                if decoded_tokens > 0 {
                    tokens.push(current);
                }
                tokens
            });
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
                    receipt_cancelled = true;
                    break;
                }
                if let (Some(config), Some(committed_token_ids)) = (
                    self.linear_proposal_ingress.as_ref(),
                    linear_context_tokens.as_mut(),
                ) {
                    let remaining_new_tokens =
                        (request.max_tokens as usize).saturating_sub(decoded_tokens);
                    // Prefill leaves the final prompt token undecoded. When whole-prompt
                    // prefill also samples the first target token, `decoded_tokens == 1`;
                    // otherwise it is zero. Those two modes therefore share this position.
                    let base_position = u64::try_from(
                        request
                            .prompt_token_ids
                            .len()
                            .saturating_sub(1)
                            .checked_add(decoded_tokens)
                            .ok_or_else(|| {
                                OpenAiError::backend("linear proposal base position exceeds usize")
                            })?,
                    )
                    .map_err(|_| {
                        OpenAiError::backend("linear proposal base position exceeds u64")
                    })?;
                    let queried = match query_linear_proposal(
                        config,
                        request.ids.request_id,
                        request.ids.session_id,
                        decoded_tokens,
                        committed_token_ids,
                        remaining_new_tokens,
                        linear_proposal_max_tokens,
                    )? {
                        LinearProposalQueryOutcome::NoProposal => None,
                        LinearProposalQueryOutcome::DeadlineExceeded {
                            proposal_elapsed_us,
                        } => {
                            let mut attrs = BTreeMap::new();
                            attrs.insert(
                                "llama_stage.linear_proposal.discard_reason".to_string(),
                                json!("deadline_exceeded"),
                            );
                            attrs.insert(
                                "llama_stage.linear_proposal.proposal_us".to_string(),
                                json!(proposal_elapsed_us),
                            );
                            self.telemetry
                                .emit("stage.openai_linear_proposal_late", attrs);
                            None
                        }
                        LinearProposalQueryOutcome::Ready(queried) => Some(queried),
                    };
                    if let Some(queried) = queried {
                        let decision_id = queried.proposal.decision_id.clone();
                        let receipt = execute_linear_proposal_with_terminal_discard(
                            config,
                            &decision_id,
                            || {
                                self.execute_local_linear_proposal(
                                    LinearProposalExecutionParams {
                                        session_id: &session_id,
                                        current,
                                        base_position,
                                        generated_len: decoded_tokens,
                                        max_new_tokens: request.max_tokens as usize,
                                        sampling: request.sampling,
                                        chat_sampling_metadata: request.chat_sampling_metadata,
                                        prompt_token_count: request.prompt_token_ids.len(),
                                    },
                                    queried,
                                    &mut emit_token,
                                )
                            },
                        )?;
                        if receipt.is_none() {
                            let discard_failed = config
                                .source()
                                .discard(
                                    &decision_id,
                                    LinearProposalDiscardReason::PositionMismatch,
                                )
                                .is_err();
                            if discard_failed {
                                self.telemetry.emit(
                                    "stage.openai_linear_proposal_discard_failed",
                                    BTreeMap::from([(
                                        "llama_stage.linear_proposal.discard_reason".to_string(),
                                        json!("position_mismatch"),
                                    )]),
                                );
                            }
                        }
                        if let Some(receipt) = receipt {
                            if report_linear_proposal_receipt(config, &receipt).is_some() {
                                let mut attrs = BTreeMap::new();
                                receipt.insert_telemetry_attrs(&mut attrs);
                                attrs.insert(
                                    "llama_stage.linear_proposal.report_outcome".to_string(),
                                    json!("failed"),
                                );
                                self.telemetry
                                    .emit("stage.openai_linear_proposal_report_failed", attrs);
                            }

                            let proposal_runtime_lock_wait_ms =
                                Duration::from_micros(receipt.runtime_lock_wait_us).as_secs_f64()
                                    * 1_000.0;
                            let proposal_runtime_lock_hold_ms =
                                Duration::from_micros(receipt.runtime_lock_hold_us).as_secs_f64()
                                    * 1_000.0;
                            runtime_lock_wait_ms += proposal_runtime_lock_wait_ms;
                            runtime_lock_wait_max_ms =
                                runtime_lock_wait_max_ms.max(proposal_runtime_lock_wait_ms);
                            runtime_lock_hold_ms += proposal_runtime_lock_hold_ms;
                            runtime_lock_hold_max_ms =
                                runtime_lock_hold_max_ms.max(proposal_runtime_lock_hold_ms);
                            runtime_lock_acquires =
                                runtime_lock_acquires.saturating_add(receipt.runtime_lock_acquires);

                            decoded_tokens = decoded_tokens
                                .checked_add(receipt.committed_tokens.len())
                                .ok_or_else(|| {
                                    OpenAiError::backend("linear proposal decode count overflow")
                                })?;
                            current = *receipt.committed_tokens.last().ok_or_else(|| {
                                OpenAiError::backend("linear proposal receipt committed no tokens")
                            })?;
                            committed_token_ids.extend_from_slice(&receipt.committed_tokens);
                            let stopped_by_proposal =
                                receipt.disposition == LinearProposalDisposition::Stopped;
                            if emit_token_debug {
                                let mut proposal_attrs = BTreeMap::new();
                                receipt.insert_telemetry_attrs(&mut proposal_attrs);
                                self.telemetry
                                    .emit_debug("stage.openai_linear_proposal", proposal_attrs);
                            }
                            if stopped_by_proposal || decoded_tokens >= request.max_tokens as usize
                            {
                                break;
                            }
                            continue;
                        }
                    }
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
                if let Some(committed_token_ids) = linear_context_tokens.as_mut() {
                    committed_token_ids.push(current);
                }
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
                if emit_token(current)? == TokenControl::Stop {
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
            let model_generation_elapsed = decode_timer.start_instant.elapsed();
            cache_stats.predicted_ms = model_generation_elapsed.as_secs_f64() * 1_000.0;
            receipt_model_generation_elapsed = Some(model_generation_elapsed);
            native_mtp_stats.insert_attrs(&mut attrs);
            self.emit_openai_summary("stage.openai_decode", decode_timer, attrs);
            Ok(())
        })();
        let receipt_observation = receipt_observation
            .as_ref()
            .and_then(|observation| observation.borrow_mut().take());
        complete_generation_before_cleanup(
            result,
            || {
                self.finalize_generation_receipt(LocalGenerationReceiptFinalization {
                    session_label: &session_id,
                    request_id: receipt_request_id,
                    session_id: receipt_session_id,
                    prompt_token_ids: receipt_prompt_token_ids,
                    observation: receipt_observation,
                    cancelled: receipt_cancelled,
                    model_generation_elapsed: receipt_model_generation_elapsed,
                })
            },
            || self.cleanup_local_generation_session(&session_id, request.ids),
        )?;
        Ok(cache_stats)
    }
}
