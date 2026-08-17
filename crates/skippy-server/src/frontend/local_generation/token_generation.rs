use crate::frontend::NativeMtpDecodeOptions;
use crate::frontend::NativeMtpDraft;
use crate::frontend::NativeMtpVerifier;
use crate::frontend::generation::GenerationCacheStats;
use crate::frontend::generation::LocalGeneration;
use crate::frontend::generation::OpenAiGenerationIds;
use crate::frontend::generation::PhaseTimer;
use crate::frontend::generation::StageOpenAiBackend;
use crate::frontend::generation::TokenControl;
use crate::frontend::generation_receipt::{
    GenerationCommit, GenerationStart, complete_generation_before_cleanup,
};
use crate::frontend::linear_proposal::greedy_linear_proposal_admitted;
use crate::frontend::util::openai_backend_error;
use crate::frontend::util::saturating_u32;
use crate::kv_integration::KvStageIntegration;
use crate::kv_integration::proactive_eviction_attrs;
use crate::kv_integration::proactive_eviction_error_kind;
use crate::runtime_state::{RuntimeSessionStats, RuntimeState};
use openai_frontend::ChatCompletionRequest;
use openai_frontend::OpenAiError;
use openai_frontend::OpenAiResult;
use serde_json::json;
use skippy_runtime::ActivationFrame;
use skippy_runtime::NativeMtpDraft as RuntimeNativeMtpDraft;
use skippy_runtime::SamplingConfig;
use std::cell::RefCell;
use std::sync::Arc;
use std::time::Duration;

use super::{LocalGenerationReceiptFinalization, prompt_fits_single_prefill_sample};

pub(super) fn commit_local_generation_token(
    config: Option<&crate::frontend::GenerationReceiptConfig>,
    request_id: u64,
    session_id: u64,
    generated_token_count: &mut usize,
    token_id: i32,
) {
    let Some(config) = config else {
        return;
    };
    *generated_token_count = generated_token_count.saturating_add(1);
    config.committed(GenerationCommit {
        request_id,
        session_id,
        generated_token_count: *generated_token_count,
        token_ids: vec![token_id].into_boxed_slice(),
    });
}

struct PromptPrefillResult {
    prompt_prefill_sample: Option<i32>,
    chat_sampling_configured: bool,
}

struct KvRecordResult {
    resident_recorded_pages: usize,
    proactive_eviction_status: &'static str,
    proactive_eviction_error_kind: Option<&'static str>,
    proactive_eviction_target_tokens: u64,
    proactive_evicted_entries: usize,
    proactive_evicted_tokens: u64,
    proactive_eviction_error: Option<anyhow::Error>,
}

pub(super) struct DecodeState {
    pub(super) decoded_tokens: usize,
    pub(super) current: i32,
    pub(super) stopped: bool,
    pub(super) runtime_lock_wait_ms: f64,
    pub(super) runtime_lock_wait_max_ms: f64,
    pub(super) runtime_lock_hold_ms: f64,
    pub(super) runtime_lock_hold_max_ms: f64,
    pub(super) runtime_lock_acquires: usize,
    pub(super) runtime_sessions_before: Option<RuntimeSessionStats>,
    pub(super) runtime_sessions_after: Option<RuntimeSessionStats>,
    pub(super) hook_request: Option<ChatCompletionRequest>,
    pub(super) hook_runtime: Option<tokio::runtime::Handle>,
    pub(super) generation_hooks_active: bool,
    pub(super) linear_proposal_max_tokens: usize,
    pub(super) linear_context_tokens: Option<Vec<i32>>,
    pub(super) pending_linear_proposal_tokens: Vec<i32>,
    pub(super) emit_token_debug: bool,
    pub(super) native_mtp_options: NativeMtpDecodeOptions,
    pub(super) native_mtp: NativeMtpVerifier,
    pub(super) post_prefill_hook_checked: bool,
    pub(super) last_mid_generation_hook_at: Option<usize>,
}

pub(super) enum LinearProposalProgress {
    NotUsed,
    Continue,
    Stop,
}

impl StageOpenAiBackend {
    pub(in crate::frontend) fn generate_local_tokens(
        &self,
        mut request: LocalGeneration<'_>,
        mut on_token: impl FnMut(i32) -> OpenAiResult<TokenControl>,
    ) -> OpenAiResult<GenerationCacheStats> {
        let session_id = request.ids.session_label.clone();
        let receipt_request_id = request.ids.request_id;
        let receipt_session_id = request.ids.session_id;
        let receipt_prompt_token_ids = self
            .generation_receipt
            .as_ref()
            .map(|_| Arc::<[i32]>::from(request.prompt_token_ids));
        if let Some(config) = self.generation_receipt.as_ref() {
            config.begin(GenerationStart {
                request_id: receipt_request_id,
                session_id: receipt_session_id,
                agent_session_id: request.ids.agent_session_id.clone(),
                prompt_token_ids: Arc::clone(
                    receipt_prompt_token_ids
                        .as_ref()
                        .expect("receipt prompt exists when receipt config exists"),
                ),
            });
        }
        let receipt_observation = self.generation_receipt.as_ref().map(|config| {
            RefCell::new(Some(
                config.observation(
                    usize::try_from(request.max_tokens)
                        .expect("supported targets represent u32 token budgets as usize"),
                ),
            ))
        });
        let mut receipt_cancelled = false;
        let mut receipt_model_generation_elapsed = None;
        let mut cache_stats = GenerationCacheStats::default();
        let mut lifecycle_committed_token_count = 0usize;
        let mut emit_token = |token_id| {
            if let Some(observation) = receipt_observation.as_ref()
                && let Some(observation) = observation.borrow_mut().as_mut()
            {
                observation.record_token(token_id, request.ids.request_started_at.elapsed());
            }
            commit_local_generation_token(
                self.generation_receipt.as_ref(),
                receipt_request_id,
                receipt_session_id,
                &mut lifecycle_committed_token_count,
                token_id,
            );
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
            if request
                .cancellation
                .is_some_and(openai_frontend::CancellationToken::is_cancelled)
            {
                return Err(OpenAiError::backend("request cancelled"));
            }
            let prefill = self.prefill_prompt(&request, &session_id, &mut cache_stats)?;
            self.configure_chat_sampling_if_needed(
                &request,
                &session_id,
                prefill.chat_sampling_configured,
            )?;
            let model_generation_elapsed = self.run_decode_loop(
                &mut request,
                &session_id,
                prefill.prompt_prefill_sample,
                &mut cache_stats,
                &mut receipt_cancelled,
                &mut emit_token,
            )?;
            receipt_model_generation_elapsed = Some(model_generation_elapsed);
            Ok(())
        })();
        let receipt_observation = receipt_observation
            .as_ref()
            .and_then(|observation| observation.borrow_mut().take());
        let generation_succeeded = result.is_ok();
        complete_generation_before_cleanup(
            result,
            || {
                self.finalize_generation_receipt(
                    LocalGenerationReceiptFinalization {
                        session_label: &session_id,
                        request_id: receipt_request_id,
                        session_id: receipt_session_id,
                        agent_session_id: request.ids.agent_session_id.as_deref(),
                        prompt_token_ids: receipt_prompt_token_ids.unwrap_or_default(),
                        observation: receipt_observation,
                        cancelled: receipt_cancelled,
                        model_generation_elapsed: receipt_model_generation_elapsed,
                    },
                    generation_succeeded,
                )
            },
            || self.cleanup_local_generation_session(&session_id, request.ids),
        )?;
        Ok(cache_stats)
    }

    fn can_sample_whole_prompt_in_prefill(
        &self,
        request: &LocalGeneration<'_>,
        session_id: &str,
    ) -> OpenAiResult<bool> {
        if request.max_tokens == 0 || request.prompt_token_ids.len() <= 1 || self.kv.is_some() {
            return Ok(false);
        }
        let mut runtime = self
            .runtime
            .lock()
            .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
        runtime
            .ensure_session_active(session_id)
            .map_err(openai_backend_error)?;
        let batch_size = runtime
            .admit_session_batch_size(session_id)
            .map_err(openai_backend_error)?;
        Ok(prompt_fits_single_prefill_sample(
            request.prompt_token_ids.len(),
            batch_size,
        ))
    }

    fn prefill_prompt(
        &self,
        request: &LocalGeneration<'_>,
        session_id: &str,
        cache_stats: &mut GenerationCacheStats,
    ) -> OpenAiResult<PromptPrefillResult> {
        if self.can_sample_whole_prompt_in_prefill(request, session_id)? {
            let chat_sampling_configured = if let Some(metadata) = request.chat_sampling_metadata {
                self.configure_chat_sampling(
                    session_id,
                    metadata,
                    request.prompt_token_ids.len(),
                    request.sampling.enabled.then_some(request.sampling),
                )?;
                true
            } else {
                false
            };
            return Ok(PromptPrefillResult {
                prompt_prefill_sample: self.prefill_whole_prompt(
                    request,
                    session_id,
                    cache_stats,
                )?,
                chat_sampling_configured,
            });
        }
        if request.prompt_token_ids.len() > 1 {
            self.restore_or_record_kv(request, session_id, cache_stats)?;
        }
        Ok(PromptPrefillResult {
            prompt_prefill_sample: None,
            chat_sampling_configured: false,
        })
    }

    fn prefill_whole_prompt(
        &self,
        request: &LocalGeneration<'_>,
        session_id: &str,
        cache_stats: &mut GenerationCacheStats,
    ) -> OpenAiResult<Option<i32>> {
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
                session_id,
                request.prompt_token_ids,
                &[],
                request.sampling.enabled.then_some(request.sampling),
                None,
            )
            .map_err(openai_backend_error)?;
        let prompt_prefill_sample = Some(predicted);
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
        Ok(prompt_prefill_sample)
    }

    fn restore_or_record_kv(
        &self,
        request: &LocalGeneration<'_>,
        session_id: &str,
        cache_stats: &mut GenerationCacheStats,
    ) -> OpenAiResult<()> {
        let prefill_timer = PhaseTimer::start();
        let prefill_tokens = &request.prompt_token_ids[..request.prompt_token_ids.len() - 1];
        let lock_timer = PhaseTimer::start();
        let mut runtime = self
            .runtime
            .lock()
            .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
        let runtime_lock_wait_ms = lock_timer.elapsed_ms();
        let runtime_lock_hold_timer = PhaseTimer::start();
        let runtime_sessions_before = runtime.session_stats();
        let (restored_prefill, restored_prefill_tokens) = if let Some(kv) = self.kv.as_ref() {
            cache_stats.status = "miss";
            self.lookup_and_restore_kv(
                kv,
                &mut runtime,
                session_id,
                request.ids,
                prefill_tokens,
                cache_stats,
            )
        } else {
            (false, 0)
        };
        let mut decoded_prefill_suffix = false;
        if restored_prefill_tokens < prefill_tokens.len() {
            decoded_prefill_suffix = true;
            if let Some(kv) = self.kv.as_ref().filter(|kv| kv.payload_is_exact_state()) {
                // Recurrent and full-state payloads cannot reconstruct a shorter
                // shared prefix from the state at the end of the request. Stop
                // at the near-tail grid boundary while prefilling and snapshot
                // native state there; the final exact state is recorded by
                // `record_and_evict_kv` below. One checkpoint bounds both the
                // extra prefill split and the very large recurrent-state export.
                let base = self.local_kv_message_base(session_id, request.ids);
                let checkpoint =
                    kv.exact_shared_checkpoint_identity(&self.config, &base, 0, prefill_tokens);
                if let Some(identity) = checkpoint.filter(|identity| {
                    identity.identity.token_count as usize > restored_prefill_tokens
                }) {
                    let boundary = identity.identity.token_count as usize;
                    runtime
                        .prefill(
                            session_id,
                            &prefill_tokens[restored_prefill_tokens..boundary],
                        )
                        .map_err(openai_backend_error)?;
                    kv.record_exact_state(&mut runtime, session_id, &identity)
                        .map_err(openai_backend_error)?;
                    runtime
                        .prefill(session_id, &prefill_tokens[boundary..])
                        .map_err(openai_backend_error)?;
                } else {
                    runtime
                        .prefill(session_id, &prefill_tokens[restored_prefill_tokens..])
                        .map_err(openai_backend_error)?;
                }
            } else {
                runtime
                    .prefill(session_id, &prefill_tokens[restored_prefill_tokens..])
                    .map_err(openai_backend_error)?;
            }
        }
        cache_stats.matched_prefix_tokens = saturating_u32(restored_prefill_tokens);
        cache_stats.suffix_prefill_tokens =
            saturating_u32(prefill_tokens.len().saturating_sub(restored_prefill_tokens));
        let record = self.record_and_evict_kv(
            &mut runtime,
            session_id,
            request.ids,
            prefill_tokens,
            restored_prefill,
            decoded_prefill_suffix,
        );
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
            json!(record.resident_recorded_pages),
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
                record.proactive_eviction_status,
                record.proactive_eviction_error_kind,
                record.proactive_eviction_target_tokens,
                record.proactive_evicted_entries,
                record.proactive_evicted_tokens,
            ),
        );
        if let Some(error) = record.proactive_eviction_error {
            return Err(openai_backend_error(error));
        }
        Ok(())
    }

    fn lookup_and_restore_kv(
        &self,
        kv: &KvStageIntegration,
        runtime: &mut RuntimeState,
        session_id: &str,
        ids: &OpenAiGenerationIds,
        prefill_tokens: &[i32],
        cache_stats: &mut GenerationCacheStats,
    ) -> (bool, usize) {
        let mut restored_prefill = false;
        let mut restored_prefill_tokens = 0usize;
        let base = self.local_kv_message_base(session_id, ids);
        let kv_identity_timer = PhaseTimer::start();
        let identities = kv.lookup_identities(&self.config, &base, 0, prefill_tokens);
        let kv_identity_ms = kv_identity_timer.elapsed_ms();
        let kv_restore_timer = PhaseTimer::start();
        match kv.restore_exact_state(runtime, session_id, &identities) {
            Ok(Some(restored)) => {
                restored_prefill = true;
                cache_stats.status = "hit";
                cache_stats.hit_kind = Some("exact_prefix");
                let mut attrs = self.openai_attrs(ids);
                attrs.insert("skippy.kv.decision".to_string(), json!("exact_hit"));
                attrs.insert(
                    "skippy.exact_cache.hit_page_id".to_string(),
                    json!(restored.page_id),
                );
                attrs.insert(
                    "skippy.exact_cache.payload_kind".to_string(),
                    json!(restored.payload_kind.to_string()),
                );
                // Which tier served it. A RAM hit and a disk hit cost
                // different amounts and fail for different reasons; without
                // this the disk tier's effect is unmeasurable from the hit
                // stream alone.
                attrs.insert(
                    "skippy.exact_cache.hit_source".to_string(),
                    json!(restored.source.as_str()),
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
                cache_stats.cached_prompt_tokens = saturating_u32(restored_prefill_tokens);
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
            Ok(None) => {
                match kv.restore_resident_prefix(runtime, session_id, &identities, prefill_tokens) {
                    Ok(Some(restored)) => {
                        restored_prefill = true;
                        cache_stats.status = "hit";
                        cache_stats.hit_kind = Some("resident_prefix");
                        let mut attrs = self.openai_attrs(ids);
                        attrs.insert("skippy.kv.decision".to_string(), json!("resident_hit"));
                        attrs.insert("skippy.kv.hit_page_id".to_string(), json!(restored.page_id));
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
                            json!(prefill_tokens.len().saturating_sub(restored.token_count)),
                        );
                        restored_prefill_tokens = restored.token_count;
                        cache_stats.cached_prompt_tokens = saturating_u32(restored_prefill_tokens);
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
                        // Resident RAM miss. For dense families the prefix may
                        // still exist in the disk tier, having outlived either
                        // eviction or a restart. Restoring it here is what
                        // turns a cold quadratic prefill into a linear read.
                        match kv.restore_dense_prefix_from_disk(runtime, session_id, &identities) {
                            Ok(Some(restored)) => {
                                let restored_tokens =
                                    (restored.token_count as usize).min(prefill_tokens.len());
                                restored_prefill = true;
                                restored_prefill_tokens = restored_tokens;
                                cache_stats.status = "hit";
                                cache_stats.hit_kind = Some("disk_prefix");
                                cache_stats.cached_prompt_tokens = saturating_u32(restored_tokens);
                                let mut attrs = self.openai_attrs(ids);
                                attrs.insert("skippy.kv.decision".to_string(), json!("disk_hit"));
                                attrs.insert(
                                    "skippy.kv.hit_page_id".to_string(),
                                    json!(restored.page_id),
                                );
                                attrs.insert(
                                    "skippy.kv.restored_tokens".to_string(),
                                    json!(restored_tokens),
                                );
                                attrs.insert(
                                    "skippy.kv.matched_prefix_tokens".to_string(),
                                    json!(restored_tokens),
                                );
                                attrs.insert(
                                    "skippy.kv.suffix_prefill_tokens".to_string(),
                                    json!(prefill_tokens.len().saturating_sub(restored_tokens)),
                                );
                                // Split the restore cost so a slow restore can
                                // be attributed to verification or to the
                                // native import without guessing.
                                attrs.insert(
                                    "skippy.kv.disk_verify_ms".to_string(),
                                    json!(restored.verify_ms),
                                );
                                attrs.insert(
                                    "skippy.kv.disk_import_ms".to_string(),
                                    json!(restored.import_ms),
                                );
                                attrs.insert(
                                    "skippy.kv.disk_payload_bytes".to_string(),
                                    json!(restored.payload_bytes),
                                );
                                // A dense disk hit is by definition served
                                // from disk; state it explicitly so the hit
                                // stream is attributable without knowing
                                // which decision label implies which tier.
                                attrs.insert(
                                    "skippy.exact_cache.hit_source".to_string(),
                                    json!("disk"),
                                );
                                kv.insert_disk_tier_attrs(&mut attrs);
                                self.telemetry
                                    .emit("stage.openai_kv_lookup_decision", attrs);
                            }
                            Ok(None) => {
                                // Use the same attribute base as every other
                                // branch. A bare map here produced a `miss`
                                // event that could not be joined with the hit
                                // and error events on session or model, which
                                // is exactly the attribution the disk tier
                                // needs to show a hit rate.
                                let mut attrs = self.openai_attrs(ids);
                                attrs.insert("skippy.kv.decision".to_string(), json!("miss"));
                                kv.insert_disk_tier_attrs(&mut attrs);
                                self.telemetry
                                    .emit("stage.openai_kv_lookup_decision", attrs);
                            }
                            Err(error) => {
                                let mut attrs = self.openai_attrs(ids);
                                attrs.insert("skippy.kv.decision".to_string(), json!("disk_error"));
                                attrs.insert(
                                    "skippy.kv.error_class".to_string(),
                                    json!(crate::kv_integration::telemetry_error_class(&error)),
                                );
                                self.telemetry
                                    .emit("stage.openai_kv_lookup_decision", attrs);
                            }
                        }
                    }
                    Err(error) => {
                        let mut attrs = self.openai_attrs(ids);
                        attrs.insert("skippy.kv.decision".to_string(), json!("resident_error"));
                        attrs.insert(
                            "skippy.kv.error_class".to_string(),
                            json!(crate::kv_integration::telemetry_error_class(&error)),
                        );
                        self.telemetry
                            .emit("stage.openai_kv_lookup_decision", attrs);
                    }
                }
            }
            Err(error) => {
                let mut attrs = self.openai_attrs(ids);
                attrs.insert("skippy.kv.decision".to_string(), json!("exact_error"));
                attrs.insert(
                    "skippy.kv.error_class".to_string(),
                    json!(crate::kv_integration::telemetry_error_class(&error)),
                );
                self.telemetry
                    .emit("stage.openai_kv_lookup_decision", attrs);
            }
        }
        let mut attrs = self.openai_attrs(ids);
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
        (restored_prefill, restored_prefill_tokens)
    }

    fn record_and_evict_kv(
        &self,
        runtime: &mut RuntimeState,
        session_id: &str,
        ids: &OpenAiGenerationIds,
        prefill_tokens: &[i32],
        restored_prefill: bool,
        decoded_prefill_suffix: bool,
    ) -> KvRecordResult {
        let mut resident_recorded_pages = 0usize;
        if let (true, Some(kv)) = (
            !restored_prefill || decoded_prefill_suffix,
            self.kv.as_ref(),
        ) {
            let base = self.local_kv_message_base(session_id, ids);
            let exact_identity = kv.prefill_identity(&self.config, &base, 0, prefill_tokens);
            if let Ok(Some(record)) = kv.record_exact_state(runtime, session_id, &exact_identity) {
                resident_recorded_pages = resident_recorded_pages.saturating_add(1);
                let mut attrs = self.openai_attrs(ids);
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
            // At most one archive per request. Each archive is a full
            // `export_kv_page` into a fresh buffer plus a synced file write,
            // all under the runtime lock on the prefill path; doing that once
            // per ladder candidate would mean several hundred-MB exports for a
            // single large prompt.
            let mut archive_candidate = crate::kv_integration::ArchiveCandidate::default();
            for identity in kv.record_identities(&self.config, &base, 0, prefill_tokens) {
                // Offer to the archive before the resident cache decides.
                // The two tiers have different cost models -- KV cells versus
                // bytes-and-a-write -- and must not share a veto. A large
                // agentic prefix is exactly the case where they disagree:
                // `max_resident_tokens` is half the context window, so on a
                // 32k-context node every candidate for a 25k prompt is
                // declined as uncacheable, and gating archival on that
                // decision meant the node persisted nothing at precisely the
                // prompt sizes the disk tier exists for.
                //
                // `ArchiveCandidate` still applies the selection policy and
                // `archive_dense_prefix` still applies the size floor and
                // dedupe check, so this remains at most one archive per
                // request. See `ArchiveCandidate` for which candidate is
                // chosen and why the obvious choices are both wrong.
                crate::kv_integration::offer_archive_candidate(
                    &mut archive_candidate,
                    &identity,
                    prefill_tokens.len(),
                );
                if let Ok(Some(record)) =
                    kv.record_resident_prefix(runtime, session_id, &identity, prefill_tokens)
                {
                    resident_recorded_pages = resident_recorded_pages.saturating_add(1);
                    let mut attrs = self.openai_attrs(ids);
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
            // Archive dense prefixes so they outlive resident eviction and
            // process restart. Done here, after recording, rather than on
            // eviction: eviction runs on the decode hot path and a
            // multi-hundred-MB export there would spike TTFT.
            if let Some(identity) = archive_candidate.take() {
                let mut attrs = self.openai_attrs(ids);
                attrs.insert("skippy.kv.decision".to_string(), json!("disk_archive"));
                kv.insert_disk_tier_attrs(&mut attrs);
                match kv.archive_dense_prefix(runtime, session_id, &identity) {
                    Ok(outcome) => outcome.insert_attrs(&identity, &mut attrs),
                    Err(error) => {
                        attrs.insert(
                            "skippy.kv.archive_status".to_string(),
                            json!("failed_error"),
                        );
                        attrs.insert(
                            "skippy.kv.archive_error_class".to_string(),
                            json!(crate::kv_integration::telemetry_error_class(&error)),
                        );
                    }
                }
                self.telemetry
                    .emit("stage.openai_kv_record_decision", attrs);
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
            match kv.evict_resident_prefix_for_decode_batch(runtime, session_id) {
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
                    proactive_eviction_error =
                        Some(error.context("evict resident-prefix KV before local OpenAI decode"));
                }
            }
        }
        KvRecordResult {
            resident_recorded_pages,
            proactive_eviction_status,
            proactive_eviction_error_kind: proactive_eviction_error_kind_attr,
            proactive_eviction_target_tokens,
            proactive_evicted_entries,
            proactive_evicted_tokens,
            proactive_eviction_error,
        }
    }

    fn configure_chat_sampling(
        &self,
        session_id: &str,
        metadata: &str,
        prompt_token_count: usize,
        sampling: Option<&SamplingConfig>,
    ) -> OpenAiResult<()> {
        let mut runtime = self
            .runtime
            .lock()
            .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
        runtime
            .configure_chat_sampling(session_id, metadata, prompt_token_count as u64, sampling)
            .map_err(openai_backend_error)
    }

    fn configure_chat_sampling_if_needed(
        &self,
        request: &LocalGeneration<'_>,
        session_id: &str,
        chat_sampling_configured: bool,
    ) -> OpenAiResult<()> {
        if let Some(metadata) = (!chat_sampling_configured)
            .then_some(request.chat_sampling_metadata)
            .flatten()
        {
            self.configure_chat_sampling(
                session_id,
                metadata,
                request.prompt_token_ids.len(),
                request.sampling.enabled.then_some(request.sampling),
            )?;
        }
        Ok(())
    }

    fn prepare_decode_state(
        &self,
        request: &mut LocalGeneration<'_>,
        session_id: &str,
        prompt_prefill_sample: Option<i32>,
        emit_token: &mut impl FnMut(i32) -> OpenAiResult<TokenControl>,
    ) -> OpenAiResult<DecodeState> {
        let mut decoded_tokens = 0usize;
        let mut current = *request
            .prompt_token_ids
            .last()
            .expect("checked non-empty prompt");
        let mut stopped = false;
        let mut pending_linear_proposal_tokens = Vec::new();
        if let Some(predicted) = prompt_prefill_sample {
            if request
                .cancellation
                .is_some_and(openai_frontend::CancellationToken::is_cancelled)
            {
                return Err(OpenAiError::backend("request cancelled"));
            }
            current = predicted;
            decoded_tokens += 1;
            pending_linear_proposal_tokens.push(current);
            stopped = emit_token(current)? == TokenControl::Stop;
        }
        let hook_request = request.hook_request.take();
        let hook_runtime = request.hook_runtime.take();
        let generation_hooks_active =
            self.generation_hooks_active(&hook_request, hook_runtime.as_ref());
        let linear_proposal_enabled = self.linear_proposal_ingress.is_some()
            && !request.native_mtp_enabled
            && !generation_hooks_active
            && greedy_linear_proposal_admitted(request.sampling, request.chat_sampling_metadata);
        let linear_proposal_max_tokens = if linear_proposal_enabled {
            let mut runtime = self
                .runtime
                .lock()
                .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
            runtime
                .admit_session_batch_size(session_id)
                .map_err(openai_backend_error)?
                .saturating_sub(1)
        } else {
            0
        };
        let linear_context_tokens = (linear_proposal_max_tokens > 0).then(|| {
            let mut tokens = request.prompt_token_ids.to_vec();
            if decoded_tokens > 0 {
                tokens.push(current);
            }
            tokens
        });
        if linear_proposal_max_tokens == 0 {
            pending_linear_proposal_tokens.clear();
        }
        Ok(DecodeState {
            decoded_tokens,
            current,
            stopped,
            runtime_lock_wait_ms: 0.0,
            runtime_lock_wait_max_ms: 0.0,
            runtime_lock_hold_ms: 0.0,
            runtime_lock_hold_max_ms: 0.0,
            runtime_lock_acquires: 0,
            runtime_sessions_before: None,
            runtime_sessions_after: None,
            hook_request,
            hook_runtime,
            generation_hooks_active,
            linear_proposal_max_tokens,
            linear_context_tokens,
            pending_linear_proposal_tokens,
            emit_token_debug: self.telemetry.is_debug_enabled(),
            native_mtp_options: NativeMtpDecodeOptions::from_config(request.speculative),
            native_mtp: NativeMtpVerifier::default(),
            post_prefill_hook_checked: false,
            last_mid_generation_hook_at: None,
        })
    }

    fn run_decode_loop(
        &self,
        request: &mut LocalGeneration<'_>,
        session_id: &str,
        prompt_prefill_sample: Option<i32>,
        cache_stats: &mut GenerationCacheStats,
        receipt_cancelled: &mut bool,
        emit_token: &mut impl FnMut(i32) -> OpenAiResult<TokenControl>,
    ) -> OpenAiResult<Duration> {
        let decode_timer = PhaseTimer::start();
        let mut state =
            self.prepare_decode_state(request, session_id, prompt_prefill_sample, emit_token)?;
        while !state.stopped && state.decoded_tokens < request.max_tokens as usize {
            if request
                .cancellation
                .is_some_and(openai_frontend::CancellationToken::is_cancelled)
            {
                *receipt_cancelled = true;
                break;
            }
            match self.try_execute_linear_proposal(request, session_id, &mut state, emit_token)? {
                LinearProposalProgress::Continue => continue,
                LinearProposalProgress::Stop => break,
                LinearProposalProgress::NotUsed => {}
            }
            let control = self.decode_one_token(request, session_id, &mut state, emit_token)?;
            if state.linear_proposal_max_tokens > 0 {
                state.pending_linear_proposal_tokens.push(state.current);
            }
            if control == TokenControl::Stop {
                break;
            }
        }
        self.emit_decode_summary(request, &mut state, cache_stats, decode_timer)
    }
}

pub(super) trait NativeMtpRuntime {
    fn decode_sampled_mtp(
        &mut self,
        session_id: &str,
        token_id: i32,
        sampling: Option<&SamplingConfig>,
        max_draft_tokens: usize,
    ) -> anyhow::Result<(i32, Option<RuntimeNativeMtpDraft>)>;

    #[allow(dead_code)]
    fn decode_frame_sampled_mtp(
        &mut self,
        session_id: &str,
        token_id: i32,
        sampling: Option<&SamplingConfig>,
        input: Option<&ActivationFrame>,
        output_capacity: usize,
        max_draft_tokens: usize,
    ) -> anyhow::Result<(i32, Option<RuntimeNativeMtpDraft>, ActivationFrame)>;
}

impl NativeMtpRuntime for RuntimeState {
    fn decode_sampled_mtp(
        &mut self,
        session_id: &str,
        token_id: i32,
        sampling: Option<&SamplingConfig>,
        max_draft_tokens: usize,
    ) -> anyhow::Result<(i32, Option<RuntimeNativeMtpDraft>)> {
        RuntimeState::decode_sampled_mtp(self, session_id, token_id, sampling, max_draft_tokens)
    }

    fn decode_frame_sampled_mtp(
        &mut self,
        session_id: &str,
        token_id: i32,
        sampling: Option<&SamplingConfig>,
        input: Option<&ActivationFrame>,
        output_capacity: usize,
        max_draft_tokens: usize,
    ) -> anyhow::Result<(i32, Option<RuntimeNativeMtpDraft>, ActivationFrame)> {
        RuntimeState::decode_frame_sampled_mtp(
            self,
            session_id,
            token_id,
            sampling,
            input,
            output_capacity,
            max_draft_tokens,
        )
    }
}

pub(super) fn decode_native_mtp(
    runtime: &mut impl NativeMtpRuntime,
    session_id: &str,
    token_id: i32,
    sampling: Option<&SamplingConfig>,
    max_draft_tokens: usize,
) -> OpenAiResult<(i32, Option<NativeMtpDraft>)> {
    let (predicted, draft) = runtime
        .decode_sampled_mtp(session_id, token_id, sampling, max_draft_tokens)
        .map_err(openai_backend_error)?;
    Ok((
        predicted,
        draft.map(|draft| NativeMtpDraft {
            tokens: draft.token_ids,
            proposal_compute_us: draft.proposal_compute_us,
        }),
    ))
}

#[cfg(test)]
pub(in crate::frontend) fn native_mtp_dispatch_counts_for_test() -> (usize, usize) {
    struct FakeNativeMtpRuntime {
        sampled_calls: usize,
        frame_calls: usize,
    }

    impl NativeMtpRuntime for FakeNativeMtpRuntime {
        fn decode_sampled_mtp(
            &mut self,
            _session_id: &str,
            _token_id: i32,
            _sampling: Option<&SamplingConfig>,
            _max_draft_tokens: usize,
        ) -> anyhow::Result<(i32, Option<RuntimeNativeMtpDraft>)> {
            self.sampled_calls += 1;
            Ok((7, None))
        }

        fn decode_frame_sampled_mtp(
            &mut self,
            _session_id: &str,
            _token_id: i32,
            _sampling: Option<&SamplingConfig>,
            _input: Option<&ActivationFrame>,
            _output_capacity: usize,
            _max_draft_tokens: usize,
        ) -> anyhow::Result<(i32, Option<RuntimeNativeMtpDraft>, ActivationFrame)> {
            self.frame_calls += 1;
            anyhow::bail!("frame MTP API must not be used by local decode");
        }
    }

    let mut runtime = FakeNativeMtpRuntime {
        sampled_calls: 0,
        frame_calls: 0,
    };
    let (predicted, draft) =
        decode_native_mtp(&mut runtime, "test", 0, None, 1).expect("sampled MTP dispatch");
    assert_eq!(predicted, 7);
    assert!(draft.is_none());
    (runtime.sampled_calls, runtime.frame_calls)
}
