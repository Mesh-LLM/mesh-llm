use super::native_mtp_decode::NativeMtpSpanProgress;
use crate::frontend::NativeMtpDecodeOptions;
use crate::frontend::NativeMtpDraft;
use crate::frontend::NativeMtpVerifier;
use crate::frontend::generation::GENERATION_RETRY_AFTER_SECS;
use crate::frontend::generation::GenerationCacheStats;
use crate::frontend::generation::LocalGeneration;
use crate::frontend::generation::OpenAiGenerationIds;
use crate::frontend::generation::PhaseTimer;
use crate::frontend::generation::StageOpenAiBackend;
use crate::frontend::generation::TokenControl;
use crate::frontend::generation_receipt::{
    GenerationCommit, GenerationStart, complete_generation_before_cleanup,
};
use crate::frontend::iteration_scheduler::ScheduledGenerationRequest;
use crate::frontend::linear_proposal::greedy_linear_proposal_admitted;
use crate::frontend::util::openai_backend_error;
use crate::frontend::util::saturating_u32;
use crate::kv_integration::proactive_eviction_attrs;
use crate::kv_integration::proactive_eviction_error_kind;
use crate::kv_integration::{KvStageIntegration, StagePrefixCachePayload};
use crate::runtime_state::{RuntimeSessionStats, RuntimeState};
use axum::http::StatusCode;
use openai_frontend::ChatCompletionRequest;
use openai_frontend::OpenAiError;
use openai_frontend::OpenAiErrorKind;
use openai_frontend::OpenAiResult;
use serde_json::json;
use skippy_metrics::attr as attr_key;
use skippy_runtime::NativeMtpDraft as RuntimeNativeMtpDraft;
use skippy_runtime::SamplingConfig;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use super::{LocalGenerationReceiptFinalization, prompt_fits_single_prefill_sample};

pub(in crate::frontend) fn resident_capacity_admission_error(
    capacity: &crate::kv_integration::ResidentCapacityDecision,
) -> OpenAiError {
    OpenAiError::from_kind(
        StatusCode::TOO_MANY_REQUESTS,
        OpenAiErrorKind::RateLimit,
        format!(
            "resident KV capacity admission rejected request: {} token deficit (capacity={}, active={}, pinned={}, request={}, minimum_free={})",
            capacity.admission_deficit_tokens,
            capacity.capacity_tokens,
            capacity.active_tokens,
            capacity.pinned_tokens,
            capacity.request_tokens,
            capacity.minimum_free_tokens,
        ),
    )
    .with_retry_after_secs(GENERATION_RETRY_AFTER_SECS)
}

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

impl Default for KvRecordResult {
    fn default() -> Self {
        Self {
            resident_recorded_pages: 0,
            proactive_eviction_status: "disabled",
            proactive_eviction_error_kind: None,
            proactive_eviction_target_tokens: 0,
            proactive_evicted_entries: 0,
            proactive_evicted_tokens: 0,
            proactive_eviction_error: None,
        }
    }
}

fn insert_resident_capacity_attrs(
    attrs: &mut BTreeMap<String, serde_json::Value>,
    decision: &crate::kv_integration::ResidentCapacityDecision,
) {
    let status = if !decision.enabled {
        "disabled"
    } else if !decision.capacity_known {
        "unknown_capacity"
    } else if !decision.admitted {
        "rejected"
    } else if decision.evicted_entries > 0 {
        "evicted"
    } else {
        "admitted"
    };
    attrs.insert(attr_key::KV_CAPACITY_STATUS.to_string(), json!(status));
    attrs.insert(
        attr_key::KV_CAPACITY_TOKENS.to_string(),
        json!(decision.capacity_tokens),
    );
    attrs.insert(
        attr_key::KV_CAPACITY_ACTIVE_TOKENS.to_string(),
        json!(decision.active_tokens),
    );
    attrs.insert(
        attr_key::KV_CAPACITY_PINNED_TOKENS.to_string(),
        json!(decision.pinned_tokens),
    );
    attrs.insert(
        attr_key::KV_CAPACITY_REQUEST_TOKENS.to_string(),
        json!(decision.request_tokens),
    );
    attrs.insert(
        attr_key::KV_CAPACITY_MINIMUM_FREE_TOKENS.to_string(),
        json!(decision.minimum_free_tokens),
    );
    attrs.insert(
        attr_key::KV_CAPACITY_TARGET_FREE_TOKENS.to_string(),
        json!(decision.target_free_tokens),
    );
    attrs.insert(
        attr_key::KV_CAPACITY_PROJECTED_FREE_TOKENS.to_string(),
        json!(decision.projected_free_tokens),
    );
    attrs.insert(
        attr_key::KV_CAPACITY_ADMISSION_DEFICIT_TOKENS.to_string(),
        json!(decision.admission_deficit_tokens),
    );
    attrs.insert(
        attr_key::KV_CAPACITY_REQUIRED_EVICTION_TOKENS.to_string(),
        json!(decision.required_eviction_tokens),
    );
    attrs.insert(
        attr_key::KV_CAPACITY_EVICTED_ENTRIES.to_string(),
        json!(decision.evicted_entries),
    );
    attrs.insert(
        attr_key::KV_CAPACITY_EVICTED_TOKENS.to_string(),
        json!(decision.evicted_tokens),
    );
    attrs.insert(
        attr_key::KV_CAPACITY_PREDICTED_RECOMPUTE_COST.to_string(),
        json!(decision.predicted_recompute_cost),
    );
}

struct KvRestoreOutcome {
    runtime_sessions_before: RuntimeSessionStats,
    runtime_sessions_after: RuntimeSessionStats,
    restored_prefill: bool,
    restored_prefill_tokens: usize,
    capacity: crate::kv_integration::ResidentCapacityDecision,
    record: KvRecordResult,
    prompt_prefill_sample: Option<i32>,
}

pub(super) struct DecodeState {
    pub(super) decoded_tokens: usize,
    pub(super) current: i32,
    /// Target-authoritative tokens emitted by this request. The runtime has
    /// consumed every token in `prompt_token_ids + generated_token_ids` except
    /// the final element: autoregressive decode consumes `current` and returns
    /// the next (still unconsumed) token. This lets us name the recurrent state
    /// at the exact token boundary that was actually captured.
    pub(super) generated_token_ids: Vec<i32>,
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
    /// Whether batched native-MTP spans may commit for this request. Token
    /// equality is only a valid acceptance test under greedy sampling.
    pub(super) native_mtp_span_admitted: bool,
    pub(super) post_prefill_hook_checked: bool,
    pub(super) last_mid_generation_hook_at: Option<usize>,
}

pub(super) enum LinearProposalProgress {
    NotUsed,
    Continue,
    Stop,
}

pub(in crate::frontend) fn post_decode_checkpoint_tokens(
    prompt_token_ids: &[i32],
    generated_token_ids: &[i32],
) -> Option<Vec<i32>> {
    if generated_token_ids.is_empty() {
        return None;
    }
    let mut checkpoint = Vec::with_capacity(
        prompt_token_ids
            .len()
            .saturating_add(generated_token_ids.len())
            .saturating_sub(1),
    );
    checkpoint.extend_from_slice(prompt_token_ids);
    checkpoint
        .extend_from_slice(&generated_token_ids[..generated_token_ids.len().saturating_sub(1)]);
    Some(checkpoint)
}

pub(in crate::frontend) fn linear_proposal_allowed(
    recurrent_checkpoint_required: bool,
    checkpoint_attempted: bool,
) -> bool {
    !recurrent_checkpoint_required || checkpoint_attempted
}

fn scheduler_generation_driver_eligible(
    max_tokens: u32,
    kv_enabled: bool,
    linear_proposal_enabled: bool,
    draft_enabled: bool,
    native_mtp_enabled: bool,
    speculative_strategy: &str,
    hooks_enabled: bool,
) -> bool {
    // This predicate selects the scheduler's built-in plain-text generation
    // driver. Speculative and extended requests keep their strategy-specific
    // proposal/checkpoint state machine, but every target iteration,
    // verification, and repair operation is submitted to the same scheduler
    // worker, including strategy-specific runtime operations.
    max_tokens > 0
        && !kv_enabled
        && !linear_proposal_enabled
        && !draft_enabled
        && !native_mtp_enabled
        && speculative_strategy == "disabled"
        && !hooks_enabled
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
            if self.uses_scheduler_builtin_driver(&request) {
                let model_generation_elapsed = self.run_scheduled_generation(
                    &request,
                    &session_id,
                    &mut cache_stats,
                    &mut emit_token,
                )?;
                receipt_model_generation_elapsed = Some(model_generation_elapsed);
                return Ok(());
            }
            let prefill = self.prefill_prompt(&request, &session_id, &mut cache_stats)?;
            self.configure_chat_sampling_if_needed(
                &request,
                &session_id,
                prefill.chat_sampling_configured,
            )?;
            let model_generation_elapsed = self.run_scheduler_feature_loop(
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

    fn uses_scheduler_builtin_driver(&self, request: &LocalGeneration<'_>) -> bool {
        let kv_enabled = self.kv.is_some();
        let linear_proposal_enabled = self.linear_proposal_ingress.is_some();
        let draft_enabled = self.draft.is_some();
        let hooks_enabled = self.hook_policy.is_some();
        let eligible = scheduler_generation_driver_eligible(
            request.max_tokens,
            kv_enabled,
            linear_proposal_enabled,
            draft_enabled,
            request.native_mtp_enabled,
            &request.speculative.effective_strategy,
            hooks_enabled,
        );
        let mut attrs = self.openai_attrs(request.ids);
        attrs.insert("skippy.scheduler.eligible".to_string(), json!(eligible));
        attrs.insert(
            "skippy.scheduler.runtime_owner".to_string(),
            json!("iteration-worker"),
        );
        attrs.insert(
            "skippy.scheduler.generation_driver".to_string(),
            json!(if eligible { "built-in" } else { "feature" }),
        );
        attrs.insert("skippy.scheduler.kv_enabled".to_string(), json!(kv_enabled));
        attrs.insert(
            "skippy.scheduler.linear_proposal_enabled".to_string(),
            json!(linear_proposal_enabled),
        );
        attrs.insert(
            "skippy.scheduler.draft_enabled".to_string(),
            json!(draft_enabled),
        );
        attrs.insert(
            "skippy.scheduler.native_mtp_enabled".to_string(),
            json!(request.native_mtp_enabled),
        );
        attrs.insert(
            "skippy.scheduler.speculative_strategy".to_string(),
            json!(request.speculative.effective_strategy),
        );
        attrs.insert(
            "skippy.scheduler.hooks_enabled".to_string(),
            json!(hooks_enabled),
        );
        self.telemetry.emit_debug("stage.scheduler_route", attrs);
        eligible
    }

    fn run_scheduled_generation(
        &self,
        request: &LocalGeneration<'_>,
        session_id: &str,
        cache_stats: &mut GenerationCacheStats,
        emit_token: &mut impl FnMut(i32) -> OpenAiResult<TokenControl>,
    ) -> OpenAiResult<Duration> {
        let timer = PhaseTimer::start();
        let stats = self.iteration_scheduler.generate(
            ScheduledGenerationRequest {
                id: session_id,
                prompt_tokens: request.prompt_token_ids,
                max_tokens: request.max_tokens,
                sampling: request.sampling.enabled.then_some(request.sampling),
                chat_sampling_metadata: request.chat_sampling_metadata,
                cancellation: request.cancellation,
            },
            emit_token,
        )?;
        cache_stats.suffix_prefill_tokens = saturating_u32(request.prompt_token_ids.len());
        cache_stats.prompt_ms = stats.prompt_ms;
        cache_stats.predicted_ms = stats.predicted_ms;
        let mut attrs = self.openai_attrs(request.ids);
        attrs.insert(
            "skippy.scheduler.serving_path".to_string(),
            json!("iteration"),
        );
        attrs.insert(
            "skippy.scheduler.prompt_tokens".to_string(),
            json!(request.prompt_token_ids.len()),
        );
        attrs.insert(
            "skippy.scheduler.max_tokens".to_string(),
            json!(request.max_tokens),
        );
        attrs.insert(
            "skippy.scheduler.prompt_ms".to_string(),
            json!(stats.prompt_ms),
        );
        attrs.insert(
            "skippy.scheduler.predicted_ms".to_string(),
            json!(stats.predicted_ms),
        );
        self.emit_openai_phase("stage.openai_scheduler_generation", timer, attrs);
        Ok(Duration::from_secs_f64(
            (stats.prompt_ms + stats.predicted_ms) / 1_000.0,
        ))
    }

    fn can_sample_whole_prompt_in_prefill(
        &self,
        request: &LocalGeneration<'_>,
        session_id: &str,
    ) -> OpenAiResult<bool> {
        if request.max_tokens == 0 || request.prompt_token_ids.len() <= 1 || self.kv.is_some() {
            return Ok(false);
        }
        let scheduler_session_id = session_id.to_string();
        let batch_size = self.iteration_scheduler.execute_runtime(
            "feature-prefill-admission",
            move |runtime| {
                runtime
                    .ensure_session_active(&scheduler_session_id)
                    .map_err(openai_backend_error)?;
                runtime
                    .admit_session_batch_size(&scheduler_session_id)
                    .map_err(openai_backend_error)
            },
        )?;
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
            let chat_sampling_configured = if self
                .kv
                .as_ref()
                .is_some_and(|kv| kv.payload == StagePrefixCachePayload::ResidentKv)
            {
                if let Some(metadata) = request.chat_sampling_metadata {
                    self.configure_chat_sampling(
                        session_id,
                        metadata,
                        request.prompt_token_ids.len(),
                        request.sampling.enabled.then_some(request.sampling),
                    )?;
                    true
                } else {
                    false
                }
            } else {
                false
            };
            let prompt_prefill_sample =
                self.restore_or_record_kv(request, session_id, cache_stats)?;
            return Ok(PromptPrefillResult {
                prompt_prefill_sample,
                chat_sampling_configured,
            });
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
        let runtime_sessions_before = self
            .iteration_scheduler
            .execute_runtime("feature-prefill-stats-before", |runtime| {
                Ok(runtime.session_stats())
            })?;
        let outcome = self.iteration_scheduler.execute_iteration(
            session_id,
            request.prompt_token_ids,
            &[],
            request.sampling.enabled.then_some(request.sampling),
            true,
            skippy_runtime::IterationBatchPhase::Prefill,
        )?;
        let prompt_prefill_sample = Some(outcome.predicted);
        cache_stats.suffix_prefill_tokens = saturating_u32(request.prompt_token_ids.len());
        let runtime_sessions_after = self
            .iteration_scheduler
            .execute_runtime("feature-prefill-stats-after", |runtime| {
                Ok(runtime.session_stats())
            })?;
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
            json!(outcome.runtime_lock_wait_ms),
        );
        attrs.insert(
            "llama_stage.runtime_lock_hold_ms".to_string(),
            json!(outcome.runtime_lock_hold_ms),
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
    ) -> OpenAiResult<Option<i32>> {
        let prefill_timer = PhaseTimer::start();
        let prefill_tokens =
            Arc::<[i32]>::from(&request.prompt_token_ids[..request.prompt_token_ids.len() - 1]);
        let recurrent_cache_prefix_token_ids = request
            .recurrent_cache_prefix_token_ids
            .map(<[i32]>::to_vec);
        let max_tokens = request.max_tokens;
        let final_prompt_token = *request
            .prompt_token_ids
            .last()
            .expect("checked non-empty prompt");
        let sampling = request.sampling.enabled.then(|| request.sampling.clone());
        let (cache_affinity, refresh_cache_affinity) = match self.kv.as_ref() {
            Some(kv) => {
                let base = self.local_kv_message_base(session_id, request.ids);
                let identities =
                    kv.lookup_identities(&self.config, &base, 0, prefill_tokens.as_ref());
                let affinity = kv.peek_cache_affinity(&self.config, &identities);
                let kv = kv.clone();
                let config = self.config.clone();
                let refresh = Box::new(move || kv.peek_cache_affinity(&config, &identities))
                    as Box<dyn Fn() -> skippy_scheduler::CacheAffinity + Send>;
                (affinity, Some(refresh))
            }
            None => (skippy_scheduler::CacheAffinity::default(), None),
        };
        let scheduler_backend = self.clone();
        let scheduler_session_id = session_id.to_string();
        let scheduler_ids = request.ids.clone();
        let mut scheduler_cache_stats = std::mem::take(cache_stats);
        let outcome = self.iteration_scheduler.execute_cache_aware_runtime_timed(
            "feature-kv-restore-prefill-record",
            cache_affinity,
            Arc::clone(&prefill_tokens),
            0,
            refresh_cache_affinity,
            move |runtime| {
                let outcome = scheduler_backend.restore_or_record_kv_on_runtime(
                    runtime,
                    &scheduler_ids,
                    &scheduler_session_id,
                    prefill_tokens.as_ref(),
                    recurrent_cache_prefix_token_ids.as_deref(),
                    final_prompt_token,
                    sampling.as_ref(),
                    max_tokens,
                    &mut scheduler_cache_stats,
                )?;
                Ok((outcome, scheduler_cache_stats))
            },
        )?;
        let runtime_lock_wait_ms = outcome.runtime_lock_wait_ms;
        let runtime_lock_hold_ms = outcome.runtime_lock_hold_ms;
        let (outcome, updated_cache_stats) = outcome.value;
        *cache_stats = updated_cache_stats;
        let KvRestoreOutcome {
            runtime_sessions_before,
            runtime_sessions_after,
            restored_prefill,
            restored_prefill_tokens,
            capacity,
            record,
            prompt_prefill_sample,
        } = outcome;
        if prompt_prefill_sample.is_some() {
            cache_stats.suffix_prefill_tokens = saturating_u32(
                request
                    .prompt_token_ids
                    .len()
                    .saturating_sub(restored_prefill_tokens),
            );
        }
        let prompt_tail_tokens = usize::from(prompt_prefill_sample.is_some());
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
        insert_resident_capacity_attrs(&mut attrs, &capacity);
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

    #[allow(clippy::too_many_arguments)]
    fn restore_or_record_kv_on_runtime(
        &self,
        runtime: &mut RuntimeState,
        ids: &OpenAiGenerationIds,
        session_id: &str,
        prefill_tokens: &[i32],
        recurrent_cache_prefix_token_ids: Option<&[i32]>,
        final_prompt_token: i32,
        sampling: Option<&SamplingConfig>,
        max_tokens: u32,
        cache_stats: &mut GenerationCacheStats,
    ) -> OpenAiResult<KvRestoreOutcome> {
        let runtime_sessions_before = runtime.session_stats();
        let capacity = if let Some(kv) = self.kv.as_ref() {
            let decode_batch_tokens = u64::from(self.config.n_batch.unwrap_or(2048));
            let target_free_tokens =
                decode_batch_tokens.saturating_add(u64::from(max_tokens).min(decode_batch_tokens));
            kv.admit_resident_capacity(
                runtime,
                session_id,
                prefill_tokens.len() as u64,
                decode_batch_tokens,
                target_free_tokens,
            )
            .map_err(openai_backend_error)?
        } else {
            crate::kv_integration::ResidentCapacityDecision {
                admitted: true,
                ..crate::kv_integration::ResidentCapacityDecision::default()
            }
        };
        if !capacity.admitted {
            return Ok(KvRestoreOutcome {
                runtime_sessions_before,
                runtime_sessions_after: runtime.session_stats(),
                restored_prefill: false,
                restored_prefill_tokens: 0,
                capacity,
                record: KvRecordResult::default(),
                prompt_prefill_sample: None,
            });
        }
        let (restored_prefill, restored_prefill_tokens) = if let Some(kv) = self.kv.as_ref() {
            cache_stats.status = "miss";
            self.lookup_and_restore_kv(kv, runtime, session_id, ids, prefill_tokens, cache_stats)
        } else {
            (false, 0)
        };
        let resident_suffix_deferred = self
            .kv
            .as_ref()
            .is_some_and(|kv| kv.payload == StagePrefixCachePayload::ResidentKv);
        let prompt_prefill_sample = if resident_suffix_deferred {
            let suffix_start = restored_prefill_tokens.min(prefill_tokens.len());
            let mut suffix = prefill_tokens[suffix_start..].to_vec();
            suffix.push(final_prompt_token);
            Some(
                runtime
                    .prefill_chunked_sampled(session_id, &suffix, sampling)
                    .map_err(openai_backend_error)?,
            )
        } else {
            None
        };
        let mut decoded_prefill_suffix =
            resident_suffix_deferred && restored_prefill_tokens < prefill_tokens.len();
        if !resident_suffix_deferred && restored_prefill_tokens < prefill_tokens.len() {
            decoded_prefill_suffix = true;
            if let Some(checkpoint_tokens) =
                recurrent_cache_prefix_token_ids.filter(|checkpoint_tokens| {
                    !checkpoint_tokens.is_empty()
                        && checkpoint_tokens.len() <= prefill_tokens.len()
                        && prefill_tokens.starts_with(checkpoint_tokens)
                        && restored_prefill_tokens < checkpoint_tokens.len()
                })
            {
                runtime
                    .prefill(session_id, &checkpoint_tokens[restored_prefill_tokens..])
                    .map_err(openai_backend_error)?;
                let _ = self.record_exact_state_at_tokens(
                    runtime,
                    session_id,
                    ids,
                    checkpoint_tokens,
                    "chat_prefix_checkpoint",
                );
                runtime
                    .prefill(session_id, &prefill_tokens[checkpoint_tokens.len()..])
                    .map_err(openai_backend_error)?;
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
                    runtime
                        .prefill(
                            session_id,
                            &prefill_tokens[restored_prefill_tokens..boundary],
                        )
                        .map_err(openai_backend_error)?;
                    kv.record_exact_state(runtime, session_id, &identity)
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
            runtime,
            session_id,
            ids,
            prefill_tokens,
            restored_prefill,
            decoded_prefill_suffix,
        );
        let runtime_sessions_after = runtime.session_stats();
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
                attrs.insert(
                    "skippy.exact_cache.lookup_ms".to_string(),
                    json!(restored.lookup_ms),
                );
                attrs.insert(
                    "skippy.exact_cache.kv_import_ms".to_string(),
                    json!(restored.kv_import_ms),
                );
                attrs.insert(
                    "skippy.exact_cache.recurrent_import_ms".to_string(),
                    json!(restored.recurrent_import_ms),
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
                        self.telemetry
                            .emit("stage.openai_kv_lookup_decision", attrs);
                    }
                    Ok(None) => {
                        let mut attrs = self.openai_attrs(ids);
                        attrs.insert("skippy.kv.decision".to_string(), json!("miss"));
                        self.telemetry
                            .emit("stage.openai_kv_lookup_decision", attrs);
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
            for identity in kv.record_identities(&self.config, &base, 0, prefill_tokens) {
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

    fn record_post_decode_exact_state(
        &self,
        request: &LocalGeneration<'_>,
        session_id: &str,
        state: &DecodeState,
    ) -> bool {
        let Some(kv) = self.kv.as_ref() else {
            return false;
        };
        if kv.payload != StagePrefixCachePayload::KvRecurrent {
            return false;
        }
        let Some(checkpoint_tokens) =
            post_decode_checkpoint_tokens(request.prompt_token_ids, &state.generated_token_ids)
        else {
            return false;
        };
        let mut attrs = self.openai_attrs(request.ids);
        let scheduler_backend = self.clone();
        let scheduler_session_id = session_id.to_string();
        let scheduler_ids = request.ids.clone();
        let Ok(outcome) = self.iteration_scheduler.execute_runtime_timed(
            "feature-post-decode-checkpoint",
            move |runtime| {
                Ok(scheduler_backend.record_exact_state_at_tokens(
                    runtime,
                    &scheduler_session_id,
                    &scheduler_ids,
                    &checkpoint_tokens,
                    "post_decode_checkpoint",
                ))
            },
        ) else {
            attrs.insert(
                "skippy.kv.decision".to_string(),
                json!("post_decode_checkpoint_scheduler_error"),
            );
            self.telemetry
                .emit("stage.openai_kv_record_decision", attrs);
            return false;
        };
        let runtime_lock_wait_ms = outcome.runtime_lock_wait_ms;
        let recorded = outcome.value;
        attrs.insert(
            "skippy.kv.decision".to_string(),
            json!(if recorded {
                "post_decode_checkpoint_recorded"
            } else {
                "post_decode_checkpoint_skipped"
            }),
        );
        attrs.insert(
            "llama_stage.runtime_lock_wait_ms".to_string(),
            json!(runtime_lock_wait_ms),
        );
        self.telemetry
            .emit("stage.openai_kv_record_decision", attrs);
        recorded
    }

    /// Record a recurrent state only when the native session is at the exact
    /// token boundary named by `checkpoint_tokens`.
    ///
    /// The caller must hold the runtime lock. This check is intentionally
    /// canonical-position based: token text or a caller-supplied count cannot
    /// authorize exporting a state at a different native position.
    fn record_exact_state_at_tokens(
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
        if kv.payload != StagePrefixCachePayload::KvRecurrent {
            return false;
        }
        let Ok(checkpoint_token_count) = u64::try_from(checkpoint_tokens.len()) else {
            return false;
        };
        let runtime_token_count = match runtime.canonical_session_position(session_id) {
            Ok(position) => position,
            Err(error) => {
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
        match kv.record_exact_state(runtime, session_id, &identity) {
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
                true
            }
            Ok(None) => false,
            Err(error) => {
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

    fn configure_chat_sampling(
        &self,
        session_id: &str,
        metadata: &str,
        prompt_token_count: usize,
        sampling: Option<&SamplingConfig>,
    ) -> OpenAiResult<()> {
        let scheduler_session_id = session_id.to_string();
        let scheduler_metadata = metadata.to_string();
        let scheduler_sampling = sampling.cloned();
        self.iteration_scheduler
            .execute_runtime("feature-chat-sampling", move |runtime| {
                runtime
                    .configure_chat_sampling(
                        &scheduler_session_id,
                        &scheduler_metadata,
                        prompt_token_count as u64,
                        scheduler_sampling.as_ref(),
                    )
                    .map_err(openai_backend_error)
            })
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
        let mut generated_token_ids = Vec::new();
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
            generated_token_ids.push(current);
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
            let scheduler_session_id = session_id.to_string();
            self.iteration_scheduler
                .execute_runtime("linear-proposal-admission", move |runtime| {
                    runtime
                        .admit_session_batch_size(&scheduler_session_id)
                        .map_err(openai_backend_error)
                })?
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
            generated_token_ids,
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
            native_mtp_span_admitted: request.native_mtp_enabled
                && !generation_hooks_active
                && greedy_linear_proposal_admitted(
                    request.sampling,
                    request.chat_sampling_metadata,
                ),
            post_prefill_hook_checked: false,
            last_mid_generation_hook_at: None,
        })
    }

    /// Drives feature-specific proposal/checkpoint state while the iteration
    /// scheduler worker remains the sole owner of target runtime execution.
    /// This is not a fallback serving scheduler: native MTP, linear proposals,
    /// hooks, and cache operations submit their target work to that worker.
    fn run_scheduler_feature_loop(
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
        let recurrent_checkpoint_required = self
            .kv
            .as_ref()
            .is_some_and(|kv| kv.payload == StagePrefixCachePayload::KvRecurrent);
        // A recurrent request needs one serial decode before proposals can
        // advance the native session beyond the prompt boundary. The serial
        // attempt is the gate: a short prompt may be below the policy's
        // minimum checkpoint size, in which case recording is correctly
        // skipped but proposals must not remain disabled for the rest of the
        // request.
        let mut first_post_decode_checkpoint_attempted = !recurrent_checkpoint_required;
        while !state.stopped && state.decoded_tokens < request.max_tokens as usize {
            if request
                .cancellation
                .is_some_and(openai_frontend::CancellationToken::is_cancelled)
            {
                *receipt_cancelled = true;
                break;
            }
            let linear_progress = if linear_proposal_allowed(
                recurrent_checkpoint_required,
                first_post_decode_checkpoint_attempted,
            ) {
                self.try_execute_linear_proposal(request, session_id, &mut state, emit_token)?
            } else {
                // A proposal can consume the final prompt token and commit
                // multiple target tokens in one call. Give recurrent caches
                // one serial decode first so the exact full-prompt boundary
                // is published before that proposal advances the session.
                LinearProposalProgress::NotUsed
            };
            match linear_progress {
                LinearProposalProgress::Continue => continue,
                LinearProposalProgress::Stop => break,
                LinearProposalProgress::NotUsed => {}
            }
            // A batched MTP span commits multiple tokens from one forward. It
            // shares the recurrent-checkpoint gate with linear proposals for
            // the same reason: it can advance the session past the prompt
            // boundary before that boundary has been published.
            if linear_proposal_allowed(
                recurrent_checkpoint_required,
                first_post_decode_checkpoint_attempted,
            ) {
                match self
                    .try_execute_native_mtp_span(request, session_id, &mut state, emit_token)?
                {
                    NativeMtpSpanProgress::Continue => continue,
                    NativeMtpSpanProgress::Stop => break,
                    NativeMtpSpanProgress::NotUsed => {}
                }
            }
            let control = self.decode_one_token(request, session_id, &mut state, emit_token)?;
            if !first_post_decode_checkpoint_attempted && state.generated_token_ids.len() == 1 {
                first_post_decode_checkpoint_attempted = true;
                self.record_post_decode_exact_state(request, session_id, &state);
            }
            if state.linear_proposal_max_tokens > 0 {
                state.pending_linear_proposal_tokens.push(state.current);
            }
            if control == TokenControl::Stop {
                break;
            }
        }
        let model_generation_elapsed =
            self.emit_decode_summary(request, &mut state, cache_stats, decode_timer)?;
        // The first-token checkpoint names the prompt boundary. Avoid
        // exporting the same recurrent state twice for a one-token response;
        // longer responses still get their final exact boundary recorded.
        if state.generated_token_ids.len() > 1 {
            let _ = self.record_post_decode_exact_state(request, session_id, &state);
        }
        Ok(model_generation_elapsed)
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
#[test]
fn resident_capacity_attrs_report_bounded_rejection_evidence() {
    let mut attrs = BTreeMap::new();
    insert_resident_capacity_attrs(
        &mut attrs,
        &crate::kv_integration::ResidentCapacityDecision {
            enabled: true,
            capacity_known: true,
            admitted: false,
            capacity_tokens: 100,
            active_tokens: 60,
            pinned_tokens: 30,
            request_tokens: 8,
            minimum_free_tokens: 5,
            target_free_tokens: 10,
            projected_free_tokens: 2,
            admission_deficit_tokens: 3,
            ..crate::kv_integration::ResidentCapacityDecision::default()
        },
    );

    assert_eq!(
        attrs.get(attr_key::KV_CAPACITY_STATUS),
        Some(&json!("rejected"))
    );
    assert_eq!(
        attrs.get(attr_key::KV_CAPACITY_ADMISSION_DEFICIT_TOKENS),
        Some(&json!(3))
    );
    assert!(!attrs.contains_key(attr_key::REQUEST_ID));
    assert!(!attrs.contains_key(attr_key::SESSION_ID));
}

#[cfg(test)]
#[test]
fn default_local_text_serving_selects_iteration_scheduler() {
    assert!(scheduler_generation_driver_eligible(
        128, false, false, false, false, "disabled", false,
    ));
    assert!(!scheduler_generation_driver_eligible(
        0, false, false, false, false, "disabled", false,
    ));
}

#[cfg(test)]
#[test]
fn speculative_and_extended_serving_paths_select_the_scheduler_feature_driver() {
    assert!(!scheduler_generation_driver_eligible(
        128, true, false, false, false, "disabled", false,
    ));
    assert!(!scheduler_generation_driver_eligible(
        128, false, true, false, false, "disabled", false,
    ));
    assert!(!scheduler_generation_driver_eligible(
        128, false, false, true, false, "disabled", false,
    ));
    assert!(!scheduler_generation_driver_eligible(
        128,
        false,
        false,
        false,
        true,
        "native-mtp",
        false,
    ));
    assert!(!scheduler_generation_driver_eligible(
        128,
        false,
        false,
        false,
        false,
        "ngram-cache",
        false,
    ));
    assert!(!scheduler_generation_driver_eligible(
        128, false, false, false, false, "disabled", true,
    ));
}

#[cfg(test)]
pub(in crate::frontend) fn native_mtp_dispatch_counts_for_test() -> (usize, usize) {
    struct FakeNativeMtpRuntime {
        sampled_calls: usize,
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
    }

    let mut runtime = FakeNativeMtpRuntime { sampled_calls: 0 };
    let (predicted, draft) =
        decode_native_mtp(&mut runtime, "test", 0, None, 1).expect("sampled MTP dispatch");
    assert_eq!(predicted, 7);
    assert!(draft.is_none());
    (runtime.sampled_calls, 0)
}
