use crate::binary_transport::PredictionReturnHub;
use crate::binary_transport::PredictionReturnReceiver;
use crate::binary_transport::WireCondition;
use crate::frontend::EmbeddedOpenAiRequestDefaults;
use crate::frontend::GenerationLifecycleConfig;
use crate::frontend::GenerationReceiptConfig;
use crate::frontend::LinearProposalIngressConfig;
use crate::frontend::NativeMtpDraft;
use crate::frontend::NativeMtpStats;
use crate::frontend::SpeculativeDecodeConfig;
use crate::frontend::admission::GenerationTokenBudget;
use crate::frontend::decode_scheduler::VerifyWindowPipelineStats;
use crate::frontend::generation::DraftRunner;
use crate::frontend::generation::GenerationConcurrencyController;
use crate::frontend::generation::GenerationServiceEstimator;
use crate::frontend::generation::GenerationTokenLimit;
use crate::frontend::generation::InferenceGenerationIds;
use crate::frontend::generation::PersistentStageLanePool;
use crate::frontend::generation::PreparedGenerationPrompt;
use crate::frontend::iteration_scheduler::IterationScheduler;
use crate::frontend::native_mtp::NativeMtpDecodeTelemetry;
use crate::frontend::prefill::PrefillChunkPolicy;
use crate::frontend::speculative::InferenceSpeculativeStats;
use crate::kv_integration::KvStageIntegration;
use crate::runtime_state::RuntimeState;
use crate::telemetry::Telemetry;
use crate::telemetry::now_unix_nanos;
use serde_json::Value;
use serde_json::json;
use skippy_inference_api::ChatCompletionRequest;
use skippy_inference_api::FinishReason;
use skippy_inference_api::InferenceHookPolicy;
use skippy_inference_api::Usage;
use skippy_protocol::StageConfig;
use skippy_protocol::binary::StageReply;
use skippy_protocol::binary::StageReplyStats;
use skippy_runtime::SamplingConfig;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicUsize;
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;

pub(in crate::frontend) struct GenerationSessionLockEntry {
    pub(in crate::frontend) semaphore: Arc<Semaphore>,
    pub(in crate::frontend) users: AtomicUsize,
}

#[derive(Clone)]
pub(in crate::frontend) struct StageOpenAiBackend {
    pub(in crate::frontend) runtime: Arc<Mutex<RuntimeState>>,
    pub(in crate::frontend) workload: Arc<super::workload::CachedModelWorkload>,
    pub(in crate::frontend) config: StageConfig,
    pub(in crate::frontend) telemetry: Telemetry,
    pub(in crate::frontend) model_id: String,
    pub(in crate::frontend) default_max_tokens: u32,
    pub(in crate::frontend) request_defaults: EmbeddedOpenAiRequestDefaults,
    /// Render-only reasoning-control observations for this loaded model.
    pub(in crate::frontend) thinking: Option<skippy_inference_api::thinking::ThinkingControls>,
    pub(in crate::frontend) ctx_size: usize,
    pub(in crate::frontend) mode: InferenceBackendMode,
    pub(in crate::frontend) draft: Option<Arc<Mutex<DraftRunner>>>,
    pub(in crate::frontend) speculative_window: usize,
    pub(in crate::frontend) adaptive_speculative_window: bool,
    pub(in crate::frontend) ngram_max: usize,
    pub(in crate::frontend) speculative: SpeculativeDecodeConfig,
    /// Decides from measured throughput whether speculation is earning its
    /// keep. `None` leaves the resolved plan's setting in force unconditionally.
    pub(in crate::frontend) speculation_governor:
        Option<std::sync::Arc<crate::frontend::speculation_gate::SpeculationGovernor>>,
    /// Searches for the run-ahead budget when the plan states `auto` instead of
    /// a number. `None` leaves the plan's own figure in force.
    pub(in crate::frontend) runahead_governor:
        Option<std::sync::Arc<crate::frontend::runahead_search::RunaheadGovernor>>,
    pub(in crate::frontend) generation_limit: Arc<GenerationConcurrencyController>,
    pub(in crate::frontend) generation_queue_depth: Arc<AtomicUsize>,
    pub(in crate::frontend) generation_queue_limit: usize,
    pub(in crate::frontend) generation_admission_timeout: Duration,
    pub(in crate::frontend) generation_service_estimator: Arc<GenerationServiceEstimator>,
    pub(in crate::frontend) generation_session_locks:
        Arc<Mutex<BTreeMap<String, Arc<GenerationSessionLockEntry>>>>,
    pub(in crate::frontend) generation_token_budget: Arc<GenerationTokenBudget>,
    pub(in crate::frontend) hook_policy: Option<Arc<dyn InferenceHookPolicy>>,
    pub(in crate::frontend) generation_receipt: Option<GenerationReceiptConfig>,
    pub(in crate::frontend) generation_lifecycle: Option<GenerationLifecycleConfig>,
    pub(in crate::frontend) linear_proposal_ingress: Option<LinearProposalIngressConfig>,
    pub(in crate::frontend) kv: Option<Arc<KvStageIntegration>>,
    pub(in crate::frontend) iteration_scheduler: IterationScheduler,
}

#[derive(Clone)]
#[allow(clippy::large_enum_variant)]
pub(in crate::frontend) enum InferenceBackendMode {
    LocalRuntime,
    EmbeddedStageZero {
        config: StageConfig,
        prefill_chunk_policy: PrefillChunkPolicy,
        activation_width: i32,
        downstream_wire_condition: WireCondition,
        prefill_reply_credit_limit: usize,
        lane_pool: Option<Arc<PersistentStageLanePool>>,
        prediction_returns: Option<Arc<PredictionReturnHub>>,
    },
}

impl InferenceBackendMode {
    pub(in crate::frontend) const EMBEDDED_STAGE_ZERO_LABEL: &'static str = "embedded-stage0";

    pub(in crate::frontend) fn label(&self) -> &'static str {
        match self {
            Self::LocalRuntime => "local-runtime",
            Self::EmbeddedStageZero { .. } => Self::EMBEDDED_STAGE_ZERO_LABEL,
        }
    }
}

pub(in crate::frontend) struct PhaseTimer {
    pub(in crate::frontend) start_unix_nanos: u64,
    pub(in crate::frontend) start_instant: Instant,
}

impl PhaseTimer {
    pub(in crate::frontend) fn start() -> Self {
        Self {
            start_unix_nanos: now_unix_nanos() as u64,
            start_instant: Instant::now(),
        }
    }

    pub(in crate::frontend) fn elapsed_ms(&self) -> f64 {
        self.start_instant.elapsed().as_secs_f64() * 1000.0
    }
}

pub(in crate::frontend) fn decode_token_phase(decode_step: u32) -> &'static str {
    match decode_step {
        0 => "cold",
        1..=7 => "warmup",
        _ => "steady",
    }
}

#[derive(Default)]
pub(in crate::frontend) struct GenerationMetrics {
    pub(in crate::frontend) detokenize_ms: f64,
    pub(in crate::frontend) text_emit_ms: f64,
    pub(in crate::frontend) eog_check_ms: f64,
}

pub(in crate::frontend) struct PreparedTextPrompt {
    pub(in crate::frontend) token_ids: Vec<i32>,
    pub(in crate::frontend) max_tokens: u32,
}

pub(in crate::frontend) struct LocalGeneration<'a> {
    pub(in crate::frontend) prompt_token_ids: &'a [i32],
    pub(in crate::frontend) recurrent_cache_prefix_token_ids: Option<&'a [i32]>,
    pub(in crate::frontend) max_tokens: u32,
    pub(in crate::frontend) sampling: &'a SamplingConfig,
    pub(in crate::frontend) chat_sampling_metadata: Option<&'a str>,
    pub(in crate::frontend) speculative: &'a SpeculativeDecodeConfig,
    pub(in crate::frontend) native_mtp_enabled: bool,
    pub(in crate::frontend) hook_request: Option<ChatCompletionRequest>,
    pub(in crate::frontend) hook_runtime: Option<tokio::runtime::Handle>,
    pub(in crate::frontend) cancellation: Option<&'a skippy_inference_api::CancellationToken>,
    pub(in crate::frontend) ids: &'a InferenceGenerationIds,
}

pub(in crate::frontend) struct EmbeddedStageZeroGeneration<'a> {
    pub(in crate::frontend) config: &'a StageConfig,
    pub(in crate::frontend) prefill_chunk_policy: &'a PrefillChunkPolicy,
    pub(in crate::frontend) activation_width: i32,
    pub(in crate::frontend) downstream_wire_condition: WireCondition,
    pub(in crate::frontend) prefill_reply_credit_limit: usize,
    pub(in crate::frontend) lane_pool: Option<Arc<PersistentStageLanePool>>,
    pub(in crate::frontend) prediction_return: Option<PredictionReturnReceiver>,
    pub(in crate::frontend) draft: Option<Arc<Mutex<DraftRunner>>>,
    pub(in crate::frontend) speculative_window: usize,
    pub(in crate::frontend) adaptive_speculative_window: bool,
    pub(in crate::frontend) ngram_max: usize,
    pub(in crate::frontend) speculative: &'a SpeculativeDecodeConfig,
    pub(in crate::frontend) native_mtp_enabled: bool,
    /// The final stage's DFlash draft when this split speculates with one;
    /// stage 0 then sizes each block from measured cost and acceptance.
    pub(in crate::frontend) split_dflash_draft: Option<&'a std::path::Path>,
    /// The final stage's draft-context counters, from its latest replies.
    pub(in crate::frontend) split_dflash_health: SplitDFlashHealth,
    pub(in crate::frontend) prompt_token_ids: &'a [i32],
    pub(in crate::frontend) recurrent_cache_prefix_token_ids: Option<&'a [i32]>,
    pub(in crate::frontend) max_tokens: u32,
    pub(in crate::frontend) sampling: &'a SamplingConfig,
    pub(in crate::frontend) chat_sampling_metadata: Option<&'a str>,
    pub(in crate::frontend) hook_request: Option<ChatCompletionRequest>,
    pub(in crate::frontend) hook_runtime: Option<tokio::runtime::Handle>,
    pub(in crate::frontend) cancellation: Option<&'a skippy_inference_api::CancellationToken>,
    pub(in crate::frontend) ids: &'a InferenceGenerationIds,
}

/// The largest split DFlash draft-context counters a final stage reported
/// for this generation's session, whose replies carry running totals, and
/// the feature-row bytes stage 0 sent it.
#[derive(Debug, Default)]
pub(in crate::frontend) struct SplitDFlashHealth {
    feature_rows_missing: std::sync::atomic::AtomicI64,
    draft_context_resets: std::sync::atomic::AtomicI64,
    verify_rollbacks_in_place: std::sync::atomic::AtomicI64,
    verify_rollbacks_checkpointed: std::sync::atomic::AtomicI64,
    draft_us: std::sync::atomic::AtomicI64,
    drafts: std::sync::atomic::AtomicI64,
    ingest_us: std::sync::atomic::AtomicI64,
    prefill_feature_bytes: std::sync::atomic::AtomicUsize,
    decode_feature_bytes: std::sync::atomic::AtomicUsize,
}

/// Split DFlash counters for a generation's response timings.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(in crate::frontend) struct SplitDFlashCounts {
    pub(in crate::frontend) feature_rows_missing: i64,
    pub(in crate::frontend) draft_context_resets: i64,
    /// Largest per-stage count of verify windows a downstream stage rolled
    /// back in place, and held in a host checkpoint.
    pub(in crate::frontend) downstream_rollbacks_in_place: i64,
    pub(in crate::frontend) downstream_rollbacks_checkpointed: i64,
    /// The final stage's draft time, block drafts, and the part of that time
    /// spent injecting feature rows.
    pub(in crate::frontend) draft_us: i64,
    pub(in crate::frontend) drafts: i64,
    pub(in crate::frontend) ingest_us: i64,
    /// The same on stage 0, when it could be read.
    pub(in crate::frontend) stage0_rollbacks: Option<skippy_runtime::VerifyRollbackCounts>,
}

impl SplitDFlashHealth {
    pub(in crate::frontend) fn observe(&self, stats: &skippy_protocol::binary::StageReplyStats) {
        use std::sync::atomic::Ordering;
        self.feature_rows_missing
            .fetch_max(stats.dflash_feature_rows_missing, Ordering::Relaxed);
        self.draft_context_resets
            .fetch_max(stats.dflash_draft_context_resets, Ordering::Relaxed);
        self.verify_rollbacks_in_place
            .fetch_max(stats.verify_rollbacks_in_place, Ordering::Relaxed);
        self.verify_rollbacks_checkpointed
            .fetch_max(stats.verify_rollbacks_checkpointed, Ordering::Relaxed);
        self.draft_us
            .fetch_max(stats.dflash_draft_us, Ordering::Relaxed);
        self.drafts
            .fetch_max(stats.dflash_drafts, Ordering::Relaxed);
        self.ingest_us
            .fetch_max(stats.dflash_ingest_us, Ordering::Relaxed);
    }

    /// Counts the DFlash feature-row bytes of a frame stage 0 forwarded, in
    /// prefill or in decode (verify windows and fused restores included).
    pub(in crate::frontend) fn record_feature_bytes(&self, prefill: bool, bytes: usize) {
        use std::sync::atomic::Ordering;
        let counter = if prefill {
            &self.prefill_feature_bytes
        } else {
            &self.decode_feature_bytes
        };
        counter.fetch_add(bytes, Ordering::Relaxed);
    }

    pub(in crate::frontend) fn feature_bytes(&self) -> SplitDFlashFeatureBytes {
        use std::sync::atomic::Ordering;
        SplitDFlashFeatureBytes {
            prefill_bytes: self.prefill_feature_bytes.load(Ordering::Relaxed),
            decode_bytes: self.decode_feature_bytes.load(Ordering::Relaxed),
        }
    }

    pub(in crate::frontend) fn counts(
        &self,
        stage0_rollbacks: Option<skippy_runtime::VerifyRollbackCounts>,
    ) -> SplitDFlashCounts {
        use std::sync::atomic::Ordering;
        SplitDFlashCounts {
            feature_rows_missing: self.feature_rows_missing.load(Ordering::Relaxed),
            draft_context_resets: self.draft_context_resets.load(Ordering::Relaxed),
            downstream_rollbacks_in_place: self.verify_rollbacks_in_place.load(Ordering::Relaxed),
            downstream_rollbacks_checkpointed: self
                .verify_rollbacks_checkpointed
                .load(Ordering::Relaxed),
            draft_us: self.draft_us.load(Ordering::Relaxed),
            drafts: self.drafts.load(Ordering::Relaxed),
            ingest_us: self.ingest_us.load(Ordering::Relaxed),
            stage0_rollbacks,
        }
    }
}

impl EmbeddedStageZeroGeneration<'_> {
    /// Whether this generation drafts with a split DFlash final stage, whose
    /// decode and verify messages say so.
    pub(in crate::frontend) fn split_dflash(&self) -> bool {
        self.split_dflash_draft.is_some()
    }

    /// Records the DFlash feature-row bytes of a frame this generation
    /// forwarded from stage 0.
    pub(in crate::frontend) fn record_forwarded_features(&self, prefill: bool, bytes: usize) {
        if self.split_dflash() {
            self.split_dflash_health
                .record_feature_bytes(prefill, bytes);
        }
    }

    /// Records the draft-context counters a downstream reply carries.
    pub(in crate::frontend) fn observe_reply_stats(
        &self,
        stats: &skippy_protocol::binary::StageReplyStats,
    ) {
        if self.split_dflash() {
            self.split_dflash_health.observe(stats);
        }
    }

    /// Draft depth to request of the final stage on decode and verify messages.
    pub(in crate::frontend) fn native_mtp_draft_tokens(&self) -> usize {
        if self.native_mtp_enabled {
            self.speculative.native_mtp.max_draft_tokens
        } else {
            0
        }
    }
}

pub(in crate::frontend) struct SplitMultimodalGeneration<'a> {
    pub(in crate::frontend) prompt: PreparedGenerationPrompt,
    pub(in crate::frontend) max_tokens: GenerationTokenLimit,
    pub(in crate::frontend) stop: Option<&'a skippy_inference_api::StopSequence>,
    pub(in crate::frontend) sampling: SamplingConfig,
    pub(in crate::frontend) cancellation: Option<&'a skippy_inference_api::CancellationToken>,
    pub(in crate::frontend) ids: InferenceGenerationIds,
    pub(in crate::frontend) config: StageConfig,
    pub(in crate::frontend) activation_width: i32,
    pub(in crate::frontend) downstream_wire_condition: WireCondition,
    pub(in crate::frontend) lane_pool: Arc<PersistentStageLanePool>,
    pub(in crate::frontend) prediction_return: Option<PredictionReturnReceiver>,
    pub(in crate::frontend) emulation_active: bool,
}

pub(in crate::frontend) struct EmbeddedLocalOutput {
    pub(in crate::frontend) output: skippy_runtime::ActivationFrame,
    pub(in crate::frontend) runtime_lock_wait_ms: f64,
    pub(in crate::frontend) runtime_lock_hold_ms: f64,
}

#[derive(Default)]
pub(in crate::frontend) struct EmbeddedExecutionStats {
    pub(in crate::frontend) stage0_compute_ms: f64,
    pub(in crate::frontend) runtime_lock_wait_ms: f64,
    pub(in crate::frontend) runtime_lock_hold_ms: f64,
    pub(in crate::frontend) activation_encode_ms: f64,
    pub(in crate::frontend) output_activation_bytes: usize,
    pub(in crate::frontend) forward_activation_bytes: usize,
    pub(in crate::frontend) forward_write_ms: f64,
    pub(in crate::frontend) downstream_wait_ms: f64,
}

pub(in crate::frontend) struct EmbeddedStageExecution {
    pub(in crate::frontend) reply: StageReply,
    pub(in crate::frontend) stats: EmbeddedExecutionStats,
    pub(in crate::frontend) elapsed_ms: f64,
}

pub(in crate::frontend) struct EmbeddedFusedFirstDecode {
    pub(in crate::frontend) predicted: i32,
    pub(in crate::frontend) predicted_tokens: Vec<i32>,
    pub(in crate::frontend) native_mtp_draft: Option<NativeMtpDraft>,
    pub(in crate::frontend) reply_stats: StageReplyStats,
    pub(in crate::frontend) execution: EmbeddedExecutionStats,
    pub(in crate::frontend) elapsed_ms: f64,
    pub(in crate::frontend) token_phase: &'static str,
    pub(in crate::frontend) message_kind: &'static str,
}

pub(in crate::frontend) struct GeneratedText {
    pub(in crate::frontend) prompt_tokens: u32,
    pub(in crate::frontend) completion_tokens: u32,
    pub(in crate::frontend) cache_status: &'static str,
    pub(in crate::frontend) cached_prompt_tokens: u32,
    pub(in crate::frontend) matched_prefix_tokens: u32,
    pub(in crate::frontend) suffix_prefill_tokens: u32,
    pub(in crate::frontend) cache_hit_kind: Option<&'static str>,
    pub(in crate::frontend) native_mtp_stats: NativeMtpStats,
    pub(in crate::frontend) native_mtp_decode_telemetry: Option<NativeMtpDecodeTelemetry>,
    pub(in crate::frontend) verify_window_pipeline_stats: Option<VerifyWindowPipelineStats>,
    pub(in crate::frontend) speculative_stats: Option<InferenceSpeculativeStats>,
    pub(in crate::frontend) prompt_ms: f64,
    pub(in crate::frontend) predicted_ms: f64,
    pub(in crate::frontend) queue_wait_ms: f64,
    pub(in crate::frontend) restore_ms: f64,
    pub(in crate::frontend) split_prefix_cache: Option<SplitPrefixCacheStats>,
    pub(in crate::frontend) text: String,
    pub(in crate::frontend) finish_reason: FinishReason,
    pub(in crate::frontend) detokenize_ms: f64,
    pub(in crate::frontend) text_emit_ms: f64,
    pub(in crate::frontend) eog_check_ms: f64,
    /// One entry per completion token, when the request asked for logprobs.
    pub(in crate::frontend) token_logprobs: Option<Vec<GeneratedTokenLogprob>>,
    pub(in crate::frontend) activation_wire: Option<ActivationWireStats>,
}

/// Encoded activation bytes stage 0 wrote to the next stage for one response:
/// the first hop only, payload bytes without frame headers. Decode includes
/// speculative verify windows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::frontend) struct ActivationWireStats {
    /// The configured codec, or `auto-lossless-v1` when the producer picks an
    /// exact encoding per frame.
    pub(in crate::frontend) codec: &'static str,
    pub(in crate::frontend) prefill_bytes: usize,
    pub(in crate::frontend) decode_bytes: usize,
    /// On a split DFlash target, the share of those bytes that were the
    /// draft's feature rows.
    pub(in crate::frontend) dflash_features: Option<SplitDFlashFeatureBytes>,
}

/// DFlash feature-row payload bytes stage 0 sent to the next stage, by phase.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(in crate::frontend) struct SplitDFlashFeatureBytes {
    pub(in crate::frontend) prefill_bytes: usize,
    pub(in crate::frontend) decode_bytes: usize,
}

impl ActivationWireStats {
    fn insert_response_timings(
        &self,
        timings: &mut BTreeMap<String, Value>,
        prefill_tokens: u32,
        completion_tokens: u32,
    ) {
        let per_token = |bytes: usize, tokens: u32| {
            if tokens == 0 {
                0.0
            } else {
                bytes as f64 / f64::from(tokens)
            }
        };
        timings.insert("activation_codec".to_string(), json!(self.codec));
        timings.insert(
            "activation_prefill_bytes".to_string(),
            json!(self.prefill_bytes),
        );
        timings.insert(
            "activation_prefill_bytes_per_token".to_string(),
            json!(per_token(self.prefill_bytes, prefill_tokens)),
        );
        timings.insert(
            "activation_decode_bytes".to_string(),
            json!(self.decode_bytes),
        );
        timings.insert(
            "activation_decode_bytes_per_token".to_string(),
            json!(per_token(self.decode_bytes, completion_tokens)),
        );
        if let Some(features) = self.dflash_features {
            timings.insert(
                "dflash_feature_prefill_bytes".to_string(),
                json!(features.prefill_bytes),
            );
            timings.insert(
                "dflash_feature_prefill_bytes_per_token".to_string(),
                json!(per_token(features.prefill_bytes, prefill_tokens)),
            );
            timings.insert(
                "dflash_feature_decode_bytes".to_string(),
                json!(features.decode_bytes),
            );
            timings.insert(
                "dflash_feature_decode_bytes_per_token".to_string(),
                json!(per_token(features.decode_bytes, completion_tokens)),
            );
        }
    }
}

/// A split's prompt prefix cache for one response, as stage 0 saw it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(in crate::frontend) struct SplitPrefixCacheStats {
    /// Prompt tokens every stage restored instead of prefilling.
    pub(in crate::frontend) hit_tokens: usize,
    /// Stages that confirmed the restore, stage 0 included.
    pub(in crate::frontend) hit_stages: u32,
    /// Activation bytes stage 0 wrote to the next stage while prefilling the
    /// rest of the prompt: the first hop, payload only.
    pub(in crate::frontend) prefill_bytes_sent: usize,
    /// Activation bytes the restore kept off the wire, over every hop between
    /// the stages that restored.
    pub(in crate::frontend) bytes_saved: usize,
}

impl SplitPrefixCacheStats {
    /// Stats for a prefill that restored `hit_tokens` on `hit_stages` stages
    /// and then sent `prefill_bytes_sent` for `prefilled_tokens` more. Saved
    /// bytes are priced at this request's measured rate per prefilled token,
    /// which includes any split DFlash feature rows, or at an f32 activation
    /// row when nothing was prefilled.
    pub(in crate::frontend) fn new(
        hit_tokens: usize,
        hit_stages: u32,
        prefill_bytes_sent: usize,
        prefilled_tokens: usize,
        activation_width: i32,
    ) -> Self {
        let bytes_per_token = prefill_bytes_sent
            .checked_div(prefilled_tokens)
            .unwrap_or_else(|| {
                usize::try_from(activation_width).unwrap_or(0) * std::mem::size_of::<f32>()
            });
        let hops = usize::try_from(hit_stages.saturating_sub(1)).unwrap_or(0);
        Self {
            hit_tokens,
            hit_stages,
            prefill_bytes_sent,
            bytes_saved: hit_tokens
                .saturating_mul(bytes_per_token)
                .saturating_mul(hops),
        }
    }

    fn insert_response_timings(&self, timings: &mut BTreeMap<String, Value>) {
        timings.insert(
            "prefix_cache_hit_tokens".to_string(),
            json!(self.hit_tokens),
        );
        timings.insert(
            "prefix_cache_hit_stages".to_string(),
            json!(self.hit_stages),
        );
        timings.insert(
            "prefix_cache_prefill_bytes_sent".to_string(),
            json!(self.prefill_bytes_sent),
        );
        timings.insert(
            "prefix_cache_bytes_saved".to_string(),
            json!(self.bytes_saved),
        );
    }
}

/// A completion token's log-probability and its alternatives, with their text.
#[derive(Debug, Clone, PartialEq)]
pub(in crate::frontend) struct GeneratedTokenLogprob {
    pub(in crate::frontend) token: GeneratedLogprobCandidate,
    pub(in crate::frontend) top: Vec<GeneratedLogprobCandidate>,
}

#[derive(Debug, Clone, PartialEq)]
pub(in crate::frontend) struct GeneratedLogprobCandidate {
    pub(in crate::frontend) text: String,
    pub(in crate::frontend) bytes: Vec<u8>,
    /// `None` when the runtime could not read the logits row.
    pub(in crate::frontend) logprob: Option<f32>,
}

impl GeneratedText {
    pub(in crate::frontend) fn usage(&self) -> Usage {
        Usage::new(self.prompt_tokens, self.completion_tokens)
            .with_cached_tokens(self.cached_prompt_tokens)
    }

    pub(in crate::frontend) fn timings(&self) -> Option<BTreeMap<String, Value>> {
        let stats = self.native_mtp_stats;
        let native_totals = self
            .native_mtp_decode_telemetry
            .and_then(NativeMtpDecodeTelemetry::composite_proposal_totals)
            .unwrap_or((stats.drafted_tokens, stats.accepted_tokens));
        let (drafted_tokens, accepted_tokens) = self
            .speculative_stats
            .as_ref()
            .filter(|stats| stats.windows > 0)
            .map_or(native_totals, |stats| {
                (
                    u64::try_from(stats.draft_tokens).unwrap_or(u64::MAX),
                    u64::try_from(stats.accepted_tokens).unwrap_or(u64::MAX),
                )
            });
        let mut timings = BTreeMap::from([
            ("prompt_n".to_string(), json!(self.prompt_tokens)),
            ("prompt_ms".to_string(), json!(self.prompt_ms)),
            ("queue_wait_ms".to_string(), json!(self.queue_wait_ms)),
            ("cache_restore_ms".to_string(), json!(self.restore_ms)),
            (
                "suffix_prefill_n".to_string(),
                json!(self.suffix_prefill_tokens),
            ),
            (
                "prompt_per_second".to_string(),
                json!(tokens_per_second(self.prompt_tokens, self.prompt_ms)),
            ),
            ("predicted_n".to_string(), json!(self.completion_tokens)),
            ("predicted_ms".to_string(), json!(self.predicted_ms)),
            (
                "predicted_per_second".to_string(),
                json!(tokens_per_second(self.completion_tokens, self.predicted_ms)),
            ),
            ("draft_n".to_string(), json!(drafted_tokens)),
            ("draft_n_accepted".to_string(), json!(accepted_tokens)),
            (
                "native_mtp_rejected".to_string(),
                json!(stats.rejected_tokens),
            ),
            (
                "native_mtp_proposal_compute_us".to_string(),
                json!(stats.proposal_compute_us),
            ),
            (
                "native_mtp_verification_compute_us".to_string(),
                json!(stats.verification_compute_us),
            ),
            (
                "native_mtp_verifications".to_string(),
                json!(stats.verification_count),
            ),
        ]);
        if let Some(telemetry) = self.native_mtp_decode_telemetry {
            telemetry.insert_response_timings(&mut timings);
        }
        if let Some(stats) = self.verify_window_pipeline_stats.as_ref() {
            stats.insert_response_timings(&mut timings);
        }
        if let Some(stats) = self.speculative_stats.as_ref() {
            stats.insert_response_timings(&mut timings);
        }
        if let Some(wire) = self.activation_wire.as_ref() {
            // Stage 0 prefills all but the last prompt token, which the first
            // decode carries; a prefix-cache hit forwards only the suffix.
            let prefill_tokens = if self.suffix_prefill_tokens > 0 {
                self.suffix_prefill_tokens
            } else {
                self.prompt_tokens.saturating_sub(1)
            };
            wire.insert_response_timings(&mut timings, prefill_tokens, self.completion_tokens);
        }
        if let Some(stats) = self.split_prefix_cache.as_ref() {
            stats.insert_response_timings(&mut timings);
        }
        Some(timings)
    }
}

fn tokens_per_second(token_count: u32, elapsed_ms: f64) -> f64 {
    if elapsed_ms > 0.0 {
        f64::from(token_count) * 1_000.0 / elapsed_ms
    } else {
        0.0
    }
}
