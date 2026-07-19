use std::{
    collections::BTreeMap,
    future::Future,
    net::{SocketAddr, TcpStream},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, anyhow, bail};
use async_trait::async_trait;
use axum::{
    Router,
    body::Body,
    extract::State,
    http::{Request, StatusCode},
    middleware::{self, Next},
    response::Response,
};
use base64::Engine;
use futures_util::{StreamExt, stream};
use openai_frontend::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, ChatCompletionStream,
    ChatHookAction, ChatHookOutcome, CompactingOpenAiBackend, CompactionConfig, CompletionChunk,
    CompletionRequest, CompletionResponse, CompletionStream, FinishReason, GenerationHookSignals,
    GuardedOpenAiBackend, GuardrailMode, GuardrailPolicy, GuardrailPolicyHandle, MessageContent,
    MessageContentPart, ModelId, ModelObject, OpenAiBackend, OpenAiError, OpenAiErrorKind,
    OpenAiHookPolicy, OpenAiRequestContext, OpenAiResult, PrefillHookSignals, ReasoningEffort,
    RetryExhaustionMode, StreamingGuardrailMode, Usage, apply_chat_hook_outcome,
    chat_mesh_hooks_enabled,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use skippy_metrics::attr as attr_key;
use skippy_protocol::binary::{
    LLAMA_TOKEN_NULL, MAX_STAGE_LOGIT_BIAS, StageLogitBias as WireLogitBias, StageReply,
    StageReplyStats, StageSamplingConfig as WireSamplingConfig, StageStateHeader, StageWireMessage,
    WireActivationDType, WireMessageKind, WireReplyKind, recv_ready, recv_reply, state_flags,
    write_stage_message,
};
use skippy_protocol::{MessageBase, SCHEMA_VERSION, StageConfig, StageTopology};
use skippy_runtime::{
    ActivationFrame, ChatReasoningFormat, ChatTemplateJsonOptions, ChatTemplateOptions,
    FlashAttentionType as RuntimeFlashAttentionType, GenerationSignalWindow,
    LogitBias as RuntimeLogitBias, MAX_LOGIT_BIAS, MediaInput, ModelInfo, RuntimeConfig,
    RuntimeLoadMode, SamplingConfig, StageModel, StageSession, TokenSignal,
};
use tokio::{
    net::TcpListener,
    sync::{OwnedSemaphorePermit, Semaphore, TryAcquireError, mpsc},
    task,
};

use crate::{
    binary_transport::{
        BinaryStageExecutionOptions, DecodeFrameBatcher, PredictionReturnHub,
        PredictionReturnReceiver, WireCondition, connect_binary_downstream,
        forwarded_stage_message, forwarded_stage_message_timed, run_binary_stage_message,
        send_client_ready_hello_if_enabled, stage_output_activation_capacity,
        write_stage_message_conditioned,
    },
    cli::{OpenAiGuardrailsCliMode, ServeOpenAiArgs},
    config::{load_json, validate_config},
    kv_integration::{KvStageIntegration, proactive_eviction_attrs, proactive_eviction_error_kind},
    runtime_state::{RuntimeSessionStats, RuntimeState, load_runtime},
    telemetry::{Telemetry, lifecycle_attrs, now_unix_nanos},
};

mod admission;
mod backend;
mod decode_batcher;
mod decode_scheduler;
mod embedded_execution;
mod embedded_generation;
mod generation_flow;
mod local_generation;
mod native_mtp;
mod prefill;
mod prefix_cache;
mod prompting;
mod request;
mod speculative;
mod tool_emulation;
mod util;

#[cfg(test)]
use prompting::parse_emulated_chat_output;
mod wire_messages;

use self::{
    admission::{GenerationTokenBudget, GenerationTokenBudgetRequest},
    decode_batcher::DecodeBatcher,
    decode_scheduler::*,
    native_mtp::*,
    prefill::*,
    request::*,
    speculative::*,
    util::*,
    wire_messages::*,
};

static OPENAI_GENERATION_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Sentinel meaning "no caller-specified max completion length; let the
/// request consume the entire remaining context window when the client
/// also omits max_tokens".
///
/// This is opt-in and should only be wired in by callers that have made
/// a deliberate decision to allow unbounded chat completions. The
/// embedded mesh-llm wiring uses [`DEFAULT_EMBEDDED_MAX_TOKENS`] instead.
pub const CONTEXT_BUDGET_MAX_TOKENS: u32 = u32::MAX;

/// Default max completion tokens for embedded mesh-llm chat serving when
/// the client omits max_tokens. Bounded so that an adversarial or
/// non-terminating generation cannot run for the full context window.
/// Clients can still request more by sending max_tokens explicitly, up
/// to the remaining context budget.
///
/// When the configured context window is smaller than this value, the
/// request is silently clamped to whatever remaining budget exists
/// rather than rejected — see [`GenerationTokenLimit::resolve`].
pub const DEFAULT_EMBEDDED_MAX_TOKENS: u32 = 4096;
const GENERATION_ADMISSION_TIMEOUT: Duration = Duration::from_secs(10);
const GENERATION_RETRY_AFTER_SECS: u64 = 1;
const MAX_EXACT_REPLAY_TOKENS: usize = 8;

pub async fn serve_openai(args: ServeOpenAiArgs) -> Result<()> {
    let config = load_json::<StageConfig>(&args.config)
        .with_context(|| format!("load stage config {}", args.config.display()))?;
    let topology = match args.topology.as_ref() {
        Some(path) => Some(
            load_json::<StageTopology>(path)
                .with_context(|| format!("load topology {}", path.display()))?,
        ),
        None => None,
    };
    validate_config(&config, topology.as_ref())?;
    if args.first_stage_addr.is_none() && config.downstream.is_some() {
        bail!("serve-openai local backend requires a final/single-stage config with no downstream");
    }
    if args.prefill_chunk_size == 0 {
        bail!("--prefill-chunk-size must be greater than zero");
    }
    if args.generation_concurrency == 0 {
        bail!("--generation-concurrency must be greater than zero");
    }
    let speculative = load_standalone_speculative_config(args.speculative_config.as_ref())?;

    let runtime = load_runtime(&config)?.ok_or_else(|| {
        anyhow!("serve-openai requires a stage config with model_path for tokenization and decode")
    })?;
    let model_id = ModelId::new(args.model_id.unwrap_or_else(|| config.model_id.clone()))
        .map_err(|error| anyhow!("invalid OpenAI model id: {error}"))?
        .into_string();
    if args.first_stage_addr.is_some() {
        bail!(
            "--first-stage-addr is no longer supported; direct prediction return requires embedded stage-0 OpenAI serving via serve-binary --openai-bind-addr"
        );
    }
    let mode = OpenAiBackendMode::LocalRuntime;
    let mode_label = mode.label();
    let telemetry = Telemetry::new(
        args.metrics_otlp_grpc,
        args.telemetry_queue_capacity,
        config.clone(),
        args.telemetry_level,
    );
    telemetry.emit("stage.openai_server_start", lifecycle_attrs(&config));
    if matches!(&mode, OpenAiBackendMode::LocalRuntime) {
        ensure_generation_concurrency_fits_lanes(
            args.generation_concurrency,
            config.lane_count,
            "--generation-concurrency",
        )?;
        prewarm_generation_sessions(
            &runtime,
            args.generation_concurrency,
            &telemetry,
            &config,
            "stage.openai_runtime_prewarm",
        )
        .context("prewarm OpenAI runtime sessions")?;
    }
    let kv = KvStageIntegration::from_config(&config)?.map(Arc::new);
    let ctx_size = usize::try_from(config.ctx_size).unwrap_or(usize::MAX);
    let decode_batcher = DecodeBatcher::new(runtime.clone(), args.generation_concurrency);
    let decode_frame_batcher =
        DecodeFrameBatcher::new(runtime.clone(), args.generation_concurrency);
    let backend: Arc<dyn OpenAiBackend> = Arc::new(StageOpenAiBackend {
        runtime,
        config,
        telemetry: telemetry.clone(),
        model_id: model_id.clone(),
        default_max_tokens: args.default_max_tokens,
        request_defaults: EmbeddedOpenAiRequestDefaults::default(),
        ctx_size,
        mode,
        draft: None,
        speculative_window: 0,
        adaptive_speculative_window: false,
        ngram_min: standalone_simple_ngram_min(&speculative),
        ngram_max: standalone_simple_ngram_max(&speculative),
        speculative,
        generation_limit: Arc::new(Semaphore::new(args.generation_concurrency)),
        generation_queue_depth: Arc::new(AtomicUsize::new(0)),
        generation_queue_limit: args.generation_concurrency,
        generation_token_budget: Arc::new(GenerationTokenBudget::new(ctx_size)),
        hook_policy: None,
        kv,
        decode_batcher,
        decode_frame_batcher,
    });
    let backend = OpenAiGuardrailsConfig::for_standalone_mode(args.openai_guardrails)
        .wrap_backend_with_context_limit(backend, Some(ctx_size));
    let app: Router = instrumented_openai_router(backend, telemetry.clone());

    println!(
        "skippy-server listening: openai={} model_id={} backend={} generation_concurrency={}",
        args.bind_addr, model_id, mode_label, args.generation_concurrency,
    );

    let listener = TcpListener::bind(args.bind_addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SpeculativeDecodeConfig {
    pub requested_strategy: String,
    pub effective_strategy: String,
    pub native_mtp: NativeMtpProposalConfig,
    pub ngram: Option<NgramProposalConfig>,
    pub extension: Option<NgramExtensionConfig>,
    pub verify_window: VerifyWindowConfig,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NativeMtpProposalConfig {
    pub enabled: bool,
    pub max_draft_tokens: usize,
    pub min_draft_tokens: usize,
    pub reject_cooldown_tokens: usize,
    pub suppress_cooldown_drafts: bool,
    pub suppress_cooldown_draft_limit: usize,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum NgramProposerKind {
    Simple,
    Cache,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NgramProposalConfig {
    pub kind: NgramProposerKind,
    pub min_ngram: usize,
    pub max_ngram: usize,
    pub max_proposal_tokens: usize,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NgramExtensionConfig {
    pub initial_tokens: usize,
    pub max_tokens: usize,
    pub tail_backoff_proposals: usize,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VerifyWindowConfig {
    pub min_tokens: usize,
    pub max_tokens: usize,
    pub pipeline_depth: usize,
}

impl Default for SpeculativeDecodeConfig {
    fn default() -> Self {
        Self {
            requested_strategy: "auto".to_string(),
            effective_strategy: "disabled".to_string(),
            native_mtp: NativeMtpProposalConfig {
                enabled: false,
                max_draft_tokens: 1,
                min_draft_tokens: 0,
                reject_cooldown_tokens: 0,
                suppress_cooldown_drafts: false,
                suppress_cooldown_draft_limit: 0,
            },
            ngram: None,
            extension: None,
            verify_window: VerifyWindowConfig {
                min_tokens: 1,
                max_tokens: 4,
                pipeline_depth: 1,
            },
        }
    }
}

impl SpeculativeDecodeConfig {
    pub fn validate(&self) -> Result<()> {
        if self.requested_strategy.trim().is_empty() || self.effective_strategy.trim().is_empty() {
            bail!("speculative decode strategies must not be empty");
        }
        if self.native_mtp.min_draft_tokens > self.native_mtp.max_draft_tokens {
            bail!("native MTP min_draft_tokens must not exceed max_draft_tokens");
        }
        if let Some(ngram) = &self.ngram
            && (ngram.min_ngram == 0
                || ngram.min_ngram > ngram.max_ngram
                || ngram.max_proposal_tokens < ngram.min_ngram)
        {
            bail!(
                "N-gram proposer requires 0 < min_ngram <= max_ngram and max_proposal_tokens >= min_ngram"
            );
        }
        if let Some(ngram) = &self.ngram
            && ngram.kind == NgramProposerKind::Cache
            && ngram.max_ngram > skippy_runtime::NGRAM_CACHE_MAX_NGRAM
        {
            bail!(
                "cache N-gram proposer max_ngram must not exceed llama.cpp limit {}",
                skippy_runtime::NGRAM_CACHE_MAX_NGRAM
            );
        }
        if self.extension.is_some() && (!self.native_mtp.enabled || self.ngram.is_none()) {
            bail!("N-gram extension requires both native MTP and an N-gram proposer");
        }
        if let Some(extension) = &self.extension
            && (extension.initial_tokens == 0
                || extension.initial_tokens > extension.max_tokens
                || extension.max_tokens == 0)
        {
            bail!("N-gram extension requires 0 < initial_tokens <= max_tokens");
        }
        if self.verify_window.min_tokens == 0
            || self.verify_window.min_tokens > self.verify_window.max_tokens
            || self.verify_window.pipeline_depth == 0
        {
            bail!("verify window requires 0 < min_tokens <= max_tokens and pipeline_depth > 0");
        }
        Ok(())
    }

    fn insert_telemetry_attrs(&self, attrs: &mut BTreeMap<String, Value>) {
        attrs.insert(
            "llama_stage.spec.requested_strategy".to_string(),
            json!(self.requested_strategy),
        );
        attrs.insert(
            "llama_stage.spec.effective_strategy".to_string(),
            json!(self.effective_strategy),
        );
    }
}

fn load_standalone_speculative_config(path: Option<&PathBuf>) -> Result<SpeculativeDecodeConfig> {
    let config = match path {
        Some(path) => load_json(path)
            .with_context(|| format!("load speculative decode config {}", path.display()))?,
        None => SpeculativeDecodeConfig::default(),
    };
    config.validate()?;
    Ok(config)
}

fn standalone_simple_ngram_min(config: &SpeculativeDecodeConfig) -> usize {
    config
        .ngram
        .as_ref()
        .filter(|ngram| ngram.kind == NgramProposerKind::Simple)
        .map_or(0, |ngram| ngram.min_ngram)
}

fn standalone_simple_ngram_max(config: &SpeculativeDecodeConfig) -> usize {
    config
        .ngram
        .as_ref()
        .filter(|ngram| ngram.kind == NgramProposerKind::Simple)
        .map_or(0, |ngram| ngram.max_proposal_tokens.min(ngram.max_ngram))
}

#[cfg(test)]
mod standalone_speculative_config_tests {
    use super::*;

    #[test]
    fn standalone_speculative_config_rejects_invalid_composite_plan() {
        let config = SpeculativeDecodeConfig {
            extension: Some(NgramExtensionConfig {
                initial_tokens: 2,
                max_tokens: 4,
                tail_backoff_proposals: 1,
            }),
            ..SpeculativeDecodeConfig::default()
        };

        let error = config.validate().expect_err("extension requires proposers");

        assert!(
            error
                .to_string()
                .contains("requires both native MTP and an N-gram proposer")
        );
    }

    #[test]
    fn standalone_speculative_config_round_trips_cache_composite() {
        let config = SpeculativeDecodeConfig {
            requested_strategy: "mtp-cache".to_string(),
            effective_strategy: "native-mtp-cache".to_string(),
            native_mtp: NativeMtpProposalConfig {
                enabled: true,
                max_draft_tokens: 2,
                ..SpeculativeDecodeConfig::default().native_mtp
            },
            ngram: Some(NgramProposalConfig {
                kind: NgramProposerKind::Cache,
                min_ngram: 2,
                max_ngram: 4,
                max_proposal_tokens: 6,
            }),
            extension: Some(NgramExtensionConfig {
                initial_tokens: 2,
                max_tokens: 6,
                tail_backoff_proposals: 2,
            }),
            ..SpeculativeDecodeConfig::default()
        };

        let json = serde_json::to_string(&config).expect("serialize plan");
        let decoded: SpeculativeDecodeConfig = serde_json::from_str(&json).expect("parse plan");

        assert_eq!(decoded, config);
        decoded.validate().expect("valid composite plan");
    }

    #[test]
    fn standalone_speculative_config_rejects_cache_windows_above_llama_limit() {
        let config = SpeculativeDecodeConfig {
            ngram: Some(NgramProposalConfig {
                kind: NgramProposerKind::Cache,
                min_ngram: 2,
                max_ngram: skippy_runtime::NGRAM_CACHE_MAX_NGRAM + 1,
                max_proposal_tokens: 6,
            }),
            ..SpeculativeDecodeConfig::default()
        };

        let error = config.validate().expect_err("cache max must be bounded");

        assert!(
            error
                .to_string()
                .contains("must not exceed llama.cpp limit 4")
        );
    }
}

#[derive(Clone)]
pub struct EmbeddedOpenAiArgs {
    pub bind_addr: SocketAddr,
    pub config: StageConfig,
    pub runtime: Arc<Mutex<RuntimeState>>,
    pub model_id: Option<String>,
    pub default_max_tokens: u32,
    pub request_defaults: EmbeddedOpenAiRequestDefaults,
    pub generation_concurrency: usize,
    pub prefill_chunk_size: usize,
    pub prefill_chunk_policy: String,
    pub prefill_chunk_schedule: Option<String>,
    pub prefill_adaptive_start: usize,
    pub prefill_adaptive_step: usize,
    pub prefill_adaptive_max: usize,
    pub draft_model_path: Option<PathBuf>,
    pub speculative_window: usize,
    pub adaptive_speculative_window: bool,
    pub draft_n_gpu_layers: Option<i32>,
    pub speculative: SpeculativeDecodeConfig,
    pub ngram_min: usize,
    pub ngram_max: usize,
    pub native_mtp_enabled: bool,
    pub native_mtp_draft_model_path: Option<PathBuf>,
    pub native_mtp_max_tokens: usize,
    pub native_mtp_min_tokens: usize,
    pub activation_width: i32,
    pub wire_dtype: WireActivationDType,
    pub reply_credit_limit: Option<usize>,
    pub downstream_connect_timeout_secs: u64,
    pub downstream_wire_condition: WireCondition,
    pub prediction_returns: Option<Arc<PredictionReturnHub>>,
    pub telemetry: Telemetry,
    pub hook_policy: Option<Arc<dyn OpenAiHookPolicy>>,
    pub openai_guardrails: Option<OpenAiGuardrailsConfig>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OpenAiGuardrailsTarget {
    Skippy,
}

impl OpenAiGuardrailsTarget {
    const fn as_status_label(self) -> &'static str {
        match self {
            Self::Skippy => "skippy",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct OpenAiGuardrailsConfig {
    pub target: OpenAiGuardrailsTarget,
    pub policy: GuardrailPolicyHandle,
    pub compaction: Option<CompactionConfig>,
}

impl OpenAiGuardrailsConfig {
    pub fn disabled_for_skippy() -> Self {
        Self {
            target: OpenAiGuardrailsTarget::Skippy,
            policy: GuardrailPolicyHandle::default(),
            compaction: None,
        }
    }

    pub fn compatibility_for_skippy() -> Self {
        Self {
            target: OpenAiGuardrailsTarget::Skippy,
            policy: GuardrailPolicy {
                mode: GuardrailMode::MetricsOnly,
                apply_to_all_models: true,
                retry_exhaustion_mode: RetryExhaustionMode::PassLastText,
                ..GuardrailPolicy::default()
            }
            .into(),
            compaction: None,
        }
    }

    pub fn for_standalone_mode(mode: OpenAiGuardrailsCliMode) -> Self {
        match mode {
            OpenAiGuardrailsCliMode::Disabled => Self::disabled_for_skippy(),
            OpenAiGuardrailsCliMode::Metrics => Self::compatibility_for_skippy(),
            OpenAiGuardrailsCliMode::Enforce => Self {
                target: OpenAiGuardrailsTarget::Skippy,
                policy: GuardrailPolicy {
                    mode: GuardrailMode::Enforce,
                    apply_to_all_models: true,
                    ..GuardrailPolicy::default()
                }
                .into(),
                compaction: None,
            },
        }
    }

    pub fn status(&self) -> OpenAiGuardrailsStatus {
        let policy = self.policy.snapshot();
        OpenAiGuardrailsStatus {
            mode: guardrail_mode_label(policy.mode),
            target: self.target.as_status_label(),
            streaming: streaming_mode_label(policy.streaming_mode),
            retry_exhaustion: retry_exhaustion_label(&policy),
            small_model_policy: small_model_policy_label(&policy),
            small_param_threshold_b: policy.small_param_threshold_b,
            max_tool_retries: policy.max_tool_retries,
            max_structured_retries: policy.max_structured_retries,
        }
    }

    fn should_wrap_guardrail_backend(&self) -> bool {
        matches!(self.target, OpenAiGuardrailsTarget::Skippy)
    }

    #[cfg(test)]
    fn wrap_backend(&self, backend: Arc<dyn OpenAiBackend>) -> Arc<dyn OpenAiBackend> {
        self.wrap_backend_with_context_limit(backend, None)
    }

    fn wrap_backend_with_context_limit(
        &self,
        backend: Arc<dyn OpenAiBackend>,
        context_limit_tokens: Option<usize>,
    ) -> Arc<dyn OpenAiBackend> {
        let backend = self.wrap_compacting_backend(backend, context_limit_tokens);
        if self.should_wrap_guardrail_backend() {
            Arc::new(GuardedOpenAiBackend::with_policy_handle(
                backend,
                self.policy.clone(),
            ))
        } else {
            backend
        }
    }

    fn wrap_compacting_backend(
        &self,
        backend: Arc<dyn OpenAiBackend>,
        context_limit_tokens: Option<usize>,
    ) -> Arc<dyn OpenAiBackend> {
        let Some(mut compaction) = self.compaction else {
            return backend;
        };
        if compaction.context_limit_tokens.is_none() {
            compaction.context_limit_tokens = context_limit_tokens;
        }
        Arc::new(CompactingOpenAiBackend::new(backend, compaction))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct OpenAiGuardrailsStatus {
    pub mode: &'static str,
    pub target: &'static str,
    pub streaming: &'static str,
    pub retry_exhaustion: &'static str,
    pub small_model_policy: &'static str,
    pub small_param_threshold_b: f32,
    pub max_tool_retries: u8,
    pub max_structured_retries: u8,
}

fn guardrail_mode_label(mode: GuardrailMode) -> &'static str {
    match mode {
        GuardrailMode::Disabled => "disabled",
        GuardrailMode::MetricsOnly => "metrics",
        GuardrailMode::Enforce => "enforce",
    }
}

fn streaming_mode_label(mode: StreamingGuardrailMode) -> &'static str {
    match mode {
        StreamingGuardrailMode::PassThrough => "pass_through",
    }
}

fn retry_exhaustion_label(policy: &GuardrailPolicy) -> &'static str {
    match format!("{:?}", policy.retry_exhaustion_mode).as_str() {
        "Error" => "error",
        "PassLastText" => "pass_last_text",
        _ => "unknown",
    }
}

fn small_model_policy_label(policy: &GuardrailPolicy) -> &'static str {
    if policy.apply_to_all_models {
        "all"
    } else {
        "small_models_only"
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct EmbeddedOpenAiRequestDefaults {
    pub stop: Option<Vec<String>>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub seed: Option<u64>,
    pub logit_bias: Option<BTreeMap<String, Value>>,
    pub top_k: Option<i32>,
    pub min_p: Option<f32>,
    pub repeat_penalty: Option<f32>,
    pub repeat_last_n: Option<i32>,
    pub reasoning_format: Option<EmbeddedReasoningFormat>,
    pub reasoning_enabled: Option<EmbeddedReasoningEnabled>,
    pub reasoning_budget: Option<EmbeddedReasoningBudget>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbeddedReasoningFormat {
    Auto,
    None,
    Deepseek,
    DeepseekLegacy,
    Hidden,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbeddedReasoningEnabled {
    Auto,
    Disabled,
    Enabled,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbeddedReasoningBudget {
    Auto,
    Tokens(u32),
    Effort(ReasoningEffort),
}

pub async fn serve_embedded_openai(args: EmbeddedOpenAiArgs) -> Result<()> {
    serve_embedded_openai_with_shutdown(args, std::future::pending::<()>()).await
}

pub async fn serve_embedded_openai_with_shutdown(
    args: EmbeddedOpenAiArgs,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<()> {
    let bind_addr = args.bind_addr;
    let binding = embedded_openai_router(args)?;

    println!(
        "skippy-server listening: openai={} model_id={} backend=embedded-stage0 generation_concurrency={}",
        bind_addr, binding.model_id, binding.generation_concurrency,
    );

    let listener = TcpListener::bind(bind_addr).await?;
    axum::serve(listener, binding.router)
        .with_graceful_shutdown(shutdown)
        .await?;
    Ok(())
}

pub struct EmbeddedOpenAiRouter {
    pub router: Router,
    pub model_id: String,
    pub generation_concurrency: usize,
}

#[derive(Clone)]
pub struct EmbeddedOpenAiBackend {
    pub backend: Arc<dyn OpenAiBackend>,
    pub model_id: String,
    pub generation_concurrency: usize,
    pub openai_guardrails: Option<OpenAiGuardrailsStatus>,
}

pub fn embedded_openai_router(args: EmbeddedOpenAiArgs) -> Result<EmbeddedOpenAiRouter> {
    let telemetry = args.telemetry.clone();
    let binding = embedded_openai_backend(args)?;
    let router = instrumented_openai_router(binding.backend.clone(), telemetry);

    Ok(EmbeddedOpenAiRouter {
        router,
        model_id: binding.model_id,
        generation_concurrency: binding.generation_concurrency,
    })
}

pub fn embedded_openai_backend(args: EmbeddedOpenAiArgs) -> Result<EmbeddedOpenAiBackend> {
    if args.prefill_chunk_size == 0 {
        bail!("--openai-prefill-chunk-size must be greater than zero");
    }
    if args.generation_concurrency == 0 {
        bail!("--openai-generation-concurrency must be greater than zero");
    }
    ensure_generation_concurrency_fits_lanes(
        args.generation_concurrency,
        args.config.lane_count,
        "--openai-generation-concurrency",
    )?;
    if args.draft_model_path.is_some() && args.speculative_window == 0 {
        bail!("--openai-speculative-window must be greater than zero when a draft model is set");
    }
    if args.native_mtp_draft_model_path.is_some() && !args.native_mtp_enabled {
        bail!("native MTP must be enabled when an MTP draft model is set");
    }
    if args.ngram_min > args.ngram_max && args.ngram_max > 0 {
        bail!("--openai-ngram-min must be less than or equal to --openai-ngram-max");
    }
    if args.config.stage_index != 0 || args.config.layer_start != 0 {
        bail!("embedded OpenAI serving is only supported on stage 0");
    }
    attach_native_mtp_draft_model(
        args.native_mtp_draft_model_path.as_deref(),
        &args.runtime,
        &args.config,
        args.draft_n_gpu_layers,
    )?;
    let draft = open_draft_runner(
        args.draft_model_path.as_deref(),
        &args.config,
        args.draft_n_gpu_layers,
        args.speculative_window,
    )?;
    let model_id = ModelId::new(
        args.model_id
            .unwrap_or_else(|| args.config.model_id.clone()),
    )
    .map_err(|error| anyhow!("invalid OpenAI model id: {error}"))?
    .into_string();
    let lane_pool = PersistentStageLanePool::new(
        &args.config,
        args.generation_concurrency,
        args.downstream_connect_timeout_secs,
        args.telemetry.clone(),
    )
    .context("create embedded OpenAI persistent downstream lanes")?;
    let prefill_reply_credit_limit = args.reply_credit_limit.unwrap_or(3);
    let mode = OpenAiBackendMode::EmbeddedStageZero {
        config: args.config.clone(),
        wire_dtype: args.wire_dtype,
        prefill_chunk_policy: PrefillChunkPolicy::parse(PrefillChunkPolicyArgs {
            policy: &args.prefill_chunk_policy,
            schedule: args.prefill_chunk_schedule.as_deref(),
            fixed_chunk_size: args.prefill_chunk_size,
            adaptive_start: args.prefill_adaptive_start,
            adaptive_step: args.prefill_adaptive_step,
            adaptive_max: args.prefill_adaptive_max,
            schedule_arg: "--openai-prefill-chunk-schedule",
            policy_arg: "--openai-prefill-chunk-policy",
        })?,
        activation_width: args.activation_width,
        downstream_wire_condition: args.downstream_wire_condition,
        prefill_reply_credit_limit,
        lane_pool,
        prediction_returns: args.prediction_returns.clone(),
    };
    args.telemetry
        .emit("stage.openai_server_start", lifecycle_attrs(&args.config));
    prewarm_generation_sessions(
        &args.runtime,
        args.generation_concurrency,
        &args.telemetry,
        &args.config,
        "stage.openai_runtime_prewarm",
    )
    .context("prewarm embedded OpenAI runtime sessions")?;
    let kv = KvStageIntegration::from_config(&args.config)?.map(Arc::new);
    let ctx_size = usize::try_from(args.config.ctx_size).unwrap_or(usize::MAX);
    let decode_batcher = DecodeBatcher::new(args.runtime.clone(), args.generation_concurrency);
    let decode_frame_batcher =
        DecodeFrameBatcher::new(args.runtime.clone(), args.generation_concurrency);
    let backend: Arc<dyn OpenAiBackend> = Arc::new(StageOpenAiBackend {
        runtime: args.runtime,
        config: args.config.clone(),
        telemetry: args.telemetry.clone(),
        model_id: model_id.clone(),
        default_max_tokens: args.default_max_tokens,
        request_defaults: args.request_defaults,
        ctx_size,
        mode,
        draft,
        speculative_window: args.speculative_window,
        adaptive_speculative_window: args.adaptive_speculative_window,
        ngram_min: args.ngram_min,
        ngram_max: args.ngram_max,
        speculative: args.speculative,
        generation_limit: Arc::new(Semaphore::new(args.generation_concurrency)),
        generation_queue_depth: Arc::new(AtomicUsize::new(0)),
        generation_queue_limit: args.generation_concurrency,
        generation_token_budget: Arc::new(GenerationTokenBudget::new(ctx_size)),
        hook_policy: args.hook_policy,
        kv,
        decode_batcher,
        decode_frame_batcher,
    });
    let openai_guardrails = args
        .openai_guardrails
        .as_ref()
        .map(OpenAiGuardrailsConfig::status);
    let backend = args
        .openai_guardrails
        .as_ref()
        .map_or(backend.clone(), |guardrails| {
            guardrails.wrap_backend_with_context_limit(backend, Some(ctx_size))
        });

    Ok(EmbeddedOpenAiBackend {
        backend,
        model_id,
        generation_concurrency: args.generation_concurrency,
        openai_guardrails,
    })
}

#[derive(Clone)]
struct StageOpenAiBackend {
    runtime: Arc<Mutex<RuntimeState>>,
    config: StageConfig,
    telemetry: Telemetry,
    model_id: String,
    default_max_tokens: u32,
    request_defaults: EmbeddedOpenAiRequestDefaults,
    ctx_size: usize,
    mode: OpenAiBackendMode,
    draft: Option<Arc<Mutex<DraftRunner>>>,
    speculative_window: usize,
    adaptive_speculative_window: bool,
    ngram_min: usize,
    ngram_max: usize,
    speculative: SpeculativeDecodeConfig,
    generation_limit: Arc<Semaphore>,
    generation_queue_depth: Arc<AtomicUsize>,
    generation_queue_limit: usize,
    generation_token_budget: Arc<GenerationTokenBudget>,
    hook_policy: Option<Arc<dyn OpenAiHookPolicy>>,
    kv: Option<Arc<KvStageIntegration>>,
    decode_batcher: DecodeBatcher,
    decode_frame_batcher: DecodeFrameBatcher,
}

struct GenerationQueueReservation {
    depth: Arc<AtomicUsize>,
}

impl Drop for GenerationQueueReservation {
    fn drop(&mut self) {
        self.depth.fetch_sub(1, Ordering::AcqRel);
    }
}

async fn acquire_generation_permit_with_queue(
    generation_limit: Arc<Semaphore>,
    generation_queue_depth: Arc<AtomicUsize>,
    generation_queue_limit: usize,
    admission_timeout: Duration,
) -> OpenAiResult<OwnedSemaphorePermit> {
    match generation_limit.clone().try_acquire_owned() {
        Ok(permit) => return Ok(permit),
        Err(TryAcquireError::Closed) => return Err(generation_lanes_busy_error()),
        Err(TryAcquireError::NoPermits) => {}
    }

    let _queue_reservation =
        reserve_generation_queue(generation_queue_depth, generation_queue_limit)
            .ok_or_else(generation_queue_full_error)?;
    tokio::time::timeout(admission_timeout, generation_limit.acquire_owned())
        .await
        .map_err(|_| generation_queue_timeout_error(admission_timeout))?
        .map_err(|_| generation_lanes_busy_error())
}

fn reserve_generation_queue(
    generation_queue_depth: Arc<AtomicUsize>,
    generation_queue_limit: usize,
) -> Option<GenerationQueueReservation> {
    let mut current = generation_queue_depth.load(Ordering::Acquire);
    loop {
        if current >= generation_queue_limit {
            return None;
        }
        match generation_queue_depth.compare_exchange_weak(
            current,
            current + 1,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => {
                return Some(GenerationQueueReservation {
                    depth: generation_queue_depth,
                });
            }
            Err(next) => current = next,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GenerationTokenLimit {
    /// Client sent a concrete `max_tokens`. Must fit in the context
    /// window; otherwise return a context_length_exceeded error so the
    /// client knows their request couldn't be honored as-asked.
    Explicit(u32),
    /// Caller didn't send `max_tokens`, but the server has a configured
    /// default cap. Clamp down to whatever fits in the remaining
    /// context budget rather than rejecting — the client didn't ask
    /// for the specific number, the server picked it.
    Default(u32),
    /// Caller didn't send `max_tokens` and the server is configured
    /// with [`CONTEXT_BUDGET_MAX_TOKENS`] (opt-in unbounded). Use the
    /// entire remaining context window.
    ContextBudget,
}

impl GenerationTokenLimit {
    fn from_request(requested: Option<u32>, default_max_tokens: u32) -> Self {
        match requested {
            Some(max_tokens) => Self::Explicit(max_tokens),
            None if default_max_tokens == CONTEXT_BUDGET_MAX_TOKENS => Self::ContextBudget,
            None => Self::Default(default_max_tokens),
        }
    }

    fn resolve(self, prompt_token_count: usize, ctx_size: usize) -> OpenAiResult<u32> {
        match self {
            Self::Explicit(max_tokens) => {
                ensure_context_capacity(prompt_token_count, max_tokens, ctx_size)?;
                Ok(max_tokens)
            }
            Self::Default(default_max_tokens) => {
                // Server-picked default. Always clamp to the remaining
                // context budget. If the prompt already exceeds the
                // window, surface that as a real error — but never
                // reject just because our default wouldn't fit.
                let remaining = context_budget_completion_tokens(prompt_token_count, ctx_size)?;
                Ok(remaining.min(default_max_tokens))
            }
            Self::ContextBudget => context_budget_completion_tokens(prompt_token_count, ctx_size),
        }
    }
}

fn instrumented_openai_router(backend: Arc<dyn OpenAiBackend>, telemetry: Telemetry) -> Router {
    openai_frontend::router_for(backend).layer(middleware::from_fn_with_state(
        telemetry,
        openai_http_telemetry,
    ))
}

fn prewarm_generation_sessions(
    runtime: &Arc<Mutex<RuntimeState>>,
    generation_concurrency: usize,
    telemetry: &Telemetry,
    config: &StageConfig,
    event_name: &'static str,
) -> Result<()> {
    let timer = PhaseTimer::start();
    let sessions = runtime
        .lock()
        .map_err(|_| anyhow!("runtime lock poisoned"))?
        .prewarm_idle_sessions(generation_concurrency)?;
    let mut attrs = lifecycle_attrs(config);
    attrs.insert(
        "llama_stage.generation_concurrency".to_string(),
        json!(generation_concurrency),
    );
    attrs.insert(
        "llama_stage.lane_count".to_string(),
        json!(sessions.lane_count),
    );
    attrs.insert(
        "llama_stage.runtime_sessions_active".to_string(),
        json!(sessions.active_sessions),
    );
    attrs.insert(
        "llama_stage.runtime_sessions_idle".to_string(),
        json!(sessions.idle_sessions),
    );
    attrs.insert(
        "llama_stage.elapsed_ms".to_string(),
        json!(timer.elapsed_ms()),
    );
    telemetry.emit_span(
        event_name,
        attrs,
        timer.start_unix_nanos,
        now_unix_nanos() as u64,
    );
    Ok(())
}

fn ensure_generation_concurrency_fits_lanes(
    generation_concurrency: usize,
    lane_count: u32,
    flag_name: &str,
) -> Result<()> {
    let lane_count = usize::try_from(lane_count).unwrap_or(usize::MAX);
    if generation_concurrency > lane_count {
        bail!(
            "{flag_name} ({generation_concurrency}) cannot exceed configured lane_count ({lane_count})"
        );
    }
    Ok(())
}

fn generation_lanes_busy_error() -> OpenAiError {
    OpenAiError::from_kind(
        StatusCode::TOO_MANY_REQUESTS,
        OpenAiErrorKind::RateLimit,
        "all execution lanes are busy",
    )
    .with_retry_after_secs(GENERATION_RETRY_AFTER_SECS)
}

fn generation_queue_full_error() -> OpenAiError {
    OpenAiError::from_kind(
        StatusCode::TOO_MANY_REQUESTS,
        OpenAiErrorKind::RateLimit,
        "generation queue is full; retry later",
    )
    .with_retry_after_secs(GENERATION_RETRY_AFTER_SECS)
}

fn generation_queue_timeout_error(timeout: Duration) -> OpenAiError {
    OpenAiError::from_kind(
        StatusCode::TOO_MANY_REQUESTS,
        OpenAiErrorKind::RateLimit,
        format!(
            "timed out waiting for an execution lane after {} seconds",
            timeout.as_secs()
        ),
    )
    .with_retry_after_secs(GENERATION_RETRY_AFTER_SECS)
}

async fn openai_http_telemetry(
    State(telemetry): State<Telemetry>,
    request: Request<Body>,
    next: Next,
) -> Response {
    let timer = PhaseTimer::start();
    let method = request.method().to_string();
    let path = request.uri().path().to_string();
    let response = next.run(request).await;
    let status = response.status().as_u16();
    let mut attrs = BTreeMap::from([
        ("llama_stage.http_method".to_string(), json!(method)),
        ("llama_stage.http_path".to_string(), json!(path)),
        ("llama_stage.http_status".to_string(), json!(status)),
    ]);
    attrs.insert(
        "llama_stage.elapsed_ms".to_string(),
        json!(timer.elapsed_ms()),
    );
    telemetry.emit_span(
        "stage.openai_http_request",
        attrs,
        timer.start_unix_nanos,
        now_unix_nanos() as u64,
    );
    response
}

#[derive(Clone)]
#[allow(clippy::large_enum_variant)]
enum OpenAiBackendMode {
    LocalRuntime,
    EmbeddedStageZero {
        config: StageConfig,
        wire_dtype: WireActivationDType,
        prefill_chunk_policy: PrefillChunkPolicy,
        activation_width: i32,
        downstream_wire_condition: WireCondition,
        prefill_reply_credit_limit: usize,
        lane_pool: Option<Arc<PersistentStageLanePool>>,
        prediction_returns: Option<Arc<PredictionReturnHub>>,
    },
}

struct PersistentStageLanePool {
    config: StageConfig,
    timeout_secs: u64,
    telemetry: Telemetry,
    lanes: Mutex<Vec<PersistentStageLane>>,
    prefill_transport: Mutex<PrefillTransportEstimate>,
    next_lane_id: AtomicU64,
    capacity: usize,
}

struct PersistentStageLane {
    id: u64,
    stream: TcpStream,
}

#[derive(Clone, Copy, Debug, Default)]
struct PrefillTransportEstimate {
    write_ms: f64,
    wait_ms: f64,
    write_to_compute: f64,
    wait_to_compute: f64,
    stage_index: i64,
    activation_bytes: i64,
    observations: u64,
}

/// Read timeout for the lane ready handshake. `recv_ready` is a blocking
/// `read_exact`; without this a stalled/half-dead downstream connection hangs
/// lane creation forever (observed as a lane stuck in "waiting ready" that
/// never completes and never errors). A single bounded deadline turns the stall
/// into an error instead. `connect_binary_downstream` already bounds the TCP
/// connect, so this handshake read is the only remaining unbounded wait — no
/// outer retry, which would only multiply the worst-case stall (see PR #1011
/// review). Applied only to the handshake read, then cleared so the persistent
/// lane's generation reads stay blocking.
const LANE_READY_READ_TIMEOUT: Duration = Duration::from_secs(20);

/// Steady-state (post-warmup) lane reconnect deadline. A live split peer at
/// LAN/WAN RTT answers the ready handshake in milliseconds; only a dead or
/// wedged downstream stage waits this long. Keeping the steady-state deadline
/// short turns "a new request routed to a dead stage" into a fast error (~3s)
/// instead of the ~20s warmup deadline. The long `LANE_READY_READ_TIMEOUT` is
/// only needed during pool warmup, when the downstream stage may still be
/// loading its model; a mid-life reconnect to an already-serving mesh has no
/// such excuse for a multi-second silence.
const LANE_STEADY_CONNECT_TIMEOUT: Duration = Duration::from_secs(3);

impl PersistentStageLanePool {
    const PREFILL_TRANSPORT_EWMA_ALPHA: f64 = 0.25;

    fn new(
        config: &StageConfig,
        capacity: usize,
        timeout_secs: u64,
        telemetry: Telemetry,
    ) -> Result<Option<Arc<Self>>> {
        if config.downstream.is_none() {
            return Ok(None);
        }
        let pool = Arc::new(Self {
            config: config.clone(),
            timeout_secs,
            telemetry,
            lanes: Mutex::new(Vec::with_capacity(capacity)),
            prefill_transport: Mutex::new(PrefillTransportEstimate::default()),
            next_lane_id: AtomicU64::new(0),
            capacity,
        });
        let timer = PhaseTimer::start();
        for _ in 0..capacity {
            // Warmup: the downstream stage may still be loading its model, so
            // allow the full ready deadline.
            let lane = pool.connect_lane(LANE_READY_READ_TIMEOUT)?;
            pool.return_lane(lane);
        }
        let mut attrs = lifecycle_attrs(config);
        attrs.insert(
            "llama_stage.openai_downstream_pool_capacity".to_string(),
            json!(capacity),
        );
        attrs.insert(
            "llama_stage.elapsed_ms".to_string(),
            json!(timer.elapsed_ms()),
        );
        pool.telemetry.emit_span(
            "stage.openai_downstream_pool_ready",
            attrs,
            timer.start_unix_nanos,
            now_unix_nanos() as u64,
        );
        Ok(Some(pool))
    }

    fn checkout(&self, ids: &OpenAiGenerationIds) -> OpenAiResult<PersistentStageLane> {
        let timer = PhaseTimer::start();
        let lane = {
            let mut lanes = self
                .lanes
                .lock()
                .map_err(|_| OpenAiError::backend("persistent lane pool lock poisoned"))?;
            lanes.pop()
        };
        // A pooled lane may be stale: if the downstream stage died while the
        // lane was checked in, the cached TCP stream is dead but reusing it
        // would block forever on the next generation read (the handshake
        // read-timeout is deliberately cleared so long generations don't
        // truncate). Probe liveness before reuse and drop a dead lane so we
        // reconnect with the short steady-state deadline instead of hanging.
        let live_pooled = lane.filter(|lane| lane_stream_is_live(&lane.stream));
        let lane = match live_pooled {
            Some(lane) => lane,
            // Steady-state reconnect: the mesh is already serving, so a healthy
            // downstream answers fast. Use the short deadline so a request
            // routed to a dead stage fails quickly instead of stalling ~20s.
            None => self
                .connect_lane(LANE_STEADY_CONNECT_TIMEOUT)
                .map_err(openai_backend_error)?,
        };
        let mut attrs = BTreeMap::from([
            (
                "llama_stage.openai_downstream_persistent".to_string(),
                json!(true),
            ),
            (
                "llama_stage.openai_downstream_lane_id".to_string(),
                json!(lane.id),
            ),
            (
                "llama_stage.openai_downstream_pool_capacity".to_string(),
                json!(self.capacity),
            ),
            (
                "llama_stage.request_id".to_string(),
                json!(ids.request_id_string()),
            ),
            (
                "llama_stage.session_id".to_string(),
                json!(ids.session_id_string()),
            ),
        ]);
        attrs.insert(
            "llama_stage.elapsed_ms".to_string(),
            json!(timer.elapsed_ms()),
        );
        self.telemetry.emit_span(
            "stage.openai_downstream_connect",
            attrs,
            timer.start_unix_nanos,
            now_unix_nanos() as u64,
        );
        Ok(lane)
    }

    fn prefill_transport_seed(&self) -> Option<PrefillChunkObservation> {
        let estimate = *self.prefill_transport.lock().ok()?;
        if estimate.observations == 0 {
            return None;
        }
        Some(PrefillChunkObservation {
            compute_ms: 1.0,
            forward_write_ms: estimate.write_to_compute,
            downstream_wait_ms: estimate.wait_to_compute,
        })
    }

    fn observe_prefill_transport(
        &self,
        stats: &StageReplyStats,
        stage0_compute_ms: f64,
        prefill_chunks: usize,
    ) {
        if stats.prefill_edge_observation_count == 0 || prefill_chunks == 0 {
            return;
        }
        let compute_ms = (stage0_compute_ms / prefill_chunks as f64).max(0.001);
        let write_ms = us_to_ms(stats.prefill_edge_write_us_max);
        let wait_ms = us_to_ms(stats.prefill_edge_wait_us_max);
        let sample = PrefillTransportEstimate {
            write_ms,
            wait_ms,
            write_to_compute: write_ms / compute_ms,
            wait_to_compute: wait_ms / compute_ms,
            stage_index: stats.prefill_edge_stage_index,
            activation_bytes: stats.prefill_edge_activation_bytes_max,
            observations: u64::try_from(stats.prefill_edge_observation_count).unwrap_or(0),
        };
        let mut estimate = match self.prefill_transport.lock() {
            Ok(estimate) => estimate,
            Err(_) => return,
        };
        if estimate.observations == 0 {
            *estimate = sample;
        } else {
            estimate.write_ms = ewma(estimate.write_ms, sample.write_ms);
            estimate.wait_ms = ewma(estimate.wait_ms, sample.wait_ms);
            estimate.write_to_compute = ewma(estimate.write_to_compute, sample.write_to_compute);
            estimate.wait_to_compute = ewma(estimate.wait_to_compute, sample.wait_to_compute);
            estimate.stage_index = sample.stage_index;
            estimate.activation_bytes = sample.activation_bytes;
            estimate.observations = estimate.observations.saturating_add(sample.observations);
        }
    }

    fn return_lane(&self, lane: PersistentStageLane) {
        match self.lanes.lock() {
            Ok(mut lanes) => lanes.push(lane),
            Err(_) => {
                let mut attrs = lifecycle_attrs(&self.config);
                attrs.insert(
                    "llama_stage.error".to_string(),
                    json!("persistent lane pool lock poisoned"),
                );
                self.telemetry
                    .emit("stage.openai_downstream_lane_return_failed", attrs);
            }
        }
    }

    fn replace_lane(&self, retired_lane_id: u64) {
        let timer = PhaseTimer::start();
        let mut attrs = lifecycle_attrs(&self.config);
        attrs.insert(
            "llama_stage.openai_downstream_retired_lane_id".to_string(),
            json!(retired_lane_id),
        );
        // Replacing a retired lane on an already-serving mesh is a steady-state
        // reconnect: a healthy downstream answers fast, a dead one should fail
        // fast rather than stall.
        match self.connect_lane(LANE_STEADY_CONNECT_TIMEOUT) {
            Ok(lane) => {
                attrs.insert(
                    "llama_stage.openai_downstream_lane_id".to_string(),
                    json!(lane.id),
                );
                attrs.insert(
                    "llama_stage.elapsed_ms".to_string(),
                    json!(timer.elapsed_ms()),
                );
                self.return_lane(lane);
                self.telemetry.emit_span(
                    "stage.openai_downstream_lane_replaced",
                    attrs,
                    timer.start_unix_nanos,
                    now_unix_nanos() as u64,
                );
            }
            Err(error) => {
                attrs.insert("llama_stage.error".to_string(), json!(error.to_string()));
                attrs.insert(
                    "llama_stage.elapsed_ms".to_string(),
                    json!(timer.elapsed_ms()),
                );
                self.telemetry.emit_span(
                    "stage.openai_downstream_lane_replace_failed",
                    attrs,
                    timer.start_unix_nanos,
                    now_unix_nanos() as u64,
                );
            }
        }
    }

    fn connect_lane(&self, ready_timeout: Duration) -> Result<PersistentStageLane> {
        let lane_id = self.next_lane_id.fetch_add(1, Ordering::Relaxed);
        let timer = PhaseTimer::start();
        // Single bounded attempt. `connect_binary_downstream` already retries
        // the TCP connect internally (its own attempt budget), and the ready
        // handshake read below is bounded by `ready_timeout`. An extra
        // outer retry here only multiplies the worst-case stall — a dead peer
        // would burn (connect budget × outer attempts) before failing (see PR
        // #1011 review). Bring-up wants a single, predictable deadline.
        let stream = self.connect_lane_once(lane_id, ready_timeout).inspect_err(|error| {
            eprintln!(
                "openai downstream lane handshake failed: stage_id={} lane_id={lane_id}: {error:#}",
                self.config.stage_id,
            );
        })?;
        let mut attrs = lifecycle_attrs(&self.config);
        attrs.insert(
            "llama_stage.openai_downstream_lane_id".to_string(),
            json!(lane_id),
        );
        attrs.insert(
            "llama_stage.openai_downstream_pool_capacity".to_string(),
            json!(self.capacity),
        );
        attrs.insert(
            "llama_stage.elapsed_ms".to_string(),
            json!(timer.elapsed_ms()),
        );
        self.telemetry.emit_span(
            "stage.openai_downstream_persistent_connect",
            attrs,
            timer.start_unix_nanos,
            now_unix_nanos() as u64,
        );
        Ok(PersistentStageLane {
            id: lane_id,
            stream,
        })
    }

    /// Perform a single connect + ready-handshake attempt for one lane.
    ///
    /// Returns the connected, ready `TcpStream` on success. `ready_timeout`
    /// bounds the handshake read: use `LANE_READY_READ_TIMEOUT` during pool
    /// warmup (the downstream stage may still be loading) and the shorter
    /// `LANE_STEADY_CONNECT_TIMEOUT` for mid-life reconnects on an
    /// already-serving mesh, so a dead stage fails fast instead of stalling.
    fn connect_lane_once(&self, lane_id: u64, ready_timeout: Duration) -> Result<TcpStream> {
        let mut stream = connect_binary_downstream(&self.config, self.timeout_secs)?
            .ok_or_else(|| anyhow!("embedded stage0 has no downstream"))?;
        let local_addr = stream.local_addr().ok();
        let peer_addr = stream.peer_addr().ok();
        eprintln!(
            "openai downstream lane waiting ready: stage_id={} lane_id={lane_id} local={local_addr:?} peer={peer_addr:?}",
            self.config.stage_id
        );
        send_client_ready_hello_if_enabled(&mut stream)
            .context("send persistent downstream lane client ready hello")?;
        receive_persistent_lane_ready(&mut stream, ready_timeout)?;
        eprintln!(
            "openai downstream lane received ready: stage_id={} lane_id={lane_id} local={local_addr:?} peer={peer_addr:?}",
            self.config.stage_id
        );
        Ok(stream)
    }
}

/// Read the lane ready handshake with a bounded deadline.
///
/// `recv_ready` is a blocking `read_exact`; without a read timeout a stalled or
/// half-dead downstream connection hangs lane creation forever (observed as a
/// lane stuck in "waiting ready" that never completes and never errors). Set the
/// deadline, read, then clear it so the persistent lane's later generation reads
/// stay blocking. Both the set and the clear are propagated: if the set fails,
/// `recv_ready` would be unbounded (defeating the fix); if the clear fails, the
/// handshake timeout would leak into generation reads and truncate long outputs.
fn receive_persistent_lane_ready(stream: &mut TcpStream, timeout: Duration) -> Result<()> {
    stream
        .set_read_timeout(Some(timeout))
        .context("set persistent downstream lane ready timeout")?;
    let ready = recv_ready(&mut *stream).context("persistent downstream lane did not become ready");
    stream
        .set_read_timeout(None)
        .context("restore persistent downstream lane read timeout")?;
    ready
}

/// Cheap liveness probe for a pooled lane before reuse.
///
/// A pooled lane whose downstream stage died is a dead TCP stream; reusing it
/// would block the next generation read forever (the handshake read-timeout is
/// cleared for pooled lanes so long generations don't truncate). A nonblocking
/// peek distinguishes cases without consuming data:
///
/// - `Ok(0)` => peer sent EOF / closed => dead, discard.
/// - `Err(WouldBlock)` => connection open, no pending data => healthy, reuse.
/// - other `Err` (reset, etc.) => dead, discard.
///
/// Between requests a healthy lane has no unread bytes, so a nonzero peek would
/// be unexpected protocol data; treat that as unsafe-to-reuse and discard too.
/// The blocking mode is always restored so the caller's reads are unaffected.
fn lane_stream_is_live(stream: &TcpStream) -> bool {
    use std::io::ErrorKind;
    if stream.set_nonblocking(true).is_err() {
        return false;
    }
    let mut probe = [0u8; 1];
    let live = match stream.peek(&mut probe) {
        Ok(0) => false,
        Ok(_) => false,
        Err(ref e) if e.kind() == ErrorKind::WouldBlock => true,
        Err(_) => false,
    };
    // Restore blocking mode; if we can't, the lane is not safe to hand back.
    if stream.set_nonblocking(false).is_err() {
        return false;
    }
    live
}

fn ewma(old: f64, sample: f64) -> f64 {
    old.mul_add(
        1.0 - PersistentStageLanePool::PREFILL_TRANSPORT_EWMA_ALPHA,
        sample * PersistentStageLanePool::PREFILL_TRANSPORT_EWMA_ALPHA,
    )
}

#[derive(Clone)]
struct OpenAiGenerationIds {
    session_label: String,
    session_id: u64,
    request_id: u64,
    cache: OpenAiCacheHints,
}

impl OpenAiGenerationIds {
    fn new(cache: OpenAiCacheHints) -> Self {
        let sequence = OPENAI_GENERATION_COUNTER.fetch_add(1, Ordering::Relaxed);
        let session_label = format!("openai-session-{}-{sequence}", now_unix_millis());
        Self {
            session_id: stable_wire_id(&[session_label.as_bytes()]),
            request_id: stable_wire_id(&[session_label.as_bytes(), b"request"]),
            session_label,
            cache,
        }
    }

    fn session_id_string(&self) -> String {
        self.session_id.to_string()
    }

    fn request_id_string(&self) -> String {
        self.request_id.to_string()
    }
}

#[derive(Clone, Default)]
struct OpenAiCacheHints {
    prompt_cache_key: Option<String>,
    prompt_cache_retention: Option<String>,
}

impl OpenAiCacheHints {
    fn from_chat_request(request: &ChatCompletionRequest) -> Self {
        Self {
            prompt_cache_key: request
                .prompt_cache_key
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToString::to_string),
            prompt_cache_retention: request
                .prompt_cache_retention
                .map(prompt_cache_retention_label)
                .map(ToString::to_string),
        }
    }

    fn from_completion_request(request: &CompletionRequest) -> Self {
        Self {
            prompt_cache_key: request
                .prompt_cache_key
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToString::to_string),
            prompt_cache_retention: request
                .prompt_cache_retention
                .map(prompt_cache_retention_label)
                .map(ToString::to_string),
        }
    }

    fn namespace(&self) -> Option<String> {
        self.prompt_cache_key
            .as_ref()
            .map(|key| format!("openai:prompt_cache_key:{key}"))
    }
}

fn prompt_cache_retention_label(retention: openai_frontend::PromptCacheRetention) -> &'static str {
    match retention {
        openai_frontend::PromptCacheRetention::InMemory => "in_memory",
        openai_frontend::PromptCacheRetention::TwentyFourHours => "24h",
    }
}

struct GenerationCacheStats {
    status: &'static str,
    cached_prompt_tokens: u32,
    matched_prefix_tokens: u32,
    suffix_prefill_tokens: u32,
    hit_kind: Option<&'static str>,
    native_mtp_stats: NativeMtpStats,
    native_mtp_decode_telemetry: Option<NativeMtpDecodeTelemetry>,
    verify_window_pipeline_stats: Option<VerifyWindowPipelineStats>,
    speculative_stats: Option<OpenAiSpeculativeStats>,
    prompt_ms: f64,
    predicted_ms: f64,
}

impl Default for GenerationCacheStats {
    fn default() -> Self {
        Self {
            status: "disabled",
            cached_prompt_tokens: 0,
            matched_prefix_tokens: 0,
            suffix_prefill_tokens: 0,
            hit_kind: None,
            native_mtp_stats: NativeMtpStats::default(),
            native_mtp_decode_telemetry: None,
            verify_window_pipeline_stats: None,
            speculative_stats: None,
            prompt_ms: 0.0,
            predicted_ms: 0.0,
        }
    }
}

struct ChainPrefixRestore {
    restored_tokens: usize,
    stats: StageReplyStats,
}

struct PhaseTimer {
    start_unix_nanos: u64,
    start_instant: Instant,
}

impl PhaseTimer {
    fn start() -> Self {
        Self {
            start_unix_nanos: now_unix_nanos() as u64,
            start_instant: Instant::now(),
        }
    }

    fn elapsed_ms(&self) -> f64 {
        self.start_instant.elapsed().as_secs_f64() * 1000.0
    }
}

fn decode_token_phase(decode_step: u32) -> &'static str {
    match decode_step {
        0 => "cold",
        1..=7 => "warmup",
        _ => "steady",
    }
}

#[derive(Default)]
struct GenerationMetrics {
    detokenize_ms: f64,
    text_emit_ms: f64,
    eog_check_ms: f64,
}

struct DraftRunner {
    path: PathBuf,
    window: usize,
    _model: StageModel,
    session: StageSession,
}

impl DraftRunner {
    fn open(
        path: &Path,
        config: &StageConfig,
        n_gpu_layers: Option<i32>,
        window: usize,
    ) -> Result<Self> {
        if !path.is_file() {
            bail!("draft model does not exist: {}", path.display());
        }
        let layer_count = model_layer_count(path)?;
        let model = StageModel::open(
            path,
            &RuntimeConfig {
                stage_index: 0,
                layer_start: 0,
                layer_end: layer_count,
                ctx_size: config.ctx_size,
                lane_count: 1,
                n_batch: None,
                n_ubatch: None,
                n_threads: None,
                n_threads_batch: None,
                n_gpu_layers: n_gpu_layers.unwrap_or(config.n_gpu_layers),
                mmap: config.mmap,
                mlock: config.mlock,
                selected_backend_device: config
                    .selected_device
                    .as_ref()
                    .map(|device| device.backend_device.clone()),
                cache_type_k: skippy_runtime::GGML_TYPE_F16,
                cache_type_v: skippy_runtime::GGML_TYPE_F16,
                flash_attn_type: RuntimeFlashAttentionType::Auto,
                load_mode: RuntimeLoadMode::RuntimeSlice,
                projector_path: None,
                include_embeddings: true,
                include_output: true,
                filter_tensors_on_load: false,
            },
        )
        .with_context(|| format!("open draft model {}", path.display()))?;
        let session = model.create_session().context("create draft session")?;
        Ok(Self {
            path: path.to_path_buf(),
            window,
            _model: model,
            session,
        })
    }

    fn reset_to_context(&mut self, context_tokens: &[i32]) -> Result<()> {
        self.session.reset().context("reset draft session")?;
        if context_tokens.len() > 1 {
            self.session
                .prefill_chunk(&context_tokens[..context_tokens.len() - 1])
                .context("prefill draft context")?;
        }
        Ok(())
    }

    fn propose(&mut self, mut current: i32, max_tokens: usize) -> Result<Vec<i32>> {
        let mut tokens = Vec::with_capacity(max_tokens);
        for _ in 0..max_tokens {
            current = self
                .session
                .decode_step(current)
                .context("draft decode step")?;
            tokens.push(current);
        }
        Ok(tokens)
    }
}

fn open_draft_runner(
    path: Option<&Path>,
    config: &StageConfig,
    n_gpu_layers: Option<i32>,
    window: usize,
) -> Result<Option<Arc<Mutex<DraftRunner>>>> {
    let Some(path) = path else {
        return Ok(None);
    };
    Ok(Some(Arc::new(Mutex::new(DraftRunner::open(
        path,
        config,
        n_gpu_layers,
        window,
    )?))))
}

fn attach_native_mtp_draft_model(
    path: Option<&Path>,
    runtime: &Arc<Mutex<RuntimeState>>,
    config: &StageConfig,
    n_gpu_layers: Option<i32>,
) -> Result<()> {
    let Some(path) = path else {
        return Ok(());
    };
    if !path.is_file() {
        bail!("MTP draft model does not exist: {}", path.display());
    }
    let layer_count = model_layer_count(path)?;
    let mut runtime = runtime
        .lock()
        .map_err(|_| anyhow!("runtime lock poisoned"))?;
    runtime
        .model
        .attach_mtp_draft_model(
            path,
            &RuntimeConfig {
                stage_index: 0,
                layer_start: 0,
                layer_end: layer_count,
                ctx_size: config.ctx_size,
                lane_count: config.lane_count,
                n_batch: None,
                n_ubatch: None,
                n_threads: None,
                n_threads_batch: None,
                n_gpu_layers: n_gpu_layers.unwrap_or(config.n_gpu_layers),
                mmap: config.mmap,
                mlock: config.mlock,
                selected_backend_device: config
                    .selected_device
                    .as_ref()
                    .map(|device| device.backend_device.clone()),
                cache_type_k: skippy_runtime::GGML_TYPE_F16,
                cache_type_v: skippy_runtime::GGML_TYPE_F16,
                flash_attn_type: RuntimeFlashAttentionType::Auto,
                load_mode: RuntimeLoadMode::RuntimeSlice,
                projector_path: None,
                include_embeddings: true,
                include_output: true,
                filter_tensors_on_load: false,
            },
        )
        .with_context(|| format!("attach MTP draft model {}", path.display()))
}

fn model_layer_count(path: &Path) -> Result<u32> {
    let info =
        ModelInfo::open(path).with_context(|| format!("open model info {}", path.display()))?;
    let layer_count = info
        .tensors()?
        .into_iter()
        .filter_map(|tensor| tensor.layer_index)
        .max()
        .map(|index| index + 1)
        .ok_or_else(|| anyhow!("could not infer layer count for {}", path.display()))?;
    Ok(layer_count)
}

impl OpenAiBackendMode {
    fn label(&self) -> &'static str {
        match self {
            Self::LocalRuntime => "local-runtime",
            Self::EmbeddedStageZero { .. } => "embedded-stage0",
        }
    }

    fn reserves_local_kv_tokens(&self) -> bool {
        match self {
            Self::LocalRuntime | Self::EmbeddedStageZero { .. } => true,
        }
    }
}

fn ensure_requested_model(advertised_model_id: &str, requested: &str) -> OpenAiResult<()> {
    if requested == advertised_model_id
        || strip_default_revision(requested) == strip_default_revision(advertised_model_id)
    {
        Ok(())
    } else {
        Err(OpenAiError::model_not_found(requested))
    }
}

/// Strip `@main` so `org/repo@main:Q4` and `org/repo:Q4` compare equal.
///
/// Only removes `@main` when it sits at a revision boundary — followed by
/// `:` (quant separator) or end-of-string.  This avoids corrupting repo
/// names that happen to contain `@main` as a prefix of a longer segment
/// (e.g. `@mainland`).
fn strip_default_revision(id: &str) -> String {
    if let Some(pos) = id.find("@main") {
        let after = pos + "@main".len();
        if after == id.len() || id.as_bytes()[after] == b':' {
            let mut s = id[..pos].to_string();
            s.push_str(&id[after..]);
            return s;
        }
    }
    id.to_string()
}

fn hook_injected_text(outcome: &ChatHookOutcome) -> Option<String> {
    let text = outcome
        .actions
        .iter()
        .filter_map(|action| match action {
            ChatHookAction::InjectText { text } if !text.is_empty() => Some(text.as_str()),
            ChatHookAction::InjectText { .. }
            | ChatHookAction::ConsumeMedia { .. }
            | ChatHookAction::None => None,
        })
        .collect::<Vec<_>>()
        .join("");
    (!text.is_empty()).then_some(text)
}

fn mid_generation_window_should_fire(
    decoded_tokens: usize,
    last_hook_at: &Option<usize>,
    window: &GenerationSignalWindow,
) -> bool {
    const MIN_DECODED_TOKENS: usize = 12;
    const COOLDOWN_TOKENS: usize = 32;
    const REPETITION_TRIGGER_COUNT: u32 = 3;

    if decoded_tokens < MIN_DECODED_TOKENS || window.token_count == 0 {
        return false;
    }
    if last_hook_at.is_some_and(|last| decoded_tokens.saturating_sub(last) < COOLDOWN_TOKENS) {
        return false;
    }
    let sustained_entropy =
        window.high_entropy_count.saturating_mul(4) >= window.token_count.saturating_mul(3);
    sustained_entropy || window.repetition_count >= REPETITION_TRIGGER_COUNT
}

fn attrs_insert_prefill_chunk_policy(
    attrs: &mut BTreeMap<String, Value>,
    policy: &PrefillChunkPolicy,
    min_chunk_size: usize,
    max_chunk_size: usize,
) {
    attrs.insert(
        "llama_stage.prefill_chunk_size".to_string(),
        json!(policy.fixed_chunk_size()),
    );
    attrs.insert(
        "llama_stage.prefill_chunk_policy".to_string(),
        json!(policy.policy_label()),
    );
    if let Some(schedule) = policy.schedule() {
        attrs.insert(
            "llama_stage.prefill_chunk_schedule".to_string(),
            json!(schedule.label()),
        );
    }
    if let Some((start, step, max)) = policy.adaptive_params() {
        attrs.insert(
            "llama_stage.prefill_adaptive_start".to_string(),
            json!(start),
        );
        attrs.insert("llama_stage.prefill_adaptive_step".to_string(), json!(step));
        attrs.insert("llama_stage.prefill_adaptive_max".to_string(), json!(max));
    }
    if min_chunk_size != usize::MAX {
        attrs.insert(
            "llama_stage.prefill_min_chunk_size".to_string(),
            json!(min_chunk_size),
        );
        attrs.insert(
            "llama_stage.prefill_max_chunk_size".to_string(),
            json!(max_chunk_size),
        );
    }
}

#[derive(Debug, Clone)]
struct PreparedGenerationPrompt {
    text: String,
    media: Vec<MediaInput>,
    chat_parse_metadata: Option<String>,
}

impl PreparedGenerationPrompt {
    fn text(text: String) -> Self {
        Self {
            text,
            media: Vec::new(),
            chat_parse_metadata: None,
        }
    }

    fn has_media(&self) -> bool {
        !self.media.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ParsedToolCalls {
    content: Option<String>,
    tool_calls: Value,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ParsedChatMessage {
    content: Option<String>,
    reasoning_content: Option<String>,
    tool_calls: Option<Value>,
}

fn tool_calls_requested(request: &ChatCompletionRequest) -> bool {
    request.tools.as_ref().is_some_and(has_requested_tools)
        && !request
            .tool_choice
            .as_ref()
            .is_some_and(|choice| matches!(choice.as_str(), Some("none")))
}

fn chat_output_parser_required(
    request: &ChatCompletionRequest,
    template_options: &ChatTemplateOptions,
) -> bool {
    tool_calls_requested(request)
        || template_options
            .reasoning_format
            .is_some_and(ChatReasoningFormat::parses_reasoning)
}

fn template_exposes_reasoning(template_options: &ChatTemplateOptions) -> bool {
    template_options
        .reasoning_format
        .is_some_and(ChatReasoningFormat::exposes_reasoning)
}

fn apply_reasoning_visibility(
    parsed: Option<ParsedChatMessage>,
    template_options: &ChatTemplateOptions,
) -> Option<ParsedChatMessage> {
    let mut parsed = parsed?;
    if !template_exposes_reasoning(template_options) {
        parsed.reasoning_content = None;
    }
    Some(parsed)
}

fn chat_response_from_generated_text(
    model: String,
    output: &GeneratedText,
    parsed_message: Option<ParsedChatMessage>,
) -> ChatCompletionResponse {
    if let Some(parsed) = parsed_message {
        let finish_reason = if parsed.tool_calls.is_some() {
            FinishReason::ToolCalls
        } else {
            output.finish_reason
        };
        return ChatCompletionResponse {
            id: openai_frontend::completion_id("chatcmpl"),
            object: "chat.completion",
            created: openai_frontend::now_unix_secs(),
            model,
            choices: vec![openai_frontend::ChatCompletionChoice {
                index: 0,
                message: openai_frontend::AssistantMessage {
                    role: "assistant",
                    content: parsed.content,
                    reasoning_content: parsed.reasoning_content,
                    tool_calls: parsed.tool_calls,
                },
                logprobs: None,
                finish_reason: Some(finish_reason),
            }],
            usage: output.usage(),
            timings: output.timings(),
        };
    }

    ChatCompletionResponse::new_with_reason(
        model,
        output.text.clone(),
        output.usage(),
        output.finish_reason,
    )
    .with_timings(output.timings())
}

fn completion_response_from_generated_text(
    model: String,
    output: &GeneratedText,
) -> CompletionResponse {
    CompletionResponse::new_with_reason(
        model,
        output.text.clone(),
        output.usage(),
        output.finish_reason,
    )
    .with_timings(output.timings())
}

fn parsed_chat_message_from_json(
    message_json: &str,
    request: &ChatCompletionRequest,
) -> Option<ParsedChatMessage> {
    let value = serde_json::from_str::<Value>(message_json).ok()?;
    let tool_calls =
        parsed_tool_calls_from_message_value(&value, request).map(|parsed| parsed.tool_calls);
    Some(ParsedChatMessage {
        content: string_field(&value, "content"),
        reasoning_content: string_field(&value, "reasoning_content"),
        tool_calls,
    })
}

#[cfg(test)]
fn parsed_tool_calls_from_message_json(
    message_json: &str,
    request: &ChatCompletionRequest,
) -> Option<ParsedToolCalls> {
    let value = serde_json::from_str::<Value>(message_json).ok()?;
    parsed_tool_calls_from_message_value(&value, request)
}

fn parsed_tool_calls_from_message_value(
    value: &Value,
    request: &ChatCompletionRequest,
) -> Option<ParsedToolCalls> {
    if !tool_calls_requested(request) {
        return None;
    }
    let allowed_names = request_allowed_tool_names(request);
    let mut tool_calls = value
        .get("tool_calls")
        .and_then(Value::as_array)?
        .iter()
        .filter(|call| tool_call_allowed(call, &allowed_names))
        .cloned()
        .collect::<Vec<_>>();
    if request.parallel_tool_calls == Some(false) {
        tool_calls.truncate(1);
    }
    if tool_calls.is_empty() {
        return None;
    }
    Some(ParsedToolCalls {
        content: string_field(value, "content"),
        tool_calls: Value::Array(tool_calls),
    })
}

fn string_field(value: &Value, field: &str) -> Option<String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .filter(|content| !content.is_empty())
        .map(ToString::to_string)
}

fn request_allowed_tool_names(request: &ChatCompletionRequest) -> Vec<String> {
    if let Some(choice_name) = request
        .tool_choice
        .as_ref()
        .and_then(tool_choice_function_name)
    {
        return vec![choice_name];
    }
    request_tool_names(request)
}

fn tool_choice_function_name(value: &Value) -> Option<String> {
    value
        .as_object()
        .and_then(|object| {
            object
                .get("function")
                .and_then(|function| function.get("name"))
                .or_else(|| object.get("name"))
        })
        .and_then(Value::as_str)
        .or_else(|| {
            value
                .as_str()
                .filter(|choice| !matches!(*choice, "auto" | "none" | "required"))
        })
        .map(ToString::to_string)
}

fn request_tool_names(request: &ChatCompletionRequest) -> Vec<String> {
    request
        .tools
        .as_ref()
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|tool| {
            tool.get("function")
                .and_then(|function| function.get("name"))
                .or_else(|| tool.get("name"))
                .and_then(Value::as_str)
                .map(ToString::to_string)
        })
        .collect()
}

fn tool_call_allowed(value: &Value, allowed_names: &[String]) -> bool {
    let Some(object) = value.as_object() else {
        return false;
    };
    let function = object.get("function").and_then(Value::as_object);
    let Some(name) = function
        .and_then(|function| function.get("name"))
        .and_then(Value::as_str)
    else {
        return false;
    };
    allowed_names.is_empty() || allowed_names.iter().any(|allowed| allowed == name)
}

fn tool_calls_stream_delta(tool_calls: Value) -> Value {
    match tool_calls {
        Value::Array(calls) => Value::Array(
            calls
                .into_iter()
                .enumerate()
                .map(|(index, call)| match call {
                    Value::Object(mut object) => {
                        object
                            .entry("index")
                            .or_insert_with(|| Value::from(index as u64));
                        Value::Object(object)
                    }
                    other => other,
                })
                .collect(),
        ),
        other => other,
    }
}

fn chat_message_generation_value(
    message: &openai_frontend::ChatMessage,
    marker: &str,
    media: &mut Vec<MediaInput>,
) -> OpenAiResult<Value> {
    let mut value = serde_json::to_value(message)
        .map_err(|error| OpenAiError::invalid_request(format!("serialize message: {error}")))?;
    let content = message
        .content
        .as_ref()
        .map(|content| message_content_to_generation_text(content, marker, media))
        .transpose()?;
    if let Some(object) = value.as_object_mut() {
        match content {
            Some(content) => {
                object.insert("content".to_string(), Value::String(content));
            }
            None => {
                object.insert("content".to_string(), Value::Null);
            }
        }
    }
    Ok(value)
}

struct LocalGeneration<'a> {
    prompt_token_ids: &'a [i32],
    max_tokens: u32,
    sampling: &'a SamplingConfig,
    chat_sampling_metadata: Option<&'a str>,
    speculative: &'a SpeculativeDecodeConfig,
    native_mtp_enabled: bool,
    hook_request: Option<ChatCompletionRequest>,
    hook_runtime: Option<tokio::runtime::Handle>,
    cancellation: Option<&'a openai_frontend::CancellationToken>,
    ids: &'a OpenAiGenerationIds,
}

struct EmbeddedStageZeroGeneration<'a> {
    config: &'a StageConfig,
    wire_dtype: WireActivationDType,
    prefill_chunk_policy: &'a PrefillChunkPolicy,
    activation_width: i32,
    downstream_wire_condition: WireCondition,
    prefill_reply_credit_limit: usize,
    lane_pool: Option<Arc<PersistentStageLanePool>>,
    prediction_return: Option<PredictionReturnReceiver>,
    draft: Option<Arc<Mutex<DraftRunner>>>,
    speculative_window: usize,
    adaptive_speculative_window: bool,
    speculative: &'a SpeculativeDecodeConfig,
    ngram_min: usize,
    ngram_max: usize,
    native_mtp_enabled: bool,
    native_mtp_max_tokens: usize,
    prompt_token_ids: &'a [i32],
    max_tokens: u32,
    sampling: &'a SamplingConfig,
    chat_sampling_metadata: Option<&'a str>,
    hook_request: Option<ChatCompletionRequest>,
    hook_runtime: Option<tokio::runtime::Handle>,
    cancellation: Option<&'a openai_frontend::CancellationToken>,
    ids: &'a OpenAiGenerationIds,
}

struct SplitMultimodalGeneration<'a> {
    prompt: PreparedGenerationPrompt,
    max_tokens: GenerationTokenLimit,
    stop: Option<&'a openai_frontend::StopSequence>,
    sampling: SamplingConfig,
    cancellation: Option<&'a openai_frontend::CancellationToken>,
    ids: OpenAiGenerationIds,
    config: StageConfig,
    wire_dtype: WireActivationDType,
    activation_width: i32,
    downstream_wire_condition: WireCondition,
    lane_pool: Arc<PersistentStageLanePool>,
    prediction_return: Option<PredictionReturnReceiver>,
    emulation_active: bool,
}

struct EmbeddedLocalOutput {
    output: skippy_runtime::ActivationFrame,
    runtime_lock_wait_ms: f64,
    runtime_lock_hold_ms: f64,
}

#[derive(Default)]
struct EmbeddedExecutionStats {
    stage0_compute_ms: f64,
    runtime_lock_wait_ms: f64,
    runtime_lock_hold_ms: f64,
    activation_encode_ms: f64,
    output_activation_bytes: usize,
    forward_activation_bytes: usize,
    forward_write_ms: f64,
    downstream_wait_ms: f64,
}

struct EmbeddedStageExecution {
    reply: StageReply,
    stats: EmbeddedExecutionStats,
    elapsed_ms: f64,
}

struct EmbeddedFusedFirstDecode {
    predicted: i32,
    predicted_tokens: Vec<i32>,
    native_mtp_draft: Option<NativeMtpDraft>,
    reply_stats: StageReplyStats,
    execution: EmbeddedExecutionStats,
    elapsed_ms: f64,
    token_phase: &'static str,
    message_kind: &'static str,
}

struct EmbeddedSessionControl {
    elapsed_ms: f64,
    local_ms: f64,
    downstream_write_ms: f64,
    downstream_wait_ms: f64,
}

type GenerationStream =
    std::pin::Pin<Box<dyn futures_util::Stream<Item = OpenAiResult<GenerationStreamEvent>> + Send>>;

enum GenerationStreamEvent {
    Delta(String),
    ReasoningDelta(String),
    ToolCalls(Value),
    Usage(Usage),
    Done(FinishReason),
}

struct ChatOutputStreamParser {
    backend: StageOpenAiBackend,
    request: ChatCompletionRequest,
    metadata: String,
    emit_reasoning: bool,
    text: String,
    emitted_content: String,
    emitted_reasoning_content: String,
    emitted_tool_calls: bool,
}

impl ChatOutputStreamParser {
    fn new(
        backend: StageOpenAiBackend,
        request: ChatCompletionRequest,
        metadata: String,
        emit_reasoning: bool,
    ) -> Self {
        Self {
            backend,
            request,
            metadata,
            emit_reasoning,
            text: String::new(),
            emitted_content: String::new(),
            emitted_reasoning_content: String::new(),
            emitted_tool_calls: false,
        }
    }

    fn push_delta(&mut self, delta: &str) -> OpenAiResult<Vec<GenerationStreamEvent>> {
        self.text.push_str(delta);
        self.events_for_text(true)
    }

    fn finish(&mut self, text: &str) -> OpenAiResult<Vec<GenerationStreamEvent>> {
        if self.text != text {
            self.text = text.to_string();
        }
        self.events_for_text(false)
    }

    fn events_for_text(&mut self, is_partial: bool) -> OpenAiResult<Vec<GenerationStreamEvent>> {
        let Some(parsed) = self.backend.parse_chat_output(
            &self.text,
            &self.request,
            Some(&self.metadata),
            is_partial,
        )?
        else {
            return Ok(Vec::new());
        };
        let mut events = Vec::new();
        if self.emit_reasoning
            && let Some(delta) = suffix_delta(
                parsed.reasoning_content.as_deref(),
                &mut self.emitted_reasoning_content,
            )
        {
            events.push(GenerationStreamEvent::ReasoningDelta(delta));
        }
        if let Some(delta) = suffix_delta(parsed.content.as_deref(), &mut self.emitted_content) {
            events.push(GenerationStreamEvent::Delta(delta));
        }
        if let (true, Some(tool_calls)) =
            (!is_partial && !self.emitted_tool_calls, parsed.tool_calls)
        {
            self.emitted_tool_calls = true;
            events.push(GenerationStreamEvent::ToolCalls(tool_calls));
        }
        Ok(events)
    }

    fn finish_reason(&self, fallback: FinishReason) -> FinishReason {
        if self.emitted_tool_calls {
            FinishReason::ToolCalls
        } else {
            fallback
        }
    }
}

fn suffix_delta(current: Option<&str>, emitted: &mut String) -> Option<String> {
    let current = current?;
    let delta = current.strip_prefix(emitted.as_str())?;
    if delta.is_empty() {
        return None;
    }
    emitted.push_str(delta);
    Some(delta.to_string())
}

fn generation_event_to_chat_chunk(
    event: OpenAiResult<GenerationStreamEvent>,
    model: &str,
) -> OpenAiResult<ChatCompletionChunk> {
    match event? {
        GenerationStreamEvent::Delta(delta) => {
            Ok(ChatCompletionChunk::delta(model.to_string(), delta))
        }
        GenerationStreamEvent::ReasoningDelta(delta) => Ok(ChatCompletionChunk {
            id: openai_frontend::completion_id("chatcmpl"),
            object: "chat.completion.chunk",
            created: openai_frontend::now_unix_secs(),
            model: model.to_string(),
            choices: vec![openai_frontend::ChatCompletionChunkChoice {
                index: 0,
                delta: openai_frontend::ChatCompletionDelta {
                    role: None,
                    content: None,
                    reasoning_content: Some(delta),
                    tool_calls: None,
                },
                logprobs: None,
                finish_reason: None,
            }],
            usage: None,
        }),
        GenerationStreamEvent::ToolCalls(tool_calls) => Ok(ChatCompletionChunk {
            id: openai_frontend::completion_id("chatcmpl"),
            object: "chat.completion.chunk",
            created: openai_frontend::now_unix_secs(),
            model: model.to_string(),
            choices: vec![openai_frontend::ChatCompletionChunkChoice {
                index: 0,
                delta: openai_frontend::ChatCompletionDelta {
                    role: None,
                    content: None,
                    reasoning_content: None,
                    tool_calls: Some(tool_calls_stream_delta(tool_calls)),
                },
                logprobs: None,
                finish_reason: None,
            }],
            usage: None,
        }),
        GenerationStreamEvent::Usage(usage) => {
            Ok(ChatCompletionChunk::usage(model.to_string(), usage))
        }
        GenerationStreamEvent::Done(reason) => Ok(ChatCompletionChunk::done_with_reason(
            model.to_string(),
            reason,
        )),
    }
}

fn generation_event_to_completion_chunk(
    event: OpenAiResult<GenerationStreamEvent>,
    model: &str,
) -> OpenAiResult<CompletionChunk> {
    match event? {
        GenerationStreamEvent::Delta(delta) => Ok(CompletionChunk::delta(model.to_string(), delta)),
        GenerationStreamEvent::ReasoningDelta(_) => {
            Ok(CompletionChunk::delta(model.to_string(), ""))
        }
        GenerationStreamEvent::ToolCalls(_) => Ok(CompletionChunk::delta(model.to_string(), "")),
        GenerationStreamEvent::Usage(usage) => Ok(CompletionChunk::usage(model.to_string(), usage)),
        GenerationStreamEvent::Done(reason) => {
            Ok(CompletionChunk::done_with_reason(model.to_string(), reason))
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TokenControl {
    Continue,
    Stop,
}

/// Whether generation for this request is running under tool-call emulation:
/// the request carries tools and the rendered prompt metadata does not report
/// native tool support (empty grammar triggers). Used to enable early
/// generation stop once a complete emulated tool call has been produced.
fn emulation_generation_active(
    hook_request: Option<&ChatCompletionRequest>,
    prompt: &PreparedGenerationPrompt,
) -> bool {
    let Some(request) = hook_request else {
        return false;
    };
    if !tool_calls_requested(request) {
        return false;
    }
    prompt
        .chat_parse_metadata
        .as_deref()
        .map(|metadata| !tool_emulation::template_supports_native_tool_calls(metadata))
        .unwrap_or(false)
}

struct TextGenerationCollector<'a, F>
where
    F: FnMut(&str) -> OpenAiResult<()>,
{
    runtime: Arc<Mutex<RuntimeState>>,
    stop_values: Vec<&'a str>,
    on_text_chunk: F,
    text: String,
    streamed_text_len: usize,
    max_stop_bytes: usize,
    generated_text_tokens: Vec<i32>,
    completion_tokens: usize,
    finish_reason: FinishReason,
    metrics: GenerationMetrics,
    emulation_active: bool,
}

impl<'a, F> TextGenerationCollector<'a, F>
where
    F: FnMut(&str) -> OpenAiResult<()>,
{
    fn new(runtime: Arc<Mutex<RuntimeState>>, stop_values: Vec<&'a str>, on_text_chunk: F) -> Self {
        let max_stop_bytes = stop_values
            .iter()
            .map(|value| value.len())
            .max()
            .unwrap_or(0);
        Self {
            runtime,
            stop_values,
            on_text_chunk,
            text: String::new(),
            streamed_text_len: 0,
            max_stop_bytes,
            generated_text_tokens: Vec::new(),
            completion_tokens: 0,
            finish_reason: finish_reason_for_generation(true),
            metrics: GenerationMetrics::default(),
            emulation_active: false,
        }
    }

    /// Enables early generation stop once a complete emulated tool call is
    /// generated. Mirrors goose's `tool_call_emitted -> Stop` so the model does
    /// not ramble after emitting a call. Only active for emulated tools
    /// requests; native requests are unaffected.
    fn with_emulation_stop(mut self, emulation_active: bool) -> Self {
        self.emulation_active = emulation_active;
        self
    }

    fn push_token(&mut self, token: i32) -> OpenAiResult<TokenControl> {
        let eog_timer = Instant::now();
        if token_is_eog_with_runtime(&self.runtime, token)? {
            self.metrics.eog_check_ms += eog_timer.elapsed().as_secs_f64() * 1000.0;
            self.finish_reason = finish_reason_for_generation(false);
            return Ok(TokenControl::Stop);
        }
        self.metrics.eog_check_ms += eog_timer.elapsed().as_secs_f64() * 1000.0;
        self.completion_tokens += 1;
        self.generated_text_tokens.push(token);
        let detokenize_timer = Instant::now();
        let candidate_bytes =
            detokenize_bytes_with_runtime(&self.runtime, &self.generated_text_tokens)?;
        self.metrics.detokenize_ms += detokenize_timer.elapsed().as_secs_f64() * 1000.0;
        let valid_len = valid_utf8_prefix_len(&candidate_bytes);
        if valid_len > 0 {
            let candidate = std::str::from_utf8(&candidate_bytes[..valid_len])
                .map_err(|error| OpenAiError::backend(error.to_string()))?;
            if let Some(delta) = candidate.strip_prefix(&self.text) {
                if !delta.is_empty() {
                    self.text = candidate.to_string();
                }
            } else if candidate != self.text {
                self.text = candidate.to_string();
            }
        }
        if self
            .stop_values
            .iter()
            .any(|stop| !stop.is_empty() && self.text.contains(stop))
        {
            self.text = trim_at_stop(&self.text, &self.stop_values).to_string();
            self.emit_safe_delta(true)?;
            self.finish_reason = finish_reason_for_generation(false);
            return Ok(TokenControl::Stop);
        }
        if self.emulation_active && tool_emulation::emulated_tool_call_complete(&self.text) {
            self.emit_safe_delta(true)?;
            self.finish_reason = finish_reason_for_generation(false);
            return Ok(TokenControl::Stop);
        }
        self.emit_safe_delta(false)?;
        Ok(TokenControl::Continue)
    }

    fn emit_safe_delta(&mut self, flush_all: bool) -> OpenAiResult<()> {
        let mut target_len = if flush_all || self.max_stop_bytes == 0 {
            self.text.len()
        } else {
            self.text
                .len()
                .saturating_sub(self.max_stop_bytes.saturating_sub(1))
        };
        while target_len > self.streamed_text_len && !self.text.is_char_boundary(target_len) {
            target_len -= 1;
        }
        if target_len < self.streamed_text_len {
            self.streamed_text_len = target_len;
            return Ok(());
        }
        if target_len > self.streamed_text_len {
            let delta = &self.text[self.streamed_text_len..target_len];
            let emit_timer = Instant::now();
            (self.on_text_chunk)(delta)?;
            self.metrics.text_emit_ms += emit_timer.elapsed().as_secs_f64() * 1000.0;
            self.streamed_text_len = target_len;
        }
        Ok(())
    }

    fn finish(
        mut self,
        prompt_token_count: usize,
        cache_stats: GenerationCacheStats,
    ) -> OpenAiResult<GeneratedText> {
        self.emit_safe_delta(true)?;
        Ok(GeneratedText {
            prompt_tokens: saturating_u32(prompt_token_count),
            completion_tokens: saturating_u32(self.completion_tokens),
            cache_status: cache_stats.status,
            cached_prompt_tokens: cache_stats.cached_prompt_tokens,
            matched_prefix_tokens: cache_stats.matched_prefix_tokens,
            suffix_prefill_tokens: cache_stats.suffix_prefill_tokens,
            cache_hit_kind: cache_stats.hit_kind,
            native_mtp_stats: cache_stats.native_mtp_stats,
            native_mtp_decode_telemetry: cache_stats.native_mtp_decode_telemetry,
            verify_window_pipeline_stats: cache_stats.verify_window_pipeline_stats,
            speculative_stats: cache_stats.speculative_stats,
            prompt_ms: cache_stats.prompt_ms,
            predicted_ms: cache_stats.predicted_ms,
            text: self.text,
            finish_reason: self.finish_reason,
            detokenize_ms: self.metrics.detokenize_ms,
            text_emit_ms: self.metrics.text_emit_ms,
            eog_check_ms: self.metrics.eog_check_ms,
        })
    }
}

struct GeneratedText {
    prompt_tokens: u32,
    completion_tokens: u32,
    cache_status: &'static str,
    cached_prompt_tokens: u32,
    matched_prefix_tokens: u32,
    suffix_prefill_tokens: u32,
    cache_hit_kind: Option<&'static str>,
    native_mtp_stats: NativeMtpStats,
    native_mtp_decode_telemetry: Option<NativeMtpDecodeTelemetry>,
    verify_window_pipeline_stats: Option<VerifyWindowPipelineStats>,
    speculative_stats: Option<OpenAiSpeculativeStats>,
    prompt_ms: f64,
    predicted_ms: f64,
    text: String,
    finish_reason: FinishReason,
    detokenize_ms: f64,
    text_emit_ms: f64,
    eog_check_ms: f64,
}

impl GeneratedText {
    fn usage(&self) -> Usage {
        Usage::new(self.prompt_tokens, self.completion_tokens)
            .with_cached_tokens(self.cached_prompt_tokens)
    }

    fn timings(&self) -> Option<BTreeMap<String, Value>> {
        let stats = self.native_mtp_stats;
        let (drafted_tokens, accepted_tokens) = self
            .native_mtp_decode_telemetry
            .and_then(NativeMtpDecodeTelemetry::composite_proposal_totals)
            .unwrap_or((stats.drafted_tokens, stats.accepted_tokens));
        let mut timings = BTreeMap::from([
            ("prompt_n".to_string(), json!(self.prompt_tokens)),
            ("prompt_ms".to_string(), json!(self.prompt_ms)),
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
                "native_mtp_verifications".to_string(),
                json!(stats.verification_count),
            ),
            (
                "native_mtp_proposal_compute_us".to_string(),
                json!(stats.proposal_compute_us),
            ),
            (
                "native_mtp_verification_compute_us".to_string(),
                json!(stats.verification_compute_us),
            ),
        ]);
        if let Some(telemetry) = self.native_mtp_decode_telemetry {
            telemetry.insert_response_timings(&mut timings);
        }
        if let Some(stats) = self.verify_window_pipeline_stats {
            stats.insert_response_timings(&mut timings);
        }
        if let Some(stats) = self.speculative_stats.as_ref() {
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

#[cfg(test)]
mod tests;
