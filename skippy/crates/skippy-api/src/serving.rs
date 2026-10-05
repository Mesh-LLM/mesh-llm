//! Product-neutral options for local and staged OpenAI model serving.
use skippy_inference_api::InferenceHookPolicy;
use skippy_protocol::StageConfig;
use skippy_serving::{
    DEFAULT_GENERATION_ADMISSION_TIMEOUT_SECS, EmbeddedOpenAiArgs, EmbeddedOpenAiRequestDefaults,
    NativeMtpProposalConfig, SpeculativeDecodeConfig,
    telemetry::{Telemetry, TelemetryLevel},
};
use std::{
    net::SocketAddr,
    path::PathBuf,
    sync::{Arc, Mutex},
};
const BUILTIN_PREFILL_CHUNK_SIZE: usize = skippy_config::local_serving::PREFILL_CHUNK_SIZE;
const BUILTIN_PREFILL_ADAPTIVE_START: usize = skippy_config::local_serving::PREFILL_ADAPTIVE_START;
const BUILTIN_PREFILL_ADAPTIVE_STEP: usize = skippy_config::local_serving::PREFILL_ADAPTIVE_STEP;
const BUILTIN_PREFILL_ADAPTIVE_MAX: usize = skippy_config::local_serving::PREFILL_ADAPTIVE_MAX;
const BUILTIN_PREFILL_ADAPTIVE_TARGET_MS: f64 =
    skippy_config::local_serving::PREFILL_ADAPTIVE_TARGET_MS;
const DEFAULT_NATIVE_MTP_MAX_TOKENS: usize = skippy_config::local_serving::NATIVE_MTP_DRAFT_TOKENS;

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct OpenAiOptions {
    pub model_id: Option<String>,
    pub default_max_tokens: u32,
    pub request_defaults: EmbeddedOpenAiRequestDefaults,
    pub generation_concurrency: usize,
    pub continuous_batching: bool,
    pub pipeline_decode_groups: Option<usize>,
    pub adaptive_generation_min_concurrency: Option<usize>,
    pub generation_queue_capacity: usize,
    pub generation_admission_timeout_secs: u64,
    pub prefill_chunk_size: usize,
    pub prefill_chunk_policy: String,
    pub prefill_chunk_schedule: Option<String>,
    pub prefill_adaptive_start: usize,
    pub prefill_adaptive_step: usize,
    pub prefill_adaptive_max: usize,
    pub prefill_adaptive_target_ms: f64,
    pub draft_model_path: Option<PathBuf>,
    pub speculative_window: usize,
    pub adaptive_speculative_window: bool,
    pub draft_n_gpu_layers: Option<i32>,
    pub speculative: SpeculativeDecodeConfig,
    pub native_mtp_enabled: bool,
    pub native_mtp_draft_model_path: Option<PathBuf>,
    pub native_mtp_max_tokens: usize,
    pub native_mtp_min_tokens: usize,
    pub activation_width: i32,
    pub reply_credit_limit: Option<usize>,
    pub downstream_connect_timeout_secs: u64,
}

impl OpenAiOptions {
    pub fn direct_single_stage_defaults(
        model_id: String,
        default_max_tokens: u32,
        generation_concurrency: usize,
        native_mtp_enabled: bool,
    ) -> Self {
        Self::embedded_stage_defaults(
            Some(model_id),
            default_max_tokens,
            generation_concurrency,
            0,
            native_mtp_enabled,
        )
    }

    pub fn embedded_stage_defaults(
        model_id: Option<String>,
        default_max_tokens: u32,
        generation_concurrency: usize,
        activation_width: i32,
        native_mtp_enabled: bool,
    ) -> Self {
        Self {
            model_id,
            default_max_tokens,
            request_defaults: EmbeddedOpenAiRequestDefaults::default(),
            generation_concurrency,
            continuous_batching: skippy_config::local_serving::CONTINUOUS_BATCHING,
            pipeline_decode_groups: None,
            adaptive_generation_min_concurrency: None,
            generation_queue_capacity: skippy_serving::frontend::default_generation_queue_capacity(
                generation_concurrency,
            ),
            generation_admission_timeout_secs: DEFAULT_GENERATION_ADMISSION_TIMEOUT_SECS,
            prefill_chunk_size: BUILTIN_PREFILL_CHUNK_SIZE,
            prefill_chunk_policy: skippy_config::local_serving::PREFILL_CHUNK_POLICY.to_string(),
            prefill_chunk_schedule: None,
            prefill_adaptive_start: BUILTIN_PREFILL_ADAPTIVE_START,
            prefill_adaptive_step: BUILTIN_PREFILL_ADAPTIVE_STEP,
            prefill_adaptive_max: BUILTIN_PREFILL_ADAPTIVE_MAX,
            prefill_adaptive_target_ms: BUILTIN_PREFILL_ADAPTIVE_TARGET_MS,
            draft_model_path: None,
            speculative_window: 0,
            adaptive_speculative_window: false,
            draft_n_gpu_layers: None,
            speculative: SpeculativeDecodeConfig {
                native_mtp: NativeMtpProposalConfig {
                    enabled: native_mtp_enabled,
                    max_draft_tokens: DEFAULT_NATIVE_MTP_MAX_TOKENS,
                    min_draft_tokens: 0,
                    reject_cooldown_tokens: 0,
                    suppress_cooldown_drafts: false,
                    suppress_cooldown_draft_limit: 0,
                },
                effective_strategy: if native_mtp_enabled {
                    "native-mtp".to_string()
                } else {
                    "disabled".to_string()
                },
                ..SpeculativeDecodeConfig::default()
            },
            native_mtp_enabled,
            native_mtp_draft_model_path: None,
            native_mtp_max_tokens: DEFAULT_NATIVE_MTP_MAX_TOKENS,
            native_mtp_min_tokens: 0,
            activation_width,
            reply_credit_limit: None,
            downstream_connect_timeout_secs:
                skippy_config::local_serving::DOWNSTREAM_CONNECT_TIMEOUT_SECS,
        }
    }

    pub fn build(
        self,
        bind_addr: SocketAddr,
        config: StageConfig,
        runtime: Arc<Mutex<skippy_serving::runtime_state::RuntimeState>>,
        telemetry: Telemetry,
        hook_policy: Option<Arc<dyn InferenceHookPolicy>>,
    ) -> EmbeddedOpenAiArgs {
        EmbeddedOpenAiArgs {
            bind_addr,
            config,
            runtime,
            model_id: self.model_id,
            default_max_tokens: self.default_max_tokens,
            request_defaults: self.request_defaults,
            generation_concurrency: self.generation_concurrency,
            continuous_batching: self.continuous_batching,
            pipeline_decode_groups: self.pipeline_decode_groups,
            adaptive_generation_min_concurrency: self.adaptive_generation_min_concurrency,
            generation_queue_capacity: self.generation_queue_capacity,
            generation_admission_timeout_secs: self.generation_admission_timeout_secs,
            prefill_chunk_size: self.prefill_chunk_size,
            prefill_chunk_policy: self.prefill_chunk_policy,
            prefill_chunk_schedule: self.prefill_chunk_schedule,
            prefill_adaptive_start: self.prefill_adaptive_start,
            prefill_adaptive_step: self.prefill_adaptive_step,
            prefill_adaptive_max: self.prefill_adaptive_max,
            prefill_adaptive_target_ms: self.prefill_adaptive_target_ms,
            draft_model_path: self.draft_model_path,
            speculative_window: self.speculative_window,
            adaptive_speculative_window: self.adaptive_speculative_window,
            draft_n_gpu_layers: self.draft_n_gpu_layers,
            speculative: self.speculative,
            native_mtp_enabled: self.native_mtp_enabled,
            native_mtp_draft_model_path: self.native_mtp_draft_model_path,
            native_mtp_max_tokens: self.native_mtp_max_tokens,
            native_mtp_min_tokens: self.native_mtp_min_tokens,
            activation_width: self.activation_width,
            reply_credit_limit: self.reply_credit_limit,
            downstream_connect_timeout_secs: self.downstream_connect_timeout_secs,
            downstream_wire_condition: skippy_serving::binary_transport::WireCondition::new(
                0.0, None,
            )
            .expect("static downstream wire condition should construct"),
            prediction_returns: None,
            telemetry,
            hook_policy,
            generation_receipt: None,
            generation_lifecycle: None,
            linear_proposal_ingress: None,
            kv_lifecycle_observer: None,
            openai_guardrails: None,
            l3_manager: None,
        }
    }
}

mod lifecycle;
pub use lifecycle::{LoadedModelBackend, ModelLoadRequest, ModelOpenEvents};

#[derive(Clone, Debug)]
pub struct ServingTelemetryOptions {
    pub metrics_otlp_grpc: Option<String>,
    pub queue_capacity: usize,
    pub level: TelemetryLevel,
}

impl Default for ServingTelemetryOptions {
    fn default() -> Self {
        Self::off()
    }
}

impl ServingTelemetryOptions {
    pub fn off() -> Self {
        Self {
            metrics_otlp_grpc: None,
            queue_capacity: 0,
            level: TelemetryLevel::Off,
        }
    }

    pub fn debug(metrics_otlp_grpc: Option<String>) -> Self {
        Self {
            metrics_otlp_grpc,
            queue_capacity: 1024,
            level: TelemetryLevel::Debug,
        }
    }

    pub fn summary(metrics_otlp_grpc: String) -> Self {
        Self {
            metrics_otlp_grpc: Some(metrics_otlp_grpc),
            queue_capacity: 1024,
            level: TelemetryLevel::Summary,
        }
    }
}

mod local;
pub use local::{LocalDiskCacheOptions, LocalOpenAiOptions, serve_local_openai_with_shutdown};

/// Startup handshakes and cancellation shared by embedded stage hosts.
pub use skippy_serving::readiness;
