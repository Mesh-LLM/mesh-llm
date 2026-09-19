//! Product-neutral options for local and staged OpenAI model serving.
use openai_frontend::OpenAiHookPolicy;
use skippy_protocol::StageConfig;
use skippy_server::{
    DEFAULT_GENERATION_ADMISSION_TIMEOUT_SECS, EmbeddedOpenAiArgs, EmbeddedOpenAiRequestDefaults,
    NativeMtpProposalConfig, SpeculativeDecodeConfig,
    telemetry::{Telemetry, TelemetryLevel},
};
use std::{
    net::SocketAddr,
    path::PathBuf,
    sync::{Arc, Mutex},
};
const BUILTIN_PREFILL_CHUNK_SIZE: usize = 64;
const BUILTIN_PREFILL_ADAPTIVE_START: usize = 64;
const BUILTIN_PREFILL_ADAPTIVE_STEP: usize = 64;
const BUILTIN_PREFILL_ADAPTIVE_MAX: usize = 512;
const BUILTIN_PREFILL_ADAPTIVE_TARGET_MS: f64 = 100.0;
const DEFAULT_NATIVE_MTP_MAX_TOKENS: usize = 3;

#[derive(Clone, Debug, PartialEq)]
pub struct OpenAiOptions {
    pub model_id: Option<String>,
    pub default_max_tokens: u32,
    pub request_defaults: EmbeddedOpenAiRequestDefaults,
    pub generation_concurrency: usize,
    pub continuous_batching: bool,
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
        Self {
            model_id: Some(model_id),
            default_max_tokens,
            request_defaults: EmbeddedOpenAiRequestDefaults::default(),
            generation_concurrency,
            continuous_batching: true,
            prefill_chunk_size: BUILTIN_PREFILL_CHUNK_SIZE,
            prefill_chunk_policy: "fixed".to_string(),
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
                    max_draft_tokens: if native_mtp_enabled {
                        DEFAULT_NATIVE_MTP_MAX_TOKENS
                    } else {
                        1
                    },
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
            native_mtp_max_tokens: if native_mtp_enabled {
                DEFAULT_NATIVE_MTP_MAX_TOKENS
            } else {
                0
            },
            native_mtp_min_tokens: 0,
            activation_width: 0,
            reply_credit_limit: None,
            downstream_connect_timeout_secs: 30,
        }
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
            continuous_batching: true,
            prefill_chunk_size: BUILTIN_PREFILL_CHUNK_SIZE,
            prefill_chunk_policy: "fixed".to_string(),
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
                    max_draft_tokens: if native_mtp_enabled {
                        DEFAULT_NATIVE_MTP_MAX_TOKENS
                    } else {
                        1
                    },
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
            native_mtp_max_tokens: if native_mtp_enabled {
                DEFAULT_NATIVE_MTP_MAX_TOKENS
            } else {
                0
            },
            native_mtp_min_tokens: 0,
            activation_width,
            reply_credit_limit: None,
            downstream_connect_timeout_secs: 30,
        }
    }

    pub fn build(
        self,
        bind_addr: SocketAddr,
        config: StageConfig,
        runtime: Arc<Mutex<skippy_server::runtime_state::RuntimeState>>,
        telemetry: Telemetry,
        hook_policy: Option<Arc<dyn OpenAiHookPolicy>>,
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
            adaptive_generation_min_concurrency: None,
            generation_queue_capacity: self.generation_concurrency.saturating_mul(8).clamp(16, 256),
            generation_admission_timeout_secs: DEFAULT_GENERATION_ADMISSION_TIMEOUT_SECS,
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
            downstream_wire_condition: skippy_server::binary_transport::WireCondition::new(
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
