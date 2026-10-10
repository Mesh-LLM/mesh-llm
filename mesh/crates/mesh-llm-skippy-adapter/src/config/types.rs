use std::path::{Path, PathBuf};

use skippy_protocol::{
    FlashAttentionType, StageKvCacheCodec, StageKvCacheMode, StageKvCachePayload,
};
use skippy_runtime::package::PackageGenerationInfo;
use skippy_serving::SpeculativeDecodeConfig;

use mesh_llm_config::{MeshConfig, ReasoningBudget, ReasoningEnabled, RequestDefaultsConfig};

pub(super) const BUILTIN_CTX_SIZE: u32 = skippy_config::local_serving::CTX_SIZE;
pub(super) const BUILTIN_BATCH: u32 = skippy_config::local_serving::BATCH;
/// Matches llama.cpp's own default (`LLAMA_SERVER_DEFAULT_N_UBATCH = 512`) and clears
/// the CUDA SSM SSD kernel gate (`n_tok > SSM_SSD_MIN_TOKENS`, 128, strict), which the
/// previous 128 default missed by exactly one token — forcing every recurrent (mamba)
/// prefill onto the sequential scan fallback. Measured on granite-4.0-h-1b: TTFT p50
/// 0.670 → 0.415 s (C1) and 6.38 → 3.97 s (C8); decode 22.2 → 39.4 tok/s at C8.
/// See WHITE_UBATCH_512_FALSIFICATION_2026_09_08 in the 2026-09-08 competitive bench.
pub(super) const BUILTIN_UBATCH: u32 = skippy_config::local_serving::UBATCH;
pub(super) const BUILTIN_PARALLEL: usize = skippy_config::local_serving::PARALLEL;
pub(super) const BUILTIN_PREFILL_CHUNK_SIZE: usize =
    skippy_config::local_serving::PREFILL_CHUNK_SIZE;
pub(super) const BUILTIN_PREFILL_ADAPTIVE_START: usize =
    skippy_config::local_serving::PREFILL_ADAPTIVE_START;
pub(super) const BUILTIN_PREFILL_ADAPTIVE_STEP: usize =
    skippy_config::local_serving::PREFILL_ADAPTIVE_STEP;
pub(super) const BUILTIN_PREFILL_ADAPTIVE_MAX: usize =
    skippy_config::local_serving::PREFILL_ADAPTIVE_MAX;
pub(super) const BUILTIN_PREFILL_ADAPTIVE_TARGET_MS: f64 =
    skippy_config::local_serving::PREFILL_ADAPTIVE_TARGET_MS;
pub(super) const BUILTIN_SAFETY_MARGIN_GB: f64 = skippy_config::capacity::BUILTIN_SAFETY_MARGIN_GB;

#[derive(Clone, Debug)]
pub struct SkippyConfigResolveRequest<'a> {
    pub mesh_config: &'a MeshConfig,
    pub model_id: &'a str,
    pub model_path: &'a Path,
    pub model_bytes: u64,
    pub allocatable_memory_bytes: Option<u64>,
    pub request_defaults: Option<&'a RequestDefaultsConfig>,
    pub package_generation: Option<&'a PackageGenerationInfo>,
    /// GGUF metadata for the model being resolved, when available. Used to
    /// guard publisher-declared quantised K/V against native load constraints
    /// such as Flash Attention and block alignment.
    pub compact_meta: Option<&'a skippy_model_artifact::gguf::GgufCompactMeta>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedSkippyConfig {
    pub model_id: String,
    pub model_path: PathBuf,
    pub model_fit: ResolvedModelFitConfig,
    pub hardware: ResolvedHardwareConfig,
    pub throughput: ResolvedThroughputConfig,
    pub skippy: ResolvedSkippyExecutionConfig,
    pub speculative: ResolvedSpeculativeConfig,
    pub request_defaults: ResolvedRequestDefaultsConfig,
    pub multimodal: ResolvedMultimodalConfig,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedMultimodalConfig {
    pub projector_url: Option<String>,
    pub projector_use_gpu: Option<bool>,
    pub media_marker: Option<String>,
    pub image_min_tokens: Option<u32>,
    pub image_max_tokens: Option<u32>,
    pub batch_max_tokens: Option<u32>,
    pub glm_dsa_policy: skippy_protocol::GlmDsaPolicy,
    pub generation_signal_window: Option<u32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedModelFitConfig {
    pub ctx_size: u32,
    pub batch: u32,
    pub ubatch: u32,
    pub cache_type_k: String,
    pub cache_type_v: String,
    pub prefix_cache: ResolvedStageKvCache,
    pub l2_max_bytes: u64,
    pub kv_cache_codec: StageKvCacheCodec,
    pub kv_offload: String,
    /// Parsed `kv_offload` for the native tri-state control. `None` covers
    /// both "auto" and any value that did not parse to a bool.
    pub kv_offload_resolved: Option<bool>,
    pub kv_unified: Option<bool>,
    pub swa_full: Option<bool>,
    pub cache_idle_slots: Option<u32>,
    pub flash_attention: FlashAttentionType,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedHardwareConfig {
    pub device: Option<String>,
    pub gpu_layers: i32,
    pub mmap: Option<bool>,
    pub mlock: bool,
    pub repack: bool,
    pub op_offload: Option<bool>,
    pub no_host_buffer: bool,
    pub check_tensors: bool,
    pub checkpoint_quantization: Option<String>,
    pub checkpoint_imatrix: Option<String>,
    pub direct_io: bool,
    pub main_gpu: Option<u32>,
    pub split_mode: skippy_protocol::SplitMode,
    pub fit_target_mib: Option<u64>,
    pub resolved_model_path: PathBuf,
    pub projector_path: Option<PathBuf>,
    pub stage_layer_start: Option<u32>,
    pub stage_layer_end: Option<u32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedThroughputConfig {
    pub parallel: usize,
    pub continuous_batching: String,
    pub pipeline_decode_groups: Option<u32>,
    /// Planned final-stage decode batching. `None` leaves it unbatched.
    pub last_stage_decode_batch: Option<bool>,
    pub threads: Option<usize>,
    pub threads_batch: Option<usize>,
    pub tuning_profile: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedSkippyExecutionConfig {
    pub binary_stage_transport: String,
    pub prefill_chunking: String,
    pub prefill_chunk_size: usize,
    pub prefill_chunk_schedule: Option<String>,
    pub prefill_controls_explicit: bool,
    pub lifecycle_startup_timeout_ms: Option<u64>,
    pub lifecycle_health_interval_ms: Option<u64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedSpeculativeConfig {
    pub strategy: String,
    pub native_mtp_enabled: bool,
    pub mode: String,
    pub draft_model_path: Option<PathBuf>,
    pub pairing_fault: String,
    pub draft_max_tokens: u32,
    pub draft_min_tokens: u32,
    pub explicit: bool,
    pub draft_n_gpu_layers: Option<i32>,
    pub decode: SpeculativeDecodeConfig,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResolvedStageKvCache {
    FamilyDefault,
    Disabled,
    Explicit(ResolvedStageKvCacheTemplate),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedStageKvCacheTemplate {
    pub mode: StageKvCacheMode,
    pub payload: StageKvCachePayload,
    pub max_entries: Option<usize>,
    pub max_bytes: Option<u64>,
    pub min_tokens: Option<u64>,
    pub shared_prefix_stride_tokens: Option<u64>,
    pub shared_prefix_record_limit: Option<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedRequestDefaultsConfig {
    pub max_tokens: Option<u32>,
    pub package_request_defaults: Option<skippy_package_format::GenerationRequestDefaults>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub presence_penalty: Option<f64>,
    pub frequency_penalty: Option<f64>,
    pub seed: Option<i64>,
    pub logit_bias: Option<toml::Value>,
    pub top_k: Option<i64>,
    pub min_p: Option<f64>,
    pub typical_p: Option<f64>,
    pub top_nsigma: Option<f64>,
    pub dynatemp_range: Option<f64>,
    pub dynatemp_exponent: Option<f64>,
    pub dry: Option<mesh_llm_config::DrySamplingConfig>,
    pub xtc: Option<mesh_llm_config::XtcSamplingConfig>,
    pub mirostat_mode: Option<mesh_llm_config::IntegerOrString>,
    pub mirostat_entropy: Option<f64>,
    pub mirostat_learning_rate: Option<f64>,
    pub samplers: Option<Vec<String>>,
    pub sampler_sequence: Option<String>,
    pub ignore_eos: Option<bool>,
    pub repeat_penalty: Option<f64>,
    pub repeat_last_n: Option<i64>,
    pub stop: Option<Vec<String>>,
    pub reasoning_format: Option<String>,
    pub reasoning_enabled: Option<ReasoningEnabled>,
    pub reasoning_budget: Option<ReasoningBudget>,
    pub chat_template: Option<String>,
    pub chat_template_file: Option<String>,
    pub allow_request_chat_template: Option<bool>,
    pub jinja: Option<bool>,
    pub chat_template_kwargs: Option<toml::Value>,
    pub skip_chat_parsing: Option<bool>,
    pub prefill_assistant: Option<toml::Value>,
    pub system_prompt: Option<String>,
    pub grammar: Option<toml::Value>,
    pub json_schema: Option<toml::Value>,
}

pub use skippy_api::serving::InferenceOptions as ResolvedEmbeddedOpenAiArgs;
