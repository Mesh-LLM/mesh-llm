mod built_in_schema;
mod schema_types;

pub use built_in_schema::{
    BuiltInConfigPathResolution, built_in_config_schema_descriptor, built_in_config_settings,
    canonicalize_built_in_config_identifier, canonicalize_built_in_config_path,
    resolve_built_in_config_identifier, resolve_built_in_config_path,
};
pub use schema_types::*;

pub use mesh_llm_types::runtime::ModelRuntimeKind;
use serde::ser::SerializeStruct;
use serde::{Deserialize, Serialize};
pub use skippy_protocol::FlashAttentionType;
use std::collections::BTreeMap;

#[derive(Clone, Debug, Default, Serialize)]
pub struct MeshConfig {
    #[serde(default)]
    pub version: Option<u32>,
    #[serde(default)]
    pub gpu: GpuConfig,
    #[serde(default)]
    pub mesh_requirements: MeshRequirementsConfig,
    #[serde(default)]
    pub owner_control: OwnerControlConfig,
    #[serde(default)]
    pub telemetry: TelemetryConfig,
    #[serde(default)]
    pub defaults: Option<ModelConfigDefaults>,
    #[serde(default)]
    pub runtime: RuntimeConfig,
    #[serde(default)]
    pub models: Vec<ModelConfigEntry>,
    #[serde(rename = "plugin", default)]
    pub plugins: Vec<PluginConfigEntry>,
    #[serde(flatten, default)]
    pub extra: BTreeMap<String, toml::Value>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct OwnerControlConfig {
    #[serde(default)]
    pub bind: Option<std::net::SocketAddr>,
    #[serde(default)]
    pub advertise_addr: Option<std::net::SocketAddr>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct GpuConfig {
    #[serde(default)]
    pub assignment: GpuAssignment,
    #[serde(default)]
    pub parallel: Option<usize>,
}

pub const DEFAULT_MODEL_TARGET_DEMAND_UPGRADE_MIN_REQUESTS: u64 = 2;
pub const DEFAULT_MODEL_TARGET_DEMAND_UPGRADE_MAX_AGE_SECS: u64 = 60 * 60;

fn default_model_target_demand_upgrade_min_requests() -> u64 {
    DEFAULT_MODEL_TARGET_DEMAND_UPGRADE_MIN_REQUESTS
}

fn default_model_target_demand_upgrade_max_age_secs() -> u64 {
    DEFAULT_MODEL_TARGET_DEMAND_UPGRADE_MAX_AGE_SECS
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct RuntimeConfig {
    #[serde(default)]
    pub debug: bool,
    #[serde(default)]
    pub listen_all: bool,
    #[serde(default)]
    pub reconcile_model_targets: bool,
    #[serde(default)]
    pub reconcile_model_target_demand_upgrades: bool,
    #[serde(default)]
    pub native_runtime: NativeRuntimeConfig,
    #[serde(default = "default_model_target_demand_upgrade_min_requests")]
    pub model_target_demand_upgrade_min_requests: u64,
    #[serde(default = "default_model_target_demand_upgrade_max_age_secs")]
    pub model_target_demand_upgrade_max_age_secs: u64,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            debug: false,
            listen_all: false,
            reconcile_model_targets: false,
            reconcile_model_target_demand_upgrades: false,
            native_runtime: NativeRuntimeConfig::default(),
            model_target_demand_upgrade_min_requests:
                DEFAULT_MODEL_TARGET_DEMAND_UPGRADE_MIN_REQUESTS,
            model_target_demand_upgrade_max_age_secs:
                DEFAULT_MODEL_TARGET_DEMAND_UPGRADE_MAX_AGE_SECS,
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct NativeRuntimeConfig {
    #[serde(default)]
    pub mesh_version: Option<String>,
    #[serde(default)]
    pub skippy_abi: Option<String>,
    #[serde(default)]
    pub selection: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct MeshRequirementsConfig {
    #[serde(default)]
    pub min_node_version: Option<String>,
    #[serde(default)]
    pub max_node_version: Option<String>,
    #[serde(default)]
    pub min_protocol_version: Option<u32>,
    #[serde(default)]
    pub max_protocol_version: Option<u32>,
    #[serde(default)]
    pub require_release_attestation: bool,
    #[serde(default)]
    pub release_signer_keys: Vec<String>,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum GpuAssignment {
    #[default]
    Auto,
    Pinned,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ModelConfigDefaults {
    #[serde(default)]
    pub model_fit: Option<ModelFitConfig>,
    #[serde(default)]
    pub hardware: Option<HardwareConfig>,
    #[serde(default)]
    pub throughput: Option<ThroughputConfig>,
    #[serde(default)]
    pub skippy: Option<SkippyConfig>,
    #[serde(default)]
    pub speculative: Option<SpeculativeConfig>,
    #[serde(default)]
    pub request_defaults: Option<RequestDefaultsConfig>,
    #[serde(default)]
    pub multimodal: Option<MultimodalConfig>,
    #[serde(default)]
    pub advanced: Option<AdvancedConfig>,
}

#[derive(Clone, Debug, Default)]
pub struct ModelConfigEntry {
    pub model: String,
    pub mmproj: Option<String>,
    pub ctx_size: Option<u32>,
    pub gpu_id: Option<String>,
    pub parallel: Option<usize>,
    pub cache_type_k: Option<String>,
    pub cache_type_v: Option<String>,
    pub batch: Option<u32>,
    pub ubatch: Option<u32>,
    pub flash_attention: Option<FlashAttentionType>,
    pub model_fit: Option<ModelFitConfig>,
    pub hardware: Option<HardwareConfig>,
    pub throughput: Option<ThroughputConfig>,
    pub skippy: Option<SkippyConfig>,
    pub speculative: Option<SpeculativeConfig>,
    pub request_defaults: Option<RequestDefaultsConfig>,
    pub multimodal: Option<MultimodalConfig>,
    pub advanced: Option<AdvancedConfig>,
    pub gpu_id_from_legacy_shim: bool,
}

impl Serialize for ModelConfigEntry {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("ModelConfigEntry", 18)?;
        state.serialize_field("model", &self.model)?;
        if let Some(value) = &self.mmproj {
            state.serialize_field("mmproj", value)?;
        }
        if let Some(value) = &self.ctx_size {
            state.serialize_field("ctx_size", value)?;
        }
        if self.gpu_id_from_legacy_shim
            && let Some(value) = &self.gpu_id
        {
            state.serialize_field("gpu_id", value)?;
        }
        if let Some(value) = &self.parallel {
            state.serialize_field("parallel", value)?;
        }
        if let Some(value) = &self.cache_type_k {
            state.serialize_field("cache_type_k", value)?;
        }
        if let Some(value) = &self.cache_type_v {
            state.serialize_field("cache_type_v", value)?;
        }
        if let Some(value) = &self.batch {
            state.serialize_field("batch", value)?;
        }
        if let Some(value) = &self.ubatch {
            state.serialize_field("ubatch", value)?;
        }
        if let Some(value) = &self.flash_attention {
            state.serialize_field("flash_attention", value)?;
        }
        if let Some(value) = &self.model_fit {
            state.serialize_field("model_fit", value)?;
        }
        if let Some(value) = &self.hardware {
            state.serialize_field("hardware", value)?;
        }
        if let Some(value) = &self.throughput {
            state.serialize_field("throughput", value)?;
        }
        if let Some(value) = &self.skippy {
            state.serialize_field("skippy", value)?;
        }
        if let Some(value) = &self.speculative {
            state.serialize_field("speculative", value)?;
        }
        if let Some(value) = &self.request_defaults {
            state.serialize_field("request_defaults", value)?;
        }
        if let Some(value) = &self.multimodal {
            state.serialize_field("multimodal", value)?;
        }
        if let Some(value) = &self.advanced {
            state.serialize_field("advanced", value)?;
        }
        state.end()
    }
}

impl ModelConfigEntry {
    /// Compute a derived profile hash from the runtime-shaping fields of this entry.
    ///
    /// The profile is derived from the fields that materially affect runtime
    /// behavior: ModelFitConfig (ctx_size, batch, ubatch, cache_type_k,
    /// cache_type_v, flash_attention), HardwareConfig (model_runtime, device,
    /// gpu_layers, tensor_split, split_mode, main_gpu, cpu_moe, n_cpu_moe,
    /// fit_target_mib, mmap, mlock), and ThroughputConfig (parallel,
    /// continuous_batching, threads, threads_batch).
    ///
    /// Returns an 8-hex-character string (e.g. "a3f2b9c1"), or empty string
    /// if all profile-input fields are at their defaults.
    /// Derive a stable profile string from the runtime-shaping config fields.
    ///
    /// Returns an 8-hex-char hash when any profile-input field is set,
    /// or an empty string (profile = default) when all inputs are at defaults.
    pub fn derived_profile(&self) -> String {
        let mut buf = Vec::new();
        Self::write_effective_fit_profile(&mut buf, self);
        Self::write_effective_hw_profile(&mut buf, self);
        Self::write_effective_tp_profile(&mut buf, self);

        if buf.is_empty() {
            return String::new();
        }

        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        buf.hash(&mut hasher);
        let hash = hasher.finish();
        format!("{:08x}", hash & 0xFFFFFFFF)
    }

    fn write_effective_fit_profile(buf: &mut Vec<u8>, entry: &ModelConfigEntry) {
        use std::io::Write;
        macro_rules! wo {
            ($key:literal, $val:expr) => {
                if let Some(ref v) = $val {
                    let _ = write!(buf, concat!($key, "={:?}\0"), v);
                }
            };
        }
        // Effective fit fields: sub-config (set by ConfigEditor) preferred,
        // top-level (set by direct Rust construction) as fallback.
        let fit = entry.model_fit.as_ref();
        wo!("ctx_size", fit.and_then(|f| f.ctx_size).or(entry.ctx_size));
        wo!("batch", fit.and_then(|f| f.batch).or(entry.batch));
        wo!("ubatch", fit.and_then(|f| f.ubatch).or(entry.ubatch));
        wo!(
            "cache_type_k",
            fit.and_then(|f| f.cache_type_k.as_ref())
                .or(entry.cache_type_k.as_ref())
        );
        wo!(
            "cache_type_v",
            fit.and_then(|f| f.cache_type_v.as_ref())
                .or(entry.cache_type_v.as_ref())
        );
        wo!(
            "flash_attention",
            fit.and_then(|f| f.flash_attention)
                .or(entry.flash_attention)
        );
    }

    fn write_effective_hw_profile(buf: &mut Vec<u8>, entry: &ModelConfigEntry) {
        use std::io::Write;
        macro_rules! wo {
            ($key:literal, $val:expr) => {
                if let Some(ref v) = $val {
                    let _ = write!(buf, concat!($key, "={:?}\0"), v);
                }
            };
        }
        let hw = entry.hardware.as_ref();
        wo!(
            "gpu_id",
            hw.and_then(|h| h.device.as_ref()).or(entry.gpu_id.as_ref())
        );
        if let Some(hw) = hw {
            wo!("model_runtime", hw.model_runtime);
            wo!("gpu_layers", hw.gpu_layers);
            wo!("tensor_split", hw.tensor_split);
            wo!("split_mode", hw.split_mode);
            wo!("main_gpu", hw.main_gpu);
            wo!("cpu_moe", hw.cpu_moe);
            wo!("n_cpu_moe", hw.n_cpu_moe);
            wo!("fit_target_mib", hw.fit_target_mib);
            wo!("mmap", hw.mmap);
            wo!("mlock", hw.mlock);
        }
    }

    fn write_effective_tp_profile(buf: &mut Vec<u8>, entry: &ModelConfigEntry) {
        use std::io::Write;
        macro_rules! wo {
            ($key:literal, $val:expr) => {
                if let Some(ref v) = $val {
                    let _ = write!(buf, concat!($key, "={:?}\0"), v);
                }
            };
        }
        let tp = entry.throughput.as_ref();
        wo!("parallel", tp.and_then(|t| t.parallel).or(entry.parallel));
        if let Some(tp) = tp {
            wo!("continuous_batching", tp.continuous_batching);
            wo!("threads", tp.threads);
            wo!("threads_batch", tp.threads_batch);
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ModelFitConfig {
    #[serde(default)]
    pub ctx_size: Option<u32>,
    #[serde(default)]
    pub batch: Option<u32>,
    #[serde(default)]
    pub ubatch: Option<u32>,
    #[serde(default)]
    pub cache_type_k: Option<String>,
    #[serde(default)]
    pub cache_type_v: Option<String>,
    #[serde(default)]
    pub kv_cache_policy: Option<String>,
    #[serde(default)]
    pub kv_offload: Option<BoolOrAuto>,
    #[serde(default)]
    pub kv_unified: Option<BoolOrAuto>,
    #[serde(default)]
    pub cache_ram_mib: Option<u64>,
    #[serde(default)]
    pub cache_idle_slots: Option<u32>,
    #[serde(default)]
    pub prompt_cache: Option<BoolOrAuto>,
    #[serde(default)]
    pub prefix_cache: Option<PrefixCacheConfig>,
    #[serde(default)]
    pub keep_tokens: Option<u32>,
    #[serde(default)]
    pub context_shift: Option<BoolOrAuto>,
    #[serde(default)]
    pub swa_full: Option<bool>,
    #[serde(default)]
    pub checkpoint_interval: Option<u32>,
    #[serde(default)]
    pub checkpoint_count: Option<u32>,
    #[serde(default)]
    pub lookup_cache_static: Option<String>,
    #[serde(default)]
    pub lookup_cache_dynamic: Option<String>,
    #[serde(default)]
    pub flash_attention: Option<FlashAttentionType>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PrefixCacheConfig {
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub max_entries: Option<u32>,
    #[serde(default)]
    pub max_bytes: Option<u64>,
    #[serde(default)]
    pub min_tokens: Option<u32>,
    #[serde(default)]
    pub shared_stride_tokens: Option<u32>,
    #[serde(default)]
    pub shared_record_limit: Option<u32>,
    #[serde(default)]
    pub payload_mode: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct HardwareConfig {
    #[serde(default)]
    pub model_runtime: Option<ModelRuntimeKind>,
    #[serde(default)]
    pub device: Option<String>,
    #[serde(default)]
    pub gpu_layers: Option<IntegerOrString>,
    #[serde(default)]
    pub stage_layer_start: Option<u32>,
    #[serde(default)]
    pub stage_layer_end: Option<u32>,
    #[serde(default)]
    pub placement: Option<String>,
    #[serde(default)]
    pub tensor_split: Option<TensorSplitConfig>,
    #[serde(default)]
    pub split_mode: Option<String>,
    #[serde(default)]
    pub main_gpu: Option<u32>,
    #[serde(default)]
    pub cpu_moe: Option<BoolOrAuto>,
    #[serde(default)]
    pub n_cpu_moe: Option<u32>,
    #[serde(default)]
    pub rpc_backend: Option<toml::Value>,
    #[serde(default)]
    pub fit_target_mib: Option<u64>,
    #[serde(default)]
    pub safety_margin_gb: Option<f64>,
    #[serde(default)]
    pub fit_context: Option<BoolOrAuto>,
    #[serde(default)]
    pub model_path: Option<String>,
    #[serde(default)]
    pub hf_repo: Option<String>,
    #[serde(default)]
    pub hf_file: Option<String>,
    #[serde(default)]
    pub mmproj: Option<String>,
    #[serde(default)]
    pub mmproj_offload: Option<BoolOrAuto>,
    #[serde(default)]
    pub lora_adapters: Vec<String>,
    #[serde(default)]
    pub control_vectors: Vec<String>,
    #[serde(default)]
    pub check_tensors: Option<bool>,
    #[serde(default)]
    pub mmap: Option<BoolOrAuto>,
    #[serde(default)]
    pub mlock: Option<bool>,
    #[serde(default)]
    pub direct_io: Option<bool>,
    #[serde(default)]
    pub repack: Option<bool>,
    #[serde(default)]
    pub op_offload: Option<bool>,
    #[serde(default)]
    pub no_host_buffer: Option<bool>,
    #[serde(default)]
    pub warmup: Option<BoolOrAuto>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ThroughputConfig {
    #[serde(default)]
    pub parallel: Option<usize>,
    #[serde(default)]
    pub continuous_batching: Option<BoolOrAuto>,
    #[serde(default)]
    pub threads: Option<usize>,
    #[serde(default)]
    pub threads_batch: Option<usize>,
    #[serde(default)]
    pub threads_http: Option<usize>,
    #[serde(default)]
    pub priority: Option<IntegerOrString>,
    #[serde(default)]
    pub poll: Option<BoolOrString>,
    #[serde(default)]
    pub cpu_affinity: Option<StringOrStringList>,
    #[serde(default)]
    pub numa: Option<String>,
    #[serde(default)]
    pub slot_prompt_similarity: Option<f64>,
    #[serde(default)]
    pub sleep_idle_seconds: Option<u64>,
    #[serde(default)]
    pub tuning_profile: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SkippyConfig {
    #[serde(default)]
    pub stage_model_path: Option<String>,
    #[serde(default)]
    pub stage_role: Option<String>,
    #[serde(default)]
    pub stage_topology: Option<String>,
    #[serde(default)]
    pub activation_wire_dtype: Option<String>,
    #[serde(default)]
    pub binary_stage_transport: Option<String>,
    #[serde(default)]
    pub openai_frontend_mode: Option<toml::Value>,
    #[serde(default)]
    pub lifecycle_startup_timeout_ms: Option<u64>,
    #[serde(default)]
    pub lifecycle_readiness_interval_ms: Option<u64>,
    #[serde(default)]
    pub lifecycle_health_interval_ms: Option<u64>,
    #[serde(default)]
    pub prefill_chunking: Option<String>,
    #[serde(default)]
    pub prefill_chunk_size: Option<u32>,
    #[serde(default)]
    pub prefill_chunk_schedule: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SpeculativeConfig {
    pub strategy: Option<String>,
    pub mode: Option<String>,
    pub draft_model: Option<String>,
    pub draft_hf_repo: Option<String>,
    pub draft_hf_file: Option<String>,
    pub draft_selection_policy: Option<String>,
    pub pairing_fault: Option<String>,
    pub draft_max_tokens: Option<u32>,
    pub draft_min_tokens: Option<u32>,
    pub draft_acceptance_threshold: Option<f64>,
    pub draft_split_probability: Option<f64>,
    pub draft_gpu_layers: Option<i32>,
    pub draft_device: Option<String>,
    pub draft_threads: Option<usize>,
    pub draft_cache_type_k: Option<String>,
    pub draft_cache_type_v: Option<String>,
    pub ngram_min: Option<u32>,
    pub ngram_max: Option<u32>,
    pub ngram_max_proposal_tokens: Option<u32>,
    pub ngram_proposer: Option<String>,
    pub extension_max_tokens: Option<u32>,
    pub native_mtp_reject_cooldown_tokens: Option<u32>,
    pub native_mtp_suppress_cooldown_drafts: Option<bool>,
    pub native_mtp_suppress_cooldown_draft_limit: Option<u32>,
    pub verify_window_min_tokens: Option<u32>,
    pub verify_window_max_tokens: Option<u32>,
    pub verify_window_pipeline_depth: Option<u32>,
    pub spec_default: Option<BoolOrAuto>,
    pub(crate) legacy_draft_model_path_used: bool,
}

impl SpeculativeConfig {
    /// Resolves the three supported policy layers without discarding fields
    /// that are not overridden by a more specific layer.
    pub fn with_precedence(
        overrides: Option<&Self>,
        model: Option<&Self>,
        defaults: Option<&Self>,
    ) -> Self {
        macro_rules! pick {
            ($field:ident) => {
                overrides
                    .and_then(|config| config.$field.clone())
                    .or_else(|| model.and_then(|config| config.$field.clone()))
                    .or_else(|| defaults.and_then(|config| config.$field.clone()))
            };
        }

        Self {
            strategy: pick!(strategy),
            mode: pick!(mode),
            draft_model: pick!(draft_model),
            draft_hf_repo: pick!(draft_hf_repo),
            draft_hf_file: pick!(draft_hf_file),
            draft_selection_policy: pick!(draft_selection_policy),
            pairing_fault: pick!(pairing_fault),
            draft_max_tokens: pick!(draft_max_tokens),
            draft_min_tokens: pick!(draft_min_tokens),
            draft_acceptance_threshold: pick!(draft_acceptance_threshold),
            draft_split_probability: pick!(draft_split_probability),
            draft_gpu_layers: pick!(draft_gpu_layers),
            draft_device: pick!(draft_device),
            draft_threads: pick!(draft_threads),
            draft_cache_type_k: pick!(draft_cache_type_k),
            draft_cache_type_v: pick!(draft_cache_type_v),
            ngram_min: pick!(ngram_min),
            ngram_max: pick!(ngram_max),
            ngram_max_proposal_tokens: pick!(ngram_max_proposal_tokens),
            ngram_proposer: pick!(ngram_proposer),
            extension_max_tokens: pick!(extension_max_tokens),
            native_mtp_reject_cooldown_tokens: pick!(native_mtp_reject_cooldown_tokens),
            native_mtp_suppress_cooldown_drafts: pick!(native_mtp_suppress_cooldown_drafts),
            native_mtp_suppress_cooldown_draft_limit: pick!(
                native_mtp_suppress_cooldown_draft_limit
            ),
            verify_window_min_tokens: pick!(verify_window_min_tokens),
            verify_window_max_tokens: pick!(verify_window_max_tokens),
            verify_window_pipeline_depth: pick!(verify_window_pipeline_depth),
            spec_default: pick!(spec_default),
            legacy_draft_model_path_used: overrides
                .filter(|config| config.draft_model.is_some())
                .or_else(|| model.filter(|config| config.draft_model.is_some()))
                .or_else(|| defaults.filter(|config| config.draft_model.is_some()))
                .is_some_and(|config| config.legacy_draft_model_path_used),
        }
    }
}

/// Raw deserialization helper that accepts both `draft_model` and the legacy
/// `draft_model_path` key. The public `SpeculativeConfig` is constructed from
/// this after detecting which key was used.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SpeculativeConfigRaw {
    #[serde(default)]
    strategy: Option<String>,
    #[serde(default)]
    mode: Option<String>,
    #[serde(default)]
    draft_model: Option<String>,
    #[serde(default)]
    draft_model_path: Option<String>,
    #[serde(default)]
    draft_hf_repo: Option<String>,
    #[serde(default)]
    draft_hf_file: Option<String>,
    #[serde(default)]
    draft_selection_policy: Option<String>,
    #[serde(default)]
    pairing_fault: Option<String>,
    #[serde(default)]
    draft_max_tokens: Option<u32>,
    #[serde(default)]
    draft_min_tokens: Option<u32>,
    #[serde(default)]
    draft_acceptance_threshold: Option<f64>,
    #[serde(default)]
    draft_split_probability: Option<f64>,
    #[serde(default)]
    draft_gpu_layers: Option<i32>,
    #[serde(default)]
    draft_device: Option<String>,
    #[serde(default)]
    draft_threads: Option<usize>,
    #[serde(default)]
    draft_cache_type_k: Option<String>,
    #[serde(default)]
    draft_cache_type_v: Option<String>,
    #[serde(default)]
    ngram_min: Option<u32>,
    #[serde(default)]
    ngram_max: Option<u32>,
    #[serde(default)]
    ngram_max_proposal_tokens: Option<u32>,
    #[serde(default)]
    ngram_proposer: Option<String>,
    #[serde(default)]
    extension_max_tokens: Option<u32>,
    #[serde(default)]
    native_mtp_reject_cooldown_tokens: Option<u32>,
    #[serde(default)]
    native_mtp_suppress_cooldown_drafts: Option<bool>,
    #[serde(default)]
    native_mtp_suppress_cooldown_draft_limit: Option<u32>,
    #[serde(default)]
    verify_window_min_tokens: Option<u32>,
    #[serde(default)]
    verify_window_max_tokens: Option<u32>,
    #[serde(default)]
    verify_window_pipeline_depth: Option<u32>,
    #[serde(default)]
    spec_default: Option<BoolOrAuto>,
}

impl<'de> Deserialize<'de> for SpeculativeConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = SpeculativeConfigRaw::deserialize(deserializer)?;
        let legacy_used = raw.draft_model_path.is_some();
        if raw.draft_model.is_some() && raw.draft_model_path.is_some() {
            return Err(serde::de::Error::custom(
                "speculative config cannot set both `draft_model` and the legacy `draft_model_path`; \
                 use `draft_model` only",
            ));
        }
        Ok(SpeculativeConfig {
            strategy: raw.strategy,
            mode: raw.mode,
            draft_model: raw.draft_model.or(raw.draft_model_path),
            draft_hf_repo: raw.draft_hf_repo,
            draft_hf_file: raw.draft_hf_file,
            draft_selection_policy: raw.draft_selection_policy,
            pairing_fault: raw.pairing_fault,
            draft_max_tokens: raw.draft_max_tokens,
            draft_min_tokens: raw.draft_min_tokens,
            draft_acceptance_threshold: raw.draft_acceptance_threshold,
            draft_split_probability: raw.draft_split_probability,
            draft_gpu_layers: raw.draft_gpu_layers,
            draft_device: raw.draft_device,
            draft_threads: raw.draft_threads,
            draft_cache_type_k: raw.draft_cache_type_k,
            draft_cache_type_v: raw.draft_cache_type_v,
            ngram_min: raw.ngram_min,
            ngram_max: raw.ngram_max,
            ngram_max_proposal_tokens: raw.ngram_max_proposal_tokens,
            ngram_proposer: raw.ngram_proposer,
            extension_max_tokens: raw.extension_max_tokens,
            native_mtp_reject_cooldown_tokens: raw.native_mtp_reject_cooldown_tokens,
            native_mtp_suppress_cooldown_drafts: raw.native_mtp_suppress_cooldown_drafts,
            native_mtp_suppress_cooldown_draft_limit: raw.native_mtp_suppress_cooldown_draft_limit,
            verify_window_min_tokens: raw.verify_window_min_tokens,
            verify_window_max_tokens: raw.verify_window_max_tokens,
            verify_window_pipeline_depth: raw.verify_window_pipeline_depth,
            spec_default: raw.spec_default,
            legacy_draft_model_path_used: legacy_used,
        })
    }
}

impl Serialize for SpeculativeConfig {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeMap;

        let mut map = serializer.serialize_map(Some(32))?;
        map.serialize_entry("strategy", &self.strategy)?;
        map.serialize_entry("mode", &self.mode)?;
        if self.legacy_draft_model_path_used {
            if let Some(ref v) = self.draft_model {
                map.serialize_entry("draft_model_path", v)?;
            }
        } else if let Some(ref v) = self.draft_model {
            map.serialize_entry("draft_model", v)?;
        }
        map.serialize_entry("draft_hf_repo", &self.draft_hf_repo)?;
        map.serialize_entry("draft_hf_file", &self.draft_hf_file)?;
        map.serialize_entry("draft_selection_policy", &self.draft_selection_policy)?;
        map.serialize_entry("pairing_fault", &self.pairing_fault)?;
        map.serialize_entry("draft_max_tokens", &self.draft_max_tokens)?;
        map.serialize_entry("draft_min_tokens", &self.draft_min_tokens)?;
        map.serialize_entry(
            "draft_acceptance_threshold",
            &self.draft_acceptance_threshold,
        )?;
        map.serialize_entry("draft_split_probability", &self.draft_split_probability)?;
        map.serialize_entry("draft_gpu_layers", &self.draft_gpu_layers)?;
        map.serialize_entry("draft_device", &self.draft_device)?;
        map.serialize_entry("draft_threads", &self.draft_threads)?;
        map.serialize_entry("draft_cache_type_k", &self.draft_cache_type_k)?;
        map.serialize_entry("draft_cache_type_v", &self.draft_cache_type_v)?;
        map.serialize_entry("ngram_min", &self.ngram_min)?;
        map.serialize_entry("ngram_max", &self.ngram_max)?;
        map.serialize_entry("ngram_max_proposal_tokens", &self.ngram_max_proposal_tokens)?;
        map.serialize_entry("ngram_proposer", &self.ngram_proposer)?;
        map.serialize_entry("extension_max_tokens", &self.extension_max_tokens)?;
        map.serialize_entry(
            "native_mtp_reject_cooldown_tokens",
            &self.native_mtp_reject_cooldown_tokens,
        )?;
        map.serialize_entry(
            "native_mtp_suppress_cooldown_drafts",
            &self.native_mtp_suppress_cooldown_drafts,
        )?;
        map.serialize_entry(
            "native_mtp_suppress_cooldown_draft_limit",
            &self.native_mtp_suppress_cooldown_draft_limit,
        )?;
        map.serialize_entry("verify_window_min_tokens", &self.verify_window_min_tokens)?;
        map.serialize_entry("verify_window_max_tokens", &self.verify_window_max_tokens)?;
        map.serialize_entry(
            "verify_window_pipeline_depth",
            &self.verify_window_pipeline_depth,
        )?;
        map.serialize_entry("spec_default", &self.spec_default)?;
        map.end()
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RequestDefaultsConfig {
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub stop: Option<StringOrStringList>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub top_k: Option<i64>,
    #[serde(default)]
    pub min_p: Option<f64>,
    #[serde(default)]
    pub typical_p: Option<f64>,
    #[serde(default)]
    pub top_nsigma: Option<f64>,
    #[serde(default)]
    pub dynatemp_range: Option<f64>,
    #[serde(default)]
    pub dynatemp_exponent: Option<f64>,
    #[serde(default)]
    pub repeat_penalty: Option<f64>,
    #[serde(default)]
    pub repeat_last_n: Option<i64>,
    #[serde(default)]
    pub presence_penalty: Option<f64>,
    #[serde(default)]
    pub frequency_penalty: Option<f64>,
    #[serde(default)]
    pub dry: Option<ReservedObjectConfig>,
    #[serde(default)]
    pub xtc: Option<ReservedObjectConfig>,
    #[serde(default)]
    pub adaptive: Option<ReservedObjectConfig>,
    #[serde(default)]
    pub mirostat_mode: Option<IntegerOrString>,
    #[serde(default)]
    pub mirostat_entropy: Option<f64>,
    #[serde(default)]
    pub mirostat_learning_rate: Option<f64>,
    #[serde(default)]
    pub samplers: Option<Vec<String>>,
    #[serde(default)]
    pub sampler_sequence: Option<String>,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub logit_bias: Option<toml::Value>,
    #[serde(default)]
    pub ignore_eos: Option<bool>,
    #[serde(default)]
    pub backend_sampling: Option<toml::Value>,
    #[serde(default)]
    pub reasoning_format: Option<String>,
    #[serde(default)]
    pub reasoning_enabled: Option<ReasoningEnabled>,
    #[serde(default)]
    pub reasoning_budget: Option<ReasoningBudget>,
    #[serde(default)]
    pub chat_template: Option<String>,
    #[serde(default)]
    pub chat_template_file: Option<String>,
    #[serde(default)]
    pub jinja: Option<bool>,
    #[serde(default)]
    pub chat_template_kwargs: Option<toml::Value>,
    #[serde(default)]
    pub skip_chat_parsing: Option<bool>,
    #[serde(default)]
    pub prefill_assistant: Option<toml::Value>,
    #[serde(default)]
    pub system_prompt: Option<String>,
    #[serde(default)]
    pub grammar: Option<toml::Value>,
    #[serde(default)]
    pub json_schema: Option<toml::Value>,
    #[serde(default)]
    pub logprobs: Option<toml::Value>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MultimodalConfig {
    #[serde(default)]
    pub mmproj: Option<String>,
    #[serde(default)]
    pub mmproj_url: Option<String>,
    #[serde(default)]
    pub mmproj_offload: Option<BoolOrAuto>,
    #[serde(default)]
    pub image_min_tokens: Option<u32>,
    #[serde(default)]
    pub image_max_tokens: Option<u32>,
    #[serde(default)]
    pub embeddings: Option<toml::Value>,
    #[serde(default)]
    pub reranking: Option<toml::Value>,
    #[serde(default)]
    pub pooling: Option<toml::Value>,
    #[serde(default)]
    pub vocoder: Option<toml::Value>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AdvancedConfig {
    #[serde(default)]
    pub server: Option<AdvancedServerConfig>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AdvancedServerConfig {
    #[serde(default)]
    pub host: Option<String>,
    #[serde(default)]
    pub port: Option<u16>,
    #[serde(default)]
    pub reuse_port: Option<bool>,
    #[serde(default)]
    pub timeout: Option<u64>,
    #[serde(default)]
    pub metrics: Option<bool>,
    #[serde(default)]
    pub slots: Option<bool>,
    #[serde(default)]
    pub props: Option<bool>,
    #[serde(default)]
    pub alias: Option<String>,
    #[serde(default)]
    pub api_prefix: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum BoolOrAuto {
    Bool(bool),
    String(String),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum BoolOrString {
    Bool(bool),
    String(String),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum IntegerOrString {
    Integer(i64),
    String(String),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum StringOrStringList {
    String(String),
    List(Vec<String>),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum TensorSplitConfig {
    Ratios(Vec<f64>),
    String(String),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum ReasoningEnabled {
    Bool(bool),
    String(String),
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum ReasoningBudget {
    Integer(u32),
    String(String),
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ReservedObjectConfig {}

#[derive(Clone, Debug, Default, Deserialize)]
struct RawMeshConfig {
    #[serde(default)]
    version: Option<u32>,
    #[serde(default)]
    gpu: GpuConfig,
    #[serde(default)]
    mesh_requirements: MeshRequirementsConfig,
    #[serde(default)]
    owner_control: OwnerControlConfig,
    #[serde(default)]
    telemetry: TelemetryConfig,
    #[serde(default)]
    defaults: Option<ModelConfigDefaults>,
    #[serde(default)]
    runtime: RuntimeConfig,
    #[serde(default)]
    models: Vec<ModelConfigEntry>,
    #[serde(rename = "plugin", default)]
    plugins: Vec<PluginConfigEntry>,
    #[serde(flatten, default)]
    extra: BTreeMap<String, toml::Value>,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct RawModelConfigDefaults {
    #[serde(default)]
    model_fit: Option<ModelFitConfig>,
    #[serde(default)]
    hardware: Option<HardwareConfig>,
    #[serde(default)]
    throughput: Option<ThroughputConfig>,
    #[serde(default)]
    skippy: Option<SkippyConfig>,
    #[serde(default)]
    speculative: Option<SpeculativeConfig>,
    #[serde(default)]
    request_defaults: Option<RequestDefaultsConfig>,
    #[serde(default)]
    multimodal: Option<MultimodalConfig>,
    #[serde(default)]
    advanced: Option<AdvancedConfig>,
    #[serde(default)]
    mmproj: Option<String>,
    #[serde(default)]
    ctx_size: Option<u32>,
    #[serde(default)]
    gpu_id: Option<String>,
    #[serde(default)]
    parallel: Option<usize>,
    #[serde(default)]
    cache_type_k: Option<String>,
    #[serde(default)]
    cache_type_v: Option<String>,
    #[serde(default)]
    batch: Option<u32>,
    #[serde(default)]
    ubatch: Option<u32>,
    #[serde(default)]
    flash_attention: Option<FlashAttentionType>,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct RawModelConfigEntry {
    model: String,
    #[serde(default)]
    mmproj: Option<String>,
    #[serde(default)]
    ctx_size: Option<u32>,
    #[serde(default)]
    gpu_id: Option<String>,
    #[serde(default)]
    parallel: Option<usize>,
    #[serde(default)]
    cache_type_k: Option<String>,
    #[serde(default)]
    cache_type_v: Option<String>,
    #[serde(default)]
    batch: Option<u32>,
    #[serde(default)]
    ubatch: Option<u32>,
    #[serde(default)]
    flash_attention: Option<FlashAttentionType>,
    #[serde(default)]
    model_fit: Option<ModelFitConfig>,
    #[serde(default)]
    hardware: Option<HardwareConfig>,
    #[serde(default)]
    throughput: Option<ThroughputConfig>,
    #[serde(default)]
    skippy: Option<SkippyConfig>,
    #[serde(default)]
    speculative: Option<SpeculativeConfig>,
    #[serde(default)]
    request_defaults: Option<RequestDefaultsConfig>,
    #[serde(default)]
    multimodal: Option<MultimodalConfig>,
    #[serde(default)]
    advanced: Option<AdvancedConfig>,
}

impl<'de> Deserialize<'de> for MeshConfig {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = RawMeshConfig::deserialize(deserializer)?;
        Ok(Self {
            version: raw.version,
            gpu: raw.gpu,
            mesh_requirements: raw.mesh_requirements,
            owner_control: raw.owner_control,
            telemetry: raw.telemetry,
            defaults: raw.defaults,
            runtime: raw.runtime,
            models: raw.models,
            plugins: raw.plugins,
            extra: raw.extra,
        })
    }
}

impl<'de> Deserialize<'de> for ModelConfigDefaults {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = RawModelConfigDefaults::deserialize(deserializer)?;
        Ok(Self::from_raw(raw))
    }
}

impl<'de> Deserialize<'de> for ModelConfigEntry {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = RawModelConfigEntry::deserialize(deserializer)?;
        Ok(Self::from_raw(raw))
    }
}

impl ModelConfigDefaults {
    fn from_raw(raw: RawModelConfigDefaults) -> Self {
        let model_fit = merge_model_fit(
            raw.model_fit,
            raw.ctx_size,
            raw.cache_type_k,
            raw.cache_type_v,
            raw.batch,
            raw.ubatch,
            raw.flash_attention,
        );
        let hardware = merge_hardware(raw.hardware, raw.gpu_id, None, None);
        let throughput = merge_throughput(raw.throughput, raw.parallel);
        let multimodal = merge_multimodal(raw.multimodal, raw.mmproj);
        Self {
            model_fit,
            hardware,
            throughput,
            skippy: raw.skippy,
            speculative: raw.speculative,
            request_defaults: raw.request_defaults,
            multimodal,
            advanced: raw.advanced,
        }
    }
}

impl ModelConfigEntry {
    fn from_raw(raw: RawModelConfigEntry) -> Self {
        let gpu_id_from_legacy_shim = raw.gpu_id.is_some();
        let model_fit = merge_model_fit(
            raw.model_fit,
            raw.ctx_size,
            raw.cache_type_k.clone(),
            raw.cache_type_v.clone(),
            raw.batch,
            raw.ubatch,
            raw.flash_attention,
        );
        let multimodal = merge_multimodal(raw.multimodal, raw.mmproj.clone());
        let hardware = merge_hardware(
            raw.hardware,
            raw.gpu_id.clone(),
            multimodal.as_ref().and_then(|m| m.mmproj.clone()),
            multimodal.as_ref().and_then(|m| m.mmproj_offload.clone()),
        );
        let throughput = merge_throughput(raw.throughput, raw.parallel);

        Self {
            model: raw.model,
            mmproj: multimodal
                .as_ref()
                .and_then(|config| config.mmproj.clone())
                .or_else(|| hardware.as_ref().and_then(|config| config.mmproj.clone()))
                .or(raw.mmproj),
            ctx_size: model_fit.as_ref().and_then(|config| config.ctx_size),
            gpu_id: hardware
                .as_ref()
                .and_then(|config| config.device.clone())
                .or(raw.gpu_id),
            parallel: throughput.as_ref().and_then(|config| config.parallel),
            cache_type_k: model_fit
                .as_ref()
                .and_then(|config| config.cache_type_k.clone())
                .or(raw.cache_type_k),
            cache_type_v: model_fit
                .as_ref()
                .and_then(|config| config.cache_type_v.clone())
                .or(raw.cache_type_v),
            batch: model_fit.as_ref().and_then(|config| config.batch),
            ubatch: model_fit.as_ref().and_then(|config| config.ubatch),
            flash_attention: model_fit
                .as_ref()
                .and_then(|config| config.flash_attention)
                .or(raw.flash_attention),
            model_fit,
            hardware,
            throughput,
            skippy: raw.skippy,
            speculative: raw.speculative,
            request_defaults: raw.request_defaults,
            multimodal,
            advanced: raw.advanced,
            gpu_id_from_legacy_shim,
        }
    }
}

pub(crate) fn merge_model_fit(
    current: Option<ModelFitConfig>,
    ctx_size: Option<u32>,
    cache_type_k: Option<String>,
    cache_type_v: Option<String>,
    batch: Option<u32>,
    ubatch: Option<u32>,
    flash_attention: Option<FlashAttentionType>,
) -> Option<ModelFitConfig> {
    let mut config = current.unwrap_or_default();
    config.ctx_size = config.ctx_size.or(ctx_size);
    config.cache_type_k = config.cache_type_k.or(cache_type_k);
    config.cache_type_v = config.cache_type_v.or(cache_type_v);
    config.batch = config.batch.or(batch);
    config.ubatch = config.ubatch.or(ubatch);
    config.flash_attention = config.flash_attention.or(flash_attention);
    if is_model_fit_empty(&config) {
        None
    } else {
        Some(config)
    }
}

pub(crate) fn merge_hardware(
    current: Option<HardwareConfig>,
    gpu_id: Option<String>,
    mmproj: Option<String>,
    mmproj_offload: Option<BoolOrAuto>,
) -> Option<HardwareConfig> {
    let mut config = current.unwrap_or_default();
    config.device = config.device.or(gpu_id);
    config.mmproj = config.mmproj.or(mmproj);
    config.mmproj_offload = config.mmproj_offload.or(mmproj_offload);
    if is_hardware_empty(&config) {
        None
    } else {
        Some(config)
    }
}

pub(crate) fn merge_throughput(
    current: Option<ThroughputConfig>,
    parallel: Option<usize>,
) -> Option<ThroughputConfig> {
    let mut config = current.unwrap_or_default();
    config.parallel = config.parallel.or(parallel);
    if is_throughput_empty(&config) {
        None
    } else {
        Some(config)
    }
}

pub(crate) fn merge_multimodal(
    current: Option<MultimodalConfig>,
    mmproj: Option<String>,
) -> Option<MultimodalConfig> {
    let mut config = current.unwrap_or_default();
    config.mmproj = config.mmproj.or(mmproj);
    if is_multimodal_empty(&config) {
        None
    } else {
        Some(config)
    }
}

fn is_model_fit_empty(config: &ModelFitConfig) -> bool {
    config == &ModelFitConfig::default()
}

fn is_hardware_empty(config: &HardwareConfig) -> bool {
    config == &HardwareConfig::default()
}

fn is_throughput_empty(config: &ThroughputConfig) -> bool {
    config == &ThroughputConfig::default()
}

fn is_multimodal_empty(config: &MultimodalConfig) -> bool {
    config == &MultimodalConfig::default()
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct TelemetryConfig {
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub service_name: Option<String>,
    #[serde(default)]
    pub endpoint: Option<String>,
    #[serde(default)]
    pub headers: BTreeMap<String, String>,
    #[serde(default)]
    pub export_interval_secs: Option<u64>,
    #[serde(default)]
    pub queue_size: Option<usize>,
    #[serde(default)]
    pub prompt_shape_metrics: bool,
    #[serde(default)]
    pub metrics: TelemetryMetricsConfig,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct TelemetryMetricsConfig {
    #[serde(default)]
    pub endpoint: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PluginConfigEntry {
    pub name: String,
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub web_ui_enabled: Option<bool>,
    #[serde(default)]
    pub command: Option<String>,
    #[serde(default)]
    pub args: Vec<String>,
    /// Optional URL passed to the plugin as `MESH_LLM_PLUGIN_URL`.
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub settings: BTreeMap<String, toml::Value>,
    #[serde(default, skip_serializing_if = "PluginStartupConfig::is_default")]
    pub startup: PluginStartupConfig,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PluginWebUiPreference {
    None,
    Enabled,
    Disabled,
}

impl PluginWebUiPreference {
    pub const fn resolve(web_ui_enabled: Option<bool>, declares_web_ui: bool) -> Self {
        match (declares_web_ui, web_ui_enabled) {
            (false, _) => Self::None,
            (true, Some(false)) => Self::Disabled,
            (true, Some(true) | None) => Self::Enabled,
        }
    }
}

impl PluginConfigEntry {
    pub const fn web_ui_preference(&self, declares_web_ui: bool) -> PluginWebUiPreference {
        PluginWebUiPreference::resolve(self.web_ui_enabled, declares_web_ui)
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct PluginStartupConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub connect_timeout_secs: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub init_timeout_secs: Option<u64>,
    #[serde(default)]
    pub optional: bool,
    #[serde(default)]
    pub lazy_start: bool,
}

impl PluginStartupConfig {
    pub fn is_default(&self) -> bool {
        self == &Self::default()
    }
}
