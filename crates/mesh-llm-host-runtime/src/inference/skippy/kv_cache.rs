use std::path::Path;

use skippy_protocol::{StageConfig, StageKvCacheConfig, StageKvCacheMode, StageKvCachePayload};

use crate::models::gguf::{GgufCompactMeta, scan_gguf_compact_meta};

const DEFAULT_PREFIX_CACHE_MIN_TOKENS: u64 = 256;
const DEFAULT_PREFIX_CACHE_MAX_ENTRIES: usize = 512;
const MIN_SHARED_PREFIX_RECORD_LIMIT: u64 = 2;
const MAX_SHARED_PREFIX_RECORD_LIMIT: u64 = 6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KvCacheType {
    F16,
    Q8_0,
    Q4_0,
}

impl KvCacheType {
    pub(crate) fn as_config_value(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Q8_0 => "q8_0",
            Self::Q4_0 => "q4_0",
        }
    }

    fn to_gguf(self) -> crate::models::gguf::GgufKvCacheType {
        use crate::models::gguf::GgufKvCacheType;
        match self {
            Self::F16 => GgufKvCacheType::F16,
            Self::Q8_0 => GgufKvCacheType::Q8_0,
            Self::Q4_0 => GgufKvCacheType::Q4_0,
        }
    }

    fn from_gguf(value: crate::models::gguf::GgufKvCacheType) -> Self {
        use crate::models::gguf::GgufKvCacheType;
        match value {
            GgufKvCacheType::F16 => Self::F16,
            GgufKvCacheType::Q8_0 => Self::Q8_0,
            GgufKvCacheType::Q4_0 => Self::Q4_0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct KvCachePolicy {
    pub(crate) k_type: KvCacheType,
    pub(crate) v_type: KvCacheType,
}

impl KvCachePolicy {
    const LARGE_MODEL_MIN_BYTES: u64 = 50 * 1024 * 1024 * 1024;

    /// Default KV cache policy, tiered by model size.
    ///
    /// Models >= 50 GB use Q4_0 K + Q4_0 V to keep KV cache small enough
    /// that unified-memory machines don't thrash.  On a 480B MoE split
    /// across two Apple Silicon nodes the difference between Q8_0 and Q4_0
    /// is the difference between swap-thrashing at 1 tok/s and running at
    /// 20+ tok/s.
    ///
    /// Smaller models use Q8_0 K + Q8_0 V which gives ~2× compression over
    /// f16 with negligible quality loss.
    ///
    /// Users can override via `--cache-type-k` / `--cache-type-v`.
    pub(crate) fn for_model_size(model_bytes: u64) -> Self {
        if model_bytes >= Self::LARGE_MODEL_MIN_BYTES {
            Self {
                k_type: KvCacheType::Q4_0,
                v_type: KvCacheType::Q4_0,
            }
        } else {
            Self {
                k_type: KvCacheType::Q8_0,
                v_type: KvCacheType::Q8_0,
            }
        }
    }

    fn as_gguf_quant(self) -> crate::models::gguf::GgufKvCacheQuant {
        crate::models::gguf::GgufKvCacheQuant::new(self.k_type.to_gguf(), self.v_type.to_gguf())
    }

    /// Downgrade this *default* policy to one the model can actually load.
    ///
    /// The size tiers above choose a quant purely from byte size; they do not
    /// know whether the model satisfies llama.cpp's quantised-KV constraints
    /// (Flash Attention availability, per-head block alignment). Without this
    /// guard, an incompatible model (e.g. Grok, or a head_dim not divisible by
    /// the q8_0/q4_0 block size of 32) fails the context build outright rather
    /// than degrading. When `meta` proves the chosen quant cannot load, fall
    /// back to f16 K/V.
    ///
    /// This is for automatic defaults only — explicit user overrides must
    /// bypass it and fail loudly. With no metadata we cannot prove
    /// incompatibility, so the policy is returned unchanged (documented limit:
    /// a metadata-only guard also cannot see a backend Flash Attention probe
    /// failure).
    pub(crate) fn guarded_for_model(
        self,
        meta: Option<&crate::models::gguf::GgufCompactMeta>,
    ) -> Self {
        let Some(meta) = meta else {
            return self;
        };
        let compatible = meta.compatible_default_kv_cache_quant(self.as_gguf_quant());
        Self {
            k_type: KvCacheType::from_gguf(compatible.k),
            v_type: KvCacheType::from_gguf(compatible.v),
        }
    }

    pub(crate) fn cache_type_k(self) -> &'static str {
        self.k_type.as_config_value()
    }

    pub(crate) fn cache_type_v(self) -> &'static str {
        self.v_type.as_config_value()
    }

    pub(crate) fn label(self) -> String {
        format!(
            "{} K + {} V",
            self.cache_type_k().to_ascii_uppercase(),
            self.cache_type_v().to_ascii_uppercase()
        )
    }
}

pub(crate) fn default_stage_prefix_cache(config: &StageConfig) -> StageKvCacheConfig {
    default_stage_prefix_cache_with_meta(config, None)
}

pub(crate) fn default_stage_prefix_cache_for_package(
    config: &StageConfig,
    package_dir: &Path,
) -> StageKvCacheConfig {
    let metadata = scan_gguf_compact_meta(&package_dir.join("shared/metadata.gguf"));
    default_stage_prefix_cache_with_meta(config, metadata.as_ref())
}

fn default_stage_prefix_cache_with_meta(
    config: &StageConfig,
    package_meta: Option<&GgufCompactMeta>,
) -> StageKvCacheConfig {
    let max_bytes = derive_stage_cache_max_bytes(config, package_meta).unwrap_or(0);
    let max_entries = derive_max_entries_from_kv_cells(
        config,
        DEFAULT_PREFIX_CACHE_MIN_TOKENS,
        DEFAULT_PREFIX_CACHE_MAX_ENTRIES,
    );
    StageKvCacheConfig {
        mode: StageKvCacheMode::LookupRecord,
        // The host requests automatic selection. The server resolves the
        // concrete payload after llama.cpp has loaded and classified the model.
        payload: StageKvCachePayload::Auto,
        max_entries,
        max_bytes,
        min_tokens: DEFAULT_PREFIX_CACHE_MIN_TOKENS,
        shared_prefix_stride_tokens: 128,
        shared_prefix_record_limit: derive_shared_prefix_record_limit(max_entries),
    }
}

/// Returns a native KV storage type that must be selected before model load.
///
/// Inkling requires q4_0 storage. The requirement comes from GGUF metadata,
/// never from a repository name or model filename.
pub(crate) fn required_native_kv_cache_type(meta: &GgufCompactMeta) -> Option<&'static str> {
    (meta.architecture == "inkling").then_some("q4_0")
}

pub(crate) fn required_native_kv_cache_type_for_model_path(
    path: impl AsRef<Path>,
) -> Option<&'static str> {
    scan_stage_cache_meta(path.as_ref())
        .as_ref()
        .and_then(required_native_kv_cache_type)
}

fn derive_shared_prefix_record_limit(max_entries: usize) -> u64 {
    let quarter_of_cache = (max_entries as u64) / 4;
    quarter_of_cache.clamp(
        MIN_SHARED_PREFIX_RECORD_LIMIT,
        MAX_SHARED_PREFIX_RECORD_LIMIT,
    )
}

fn derive_max_entries_from_kv_cells(
    config: &StageConfig,
    min_tokens: u64,
    default_max_entries: usize,
) -> usize {
    if min_tokens == 0 {
        return default_max_entries;
    }
    let n_ctx = u64::from(config.ctx_size.max(1));
    let cache_budget_cells = n_ctx / 2;
    let kv_capped = (cache_budget_cells / min_tokens) as usize;
    kv_capped.clamp(1, default_max_entries)
}

fn derive_stage_cache_max_bytes(
    config: &StageConfig,
    package_meta: Option<&GgufCompactMeta>,
) -> Option<u64> {
    if let Some(max_bytes) =
        package_meta.and_then(|meta| estimate_stage_cache_max_bytes(config, meta))
    {
        return Some(max_bytes);
    }

    [
        config.materialized_path.as_deref(),
        config.source_model_path.as_deref(),
        config.model_path.as_deref(),
    ]
    .into_iter()
    .flatten()
    .find_map(|path| scan_stage_cache_meta(Path::new(path)))
    .and_then(|meta| estimate_stage_cache_max_bytes(config, &meta))
}

fn scan_stage_cache_meta(path: &Path) -> Option<GgufCompactMeta> {
    scan_gguf_compact_meta(path)
        .or_else(|| scan_gguf_compact_meta(&path.join("shared/metadata.gguf")))
}

fn estimate_stage_cache_max_bytes(config: &StageConfig, meta: &GgufCompactMeta) -> Option<u64> {
    let stage_layers = config.layer_end.checked_sub(config.layer_start)?;
    if stage_layers == 0 {
        return None;
    }

    let kv_heads = if meta.kv_head_count > 0 {
        meta.kv_head_count
    } else {
        meta.head_count
    };
    let key_width = if meta.key_length > 0 {
        meta.key_length
    } else if meta.embedding_size > 0 && kv_heads > 0 {
        meta.embedding_size.checked_div(kv_heads)?
    } else {
        return None;
    };
    let value_width = if meta.value_length > 0 {
        meta.value_length
    } else if meta.embedding_size > 0 && kv_heads > 0 {
        meta.embedding_size.checked_div(kv_heads)?
    } else {
        return None;
    };

    let key_elems_per_token = u64::from(key_width).checked_mul(u64::from(kv_heads))?;
    let value_elems_per_token = u64::from(value_width).checked_mul(u64::from(kv_heads))?;
    let key_bytes_per_token = dtype_bytes(key_elems_per_token, &config.cache_type_k)?;
    let value_bytes_per_token = dtype_bytes(value_elems_per_token, &config.cache_type_v)?;
    let bytes_per_token_layer = key_bytes_per_token.checked_add(value_bytes_per_token)?;

    let full_pool_bytes = bytes_per_token_layer
        .checked_mul(u64::from(stage_layers))?
        .checked_mul(u64::from(config.ctx_size.max(1)))?;
    let cache_budget_bytes = full_pool_bytes / 2;
    (cache_budget_bytes > 0).then_some(cache_budget_bytes)
}

fn dtype_bytes(elements: u64, dtype: &str) -> Option<u64> {
    match dtype.trim().to_ascii_lowercase().as_str() {
        "f32" => elements.checked_mul(4),
        "f16" | "bf16" => elements.checked_mul(2),
        "q8" | "q8_0" => ggml_block_bytes(elements, 32, 34),
        "q8_1" => ggml_block_bytes(elements, 32, 36),
        "q4" | "q4_0" | "iq4_nl" => ggml_block_bytes(elements, 32, 18),
        "q4_1" => ggml_block_bytes(elements, 32, 20),
        _ => None,
    }
}

fn ggml_block_bytes(elements: u64, block_size: u64, type_size: u64) -> Option<u64> {
    elements.div_ceil(block_size).checked_mul(type_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_model_uses_q8_0() {
        let policy = KvCachePolicy::for_model_size(10 * 1024 * 1024 * 1024);
        assert_eq!(policy.k_type, KvCacheType::Q8_0);
        assert_eq!(policy.v_type, KvCacheType::Q8_0);
    }

    #[test]
    fn large_model_uses_q4_0() {
        let policy = KvCachePolicy::for_model_size(50 * 1024 * 1024 * 1024);
        assert_eq!(policy.k_type, KvCacheType::Q4_0);
        assert_eq!(policy.v_type, KvCacheType::Q4_0);
    }

    fn meta(architecture: &str, head_dim: u32) -> crate::models::gguf::GgufCompactMeta {
        crate::models::gguf::GgufCompactMeta {
            architecture: architecture.to_string(),
            head_count: 32,
            kv_head_count: 8,
            layer_count: 32,
            key_length: head_dim,
            value_length: head_dim,
            ..Default::default()
        }
    }

    #[test]
    fn guard_keeps_quant_for_block_aligned_model() {
        let policy = KvCachePolicy::for_model_size(10 * 1024 * 1024 * 1024);
        let guarded = policy.guarded_for_model(Some(&meta("qwen3", 128)));
        assert_eq!(guarded, policy);
    }

    #[test]
    fn guard_falls_back_to_f16_for_unaligned_head_dim() {
        let policy = KvCachePolicy::for_model_size(10 * 1024 * 1024 * 1024);
        let guarded = policy.guarded_for_model(Some(&meta("phi2", 80)));
        assert_eq!(guarded.k_type, KvCacheType::F16);
        assert_eq!(guarded.v_type, KvCacheType::F16);
    }

    #[test]
    fn guard_falls_back_to_f16_for_grok() {
        let policy = KvCachePolicy::for_model_size(60 * 1024 * 1024 * 1024);
        let guarded = policy.guarded_for_model(Some(&meta("grok", 128)));
        assert_eq!(guarded.k_type, KvCacheType::F16);
        assert_eq!(guarded.v_type, KvCacheType::F16);
    }

    #[test]
    fn guard_is_noop_without_metadata() {
        let policy = KvCachePolicy::for_model_size(10 * 1024 * 1024 * 1024);
        assert_eq!(policy.guarded_for_model(None), policy);
    }

    fn stage_config() -> StageConfig {
        StageConfig {
            model_id: "misleading/model-name".to_string(),
            layer_start: 0,
            layer_end: 2,
            ctx_size: 1024,
            cache_type_k: "f16".to_string(),
            cache_type_v: "q8_0".to_string(),
            ..StageConfig::default()
        }
    }

    #[test]
    fn every_model_name_requests_runtime_selected_payload() {
        for model_id in [
            "nvidia/Nemotron-3-Super-120B-A12B-NVFP4-MTPv2",
            "Qwen/Qwen3-8B",
            "future/unknown-architecture",
        ] {
            let mut config = stage_config();
            config.model_id = model_id.to_string();
            assert_eq!(
                default_stage_prefix_cache(&config).payload,
                StageKvCachePayload::Auto,
                "{model_id}"
            );
        }
    }

    #[test]
    fn gguf_architecture_only_controls_required_native_kv_storage_type() {
        assert_eq!(
            required_native_kv_cache_type(&meta("inkling", 64)),
            Some("q4_0")
        );
        assert_eq!(
            required_native_kv_cache_type(&meta("nemotron_h_moe", 64)),
            None
        );
    }

    #[test]
    fn stage_cache_cap_tracks_ctx_layers_and_kv_types() {
        let config = stage_config();
        let mut cache_meta = meta("future_arch", 64);
        cache_meta.head_count = 8;
        cache_meta.kv_head_count = 4;
        let bytes = estimate_stage_cache_max_bytes(&config, &cache_meta);
        // Per token/layer: K = 4*64*2, V = 4*64*34/32. Two layers and
        // 1024 context cells, with half reserved for active lanes.
        assert_eq!(bytes, Some((512 + 272) * 2 * 1024 / 2));
    }

    #[test]
    fn record_limit_stays_bounded() {
        assert_eq!(derive_shared_prefix_record_limit(1), 2);
        assert_eq!(derive_shared_prefix_record_limit(16), 4);
        assert_eq!(derive_shared_prefix_record_limit(512), 6);
    }
}
