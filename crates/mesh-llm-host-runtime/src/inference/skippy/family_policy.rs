use std::path::Path;

use skippy_protocol::{StageConfig, StageKvCacheConfig, StageKvCacheMode, StageKvCachePayload};

use crate::models::gguf::{GgufCompactMeta, scan_gguf_compact_meta};

const DEFAULT_PREFIX_CACHE_MIN_TOKENS: u64 = 256;
const DEFAULT_PREFIX_CACHE_MAX_ENTRIES: usize = 512;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct FamilyPolicy {
    pub(crate) default_kv_cache_type: Option<&'static str>,
    pub(crate) prefix_cache: FamilyPrefixCachePolicy,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum FamilyPrefixCachePolicy {
    Auto { min_tokens: u64, max_entries: usize },
}

impl FamilyPolicy {
    pub(crate) fn stage_kv_cache_config_for_stage(
        &self,
        config: &StageConfig,
    ) -> Option<StageKvCacheConfig> {
        self.stage_kv_cache_config_for_stage_with_meta(config, None)
    }

    pub(crate) fn stage_kv_cache_config_for_package(
        &self,
        config: &StageConfig,
        package_dir: &Path,
    ) -> Option<StageKvCacheConfig> {
        let metadata = scan_gguf_compact_meta(&package_dir.join("shared/metadata.gguf"));
        self.stage_kv_cache_config_for_stage_with_meta(config, metadata.as_ref())
    }

    fn stage_kv_cache_config_for_stage_with_meta(
        &self,
        config: &StageConfig,
        package_meta: Option<&GgufCompactMeta>,
    ) -> Option<StageKvCacheConfig> {
        let FamilyPrefixCachePolicy::Auto {
            min_tokens,
            max_entries,
        } = self.prefix_cache;
        let max_bytes = derive_stage_cache_max_bytes(config, package_meta).unwrap_or(0);
        let bounded_entries = derive_max_entries_from_kv_cells(config, min_tokens, max_entries);
        Some(StageKvCacheConfig {
            mode: StageKvCacheMode::LookupRecord,
            // The host only requests automatic selection. The server resolves
            // the concrete payload after llama.cpp has loaded and classified
            // the actual model.
            payload: StageKvCachePayload::Auto,
            max_entries: bounded_entries,
            max_bytes,
            min_tokens,
            shared_prefix_stride_tokens: 128,
            shared_prefix_record_limit: derive_shared_prefix_record_limit(bounded_entries),
        })
    }
}

const MIN_SHARED_PREFIX_RECORD_LIMIT: u64 = 2;
const MAX_SHARED_PREFIX_RECORD_LIMIT: u64 = 6;

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
    policy_default: usize,
) -> usize {
    if min_tokens == 0 {
        return policy_default;
    }
    let n_ctx = u64::from(config.ctx_size.max(1));
    let cache_budget_cells = n_ctx / 2;
    let kv_capped = (cache_budget_cells / min_tokens) as usize;
    kv_capped.clamp(1, policy_default)
}

pub(crate) fn family_policy_for_stage_config(config: &StageConfig) -> FamilyPolicy {
    let metadata = [
        config.materialized_path.as_deref(),
        config.source_model_path.as_deref(),
        config.model_path.as_deref(),
    ]
    .into_iter()
    .flatten()
    .find_map(|path| scan_stage_cache_meta(Path::new(path)));
    generic_model_policy(metadata.as_ref())
}

pub(crate) fn family_policy_for_compact_meta(meta: &GgufCompactMeta) -> FamilyPolicy {
    generic_model_policy(Some(meta))
}

pub(crate) fn family_policy_for_model_path(path: impl AsRef<Path>) -> FamilyPolicy {
    let metadata = scan_stage_cache_meta(path.as_ref());
    generic_model_policy(metadata.as_ref())
}

fn generic_model_policy(meta: Option<&GgufCompactMeta>) -> FamilyPolicy {
    // Inkling requires q4_0 native KV storage. This is read from the GGUF
    // architecture field, never inferred from a repository or filename.
    let default_kv_cache_type = meta
        .is_some_and(|meta| meta.architecture == "inkling")
        .then_some("q4_0");
    FamilyPolicy {
        default_kv_cache_type,
        prefix_cache: FamilyPrefixCachePolicy::Auto {
            min_tokens: DEFAULT_PREFIX_CACHE_MIN_TOKENS,
            max_entries: DEFAULT_PREFIX_CACHE_MAX_ENTRIES,
        },
    }
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

    fn kv_meta(architecture: &str) -> GgufCompactMeta {
        GgufCompactMeta {
            architecture: architecture.to_string(),
            head_count: 8,
            kv_head_count: 4,
            key_length: 64,
            value_length: 64,
            ..Default::default()
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
            let cache = family_policy_for_stage_config(&config)
                .stage_kv_cache_config_for_stage(&config)
                .expect("generic cache policy");
            assert_eq!(cache.payload, StageKvCachePayload::Auto, "{model_id}");
        }
    }

    #[test]
    fn gguf_architecture_only_controls_required_native_kv_storage_type() {
        assert_eq!(
            family_policy_for_compact_meta(&kv_meta("inkling")).default_kv_cache_type,
            Some("q4_0")
        );
        assert_eq!(
            family_policy_for_compact_meta(&kv_meta("nemotron_h_moe")).default_kv_cache_type,
            None
        );
    }

    #[test]
    fn stage_cache_cap_tracks_ctx_layers_and_kv_types() {
        let config = stage_config();
        let bytes = estimate_stage_cache_max_bytes(&config, &kv_meta("future_arch"));
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
