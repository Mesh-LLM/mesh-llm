use std::path::Path;

use skippy_protocol::{StageConfig, StageKvCacheConfig, StageKvCacheMode};

use skippy_model_artifact::gguf::{GgufCompactMeta, scan_gguf_compact_meta};

const DEFAULT_PREFIX_CACHE_MIN_TOKENS: u64 = 256;
const DEFAULT_PREFIX_CACHE_MAX_ENTRIES: usize = 512;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FamilyPolicy {
    pub prefix_cache: FamilyPrefixCachePolicy,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FamilyPrefixCachePolicy {
    Auto { min_tokens: u64, max_entries: usize },
}

impl FamilyPolicy {
    pub fn stage_kv_cache_config_for_stage(
        &self,
        config: &StageConfig,
    ) -> Option<StageKvCacheConfig> {
        self.stage_kv_cache_config_for_stage_with_meta(config, None)
    }

    pub fn stage_kv_cache_config_for_package(
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
            exact_max_bytes: None,
            mode: StageKvCacheMode::LookupRecord,
            // The host sets cache budgets. The loaded native graph and memory
            // determine the concrete representation inside the server.
            max_entries: bounded_entries,
            max_bytes,
            l2_max_bytes: 0,
            codec: skippy_protocol::StageKvCacheCodec::Native,
            min_tokens,
            shared_prefix_stride_tokens: 128,
            shared_prefix_record_limit: derive_shared_prefix_record_limit(bounded_entries),
            prefill_admission: Default::default(),
            retention: Default::default(),
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

pub fn family_policy_for_stage_config(config: &StageConfig) -> FamilyPolicy {
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

pub fn family_policy_for_compact_meta(meta: &GgufCompactMeta) -> FamilyPolicy {
    generic_model_policy(Some(meta))
}

pub fn family_policy_for_model_path(path: impl AsRef<Path>) -> FamilyPolicy {
    let metadata = scan_stage_cache_meta(path.as_ref());
    generic_model_policy(metadata.as_ref())
}

fn generic_model_policy(_meta: Option<&GgufCompactMeta>) -> FamilyPolicy {
    FamilyPolicy {
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

/// The exact-state catalog's soft budget for a stage: half of one context
/// window of the attention KV the stage's layers keep, plus the recurrent
/// state of one snapshot per lane. Layers that keep recurrent state instead of
/// KV cost no KV.
fn estimate_stage_cache_max_bytes(config: &StageConfig, meta: &GgufCompactMeta) -> Option<u64> {
    let stage_layers = config.layer_start..config.layer_end;
    if stage_layers.is_empty() {
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
    let layer_kv_bytes_per_token = |heads: u32| -> Option<u64> {
        let key_bytes = dtype_bytes(
            u64::from(key_width) * u64::from(heads),
            &config.cache_type_k,
        )?;
        let value_bytes = dtype_bytes(
            u64::from(value_width) * u64::from(heads),
            &config.cache_type_v,
        )?;
        key_bytes.checked_add(value_bytes)
    };

    let recurrent = attention_free_layers(meta);
    let per_layer_heads = (meta.kv_head_counts.len() == meta.layer_count as usize)
        .then_some(meta.kv_head_counts.as_slice());
    let kv_bytes_per_token = stage_layers
        .clone()
        .filter(|layer| !recurrent.get(*layer as usize).copied().unwrap_or(false))
        .map(|layer| {
            let heads = per_layer_heads
                .and_then(|heads| heads.get(layer as usize).copied())
                .filter(|heads| *heads > 0)
                .unwrap_or(kv_heads);
            layer_kv_bytes_per_token(heads)
        })
        .try_fold(0u64, |total, bytes| total.checked_add(bytes?))?;
    let attention_budget = kv_bytes_per_token.checked_mul(u64::from(config.ctx_size.max(1)))? / 2;

    let plane_bytes = meta.recurrent_plane_bytes_by_layer();
    let recurrent_bytes_per_snapshot = stage_layers
        .filter_map(|layer| plane_bytes.get(layer as usize).copied())
        .fold(0u64, u64::saturating_add);
    let recurrent_allowance =
        recurrent_bytes_per_snapshot.saturating_mul(u64::from(config.lane_count.max(1)));

    let cache_budget_bytes = attention_budget.saturating_add(recurrent_allowance);
    (cache_budget_bytes > 0).then_some(cache_budget_bytes)
}

/// The layers that keep recurrent state and no attention KV. A mask that marks
/// every layer recurrent although the model has attention heads is a guess
/// from missing metadata, so every layer is charged KV instead.
fn attention_free_layers(meta: &GgufCompactMeta) -> Vec<bool> {
    let mask = meta.recurrent_layer_mask();
    let guessed = mask.iter().all(|recurrent| *recurrent)
        && !meta.has_only_recurrent_layers_without_kv_cache();
    if guessed { Vec::new() } else { mask }
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
            assert_eq!(cache.mode, StageKvCacheMode::LookupRecord);
        }
    }

    #[test]
    fn stage_cache_cap_tracks_ctx_layers_and_kv_types() {
        let config = stage_config();
        let bytes = estimate_stage_cache_max_bytes(&config, &kv_meta("future_arch"));
        // Per token/layer: K = 4*64*2, V = 4*64*34/32. Two layers and
        // 1024 context cells, with half reserved for active lanes.
        assert_eq!(bytes, Some((512 + 272) * 2 * 1024 / 2));
    }

    /// Four layers with a full-attention layer every fourth: one attention
    /// layer and three gated-delta layers, as in Qwen3.5.
    fn hybrid_meta() -> GgufCompactMeta {
        GgufCompactMeta {
            layer_count: 4,
            full_attention_interval: 4,
            ssm_conv_kernel: 4,
            ssm_inner_size: 64,
            ssm_state_size: 16,
            ssm_group_count: 2,
            ..kv_meta("hybrid_arch")
        }
    }

    #[test]
    fn hybrid_stages_pay_kv_only_for_attention_layers_and_state_per_lane() {
        let mut config = stage_config();
        config.layer_end = 4;
        config.lane_count = 2;
        let meta = hybrid_meta();
        // One attention layer: K = 4*64*2, V = 4*64*34/32 per token.
        let attention = (512 + 272) * 1024 / 2;
        // Per gated-delta layer, in f32: a 3-wide conv over 64 + 2*2*16
        // channels, and a 16x64 state.
        let plane = (3 * (64 + 2 * 2 * 16) + 16 * 64) * 4;
        assert_eq!(
            estimate_stage_cache_max_bytes(&config, &meta),
            Some(attention + 3 * plane * 2)
        );
    }

    #[test]
    fn a_stage_of_only_recurrent_layers_pays_no_kv() {
        let mut config = stage_config();
        config.layer_start = 0;
        config.layer_end = 3;
        config.lane_count = 1;
        let plane = (3 * (64 + 2 * 2 * 16) + 16 * 64) * 4;
        assert_eq!(
            estimate_stage_cache_max_bytes(&config, &hybrid_meta()),
            Some(3 * plane)
        );
    }

    #[test]
    fn a_recurrent_mask_guessed_from_missing_metadata_still_charges_kv() {
        let mut config = stage_config();
        config.lane_count = 1;
        // State dimensions but no mask, KV head counts or cadence: the layer
        // mask marks every layer recurrent, which the attention heads deny.
        let meta = GgufCompactMeta {
            layer_count: 2,
            full_attention_interval: 0,
            ..hybrid_meta()
        };
        let plane = (3 * (64 + 2 * 2 * 16) + 16 * 64) * 4;
        assert_eq!(
            estimate_stage_cache_max_bytes(&config, &meta),
            Some((512 + 272) * 2 * 1024 / 2 + 2 * plane)
        );
    }

    #[test]
    fn record_limit_stays_bounded() {
        assert_eq!(derive_shared_prefix_record_limit(1), 2);
        assert_eq!(derive_shared_prefix_record_limit(16), 4);
        assert_eq!(derive_shared_prefix_record_limit(512), 6);
    }
}
