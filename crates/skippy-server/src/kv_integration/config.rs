use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use anyhow::Result;
use skippy_cache::{
    ExactStateCache, PrefixCandidatePolicy, ResidentActivationCache, ResidentCacheConfig,
    ResidentPrefixCache,
};
use skippy_protocol::{
    LoadMode, StageConfig, StageKvCacheConfig, StageKvCacheMode, StageKvCachePayload,
};
use skippy_runtime::ModelInfo;
use skippy_topology::{STAGE_RUNTIME_LLAMA_FAMILY_EXPECTATIONS, infer_family_capability};

use super::{
    EXACT_STATE_RECORD_CAPACITY, ExactStateExtra, KvStageIntegration, PendingExactStateRecord,
    StageKvMode, StagePrefixCachePayload,
};

impl KvStageIntegration {
    pub fn from_config(config: &StageConfig) -> Result<Option<Self>> {
        let Some(cache_config) = effective_cache_config(config) else {
            return Ok(None);
        };
        let mode = match cache_config.mode {
            StageKvCacheMode::Disabled | StageKvCacheMode::Auto => StageKvMode::Disabled,
            StageKvCacheMode::Record => StageKvMode::Record,
            StageKvCacheMode::LookupRecord => StageKvMode::LookupRecord,
        };
        if mode == StageKvMode::Disabled {
            return Ok(None);
        }
        let payload = effective_cache_payload(config, cache_config.payload);
        if payload == StagePrefixCachePayload::Disabled {
            return Ok(None);
        }
        if model_requires_recurrent_state(config)
            && matches!(payload, StagePrefixCachePayload::ResidentKv)
        {
            return Ok(None);
        }
        let mut candidate_policy = PrefixCandidatePolicy::from_cache(&cache_config);
        let resident_config = ResidentCacheConfig::from_stage(config, &cache_config);
        // Bound the record ladder by the same token budget the resident cache
        // enforces, so a single request cannot record more than it can hold.
        candidate_policy.max_resident_tokens_hint = resident_config.max_resident_tokens;
        let exact_states = ExactStateCache::<ExactStateExtra>::new(
            cache_config.max_entries.clamp(1, 512),
            cache_config.max_bytes,
        );
        let exact_states = Arc::new(Mutex::new(exact_states));
        let (exact_state_record_tx, exact_state_record_rx) =
            std::sync::mpsc::sync_channel::<PendingExactStateRecord>(EXACT_STATE_RECORD_CAPACITY);
        let worker_exact_states = exact_states.clone();
        let inflight_records = Arc::new(Mutex::new(BTreeSet::new()));
        let worker_inflight_records = inflight_records.clone();
        let exact_state_records_queued = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let exact_state_records_dropped = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let exact_state_records_pending = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let worker_exact_state_records_pending = exact_state_records_pending.clone();
        std::thread::Builder::new()
            .name(format!("skippy-exact-cache-{}", config.stage_id))
            .spawn(move || {
                while let Ok(pending) = exact_state_record_rx.recv() {
                    let page_id = pending.page_id.clone();
                    worker_exact_states
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .record(
                            pending.page_id,
                            pending.token_count,
                            pending.payload,
                            pending.extra,
                        );
                    worker_inflight_records
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .remove(&page_id);
                    worker_exact_state_records_pending
                        .fetch_sub(1, std::sync::atomic::Ordering::Release);
                }
            })?;
        Ok(Some(Self {
            mode,
            payload,
            correctness_mode: false,
            trust_local_writes: true,
            candidate_policy,
            inflight_records,
            resident: Arc::new(Mutex::new(ResidentPrefixCache::new(resident_config))),
            activations: Arc::new(Mutex::new(ResidentActivationCache::new(resident_config))),
            exact_states,
            exact_state_record_tx,
            exact_state_records_queued,
            exact_state_records_dropped,
            exact_state_records_pending,
            first_tokens: Arc::new(Mutex::new(BTreeMap::new())),
            replay_tokens: Arc::new(Mutex::new(BTreeMap::new())),
            split_prefill_tokens: Arc::new(Mutex::new(BTreeMap::new())),
        }))
    }
}

fn effective_cache_payload(
    config: &StageConfig,
    requested: StageKvCachePayload,
) -> StagePrefixCachePayload {
    if matches!(requested, StageKvCachePayload::Auto) && model_requires_recurrent_state(config) {
        return StagePrefixCachePayload::KvRecurrent;
    }
    match requested {
        StageKvCachePayload::ResidentKv => StagePrefixCachePayload::ResidentKv,
        StageKvCachePayload::KvRecurrent => StagePrefixCachePayload::KvRecurrent,
        StageKvCachePayload::FullState => StagePrefixCachePayload::FullState,
        StageKvCachePayload::Auto => infer_cache_payload(config),
    }
}

pub(crate) fn model_requires_recurrent_state(config: &StageConfig) -> bool {
    let Some(path) = kv_cache_inspection_path(config) else {
        return false;
    };
    let Ok(info) = ModelInfo::open(path) else {
        return false;
    };
    let Ok(tensors) = info.tensors() else {
        return false;
    };
    tensors
        .iter()
        .any(|tensor| tensor_name_requires_recurrent_state(&tensor.name))
}

fn kv_cache_inspection_path(config: &StageConfig) -> Option<PathBuf> {
    let path = config.model_path.as_deref()?;
    match config.load_mode {
        LoadMode::LayerPackage => {
            let package_dir = std::path::Path::new(path);
            layer_package_inspection_path(package_dir, config.layer_start, config.layer_end)
                .or_else(|| layer_package_metadata_path(package_dir))
                .or_else(|| Some(PathBuf::from(path)))
        }
        LoadMode::RuntimeSlice | LoadMode::ArtifactSlice => Some(PathBuf::from(path)),
    }
}

fn layer_package_inspection_path(
    package_dir: &Path,
    layer_start: u32,
    layer_end: u32,
) -> Option<PathBuf> {
    let manifest_path = package_dir.join("model-package.json");
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(manifest_path).ok()?).ok()?;
    let layers = manifest.get("layers")?.as_array()?;
    let selected = layers
        .iter()
        .enumerate()
        .find(|(index, layer)| {
            let layer_index = layer
                .get("layer_index")
                .and_then(|value| value.as_u64())
                .and_then(|value| u32::try_from(value).ok())
                .unwrap_or(*index as u32);
            layer_index >= layer_start && layer_index < layer_end
        })
        .or_else(|| layers.first().map(|layer| (0, layer)))?
        .1;
    let path = selected.get("path")?.as_str()?;
    let path = PathBuf::from(path);
    if path.is_absolute() {
        return None;
    }
    let absolute = package_dir.join(path);
    absolute.is_file().then_some(absolute)
}

fn layer_package_metadata_path(package_dir: &Path) -> Option<PathBuf> {
    let metadata = package_dir.join("shared/metadata.gguf");
    metadata.is_file().then_some(metadata)
}

fn tensor_name_requires_recurrent_state(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.contains(".ssm")
        || lower.contains("ssm_")
        || lower.contains("time_mix")
        || lower.contains("recurrent")
        || lower.contains("rwkv")
}

fn infer_cache_payload(config: &StageConfig) -> StagePrefixCachePayload {
    let identity = format!(
        "{} {}",
        config.model_id,
        config.model_path.as_deref().unwrap_or_default()
    )
    .to_ascii_lowercase();

    // Prefer the shared family capability table over substring guesses. It
    // already knows which families are recurrent or hybrid, so a new release
    // that reuses an existing llama.cpp architecture is classified correctly
    // without adding another literal here.
    if let Some(capability) = infer_family_capability(&identity, 0, 0)
        && let Some(expectation) = STAGE_RUNTIME_LLAMA_FAMILY_EXPECTATIONS
            .iter()
            .find(|expectation| expectation.family_id == capability.family_id)
    {
        return if expectation.recurrent_or_hybrid {
            StagePrefixCachePayload::KvRecurrent
        } else {
            StagePrefixCachePayload::ResidentKv
        };
    }

    if identity.contains("falcon-h1")
        || identity.contains("qwen3next")
        || identity.contains("qwen3-next")
        || identity.contains("kimi-linear")
        || identity.contains("kimi_linear")
    {
        return StagePrefixCachePayload::KvRecurrent;
    }
    if identity.contains("llama")
        || identity.contains("qwen3")
        || identity.contains("deepseek")
        || identity.contains("glm4")
        || identity.contains("glm-4.7")
        || identity.contains("glm47")
        || identity.contains("glm4.7")
        || identity.contains("olmo")
        || identity.contains("gemma")
        || identity.contains("minimax")
    {
        return StagePrefixCachePayload::ResidentKv;
    }
    StagePrefixCachePayload::Disabled
}

/// Either kill-switch variable resolving to `off` disables the cache, so an
/// explicit `SKIPPY_KV_CACHE=on` cannot mask `SKIPPY_PREFIX_CACHE=off`.
fn cache_disabled_by_env(kv_cache: Option<&str>, prefix_cache: Option<&str>) -> bool {
    [kv_cache, prefix_cache]
        .into_iter()
        .flatten()
        .any(|value| parse_cache_mode(value) == Some(StageKvCacheMode::Disabled))
}

fn effective_cache_config(config: &StageConfig) -> Option<StageKvCacheConfig> {
    // An explicit environment kill-switch beats the planned stage config so
    // benches and incident response can turn the prefix cache off without
    // replanning the topology.
    if cache_disabled_by_env(
        std::env::var("SKIPPY_KV_CACHE").ok().as_deref(),
        std::env::var("SKIPPY_PREFIX_CACHE").ok().as_deref(),
    ) {
        return None;
    }
    if let Some(cache) = config.kv_cache.clone() {
        return Some(cache);
    }
    let mode = std::env::var("SKIPPY_KV_CACHE")
        .or_else(|_| std::env::var("SKIPPY_PREFIX_CACHE"))
        .ok()
        .and_then(|value| parse_cache_mode(&value));
    let mode = mode?;
    let max_entries = std::env::var("SKIPPY_KV_CACHE_MAX_ENTRIES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(64);
    let max_bytes = std::env::var("SKIPPY_KV_CACHE_MAX_BYTES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    let min_tokens = std::env::var("SKIPPY_KV_CACHE_MIN_TOKENS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(64);
    let shared_prefix_stride_tokens = std::env::var("SKIPPY_KV_CACHE_SHARED_STRIDE_TOKENS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(128);
    let shared_prefix_record_limit = std::env::var("SKIPPY_KV_CACHE_SHARED_RECORD_LIMIT")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(2);
    let payload = std::env::var("SKIPPY_KV_CACHE_PAYLOAD")
        .ok()
        .and_then(|value| parse_cache_payload(&value))
        .unwrap_or(StageKvCachePayload::Auto);
    Some(StageKvCacheConfig {
        mode,
        payload,
        max_entries,
        max_bytes,
        min_tokens,
        shared_prefix_stride_tokens,
        shared_prefix_record_limit,
    })
}

fn parse_cache_payload(value: &str) -> Option<StageKvCachePayload> {
    match value.trim().to_ascii_lowercase().replace('_', "-").as_str() {
        "" | "auto" => Some(StageKvCachePayload::Auto),
        "resident" | "resident-kv" | "kv" => Some(StageKvCachePayload::ResidentKv),
        "kv-recurrent" | "kvrecurrent" => Some(StageKvCachePayload::KvRecurrent),
        "full" | "full-state" | "fullstate" => Some(StageKvCachePayload::FullState),
        _ => None,
    }
}

fn parse_cache_mode(value: &str) -> Option<StageKvCacheMode> {
    match value.trim().to_ascii_lowercase().replace('_', "-").as_str() {
        "" | "auto" => Some(StageKvCacheMode::Auto),
        "0" | "off" | "false" | "disabled" | "disable" => Some(StageKvCacheMode::Disabled),
        "record" => Some(StageKvCacheMode::Record),
        "1" | "on" | "true" | "lookup-record" | "lookuprecord" | "exact" => {
            Some(StageKvCacheMode::LookupRecord)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use skippy_protocol::FlashAttentionType;

    #[test]
    fn either_cache_kill_switch_disables_the_cache() {
        assert!(!cache_disabled_by_env(None, None));
        assert!(!cache_disabled_by_env(Some("on"), None));
        assert!(cache_disabled_by_env(Some("off"), None));
        assert!(cache_disabled_by_env(None, Some("off")));
        assert!(cache_disabled_by_env(Some("on"), Some("off")));
        assert!(cache_disabled_by_env(Some("off"), Some("on")));
    }

    #[test]
    fn recurrent_tensor_names_require_exact_state_cache() {
        assert!(tensor_name_requires_recurrent_state("blk.0.ssm_a"));
        assert!(tensor_name_requires_recurrent_state(
            "blk.0.ssm_conv1d.weight"
        ));
        assert!(tensor_name_requires_recurrent_state(
            "blk.0.time_mix_k.weight"
        ));
        assert!(tensor_name_requires_recurrent_state(
            "blk.0.rwkv_gate.weight"
        ));
        assert!(!tensor_name_requires_recurrent_state("blk.0.attn_q.weight"));
        assert!(!tensor_name_requires_recurrent_state(
            "blk.0.ffn_down.weight"
        ));
    }

    #[test]
    fn qwen38_identities_infer_recurrent_cache_payload() {
        // Qwen3.8 loads as the qwen35/qwen35moe recurrent pair. These must not
        // fall through to ResidentKv, which is the wrong payload shape for a
        // recurrent family.
        for model_id in [
            "unsloth/Qwen3.8-2.4T-A95B-GGUF:UD-Q1_0",
            "meshllm/Qwen3.8-2.4T-A95B-UD-Q1_0-layers",
            "unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL",
            "qwen38moe",
            "qwen38",
            // Architecture strings, which is what a stage sees when the model
            // id is absent.
            "qwen35moe",
            "qwen35",
        ] {
            assert_eq!(
                infer_cache_payload(&test_config(model_id)),
                StagePrefixCachePayload::KvRecurrent,
                "{model_id} must select KvRecurrent"
            );
        }
    }

    #[test]
    fn qwen3_parameter_sized_identities_stay_resident_kv() {
        // `Qwen3-8B` compacts to `qwen38b`: the digit is a parameter count, not
        // a series number, so it must stay on the non-recurrent Qwen3 path.
        for model_id in [
            "Qwen/Qwen3-8B-GGUF:Q4_K_M",
            "Qwen/Qwen3-5B-GGUF:Q4_K_M",
            "qwen38b",
        ] {
            assert_eq!(
                infer_cache_payload(&test_config(model_id)),
                StagePrefixCachePayload::ResidentKv,
                "{model_id} must stay on ResidentKv"
            );
        }
    }

    #[test]
    fn layer_package_inspection_uses_representative_layer_file() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(dir.path().join("shared")).unwrap();
        fs::create_dir_all(dir.path().join("layers")).unwrap();
        fs::write(dir.path().join("shared/metadata.gguf"), b"metadata").unwrap();
        fs::write(dir.path().join("layers/00000.gguf"), b"layer0").unwrap();
        fs::write(dir.path().join("layers/00001.gguf"), b"layer1").unwrap();
        let manifest = serde_json::json!({
            "layers": [
                { "layer_index": 0, "path": "layers/00000.gguf" },
                { "layer_index": 1, "path": "layers/00001.gguf" }
            ]
        });
        fs::write(
            dir.path().join("model-package.json"),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
        let mut config = test_config("example/recurrent-package");
        config.load_mode = LoadMode::LayerPackage;
        config.model_path = Some(dir.path().to_string_lossy().to_string());
        config.layer_start = 1;
        config.layer_end = 2;

        assert_eq!(
            kv_cache_inspection_path(&config),
            Some(dir.path().join("layers/00001.gguf"))
        );
    }

    #[test]
    fn layer_package_inspection_falls_back_to_shared_metadata() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(dir.path().join("shared")).unwrap();
        fs::write(dir.path().join("shared/metadata.gguf"), b"metadata").unwrap();
        let manifest = serde_json::json!({
            "layers": [
                { "layer_index": 0, "path": "layers/missing.gguf" }
            ]
        });
        fs::write(
            dir.path().join("model-package.json"),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
        let mut config = test_config("example/recurrent-package");
        config.load_mode = LoadMode::LayerPackage;
        config.model_path = Some(dir.path().to_string_lossy().to_string());

        assert_eq!(
            kv_cache_inspection_path(&config),
            Some(dir.path().join("shared/metadata.gguf"))
        );
    }

    #[test]
    fn cache_payload_inference_selects_recurrent_and_dense_families() {
        assert_eq!(
            infer_cache_payload(&test_config("tiiuae/Falcon-H1-0.5B-Instruct-GGUF:Q4_K_M")),
            StagePrefixCachePayload::KvRecurrent
        );
        assert_eq!(
            infer_cache_payload(&test_config(
                "bartowski/moonshotai_Kimi-Linear-48B-A3B-Instruct-GGUF:IQ2_XXS"
            )),
            StagePrefixCachePayload::KvRecurrent
        );
        assert_eq!(
            infer_cache_payload(&test_config(
                "hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF:Q4_K_M"
            )),
            StagePrefixCachePayload::ResidentKv
        );
        assert_eq!(
            infer_cache_payload(&test_config("unsloth/gemma-4-E4B-it-GGUF:Q4_K_M")),
            StagePrefixCachePayload::ResidentKv
        );
        assert_eq!(
            infer_cache_payload(&test_config("unsloth/GLM-4.7-Flash-GGUF:Q4_K_M")),
            StagePrefixCachePayload::ResidentKv
        );
        assert_eq!(
            infer_cache_payload(&test_config("example/unknown-model:Q4_K_M")),
            StagePrefixCachePayload::Disabled
        );
    }

    #[test]
    fn explicit_cache_payload_overrides_identity_inference() {
        let config = test_config("tiiuae/Falcon-H1-0.5B-Instruct-GGUF:Q4_K_M");

        assert_eq!(
            effective_cache_payload(&config, StageKvCachePayload::ResidentKv),
            StagePrefixCachePayload::ResidentKv
        );
        assert_eq!(
            effective_cache_payload(&config, StageKvCachePayload::KvRecurrent),
            StagePrefixCachePayload::KvRecurrent
        );
        assert_eq!(
            effective_cache_payload(&config, StageKvCachePayload::FullState),
            StagePrefixCachePayload::FullState
        );
    }

    #[test]
    fn parses_cache_mode_and_payload_aliases() {
        assert_eq!(
            parse_cache_mode("lookup_record"),
            Some(StageKvCacheMode::LookupRecord)
        );
        assert_eq!(
            parse_cache_mode("exact"),
            Some(StageKvCacheMode::LookupRecord)
        );
        assert_eq!(parse_cache_mode("off"), Some(StageKvCacheMode::Disabled));
        assert_eq!(
            parse_cache_payload("kv_recurrent"),
            Some(StageKvCachePayload::KvRecurrent)
        );
        assert_eq!(
            parse_cache_payload("resident"),
            Some(StageKvCachePayload::ResidentKv)
        );
        assert_eq!(parse_cache_payload("nope"), None);
    }

    fn test_config(model_id: &str) -> StageConfig {
        StageConfig {
            run_id: "test-run".to_string(),
            topology_id: "test-topology".to_string(),
            model_id: model_id.to_string(),
            package_ref: None,
            manifest_sha256: None,
            source_model_path: None,
            source_model_sha256: None,
            source_model_bytes: None,
            materialized_path: None,
            materialized_pinned: false,
            model_path: None,
            projector_path: None,
            stage_id: "stage-0".to_string(),
            stage_index: 0,
            layer_start: 0,
            layer_end: 1,
            ctx_size: 256,
            lane_count: 1,
            n_batch: None,
            n_ubatch: None,
            n_gpu_layers: 0,
            mmap: None,
            mlock: false,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            flash_attn_type: FlashAttentionType::Auto,
            filter_tensors_on_load: false,
            selected_device: None,
            kv_cache: None,
            native_mtp_enabled: true,
            load_mode: LoadMode::RuntimeSlice,
            bind_addr: "127.0.0.1:0".to_string(),
            upstream: None,
            downstream: None,
        }
    }
}
