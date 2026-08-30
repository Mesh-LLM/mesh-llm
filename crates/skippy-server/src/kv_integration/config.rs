use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use anyhow::Result;
use skippy_cache::{
    CacheBlobStore, ResidentActivationCache, ResidentCacheConfig, SparseCheckpointPolicy,
    UnifiedRadixCache,
};
use skippy_protocol::{
    LoadMode, StageConfig, StageKvCacheConfig, StageKvCacheMode, StageKvCachePayload,
};
use skippy_runtime::ModelInfo;
use skippy_topology::{STAGE_RUNTIME_LLAMA_FAMILY_EXPECTATIONS, infer_family_capability};

use super::{
    EXACT_STATE_RECORD_CAPACITY, KvStageIntegration, PendingExactStateRecord, RadixExactEntry,
    ResidentSequencePool, StageKvMode, StagePrefixCachePayload,
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
        let mut checkpoint_policy = SparseCheckpointPolicy::from_cache(&cache_config);
        let resident_config = ResidentCacheConfig::from_stage(config, &cache_config);
        if resident_config.max_entries == 0 {
            return Ok(None);
        }
        // Activation checkpoints still use their own sparse policy; serving KV
        // contributes one full token path to the radix tree.
        checkpoint_policy.max_resident_tokens_hint = resident_config.max_resident_tokens;
        let exact_max_entries = cache_config.max_entries.clamp(1, 512);
        let exact_max_bytes = cache_config.max_bytes;
        let radix = Arc::new(Mutex::new(UnifiedRadixCache::new()));
        let exact_blobs = Arc::new(Mutex::new(CacheBlobStore::default()));
        let (exact_state_record_tx, exact_state_record_rx) =
            std::sync::mpsc::sync_channel::<PendingExactStateRecord>(EXACT_STATE_RECORD_CAPACITY);
        let worker_radix = radix.clone();
        let worker_exact_blobs = exact_blobs.clone();
        let inflight_records = Arc::new(Mutex::new(BTreeSet::new()));
        let worker_inflight_records = inflight_records.clone();
        let exact_state_records_queued = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let exact_state_records_dropped = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let worker_exact_state_records_dropped = exact_state_records_dropped.clone();
        let exact_state_records_pending = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let worker_exact_state_records_pending = exact_state_records_pending.clone();
        std::thread::Builder::new()
            .name(format!("skippy-exact-cache-{}", config.stage_id))
            .spawn(move || {
                while let Ok(pending) = exact_state_record_rx.recv() {
                    let page_id = pending.page_id.clone();
                    if store_exact_radix_record(
                        &worker_radix,
                        &worker_exact_blobs,
                        exact_max_entries,
                        exact_max_bytes,
                        pending,
                    )
                    .is_err()
                    {
                        worker_exact_state_records_dropped
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
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
            checkpoint_policy,
            inflight_records,
            resident_config,
            resident_sequences: Arc::new(Mutex::new(ResidentSequencePool::new(
                resident_config.reserved_seq_count,
            ))),
            activations: Arc::new(Mutex::new(ResidentActivationCache::new(resident_config))),
            radix,
            exact_blobs,
            exact_max_entries,
            exact_max_bytes,
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

fn store_exact_radix_record(
    radix: &Mutex<UnifiedRadixCache<super::RadixResidentEntry, RadixExactEntry>>,
    blobs: &Mutex<CacheBlobStore>,
    max_entries: usize,
    max_bytes: u64,
    pending: PendingExactStateRecord,
) -> Result<()> {
    let logical_bytes = pending.payload.byte_len();
    let (payload, _) = pending.payload.dedupe_into(
        &mut blobs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
    );
    // Cloning retains the Arc-backed blocks without changing blob-store
    // accounting, leaving `payload` available to roll that accounting back if
    // the radix rejects the insert.
    let mut released = Vec::new();
    let insert_result = {
        let mut radix = radix
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let insert_result = radix.insert_recurrent(
            pending.namespace,
            &pending.token_ids,
            logical_bytes,
            RadixExactEntry {
                page_id: pending.page_id,
                payload: payload.clone(),
                extra: pending.extra,
            },
        );
        match insert_result {
            Err(error) => Err(error),
            Ok(replaced) => {
                if let Some(replaced) = replaced {
                    released.push(replaced.payload);
                }
                while radix.stats().recurrent_entries > max_entries {
                    let Some(evicted) = radix.evict_lru_recurrent() else {
                        break;
                    };
                    released.push(evicted.value.payload);
                }
                Ok(())
            }
        }
    };
    if let Err(error) = insert_result {
        payload.release_from(
            &mut blobs
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        );
        return Err(error);
    }
    if !released.is_empty() {
        let mut blobs = blobs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for payload in released {
            payload.release_from(&mut blobs);
        }
    }
    while max_bytes > 0
        && blobs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .physical_bytes()
            > max_bytes
    {
        let evicted = radix
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .evict_lru_recurrent();
        let Some(evicted) = evicted else {
            break;
        };
        evicted.value.payload.release_from(
            &mut blobs
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        );
    }
    Ok(())
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
    // A hybrid stage whose first layer file is attention-only still needs
    // KvRecurrent: probe every layer file in [layer_start, layer_end), not
    // just the first one, so interleaved hybrids are never misdetected.
    kv_cache_inspection_paths(config).into_iter().any(|path| {
        let Ok(info) = ModelInfo::open(&path) else {
            return false;
        };
        let Ok(tensors) = info.tensors() else {
            return false;
        };
        tensors
            .iter()
            .any(|tensor| tensor_name_requires_recurrent_state(&tensor.name))
    })
}

fn kv_cache_inspection_paths(config: &StageConfig) -> Vec<PathBuf> {
    let Some(path) = config.model_path.as_deref() else {
        return Vec::new();
    };
    match config.load_mode {
        LoadMode::LayerPackage => {
            let package_dir = std::path::Path::new(path);
            let mut paths =
                layer_package_inspection_paths(package_dir, config.layer_start, config.layer_end);
            if paths.is_empty() {
                if let Some(metadata) = layer_package_metadata_path(package_dir) {
                    paths.push(metadata);
                } else {
                    paths.push(PathBuf::from(path));
                }
            }
            paths
        }
        LoadMode::RuntimeSlice | LoadMode::ArtifactSlice => vec![PathBuf::from(path)],
    }
}

fn layer_package_inspection_paths(
    package_dir: &Path,
    layer_start: u32,
    layer_end: u32,
) -> Vec<PathBuf> {
    let Some(manifest_path) = package_inspection_file(package_dir, Path::new("model-package.json"))
    else {
        return Vec::new();
    };
    let Ok(manifest) =
        serde_json::from_slice::<serde_json::Value>(&fs::read(manifest_path).unwrap_or_default())
    else {
        return Vec::new();
    };
    let Some(layers) = manifest.get("layers").and_then(|value| value.as_array()) else {
        return Vec::new();
    };
    layers
        .iter()
        .enumerate()
        .filter_map(|(index, layer)| {
            let layer_index = layer
                .get("layer_index")
                .and_then(|value| value.as_u64())
                .and_then(|value| u32::try_from(value).ok())
                .unwrap_or(index as u32);
            if layer_index < layer_start || layer_index >= layer_end {
                return None;
            }
            let path = layer.get("path")?.as_str()?;
            package_inspection_file(package_dir, Path::new(path))
        })
        .collect()
}

fn layer_package_metadata_path(package_dir: &Path) -> Option<PathBuf> {
    package_inspection_file(package_dir, Path::new("shared/metadata.gguf"))
}

fn package_inspection_file(package_dir: &Path, relative_path: &Path) -> Option<PathBuf> {
    if relative_path.as_os_str().is_empty()
        || !relative_path
            .components()
            .all(|component| matches!(component, std::path::Component::Normal(_)))
    {
        return None;
    }

    let canonical_package = fs::canonicalize(package_dir).ok()?;
    if !canonical_package.is_dir() {
        return None;
    }
    let canonical_candidate = fs::canonicalize(canonical_package.join(relative_path)).ok()?;
    if !canonical_candidate.is_file() {
        return None;
    }

    let containment_root = hugging_face_repo_cache_root(&canonical_package)
        .unwrap_or_else(|| canonical_package.clone());
    canonical_candidate
        .starts_with(containment_root)
        .then_some(canonical_candidate)
}

fn hugging_face_repo_cache_root(canonical_package: &Path) -> Option<PathBuf> {
    let snapshots_dir = canonical_package.parent()?;
    if snapshots_dir.file_name()?.to_str()? != "snapshots" {
        return None;
    }
    let repo_root = snapshots_dir.parent()?;
    let encoded_repo = repo_root.file_name()?.to_str()?.strip_prefix("models--")?;
    let mut repo_parts = encoded_repo.split("--");
    if !matches!(
        (repo_parts.next(), repo_parts.next(), repo_parts.next()),
        (Some(owner), Some(repo), None) if !owner.is_empty() && !repo.is_empty()
    ) {
        return None;
    }
    fs::canonicalize(repo_root).ok()
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

    fn pending(page_id: &str, tokens: &[i32], bytes: &[u8]) -> PendingExactStateRecord {
        PendingExactStateRecord {
            page_id: page_id.to_string(),
            payload: skippy_cache::ExactStatePayload::full_state(bytes.to_vec()),
            extra: super::super::ExactStateExtra::default(),
            namespace: "model".to_string(),
            token_ids: tokens.to_vec(),
        }
    }

    #[test]
    fn exact_payloads_live_on_radix_nodes_and_release_deduped_blocks_on_eviction() {
        let radix = Mutex::new(UnifiedRadixCache::new());
        let blobs = Mutex::new(CacheBlobStore::new(4));

        store_exact_radix_record(&radix, &blobs, 1, 0, pending("first", &[1, 2], b"aaaabbbb"))
            .unwrap();
        store_exact_radix_record(
            &radix,
            &blobs,
            1,
            0,
            pending("second", &[1, 3], b"aaaacccc"),
        )
        .unwrap();

        let mut radix = radix.lock().unwrap();
        let blobs = blobs.lock().unwrap();
        assert_eq!(radix.stats().recurrent_entries, 1);
        assert_eq!(blobs.physical_bytes(), 8);
        assert!(radix.lookup_recurrent("model", &[1, 2]).is_none());
        assert_eq!(
            radix
                .lookup_recurrent("model", &[1, 3])
                .expect("second exact payload should remain")
                .value
                .page_id,
            "second"
        );
    }

    #[test]
    fn invalid_exact_radix_key_releases_deduped_payload() {
        let radix = Mutex::new(UnifiedRadixCache::new());
        let blobs = Mutex::new(CacheBlobStore::new(4));

        let error =
            store_exact_radix_record(&radix, &blobs, 1, 0, pending("empty", &[], b"aaaabbbb"))
                .unwrap_err();

        assert_eq!(
            error.to_string(),
            "radix cache key must contain at least one token"
        );
        assert_eq!(blobs.lock().unwrap().physical_bytes(), 0);
        assert_eq!(radix.lock().unwrap().stats().recurrent_entries, 0);
    }

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
            kv_cache_inspection_paths(&config),
            vec![fs::canonicalize(dir.path().join("layers/00001.gguf")).unwrap()]
        );
    }

    #[test]
    fn layer_package_inspection_selects_every_in_range_layer_file() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(dir.path().join("layers")).unwrap();
        fs::write(dir.path().join("layers/00000.gguf"), b"layer0").unwrap();
        fs::write(dir.path().join("layers/00001.gguf"), b"layer1").unwrap();
        fs::write(dir.path().join("layers/00002.gguf"), b"layer2").unwrap();
        let manifest = serde_json::json!({
            "layers": [
                { "layer_index": 0, "path": "layers/00000.gguf" },
                { "layer_index": 1, "path": "layers/00001.gguf" },
                { "layer_index": 2, "path": "layers/00002.gguf" }
            ]
        });
        fs::write(
            dir.path().join("model-package.json"),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
        let mut config = test_config("example/hybrid-package");
        config.load_mode = LoadMode::LayerPackage;
        config.model_path = Some(dir.path().to_string_lossy().to_string());
        config.layer_start = 0;
        config.layer_end = 3;

        // a hybrid stage must probe every layer file, not just the first:
        // an attention-first stage still carries recurrent tensors later
        // in the range
        assert_eq!(
            kv_cache_inspection_paths(&config),
            vec![
                fs::canonicalize(dir.path().join("layers/00000.gguf")).unwrap(),
                fs::canonicalize(dir.path().join("layers/00001.gguf")).unwrap(),
                fs::canonicalize(dir.path().join("layers/00002.gguf")).unwrap(),
            ]
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
            kv_cache_inspection_paths(&config),
            vec![fs::canonicalize(dir.path().join("shared/metadata.gguf")).unwrap()]
        );
    }

    #[test]
    fn layer_package_inspection_rejects_parent_directory_escape() {
        let dir = tempfile::tempdir().unwrap();
        let package = dir.path().join("package");
        fs::create_dir_all(&package).unwrap();
        fs::write(dir.path().join("outside.gguf"), b"outside").unwrap();
        fs::write(
            package.join("model-package.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "layers": [{ "layer_index": 0, "path": "../outside.gguf" }]
            }))
            .unwrap(),
        )
        .unwrap();

        assert!(layer_package_inspection_paths(&package, 0, 1).is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn layer_package_inspection_rejects_symlink_outside_local_package() {
        use std::os::unix::fs::symlink;

        let dir = tempfile::tempdir().unwrap();
        let package = dir.path().join("package");
        fs::create_dir_all(package.join("layers")).unwrap();
        fs::write(dir.path().join("outside.gguf"), b"outside").unwrap();
        symlink(
            dir.path().join("outside.gguf"),
            package.join("layers/00000.gguf"),
        )
        .unwrap();
        fs::write(
            package.join("model-package.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "layers": [{ "layer_index": 0, "path": "layers/00000.gguf" }]
            }))
            .unwrap(),
        )
        .unwrap();

        assert!(layer_package_inspection_paths(&package, 0, 1).is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn layer_package_inspection_rejects_intermediate_symlink_outside_local_package() {
        use std::os::unix::fs::symlink;

        let dir = tempfile::tempdir().unwrap();
        let package = dir.path().join("package");
        let outside = dir.path().join("outside");
        fs::create_dir_all(&package).unwrap();
        fs::create_dir_all(&outside).unwrap();
        fs::write(outside.join("00000.gguf"), b"outside").unwrap();
        symlink(&outside, package.join("layers")).unwrap();
        fs::write(
            package.join("model-package.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "layers": [{ "layer_index": 0, "path": "layers/00000.gguf" }]
            }))
            .unwrap(),
        )
        .unwrap();

        assert!(layer_package_inspection_paths(&package, 0, 1).is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn layer_package_inspection_accepts_hf_snapshot_blob_symlink() {
        use std::os::unix::fs::symlink;

        let dir = tempfile::tempdir().unwrap();
        let repo = dir.path().join("models--owner--package");
        let package = repo.join("snapshots/revision");
        let blob = repo.join("blobs/layer-blob");
        fs::create_dir_all(package.join("layers")).unwrap();
        fs::create_dir_all(blob.parent().unwrap()).unwrap();
        fs::write(&blob, b"layer").unwrap();
        symlink(
            "../../../blobs/layer-blob",
            package.join("layers/00000.gguf"),
        )
        .unwrap();
        fs::write(
            package.join("model-package.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "layers": [{ "layer_index": 0, "path": "layers/00000.gguf" }]
            }))
            .unwrap(),
        )
        .unwrap();

        assert_eq!(
            layer_package_inspection_paths(&package, 0, 1),
            vec![fs::canonicalize(blob).unwrap()]
        );
    }

    #[cfg(unix)]
    #[test]
    fn layer_package_inspection_rejects_hf_snapshot_symlink_to_sibling_repo() {
        use std::os::unix::fs::symlink;

        let dir = tempfile::tempdir().unwrap();
        let repo = dir.path().join("models--owner--package");
        let sibling_repo = dir.path().join("models--owner--other");
        let package = repo.join("snapshots/revision");
        let sibling_blob = sibling_repo.join("blobs/layer-blob");
        fs::create_dir_all(package.join("layers")).unwrap();
        fs::create_dir_all(sibling_blob.parent().unwrap()).unwrap();
        fs::write(&sibling_blob, b"layer").unwrap();
        symlink(
            "../../../../models--owner--other/blobs/layer-blob",
            package.join("layers/00000.gguf"),
        )
        .unwrap();
        fs::write(
            package.join("model-package.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "layers": [{ "layer_index": 0, "path": "layers/00000.gguf" }]
            }))
            .unwrap(),
        )
        .unwrap();

        assert!(layer_package_inspection_paths(&package, 0, 1).is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn layer_package_inspection_rejects_malformed_hf_repo_root_exception() {
        use std::os::unix::fs::symlink;

        let dir = tempfile::tempdir().unwrap();
        let malformed_repo = dir.path().join("models--owner");
        let package = malformed_repo.join("snapshots/revision");
        let outside_package = malformed_repo.join("private.gguf");
        fs::create_dir_all(package.join("layers")).unwrap();
        fs::write(&outside_package, b"outside package").unwrap();
        symlink("../../../private.gguf", package.join("layers/00000.gguf")).unwrap();
        fs::write(
            package.join("model-package.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "layers": [{ "layer_index": 0, "path": "layers/00000.gguf" }]
            }))
            .unwrap(),
        )
        .unwrap();

        assert!(layer_package_inspection_paths(&package, 0, 1).is_empty());
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
            repack: false,
            op_offload: None,
            no_host_buffer: false,
            check_tensors: false,
            direct_io: false,
            main_gpu: None,
            split_mode: skippy_protocol::SplitMode::Auto,
            cache_type_k: "f16".to_string(),
            cache_type_v: "f16".to_string(),
            flash_attn_type: FlashAttentionType::Auto,
            kv_offload: None,
            kv_unified: None,
            swa_full: None,
            cache_idle_slots: None,
            filter_tensors_on_load: false,
            selected_device: None,
            kv_cache: None,
            native_mtp_enabled: true,
            load_mode: LoadMode::RuntimeSlice,
            bind_addr: "127.0.0.1:0".to_string(),
            upstream: None,
            downstream: None,
            ..StageConfig::default()
        }
    }
}
