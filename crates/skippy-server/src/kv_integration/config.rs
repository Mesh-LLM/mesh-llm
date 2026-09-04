use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex},
};

use anyhow::Result;
use mesh_llm_events::OutputEvent;
use skippy_cache::{
    CacheBlobStore, ResidentActivationCache, ResidentCacheConfig, SparseCheckpointPolicy,
    UnifiedRadixCache,
};
use skippy_protocol::{StageConfig, StageKvCacheConfig, StageKvCacheMode, StageKvCachePayload};
use skippy_runtime::ModelStateKind;

use super::{
    EXACT_STATE_RECORD_CAPACITY, KvStageIntegration, PendingExactStateRecord, RadixExactEntry,
    ResidentSequencePool, StageKvMode, StagePrefixCachePayload,
    model_capability::{ModelKvCapability, loaded_model_kv_capability},
    output_tokens::OutputTokenCache,
};

// Recurrent and hybrid payloads share the native n_ctx cell pool across
// sequence lanes, so their exact-state catalog must remain deliberately small.
const RECURRENT_CACHE_MAX_ENTRIES: usize = 16;

impl KvStageIntegration {
    pub fn from_loaded_model(
        config: &StageConfig,
        model_state_kind: Option<ModelStateKind>,
    ) -> Result<Option<Self>> {
        let Some(mut cache_config) = effective_cache_config(config) else {
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
        let model_capability = loaded_model_kv_capability(model_state_kind);
        if let ModelKvCapability::Unknown(reason) = &model_capability {
            emit_cache_disabled_warning(config, reason);
            return Ok(None);
        }
        let payload = effective_cache_payload(cache_config.payload, &model_capability);
        if payload == StagePrefixCachePayload::Disabled {
            return Ok(None);
        }
        if matches!(model_capability, ModelKvCapability::KnownRecurrent)
            && matches!(payload, StagePrefixCachePayload::ResidentKv)
        {
            emit_cache_disabled_warning(
                config,
                "resident KV was requested for a recurrent-state model",
            );
            return Ok(None);
        }
        if matches!(model_capability, ModelKvCapability::KnownDense)
            && matches!(payload, StagePrefixCachePayload::KvRecurrent)
        {
            emit_cache_disabled_warning(
                config,
                "recurrent KV state was requested for a dense model",
            );
            return Ok(None);
        }
        // FullState is architecture-neutral: the native runtime serializes the
        // complete session state for both dense and recurrent model families.
        if matches!(model_capability, ModelKvCapability::KnownRecurrent) {
            cache_config.max_entries = cache_config.max_entries.min(RECURRENT_CACHE_MAX_ENTRIES);
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
        let exact_state_record_worker_healthy = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let worker_exact_state_record_worker_healthy = exact_state_record_worker_healthy.clone();
        let exact_state_record_worker_panics = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let worker_exact_state_record_worker_panics = exact_state_record_worker_panics.clone();
        std::thread::Builder::new()
            .name(format!("skippy-exact-cache-{}", config.stage_id))
            .spawn(move || {
                while let Ok(pending) = exact_state_record_rx.recv() {
                    super::run_exact_state_record_job(
                        &worker_inflight_records,
                        &worker_exact_state_records_dropped,
                        &worker_exact_state_records_pending,
                        &worker_exact_state_record_worker_healthy,
                        &worker_exact_state_record_worker_panics,
                        pending,
                        |pending| {
                            store_exact_radix_record(
                                &worker_radix,
                                &worker_exact_blobs,
                                exact_max_entries,
                                exact_max_bytes,
                                pending,
                            )
                        },
                    );
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
            resident_capacity_reservations: Default::default(),
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
            exact_state_record_worker_healthy,
            exact_state_record_worker_panics,
            cache_healthy: Arc::new(std::sync::atomic::AtomicBool::new(true)),
            output_tokens: Arc::new(Mutex::new(OutputTokenCache::new(exact_max_entries))),
            split_prefill_tokens: Arc::new(Mutex::new(BTreeMap::new())),
        }))
    }

    #[cfg(test)]
    pub(crate) fn from_config(
        config: &StageConfig,
        model_state_kind: ModelStateKind,
    ) -> Result<Option<Self>> {
        Self::from_loaded_model(config, Some(model_state_kind))
    }
}

fn emit_cache_disabled_warning(config: &StageConfig, reason: &str) {
    let _ = mesh_llm_events::emit_event(OutputEvent::Warning {
        message: "Skippy KV cache disabled for this model stage".to_string(),
        context: Some(format!(
            "stage_id={} model_id={} reason={reason}",
            config.stage_id, config.model_id
        )),
    });
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
        )?;
        return Err(error);
    }
    if !released.is_empty() {
        let mut blobs = blobs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for payload in released {
            payload.release_from(&mut blobs)?;
        }
    }
    // Exact snapshots are indivisible, and the default cap is estimated from
    // attention KV metadata that cannot include architecture-specific
    // recurrent state. Treat the byte limit as a soft cap for one entry;
    // otherwise an oversized checkpoint self-evicts immediately and exact
    // prefix caching is permanently disabled for that stage.
    if max_bytes > 0 {
        loop {
            let over_bytes = blobs
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .physical_bytes()
                > max_bytes;
            let multiple_entries = radix
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .stats()
                .recurrent_entries
                > 1;
            if !over_bytes || !multiple_entries {
                break;
            }
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
            )?;
        }
    }
    Ok(())
}

fn effective_cache_payload(
    requested: StageKvCachePayload,
    capability: &ModelKvCapability,
) -> StagePrefixCachePayload {
    match requested {
        StageKvCachePayload::ResidentKv => StagePrefixCachePayload::ResidentKv,
        StageKvCachePayload::KvRecurrent => StagePrefixCachePayload::KvRecurrent,
        StageKvCachePayload::FullState => StagePrefixCachePayload::FullState,
        StageKvCachePayload::Auto => match capability {
            ModelKvCapability::KnownDense => StagePrefixCachePayload::ResidentKv,
            ModelKvCapability::KnownRecurrent => StagePrefixCachePayload::KvRecurrent,
            ModelKvCapability::Unknown(_) => StagePrefixCachePayload::Disabled,
        },
    }
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
    use skippy_protocol::{FlashAttentionType, LoadMode};

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
    fn oversized_exact_payload_retains_one_reusable_entry() {
        let radix = Mutex::new(UnifiedRadixCache::new());
        let blobs = Mutex::new(CacheBlobStore::new(4));

        store_exact_radix_record(
            &radix,
            &blobs,
            4,
            4,
            pending("checkpoint", &[1, 2], b"aaaabbbb"),
        )
        .unwrap();

        assert_eq!(radix.lock().unwrap().stats().recurrent_entries, 1);
        assert_eq!(blobs.lock().unwrap().physical_bytes(), 8);

        store_exact_radix_record(
            &radix,
            &blobs,
            4,
            4,
            pending("replacement", &[1, 3], b"ccccdddd"),
        )
        .unwrap();

        let mut radix = radix.lock().unwrap();
        assert_eq!(radix.stats().recurrent_entries, 1);
        assert!(radix.lookup_recurrent("model", &[1, 2]).is_none());
        assert!(radix.lookup_recurrent("model", &[1, 3]).is_some());
        assert_eq!(blobs.lock().unwrap().physical_bytes(), 8);
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
    fn explicit_cache_payload_is_checked_against_loaded_model_capability() {
        assert_eq!(
            effective_cache_payload(
                StageKvCachePayload::ResidentKv,
                &ModelKvCapability::KnownDense,
            ),
            StagePrefixCachePayload::ResidentKv
        );
        assert_eq!(
            effective_cache_payload(
                StageKvCachePayload::KvRecurrent,
                &ModelKvCapability::KnownRecurrent,
            ),
            StagePrefixCachePayload::KvRecurrent
        );
        assert_eq!(
            effective_cache_payload(
                StageKvCachePayload::FullState,
                &ModelKvCapability::KnownRecurrent,
            ),
            StagePrefixCachePayload::FullState
        );
    }

    #[test]
    fn loaded_hybrid_state_overrides_a_misleading_dense_model_name() {
        let config = enabled_auto_config("nvidia/Nemotron-3-Super-120B-A12B-NVFP4-MTPv2");

        let kv = KvStageIntegration::from_loaded_model(&config, Some(ModelStateKind::Hybrid))
            .unwrap()
            .expect("hybrid loaded model should enable the recurrent cache");

        assert_eq!(kv.payload, StagePrefixCachePayload::KvRecurrent);
    }

    #[test]
    fn loaded_dense_state_selects_resident_kv_independent_of_model_name() {
        let config = enabled_auto_config("future/unknown-architecture-name");

        let kv = KvStageIntegration::from_loaded_model(&config, Some(ModelStateKind::Dense))
            .unwrap()
            .expect("dense loaded model should enable resident KV");

        assert_eq!(kv.payload, StagePrefixCachePayload::ResidentKv);
    }

    #[test]
    fn missing_loaded_descriptor_disables_cache() {
        let config = enabled_auto_config("Qwen/Qwen3-8B");
        assert!(
            KvStageIntegration::from_loaded_model(&config, None)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn explicit_resident_kv_is_rejected_for_loaded_hybrid_model() {
        let mut config = enabled_auto_config("future/model");
        config.kv_cache.as_mut().unwrap().payload = StageKvCachePayload::ResidentKv;

        assert!(
            KvStageIntegration::from_loaded_model(&config, Some(ModelStateKind::Hybrid))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn explicit_recurrent_kv_is_rejected_for_loaded_dense_model() {
        let mut config = enabled_auto_config("future/model");
        config.kv_cache.as_mut().unwrap().payload = StageKvCachePayload::KvRecurrent;

        assert!(
            KvStageIntegration::from_loaded_model(&config, Some(ModelStateKind::Dense))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn explicit_full_state_remains_architecture_neutral() {
        let mut config = enabled_auto_config("future/model");
        config.kv_cache.as_mut().unwrap().payload = StageKvCachePayload::FullState;

        for state_kind in [ModelStateKind::Dense, ModelStateKind::Recurrent] {
            let kv = KvStageIntegration::from_loaded_model(&config, Some(state_kind))
                .unwrap()
                .expect("full-state caching should support every loaded model state kind");
            assert_eq!(kv.payload, StagePrefixCachePayload::FullState);
        }
    }

    #[test]
    fn recurrent_cache_cardinality_is_capped_after_load() {
        let mut config = enabled_auto_config("future/model");
        config.ctx_size = 65_536;
        let kv = KvStageIntegration::from_loaded_model(&config, Some(ModelStateKind::Recurrent))
            .unwrap()
            .expect("recurrent loaded model should enable exact-state caching");

        assert_eq!(kv.payload, StagePrefixCachePayload::KvRecurrent);
        assert_eq!(kv.exact_max_entries, RECURRENT_CACHE_MAX_ENTRIES);
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

    fn enabled_auto_config(model_id: &str) -> StageConfig {
        let mut config = test_config(model_id);
        config.kv_cache = Some(StageKvCacheConfig {
            mode: StageKvCacheMode::LookupRecord,
            payload: StageKvCachePayload::Auto,
            max_entries: 512,
            max_bytes: 0,
            min_tokens: 1,
            shared_prefix_stride_tokens: 1,
            shared_prefix_record_limit: 1,
        });
        config
    }
}
