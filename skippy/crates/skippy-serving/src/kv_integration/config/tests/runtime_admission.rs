//! Loaded memory, cache policy, and durable-tier admission remain independent.
use super::*;
use skippy_protocol::StageDevice;

#[test]
fn absent_weight_identity_disables_disk_without_disabling_native_cache() {
    let root = tempfile::tempdir().unwrap();
    let manager = L3CacheManager::acquire(root.path(), StoreLimits::new(1_000_000, 0)).unwrap();
    let mut config = enabled_auto_config("mutable-model-id");
    config.source_model_sha256 = None;
    config.manifest_sha256 = None;
    config.package_ref = Some("mutable/package".into());
    let kv = KvStageIntegration::from_loaded_model_with_l3_manager(
        &config,
        Some(ModelStateKind::Dense),
        Some(skippy_runtime::MemoryCacheCapabilities {
            resident: true,
            resident_attention: true,
            resident_checkpoint: false,
            kv_recurrent: true,
            full_state: true,
        }),
        Some(manager),
        None,
    )
    .unwrap()
    .unwrap();
    assert_eq!(kv.payload, StagePrefixCachePayload::ResidentKv);
    assert!(kv.l3.is_none());
    assert!(kv.l2.is_none());
    assert!(kv.durable_payload.is_none());
}

#[test]
fn well_formed_declared_hashes_do_not_authorize_raw_config_disk_reuse() {
    let root = tempfile::tempdir().unwrap();
    let manager = L3CacheManager::acquire(root.path(), StoreLimits::new(1_000_000, 0)).unwrap();
    let mut config = enabled_auto_config("forged-weight-identity");
    config.source_model_sha256 = Some("ab".repeat(32));
    config.manifest_sha256 = Some("cd".repeat(32));
    let kv = KvStageIntegration::from_loaded_model_with_l3_manager(
        &config,
        Some(ModelStateKind::Dense),
        Some(skippy_runtime::MemoryCacheCapabilities {
            resident: true,
            resident_attention: true,
            resident_checkpoint: false,
            kv_recurrent: true,
            full_state: true,
        }),
        Some(manager),
        None,
    )
    .unwrap()
    .unwrap();
    assert!(kv.records_resident_prefixes());
    assert!(kv.l3.is_none());
    assert!(kv.durable_payload.is_none());
}

#[test]
fn actual_weight_content_separates_disk_and_prefix_keys_under_one_claimed_identity() {
    use crate::runtime_state::model_content::VerifiedModelContent;
    let root = tempfile::tempdir().unwrap();
    let model = root.path().join("model.gguf");
    let disk = root.path().join("cache");
    let manager = L3CacheManager::acquire(&disk, StoreLimits::new(1_000_000, 0)).unwrap();
    let mut config = enabled_auto_config("same-model-name");
    config.model_path = Some(model.to_string_lossy().into_owned());
    config.source_model_sha256 = Some("ab".repeat(32));
    config.manifest_sha256 = Some("cd".repeat(32));
    let create = || {
        let content = VerifiedModelContent::observe(&config)
            .unwrap()
            .map(Arc::new);
        KvStageIntegration::from_loaded_model_with_l3(
            &config,
            Some(ModelStateKind::Dense),
            Some(skippy_runtime::MemoryCacheCapabilities {
                resident: true,
                resident_attention: true,
                resident_checkpoint: false,
                kv_recurrent: true,
                full_state: true,
            }),
            content,
            || Ok(Some(manager.clone())),
            None,
        )
        .unwrap()
        .unwrap()
    };
    std::fs::write(&model, b"first numerical weights").unwrap();
    let first = create();
    let same = create();
    assert_eq!(
        first.l3.as_ref().unwrap().state_identity(),
        same.l3.as_ref().unwrap().state_identity()
    );
    std::fs::write(&model, b"other numerical weights").unwrap();
    let changed = create();
    let first_l3 = first.l3.as_ref().unwrap();
    let changed_l3 = changed.l3.as_ref().unwrap();
    assert_eq!(
        first_l3.model_identity(),
        changed_l3.model_identity(),
        "canonical model lifecycle remains shared"
    );
    assert_ne!(first_l3.state_identity(), changed_l3.state_identity());
    let base = skippy_protocol::MessageBase {
        schema_version: skippy_protocol::SCHEMA_VERSION,
        run_id: "run".into(),
        request_id: "request".into(),
        session_id: "session".into(),
        stage_id: "stage-0".into(),
        stage_index: 0,
        topology_id: "topology".into(),
        model_id: Some(config.model_id.clone()),
        tokenizer_id: None,
        chat_template_id: None,
        seq: None,
    };
    let old_prefix = first.prefill_identity(&config, &base, 0, &[1, 2, 3]);
    let changed_prefix = changed.prefill_identity(&config, &base, 0, &[1, 2, 3]);
    assert_ne!(old_prefix.namespace, changed_prefix.namespace);
    assert_ne!(old_prefix.page_id, changed_prefix.page_id);
    assert_ne!(
        old_prefix.identity.prefix_hash,
        changed_prefix.identity.prefix_hash
    );
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
fn injected_manager_is_shared_across_placement_equivalent_stages() {
    let root = std::env::temp_dir()
        .join("skippy-server-l3-manager-tests")
        .join(format!("shared-stages-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let manager = L3CacheManager::acquire(&root, StoreLimits::new(1_000_000, 0)).unwrap();
    let first = enabled_auto_config("future/model");
    let second = StageConfig {
        stage_id: "replica-stage".to_string(),
        stage_index: 7,
        topology_id: "other-topology".to_string(),
        run_id: "other-run".to_string(),
        ..first.clone()
    };

    let first = KvStageIntegration::from_test_model_with_l3_manager(
        &first,
        Some(ModelStateKind::Dense),
        Some(skippy_runtime::MemoryCacheCapabilities {
            resident_attention: false,
            resident_checkpoint: false,
            resident: false,
            kv_recurrent: true,
            full_state: true,
        }),
        Some(manager.clone()),
        None,
    )
    .unwrap()
    .unwrap();
    let second = KvStageIntegration::from_test_model_with_l3_manager(
        &second,
        Some(ModelStateKind::Dense),
        Some(skippy_runtime::MemoryCacheCapabilities {
            resident_attention: false,
            resident_checkpoint: false,
            resident: false,
            kv_recurrent: true,
            full_state: true,
        }),
        Some(manager),
        None,
    )
    .unwrap()
    .unwrap();
    let first_l3 = first.l3.as_ref().unwrap();
    let second_l3 = second.l3.as_ref().unwrap();

    assert!(first_l3.manager().shares_root_with(second_l3.manager()));
    assert_eq!(first_l3.state_identity(), second_l3.state_identity());
    assert_eq!(first_l3.model_identity(), second_l3.model_identity());
    assert!(Arc::ptr_eq(&first.inflight_fills, &second.inflight_fills));
    assert!(Arc::ptr_eq(
        &first.inflight_records,
        &second.inflight_records
    ));
    assert!(first.try_begin_record("shared-page"));
    assert!(
        !second.try_begin_record("shared-page"),
        "placement replicas performed duplicate record work"
    );
    first.finish_record("shared-page");
    assert!(second.try_begin_record("shared-page"));
    second.finish_record("shared-page");
}

#[test]
fn native_resident_cache_does_not_materialize_a_duplicate_disk_payload() {
    let root = std::env::temp_dir()
        .join("skippy-server-l3-manager-tests")
        .join(format!("dense-export-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let manager = L3CacheManager::acquire(&root, StoreLimits::new(1_000_000, 0)).unwrap();
    let config = enabled_auto_config("future/model");

    let kv = KvStageIntegration::from_test_model_with_l3_manager(
        &config,
        Some(ModelStateKind::Dense),
        Some(skippy_runtime::MemoryCacheCapabilities {
            resident_attention: true,
            resident_checkpoint: false,
            resident: true,
            kv_recurrent: true,
            full_state: true,
        }),
        Some(manager),
        None,
    )
    .unwrap()
    .expect("dense disk cache should remain enabled");

    assert_eq!(kv.payload, StagePrefixCachePayload::ResidentKv);
    assert_eq!(
        kv.durable_payload,
        Some(StagePrefixCachePayload::KvRecurrent)
    );
    assert_eq!(
        kv.exact_state_payload(),
        Some(StagePrefixCachePayload::KvRecurrent)
    );
    assert!(!kv.payload_is_exact_state());
    assert!(kv.l3.is_some());

    let mut invalid = kv;
    invalid.payload = StagePrefixCachePayload::Disabled;
    invalid.durable_payload = Some(StagePrefixCachePayload::ResidentKv);
    assert_eq!(invalid.exact_state_payload(), None);
}

#[test]
fn cache_ram_budget_enables_stage_l2_with_disk_authority() {
    let mut config = enabled_auto_config("future/model");
    config
        .kv_cache
        .as_mut()
        .expect("enabled cache config")
        .l2_max_bytes = 64 * 1024 * 1024;
    let root = tempfile::tempdir().unwrap();
    let manager = L3CacheManager::acquire(root.path(), StoreLimits::new(1 << 30, 0)).unwrap();

    let kv = KvStageIntegration::from_test_model_with_l3_manager(
        &config,
        Some(ModelStateKind::Dense),
        Some(skippy_runtime::MemoryCacheCapabilities {
            resident_attention: false,
            resident_checkpoint: false,
            resident: false,
            kv_recurrent: true,
            full_state: true,
        }),
        Some(manager),
        None,
    )
    .unwrap()
    .expect("prefix cache should remain enabled");

    assert!(kv.l2.is_some());
    assert!(kv.l3.is_some());
    let attrs = kv.attrs().into_iter().collect::<BTreeMap<_, _>>();
    assert_eq!(attrs["skippy.kv.l2.enabled"], serde_json::json!(true));
    assert_eq!(
        attrs["skippy.kv.l2.budget_bytes"],
        serde_json::json!(64 * 1024 * 1024u64)
    );
}

#[test]
fn native_checkpoint_ram_budget_works_without_disk() {
    let mut config = enabled_auto_config("future/model");
    config
        .kv_cache
        .as_mut()
        .expect("enabled cache config")
        .l2_max_bytes = 64 * 1024 * 1024;

    let kv = KvStageIntegration::from_test_model_with_l3_manager(
        &config,
        Some(ModelStateKind::Hybrid),
        Some(skippy_runtime::MemoryCacheCapabilities {
            resident_checkpoint: true,
            resident_attention: true,
            resident: true,
            kv_recurrent: true,
            full_state: true,
        }),
        None,
        None,
    )
    .unwrap()
    .expect("L1 prefix cache should remain enabled");

    assert_eq!(kv.resident_checkpoint_max_bytes, 64 * 1024 * 1024);
    assert!(kv.resident_requires_checkpoint());
    assert!(!kv.payload_is_exact_state());
    assert!(kv.l2.is_none());
    assert!(kv.l3.is_none());
}

#[test]
fn parses_cache_mode_aliases() {
    assert_eq!(
        parse_cache_mode("lookup_record"),
        Some(StageKvCacheMode::LookupRecord)
    );
    assert_eq!(
        parse_cache_mode("exact"),
        Some(StageKvCacheMode::LookupRecord)
    );
    assert_eq!(parse_cache_mode("off"), Some(StageKvCacheMode::Disabled));
}

#[test]
fn cachegen_selector_enables_only_qualified_metal_layouts() {
    let mut config = test_config("future/model");
    config.selected_device = Some(StageDevice {
        backend_device: "MTL0".to_string(),
        stable_id: None,
        index: Some(0),
        vram_bytes: None,
    });
    config.cache_type_k = "f32".to_string();
    config.cache_type_v = "f16".to_string();
    assert!(cachegen_serving_enabled(
        &config,
        StageKvCacheCodec::CacheGen
    ));

    config.cache_type_k = "f16".to_string();
    assert!(!cachegen_serving_enabled(
        &config,
        StageKvCacheCodec::CacheGen
    ));
    config.cache_type_k = "f32".to_string();
    config.selected_device.as_mut().unwrap().backend_device = "CUDA0".to_string();
    assert!(!cachegen_serving_enabled(
        &config,
        StageKvCacheCodec::CacheGen
    ));
    config.selected_device = None;
    assert!(!cachegen_serving_enabled(
        &config,
        StageKvCacheCodec::CacheGen
    ));
    assert!(!cachegen_serving_enabled(
        &config,
        StageKvCacheCodec::Native
    ));
}
