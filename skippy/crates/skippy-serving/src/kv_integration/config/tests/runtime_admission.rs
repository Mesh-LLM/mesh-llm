//! Loaded memory, cache policy, and durable-tier admission remain independent.
use super::*;
use skippy_protocol::StageDevice;

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

    let first = KvStageIntegration::from_loaded_model_with_l3_manager(
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
    let second = KvStageIntegration::from_loaded_model_with_l3_manager(
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

    let kv = KvStageIntegration::from_loaded_model_with_l3_manager(
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

    let kv = KvStageIntegration::from_loaded_model_with_l3_manager(
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

    let kv = KvStageIntegration::from_loaded_model_with_l3_manager(
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
