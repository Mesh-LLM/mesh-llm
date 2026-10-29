//! Only complete native state representations authorize prefix caching.
use super::*;

#[test]
fn loaded_capabilities_select_complete_state_without_configuration_labels() {
    for (kind, resident, kv_recurrent, expected) in [
        (
            ModelStateKind::Dense,
            true,
            true,
            StagePrefixCachePayload::ResidentKv,
        ),
        (
            ModelStateKind::Recurrent,
            false,
            true,
            StagePrefixCachePayload::KvRecurrent,
        ),
        (
            ModelStateKind::Hybrid,
            false,
            true,
            StagePrefixCachePayload::KvRecurrent,
        ),
        (
            ModelStateKind::Hybrid,
            false,
            false,
            StagePrefixCachePayload::FullState,
        ),
    ] {
        let config = enabled_auto_config("arbitrary-model-name");
        let kv = KvStageIntegration::from_loaded_model(
            &config,
            Some(kind),
            Some(skippy_runtime::MemoryCacheCapabilities {
                resident,
                kv_recurrent,
                full_state: true,
                resident_kv_recurrent: false,
                kv_cell_access: Default::default(),
            }),
            None,
        )
        .unwrap()
        .unwrap();
        assert_eq!(kv.payload, expected);
        assert_eq!(kv.payload_selection_reason, "loaded_graph_and_memory");
        if matches!(kind, ModelStateKind::Recurrent | ModelStateKind::Hybrid) {
            assert_eq!(kv.exact_max_entries, RECURRENT_CACHE_MAX_ENTRIES);
        }
    }
}

#[test]
fn resident_hybrid_keeps_exact_host_fallback_without_disk() {
    let mut config = enabled_auto_config("hybrid-model");
    config.kv_cache.as_mut().unwrap().max_bytes = 8 * 1024 * 1024 * 1024;
    let kv = KvStageIntegration::from_loaded_model(
        &config,
        Some(ModelStateKind::Hybrid),
        Some(skippy_runtime::MemoryCacheCapabilities {
            resident: false,
            kv_recurrent: true,
            full_state: true,
            resident_kv_recurrent: true,
            kv_cell_access: Default::default(),
        }),
        None,
    )
    .unwrap()
    .unwrap();

    assert_eq!(kv.payload, StagePrefixCachePayload::ResidentKv);
    assert_eq!(
        kv.exact_state_payload(),
        Some(StagePrefixCachePayload::KvRecurrent)
    );
    assert_eq!(kv.exact_state_record_queue_limit, 8 * 1024 * 1024 * 1024);
    assert!(kv.l3.is_none());

    let dense = KvStageIntegration::from_loaded_model(
        &config,
        Some(ModelStateKind::Dense),
        Some(skippy_runtime::MemoryCacheCapabilities {
            resident: true,
            kv_recurrent: true,
            full_state: true,
            resident_kv_recurrent: false,
            kv_cell_access: Default::default(),
        }),
        None,
    )
    .unwrap()
    .unwrap();
    assert_eq!(dense.exact_state_payload(), None);
}

#[test]
fn no_loaded_runtime_cannot_authorize_a_cache() {
    let config = enabled_auto_config("arbitrary-model-name");
    assert!(
        KvStageIntegration::from_loaded_model(&config, None, None, None)
            .unwrap()
            .is_none()
    );
}

#[test]
fn loaded_runtime_without_complete_state_export_disables_cache() {
    for kind in [
        ModelStateKind::Dense,
        ModelStateKind::Recurrent,
        ModelStateKind::Hybrid,
    ] {
        let config = enabled_auto_config("arbitrary-model-name");
        assert!(
            KvStageIntegration::from_loaded_model(
                &config,
                Some(kind),
                Some(skippy_runtime::MemoryCacheCapabilities::default()),
                None,
            )
            .unwrap()
            .is_none()
        );
    }
}
