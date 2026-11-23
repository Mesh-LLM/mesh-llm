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
                resident_attention: resident,
                resident_checkpoint: false,
                resident,
                kv_recurrent,
                full_state: true,
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
