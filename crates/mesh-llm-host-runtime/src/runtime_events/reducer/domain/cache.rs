//! `cache` category: a single latest-wins KV-cache pressure/capacity signal.

use mesh_llm_runtime_event_contracts::KvRuntimeStateEventKind;

/// Last known KV-cache capacity/pressure signal: `cache` is a single
/// latest-wins object, not a per-entity collection.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CacheDomainState {
    pub pressure: Option<String>,
    pub capacity_state: Option<String>,
}

pub(super) fn apply_kv_runtime_state(cache: &mut CacheDomainState, kind: KvRuntimeStateEventKind) {
    match kind {
        KvRuntimeStateEventKind::CachePressureCrossed => {
            cache.pressure = Some("pressure".to_string());
        }
        KvRuntimeStateEventKind::CachePressureCleared => {
            cache.pressure = Some("normal".to_string());
        }
        KvRuntimeStateEventKind::ContextCapacityApproachingLimit => {
            cache.capacity_state = Some("approaching_limit".to_string());
        }
        KvRuntimeStateEventKind::ContextExhausted => {
            cache.capacity_state = Some("exhausted".to_string());
        }
        KvRuntimeStateEventKind::CacheReset => {
            cache.pressure = Some("normal".to_string());
            cache.capacity_state = Some("reset".to_string());
        }
        KvRuntimeStateEventKind::KvCacheInitializationStarted
        | KvRuntimeStateEventKind::KvCacheInitializationCompleted
        | KvRuntimeStateEventKind::KvCacheInitializationFailed
        | KvRuntimeStateEventKind::CacheLookupHit
        | KvRuntimeStateEventKind::CacheLookupMiss
        | KvRuntimeStateEventKind::CacheLookupPartial
        | KvRuntimeStateEventKind::CacheLookupError
        | KvRuntimeStateEventKind::PrefixRestored
        | KvRuntimeStateEventKind::CheckpointRestored
        | KvRuntimeStateEventKind::CacheRecordCompleted
        | KvRuntimeStateEventKind::CacheRecordFailed
        | KvRuntimeStateEventKind::CacheTrim
        | KvRuntimeStateEventKind::CacheEviction
        | KvRuntimeStateEventKind::RuntimeStateImportCompleted
        | KvRuntimeStateEventKind::RuntimeStateImportFailed
        | KvRuntimeStateEventKind::RuntimeStateExportCompleted
        | KvRuntimeStateEventKind::RuntimeStateExportFailed => {}
    }
}
