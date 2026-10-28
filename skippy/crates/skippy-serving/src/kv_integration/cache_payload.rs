//! Select a complete representation from the loaded runtime's graph admission.
use skippy_runtime::MemoryCacheCapabilities;

use super::StagePrefixCachePayload;

pub(super) fn effective_cache_payload(
    admitted: MemoryCacheCapabilities,
) -> Option<StagePrefixCachePayload> {
    // A hybrid whose attention cells can be lent serves resident prefixes
    // too; its recurrent half rides on each entry as a snapshot.
    if admitted.resident || admitted.resident_kv_recurrent {
        Some(StagePrefixCachePayload::ResidentKv)
    } else if admitted.kv_recurrent {
        Some(StagePrefixCachePayload::KvRecurrent)
    } else if admitted.full_state {
        Some(StagePrefixCachePayload::FullState)
    } else {
        None
    }
}

/// Durable tiers require an admitted byte exporter; resident handles cannot spill.
pub(super) fn durable_cache_payload(
    admitted: MemoryCacheCapabilities,
) -> Option<StagePrefixCachePayload> {
    effective_cache_payload(MemoryCacheCapabilities {
        resident: false,
        resident_kv_recurrent: false,
        ..admitted
    })
}

/// A resident entry on this stage must carry a recurrent snapshot: the
/// attention cells alone are not a complete state.
pub(super) fn resident_entries_carry_recurrent(admitted: MemoryCacheCapabilities) -> bool {
    admitted.resident_kv_recurrent && !admitted.resident
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_lending_hybrid_is_resident_for_serving_and_exports_for_durability() {
        let lending = MemoryCacheCapabilities {
            resident: false,
            kv_recurrent: true,
            full_state: true,
            resident_kv_recurrent: true,
            kv_cell_access: Default::default(),
        };
        assert_eq!(
            effective_cache_payload(lending),
            Some(StagePrefixCachePayload::ResidentKv)
        );
        assert_eq!(
            durable_cache_payload(lending),
            Some(StagePrefixCachePayload::KvRecurrent)
        );
        assert!(resident_entries_carry_recurrent(lending));
        assert!(!resident_entries_carry_recurrent(MemoryCacheCapabilities {
            resident: true,
            ..lending
        }));
    }

    #[test]
    fn only_admitted_complete_representations_are_selected() {
        for (resident, kv_recurrent, full_state, expected) in [
            (true, true, true, Some(StagePrefixCachePayload::ResidentKv)),
            (
                true,
                false,
                false,
                Some(StagePrefixCachePayload::ResidentKv),
            ),
            (
                false,
                true,
                true,
                Some(StagePrefixCachePayload::KvRecurrent),
            ),
            (
                false,
                true,
                false,
                Some(StagePrefixCachePayload::KvRecurrent),
            ),
            (false, false, true, Some(StagePrefixCachePayload::FullState)),
            (false, false, false, None),
        ] {
            assert_eq!(
                effective_cache_payload(MemoryCacheCapabilities {
                    resident,
                    kv_recurrent,
                    full_state,
                    resident_kv_recurrent: false,
                    kv_cell_access: Default::default(),
                }),
                expected,
            );
            let durable = if kv_recurrent {
                Some(StagePrefixCachePayload::KvRecurrent)
            } else if full_state {
                Some(StagePrefixCachePayload::FullState)
            } else {
                None
            };
            assert_eq!(
                durable_cache_payload(MemoryCacheCapabilities {
                    resident,
                    kv_recurrent,
                    full_state,
                    resident_kv_recurrent: false,
                    kv_cell_access: Default::default(),
                }),
                durable,
            );
        }
    }
}
