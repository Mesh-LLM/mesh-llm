//! Select a complete representation from the loaded runtime's graph admission.
use skippy_runtime::MemoryCacheCapabilities;

use super::StagePrefixCachePayload;

pub(super) fn effective_cache_payload(
    admitted: MemoryCacheCapabilities,
) -> Option<StagePrefixCachePayload> {
    if admitted.resident {
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
        ..admitted
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
                }),
                durable,
            );
        }
    }
}
