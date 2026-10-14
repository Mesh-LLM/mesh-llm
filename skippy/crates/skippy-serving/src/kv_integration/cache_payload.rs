//! Select a complete representation from the loaded runtime's graph admission.
use skippy_runtime::MemoryCacheCapabilities;

use super::StagePrefixCachePayload;

pub(super) fn effective_cache_payload(
    admitted: MemoryCacheCapabilities,
) -> StagePrefixCachePayload {
    if admitted.resident {
        StagePrefixCachePayload::ResidentKv
    } else if admitted.kv_recurrent {
        StagePrefixCachePayload::KvRecurrent
    } else {
        StagePrefixCachePayload::FullState
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_admitted_complete_representations_are_selected() {
        for (resident, kv_recurrent, expected) in [
            (true, true, StagePrefixCachePayload::ResidentKv),
            (true, false, StagePrefixCachePayload::ResidentKv),
            (false, true, StagePrefixCachePayload::KvRecurrent),
            (false, false, StagePrefixCachePayload::FullState),
        ] {
            assert_eq!(
                effective_cache_payload(MemoryCacheCapabilities {
                    resident,
                    kv_recurrent
                }),
                expected,
            );
        }
    }
}
