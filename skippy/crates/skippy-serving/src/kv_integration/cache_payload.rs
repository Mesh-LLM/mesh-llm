//! Select cache representations from the loaded native memory adapter contract.
use skippy_protocol::StageKvCachePayload;

use super::{StagePrefixCachePayload, model_capability::ModelKvCapability};

pub(super) fn effective_cache_payload(
    requested: StageKvCachePayload,
    capability: &ModelKvCapability,
    memory_cache: Option<skippy_runtime::MemoryCacheCapabilities>,
) -> StagePrefixCachePayload {
    if matches!(
        (requested, capability),
        (
            StageKvCachePayload::ResidentKv,
            ModelKvCapability::KnownRecurrent
        ) | (
            StageKvCachePayload::KvRecurrent,
            ModelKvCapability::KnownDense
        )
    ) {
        return match requested {
            StageKvCachePayload::ResidentKv => StagePrefixCachePayload::ResidentKv,
            _ => StagePrefixCachePayload::KvRecurrent,
        };
    }
    let supported = memory_cache.unwrap_or_default();
    match requested {
        StageKvCachePayload::FullState => StagePrefixCachePayload::FullState,
        StageKvCachePayload::Auto => match capability {
            ModelKvCapability::KnownDense if supported.resident => {
                StagePrefixCachePayload::ResidentKv
            }
            ModelKvCapability::KnownRecurrent if supported.kv_recurrent => {
                StagePrefixCachePayload::KvRecurrent
            }
            ModelKvCapability::Unknown(_) => StagePrefixCachePayload::Disabled,
            _ => StagePrefixCachePayload::FullState,
        },
        StageKvCachePayload::ResidentKv if supported.resident => {
            StagePrefixCachePayload::ResidentKv
        }
        StageKvCachePayload::KvRecurrent if supported.kv_recurrent => {
            StagePrefixCachePayload::KvRecurrent
        }
        _ => StagePrefixCachePayload::FullState,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unknown_memory_uses_full_state_for_dense_and_hybrid_models() {
        for capability in [
            ModelKvCapability::KnownDense,
            ModelKvCapability::KnownRecurrent,
        ] {
            assert_eq!(
                effective_cache_payload(StageKvCachePayload::Auto, &capability, None),
                StagePrefixCachePayload::FullState
            );
        }
    }

    #[test]
    fn supported_memory_retains_fast_payloads() {
        let supported = Some(skippy_runtime::MemoryCacheCapabilities {
            resident: true,
            kv_recurrent: true,
        });
        assert_eq!(
            effective_cache_payload(
                StageKvCachePayload::Auto,
                &ModelKvCapability::KnownDense,
                supported
            ),
            StagePrefixCachePayload::ResidentKv
        );
        assert_eq!(
            effective_cache_payload(
                StageKvCachePayload::Auto,
                &ModelKvCapability::KnownRecurrent,
                supported
            ),
            StagePrefixCachePayload::KvRecurrent
        );
    }

    #[test]
    fn resident_only_memory_cannot_select_partial_snapshots() {
        let supported = Some(skippy_runtime::MemoryCacheCapabilities {
            resident: true,
            kv_recurrent: false,
        });
        assert_eq!(
            effective_cache_payload(
                StageKvCachePayload::Auto,
                &ModelKvCapability::KnownDense,
                supported
            ),
            StagePrefixCachePayload::ResidentKv
        );
        assert_eq!(
            effective_cache_payload(
                StageKvCachePayload::Auto,
                &ModelKvCapability::KnownRecurrent,
                supported
            ),
            StagePrefixCachePayload::FullState
        );
        assert_eq!(
            effective_cache_payload(
                StageKvCachePayload::KvRecurrent,
                &ModelKvCapability::KnownRecurrent,
                supported
            ),
            StagePrefixCachePayload::FullState
        );
    }
}
