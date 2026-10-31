//! Scheduling follows the loaded state contract, independently of storage tiers.
use super::KvStageIntegration;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum CacheSchedulingPolicy {
    #[default]
    Serialized,
    Cooperative,
}

impl CacheSchedulingPolicy {
    pub(crate) fn cooperative(self) -> bool {
        self == Self::Cooperative
    }
}

impl KvStageIntegration {
    pub(crate) fn scheduling_policy(&self) -> CacheSchedulingPolicy {
        if self.retains_recurrent_checkpoints() {
            CacheSchedulingPolicy::Cooperative
        } else {
            CacheSchedulingPolicy::Serialized
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use skippy_protocol::{StageConfig, StageKvCacheConfig, StageKvCacheMode};
    use skippy_runtime::{MemoryCacheCapabilities, ModelStateKind};

    #[test]
    fn loaded_checkpoint_memory_is_cooperative_without_disk_or_family_labels() {
        for checkpoint in [false, true] {
            let config = StageConfig {
                kv_cache: Some(StageKvCacheConfig {
                    mode: StageKvCacheMode::LookupRecord,
                    min_tokens: 1,
                    max_entries: 8,
                    max_bytes: 0,
                    l2_max_bytes: 1024,
                    exact_max_bytes: None,
                    codec: Default::default(),
                    shared_prefix_stride_tokens: 1,
                    shared_prefix_record_limit: 0,
                }),
                ..StageConfig::default()
            };
            let kv = KvStageIntegration::from_loaded_model_with_l3_manager(
                &config,
                Some(ModelStateKind::Dense),
                Some(MemoryCacheCapabilities {
                    resident: true,
                    resident_checkpoint: checkpoint,
                    resident_attention: true,
                    kv_recurrent: true,
                    full_state: true,
                }),
                None,
                None,
            )
            .unwrap()
            .unwrap();
            assert!(kv.l3.is_none());
            assert!(kv.durable_payload.is_none());
            assert_eq!(kv.scheduling_policy().cooperative(), checkpoint);
        }
    }
}
