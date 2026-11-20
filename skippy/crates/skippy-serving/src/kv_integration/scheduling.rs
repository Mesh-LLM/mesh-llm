//! Prompt admission: who prefills a prompt once its prefix is looked up.
//!
//! The policy is configuration (`kv_cache.prefill_admission`), independent of
//! the model's state kind and of storage tiers. `SKIPPY_PREFILL_ADMISSION`
//! overrides it for a bench or an incident without a replan, the same shape
//! as `SKIPPY_KV_CACHE`.
use super::KvStageIntegration;
use skippy_protocol::StagePrefillAdmission;

pub(crate) const PREFILL_ADMISSION_ENV: &str = "SKIPPY_PREFILL_ADMISSION";

// The internal default stays Serialized: it is what a cache operation without a
// configured policy (no kv integration, lazy queue entries) assumes. The configured
// default lives in `StagePrefillAdmission`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum CacheSchedulingPolicy {
    /// The cache operation prefills the whole prompt under the runtime lock.
    #[default]
    Serialized,
    /// The cache operation restores the prefix; the suffix runs as iteration
    /// batches beside other requests' chunks and decode rows.
    Cooperative,
}

impl CacheSchedulingPolicy {
    pub(crate) fn cooperative(self) -> bool {
        self == Self::Cooperative
    }
}

impl From<StagePrefillAdmission> for CacheSchedulingPolicy {
    fn from(admission: StagePrefillAdmission) -> Self {
        match admission {
            StagePrefillAdmission::Serialized => Self::Serialized,
            StagePrefillAdmission::Cooperative => Self::Cooperative,
        }
    }
}

pub(crate) fn prefill_admission_from_value(value: Option<&str>) -> Option<CacheSchedulingPolicy> {
    match value?.trim().to_ascii_lowercase().as_str() {
        "serialized" => Some(CacheSchedulingPolicy::Serialized),
        "cooperative" => Some(CacheSchedulingPolicy::Cooperative),
        _ => None,
    }
}

/// The environment override wins when it parses; otherwise the configured policy.
pub(crate) fn resolve_prefill_admission(
    env_value: Option<&str>,
    configured: StagePrefillAdmission,
) -> CacheSchedulingPolicy {
    prefill_admission_from_value(env_value).unwrap_or_else(|| configured.into())
}

impl KvStageIntegration {
    pub(crate) fn scheduling_policy(&self) -> CacheSchedulingPolicy {
        self.prefill_admission
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use skippy_protocol::{StageConfig, StageKvCacheConfig, StageKvCacheMode};
    use skippy_runtime::{MemoryCacheCapabilities, ModelStateKind};

    fn integration(admission: StagePrefillAdmission, checkpoint: bool) -> KvStageIntegration {
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
                prefill_admission: admission,
                retention: Default::default(),
            }),
            ..StageConfig::default()
        };
        KvStageIntegration::from_test_model_with_l3_manager(
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
        .unwrap()
    }

    #[test]
    fn admission_follows_configuration_not_checkpoint_retention() {
        for checkpoint in [false, true] {
            let cooperative = integration(StagePrefillAdmission::Cooperative, checkpoint);
            assert!(cooperative.l3.is_none());
            assert!(cooperative.durable_payload.is_none());
            assert!(cooperative.scheduling_policy().cooperative());
            assert!(
                !integration(StagePrefillAdmission::Serialized, checkpoint)
                    .scheduling_policy()
                    .cooperative()
            );
        }
    }

    #[test]
    fn cooperative_is_the_default() {
        assert_eq!(
            CacheSchedulingPolicy::from(StagePrefillAdmission::default()),
            CacheSchedulingPolicy::Cooperative
        );
    }

    #[test]
    fn environment_override_parses_or_yields_to_configuration() {
        assert_eq!(
            resolve_prefill_admission(Some(" Serialized "), StagePrefillAdmission::Cooperative),
            CacheSchedulingPolicy::Serialized
        );
        assert_eq!(
            resolve_prefill_admission(Some("cooperative"), StagePrefillAdmission::Serialized),
            CacheSchedulingPolicy::Cooperative
        );
        assert_eq!(
            resolve_prefill_admission(Some("sometimes"), StagePrefillAdmission::Serialized),
            CacheSchedulingPolicy::Serialized
        );
        assert_eq!(
            resolve_prefill_admission(None, StagePrefillAdmission::Serialized),
            CacheSchedulingPolicy::Serialized
        );
        assert_eq!(prefill_admission_from_value(None), None);
    }
}
