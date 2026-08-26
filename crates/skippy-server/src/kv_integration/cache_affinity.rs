use skippy_protocol::StageConfig;
use skippy_scheduler::{CacheAffinity, StageCacheAffinity};

use super::{KvStageIntegration, PrefillKvIdentity, StagePrefixCachePayload};

impl KvStageIntegration {
    /// Inspect cache value for scheduling without mutating LRU recency.
    pub fn peek_cache_affinity(
        &self,
        config: &StageConfig,
        identities: &[PrefillKvIdentity],
    ) -> CacheAffinity {
        if !self.should_lookup() {
            return CacheAffinity::default();
        }
        let radix = self
            .radix
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let cache_epoch = radix.epoch();
        let matched_tokens = identities
            .iter()
            .filter_map(|identity| match self.payload {
                StagePrefixCachePayload::ResidentKv => radix
                    .peek_resident(&identity.namespace, &identity.token_ids)
                    .map(|hit| hit.matched_tokens),
                StagePrefixCachePayload::KvRecurrent | StagePrefixCachePayload::FullState => radix
                    .peek_recurrent(&identity.namespace, &identity.token_ids)
                    .map(|hit| hit.matched_tokens),
                StagePrefixCachePayload::Disabled => None,
            })
            .max()
            .unwrap_or(0);
        if matched_tokens == 0 {
            return CacheAffinity::default();
        }
        // Layer count is a deterministic first-order proxy for work saved on
        // heterogeneous stages. The policy type supports measured weights once
        // stage timing calibration is available.
        let prefill_cost_per_token =
            u64::from(config.layer_end.saturating_sub(config.layer_start).max(1));
        CacheAffinity::from_stage(StageCacheAffinity {
            stage_index: config.stage_index,
            matched_tokens,
            prefill_cost_per_token,
            restore_cost: 0,
            cache_epoch,
        })
    }
}
