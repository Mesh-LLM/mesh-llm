//! Native memory determines whether prefix matches may truncate attention state.
use super::*;
use crate::kv_integration::RadixExactEntry;

impl KvStageIntegration {
    pub(crate) fn retains_recurrent_checkpoints(&self) -> bool {
        self.payload_is_exact_state() || self.resident_requires_checkpoint()
    }

    pub(crate) fn resident_requires_checkpoint(&self) -> bool {
        self.loaded_memory_cache
            .is_some_and(|caps| caps.resident_checkpoint)
    }

    pub(crate) fn peek_native_prefix(
        &self,
        radix: &skippy_cache::UnifiedRadixCache<RadixResidentEntry, RadixExactEntry>,
        namespace: &str,
        tokens: &[i32],
    ) -> Option<skippy_cache::RadixMatch<RadixResidentEntry>> {
        if self.resident_requires_checkpoint() {
            radix.peek_resident_checkpoint(namespace, tokens)
        } else {
            radix.peek_resident(namespace, tokens)
        }
    }

    /// What the cache held for identities whose lookup restored nothing.
    /// Reads without changing recency or references.
    pub(crate) fn diagnose_resident_miss(
        &self,
        identities: &[PrefillKvIdentity],
    ) -> ResidentMissDiagnosis {
        if !self.should_lookup() || self.payload != StagePrefixCachePayload::ResidentKv {
            return ResidentMissDiagnosis {
                reason: ResidentMissReason::LookupOff,
                resident_entries: 0,
                shared_prefix_tokens: 0,
                stored_prefix_tokens: 0,
                checkpoint_prefix_tokens: 0,
            };
        }
        let radix = self
            .radix
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let resident_entries = radix.stats().resident_entries;
        let (shared_prefix_tokens, stored_prefix_tokens) = identities
            .iter()
            .filter_map(|identity| radix.peek_resident(&identity.namespace, &identity.token_ids))
            .map(|shared| (shared.matched_tokens, shared.stored_tokens.len()))
            .max()
            .unwrap_or_default();
        let checkpoint_prefix_tokens = identities
            .iter()
            .filter_map(|identity| {
                self.peek_native_prefix(&radix, &identity.namespace, &identity.token_ids)
            })
            .map(|checkpoint| checkpoint.matched_tokens)
            .max()
            .unwrap_or_default();
        ResidentMissDiagnosis {
            reason: ResidentMissReason::classify(
                resident_entries,
                shared_prefix_tokens,
                checkpoint_prefix_tokens,
                self.meets_shared_prefix_min_tokens(checkpoint_prefix_tokens),
            ),
            resident_entries,
            shared_prefix_tokens,
            stored_prefix_tokens,
            checkpoint_prefix_tokens,
        }
    }
}
