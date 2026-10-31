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
}
