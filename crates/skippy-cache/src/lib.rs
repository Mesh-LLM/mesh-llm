pub mod config;
pub mod identity;
pub mod payload;
pub mod radix;
pub mod resident;

pub use config::{ResidentCacheConfig, SparseCheckpointPolicy};
pub use identity::{
    NATIVE_KV_DTYPE, NATIVE_KV_RUNTIME_ABI_VERSION, PrefixIdentity, activation_page_id,
    prefix_hash, prefix_hash_with_namespace, prefix_identity, prefix_identity_with_namespace,
    prefix_namespace_hash,
};
pub use payload::{
    CacheBlobStore, CacheBytes, CacheBytesReconstructStats, CacheDedupeStats, ExactStatePayload,
    ExactStatePayloadKind,
};
pub use radix::{
    RadixEviction, RadixEvictionCandidate, RadixMatch, UnifiedRadixCache, UnifiedRadixCacheStats,
};
pub use resident::{
    ResidentActivationCache, ResidentActivationLookup, ResidentActivationRecordOutcome,
    ResidentActivationStats,
};

/// llama.cpp's hard sequence-id capacity for one context.
pub const LLAMA_MAX_SEQ: i32 = 256;

#[cfg(test)]
mod legacy_prefix_index_absence_tests {
    use std::path::Path;

    #[test]
    fn removed_flat_prefix_indexes_cannot_reappear() {
        let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        for removed in ["exact_state.rs", "resident/prefix.rs"] {
            assert!(
                !source.join(removed).exists(),
                "removed flat prefix index reappeared: {removed}"
            );
        }
    }
}
