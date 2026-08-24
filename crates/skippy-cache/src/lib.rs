pub mod config;
pub mod exact_state;
pub mod identity;
pub mod payload;
pub mod resident;

pub use config::{PrefixCandidatePolicy, ResidentCacheConfig};
pub use exact_state::{
    ExactStateCache, ExactStateCacheStats, ExactStateLookup, ExactStateRecordOutcome,
};
pub use identity::{
    NATIVE_KV_DTYPE, NATIVE_KV_RUNTIME_ABI_VERSION, PrefixIdentity, activation_page_id,
    prefix_hash, prefix_hash_with_namespace, prefix_identity, prefix_identity_with_namespace,
};
pub use payload::{
    CacheBlobStore, CacheBytes, CacheBytesReconstructStats, CacheDedupeStats, ExactStatePayload,
    ExactStatePayloadKind,
};
pub use resident::{
    ResidentActivationCache, ResidentActivationLookup, ResidentActivationRecordOutcome,
    ResidentActivationStats, ResidentPrefixAllocation, ResidentPrefixCache,
    ResidentPrefixCacheStats, ResidentPrefixEviction, ResidentPrefixLookup,
};

/// llama.cpp's hard sequence-id capacity for one context.
pub const LLAMA_MAX_SEQ: i32 = 256;
