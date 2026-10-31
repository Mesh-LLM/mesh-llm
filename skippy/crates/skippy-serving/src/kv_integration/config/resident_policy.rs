//! Native radix prefixes own shared KV references and bounded recurrent host state.
use skippy_protocol::StageKvCacheConfig;

/// Zero RAM mirrors means use the native checkpoint default, not disable prefix reuse.
/// Disabling the stage cache itself remains the explicit off switch.
const DEFAULT_CHECKPOINT_BYTES: u64 = 1024 * 1024 * 1024;

pub(super) fn checkpoint_budget(cache: &StageKvCacheConfig) -> u64 {
    if cache.l2_max_bytes > 0 {
        cache.l2_max_bytes
    } else {
        DEFAULT_CHECKPOINT_BYTES
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_ram_budget_is_independent_of_disk_and_attention_estimates() {
        let cache = StageKvCacheConfig {
            max_bytes: 32,
            exact_max_bytes: Some(64),
            l2_max_bytes: 256,
            mode: skippy_protocol::StageKvCacheMode::LookupRecord,
            max_entries: 64,
            codec: Default::default(),
            min_tokens: 64,
            shared_prefix_stride_tokens: 128,
            shared_prefix_record_limit: 2,
        };
        assert_eq!(checkpoint_budget(&cache), 256);
        assert_eq!(
            checkpoint_budget(&StageKvCacheConfig {
                l2_max_bytes: 0,
                ..cache
            }),
            DEFAULT_CHECKPOINT_BYTES
        );
    }
}
