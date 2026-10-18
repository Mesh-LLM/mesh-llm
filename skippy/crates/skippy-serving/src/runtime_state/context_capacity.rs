use super::RuntimeState;

impl RuntimeState {
    /// Total native KV pool, including the context reserved for every lane.
    /// Native model loading validates that this product fits its context range.
    /// A modelless runtime with zero context retains its zero-capacity signal.
    pub fn kv_pool_tokens(&self) -> u32 {
        self.ctx_size.saturating_mul(self.lane_count.max(1))
    }

    /// One sequence's context limit, used when bounding its sampling history.
    pub fn context_tokens_per_lane(&self) -> u32 {
        self.ctx_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admission_counts_all_reserved_lanes_without_expanding_sampling_history() {
        for lanes in [1, 2, 4] {
            let runtime = RuntimeState::new_modelless_with_capacity_for_test(lanes, 262_144);
            assert_eq!(runtime.kv_pool_tokens(), 262_144 * lanes);
            assert_eq!(runtime.context_tokens_per_lane(), 262_144);
        }
    }

    #[test]
    fn empty_test_runtime_preserves_zero_capacity() {
        let runtime = RuntimeState::new_modelless_for_test(4);
        assert_eq!(runtime.kv_pool_tokens(), 0);
        assert_eq!(runtime.context_tokens_per_lane(), 0);
    }
}
