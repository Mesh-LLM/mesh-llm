/// Cache work saved at one split-model stage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageCacheAffinity {
    pub stage_index: u32,
    pub matched_tokens: usize,
    pub prefill_cost_per_token: u64,
    pub restore_cost: u64,
    pub cache_epoch: u64,
}

impl StageCacheAffinity {
    pub fn estimated_saved_cost(&self) -> u64 {
        u64::try_from(self.matched_tokens)
            .unwrap_or(u64::MAX)
            .saturating_mul(self.prefill_cost_per_token)
            .saturating_sub(self.restore_cost)
    }
}

/// Per-stage cache affinity for one waiting request.
///
/// Keeping the stages separate matters for split serving: a downstream stage
/// may have a useful prefix even when stage zero misses.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CacheAffinity {
    pub stages: Vec<StageCacheAffinity>,
}

impl CacheAffinity {
    pub fn from_stage(stage: StageCacheAffinity) -> Self {
        Self {
            stages: vec![stage],
        }
    }

    pub fn estimated_saved_cost(&self) -> u64 {
        self.stages
            .iter()
            .map(StageCacheAffinity::estimated_saved_cost)
            .fold(0u64, u64::saturating_add)
    }

    pub fn matched_tokens(&self) -> usize {
        self.stages
            .iter()
            .map(|stage| stage.matched_tokens)
            .fold(0usize, usize::saturating_add)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CacheAwareCandidate<'a> {
    pub index: usize,
    pub priority: u64,
    pub affinity: &'a CacheAffinity,
    pub enqueued_turn: u64,
    pub order: u64,
}

/// Select the highest-priority, highest-value cache candidate.
///
/// Equal-priority requests gain `aging_cost_per_turn` for every turn they wait,
/// which bounds starvation even when hot-prefix requests keep arriving.
pub fn select_cache_aware_candidate<'a>(
    candidates: impl IntoIterator<Item = CacheAwareCandidate<'a>>,
    current_turn: u64,
    aging_cost_per_turn: u64,
) -> Option<usize> {
    candidates
        .into_iter()
        .max_by(|left, right| {
            let left_age = current_turn.saturating_sub(left.enqueued_turn);
            let right_age = current_turn.saturating_sub(right.enqueued_turn);
            let left_score = left
                .affinity
                .estimated_saved_cost()
                .saturating_add(left_age.saturating_mul(aging_cost_per_turn));
            let right_score = right
                .affinity
                .estimated_saved_cost()
                .saturating_add(right_age.saturating_mul(aging_cost_per_turn));
            left.priority
                .cmp(&right.priority)
                .then_with(|| left_score.cmp(&right_score))
                // Earlier enqueue order wins an exact tie.
                .then_with(|| right.order.cmp(&left.order))
        })
        .map(|candidate| candidate.index)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn affinity(saved_cost: u64) -> CacheAffinity {
        CacheAffinity::from_stage(StageCacheAffinity {
            stage_index: 0,
            matched_tokens: 1,
            prefill_cost_per_token: saved_cost,
            restore_cost: 0,
            cache_epoch: 0,
        })
    }

    #[test]
    fn cache_value_orders_equal_priority_candidates() {
        let cold = affinity(0);
        let hot = affinity(100);
        let selected = select_cache_aware_candidate(
            [
                CacheAwareCandidate {
                    index: 0,
                    priority: 0,
                    affinity: &cold,
                    enqueued_turn: 0,
                    order: 0,
                },
                CacheAwareCandidate {
                    index: 1,
                    priority: 0,
                    affinity: &hot,
                    enqueued_turn: 0,
                    order: 1,
                },
            ],
            0,
            10,
        );
        assert_eq!(selected, Some(1));
    }

    #[test]
    fn aging_eventually_promotes_a_cold_request() {
        let cold = affinity(0);
        let hot = affinity(100);
        let selected = select_cache_aware_candidate(
            [
                CacheAwareCandidate {
                    index: 0,
                    priority: 0,
                    affinity: &cold,
                    enqueued_turn: 0,
                    order: 0,
                },
                CacheAwareCandidate {
                    index: 1,
                    priority: 0,
                    affinity: &hot,
                    enqueued_turn: 11,
                    order: 1,
                },
            ],
            11,
            10,
        );
        assert_eq!(selected, Some(0));
    }

    #[test]
    fn explicit_priority_precedes_cache_value() {
        let cold = affinity(0);
        let hot = affinity(1_000);
        let selected = select_cache_aware_candidate(
            [
                CacheAwareCandidate {
                    index: 0,
                    priority: 1,
                    affinity: &cold,
                    enqueued_turn: 0,
                    order: 0,
                },
                CacheAwareCandidate {
                    index: 1,
                    priority: 0,
                    affinity: &hot,
                    enqueued_turn: 0,
                    order: 1,
                },
            ],
            0,
            10,
        );
        assert_eq!(selected, Some(0));
    }
}
