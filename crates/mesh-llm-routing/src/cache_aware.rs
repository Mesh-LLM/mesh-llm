//! Cost-based selection from positive cache evidence.

use crate::InferenceTarget;
use crate::cache_inventory::CacheAffinityEntry;

/// Minimum evidence required before cache locality may change routing.
pub const DEFAULT_MIN_SAVED_TOKENS: u32 = 256;
/// Conservative prefill-cost estimate used when a runtime has not advertised
/// a measured value yet.
pub const DEFAULT_PREFILL_MICROS_PER_TOKEN: u64 = 1_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheAwareConfig {
    pub min_saved_tokens: u32,
    pub prefill_micros_per_token: u64,
}

impl Default for CacheAwareConfig {
    fn default() -> Self {
        Self {
            min_saved_tokens: DEFAULT_MIN_SAVED_TOKENS,
            prefill_micros_per_token: DEFAULT_PREFILL_MICROS_PER_TOKEN,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TargetCacheEvidence {
    pub target: InferenceTarget,
    pub entry: CacheAffinityEntry,
}

impl TargetCacheEvidence {
    pub fn estimated_cost_micros(&self, config: CacheAwareConfig) -> u64 {
        self.entry
            .queue_delay_micros
            .saturating_add(self.entry.restore_micros)
            .saturating_add(
                u64::from(self.entry.suffix_prefill_tokens)
                    .saturating_mul(config.prefill_micros_per_token),
            )
    }
}

/// Choose the lowest estimated-cost material cache hit. Candidate order is the
/// deterministic tie-breaker, so normal health/context ordering remains stable.
pub fn select_cache_target(
    candidates: &[InferenceTarget],
    evidence: &[TargetCacheEvidence],
    config: CacheAwareConfig,
) -> Option<TargetCacheEvidence> {
    let cold_prefill_tokens = evidence
        .iter()
        .map(|item| {
            u64::from(item.entry.matched_tokens)
                .saturating_add(u64::from(item.entry.suffix_prefill_tokens))
        })
        .max()?;
    let cold_cost_micros = cold_prefill_tokens.saturating_mul(config.prefill_micros_per_token);
    candidates
        .iter()
        .enumerate()
        .filter_map(|(candidate_order, candidate)| {
            evidence
                .iter()
                .find(|item| &item.target == candidate)
                .filter(|item| item.entry.matched_tokens >= config.min_saved_tokens)
                .map(|item| {
                    (
                        item.estimated_cost_micros(config),
                        std::cmp::Reverse(item.entry.matched_tokens),
                        candidate_order,
                        item.clone(),
                    )
                })
        })
        .min_by_key(|(cost, matched, order, _)| (*cost, *matched, *order))
        .filter(|(cost, _, _, _)| *cost < cold_cost_micros)
        .map(|(_, _, _, evidence)| evidence)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache_inventory::CacheTier;
    use iroh::SecretKey;

    fn remote(seed: u8) -> InferenceTarget {
        let mut bytes = [0u8; 32];
        bytes[0] = seed;
        InferenceTarget::Remote(SecretKey::from_bytes(&bytes).public())
    }

    fn evidence(
        target: InferenceTarget,
        matched: u32,
        suffix: u32,
        queue: u64,
    ) -> TargetCacheEvidence {
        TargetCacheEvidence {
            target,
            entry: CacheAffinityEntry {
                model: "model".to_string(),
                prefix_digest: [7; 16],
                matched_tokens: matched,
                suffix_prefill_tokens: suffix,
                tier: CacheTier::L1,
                restore_micros: 0,
                queue_delay_micros: queue,
            },
        }
    }

    #[test]
    fn saved_work_beats_raw_matched_depth() {
        let first = remote(1);
        let second = remote(2);
        let selected = select_cache_target(
            &[first.clone(), second.clone()],
            &[
                evidence(first, 2_000, 100, 2_000_000),
                evidence(second.clone(), 1_500, 200, 0),
            ],
            CacheAwareConfig::default(),
        )
        .expect("material hit");

        assert_eq!(selected.target, second);
    }

    #[test]
    fn small_hits_do_not_override_normal_routing() {
        let target = remote(1);
        assert!(
            select_cache_target(
                std::slice::from_ref(&target),
                &[evidence(target.clone(), DEFAULT_MIN_SAVED_TOKENS - 1, 0, 0)],
                CacheAwareConfig::default(),
            )
            .is_none()
        );
    }

    #[test]
    fn queue_delay_can_make_a_hit_worse_than_cold_prefill() {
        let target = remote(1);
        assert!(
            select_cache_target(
                std::slice::from_ref(&target),
                &[evidence(target.clone(), 512, 32, 1_000_000)],
                CacheAwareConfig::default(),
            )
            .is_none()
        );
    }
}
