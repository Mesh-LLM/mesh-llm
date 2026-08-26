use std::collections::BTreeMap;

/// Requests whose cache-plus-aging scores differ by fewer than this many
/// scheduler turns remain eligible for waiting-prefix grouping. The band keeps
/// locality reachable for naturally staggered arrivals without allowing a
/// prefix group to outrank materially older or more valuable work.
const PREFIX_GROUPING_SCORE_BAND_TURNS: u64 = 4;

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
    pub prompt_tokens: &'a [i32],
    pub enqueued_turn: u64,
    pub order: u64,
}

/// Order cache candidates by priority, saved work plus aging, then waiting
/// prefix locality.
///
/// Equal-priority requests gain `aging_cost_per_turn` for every turn they wait,
/// which bounds starvation even when hot-prefix requests keep arriving. The
/// Within a four-turn score band, the locality tie-break builds an ephemeral
/// radix order over waiting prompts and visits the heaviest shared-prefix
/// subtrees first. It never touches the materialized cache or its LRU recency.
pub fn order_cache_aware_candidates<'a>(
    candidates: impl IntoIterator<Item = CacheAwareCandidate<'a>>,
    current_turn: u64,
    aging_cost_per_turn: u64,
    group_waiting_prefixes: bool,
) -> Vec<usize> {
    let candidates = candidates.into_iter().collect::<Vec<_>>();
    let mut dfs_ranks = vec![0usize; candidates.len()];
    if group_waiting_prefixes {
        let mut dfs_order = Vec::with_capacity(candidates.len());
        append_dfs_weight_order(
            &candidates,
            (0..candidates.len()).collect(),
            0,
            &mut dfs_order,
        );
        for (rank, position) in dfs_order.into_iter().enumerate() {
            dfs_ranks[position] = rank;
        }
    }

    let mut ranked = candidates
        .into_iter()
        .enumerate()
        .map(|(position, candidate)| (candidate, dfs_ranks[position]))
        .collect::<Vec<_>>();
    ranked.sort_by(|(left, left_dfs_rank), (right, right_dfs_rank)| {
        let left_score = effective_score(left, current_turn, aging_cost_per_turn);
        let right_score = effective_score(right, current_turn, aging_cost_per_turn);
        right
            .priority
            .cmp(&left.priority)
            .then_with(|| {
                if group_waiting_prefixes {
                    grouping_score(right, current_turn, aging_cost_per_turn).cmp(&grouping_score(
                        left,
                        current_turn,
                        aging_cost_per_turn,
                    ))
                } else {
                    right_score.cmp(&left_score)
                }
            })
            .then_with(|| {
                if group_waiting_prefixes && left.prompt_tokens != right.prompt_tokens {
                    left_dfs_rank.cmp(right_dfs_rank)
                } else {
                    std::cmp::Ordering::Equal
                }
            })
            .then_with(|| right_score.cmp(&left_score))
            .then_with(|| left.order.cmp(&right.order))
    });
    ranked
        .into_iter()
        .map(|(candidate, _)| candidate.index)
        .collect()
}

fn grouping_score(
    candidate: &CacheAwareCandidate<'_>,
    current_turn: u64,
    aging_cost_per_turn: u64,
) -> u64 {
    let width = aging_cost_per_turn
        .max(1)
        .saturating_mul(PREFIX_GROUPING_SCORE_BAND_TURNS);
    let age = current_turn.saturating_sub(candidate.enqueued_turn);
    let banded_age = age.saturating_add(PREFIX_GROUPING_SCORE_BAND_TURNS.saturating_sub(1))
        / PREFIX_GROUPING_SCORE_BAND_TURNS;
    candidate
        .affinity
        .estimated_saved_cost()
        .saturating_add(banded_age.saturating_mul(width))
}

/// Select the first candidate from [`order_cache_aware_candidates`].
pub fn select_cache_aware_candidate<'a>(
    candidates: impl IntoIterator<Item = CacheAwareCandidate<'a>>,
    current_turn: u64,
    aging_cost_per_turn: u64,
    group_waiting_prefixes: bool,
) -> Option<usize> {
    order_cache_aware_candidates(
        candidates,
        current_turn,
        aging_cost_per_turn,
        group_waiting_prefixes,
    )
    .into_iter()
    .next()
}

fn effective_score(
    candidate: &CacheAwareCandidate<'_>,
    current_turn: u64,
    aging_cost_per_turn: u64,
) -> u64 {
    let age = current_turn.saturating_sub(candidate.enqueued_turn);
    candidate
        .affinity
        .estimated_saved_cost()
        .saturating_add(age.saturating_mul(aging_cost_per_turn))
}

/// Append a compressed-radix DFS order for candidate positions.
///
/// Common token runs are skipped before branching, so recursion depth follows
/// radix branch points rather than prompt length.
fn append_dfs_weight_order(
    candidates: &[CacheAwareCandidate<'_>],
    positions: Vec<usize>,
    mut depth: usize,
    output: &mut Vec<usize>,
) {
    if positions.len() <= 1 {
        output.extend(positions);
        return;
    }

    let common_limit = positions
        .iter()
        .map(|position| candidates[*position].prompt_tokens.len())
        .min()
        .unwrap_or(depth);
    while depth < common_limit {
        let token = candidates[positions[0]].prompt_tokens[depth];
        if positions
            .iter()
            .all(|position| candidates[*position].prompt_tokens[depth] == token)
        {
            depth += 1;
        } else {
            break;
        }
    }

    let mut terminal = Vec::new();
    let mut children = BTreeMap::<i32, Vec<usize>>::new();
    for position in positions {
        match candidates[position].prompt_tokens.get(depth).copied() {
            Some(token) => children.entry(token).or_default().push(position),
            None => terminal.push(position),
        }
    }
    let mut children = children.into_iter().collect::<Vec<_>>();
    children.sort_by(|(left_token, left), (right_token, right)| {
        right
            .len()
            .cmp(&left.len())
            .then_with(|| {
                minimum_enqueue_order(candidates, left)
                    .cmp(&minimum_enqueue_order(candidates, right))
            })
            .then_with(|| left_token.cmp(right_token))
    });
    for (_, child) in children {
        append_dfs_weight_order(candidates, child, depth.saturating_add(1), output);
    }
    terminal.sort_by_key(|position| candidates[*position].order);
    output.extend(terminal);
}

fn minimum_enqueue_order(candidates: &[CacheAwareCandidate<'_>], positions: &[usize]) -> u64 {
    positions
        .iter()
        .map(|position| candidates[*position].order)
        .min()
        .unwrap_or(u64::MAX)
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
                    prompt_tokens: &[1, 2, 3],
                    enqueued_turn: 0,
                    order: 0,
                },
                CacheAwareCandidate {
                    index: 1,
                    priority: 0,
                    affinity: &hot,
                    prompt_tokens: &[4, 5, 6],
                    enqueued_turn: 0,
                    order: 1,
                },
            ],
            0,
            10,
            true,
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
                    prompt_tokens: &[1, 2, 3],
                    enqueued_turn: 0,
                    order: 0,
                },
                CacheAwareCandidate {
                    index: 1,
                    priority: 0,
                    affinity: &hot,
                    prompt_tokens: &[4, 5, 6],
                    enqueued_turn: 11,
                    order: 1,
                },
            ],
            11,
            10,
            true,
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
                    prompt_tokens: &[1, 2, 3],
                    enqueued_turn: 0,
                    order: 0,
                },
                CacheAwareCandidate {
                    index: 1,
                    priority: 0,
                    affinity: &hot,
                    prompt_tokens: &[4, 5, 6],
                    enqueued_turn: 0,
                    order: 1,
                },
            ],
            0,
            10,
            true,
        );
        assert_eq!(selected, Some(0));
    }

    #[test]
    fn dfs_weight_groups_the_heaviest_equal_score_prefix_subtree() {
        let affinity = CacheAffinity::default();
        let ordered = order_cache_aware_candidates(
            [
                CacheAwareCandidate {
                    index: 0,
                    priority: 0,
                    affinity: &affinity,
                    prompt_tokens: &[9, 9, 9],
                    enqueued_turn: 0,
                    order: 0,
                },
                CacheAwareCandidate {
                    index: 1,
                    priority: 0,
                    affinity: &affinity,
                    prompt_tokens: &[1, 2, 3],
                    enqueued_turn: 0,
                    order: 1,
                },
                CacheAwareCandidate {
                    index: 2,
                    priority: 0,
                    affinity: &affinity,
                    prompt_tokens: &[1, 2, 4],
                    enqueued_turn: 0,
                    order: 2,
                },
            ],
            0,
            10,
            true,
        );

        assert_eq!(ordered, [1, 2, 0]);
    }

    #[test]
    fn dfs_weight_groups_staggered_arrivals_within_score_band() {
        let affinity = CacheAffinity::default();
        let ordered = order_cache_aware_candidates(
            [
                CacheAwareCandidate {
                    index: 0,
                    priority: 0,
                    affinity: &affinity,
                    prompt_tokens: &[9, 9, 9],
                    enqueued_turn: 0,
                    order: 0,
                },
                CacheAwareCandidate {
                    index: 1,
                    priority: 0,
                    affinity: &affinity,
                    prompt_tokens: &[1, 2, 3],
                    enqueued_turn: 1,
                    order: 1,
                },
                CacheAwareCandidate {
                    index: 2,
                    priority: 0,
                    affinity: &affinity,
                    prompt_tokens: &[1, 2, 4],
                    enqueued_turn: 2,
                    order: 2,
                },
            ],
            3,
            4_096,
            true,
        );

        assert_eq!(ordered, [1, 2, 0]);
    }

    #[test]
    fn materially_older_work_precedes_prefix_group() {
        let affinity = CacheAffinity::default();
        let ordered = order_cache_aware_candidates(
            [
                CacheAwareCandidate {
                    index: 0,
                    priority: 0,
                    affinity: &affinity,
                    prompt_tokens: &[9, 9, 9],
                    enqueued_turn: 0,
                    order: 0,
                },
                CacheAwareCandidate {
                    index: 1,
                    priority: 0,
                    affinity: &affinity,
                    prompt_tokens: &[1, 2, 3],
                    enqueued_turn: 9,
                    order: 1,
                },
                CacheAwareCandidate {
                    index: 2,
                    priority: 0,
                    affinity: &affinity,
                    prompt_tokens: &[1, 2, 4],
                    enqueued_turn: 10,
                    order: 2,
                },
            ],
            10,
            4_096,
            true,
        );

        assert_eq!(ordered[0], 0);
    }

    #[test]
    fn disabling_prefix_grouping_restores_enqueue_order() {
        let affinity = CacheAffinity::default();
        let ordered = order_cache_aware_candidates(
            [
                CacheAwareCandidate {
                    index: 0,
                    priority: 0,
                    affinity: &affinity,
                    prompt_tokens: &[9, 9, 9],
                    enqueued_turn: 0,
                    order: 0,
                },
                CacheAwareCandidate {
                    index: 1,
                    priority: 0,
                    affinity: &affinity,
                    prompt_tokens: &[1, 2, 3],
                    enqueued_turn: 0,
                    order: 1,
                },
                CacheAwareCandidate {
                    index: 2,
                    priority: 0,
                    affinity: &affinity,
                    prompt_tokens: &[1, 2, 4],
                    enqueued_turn: 0,
                    order: 2,
                },
            ],
            0,
            10,
            false,
        );

        assert_eq!(ordered, [0, 1, 2]);
    }

    #[test]
    fn materialized_cache_value_precedes_waiting_prefix_weight() {
        let cold = affinity(0);
        let hot = affinity(100);
        let ordered = order_cache_aware_candidates(
            [
                CacheAwareCandidate {
                    index: 0,
                    priority: 0,
                    affinity: &cold,
                    prompt_tokens: &[1, 2, 3],
                    enqueued_turn: 0,
                    order: 0,
                },
                CacheAwareCandidate {
                    index: 1,
                    priority: 0,
                    affinity: &cold,
                    prompt_tokens: &[1, 2, 4],
                    enqueued_turn: 0,
                    order: 1,
                },
                CacheAwareCandidate {
                    index: 2,
                    priority: 0,
                    affinity: &hot,
                    prompt_tokens: &[9, 9, 9],
                    enqueued_turn: 0,
                    order: 2,
                },
            ],
            0,
            10,
            true,
        );

        assert_eq!(ordered[0], 2);
    }

    #[test]
    fn long_common_prefix_does_not_drive_recursion_depth() {
        let affinity = CacheAffinity::default();
        let mut left = vec![1; 100_000];
        let mut right = left.clone();
        left.push(2);
        right.push(3);
        let ordered = order_cache_aware_candidates(
            [
                CacheAwareCandidate {
                    index: 0,
                    priority: 0,
                    affinity: &affinity,
                    prompt_tokens: &left,
                    enqueued_turn: 0,
                    order: 0,
                },
                CacheAwareCandidate {
                    index: 1,
                    priority: 0,
                    affinity: &affinity,
                    prompt_tokens: &right,
                    enqueued_turn: 0,
                    order: 1,
                },
            ],
            0,
            10,
            true,
        );

        assert_eq!(ordered, [0, 1]);
    }
}
