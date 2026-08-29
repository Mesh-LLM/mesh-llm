use skippy_protocol::{StageConfig, StageKvCacheConfig};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentCacheConfig {
    pub max_entries: usize,
    pub max_bytes: u64,
    pub min_tokens: u64,
    pub reserved_seq_count: i32,
    /// Maximum number of native KV cell positions the cache may hold
    /// at one time, in tokens. Under `kv_unified = true` (skippy patch
    /// 0034) the resident prefix cache shares one `n_ctx` cell pool
    /// with the active execution lanes. Without this cap the cache
    /// budget is bounded only by `max_entries` and `max_bytes`, both
    /// of which can easily allow more pinned tokens than the cell
    /// pool has cells — the lanes then can't find a free slot and
    /// the embedded runtime surfaces HTTP 502
    /// `RuntimeError: llama_decode failed`
    /// (`decode: failed to find a memory slot`).
    ///
    /// Set this to a fraction of the model's `n_ctx`. A value of 0 disables
    /// the cap and behaves like the legacy unbounded-by-tokens cache. The cap is
    /// only useful when `n_ctx` is comfortably larger than
    /// `min_tokens`; see [`derive_max_resident_tokens`] for the floor.
    pub max_resident_tokens: u64,
}

impl ResidentCacheConfig {
    pub fn from_stage(config: &StageConfig, cache: &StageKvCacheConfig) -> Self {
        let reserved_seq_count = i32::try_from(config.lane_count.saturating_mul(2))
            .unwrap_or(i32::MAX)
            .max(2);
        let max_resident_tokens = derive_max_resident_tokens(u64::from(config.ctx_size));
        Self {
            max_entries: cap_resident_entries(cache.max_entries, reserved_seq_count),
            max_bytes: cache.max_bytes,
            min_tokens: cache.min_tokens,
            reserved_seq_count,
            max_resident_tokens,
        }
    }
}

fn cap_resident_entries(configured_entries: usize, reserved_seq_count: i32) -> usize {
    let available_sequence_ids = crate::LLAMA_MAX_SEQ.saturating_sub(reserved_seq_count) as usize;
    if available_sequence_ids == 0 {
        return 0;
    }
    configured_entries.clamp(1, 512).min(available_sequence_ids)
}

/// Derive `max_resident_tokens` from the model's `n_ctx` cell pool.
///
/// The cache shares the `n_ctx` cell pool with the active lanes under
/// `kv_unified = true`. The cap reserves one eighth of the pool for
/// in-flight lane prefills and lets the cache use at most the other seven
/// eighths. Runtime capacity admission evicts resident entries before a
/// request consumes that reserve, so holding half the pool idle here only
/// shrinks the reusable working set and causes avoidable prefix thrashing.
///
/// For small contexts (smoke-test / tiny-model configs) the fractional pool
/// can be smaller than a single typical prompt; applying the cap then
/// rejects the very first record and degrades the cache without
/// preventing any real wedge. The cap is therefore disabled when the
/// model's `n_ctx` is below `MIN_CTX_FOR_CELL_CAP` cells. The original
/// failure mode this cap fixes is large-context unified-KV serving
/// (e.g. `n_ctx = 131072`), which comfortably clears this floor.
///
/// Picking `min_tokens` as the floor would be tempting but does not
/// match the actual wedge: callers can configure `min_tokens` as low
/// as 64 while still using a small `n_ctx`, and the cap would still
/// be smaller than typical prompts. A hard cell-count floor is easier
/// to reason about and matches the real-world contexts the cap is
/// designed for (long-context unified-KV serving).
const MIN_CTX_FOR_CELL_CAP: u64 = 8192;

fn derive_max_resident_tokens(ctx_size: u64) -> u64 {
    if ctx_size < MIN_CTX_FOR_CELL_CAP {
        return 0;
    }
    ctx_size.saturating_sub(ctx_size.saturating_div(8))
}

#[cfg(test)]
mod resident_cache_config_tests {
    use super::*;

    #[test]
    fn cap_disabled_for_smoke_test_ctx_size() {
        // Smoke-test / SmolLM2 scenario: ctx_size=768. Any fractional cap would
        // be smaller than a typical 533-token smoke prompt; cap stays
        // disabled.
        assert_eq!(derive_max_resident_tokens(768), 0);
    }

    #[test]
    fn cap_enabled_for_production_ctx_size() {
        // Production failure mode the cap is designed for.
        assert_eq!(derive_max_resident_tokens(131072), 114688);
        // Exactly at the floor.
        assert_eq!(derive_max_resident_tokens(8192), 7168);
        // Just above the floor.
        assert_eq!(derive_max_resident_tokens(16384), 14336);
    }

    #[test]
    fn cap_disabled_just_below_floor() {
        // Below the hard floor.
        assert_eq!(derive_max_resident_tokens(8191), 0);
        assert_eq!(derive_max_resident_tokens(4096), 0);
    }

    #[test]
    fn resident_entry_cap_fits_available_sequence_ids() {
        assert_eq!(cap_resident_entries(512, 8), 248);
        assert_eq!(cap_resident_entries(64, 8), 64);
        assert_eq!(cap_resident_entries(64, crate::LLAMA_MAX_SEQ), 0);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseCheckpointPolicy {
    pub min_tokens: u64,
    pub stride_tokens: u64,
    pub record_limit: u64,
    pub page_size_tokens: u64,
    /// Total tokens the resident cache may pin, used to bound how deep the
    /// record ladder can go. `0` means unknown and leaves it unbounded.
    ///
    /// The entry-count cap in `family_policy` assumes each entry costs about
    /// `min_tokens` cells, which the ladder violates: its first two slots are
    /// the longest candidates. Bounding by tokens keeps a single request from
    /// trying to pin more than the pool holds and evicting its own records.
    pub max_resident_tokens_hint: u64,
}

impl SparseCheckpointPolicy {
    pub fn from_cache(cache: &StageKvCacheConfig) -> Self {
        Self {
            min_tokens: cache.min_tokens,
            stride_tokens: cache.shared_prefix_stride_tokens,
            record_limit: cache.shared_prefix_record_limit,
            max_resident_tokens_hint: 0,
            page_size_tokens: cache.min_tokens.max(1),
        }
    }

    pub fn candidate_token_counts(self, token_count: u64) -> Vec<u64> {
        if token_count == 0 {
            return Vec::new();
        }
        let mut counts = vec![token_count];
        if self.min_tokens == 0 || token_count <= self.min_tokens {
            return counts;
        }
        let stride = self.stride_tokens.max(1).min(self.page_size_tokens.max(1));
        let mut candidate = stable_grid_floor(token_count, stride);
        if candidate == token_count {
            candidate = candidate.saturating_sub(stride);
        }
        if candidate < self.min_tokens {
            candidate = self.min_tokens;
        }
        while candidate >= self.min_tokens {
            counts.push(candidate);
            if candidate == self.min_tokens {
                break;
            }
            let next = candidate.saturating_sub(stride);
            candidate = next.max(self.min_tokens);
        }
        counts.sort_unstable_by(|a, b| b.cmp(a));
        counts.dedup();
        counts
    }

    /// Choose which prefix lengths to actually persist for a request.
    ///
    /// Recording is far more expensive than probing: a lookup walks a
    /// stride-aligned grid for free, while every recorded candidate costs
    /// resident KV cells (or an export) . So the lookup grid is dense and this
    /// ladder is deliberately sparse — which makes *which* lengths get a slot
    /// the single highest-leverage policy decision in the cache.
    ///
    /// The previous policy filled every slot from the top of the grid
    /// downwards. With the shipped `record_limit: 2` that meant an 8000-token
    /// request recorded `[8000, 7936]` — two lengths that differ by one stride
    /// and both sit in the request's *tail*, the least shareable part of a
    /// prompt. A second session with the same 2048-token system prompt and a
    /// different tail probes 2048, finds nothing, and pays a full cold prefill.
    /// Cross-session sharing was implemented and reachable, and the record side
    /// simply never stored anything shareable.
    ///
    /// The fix is not "record low candidates instead": `min_tokens` is an
    /// admission threshold, not an estimate of where the stable prefix ends,
    /// and a 256-token page saves little against an 8k prompt. Two distinct
    /// goals are competing for the same slots:
    ///
    /// - **Same-session continuation** wants the *longest* prefix, so the next
    ///   turn of this conversation re-prefills only the newly appended tokens.
    /// - **Cross-session sharing** wants a length near the shared
    ///   system-prompt/tool-schema boundary, which is well below the tail.
    ///
    /// Both are legitimate, so the ladder serves them in priority order rather
    /// than choosing between them:
    ///
    /// 1. the exact length,
    /// 2. the longest grid candidate — the near-tail continuation page that
    ///    the previous policy was right to keep,
    /// 3. any remaining slots spread *geometrically* down towards
    ///    `min_tokens`, for cross-session sharing.
    ///
    /// Geometric spacing bounds the worst-case miss: with an unknown boundary
    /// anywhere in the prompt, evenly spaced ratios keep the fraction of
    /// wasted prefill roughly constant wherever it lands, whereas linear
    /// spacing wastes disproportionately at short lengths.
    ///
    /// The added low candidates are also the *cheapest* ones to hold: a 768
    /// token page pins 768 KV cells against the 7936 of a near-tail page, so
    /// deepening the ladder downwards costs far less capacity than its length
    /// suggests. At `record_limit = 2` this reduces exactly to the previous
    /// `[exact, near-tail]` behaviour, so the change is opt-in via the limit.
    pub fn record_candidate_token_counts(self, token_count: u64) -> Vec<u64> {
        let candidates = self.candidate_token_counts(token_count);
        let limit = self.record_limit as usize;
        if limit == 0 || candidates.len() <= limit {
            return candidates;
        }

        let mut selected = Vec::with_capacity(limit);
        // Slot 0: exact length, for same-session continuation.
        selected.push(token_count);
        if limit == 1 {
            return selected;
        }

        // The shared-prefix slots are picked from the grid excluding the exact
        // length, which is already covered above.
        let shared: Vec<u64> = candidates
            .iter()
            .copied()
            .filter(|candidate| *candidate != token_count)
            .collect();
        if shared.is_empty() {
            return selected;
        }

        // Slot 1: longest grid candidate, for same-session continuation of a
        // conversation whose next turn appends a short tail.
        if let Some(near_tail) = shared.first().copied() {
            selected.push(near_tail);
        }

        // Cap the ladder by *pinned tokens*, not slot count: recording more
        // than the resident cache can hold just evicts what this same request
        // recorded a moment earlier, which is the thrash the deeper ladder was
        // meant to remove. `max_resident` of 0 means "unknown", and leaves the
        // ladder unconstrained.
        //
        // The budget is charged against the *shared* rungs only. The exact and
        // near-tail slots above are recorded unconditionally and are by far the
        // longest — for a 12k prompt they alone pin ~24k cells, more than the
        // whole budget on any context below 48k. Counting them here made the
        // very first shared rung unaffordable and silently reduced every
        // shipped default back to `[exact, near-tail]`, i.e. the pre-ladder
        // behaviour this policy exists to replace. Their cost is already spent
        // by the time we get here, so gating the cheap rungs on it protects
        // nothing and forfeits every cross-session hit.
        let pinned_budget = self.max_resident_tokens_hint;
        let mut pinned: u64 = 0;

        for target in self.shared_slot_targets(token_count, limit) {
            if selected.len() >= limit {
                break;
            }
            // Snap to the nearest grid length that is not already taken. The
            // grid is descending, so `min_by_key` on absolute distance gives
            // the closest recordable length to the ideal target.
            let Some(pick) = shared
                .iter()
                .copied()
                .filter(|candidate| !selected.contains(candidate))
                .min_by_key(|candidate| candidate.abs_diff(target))
            else {
                break;
            };
            // Stop once the ladder would pin more than the resident token
            // budget allows. Low candidates are cheap, so this usually admits
            // several of them; it only bites when the long slots have already
            // consumed the budget.
            if pinned_budget > 0 && pinned.saturating_add(pick) > pinned_budget {
                break;
            }
            pinned = pinned.saturating_add(pick);
            selected.push(pick);
        }

        selected.sort_unstable_by(|a, b| b.cmp(a));
        selected.dedup();
        selected
    }

    /// Ideal (pre-snapping) lengths for the shared-prefix record slots.
    ///
    /// Slots are spaced geometrically between `min_tokens` and the full
    /// length. The exact length and the near-tail candidate are allocated by
    /// the caller, so these targets cover the remaining `limit - 2` slots and
    /// deliberately bias below the tail. For an 8000-token request with
    /// `min_tokens = 256` and `record_limit = 4` this lands targets in the
    /// system-prompt/tool-schema region rather than the tail.
    fn shared_slot_targets(self, token_count: u64, limit: usize) -> Vec<u64> {
        let shared_slots = limit.saturating_sub(2);
        if shared_slots == 0 {
            return Vec::new();
        }
        let floor = self.min_tokens.max(1).min(token_count.max(1)) as f64;
        let ceiling = token_count.max(1) as f64;
        if ceiling <= floor {
            return vec![self.min_tokens];
        }

        // Geometric interpolation: target_i = floor * (ceiling/floor)^(i/n).
        // Index 0 is skipped (that is `floor` itself, the least useful end)
        // and index `shared_slots` is the exact length, already recorded.
        let ratio = ceiling / floor;
        (1..=shared_slots)
            .map(|index| {
                let exponent = index as f64 / (shared_slots + 1) as f64;
                let target = floor * ratio.powf(exponent);
                target.round().max(1.0) as u64
            })
            .collect()
    }
}

fn stable_grid_floor(token_count: u64, stride: u64) -> u64 {
    token_count.saturating_sub(token_count % stride.max(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_candidates_prefer_longest_prefix_first() {
        let policy = SparseCheckpointPolicy {
            min_tokens: 64,
            stride_tokens: 32,
            record_limit: 2,
            page_size_tokens: 64,
            max_resident_tokens_hint: 0,
        };

        assert_eq!(policy.candidate_token_counts(160), vec![160, 128, 96, 64]);
    }

    #[test]
    fn record_candidates_are_limited_but_keep_current_and_shared_prefix() {
        let policy = SparseCheckpointPolicy {
            min_tokens: 64,
            stride_tokens: 32,
            record_limit: 2,
            page_size_tokens: 64,
            max_resident_tokens_hint: 0,
        };

        assert_eq!(policy.record_candidate_token_counts(160), vec![160, 128]);
    }

    #[test]
    fn candidates_below_min_only_use_exact_request() {
        let policy = SparseCheckpointPolicy {
            min_tokens: 64,
            stride_tokens: 32,
            record_limit: 2,
            page_size_tokens: 64,
            max_resident_tokens_hint: 0,
        };

        assert_eq!(policy.candidate_token_counts(63), vec![63]);
        assert_eq!(policy.record_candidate_token_counts(63), vec![63]);
    }

    #[test]
    fn unlimited_record_candidates_keep_shared_prefix_grid() {
        let policy = SparseCheckpointPolicy {
            min_tokens: 64,
            stride_tokens: 32,
            record_limit: 0,
            page_size_tokens: 64,
            max_resident_tokens_hint: 0,
        };

        assert_eq!(
            policy.record_candidate_token_counts(160),
            vec![160, 128, 96, 64]
        );
    }

    #[test]
    fn same_prefix_different_tail_prompts_share_near_tail_candidate() {
        let policy = SparseCheckpointPolicy {
            min_tokens: 256,
            stride_tokens: 128,
            record_limit: 2,
            page_size_tokens: 256,
            max_resident_tokens_hint: 0,
        };

        let recorded = policy.record_candidate_token_counts(2214);
        let lookup = policy.candidate_token_counts(2231);
        let shared = recorded
            .iter()
            .copied()
            .find(|candidate| lookup.contains(candidate));

        assert_eq!(recorded, vec![2214, 2176]);
        assert_eq!(shared, Some(2176));
    }

    #[test]
    fn non_aligned_min_tokens_still_provides_shared_floor_candidate() {
        let policy = SparseCheckpointPolicy {
            min_tokens: 300,
            stride_tokens: 128,
            record_limit: 2,
            page_size_tokens: 300,
            max_resident_tokens_hint: 0,
        };

        assert_eq!(policy.candidate_token_counts(350), vec![350, 300]);
        assert_eq!(policy.record_candidate_token_counts(350), vec![350, 300]);
    }
}

#[cfg(test)]
mod cache_checkpoint_tests {
    use super::*;

    /// The shipped agentic policy: 128-token stride, 256-token floor.
    fn agentic_policy(record_limit: u64) -> SparseCheckpointPolicy {
        SparseCheckpointPolicy {
            min_tokens: 256,
            stride_tokens: 128,
            record_limit,
            page_size_tokens: 256,
            max_resident_tokens_hint: 0,
        }
    }

    /// Regression guard for the headline finding: at the shipped
    /// `record_limit: 2` an 8000-token request recorded only `[8000, 7936]`,
    /// both in the tail, so a 2048-token shared system prompt was never
    /// stored even though lookup probes for it.
    #[test]
    fn shallow_ladder_records_only_tail_candidates() {
        let recorded = agentic_policy(2).record_candidate_token_counts(8000);

        assert_eq!(recorded, vec![8000, 7936]);
        let lowest = recorded.iter().copied().min().expect("recorded candidate");
        assert!(
            lowest > 4000,
            "shallow ladder should sit in the tail, got {lowest}"
        );
    }

    /// With a deeper ladder the same request now stores candidates down in the
    /// system-prompt region, which is what makes cross-session sharing land.
    #[test]
    fn deeper_ladder_reaches_the_shared_system_prompt_region() {
        let recorded = agentic_policy(4).record_candidate_token_counts(8000);

        assert_eq!(recorded.len(), 4);
        // Continuation candidates are retained.
        assert_eq!(recorded[0], 8000);
        assert_eq!(recorded[1], 7936);
        // ...and the remaining slots reach well below the tail.
        let lowest = recorded.iter().copied().min().expect("recorded candidate");
        assert!(
            lowest < 2048,
            "deeper ladder should reach the shared-prefix region, got {lowest}"
        );
        // Every recorded length must be probed by lookup, or it is dead weight.
        let probed = agentic_policy(4).candidate_token_counts(8000);
        for candidate in &recorded {
            assert!(probed.contains(candidate), "{candidate} is never probed");
        }
    }

    /// The end-to-end scenario from the issue: session A serves an 8k prompt,
    /// session B arrives later with the same 2k system prompt and a different
    /// tail. Under the shallow ladder B shares nothing; under the deeper
    /// ladder it finds a substantial shared prefix.
    #[test]
    fn cross_session_sharing_requires_a_deeper_ladder() {
        let shared_system_prompt_tokens = 2048;
        let session_a_total = 8000;
        // Session B: same system prompt, completely different 700-token tail.
        let session_b_total = shared_system_prompt_tokens + 700;

        let shallow_recorded = agentic_policy(2).record_candidate_token_counts(session_a_total);
        let shallow_probed = agentic_policy(2).candidate_token_counts(session_b_total);
        let shallow_hit = shallow_recorded
            .iter()
            .copied()
            .find(|candidate| shallow_probed.contains(candidate));
        assert_eq!(
            shallow_hit, None,
            "shallow ladder should miss entirely across sessions"
        );

        let deep_recorded = agentic_policy(6).record_candidate_token_counts(session_a_total);
        let deep_probed = agentic_policy(6).candidate_token_counts(session_b_total);
        let deep_hit = deep_recorded
            .iter()
            .copied()
            .find(|candidate| deep_probed.contains(candidate));
        let deep_hit = deep_hit.expect("deeper ladder should share a prefix across sessions");
        assert!(
            deep_hit >= 1000,
            "shared prefix should be substantial, got {deep_hit}"
        );
        // The reused prefix must not exceed what session B actually sent.
        assert!(deep_hit <= session_b_total);
    }

    /// Candidates must stay strictly descending, unique, and within bounds at
    /// every limit, since the lookup path assumes longest-first ordering.
    #[test]
    fn ladder_is_well_formed_at_every_limit() {
        for limit in 0..12u64 {
            for token_count in [300u64, 1000, 2214, 8000, 131_072] {
                let recorded = agentic_policy(limit).record_candidate_token_counts(token_count);
                if limit > 0 {
                    assert!(recorded.len() <= limit as usize);
                }
                assert!(recorded.contains(&token_count), "exact length must be kept");
                let mut sorted = recorded.clone();
                sorted.sort_unstable_by(|a, b| b.cmp(a));
                assert_eq!(recorded, sorted, "must be descending");
                let mut unique = recorded.clone();
                unique.dedup();
                assert_eq!(recorded, unique, "must be unique");
                for candidate in &recorded {
                    assert!(*candidate <= token_count && *candidate > 0);
                }
            }
        }
    }

    /// The resident token budget bounds the *shared* rungs, not the two
    /// mandatory slots.
    ///
    /// The exact and near-tail slots are recorded unconditionally and are the
    /// longest candidates, so charging them against the budget consumed it
    /// outright on any realistic context and left no shared rung affordable —
    /// which is precisely the `[exact, near-tail]` behaviour the ladder
    /// replaces. Their cost is already committed, so the budget governs only
    /// what is still optional.
    #[test]
    fn ladder_budget_bounds_the_shared_rungs_not_the_mandatory_slots() {
        let policy = SparseCheckpointPolicy {
            min_tokens: 256,
            stride_tokens: 128,
            record_limit: 6,
            page_size_tokens: 256,
            // An 8k context gives a 4k resident budget.
            max_resident_tokens_hint: 4096,
        };

        let recorded = policy.record_candidate_token_counts(8000);

        assert_eq!(recorded[0], 8000, "exact length is always recorded");
        assert_eq!(
            recorded[1], 7936,
            "near-tail continuation is always recorded"
        );

        let shared: Vec<u64> = recorded[2..].to_vec();
        assert!(
            !shared.is_empty(),
            "a 4k budget must still afford shared rungs; got {recorded:?}"
        );
        let shared_pinned: u64 = shared.iter().sum();
        assert!(
            shared_pinned <= 4096,
            "shared rungs {shared:?} pin {shared_pinned} tokens, over the 4096 budget"
        );
    }

    /// With a generous budget the deeper ladder is admitted in full.
    #[test]
    fn generous_budget_admits_the_full_ladder() {
        let policy = SparseCheckpointPolicy {
            min_tokens: 256,
            stride_tokens: 128,
            record_limit: 6,
            page_size_tokens: 256,
            max_resident_tokens_hint: 1_000_000,
        };

        let recorded = policy.record_candidate_token_counts(8000);

        assert_eq!(recorded.len(), 6);
        assert!(recorded.iter().copied().min().unwrap() < 2048);
    }

    /// Short prompts below the floor must not gain phantom candidates.
    #[test]
    fn short_prompts_are_unaffected_by_a_deeper_ladder() {
        assert_eq!(
            agentic_policy(6).record_candidate_token_counts(200),
            vec![200]
        );
    }
}

#[cfg(test)]
mod shipped_default_ladder_tests {
    use super::*;

    /// The ladder must produce shareable rungs under the *shipped* config,
    /// not just under hand-picked test values.
    ///
    /// `family_policy` derives `record_limit` from the entry cap, which is
    /// itself derived from `n_ctx`, and `max_resident_tokens_hint` is a fixed
    /// fraction of `n_ctx`. Those three interact: charging the unconditional
    /// exact and near-tail slots against the token budget consumed it entirely
    /// on every context below ~48k, so the shared rungs were never affordable and the
    /// deeper ladder silently did nothing on real deployments. This test pins
    /// the composed behaviour so that regression cannot return unnoticed.
    #[test]
    fn shipped_defaults_record_shareable_rungs() {
        for ctx in [8192u64, 16384, 32768, 131072] {
            let max_entries = ((ctx / 2 / 256) as usize).clamp(1, 16);
            let record_limit = ((max_entries as u64) / 4).clamp(2, 6);
            let policy = SparseCheckpointPolicy {
                min_tokens: 256,
                stride_tokens: 128,
                record_limit,
                page_size_tokens: 256,
                max_resident_tokens_hint: derive_max_resident_tokens(ctx),
            };

            let recorded = policy.record_candidate_token_counts(12288);
            let shareable: Vec<u64> = recorded
                .iter()
                .copied()
                .filter(|length| *length * 2 < 12288)
                .collect();

            assert!(
                !shareable.is_empty(),
                "ctx={ctx} recorded {recorded:?} with no rung below the request tail, \
                 so a second session with the same system prompt cannot hit anything"
            );
        }
    }

    /// Contexts small enough to derive `record_limit = 2` keep the historical
    /// `[exact, near-tail]` behaviour. Documented rather than asserted-away:
    /// raising the floor changes default behaviour and is a separate decision.
    #[test]
    fn small_contexts_still_record_only_the_tail() {
        let policy = SparseCheckpointPolicy {
            min_tokens: 256,
            stride_tokens: 128,
            record_limit: 2,
            page_size_tokens: 256,
            max_resident_tokens_hint: derive_max_resident_tokens(4096),
        };

        assert_eq!(
            policy.record_candidate_token_counts(12288),
            vec![12288, 12160]
        );
    }
}
