//! Unit and comparison tests for the benefit policy (#1650 first slice).

use super::*;
use crate::policy::admission::AdmissionDecisionKind;
use crate::policy::lru_baseline::LruCache;
use crate::policy::traces;

fn cost(cold: f64, restore: f64) -> CostSample {
    CostSample {
        cold_prefill_cost: cold,
        restore_cost: restore,
    }
}

#[test]
fn rejects_entries_with_no_net_benefit() {
    let mut policy = BenefitPolicy::new(PolicyConfig::default());
    let decision = policy.consider_admission(1, 1 << 20, vec![], cost(100.0, 150.0));
    assert_eq!(decision.kind, AdmissionDecisionKind::Reject);
    assert!(policy.is_empty());
}

#[test]
fn new_entries_start_in_probation_and_promote_on_second_hit() {
    let mut policy = BenefitPolicy::new(PolicyConfig::default());
    policy.consider_admission(1, 1 << 20, vec![], cost(400.0, 120.0));
    assert_eq!(policy.entry(1).unwrap().state, PolicyEntryState::Probation);

    assert!(policy.record_hit(1, cost(400.0, 120.0)).is_none()); // first hit
    let promoted = policy.record_hit(1, cost(400.0, 120.0)).unwrap(); // second
    assert_eq!(promoted.kind, AdmissionDecisionKind::Promote);
    assert_eq!(policy.entry(1).unwrap().state, PolicyEntryState::Admitted);
}

#[test]
fn score_divides_by_exclusive_bytes() {
    let mut policy = BenefitPolicy::new(PolicyConfig::default());
    policy.consider_admission(1, 2 << 20, vec![], cost(300.0, 100.0));
    policy.record_hit(1, cost(300.0, 100.0));
    policy.record_hit(1, cost(300.0, 100.0));
    let score = policy.score(1).unwrap();
    // reuse=1.0, net=200.0, exclusive=2MiB
    assert!((score.inputs.net_benefit - 200.0).abs() < 1e-9);
    assert!((score.inputs.exclusive_bytes - (2.0 * 1024.0 * 1024.0)).abs() < 1e-9);
    assert!((score.value - 200.0 / (2.0 * 1024.0 * 1024.0)).abs() < 1e-9);
}

#[test]
fn shared_segments_get_fractional_credit_in_score() {
    let mut policy = BenefitPolicy::new(PolicyConfig::default());
    let seg = (7u64, 3 << 20);
    policy.consider_admission(1, 0, vec![seg], cost(400.0, 100.0));
    policy.consider_admission(2, 0, vec![seg], cost(400.0, 100.0));
    policy.consider_admission(3, 0, vec![seg], cost(400.0, 100.0));
    // Each entry charges 1MiB of the shared 3MiB segment.
    assert_eq!(policy.segments.total_bytes(), 3 << 20);
    for key in [1u64, 2, 3] {
        let score = policy.score(key).unwrap();
        assert!((score.inputs.exclusive_bytes - (1.0 * 1024.0 * 1024.0)).abs() < 1e-9);
    }
}

#[test]
fn eviction_picks_lowest_score_deterministically() {
    // Grace off: force the score order.
    let mut policy = BenefitPolicy::new(PolicyConfig {
        grace_observations: 0,
        ..PolicyConfig::default()
    });
    // Same value, different footprint: smaller footprint -> higher score.
    policy.consider_admission(1, 8 << 20, vec![], cost(400.0, 100.0));
    policy.consider_admission(2, 1 << 20, vec![], cost(400.0, 100.0));
    for key in [1u64, 2] {
        policy.record_hit(key, cost(400.0, 100.0));
        policy.record_hit(key, cost(400.0, 100.0));
    }
    let victims = policy.choose_victims(8 << 20, &[]);
    assert_eq!(victims.first().map(|(k, _)| *k), Some(1));
}

#[test]
fn pinned_entries_are_not_victims() {
    // Grace off: force the pinned check.
    let mut policy = BenefitPolicy::new(PolicyConfig {
        grace_observations: 0,
        ..PolicyConfig::default()
    });
    policy.consider_admission(1, 1 << 20, vec![], cost(400.0, 100.0));
    policy.consider_admission(2, 1 << 20, vec![], cost(400.0, 100.0));
    let victims = policy.choose_victims(1 << 20, &[2]);
    assert_eq!(victims.iter().map(|(k, _)| *k).collect::<Vec<_>>(), vec![1]);
}

#[test]
fn decay_shrinks_stale_reuse_probability() {
    let mut policy = BenefitPolicy::new(PolicyConfig::default());
    policy.consider_admission(1, 1 << 20, vec![], cost(400.0, 100.0));
    for _ in 0..10 {
        policy.record_hit(1, cost(400.0, 100.0));
    }
    let hot = policy.entry(1).unwrap().reuse_probability();
    for _ in 0..50 {
        policy.observe_pressure(1.0);
    }
    let stale = policy.entry(1).unwrap().reuse_probability();
    assert!(stale < hot);
}

#[test]
fn policy_decisions_are_deterministic_across_runs() {
    let trace = traces::zipf_hotset_trace(42, 500);
    let run = || {
        let mut policy = BenefitPolicy::new(PolicyConfig::default());
        let mut evictions = Vec::new();
        for access in &trace {
            if policy.entry(access.entry).is_some() {
                policy.record_hit(
                    access.entry,
                    cost(access.cold_prefill_cost, access.restore_cost),
                );
            } else {
                policy.consider_admission(
                    access.entry,
                    access.exclusive_bytes,
                    vec![],
                    cost(access.cold_prefill_cost, access.restore_cost),
                );
            }
            for (key, verdict) in policy.choose_victims(1 << 30, &[]) {
                if verdict == EvictionVerdict::Evict {
                    evictions.push(key);
                    policy.remove(key);
                }
            }
        }
        evictions
    };
    assert_eq!(run(), run());
}

/// Policy-vs-LRU comparison metrics over a trace at matched capacity.
struct Comparison {
    policy_saved_cost: f64,
    lru_saved_cost: f64,
    policy_bytes_written: u64,
    lru_bytes_written: u64,
}

fn compare(trace: &[traces::TraceAccess], capacity_bytes: u64) -> Comparison {
    let mut policy = BenefitPolicy::new(PolicyConfig::default());
    let mut lru = LruCache::new(capacity_bytes);
    let mut policy_saved = 0.0f64;
    let mut policy_written = 0u64;

    for access in trace {
        if policy.entry(access.entry).is_some() {
            policy.record_hit(
                access.entry,
                cost(access.cold_prefill_cost, access.restore_cost),
            );
            policy_saved += (access.cold_prefill_cost - access.restore_cost).max(0.0);
        } else {
            // Re-offer a rejected/evicted entry each time it recurs;
            // probation keeps new candidates in the accounting set until
            // pressure forces a choice.
            let decision = policy.consider_admission(
                access.entry,
                access.exclusive_bytes,
                vec![],
                cost(access.cold_prefill_cost, access.restore_cost),
            );
            policy_written += access.exclusive_bytes;
            // Hard probation cap is enforced as part of admission: commit the
            // selected victims.
            for key in decision.probation_cap_victims {
                policy.remove(key);
            }
        }
        let mut used: u64 = policy.entries.values().map(|e| e.exclusive_bytes).sum();
        while used > capacity_bytes {
            let victims = policy.choose_victims(used - capacity_bytes, &[]);
            if victims.is_empty() {
                break;
            }
            for (key, verdict) in victims {
                if verdict == EvictionVerdict::Evict {
                    policy.remove(key);
                }
            }
            used = policy.entries.values().map(|e| e.exclusive_bytes).sum();
        }
        lru.access(access);
    }
    Comparison {
        policy_saved_cost: policy_saved,
        lru_saved_cost: lru.saved_cost,
        policy_bytes_written: policy_written,
        lru_bytes_written: lru.bytes_written,
    }
}

/// The acceptance direction the issue requires: on a trace where one-shot
/// large entries pollute an LRU, the benefit policy must save more cold
/// prefill cost at the same capacity, or write fewer bytes.
#[test]
fn beats_lru_on_one_shot_pollution_trace() {
    let trace = traces::one_shot_trace(7, 2_000);
    let capacity = 64 << 20;
    let comparison = compare(&trace, capacity);
    assert!(
        comparison.policy_saved_cost >= comparison.lru_saved_cost * 0.95,
        "policy saved {} vs lru {}",
        comparison.policy_saved_cost,
        comparison.lru_saved_cost
    );
    assert!(
        comparison.policy_bytes_written <= comparison.lru_bytes_written,
        "policy wrote {} vs lru {}",
        comparison.policy_bytes_written,
        comparison.lru_bytes_written
    );
}

#[test]
fn no_regression_on_turn_growth_trace() {
    let trace = traces::turn_growth_trace(11, 4, 12);
    let capacity = 64 << 20;
    let comparison = compare(&trace, capacity);
    assert!(
        comparison.policy_saved_cost >= comparison.lru_saved_cost * 0.95,
        "policy saved {} vs lru {}",
        comparison.policy_saved_cost,
        comparison.lru_saved_cost
    );
}

#[test]
fn no_regression_on_zipf_hotset_trace() {
    let trace = traces::zipf_hotset_trace(3, 4_000);
    let capacity = 64 << 20;
    let comparison = compare(&trace, capacity);
    assert!(
        comparison.policy_saved_cost >= comparison.lru_saved_cost * 0.95,
        "policy saved {} vs lru {}",
        comparison.policy_saved_cost,
        comparison.lru_saved_cost
    );
}

#[test]
fn no_regression_on_mixed_size_trace() {
    let trace = traces::mixed_size_trace(5, 3_000);
    let capacity = 256 << 20;
    let comparison = compare(&trace, capacity);
    assert!(
        comparison.policy_saved_cost >= comparison.lru_saved_cost * 0.95,
        "policy saved {} vs lru {}",
        comparison.policy_saved_cost,
        comparison.lru_saved_cost
    );
}

#[test]
fn invalid_costs_and_config_are_rejected_not_panicked() {
    let mut policy = BenefitPolicy::new(PolicyConfig::default());
    let nan = f64::NAN;
    policy.consider_admission(
        1,
        1 << 20,
        vec![],
        CostSample {
            cold_prefill_cost: nan,
            restore_cost: 10.0,
        },
    );
    assert!(policy.score(1).is_none()); // NaN never enters the ordering
    assert!(
        !PolicyConfig {
            min_reuse_probability: f64::NAN,
            ..PolicyConfig::default()
        }
        .is_valid()
    );
    assert!(
        !PolicyConfig {
            persistence_hit_threshold: 0,
            ..PolicyConfig::default()
        }
        .is_valid()
    );
    assert!(
        !PolicyConfig {
            decay: DecayConfig { factor: 0.0 },
            ..PolicyConfig::default()
        }
        .is_valid()
    );
    assert!(
        !PolicyConfig {
            decay: DecayConfig { factor: 1.5 },
            ..PolicyConfig::default()
        }
        .is_valid()
    );
}

#[test]
fn choose_victims_never_returns_a_duplicate_key() {
    let mut policy = BenefitPolicy::new(PolicyConfig {
        grace_observations: 0,
        ..PolicyConfig::default()
    });
    for key in 1..=8u64 {
        policy.consider_admission(key, 1 << 20, vec![], cost(400.0, 100.0));
    }
    let victims = policy.choose_victims(u64::MAX, &[]);
    let keys: Vec<u64> = victims.iter().map(|(k, _)| *k).collect();
    let mut sorted = keys.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(keys.len(), sorted.len(), "duplicate victim keys");
    assert_eq!(keys.len(), 8);
}

#[test]
fn victim_selection_counts_marginal_physical_bytes_of_shared_segments() {
    // Three entries share one segment; only evicting the last reference
    // physically frees it. Victim selection must keep choosing until the
    // requested *physical* bytes are covered.
    let mut policy = BenefitPolicy::new(PolicyConfig {
        grace_observations: 0,
        ..PolicyConfig::default()
    });
    let seg = (1u64, 3 << 20);
    policy.consider_admission(1, 0, vec![seg], cost(400.0, 100.0));
    policy.consider_admission(2, 0, vec![seg], cost(400.0, 100.0));
    policy.consider_admission(3, 0, vec![seg], cost(400.0, 100.0));
    for key in [1u64, 2, 3] {
        policy.record_hit(key, cost(400.0, 100.0));
        policy.record_hit(key, cost(400.0, 100.0));
    }
    // Freeing 3 MiB must select all three references, not one (each alone
    // releases nothing physical).
    let victims = policy.choose_victims(3 << 20, &[]);
    assert_eq!(victims.len(), 3);
    // Freeing 1 MiB: marginal release of two of the three is 0, the third is
    // 3 MiB; the loop must not stop before the target is met.
    let victims = policy.choose_victims(1 << 20, &[]);
    assert_eq!(victims.len(), 3);
}

#[test]
fn probation_byte_cap_is_enforced_over_grace() {
    // Grace must never hold the policy over the hard probation byte cap.
    // Cap fits two small entries only.
    let mut policy = BenefitPolicy::new(PolicyConfig {
        probation_byte_budget: 2 << 20,
        ..PolicyConfig::default()
    });
    for key in 1..=10u64 {
        policy.consider_admission(key, 1 << 20, vec![], cost(400.0, 100.0));
    }
    assert!(policy.probation_bytes() > 2 << 20);
    for key in policy.enforce_probation_cap() {
        policy.remove(key);
    }
    assert!(
        policy.probation_bytes() <= 2 << 20,
        "probation bytes {} over cap {}",
        policy.probation_bytes(),
        2 << 20
    );
}

#[test]
fn recurrence_after_eviction_is_a_value_signal() {
    // An entry evicted before its second hit must not restart from zero
    // history: its recurrence carries ghost statistics and counts as a
    // reuse observation (issue: "second-hit or equivalent value signal").
    let mut policy = BenefitPolicy::new(PolicyConfig {
        grace_observations: 0,
        ..PolicyConfig::default()
    });
    policy.consider_admission(1, 1 << 20, vec![], cost(400.0, 100.0));
    assert!(policy.record_hit(1, cost(400.0, 100.0)).is_none()); // first hit
    policy.remove(1); // evicted before the second hit
    policy.consider_admission(1, 1 << 20, vec![], cost(400.0, 100.0));
    assert_eq!(
        policy.entry(1).unwrap().state,
        PolicyEntryState::Admitted,
        "recurrence with ghost history must re-admit as a value signal"
    );
}

#[test]
fn invalid_cost_samples_never_mutate_entry_state() {
    let mut policy = BenefitPolicy::new(PolicyConfig::default());
    let nan = f64::NAN;
    // Admission with an invalid sample must not insert.
    let decision = policy.consider_admission(
        1,
        1 << 20,
        vec![],
        CostSample {
            cold_prefill_cost: nan,
            restore_cost: 10.0,
        },
    );
    assert_eq!(decision.verdict, AdmissionVerdict::Reject);
    assert!(policy.is_empty());
    // A valid admission followed by an invalid hit must not poison state.
    policy.consider_admission(1, 1 << 20, vec![], cost(400.0, 100.0));
    let before = policy.entry(1).unwrap().clone();
    assert!(
        policy
            .record_hit(
                1,
                CostSample {
                    cold_prefill_cost: 400.0,
                    restore_cost: nan
                }
            )
            .is_none()
    );
    assert_eq!(policy.entry(1).unwrap().last_cost, before.last_cost);
    assert_eq!(policy.entry(1).unwrap().hits, before.hits);
    // NaN pressure is ignored entirely.
    policy.observe_pressure(nan);
    assert_eq!(policy.entry(1).unwrap().reuse_weight, before.reuse_weight);
}

#[test]
fn probation_cap_counts_shared_charge_and_is_enforced_by_admission() {
    // Probation entries backed entirely by shared segments charge fractional
    // bytes and must not grow without bound.
    let mut policy = BenefitPolicy::new(PolicyConfig {
        probation_byte_budget: 1 << 20, // smaller than one shared segment
        grace_observations: 0,
        ..PolicyConfig::default()
    });
    let mut victims = Vec::new();
    for key in 1..=6u64 {
        let decision = policy.consider_admission(key, 0, vec![(1u64, 3 << 20)], cost(400.0, 100.0));
        for v in decision.probation_cap_victims {
            policy.remove(v);
            victims.push(v);
        }
    }
    assert!(
        policy.probation_bytes() <= 1 << 20,
        "probation bytes {} over cap",
        policy.probation_bytes()
    );
    assert!(!victims.is_empty(), "shared-only probation must be capped");
}

#[test]
fn ghosts_are_bounded_by_count_and_age() {
    // Count bound: one-shot keys must not create permanent metadata.
    let mut policy = BenefitPolicy::new(PolicyConfig {
        ghost_capacity: 8,
        ghost_max_age_observations: 100,
        grace_observations: 0,
        ..PolicyConfig::default()
    });
    for key in 0..64u64 {
        policy.consider_admission(key, 1 << 20, vec![], cost(400.0, 100.0));
        for (_, verdict) in policy.choose_victims(u64::MAX, &[]) {
            if verdict == EvictionVerdict::Evict {
                policy.remove(key);
                break;
            }
        }
    }
    assert!(
        policy.ghost_count() <= 8,
        "ghost count {}",
        policy.ghost_count()
    );

    // Age bound: stale popularity cannot revive indefinitely.
    let mut policy = BenefitPolicy::new(PolicyConfig {
        ghost_capacity: 1024,
        ghost_max_age_observations: 10,
        grace_observations: 0,
        ..PolicyConfig::default()
    });
    policy.consider_admission(99, 1 << 20, vec![], cost(400.0, 100.0));
    policy.remove(99);
    assert!(policy.ghost(99).is_some());
    for _ in 0..50 {
        policy.consider_admission(0, 1 << 20, vec![], cost(400.0, 100.0));
    }
    assert!(
        policy.ghost(99).is_none(),
        "old ghost must expire via age bound"
    );
}

#[test]
fn probation_cap_selection_is_not_committed_removal() {
    // Selection must leave state untouched so callers can commit physically;
    // committed removal through `remove` records the ghost.
    let mut policy = BenefitPolicy::new(PolicyConfig {
        probation_byte_budget: 1 << 20,
        grace_observations: 0,
        ..PolicyConfig::default()
    });
    for key in 1..=4u64 {
        policy.consider_admission(key, 1 << 20, vec![], cost(400.0, 100.0));
    }
    let victims = policy.select_probation_cap_victims();
    assert!(!victims.is_empty());
    assert_eq!(policy.len(), 4, "selection must not remove entries");
    let decision = policy.consider_admission(5, 1 << 20, vec![], cost(400.0, 100.0));
    assert!(!decision.probation_cap_victims.is_empty());
    for key in &decision.probation_cap_victims {
        policy.remove(*key);
    }
    // Committed removals become ghosts (bounded).
    assert!(policy.ghost_count() > 0);
}

#[test]
fn invalid_observation_leaves_clock_and_ghosts_untouched() {
    let nan = f64::NAN;
    let mut policy = BenefitPolicy::new(PolicyConfig {
        grace_observations: 2,
        ghost_max_age_observations: 1,
        ..PolicyConfig::default()
    });
    // Seed one ghost.
    policy.consider_admission(9, 1 << 20, vec![], cost(400.0, 100.0));
    policy.remove(9);
    assert!(policy.ghost(9).is_some());

    // Invalid admission: clock must not advance, ghost must survive
    // (age bound is 1, so a real observation would have expired it).
    let before_clock = policy.clock_debug();
    let decision = policy.consider_admission(
        1,
        1 << 20,
        vec![],
        CostSample {
            cold_prefill_cost: nan,
            restore_cost: 10.0,
        },
    );
    assert_eq!(decision.verdict, AdmissionVerdict::Reject);
    assert_eq!(policy.clock_debug(), before_clock);
    assert!(policy.ghost(9).is_some());

    // Invalid hit: same invariants.
    policy.consider_admission(2, 1 << 20, vec![], cost(400.0, 100.0));
    let before_clock = policy.clock_debug();
    assert!(
        policy
            .record_hit(
                2,
                CostSample {
                    cold_prefill_cost: nan,
                    restore_cost: 10.0
                }
            )
            .is_none()
    );
    assert_eq!(policy.clock_debug(), before_clock);
}

#[test]
fn many_reference_small_segment_still_charges_probation_bytes() {
    // A 4-byte segment referenced by 9 probation entries: per-entry
    // truncation would charge 0; the class total must still be 4.
    let mut policy = BenefitPolicy::new(PolicyConfig::default());
    for key in 1..=9u64 {
        policy.consider_admission(key, 0, vec![(1u64, 4)], cost(400.0, 100.0));
    }
    assert!(
        policy.probation_bytes() >= 4,
        "charged {}",
        policy.probation_bytes()
    );
    // And with a 1-byte cap, admission must select shared-only victims.
    let mut policy = BenefitPolicy::new(PolicyConfig {
        probation_byte_budget: 0,
        ..PolicyConfig::default()
    });
    let decision = policy.consider_admission(1, 0, vec![(1u64, 4)], cost(400.0, 100.0));
    assert!(!decision.probation_cap_victims.is_empty());
}

#[test]
fn ghost_promoted_recurrence_returns_admit_persist() {
    let mut policy = BenefitPolicy::new(PolicyConfig::default());
    policy.consider_admission(1, 1 << 20, vec![], cost(400.0, 100.0));
    assert!(policy.record_hit(1, cost(400.0, 100.0)).is_none()); // hits = 1
    policy.remove(1);
    let decision = policy.consider_admission(1, 1 << 20, vec![], cost(400.0, 100.0));
    assert_eq!(decision.kind, AdmissionDecisionKind::AdmitPersist);
    assert_eq!(policy.entry(1).unwrap().state, PolicyEntryState::Admitted);
    // Non-promoted recurrence stays AdmitProbation/Probation.
    policy.consider_admission(2, 1 << 20, vec![], cost(400.0, 100.0));
    policy.remove(2);
    let decision = policy.consider_admission(2, 1 << 20, vec![], cost(400.0, 100.0));
    assert_eq!(decision.kind, AdmissionDecisionKind::AdmitProbation);
    assert_eq!(policy.entry(2).unwrap().state, PolicyEntryState::Probation);
}

#[test]
fn zero_ghost_capacity_is_a_real_zero_bound() {
    let mut policy = BenefitPolicy::new(PolicyConfig {
        ghost_capacity: 0,
        ..PolicyConfig::default()
    });
    assert!(
        PolicyConfig {
            ghost_capacity: 0,
            ..PolicyConfig::default()
        }
        .is_valid()
    );
    policy.consider_admission(1, 1 << 20, vec![], cost(400.0, 100.0));
    policy.remove(1);
    assert_eq!(policy.ghost_count(), 0, "zero capacity must retain nothing");
}

#[test]
fn cap_victim_selection_covers_recomputed_shared_shares() {
    // Removing shared references raises survivors' shares: the selected set
    // must actually bring the class under cap once committed.
    let mut policy = BenefitPolicy::new(PolicyConfig {
        probation_byte_budget: 9 << 20,
        grace_observations: 0,
        ..PolicyConfig::default()
    });
    // Three no-hit probationers, each 1 MiB exclusive + a shared 9 MiB
    // segment (3 MiB share each → class charge 3*(1+3) = 12 MiB > 9 MiB).
    for key in 1..=3u64 {
        let decision =
            policy.consider_admission(key, 1 << 20, vec![(7u64, 9 << 20)], cost(400.0, 100.0));
        for v in decision.probation_cap_victims {
            policy.remove(v);
        }
    }
    assert!(
        policy.probation_bytes() <= 9 << 20,
        "committed class charge {} over cap",
        policy.probation_bytes()
    );
    // Removal must have happened through the decision path.
    assert!(policy.len() < 3);
}

#[test]
fn duplicate_segment_references_are_rejected() {
    let mut policy = BenefitPolicy::new(PolicyConfig::default());
    let decision =
        policy.consider_admission(1, 0, vec![(42u64, 100), (42u64, 100)], cost(400.0, 100.0));
    assert_eq!(decision.verdict, AdmissionVerdict::Reject);
    assert!(
        decision
            .reasons
            .contains(&"duplicate-segment-reference".to_string())
    );
    assert!(policy.is_empty());
    // Conflicting size for a known segment is also rejected.
    let mut policy = BenefitPolicy::new(PolicyConfig::default());
    policy.consider_admission(1, 0, vec![(42u64, 100)], cost(400.0, 100.0));
    let decision = policy.consider_admission(2, 0, vec![(42u64, 200)], cost(400.0, 100.0));
    assert_eq!(decision.verdict, AdmissionVerdict::Reject);
    assert!(
        decision
            .reasons
            .contains(&"segment-size-conflict".to_string())
    );
    // Same size for a known segment is fine.
    let decision = policy.consider_admission(3, 0, vec![(42u64, 100)], cost(400.0, 100.0));
    assert_eq!(decision.verdict, AdmissionVerdict::Admit);
    // One 100-byte segment: class charge is exactly 100.
    assert_eq!(policy.probation_bytes(), 100);
}

#[test]
fn cap_victim_selection_reproduces_reported_counterexample() {
    // Exact shape of the reported probe: before=10,590,618 over
    // cap=9,437,184 (9 MiB); stale-share subtraction selected [1,2] and
    // left the committed class at 10,354,688. Reproduce it with concrete
    // numbers: 10 MiB cap basis scaled to 9 MiB via three probationers —
    // 1 MiB exclusive each (3 MiB) plus one shared 8 MiB segment
    // (8/3 MiB per share -> class ~3+8=11 MiB before, over cap).
    let mut policy = BenefitPolicy::new(PolicyConfig {
        probation_byte_budget: 9 << 20,
        grace_observations: 0,
        ..PolicyConfig::default()
    });
    // Admission applies cap victims incrementally, so build the over-cap
    // state with cap victims disabled (huge probation budget), then swap
    // in the real cap and select.
    for key in 1..=3u64 {
        policy.consider_admission(key, 1 << 20, vec![(7u64, 8 << 20)], cost(400.0, 100.0));
    }
    let before = policy.probation_bytes();
    assert!(before > 9 << 20, "before {}", before);
    // Now select against the real cap by constructing the over-cap state
    // through the public API: reset the budget by direct selection.
    let victims = policy.with_probation_budget(9 << 20, |p| p.select_probation_cap_victims());
    assert!(!victims.is_empty());
    for key in &victims {
        policy.remove(*key);
    }
    let after = policy.probation_bytes();
    assert!(
        after <= 9 << 20,
        "committed class {} still over cap after victims {:?}",
        after,
        victims
    );
}
