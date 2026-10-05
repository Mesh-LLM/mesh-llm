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
    let decision = policy.consider_admission(1, 1 << 20, vec![], cost(100.0, 150.0), &[]);
    assert_eq!(decision.kind, AdmissionDecisionKind::Reject);
    assert!(policy.is_empty());
}

#[test]
fn new_entries_start_in_probation_and_promote_on_second_hit() {
    let mut policy = BenefitPolicy::new(PolicyConfig::default());
    policy.consider_admission(1, 1 << 20, vec![], cost(400.0, 120.0), &[]);
    assert_eq!(policy.entry(1).unwrap().state, PolicyEntryState::Probation);

    assert!(policy.record_hit(1, cost(400.0, 120.0)).is_none()); // first hit
    let promoted = policy.record_hit(1, cost(400.0, 120.0)).unwrap(); // second
    assert_eq!(promoted.kind, AdmissionDecisionKind::Promote);
    assert_eq!(policy.entry(1).unwrap().state, PolicyEntryState::Admitted);
}

#[test]
fn score_divides_by_exclusive_bytes() {
    let mut policy = BenefitPolicy::new(PolicyConfig::default());
    policy.consider_admission(1, 2 << 20, vec![], cost(300.0, 100.0), &[]);
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
    policy.consider_admission(1, 0, vec![seg], cost(400.0, 100.0), &[]);
    policy.consider_admission(2, 0, vec![seg], cost(400.0, 100.0), &[]);
    policy.consider_admission(3, 0, vec![seg], cost(400.0, 100.0), &[]);
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
    policy.consider_admission(1, 8 << 20, vec![], cost(400.0, 100.0), &[]);
    policy.consider_admission(2, 1 << 20, vec![], cost(400.0, 100.0), &[]);
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
    policy.consider_admission(1, 1 << 20, vec![], cost(400.0, 100.0), &[]);
    policy.consider_admission(2, 1 << 20, vec![], cost(400.0, 100.0), &[]);
    let victims = policy.choose_victims(1 << 20, &[2]);
    assert_eq!(victims.iter().map(|(k, _)| *k).collect::<Vec<_>>(), vec![1]);
}

#[test]
fn decay_shrinks_stale_reuse_probability() {
    let mut policy = BenefitPolicy::new(PolicyConfig::default());
    policy.consider_admission(1, 1 << 20, vec![], cost(400.0, 100.0), &[]);
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
                    &[],
                );
            }
            for (key, verdict) in policy.choose_victims(1 << 30, &[]) {
                if verdict == EvictionVerdict::Evict {
                    evictions.push(key);
                    policy.remove(key, &[]);
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
            let decision = policy.record_hit(
                access.entry,
                cost(access.cold_prefill_cost, access.restore_cost),
            );
            policy_saved += (access.cold_prefill_cost - access.restore_cost).max(0.0);
            // Promotion is the persistence event for a probational entry:
            // the bytes written are the resident entry's retained size at
            // admission, not the current access's (re-sampled) size.
            if let Some(AdmissionDecision {
                kind: AdmissionDecisionKind::Promote,
                ..
            }) = &decision
            {
                let persisted = policy
                    .entry(access.entry)
                    .map(|e| e.exclusive_bytes)
                    .expect("promoted entry resident");
                policy_written += persisted;
            }
        } else {
            // Re-offer a rejected/evicted entry each time it recurs;
            // probation keeps new candidates in the accounting set until
            // pressure forces a choice.
            let decision = policy.consider_admission(
                access.entry,
                access.exclusive_bytes,
                vec![],
                cost(access.cold_prefill_cost, access.restore_cost),
                &[],
            );
            // Count bytes only when the decision actually persists.
            // AdmitProbation is memory-only accounting; the entry becomes
            // written when a later hit promotes it to Admitted (persist).
            if decision.kind == AdmissionDecisionKind::AdmitPersist {
                policy_written += access.exclusive_bytes;
            }
            // Hard probation cap is enforced as part of admission: commit the
            // selected victims.
            for key in decision.probation_cap_repair.victims().iter().copied() {
                policy.remove(key, &[]);
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
                    policy.remove(key, &[]);
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
/// large entries pollute an LRU, the benefit policy must STRICTLY save
/// more cold prefill cost or STRICTLY write fewer persisted bytes, with
/// the non-winning dimension held inside an explicit regression
/// tolerance.
#[test]
fn beats_lru_on_one_shot_pollution_trace() {
    let trace = traces::one_shot_trace(7, 2_000);
    let capacity = 64 << 20;
    let comparison = compare(&trace, capacity);
    const TOLERANCE: f64 = 0.05;
    let wins_saved = comparison.policy_saved_cost > comparison.lru_saved_cost;
    let wins_written = comparison.policy_bytes_written < comparison.lru_bytes_written;
    // Strict improvement in at least one dimension (#1650 acceptance).
    assert!(
        wins_saved || wins_written,
        "one-shot pollution @ {capacity}: saved policy={:.1} lru={:.1}, written policy={} lru={} — no strict improvement",
        comparison.policy_saved_cost,
        comparison.lru_saved_cost,
        comparison.policy_bytes_written,
        comparison.lru_bytes_written
    );
    // The non-winning dimension must stay inside an explicit regression
    // tolerance rather than degrading unboundedly.
    if !wins_saved {
        assert!(
            comparison.policy_saved_cost >= comparison.lru_saved_cost * (1.0 - TOLERANCE),
            "policy saved {} vs lru {} — saved cost regressed beyond {:.0}%",
            comparison.policy_saved_cost,
            comparison.lru_saved_cost,
            TOLERANCE * 100.0
        );
    }
    if !wins_written {
        assert!(
            (comparison.policy_bytes_written as f64)
                <= comparison.lru_bytes_written as f64 * (1.0 + TOLERANCE),
            "policy wrote {} vs lru {} — writes regressed beyond {:.0}%",
            comparison.policy_bytes_written,
            comparison.lru_bytes_written,
            TOLERANCE * 100.0
        );
    }
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

/// The mixed-size trace must actually exercise mixed resident size
/// classes and large-entry eviction at a capacity smaller than the
/// 256 MiB class.
#[test]
fn mixed_size_trace_has_real_mixed_residency_and_large_eviction() {
    let trace = traces::mixed_size_trace(5, 300);
    // Distinct keys per class, each consistently sized.
    let mut sizes = std::collections::BTreeMap::new();
    let mut classes = std::collections::BTreeSet::new();
    for a in &trace {
        let prev = sizes.insert(a.entry, a.exclusive_bytes);
        assert!(
            prev.is_none_or(|s| s == a.exclusive_bytes),
            "key {} changed size class",
            a.entry
        );
        classes.insert(a.exclusive_bytes);
    }
    assert_eq!(classes.len(), 3, "all three size classes present");
    // Run the trace at a capacity that cannot hold a 256 MiB entry
    // alongside the hot 4 MiB set: large entries must be admitted
    // (probation) and evicted, never violating capacity.
    let capacity = 96 << 20;
    let mut policy = BenefitPolicy::new(PolicyConfig::default());
    let mut resident_classes = std::collections::BTreeSet::new();
    for access in &trace {
        if policy.entry(access.entry).is_some() {
            policy.record_hit(
                access.entry,
                cost(access.cold_prefill_cost, access.restore_cost),
            );
        } else {
            let decision = policy.consider_admission(
                access.entry,
                access.exclusive_bytes,
                vec![],
                cost(access.cold_prefill_cost, access.restore_cost),
                &[],
            );
            for key in decision.probation_cap_repair.victims().iter().copied() {
                policy.remove(key, &[]);
            }
        }
        let mut used: u64 = policy.entries.values().map(|e| e.exclusive_bytes).sum();
        while used > capacity {
            let victims = policy.choose_victims(used - capacity, &[]);
            if victims.is_empty() {
                break;
            }
            for (key, verdict) in victims {
                if verdict == EvictionVerdict::Evict {
                    policy.remove(key, &[]);
                }
            }
            used = policy.entries.values().map(|e| e.exclusive_bytes).sum();
        }
        assert!(used <= capacity, "used {} over capacity after access", used);
        for e in policy.entries.values() {
            resident_classes.insert(e.exclusive_bytes);
        }
    }
    assert!(
        resident_classes.len() >= 2,
        "mixed resident sizes never observed: {:?}",
        resident_classes
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
        &[],
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
        policy.consider_admission(key, 1 << 20, vec![], cost(400.0, 100.0), &[]);
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
    policy.consider_admission(1, 0, vec![seg], cost(400.0, 100.0), &[]);
    policy.consider_admission(2, 0, vec![seg], cost(400.0, 100.0), &[]);
    policy.consider_admission(3, 0, vec![seg], cost(400.0, 100.0), &[]);
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
        policy.consider_admission(key, 1 << 20, vec![], cost(400.0, 100.0), &[]);
    }
    assert!(policy.probation_bytes() > 2 << 20);
    for key in policy.enforce_probation_cap() {
        policy.remove(key, &[]);
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
    policy.consider_admission(1, 1 << 20, vec![], cost(400.0, 100.0), &[]);
    assert!(policy.record_hit(1, cost(400.0, 100.0)).is_none()); // first hit
    policy.remove(1, &[]); // evicted before the second hit
    policy.consider_admission(1, 1 << 20, vec![], cost(400.0, 100.0), &[]);
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
        &[],
    );
    assert_eq!(decision.verdict, AdmissionVerdict::Reject);
    assert!(policy.is_empty());
    // A valid admission followed by an invalid hit must not poison state.
    policy.consider_admission(1, 1 << 20, vec![], cost(400.0, 100.0), &[]);
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
        let decision =
            policy.consider_admission(key, 0, vec![(1u64, 3 << 20)], cost(400.0, 100.0), &[]);
        for v in decision.probation_cap_repair.victims().iter().copied() {
            policy.remove(v, &[]);
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
        policy.consider_admission(key, 1 << 20, vec![], cost(400.0, 100.0), &[]);
        for (_, verdict) in policy.choose_victims(u64::MAX, &[]) {
            if verdict == EvictionVerdict::Evict {
                policy.remove(key, &[]);
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
    policy.consider_admission(99, 1 << 20, vec![], cost(400.0, 100.0), &[]);
    policy.remove(99, &[]);
    assert!(policy.ghost(99).is_some());
    for i in 0..50u64 {
        // Unique keys: an already-resident key is rejected before the clock
        // advances, so age must be driven by fresh observations.
        policy.consider_admission(1000 + i, 1 << 20, vec![], cost(400.0, 100.0), &[]);
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
        policy.consider_admission(key, 1 << 20, vec![], cost(400.0, 100.0), &[]);
    }
    let victims = policy.select_probation_cap_victims(&[]).victims().to_vec();
    assert!(!victims.is_empty());
    assert_eq!(policy.len(), 4, "selection must not remove entries");
    let decision = policy.consider_admission(5, 1 << 20, vec![], cost(400.0, 100.0), &[]);
    assert!(!decision.probation_cap_repair.victims().is_empty());
    for key in decision.probation_cap_repair.victims() {
        policy.remove(*key, &[]);
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
    policy.consider_admission(9, 1 << 20, vec![], cost(400.0, 100.0), &[]);
    policy.remove(9, &[]);
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
        &[],
    );
    assert_eq!(decision.verdict, AdmissionVerdict::Reject);
    assert_eq!(policy.clock_debug(), before_clock);
    assert!(policy.ghost(9).is_some());

    // Invalid hit: same invariants.
    policy.consider_admission(2, 1 << 20, vec![], cost(400.0, 100.0), &[]);
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
        policy.consider_admission(key, 0, vec![(1u64, 4)], cost(400.0, 100.0), &[]);
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
    let decision = policy.consider_admission(1, 0, vec![(1u64, 4)], cost(400.0, 100.0), &[]);
    assert!(!decision.probation_cap_repair.victims().is_empty());
}

#[test]
fn ghost_promoted_recurrence_returns_admit_persist() {
    let mut policy = BenefitPolicy::new(PolicyConfig::default());
    policy.consider_admission(1, 1 << 20, vec![], cost(400.0, 100.0), &[]);
    assert!(policy.record_hit(1, cost(400.0, 100.0)).is_none()); // hits = 1
    policy.remove(1, &[]);
    let decision = policy.consider_admission(1, 1 << 20, vec![], cost(400.0, 100.0), &[]);
    assert_eq!(decision.kind, AdmissionDecisionKind::AdmitPersist);
    assert_eq!(policy.entry(1).unwrap().state, PolicyEntryState::Admitted);
    // Non-promoted recurrence stays AdmitProbation/Probation.
    policy.consider_admission(2, 1 << 20, vec![], cost(400.0, 100.0), &[]);
    policy.remove(2, &[]);
    let decision = policy.consider_admission(2, 1 << 20, vec![], cost(400.0, 100.0), &[]);
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
    policy.consider_admission(1, 1 << 20, vec![], cost(400.0, 100.0), &[]);
    policy.remove(1, &[]);
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
            policy.consider_admission(key, 1 << 20, vec![(7u64, 9 << 20)], cost(400.0, 100.0), &[]);
        for v in decision.probation_cap_repair.victims().iter().copied() {
            policy.remove(v, &[]);
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
    let decision = policy.consider_admission(
        1,
        0,
        vec![(42u64, 100), (42u64, 100)],
        cost(400.0, 100.0),
        &[],
    );
    assert_eq!(decision.verdict, AdmissionVerdict::Reject);
    assert!(
        decision
            .reasons
            .contains(&"duplicate-segment-reference".to_string())
    );
    assert!(policy.is_empty());
    // Conflicting size for a known segment is also rejected.
    let mut policy = BenefitPolicy::new(PolicyConfig::default());
    policy.consider_admission(1, 0, vec![(42u64, 100)], cost(400.0, 100.0), &[]);
    let decision = policy.consider_admission(2, 0, vec![(42u64, 200)], cost(400.0, 100.0), &[]);
    assert_eq!(decision.verdict, AdmissionVerdict::Reject);
    assert!(
        decision
            .reasons
            .contains(&"segment-size-conflict".to_string())
    );
    // Same size for a known segment is fine.
    let decision = policy.consider_admission(3, 0, vec![(42u64, 100)], cost(400.0, 100.0), &[]);
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
        policy.consider_admission(key, 1 << 20, vec![(7u64, 8 << 20)], cost(400.0, 100.0), &[]);
    }
    let before = policy.probation_bytes();
    assert!(before > 9 << 20, "before {}", before);
    // Now select against the real cap by constructing the over-cap state
    // through the public API: reset the budget by direct selection.
    let victims = policy
        .with_probation_budget(9 << 20, |p| p.select_probation_cap_victims(&[]))
        .victims()
        .to_vec();
    assert!(!victims.is_empty());
    for key in &victims {
        policy.remove(*key, &[]);
    }
    let after = policy.probation_bytes();
    assert!(
        after <= 9 << 20,
        "committed class {} still over cap after victims {:?}",
        after,
        victims
    );
}

#[test]
fn conflict_on_a_later_segment_leaves_state_untouched() {
    // Admit segment 1 at size 100, then offer a new key whose later segment
    // conflicts: [(2,100),(1,200)]. The reject must leave the ledger without
    // segment 2, no entry for the rejected key, and any matching ghost intact.
    let mut policy = BenefitPolicy::new(PolicyConfig::default());
    policy.consider_admission(1, 0, vec![(1u64, 100)], cost(400.0, 100.0), &[]);

    // Seed a ghost for the key that will be rejected so ghost survival is
    // observable.
    policy.consider_admission(9, 1 << 20, vec![], cost(400.0, 100.0), &[]);
    policy.remove(9, &[]);
    assert!(policy.ghost(9).is_some());

    let decision = policy.consider_admission(
        9,
        1 << 20,
        vec![(2u64, 100), (1u64, 200)],
        cost(400.0, 100.0),
        &[],
    );
    assert_eq!(decision.verdict, AdmissionVerdict::Reject);
    assert!(
        decision
            .reasons
            .contains(&"segment-size-conflict".to_string())
    );
    // No entry for the rejected key; the ghost survived the reject.
    assert!(policy.entry(9).is_none());
    assert!(
        policy.ghost(9).is_some(),
        "rejected re-offer must not lose the ghost"
    );
    // Segment 2 was never registered.
    assert!(policy.segments.segment_record(2).is_none());
    // Segment 1 still has exactly one reference (the original entry).
    let record = policy.segments.segment_record(1).expect("segment 1 intact");
    assert_eq!(record.references, std::collections::BTreeSet::from([1u64]));
    assert_eq!(record.size, 100);
    // Class charge unchanged: the original entry's shared 100 bytes only.
    assert_eq!(policy.probation_bytes(), 100);
}

#[test]
fn structural_rejects_leave_clock_and_zero_horizon_ghosts_untouched() {
    // ghost_max_age_observations = 0: any real observation would expire the
    // ghost. Duplicate-ID and size-conflict rejects must not.
    let mk = || {
        BenefitPolicy::new(PolicyConfig {
            ghost_max_age_observations: 0,
            grace_observations: 4,
            ..PolicyConfig::default()
        })
    };

    // Duplicate segment IDs.
    let mut policy = mk();
    policy.consider_admission(9, 1 << 20, vec![], cost(400.0, 100.0), &[]);
    policy.remove(9, &[]);
    assert!(policy.ghost(9).is_some());
    let clock = policy.clock_debug();
    let decision = policy.consider_admission(
        1,
        0,
        vec![(42u64, 100), (42u64, 100)],
        cost(400.0, 100.0),
        &[],
    );
    assert_eq!(decision.verdict, AdmissionVerdict::Reject);
    assert_eq!(
        policy.clock_debug(),
        clock,
        "duplicate reject must not advance clock"
    );
    assert!(
        policy.ghost(9).is_some(),
        "duplicate reject must not expire ghosts"
    );

    // Size conflict on a later segment.
    let mut policy = mk();
    policy.consider_admission(1, 0, vec![(1u64, 100)], cost(400.0, 100.0), &[]);
    policy.consider_admission(9, 1 << 20, vec![], cost(400.0, 100.0), &[]);
    policy.remove(9, &[]);
    assert!(policy.ghost(9).is_some());
    let clock = policy.clock_debug();
    let decision = policy.consider_admission(
        9,
        1 << 20,
        vec![(2u64, 100), (1u64, 200)],
        cost(400.0, 100.0),
        &[],
    );
    assert_eq!(decision.verdict, AdmissionVerdict::Reject);
    assert_eq!(
        policy.clock_debug(),
        clock,
        "conflict reject must not advance clock"
    );
    assert!(
        policy.ghost(9).is_some(),
        "conflict reject must not expire ghosts"
    );
}

#[test]
fn already_resident_key_rejection_leaves_no_stale_references() {
    let mut policy = BenefitPolicy::new(PolicyConfig::default());
    policy.consider_admission(1, 0, vec![(1u64, 100)], cost(400.0, 100.0), &[]);
    // Re-admitting a resident key is rejected and must not swap the entry's
    // segment list or leak old ledger references.
    let decision = policy.consider_admission(1, 0, vec![(2u64, 50)], cost(400.0, 100.0), &[]);
    assert_eq!(decision.verdict, AdmissionVerdict::Reject);
    assert!(
        decision
            .reasons
            .contains(&"already-resident-key".to_string())
    );
    let record = policy.segments.segment_record(1).expect("segment 1 intact");
    assert_eq!(record.references, std::collections::BTreeSet::from([1u64]));
    assert!(
        policy.segments.segment_record(2).is_none(),
        "segment 2 must not be registered"
    );
    assert_eq!(policy.probation_bytes(), 100);
}

#[test]
fn admitted_coreference_removal_repairs_the_probation_cap() {
    // Ghost-promoted admitted A on shared segment S; probation P shares S
    // (half-share fits the cap); one hit on P (still probation at
    // threshold 2); remove A -> P's charge rises to all of S and exceeds
    // the cap. The removal response must carry P as a cap victim.
    let mut policy = BenefitPolicy::new(PolicyConfig {
        probation_byte_budget: 1 << 20, // 1 MiB cap; S is 1.5 MiB
        grace_observations: 0,
        ..PolicyConfig::default()
    });
    // Build A as ghost-promoted: admit, one hit, evict (ghost), recur.
    policy.consider_admission(1, 0, vec![(7u64, (3 << 20) / 2)], cost(400.0, 100.0), &[]);
    policy.record_hit(1, cost(400.0, 100.0));
    policy.remove_without_cap_repair(1);
    let decision =
        policy.consider_admission(1, 0, vec![(7u64, (3 << 20) / 2)], cost(400.0, 100.0), &[]);
    assert_eq!(decision.kind, AdmissionDecisionKind::AdmitPersist);
    for v in decision.probation_cap_repair.victims().iter().copied() {
        policy.remove_without_cap_repair(v);
    }
    assert_eq!(policy.entry(1).unwrap().state, PolicyEntryState::Admitted);

    // P shares S: half-share = 0.75 MiB fits the 1 MiB cap.
    let decision =
        policy.consider_admission(2, 0, vec![(7u64, (3 << 20) / 2)], cost(400.0, 100.0), &[]);
    assert_eq!(decision.verdict, AdmissionVerdict::Admit);
    for v in decision.probation_cap_repair.victims().iter().copied() {
        policy.remove_without_cap_repair(v);
    }
    // One hit on P: still probation at threshold 2.
    policy.record_hit(2, cost(400.0, 100.0));
    assert_eq!(policy.entry(2).unwrap().state, PolicyEntryState::Probation);
    assert!(
        policy.probation_bytes() <= 1 << 20,
        "half-share state over cap: {}",
        policy.probation_bytes()
    );

    // Remove the admitted co-reference: P's charge rises to all of S.
    let outcome = policy.remove(1, &[]).expect("A removed");
    assert!(
        !outcome.probation_cap_repair.victims().is_empty(),
        "removal must carry cap victims for the risen share"
    );
    for v in outcome.probation_cap_repair.victims() {
        policy.remove_without_cap_repair(*v);
    }
    assert!(
        policy.probation_bytes() <= 1 << 20,
        "class {} still over cap after repair",
        policy.probation_bytes()
    );
    assert!(policy.entry(2).is_none(), "P must be the repair victim");
}

#[test]
fn cap_repair_prefers_unpinned_over_older_pinned_probationer() {
    // Older pinned probationer (key 1) must never be selected while the
    // younger unpinned probationer (key 2) can repair the cap (#1650).
    let mut policy = BenefitPolicy::new(PolicyConfig {
        probation_byte_budget: 1 << 20,
        ..PolicyConfig::default()
    });
    for key in 1..=2u64 {
        let decision = policy.consider_admission(key, 1 << 20, vec![], cost(400.0, 100.0), &[1]);
        // Do not commit yet: build the over-cap state first.
        assert!(
            !decision.probation_cap_repair.victims().contains(&1),
            "admission never selects a pin"
        );
    }
    let repair = policy.select_probation_cap_victims(&[1]);
    assert!(
        !repair.victims().contains(&1),
        "pinned key selected as victim: {:?}",
        repair.victims()
    );
    assert!(repair.victims().contains(&2));
    assert!(!repair.is_deferred());
    for v in repair.victims().iter().copied() {
        policy.remove(v, &[1]);
    }
    assert!(policy.probation_bytes() <= 1 << 20);
    assert!(policy.entries.contains_key(&1), "pin must survive repair");
}

#[test]
fn all_pinned_cap_is_deferred_with_shortfall_never_a_pin() {
    let mut policy = BenefitPolicy::new(PolicyConfig {
        probation_byte_budget: 1 << 20,
        ..PolicyConfig::default()
    });
    let pinned = [1u64, 2];
    for (n, key) in pinned.iter().enumerate() {
        let decision =
            policy.consider_admission(*key, 1 << 20, vec![], cost(400.0, 100.0), &pinned);
        let repair = &decision.probation_cap_repair;
        assert!(
            repair.victims().is_empty(),
            "all candidates pinned: no victim may be selected"
        );
        // First admission is under cap; the second pushes the all-pinned
        // class over cap with no selectable candidate.
        if n == 1 {
            assert!(
                repair.is_deferred(),
                "unsatisfiable cap must be Deferred, got {:?}",
                repair
            );
            assert_eq!(repair.shortfall_bytes(), 1 << 20);
        }
        assert!(
            !repair.victims().contains(&1) && !repair.victims().contains(&2),
            "pins never selected even when cap is unsatisfiable"
        );
    }
    // The removal path honors pins identically: a caller-forced removal of
    // the pinned key 1 leaves key 2 at exactly the cap — satisfied, no pin
    // selected.
    let outcome = policy.remove(1, &pinned).expect("entry 1 removed");
    assert!(!outcome.probation_cap_repair.is_deferred());
    assert!(outcome.probation_cap_repair.victims().is_empty());
    assert!(policy.entries.contains_key(&2));
    // Once the pin releases, repair becomes satisfiable again (key 2 is
    // selectable again).
    let repair = policy.select_probation_cap_victims(&[]);
    assert!(!repair.is_deferred());
}

/// Grace must expire under repeated misses: a miss is a real observation
/// that advances the clock but is a negative value signal (no recency
/// refresh), so a miss-only stream ages an entry out of grace and makes
/// it evictable instead of holding it indefinitely.
#[test]
fn grace_expires_under_repeated_misses() {
    let grace = 8u64;
    let mut policy = BenefitPolicy::new(PolicyConfig {
        grace_observations: grace,
        ..PolicyConfig::default()
    });
    policy.consider_admission(1, 1 << 20, vec![], cost(400.0, 100.0), &[]);
    policy.consider_admission(2, 1 << 20, vec![], cost(400.0, 100.0), &[]);
    // Miss key 1 seven times (each miss advances the clock): the clock at
    // admission was 2, so key 1's age becomes 8 (out of grace) while key
    // 2's age becomes 7 (still in grace) — exactly one observation on
    // either side of the grace boundary, not an ordering artifact.
    let clock_at_admission = policy.clock_debug();
    for _ in 0..grace - 1 {
        policy.record_miss(1);
    }
    assert!(
        policy.clock_debug() == clock_at_admission + grace - 1,
        "misses must advance the clock"
    );
    let e1 = policy.entry(1).unwrap();
    let e2 = policy.entry(2).unwrap();
    assert_eq!(
        policy.clock_debug().saturating_sub(e1.last_observation),
        grace,
        "key 1 exactly out of grace"
    );
    assert_eq!(
        policy.clock_debug().saturating_sub(e2.last_observation),
        grace - 1,
        "key 2 exactly one observation inside grace"
    );
    // Out-of-grace entry 1 is now evictable in pass 1 (grace honored);
    // requesting only its bytes must never touch in-grace entry 2.
    let victims = policy.choose_victims(1 << 20, &[]);
    let evictable: Vec<u64> = victims.iter().map(|(k, _)| *k).collect();
    assert!(
        evictable.contains(&1),
        "miss-only entry must age out of grace, victims {:?}",
        evictable
    );
    assert!(
        !evictable.contains(&2),
        "in-grace entry must not be evicted first, victims {:?}",
        evictable
    );
    // A miss-only stream cannot hold grace forever — already asserted above:
    // entry 1's age is exactly `grace` (evictable) after only misses.
}
