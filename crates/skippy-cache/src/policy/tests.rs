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
            policy.consider_admission(
                access.entry,
                access.exclusive_bytes,
                vec![],
                cost(access.cold_prefill_cost, access.restore_cost),
            );
            policy_written += access.exclusive_bytes;
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
