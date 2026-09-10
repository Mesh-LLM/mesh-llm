//! Admission, probation, promotion, and eviction selection (#1650 first
//! slice). Pure decision logic over `BenefitPolicy` state.

use serde::Serialize;

use crate::policy::{
    BenefitPolicy, CostSample, EntryKey, EvictionVerdict, GhostStats, PolicyEntry, SegmentId,
};

/// What the policy decided to do with a candidate or admitted entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum AdmissionDecisionKind {
    AdmitProbation,
    AdmitPersist,
    Promote,
    Reject,
}

impl AdmissionDecision {
    /// A rejected offer whose measured cost was invalid. Constructed without
    /// touching policy state.
    pub fn rejected_invalid_cost() -> Self {
        Self {
            kind: AdmissionDecisionKind::Reject,
            verdict: AdmissionVerdict::Reject,
            reasons: vec!["invalid-cost-sample".into()],
            probation_cap_victims: Vec::new(),
        }
    }
}

/// The verdict plus opaque reasons for logging.
#[derive(Debug, Clone, PartialEq)]
pub struct AdmissionDecision {
    pub kind: AdmissionDecisionKind,
    pub verdict: AdmissionVerdict,
    pub reasons: Vec<String>,
    /// Keys the caller must physically evict to keep the probation class
    /// under its hard byte cap after this admission. Empty when the cap
    /// holds. The policy does not remove them itself: committed removal
    /// stays with `BenefitPolicy::remove`.
    pub probation_cap_victims: Vec<crate::policy::EntryKey>,
}

/// Lifecycle state of a policy entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyEntryState {
    /// Resident in memory only; not persisted to disk.
    Probation,
    /// Persist-eligible: survived the hit threshold.
    Admitted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmissionVerdict {
    Admit,
    Reject,
}

pub(crate) fn consider(
    policy: &mut BenefitPolicy,
    key: EntryKey,
    exclusive_bytes: u64,
    shared: Vec<(SegmentId, u64)>,
    cost: CostSample,
) -> AdmissionDecision {
    let reasons: Vec<String>;
    let kind;

    let mut seen = std::collections::BTreeSet::new();
    let duplicate_segment = shared.iter().any(|(s, _)| !seen.insert(*s));
    if !cost.is_valid() {
        // Invalid measured costs (NaN/infinite/negative) are rejected before
        // any mutation: they must never enter entry state.
        kind = AdmissionDecisionKind::Reject;
        reasons = vec!["invalid-cost-sample".into()];
    } else if duplicate_segment {
        kind = AdmissionDecisionKind::Reject;
        reasons = vec!["duplicate-segment-reference".into()];
    } else if cost.net_benefit() <= 0.0 {
        kind = AdmissionDecisionKind::Reject;
        reasons = vec!["no-net-benefit".into()];
    } else if exclusive_bytes == 0 && shared.is_empty() {
        // Zero-footprint entries are free to keep.
        kind = AdmissionDecisionKind::AdmitProbation;
        reasons = vec!["zero-exclusive-bytes".into()];
    } else {
        kind = AdmissionDecisionKind::AdmitProbation;
        reasons = vec!["probation-new-entry".into()];
    }

    if matches!(kind, AdmissionDecisionKind::Reject) {
        return AdmissionDecision {
            kind,
            verdict: AdmissionVerdict::Reject,
            reasons,
            probation_cap_victims: Vec::new(),
        };
    }
    // Carry ghost history in: a recurring entry re-enters with its past
    // reuse signal, and an entry whose history already clears the hit
    // threshold admits straight to the admitted class (equivalent value
    // signal).
    let had_ghost = policy.ghosts.contains_key(&key);
    let mut ghost = policy.ghosts.remove(&key).unwrap_or(GhostStats {
        hits: 0,
        reuse_weight: 0.0,
        observation_weight: 0.0,
        last_observation: 0,
    });
    if had_ghost {
        // This offer *is* a recurrence: count it as a reuse observation, the
        // same way record_hit would, so eviction cannot erase the value
        // signal the recurrence just demonstrated.
        ghost.hits += 1;
        ghost.reuse_weight = ghost.reuse_weight * policy.config.decay.factor + 1.0;
        ghost.observation_weight = ghost.observation_weight * policy.config.decay.factor + 1.0;
    }
    let ghost_promoted = ghost.hits >= policy.config.persistence_hit_threshold as u64;
    let state = if ghost_promoted {
        PolicyEntryState::Admitted
    } else {
        PolicyEntryState::Probation
    };
    // The decision kind must match the resulting entry state so a
    // store-facing caller persists exactly what the policy admitted.
    let kind = if ghost_promoted {
        AdmissionDecisionKind::AdmitPersist
    } else {
        kind
    };
    let segment_ids: Vec<SegmentId> = shared.iter().map(|(s, _)| *s).collect();
    for (segment, size) in &shared {
        if let Some(record) = policy.segments.segment_record(*segment)
            && !record.references.contains(&key)
            && record.size != *size
        {
            // Conflicting physical size for the same segment identity.
            return AdmissionDecision {
                kind: AdmissionDecisionKind::Reject,
                verdict: AdmissionVerdict::Reject,
                reasons: vec!["segment-size-conflict".into()],
                probation_cap_victims: Vec::new(),
            };
        }
        policy.segments.add(*segment, *size, key);
    }
    policy.entries.insert(
        key,
        PolicyEntry {
            state,
            hits: ghost.hits,
            misses: 0,
            reuse_weight: ghost.reuse_weight,
            observation_weight: ghost.observation_weight,
            last_observation: policy.clock,
            last_cost: Some(cost),
            exclusive_bytes,
            segments: segment_ids,
        },
    );
    AdmissionDecision {
        kind,
        verdict: AdmissionVerdict::Admit,
        reasons,
        probation_cap_victims: Vec::new(),
    }
}

pub(crate) fn record_hit(
    policy: &mut BenefitPolicy,
    key: EntryKey,
    cost: CostSample,
) -> Option<AdmissionDecision> {
    if !cost.is_valid() {
        // Invalid measured costs never mutate entry state.
        return None;
    }
    let entry = policy.entries.get_mut(&key)?;
    entry.hits += 1;
    entry.last_observation = policy.clock;
    entry.last_cost = Some(cost);
    entry.reuse_weight = entry.reuse_weight * policy.config.decay.factor + 1.0;
    entry.observation_weight = entry.observation_weight * policy.config.decay.factor + 1.0;

    if entry.state == PolicyEntryState::Probation
        && entry.hits >= policy.config.persistence_hit_threshold as u64
    {
        entry.state = PolicyEntryState::Admitted;
        return Some(AdmissionDecision {
            kind: AdmissionDecisionKind::Promote,
            verdict: AdmissionVerdict::Admit,
            reasons: vec!["probation-second-hit".into()],
            probation_cap_victims: Vec::new(),
        });
    }
    None
}

pub(crate) fn choose_victims(
    policy: &mut BenefitPolicy,
    bytes_to_free: u64,
    pinned: &[EntryKey],
) -> Vec<(EntryKey, EvictionVerdict)> {
    let grace = policy.config.grace_observations;
    let clock = policy.clock;
    let in_grace = |e: &PolicyEntry| clock.saturating_sub(e.last_observation) < grace;

    // NaN-free scores only (score::compute rejects invalid costs/config), so
    // a total ordering is safe: finite values ordered normally, keys
    // tie-break. We still avoid partial_cmp().unwrap() defensively.
    let mut candidates: Vec<(f64, EntryKey)> = policy
        .entries
        .iter()
        .filter(|(k, _)| !pinned.contains(k))
        .filter_map(|(k, e)| {
            let score = super::score::compute(&policy.config, *k, e, &policy.segments)?;
            Some((score.value, *k))
        })
        .collect();
    candidates.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

    // Marginal physical release for `key` given `selected` victims already
    // chosen: exclusive bytes plus segments whose remaining references are
    // all inside the selected victim set (or the entry itself).
    let marginal_release =
        |key: EntryKey, selected: &std::collections::BTreeSet<EntryKey>| -> u64 {
            let entry = policy.entries.get(&key).expect("candidate present");
            let mut release = entry.exclusive_bytes;
            for segment in &entry.segments {
                if let Some(record) = policy.segments.segment_record(*segment) {
                    let others_kept = record
                        .references
                        .iter()
                        .any(|r| *r != key && !selected.contains(r));
                    if !others_kept {
                        release += record.size;
                    }
                }
            }
            release
        };

    // Eviction order: no-hit probationers first (they have not earned
    // bytes), oldest observation first — LRU within the probation class so a
    // recently offered candidate is not the automatic first victim. Then
    // ascending benefit score. Grace waives under a hard capacity request:
    // the byte budget always wins.
    let mut order: Vec<(u64, EntryKey)> = policy
        .entries
        .iter()
        .filter(|(k, e)| {
            !pinned.contains(k) && e.state == PolicyEntryState::Probation && e.hits == 0
        })
        .map(|(k, e)| (e.last_observation, *k))
        .collect();
    order.sort(); // (oldest observation, key): deterministic
    let mut ordered_keys: Vec<EntryKey> = order.into_iter().map(|(_, k)| k).collect();
    ordered_keys.extend(candidates.into_iter().map(|(_, key)| key));
    let order = ordered_keys;

    let mut freed: u64 = 0;
    let mut victims = Vec::new();
    let mut selected: std::collections::BTreeSet<EntryKey> = Default::default();
    // Pass 1: grace honored. Pass 2 (only if the hard budget is still unmet):
    // grace waived — the hard byte budget always wins under pressure.
    for waive_grace in [false, true] {
        for key in &order {
            if freed >= bytes_to_free {
                break;
            }
            if selected.contains(key) || !policy.entries.contains_key(key) {
                continue; // dedup: probation-first and scored paths overlap
            }
            let evictable = waive_grace || policy.entries.get(key).is_some_and(|e| !in_grace(e));
            if !evictable {
                continue;
            }
            let release = marginal_release(*key, &selected);
            selected.insert(*key);
            victims.push((*key, EvictionVerdict::Evict));
            freed += release;
        }
        if freed >= bytes_to_free {
            break;
        }
    }
    victims
}
