//! Admission, probation, promotion, and eviction selection (#1650 first
//! slice). Pure decision logic over `BenefitPolicy` state.

use serde::Serialize;

use crate::policy::{
    BenefitPolicy, CostSample, DecisionReason, EntryKey, EvictionVerdict, PolicyEntry, SegmentId,
};

/// What the policy decided to do with a candidate or admitted entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum AdmissionDecisionKind {
    AdmitProbation,
    AdmitPersist,
    Promote,
    Reject,
}

/// The verdict plus opaque reasons for logging.
#[derive(Debug, Clone, PartialEq)]
pub struct AdmissionDecision {
    pub kind: AdmissionDecisionKind,
    pub verdict: AdmissionVerdict,
    pub reasons: Vec<String>,
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

    if cost.net_benefit() <= 0.0 {
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
        };
    }

    let segment_ids: Vec<SegmentId> = shared.iter().map(|(s, _)| *s).collect();
    for (segment, size) in &shared {
        policy.segments.add(*segment, *size, key);
    }
    policy.entries.insert(
        key,
        PolicyEntry {
            state: PolicyEntryState::Probation,
            hits: 0,
            misses: 0,
            reuse_weight: 0.0,
            observation_weight: 0.0,
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
    }
}

pub(crate) fn record_hit(
    policy: &mut BenefitPolicy,
    key: EntryKey,
    cost: CostSample,
) -> Option<AdmissionDecision> {
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
        });
    }
    None
}

pub(crate) fn choose_victims(
    policy: &mut BenefitPolicy,
    bytes_to_free: u64,
    pinned: &[EntryKey],
) -> Vec<(EntryKey, EvictionVerdict)> {
    let mut candidates: Vec<(f64, EntryKey, f64)> = policy
        .entries
        .iter()
        .filter(|(k, _)| !pinned.contains(k))
        .filter_map(|(k, e)| {
            let score = super::score::compute(&policy.config, *k, e, &policy.segments)?;
            let footprint =
                e.exclusive_bytes as f64 + policy.segments.fractional_bytes(*k, &e.segments);
            Some((score.value, *k, footprint))
        })
        .collect();
    // Deterministic: ascending score, ascending key tie-break.
    candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));

    // Probation entries with no hits are always droppable first: they have
    // not yet earned bytes under second-hit admission.
    let mut freed = 0.0f64;
    let mut victims = Vec::new();
    let grace = policy.config.grace_observations;
    let clock = policy.clock;
    let in_grace = |e: &PolicyEntry| clock.saturating_sub(e.last_observation) < grace;
    let probation_first: Vec<EntryKey> = policy
        .entries
        .iter()
        .filter(|(k, e)| {
            !pinned.contains(k)
                && e.state == PolicyEntryState::Probation
                && e.hits == 0
                && !in_grace(e)
        })
        .map(|(k, _)| *k)
        .collect();
    for key in probation_first {
        if freed >= bytes_to_free as f64 {
            break;
        }
        let entry = policy.entries.get(&key).unwrap();
        let footprint =
            entry.exclusive_bytes as f64 + policy.segments.fractional_bytes(key, &entry.segments);
        victims.push((key, EvictionVerdict::Evict));
        freed += footprint;
    }
    for (_score, key, footprint) in candidates {
        if freed >= bytes_to_free as f64 {
            break;
        }
        if policy.entries.get(&key).is_some_and(in_grace) {
            continue; // grace window: not yet evictable
        }
        victims.push((key, EvictionVerdict::Evict));
        freed += footprint;
    }
    victims
}

#[allow(dead_code)]
fn reason(
    entry: EntryKey,
    decision: AdmissionDecisionKind,
    reasons: Vec<String>,
    score: Option<f64>,
) -> DecisionReason {
    DecisionReason {
        entry,
        decision,
        reasons,
        score,
    }
}
