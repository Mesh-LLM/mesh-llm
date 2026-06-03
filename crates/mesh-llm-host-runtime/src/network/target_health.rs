//! Local outcome-aware target health for routing decisions.
//!
//! This state is intentionally process-local. It helps the local proxy avoid a
//! target that just timed out or repeatedly failed, but it is not a mesh
//! protocol signal, not cryptographic trust, and should not be gossiped.

use crate::inference::election::InferenceTarget;
use serde::Serialize;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const DEFAULT_BASE_COOLDOWN: Duration = Duration::from_secs(30);
const DEFAULT_MAX_COOLDOWN: Duration = Duration::from_secs(5 * 60);
const DEFAULT_MAX_ENTRIES: usize = 2048;
const DEFAULT_REPUTATION_TTL: Duration = Duration::from_secs(20 * 60);
const DEFAULT_REPUTATION_RECOVERY_SUCCESSES: u32 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TargetHealthOutcome {
    Success,
    Timeout,
    Unavailable,
    ContextOverflow,
    Rejected,
    ClientDisconnected,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct TargetKey {
    model: String,
    target: InferenceTarget,
}

#[derive(Clone, Debug)]
struct TargetEntry {
    failures: u32,
    cool_until: Instant,
}

#[derive(Clone, Debug)]
struct ReputationEntry {
    penalty: u32,
    recovery_successes: u32,
    last_observed: Instant,
}

#[derive(Clone, Copy, Debug)]
struct TargetHealthConfig {
    base_cooldown: Duration,
    max_cooldown: Duration,
    max_entries: usize,
    reputation_ttl: Duration,
    reputation_recovery_successes: u32,
}

impl Default for TargetHealthConfig {
    fn default() -> Self {
        Self {
            base_cooldown: DEFAULT_BASE_COOLDOWN,
            max_cooldown: DEFAULT_MAX_COOLDOWN,
            max_entries: DEFAULT_MAX_ENTRIES,
            reputation_ttl: DEFAULT_REPUTATION_TTL,
            reputation_recovery_successes: DEFAULT_REPUTATION_RECOVERY_SUCCESSES,
        }
    }
}

#[derive(Default)]
struct TargetHealthState {
    entries: HashMap<TargetKey, TargetEntry>,
    reputation: HashMap<TargetKey, ReputationEntry>,
    lru: VecDeque<TargetKey>,
    routes_avoided: u64,
    routes_penalized: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct TargetReputationStats {
    pub penalized_targets: usize,
    pub routes_penalized: u64,
}

#[cfg(test)]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct TargetHealthSnapshot {
    pub cooling_targets: usize,
    pub routes_avoided: u64,
    pub reputation: TargetReputationStats,
}

#[derive(Clone)]
pub(crate) struct TargetHealth {
    inner: Arc<Mutex<TargetHealthState>>,
    config: TargetHealthConfig,
}

impl Default for TargetHealth {
    fn default() -> Self {
        Self::new()
    }
}

impl TargetHealth {
    pub(crate) fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(TargetHealthState::default())),
            config: TargetHealthConfig::default(),
        }
    }

    #[cfg(test)]
    fn with_config(base_cooldown: Duration, max_cooldown: Duration, max_entries: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(TargetHealthState::default())),
            config: TargetHealthConfig {
                base_cooldown,
                max_cooldown,
                max_entries,
                reputation_ttl: DEFAULT_REPUTATION_TTL,
                reputation_recovery_successes: DEFAULT_REPUTATION_RECOVERY_SUCCESSES,
            },
        }
    }

    pub(crate) fn record_outcome(
        &self,
        model: Option<&str>,
        target: &InferenceTarget,
        outcome: TargetHealthOutcome,
    ) {
        if matches!(target, InferenceTarget::None) {
            return;
        }
        let Some(model) = normalized_model(model) else {
            return;
        };
        let key = TargetKey {
            model,
            target: target.clone(),
        };
        let mut state = self.inner.lock().unwrap();
        let now = Instant::now();
        state.prune_expired(now, self.config);

        match outcome {
            TargetHealthOutcome::Success => {
                state.remove_key(&key);
                state.record_reputation_success(&key, now, self.config);
            }
            TargetHealthOutcome::Timeout | TargetHealthOutcome::Unavailable => {
                state.record_failure(key.clone(), now, self.config);
                state.record_reputation_penalty(key, now, 1, self.config);
            }
            TargetHealthOutcome::ContextOverflow
            | TargetHealthOutcome::Rejected
            | TargetHealthOutcome::ClientDisconnected => {}
        }
    }

    pub(crate) fn eligible_candidates(
        &self,
        model: &str,
        candidates: &[InferenceTarget],
    ) -> Vec<InferenceTarget> {
        self.eligible_candidates_inner(model, candidates, true)
    }

    pub(crate) fn strict_eligible_candidates(
        &self,
        model: &str,
        candidates: &[InferenceTarget],
    ) -> Vec<InferenceTarget> {
        self.eligible_candidates_inner(model, candidates, false)
    }

    fn eligible_candidates_inner(
        &self,
        model: &str,
        candidates: &[InferenceTarget],
        preserve_availability: bool,
    ) -> Vec<InferenceTarget> {
        if preserve_availability && candidates.len() <= 1 {
            return candidates.to_vec();
        }
        let Some(model) = normalized_model(Some(model)) else {
            return candidates.to_vec();
        };
        let now = Instant::now();
        let mut state = self.inner.lock().unwrap();
        state.prune_expired(now, self.config);

        let mut eligible = Vec::with_capacity(candidates.len());
        let mut cooling = 0usize;
        for candidate in candidates {
            let key = TargetKey {
                model: model.clone(),
                target: candidate.clone(),
            };
            if state.is_cooling(&key, now) {
                cooling += 1;
            } else {
                eligible.push(candidate.clone());
            }
        }

        if cooling == 0 && state.no_reputation_penalties(&model, candidates) {
            candidates.to_vec()
        } else if preserve_availability && !has_routable_candidate(&eligible) {
            state.reputation_ordered_candidates(&model, candidates, now)
        } else {
            state.routes_avoided = state.routes_avoided.saturating_add(cooling as u64);
            state.reputation_ordered_candidates(&model, &eligible, now)
        }
    }

    pub(crate) fn reputation_stats(&self) -> TargetReputationStats {
        let mut state = self.inner.lock().unwrap();
        state.prune_expired(Instant::now(), self.config);
        state.reputation_stats()
    }

    #[cfg(test)]
    pub(crate) fn snapshot(&self) -> TargetHealthSnapshot {
        let mut state = self.inner.lock().unwrap();
        state.prune_expired(Instant::now(), self.config);
        TargetHealthSnapshot {
            cooling_targets: state.entries.len(),
            routes_avoided: state.routes_avoided,
            reputation: state.reputation_stats(),
        }
    }
}

impl TargetHealthState {
    fn record_failure(&mut self, key: TargetKey, now: Instant, config: TargetHealthConfig) {
        let failures = self
            .entries
            .get(&key)
            .map(|entry| entry.failures.saturating_add(1))
            .unwrap_or(1);
        let cooldown = cooldown_for_failure(failures, config);
        self.entries.insert(
            key.clone(),
            TargetEntry {
                failures,
                cool_until: now + cooldown,
            },
        );
        self.touch_key(&key);
        self.prune_over_capacity(config.max_entries);
    }

    fn is_cooling(&self, key: &TargetKey, now: Instant) -> bool {
        self.entries
            .get(key)
            .map(|entry| entry.cool_until > now)
            .unwrap_or(false)
    }

    fn record_reputation_penalty(
        &mut self,
        key: TargetKey,
        now: Instant,
        penalty: u32,
        config: TargetHealthConfig,
    ) {
        let entry = self
            .reputation
            .entry(key.clone())
            .or_insert_with(|| ReputationEntry {
                penalty: 0,
                recovery_successes: 0,
                last_observed: now,
            });
        entry.penalty = entry.penalty.saturating_add(penalty).min(16);
        entry.recovery_successes = 0;
        entry.last_observed = now;
        self.touch_key(&key);
        self.prune_over_capacity(config.max_entries);
    }

    fn record_reputation_success(
        &mut self,
        key: &TargetKey,
        now: Instant,
        config: TargetHealthConfig,
    ) {
        let Some(entry) = self.reputation.get_mut(key) else {
            return;
        };
        entry.last_observed = now;
        entry.recovery_successes = entry.recovery_successes.saturating_add(1);
        if entry.recovery_successes < config.reputation_recovery_successes {
            return;
        }
        entry.recovery_successes = 0;
        entry.penalty = entry.penalty.saturating_sub(1);
        if entry.penalty == 0 {
            self.reputation.remove(key);
            self.lru.retain(|existing| existing != key);
        }
    }

    fn prune_expired(&mut self, now: Instant, config: TargetHealthConfig) {
        let expired: Vec<TargetKey> = self
            .entries
            .iter()
            .filter_map(|(key, entry)| (entry.cool_until <= now).then_some(key.clone()))
            .collect();
        for key in expired {
            self.remove_key(&key);
        }
        let stale_reputation: Vec<TargetKey> = self
            .reputation
            .iter()
            .filter_map(|(key, entry)| {
                (now.duration_since(entry.last_observed) >= config.reputation_ttl)
                    .then_some(key.clone())
            })
            .collect();
        for key in stale_reputation {
            self.reputation.remove(&key);
            if !self.entries.contains_key(&key) {
                self.lru.retain(|existing| existing != &key);
            }
        }
    }

    fn prune_over_capacity(&mut self, max_entries: usize) {
        while self.entries.len().max(self.reputation.len()) > max_entries {
            let Some(key) = self.lru.pop_front() else {
                break;
            };
            self.entries.remove(&key);
            self.reputation.remove(&key);
        }
    }

    fn touch_key(&mut self, key: &TargetKey) {
        self.lru.retain(|existing| existing != key);
        self.lru.push_back(key.clone());
    }

    fn remove_key(&mut self, key: &TargetKey) {
        self.entries.remove(key);
        if !self.reputation.contains_key(key) {
            self.lru.retain(|existing| existing != key);
        }
    }

    fn no_reputation_penalties(&self, model: &str, candidates: &[InferenceTarget]) -> bool {
        candidates.iter().all(|target| {
            let key = TargetKey {
                model: model.to_string(),
                target: target.clone(),
            };
            self.reputation_score(&key) == 0
        })
    }

    fn reputation_ordered_candidates(
        &mut self,
        model: &str,
        candidates: &[InferenceTarget],
        now: Instant,
    ) -> Vec<InferenceTarget> {
        let mut ordered = candidates.to_vec();
        ordered.sort_by_key(|target| {
            let key = TargetKey {
                model: model.to_string(),
                target: target.clone(),
            };
            (
                matches!(target, InferenceTarget::None),
                self.reputation_score(&key),
            )
        });
        let penalized = candidates.iter().filter(|target| {
            let key = TargetKey {
                model: model.to_string(),
                target: (*target).clone(),
            };
            self.reputation_score(&key) > 0
        });
        let penalized_count = penalized.count();
        if penalized_count > 0 && ordered != candidates {
            self.routes_penalized = self.routes_penalized.saturating_add(penalized_count as u64);
            for target in candidates {
                let key = TargetKey {
                    model: model.to_string(),
                    target: target.clone(),
                };
                if let Some(entry) = self.reputation.get_mut(&key) {
                    entry.last_observed = now;
                }
            }
        }
        ordered
    }

    fn reputation_score(&self, key: &TargetKey) -> u32 {
        self.reputation
            .get(key)
            .map(|entry| entry.penalty)
            .unwrap_or(0)
    }

    fn reputation_stats(&self) -> TargetReputationStats {
        TargetReputationStats {
            penalized_targets: self
                .reputation
                .values()
                .filter(|entry| entry.penalty > 0)
                .count(),
            routes_penalized: self.routes_penalized,
        }
    }
}

fn normalized_model(model: Option<&str>) -> Option<String> {
    model
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(ToOwned::to_owned)
}

fn has_routable_candidate(candidates: &[InferenceTarget]) -> bool {
    candidates
        .iter()
        .any(|candidate| !matches!(candidate, InferenceTarget::None))
}

fn cooldown_for_failure(failures: u32, config: TargetHealthConfig) -> Duration {
    let multiplier = 1u32
        .checked_shl(failures.saturating_sub(1).min(6))
        .unwrap_or(64);
    config
        .base_cooldown
        .saturating_mul(multiplier)
        .min(config.max_cooldown)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn local(port: u16) -> InferenceTarget {
        InferenceTarget::Local(port)
    }

    #[test]
    fn retryable_failure_cools_target_when_alternatives_exist() {
        let health = TargetHealth::default();
        let candidates = vec![local(9001), local(9002)];

        health.record_outcome(Some("qwen"), &local(9001), TargetHealthOutcome::Unavailable);

        assert_eq!(
            health.eligible_candidates("qwen", &candidates),
            vec![local(9002)]
        );
        assert_eq!(
            health.snapshot(),
            TargetHealthSnapshot {
                cooling_targets: 1,
                routes_avoided: 1,
                reputation: TargetReputationStats {
                    penalized_targets: 1,
                    ..TargetReputationStats::default()
                },
            }
        );
    }

    #[test]
    fn success_clears_target_cooldown_before_reputation_fully_recovers() {
        let health = TargetHealth::default();
        let candidates = vec![local(9001), local(9002)];

        health.record_outcome(Some("qwen"), &local(9001), TargetHealthOutcome::Timeout);
        health.record_outcome(Some("qwen"), &local(9001), TargetHealthOutcome::Success);

        assert_eq!(
            health.eligible_candidates("qwen", &candidates),
            vec![local(9002), local(9001)]
        );
        assert_eq!(health.snapshot().cooling_targets, 0);
        assert_eq!(health.snapshot().reputation.penalized_targets, 1);
    }

    #[test]
    fn context_overflow_and_rejected_do_not_cool_target() {
        let health = TargetHealth::default();
        let candidates = vec![local(9001), local(9002)];

        health.record_outcome(
            Some("qwen"),
            &local(9001),
            TargetHealthOutcome::ContextOverflow,
        );
        health.record_outcome(Some("qwen"), &local(9001), TargetHealthOutcome::Rejected);

        assert_eq!(health.eligible_candidates("qwen", &candidates), candidates);
        assert_eq!(health.snapshot().cooling_targets, 0);
        assert_eq!(health.snapshot().reputation.penalized_targets, 0);
    }

    #[test]
    fn all_cooling_candidates_remain_eligible_to_preserve_availability() {
        let health = TargetHealth::default();
        let candidates = vec![local(9001), local(9002)];

        health.record_outcome(Some("qwen"), &local(9001), TargetHealthOutcome::Timeout);
        health.record_outcome(Some("qwen"), &local(9002), TargetHealthOutcome::Unavailable);

        assert_eq!(health.eligible_candidates("qwen", &candidates), candidates);
        assert_eq!(health.snapshot().cooling_targets, 2);
    }

    #[test]
    fn none_does_not_count_as_a_routable_cooldown_alternative() {
        let health = TargetHealth::default();
        let candidates = vec![local(9001), InferenceTarget::None];

        health.record_outcome(Some("qwen"), &local(9001), TargetHealthOutcome::Timeout);

        assert_eq!(health.eligible_candidates("qwen", &candidates), candidates);
        assert_eq!(
            health.snapshot(),
            TargetHealthSnapshot {
                cooling_targets: 1,
                routes_avoided: 0,
                reputation: TargetReputationStats {
                    penalized_targets: 1,
                    ..TargetReputationStats::default()
                },
            }
        );
    }

    #[test]
    fn strict_candidates_exclude_single_cooling_target_for_auto_fallback() {
        let health = TargetHealth::default();
        let candidates = vec![local(9001)];

        health.record_outcome(Some("qwen"), &local(9001), TargetHealthOutcome::Timeout);

        assert_eq!(health.eligible_candidates("qwen", &candidates), candidates);
        assert!(
            health
                .strict_eligible_candidates("qwen", &candidates)
                .is_empty()
        );
        assert_eq!(
            health.snapshot(),
            TargetHealthSnapshot {
                cooling_targets: 1,
                routes_avoided: 1,
                reputation: TargetReputationStats {
                    penalized_targets: 1,
                    ..TargetReputationStats::default()
                },
            }
        );
    }

    #[test]
    fn strict_candidates_exclude_all_cooling_targets_for_auto_fallback() {
        let health = TargetHealth::default();
        let candidates = vec![local(9001), local(9002)];

        health.record_outcome(Some("qwen"), &local(9001), TargetHealthOutcome::Timeout);
        health.record_outcome(Some("qwen"), &local(9002), TargetHealthOutcome::Unavailable);

        assert_eq!(health.eligible_candidates("qwen", &candidates), candidates);
        assert!(
            health
                .strict_eligible_candidates("qwen", &candidates)
                .is_empty()
        );
        assert_eq!(
            health.snapshot(),
            TargetHealthSnapshot {
                cooling_targets: 2,
                routes_avoided: 2,
                reputation: TargetReputationStats {
                    penalized_targets: 2,
                    ..TargetReputationStats::default()
                },
            }
        );
    }

    #[test]
    fn cooldowns_are_scoped_by_model_and_target() {
        let health = TargetHealth::default();
        let candidates = vec![local(9001), local(9002)];

        health.record_outcome(Some("qwen"), &local(9001), TargetHealthOutcome::Timeout);

        assert_eq!(
            health.eligible_candidates("qwen", &candidates),
            vec![local(9002)]
        );
        assert_eq!(health.eligible_candidates("llama", &candidates), candidates);
    }

    #[test]
    fn expired_cooldowns_are_pruned() {
        let health = TargetHealth::with_config(
            Duration::from_millis(0),
            Duration::from_millis(0),
            DEFAULT_MAX_ENTRIES,
        );
        let candidates = vec![local(9001), local(9002)];

        health.record_outcome(Some("qwen"), &local(9001), TargetHealthOutcome::Timeout);

        assert_eq!(
            health.eligible_candidates("qwen", &candidates),
            vec![local(9002), local(9001)]
        );
        assert_eq!(health.snapshot().cooling_targets, 0);
    }

    #[test]
    fn entry_limit_evicts_oldest_cooldown() {
        let health = TargetHealth::with_config(Duration::from_secs(60), Duration::from_secs(60), 1);
        let candidates = vec![local(9001), local(9002)];

        health.record_outcome(Some("qwen"), &local(9001), TargetHealthOutcome::Timeout);
        health.record_outcome(Some("qwen"), &local(9002), TargetHealthOutcome::Timeout);

        assert_eq!(
            health.eligible_candidates("qwen", &candidates),
            vec![local(9001)]
        );
        assert_eq!(health.snapshot().cooling_targets, 1);
    }

    #[test]
    fn retryable_failure_leaves_behavioral_penalty_after_cooldown() {
        let health = TargetHealth::with_config(
            Duration::from_millis(0),
            Duration::from_millis(0),
            DEFAULT_MAX_ENTRIES,
        );
        let candidates = vec![local(9001), local(9002)];

        health.record_outcome(Some("qwen"), &local(9001), TargetHealthOutcome::Unavailable);

        assert_eq!(
            health.eligible_candidates("qwen", &candidates),
            vec![local(9002), local(9001)]
        );
        let snapshot = health.snapshot();
        assert_eq!(snapshot.cooling_targets, 0);
        assert_eq!(snapshot.reputation.penalized_targets, 1);
        assert_eq!(snapshot.reputation.routes_penalized, 1);
    }

    #[test]
    fn behavioral_success_rebuilds_target_reputation() {
        let health = TargetHealth::default();
        let candidates = vec![local(9001), local(9002)];

        health.record_outcome(Some("qwen"), &local(9001), TargetHealthOutcome::Unavailable);
        health.record_outcome(Some("qwen"), &local(9001), TargetHealthOutcome::Success);
        health.record_outcome(Some("qwen"), &local(9001), TargetHealthOutcome::Success);

        assert_eq!(health.eligible_candidates("qwen", &candidates), candidates);
        assert_eq!(health.snapshot().reputation.penalized_targets, 0);
    }
}
