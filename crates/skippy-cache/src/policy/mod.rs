//! Benefit-per-exclusive-byte admission, probation, and eviction policy
//! (issue #1650, first slice).
//!
//! This module is deliberately a *pure policy*: it consumes observed events
//! (candidate offers, hits, misses, cost samples) and produces decisions
//! (admit / probation / persist / evict) with opaque reasons. It performs no
//! I/O, holds no locks on the restore path, and never sees prompt content —
//! entries are addressed by an opaque `EntryKey` the caller assigns.
//!
//! The score is the one the issue prescribes:
//!
//! ```text
//! reuse_probability * max(cold_prefill_cost - restore_cost, 0)
//! -----------------------------------------------------------
//!                exclusive_physical_bytes
//! ```
//!
//! Shared segments are credited fractionally: a physical byte referenced by
//! N entries counts as `bytes / N` against each of them, so total accounted
//! bytes never double-count a segment.
//!
//! Determinism: every ordering falls back to `(score, entry_key)` so two runs
//! over the same trace make identical decisions.

mod accounting;
mod admission;
mod decay;
#[cfg(test)]
mod lru_baseline;
mod score;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod traces;

pub use accounting::{SegmentId, SharedSegmentLedger};
pub use admission::{AdmissionDecision, AdmissionDecisionKind, AdmissionVerdict, PolicyEntryState};
pub use decay::DecayConfig;
pub use score::{BenefitScore, ScoreInputs};

use std::collections::BTreeMap;

use serde::Serialize;

/// Opaque, caller-assigned entry identity. Ordered so tie-breaks are
/// deterministic; content-free so policy logs leak nothing about prompts.
pub type EntryKey = u64;

/// Opaque shared-segment identity.
pub type SegmentRef = SegmentId;

/// Observed costs, in caller-defined units (the policy only compares them).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CostSample {
    /// Cold prefill cost for the entry's token range (e.g. ms or tokens).
    pub cold_prefill_cost: f64,
    /// Measured `queue + restore + suffix-prefill` cost for the same range.
    pub restore_cost: f64,
}

impl CostSample {
    /// Net benefit of a restore hit over recomputing cold. Never negative.
    pub fn net_benefit(&self) -> f64 {
        (self.cold_prefill_cost - self.restore_cost).max(0.0)
    }

    /// A usable sample must be finite and nonnegative; measured costs that
    /// arrive NaN/infinite (or negative) are rejected so scores and orderings
    /// stay total and panic-free.
    pub fn is_valid(&self) -> bool {
        self.cold_prefill_cost.is_finite()
            && self.restore_cost.is_finite()
            && self.cold_prefill_cost >= 0.0
            && self.restore_cost >= 0.0
    }
}

/// Policy decision log line: what was decided, and why, without content.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DecisionReason {
    pub entry: EntryKey,
    pub decision: AdmissionDecisionKind,
    /// Machine-readable reason tokens, e.g. `probation-second-hit`.
    pub reasons: Vec<String>,
    pub score: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum EvictionVerdict {
    Keep,
    Evict,
}

/// Tunables. Defaults follow the issue's guidance; every field is `Copy` and
/// plain so config files can carry it verbatim later.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PolicyConfig {
    /// Bytes probation entries may collectively charge before the policy
    /// must start dropping the least valuable probationers.
    pub probation_byte_budget: u64,
    /// Hits required to leave probation and become persist-eligible.
    /// The issue names second-hit admission: `2`.
    pub persistence_hit_threshold: u32,
    /// Minimum reuse probability the estimator may report (floor so a single
    /// hit still admits under pressure, and division stays sane).
    pub min_reuse_probability: f64,
    pub decay: DecayConfig,
    /// Observations of grace after admission during which an entry cannot be
    /// chosen as an eviction victim: probation must get a fair chance to land
    /// its second hit before pressure can reclaim its bytes.
    pub grace_observations: u64,
    /// Maximum retained ghost records. One-shot keys must not create
    /// permanent metadata.
    pub ghost_capacity: usize,
    /// Ghosts older than this many observations are expired so stale
    /// popularity cannot revive indefinitely.
    pub ghost_max_age_observations: u64,
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self {
            probation_byte_budget: 256 << 20,
            persistence_hit_threshold: 2,
            min_reuse_probability: 0.01,
            decay: DecayConfig::default(),
            grace_observations: 32,
            ghost_capacity: 4096,
            ghost_max_age_observations: 1024,
        }
    }
}

impl PolicyConfig {
    /// Config bounds: NaN/empty decay or an out-of-range reuse floor would
    /// poison every score. `probation_byte_budget` may be 0 (probation off).
    pub fn is_valid(&self) -> bool {
        self.min_reuse_probability.is_finite()
            && (0.0..=1.0).contains(&self.min_reuse_probability)
            && self.decay.is_valid()
            && self.persistence_hit_threshold >= 1
    }
}

/// Per-entry policy state and statistics.
#[derive(Debug, Clone, PartialEq)]
pub struct PolicyEntry {
    pub state: PolicyEntryState,
    pub hits: u64,
    pub misses: u64,
    /// Decayed reuse-estimator numerator/denominator inputs.
    pub reuse_weight: f64,
    pub observation_weight: f64,
    pub last_cost: Option<CostSample>,
    /// Exclusive (non-shared) physical bytes charged to this entry.
    pub exclusive_bytes: u64,
    /// Segments this entry references; fractional credit lives in the ledger.
    pub segments: Vec<SegmentId>,
    /// Clock value at this entry's last admission or hit; drives grace.
    pub last_observation: u64,
}

impl PolicyEntry {
    /// Estimated reuse probability under the configured decay window: a
    /// smoothed hit ratio in `[0, 1]`.
    pub fn reuse_probability(&self) -> f64 {
        if self.observation_weight <= 0.0 {
            return 0.0;
        }
        (self.reuse_weight / self.observation_weight).clamp(0.0, 1.0)
    }
}

/// The policy engine. Owns per-entry statistics and the shared-segment ledger;
/// the caller drives it from cache events.
pub struct BenefitPolicy {
    pub(crate) config: PolicyConfig,
    pub(crate) entries: BTreeMap<EntryKey, PolicyEntry>,
    pub(crate) segments: SharedSegmentLedger,
    /// Monotonic observation counter driving the probation grace window.
    pub(crate) clock: u64,
    /// Reuse statistics that outlive eviction ("ghosts"): an entry that
    /// recurs after eviction carries its history back in, so the second-hit
    /// value signal survives cache pressure. Bounded by count and age.
    pub(crate) ghosts: BTreeMap<EntryKey, GhostStats>,
}

impl BenefitPolicy {
    /// Insert a ghost, evicting the oldest ghost when the count bound is
    /// exceeded. `ghost_capacity = 0` disables ghost retention entirely.
    fn insert_ghost(&mut self, key: EntryKey, stats: GhostStats) {
        if self.config.ghost_capacity == 0 {
            return;
        }
        while self.ghosts.len() >= self.config.ghost_capacity {
            let oldest = self
                .ghosts
                .iter()
                .min_by_key(|(k, g)| (g.last_observation, **k))
                .map(|(k, _)| *k);
            match oldest {
                Some(k) => {
                    self.ghosts.remove(&k);
                }
                None => break,
            }
        }
        self.ghosts.insert(key, stats);
    }

    /// Drop ghosts older than the configured age bound. Called from the
    /// observation-driven entry points so expiry is deterministic over a
    /// trace without a background timer.
    fn expire_ghosts(&mut self) {
        let horizon = self
            .clock
            .saturating_sub(self.config.ghost_max_age_observations);
        self.ghosts.retain(|_, g| g.last_observation >= horizon);
    }
}

/// Surviving statistics for an evicted entry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GhostStats {
    pub hits: u64,
    pub reuse_weight: f64,
    pub observation_weight: f64,
    /// Clock value when the ghost was created; drives age expiry.
    pub last_observation: u64,
}

impl BenefitPolicy {
    /// Panics on invalid config so misconfiguration fails at startup
    /// rather than producing NaN scores later.
    pub fn new(config: PolicyConfig) -> Self {
        assert!(config.is_valid(), "invalid PolicyConfig: {:?}", config);
        Self {
            config,
            entries: BTreeMap::new(),
            segments: SharedSegmentLedger::default(),
            clock: 0,
            ghosts: BTreeMap::new(),
        }
    }

    pub fn config(&self) -> &PolicyConfig {
        &self.config
    }

    pub fn ghost(&self, key: EntryKey) -> Option<&GhostStats> {
        self.ghosts.get(&key)
    }

    pub fn ghost_count(&self) -> usize {
        self.ghosts.len()
    }

    /// Current observation clock (test-only access).
    #[cfg(test)]
    pub fn clock_debug(&self) -> u64 {
        self.clock
    }

    pub fn entry(&self, key: EntryKey) -> Option<&PolicyEntry> {
        self.entries.get(&key)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Offer a new candidate for admission. `exclusive_bytes` are bytes only
    /// this entry would reference; `shared` lists `(segment, size)` segments
    /// it would join (size is ignored if the segment is already registered).
    pub fn consider_admission(
        &mut self,
        key: EntryKey,
        exclusive_bytes: u64,
        shared: Vec<(SegmentId, u64)>,
        cost: CostSample,
    ) -> AdmissionDecision {
        // Full structural prevalidation before any policy mutation: an
        // invalid cost, duplicate segment IDs, a size conflict, or an
        // already-resident key must not advance the clock (aging grace) or
        // expire ghosts.
        if !cost.is_valid() {
            return AdmissionDecision::rejected_invalid_cost();
        }
        if self.entries.contains_key(&key) {
            return AdmissionDecision::rejected("already-resident-key");
        }
        let mut seen = std::collections::BTreeSet::new();
        if shared.iter().any(|(s, _)| !seen.insert(*s)) {
            return AdmissionDecision::rejected("duplicate-segment-reference");
        }
        if shared.iter().any(|(segment, size)| {
            self.segments
                .segment_record(*segment)
                .is_some_and(|r| !r.references.contains(&key) && r.size != *size)
        }) {
            return AdmissionDecision::rejected("segment-size-conflict");
        }
        self.clock += 1;
        self.expire_ghosts();
        let mut decision = admission::consider(self, key, exclusive_bytes, shared, cost);
        if decision.verdict == crate::policy::AdmissionVerdict::Admit {
            // Hard probation cap is part of admission: the decision carries
            // the keys the caller must physically evict. Selection only —
            // committed removal stays with `remove` so victims become ghosts.
            decision.probation_cap_victims = self.select_probation_cap_victims();
        }
        decision
    }

    /// Record a restore hit on an admitted entry; may promote out of probation.
    pub fn record_hit(&mut self, key: EntryKey, cost: CostSample) -> Option<AdmissionDecision> {
        // Validate before any mutation (including the clock).
        if !cost.is_valid() {
            return None;
        }
        self.clock += 1;
        admission::record_hit(self, key, cost)
    }

    /// Record a miss/cold recompute for an admitted entry (decays reuse).
    pub fn record_miss(&mut self, key: EntryKey) {
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.misses += 1;
            entry.observation_weight = entry.observation_weight * self.config.decay.factor + 1.0;
        }
    }

    /// Observe demand pressure (0 = idle, 1 = saturated). Higher pressure
    /// decays reuse history faster than the observation base, so stale
    /// popularity cannot pin bytes forever.
    pub fn observe_pressure(&mut self, pressure: f64) {
        // Non-finite pressure never poisons history: NaN is ignored,
        // +inf saturates to full pressure, -inf to none.
        if pressure.is_nan() {
            return;
        }
        let pressure = if pressure == f64::INFINITY {
            1.0
        } else if pressure == f64::NEG_INFINITY {
            0.0
        } else {
            pressure.clamp(0.0, 1.0)
        };
        let base = self.config.decay.factor;
        let reuse_factor = base * (1.0 - pressure);
        for entry in self.entries.values_mut() {
            entry.reuse_weight *= reuse_factor;
            entry.observation_weight *= base + (1.0 - base) * pressure;
        }
        for ghost in self.ghosts.values_mut() {
            ghost.reuse_weight *= reuse_factor;
            ghost.observation_weight *= base + (1.0 - base) * pressure;
        }
        self.expire_ghosts();
    }

    /// Score an entry under the current statistics. Returns `None` for
    /// entries with no cost observation yet.
    pub fn score(&self, key: EntryKey) -> Option<BenefitScore> {
        let entry = self.entries.get(&key)?;
        score::compute(&self.config, key, entry, &self.segments)
    }

    /// Pick eviction victims until `bytes_to_free` exclusive-and-fractional
    /// bytes are released. Lowest score first, deterministic `(score, key)`
    /// tie-break. Pinned entries are never chosen while an unpinned
    /// candidate remains.
    pub fn choose_victims(
        &mut self,
        bytes_to_free: u64,
        pinned: &[EntryKey],
    ) -> Vec<(EntryKey, EvictionVerdict)> {
        admission::choose_victims(self, bytes_to_free, pinned)
    }

    /// Bytes charged to the probation class: exclusive bytes plus the
    /// fractional shared-segment credit. Shares are summed exactly (f64)
    /// across all probation entries before a single rounding, so a small
    /// segment shared by many references still charges its physical bytes
    /// in aggregate — per-entry truncation cannot zero it out.
    pub fn probation_bytes(&self) -> u64 {
        let mut exact = 0.0f64;
        for (k, e) in &self.entries {
            if e.state == PolicyEntryState::Probation {
                exact += e.exclusive_bytes as f64 + self.segments.fractional_bytes(*k, &e.segments);
            }
        }
        // Deterministic round-half-up; the class charge is a hard bound, so
        // we always round up any fractional residue.
        exact.ceil() as u64
    }

    /// Select the no-hit probationers that must be evicted to bring the
    /// probation class back under its byte cap, oldest observation first
    /// (grace waived: the hard cap always wins). Selection only — this does
    /// not mutate policy state; the caller commits each removal via
    /// `remove`, which also records the ghost. Keys are deterministic
    /// `(last_observation, key)` order.
    pub fn select_probation_cap_victims(&self) -> Vec<EntryKey> {
        let cap = self.config.probation_byte_budget;
        if self.probation_bytes() <= cap {
            return Vec::new();
        }
        let mut probationers: Vec<(u64, EntryKey)> = self
            .entries
            .iter()
            .filter(|(_, e)| e.state == PolicyEntryState::Probation && e.hits == 0)
            .map(|(k, e)| (e.last_observation, *k))
            .collect();
        probationers.sort();
        // Simulate each removal against a scratch ledger: removing a shared
        // reference raises the survivors' fractional shares, so the remaining
        // class charge must be recomputed, not decremented by stale shares.
        let mut scratch = self.segments.clone();
        let mut removed: std::collections::BTreeSet<EntryKey> = Default::default();
        let mut victims = Vec::new();
        for (_, key) in probationers {
            let Some(entry) = self.entries.get(&key) else {
                continue;
            };
            scratch.release(&entry.segments, key);
            removed.insert(key);
            let charge: f64 = self
                .entries
                .iter()
                .filter(|(k, e)| e.state == PolicyEntryState::Probation && !removed.contains(*k))
                .map(|(k, e)| e.exclusive_bytes as f64 + scratch.fractional_bytes(*k, &e.segments))
                .sum();
            victims.push(key);
            if charge.ceil() as u64 <= cap {
                break;
            }
        }
        victims
    }

    /// Test helper: run `f` with a temporarily different probation budget.
    #[cfg(test)]
    pub fn with_probation_budget<R>(&mut self, budget: u64, f: impl FnOnce(&Self) -> R) -> R {
        let saved = std::mem::replace(
            // Safety of construction: same struct, one field changed.
            &mut self.config.probation_byte_budget,
            budget,
        );
        let result = f(self);
        self.config.probation_byte_budget = saved;
        result
    }

    /// Compatibility wrapper for callers that want cap enforcement applied
    /// immediately: selects victims (see `select_probation_cap_victims`)
    /// and commits their removals through `remove`, so they become ghosts.
    pub fn enforce_probation_cap(&mut self) -> Vec<EntryKey> {
        let victims = self.select_probation_cap_victims();
        for key in &victims {
            self.remove(*key);
        }
        victims
    }

    /// Remove an entry the caller has actually evicted, releasing its
    /// fractional segment credit and stashing its reuse statistics as a
    /// ghost so a recurrence is recognized as a value signal.
    pub fn remove(&mut self, key: EntryKey) -> Option<PolicyEntry> {
        let entry = self.entries.remove(&key)?;
        self.segments.release(&entry.segments, key);
        self.insert_ghost(
            key,
            GhostStats {
                hits: entry.hits,
                reuse_weight: entry.reuse_weight,
                observation_weight: entry.observation_weight,
                last_observation: self.clock,
            },
        );
        Some(entry)
    }
}
