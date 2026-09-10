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
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self {
            probation_byte_budget: 256 << 20,
            persistence_hit_threshold: 2,
            min_reuse_probability: 0.01,
            decay: DecayConfig::default(),
            grace_observations: 32,
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
    /// value signal survives cache pressure.
    pub(crate) ghosts: BTreeMap<EntryKey, GhostStats>,
}

/// Surviving statistics for an evicted entry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GhostStats {
    pub hits: u64,
    pub reuse_weight: f64,
    pub observation_weight: f64,
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
        self.clock += 1;
        admission::consider(self, key, exclusive_bytes, shared, cost)
    }

    /// Record a restore hit on an admitted entry; may promote out of probation.
    pub fn record_hit(&mut self, key: EntryKey, cost: CostSample) -> Option<AdmissionDecision> {
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
        let pressure = pressure.clamp(0.0, 1.0);
        let base = self.config.decay.factor;
        let reuse_factor = base * (1.0 - pressure);
        for entry in self.entries.values_mut() {
            entry.reuse_weight *= reuse_factor;
            entry.observation_weight *= base + (1.0 - base) * pressure;
        }
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

    /// Total exclusive bytes held by probation-state entries.
    pub fn probation_bytes(&self) -> u64 {
        self.entries
            .values()
            .filter(|e| e.state == PolicyEntryState::Probation)
            .map(|e| e.exclusive_bytes)
            .sum()
    }

    /// Enforce the hard probation byte cap: evict no-hit probationers until
    /// the cap holds, waiving grace under pressure (the hard cap always
    /// wins). Oldest-observation first, then key order for determinism.
    /// Returns the keys the caller must actually evict.
    pub fn enforce_probation_cap(&mut self) -> Vec<EntryKey> {
        let cap = self.config.probation_byte_budget;
        let mut over = self.probation_bytes().saturating_sub(cap);
        if over == 0 {
            return Vec::new();
        }
        let mut probationers: Vec<(u64, EntryKey)> = self
            .entries
            .iter()
            .filter(|(_, e)| e.state == PolicyEntryState::Probation && e.hits == 0)
            .map(|(k, e)| (e.last_observation, *k))
            .collect();
        probationers.sort();
        let mut victims = Vec::new();
        for (_, key) in probationers {
            if over == 0 {
                break;
            }
            if let Some(entry) = self.entries.remove(&key) {
                self.segments.release(&entry.segments, key);
                over = over.saturating_sub(entry.exclusive_bytes);
                victims.push(key);
            }
        }
        victims
    }

    /// Remove an entry the caller has actually evicted, releasing its
    /// fractional segment credit and stashing its reuse statistics as a
    /// ghost so a recurrence is recognized as a value signal.
    pub fn remove(&mut self, key: EntryKey) -> Option<PolicyEntry> {
        let entry = self.entries.remove(&key)?;
        self.segments.release(&entry.segments, key);
        self.ghosts.insert(
            key,
            GhostStats {
                hits: entry.hits,
                reuse_weight: entry.reuse_weight,
                observation_weight: entry.observation_weight,
            },
        );
        Some(entry)
    }
}
