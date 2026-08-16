//! Why a prefix lookup missed, and how long ago the prefix was last seen.
//!
//! This is the measurement gate for the expensive retention work. A disk tier
//! only pays for itself if misses are dominated by prefixes this node *had*
//! and evicted. If misses are mostly `NeverSeen`, no amount of retention
//! capacity helps and the effort should be spent on routing instead.
//!
//! Distinguishing those cases needs a small amount of memory about prefixes
//! that are no longer resident, so eviction leaves a **tombstone**: the page
//! id, when it was evicted, and how many tokens it held. Tombstones are
//! strictly bounded and hold no token data, so the memory cost is a few tens
//! of KiB regardless of traffic.
//!
//! Privacy: page ids are BLAKE3 digests over tokens and never leave the
//! process — only the bounded enum and bucket labels below are exportable.
//! See `.agents/skills/telemetry-privacy-review`.

use std::collections::{BTreeMap, HashMap};

/// Why a lookup for a prefix did not produce a usable cache entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PrefixMissReason {
    /// This node held the prefix and evicted it. These are the misses a
    /// retention tier converts into hits.
    EvictedRecently,
    /// This node has no record of ever holding the prefix. Retention cannot
    /// help; only routing the request to a node that has it can.
    NeverSeen,
    /// The prefix was held under a different stage configuration — a
    /// different KV dtype, backend, or layer split — so the bytes are not
    /// usable. Persistently non-zero means topology or config churn is
    /// destroying retention value.
    IdentityMismatch,
}

impl PrefixMissReason {
    /// Stable, bounded label for telemetry export.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::EvictedRecently => "evicted_recently",
            Self::NeverSeen => "never_seen",
            Self::IdentityMismatch => "identity_mismatch",
        }
    }

    pub const ALL: [Self; 3] = [
        Self::EvictedRecently,
        Self::NeverSeen,
        Self::IdentityMismatch,
    ];
}

/// Bucketed time since a prefix was last resident.
///
/// The gap distribution is the decisive input: it says how long a tier must
/// retain a prefix to convert misses into hits. Buckets are coarse and fixed
/// so the attribute stays low-cardinality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PrefixGapBucket {
    /// Under a minute — a tight agent tool loop.
    UnderMinute,
    /// Under five minutes — an active conversation with think time.
    UnderFiveMinutes,
    /// Under an hour — a user who stepped away.
    UnderHour,
    /// An hour or more — only a persistent tier can serve this.
    OverHour,
}

impl PrefixGapBucket {
    pub fn from_seconds(seconds: u64) -> Self {
        match seconds {
            0..=59 => Self::UnderMinute,
            60..=299 => Self::UnderFiveMinutes,
            300..=3599 => Self::UnderHour,
            _ => Self::OverHour,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::UnderMinute => "under_minute",
            Self::UnderFiveMinutes => "under_5m",
            Self::UnderHour => "under_1h",
            Self::OverHour => "over_1h",
        }
    }

    pub const ALL: [Self; 4] = [
        Self::UnderMinute,
        Self::UnderFiveMinutes,
        Self::UnderHour,
        Self::OverHour,
    ];
}

/// A bounded record that a prefix was once held and is no longer resident.
#[derive(Debug, Clone, Copy)]
struct PrefixTombstone {
    evicted_at_secs: u64,
    token_count: u64,
    /// Insertion order, used for FIFO trimming when the table is full.
    sequence: u64,
}

/// Bounded miss-reason accounting for the prefix cache.
///
/// Tombstones are trimmed FIFO rather than LRU: the question being answered is
/// "how long ago was this evicted", so the oldest tombstones are both the
/// largest and the least actionable, and dropping them only ever moves a miss
/// from `EvictedRecently` to `NeverSeen` — a conservative direction that
/// understates rather than overstates the value of a retention tier.
#[derive(Debug)]
pub struct PrefixMissTracker {
    max_tombstones: usize,
    sequence: u64,
    tombstones: HashMap<String, PrefixTombstone>,
    /// Insertion order index, so trimming is O(log n) rather than a linear
    /// scan. This runs on the eviction path, which is itself on the decode
    /// hot path, so a full scan of 8192 entries per eviction is not
    /// acceptable — and it would cost that on the default path too, since
    /// miss tracking is always on.
    by_sequence: BTreeMap<u64, String>,
    hits: u64,
    misses: HashMap<(PrefixMissReason, PrefixGapBucket), u64>,
    /// Tokens that had to be re-prefilled because a held prefix was evicted.
    /// This is the concrete cost of insufficient retention.
    evicted_miss_tokens: u64,
}

/// Enough to cover the working set of a busy agentic node without unbounded
/// growth. At ~120 bytes per entry this is well under 1 MiB.
const DEFAULT_MAX_TOMBSTONES: usize = 8192;

impl Default for PrefixMissTracker {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_TOMBSTONES)
    }
}

impl PrefixMissTracker {
    pub fn new(max_tombstones: usize) -> Self {
        Self {
            max_tombstones: max_tombstones.max(1),
            sequence: 0,
            tombstones: HashMap::new(),
            by_sequence: BTreeMap::new(),
            hits: 0,
            misses: HashMap::new(),
            evicted_miss_tokens: 0,
        }
    }

    /// Record that a prefix left the resident cache.
    pub fn note_evicted(&mut self, page_id: &str, token_count: u64, now_secs: u64) {
        self.trim_to_capacity();
        self.sequence = self.sequence.saturating_add(1);
        let sequence = self.sequence;
        if let Some(previous) = self.tombstones.insert(
            page_id.to_string(),
            PrefixTombstone {
                evicted_at_secs: now_secs,
                token_count,
                sequence,
            },
        ) {
            self.by_sequence.remove(&previous.sequence);
        }
        self.by_sequence.insert(sequence, page_id.to_string());
    }

    /// Record that a prefix was found. Clears any tombstone, since the prefix
    /// is resident again and a later miss on it would be a fresh eviction.
    pub fn note_hit(&mut self, page_id: &str) {
        self.hits = self.hits.saturating_add(1);
        self.forget(page_id);
    }

    /// Record that a prefix was recorded into the cache.
    pub fn note_recorded(&mut self, page_id: &str) {
        self.forget(page_id);
    }

    /// Classify a miss and fold it into the histogram.
    pub fn note_miss(&mut self, page_id: &str, now_secs: u64) -> PrefixMissReason {
        let (reason, bucket) = match self.tombstones.get(page_id) {
            Some(tombstone) => {
                let gap = now_secs.saturating_sub(tombstone.evicted_at_secs);
                self.evicted_miss_tokens = self
                    .evicted_miss_tokens
                    .saturating_add(tombstone.token_count);
                (
                    PrefixMissReason::EvictedRecently,
                    PrefixGapBucket::from_seconds(gap),
                )
            }
            None => (
                PrefixMissReason::NeverSeen,
                PrefixGapBucket::from_seconds(u64::MAX),
            ),
        };
        *self.misses.entry((reason, bucket)).or_default() += 1;
        reason
    }

    /// Record a miss caused by a prefix that exists under an incompatible
    /// stage configuration.
    pub fn note_identity_mismatch(&mut self) {
        *self
            .misses
            .entry((
                PrefixMissReason::IdentityMismatch,
                PrefixGapBucket::UnderMinute,
            ))
            .or_default() += 1;
    }

    pub fn stats(&self) -> PrefixMissStats {
        let mut by_reason = [0u64; 3];
        for (index, reason) in PrefixMissReason::ALL.iter().enumerate() {
            by_reason[index] = self
                .misses
                .iter()
                .filter(|((entry_reason, _), _)| entry_reason == reason)
                .map(|(_, count)| *count)
                .sum();
        }
        let mut evicted_by_gap = [0u64; 4];
        for (index, bucket) in PrefixGapBucket::ALL.iter().enumerate() {
            evicted_by_gap[index] = self
                .misses
                .get(&(PrefixMissReason::EvictedRecently, *bucket))
                .copied()
                .unwrap_or_default();
        }
        PrefixMissStats {
            hits: self.hits,
            misses_by_reason: by_reason,
            evicted_misses_by_gap: evicted_by_gap,
            evicted_miss_tokens: self.evicted_miss_tokens,
            tombstones: self.tombstones.len(),
        }
    }

    fn trim_to_capacity(&mut self) {
        while self.tombstones.len() >= self.max_tombstones {
            let Some((sequence, page_id)) = self
                .by_sequence
                .iter()
                .next()
                .map(|(sequence, page_id)| (*sequence, page_id.clone()))
            else {
                break;
            };
            self.by_sequence.remove(&sequence);
            self.tombstones.remove(&page_id);
        }
    }

    fn forget(&mut self, page_id: &str) {
        if let Some(tombstone) = self.tombstones.remove(page_id) {
            self.by_sequence.remove(&tombstone.sequence);
        }
    }
}

/// Snapshot of miss accounting, safe to export as bounded metrics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PrefixMissStats {
    pub hits: u64,
    /// Indexed by [`PrefixMissReason::ALL`].
    pub misses_by_reason: [u64; 3],
    /// `EvictedRecently` misses indexed by [`PrefixGapBucket::ALL`].
    pub evicted_misses_by_gap: [u64; 4],
    pub evicted_miss_tokens: u64,
    pub tombstones: usize,
}

impl PrefixMissStats {
    pub fn total_misses(&self) -> u64 {
        self.misses_by_reason.iter().copied().sum()
    }

    pub fn misses_for(&self, reason: PrefixMissReason) -> u64 {
        PrefixMissReason::ALL
            .iter()
            .position(|candidate| *candidate == reason)
            .map(|index| self.misses_by_reason[index])
            .unwrap_or_default()
    }

    pub fn evicted_misses_in(&self, bucket: PrefixGapBucket) -> u64 {
        PrefixGapBucket::ALL
            .iter()
            .position(|candidate| *candidate == bucket)
            .map(|index| self.evicted_misses_by_gap[index])
            .unwrap_or_default()
    }

    /// Fraction of misses a retention tier could plausibly convert into hits.
    ///
    /// This is the number the disk-tier decision turns on. Low means build
    /// routing (W6), not storage.
    pub fn recoverable_miss_ratio(&self) -> f64 {
        let total = self.total_misses();
        if total == 0 {
            return 0.0;
        }
        self.misses_for(PrefixMissReason::EvictedRecently) as f64 / total as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unseen_prefix_is_never_seen() {
        let mut tracker = PrefixMissTracker::default();

        assert_eq!(
            tracker.note_miss("page-a", 100),
            PrefixMissReason::NeverSeen
        );
        assert_eq!(tracker.stats().misses_for(PrefixMissReason::NeverSeen), 1);
    }

    #[test]
    fn evicted_prefix_is_attributed_to_eviction_with_a_gap_bucket() {
        let mut tracker = PrefixMissTracker::default();
        tracker.note_evicted("page-a", 2048, 100);

        assert_eq!(
            tracker.note_miss("page-a", 100 + 600),
            PrefixMissReason::EvictedRecently
        );
        let stats = tracker.stats();
        assert_eq!(stats.evicted_misses_in(PrefixGapBucket::UnderHour), 1);
        // The re-prefill cost of that miss is attributed too.
        assert_eq!(stats.evicted_miss_tokens, 2048);
    }

    #[test]
    fn gap_buckets_cover_the_full_range() {
        assert_eq!(
            PrefixGapBucket::from_seconds(0),
            PrefixGapBucket::UnderMinute
        );
        assert_eq!(
            PrefixGapBucket::from_seconds(59),
            PrefixGapBucket::UnderMinute
        );
        assert_eq!(
            PrefixGapBucket::from_seconds(60),
            PrefixGapBucket::UnderFiveMinutes
        );
        assert_eq!(
            PrefixGapBucket::from_seconds(3599),
            PrefixGapBucket::UnderHour
        );
        assert_eq!(
            PrefixGapBucket::from_seconds(3600),
            PrefixGapBucket::OverHour
        );
        assert_eq!(
            PrefixGapBucket::from_seconds(u64::MAX),
            PrefixGapBucket::OverHour
        );
    }

    /// A prefix that comes back and is evicted again must be attributed to the
    /// *second* eviction, not the first, or gap measurements drift long.
    #[test]
    fn a_hit_clears_the_tombstone() {
        let mut tracker = PrefixMissTracker::default();
        tracker.note_evicted("page-a", 128, 0);
        tracker.note_hit("page-a");

        assert_eq!(
            tracker.note_miss("page-a", 10_000),
            PrefixMissReason::NeverSeen
        );
    }

    #[test]
    fn recording_a_prefix_clears_the_tombstone() {
        let mut tracker = PrefixMissTracker::default();
        tracker.note_evicted("page-a", 128, 0);
        tracker.note_recorded("page-a");

        assert_eq!(tracker.note_miss("page-a", 10), PrefixMissReason::NeverSeen);
    }

    /// Tombstones must never grow without bound, whatever the traffic.
    #[test]
    fn tombstone_table_is_bounded() {
        let mut tracker = PrefixMissTracker::new(16);
        for index in 0..1000 {
            tracker.note_evicted(&format!("page-{index}"), 64, index as u64);
        }

        assert!(tracker.stats().tombstones <= 16);
    }

    /// Trimming must degrade towards `NeverSeen`, never invent an eviction.
    #[test]
    fn trimming_drops_oldest_tombstones_first() {
        let mut tracker = PrefixMissTracker::new(4);
        for index in 0..8 {
            tracker.note_evicted(&format!("page-{index}"), 64, index as u64);
        }

        // The oldest are gone...
        assert_eq!(
            tracker.note_miss("page-0", 100),
            PrefixMissReason::NeverSeen
        );
        // ...and the newest survive.
        assert_eq!(
            tracker.note_miss("page-7", 100),
            PrefixMissReason::EvictedRecently
        );
    }

    #[test]
    fn recoverable_ratio_drives_the_tier_decision() {
        let mut tracker = PrefixMissTracker::default();
        assert_eq!(tracker.stats().recoverable_miss_ratio(), 0.0);

        tracker.note_evicted("page-a", 100, 0);
        tracker.note_miss("page-a", 10);
        tracker.note_miss("page-b", 10);
        tracker.note_miss("page-c", 10);

        // One of three misses is recoverable by a retention tier.
        let ratio = tracker.stats().recoverable_miss_ratio();
        assert!((ratio - 1.0 / 3.0).abs() < 1e-9, "got {ratio}");
    }

    #[test]
    fn identity_mismatch_is_tracked_separately() {
        let mut tracker = PrefixMissTracker::default();
        tracker.note_identity_mismatch();

        let stats = tracker.stats();
        assert_eq!(stats.misses_for(PrefixMissReason::IdentityMismatch), 1);
        assert_eq!(stats.total_misses(), 1);
        // An incompatible page is not something a bigger cache would fix.
        assert_eq!(stats.recoverable_miss_ratio(), 0.0);
    }

    #[test]
    fn miss_reason_labels_are_stable_and_bounded() {
        let labels: Vec<&str> = PrefixMissReason::ALL
            .iter()
            .map(|reason| reason.as_str())
            .collect();

        assert_eq!(
            labels,
            vec!["evicted_recently", "never_seen", "identity_mismatch"]
        );
    }
}
