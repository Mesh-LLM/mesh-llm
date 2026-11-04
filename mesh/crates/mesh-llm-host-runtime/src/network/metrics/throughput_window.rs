//! Recent throughput evidence for routing: a bounded, timestamped window of
//! valid samples, kept apart from target activity.
//!
//! Activity (`last_updated`) is refreshed by every attempt, timeouts and
//! rejections included, so it says nothing about when a throughput sample was
//! measured. Routing reads this window instead. Only a token-bearing success
//! with a positive duration adds a sample, a sample stops counting once it is
//! older than [`THROUGHPUT_EVIDENCE_MAX_AGE`], and a window with no surviving
//! sample is no evidence at all. The cumulative counters next to it keep their
//! diagnostic meaning.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// How long one throughput sample counts as routing evidence. Well inside the
/// one-hour activity TTL, so a target kept alive by failures loses its credit.
pub(super) const THROUGHPUT_EVIDENCE_MAX_AGE: Duration = Duration::from_secs(15 * 60);

/// Most samples one window keeps; a new sample evicts the oldest.
pub(super) const THROUGHPUT_EVIDENCE_MAX_SAMPLES: usize = 32;

/// The samples of a window still inside it at one instant.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct RecentThroughput {
    pub(super) tps_milli_sum: u64,
    pub(super) samples: u64,
}

impl RecentThroughput {
    pub(super) fn add(self, other: Self) -> Self {
        Self {
            tps_milli_sum: self.tps_milli_sum.saturating_add(other.tps_milli_sum),
            samples: self.samples.saturating_add(other.samples),
        }
    }
}

#[derive(Default)]
pub(super) struct ThroughputWindow {
    /// Observation instant and tokens per second (milli), oldest first.
    samples: VecDeque<(Instant, u64)>,
}

impl ThroughputWindow {
    /// Add one sample observed at `now`. Concurrent attempts can take the
    /// shard lock out of timestamp order, so the sample goes to its place in
    /// time and the cap then evicts the oldest observation. A rate that rounds
    /// down to zero is no evidence and is not kept.
    pub(super) fn record(&mut self, now: Instant, tps_milli: u64) {
        if tps_milli == 0 {
            return;
        }
        self.expire(now);
        let position = self
            .samples
            .partition_point(|(observed, _)| *observed <= now);
        self.samples.insert(position, (now, tps_milli));
        if self.samples.len() > THROUGHPUT_EVIDENCE_MAX_SAMPLES {
            self.samples.pop_front();
        }
    }

    /// The samples still inside the window at `now`; empty once all expired.
    pub(super) fn recent(&self, now: Instant) -> RecentThroughput {
        self.samples
            .iter()
            .filter(|(observed, _)| is_fresh(*observed, now))
            .fold(RecentThroughput::default(), |recent, (_, tps_milli)| {
                recent.add(RecentThroughput {
                    tps_milli_sum: *tps_milli,
                    samples: 1,
                })
            })
    }

    fn expire(&mut self, now: Instant) {
        while self
            .samples
            .front()
            .is_some_and(|(observed, _)| !is_fresh(*observed, now))
        {
            self.samples.pop_front();
        }
    }
}

/// A sample taken after `now` (a caller holding an earlier instant) has age
/// zero rather than a negative one.
fn is_fresh(observed: Instant, now: Instant) -> bool {
    now.saturating_duration_since(observed) <= THROUGHPUT_EVIDENCE_MAX_AGE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_sample_counts_up_to_the_max_age_and_not_one_nanosecond_after() {
        let start = Instant::now();
        let mut window = ThroughputWindow::default();
        window.record(start, 100_000);

        let at_limit = start + THROUGHPUT_EVIDENCE_MAX_AGE;
        assert_eq!(
            window.recent(at_limit),
            RecentThroughput {
                tps_milli_sum: 100_000,
                samples: 1
            }
        );
        let past_limit = at_limit + Duration::from_nanos(1);
        assert_eq!(window.recent(past_limit), RecentThroughput::default());
    }

    #[test]
    fn an_empty_window_is_no_evidence() {
        assert_eq!(
            ThroughputWindow::default().recent(Instant::now()),
            RecentThroughput::default()
        );
    }

    #[test]
    fn a_new_sample_restores_evidence_without_the_expired_ones() {
        let start = Instant::now();
        let mut window = ThroughputWindow::default();
        window.record(start, 100_000);
        let later = start + THROUGHPUT_EVIDENCE_MAX_AGE + Duration::from_secs(60);
        window.record(later, 40_000);

        assert_eq!(
            window.recent(later),
            RecentThroughput {
                tps_milli_sum: 40_000,
                samples: 1
            }
        );
        assert_eq!(window.samples.len(), 1, "the expired sample is dropped");
    }

    #[test]
    fn high_volume_keeps_only_the_newest_samples() {
        let start = Instant::now();
        let mut window = ThroughputWindow::default();
        for i in 0..100_u64 {
            window.record(start + Duration::from_millis(i), i * 1_000);
        }

        let recent = window.recent(start + Duration::from_secs(1));
        let newest = (100 - THROUGHPUT_EVIDENCE_MAX_SAMPLES as u64)..100;
        assert_eq!(recent.samples, THROUGHPUT_EVIDENCE_MAX_SAMPLES as u64);
        assert_eq!(recent.tps_milli_sum, newest.map(|i| i * 1_000).sum::<u64>());
        assert_eq!(window.samples.len(), THROUGHPUT_EVIDENCE_MAX_SAMPLES);
    }

    #[test]
    fn a_sample_newer_than_the_reading_instant_counts_as_fresh() {
        let start = Instant::now();
        let mut window = ThroughputWindow::default();
        window.record(start + Duration::from_secs(5), 7_000);

        assert_eq!(window.recent(start).samples, 1);
    }

    #[test]
    fn a_late_recorded_older_sample_is_the_one_evicted() {
        let start = Instant::now();
        let mut window = ThroughputWindow::default();
        for i in 1..=THROUGHPUT_EVIDENCE_MAX_SAMPLES as u64 {
            window.record(start + Duration::from_millis(i), 1_000);
        }
        // Took the lock last, but was observed first: it is the oldest.
        window.record(start, 9_000);

        assert_eq!(window.samples.len(), THROUGHPUT_EVIDENCE_MAX_SAMPLES);
        assert!(window.samples.iter().all(|(_, tps)| *tps == 1_000));

        let mut ordered = ThroughputWindow::default();
        ordered.record(start + Duration::from_millis(2), 2_000);
        ordered.record(start + Duration::from_millis(1), 1_000);
        let times: Vec<_> = ordered.samples.iter().map(|(at, _)| *at).collect();
        assert_eq!(
            times,
            [
                start + Duration::from_millis(1),
                start + Duration::from_millis(2)
            ]
        );
    }

    #[test]
    fn a_zero_rate_is_not_evidence() {
        let mut window = ThroughputWindow::default();
        window.record(Instant::now(), 0);
        assert!(window.samples.is_empty());
    }
}
