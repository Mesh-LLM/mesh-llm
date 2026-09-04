//! Coalesced engine health counters.
//!
//! Health is out-of-band from the primary terminal/state/progress lanes: it
//! is never recursively submitted through those lanes, and publication is
//! cadence-gated (at most one frame per second) rather than emitted per
//! counter increment.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(test)]
use std::time::Duration;
use std::time::Instant;

use super::config::HEALTH_PUBLISH_MIN_INTERVAL;

/// Point-in-time counters, safe to clone and hand to a consumer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EngineHealthSnapshot {
    pub rebuild_generation: u64,
    pub reservation_exhausted: u64,
    pub terminal_delivery_failed: u64,
    pub dropped_progress: u64,
    pub dropped_diagnostic: u64,
    pub replay_evicted: u64,
    pub subscriber_disconnected: u64,
    pub shutdown_degraded: u64,
    pub reducer_rejected: u64,
    pub event_cutover_divergence: u64,
}

#[derive(Debug, Default)]
struct Counters {
    rebuild_generation: AtomicU64,
    reservation_exhausted: AtomicU64,
    terminal_delivery_failed: AtomicU64,
    dropped_progress: AtomicU64,
    dropped_diagnostic: AtomicU64,
    replay_evicted: AtomicU64,
    subscriber_disconnected: AtomicU64,
    shutdown_degraded: AtomicU64,
    reducer_rejected: AtomicU64,
    event_cutover_divergence: AtomicU64,
}

/// Engine health: coalesced counters plus a cadence-gated publish gate.
#[derive(Debug)]
pub struct EngineHealth {
    counters: Counters,
    last_published: Mutex<Option<Instant>>,
}

impl Default for EngineHealth {
    fn default() -> Self {
        Self {
            counters: Counters::default(),
            last_published: Mutex::new(None),
        }
    }
}

impl EngineHealth {
    pub fn bump_reservation_exhausted(&self) {
        self.counters
            .reservation_exhausted
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn bump_terminal_delivery_failed(&self) {
        self.counters
            .terminal_delivery_failed
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn bump_dropped_progress(&self) {
        self.counters
            .dropped_progress
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn bump_dropped_diagnostic(&self) {
        self.counters
            .dropped_diagnostic
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn bump_replay_evicted(&self) {
        self.counters.replay_evicted.fetch_add(1, Ordering::Relaxed);
    }

    pub fn bump_subscriber_disconnected(&self) {
        self.counters
            .subscriber_disconnected
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn bump_shutdown_degraded(&self) {
        self.counters
            .shutdown_degraded
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn bump_reducer_rejected(&self) {
        self.counters
            .reducer_rejected
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Task 6 (`.omo/plans/event-system-fixes.md`, defect D14): a
    /// `runtime_data::event_cutover` shadow comparison found a legacy
    /// value that disagreed with the reducer's own projection. The legacy
    /// value stays authoritative regardless -- this counter is
    /// observability only, never a cutover trigger.
    pub fn bump_event_cutover_divergence(&self) {
        self.counters
            .event_cutover_divergence
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn set_rebuild_generation(&self, value: u64) {
        self.counters
            .rebuild_generation
            .store(value, Ordering::Relaxed);
    }

    /// Snapshot counters without applying the publish cadence gate.
    #[must_use]
    pub fn snapshot(&self) -> EngineHealthSnapshot {
        EngineHealthSnapshot {
            rebuild_generation: self.counters.rebuild_generation.load(Ordering::Relaxed),
            reservation_exhausted: self.counters.reservation_exhausted.load(Ordering::Relaxed),
            terminal_delivery_failed: self
                .counters
                .terminal_delivery_failed
                .load(Ordering::Relaxed),
            dropped_progress: self.counters.dropped_progress.load(Ordering::Relaxed),
            dropped_diagnostic: self.counters.dropped_diagnostic.load(Ordering::Relaxed),
            replay_evicted: self.counters.replay_evicted.load(Ordering::Relaxed),
            subscriber_disconnected: self
                .counters
                .subscriber_disconnected
                .load(Ordering::Relaxed),
            shutdown_degraded: self.counters.shutdown_degraded.load(Ordering::Relaxed),
            reducer_rejected: self.counters.reducer_rejected.load(Ordering::Relaxed),
            event_cutover_divergence: self
                .counters
                .event_cutover_divergence
                .load(Ordering::Relaxed),
        }
    }

    /// Cadence-gated publish: `Some(snapshot)` at most once per
    /// [`HEALTH_PUBLISH_MIN_INTERVAL`], `None` otherwise (coalesced).
    ///
    /// `now` is caller-supplied so tests stay deterministic without sleeping
    /// on the wall clock.
    pub fn publish_at(&self, now: Instant) -> Option<EngineHealthSnapshot> {
        let mut last = self
            .last_published
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let due = match *last {
            None => true,
            Some(previous) => now.duration_since(previous) >= HEALTH_PUBLISH_MIN_INTERVAL,
        };
        if !due {
            return None;
        }
        *last = Some(now);
        Some(self.snapshot())
    }

    #[cfg(test)]
    pub fn min_interval() -> Duration {
        HEALTH_PUBLISH_MIN_INTERVAL
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn publish_is_coalesced_within_the_cadence_window() {
        let health = EngineHealth::default();
        let start = Instant::now();
        health.bump_reservation_exhausted();

        assert!(health.publish_at(start).is_some());
        assert!(
            health
                .publish_at(start + Duration::from_millis(1))
                .is_none()
        );
        assert!(
            health
                .publish_at(start + EngineHealth::min_interval())
                .is_some()
        );
    }

    #[test]
    fn snapshot_reflects_every_counter() {
        let health = EngineHealth::default();
        health.bump_reservation_exhausted();
        health.bump_terminal_delivery_failed();
        health.bump_dropped_progress();
        health.bump_dropped_diagnostic();
        health.bump_replay_evicted();
        health.bump_subscriber_disconnected();
        health.bump_shutdown_degraded();
        health.bump_reducer_rejected();
        health.bump_event_cutover_divergence();
        health.set_rebuild_generation(2);

        let snapshot = health.snapshot();
        assert_eq!(
            snapshot,
            EngineHealthSnapshot {
                rebuild_generation: 2,
                reservation_exhausted: 1,
                terminal_delivery_failed: 1,
                dropped_progress: 1,
                dropped_diagnostic: 1,
                replay_evicted: 1,
                subscriber_disconnected: 1,
                shutdown_degraded: 1,
                reducer_rejected: 1,
                event_cutover_divergence: 1,
            }
        );
    }
}
