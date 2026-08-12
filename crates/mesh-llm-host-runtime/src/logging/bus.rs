//! Bounded nonblocking replay bus for logging events.
//!
//! **Overflow policy: drop-oldest.** When the queue is full, the oldest entry is evicted to make room. This preserves recent context at the cost of losing aged entries. Evictions and rejected entries are counted separately: an eviction records the displaced replay entry, while a drop records the new event that could not be accepted.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use tokio::sync::Notify;

/// Entry on the replay bus carrying a serialized event payload.
#[derive(Clone, Debug)]
pub struct BusEntry {
    /// Serialized JSON/event content to persist. The sink deserializes this into domain types (lifecycle events, summaries, etc.). Payloads are sanitized before enqueue via the privacy policy redactor.
    pub payload: String,

    /// Channel routing hint for downstream consumers (requests/operations/system). This helps workers route entries without parsing JSON.
    #[allow(dead_code)]
    pub channel_hint: u8, // 0=requests, 1=operations, 2=system — matches ReplayChannel discriminant
}

/// Result of a nonblocking replay-bus enqueue attempt.
///
/// A full enabled bus always accepts the new entry by evicting its oldest
/// replay entry. A zero-capacity bus is intentionally disabled and rejects
/// every entry without allocating or notifying a consumer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PushOutcome {
    /// The entry was appended without evicting a prior replay entry.
    Enqueued,
    /// The entry was appended after evicting the oldest replay entry.
    EvictedOldest,
    /// The entry was not accepted because this bus has no configured capacity.
    Rejected,
}

/// Bounded nonblocking replay bus with drop-oldest overflow policy.
///
/// When `push` is called and the queue is already at capacity, the oldest entry
/// is evicted (popped from the front) before the new entry is appended. This ensures
/// recent context survives under pressure while older entries are discarded.
#[derive(Debug)]
pub struct ReplayBus {
    state: Mutex<ReplayBusState>,
    notify: Notify,

    /// Number of new events that could not be accepted by the bus.
    pub drops: Arc<AtomicU64>,

    /// Number of oldest entries evicted to make room for new ones (under drop-oldest).
    pub evictions: Arc<AtomicU64>,
}

#[derive(Debug)]
struct ReplayBusState {
    capacity: usize,
    entries: VecDeque<BusEntry>,
}

impl ReplayBus {
    /// Create a new bus with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            state: Mutex::new(ReplayBusState {
                capacity,
                entries: VecDeque::with_capacity(capacity),
            }),
            notify: Notify::new(),
            drops: Arc::new(AtomicU64::new(0)),
            evictions: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Push an entry onto the bus. If full, drop-oldest applies (evict front, push back).
    pub fn push(&self, payload: String) -> PushOutcome {
        self.push_with_hint(payload, 0)
    }

    /// Push with a channel hint for downstream routing.
    #[allow(dead_code)]
    pub fn push_with_hint(&self, payload: String, channel_hint: u8) -> PushOutcome {
        let mut state = self.state.lock().expect("bus mutex poisoned");

        if state.capacity == 0 {
            self.drops.fetch_add(1, Ordering::Relaxed);
            return PushOutcome::Rejected;
        }

        let outcome = if state.entries.len() >= state.capacity {
            // Drop oldest to make room.
            state.entries.pop_front();
            self.evictions.fetch_add(1, Ordering::Relaxed);
            PushOutcome::EvictedOldest
        } else {
            PushOutcome::Enqueued
        };

        state.entries.push_back(BusEntry {
            payload,
            channel_hint,
        });
        drop(state);
        self.notify.notify_one();
        outcome
    }

    /// Drain all entries from the bus for batch processing by the persistence worker.
    pub fn drain(&self) -> Vec<BusEntry> {
        let mut state = self.state.lock().expect("bus mutex poisoned");
        state.entries.drain(..).collect()
    }

    /// Current number of buffered entries (for observability / tests).
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.state.lock().expect("bus mutex poisoned").entries.len()
    }

    /// Current replay capacity. This is read under the same lock as entries,
    /// so callers never observe a capacity adjustment half-applied.
    #[allow(dead_code)]
    pub fn capacity(&self) -> usize {
        self.state.lock().expect("bus mutex poisoned").capacity
    }

    /// Atomically update replay capacity and trim oldest entries as required.
    ///
    /// Shrinking is deterministic: the oldest replay entries are evicted first,
    /// preserving the newest contiguous suffix. Each removed entry increments
    /// the existing eviction counter; a grow never fabricates eviction/drop
    /// accounting. A zero capacity remains a valid disabled replay bus.
    pub fn set_capacity(&self, capacity: usize) -> u64 {
        let mut state = self.state.lock().expect("bus mutex poisoned");
        state.capacity = capacity;
        let mut evicted = 0;
        while state.entries.len() > capacity {
            state.entries.pop_front();
            evicted += 1;
        }
        if evicted != 0 {
            self.evictions.fetch_add(evicted, Ordering::Relaxed);
        }
        evicted
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Wait until at least one entry is available (or the bus has been signalled).
    pub async fn notified(&self) {
        self.notify.notified().await;
    }

    /// Clone the drops counter for external observation.
    #[allow(dead_code)]
    pub fn drops_clone(&self) -> Arc<AtomicU64> {
        self.drops.clone()
    }

    /// Clone the evictions counter for external observation.
    #[allow(dead_code)]
    pub fn evictions_clone(&self) -> Arc<AtomicU64> {
        self.evictions.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_within_capacity_no_eviction() {
        let bus = ReplayBus::new(3);
        assert_eq!(bus.push("a".into()), PushOutcome::Enqueued);
        assert_eq!(bus.push("b".into()), PushOutcome::Enqueued);
        assert_eq!(bus.len(), 2);
        assert_eq!(bus.drops.load(Ordering::Relaxed), 0);
        assert_eq!(bus.evictions.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn overflow_drops_oldest() {
        let bus = ReplayBus::new(2);
        assert_eq!(bus.push("old".into()), PushOutcome::Enqueued);
        assert_eq!(bus.push("keep".into()), PushOutcome::Enqueued);
        assert_eq!(bus.len(), 2);

        // Push third → evicts "old"
        assert_eq!(bus.push("new".into()), PushOutcome::EvictedOldest);
        assert_eq!(bus.len(), 2);
        assert_eq!(bus.drops.load(Ordering::Relaxed), 0);
        assert_eq!(bus.evictions.load(Ordering::Relaxed), 1);

        let entries = bus.drain();
        assert_eq!(entries.len(), 2);
        // First entry should be "keep", not "old" (oldest was dropped).
        assert_eq!(entries[0].payload, "keep");
        assert_eq!(entries[1].payload, "new");
    }

    #[test]
    fn drain_is_empty_after() {
        let bus = ReplayBus::new(4);
        bus.push("x".into());
        bus.drain();
        assert!(bus.is_empty());
    }

    #[test]
    fn zero_capacity_rejects_without_eviction() {
        let bus = ReplayBus::new(0);
        assert_eq!(bus.push_with_hint("a".into(), 2), PushOutcome::Rejected);
        assert_eq!(bus.len(), 0);
        assert_eq!(bus.drops.load(Ordering::Relaxed), 1);
        assert_eq!(bus.evictions.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn hinted_overflow_preserves_new_hint_and_counts_only_eviction() {
        let bus = ReplayBus::new(1);
        assert_eq!(bus.push_with_hint("old".into(), 1), PushOutcome::Enqueued);
        assert_eq!(
            bus.push_with_hint("new".into(), 2),
            PushOutcome::EvictedOldest
        );

        let entries = bus.drain();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].payload, "new");
        assert_eq!(entries[0].channel_hint, 2);
        assert_eq!(bus.drops.load(Ordering::Relaxed), 0);
        assert_eq!(bus.evictions.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn shrinking_trims_oldest_entries_and_counts_each_eviction() {
        let bus = ReplayBus::new(4);
        for value in ["a", "b", "c", "d"] {
            assert_eq!(bus.push(value.into()), PushOutcome::Enqueued);
        }

        assert_eq!(bus.set_capacity(2), 2);
        assert_eq!(bus.capacity(), 2);
        assert_eq!(bus.evictions.load(Ordering::Relaxed), 2);
        let entries = bus.drain();
        assert_eq!(
            entries
                .into_iter()
                .map(|entry| entry.payload)
                .collect::<Vec<_>>(),
            vec!["c", "d"]
        );
    }

    #[test]
    fn growing_preserves_existing_replay_without_evictions() {
        let bus = ReplayBus::new(1);
        assert_eq!(bus.push("a".into()), PushOutcome::Enqueued);

        assert_eq!(bus.set_capacity(3), 0);
        assert_eq!(bus.capacity(), 3);
        assert_eq!(bus.push("b".into()), PushOutcome::Enqueued);
        assert_eq!(bus.evictions.load(Ordering::Relaxed), 0);
        assert_eq!(bus.len(), 2);
    }
}
