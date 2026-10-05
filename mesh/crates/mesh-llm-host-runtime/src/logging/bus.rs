//! Bounded nonblocking replay bus for logging events.
//!
//! **Overflow policy: drop-oldest.** When the queue is full, the oldest entry is evicted to make room. This preserves recent context at the cost of losing aged entries. Evictions and rejected entries are counted separately: an eviction records the displaced replay entry, while a drop records the new event that could not be accepted.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use mesh_llm_events::logging::replay::{ReplayChannel, ReplaySequence};
use tokio::sync::{Notify, broadcast};

use super::metrics::{LoggingMetric, LoggingMetrics};

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
pub struct ReplayBus {
    state: Mutex<ReplayBusState>,
    notify: Notify,
    updates: broadcast::Sender<ReplayUpdate>,
    metrics: LoggingMetrics,

    /// Number of new events that could not be accepted by the bus.
    pub drops: Arc<AtomicU64>,

    /// Number of oldest entries evicted to make room for new ones (under drop-oldest).
    pub evictions: Arc<AtomicU64>,
}

#[derive(Debug)]
struct ReplayBusState {
    capacity: usize,
    /// One-time delivery ownership for the persistence worker. Draining this
    /// queue must never consume the independently readable replay history.
    entries: VecDeque<BusEntry>,
    replay: ReplayHistory,
    audit_replay: AuditReplayHistory,
}

/// Per-channel position carried by an SSE `id:` value. A vector cursor is
/// necessary because lifecycle sequences are monotonic per replay channel.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ReplayCursor {
    requests: u64,
    operations: u64,
    system: u64,
}

impl ReplayCursor {
    pub fn sequence(self, channel: ReplayChannel) -> u64 {
        match channel {
            ReplayChannel::Requests => self.requests,
            ReplayChannel::Operations => self.operations,
            ReplayChannel::System => self.system,
        }
    }

    fn advance(&mut self, replay: ReplaySequence) {
        let slot = match replay.channel {
            ReplayChannel::Requests => &mut self.requests,
            ReplayChannel::Operations => &mut self.operations,
            ReplayChannel::System => &mut self.system,
        };
        *slot = (*slot).max(replay.sequence);
    }
}

/// A replayable copy of an admitted bus entry. The payload is still internal;
/// the logs SSE semantic module projects it to a privacy-safe public DTO.
#[derive(Clone, Debug)]
pub struct ReplayRecord {
    pub entry: BusEntry,
    pub replay: ReplaySequence,
    pub cursor: ReplayCursor,
}

/// Non-destructive snapshot of the bounded replay window.
#[derive(Clone, Debug)]
pub struct ReplayWindow {
    pub records: Vec<Arc<ReplayRecord>>,
    /// Highest sequence that was evicted for each channel. A reconnect whose
    /// cursor is behind this boundary must receive a replay-gap frame.
    pub evicted_through: ReplayCursor,
    pub latest: ReplayCursor,
}

/// A replayable copy of an admitted operational audit entry. Audit entries
/// use their own monotonic sequence, independent of the lifecycle channels.
#[derive(Clone, Debug)]
pub struct AuditReplayRecord {
    pub entry: BusEntry,
    pub sequence: u64,
    pub cursor: u64,
}

/// Non-destructive snapshot of the bounded audit replay window.
#[derive(Clone, Debug)]
pub struct AuditReplayWindow {
    pub records: Vec<Arc<AuditReplayRecord>>,
    /// Highest audit sequence that was evicted. A reconnect whose audit cursor
    /// is behind this boundary must receive an audit replay-gap frame.
    pub evicted_through: u64,
    pub latest: u64,
}

/// One newly admitted replay record plus the bounded-history state at its
/// admission point. SSE subscribers consume this delta during normal live
/// delivery and fall back to a replay snapshot only after receiver lag.
#[derive(Clone, Debug)]
pub(crate) enum ReplayUpdate {
    Lifecycle {
        record: Arc<ReplayRecord>,
        evicted_through: ReplayCursor,
        latest: ReplayCursor,
    },
    Audit {
        record: Arc<AuditReplayRecord>,
        evicted_through: u64,
    },
}

#[derive(Debug)]
struct ReplayHistory {
    records: VecDeque<Arc<ReplayRecord>>,
    evicted_through: ReplayCursor,
    latest: ReplayCursor,
}

#[derive(Debug)]
struct AuditReplayHistory {
    records: VecDeque<Arc<AuditReplayRecord>>,
    evicted_through: u64,
    latest: u64,
}

impl AuditReplayHistory {
    fn new(capacity: usize) -> Self {
        Self {
            records: VecDeque::with_capacity(capacity),
            evicted_through: 0,
            latest: 0,
        }
    }

    fn push(&mut self, capacity: usize, entry: BusEntry) -> (bool, Arc<AuditReplayRecord>) {
        let sequence = self.latest + 1;
        let evicted = if self.records.len() >= capacity {
            if let Some(record) = self.records.pop_front() {
                self.evicted_through = self.evicted_through.max(record.sequence);
            }
            true
        } else {
            false
        };
        let record = Arc::new(AuditReplayRecord {
            entry,
            sequence,
            cursor: sequence,
        });
        self.records.push_back(Arc::clone(&record));
        self.latest = sequence;
        (evicted, record)
    }

    fn trim_to(&mut self, capacity: usize) -> u64 {
        let mut evicted = 0;
        while self.records.len() > capacity {
            if let Some(record) = self.records.pop_front() {
                self.evicted_through = self.evicted_through.max(record.sequence);
                evicted += 1;
            }
        }
        evicted
    }

    fn snapshot(&self) -> AuditReplayWindow {
        AuditReplayWindow {
            records: self.records.iter().cloned().collect(),
            evicted_through: self.evicted_through,
            latest: self.latest,
        }
    }
}

impl ReplayHistory {
    fn new(capacity: usize) -> Self {
        Self {
            records: VecDeque::with_capacity(capacity),
            evicted_through: ReplayCursor::default(),
            latest: ReplayCursor::default(),
        }
    }

    fn push(
        &mut self,
        capacity: usize,
        entry: BusEntry,
        replay: ReplaySequence,
    ) -> (bool, Arc<ReplayRecord>) {
        self.latest.advance(replay);
        let evicted = if self.records.len() >= capacity {
            if let Some(record) = self.records.pop_front() {
                self.evicted_through.advance(record.replay);
            }
            true
        } else {
            false
        };
        let record = Arc::new(ReplayRecord {
            entry,
            replay,
            cursor: self.latest,
        });
        self.records.push_back(Arc::clone(&record));
        (evicted, record)
    }

    fn trim_to(&mut self, capacity: usize) -> u64 {
        let mut evicted = 0;
        while self.records.len() > capacity {
            if let Some(record) = self.records.pop_front() {
                self.evicted_through.advance(record.replay);
                evicted += 1;
            }
        }
        evicted
    }

    fn snapshot(&self) -> ReplayWindow {
        ReplayWindow {
            records: self.records.iter().cloned().collect(),
            evicted_through: self.evicted_through,
            latest: self.latest,
        }
    }
}

impl ReplayBus {
    /// Create a new bus with the given capacity.
    pub fn new(capacity: usize) -> Self {
        let (updates, _) = broadcast::channel(capacity.max(1));
        Self {
            state: Mutex::new(ReplayBusState {
                capacity,
                entries: VecDeque::with_capacity(capacity),
                replay: ReplayHistory::new(capacity),
                audit_replay: AuditReplayHistory::new(capacity),
            }),
            notify: Notify::new(),
            updates,
            metrics: LoggingMetrics::default(),
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
        self.push_inner(payload, channel_hint, None)
    }

    /// Push a canonical lifecycle event and retain an independently readable
    /// replay copy. Persistence workers still receive only the delivery queue.
    pub fn push_replay(
        &self,
        payload: String,
        channel_hint: u8,
        replay: ReplaySequence,
    ) -> PushOutcome {
        self.push_inner(payload, channel_hint, Some(replay))
    }

    /// Push an audit entry onto the bus and retain a copy in the audit replay
    /// window. Uses an independent monotonic sequence (not ReplaySequence).
    /// Mirrors the delivery-queue + notify + broadcast behavior of push_inner.
    pub fn push_audit_replay(&self, payload: String, channel_hint: u8) -> PushOutcome {
        let mut state = self.state.lock().expect("bus mutex poisoned");

        if state.capacity == 0 {
            self.drops.fetch_add(1, Ordering::Relaxed);
            drop(state);
            self.metrics
                .record(LoggingMetric::ReplayDropped { count: 1 });
            return PushOutcome::Rejected;
        }

        let delivery_evicted = if state.entries.len() >= state.capacity {
            state.entries.pop_front();
            true
        } else {
            false
        };

        let entry = BusEntry {
            payload,
            channel_hint,
        };
        state.entries.push_back(entry.clone());
        let capacity = state.capacity;
        let (audit_evicted, record) = state.audit_replay.push(capacity, entry);
        let update = ReplayUpdate::Audit {
            record,
            evicted_through: state.audit_replay.evicted_through,
        };
        let outcome = if delivery_evicted || audit_evicted {
            self.evictions.fetch_add(1, Ordering::Relaxed);
            PushOutcome::EvictedOldest
        } else {
            PushOutcome::Enqueued
        };
        drop(state);
        if matches!(outcome, PushOutcome::EvictedOldest) {
            self.metrics
                .record(LoggingMetric::ReplayEvicted { count: 1 });
        }
        self.notify.notify_one();
        let _ = self.updates.send(update);
        outcome
    }

    /// Snapshot the audit replay history without consuming it.
    pub fn audit_replay_window(&self) -> AuditReplayWindow {
        self.state
            .lock()
            .expect("bus mutex poisoned")
            .audit_replay
            .snapshot()
    }

    fn push_inner(
        &self,
        payload: String,
        channel_hint: u8,
        replay: Option<ReplaySequence>,
    ) -> PushOutcome {
        let mut state = self.state.lock().expect("bus mutex poisoned");

        if state.capacity == 0 {
            self.drops.fetch_add(1, Ordering::Relaxed);
            drop(state);
            self.metrics
                .record(LoggingMetric::ReplayDropped { count: 1 });
            return PushOutcome::Rejected;
        }

        let delivery_evicted = if state.entries.len() >= state.capacity {
            // Drop oldest to make room.
            state.entries.pop_front();
            true
        } else {
            false
        };

        let entry = BusEntry {
            payload,
            channel_hint,
        };
        state.entries.push_back(entry.clone());
        let capacity = state.capacity;
        let (replay_evicted, update) = replay.map_or((false, None), |replay| {
            let (evicted, record) = state.replay.push(capacity, entry, replay);
            let update = ReplayUpdate::Lifecycle {
                record,
                evicted_through: state.replay.evicted_through,
                latest: state.replay.latest,
            };
            (evicted, Some(update))
        });
        let outcome = if delivery_evicted || replay_evicted {
            self.evictions.fetch_add(1, Ordering::Relaxed);
            PushOutcome::EvictedOldest
        } else {
            PushOutcome::Enqueued
        };
        drop(state);
        if matches!(outcome, PushOutcome::EvictedOldest) {
            self.metrics
                .record(LoggingMetric::ReplayEvicted { count: 1 });
        }
        self.notify.notify_one();
        if let Some(update) = update {
            let _ = self.updates.send(update);
        }
        outcome
    }

    /// Drain all entries from the bus for batch processing by the persistence worker.
    pub fn drain(&self) -> Vec<BusEntry> {
        let mut state = self.state.lock().expect("bus mutex poisoned");
        state.entries.drain(..).collect()
    }

    /// Snapshot the separately retained replay history without consuming it.
    /// This remains available after the persistence worker drains delivery.
    pub fn replay_window(&self) -> ReplayWindow {
        self.state
            .lock()
            .expect("bus mutex poisoned")
            .replay
            .snapshot()
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
        let mut delivery_evicted = 0;
        while state.entries.len() > capacity {
            state.entries.pop_front();
            delivery_evicted += 1;
        }
        let replay_evicted = state.replay.trim_to(capacity);
        let audit_evicted = state.audit_replay.trim_to(capacity);
        let evicted = delivery_evicted.max(replay_evicted).max(audit_evicted);
        if evicted != 0 {
            self.evictions.fetch_add(evicted, Ordering::Relaxed);
        }
        drop(state);
        if evicted != 0 {
            self.metrics
                .record(LoggingMetric::ReplayEvicted { count: evicted });
        }
        evicted
    }

    /// Record replay gaps only where an existing SSE session has determined
    /// that an evicted cursor requires recovery. This does not alter the
    /// replay protocol or expose its cursor/channel values to telemetry.
    pub(crate) fn record_replay_gaps(&self, count: u64) {
        if count != 0 {
            self.metrics.record(LoggingMetric::ReplayGap { count });
        }
    }

    pub(crate) fn metrics(&self) -> LoggingMetrics {
        self.metrics.clone()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Wait until at least one entry is available (or the bus has been signalled).
    pub async fn notified(&self) {
        self.notify.notified().await;
    }

    /// Subscribe to accepted replay deltas. Each SSE connection owns its
    /// receiver, so one slow or cancelled subscriber cannot consume another
    /// connection's update.
    pub(crate) fn subscribe_updates(&self) -> broadcast::Receiver<ReplayUpdate> {
        self.updates.subscribe()
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

    #[test]
    fn replay_window_survives_a_persistence_drain() {
        let bus = ReplayBus::new(2);
        bus.push_replay(
            "first".into(),
            0,
            ReplaySequence::next(ReplayChannel::Requests, 1),
        );
        bus.push_replay(
            "second".into(),
            1,
            ReplaySequence::next(ReplayChannel::Operations, 1),
        );

        assert_eq!(bus.drain().len(), 2);
        assert!(bus.is_empty(), "delivery queue was consumed");
        let replay = bus.replay_window();
        assert_eq!(replay.records.len(), 2, "replay remains readable");
        assert_eq!(replay.records[0].replay.sequence, 1);
        assert_eq!(replay.records[1].replay.channel, ReplayChannel::Operations);
    }

    #[test]
    fn replay_window_reports_per_channel_eviction_boundary() {
        let bus = ReplayBus::new(1);
        bus.push_replay(
            "old".into(),
            0,
            ReplaySequence::next(ReplayChannel::Requests, 1),
        );
        bus.push_replay(
            "new".into(),
            0,
            ReplaySequence::next(ReplayChannel::Requests, 2),
        );

        let replay = bus.replay_window();
        assert_eq!(replay.evicted_through.sequence(ReplayChannel::Requests), 1);
        assert_eq!(replay.latest.sequence(ReplayChannel::Requests), 2);
        assert_eq!(replay.records[0].replay.sequence, 2);
    }

    #[tokio::test]
    async fn replay_updates_fan_out_to_each_subscriber() {
        let bus = ReplayBus::new(1);
        let mut first = bus.subscribe_updates();
        let mut second = bus.subscribe_updates();
        bus.push_replay(
            "event".into(),
            0,
            ReplaySequence::next(ReplayChannel::Requests, 1),
        );

        for update in [first.recv().await, second.recv().await] {
            let ReplayUpdate::Lifecycle { record, .. } = update.expect("replay update") else {
                panic!("lifecycle pushes must broadcast lifecycle deltas");
            };
            assert_eq!(record.replay.sequence, 1);
            assert_eq!(record.entry.payload, "event");
        }
    }

    #[test]
    fn audit_window_retains_entries_in_order_with_monotonic_sequences() {
        let bus = ReplayBus::new(3);
        assert_eq!(
            bus.push_audit_replay("audit-a".into(), 2),
            PushOutcome::Enqueued
        );
        assert_eq!(
            bus.push_audit_replay("audit-b".into(), 2),
            PushOutcome::Enqueued
        );
        assert_eq!(
            bus.push_audit_replay("audit-c".into(), 2),
            PushOutcome::Enqueued
        );

        let window = bus.audit_replay_window();
        assert_eq!(window.records.len(), 3);
        assert_eq!(window.records[0].sequence, 1);
        assert_eq!(window.records[0].cursor, 1);
        assert_eq!(window.records[0].entry.payload, "audit-a");
        assert_eq!(window.records[1].sequence, 2);
        assert_eq!(window.records[2].sequence, 3);
        assert_eq!(window.latest, 3);
        assert_eq!(window.evicted_through, 0);
    }

    #[test]
    fn audit_window_drop_oldest_eviction_advances_evicted_through() {
        let bus = ReplayBus::new(2);
        assert_eq!(
            bus.push_audit_replay("old".into(), 2),
            PushOutcome::Enqueued
        );
        assert_eq!(
            bus.push_audit_replay("keep".into(), 2),
            PushOutcome::Enqueued
        );

        // Push third → evicts "old" (sequence 1)
        assert_eq!(
            bus.push_audit_replay("new".into(), 2),
            PushOutcome::EvictedOldest
        );
        assert_eq!(bus.evictions.load(Ordering::Relaxed), 1);

        let window = bus.audit_replay_window();
        assert_eq!(window.records.len(), 2);
        assert_eq!(window.records[0].entry.payload, "keep");
        assert_eq!(window.records[1].entry.payload, "new");
        assert_eq!(window.evicted_through, 1);
        assert_eq!(window.latest, 3);
    }

    #[test]
    fn audit_window_survives_drain() {
        let bus = ReplayBus::new(2);
        bus.push_audit_replay("audit-1".into(), 2);
        bus.push_audit_replay("audit-2".into(), 2);

        assert_eq!(bus.drain().len(), 2);
        assert!(bus.is_empty(), "delivery queue was consumed");

        let window = bus.audit_replay_window();
        assert_eq!(window.records.len(), 2, "audit replay remains readable");
        assert_eq!(window.records[0].sequence, 1);
        assert_eq!(window.records[1].sequence, 2);
    }

    #[tokio::test]
    async fn audit_updates_broadcast_to_subscribers() {
        let bus = ReplayBus::new(1);
        let mut first = bus.subscribe_updates();
        let mut second = bus.subscribe_updates();
        bus.push_audit_replay("audit-event".into(), 2);

        for update in [first.recv().await, second.recv().await] {
            let ReplayUpdate::Audit { record, .. } = update.expect("audit update") else {
                panic!("audit pushes must broadcast audit deltas");
            };
            assert_eq!(record.sequence, 1);
            assert_eq!(record.entry.payload, "audit-event");
        }
    }

    #[test]
    fn audit_zero_capacity_rejects_without_allocation() {
        let bus = ReplayBus::new(0);
        assert_eq!(
            bus.push_audit_replay("audit".into(), 2),
            PushOutcome::Rejected
        );
        assert_eq!(bus.len(), 0);
        assert_eq!(bus.drops.load(Ordering::Relaxed), 1);
        assert_eq!(bus.evictions.load(Ordering::Relaxed), 0);
        let window = bus.audit_replay_window();
        assert_eq!(window.records.len(), 0);
        assert_eq!(window.evicted_through, 0);
        assert_eq!(window.latest, 0);
    }
}
