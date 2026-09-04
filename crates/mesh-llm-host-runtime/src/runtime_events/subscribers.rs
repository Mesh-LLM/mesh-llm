//! Bounded in-process subscriber handles.
//!
//! `tokio::sync::broadcast` already provides a fixed-capacity ring buffer
//! with lag detection (`Lagged(n)`), which matches the plan's per-subscriber
//! FRAME-COUNT lag bound directly; a subscriber that falls more than
//! `SUBSCRIBER_LAG_MAX_FRAMES` behind is disconnected by the channel itself
//! (its next `recv()` returns `Lagged`) rather than by a hand-rolled queue
//! here. The AGE and BYTES dimensions of the same frozen bound have no
//! channel-native equivalent (the ring only tracks position, not staleness
//! or size), so [`lag_bound_exceeded`] checks them explicitly against each
//! received frame; the caller (`api/routes/runtime_events/stream.rs`)
//! disconnects on either signal, matching "disconnect at the first
//! configured limit" for all three dimensions.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use tokio::sync::broadcast;

use super::config::{
    MAX_CONCURRENT_SUBSCRIBERS, SUBSCRIBER_LAG_MAX_AGE, SUBSCRIBER_LAG_MAX_BYTES,
    SUBSCRIBER_LAG_MAX_FRAMES,
};
use super::health::EngineHealth;
use super::replay::ReplayFrame;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubscribeError {
    CapacityReached,
}

struct RegistryInner {
    sender: broadcast::Sender<ReplayFrame>,
    active: AtomicUsize,
}

/// Shared registry: `publish` fans a frame out to every live subscriber.
#[derive(Clone)]
pub struct SubscriberRegistry {
    inner: Arc<RegistryInner>,
}

/// A live subscription. Dropping it frees a slot back to the registry.
pub struct SubscriptionHandle {
    registry: Arc<RegistryInner>,
    receiver: broadcast::Receiver<ReplayFrame>,
}

impl SubscriberRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::with_capacity(SUBSCRIBER_LAG_MAX_FRAMES)
    }

    /// Same as [`Self::new`] with an explicit frame-count lag capacity;
    /// tests use this to shrink the ring so a "slow subscriber" scenario
    /// (falls behind while events flow, discovers it on its next `recv`)
    /// is reachable in a handful of publishes instead of the frozen 1,024.
    #[must_use]
    pub fn with_capacity(lag_frames: usize) -> Self {
        let (sender, _receiver) = broadcast::channel(lag_frames);
        Self {
            inner: Arc::new(RegistryInner {
                sender,
                active: AtomicUsize::new(0),
            }),
        }
    }

    pub fn subscribe(&self) -> Result<SubscriptionHandle, SubscribeError> {
        let active = self.inner.active.fetch_add(1, Ordering::AcqRel);
        if active >= MAX_CONCURRENT_SUBSCRIBERS {
            self.inner.active.fetch_sub(1, Ordering::AcqRel);
            return Err(SubscribeError::CapacityReached);
        }
        Ok(SubscriptionHandle {
            registry: Arc::clone(&self.inner),
            receiver: self.inner.sender.subscribe(),
        })
    }

    /// Fan `frame` out to every live subscriber; a subscriber past its lag
    /// bound observes `Lagged` on its next `recv` rather than blocking this
    /// call, so publish never waits on a consumer.
    pub fn publish(&self, frame: ReplayFrame) {
        let _ = self.inner.sender.send(frame);
    }

    #[must_use]
    pub fn active_count(&self) -> usize {
        self.inner.active.load(Ordering::Acquire)
    }
}

impl Default for SubscriberRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl SubscriptionHandle {
    /// Receive the next frame, or `Err` on lag/close. A `Lagged(n)` error
    /// means this subscriber missed `n` frames and should be counted as
    /// disconnected by the caller via `health.bump_subscriber_disconnected`.
    pub async fn recv(&mut self) -> Result<ReplayFrame, broadcast::error::RecvError> {
        self.receiver.recv().await
    }

    /// The number of published messages this subscriber has not yet
    /// observed. Real tokio-broadcast introspection (`Receiver::len`), used
    /// by [`lag_bound_exceeded`] to approximate this subscriber's current
    /// unread-backlog bytes without walking the whole channel.
    #[must_use]
    pub fn backlog_len(&self) -> usize {
        self.receiver.len()
    }

    pub fn record_disconnect(&self, health: &EngineHealth) {
        health.bump_subscriber_disconnected();
    }
}

/// Whether receiving `frame` (with `backlog_len` still-unread messages
/// behind it, per [`SubscriptionHandle::backlog_len`]) means this
/// subscriber has fallen behind the frozen per-subscriber lag bound on the
/// AGE or BYTES dimension. The FRAME-COUNT dimension is already enforced by
/// the broadcast channel's fixed ring capacity and surfaces as
/// `RecvError::Lagged` before this function is ever reached — first limit
/// wins, and the channel's own limit is checked first by construction (a
/// `Lagged` receive short-circuits the caller's match arm).
///
/// Task 9 (`.omo/plans/event-system-fixes.md`, defect D11): the BYTES
/// check now multiplies `backlog_len` by `frame`'s own REAL
/// `wire_bytes.len()` instead of the removed fixed
/// `APPROX_FRAME_BYTE_COST` estimate. This is still an approximation of
/// the true backlog byte total -- `tokio::sync::broadcast::Receiver` has
/// no API to inspect the SIZE of every still-queued message without
/// consuming them (consuming would defeat the purpose: it would drain the
/// very backlog being measured) -- but it is now grounded in a REAL,
/// currently-observed frame size rather than a compile-time guess,
/// consistent with replay retention's own real-byte accounting
/// (`replay::ReplayBuffer::push_at`).
#[must_use]
pub fn lag_bound_exceeded(frame: &ReplayFrame, backlog_len: usize, now: Instant) -> bool {
    if now.saturating_duration_since(frame.recorded_at) > SUBSCRIBER_LAG_MAX_AGE {
        return true;
    }
    backlog_len.saturating_mul(frame.wire_bytes.len()) > SUBSCRIBER_LAG_MAX_BYTES
}

impl Drop for SubscriptionHandle {
    fn drop(&mut self) {
        self.registry.active.fetch_sub(1, Ordering::AcqRel);
    }
}

#[cfg(test)]
mod tests {
    use mesh_llm_runtime_event_contracts::{
        EventSequence, FamilyFact, NativeRuntimeEventKind, OperationId, OperationScope, RuntimeFact,
    };
    use std::time::Instant;

    use super::*;

    fn frame(sequence: u64) -> ReplayFrame {
        ReplayFrame {
            sequence: EventSequence::new(sequence),
            rebuild_generation: 0,
            scope: OperationScope::root_only(OperationId::new()),
            fact: std::sync::Arc::new(RuntimeFact::NativeRuntime(FamilyFact::new(
                NativeRuntimeEventKind::RuntimeStopped,
            ))),
            recorded_at: Instant::now(),
            wire_bytes: std::sync::Arc::from(Vec::new()),
        }
    }

    #[tokio::test]
    async fn subscriber_receives_published_frames() {
        let registry = SubscriberRegistry::new();
        let mut handle = registry.subscribe().expect("subscribe");

        registry.publish(frame(1));
        let received = handle.recv().await.expect("recv");
        assert_eq!(received.sequence.get(), 1);
    }

    #[test]
    fn subscription_beyond_the_cap_is_rejected() {
        let registry = SubscriberRegistry::new();
        let mut handles = Vec::new();
        for _ in 0..MAX_CONCURRENT_SUBSCRIBERS {
            handles.push(registry.subscribe().expect("subscribe under cap"));
        }

        assert!(matches!(
            registry.subscribe(),
            Err(SubscribeError::CapacityReached)
        ));
        assert_eq!(registry.active_count(), MAX_CONCURRENT_SUBSCRIBERS);
    }

    #[test]
    fn dropping_a_subscription_frees_its_slot() {
        let registry = SubscriberRegistry::new();
        let handle = registry.subscribe().expect("subscribe");
        assert_eq!(registry.active_count(), 1);
        drop(handle);
        assert_eq!(registry.active_count(), 0);
    }

    // ─── Slow subscriber: frame-count dimension ────────────────────────

    #[tokio::test]
    async fn a_subscriber_that_stops_draining_is_lagged_once_it_falls_too_far_behind() {
        let registry = SubscriberRegistry::with_capacity(4);
        let mut handle = registry.subscribe().expect("subscribe");

        // Events keep flowing while this subscriber never calls `recv`.
        for sequence in 0..6 {
            registry.publish(frame(sequence));
        }

        let result = handle.recv().await;
        assert!(
            matches!(result, Err(broadcast::error::RecvError::Lagged(_))),
            "a subscriber that never drained 6 publishes into a 4-slot ring must observe Lagged, got {result:?}"
        );
    }

    #[test]
    fn record_disconnect_bumps_the_subscriber_disconnected_health_counter() {
        let registry = SubscriberRegistry::new();
        let handle = registry.subscribe().expect("subscribe");
        let health = EngineHealth::default();

        handle.record_disconnect(&health);

        assert_eq!(health.snapshot().subscriber_disconnected, 1);
    }

    // ─── Slow subscriber: age and bytes dimensions ─────────────────────

    #[test]
    fn lag_bound_exceeded_by_age_even_with_zero_backlog() {
        let start = Instant::now();
        let mut stale = frame(0);
        stale.recorded_at = start;

        let now = start + SUBSCRIBER_LAG_MAX_AGE + std::time::Duration::from_secs(1);
        assert!(lag_bound_exceeded(&stale, 0, now));
    }

    #[test]
    fn lag_bound_exceeded_by_backlog_bytes_even_for_a_fresh_frame() {
        // Task 9 (`.omo/plans/event-system-fixes.md`): a real, controlled
        // 100-byte wire frame -- not the removed fixed
        // `APPROX_FRAME_BYTE_COST` constant -- drives the backlog-bytes
        // approximation now.
        let mut fresh = frame(0);
        fresh.wire_bytes = std::sync::Arc::from(vec![0u8; 100]);
        let now = fresh.recorded_at;

        // A backlog whose approximate byte cost alone already exceeds the
        // frozen byte bound must trip it regardless of age.
        let backlog_len = SUBSCRIBER_LAG_MAX_BYTES / fresh.wire_bytes.len() + 2;
        assert!(lag_bound_exceeded(&fresh, backlog_len, now));
    }

    #[test]
    fn lag_bound_not_exceeded_for_a_fresh_frame_with_a_small_backlog() {
        let fresh = frame(0);
        let now = fresh.recorded_at;
        assert!(!lag_bound_exceeded(&fresh, 1, now));
    }
}
