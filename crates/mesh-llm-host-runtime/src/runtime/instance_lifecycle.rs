//! Per-instance lifecycle state machine, admission guard, and drain logic.
//!
//! Each runtime model instance (identified by its `instance_id`) owns an
//! `InstanceLifecycleRecord` that tracks its current state, in-flight request
//! count, drain deadline, and bounded transition history. The state machine is
//! transition-safe: only valid transitions are accepted; invalid ones return
//! an error.
//!
//! Drain semantics:
//! 1. `mark_draining(deadline)` atomically sets state to `Draining` and records
//!    the effective deadline (zero = force immediate).
//! 2. While `Draining`, `is_accepting_work()` returns false — new work is
//!    rejected at admission time.
//! 3. Already-admitted work (in-flight > 0) continues until completion or
//!    deadline expiry.
//! 4. At zero in-flight, the instance transitions to `Unloading` immediately.
//! 5. If the deadline expires with in-flight still > 0, force-cancel is
//!    triggered and state moves to `Unloading`.
//!
//! Model-target drain resolves to exactly one instance or returns an ambiguity
//! error. Instance-target drain never affects sibling instances sharing the same
//! model name.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

// ── Lifecycle State ──────────────────────────────────────────────────────────

/// Per-instance lifecycle state. Transitions are validated by the state machine.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum InstanceLifecycleState {
    /// Instance is planned but not yet started (reconciliation intent exists).
    Planned,
    /// Model path/catalog resolution in progress.
    Resolving,
    /// Native runtime loading the model weights into memory.
    Loading,
    /// Model loaded; warming up slots or waiting for readiness probes.
    Warming,
    /// Fully serving requests. The steady state.
    Serving,
    /// Drain initiated: rejecting new work, waiting for in-flight to clear
    /// or deadline to expire.
    Draining,
    /// In-flight cleared (or force-cancelled); unloading resources now.
    Unloading,
    /// Instance encountered an unrecoverable error.
    Failed,
    /// Instance completed its lifecycle cleanly (stopped after unload).
    Stopped,
}

impl InstanceLifecycleState {
    /// Whether this state is a terminal state (no further transitions possible).
    pub(crate) fn is_terminal(self) -> bool {
        matches!(self, Self::Failed | Self::Stopped)
    }

    /// Whether the instance can accept new work in this state.
    pub(crate) fn accepts_work(self) -> bool {
        matches!(self, Self::Serving)
    }

    /// Human-readable label for logging and status output.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "stable lifecycle labels are covered by unit tests"
        )
    )]
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Planned => "planned",
            Self::Resolving => "resolving",
            Self::Loading => "loading",
            Self::Warming => "warming",
            Self::Serving => "serving",
            Self::Draining => "draining",
            Self::Unloading => "unloading",
            Self::Failed => "failed",
            Self::Stopped => "stopped",
        }
    }
}

// ── Transition Validation ────────────────────────────────────────────────────

/// Result of attempting a state transition.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum TransitionResult {
    /// Transition succeeded; previous state is returned for logging.
    Ok(InstanceLifecycleState),
    /// Transition not allowed from current state.
    Invalid {
        from: InstanceLifecycleState,
        to: InstanceLifecycleState,
    },
}

/// Valid transitions between lifecycle states.
fn valid_transitions() -> &'static [(InstanceLifecycleState, &'static [InstanceLifecycleState])] {
    &[
        (
            InstanceLifecycleState::Planned,
            &[
                InstanceLifecycleState::Resolving,
                InstanceLifecycleState::Failed,
                InstanceLifecycleState::Stopped,
            ],
        ),
        (
            InstanceLifecycleState::Resolving,
            &[
                InstanceLifecycleState::Loading,
                InstanceLifecycleState::Failed,
                InstanceLifecycleState::Stopped,
            ],
        ),
        (
            InstanceLifecycleState::Loading,
            &[
                InstanceLifecycleState::Warming,
                InstanceLifecycleState::Failed,
                InstanceLifecycleState::Stopped,
            ],
        ),
        (
            InstanceLifecycleState::Warming,
            &[
                InstanceLifecycleState::Serving,
                InstanceLifecycleState::Failed,
                InstanceLifecycleState::Stopped,
            ],
        ),
        (
            InstanceLifecycleState::Serving,
            &[
                InstanceLifecycleState::Draining,
                InstanceLifecycleState::Failed,
                InstanceLifecycleState::Stopped,
            ],
        ),
        (
            InstanceLifecycleState::Draining,
            &[
                InstanceLifecycleState::Unloading,
                InstanceLifecycleState::Failed,
            ],
        ),
        (
            InstanceLifecycleState::Unloading,
            &[
                InstanceLifecycleState::Stopped,
                InstanceLifecycleState::Failed,
            ],
        ),
        // Terminal states: no outgoing transitions.
        (InstanceLifecycleState::Failed, &[]),
        (InstanceLifecycleState::Stopped, &[]),
    ]
}

// ── In-Flight Tracker ────────────────────────────────────────────────────────

/// Per-instance in-flight request counter. Thread-safe for use across async tasks.
#[derive(Debug)]
pub(crate) struct InFlightTracker {
    inner: std::sync::Mutex<InFlightInner>,
}

#[derive(Debug)]
struct InFlightInner {
    count: u64,
}

impl InFlightTracker {
    pub(crate) fn new() -> Self {
        Self {
            inner: std::sync::Mutex::new(InFlightInner { count: 0 }),
        }
    }

    /// Increment in-flight count. Returns the new count.
    pub(crate) fn increment(&self) -> u64 {
        let mut guard = self.inner.lock().expect("inflight mutex poisoned");
        guard.count = guard.count.saturating_add(1);
        guard.count
    }

    /// Decrement in-flight count. Returns the new count (saturates at 0).
    pub(crate) fn decrement(&self) -> u64 {
        let mut guard = self.inner.lock().expect("inflight mutex poisoned");
        guard.count = guard.count.saturating_sub(1);
        guard.count
    }

    /// Current in-flight count.
    pub(crate) fn get(&self) -> u64 {
        let guard = self.inner.lock().expect("inflight mutex poisoned");
        guard.count
    }

    #[cfg(test)]
    fn set(&self, value: u64) {
        let mut guard = self.inner.lock().expect("inflight mutex poisoned");
        guard.count = value;
    }
}

impl Default for InFlightTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ── Lifecycle Record ─────────────────────────────────────────────────────────

/// Bounded history of lifecycle transitions for status reporting.
#[derive(Clone, Debug)]
pub(crate) struct TransitionEntry {
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "transition history is test-observable")
    )]
    pub(crate) from: InstanceLifecycleState,
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "transition history is test-observable")
    )]
    pub(crate) to: InstanceLifecycleState,
    #[expect(dead_code, reason = "transition timestamps support bounded history")]
    pub(crate) at: Instant,
}

/// Complete lifecycle record for a single runtime instance.
///
/// This is the authoritative source of truth for an instance's state, admission
/// status, and drain progress. Access is synchronized via a tokio mutex so it
/// can be shared across the control loop, proxy admission checks, and unload
/// logic.
#[derive(Debug)]
pub(crate) struct InstanceLifecycleRecord {
    /// Current lifecycle state.
    state: InstanceLifecycleState,

    /// In-flight request counter for this instance.
    in_flight: std::sync::Arc<InFlightTracker>,

    /// When `Draining`, the deadline after which force-cancel is triggered.
    /// `None` means no deadline (wait indefinitely for zero in-flight).
    drain_deadline: Option<Instant>,

    /// Whether a drain has been initiated (idempotency guard).
    draining_initiated: bool,

    /// Bounded history of recent transitions.
    history: VecDeque<TransitionEntry>,

    /// Maximum number of transition entries to retain.
    max_history: usize,

    /// Timestamp when the instance was created.
    #[expect(dead_code, reason = "instance age is test-observable")]
    created_at: Instant,

    /// Optional error message if state is `Failed`.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "lifecycle errors are test-observable")
    )]
    last_error: Option<String>,
}

impl InstanceLifecycleRecord {
    /// Create a new lifecycle record starting in the given state.
    pub(crate) fn new(initial_state: InstanceLifecycleState, max_history: usize) -> Self {
        Self {
            state: initial_state,
            in_flight: std::sync::Arc::new(InFlightTracker::new()),
            drain_deadline: None,
            draining_initiated: false,
            history: VecDeque::with_capacity(max_history.max(1)),
            max_history: max_history.max(1),
            created_at: Instant::now(),
            last_error: None,
        }
    }

    /// Current lifecycle state.
    pub(crate) fn state(&self) -> InstanceLifecycleState {
        self.state
    }

    /// Whether this instance is currently accepting new work.
    /// Only `Serving` state accepts work; all other states reject.
    pub(crate) fn is_accepting_work(&self) -> bool {
        self.state.accepts_work()
    }

    /// Current in-flight request count for this instance.
    pub(crate) fn in_flight_count(&self) -> u64 {
        self.in_flight.get()
    }

    pub(crate) fn in_flight_tracker(&self) -> std::sync::Arc<InFlightTracker> {
        self.in_flight.clone()
    }

    /// Increment the in-flight counter (admit a new request).
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "direct counter API is retained for lifecycle tests"
        )
    )]
    pub(crate) fn admit_request(&self) -> u64 {
        self.in_flight.increment()
    }

    /// Decrement the in-flight counter (complete a request).
    /// Returns `true` if this decrement brought the count to zero while
    /// draining, signaling that unload can proceed immediately.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "direct counter API is retained for lifecycle tests"
        )
    )]
    pub(crate) fn complete_request(&self) -> bool {
        let new_count = self.in_flight.decrement();
        new_count == 0 && self.state == InstanceLifecycleState::Draining
    }

    /// Mark this instance as draining with the given deadline.
    ///
    /// Returns `Ok(())` on success, or an error if:
    /// - The instance is not in a drainable state (not Serving)
    /// - A drain was already initiated (idempotent rejection)
    pub(crate) fn mark_draining(
        &mut self,
        deadline: Instant,
    ) -> Result<(), InstanceLifecycleError> {
        if self.draining_initiated {
            if self.state == InstanceLifecycleState::Draining {
                self.drain_deadline = Some(match self.drain_deadline {
                    Some(current) => current.min(deadline),
                    None => deadline,
                });
            }
            return Ok(());
        }

        if self.state != InstanceLifecycleState::Serving {
            return Err(InstanceLifecycleError::NotDrainable { state: self.state });
        }

        self.transition_to(InstanceLifecycleState::Draining)?;
        self.drain_deadline = Some(deadline);
        self.draining_initiated = true;
        Ok(())
    }

    /// Mark this instance as force-draining (deadline = now).
    pub(crate) fn mark_draining_force(&mut self) -> Result<(), InstanceLifecycleError> {
        let now = Instant::now();
        self.mark_draining(now)
    }

    /// Check if drain deadline has expired. Returns `true` if force-cancel
    /// should be triggered.
    pub(crate) fn is_drain_deadline_expired(&self) -> bool {
        match (self.state, self.drain_deadline) {
            (InstanceLifecycleState::Draining, Some(deadline)) => {
                Instant::now() >= deadline && self.in_flight.get() > 0
            }
            _ => false,
        }
    }

    /// Check if the instance can transition to Unloading.
    /// Returns true if in-flight is zero (graceful) or deadline expired (force).
    pub(crate) fn can_transition_to_unloading(&self) -> bool {
        self.state == InstanceLifecycleState::Draining
            && (self.in_flight.get() == 0 || self.is_drain_deadline_expired())
    }

    /// Transition to Unloading. Only valid from Draining when in-flight is zero
    /// or deadline expired.
    pub(crate) fn transition_to_unloading(&mut self) -> Result<(), InstanceLifecycleError> {
        if !self.can_transition_to_unloading() {
            return Err(InstanceLifecycleError::CannotUnloadYet {
                state: self.state,
                in_flight: self.in_flight.get(),
                deadline_expired: self.is_drain_deadline_expired(),
            });
        }
        self.transition_to(InstanceLifecycleState::Unloading)?;
        Ok(())
    }

    /// Transition to a new state. Validates the transition and records history.
    pub(crate) fn transition_to(
        &mut self,
        next: InstanceLifecycleState,
    ) -> Result<(), InstanceLifecycleError> {
        match Self::validate_transition(self.state, next) {
            TransitionResult::Ok(prev) => {
                let entry = TransitionEntry {
                    from: prev,
                    to: next,
                    at: Instant::now(),
                };
                self.history.push_back(entry);
                while self.history.len() > self.max_history {
                    self.history.pop_front();
                }
                self.state = next;
                Ok(())
            }
            TransitionResult::Invalid { from, to } => {
                Err(InstanceLifecycleError::InvalidTransition { from, to })
            }
        }
    }

    /// Validate a transition without applying it.
    pub(crate) fn validate_transition(
        current: InstanceLifecycleState,
        next: InstanceLifecycleState,
    ) -> TransitionResult {
        let table = valid_transitions();
        for &(state, allowed) in table {
            if state == current {
                if allowed.contains(&next) {
                    return TransitionResult::Ok(current);
                } else {
                    return TransitionResult::Invalid {
                        from: current,
                        to: next,
                    };
                }
            }
        }
        // Terminal states have no valid transitions.
        TransitionResult::Invalid {
            from: current,
            to: next,
        }
    }

    /// Set the error message for a failed instance.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "lifecycle error storage is covered by tests")
    )]
    pub(crate) fn set_error(&mut self, error: String) {
        self.last_error = Some(error);
    }

    /// Get the last error message, if any.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "lifecycle error storage is covered by tests")
    )]
    pub(crate) fn last_error(&self) -> Option<&str> {
        self.last_error.as_deref()
    }

    /// Recent transition history (most recent first).
    #[expect(dead_code, reason = "reserved bounded history status surface")]
    pub(crate) fn recent_history(&self, limit: usize) -> Vec<TransitionEntry> {
        let mut entries: Vec<_> = self.history.iter().cloned().rev().take(limit).collect();
        entries.reverse();
        entries
    }

    /// Full transition history as a vector clone (VecDeque doesn't coerce to slice refs stably).
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "full history access is covered by tests")
    )]
    pub(crate) fn full_history(&self) -> Vec<TransitionEntry> {
        self.history.iter().cloned().collect()
    }

    /// Time since the instance was created.
    #[expect(dead_code, reason = "reserved instance age status surface")]
    pub(crate) fn age(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Whether this instance is in a terminal state.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "terminal-state classification is covered by tests"
        )
    )]
    pub(crate) fn is_terminal(&self) -> bool {
        self.state.is_terminal()
    }

    /// Reset drain state (for testing).
    #[cfg(test)]
    #[expect(dead_code, reason = "retained for focused drain state tests")]
    fn reset_drain(&mut self) {
        self.draining_initiated = false;
        self.drain_deadline = None;
    }

    /// Set in-flight count directly (for testing).
    #[cfg(test)]
    fn set_in_flight(&self, value: u64) {
        self.in_flight.set(value);
    }
}

/// RAII token for a request admitted to one concrete runtime instance.
pub(crate) struct InstanceRequestGuard {
    tracker: std::sync::Arc<InFlightTracker>,
}

impl InstanceRequestGuard {
    pub(crate) fn new(tracker: std::sync::Arc<InFlightTracker>) -> Self {
        tracker.increment();
        Self { tracker }
    }
}

impl Drop for InstanceRequestGuard {
    fn drop(&mut self) {
        self.tracker.decrement();
    }
}

// ── Errors ────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub(crate) enum InstanceLifecycleError {
    #[error("invalid transition from {from:?} to {to:?}")]
    InvalidTransition {
        from: InstanceLifecycleState,
        to: InstanceLifecycleState,
    },

    #[error("cannot drain instance in state {state:?}; must be Serving")]
    NotDrainable { state: InstanceLifecycleState },

    #[error(
        "cannot unload yet: state={state:?}, in_flight={in_flight}, deadline_expired={deadline_expired}"
    )]
    CannotUnloadYet {
        state: InstanceLifecycleState,
        in_flight: u64,
        deadline_expired: bool,
    },

    #[error("instance already terminal (state={state:?})")]
    #[expect(dead_code, reason = "reserved explicit terminal-state diagnostic")]
    AlreadyTerminal { state: InstanceLifecycleState },

    #[error("model '{0}' has multiple loaded instances; specify instance ID to drain a single one")]
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "model-target ambiguity is covered by lifecycle tests"
        )
    )]
    AmbiguousModelTarget(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub(crate) enum InstanceAdmissionError {
    #[error("runtime instance is not accepting work while {state:?}")]
    NotAccepting { state: InstanceLifecycleState },
}

// ── Drain Coordinator ────────────────────────────────────────────────────────

/// Coordinates the drain process for a single instance.
///
/// This is a helper that encapsulates the drain loop logic: wait for zero
/// in-flight or deadline expiry, then transition to unloading.
pub(crate) struct DrainCoordinator {
    /// How often to poll the in-flight counter during drain.
    pub(crate) poll_interval: Duration,
}

impl Default for DrainCoordinator {
    fn default() -> Self {
        Self {
            poll_interval: Duration::from_millis(100),
        }
    }
}

impl DrainCoordinator {
    /// Wait for the instance to become ready for unloading.
    ///
    /// Returns `DrainResult::Graceful` if in-flight reached zero before deadline,
    /// or `DrainResult::ForceCancelled` if the deadline expired with remaining work.
    pub(crate) async fn wait_for_unload_ready(
        &self,
        record: &std::sync::Arc<tokio::sync::Mutex<InstanceLifecycleRecord>>,
    ) -> DrainResult {
        loop {
            {
                let record = record.lock().await;
                if record.state() != InstanceLifecycleState::Draining {
                    return DrainResult::ForceCancelled;
                }
                if record.in_flight_count() == 0 {
                    return DrainResult::Graceful;
                }
                if record.is_drain_deadline_expired() {
                    return DrainResult::ForceCancelled;
                }
            }
            tokio::time::sleep(self.poll_interval).await;
        }
    }

    /// Wait for unload readiness with a fake clock (for testing).
    #[cfg(test)]
    pub(crate) fn wait_for_unload_ready_fake_clock(
        &self,
        record: &InstanceLifecycleRecord,
        _fake_now: Instant,
    ) -> DrainResult {
        if record.state() != InstanceLifecycleState::Draining {
            return DrainResult::ForceCancelled;
        }
        if record.in_flight_count() == 0 {
            return DrainResult::Graceful;
        }
        // In tests, we check the deadline against the fake clock by directly
        // checking the drain_deadline field.
        match record.drain_deadline {
            Some(deadline) => {
                if _fake_now >= deadline && record.in_flight_count() > 0 {
                    return DrainResult::ForceCancelled;
                }
                DrainResult::Waiting
            }
            None => DrainResult::Waiting,
        }
    }
}

/// Result of the drain wait process.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DrainResult {
    /// In-flight reached zero before deadline; unload gracefully.
    Graceful,
    /// Deadline expired with remaining in-flight; force-cancel and unload.
    ForceCancelled,
    /// Still waiting (used by fake-clock tests).
    #[cfg(test)]
    Waiting,
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tokio::sync::Mutex;

    // ── State Machine Transitions ────────────────────────────────────────────

    #[test]
    fn valid_forward_transitions_through_full_lifecycle() {
        let mut record = InstanceLifecycleRecord::new(InstanceLifecycleState::Planned, 20);

        // Planned → Resolving → Loading → Warming → Serving
        assert!(
            record
                .transition_to(InstanceLifecycleState::Resolving)
                .is_ok()
        );
        assert_eq!(record.state(), InstanceLifecycleState::Resolving);

        assert!(
            record
                .transition_to(InstanceLifecycleState::Loading)
                .is_ok()
        );
        assert_eq!(record.state(), InstanceLifecycleState::Loading);

        assert!(
            record
                .transition_to(InstanceLifecycleState::Warming)
                .is_ok()
        );
        assert_eq!(record.state(), InstanceLifecycleState::Warming);

        assert!(
            record
                .transition_to(InstanceLifecycleState::Serving)
                .is_ok()
        );
        assert_eq!(record.state(), InstanceLifecycleState::Serving);
    }

    #[test]
    fn valid_drain_and_unload_path() {
        let mut record = InstanceLifecycleRecord::new(InstanceLifecycleState::Serving, 20);

        // Serving → Draining (via mark_draining)
        assert!(
            record
                .mark_draining(Instant::now() + Duration::from_secs(30))
                .is_ok()
        );
        assert_eq!(record.state(), InstanceLifecycleState::Draining);

        // Draining → Unloading (when in-flight is zero)
        assert!(record.transition_to_unloading().is_ok());
        assert_eq!(record.state(), InstanceLifecycleState::Unloading);

        // Unloading → Stopped
        assert!(
            record
                .transition_to(InstanceLifecycleState::Stopped)
                .is_ok()
        );
        assert_eq!(record.state(), InstanceLifecycleState::Stopped);
    }

    #[test]
    fn failure_transition_from_any_non_terminal_state() {
        let states = [
            InstanceLifecycleState::Planned,
            InstanceLifecycleState::Resolving,
            InstanceLifecycleState::Loading,
            InstanceLifecycleState::Warming,
            InstanceLifecycleState::Serving,
            InstanceLifecycleState::Draining,
            InstanceLifecycleState::Unloading,
        ];

        for state in states {
            let mut record = InstanceLifecycleRecord::new(state, 20);
            assert!(
                record.transition_to(InstanceLifecycleState::Failed).is_ok(),
                "should be able to transition from {:?} to Failed",
                state
            );
        }
    }

    #[test]
    fn invalid_transitions_are_rejected() {
        // Cannot go backwards: Serving → Loading
        let mut record = InstanceLifecycleRecord::new(InstanceLifecycleState::Serving, 20);
        let err = record
            .transition_to(InstanceLifecycleState::Loading)
            .unwrap_err();
        assert!(matches!(
            err,
            InstanceLifecycleError::InvalidTransition { .. }
        ));

        // Cannot skip states: Planned → Serving
        let mut record = InstanceLifecycleRecord::new(InstanceLifecycleState::Planned, 20);
        let err = record
            .transition_to(InstanceLifecycleState::Serving)
            .unwrap_err();
        assert!(matches!(
            err,
            InstanceLifecycleError::InvalidTransition { .. }
        ));

        // Cannot go from Draining to Serving
        let mut record = InstanceLifecycleRecord::new(InstanceLifecycleState::Draining, 20);
        let err = record
            .transition_to(InstanceLifecycleState::Serving)
            .unwrap_err();
        assert!(matches!(
            err,
            InstanceLifecycleError::InvalidTransition { .. }
        ));

        // Cannot transition from terminal states
        let mut record = InstanceLifecycleRecord::new(InstanceLifecycleState::Stopped, 20);
        for target in [
            InstanceLifecycleState::Planned,
            InstanceLifecycleState::Serving,
            InstanceLifecycleState::Failed,
        ] {
            let err = record.transition_to(target).unwrap_err();
            assert!(matches!(
                err,
                InstanceLifecycleError::InvalidTransition { .. }
            ));
        }
    }

    // ── Admission Guard ──────────────────────────────────────────────────────

    #[test]
    fn only_serving_accepts_work() {
        let states = [
            (InstanceLifecycleState::Planned, false),
            (InstanceLifecycleState::Resolving, false),
            (InstanceLifecycleState::Loading, false),
            (InstanceLifecycleState::Warming, false),
            (InstanceLifecycleState::Serving, true),
            (InstanceLifecycleState::Draining, false),
            (InstanceLifecycleState::Unloading, false),
            (InstanceLifecycleState::Failed, false),
            (InstanceLifecycleState::Stopped, false),
        ];

        for (state, expected) in states {
            let record = InstanceLifecycleRecord::new(state, 20);
            assert_eq!(
                record.is_accepting_work(),
                expected,
                "state {:?} should accept_work={}",
                state,
                expected
            );
        }
    }

    // ── In-Flight Tracking ───────────────────────────────────────────────────

    #[test]
    fn in_flight_counter_tracks_requests() {
        let record = InstanceLifecycleRecord::new(InstanceLifecycleState::Serving, 20);

        assert_eq!(record.in_flight_count(), 0);

        record.admit_request();
        assert_eq!(record.in_flight_count(), 1);

        record.admit_request();
        assert_eq!(record.in_flight_count(), 2);

        let is_zero = record.complete_request();
        // Not draining, so even though count goes to 1, the "unload-ready" flag is false.
        assert!(!is_zero, "should not signal unload-ready while Serving");
        assert_eq!(record.in_flight_count(), 1);

        let is_zero = record.complete_request();
        // Count reaches zero but state is Serving (not Draining), so no unload signal.
        assert!(
            !is_zero,
            "Serving state should not trigger unload-ready even at zero in-flight"
        );
        assert_eq!(record.in_flight_count(), 0);
    }

    #[test]
    fn complete_request_returns_true_on_zero_while_draining() {
        let mut record = InstanceLifecycleRecord::new(InstanceLifecycleState::Serving, 20);
        assert!(
            record
                .mark_draining(Instant::now() + Duration::from_secs(30))
                .is_ok()
        );

        // Simulate one in-flight request
        record.admit_request();
        assert_eq!(record.in_flight_count(), 1);

        let is_zero = record.complete_request();
        // This should return true because we're draining and count went to zero.
        assert!(
            is_zero,
            "should signal unload-ready when last request completes during drain"
        );
    }

    #[test]
    fn in_flight_saturates_at_zero() {
        let record = InstanceLifecycleRecord::new(InstanceLifecycleState::Serving, 20);
        // Decrement from zero should saturate.
        record.complete_request();
        assert_eq!(record.in_flight_count(), 0);
    }

    // ── Drain Logic ──────────────────────────────────────────────────────────

    #[test]
    fn drain_rejects_non_serving_state() {
        let mut record = InstanceLifecycleRecord::new(InstanceLifecycleState::Loading, 20);
        let err = record.mark_draining(Instant::now()).unwrap_err();
        assert!(matches!(err, InstanceLifecycleError::NotDrainable { .. }));
    }

    #[test]
    fn drain_is_idempotent() {
        let mut record = InstanceLifecycleRecord::new(InstanceLifecycleState::Serving, 20);
        assert!(
            record
                .mark_draining(Instant::now() + Duration::from_secs(30))
                .is_ok()
        );
        // Second call should succeed (idempotent), not error.
        assert!(
            record
                .mark_draining(Instant::now() + Duration::from_secs(60))
                .is_ok()
        );
    }

    #[test]
    fn drain_idempotency_tightens_to_shorter_deadline() {
        let mut record = InstanceLifecycleRecord::new(InstanceLifecycleState::Serving, 20);
        let first_deadline = Instant::now() + Duration::from_secs(300);
        record.mark_draining(first_deadline).unwrap();

        record
            .mark_draining(first_deadline + Duration::from_secs(60))
            .unwrap();
        assert_eq!(record.drain_deadline, Some(first_deadline));

        let shorter_deadline = Instant::now() + Duration::from_secs(5);
        record.mark_draining(shorter_deadline).unwrap();
        assert_eq!(record.drain_deadline, Some(shorter_deadline));

        record.mark_draining_force().unwrap();
        let force_deadline = record.drain_deadline.unwrap();
        assert!(force_deadline <= Instant::now());
    }

    #[test]
    fn drain_deadline_expired_returns_true_correctly() {
        let mut record = InstanceLifecycleRecord::new(InstanceLifecycleState::Serving, 20);
        assert!(
            record
                .mark_draining(Instant::now() + Duration::from_millis(1))
                .is_ok()
        );

        // Immediately: not expired yet (deadline is in the future)
        // But we set it to 1ms in the future, so let's check after a tiny delay.
        std::thread::sleep(Duration::from_millis(5));

        // With zero in-flight, deadline expiry doesn't matter for force-cancel.
        assert!(!record.is_drain_deadline_expired());

        // Simulate in-flight > 0
        record.set_in_flight(1);
        assert!(record.is_drain_deadline_expired());
    }

    #[test]
    fn drain_force_sets_deadline_to_now() {
        let mut record = InstanceLifecycleRecord::new(InstanceLifecycleState::Serving, 20);
        assert!(record.mark_draining_force().is_ok());
        assert_eq!(record.state(), InstanceLifecycleState::Draining);

        // With in-flight > 0 and deadline at now, should be expired.
        record.set_in_flight(1);
        assert!(record.is_drain_deadline_expired());
    }

    #[test]
    fn can_transition_to_unloading_when_zero_inflight() {
        let mut record = InstanceLifecycleRecord::new(InstanceLifecycleState::Serving, 20);
        assert!(
            record
                .mark_draining(Instant::now() + Duration::from_secs(30))
                .is_ok()
        );

        // Zero in-flight → can unload immediately.
        assert!(record.can_transition_to_unloading());
    }

    #[test]
    fn can_not_transition_to_unloading_with_inflight_and_no_deadline_expiry() {
        let mut record = InstanceLifecycleRecord::new(InstanceLifecycleState::Serving, 20);
        assert!(
            record
                .mark_draining(Instant::now() + Duration::from_secs(300))
                .is_ok()
        );

        // In-flight > 0 and deadline far in future → cannot unload yet.
        record.set_in_flight(5);
        assert!(!record.can_transition_to_unloading());
    }

    #[test]
    fn can_transition_to_unloading_when_deadline_expired() {
        let mut record = InstanceLifecycleRecord::new(InstanceLifecycleState::Serving, 20);
        // Set deadline to the past.
        assert!(
            record
                .mark_draining(Instant::now() - Duration::from_secs(1))
                .is_ok()
        );

        record.set_in_flight(3);
        assert!(record.can_transition_to_unloading());
    }

    #[test]
    fn transition_to_unloading_succeeds_when_ready() {
        let mut record = InstanceLifecycleRecord::new(InstanceLifecycleState::Serving, 20);
        assert!(
            record
                .mark_draining(Instant::now() + Duration::from_secs(30))
                .is_ok()
        );

        assert!(record.transition_to_unloading().is_ok());
        assert_eq!(record.state(), InstanceLifecycleState::Unloading);
    }

    #[test]
    fn transition_to_unloading_fails_when_not_ready() {
        let mut record = InstanceLifecycleRecord::new(InstanceLifecycleState::Serving, 20);
        assert!(
            record
                .mark_draining(Instant::now() + Duration::from_secs(300))
                .is_ok()
        );

        record.set_in_flight(5);
        let err = record.transition_to_unloading().unwrap_err();
        assert!(matches!(
            err,
            InstanceLifecycleError::CannotUnloadYet { .. }
        ));
    }

    // ── History ──────────────────────────────────────────────────────────────

    #[test]
    fn history_records_transitions() {
        let mut record = InstanceLifecycleRecord::new(InstanceLifecycleState::Planned, 20);
        record
            .transition_to(InstanceLifecycleState::Resolving)
            .unwrap();
        record
            .transition_to(InstanceLifecycleState::Loading)
            .unwrap();

        let history = record.full_history();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].from, InstanceLifecycleState::Planned);
        assert_eq!(history[0].to, InstanceLifecycleState::Resolving);
        assert_eq!(history[1].from, InstanceLifecycleState::Resolving);
        assert_eq!(history[1].to, InstanceLifecycleState::Loading);
    }

    #[test]
    fn history_is_bounded() {
        let mut record = InstanceLifecycleRecord::new(InstanceLifecycleState::Planned, 3);
        for state in [
            InstanceLifecycleState::Resolving,
            InstanceLifecycleState::Loading,
            InstanceLifecycleState::Warming,
            InstanceLifecycleState::Serving,
        ] {
            record.transition_to(state).unwrap();
        }
        record
            .mark_draining(Instant::now() + Duration::from_secs(1))
            .unwrap();
        record.transition_to_unloading().unwrap();
        record
            .transition_to(InstanceLifecycleState::Stopped)
            .unwrap();

        let history = record.full_history();
        assert_eq!(history.len(), 3);
        assert_eq!(history[0].from, InstanceLifecycleState::Serving);
        assert_eq!(history[0].to, InstanceLifecycleState::Draining);
        assert_eq!(history[2].from, InstanceLifecycleState::Unloading);
        assert_eq!(history[2].to, InstanceLifecycleState::Stopped);
    }

    // ── Terminal States ──────────────────────────────────────────────────────

    #[test]
    fn terminal_states_are_correctly_identified() {
        let stopped = InstanceLifecycleRecord::new(InstanceLifecycleState::Stopped, 20);
        assert!(stopped.is_terminal());

        let failed = InstanceLifecycleRecord::new(InstanceLifecycleState::Failed, 20);
        assert!(failed.is_terminal());

        let serving = InstanceLifecycleRecord::new(InstanceLifecycleState::Serving, 20);
        assert!(!serving.is_terminal());
    }

    // ── Error Messages ───────────────────────────────────────────────────────

    #[test]
    fn error_message_is_set_and_retrieved() {
        let mut record = InstanceLifecycleRecord::new(InstanceLifecycleState::Serving, 20);
        assert!(record.last_error().is_none());

        record.set_error("OOM during loading".to_string());
        assert_eq!(record.last_error(), Some("OOM during loading"));
    }

    // ── State Labels ─────────────────────────────────────────────────────────

    #[test]
    fn state_as_str_labels() {
        assert_eq!(InstanceLifecycleState::Planned.as_str(), "planned");
        assert_eq!(InstanceLifecycleState::Resolving.as_str(), "resolving");
        assert_eq!(InstanceLifecycleState::Loading.as_str(), "loading");
        assert_eq!(InstanceLifecycleState::Warming.as_str(), "warming");
        assert_eq!(InstanceLifecycleState::Serving.as_str(), "serving");
        assert_eq!(InstanceLifecycleState::Draining.as_str(), "draining");
        assert_eq!(InstanceLifecycleState::Unloading.as_str(), "unloading");
        assert_eq!(InstanceLifecycleState::Failed.as_str(), "failed");
        assert_eq!(InstanceLifecycleState::Stopped.as_str(), "stopped");
    }

    // ── Ambiguous Model Target Error ─────────────────────────────────────────

    #[test]
    fn ambiguous_model_target_error_displays_correctly() {
        let err = InstanceLifecycleError::AmbiguousModelTarget("Qwen3-8B".to_string());
        let msg = err.to_string();
        assert!(msg.contains("Qwen3-8B"));
        assert!(msg.contains("multiple loaded instances"));
    }

    // ── Drain Coordinator (fake clock tests) ────────────────────────────────

    #[test]
    fn drain_coordinator_graceful_with_zero_inflight() {
        let mut record = InstanceLifecycleRecord::new(InstanceLifecycleState::Serving, 20);
        record
            .mark_draining(Instant::now() + Duration::from_secs(1))
            .unwrap();
        let coord = DrainCoordinator::default();
        let result = coord.wait_for_unload_ready_fake_clock(&record, Instant::now());
        assert_eq!(result, DrainResult::Graceful);
    }

    #[test]
    fn drain_coordinator_force_cancel_when_deadline_expired() {
        let mut record = InstanceLifecycleRecord::new(InstanceLifecycleState::Serving, 20);
        assert!(
            record
                .mark_draining(Instant::now() - Duration::from_secs(1))
                .is_ok()
        );
        record.set_in_flight(3);

        let coord = DrainCoordinator::default();
        let result = coord.wait_for_unload_ready_fake_clock(&record, Instant::now());
        assert_eq!(result, DrainResult::ForceCancelled);
    }

    #[test]
    fn drain_coordinator_waiting_when_inflight_and_no_expiry() {
        let mut record = InstanceLifecycleRecord::new(InstanceLifecycleState::Serving, 20);
        assert!(
            record
                .mark_draining(Instant::now() + Duration::from_secs(300))
                .is_ok()
        );
        record.set_in_flight(5);

        let coord = DrainCoordinator::default();
        let result = coord.wait_for_unload_ready_fake_clock(&record, Instant::now());
        assert_eq!(result, DrainResult::Waiting);
    }

    #[tokio::test]
    async fn instance_drain_completes_when_inflight_reaches_zero() {
        let record = Arc::new(Mutex::new(InstanceLifecycleRecord::new(
            InstanceLifecycleState::Serving,
            20,
        )));
        let request = {
            let mut locked = record.lock().await;
            let request = InstanceRequestGuard::new(locked.in_flight_tracker());
            locked
                .mark_draining(Instant::now() + Duration::from_secs(1))
                .unwrap();
            request
        };

        {
            let locked = record.lock().await;
            assert!(!locked.is_accepting_work());
            assert_eq!(locked.in_flight_count(), 1);
        }

        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(10)).await;
            drop(request);
        });
        let result = DrainCoordinator {
            poll_interval: Duration::from_millis(1),
        }
        .wait_for_unload_ready(&record)
        .await;

        assert_eq!(result, DrainResult::Graceful);
        let mut locked = record.lock().await;
        assert_eq!(locked.in_flight_count(), 0);
        locked.transition_to_unloading().unwrap();
        assert_eq!(locked.state(), InstanceLifecycleState::Unloading);
    }

    #[tokio::test]
    async fn drain_wait_force_cancels_after_failed_transition() {
        let record = Arc::new(Mutex::new(InstanceLifecycleRecord::new(
            InstanceLifecycleState::Serving,
            20,
        )));
        {
            let mut locked = record.lock().await;
            locked
                .mark_draining(Instant::now() + Duration::from_secs(300))
                .unwrap();
            locked
                .transition_to(InstanceLifecycleState::Failed)
                .unwrap();
        }

        let result = DrainCoordinator::default()
            .wait_for_unload_ready(&record)
            .await;

        assert_eq!(result, DrainResult::ForceCancelled);
    }

    #[tokio::test(start_paused = true)]
    async fn drain_wait_keeps_lifecycle_mutex_acquirable_while_pending() {
        let record = Arc::new(Mutex::new(InstanceLifecycleRecord::new(
            InstanceLifecycleState::Serving,
            20,
        )));
        let request = {
            let mut locked = record.lock().await;
            let request = InstanceRequestGuard::new(locked.in_flight_tracker());
            locked
                .mark_draining(Instant::now() + Duration::from_secs(300))
                .unwrap();
            request
        };

        let waiter_record = Arc::clone(&record);
        let waiter = tokio::spawn(async move {
            DrainCoordinator {
                poll_interval: Duration::from_secs(60),
            }
            .wait_for_unload_ready(&waiter_record)
            .await
        });
        tokio::task::yield_now().await;

        let locked = tokio::time::timeout(Duration::from_millis(1), record.lock())
            .await
            .expect("drain waiter must not hold the lifecycle mutex while pending");
        assert_eq!(locked.state(), InstanceLifecycleState::Draining);
        assert!(!locked.is_accepting_work());
        drop(locked);
        assert!(!waiter.is_finished());

        drop(request);
        tokio::time::advance(Duration::from_secs(60)).await;
        assert_eq!(waiter.await.unwrap(), DrainResult::Graceful);
    }

    #[test]
    fn instance_drain_deadline_cancels_only_target_instance() {
        let mut target = InstanceLifecycleRecord::new(InstanceLifecycleState::Serving, 20);
        let sibling = InstanceLifecycleRecord::new(InstanceLifecycleState::Serving, 20);
        let target_request = InstanceRequestGuard::new(target.in_flight_tracker());
        let sibling_request = InstanceRequestGuard::new(sibling.in_flight_tracker());

        target
            .mark_draining(Instant::now() - Duration::from_millis(1))
            .unwrap();
        let result =
            DrainCoordinator::default().wait_for_unload_ready_fake_clock(&target, Instant::now());

        assert_eq!(result, DrainResult::ForceCancelled);
        target.transition_to_unloading().unwrap();
        assert_eq!(target.state(), InstanceLifecycleState::Unloading);
        assert_eq!(target.in_flight_count(), 1);
        assert_eq!(sibling.state(), InstanceLifecycleState::Serving);
        assert!(sibling.is_accepting_work());
        assert_eq!(sibling.in_flight_count(), 1);

        drop(target_request);
        drop(sibling_request);
    }
}
