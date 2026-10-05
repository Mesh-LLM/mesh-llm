//! Lifecycle guard owning exactly ONE terminal transition per request.
//!
//! A `LifecycleGuard` starts in an active state and allows transitions to any terminal state
//! (Completed/Failed/Rejected/Cancelled/Dropped) exactly once. After the first terminal transition,
//! all further terminal attempts are rejected with [`DuplicateTerminalError`] — making the guard idempotent.
//! Per-attempt events (e.g., retry `AttemptStarted`) do NOT terminate the parent request; they are
//! always allowed regardless of current state and produce a separate attempt-scoped log entry.

use std::fmt;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex, Weak};

use mesh_llm_events::logging::events::TokenUsage;
use mesh_llm_events::logging::identifiers::RequestId;

/// Terminal outcomes for a lifecycle guard transition.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TerminalOutcome {
    /// Request completed successfully.
    Completed,
    /// Request completed successfully with its actual HTTP terminal status.
    CompletedWithStatus(u16),
    /// Request completed with its actual status and authoritative usage.
    CompletedWithUsage { status_code: u16, usage: TokenUsage },
    /// Request failed with an error condition.
    Failed(String),
    /// Request failed after an HTTP response was received.
    FailedWithStatus { error: String, status_code: u16 },
    /// Request rejected before processing (e.g., invalid input).
    Rejected(Option<String>),
    /// Request was rejected with its actual HTTP terminal status.
    RejectedWithStatus {
        reason: Option<String>,
        status_code: u16,
    },
    /// Request cancelled by caller or system.
    Cancelled(Option<String>),
    /// Request dropped without terminal processing (queue overflow, timeout, etc.).
    Dropped(Option<String>),
}

impl TerminalOutcome {
    #[allow(dead_code)]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Completed | Self::CompletedWithStatus(_) | Self::CompletedWithUsage { .. } => {
                "completed"
            }
            Self::Failed(_) | Self::FailedWithStatus { .. } => "failed",
            Self::Rejected(_) | Self::RejectedWithStatus { .. } => "rejected",
            Self::Cancelled(_) => "cancelled",
            Self::Dropped(_) => "dropped",
        }
    }

    /// Returns the error/reason string if this is a non-success terminal outcome.
    #[allow(dead_code)]
    pub fn reason(&self) -> Option<&str> {
        match self {
            Self::Completed | Self::CompletedWithStatus(_) | Self::CompletedWithUsage { .. } => {
                None
            }
            Self::Failed(e) => Some(e),
            Self::FailedWithStatus { error, .. } => Some(error),
            Self::Rejected(r) => r.as_deref(),
            Self::RejectedWithStatus { reason, .. } => reason.as_deref(),
            Self::Cancelled(r) => r.as_deref(),
            Self::Dropped(r) => r.as_deref(),
        }
    }

    /// Is this a success outcome?
    #[allow(dead_code)]
    pub fn is_success(&self) -> bool {
        matches!(
            self,
            Self::Completed | Self::CompletedWithStatus(_) | Self::CompletedWithUsage { .. }
        )
    }

    pub(crate) const fn status_code(&self) -> Option<u16> {
        match self {
            Self::CompletedWithStatus(status) => Some(*status),
            Self::CompletedWithUsage { status_code, .. } => Some(*status_code),
            Self::FailedWithStatus { status_code, .. }
            | Self::RejectedWithStatus { status_code, .. } => Some(*status_code),
            _ => None,
        }
    }
}

impl fmt::Display for TerminalOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Completed | Self::CompletedWithStatus(_) | Self::CompletedWithUsage { .. } => {
                write!(f, "completed")
            }
            Self::Failed(e) => write!(f, "failed: {}", e),
            Self::FailedWithStatus { error, .. } => write!(f, "failed: {}", error),
            Self::Rejected(r) => write!(f, "rejected: {}", r.as_deref().unwrap_or("unknown")),
            Self::RejectedWithStatus { reason, .. } => {
                write!(f, "rejected: {}", reason.as_deref().unwrap_or("unknown"))
            }
            Self::Cancelled(r) => write!(f, "cancelled: {}", r.as_deref().unwrap_or("unknown")),
            Self::Dropped(r) => write!(f, "dropped: {}", r.as_deref().unwrap_or("unknown")),
        }
    }
}

/// Error returned when a duplicate terminal transition is attempted.
#[derive(Clone, Debug)]
pub struct DuplicateTerminalError {
    pub existing: Box<TerminalOutcome>,
    pub attempted: Box<TerminalOutcome>,
}

impl fmt::Display for DuplicateTerminalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "duplicate terminal transition: {} → {}",
            self.existing.as_str(),
            self.attempted.as_str()
        )
    }
}

impl std::error::Error for DuplicateTerminalError {}

/// Internal state encoding as a single byte.
const STATE_ACTIVE: u8 = 0;
// Terminal states are encoded by outcome variant index + data pointer stored separately via Arc<Mutex<...>>.

/// Service callback invoked after this guard wins its one terminal transition.
///
/// The callback is deliberately weakly held by the guard. A request handle must
/// not keep a logging service alive after the runtime has shut down.
pub(crate) trait LifecycleRecorder: Send + Sync {
    /// Record the terminal event after the guard has atomically claimed it.
    fn record_terminal(&self, request_id: RequestId, outcome: TerminalOutcome);
}

/// Shared terminal state storage (only written once, read many).
#[derive(Debug, Default)]
struct TerminalRecord {
    outcome: Option<TerminalOutcome>,
}

struct LifecycleInner {
    state_flag: Arc<AtomicU8>,
    record: Mutex<TerminalRecord>,
    request_id: Option<RequestId>,
    recorder: Option<Weak<dyn LifecycleRecorder>>,
}

/// Lifecycle guard with exactly-one-terminal-transition invariant.
///
/// The guard is `Clone` — all clones share the same underlying terminal state via `Arc`. This means
/// cloning a guard does not create independent lifecycle tracks; they all observe and enforce the same
/// single-terminal constraint. Per-attempt events are recorded separately (via the service) without
/// affecting this parent request's terminal status.
#[derive(Clone)]
pub struct LifecycleGuard {
    inner: Arc<LifecycleInner>,
}

impl fmt::Debug for LifecycleGuard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LifecycleGuard")
            .field("active", &self.is_active())
            .field("request_id", &self.inner.request_id)
            .finish_non_exhaustive()
    }
}

impl LifecycleGuard {
    /// Create a new guard starting in Active state.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(LifecycleInner {
                state_flag: Arc::new(AtomicU8::new(STATE_ACTIVE)),
                record: Mutex::new(TerminalRecord::default()),
                request_id: None,
                recorder: None,
            }),
        }
    }

    /// Create a request-owned guard whose terminal records are delivered by the
    /// logging service. This is kept crate-private so every callback remains
    /// bound to a service-owned request registration.
    pub(crate) fn for_request(
        request_id: RequestId,
        recorder: Weak<dyn LifecycleRecorder>,
    ) -> Self {
        Self {
            inner: Arc::new(LifecycleInner {
                state_flag: Arc::new(AtomicU8::new(STATE_ACTIVE)),
                record: Mutex::new(TerminalRecord::default()),
                request_id: Some(request_id),
                recorder: Some(recorder),
            }),
        }
    }

    /// Check if this guard is still in an active (non-terminal) state.
    #[allow(dead_code)]
    pub fn is_active(&self) -> bool {
        self.inner.state_flag.load(Ordering::Acquire) == STATE_ACTIVE
    }

    /// Transition to a terminal outcome. Returns `Err(DuplicateTerminalError)` if already terminated.
    /// This transition is exactly-once: the first call succeeds, all subsequent calls are rejected.
    pub fn terminate(&self, outcome: TerminalOutcome) -> Result<(), DuplicateTerminalError> {
        // Record the outcome before publishing the terminal flag. The mutex is
        // the transition arbiter, so a losing caller always observes the real
        // winner rather than fabricating a placeholder outcome.
        let mut record = lock_recover(&self.inner.record);
        if let Some(existing) = &record.outcome {
            return Err(DuplicateTerminalError {
                existing: Box::new(existing.clone()),
                attempted: Box::new(outcome),
            });
        }

        record.outcome = Some(outcome.clone());
        self.inner.state_flag.store(1, Ordering::Release);
        drop(record);
        self.record_terminal(outcome);
        Ok(())
    }

    /// Attempt to transition idempotently — if already at the same terminal state, return Ok; otherwise attempt.
    pub fn terminate_idempotent(
        &self,
        outcome: TerminalOutcome,
    ) -> Result<(), DuplicateTerminalError> {
        match self.terminate(outcome) {
            Ok(()) => Ok(()),
            Err(DuplicateTerminalError {
                existing,
                attempted,
            }) if existing == attempted => Ok(()),
            Err(e) => Err(e),
        }
    }

    /// Get the current terminal outcome (if any). Returns None while active.
    #[allow(dead_code)]
    pub fn terminal_outcome(&self) -> Option<TerminalOutcome> {
        let record = lock_recover(&self.inner.record);
        record.outcome.clone()
    }

    /// Allocate a typed transport-attempt identifier. Attempts are deliberately
    /// independent from the request terminal state, so retries can be recorded
    /// before or after a parent terminal transition without mutating it.
    pub fn record_attempt(&self) -> mesh_llm_events::logging::identifiers::AttemptId {
        mesh_llm_events::logging::identifiers::AttemptId::new()
    }

    /// Clone the internal state flag for external observation (drop counters etc.).
    #[allow(dead_code)]
    pub fn state_flag_clone(&self) -> Arc<AtomicU8> {
        Arc::clone(&self.inner.state_flag)
    }

    /// Whether two guards share the same terminal track.
    #[allow(dead_code)]
    pub fn shares_track_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    fn record_terminal(&self, outcome: TerminalOutcome) {
        let Some(request_id) = self.inner.request_id else {
            return;
        };
        let Some(recorder) = self.inner.recorder.as_ref().and_then(Weak::upgrade) else {
            return;
        };

        // A terminal event must never turn a logging problem into a request-path
        // panic. The service recorder is also fail-open, but Drop requires this
        // boundary to be panic-safe even if a future recorder regresses.
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            recorder.record_terminal(request_id, outcome);
        }));
    }
}

impl Drop for LifecycleGuard {
    fn drop(&mut self) {
        if Arc::strong_count(&self.inner) != 1 || !self.is_active() {
            return;
        }

        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = self.terminate(TerminalOutcome::Dropped(Some(
                "request lifecycle guard dropped before terminal outcome".into(),
            )));
        }));
    }
}

fn lock_recover<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

impl Default for LifecycleGuard {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_guard_is_active() {
        let guard = LifecycleGuard::new();
        assert!(guard.is_active());
        assert_eq!(guard.terminal_outcome(), None);
    }

    #[test]
    fn terminate_completed_succeeds() {
        let guard = LifecycleGuard::new();
        assert!(guard.terminate(TerminalOutcome::Completed).is_ok());
        assert!(!guard.is_active());
        assert_eq!(guard.terminal_outcome(), Some(TerminalOutcome::Completed));
    }

    #[test]
    fn terminate_failed_succeeds() {
        let guard = LifecycleGuard::new();
        assert!(
            guard
                .terminate(TerminalOutcome::Failed("timeout".into()))
                .is_ok()
        );
        assert_eq!(
            guard.terminal_outcome(),
            Some(TerminalOutcome::Failed("timeout".into()))
        );
    }

    #[test]
    fn terminate_rejected_succeeds() {
        let guard = LifecycleGuard::new();
        assert!(
            guard
                .terminate(TerminalOutcome::Rejected(Some("invalid model".into())))
                .is_ok()
        );
    }

    #[test]
    fn terminate_cancelled_succeeds() {
        let guard = LifecycleGuard::new();
        assert!(guard.terminate(TerminalOutcome::Cancelled(None)).is_ok());
    }

    #[test]
    fn terminate_dropped_succeeds() {
        let guard = LifecycleGuard::new();
        assert!(
            guard
                .terminate(TerminalOutcome::Dropped(Some("queue full".into())))
                .is_ok()
        );
    }

    #[test]
    fn second_terminal_rejected_as_duplicate() {
        let guard = LifecycleGuard::new();
        guard.terminate(TerminalOutcome::Completed).unwrap();

        let err = guard
            .terminate(TerminalOutcome::Failed("oops".into()))
            .unwrap_err();
        assert_eq!(*err.existing, TerminalOutcome::Completed);
        assert_eq!(*err.attempted, TerminalOutcome::Failed("oops".into()));
    }

    #[test]
    fn idempotent_same_terminal_ok() {
        let guard = LifecycleGuard::new();
        guard.terminate(TerminalOutcome::Cancelled(None)).unwrap();

        // Same terminal → Ok (idempotent)
        assert!(
            guard
                .terminate_idempotent(TerminalOutcome::Cancelled(None))
                .is_ok()
        );
    }

    #[test]
    fn idempotent_different_terminal_rejected() {
        let guard = LifecycleGuard::new();
        guard.terminate(TerminalOutcome::Completed).unwrap();

        // Different terminal → Error
        assert!(
            guard
                .terminate_idempotent(TerminalOutcome::Failed("x".into()))
                .is_err()
        );
    }

    #[test]
    fn cloned_guard_shares_terminal_state() {
        let guard = LifecycleGuard::new();
        let cloned = guard.clone();

        guard.terminate(TerminalOutcome::Completed).unwrap();
        assert!(!cloned.is_active());
        assert_eq!(cloned.terminal_outcome(), Some(TerminalOutcome::Completed));

        // Cannot terminate via clone either.
        assert!(
            cloned
                .terminate(TerminalOutcome::Failed("x".into()))
                .is_err()
        );
    }

    #[test]
    fn record_attempt_allocates_distinct_typed_ids_without_terminating_parent() {
        let guard = LifecycleGuard::new();
        let first = guard.record_attempt();

        // Even after terminal, attempt recording is allowed and remains independent.
        guard.terminate(TerminalOutcome::Completed).unwrap();
        let second = guard.record_attempt();
        assert_ne!(first, second);
        assert_eq!(guard.terminal_outcome(), Some(TerminalOutcome::Completed));
    }

    #[test]
    fn shares_track_with_same_instance() {
        let g = LifecycleGuard::new();
        assert!(g.shares_track_with(&g));
    }

    #[test]
    fn shares_track_with_clone() {
        let g1 = LifecycleGuard::new();
        let g2 = g1.clone();
        assert!(g1.shares_track_with(&g2));
    }

    #[test]
    fn does_not_share_track_independent() {
        let g1 = LifecycleGuard::new();
        let g2 = LifecycleGuard::new();
        assert!(!g1.shares_track_with(&g2));
    }

    #[test]
    fn terminal_outcome_as_str() {
        for outcome in [
            TerminalOutcome::Completed,
            TerminalOutcome::Failed("e".into()),
            TerminalOutcome::Rejected(None),
            TerminalOutcome::Cancelled(Some("r".into())),
            TerminalOutcome::Dropped(None),
        ] {
            assert!(!outcome.as_str().is_empty());
        }

        let g = LifecycleGuard::new();
        g.terminate(TerminalOutcome::Completed).unwrap();
        assert!(g.terminal_outcome().unwrap().is_success());
    }

    #[test]
    fn terminal_outcome_reason() {
        assert_eq!(TerminalOutcome::Completed.reason(), None);
        assert_eq!(TerminalOutcome::Failed("err".into()).reason(), Some("err"));
        assert_eq!(
            TerminalOutcome::Rejected(Some("r".into())).reason(),
            Some("r")
        );
        assert_eq!(TerminalOutcome::Cancelled(None).reason(), None);
    }

    #[test]
    fn default_guard_is_active() {
        let guard = LifecycleGuard::default();
        assert!(guard.is_active());
    }

    #[test]
    fn display_for_terminal_outcome() {
        let msg = format!("{}", TerminalOutcome::Failed("timeout".into()));
        assert!(msg.contains("failed"));
        assert!(msg.contains("timeout"));
    }

    #[test]
    fn duplicate_terminal_error_display() {
        let err = DuplicateTerminalError {
            existing: Box::new(TerminalOutcome::Completed),
            attempted: Box::new(TerminalOutcome::Failed("x".into())),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("completed"));
        assert!(msg.contains("failed"));
    }

    #[test]
    fn duplicate_terminal_error_is_std_error() {
        let err: Box<dyn std::error::Error> = Box::new(DuplicateTerminalError {
            existing: Box::new(TerminalOutcome::Completed),
            attempted: Box::new(TerminalOutcome::Failed("x".into())),
        });
        assert!(!err.to_string().is_empty());
    }

    #[test]
    fn concurrent_terminate_only_one_succeeds() {
        use std::thread;

        let guard = LifecycleGuard::new();
        let mut handles = vec![];

        for _i in 0..10 {
            let g = guard.clone();
            handles.push(thread::spawn(move || {
                match g.terminate(TerminalOutcome::Completed) {
                    Ok(()) => true,
                    Err(_) => false,
                }
            }));
        }

        let successes: usize = handles
            .into_iter()
            .map(|h| h.join().unwrap() as usize)
            .sum();
        assert_eq!(successes, 1, "exactly one thread should succeed");
    }
}
