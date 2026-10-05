//! Lifecycle state machine with exactly-one-terminal-transition invariant.

use std::{
    fmt,
    sync::{
        Arc,
        atomic::{AtomicU8, Ordering},
    },
};

/// Terminal or active lifecycle states for a request or attempt.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum LifecycleState {
    /// The entity is currently being processed.
    Active,
    /// Processing finished successfully.
    Completed,
    /// Processing ended with an error.
    Failed,
    /// Processing was rejected before beginning (e.g., invalid input).
    Rejected,
    /// Processing was cancelled by the caller or system.
    Cancelled,
}

impl LifecycleState {
    /// Returns `true` if this state is terminal (no further transitions allowed).
    pub fn is_terminal(&self) -> bool {
        !matches!(self, LifecycleState::Active)
    }

    #[allow(dead_code)]
    pub fn as_str(&self) -> &'static str {
        match self {
            LifecycleState::Active => "active",
            LifecycleState::Completed => "completed",
            LifecycleState::Failed => "failed",
            LifecycleState::Rejected => "rejected",
            LifecycleState::Cancelled => "cancelled",
        }
    }

    const fn as_u8(self) -> u8 {
        match self {
            Self::Active => 0,
            Self::Completed => 1,
            Self::Failed => 2,
            Self::Rejected => 3,
            Self::Cancelled => 4,
        }
    }

    fn from_u8(value: u8) -> Self {
        match value {
            0 => Self::Active,
            1 => Self::Completed,
            2 => Self::Failed,
            3 => Self::Rejected,
            4 => Self::Cancelled,
            _ => unreachable!("LifecycleGuard only stores valid LifecycleState values"),
        }
    }
}

/// Error returned when a lifecycle transition is invalid.
#[derive(Clone, Copy, Debug)]
pub struct LifecycleTransitionError {
    pub from: LifecycleState,
    pub to: LifecycleState,
}

impl fmt::Display for LifecycleTransitionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid transition {} → {}",
            self.from.as_str(),
            self.to.as_str()
        )
    }
}

impl std::error::Error for LifecycleTransitionError {}

/// Guard that owns the current lifecycle state and enforces exactly-one-terminal-transition.
///
/// A guard starts in `Active` (or a provided initial state). Once it transitions to any terminal
/// state, all further transition attempts are rejected with [`LifecycleTransitionError`].
#[derive(Clone, Debug)]
pub struct LifecycleGuard {
    current: Arc<AtomicU8>,
}

impl PartialEq for LifecycleGuard {
    fn eq(&self, other: &Self) -> bool {
        self.state() == other.state()
    }
}

impl Eq for LifecycleGuard {}

impl LifecycleGuard {
    /// Create a new guard starting in the given state (typically `Active`).
    pub fn new(state: LifecycleState) -> Self {
        Self {
            current: Arc::new(AtomicU8::new(state.as_u8())),
        }
    }

    /// Returns a guard initialized to `Active`.
    #[allow(dead_code)]
    pub fn active() -> Self {
        Self::new(LifecycleState::Active)
    }

    /// Return the current lifecycle state.
    #[allow(dead_code)]
    pub fn state(&self) -> LifecycleState {
        LifecycleState::from_u8(self.current.load(Ordering::Acquire))
    }

    /// Transition to a new state, returning an error if:
    /// - The current state is already terminal (no further transitions allowed), OR
    /// - Attempting the same non-terminal transition twice where idempotency does not apply.
    ///
    /// Note: transitioning from `Active` → `Active` is allowed (idempotent no-op).
    pub fn transition(
        &mut self,
        new_state: LifecycleState,
    ) -> Result<(), LifecycleTransitionError> {
        if new_state == LifecycleState::Active {
            let current = self.state();
            if current == LifecycleState::Active {
                return Ok(());
            }
            return Err(LifecycleTransitionError {
                from: current,
                to: new_state,
            });
        }

        match self.current.compare_exchange(
            LifecycleState::Active.as_u8(),
            new_state.as_u8(),
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => Ok(()),
            Err(current) => Err(LifecycleTransitionError {
                from: LifecycleState::from_u8(current),
                to: new_state,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_active_to_completed_works() {
        let mut guard = LifecycleGuard::active();
        assert!(guard.transition(LifecycleState::Completed).is_ok());
        assert_eq!(guard.state(), LifecycleState::Completed);
    }

    #[test]
    fn test_completed_to_failed_fails() {
        let mut guard = LifecycleGuard::active();
        guard.transition(LifecycleState::Completed).unwrap();
        let err = guard.transition(LifecycleState::Failed).unwrap_err();
        assert!(matches!(
            err,
            LifecycleTransitionError {
                from: LifecycleState::Completed,
                to: LifecycleState::Failed,
            }
        ));
        assert_eq!(guard.state(), LifecycleState::Completed);
    }

    #[test]
    fn test_second_terminal_rejected() {
        let mut guard = LifecycleGuard::active();
        guard.transition(LifecycleState::Completed).unwrap();
        let err = guard.transition(LifecycleState::Completed).unwrap_err();
        assert!(matches!(
            err,
            LifecycleTransitionError {
                from: LifecycleState::Completed,
                to: LifecycleState::Completed,
            }
        ));
        assert_eq!(guard.state(), LifecycleState::Completed);
    }

    #[test]
    fn test_active_to_active_idempotent() {
        let mut guard = LifecycleGuard::active();
        assert!(guard.transition(LifecycleState::Active).is_ok());
        assert_eq!(guard.state(), LifecycleState::Active);
    }

    #[test]
    fn test_terminal_states_are_terminal() {
        for state in [
            LifecycleState::Completed,
            LifecycleState::Failed,
            LifecycleState::Rejected,
            LifecycleState::Cancelled,
        ] {
            assert!(state.is_terminal(), "{:?} should be terminal", state);
        }
        assert!(!LifecycleState::Active.is_terminal());
    }

    #[test]
    fn test_active_to_all_terminals_work() {
        for target in [
            LifecycleState::Completed,
            LifecycleState::Failed,
            LifecycleState::Rejected,
            LifecycleState::Cancelled,
        ] {
            let mut guard = LifecycleGuard::active();
            assert!(
                guard.transition(target).is_ok(),
                "Active→{:?} should work",
                target
            );
        }
    }

    #[test]
    fn test_lifecycle_transition_error_display() {
        let err = LifecycleTransitionError {
            from: LifecycleState::Completed,
            to: LifecycleState::Failed,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("completed"));
        assert!(msg.contains("failed"));
    }

    #[test]
    fn test_guard_clones_share_terminal_transition() {
        let mut guard1 = LifecycleGuard::active();
        let mut guard2 = guard1.clone();

        guard2.transition(LifecycleState::Failed).unwrap();

        assert_eq!(guard1.state(), LifecycleState::Failed);
        let error = guard1.transition(LifecycleState::Completed).unwrap_err();
        assert!(matches!(
            error,
            LifecycleTransitionError {
                from: LifecycleState::Failed,
                to: LifecycleState::Completed,
            }
        ));
    }

    #[test]
    fn test_concurrent_clones_accept_exactly_one_terminal_transition() {
        use std::{sync::Barrier, thread};

        let guard = LifecycleGuard::active();
        let mut completed = guard.clone();
        let mut failed = guard.clone();
        let barrier = Arc::new(Barrier::new(3));
        let completed_barrier = Arc::clone(&barrier);
        let failed_barrier = Arc::clone(&barrier);

        let completed_handle = thread::spawn(move || {
            completed_barrier.wait();
            completed.transition(LifecycleState::Completed)
        });
        let failed_handle = thread::spawn(move || {
            failed_barrier.wait();
            failed.transition(LifecycleState::Failed)
        });

        barrier.wait();
        let results = [
            completed_handle.join().unwrap(),
            failed_handle.join().unwrap(),
        ];
        assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
        assert!(matches!(
            guard.state(),
            LifecycleState::Completed | LifecycleState::Failed
        ));
    }

    #[test]
    fn test_new_guard_with_custom_state() {
        let mut guard = LifecycleGuard::new(LifecycleState::Cancelled);
        assert_eq!(guard.state(), LifecycleState::Cancelled);
        // Already terminal → any transition fails
        let err = guard.transition(LifecycleState::Active).unwrap_err();
        assert!(matches!(
            err,
            LifecycleTransitionError {
                from: LifecycleState::Cancelled,
                to: LifecycleState::Active,
            }
        ));
        assert_eq!(guard.state(), LifecycleState::Cancelled);
    }

    #[test]
    fn test_state_as_str() {
        assert_eq!(LifecycleState::Active.as_str(), "active");
        assert_eq!(LifecycleState::Completed.as_str(), "completed");
        assert_eq!(LifecycleState::Failed.as_str(), "failed");
        assert_eq!(LifecycleState::Rejected.as_str(), "rejected");
        assert_eq!(LifecycleState::Cancelled.as_str(), "cancelled");
    }
}
