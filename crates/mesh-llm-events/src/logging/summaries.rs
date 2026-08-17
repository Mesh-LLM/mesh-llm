//! Request summary for lifecycle tracking and reporting.

use std::sync::{Arc, Mutex, MutexGuard};

use super::identifiers::RequestId;
use super::lifecycle::{LifecycleState, LifecycleTransitionError};

/// Immutable terminal outcome accepted for a request.
///
/// A request summary has at most one terminal outcome. Once accepted, its state, applicable
/// metadata, and timestamp are retained together and cannot be changed by later transitions.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RequestTerminalOutcome {
    pub state: LifecycleState,
    /// ISO 8601 timestamp when the request reached its terminal state.
    pub terminal_at: String,
    /// Error associated with a failed request.
    pub error: Option<String>,
    /// Reason associated with a rejected or cancelled request.
    pub reason: Option<String>,
}

#[derive(Debug)]
struct SummaryLifecycle {
    outcome: Option<RequestTerminalOutcome>,
}

/// Compact view of a request's lifecycle state and routing metadata.
#[derive(Clone, Debug)]
pub struct RequestSummary {
    pub request_id: RequestId,
    lifecycle: Arc<Mutex<SummaryLifecycle>>,
    /// ISO 8601 timestamp when the request was created.
    pub created_at: String,

    // Routing metadata.
    #[allow(dead_code)]
    pub route: Option<String>,
    #[allow(dead_code)]
    pub model: Option<String>,
    #[allow(dead_code)]
    pub provider: Option<String>,
    #[allow(dead_code)]
    pub engine: Option<String>,

    // Outcome metadata.
    #[allow(dead_code)]
    pub status_code: Option<u16>,

    // Nullable reserved identity fields.
    #[allow(dead_code)]
    pub tenant_id: Option<String>,
    #[allow(dead_code)]
    pub account_id: Option<String>,
    #[allow(dead_code)]
    pub user_id: Option<String>,
}

impl RequestSummary {
    /// Create a new summary in the Active state.
    #[allow(dead_code)]
    pub fn new(request_id: RequestId, created_at: String) -> Self {
        Self {
            request_id,
            lifecycle: Arc::new(Mutex::new(SummaryLifecycle { outcome: None })),
            created_at,
            route: None,
            model: None,
            provider: None,
            engine: None,
            status_code: None,
            tenant_id: None,
            account_id: None,
            user_id: None,
        }
    }

    /// Mark the request as completed if it has not already reached a terminal state.
    #[allow(dead_code)]
    pub fn set_completed(&self, terminal_at: String) -> Result<(), LifecycleTransitionError> {
        self.set_terminal(LifecycleState::Completed, terminal_at, None, None)
    }

    /// Mark the request as failed with an error message if it has not already reached a terminal
    /// state. The error is only recorded when the transition succeeds.
    #[allow(dead_code)]
    pub fn set_failed(
        &self,
        error: String,
        terminal_at: String,
    ) -> Result<(), LifecycleTransitionError> {
        self.set_terminal(LifecycleState::Failed, terminal_at, Some(error), None)
    }

    /// Mark the request as rejected before processing. The reason is only recorded when the
    /// transition succeeds.
    #[allow(dead_code)]
    pub fn set_rejected(
        &self,
        reason: Option<String>,
        terminal_at: String,
    ) -> Result<(), LifecycleTransitionError> {
        self.set_terminal(LifecycleState::Rejected, terminal_at, None, reason)
    }

    /// Mark the request as cancelled. The reason is only recorded when the transition succeeds.
    #[allow(dead_code)]
    pub fn set_cancelled(
        &self,
        reason: Option<String>,
        terminal_at: String,
    ) -> Result<(), LifecycleTransitionError> {
        self.set_terminal(LifecycleState::Cancelled, terminal_at, None, reason)
    }

    /// Return the current lifecycle state.
    #[allow(dead_code)]
    pub fn state(&self) -> LifecycleState {
        self.lifecycle()
            .outcome
            .as_ref()
            .map_or(LifecycleState::Active, |outcome| outcome.state)
    }

    /// Return the immutable terminal outcome, if one has been accepted.
    #[allow(dead_code)]
    pub fn terminal_outcome(&self) -> Option<RequestTerminalOutcome> {
        self.lifecycle().outcome.clone()
    }

    /// Return the terminal timestamp, if the request has reached a terminal state.
    #[allow(dead_code)]
    pub fn terminal_at(&self) -> Option<String> {
        self.terminal_outcome().map(|outcome| outcome.terminal_at)
    }

    /// Return the failure error, if the request failed.
    #[allow(dead_code)]
    pub fn error(&self) -> Option<String> {
        self.terminal_outcome().and_then(|outcome| outcome.error)
    }

    /// Return the rejection or cancellation reason, if present.
    #[allow(dead_code)]
    pub fn reason(&self) -> Option<String> {
        self.terminal_outcome().and_then(|outcome| outcome.reason)
    }

    /// Check if this summary is in a terminal state.
    #[allow(dead_code)]
    pub fn is_terminal(&self) -> bool {
        self.state().is_terminal()
    }

    fn lifecycle(&self) -> MutexGuard<'_, SummaryLifecycle> {
        self.lifecycle
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn set_terminal(
        &self,
        state: LifecycleState,
        terminal_at: String,
        error: Option<String>,
        reason: Option<String>,
    ) -> Result<(), LifecycleTransitionError> {
        let mut lifecycle = self.lifecycle();
        if let Some(outcome) = &lifecycle.outcome {
            return Err(LifecycleTransitionError {
                from: outcome.state,
                to: state,
            });
        }

        lifecycle.outcome = Some(RequestTerminalOutcome {
            state,
            terminal_at,
            error,
            reason,
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{Arc, Barrier},
        thread,
    };

    use super::*;

    const CREATED_AT: &str = "2025-01-01T00:00:00Z";

    #[derive(Clone, Debug)]
    enum CompetingOutcome {
        Completed,
        Failed,
        Rejected,
        Cancelled,
    }

    impl CompetingOutcome {
        fn state(&self) -> LifecycleState {
            match self {
                Self::Completed => LifecycleState::Completed,
                Self::Failed => LifecycleState::Failed,
                Self::Rejected => LifecycleState::Rejected,
                Self::Cancelled => LifecycleState::Cancelled,
            }
        }

        fn apply(
            &self,
            summary: &RequestSummary,
            terminal_at: String,
        ) -> Result<(), LifecycleTransitionError> {
            match self {
                Self::Completed => summary.set_completed(terminal_at),
                Self::Failed => summary.set_failed("timeout".into(), terminal_at),
                Self::Rejected => {
                    summary.set_rejected(Some("admission policy".into()), terminal_at)
                }
                Self::Cancelled => {
                    summary.set_cancelled(Some("caller disconnected".into()), terminal_at)
                }
            }
        }
    }

    fn summary() -> RequestSummary {
        RequestSummary::new(RequestId::new(), CREATED_AT.into())
    }

    #[test]
    fn test_new_summary_is_active() {
        let summary = summary();
        assert_eq!(summary.state(), LifecycleState::Active);
        assert!(summary.terminal_outcome().is_none());
        assert!(!summary.is_terminal());
    }

    #[test]
    fn test_set_completed_records_terminal_timestamp() {
        let summary = summary();
        summary
            .set_completed("2025-01-01T00:00:01Z".into())
            .unwrap();

        assert_eq!(summary.state(), LifecycleState::Completed);
        assert_eq!(
            summary.terminal_outcome(),
            Some(RequestTerminalOutcome {
                state: LifecycleState::Completed,
                terminal_at: "2025-01-01T00:00:01Z".into(),
                error: None,
                reason: None,
            })
        );
        assert!(summary.is_terminal());
    }

    #[test]
    fn test_set_failed_records_error_and_timestamp() {
        let summary = summary();
        summary
            .set_failed("timeout".into(), "2025-01-01T00:00:01Z".into())
            .unwrap();

        assert_eq!(summary.state(), LifecycleState::Failed);
        assert_eq!(summary.error().as_deref(), Some("timeout"));
        assert_eq!(
            summary.terminal_at().as_deref(),
            Some("2025-01-01T00:00:01Z")
        );
        assert!(summary.reason().is_none());
    }

    #[test]
    fn test_set_rejected_records_reason_and_timestamp() {
        let summary = summary();
        summary
            .set_rejected(
                Some("admission policy".into()),
                "2025-01-01T00:00:01Z".into(),
            )
            .unwrap();

        assert_eq!(summary.state(), LifecycleState::Rejected);
        assert_eq!(summary.reason().as_deref(), Some("admission policy"));
        assert_eq!(
            summary.terminal_at().as_deref(),
            Some("2025-01-01T00:00:01Z")
        );
        assert!(summary.error().is_none());
    }

    #[test]
    fn test_set_cancelled_records_reason_and_timestamp() {
        let summary = summary();
        summary
            .set_cancelled(
                Some("caller disconnected".into()),
                "2025-01-01T00:00:01Z".into(),
            )
            .unwrap();

        assert_eq!(summary.state(), LifecycleState::Cancelled);
        assert_eq!(summary.reason().as_deref(), Some("caller disconnected"));
        assert_eq!(
            summary.terminal_at().as_deref(),
            Some("2025-01-01T00:00:01Z")
        );
        assert!(summary.error().is_none());
    }

    #[test]
    fn test_duplicate_terminal_does_not_mutate_outcome() {
        let summary = summary();
        summary
            .set_failed("timeout".into(), "2025-01-01T00:00:01Z".into())
            .unwrap();
        let outcome = summary.terminal_outcome();

        let error = summary
            .set_cancelled(
                Some("caller disconnected".into()),
                "2025-01-01T00:00:02Z".into(),
            )
            .unwrap_err();
        assert!(matches!(
            error,
            LifecycleTransitionError {
                from: LifecycleState::Failed,
                to: LifecycleState::Cancelled,
            }
        ));
        assert_eq!(summary.terminal_outcome(), outcome);
    }

    #[test]
    fn test_summary_clones_share_identical_winning_terminal_outcome() {
        let summary = summary();
        let clone = summary.clone();

        clone
            .set_rejected(
                Some("admission policy".into()),
                "2025-01-01T00:00:01Z".into(),
            )
            .unwrap();
        let error = summary
            .set_cancelled(
                Some("caller disconnected".into()),
                "2025-01-01T00:00:02Z".into(),
            )
            .unwrap_err();

        assert!(matches!(
            error,
            LifecycleTransitionError {
                from: LifecycleState::Rejected,
                to: LifecycleState::Cancelled,
            }
        ));
        assert_eq!(summary.terminal_outcome(), clone.terminal_outcome());
        assert_eq!(summary.reason().as_deref(), Some("admission policy"));
        assert_eq!(
            summary.terminal_at().as_deref(),
            Some("2025-01-01T00:00:01Z")
        );
    }

    #[test]
    fn test_concurrent_duplicate_terminal_cannot_mutate_winning_metadata() {
        let summary = summary();
        let first_summary = summary.clone();
        let second_summary = summary.clone();
        let barrier = Arc::new(Barrier::new(3));
        let first_barrier = Arc::clone(&barrier);
        let second_barrier = Arc::clone(&barrier);

        let first_handle = thread::spawn(move || {
            first_barrier.wait();
            let result =
                first_summary.set_failed("first timeout".into(), "2025-01-01T00:00:01Z".into());
            (first_summary, result)
        });
        let second_handle = thread::spawn(move || {
            second_barrier.wait();
            let result =
                second_summary.set_failed("second timeout".into(), "2025-01-01T00:00:02Z".into());
            (second_summary, result)
        });

        barrier.wait();
        let (first_summary, first_result) = first_handle.join().unwrap();
        let (second_summary, second_result) = second_handle.join().unwrap();
        assert_eq!(
            usize::from(first_result.is_ok()) + usize::from(second_result.is_ok()),
            1
        );
        assert!(matches!(
            if first_result.is_err() {
                first_result
            } else {
                second_result
            },
            Err(LifecycleTransitionError {
                from: LifecycleState::Failed,
                to: LifecycleState::Failed,
            })
        ));

        let winner = summary.terminal_outcome().unwrap();
        assert!(matches!(
            (winner.error.as_deref(), winner.terminal_at.as_str()),
            (Some("first timeout"), "2025-01-01T00:00:01Z")
                | (Some("second timeout"), "2025-01-01T00:00:02Z")
        ));
        assert_eq!(first_summary.terminal_outcome(), Some(winner.clone()));
        assert_eq!(second_summary.terminal_outcome(), Some(winner));
    }

    #[test]
    fn test_concurrent_clones_accept_one_immutable_outcome_for_each_terminal_pair() {
        let outcomes = [
            CompetingOutcome::Completed,
            CompetingOutcome::Failed,
            CompetingOutcome::Rejected,
            CompetingOutcome::Cancelled,
        ];

        for (first_index, first) in outcomes.iter().enumerate() {
            for second in outcomes.iter().skip(first_index + 1) {
                let summary = summary();
                let first_summary = summary.clone();
                let second_summary = summary.clone();
                let barrier = Arc::new(Barrier::new(3));
                let first_barrier = Arc::clone(&barrier);
                let second_barrier = Arc::clone(&barrier);
                let first = first.clone();
                let second = second.clone();
                let first_state = first.state();
                let second_state = second.state();

                let first_handle = thread::spawn(move || {
                    first_barrier.wait();
                    let result = first.apply(&first_summary, "2025-01-01T00:00:01Z".into());
                    (first_summary, result)
                });
                let second_handle = thread::spawn(move || {
                    second_barrier.wait();
                    let result = second.apply(&second_summary, "2025-01-01T00:00:02Z".into());
                    (second_summary, result)
                });

                barrier.wait();
                let (first_summary, first_result) = first_handle.join().unwrap();
                let (second_summary, second_result) = second_handle.join().unwrap();
                assert_eq!(
                    usize::from(first_result.is_ok()) + usize::from(second_result.is_ok()),
                    1
                );

                let winning_state = if first_result.is_ok() {
                    first_state
                } else {
                    second_state
                };
                let losing_state = if first_result.is_ok() {
                    second_state
                } else {
                    first_state
                };
                let losing_result = if first_result.is_err() {
                    first_result
                } else {
                    second_result
                };
                assert!(matches!(
                    losing_result,
                    Err(LifecycleTransitionError { from, to }) if from == winning_state && to == losing_state
                ));

                let winner = summary.terminal_outcome().unwrap();
                assert_eq!(winner.state, winning_state);
                assert_eq!(first_summary.terminal_outcome(), Some(winner.clone()));
                assert_eq!(second_summary.terminal_outcome(), Some(winner));
            }
        }
    }
}
