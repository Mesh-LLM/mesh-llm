//! What the host observed of one native model open, and how that
//! observation is reconciled against the authoritative native return.
//!
//! The native return settles the load terminal. Callbacks are only
//! observations: they can be lost to a full queue or refused at the
//! boundary, and they can disagree with the return. This module reports
//! both conditions; it never decides the terminal.

use skippy_api::serving::{ModelLoadOutcome, ModelOpenSummary};

/// Host-side summary of the records one model-open queue produced.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct ModelOpenObservation {
    pub(crate) drained: u64,
    pub(crate) dropped: u64,
    pub(crate) rejected: u64,
    pub(crate) saw_finished: bool,
    pub(crate) saw_failed_handled: bool,
    pub(crate) last_sequence: Option<u64>,
}

/// The authoritative outcome of the blocking native open.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ModelOpenReturn {
    Succeeded,
    Failed,
}

impl ModelOpenReturn {
    pub(super) fn of(outcome: ModelLoadOutcome) -> Self {
        match outcome {
            ModelLoadOutcome::Succeeded => Self::Succeeded,
            ModelLoadOutcome::Failed => Self::Failed,
        }
    }
}

/// A terminal callback that disagrees with the native return.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ModelOpenContradiction {
    FinishedButReturnedFailure,
    FailedHandledButReturnedSuccess,
}

impl ModelOpenObservation {
    /// What Skippy's final summary says of the queue after its last drain.
    pub(super) fn of(summary: &ModelOpenSummary) -> Self {
        Self {
            drained: summary.records_observed,
            dropped: summary.dropped,
            rejected: summary.rejected,
            saw_finished: summary.saw_finished,
            saw_failed_handled: summary.saw_failed_handled,
            last_sequence: summary.last_sequence,
        }
    }

    /// Records the native side produced that never reached the host.
    pub(crate) fn lost(&self) -> u64 {
        self.dropped.saturating_add(self.rejected)
    }

    /// A missing terminal callback is not a contradiction: callbacks are
    /// best-effort and may have been dropped.
    pub(crate) fn contradiction(
        &self,
        returned: ModelOpenReturn,
    ) -> Option<ModelOpenContradiction> {
        match returned {
            ModelOpenReturn::Failed if self.saw_finished => {
                Some(ModelOpenContradiction::FinishedButReturnedFailure)
            }
            ModelOpenReturn::Succeeded if self.saw_failed_handled => {
                Some(ModelOpenContradiction::FailedHandledButReturnedSuccess)
            }
            ModelOpenReturn::Failed | ModelOpenReturn::Succeeded => None,
        }
    }
}
