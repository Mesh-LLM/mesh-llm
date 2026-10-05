//! What the host observed of one native model open, and how that
//! observation is reconciled against the authoritative native return.
//!
//! The native return settles the load terminal. Callbacks are only
//! observations: they can be lost to a full queue or refused at the
//! boundary, and they can disagree with the return. This module reports
//! both conditions; it never decides the terminal.

use skippy_ffi::SkippyRuntimeEventKind;
use skippy_runtime::{ModelOpenEventQueue, NativeEventRecord};

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
    pub(crate) fn of<T, E>(result: &Result<T, E>) -> Self {
        match result {
            Ok(_) => Self::Succeeded,
            Err(_) => Self::Failed,
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
    pub(super) fn record(&mut self, record: &NativeEventRecord) {
        self.drained += 1;
        self.last_sequence = Some(record.sequence);
        let kind = SkippyRuntimeEventKind(record.kind);
        self.saw_finished |= kind == SkippyRuntimeEventKind::MODEL_OPEN_FINISHED;
        self.saw_failed_handled |= kind == SkippyRuntimeEventKind::MODEL_OPEN_FAILED_HANDLED;
    }

    /// Copy the queue's loss counters. Called once the final drain is done.
    pub(super) fn with_losses_from(self, queue: &ModelOpenEventQueue) -> Self {
        Self {
            dropped: queue.dropped(),
            rejected: queue.rejected(),
            ..self
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
