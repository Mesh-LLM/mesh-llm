use std::io;
use std::process::ExitStatus;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stream {
    Stdout,
    Stderr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Exited,
    Ready,
    EarlyExit,
    Deadline,
    ReadinessDeadline,
    Cancelled,
    IoFailure,
    ObservationRejected,
}

#[derive(Debug, Default)]
pub struct StreamReport {
    pub bytes_seen: u64,
    pub bytes_retained: Vec<u8>,
    pub truncated: bool,
    pub suppressed_lines: u64,
    /// All pipe bytes reached bounded line classification and EOF, with no oversized lines.
    /// Independent of persisted diagnostic redaction/truncation.
    pub line_capture_complete: bool,
    pub oversized_lines: u64,
}

#[derive(Debug, Default)]
pub struct Cleanup {
    pub complete: bool,
    pub forced: bool,
    pub graceful_signal_failed: bool,
    pub failure: Option<Failure>,
}

/// Metadata only; elapsed is measured on the supervisor's single start clock
/// after classification and the initial live-leader check complete.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadinessObservation {
    pub stream: Stream,
    pub elapsed: Duration,
}

/// Ordered observations, never proof that a signal caused exit or was handled.
/// Rejected live-stop requests still permit ordinary descendant cleanup signals.
#[derive(Debug)]
pub enum GracefulRequest {
    RequestedAfterLiveObservation,
    LeaderExitedBeforeRequest,
    LeaderObservationFailed(Failure),
    SkippedInactiveTree,
    TreeObservationFailed,
    RequestFailed(Failure),
}

#[derive(Debug, Default)]
pub enum ReadinessStop {
    #[default]
    NotAdmitted,
    Admitted {
        observation: ReadinessObservation,
        request: GracefulRequest,
    },
    ProbeAdmitted {
        elapsed: Duration,
        request: GracefulRequest,
    },
}

pub(super) enum StopObservation {
    Line(ReadinessObservation),
    Probe(Duration),
}

impl StopObservation {
    pub(super) fn receipt(self, request: GracefulRequest) -> ReadinessStop {
        match self {
            Self::Line(observation) => ReadinessStop::Admitted {
                observation,
                request,
            },
            Self::Probe(elapsed) => ReadinessStop::ProbeAdmitted { elapsed, request },
        }
    }
}

#[derive(Debug)]
pub struct ProcessReport {
    pub pid: u32,
    pub outcome: Outcome,
    pub status: Option<ExitStatus>,
    pub ready: bool,
    /// ObservedLines and probe admission have distinct receipts; legacy semantics remain.
    pub readiness_stop: ReadinessStop,
    pub elapsed: Duration,
    pub stdout: StreamReport,
    pub stderr: StreamReport,
    pub cleanup: Cleanup,
    pub failure: Option<Failure>,
}

impl ProcessReport {
    pub fn success(&self) -> bool {
        let completed = match self.outcome {
            Outcome::Exited | Outcome::Ready => true,
            Outcome::EarlyExit
            | Outcome::Deadline
            | Outcome::ReadinessDeadline
            | Outcome::Cancelled
            | Outcome::ObservationRejected
            | Outcome::IoFailure => false,
        };
        completed
            && match &self.readiness_stop {
                ReadinessStop::NotAdmitted => true,
                ReadinessStop::Admitted { request, .. }
                | ReadinessStop::ProbeAdmitted { request, .. } => match request {
                    GracefulRequest::RequestedAfterLiveObservation => true,
                    GracefulRequest::LeaderExitedBeforeRequest
                    | GracefulRequest::LeaderObservationFailed(_)
                    | GracefulRequest::SkippedInactiveTree
                    | GracefulRequest::TreeObservationFailed
                    | GracefulRequest::RequestFailed(_) => false,
                },
            }
            && self.status.is_some_and(|status| status.success())
            && self.cleanup.complete
            && !self.cleanup.forced
            && !self.cleanup.graceful_signal_failed
            && self.cleanup.failure.is_none()
            && self.failure.is_none()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Failure {
    #[error(
        "retained member name must contain 1..=64 ASCII letters, digits, hyphens or underscores"
    )]
    InvalidMemberName,
    #[error("retained member restart generation exhausted")]
    MemberGenerationExhausted,
    #[error("retained session exceeds {limit} concurrent members")]
    RetainedMemberLimit { limit: usize },
    #[error("raw {stream:?} capture exceeded {limit} bytes")]
    RawCaptureOverflow { stream: Stream, limit: usize },
    #[error("raw {stream:?} capture did not reach EOF")]
    RawCaptureIncomplete { stream: Stream },
    #[error("invalid process specification: {0}")]
    InvalidSpec(&'static str),
    #[error("process {operation} failed ({kind:?}, OS code {code:?})")]
    Io {
        operation: &'static str,
        kind: io::ErrorKind,
        code: Option<i32>,
    },
    #[error("owned process cleanup deadline exceeded")]
    CleanupDeadline,
    #[error("process enumeration exceeded its bound")]
    EnumerationLimit,
}

impl Failure {
    pub(super) fn io(operation: &'static str, error: io::Error) -> Self {
        Self::Io {
            operation,
            kind: error.kind(),
            code: error.raw_os_error(),
        }
    }
}
