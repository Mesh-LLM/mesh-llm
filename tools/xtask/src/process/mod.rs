//! Bounded supervision of cooperative child trees. No existing caller is switched.
//!
//! Unix children must retain their inherited process group: this is lifecycle
//! ownership, not a sandbox for programs that call setsid/setpgid. The leader is
//! left unreaped until the last group signal, preventing PGID reuse. Windows
//! children start suspended, enter a non-breakaway kill-on-close job, then resume.
//! A supervisor forcibly killed during suspended Windows assignment can orphan
//! that suspended child; normal errors and unwinding retain cleanup ownership.
//!
//! Deadlines cover monitoring and cleanup, not a kernel-stalled spawn, filesystem
//! syscall, or scheduler starvation. No blocking pipe reader or detached thread
//! can outlive a run. Output is drained fairly even after retention limits fill.
//! Native runtime log files are never deleted or read implicitly. Callers retain
//! their private runtime directories and explicitly select evidence destinations.
//! macOS capture uses private FIFOs, opened with atomic O_CLOEXEC and unlinked
//! before spawn; unrelated launches need no shared lock. EOF is read directly,
//! not inferred from FIFO poll/kqueue notifications or from leader exit.

mod capture;
mod control;
mod inherited;
mod line;
mod observed;
mod output;
mod probe;
#[cfg(test)]
#[path = "probe_stop_tests.rs"]
mod probe_stop_tests;
#[cfg(test)]
#[path = "probe_tests.rs"]
mod probe_tests;
#[cfg(test)]
#[path = "probe_tick_tests.rs"]
mod probe_tick_tests;
mod raw;
mod report;
pub mod retained;
mod spec;
mod supervisor;
#[cfg(unix)]
mod unix;
#[cfg(windows)]
mod windows;

pub use inherited::{InheritedReport, supervise_inherited, supervise_inherited_closed_stdin};
pub use line::{LineEnding, LineMatcher, ObservedLine};
pub use probe::{
    Probe, ProbeContext, ProbeDecision, ProbeReport, ReadinessProbe, supervise_with_probe,
};
pub use raw::{RawBytes, RawCaptureOptions, RawProcessReport};
pub use report::{
    Cleanup, Failure, GracefulRequest, Outcome, ProcessReport, ReadinessObservation, ReadinessStop,
    Stream, StreamReport,
};
pub use spec::{Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value};
pub use supervisor::{supervise, supervise_raw};

#[cfg(unix)]
use unix as platform;
#[cfg(windows)]
use windows as platform;
