use super::control::{Leader, Tree, shutdown_ready};
use super::observed::Decision;
use super::probe::{Admission, ProbeLines, Session};
use super::report::StopObservation;
use super::*;
use std::collections::VecDeque;
use std::process::ExitStatus;
use std::time::Duration;

enum Step {
    Leader(bool),
    Active(bool),
    Graceful,
    Reap,
}
struct Script(VecDeque<Step>);
impl Leader for Script {
    fn exited(&mut self) -> Result<bool, Failure> {
        match self.0.pop_front() {
            Some(Step::Leader(exited)) => Ok(exited),
            _ => panic!("leader query out of order"),
        }
    }
}
impl Tree for Script {
    fn active(&mut self) -> Result<bool, Failure> {
        match self.0.pop_front() {
            Some(Step::Active(active)) => Ok(active),
            _ => panic!("tree query out of order"),
        }
    }
    fn graceful(&mut self) -> Result<(), Failure> {
        assert!(matches!(self.0.pop_front(), Some(Step::Graceful)));
        Ok(())
    }
    fn force(&mut self) -> Result<(), Failure> {
        panic!("unexpected force")
    }
    fn reap(&mut self) -> Result<Option<ExitStatus>, Failure> {
        assert!(matches!(self.0.pop_front(), Some(Step::Reap)));
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;
            Ok(Some(ExitStatus::from_raw(0)))
        }
        #[cfg(windows)]
        {
            use std::os::windows::process::ExitStatusExt;
            Ok(Some(ExitStatus::from_raw(0)))
        }
    }
}
struct Ready;
impl ReadinessProbe for Ready {
    type Rejection = ();
    fn tick(&mut self, _: ProbeContext) -> ProbeDecision<()> {
        ProbeDecision::Candidate
    }
    fn line(&mut self, _: ObservedLine<'_>) -> ProbeDecision<()> {
        panic!("candidate must freeze lines")
    }
}
struct Output;
impl ProbeLines for Output {
    fn poll_probe(&mut self, callback: &mut dyn FnMut(ObservedLine<'_>)) -> Result<(), Failure> {
        callback(ObservedLine {
            stream: Stream::Stderr,
            bytes: b"cleanup",
            ending: LineEnding::Lf,
        });
        Ok(())
    }
}

#[test]
fn probe_pre_request_exit_keeps_admission_but_rejects_success() {
    for exited in [false, true] {
        let mut child = Script(VecDeque::from([
            Step::Leader(false),
            Step::Active(true),
            Step::Leader(exited),
            Step::Graceful,
            Step::Active(false),
            Step::Active(false),
            Step::Reap,
        ]));
        let limits = Limits {
            execution: Duration::from_secs(2),
            graceful_shutdown: Duration::from_millis(1),
            forced_shutdown: Duration::from_millis(1),
            retained_bytes_per_stream: 0,
            readiness: Readiness::None,
            completion: Completion::Exit,
        };
        let cancellation = Cancellation::default();
        let now = std::cell::Cell::new(Duration::from_secs(1) - Duration::from_nanos(1));
        let admission = Admission {
            limits: &limits,
            cancellation: &cancellation,
            elapsed: || now.get(),
            deadline: Duration::from_secs(1),
            pid: 42,
        };
        let decision = Session::new(&mut Ready, &admission).monitor(&mut child, &mut Output);
        let elapsed = match decision {
            Decision::Admitted(((), elapsed)) => elapsed,
            _ => panic!("expected admission"),
        };
        now.set(Duration::from_secs(3));
        cancellation.cancel();
        let (cleanup, status, request) = shutdown_ready(&mut child, &limits, || {});
        let report = ProcessReport {
            pid: 42,
            outcome: Outcome::Ready,
            status,
            ready: true,
            readiness_stop: StopObservation::Probe(elapsed).receipt(request),
            elapsed: now.get(),
            stdout: StreamReport::default(),
            stderr: StreamReport::default(),
            cleanup,
            failure: None,
        };
        assert_eq!(report.success(), !exited);
        assert!(report.cleanup.complete);
        match report.readiness_stop {
            ReadinessStop::ProbeAdmitted {
                elapsed: actual,
                request,
            } => {
                assert_eq!(actual, elapsed);
                assert_eq!(
                    matches!(request, GracefulRequest::LeaderExitedBeforeRequest),
                    exited
                );
                assert_eq!(
                    matches!(request, GracefulRequest::RequestedAfterLiveObservation),
                    !exited
                );
            }
            ReadinessStop::Admitted { .. } | ReadinessStop::NotAdmitted => panic!("wrong receipt"),
        }
        assert!(child.0.is_empty());
    }
}
