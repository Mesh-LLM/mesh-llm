use super::control::Leader;
use super::observed::Decision;
use super::probe::{Admission, ProbeLines, Session};
use super::*;
use std::cell::Cell;
use std::time::Duration;

struct Observer<'a> {
    now: &'a Cell<Duration>,
    advance: Duration,
    ticks: usize,
}
impl ReadinessProbe for Observer<'_> {
    type Rejection = ();
    fn tick(&mut self, context: ProbeContext) -> ProbeDecision<()> {
        assert_eq!(context.elapsed, self.now.get());
        assert_eq!(context.remaining, Duration::from_secs(1) - context.elapsed);
        self.ticks += 1;
        self.now.set(self.advance);
        ProbeDecision::Candidate
    }
    fn line(&mut self, _: ObservedLine<'_>) -> ProbeDecision<()> {
        panic!("tick candidate must freeze lines")
    }
}
struct Child {
    result: Result<bool, Failure>,
    calls: usize,
}
impl Leader for Child {
    fn exited(&mut self) -> Result<bool, Failure> {
        self.calls += 1;
        std::mem::replace(&mut self.result, Ok(false))
    }
}
struct Output;
impl ProbeLines for Output {
    fn poll_probe(&mut self, callback: &mut dyn FnMut(ObservedLine<'_>)) -> Result<(), Failure> {
        callback(ObservedLine {
            stream: Stream::Stdout,
            bytes: b"READY",
            ending: LineEnding::Lf,
        });
        Ok(())
    }
}
fn limits() -> Limits {
    Limits {
        execution: Duration::from_secs(2),
        graceful_shutdown: Duration::from_millis(1),
        forced_shutdown: Duration::from_millis(1),
        retained_bytes_per_stream: 0,
        readiness: Readiness::None,
        completion: Completion::Exit,
    }
}

#[test]
fn probe_tick_candidate_checks_deadline_before_live_query() {
    let deadline = Duration::from_secs(1);
    for elapsed in [
        deadline - Duration::from_nanos(1),
        deadline,
        deadline + Duration::from_nanos(1),
    ] {
        let now = Cell::new(Duration::ZERO);
        let cancellation = Cancellation::default();
        let limits = limits();
        let admission = Admission {
            limits: &limits,
            cancellation: &cancellation,
            elapsed: || now.get(),
            deadline,
            pid: 42,
        };
        let mut observer = Observer {
            now: &now,
            advance: elapsed,
            ticks: 0,
        };
        let mut child = Child {
            result: Ok(false),
            calls: 0,
        };
        let decision = Session::new(&mut observer, &admission).monitor(&mut child, &mut Output);
        assert_eq!(observer.ticks, 1);
        assert_eq!(child.calls, usize::from(elapsed < deadline));
        if elapsed < deadline {
            assert!(matches!(decision, Decision::Admitted(((), actual)) if actual == elapsed));
        } else {
            assert!(matches!(
                decision,
                Decision::Terminal(Outcome::ReadinessDeadline)
            ));
        }
    }
}

#[test]
fn probe_expired_start_freezes_tick_and_line_callbacks() {
    for (execution, expected) in [
        (Duration::from_secs(2), Outcome::ReadinessDeadline),
        (Duration::from_secs(1), Outcome::Deadline),
    ] {
        let now = Cell::new(Duration::from_secs(1));
        let cancellation = Cancellation::default();
        let mut limits = limits();
        limits.execution = execution;
        let admission = Admission {
            limits: &limits,
            cancellation: &cancellation,
            elapsed: || now.get(),
            deadline: Duration::from_secs(1),
            pid: 42,
        };
        let mut observer = Observer {
            now: &now,
            advance: Duration::ZERO,
            ticks: 0,
        };
        let mut child = Child {
            result: Ok(false),
            calls: 0,
        };
        let decision = Session::new(&mut observer, &admission).monitor(&mut child, &mut Output);
        assert!(matches!(decision, Decision::Terminal(actual) if actual == expected));
        assert_eq!(observer.ticks, 0);
        assert_eq!(child.calls, 0);
    }
}

#[test]
fn probe_candidate_rejects_exited_or_unknown_leader() {
    for exited in [Ok(true), Err(Failure::CleanupDeadline)] {
        let now = Cell::new(Duration::ZERO);
        let cancellation = Cancellation::default();
        let limits = limits();
        let admission = Admission {
            limits: &limits,
            cancellation: &cancellation,
            elapsed: || now.get(),
            deadline: Duration::from_secs(1),
            pid: 42,
        };
        let failed = exited.is_err();
        let mut child = Child {
            result: exited,
            calls: 0,
        };
        let mut observer = Observer {
            now: &now,
            advance: Duration::ZERO,
            ticks: 0,
        };
        let decision = Session::new(&mut observer, &admission).monitor(&mut child, &mut Output);
        if failed {
            assert!(matches!(
                decision,
                Decision::Failed(Failure::CleanupDeadline)
            ));
        } else {
            assert!(matches!(decision, Decision::Terminal(Outcome::EarlyExit)));
        }
        assert_eq!(child.calls, 1);
    }
}
