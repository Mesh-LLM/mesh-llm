use super::control::Leader;
use super::observed::Decision;
use super::probe::{Admission, ProbeLines, Session};
use super::*;
use std::cell::Cell;
use std::time::Duration;

#[derive(Clone, Copy)]
enum Boundary {
    Start,
    Tick,
    Line,
    Leader,
}

struct Observer<'a> {
    now: &'a Cell<Duration>,
    cancellation: &'a Cancellation,
    cancel_at: Option<Boundary>,
    advance: Duration,
    reject: bool,
    calls: Vec<&'static str>,
}

impl ReadinessProbe for Observer<'_> {
    type Rejection = u8;

    fn tick(&mut self, context: ProbeContext) -> ProbeDecision<u8> {
        self.calls.push("tick");
        assert_eq!(context.pid, 42);
        assert_eq!(context.remaining, Duration::from_secs(1));
        if matches!(self.cancel_at, Some(Boundary::Tick)) {
            self.cancellation.cancel();
        }
        ProbeDecision::Pending
    }

    fn line(&mut self, line: ObservedLine<'_>) -> ProbeDecision<u8> {
        self.calls.push("line");
        assert_eq!(line.bytes, b"READY");
        self.now.set(self.advance);
        if matches!(self.cancel_at, Some(Boundary::Line)) {
            self.cancellation.cancel();
        }
        if self.reject {
            ProbeDecision::Rejected(23)
        } else {
            ProbeDecision::Candidate
        }
    }
}

struct Child<'a> {
    now: &'a Cell<Duration>,
    advance: Duration,
    cancellation: Option<&'a Cancellation>,
    exited: bool,
    calls: usize,
}

impl Leader for Child<'_> {
    fn exited(&mut self) -> Result<bool, Failure> {
        self.calls += 1;
        self.now.set(self.advance);
        if let Some(cancel) = self.cancellation {
            cancel.cancel();
        }
        Ok(self.exited)
    }
}

struct Output(bool);
impl ProbeLines for Output {
    fn poll_probe(&mut self, callback: &mut dyn FnMut(ObservedLine<'_>)) -> Result<(), Failure> {
        for stream in [Stream::Stdout, Stream::Stderr] {
            callback(ObservedLine {
                stream,
                bytes: b"READY",
                ending: LineEnding::Lf,
            });
        }
        if self.0 {
            Err(Failure::io(
                "write output",
                std::io::ErrorKind::WriteZero.into(),
            ))
        } else {
            Ok(())
        }
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
fn probe_admission_requires_all_observations_strictly_before_deadline() {
    let deadline = Duration::from_secs(1);
    for elapsed in [
        deadline - Duration::from_nanos(1),
        deadline,
        deadline + Duration::from_nanos(1),
    ] {
        for during_live_check in [false, true] {
            let now = Cell::new(Duration::ZERO);
            let cancellation = Cancellation::default();
            let limits = limits();
            let mut observer = Observer {
                now: &now,
                cancellation: &cancellation,
                cancel_at: None,
                advance: if during_live_check {
                    Duration::ZERO
                } else {
                    elapsed
                },
                reject: false,
                calls: Vec::new(),
            };
            let admission = Admission {
                limits: &limits,
                cancellation: &cancellation,
                elapsed: || now.get(),
                deadline,
                pid: 42,
            };
            let mut child = Child {
                now: &now,
                advance: elapsed,
                cancellation: None,
                exited: false,
                calls: 0,
            };
            let decision =
                Session::new(&mut observer, &admission).monitor(&mut child, &mut Output(false));
            if elapsed < deadline {
                assert!(matches!(decision, Decision::Admitted(((), actual)) if actual == elapsed));
            } else {
                assert!(matches!(
                    decision,
                    Decision::Terminal(Outcome::ReadinessDeadline)
                ));
                assert_eq!(child.calls, usize::from(during_live_check));
            }
            assert_eq!(observer.calls, ["tick", "line"]);
        }
    }
}

#[test]
fn probe_cancellation_wins_at_every_admission_boundary() {
    for boundary in [
        Boundary::Start,
        Boundary::Tick,
        Boundary::Line,
        Boundary::Leader,
    ] {
        let now = Cell::new(Duration::ZERO);
        let cancellation = Cancellation::default();
        if matches!(boundary, Boundary::Start) {
            cancellation.cancel();
        }
        let limits = limits();
        let mut observer = Observer {
            now: &now,
            cancellation: &cancellation,
            cancel_at: Some(boundary),
            advance: Duration::ZERO,
            reject: false,
            calls: Vec::new(),
        };
        let admission = Admission {
            limits: &limits,
            cancellation: &cancellation,
            elapsed: || now.get(),
            deadline: Duration::from_secs(1),
            pid: 42,
        };
        let mut child = Child {
            now: &now,
            advance: Duration::ZERO,
            cancellation: matches!(boundary, Boundary::Leader).then_some(&cancellation),
            exited: false,
            calls: 0,
        };
        let decision =
            Session::new(&mut observer, &admission).monitor(&mut child, &mut Output(false));
        assert!(matches!(decision, Decision::Terminal(Outcome::Cancelled)));
        let expected = match boundary {
            Boundary::Start => vec![],
            Boundary::Tick => vec!["tick"],
            Boundary::Line | Boundary::Leader => vec!["tick", "line"],
        };
        assert_eq!(observer.calls, expected);
        assert_eq!(
            child.calls,
            usize::from(matches!(boundary, Boundary::Leader))
        );
    }
}

#[test]
fn probe_rejection_and_output_error_never_admit_or_repeat_callbacks() {
    for output_error in [false, true] {
        let now = Cell::new(Duration::ZERO);
        let cancellation = Cancellation::default();
        let limits = limits();
        let mut observer = Observer {
            now: &now,
            cancellation: &cancellation,
            cancel_at: None,
            advance: Duration::ZERO,
            reject: true,
            calls: Vec::new(),
        };
        let admission = Admission {
            limits: &limits,
            cancellation: &cancellation,
            elapsed: || now.get(),
            deadline: Duration::from_secs(1),
            pid: 42,
        };
        let mut child = Child {
            now: &now,
            advance: Duration::ZERO,
            cancellation: None,
            exited: false,
            calls: 0,
        };
        let mut session = Session::new(&mut observer, &admission);
        let decision = session.monitor(&mut child, &mut Output(output_error));
        assert_eq!(session.rejection, Some(23));
        if output_error {
            assert!(matches!(
                decision,
                Decision::Failed(Failure::Io {
                    operation: "write output",
                    ..
                })
            ));
        } else {
            assert!(matches!(
                decision,
                Decision::Terminal(Outcome::ObservationRejected)
            ));
        }
        assert_eq!(observer.calls, ["tick", "line"]);
        assert_eq!(child.calls, 0);
    }
}
