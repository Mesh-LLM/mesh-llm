use super::*;
use crate::process::{Completion, LineEnding};
use std::cell::Cell;

struct Script<'a> {
    now: &'a Cell<Duration>,
    advance: Duration,
    cancellation: Option<&'a Cancellation>,
    exited: Result<bool, Failure>,
    calls: usize,
}

impl Leader for Script<'_> {
    fn exited(&mut self) -> Result<bool, Failure> {
        self.calls += 1;
        self.now.set(self.advance);
        if let Some(cancellation) = self.cancellation {
            cancellation.cancel();
        }
        std::mem::replace(&mut self.exited, Ok(false))
    }
}

struct Batch<Callback>(Callback);
impl<Callback: FnMut() -> Result<Option<Stream>, Failure>> Lines for Batch<Callback> {
    fn poll_lines(&mut self, _: &Readiness) -> Result<Option<Stream>, Failure> {
        (self.0)()
    }
}

fn limits() -> Limits {
    Limits {
        execution: Duration::from_secs(2),
        graceful_shutdown: Duration::from_millis(10),
        forced_shutdown: Duration::from_millis(10),
        retained_bytes_per_stream: 0,
        readiness: Readiness::ObservedLines {
            deadline: Duration::from_secs(1),
            matcher: |line| line.ending == LineEnding::Lf && line.bytes == b"READY",
        },
        completion: Completion::StopAfterReady,
    }
}

#[test]
fn migration_process_observed_exit_before_admission_rejects_buffered_ready() {
    let now = Cell::new(Duration::ZERO);
    let mut child = Script {
        now: &now,
        advance: Duration::ZERO,
        cancellation: None,
        exited: Ok(true),
        calls: 0,
    };
    let mut batch = Batch(|| Ok(Some(Stream::Stdout)));
    let decision = monitor(
        &mut child,
        &mut batch,
        &Admission {
            limits: &limits(),
            cancellation: &Cancellation::default(),
            elapsed: || now.get(),
        },
    );
    assert!(matches!(decision, Decision::Terminal(Outcome::EarlyExit)));
    assert_eq!(child.calls, 1);
}

#[test]
fn migration_process_observed_leader_query_failure_is_typed_not_live() {
    let now = Cell::new(Duration::ZERO);
    let mut child = Script {
        now: &now,
        advance: Duration::ZERO,
        cancellation: None,
        exited: Err(Failure::io(
            "observe exit",
            std::io::ErrorKind::PermissionDenied.into(),
        )),
        calls: 0,
    };
    let mut batch = Batch(|| Ok(Some(Stream::Stdout)));
    let decision = monitor(
        &mut child,
        &mut batch,
        &Admission {
            limits: &limits(),
            cancellation: &Cancellation::default(),
            elapsed: || now.get(),
        },
    );
    assert!(matches!(
        decision,
        Decision::Failed(Failure::Io {
            operation: "observe exit",
            ..
        })
    ));
}

#[test]
fn migration_process_observed_admission_checks_clock_after_poll_and_live_query() {
    let deadline = Duration::from_secs(1);
    for elapsed in [
        deadline - Duration::from_nanos(1),
        deadline,
        deadline + Duration::from_nanos(1),
    ] {
        for cross_in_query in [false, true] {
            let now = Cell::new(Duration::ZERO);
            let mut child = Script {
                now: &now,
                advance: elapsed,
                cancellation: None,
                exited: Ok(false),
                calls: 0,
            };
            let mut batch = Batch(|| {
                if !cross_in_query {
                    now.set(elapsed);
                }
                Ok(Some(Stream::Stderr))
            });
            let decision = monitor(
                &mut child,
                &mut batch,
                &Admission {
                    limits: &limits(),
                    cancellation: &Cancellation::default(),
                    elapsed: || now.get(),
                },
            );
            if elapsed < deadline {
                assert!(
                    matches!(decision, Decision::Admitted(ReadinessObservation { stream: Stream::Stderr, elapsed: actual }) if actual == elapsed)
                );
            } else {
                assert!(matches!(
                    decision,
                    Decision::Terminal(Outcome::ReadinessDeadline)
                ));
                assert_eq!(child.calls, usize::from(cross_in_query));
            }
        }
    }
}

#[test]
fn migration_process_observed_cancellation_precedes_admission_at_each_boundary() {
    for boundary in 0..3 {
        let now = Cell::new(Duration::ZERO);
        let cancellation = Cancellation::default();
        if boundary == 0 {
            cancellation.cancel();
        }
        let mut child = Script {
            now: &now,
            advance: Duration::ZERO,
            cancellation: (boundary == 2).then_some(&cancellation),
            exited: Ok(false),
            calls: 0,
        };
        let mut polls = 0;
        let mut batch = Batch(|| {
            polls += 1;
            if boundary == 1 {
                cancellation.cancel();
            }
            Ok(Some(Stream::Stdout))
        });
        let decision = monitor(
            &mut child,
            &mut batch,
            &Admission {
                limits: &limits(),
                cancellation: &cancellation,
                elapsed: || now.get(),
            },
        );
        assert!(matches!(decision, Decision::Terminal(Outcome::Cancelled)));
        assert_eq!(polls, usize::from(boundary > 0));
        assert_eq!(child.calls, usize::from(boundary == 2));
    }
}

#[test]
fn migration_process_observed_expired_budget_does_not_poll_cleanup_readiness() {
    let now = Cell::new(Duration::from_secs(1));
    let mut child = Script {
        now: &now,
        advance: Duration::ZERO,
        cancellation: None,
        exited: Ok(false),
        calls: 0,
    };
    let mut batch = Batch(|| panic!("deadline must freeze admission before output"));
    let decision = monitor(
        &mut child,
        &mut batch,
        &Admission {
            limits: &limits(),
            cancellation: &Cancellation::default(),
            elapsed: || now.get(),
        },
    );
    assert!(matches!(
        decision,
        Decision::Terminal(Outcome::ReadinessDeadline)
    ));
    assert_eq!(child.calls, 0);
}

#[test]
fn migration_process_observed_output_failure_is_terminal_before_live_query() {
    let now = Cell::new(Duration::ZERO);
    let mut child = Script {
        now: &now,
        advance: Duration::ZERO,
        cancellation: None,
        exited: Ok(false),
        calls: 0,
    };
    let mut batch = Batch(|| {
        Err(Failure::io(
            "write output",
            std::io::ErrorKind::WriteZero.into(),
        ))
    });
    let decision = monitor(
        &mut child,
        &mut batch,
        &Admission {
            limits: &limits(),
            cancellation: &Cancellation::default(),
            elapsed: || now.get(),
        },
    );
    assert!(matches!(
        decision,
        Decision::Failed(Failure::Io {
            operation: "write output",
            ..
        })
    ));
    assert_eq!(child.calls, 0);
}
