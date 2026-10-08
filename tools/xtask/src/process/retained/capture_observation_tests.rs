//! Retained callbacks classify only fixed test markers and never retain raw lines.
use super::*;
use crate::process::{Completion, Limits, Readiness, Value};
use std::{collections::BTreeMap, time::Duration};
#[derive(Clone, Copy, Debug)]
enum End {
    Expected,
    Stop,
    Cleanup,
    Reject,
}
struct Counter {
    end: End,
    start: Option<Launch>,
    early: usize,
    tail: usize,
    stopped: bool,
}
impl Coordinator for Counter {
    type Rejection = &'static str;
    fn captured_line(&mut self, _: MemberId, line: ObservedLine<'_>) {
        if line.bytes == b"early" {
            self.early += 1;
        }
        if line.bytes == b"tail" {
            self.tail += 1;
        }
    }
    fn line(&mut self, _: MemberId, _: ObservedLine<'_>) -> ProbeDecision<Self::Rejection> {
        ProbeDecision::Pending
    }
    fn tick(&mut self, context: Context<'_>) -> Action<Self::Rejection> {
        if let Some(launch) = self.start.take() {
            return if matches!(self.end, End::Expected) {
                Action::StartExpected {
                    launch,
                    policy: ExpectedExit::new(&[0], Duration::from_secs(3)).unwrap(),
                }
            } else {
                Action::Start(launch)
            };
        }
        if self.stopped {
            return Action::Complete;
        }
        if let Some(member) = context.members.first() {
            if matches!(self.end, End::Expected) {
                return if matches!(member.state, MemberState::ExpectedExit { .. }) {
                    Action::Complete
                } else {
                    Action::Pending
                };
            }
            if matches!(member.state, MemberState::Starting) {
                return Action::Admit(member.member);
            }
            if matches!(member.state, MemberState::Ready { .. }) && self.early == 1 {
                return match self.end {
                    End::Stop => {
                        self.stopped = true;
                        Action::Stop(member.member)
                    }
                    End::Cleanup => Action::Complete,
                    End::Reject => Action::Reject("fixed fixture rejection"),
                    End::Expected => unreachable!(),
                };
            }
        }
        Action::Pending
    }
}
#[test]
fn retained_observer_receives_each_line_once_in_expected_stop_cleanup_and_rejection_paths() {
    for end in [End::Expected, End::Stop, End::Cleanup, End::Reject] {
        let directory = tempfile::tempdir().unwrap();
        let script = if matches!(end, End::Expected) {
            "printf 'early\\n'; printf tail"
        } else {
            "trap 'printf tail; exit 0' TERM; printf 'early\\n'; while :; do /bin/sleep 0.01; done"
        };
        let mut counter = Counter {
            end,
            start: Some(Launch {
                member: MemberId::Seed,
                spec: ProcessSpec {
                    executable: "/bin/sh".into(),
                    cwd: directory.path().into(),
                    arguments: vec![Value::Public("-c".into()), Value::Public(script.into())],
                    environment: BTreeMap::new(),
                },
                files: OutputFiles::default(),
                readiness_deadline: Duration::from_secs(3),
            }),
            early: 0,
            tail: 0,
            stopped: false,
        };
        let limits = Limits {
            execution: Duration::from_secs(4),
            graceful_shutdown: Duration::from_millis(250),
            forced_shutdown: Duration::from_millis(250),
            retained_bytes_per_stream: 128,
            readiness: Readiness::None,
            completion: Completion::Exit,
        };
        let report = run(
            &mut counter,
            &limits,
            &crate::process::Cancellation::default(),
        )
        .unwrap();
        assert_eq!(
            (counter.early, counter.tail),
            (1, 1),
            "{end:?}: {:?}",
            report.members
        );
        assert!(report.members.iter().all(|member| {
            member.process.cleanup.complete
                && !member.process.cleanup.forced
                && member.process.failure.is_none()
                && [&member.process.stdout, &member.process.stderr]
                    .iter()
                    .all(|stream| stream.line_capture_complete)
        }));
        if matches!(end, End::Reject) {
            assert_eq!(report.rejection, Some("fixed fixture rejection"));
        } else {
            assert!(report.recovery_success());
        }
        assert!(
            report
                .members
                .iter()
                .all(|member| member.process.stdout.suppressed_lines == 0
                    && member.process.stderr.suppressed_lines == 0)
        );
        directory.close().unwrap();
    }
}

#[derive(Clone, Copy, Debug)]
enum PairEnd {
    Stop,
    Cancel,
    Deadline,
}
struct PairCounter {
    launches: std::collections::VecDeque<Launch>,
    end: PairEnd,
    cancellation: crate::process::Cancellation,
    ready: [bool; 2],
    health: [usize; 2],
    stopped: bool,
}
impl PairCounter {
    fn index(member: MemberId) -> usize {
        usize::from(member == MemberId::WorkerOne)
    }
}
impl Coordinator for PairCounter {
    type Rejection = &'static str;
    fn captured_line(&mut self, member: MemberId, line: ObservedLine<'_>) {
        let index = Self::index(member);
        if line.bytes == b"ready" {
            self.ready[index] = true;
        }
        // Typed closed projection only: secret values and raw lines are never retained.
        if let Ok(value) = serde_json::from_slice::<serde_json::Value>(line.bytes)
            && value.get("context").and_then(|v| v.as_str()) == Some("event_system_health")
            && value.get("message").and_then(|v| v.as_str())
                == Some("dropped_progress=7 secret=private")
        {
            self.health[index] += 1;
        }
    }
    fn line(&mut self, _: MemberId, _: ObservedLine<'_>) -> ProbeDecision<Self::Rejection> {
        ProbeDecision::Pending
    }
    fn tick(&mut self, context: Context<'_>) -> Action<Self::Rejection> {
        if let Some(launch) = self.launches.pop_front() {
            return Action::Start(launch);
        }
        if let Some(member) = context
            .members
            .iter()
            .find(|m| matches!(m.state, MemberState::Starting))
        {
            return Action::Admit(member.member);
        }
        if self.ready == [true, true] && !self.stopped {
            self.stopped = true;
            match self.end {
                PairEnd::Stop => return Action::Stop(MemberId::Seed),
                PairEnd::Cancel => self.cancellation.cancel(),
                PairEnd::Deadline => {}
            }
        }
        if matches!(self.end, PairEnd::Stop) && self.stopped && self.health == [1, 1] {
            Action::Complete
        } else {
            Action::Pending
        }
    }
}
fn pair_fixture(end: PairEnd) {
    let execution = if matches!(end, PairEnd::Deadline) {
        Duration::from_millis(400)
    } else {
        Duration::from_secs(3)
    };
    let directory = tempfile::tempdir().unwrap();
    let health =
        r#"{"context":"event_system_health","message":"dropped_progress=7 secret=private"}"#;
    let seed = format!(
        "health='{health}'; trap 'printf \"%s\\n\" \"$health\" >&2; : > stopped; while [ ! -f acknowledged ]; do /bin/sleep 0.01; done; exit 0' TERM; printf 'ready\\n'; while :; do /bin/sleep 0.01; done"
    );
    let survivor = format!(
        "health='{health}'; trap 'if [ ! -f acknowledged ]; then printf \"%s\\n\" \"$health\" >&2; : > acknowledged; fi; exit 0' TERM; printf 'ready\\n'; while [ ! -f stopped ]; do /bin/sleep 0.01; done; printf '%s\\n' \"$health\" >&2; : > acknowledged; while :; do /bin/sleep 0.01; done"
    );
    let launches = [(MemberId::Seed, seed), (MemberId::WorkerOne, survivor)]
        .into_iter()
        .map(|(member, script)| Launch {
            member,
            spec: ProcessSpec {
                executable: "/bin/sh".into(),
                cwd: directory.path().into(),
                arguments: vec![Value::Public("-c".into()), Value::Public(script.into())],
                environment: BTreeMap::new(),
            },
            files: OutputFiles {
                stdout: None,
                stderr: Some(
                    directory
                        .path()
                        .join(format!("{}.stderr.log", PairCounter::index(member))),
                ),
            },
            readiness_deadline: execution,
        })
        .collect();
    let cancellation = crate::process::Cancellation::default();
    let mut counter = PairCounter {
        launches,
        end,
        cancellation: cancellation.clone(),
        ready: [false; 2],
        health: [0; 2],
        stopped: false,
    };
    let limits = Limits {
        execution,
        graceful_shutdown: Duration::from_millis(500),
        forced_shutdown: Duration::from_millis(250),
        retained_bytes_per_stream: 4096,
        readiness: Readiness::None,
        completion: Completion::Exit,
    };
    let report = run(&mut counter, &limits, &cancellation).unwrap();
    assert!(report.failure.is_none(), "{end:?}: {:?}", report.failure);
    assert_eq!(counter.health, [1, 1], "{end:?}: {:?}", report.members);
    assert_eq!(report.members.len(), 2);
    assert_eq!(
        report.outcome,
        match end {
            PairEnd::Stop => crate::process::Outcome::Ready,
            PairEnd::Cancel => crate::process::Outcome::Cancelled,
            PairEnd::Deadline => crate::process::Outcome::Deadline,
        }
    );
    for member in &report.members {
        assert!(member.process.failure.is_none());
        assert!(
            member.process.cleanup.complete
                && !member.process.cleanup.forced
                && member.process.cleanup.failure.is_none()
                && !member.process.cleanup.graceful_signal_failed
        );
        assert!(
            member.process.stdout.line_capture_complete
                && member.process.stderr.line_capture_complete
        );
        assert_eq!(member.process.stderr.suppressed_lines, 1);
        let persisted = std::fs::read_to_string(
            directory
                .path()
                .join(format!("{}.stderr.log", PairCounter::index(member.member))),
        )
        .unwrap();
        assert!(!persisted.contains("private") && !persisted.contains("dropped_progress"));
    }
    directory.close().unwrap();
}
#[test]
fn typed_survivor_tail_is_seen_once_while_another_member_stops() {
    pair_fixture(PairEnd::Stop);
}
#[test]
fn cancellation_and_deadline_cleanup_classify_typed_shutdown_tails_once() {
    for end in [PairEnd::Cancel, PairEnd::Deadline] {
        pair_fixture(end);
    }
}
