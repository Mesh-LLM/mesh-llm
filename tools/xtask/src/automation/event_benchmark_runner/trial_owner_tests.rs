use super::*;
use crate::process::retained::Snapshot;
use crate::process::{OutputFiles, ProcessSpec};
use std::time::Instant;

fn launch(member: MemberId) -> Launch {
    Launch {
        member,
        spec: ProcessSpec {
            executable: "/fixture".into(),
            arguments: vec![],
            cwd: std::env::temp_dir(),
            environment: Default::default(),
        },
        files: OutputFiles::default(),
        readiness_deadline: Duration::from_secs(5),
    }
}
fn owner() -> Owner {
    Owner {
        server: Some(launch(MemberId::Seed)),
        worker: Some(launch(MemberId::WorkerOne)),
        worker_policy: ExpectedExit::new(&[0, 1], Duration::from_secs(5)).unwrap(),
        stopping: false,
        setup_ms: None,
        stop_started: None,
        shutdown_ms: None,
        api_readiness: None,
        listener_ready: true,
        host_readiness_timeout: Duration::from_secs(5),
        host_started: None,
        health_streams: [None, None],
    }
}
fn context(members: &[Snapshot], millis: u64) -> Context<'_> {
    Context {
        elapsed: Duration::from_millis(millis),
        remaining: Duration::from_secs(5),
        members,
    }
}

#[test]
fn ownership_admission_precedes_worker_and_http_readiness_is_not_synthesized() {
    let mut owner = owner();
    assert!(matches!(owner.tick(context(&[], 0)), Action::Start(_)));
    assert!(matches!(owner.tick(context(&[], 1)), Action::Pending));
    let mut members = vec![Snapshot {
        member: MemberId::Seed,
        pid: 1,
        started: Instant::now(),
        state: MemberState::Starting,
    }];
    assert!(matches!(
        owner.tick(context(&members, 3)),
        Action::Admit(MemberId::Seed)
    ));
    assert_eq!(owner.setup_ms, Some(3.0));
    members[0].state = MemberState::Ready {
        elapsed: Duration::from_millis(3),
    };
    assert!(matches!(
        owner.tick(context(&members, 4)),
        Action::StartExpected { .. }
    ));
    assert!(matches!(owner.tick(context(&members, 5)), Action::Pending));
}

#[test]
fn failed_worker_still_stops_host_and_records_observed_stop_duration() {
    let mut owner = owner();
    owner.server = None;
    owner.worker = None;
    let members = [Snapshot {
        member: MemberId::WorkerOne,
        pid: 2,
        started: Instant::now(),
        state: MemberState::ExpectedExit {
            status: 1,
            elapsed: Duration::from_millis(20),
        },
    }];
    assert!(matches!(
        owner.tick(context(&members, 25)),
        Action::Stop(MemberId::Seed)
    ));
    assert!(matches!(owner.tick(context(&[], 40)), Action::Complete));
    assert_eq!(owner.shutdown_ms, Some(15.0));
}

#[cfg(unix)]
fn fixture_launch(member: MemberId, script: &str, directory: &std::path::Path) -> Launch {
    use crate::process::Value;
    Launch {
        member,
        spec: ProcessSpec {
            executable: "/bin/sh".into(),
            cwd: directory.into(),
            arguments: vec![Value::Public("-c".into()), Value::Public(script.into())],
            environment: Default::default(),
        },
        files: OutputFiles::default(),
        readiness_deadline: Duration::from_secs(5),
    }
}

#[cfg(unix)]
fn fixture_limits() -> crate::process::Limits {
    use crate::process::{Completion, Limits, Readiness};
    Limits {
        execution: Duration::from_secs(5),
        graceful_shutdown: Duration::from_millis(250),
        forced_shutdown: Duration::from_secs(1),
        retained_bytes_per_stream: 1024,
        readiness: Readiness::None,
        completion: Completion::Exit,
    }
}

#[cfg(unix)]
#[test]
fn retained_failed_worker_reaps_owned_host_and_descendant() {
    let directory = tempfile::tempdir().unwrap();
    let mut owner = owner();
    owner.server = Some(fixture_launch(
        MemberId::Seed,
        "sleep 60 & wait",
        directory.path(),
    ));
    owner.worker = Some(fixture_launch(
        MemberId::WorkerOne,
        "exit 1",
        directory.path(),
    ));
    let report = crate::process::retained::run(
        &mut owner,
        &fixture_limits(),
        &crate::process::Cancellation::default(),
    )
    .unwrap();
    assert!(report.recovery_success());
    assert_eq!(report.members.len(), 2);
    assert!(
        report
            .members
            .iter()
            .all(|member| member.process.cleanup.complete)
    );
    let worker = report
        .members
        .iter()
        .find(|member| member.member == MemberId::WorkerOne)
        .unwrap();
    assert_eq!(
        worker
            .process
            .status
            .as_ref()
            .and_then(std::process::ExitStatus::code),
        Some(1)
    );
}

#[cfg(unix)]
#[test]
fn early_host_exit_is_not_a_successful_trial_and_owned_members_are_cleaned() {
    let directory = tempfile::tempdir().unwrap();
    let mut owner = owner();
    owner.server = Some(fixture_launch(MemberId::Seed, "exit 7", directory.path()));
    owner.worker = Some(fixture_launch(
        MemberId::WorkerOne,
        "sleep 60",
        directory.path(),
    ));
    let report = crate::process::retained::run(
        &mut owner,
        &fixture_limits(),
        &crate::process::Cancellation::default(),
    )
    .unwrap();
    assert!(!report.recovery_success());
    assert!(
        report
            .members
            .iter()
            .all(|member| member.process.cleanup.complete)
    );
}

#[cfg(unix)]
#[test]
fn cancellation_during_live_trial_reaps_both_owned_trees() {
    let directory = tempfile::tempdir().unwrap();
    let mut owner = owner();
    owner.server = Some(fixture_launch(
        MemberId::Seed,
        "sleep 60 & wait",
        directory.path(),
    ));
    owner.worker = Some(fixture_launch(
        MemberId::WorkerOne,
        "sleep 60 & wait",
        directory.path(),
    ));
    let cancellation = crate::process::Cancellation::default();
    let signal = cancellation.clone();
    let cancel_thread = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(100));
        signal.cancel();
    });
    let report =
        crate::process::retained::run(&mut owner, &fixture_limits(), &cancellation).unwrap();
    cancel_thread.join().unwrap();
    assert_eq!(report.outcome, crate::process::Outcome::Cancelled);
    assert!(!report.recovery_success());
    assert!(
        report
            .members
            .iter()
            .all(|member| member.process.cleanup.complete)
    );
}

#[test]
fn cancellation_before_start_refuses_without_consuming_launches() {
    let cancellation = crate::process::Cancellation::default();
    cancellation.cancel();
    let mut owner = owner();
    let limits = crate::process::Limits {
        execution: Duration::from_secs(5),
        graceful_shutdown: Duration::from_secs(1),
        forced_shutdown: Duration::from_secs(1),
        retained_bytes_per_stream: 1024,
        readiness: crate::process::Readiness::None,
        completion: crate::process::Completion::Exit,
    };
    assert!(crate::process::retained::run(&mut owner, &limits, &cancellation).is_err());
    assert!(owner.server.is_some());
    assert!(owner.worker.is_some());
}

fn log(owner: &mut Owner, member: MemberId, bytes: &[u8]) {
    assert!(matches!(
        owner.line(
            member,
            crate::process::ObservedLine {
                stream: crate::process::Stream::Stdout,
                bytes,
                ending: crate::process::LineEnding::Lf,
            }
        ),
        ProbeDecision::Pending
    ));
}

fn gated_owner() -> Owner {
    let mut owner = owner();
    owner.api_readiness = Some(("http://127.0.0.1:12345/v1".into(), 12345));
    owner.listener_ready = false;
    owner.host_readiness_timeout = Duration::from_secs(1);
    owner
}

#[test]
fn malformed_unrelated_wrong_port_and_unowned_readiness_do_not_admit_worker() {
    let mut owner = gated_owner();
    for bytes in [
        b"not-json".as_slice(),
        br#"{"event":"ready","api_url":"http://127.0.0.1:12345/v1","api_port":12346}"#,
        br#"{"event":"api_ready","url":"http://127.0.0.1:12346/v1"}"#,
        br#"{"event":"info","url":"http://127.0.0.1:12345/v1"}"#,
    ] {
        log(&mut owner, MemberId::Seed, bytes);
        assert!(!owner.listener_ready);
    }
    log(
        &mut owner,
        MemberId::WorkerOne,
        br#"{"event":"api_ready","url":"http://127.0.0.1:12345/v1"}"#,
    );
    assert!(!owner.listener_ready);
}

#[test]
fn exact_owned_api_or_runtime_readiness_unblocks_worker() {
    for bytes in [
        br#"{"event":"api_ready","url":"http://127.0.0.1:12345/v1"}"#.as_slice(),
        br#"{"event":"ready","api_url":"http://127.0.0.1:12345/v1","api_port":12345}"#,
    ] {
        let mut owner = gated_owner();
        owner.server = None;
        let members = [Snapshot {
            member: MemberId::Seed,
            pid: 1,
            started: Instant::now(),
            state: MemberState::Ready {
                elapsed: Duration::ZERO,
            },
        }];
        assert!(matches!(owner.tick(context(&members, 0)), Action::Pending));
        log(&mut owner, MemberId::Seed, bytes);
        assert!(owner.listener_ready);
        assert!(matches!(
            owner.tick(context(&members, 1)),
            Action::StartExpected { .. }
        ));
    }
}

#[test]
fn owned_readiness_deadline_rejects_even_after_ownership_admission() {
    let mut owner = gated_owner();
    assert!(matches!(owner.tick(context(&[], 0)), Action::Start(_)));
    let members = [Snapshot {
        member: MemberId::Seed,
        pid: 1,
        started: Instant::now(),
        state: MemberState::Ready {
            elapsed: Duration::ZERO,
        },
    }];
    assert!(matches!(
        owner.tick(context(&members, 1000)),
        Action::Reject(_)
    ));
    assert!(owner.worker.is_some());
}

#[cfg(unix)]
#[test]
fn exact_owned_stub_listener_readiness_allows_native_worker_and_cleanup() {
    let directory = tempfile::tempdir().unwrap();
    let mut owner = gated_owner();
    owner.server = Some(fixture_launch(
        MemberId::Seed,
        "printf '%s\\n' '{\"event\":\"api_ready\",\"url\":\"http://127.0.0.1:12345/v1\"}'; sleep 60 & wait",
        directory.path(),
    ));
    owner.worker = Some(fixture_launch(
        MemberId::WorkerOne,
        "exit 0",
        directory.path(),
    ));
    let report = crate::process::retained::run(
        &mut owner,
        &fixture_limits(),
        &crate::process::Cancellation::default(),
    )
    .unwrap();
    assert!(report.recovery_success());
    assert!(owner.listener_ready);
    assert_eq!(report.members.len(), 2);
}

#[cfg(unix)]
#[test]
fn unrelated_stub_readiness_followed_by_exit_never_starts_worker() {
    let directory = tempfile::tempdir().unwrap();
    let mut owner = gated_owner();
    owner.server = Some(fixture_launch(
        MemberId::Seed,
        "printf '%s\\n' '{\"event\":\"api_ready\",\"url\":\"http://127.0.0.1:9999/v1\"}'; exit 7",
        directory.path(),
    ));
    owner.worker = Some(fixture_launch(
        MemberId::WorkerOne,
        "exit 0",
        directory.path(),
    ));
    let report = crate::process::retained::run(
        &mut owner,
        &fixture_limits(),
        &crate::process::Cancellation::default(),
    )
    .unwrap();
    assert!(!report.recovery_success());
    assert!(!owner.listener_ready);
    assert!(owner.worker.is_some());
    assert!(
        report
            .members
            .iter()
            .all(|member| member.member == MemberId::Seed && member.process.cleanup.complete)
    );
}
