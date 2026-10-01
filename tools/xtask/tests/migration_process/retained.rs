use crate::{
    process::{retained::*, *},
    support::*,
};
use std::collections::VecDeque;
use std::sync::mpsc::{SyncSender, sync_channel};
use std::time::Duration;
#[path = "retained_completion.rs"]
mod completion_tests;
#[path = "retained_generations.rs"]
mod generation_tests;
#[path = "retained_shutdown.rs"]
mod shutdown_tests;
struct Observer {
    launches: VecDeque<Launch>,
    stop: Option<MemberId>,
    stopped: bool,
    complete: bool,
    cancellation: Option<Cancellation>,
    ready_notice: Option<SyncSender<()>>,
    retained_ticks: usize,
    after_stop: Option<Launch>,
}
impl Coordinator for Observer {
    type Rejection = ();
    fn line(&mut self, _: MemberId, line: ObservedLine<'_>) -> ProbeDecision<()> {
        if line.ending == LineEnding::Lf
            && [b"READY".as_slice(), b"TREE_READY"]
                .contains(&line.bytes.strip_suffix(b"\r").unwrap_or(line.bytes))
        {
            ProbeDecision::Candidate
        } else {
            ProbeDecision::Pending
        }
    }
    fn tick(&mut self, context: Context<'_>) -> Action<()> {
        if self.stopped
            && let Some(launch) = self.after_stop.take()
        {
            return Action::Start(launch);
        }
        if context
            .members
            .iter()
            .any(|member| matches!(member.state, MemberState::Starting))
        {
            return Action::Pending;
        }
        if let Some(launch) = self.launches.pop_front() {
            return Action::Start(launch);
        }
        self.retained_ticks += 1;
        if let Some(notice) = self.ready_notice.take() {
            notice.try_send(()).unwrap();
        }
        if let Some(cancel) = self.cancellation.take() {
            cancel.cancel();
        }
        if !self.stopped
            && let Some(member) = self.stop
        {
            self.stopped = true;
            return Action::Stop(member);
        }
        if self.complete && self.retained_ticks >= 3 {
            Action::Complete
        } else {
            Action::Pending
        }
    }
}
fn observer(launches: Vec<Launch>) -> Observer {
    Observer {
        launches: launches.into(),
        stop: None,
        stopped: false,
        complete: false,
        cancellation: None,
        ready_notice: None,
        retained_ticks: 0,
        after_stop: None,
    }
}
fn launch(root: &std::path::Path, member: MemberId, mode: &str) -> Launch {
    let mut spec = spec(root, mode);
    let example = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("examples")
        .join(format!(
            "migration_process_driver{}",
            std::env::consts::EXE_SUFFIX
        ));
    spec.executable = example;
    spec.arguments.clear();
    spec.environment
        .insert("MIGRATION_PROCESS_DRIVER".into(), Value::Public("1".into()));
    Launch {
        member,
        spec,
        files: OutputFiles::default(),
        readiness_deadline: Duration::from_secs(2),
    }
}
#[test]
fn retained_seed_survives_worker_intentional_stop() {
    let roots: Vec<_> = (0..3).map(|_| tempfile::tempdir().unwrap()).collect();
    let mut sentinel = Sentinel::new(roots[0].path());
    let mut observer = observer(
        roots
            .iter()
            .zip([MemberId::Seed, MemberId::WorkerOne, MemberId::WorkerTwo])
            .map(|(root, member)| launch(root.path(), member, "ready-hang"))
            .collect(),
    );
    observer.stop = Some(MemberId::WorkerTwo);
    observer.complete = true;
    let report = run(&mut observer, &limits(), &Cancellation::default()).unwrap();
    assert!(report.success());
    assert!(observer.retained_ticks >= 3);
    assert!(matches!(
        report.members[2].disposition,
        Disposition::IntentionalStop
    ));
    for root in &roots {
        assert_stopped(root.path(), &["ready-hang"]);
    }
    sentinel.assert_alive();
}
#[test]
fn retained_unexpected_ready_member_crash_fails_session() {
    let root = tempfile::tempdir().unwrap();
    let (notice, ready) = sync_channel(1);
    let crash = root.path().join("crash-now");
    let worker = std::thread::spawn(move || {
        ready.recv_timeout(Duration::from_secs(3)).unwrap();
        std::fs::write(crash, b"crash").unwrap();
    });
    let mut observer = observer(vec![launch(root.path(), MemberId::Seed, "retained-crash")]);
    observer.ready_notice = Some(notice);
    let report = run(&mut observer, &limits(), &Cancellation::default()).unwrap();
    worker.join().unwrap();
    assert_eq!(report.outcome, Outcome::EarlyExit);
    assert!(report.members[0].admitted.is_some());
    assert_eq!(report.members[0].process.status.unwrap().code(), Some(23));
    assert!(!report.success());
    assert_stopped(root.path(), &["retained-crash"]);
}
#[test]
fn retained_execution_deadline_continues_after_admission() {
    let root = tempfile::tempdir().unwrap();
    let mut observer = observer(vec![launch(root.path(), MemberId::Seed, "ready-hang")]);
    let mut limits = limits();
    limits.execution = Duration::from_secs(3);
    let report = run(&mut observer, &limits, &Cancellation::default()).unwrap();
    assert_eq!(report.outcome, Outcome::Deadline);
    assert!(report.members[0].admitted.is_some());
    assert!(report.members[0].process.cleanup.complete);
    assert!(!report.success());
    assert_stopped(root.path(), &["ready-hang"]);
}
#[test]
fn retained_cancel_cleans_descendants_and_preserves_sentinel() {
    let root = tempfile::tempdir().unwrap();
    let mut sentinel = Sentinel::new(root.path());
    let cancel = Cancellation::default();
    let mut observer = observer(vec![launch(root.path(), MemberId::Seed, "tree-hang")]);
    observer.cancellation = Some(cancel.clone());
    let report = run(&mut observer, &limits(), &cancel).unwrap();
    assert_eq!(report.outcome, Outcome::Cancelled);
    assert!(report.members[0].admitted.is_some());
    assert!(report.members[0].process.cleanup.complete);
    assert_stopped(root.path(), &["tree-hang", "branch", "leaf"]);
    sentinel.assert_alive();
}
#[test]
fn retained_intentional_stop_forces_stubborn_descendants_without_failure() {
    let root = tempfile::tempdir().unwrap();
    let mut sentinel = Sentinel::new(root.path());
    let mut observer = observer(vec![launch(root.path(), MemberId::WorkerOne, "tree-hang")]);
    observer.stop = Some(MemberId::WorkerOne);
    observer.complete = true;
    let report = run(&mut observer, &limits(), &Cancellation::default()).unwrap();
    assert!(report.recovery_success());
    assert!(!report.success());
    assert!(report.members[0].process.cleanup.forced);
    assert!(matches!(
        report.members[0].disposition,
        Disposition::IntentionalStop
    ));
    assert_stopped(root.path(), &["tree-hang", "branch", "leaf"]);
    sentinel.assert_alive();
}
#[test]
fn retained_readiness_deadline_cannot_use_cleanup_only_marker() {
    let root = tempfile::tempdir().unwrap();
    let mut member = launch(root.path(), MemberId::Seed, "observed-timeout");
    member.readiness_deadline = Duration::from_millis(500);
    let mut observer = observer(vec![member]);
    let report = run(&mut observer, &limits(), &Cancellation::default()).unwrap();
    assert_eq!(report.outcome, Outcome::ReadinessDeadline);
    assert!(report.members[0].admitted.is_none());
    assert!(report.members[0].process.cleanup.complete);
    assert!(!report.success());
}
#[test]
fn retained_duplicate_member_cannot_spawn_twice() {
    let root = tempfile::tempdir().unwrap();
    let mut observer = observer(vec![
        launch(root.path(), MemberId::Seed, "ready-hang"),
        launch(root.path(), MemberId::Seed, "ready-hang"),
    ]);
    let report = run(&mut observer, &limits(), &Cancellation::default()).unwrap();
    assert_eq!(report.outcome, Outcome::IoFailure);
    assert!(matches!(report.failure, Some(Failure::InvalidSpec(_))));
    assert_eq!(report.members.len(), 1);
    assert_stopped(root.path(), &["ready-hang"]);
}

#[test]
fn retained_failed_second_spawn_cleans_previous_member() {
    let root = tempfile::tempdir().unwrap();
    let mut broken = launch(root.path(), MemberId::WorkerOne, "retained-crash");
    broken.spec.executable = root.path().join("missing-executable");
    let mut observer = observer(vec![
        launch(root.path(), MemberId::Seed, "ready-hang"),
        broken,
    ]);
    let report = run(&mut observer, &limits(), &Cancellation::default()).unwrap();
    assert_eq!(report.outcome, Outcome::IoFailure);
    assert!(report.failure.is_some());
    assert_eq!(report.members.len(), 1);
    assert!(report.members[0].process.cleanup.complete);
    assert_stopped(root.path(), &["ready-hang"]);
}
