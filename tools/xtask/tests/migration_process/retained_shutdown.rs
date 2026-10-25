use super::*;
use std::path::{Path, PathBuf};
use std::time::Instant;

struct BarrierObserver {
    inner: Observer,
    survivor: PathBuf,
    window: bool,
    frozen: bool,
    complete: bool,
}

impl Coordinator for BarrierObserver {
    type Rejection = ();
    fn line(&mut self, member: MemberId, line: ObservedLine<'_>) -> ProbeDecision<()> {
        assert!(!self.frozen, "mutation callback after completion");
        if line.bytes == b"SURVIVOR_WINDOW" {
            self.window = true;
            std::fs::write(self.survivor.join("window-observed"), b"observed").unwrap();
        }
        self.inner.line(member, line)
    }
    fn tick(&mut self, context: Context<'_>) -> Action<()> {
        assert!(!self.frozen, "tick after completion");
        if self.complete
            && self.inner.launches.is_empty()
            && context.members.len() == 2
            && context
                .members
                .iter()
                .all(|member| matches!(member.state, MemberState::Ready { .. }))
        {
            self.frozen = true;
            return Action::Complete;
        }
        self.inner.tick(context)
    }
}

fn barrier_launch(root: &Path, member: MemberId) -> Launch {
    let mut launch = launch(root, member, "retained-crash");
    launch
        .spec
        .environment
        .insert("RETAINED_STOP_BARRIER".into(), Value::Public("1".into()));
    launch
}

fn wait_file(path: &Path) {
    let until = Instant::now() + Duration::from_secs(4);
    while !path.is_file() {
        assert!(Instant::now() < until, "barrier was not reached: {path:?}");
        std::thread::park_timeout(Duration::from_millis(1));
    }
}

#[test]
fn survivor_exit_is_observed_during_target_graceful_stop() {
    let target = tempfile::tempdir().unwrap();
    let survivor = tempfile::tempdir().unwrap();
    let mut live = launch(survivor.path(), MemberId::WorkerOne, "retained-crash");
    live.spec.environment.insert(
        "RETAINED_SURVIVOR_BARRIER".into(),
        Value::Public(target.path().join("stop-entered").into()),
    );
    let mut inner = observer(vec![barrier_launch(target.path(), MemberId::Seed), live]);
    inner.stop = Some(MemberId::Seed);
    let mut observer = BarrierObserver {
        inner,
        survivor: survivor.path().into(),
        window: false,
        frozen: false,
        complete: false,
    };
    let mut limits = limits();
    limits.graceful_shutdown = Duration::from_secs(2);
    let report = run(&mut observer, &limits, &Cancellation::default()).unwrap();
    assert!(
        observer.window,
        "survivor output must be observed before target stop finishes"
    );
    assert_eq!(report.outcome, Outcome::EarlyExit);
    assert_eq!(report.members[1].process.status.unwrap().code(), Some(23));
    assert!(
        report
            .members
            .iter()
            .all(|member| member.process.cleanup.complete)
    );
    assert!(
        report.members[0]
            .process
            .stdout
            .bytes_retained
            .windows(11)
            .any(|bytes| bytes == b"TARGET_STOP")
    );
}

fn cancellation_during_stop(complete: bool) {
    let target = tempfile::tempdir().unwrap();
    let survivor = tempfile::tempdir().unwrap();
    let cancel = Cancellation::default();
    let worker_cancel = cancel.clone();
    let barrier = target.path().join("stop-entered");
    let release = target.path().join("stop-release");
    let mut inner = observer(vec![
        barrier_launch(target.path(), MemberId::Seed),
        launch(survivor.path(), MemberId::WorkerOne, "retained-crash"),
    ]);
    inner.stop = Some(MemberId::Seed);
    let mut observer = BarrierObserver {
        inner,
        survivor: survivor.path().into(),
        window: false,
        frozen: false,
        complete,
    };
    std::thread::scope(|scope| {
        let worker = scope.spawn(move || {
            wait_file(&barrier);
            worker_cancel.cancel();
            std::fs::write(release, b"release").unwrap();
        });
        let report = run(&mut observer, &limits(), &cancel).unwrap();
        worker.join().unwrap();
        assert_eq!(report.outcome, Outcome::Cancelled);
        assert!(!report.success());
        assert!(
            report
                .members
                .iter()
                .all(|member| member.process.cleanup.complete)
        );
    });
}

#[test]
fn cancellation_is_observed_during_intentional_stop() {
    cancellation_during_stop(false);
}

#[test]
fn cancellation_during_final_cleanup_cannot_become_success() {
    cancellation_during_stop(true);
}

#[test]
fn graceful_final_cleanup_preserves_success_without_mutation_callbacks() {
    let target = tempfile::tempdir().unwrap();
    let survivor = tempfile::tempdir().unwrap();
    let barrier = target.path().join("stop-entered");
    let release = target.path().join("stop-release");
    let inner = observer(vec![
        barrier_launch(target.path(), MemberId::Seed),
        launch(survivor.path(), MemberId::WorkerOne, "retained-crash"),
    ]);
    let mut observer = BarrierObserver {
        inner,
        survivor: survivor.path().into(),
        window: false,
        frozen: false,
        complete: true,
    };
    std::thread::scope(|scope| {
        let worker = scope.spawn(move || {
            wait_file(&barrier);
            std::fs::write(release, b"release").unwrap();
        });
        let mut limits = limits();
        limits.graceful_shutdown = Duration::from_secs(2);
        let report = run(&mut observer, &limits, &Cancellation::default()).unwrap();
        worker.join().unwrap();
        assert!(report.success());
    });
}

fn deadline_during_stop(complete: bool) {
    let target = tempfile::tempdir().unwrap();
    let survivor = tempfile::tempdir().unwrap();
    let mut target_launch = barrier_launch(target.path(), MemberId::Seed);
    target_launch.readiness_deadline = Duration::from_millis(900);
    let mut live = launch(survivor.path(), MemberId::WorkerOne, "retained-crash");
    live.readiness_deadline = Duration::from_millis(900);
    let mut inner = observer(vec![target_launch, live]);
    inner.stop = Some(MemberId::Seed);
    let mut observer = BarrierObserver {
        inner,
        survivor: survivor.path().into(),
        window: false,
        frozen: false,
        complete,
    };
    let mut limits = limits();
    limits.execution = Duration::from_secs(1);
    limits.graceful_shutdown = Duration::from_secs(2);
    let report = run(&mut observer, &limits, &Cancellation::default()).unwrap();
    assert!(target.path().join("stop-entered").is_file());
    assert_eq!(report.outcome, Outcome::Deadline);
    assert!(!report.success());
    assert!(
        report
            .members
            .iter()
            .all(|member| member.process.cleanup.complete)
    );
}

#[test]
fn deadline_crossing_is_observed_during_intentional_stop() {
    deadline_during_stop(false);
}

#[test]
fn deadline_during_final_cleanup_cannot_become_success() {
    deadline_during_stop(true);
}

#[test]
fn survivor_exit_during_final_cleanup_cannot_become_success() {
    let target = tempfile::tempdir().unwrap();
    let survivor = tempfile::tempdir().unwrap();
    let barrier = target.path().join("stop-entered");
    let crash = survivor.path().join("crash-now");
    let inner = observer(vec![
        barrier_launch(target.path(), MemberId::Seed),
        launch(survivor.path(), MemberId::WorkerOne, "retained-crash"),
    ]);
    let mut observer = BarrierObserver {
        inner,
        survivor: survivor.path().into(),
        window: false,
        frozen: false,
        complete: true,
    };
    std::thread::scope(|scope| {
        let worker = scope.spawn(move || {
            wait_file(&barrier);
            std::fs::write(crash, b"crash").unwrap();
        });
        let mut limits = limits();
        limits.graceful_shutdown = Duration::from_secs(2);
        let report = run(&mut observer, &limits, &Cancellation::default()).unwrap();
        worker.join().unwrap();
        assert_eq!(report.outcome, Outcome::EarlyExit);
        assert_eq!(report.members[1].process.status.unwrap().code(), Some(23));
        assert!(!report.success());
        assert!(
            report
                .members
                .iter()
                .all(|member| member.process.cleanup.complete)
        );
    });
}
