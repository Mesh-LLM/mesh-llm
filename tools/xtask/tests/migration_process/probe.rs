use crate::{process::*, support::*};
use std::time::Duration;

#[derive(Debug, PartialEq, Eq)]
enum Rejection {
    Fixture,
}

#[derive(Default)]
struct Observer {
    pid: Option<u32>,
    stdout: usize,
    stderr: usize,
    ready: bool,
    reject: bool,
    ticks: usize,
}

impl ReadinessProbe for Observer {
    type Rejection = Rejection;
    fn line(&mut self, line: ObservedLine<'_>) -> ProbeDecision<Rejection> {
        match line.stream {
            Stream::Stdout => self.stdout += 1,
            Stream::Stderr => self.stderr += 1,
        }
        self.ready |= line.ending == LineEnding::Lf && line.bytes == b"READY\r";
        ProbeDecision::Pending
    }
    fn tick(&mut self, context: ProbeContext) -> ProbeDecision<Rejection> {
        self.ticks += 1;
        assert_eq!(*self.pid.get_or_insert(context.pid), context.pid);
        assert_eq!(context.elapsed + context.remaining, Duration::from_secs(1));
        // Separate pipes do not promise cross-stream delivery order. Admit only
        // after the complete intended fixture census has reached the observer.
        if self.ready && self.stdout >= 32 && self.stderr >= 33 {
            if self.reject {
                ProbeDecision::Rejected(Rejection::Fixture)
            } else {
                ProbeDecision::Candidate
            }
        } else {
            ProbeDecision::Pending
        }
    }
}

#[test]
fn probe_stateful_tick_admission_and_rejection_clean_only_owned_tree() {
    for reject in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let mut sentinel = Sentinel::new(root.path());
        let mut observer = Observer {
            reject,
            ..Observer::default()
        };
        let mut limits = limits();
        limits.retained_bytes_per_stream = 0;
        limits.graceful_shutdown = Duration::from_secs(1);
        let result = supervise_with_probe(
            &spec(root.path(), "observed-ready"),
            &limits,
            &Cancellation::default(),
            OutputFiles::default(),
            Probe {
                observer: &mut observer,
                deadline: Duration::from_secs(1),
            },
        )
        .unwrap();
        let report = result.process;
        assert_eq!(observer.pid, Some(report.pid));
        assert!(observer.ticks >= 2);
        assert!(observer.stdout >= 32);
        assert_eq!(observer.stderr, 33, "cleanup-only line invoked observer");
        assert!(report.stdout.bytes_retained.is_empty() && report.stderr.bytes_retained.is_empty());
        assert!(report.cleanup.complete && !report.cleanup.forced);
        assert_eq!(report.status.unwrap().code(), Some(0));
        assert!(root.path().join("observed-stop").is_file());
        if reject {
            assert_eq!(report.outcome, Outcome::ObservationRejected);
            assert_eq!(result.rejection, Some(Rejection::Fixture));
            assert!(!report.ready && !report.success());
            assert!(matches!(report.readiness_stop, ReadinessStop::NotAdmitted));
        } else {
            assert_eq!(report.outcome, Outcome::Ready);
            assert!(report.success() && result.rejection.is_none());
            assert!(
                matches!(report.readiness_stop, ReadinessStop::ProbeAdmitted { elapsed, request: GracefulRequest::RequestedAfterLiveObservation } if elapsed < Duration::from_secs(1))
            );
        }
        eprintln!(
            "probe reject={reject} report={report:?} ticks={} stdout={} stderr={}",
            observer.ticks, observer.stdout, observer.stderr
        );
        sentinel.assert_alive();
        assert_stopped(root.path(), &["observed-ready"]);
    }
}

#[test]
fn probe_early_zero_and_nonzero_leader_exit_cannot_use_live_descendants() {
    for (mode, code) in [("tree-exit", 0), ("tree-crash", 23)] {
        let root = tempfile::tempdir().unwrap();
        let mut sentinel = Sentinel::new(root.path());
        let mut observer = Observer::default();
        let report = supervise_with_probe(
            &spec(root.path(), mode),
            &limits(),
            &Cancellation::default(),
            OutputFiles::default(),
            Probe {
                observer: &mut observer,
                deadline: Duration::from_secs(1),
            },
        )
        .unwrap()
        .process;
        assert_eq!(report.outcome, Outcome::EarlyExit);
        assert_eq!(report.status.unwrap().code(), Some(code));
        assert!(report.cleanup.complete && !report.success());
        assert!(matches!(report.readiness_stop, ReadinessStop::NotAdmitted));
        sentinel.assert_alive();
        assert_stopped(root.path(), &[mode, "branch", "leaf"]);
    }
}

#[test]
fn probe_cleanup_only_lines_do_not_admit_or_reenter_observer() {
    let root = tempfile::tempdir().unwrap();
    let mut sentinel = Sentinel::new(root.path());
    let mut observer = Observer::default();
    let report = supervise_with_probe(
        &spec(root.path(), "observed-timeout"),
        &limits(),
        &Cancellation::default(),
        OutputFiles::default(),
        Probe {
            observer: &mut observer,
            deadline: Duration::from_secs(1),
        },
    )
    .unwrap()
    .process;
    assert_eq!(report.outcome, Outcome::ReadinessDeadline);
    assert!(matches!(report.readiness_stop, ReadinessStop::NotAdmitted));
    assert_eq!(observer.stderr, 32);
    assert!(report.stderr.bytes_retained.ends_with(b"READY\n"));
    assert!(report.cleanup.complete && !report.success());
    sentinel.assert_alive();
    assert_stopped(root.path(), &["observed-timeout"]);
}

#[test]
fn probe_setup_failures_and_pre_cancel_never_invoke_observer() {
    for scenario in ["missing", "output", "cancelled", "deadline", "line"] {
        let root = tempfile::tempdir().unwrap();
        let mut observer = Observer::default();
        let mut spec = spec(root.path(), "observed-ready");
        let mut limits = limits();
        let cancellation = Cancellation::default();
        let mut files = OutputFiles::default();
        let mut deadline = Duration::from_secs(1);
        match scenario {
            "missing" => spec.executable = root.path().join("missing"),
            "output" => {
                let path = root.path().join("out");
                std::fs::write(&path, b"preserved").unwrap();
                files.stdout = Some(path);
            }
            "cancelled" => cancellation.cancel(),
            "deadline" => deadline = Duration::ZERO,
            "line" => ready(&mut limits, b"READY"),
            _ => unreachable!(),
        }
        let result = supervise_with_probe(
            &spec,
            &limits,
            &cancellation,
            files,
            Probe {
                observer: &mut observer,
                deadline,
            },
        );
        assert!(result.is_err());
        assert_eq!(observer.ticks, 0);
        assert_eq!(observer.stdout + observer.stderr, 0);
        assert!(!root.path().join("observed-ready.pid").exists());
    }
}
