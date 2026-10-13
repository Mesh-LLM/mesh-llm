use super::{Interrupt, platform};
use crate::process::{retained::*, *};
use std::sync::atomic::{AtomicBool, Ordering};

static UNRELATED: AtomicBool = AtomicBool::new(false);

extern "C" fn unrelated(_: libc::c_int) {
    UNRELATED.store(true, Ordering::SeqCst);
}

#[test]
fn migration_lifecycle_interrupt_scope_restores_and_refuses_other_owners() {
    if std::env::var_os("TASK20_INTERRUPT_SCOPE_PROBE").is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "command_interrupt::tests::migration_lifecycle_interrupt_scope_restores_and_refuses_other_owners",
                "--nocapture",
            ])
            .env("TASK20_INTERRUPT_SCOPE_PROBE", "1")
            .output()
            .unwrap();
        let status = output.status;
        assert!(status.success());
        assert!(
            String::from_utf8_lossy(&output.stdout).contains("1 passed; 0 failed"),
            "{output:?}"
        );
        eprintln!("{}", String::from_utf8_lossy(&output.stdout));
        return;
    }

    let original_int = platform::action(libc::SIGINT).unwrap();
    let original_term = platform::action(libc::SIGTERM).unwrap();
    let mut other = original_term;
    other.sa_sigaction = (unrelated as *const ()).expose_provenance();
    platform::replace(libc::SIGTERM, &other).unwrap();
    assert!(Interrupt::install().is_err());
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().canonicalize().unwrap();
    let arguments = vec![
        "--binary".into(),
        std::env::current_exe().unwrap().to_str().unwrap().into(),
        "--native-runtime-root".into(),
        root.to_str().unwrap().into(),
        "--state-parent".into(),
        root.to_str().unwrap().into(),
    ];
    let error = crate::automation::client_readiness::run(&root, &arguments).unwrap_err();
    assert!(error.to_string().contains("existing signal handler"));
    assert_eq!(std::fs::read_dir(&root).unwrap().count(), 0);
    assert_eq!(
        platform::action(libc::SIGINT).unwrap().sa_sigaction,
        original_int.sa_sigaction
    );
    raise(libc::SIGTERM);
    assert!(UNRELATED.swap(false, Ordering::SeqCst));
    platform::replace(libc::SIGTERM, &original_term).unwrap();

    let scope = Interrupt::install().unwrap();
    assert!(Interrupt::install().is_err());
    assert!(!scope.cancellation().is_cancelled());
    raise(libc::SIGINT);
    raise(libc::SIGTERM);
    assert!(scope.cancellation().is_cancelled());
    assert!(scope.finish().is_err());
    assert_eq!(
        platform::action(libc::SIGINT).unwrap().sa_sigaction,
        original_int.sa_sigaction
    );
    assert_eq!(
        platform::action(libc::SIGTERM).unwrap().sa_sigaction,
        original_term.sa_sigaction
    );

    let scope = Interrupt::install().unwrap();
    assert!(!scope.cancellation().is_cancelled());
    platform::replace(libc::SIGTERM, &other).unwrap();
    scope.finish().unwrap();
    raise(libc::SIGTERM);
    assert!(UNRELATED.load(Ordering::SeqCst));
    platform::replace(libc::SIGTERM, &original_term).unwrap();

    let mut ignored = original_int;
    ignored.sa_sigaction = libc::SIG_IGN;
    platform::replace(libc::SIGINT, &ignored).unwrap();
    let scope = Interrupt::install().unwrap();
    drop(scope);
    assert_eq!(
        platform::action(libc::SIGINT).unwrap().sa_sigaction,
        libc::SIG_IGN
    );
    platform::replace(libc::SIGINT, &original_int).unwrap();
    let mut retained = RetainedInterrupt(false);
    let limits = crate::process::Limits {
        execution: std::time::Duration::from_secs(1),
        graceful_shutdown: std::time::Duration::from_millis(10),
        forced_shutdown: std::time::Duration::from_millis(10),
        retained_bytes_per_stream: 0,
        readiness: crate::process::Readiness::None,
        completion: crate::process::Completion::Exit,
    };
    assert!(matches!(
        crate::automation::retained_session::run(&mut retained, &limits),
        Err(crate::automation::retained_session::Error::Finalization {
            reason: super::Reason::Interrupted,
            preceding: Ok(crate::process::retained::Report {
                outcome: crate::process::Outcome::Cancelled,
                ..
            }),
        })
    ));
    assert!(retained.0);
}

struct RetainedInterrupt(bool);
impl crate::process::retained::Coordinator for RetainedInterrupt {
    type Rejection = ();
    fn line(
        &mut self,
        _: crate::process::retained::MemberId,
        _: crate::process::ObservedLine<'_>,
    ) -> crate::process::ProbeDecision<()> {
        panic!("no child was started")
    }
    fn tick(
        &mut self,
        _: crate::process::retained::Context<'_>,
    ) -> crate::process::retained::Action<()> {
        self.0 = true;
        raise(libc::SIGINT);
        crate::process::retained::Action::Complete
    }
}

fn raise(signal: libc::c_int) {
    // SAFETY: isolated subprocess owns both signal dispositions for this test.
    assert_eq!(unsafe { libc::raise(signal) }, 0);
}

#[test]
fn retained_interrupt_after_admission_preserves_real_member_report() {
    if std::env::var_os("RETAINED_MEMBER_INTERRUPT_PROBE").is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "command_interrupt::tests::retained_interrupt_after_admission_preserves_real_member_report",
                "--nocapture",
            ])
            .env("RETAINED_MEMBER_INTERRUPT_PROBE", "1")
            .output()
            .unwrap();
        assert!(output.status.success(), "{output:?}");
        assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed; 0 failed"));
        return;
    }
    let root = tempfile::tempdir().unwrap();
    let fixture = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("examples/migration_process_driver");
    assert!(fixture.is_file(), "build migration_process_driver first");
    let launch = Launch {
        member: MemberId::Seed,
        spec: ProcessSpec {
            executable: fixture,
            arguments: Vec::new(),
            cwd: root.path().to_path_buf(),
            environment: std::collections::BTreeMap::from([
                (
                    "MIGRATION_PROCESS_MODE".into(),
                    Value::Public("ready-hang".into()),
                ),
                (
                    "MIGRATION_PROCESS_ROOT".into(),
                    Value::Public(root.path().into()),
                ),
                ("MIGRATION_PROCESS_DRIVER".into(), Value::Public("1".into())),
            ]),
        },
        files: OutputFiles::default(),
        readiness_deadline: std::time::Duration::from_secs(2),
    };
    let mut observer = AdmittedInterrupt {
        launch: Some(launch),
        pid: None,
    };
    let limits = Limits {
        execution: std::time::Duration::from_secs(5),
        graceful_shutdown: std::time::Duration::from_secs(1),
        forced_shutdown: std::time::Duration::from_secs(1),
        retained_bytes_per_stream: 1024,
        readiness: Readiness::None,
        completion: Completion::Exit,
    };
    let result = crate::automation::retained_session::run(&mut observer, &limits);
    let Err(crate::automation::retained_session::Error::Finalization {
        reason: super::Reason::Interrupted,
        preceding: Ok(report),
    }) = result
    else {
        panic!("interruption must preserve the real member report");
    };
    assert_eq!(report.outcome, Outcome::Cancelled);
    assert!(!report.success());
    assert!(!report.recovery_success());
    assert_eq!(report.members.len(), 1);
    let member = &report.members[0];
    let pid: u32 = std::fs::read_to_string(root.path().join("ready-hang.pid"))
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(Some(member.process.pid), observer.pid);
    assert_eq!(member.process.pid, pid);
    assert!(member.admitted.is_some());
    assert_eq!(member.process.outcome, Outcome::Cancelled);
    assert_eq!(member.process.status.unwrap().code(), Some(0));
    assert!(member.process.cleanup.complete);
    assert!(!member.process.cleanup.forced);
    assert!(!member.process.cleanup.graceful_signal_failed);
    assert!(member.process.cleanup.failure.is_none());
    assert!(member.process.failure.is_none());
    assert_eq!(member.process.stdout.bytes_retained, b"READY\n");
    assert_eq!(member.process.stdout.bytes_seen, 6);
}

struct AdmittedInterrupt {
    launch: Option<Launch>,
    pid: Option<u32>,
}
impl Coordinator for AdmittedInterrupt {
    type Rejection = ();
    fn line(&mut self, _: MemberId, line: ObservedLine<'_>) -> ProbeDecision<()> {
        if line.bytes == b"READY" && line.ending == LineEnding::Lf {
            ProbeDecision::Candidate
        } else {
            ProbeDecision::Pending
        }
    }
    fn tick(&mut self, context: Context<'_>) -> Action<()> {
        if let Some(launch) = self.launch.take() {
            return Action::Start(launch);
        }
        match context.members[0].state {
            MemberState::Starting => Action::Pending,
            MemberState::Ready { .. } => {
                self.pid = Some(context.members[0].pid);
                raise(libc::SIGINT);
                Action::Pending
            }
            MemberState::IntentionalStop | MemberState::ExpectedExit { .. } => {
                panic!("member must remain live until interruption")
            }
        }
    }
}
