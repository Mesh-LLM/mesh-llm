use super::*;
use crate::process::Cleanup;
use std::os::unix::process::ExitStatusExt;
use std::{process::Command, thread, time::Instant};

#[test]
fn status_preserves_timeout_cleanup_signals_and_child_failure() {
    let mut report = InheritedReport {
        pid: 1,
        outcome: Outcome::Exited,
        status: Some(std::process::ExitStatus::from_raw(7 << 8)),
        elapsed: Duration::ZERO,
        cleanup: Cleanup {
            complete: true,
            ..Cleanup::default()
        },
        failure: None,
    };
    assert_eq!(status(&report, None), 7);
    report.status = Some(std::process::ExitStatus::from_raw(libc::SIGSEGV));
    assert_eq!(status(&report, None), 128 + libc::SIGSEGV);
    report.outcome = Outcome::Deadline;
    assert_eq!(status(&report, None), 124);
    assert_eq!(status(&report, Some(libc::SIGTERM)), 143);
    assert_eq!(status(&report, Some(libc::SIGINT)), 130);
    report.cleanup.complete = false;
    assert_eq!(status(&report, Some(libc::SIGTERM)), 125);
}

#[test]
fn actual_interrupts_return_exact_status_after_cleanup() {
    const CHILD: &str = "MESH_CANARY_TIMEOUT_SIGNAL_FIXTURE";
    if let Some(root) = std::env::var_os(CHILD) {
        let root = PathBuf::from(root);
        let report = execute(&Input {
            label: "signal fixture".into(),
            seconds: 8,
            cwd: root.clone(),
            executable: "/bin/sh".into(),
            arguments: vec![
                "-c".into(),
                "echo $$ > leader; trap '' TERM; /bin/sleep 30 & echo $! > descendant; wait".into(),
            ],
        });
        fs::write(root.join("report"), report.code.to_string()).unwrap();
        assert!(matches!(report.code, 130 | 143));
        return;
    }
    for signal in [libc::SIGINT, libc::SIGTERM] {
        let directory = tempfile::tempdir().unwrap();
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "automation::canary_timeout::tests::actual_interrupts_return_exact_status_after_cleanup", "--nocapture"])
            .env(CHILD,directory.path()).spawn().unwrap();
        let until = Instant::now() + Duration::from_secs(6);
        while !directory.path().join("descendant").is_file() && Instant::now() < until {
            thread::sleep(Duration::from_millis(10));
        }
        if !directory.path().join("descendant").is_file() {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!("fixture failed to become ready");
        }
        // Only the exact still-owned harness PID receives the operator signal.
        assert_eq!(
            unsafe { libc::kill(i32::try_from(child.id()).unwrap(), signal) },
            0
        );
        assert!(child.wait().unwrap().success());
        assert_eq!(
            fs::read_to_string(directory.path().join("report")).unwrap(),
            (128 + signal).to_string()
        );
    }
}
