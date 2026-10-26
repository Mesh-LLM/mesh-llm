use crate::{process::*, support::*};
#[cfg(target_os = "macos")]
use std::sync::{Arc, Barrier, mpsc};
#[cfg(target_os = "macos")]
use std::time::Duration;

#[test]
fn migration_process_buffered_eof_ready_exit() {
    let root = tempfile::tempdir().unwrap();
    let mut limits = limits();
    ready(&mut limits, b"READY");
    limits.retained_bytes_per_stream = 65536;
    let report = supervise(
        &spec(root.path(), "buffered-eof"),
        &limits,
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap();
    assert_stopped(root.path(), &["buffered-eof"]);
    assert!(
        report.success(),
        "outcome={:?} ready={} status={:?} cleanup={:?} failure={:?} seen={}",
        report.outcome,
        report.ready,
        report.status,
        report.cleanup,
        report.failure,
        report.stdout.bytes_seen
    );
    assert!(report.ready);
    assert!(report.stdout.bytes_retained.ends_with(b"READY"));
}

#[test]
#[cfg(target_os = "macos")]
fn migration_process_concurrent_captures_exclude_unrelated_held_children() {
    let sentinel_root = tempfile::tempdir().unwrap();
    let mut sentinel = Sentinel::new(sentinel_root.path());
    for batch in 0..64 {
        let gate = Arc::new(Barrier::new(12));
        let (sender, receiver) = mpsc::sync_channel(4);
        let roots: Vec<_> = (0..12).map(|_| tempfile::tempdir().unwrap()).collect();
        let fast = std::thread::scope(|scope| {
            let mut workers = Vec::new();
            for (index, root) in roots.iter().enumerate() {
                let gate = Arc::clone(&gate);
                let sender = sender.clone();
                workers.push(scope.spawn(move || {
                    let mode = if index < 4 {
                        "capture-fast"
                    } else {
                        "capture-held"
                    };
                    let spec = spec(root.path(), mode);
                    gate.wait();
                    if index < 8 {
                        let mut limits = limits();
                        limits.forced_shutdown = Duration::from_millis(400);
                        let report = supervise(
                            &spec,
                            &limits,
                            &Cancellation::default(),
                            OutputFiles::default(),
                        )
                        .unwrap();
                        if index < 4 {
                            sender.send(report).unwrap();
                        } else {
                            assert!(report.success(), "held report: {report:?}");
                        }
                    } else {
                        let mut command = std::process::Command::new(&spec.executable);
                        command
                            .args(["--exact", "migration_process_fixture", "--nocapture"])
                            .env("MIGRATION_PROCESS_MODE", mode)
                            .env("MIGRATION_PROCESS_ROOT", root.path())
                            .stdin(std::process::Stdio::null())
                            .stdout(std::process::Stdio::null())
                            .stderr(std::process::Stdio::null());
                        let mut child = command.spawn().unwrap();
                        assert!(child.wait().unwrap().success());
                    }
                }));
            }
            let reports: Vec<_> = (0..4)
                .map(|_| receiver.recv_timeout(Duration::from_secs(4)).unwrap())
                .collect();
            for root in &roots[4..] {
                let until = std::time::Instant::now() + Duration::from_secs(2);
                while !root.path().join("holding").is_file() {
                    assert!(
                        std::time::Instant::now() < until,
                        "held child handshake absent"
                    );
                    std::thread::park_timeout(Duration::from_millis(1));
                }
                let pid: u32 = std::fs::read_to_string(root.path().join("capture-held.pid"))
                    .unwrap()
                    .parse()
                    .unwrap();
                assert!(alive(pid), "unrelated holder exited before release");
                std::fs::write(root.path().join("release"), b"release").unwrap();
            }
            for worker in workers {
                worker.join().unwrap();
            }
            reports
        });
        sentinel.assert_alive();
        for report in fast {
            assert!(!alive(report.pid));
            assert!(report.success(), "batch={batch} fast report: {report:?}");
        }
    }
}
