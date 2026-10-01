use super::*;
use std::io::Read;
#[cfg(not(target_os = "macos"))]
use std::os::fd::OwnedFd;
use std::process::ChildStdout;
use std::time::Duration;

fn capture() -> (Capture<ChildStdout>, std::io::PipeWriter) {
    #[cfg(target_os = "macos")]
    let (reader, writer) = {
        let (reader, writer) = platform::test_pipe().unwrap();
        (reader, std::io::PipeWriter::from(writer))
    };
    #[cfg(not(target_os = "macos"))]
    let (reader, writer) = {
        let (reader, writer) = std::io::pipe().unwrap();
        (OwnedFd::from(reader), writer)
    };
    let capture = Capture::new(
        ChildStdout::from(reader),
        Stream::Stdout,
        None,
        128,
        Vec::new(),
    )
    .unwrap();
    (capture, writer)
}

fn readiness() -> Readiness {
    Readiness::Line {
        stream: Stream::Stdout,
        bytes: b"READY".to_vec(),
        deadline: Duration::from_secs(1),
    }
}

#[test]
fn migration_process_readiness_snapshot_preserves_pre_exit_buffered_marker() {
    let (mut capture, mut writer) = capture();
    writer.write_all(b"READY\n").unwrap();
    let eligible = capture.pending_bytes().unwrap();
    let ready = capture.poll_snapshot(&readiness(), eligible).unwrap();
    assert!(ready);
}

#[test]
fn migration_process_readiness_snapshot_excludes_new_marker_suffix() {
    let (mut capture, mut writer) = capture();
    writer.write_all(b"REA").unwrap();
    let eligible = capture.pending_bytes().unwrap();
    writer.write_all(b"DY\n").unwrap();
    let ready = capture.poll_snapshot(&readiness(), eligible).unwrap();
    assert!(!ready);
    assert!(!capture.poll(&Readiness::None).unwrap());
    let (report, error) = capture.finish();
    assert!(error.is_none());
    assert_eq!(report.bytes_retained, b"READY\n");
}

#[test]
fn migration_process_readiness_snapshot_flushes_observed_eof() {
    #[cfg(target_os = "macos")]
    if std::env::var_os("MIGRATION_PROCESS_EOF_CHILD").is_none() {
        let test_name =
            "process::capture::tests::migration_process_readiness_snapshot_flushes_observed_eof";
        let output_dir = tempfile::tempdir().unwrap();
        let stdout_path = output_dir.path().join("stdout");
        let stderr_path = output_dir.path().join("stderr");
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", test_name, "--nocapture"])
            .env("MIGRATION_PROCESS_EOF_CHILD", "1")
            .stdout(std::fs::File::create(&stdout_path).unwrap())
            .stderr(std::fs::File::create(&stderr_path).unwrap())
            .spawn()
            .unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break status,
                Ok(None) if std::time::Instant::now() < deadline => {
                    std::thread::park_timeout(Duration::from_millis(5));
                }
                result => {
                    let kill = child.kill();
                    let reap = child.wait();
                    assert!(kill.is_ok() && reap.is_ok(), "kill={kill:?} reap={reap:?}");
                    panic!("isolated EOF test did not exit before deadline: {result:?}");
                }
            }
        };
        let mut stdout_bytes = Vec::new();
        let mut stderr_bytes = Vec::new();
        std::fs::File::open(stdout_path)
            .unwrap()
            .take(4096)
            .read_to_end(&mut stdout_bytes)
            .unwrap();
        std::fs::File::open(stderr_path)
            .unwrap()
            .take(4096)
            .read_to_end(&mut stderr_bytes)
            .unwrap();
        let stdout = String::from_utf8_lossy(&stdout_bytes);
        let stderr = String::from_utf8_lossy(&stderr_bytes);
        assert!(
            status.success()
                && stdout.contains(&format!("test {test_name} ... ok"))
                && stdout.contains("1 passed; 0 failed"),
            "isolated EOF test status={:?} stdout={stdout} stderr={stderr}",
            status
        );
        return;
    }

    let (mut capture, mut writer) = capture();
    writer.write_all(b"READY").unwrap();
    let eligible = capture.pending_bytes().unwrap();
    drop(writer);
    let ready = capture.poll_snapshot(&readiness(), eligible).unwrap();
    assert!(
        ready,
        "eligible={eligible} pending={} eof={} read={:?}",
        capture.pending.len(),
        capture.eof(),
        platform::read_pipe(&mut capture.pipe, &mut [0; 1])
    );
    assert!(capture.eof());
}

#[test]
fn migration_process_readiness_snapshot_does_not_invent_eof_for_held_writer() {
    let (mut capture, mut writer) = capture();
    writer.write_all(b"READY").unwrap();
    let eligible = capture.pending_bytes().unwrap();
    let ready = capture.poll_snapshot(&readiness(), eligible).unwrap();
    assert!(!ready);
    assert!(!capture.eof());
    drop(writer);
    assert!(!capture.poll(&Readiness::None).unwrap());
}

#[test]
fn migration_process_readiness_snapshot_rejects_post_cutoff_newline_and_eof() {
    let (mut capture, mut writer) = capture();
    writer.write_all(b"READY").unwrap();
    let eligible = capture.pending_bytes().unwrap();
    writer.write_all(b"\n").unwrap();
    drop(writer);
    let ready = capture.poll_snapshot(&readiness(), eligible).unwrap();
    assert!(!ready);
    assert!(!capture.poll(&Readiness::None).unwrap());
    let (report, error) = capture.finish();
    assert!(error.is_none());
    assert_eq!(report.bytes_retained, b"READY\n");
}
