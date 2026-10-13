use super::*;
use std::io::Write;
fn pipe() -> (std::os::fd::OwnedFd, std::io::PipeWriter) {
    #[cfg(target_os = "macos")]
    {
        let (reader, writer) = super::super::platform::test_pipe().unwrap();
        (reader, std::io::PipeWriter::from(writer))
    }
    #[cfg(not(target_os = "macos"))]
    {
        let (reader, writer) = std::io::pipe().unwrap();
        (reader.into(), writer)
    }
}
#[test]
fn retained_stderr_projection_survives_stdout_failure_without_readiness_admission() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("read-only.log");
    std::fs::write(&path, b"").unwrap();
    let (stdout, mut out) = pipe();
    let (stderr, mut err) = pipe();
    out.write_all(b"stdout\n").unwrap();
    err.write_all(b"health=7 secret=private\n").unwrap();
    drop(out);
    drop(err);
    let mut output = Output {
        stdout: Capture::new(
            ChildStdout::from(stdout),
            Stream::Stdout,
            Some(File::open(&path).unwrap()),
            128,
            vec![],
        )
        .unwrap(),
        stderr: Capture::new(ChildStderr::from(stderr), Stream::Stderr, None, 128, vec![]).unwrap(),
        failure: None,
    };
    let (mut stdout_count, mut stderr_count, mut readiness_count) = (0, 0, 0);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
    let mut failure = false;
    loop {
        let result = output.poll_captured(&mut |line, allowed| match line.stream {
            Stream::Stdout => stdout_count += 1,
            Stream::Stderr => {
                stderr_count += 1;
                if allowed {
                    readiness_count += 1;
                }
                assert_eq!(line.bytes, b"health=7 secret=private");
            }
        });
        failure |= result.is_err();
        if stderr_count == 1 {
            break;
        }
        assert!(std::time::Instant::now() < deadline);
        std::thread::yield_now();
    }
    assert!(failure);
    assert_eq!((stdout_count, stderr_count, readiness_count), (1, 1, 0));
    let (stdout, _) = output.stdout.finish();
    let (stderr, _) = output.stderr.finish();
    assert!(!stdout.line_capture_complete);
    assert!(stderr.line_capture_complete);
    assert_eq!(stderr.suppressed_lines, 1);
    assert!(!String::from_utf8_lossy(&stderr.bytes_retained).contains("private"));
    directory.close().unwrap();
}
