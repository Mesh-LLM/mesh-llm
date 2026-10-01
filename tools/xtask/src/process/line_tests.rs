use super::*;
use std::time::Duration;

fn capture(stream: Stream, limit: usize) -> Capture<std::process::ChildStdout> {
    #[cfg(target_os = "macos")]
    let (reader, _writer) = platform::test_pipe().unwrap();
    #[cfg(not(target_os = "macos"))]
    let (reader, _writer) = {
        let (reader, writer) = std::io::pipe().unwrap();
        (std::os::fd::OwnedFd::from(reader), writer)
    };
    Capture::new(reader.into(), stream, None, limit, vec![b"a\"b".to_vec()]).unwrap()
}

fn readiness(matcher: super::super::LineMatcher) -> Readiness {
    Readiness::ObservedLines {
        deadline: Duration::from_secs(1),
        matcher,
    }
}

fn marker(line: ObservedLine<'_>) -> bool {
    line.ending == LineEnding::Lf
        && line.bytes.strip_suffix(b"\r").unwrap_or(line.bytes) == b"READY"
}

#[test]
fn migration_process_observed_lines_ignore_retention_and_stream_choice() {
    for stream in [Stream::Stdout, Stream::Stderr] {
        for limit in [0, 128] {
            let mut capture = capture(stream, limit);
            assert!(
                !capture
                    .consume(&b"harmless\n".repeat(32), &readiness(marker))
                    .unwrap()
            );
            let admitted = capture.consume(b"READY\n", &readiness(marker)).unwrap();
            assert!(admitted);
            assert_eq!(capture.report.bytes_retained.len(), limit);
            assert!(capture.report.truncated);
        }
    }
}

#[test]
fn migration_process_observed_lines_classify_before_cap_splits_record_or_lf() {
    for limit in [3, 5] {
        let mut capture = capture(Stream::Stdout, limit);
        let admitted = capture.consume(b"READY\n", &readiness(marker)).unwrap();
        assert!(admitted);
        assert_eq!(capture.report.bytes_retained, &b"READY\n"[..limit]);
        assert!(capture.report.truncated);
    }
}

#[test]
fn migration_process_observed_lines_classify_original_sensitive_and_secret_bytes() {
    for bytes in [b"READY tokens:0\n".as_slice(), b"READY a\"b\n"] {
        let mut capture = capture(Stream::Stdout, 128);
        let readiness =
            readiness(|line| line.ending == LineEnding::Lf && line.bytes.starts_with(b"READY "));
        let admitted = capture.consume(bytes, &readiness).unwrap();
        assert!(admitted);
        assert!(
            !capture
                .report
                .bytes_retained
                .windows(3)
                .any(|part| part == b"a\"b")
        );
        assert!(
            !capture
                .report
                .bytes_retained
                .windows(6)
                .any(|part| part == b"tokens")
        );
    }
}

#[test]
fn migration_process_observed_lines_do_not_match_redaction_created_bytes() {
    let mut capture = capture(Stream::Stdout, 128);
    let readiness = readiness(|line| line.bytes == b"READY ***");
    let admitted = capture.consume(b"READY a\"b\n", &readiness).unwrap();
    assert!(!admitted);
    assert_eq!(capture.report.bytes_retained, b"READY ***\n");
}

#[test]
fn migration_process_observed_lines_wait_for_lf_and_preserve_cr() {
    let mut capture = capture(Stream::Stderr, 128);
    let readiness = readiness(|line| {
        line.stream == Stream::Stderr && line.ending == LineEnding::Lf && line.bytes == b"READY\r"
    });
    assert!(!capture.consume(b"REA", &readiness).unwrap());
    assert!(!capture.consume(b"DY\r", &readiness).unwrap());
    assert!(capture.consume(b"\n", &readiness).unwrap());
}

#[test]
fn migration_process_observed_lines_pending_fragment_cannot_match_during_cleanup() {
    let mut capture = capture(Stream::Stdout, 128);
    assert!(!capture.consume(b"READY", &readiness(marker)).unwrap());
    let candidate = capture.consume(b"\n", &Readiness::None).unwrap();
    assert!(!candidate);
    assert_eq!(capture.report.bytes_retained, b"READY\n");
}

#[test]
fn migration_process_observed_lines_do_not_join_stream_fragments() {
    let mut stdout = capture(Stream::Stdout, 128);
    let mut stderr = capture(Stream::Stderr, 128);
    assert!(!stdout.consume(b"REA", &readiness(marker)).unwrap());
    assert!(!stderr.consume(b"DY\n", &readiness(marker)).unwrap());
}

#[test]
fn migration_process_observed_lines_enforce_raw_ceiling_including_cr() {
    for (length, expected) in [(8192, true), (8193, false)] {
        let mut capture = capture(Stream::Stdout, 128);
        let readiness = readiness(|line| {
            line.ending == LineEnding::Lf && line.bytes.len() == 8192 && line.bytes.ends_with(b"\r")
        });
        assert!(
            !capture
                .consume(&vec![b'x'; length - 1], &readiness)
                .unwrap()
        );
        let admitted = capture.consume(b"\r\n", &readiness).unwrap();
        assert_eq!(admitted, expected);
        assert_eq!(capture.report.suppressed_lines, u64::from(!expected));
    }
}

#[test]
fn migration_process_observed_lines_distinguish_eof_and_disable_finish_readiness() {
    let mut capture = capture(Stream::Stdout, 128);
    assert!(!capture.consume(b"READY", &readiness(marker)).unwrap());
    assert!(!capture.flush_line(&readiness(marker)).unwrap());
    assert!(!capture.consume(b"READY", &readiness(marker)).unwrap());
    assert!(
        capture
            .flush_line(&readiness(
                |line| line.ending == LineEnding::Eof && line.bytes == b"READY"
            ))
            .unwrap()
    );
    assert!(!capture.consume(b"READY", &readiness(marker)).unwrap());
    let (report, error) = capture.finish();
    assert!(error.is_none());
    assert_eq!(report.bytes_retained, b"READYREADYREADY");
}

#[test]
fn migration_process_observed_lines_output_write_failure_rejects_candidate() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("evidence");
    std::fs::write(&path, b"preserved").unwrap();
    let mut capture = capture(Stream::Stdout, 128);
    capture.file = Some(File::open(&path).unwrap());
    let result = capture.consume(b"READY\n", &readiness(marker));
    assert!(matches!(
        result,
        Err(Failure::Io {
            operation: "write output",
            ..
        })
    ));
    assert_eq!(std::fs::read(path).unwrap(), b"preserved");
}
