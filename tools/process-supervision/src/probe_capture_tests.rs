use super::*;

fn capture(stream: Stream, limit: usize) -> Capture<std::process::ChildStdout> {
    #[cfg(target_os = "macos")]
    let (reader, _writer) = platform::test_pipe().unwrap();
    #[cfg(not(target_os = "macos"))]
    let (reader, _writer) = {
        let (reader, writer) = std::io::pipe().unwrap();
        (std::os::fd::OwnedFd::from(reader), writer)
    };
    Capture::new(
        reader.into(),
        stream,
        None,
        limit,
        vec![b"private".to_vec()],
    )
    .unwrap()
}

#[test]
fn probe_raw_lines_precede_redaction_and_zero_retention() {
    for stream in [Stream::Stdout, Stream::Stderr] {
        for limit in [0, 128] {
            let mut capture = capture(stream, limit);
            let mut count = 0;
            capture
                .consume_with(b"token=private\r\n", &mut |line| {
                    count += 1;
                    assert_eq!(line.stream, stream);
                    assert_eq!(line.bytes, b"token=private\r");
                    assert_eq!(line.ending, LineEnding::Lf);
                    false
                })
                .unwrap();
            assert_eq!(count, 1);
            assert_eq!(capture.report.suppressed_lines, 1);
            assert_eq!(
                capture.report.bytes_retained,
                &b"[output line suppressed]\n"[..limit.min(25)]
            );
        }
    }
}

#[test]
fn probe_raw_size_ceiling_includes_cr_but_not_lf() {
    for length in [8192, 8193] {
        let mut capture = capture(Stream::Stdout, 0);
        let mut count = 0;
        let mut callback = |line: ObservedLine<'_>| {
            count += 1;
            assert_eq!(line.bytes.len(), 8192);
            assert!(line.bytes.ends_with(b"\r"));
            false
        };
        capture
            .consume_with(&vec![b'x'; length - 1], &mut callback)
            .unwrap();
        capture.consume_with(b"\r\n", &mut callback).unwrap();
        assert_eq!(count, usize::from(length == 8192));
    }
}

#[test]
fn probe_fragments_keep_their_ending_and_cleanup_has_no_callback() {
    let mut capture = capture(Stream::Stderr, 128);
    capture
        .consume_with(b"READY", &mut |_| panic!("fragment callback"))
        .unwrap();
    let mut count = 0;
    capture
        .flush_with(&mut |line| {
            count += 1;
            assert_eq!(line.ending, LineEnding::Eof);
            assert_eq!(line.bytes, b"READY");
            false
        })
        .unwrap();
    capture
        .consume_with(b"READY", &mut |_| panic!("fragment callback"))
        .unwrap();
    capture.consume(b"\n", &Readiness::None).unwrap();
    let (report, failure) = capture.finish();
    assert!(failure.is_none());
    assert_eq!(count, 1);
    assert_eq!(report.bytes_retained, b"READYREADY\n");
}

#[test]
fn probe_output_write_error_survives_raw_classification() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("evidence");
    std::fs::write(&path, b"preserved").unwrap();
    let mut capture = capture(Stream::Stdout, 128);
    capture.file = Some(File::open(&path).unwrap());
    let mut count = 0;
    let result = capture.consume_with(b"READY\n", &mut |_| {
        count += 1;
        false
    });
    assert_eq!(count, 1);
    assert!(matches!(
        result,
        Err(Failure::Io {
            operation: "write output",
            ..
        })
    ));
    assert_eq!(std::fs::read(path).unwrap(), b"preserved");
}
