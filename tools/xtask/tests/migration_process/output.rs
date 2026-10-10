use crate::{process::*, support::*};
use std::time::Duration;

#[test]
fn migration_process_flood_drains_both_pipes_with_bounded_retention() {
    let root = tempfile::tempdir().unwrap();
    let mut limits = limits();
    limits.retained_bytes_per_stream = 128;
    ready(&mut limits, b"READY");
    let report = supervise(
        &spec(root.path(), "flood"),
        &limits,
        &Cancellation::default(),
        OutputFiles {
            stdout: Some(root.path().join("stdout")),
            stderr: Some(root.path().join("stderr")),
        },
    )
    .unwrap();
    assert!(report.success(), "{report:?}");
    for stream in [&report.stdout, &report.stderr] {
        assert!(stream.bytes_seen >= 4 * 1024 * 1024);
        assert!(stream.bytes_retained.len() <= 128);
        assert!(stream.truncated);
    }
    assert_eq!(
        std::fs::read(root.path().join("stdout")).unwrap(),
        report.stdout.bytes_retained
    );
    assert_eq!(
        std::fs::read(root.path().join("stderr")).unwrap(),
        report.stderr.bytes_retained
    );
}

#[test]
fn migration_process_infinite_flood_cannot_starve_deadline() {
    let root = tempfile::tempdir().unwrap();
    let mut limits = limits();
    limits.execution = Duration::from_millis(100);
    limits.retained_bytes_per_stream = 128;
    let report = supervise(
        &spec(root.path(), "forever-flood"),
        &limits,
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap();
    assert_eq!(report.outcome, Outcome::Deadline);
    assert!(report.cleanup.complete);
    assert!(report.elapsed < Duration::from_secs(3));
}

#[test]
fn migration_process_invalid_encoding_is_preserved_without_panicking() {
    let root = tempfile::tempdir().unwrap();
    let report = supervise(
        &spec(root.path(), "bytes"),
        &limits(),
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap();
    assert!(report.success(), "{report:?}");
    assert!(
        report
            .stdout
            .bytes_retained
            .windows(6)
            .any(|bytes| bytes == b"bad\xff\xfe\n")
    );
    assert!(report.stderr.bytes_retained.ends_with(b"err\x80\n"));
}

#[test]
fn migration_process_secrets_are_redacted_in_split_writes_and_unterminated_output() {
    let root = tempfile::tempdir().unwrap();
    let mut spec = spec(root.path(), "secrets");
    let secret = "fixture-credential-value";
    spec.environment
        .insert("CREDENTIAL".into(), Value::Secret(secret.into()));
    let report = supervise(
        &spec,
        &limits(),
        &Cancellation::default(),
        OutputFiles {
            stdout: Some(root.path().join("out")),
            stderr: None,
        },
    )
    .unwrap();
    assert!(report.success(), "{report:?}");
    for bytes in [&report.stdout.bytes_retained, &report.stderr.bytes_retained] {
        assert!(
            !bytes
                .windows(secret.len())
                .any(|part| part == secret.as_bytes())
        );
        assert!(!bytes.windows(15).any(|part| part == b"fixture-private"));
    }
    assert_eq!(
        std::fs::read(root.path().join("out")).unwrap(),
        report.stdout.bytes_retained
    );
}

#[test]
fn migration_process_explicit_cwd_environment_and_literal_argv() {
    let root = tempfile::Builder::new()
        .prefix("process path spaces ")
        .tempdir()
        .unwrap();
    let mut spec = spec(root.path(), "args");
    spec.environment.insert(
        "PAYLOAD".into(),
        Value::Public("literal payload; $(not-a-command)".into()),
    );
    spec.arguments.push(Value::Public("--skip".into()));
    spec.arguments
        .push(Value::Public("literal argv; $(not-a-command)".into()));
    let report = supervise(
        &spec,
        &limits(),
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap();
    assert!(report.success(), "{report:?}");
    assert_eq!(
        std::fs::read(root.path().join("cwd")).unwrap(),
        std::fs::canonicalize(root.path())
            .unwrap()
            .as_os_str()
            .as_encoded_bytes()
    );
    assert!(
        String::from_utf8_lossy(&report.stdout.bytes_retained)
            .contains("literal payload; $(not-a-command)")
    );
    let args: Vec<String> =
        serde_json::from_slice(&std::fs::read(root.path().join("args.json")).unwrap()).unwrap();
    assert_eq!(args.last().unwrap(), "literal argv; $(not-a-command)");
    assert_eq!(
        std::fs::read_to_string(root.path().join("inherited-home")).unwrap(),
        "absent"
    );
}

#[test]
fn migration_process_existing_output_is_not_overwritten() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("evidence");
    std::fs::write(&path, b"previous evidence").unwrap();
    let result = supervise(
        &spec(root.path(), "exit"),
        &limits(),
        &Cancellation::default(),
        OutputFiles {
            stdout: Some(path.clone()),
            stderr: None,
        },
    );
    assert!(matches!(
        result,
        Err(Failure::Io {
            operation: "create output",
            ..
        })
    ));
    assert_eq!(std::fs::read(path).unwrap(), b"previous evidence");
    assert!(!root.path().join("exit.pid").exists());
}

#[test]
fn migration_process_multiline_secret_is_rejected_before_spawn() {
    let root = tempfile::tempdir().unwrap();
    let mut spec = spec(root.path(), "exit");
    spec.environment
        .insert("CREDENTIAL".into(), Value::Secret("one\ntwo".into()));
    let result = supervise(
        &spec,
        &limits(),
        &Cancellation::default(),
        OutputFiles::default(),
    );
    assert!(matches!(result, Err(Failure::InvalidSpec(_))));
    assert!(!root.path().join("exit.pid").exists());
}
