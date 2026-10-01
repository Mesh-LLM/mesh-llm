use super::{Error, NativeLipo, fixtures::Fixture};
use crate::process::{Cancellation, Outcome};
use std::{collections::BTreeMap, fs, num::NonZeroUsize, path::PathBuf, time::Duration};

fn native(fixture: &Fixture) -> NativeLipo {
    let executable = PathBuf::from(std::env::var_os("CARGO_TARGET_DIR").unwrap())
        .join("debug/examples/migration_xcframework_lipo_fixture");
    assert!(
        executable.is_file(),
        "build the authored inert Rust example first"
    );
    NativeLipo {
        executable,
        cwd: fixture.directory.path().to_owned(),
        environment: BTreeMap::new(),
        timeout: Duration::from_secs(5),
        max_bytes: NonZeroUsize::new(65536).unwrap(),
    }
}

#[test]
fn native_adapter_when_exact_full_matrix() {
    let fixture = Fixture::full();
    fixture.materialize();
    fixture.write(true);
    let adapter = native(&fixture);
    let result = super::verify_with_native(
        &fixture.root,
        Some(super::Mode::Full),
        &adapter,
        &Cancellation::default(),
    );
    assert_eq!(result.unwrap(), 4);
}

#[test]
fn empty_lipo_when_stdout_has_no_architecture() {
    let fixture = Fixture::host();
    fixture.materialize();
    let binary = fixture.framework(0).join("MeshLLMFFI");
    fs::write(&binary, "!empty").unwrap();
    let result = native(&fixture).inspect(&binary, &Cancellation::default());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("reported no architectures")
    );
}

#[test]
fn raw_lipo_when_redaction_triggers_and_python_separators_are_architectures() {
    let fixture = Fixture::host();
    fixture.materialize();
    let binary = fixture.framework(0).join("MeshLLMFFI");
    fs::write(&binary, "!raw").unwrap();
    let result = native(&fixture).inspect(&binary, &Cancellation::default());
    assert_eq!(
        result.unwrap(),
        ["authorization", "invite", "password", "secret", "token"]
            .into_iter()
            .map(str::to_owned)
            .collect()
    );
}

#[test]
fn nonzero_lipo_when_stderr_has_private_payload() {
    let fixture = Fixture::host();
    fixture.materialize();
    let binary = fixture.framework(0).join("MeshLLMFFI");
    fs::write(&binary, "!nonzero").unwrap();
    let result = native(&fixture).inspect(&binary, &Cancellation::default());
    let error = result.unwrap_err();
    assert!(!error.to_string().contains("private-error-sentinel"));
    match error {
        Error::Native { report, .. } => {
            assert_eq!(report.process.status.unwrap().code(), Some(7));
            assert_eq!(
                report.stderr.unwrap().as_bytes(),
                b"private-error-sentinel\n"
            );
        }
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn output_bound_when_native_stdout_exceeds_limit() {
    let fixture = Fixture::host();
    fixture.materialize();
    let binary = fixture.framework(0).join("MeshLLMFFI");
    fs::write(&binary, "!overflow").unwrap();
    let mut adapter = native(&fixture);
    adapter.max_bytes = NonZeroUsize::new(16).unwrap();
    let result = adapter.inspect(&binary, &Cancellation::default());
    assert!(matches!(
        result,
        Err(Error::Native { .. }) | Err(Error::Spawn { .. })
    ));
}

#[test]
fn adapter_is_read_only_when_native_verification_succeeds() {
    let fixture = Fixture::host();
    fixture.materialize();
    fixture.write(false);
    let before = super::layout_tests::snapshot(&fixture.root);
    let result = super::verify_with_native(
        &fixture.root,
        None,
        &native(&fixture),
        &Cancellation::default(),
    );
    assert_eq!(result.unwrap(), 1);
    assert_eq!(super::layout_tests::snapshot(&fixture.root), before);
}

#[test]
fn timeout_stops_descendant_when_native_child_tree_stalls() {
    let fixture = Fixture::host();
    fixture.materialize();
    let binary = fixture.framework(0).join("MeshLLMFFI");
    fs::write(&binary, "!tree").unwrap();
    let mut adapter = native(&fixture);
    adapter.timeout = Duration::from_millis(500);
    let sentinel = fixture.directory.path().join("sentinel");
    let ready = fixture.directory.path().join("ready");
    adapter.environment.insert(
        "XCFRAMEWORK_SENTINEL".into(),
        sentinel.clone().into_os_string(),
    );
    adapter
        .environment
        .insert("XCFRAMEWORK_READY".into(), ready.clone().into_os_string());
    let result = adapter.inspect(&binary, &Cancellation::default());
    assert_tree_stopped(result, Outcome::Deadline, &ready, &sentinel);
}

#[test]
fn cancellation_stops_descendant_when_shared_scope_is_cancelled() {
    let fixture = Fixture::host();
    fixture.materialize();
    let binary = fixture.framework(0).join("MeshLLMFFI");
    fs::write(&binary, "!tree").unwrap();
    let mut adapter = native(&fixture);
    let sentinel = fixture.directory.path().join("sentinel");
    let ready = fixture.directory.path().join("ready");
    adapter.environment.insert(
        "XCFRAMEWORK_SENTINEL".into(),
        sentinel.clone().into_os_string(),
    );
    adapter
        .environment
        .insert("XCFRAMEWORK_READY".into(), ready.clone().into_os_string());
    let cancellation = Cancellation::default();
    let signal = cancellation.clone();
    let readiness = ready.clone();
    let trigger = std::thread::spawn(move || {
        let deadline = std::time::Instant::now() + Duration::from_secs(3);
        while !readiness.is_file() {
            assert!(
                std::time::Instant::now() < deadline,
                "fixture did not signal readiness"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
        signal.cancel();
    });
    let result = adapter.inspect(&binary, &cancellation);
    trigger.join().unwrap();
    assert_tree_stopped(result, Outcome::Cancelled, &ready, &sentinel);
}

fn assert_tree_stopped(
    result: Result<std::collections::BTreeSet<String>, Error>,
    outcome: Outcome,
    ready: &std::path::Path,
    sentinel: &std::path::Path,
) {
    match result.unwrap_err() {
        Error::Native { report, .. } => {
            assert_eq!(report.process.outcome, outcome);
            assert!(report.process.cleanup.complete);
            let pid = fs::read_to_string(ready).unwrap();
            let probe = std::process::Command::new("/bin/kill")
                .args(["-0", pid.trim()])
                .output()
                .unwrap();
            assert!(!probe.status.success(), "descendant remains live");
            assert!(!sentinel.exists());
        }
        other => panic!("unexpected {other:?}"),
    }
}
