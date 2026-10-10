//! Actual frontdoor with inert local peers; no mesh/runtime/payment claim.
use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
};
use serde_json::Value as Json;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
fn quote(path: &Path) -> String {
    format!("'{}'", path.to_str().unwrap().replace('\'', "'\\''"))
}
fn binary(root: &Path, name: &str, mode: &str) -> PathBuf {
    use std::os::unix::fs::PermissionsExt as _;
    let directory = root.join(name);
    std::fs::create_dir(&directory).unwrap();
    std::fs::create_dir(directory.join("native-runtimes")).unwrap();
    std::fs::write(
        directory.join("native-runtimes/offered-fixture"),
        b"not a loaded or validated package",
    )
    .unwrap();
    let helper = Path::new(env!("CARGO_BIN_EXE_xtask"))
        .parent()
        .unwrap()
        .join("examples/l7_daemon_fixture");
    let path = directory.join("mesh-llm");
    std::fs::write(
        &path,
        format!(
            "#!/bin/sh\nexec {} lightning-peer {mode} \"$@\"\n",
            quote(&helper)
        ),
    )
    .unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).unwrap();
    path
}
fn spec(root: &Path, mode: &str) -> ProcessSpec {
    let current = binary(root, "current", "current");
    let released = binary(root, "released", mode);
    let model = root.join("inert.gguf");
    std::fs::write(&model, b"not actual GGUF, inert child does not load model").unwrap();
    ProcessSpec {
        executable: env!("CARGO_BIN_EXE_xtask").into(),
        cwd: root.into(),
        environment: BTreeMap::from([("PATH".into(), Value::Public("/usr/bin:/bin".into()))]),
        arguments: [
            "automation",
            "lightning-compatibility",
            "--current-binary",
            current.to_str().unwrap(),
            "--released-binary",
            released.to_str().unwrap(),
            "--released-dialect",
            "serve-client-bind-port",
            "--model",
            model.to_str().unwrap(),
            "--output",
            root.join("evidence").to_str().unwrap(),
            "--timeout-secs",
            "30",
        ]
        .into_iter()
        .map(|s| Value::Public(s.into()))
        .collect(),
    }
}
fn execute(spec: &ProcessSpec, cancel: &Cancellation) -> process::RawProcessReport {
    process::supervise_raw(
        spec,
        &Limits {
            execution: Duration::from_secs(40),
            graceful_shutdown: Duration::from_secs(14),
            forced_shutdown: Duration::from_secs(2),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        cancel,
        RawCaptureOptions {
            stdout: std::num::NonZeroUsize::new(65536),
            stderr: std::num::NonZeroUsize::new(65536),
        },
    )
    .unwrap()
}
fn clean(raw: &process::RawProcessReport) {
    let p = &raw.process;
    assert!(
        p.failure.is_none()
            && p.cleanup.complete
            && !p.cleanup.forced
            && !p.cleanup.graceful_signal_failed
            && p.cleanup.failure.is_none()
    );
    for (bytes, stream) in [
        (raw.stdout.as_ref().unwrap(), &p.stdout),
        (raw.stderr.as_ref().unwrap(), &p.stderr),
    ] {
        assert_eq!(bytes.as_bytes().len() as u64, stream.bytes_seen);
        assert!(!stream.truncated && stream.line_capture_complete);
    }
}
fn report(root: &Path) -> Json {
    serde_json::from_slice(&std::fs::read(root.join("evidence/results.json")).unwrap()).unwrap()
}
#[test]
fn lightning_actual_cli_runs_three_cases_and_isolated_bundle_state() {
    let root = tempfile::tempdir().unwrap();
    let raw = execute(&spec(root.path(), "released"), &Cancellation::default());
    clean(&raw);
    assert_eq!(raw.process.outcome, process::Outcome::Exited);
    assert!(raw.process.success());
    let report = report(root.path());
    assert_eq!(report["passed"], true);
    assert_eq!(
        report["cases"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["status"].as_u64().unwrap())
            .collect::<Vec<_>>(),
        vec![200, 402, 200]
    );
    assert_eq!(report["members"].as_array().unwrap().len(), 5);
    assert_eq!(report["source_unchanged"], true);
    assert_eq!(report["logs"].as_object().unwrap().len(), 15);
    for name in [
        "current-provider",
        "released-client-free",
        "released-client-paid",
        "released-provider",
        "current-client",
    ] {
        let path = root.path().join("evidence").join(name);
        assert!(
            path.join("config.toml").is_file()
                && path.join("stdout.log").is_file()
                && path.join("stderr.log").is_file()
        );
        let args: Vec<String> =
            serde_json::from_slice(&std::fs::read(path.join("inert-argv.json")).unwrap()).unwrap();
        assert!(args.contains(&"--bind-port".into()));
        if name.contains("client") {
            assert!(args.contains(&"<redacted>".into()));
        }
        assert!(
            !std::fs::read_to_string(root.path().join("evidence/results.json"))
                .unwrap()
                .contains("inert:")
        );
    }
    root.close().unwrap();
}
#[test]
fn lightning_actual_cli_refuses_paid_bypass_bad_usage_and_existing_output() {
    for (mode, count) in [("bypass", 2), ("missing-usage", 1)] {
        let root = tempfile::tempdir().unwrap();
        let raw = execute(&spec(root.path(), mode), &Cancellation::default());
        clean(&raw);
        assert_eq!(raw.process.outcome, process::Outcome::Exited);
        assert!(!raw.process.status.unwrap().success());
        let report = report(root.path());
        assert_eq!(report["passed"], false);
        assert_eq!(report["cases"].as_array().unwrap().len(), count);
        assert_eq!(report["cases"][count - 1]["passed"], false);
        root.close().unwrap();
    }
    let root = tempfile::tempdir().unwrap();
    let invocation = spec(root.path(), "released");
    std::fs::create_dir(root.path().join("evidence")).unwrap();
    std::fs::write(root.path().join("evidence/preserve"), b"prior").unwrap();
    let raw = execute(&invocation, &Cancellation::default());
    clean(&raw);
    assert!(!raw.process.status.unwrap().success());
    assert_eq!(
        std::fs::read(root.path().join("evidence/preserve")).unwrap(),
        b"prior"
    );
    assert!(!root.path().join("evidence/current-provider").exists());
    root.close().unwrap();
    let root = tempfile::tempdir().unwrap();
    let held = execute(&spec(root.path(), "hold"), &Cancellation::default());
    clean(&held);
    assert_eq!(held.process.outcome, process::Outcome::Exited);
    assert!(!held.process.status.unwrap().success());
    assert_eq!(report(root.path())["passed"], false);
    assert!(
        root.path()
            .join("evidence/released-client-free/received-inference")
            .exists()
    );
    root.close().unwrap();
}
#[test]
fn lightning_actual_cli_inflight_cancellation_preserves_failed_evidence_and_cleanup() {
    let root = tempfile::tempdir().unwrap();
    let spec = spec(root.path(), "hold");
    let cancellation = Cancellation::default();
    let held = cancellation.clone();
    let peer = std::thread::spawn(move || execute(&spec, &held));
    let until = Instant::now() + Duration::from_secs(20);
    let marker = root
        .path()
        .join("evidence/released-client-free/received-inference");
    while !marker.exists() && Instant::now() < until {
        std::thread::sleep(Duration::from_millis(5));
    }
    let observed = marker.exists();
    cancellation.cancel();
    let raw = peer.join().unwrap();
    clean(&raw);
    assert!(observed && cancellation.is_cancelled());
    assert!(!raw.process.status.unwrap().success());
    assert_eq!(report(root.path())["passed"], false);
    root.close().unwrap();
}
