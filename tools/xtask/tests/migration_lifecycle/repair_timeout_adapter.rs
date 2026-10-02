use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
};
use std::{
    collections::BTreeMap,
    fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    time::Duration,
};
struct Fixture {
    _temporary: tempfile::TempDir,
    root: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().canonicalize().unwrap();
        fs::create_dir(root.join("tmp")).unwrap();
        let source = fs::read_to_string(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../scripts/llama-canary-agent-repair.sh"),
        )
        .unwrap();
        let selection = source
            .split("# Legacy workload automation selection begins.\n")
            .nth(1)
            .unwrap()
            .split("# Legacy workload automation selection ends.")
            .next()
            .unwrap();
        let function = source
            .split("run_for() {\n")
            .nth(1)
            .unwrap()
            .split("\nrecord_failure_class()")
            .next()
            .unwrap();
        fs::write(
            root.join("adapter.sh"),
            format!("set -euo pipefail\n{selection}\nrun_for() {{\n{function}"),
        )
        .unwrap();
        fs::copy(env!("CARGO_BIN_EXE_xtask"), root.join("controller")).unwrap();
        let child = root.join("child");
        fs::write(&child,"#!/bin/bash\nset -euo pipefail\n[[ \"$ADAPTER_ENV\" == 'kept value' ]] || exit 97\nfor arg in \"$@\"; do printf '%s\\0' \"$arg\"; done > child.args\nprintf 'native output\\n'\nprintf 'native error\\n' >&2\nexit 7\n").unwrap();
        fs::set_permissions(child, fs::Permissions::from_mode(0o755)).unwrap();
        Self {
            _temporary: temporary,
            root,
        }
    }
    fn run(&self, body: &str) -> process::RawProcessReport {
        self.run_with_env(body, "kept value")
    }
    fn run_with_env(&self, body: &str, adapter_env: &str) -> process::RawProcessReport {
        let environment: BTreeMap<_, _> = [
            ("PATH", std::env::var("PATH").unwrap()),
            ("RUNNER_TEMP", self.root.join("tmp").display().to_string()),
            (
                "MESH_LLM_AUTOMATION_BIN",
                self.root.join("controller").display().to_string(),
            ),
            ("HARNESS_MODE", "repair".into()),
            ("ADAPTER_ENV", adapter_env.into()),
        ]
        .into_iter()
        .map(|(key, value)| (key.into(), Value::Public(value.into())))
        .collect();
        let report = process::supervise_raw(
            &ProcessSpec {
                executable: "/bin/bash".into(),
                cwd: self.root.clone(),
                environment,
                arguments: vec!["-c".into(), format!("source ./adapter.sh\n{body}").into()]
                    .into_iter()
                    .map(Value::Public)
                    .collect(),
            },
            &Limits {
                execution: Duration::from_secs(25),
                graceful_shutdown: Duration::from_secs(1),
                forced_shutdown: Duration::from_secs(1),
                retained_bytes_per_stream: 65536,
                readiness: Readiness::None,
                completion: Completion::Exit,
            },
            &Cancellation::default(),
            RawCaptureOptions {
                stdout: std::num::NonZeroUsize::new(65536),
                stderr: std::num::NonZeroUsize::new(65536),
            },
        )
        .unwrap();
        assert!(report.process.failure.is_none(), "{:?}", report.process);
        assert!(report.process.cleanup.complete, "{:?}", report.process);
        report
    }
}
#[test]
fn repair_timeout_adapter_uses_frozen_owner_with_relative_native_argv_environment_and_exit() {
    let fixture = Fixture::new();
    let report=fixture.run("if run_for 'agent developer task' 2 ./child '' 'space value' '*.txt'; then exit 96; else status=$?; exit \"$status\"; fi");
    assert_eq!(report.process.status.unwrap().code(), Some(7));
    assert_eq!(
        fs::read(fixture.root.join("child.args")).unwrap(),
        b"\0space value\0*.txt\0"
    );
    assert!(String::from_utf8_lossy(report.stdout.unwrap().as_bytes()).contains("native output"));
    assert!(String::from_utf8_lossy(report.stderr.unwrap().as_bytes()).contains("native error"));
    assert_eq!(fs::read_dir(fixture.root.join("tmp")).unwrap().count(), 0);
}
#[test]
fn repair_timeout_adapter_rejects_changed_controller_before_request_or_launch_and_bounds_deadline()
{
    let changed = Fixture::new();
    let report=changed.run("printf changed >> \"$repair_workload_controller\"\nif run_for changed 2 ./child; then exit 96; else exit \"$?\"; fi");
    assert_eq!(report.process.status.unwrap().code(), Some(125));
    assert!(!changed.root.join("child.args").exists());
    assert_eq!(fs::read_dir(changed.root.join("tmp")).unwrap().count(), 0);
    let missing = Fixture::new();
    let report = missing.run("if run_for missing 2 nonexistent-repair-fixture-command; then exit 96; else exit \"$?\"; fi");
    assert_eq!(report.process.status.unwrap().code(), Some(125));
    assert_eq!(fs::read_dir(missing.root.join("tmp")).unwrap().count(), 0);
    let deadline = Fixture::new();
    let report = deadline
        .run("if run_for 'deadline fixture' 1 /bin/sleep 20; then exit 96; else exit \"$?\"; fi");
    assert_eq!(report.process.status.unwrap().code(), Some(124));
    assert!(
        String::from_utf8_lossy(report.stderr.unwrap().as_bytes())
            .contains("deadline fixture timed out after 1s")
    );
    assert_eq!(fs::read_dir(deadline.root.join("tmp")).unwrap().count(), 0);
}

#[test]
fn repair_timeout_adapter_rejects_wrong_native_environment_before_recording_arguments() {
    let fixture = Fixture::new();
    let report = fixture.run_with_env(
        "if run_for environment 2 ./child value; then exit 96; else exit \"$?\"; fi",
        "wrong value",
    );
    assert_eq!(report.process.status.unwrap().code(), Some(97));
    assert!(!fixture.root.join("child.args").exists());
    assert_eq!(fs::read_dir(fixture.root.join("tmp")).unwrap().count(), 0);
}
