use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
};
use crate::workflow_yaml;
use std::{
    collections::BTreeMap,
    fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    time::Duration,
};
fn source() -> String {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let text =
        fs::read_to_string(root.join(".github/workflows/agentic-replay-nightly.yml")).unwrap();
    let tree = workflow_yaml::parse(&text).unwrap();
    let workflow_yaml::Node::Seq(steps) = tree
        .get("jobs")
        .unwrap()
        .get("replay")
        .unwrap()
        .get("steps")
        .unwrap()
    else {
        panic!("steps")
    };
    steps
        .iter()
        .find(|step| step.get("id").and_then(workflow_yaml::Node::text) == Some("repair"))
        .unwrap()
        .get("run")
        .unwrap()
        .text()
        .unwrap()
        .to_owned()
}
fn bash() -> PathBuf {
    std::env::split_paths(&std::env::var_os("PATH").unwrap())
        .map(|path| path.join("bash"))
        .find(|path| {
            path.is_file() && fs::metadata(path).unwrap().permissions().mode() & 0o111 != 0
        })
        .unwrap()
}
struct RepairResult {
    prepared: String,
    log: String,
    calls: String,
}
fn execute(code: u8) -> RepairResult {
    let root = tempfile::tempdir().unwrap();
    fs::create_dir(root.path().join("scripts")).unwrap();
    fs::create_dir(root.path().join("agentic-replay-artifacts")).unwrap();
    let script = root.path().join("scripts/agentic-replay-repair.sh");
    fs::write(&script, "#!/bin/sh\nset -eu\n[ \"$#\" = 1 ] || exit 90\n[ \"$1\" = \"$RUNNER_TEMP/agentic-replay-artifacts\" ] || exit 91\nprintf '%s\\n' \"$REPLAY_AGENT_PROVIDER/$REPLAY_AGENT_MODEL\" > \"$RUNNER_TEMP/calls\"\nprintf 'repair stdout\\n'\nprintf 'repair stderr\\n' >&2\nexit \"$FIXTURE_STATUS\"\n").unwrap();
    fs::set_permissions(&script, fs::Permissions::from_mode(0o700)).unwrap();
    let output = root.path().join("outputs");
    let environment = [
        ("PATH", "/usr/bin:/bin".into()),
        ("RUNNER_TEMP", root.path().display().to_string()),
        ("GITHUB_OUTPUT", output.display().to_string()),
        ("FIXTURE_STATUS", code.to_string()),
        ("REPLAY_AGENT_PROVIDER", "finite-provider".into()),
        ("REPLAY_AGENT_MODEL", "finite-model".into()),
    ]
    .into_iter()
    .map(|(key, value)| (key.into(), Value::Public(value.into())))
    .collect::<BTreeMap<_, _>>();
    let spec = ProcessSpec {
        executable: bash(),
        cwd: root.path().to_owned(),
        arguments: vec![Value::Public("-c".into()), Value::Public(source().into())],
        environment,
    };
    let limits = Limits {
        execution: Duration::from_secs(4),
        graceful_shutdown: Duration::from_secs(1),
        forced_shutdown: Duration::from_secs(1),
        retained_bytes_per_stream: 8192,
        readiness: Readiness::None,
        completion: Completion::Exit,
    };
    let report = process::supervise_raw(
        &spec,
        &limits,
        &Cancellation::default(),
        RawCaptureOptions {
            stdout: std::num::NonZeroUsize::new(8192),
            stderr: std::num::NonZeroUsize::new(8192),
        },
    )
    .unwrap();
    assert!(
        report.process.cleanup.complete && report.process.failure.is_none(),
        "{:?}",
        report.process
    );
    assert_eq!(report.process.outcome, process::Outcome::Exited);
    assert!(
        report.process.status.unwrap().success(),
        "repair output branch intentionally reports prepared=false on child failure"
    );
    RepairResult {
        prepared: fs::read_to_string(output).unwrap(),
        log: fs::read_to_string(root.path().join("agentic-replay-artifacts/repair.log")).unwrap(),
        calls: fs::read_to_string(root.path().join("calls")).unwrap(),
    }
}
#[test]
fn actual_repair_step_records_both_streams_and_reports_only_successful_preparation() {
    for code in [0, 7] {
        let result = execute(code);
        assert_eq!(
            result.prepared,
            if code == 0 {
                "prepared=true\n"
            } else {
                "prepared=false\n"
            }
        );
        assert_eq!(result.log, "repair stdout\nrepair stderr\n");
        assert_eq!(result.calls, "finite-provider/finite-model\n");
    }
}
