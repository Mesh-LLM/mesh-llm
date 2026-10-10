//! Actual admitted-mode timeout routing after removal of unreachable interpreter fallbacks.
use crate::process::{
    self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value,
};
use std::{collections::BTreeMap, fs, path::Path, time::Duration};

fn function(source: &str, name: &str) -> String {
    let marker = format!("{name}() {{\n");
    let body = source
        .split_once(&marker)
        .unwrap()
        .1
        .split_once("\n}\n")
        .unwrap()
        .0;
    format!("{marker}{body}\n}}\n")
}
fn run(root: &Path, mode: &str) -> process::ProcessReport {
    let source = fs::read_to_string(
        super::support::repository().join("scripts/llama-canary-agent-repair.sh"),
    )
    .unwrap();
    let admission = source.split_once("if [[ ! \"$HARNESS_MODE\" =~ ^(repair|verify|repair-build|verify-build|pinned-build)$ ]]; then\n").unwrap().1.split_once("\nfi\n").unwrap().0;
    let admission = format!(
        "if [[ ! \"$HARNESS_MODE\" =~ ^(repair|verify|repair-build|verify-build|pinned-build)$ ]]; then\n{admission}\nfi\n"
    );
    let selection = source
        .split_once("# Legacy workload automation selection begins.\n")
        .unwrap()
        .1
        .split_once("# Legacy workload automation selection ends.")
        .unwrap()
        .0;
    let timeout = function(&source, "run_for");
    let script = format!(
        r#"set -euo pipefail
{admission}
{selection}
{timeout}
if run_for 'admitted mode proof' 3 /bin/bash -c 'printf exercised > invoked; printf "%s\0" "$@" > arguments; exit 23' child '' 'space value' '*.txt'; then exit 96; else exit "$?"; fi
"#
    );
    let report = process::supervise(
        &ProcessSpec {
            executable: "/bin/bash".into(),
            cwd: root.into(),
            arguments: vec![Value::Public("-c".into()), Value::Public(script.into())],
            environment: BTreeMap::from([
                ("HARNESS_MODE".into(), Value::Public(mode.into())),
                (
                    "MESH_LLM_AUTOMATION_BIN".into(),
                    Value::Public(env!("CARGO_BIN_EXE_xtask").into()),
                ),
                ("RUNNER_TEMP".into(), Value::Public(root.into())),
                (
                    "PATH".into(),
                    Value::Public("/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin".into()),
                ),
            ]),
        },
        &Limits {
            execution: Duration::from_secs(8),
            graceful_shutdown: Duration::from_secs(2),
            forced_shutdown: Duration::from_secs(2),
            retained_bytes_per_stream: 32768,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap();
    assert!(
        report.cleanup.complete && !report.stdout.truncated && !report.stderr.truncated,
        "{report:?}"
    );
    report
}
#[test]
fn all_five_admitted_modes_use_actual_typed_timeout_and_preserve_child_status() {
    for mode in [
        "repair",
        "verify",
        "repair-build",
        "verify-build",
        "pinned-build",
    ] {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().canonicalize().unwrap();
        let report = run(&root, mode);
        assert_eq!(
            report.status.unwrap().code(),
            Some(23),
            "{mode}: {report:?}"
        );
        assert_eq!(fs::read(root.join("invoked")).unwrap(), b"exercised");
        assert_eq!(
            fs::read(root.join("arguments")).unwrap(),
            b"\0space value\0*.txt\0"
        );
        assert!(!fs::read_dir(root).unwrap().any(|entry| {
            entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with("canary-timeout.")
        }));
    }
}
#[test]
fn copied_actual_admission_refuses_other_modes_before_controller_or_child() {
    for mode in [
        "",
        "verify-extra",
        "custom-build",
        "REPAIR",
        "repair; /usr/bin/true",
    ] {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().canonicalize().unwrap();
        let report = run(&root, mode);
        assert_eq!(report.status.unwrap().code(), Some(1), "{report:?}");
        assert!(!root.join("invoked").exists());
        assert_eq!(fs::read_dir(root).unwrap().count(), 0);
    }
}
