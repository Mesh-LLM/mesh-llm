//! Actual typed CLI on finite copies of maintained workflows; no CI dispatch.
use crate::{
    process::{
        self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
    },
    support::{Golden, expected_jobs, lane_workflow, needs, repository_root},
};
use std::{collections::BTreeMap, fs, num::NonZeroUsize, path::Path, time::Duration};

fn actual(workflow: &Path) -> (i32, String) {
    let golden = Golden::load("control-main").unwrap();
    let jobs = expected_jobs().unwrap();
    let names = jobs["control-main"]["quality"]
        .as_array()
        .unwrap()
        .iter()
        .map(|name| name.as_str().unwrap())
        .collect::<Vec<_>>();
    let needs = needs("quality", &names);
    let mut environment = BTreeMap::new();
    for key in ["PATH", "SystemRoot", "WINDIR", "TEMP", "TMP"] {
        if let Some(value) = std::env::var_os(key) {
            environment.insert(key.into(), Value::Public(value));
        }
    }
    let result = process::supervise_raw(
        &ProcessSpec {
            executable: Path::new(env!("CARGO_BIN_EXE_xtask")).to_owned(),
            cwd: repository_root(),
            environment,
            arguments: [
                "ci",
                "validate-lane",
                "--lane-plan",
                &golden.lane_plan("quality"),
                "--needs",
                &needs,
                "--workflow",
                workflow.to_str().unwrap(),
            ]
            .into_iter()
            .map(|arg| Value::Public(arg.into()))
            .collect(),
        },
        &Limits {
            execution: Duration::from_secs(10),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(2),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &Cancellation::default(),
        RawCaptureOptions {
            stdout: NonZeroUsize::new(65536),
            stderr: NonZeroUsize::new(65536),
        },
    )
    .unwrap();
    assert!(result.process.failure.is_none(), "{:?}", result.process);
    assert!(result.process.cleanup.complete);
    assert!(result.stdout.unwrap().as_bytes().is_empty());
    (
        result.process.status.unwrap().code().unwrap(),
        String::from_utf8_lossy(result.stderr.unwrap().as_bytes()).into_owned(),
    )
}
#[test]
fn migration_ci_graph_workflow_uniqueness_current_graph_is_admitted() {
    let (status, diagnostic) = actual(&lane_workflow("quality"));
    assert_eq!(status, 0, "{diagnostic}");
    assert!(diagnostic.is_empty());
}
#[test]
fn migration_ci_graph_workflow_uniqueness_duplicate_jobs_refuse() {
    let source = fs::read_to_string(lane_workflow("quality")).unwrap();
    let temp = tempfile::tempdir().unwrap();
    for key in ["summary", "quality"] {
        let workflow = temp.path().join(format!("duplicate-{key}.yml"));
        fs::write(
            &workflow,
            format!("{source}\n  {key}:\n    name: decoy repeated owner\n"),
        )
        .unwrap();
        let (status, diagnostic) = actual(&workflow);
        assert_eq!(status, 2, "{diagnostic}");
        assert!(
            diagnostic.contains(&format!("duplicate workflow key '{key}'")),
            "{diagnostic}"
        );
    }
}
#[test]
fn migration_ci_graph_workflow_uniqueness_missing_summary_refuses() {
    let source = fs::read_to_string(lane_workflow("quality")).unwrap();
    assert_eq!(source.matches("\n  summary:\n").count(), 1);
    let temp = tempfile::tempdir().unwrap();
    let workflow = temp.path().join("missing-summary.yml");
    fs::write(
        &workflow,
        source.replace("\n  summary:\n", "\n  former_summary:\n"),
    )
    .unwrap();
    let (status, diagnostic) = actual(&workflow);
    assert_eq!(status, 2, "{diagnostic}");
    assert!(
        diagnostic.contains("exactly one summary job, found 0"),
        "{diagnostic}"
    );
}
