//! Execute the actual catalog-owned jq step with finite, synthetic catalogs.
#![cfg(unix)]
use crate::process::{
    self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value,
};
use crate::workflow_yaml;
use serde_json::{Value as Json, json};
use std::{collections::BTreeMap, fs, path::Path, time::Duration};
fn row() -> Json {
    json!({"id":"linux-cpu", "platform":"linux", "architecture":"amd64", "backend":"cpu", "target":"x86_64-unknown-linux-gnu", "build_dir":"synthetic build directory", "container_image":format!("ghcr.io/mesh-llm/runner@sha256:{}", "a".repeat(64)), "toolchain_epoch":"synthetic epoch", "verify_backend":"public"})
}
fn source_step() -> String {
    let source = include_str!("../../../../.github/workflows/ci-pr-canary-lane.yml");
    let document = workflow_yaml::parse(source).unwrap();
    let steps = document
        .get("jobs")
        .unwrap()
        .get("plan")
        .unwrap()
        .get("steps")
        .unwrap();
    let workflow_yaml::Node::Seq(steps) = steps else {
        panic!("steps sequence")
    };
    steps
        .iter()
        .find(|step| step.get("id").and_then(workflow_yaml::Node::text) == Some("source_matrix"))
        .unwrap()
        .get("run")
        .and_then(workflow_yaml::Node::text)
        .unwrap()
        .into()
}
fn execute(catalog: Json) -> (process::ProcessReport, BTreeMap<String, Json>) {
    let directory = tempfile::tempdir().unwrap();
    fs::create_dir(directory.path().join("ci")).unwrap();
    fs::write(
        directory.path().join("ci/slices.yml"),
        serde_json::to_vec(&catalog).unwrap(),
    )
    .unwrap();
    let output = directory.path().join("github-output");
    let spec = ProcessSpec {
        executable: "/bin/bash".into(),
        arguments: vec![
            Value::Public("-c".into()),
            Value::Public(source_step().into()),
        ],
        cwd: directory.path().into(),
        environment: BTreeMap::from([
            (
                "PATH".into(),
                Value::Public("/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin".into()),
            ),
            (
                "GITHUB_OUTPUT".into(),
                Value::Public(output.to_str().unwrap().into()),
            ),
        ]),
    };
    let report = process::supervise(
        &spec,
        &Limits {
            execution: Duration::from_secs(10),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 32768,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap();
    assert!(report.cleanup.complete, "{report:?}");
    let matrices = fs::read_to_string(output)
        .unwrap_or_default()
        .lines()
        .map(|line| {
            let (key, value) = line.split_once('=').expect("matrix output binding");
            (
                key.into(),
                serde_json::from_str(value).expect("matrix JSON"),
            )
        })
        .collect();
    (report, matrices)
}
#[test]
fn actual_catalog_step_propagates_row_identity_without_hardcoded_build_inputs() {
    for changed in [false, true] {
        let mut selected = row();
        if changed {
            selected["build_dir"] = "different directory".into();
            selected["toolchain_epoch"] = "different epoch".into();
            selected["container_image"] =
                format!("ghcr.io/mesh-llm/new-runner@sha256:{}", "b".repeat(64)).into();
        }
        let (report, output) = execute(json!({"runtime_rows":[selected.clone()]}));
        assert!(report.success(), "{report:?}");
        assert_eq!(output["runtime_matrix"], json!([selected]));
        assert_eq!(
            output["host_matrix"],
            json!([{"id":"linux-cpu","platform":"linux","architecture":"amd64"}])
        );
        assert_eq!(
            output["product_matrix"],
            json!([{"id":"linux-cpu","platform":"linux","architecture":"amd64","backend":"cpu"}])
        );
    }
}
#[test]
fn actual_catalog_step_rejects_absent_duplicate_and_wrong_or_unbound_cpu_rows() {
    let mut cases = vec![
        json!({"runtime_rows":[]}),
        json!({"runtime_rows":[row(),row()]}),
    ];
    for (key, value) in [
        ("backend", "cuda"),
        ("architecture", "arm64"),
        ("platform", "macos"),
        ("target", "wrong-target"),
        ("verify_backend", "private"),
        ("build_dir", ""),
        ("toolchain_epoch", ""),
        ("container_image", "ghcr.io/mesh-llm/runner:latest"),
    ] {
        let mut invalid = row();
        invalid[key] = value.into();
        cases.push(json!({"runtime_rows":[invalid]}));
    }
    for catalog in cases {
        let (report, output) = execute(catalog.clone());
        assert!(!report.success(), "{catalog}: {report:?}");
        assert!(
            output.is_empty(),
            "failed selector must publish no partial matrix"
        );
    }
}
#[test]
fn actual_runner_routing_admits_canary_changes_and_not_unrelated_docs() {
    let source = fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../.github/actions/compute-changes/derive-outputs.sh"),
    )
    .unwrap();
    let begin = source.find("RUNNER_CONTRACT_REQUIRED=\"false\"").unwrap();
    let end = source[begin..].find("# Determine docs_only:").unwrap() + begin;
    let script = format!(
        "set -euo pipefail\n{}\nprintf '%s' \"$RUNNER_CONTRACT_REQUIRED\"\n",
        &source[begin..end]
    );
    for (changed, expected) in [
        (".github/workflows/pr_ci_canary.yml", "true"),
        ("docs/MESHES.md", "false"),
    ] {
        let directory = tempfile::tempdir().unwrap();
        let spec = ProcessSpec {
            executable: "/bin/bash".into(),
            arguments: vec![
                Value::Public("-c".into()),
                Value::Public(script.clone().into()),
            ],
            cwd: directory.path().into(),
            environment: BTreeMap::from([
                ("PATH".into(), Value::Public("/usr/bin:/bin".into())),
                ("EVENT_NAME".into(), Value::Public("pull_request".into())),
                ("CHANGED_FILES".into(), Value::Public(changed.into())),
            ]),
        };
        let report = process::supervise(
            &spec,
            &Limits {
                execution: Duration::from_secs(5),
                graceful_shutdown: Duration::from_secs(1),
                forced_shutdown: Duration::from_secs(1),
                retained_bytes_per_stream: 32768,
                readiness: Readiness::None,
                completion: Completion::Exit,
            },
            &Cancellation::default(),
            OutputFiles::default(),
        )
        .unwrap();
        assert!(report.success() && report.cleanup.complete, "{report:?}");
        assert_eq!(report.stdout.bytes_retained, expected.as_bytes());
    }
}
#[test]
fn actual_current_catalog_selects_exactly_one_public_amd64_cpu_row() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../ci/slices.yml");
    let catalog: Json = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
    let selected = catalog["runtime_rows"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|row| row["id"] == "linux-cpu")
        .collect::<Vec<_>>();
    assert_eq!(selected.len(), 1);
    let (report, output) = execute(catalog.clone());
    assert!(report.success(), "{report:?}");
    assert_eq!(output["runtime_matrix"], json!([selected[0]]));
}
