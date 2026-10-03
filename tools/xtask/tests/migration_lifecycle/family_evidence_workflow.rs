//! Finite real producer and workflow boundaries; never execute models/native artifacts.
use crate::{
    process,
    workflow_yaml::{self, Node},
};
use serde_json::Value as Json;
use std::{collections::BTreeMap, fs, num::NonZeroUsize, path::Path, time::Duration};

fn run(script: &str, root: &Path, environment: &[(&str, &str)]) -> process::RawProcessReport {
    let mut env = BTreeMap::from([(
        "PATH".into(),
        process::Value::Public("/usr/bin:/bin:/opt/homebrew/bin".into()),
    )]);
    for (key, value) in environment {
        env.insert((*key).into(), process::Value::Public((*value).into()));
    }
    let spec = process::ProcessSpec {
        executable: "/bin/bash".into(),
        cwd: root.into(),
        arguments: vec![
            process::Value::Public("-c".into()),
            process::Value::Public(script.into()),
        ],
        environment: env,
    };
    let report = process::supervise_raw(
        &spec,
        &process::Limits {
            execution: Duration::from_secs(5),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 65536,
            readiness: process::Readiness::None,
            completion: process::Completion::Exit,
        },
        &process::Cancellation::default(),
        process::RawCaptureOptions {
            stdout: NonZeroUsize::new(65536),
            stderr: NonZeroUsize::new(65536),
        },
    )
    .unwrap();
    assert!(report.process.cleanup.complete);
    report
}
fn field<'a>(node: &'a Node, key: &str) -> &'a str {
    node.get(key).and_then(Node::text).unwrap()
}
fn steps(node: &Node) -> &[Node] {
    let Node::Seq(items) = node.get("steps").unwrap() else {
        panic!("steps")
    };
    items
}
fn named<'a>(items: &'a [Node], key: &str, value: &str) -> (usize, &'a Node) {
    items
        .iter()
        .enumerate()
        .find(|(_, item)| item.get(key).and_then(Node::text) == Some(value))
        .unwrap()
}
fn expand(text: &str, attempt: &str, identity: &str) -> String {
    let mut expanded = text.to_owned();
    for (key, value) in [
        ("github.run_id", "123"),
        ("github.run_attempt", attempt),
        ("needs.build.outputs.identity", identity),
        ("inputs.pass_id", "repair-1"),
        ("matrix.shard_index", "0"),
    ] {
        expanded = expanded.replace(&format!("${{{{ {key} }}}}"), value);
    }
    assert!(!expanded.contains("${{"));
    expanded
}
fn workflow() -> Node {
    workflow_yaml::parse(
        &fs::read_to_string(
            super::support::repository().join(".github/workflows/llama-canary-family-pass.yml"),
        )
        .unwrap(),
    )
    .unwrap()
}
#[test]
fn parsed_producer_bound_artifacts_select_prior_siblings_and_never_foreign_identity() {
    let document = workflow();
    let jobs = document.get("jobs").unwrap();
    let build = jobs.get("build").unwrap();
    let (_, upload_package) = named(steps(build), "id", "upload_package");
    assert_eq!(
        field(build.get("outputs").unwrap(), "package"),
        field(upload_package.get("with").unwrap(), "name")
    );
    let family = jobs.get("family").unwrap();
    let aggregate = jobs.get("aggregate").unwrap();
    let (_, upload) = named(steps(family), "name", "Upload family evidence");
    let download = steps(aggregate)
        .iter()
        .find(|step| step.get("with").is_some_and(|v| v.get("pattern").is_some()))
        .unwrap();
    assert!(
        download
            .get("with")
            .unwrap()
            .get("merge-multiple")
            .is_none()
    );
    let identity = "a".repeat(64);
    let pattern = expand(
        field(download.get("with").unwrap(), "pattern"),
        "3",
        &identity,
    );
    let root = tempfile::tempdir().unwrap();
    for (attempt, id, expected) in [
        ("2", identity.as_str(), true),
        ("3", identity.as_str(), true),
        ("2", "foreign", false),
    ] {
        let name = expand(field(upload.get("with").unwrap(), "name"), attempt, id);
        let result = run(
            "case \"$NAME\" in $PATTERN) exit 0;; *) exit 1;; esac",
            root.path(),
            &[("NAME", &name), ("PATTERN", &pattern)],
        );
        assert_eq!(result.process.status.unwrap().success(), expected);
    }
    let inputs = document
        .get("on")
        .unwrap()
        .get("workflow_call")
        .unwrap()
        .get("inputs")
        .unwrap();
    assert!(inputs.get("feedback_pattern").is_none());
    assert!(inputs.get("feedback_build_pattern").is_none());
    // Every executable workflow step is inspected, rather than comments or arbitrary YAML bytes.
    for (_, job) in jobs.entries() {
        for step in steps(job) {
            if let Some(body) = step.get("run").and_then(Node::text) {
                assert!(!body.contains("canary-feedback"));
            }
        }
    }
}
#[test]
fn actual_family_and_aggregate_gates_refuse_failure_after_evidence_is_uploaded() {
    let document = workflow();
    let jobs = document.get("jobs").unwrap();
    let family = jobs.get("family").unwrap();
    let family_steps = steps(family);
    let (certify, cert) = named(family_steps, "id", "certify");
    assert_eq!(field(cert, "continue-on-error"), "true");
    let (upload, upload_step) = named(family_steps, "id", "upload_evidence");
    let (gate, gate_step) = named(
        family_steps,
        "name",
        "Require successful family certification",
    );
    assert!(certify < upload && upload < gate);
    assert!(gate_step.get("continue-on-error").is_none());
    assert!(field(upload_step, "if").contains("failure()"));
    assert!(field(upload_step, "if").contains("cancelled()"));
    assert_eq!(
        field(gate_step.get("env").unwrap(), "OUTCOME"),
        "${{ steps.certify.outcome }}"
    );
    let (_, aggregate) = named(steps(jobs.get("aggregate").unwrap()), "id", "aggregate");
    assert_eq!(
        field(aggregate.get("env").unwrap(), "FAMILY_RESULT"),
        "${{ needs.family.result }}"
    );
    let root = tempfile::tempdir().unwrap();
    let recorder = root.path().join("automation");
    fs::write(
        &recorder,
        "#!/bin/bash\nprintf '%s\\n' called >> \"$CALLS\"\n",
    )
    .unwrap();
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(&recorder, fs::Permissions::from_mode(0o755)).unwrap();
    let calls = root.path().join("calls");
    for outcome in ["success", "failure", "cancelled", "skipped", ""] {
        let result = run(
            field(gate_step, "run"),
            root.path(),
            &[("OUTCOME", outcome)],
        );
        assert_eq!(
            result.process.status.unwrap().success(),
            outcome == "success"
        );
        let result = run(
            field(aggregate, "run"),
            root.path(),
            &[
                ("FAMILY_RESULT", outcome),
                ("MESH_LLM_AUTOMATION_BIN", recorder.to_str().unwrap()),
                ("CALLS", calls.to_str().unwrap()),
                ("RUNNER_TEMP", root.path().to_str().unwrap()),
                ("IDENTITY", "finite-identity"),
                ("GITHUB_RUN_ID", "123"),
                ("GITHUB_RUN_ATTEMPT", "3"),
                ("CANARY_CONTROLLER_SHA", "finite-controller"),
                ("CANARY_MESH_SOURCE", ""),
            ],
        );
        assert_eq!(
            result.process.status.unwrap().success(),
            outcome == "success"
        );
        assert_eq!(fs::read_to_string(&calls).unwrap(), "called\n");
    }
}
#[test]
fn actual_battery_json_writers_emit_one_record_each_and_preserve_escaped_payload() {
    let source =
        fs::read_to_string(super::support::repository().join("scripts/skippy-family-battery.sh"))
            .unwrap();
    let root = tempfile::tempdir().unwrap();
    let results = root.path().join("results.jsonl");
    let manifest = root.path().join("manifest.json");
    fs::write(
        &manifest,
        r#"{"commands":[{"name":"chain","status":"pass","exit_code":0}]}"#,
    )
    .unwrap();
    let mut commands = Vec::new();
    for preceding in source
        .split(">> \"$RESULTS_JSONL\"")
        .take(source.matches(">> \"$RESULTS_JSONL\"").count())
    {
        let start = preceding
            .rfind("\n  jq ")
            .into_iter()
            .chain(preceding.rfind("\n    jq "))
            .max()
            .unwrap();
        commands.push(format!(
            "{} >> \"$RESULTS_JSONL\"",
            preceding[start..].trim()
        ));
    }
    assert_eq!(
        commands.len(),
        7,
        "new writers require finite fixture inputs"
    );
    let script = format!("set -euo pipefail\n{}", commands.join("\n"));
    let note = "a note with \"quotes\" and\na second line";
    let result = run(
        &script,
        root.path(),
        &[
            ("RESULTS_JSONL", results.to_str().unwrap()),
            ("manifest_path", manifest.to_str().unwrap()),
            ("name", "environment-preflight"),
            ("family", "finite-family"),
            ("model_id", "finite-model"),
            ("status", "pass"),
            ("outcome", "pass"),
            ("note", note),
            ("target", "finite.gguf"),
            ("scan_log", "scan.log"),
            ("source_revision", "finite-source"),
            ("split_layer", "1"),
            ("model_size_bytes", "1"),
            ("activation_width", "1"),
            ("startup_timeout", "1"),
            ("cert_timeout", "1"),
            ("elapsed_seconds", "1"),
            ("native_mtp", "0"),
            ("exit_code", "0"),
            ("model_class", "embedding"),
            ("smoke_lane", "embedding-smoke"),
            ("oracle_lane", "embedding-oracle"),
            ("log_path", "finite.log"),
            ("oracle_requested", "1"),
        ],
    );
    assert!(result.process.status.unwrap().success(), "{:?}", result);
    let bytes = fs::read_to_string(results).unwrap();
    let rows: Vec<Json> = bytes
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(rows.len(), commands.len());
    assert!(rows.iter().all(Json::is_object));
    assert_eq!(rows[0]["outcomes"][0]["note"], note);
    assert_eq!(rows[0]["exit_code"], 0);
    assert_eq!(rows[3]["outcomes"][0]["name"], "chain");
    assert_eq!(rows[6]["workload_class"], "embedding");
    assert_eq!(rows[6]["outcomes"].as_array().unwrap().len(), 2);
}
