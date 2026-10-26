use super::*;
use crate::ci_validation::lane_results::workflow_yaml;
use std::{fs, path::Path};
fn actual() -> BTreeMap<String, Node> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    BTreeMap::from([(
        "ci-linux-runtime-slice.yml".into(),
        workflow_yaml::parse(
            &fs::read_to_string(root.join(".github/workflows/ci-linux-runtime-slice.yml")).unwrap(),
        )
        .unwrap(),
    )])
}
fn steps(workflows: &mut BTreeMap<String, Node>) -> &mut Vec<Node> {
    let document = workflows.get_mut("ci-linux-runtime-slice.yml").unwrap();
    let job = h::mutable(h::mutable(document, "jobs"), "linux_runtime");
    let Node::Seq(steps) = h::mutable(job, "steps") else {
        panic!("steps required")
    };
    steps
}
fn step<'a>(workflows: &'a mut BTreeMap<String, Node>, name: &str) -> &'a mut Node {
    steps(workflows)
        .iter_mut()
        .find(|step| super::super::field(step, "name") == Some(name))
        .unwrap()
}
fn replace(node: &mut Node, key: &str, value: &str) {
    *h::mutable(node, key) = Node::Scalar(value.into());
}
#[test]
fn actual_cpu_reporter_gate_retains_model_bundle_and_failure_evidence() {
    check(&actual()).unwrap();
}
#[test]
fn every_native_reporter_stage_rejects_unqualified_backend_or_success_only_upload() {
    for name in [
        "Restore runtime-event gate model",
        "Run native runtime-event gate",
        "Upload native runtime-event gate evidence",
    ] {
        let mut workflows = actual();
        replace(step(&mut workflows, name), "if", "${{ success() }}");
        assert!(check(&workflows).is_err(), "{name}");
    }
}
#[test]
fn restore_model_identity_and_original_event_cadence_cannot_be_substituted() {
    for (key, value) in [
        ("model_manifest", "untrusted.json"),
        ("model_artifact_id", "another-model"),
        ("model_cadence", "manual"),
    ] {
        let mut workflows = actual();
        replace(
            h::mutable(
                step(&mut workflows, "Restore runtime-event gate model"),
                "with",
            ),
            key,
            value,
        );
        assert!(check(&workflows).is_err());
    }
}
#[test]
fn native_reporter_requires_restored_bundle_parent_model_and_same_evidence_path() {
    for (before, after) in [
        ("$(dirname \"$RUNTIME_DIR\")", "$RUNTIME_DIR"),
        ("$MODEL_PATH", "unrelated-model.gguf"),
        (
            "--evidence runtime-events-native-evidence.txt",
            "--evidence unrelated.txt",
        ),
    ] {
        let mut workflows = actual();
        let gate = step(&mut workflows, "Run native runtime-event gate");
        let run = super::super::field(gate, "run")
            .unwrap()
            .replace(before, after);
        replace(gate, "run", &run);
        assert!(check(&workflows).is_err());
    }
}
#[test]
fn missing_addressable_producer_or_model_before_execution_rejects() {
    for (name, id) in [
        ("Prepare immutable Linux native runtime", "unbound"),
        ("Restore runtime-event gate model", "unbound"),
    ] {
        let mut workflows = actual();
        replace(step(&mut workflows, name), "id", id);
        assert!(check(&workflows).is_err());
    }
    let mut workflows = actual();
    let steps = steps(&mut workflows);
    let restore = steps
        .iter()
        .position(|step| {
            super::super::field(step, "name") == Some("Restore runtime-event gate model")
        })
        .unwrap();
    let execute = steps
        .iter()
        .position(|step| super::super::field(step, "name") == Some("Run native runtime-event gate"))
        .unwrap();
    steps.swap(restore, execute);
    assert!(check(&workflows).is_err());
}
#[test]
fn evidence_is_required_and_uploaded_after_failure_without_masking_execution() {
    for (key, value) in [("path", "unrelated.txt"), ("if-no-files-found", "ignore")] {
        let mut workflows = actual();
        replace(
            h::mutable(
                step(&mut workflows, "Upload native runtime-event gate evidence"),
                "with",
            ),
            key,
            value,
        );
        assert!(check(&workflows).is_err());
    }
    for skipped in [
        "if false; then\nscripts/ci-runtime-events-native-gate.sh\nfi",
        "echo scripts/ci-runtime-events-native-gate.sh",
        "exit 0\nscripts/ci-runtime-events-native-gate.sh",
    ] {
        let mut workflows = actual();
        replace(
            step(&mut workflows, "Run native runtime-event gate"),
            "run",
            skipped,
        );
        assert!(check(&workflows).is_err());
    }
}
#[test]
fn direct_argument_order_changes_preserve_the_same_native_admission() {
    let mut workflows = actual();
    let gate = step(&mut workflows, "Run native runtime-event gate");
    let mut lines: Vec<_> = super::super::field(gate, "run")
        .unwrap()
        .lines()
        .map(str::to_owned)
        .collect();
    let first = lines.remove(0);
    lines.reverse();
    let mut owned: Vec<_> = std::iter::once(first.trim().trim_end_matches('\\').trim().to_owned())
        .chain(
            lines
                .into_iter()
                .map(|line| line.trim().trim_end_matches('\\').trim().to_owned()),
        )
        .collect();
    let last = owned.len() - 1;
    for line in &mut owned[..last] {
        line.push_str(" \\");
    }
    replace(gate, "run", &owned.join("\n"));
    check(&workflows).unwrap();
}

#[test]
fn cached_or_prepared_runtime_requires_both_verified_producer_bindings() {
    for (name, key, value) in [
        ("Verify restored Linux CPU runtime", "id", "unverified"),
        ("Verify restored Linux CPU runtime", "if", "true"),
        ("Prepare immutable Linux native runtime", "if", "true"),
    ] {
        let mut workflows = actual();
        replace(step(&mut workflows, name), key, value);
        assert!(check(&workflows).is_err());
    }
    for (key, value) in [
        (
            "RUNTIME_DIR",
            "${{ steps.native_runtime.outputs.runtime_dir }}",
        ),
        ("MODEL_PATH", "unrelated.gguf"),
    ] {
        let mut workflows = actual();
        replace(
            h::mutable(step(&mut workflows, "Run native runtime-event gate"), "env"),
            key,
            value,
        );
        assert!(check(&workflows).is_err());
    }
}

#[test]
fn cached_runtime_cannot_drop_or_swap_exact_planned_row_expectations() {
    for (before, after) in [
        ("--expected-backend \"$EXPECTED_BACKEND\"", ""),
        ("--expected-target \"$EXPECTED_TARGET\"", ""),
        (
            "--expected-backend \"$EXPECTED_BACKEND\"",
            "--expected-backend cpu",
        ),
        (
            "--expected-target \"$EXPECTED_TARGET\"",
            "--expected-target \"$EXPECTED_BACKEND\"",
        ),
    ] {
        let mut workflows = actual();
        let verification = step(&mut workflows, "Verify restored Linux CPU runtime");
        let run = super::super::field(verification, "run")
            .unwrap()
            .replace(before, after);
        replace(verification, "run", &run);
        assert!(check(&workflows).is_err(), "{before}");
    }
}
