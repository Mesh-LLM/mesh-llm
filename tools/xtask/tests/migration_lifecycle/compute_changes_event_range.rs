//! Execute the production event step against private, real Git histories.
use super::Fixture;
use crate::workflow_yaml::{self, Node};
use std::{collections::BTreeSet, fs};

fn finish(fixture: Fixture) {
    fixture
        .directory
        .close()
        .expect("owned event Git fixture deletion failed");
}

fn changed_files(fixture: &Fixture, base: &str, head: &str, event: &str) -> BTreeSet<String> {
    let source = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../.github/actions/compute-changes/action.yml"
    ));
    let tree = workflow_yaml::parse(source).unwrap();
    let Node::Seq(steps) = tree.get("runs").unwrap().get("steps").unwrap() else {
        panic!("compute action steps")
    };
    let matching: Vec<_> = steps
        .iter()
        .filter(|step| step.get("id").and_then(Node::text) == Some("files"))
        .collect();
    assert_eq!(matching.len(), 1, "unique production file-list step");
    let mut script = matching[0]
        .get("run")
        .and_then(Node::text)
        .unwrap()
        .to_owned();
    // Bind the three GitHub inputs as process data. Redirect only the fixed
    // scratch location; the event branches and Git commands remain production.
    for (expression, variable) in [
        ("${{ inputs.event_name }}", "${EVENT_NAME}"),
        ("${{ inputs.base_sha }}", "${BASE_SHA}"),
        ("${{ inputs.head_sha }}", "${HEAD_SHA}"),
    ] {
        script = script.replace(expression, variable);
    }
    assert!(!script.contains("${{"), "unbound GitHub expression");
    assert!(script.contains("/tmp/changed_files.txt"));
    script = script.replace("/tmp/changed_files.txt", "\"$CHANGED_FILES_OUTPUT\"");
    let destination = fixture.scratch.join("changed files.txt");
    let output_path = destination.to_str().unwrap();
    let output = fixture.execute(
        "/bin/bash".into(),
        vec!["-euo".into(), "pipefail".into(), "-c".into(), script],
        &[
            ("BASE_SHA", base),
            ("HEAD_SHA", head),
            ("EVENT_NAME", event),
            ("CHANGED_FILES_OUTPUT", output_path),
        ],
    );
    assert_eq!(output, fs::read(destination).unwrap());
    String::from_utf8(output)
        .unwrap()
        .lines()
        .map(str::to_owned)
        .collect()
}

#[test]
fn push_diff_includes_earlier_commits_and_uses_payload_head_not_checkout_head() {
    let f = Fixture::new();
    f.write("README.md", "base");
    let base = f.commit();
    f.write("scripts/package-native-runtime.sh", "earlier native change");
    f.commit();
    f.write("sdk/node/index.js", "last sdk change");
    let head = f.commit();
    f.git(&["checkout", "--quiet", "--detach", &base]);
    assert_eq!(
        changed_files(&f, &base, &head, "push"),
        BTreeSet::from([
            "scripts/package-native-runtime.sh".into(),
            "sdk/node/index.js".into(),
        ])
    );
    finish(f);
}

#[test]
fn push_unknown_or_zero_event_history_and_manual_dispatch_fail_open() {
    let f = Fixture::new();
    f.write("README.md", "base");
    let head = f.commit();
    let zeros = "0".repeat(40);
    let missing = "1".repeat(40);
    for (base, candidate_head, event) in [
        (zeros.as_str(), head.as_str(), "push"),
        ("", head.as_str(), "push"),
        (head.as_str(), "", "push"),
        (missing.as_str(), head.as_str(), "push"),
        (head.as_str(), missing.as_str(), "push"),
        ("", "", "workflow_dispatch"),
    ] {
        assert_eq!(
            changed_files(&f, base, candidate_head, event),
            BTreeSet::from(["__force_all__".into()]),
            "{event} {base} {candidate_head}"
        );
    }
    finish(f);
}
