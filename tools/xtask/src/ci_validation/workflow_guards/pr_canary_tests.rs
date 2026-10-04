use super::super::workflow_yaml;
use super::*;
use std::path::Path;
fn actual() -> BTreeMap<String, Node> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    ["pr_ci_canary.yml", "ci-pr-canary-lane.yml"]
        .into_iter()
        .chain(SLICES.iter().map(|(_, name)| *name))
        .map(|name| {
            (
                name.into(),
                workflow_yaml::parse(
                    &std::fs::read_to_string(root.join(".github/workflows").join(name)).unwrap(),
                )
                .unwrap(),
            )
        })
        .collect()
}
fn entry(node: &mut Node, key: &str, value: Node) {
    let Node::Map(entries) = node else {
        panic!("expected mapping")
    };
    if let Some((_, old)) = entries.iter_mut().find(|(name, _)| name == key) {
        *old = value;
    } else {
        entries.push((key.into(), value));
    }
}
fn scalar(value: &str) -> Node {
    Node::Scalar(value.into())
}
#[test]
fn actual_pr_canary_retains_credential_free_six_workflow_closure() {
    check(&actual()).unwrap();
}
#[test]
fn unrelated_labels_cannot_cancel_or_start_active_diagnostic() {
    for change in 0..4 {
        let mut workflows = actual();
        let node = workflows.get_mut("pr_ci_canary.yml").unwrap();
        match change {
            0 => entry(
                h::mutable(node, "concurrency"),
                "group",
                scalar("pr-${{ github.event.pull_request.number }}"),
            ),
            1 => entry(
                h::mutable(h::mutable(node, "jobs"), "canary"),
                "if",
                scalar("true"),
            ),
            2 => entry(
                h::mutable(h::mutable(node, "on"), "pull_request"),
                "paths",
                Node::Seq(vec![scalar("crates/**")]),
            ),
            _ => entry(
                h::mutable(h::mutable(node, "on"), "pull_request"),
                "types",
                Node::Seq(vec![scalar("synchronize")]),
            ),
        }
        assert!(check(&workflows).is_err(), "change {change}");
    }
}
#[test]
fn pr_source_cannot_replace_protected_lane_or_runner_policy() {
    for change in 0..3 {
        let mut workflows = actual();
        if change == 0 {
            let node = workflows.get_mut("pr_ci_canary.yml").unwrap();
            entry(
                h::mutable(h::mutable(node, "jobs"), "canary"),
                "uses",
                scalar("./.github/workflows/ci-pr-canary-lane.yml"),
            );
        } else {
            let lane = workflows.get_mut("ci-pr-canary-lane.yml").unwrap();
            let inputs = h::mutable(h::mutable(h::mutable(lane, "jobs"), "hosts"), "with");
            if change == 1 {
                entry(
                    inputs,
                    "policy_source_sha",
                    scalar("${{ inputs.merge_sha }}"),
                );
            } else {
                entry(inputs, "force_hosted", scalar("false"));
            }
        }
        assert!(check(&workflows).is_err());
    }
}
#[test]
fn slice_and_matrix_handoffs_cannot_be_dropped_or_substituted() {
    for change in 0..4 {
        let mut workflows = actual();
        let lane = workflows.get_mut("ci-pr-canary-lane.yml").unwrap();
        let hosts = h::mutable(h::mutable(lane, "jobs"), "hosts");
        match change {
            0 => entry(
                hosts,
                "uses",
                scalar("./.github/workflows/ci-linux-lane.yml"),
            ),
            1 => entry(h::mutable(hosts, "with"), "hosts_matrix", scalar("[]")),
            2 => entry(
                h::mutable(hosts, "with"),
                "source_sha",
                scalar("${{ inputs.head_sha }}"),
            ),
            _ => entry(
                h::mutable(h::mutable(h::mutable(lane, "jobs"), "plan"), "outputs"),
                "host_matrix",
                scalar("[]"),
            ),
        }
        assert!(check(&workflows).is_err());
    }
}
#[test]
fn every_slice_requires_protected_credential_free_policy_checkout() {
    for (_, name) in SLICES {
        let mut workflows = actual();
        let node = workflows.get_mut(*name).unwrap();
        let Node::Map(jobs) = h::mutable(node, "jobs") else {
            unreachable!()
        };
        for (_, job) in jobs {
            if let Some(Node::Seq(steps)) = match job {
                Node::Map(entries) => entries
                    .iter_mut()
                    .find(|(key, _)| key == "steps")
                    .map(|(_, value)| value),
                _ => None,
            } {
                for step in steps {
                    if field(step, "uses")
                        .is_some_and(|value| value.starts_with("actions/checkout@"))
                        && step.get("with").and_then(|inputs| field(inputs, "ref"))
                            == Some(
                                "${{ inputs.policy_source_sha || github.event.repository.default_branch }}",
                            )
                    {
                        entry(
                            h::mutable(step, "with"),
                            "ref",
                            scalar("${{ inputs.source_sha }}"),
                        );
                    }
                }
            }
        }
        assert!(check(&workflows).is_err(), "{name}");
    }
}
#[test]
fn every_reachable_workflow_rejects_credentials_write_or_privileged_execution() {
    for name in actual().keys() {
        for (key, value) in [
            ("permissions", scalar("write-all")),
            ("secrets", scalar("inherit")),
            ("environment", scalar("production")),
            ("id-token", scalar("write")),
            (
                "runs-on",
                Node::Seq(vec![scalar("self-hosted"), scalar("linux")]),
            ),
            ("use_depot", scalar("true")),
        ] {
            let mut workflows = actual();
            entry(workflows.get_mut(name).unwrap(), key, value);
            assert!(check(&workflows).is_err(), "{name}: {key}");
        }
    }
}

#[test]
fn immutable_merge_identity_and_bounded_summary_cannot_be_dropped() {
    let source = include_str!("../../../../../.github/workflows/ci-pr-canary-lane.yml");
    for (from, to) in [
        ("name: Canary / CI", "name: Unbound result"),
        (
            "[plan, ui_artifact, hosts, native_runtime, product]",
            "[plan, hosts]",
        ),
        ("[[ \"$PRODUCT_RESULT\" == success ]]", "true"),
        ("native runtime-event gate", "unqualified gate"),
        ("Linux lane orchestration", "unqualified orchestration"),
        ("refs/pull/${PR_NUMBER}/head", "refs/heads/main"),
        (
            "ref: ${{ inputs.merge_sha }}",
            "ref: ${{ inputs.head_sha }}",
        ),
        (
            "git merge-base --is-ancestor \"$HEAD_SHA\" \"$MERGE_SHA\"",
            "true",
        ),
        ("parent_count < 3", "parent_count < 2"),
        ("-- ci/ownership.yml ci/slices.yml", "-- unrelated.yml"),
        ("PR_NUMBER: ${{ inputs.pr_number }}", "PR_NUMBER: 1"),
    ] {
        assert!(source.contains(from));
        let mut workflows = actual();
        workflows.insert(
            "ci-pr-canary-lane.yml".into(),
            workflow_yaml::parse(&source.replace(from, to)).unwrap(),
        );
        assert!(check(&workflows).is_err(), "{from}");
    }
}
