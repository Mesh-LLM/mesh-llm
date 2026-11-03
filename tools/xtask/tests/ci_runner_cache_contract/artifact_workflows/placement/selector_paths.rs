//! Track reusable callers that can reach the centralized selector.

use super::super as workflows;
use super::{caller_census, support};
use std::collections::BTreeSet;

fn selector_paths() -> BTreeSet<String> {
    let seeds = caller_census::direct_workflow_callers(&support::root())
        .into_iter()
        .map(|(file, _, _)| file)
        .collect();
    workflows::reusable_paths::callers(&seeds)
}

fn only_main(branches: &workflows::Node) -> bool {
    match branches {
        workflows::Node::Scalar(value) => value == "main",
        workflows::Node::Seq(values) => values.len() == 1 && values[0].text() == Some("main"),
        workflows::Node::Map(_) => false,
    }
}

fn allowed_direct_default(
    edge: &str,
    caller: &workflows::Node,
    inputs: &workflows::Node,
    supplied: Option<&str>,
) -> bool {
    if supplied.is_some() {
        return false;
    }
    let event = match edge {
        "main_linux.yml/lane -> ci-linux-lane.yml"
        | "main_quality.yml/lane -> ci-quality-lane.yml" => "push",
        "pr_linux.yml/lane -> ci-linux-lane.yml@main"
        | "pr_quality.yml/lane -> ci-quality-lane.yml@main" => "pull_request",
        _ => return false,
    };
    let Some(triggers) = caller.get("on") else {
        return false;
    };
    let Some(trigger) = triggers.get(event) else {
        return false;
    };
    let Some(input) = inputs.get("use_depot") else {
        return false;
    };
    triggers.entries().len() == 1
        && (event != "push" || trigger.get("branches").is_some_and(only_main))
        && workflows::text(input, "required") == Some("false")
        && workflows::text(input, "type") == Some("boolean")
        && workflows::text(input, "default") == Some("false")
}

fn standard_lane(file: &str) -> bool {
    matches!(
        file,
        "ci-linux-lane.yml"
            | "ci-macos-lane.yml"
            | "ci-windows-lane.yml"
            | "ci-quality-lane.yml"
            | "ci-website-lane.yml"
    )
}

/// Standard lane calls must carry the same source and trust context into a
/// selected slice. Merely supplying a nonempty expression is not enough.
fn lane_forwarding_mismatches(
    file: &str,
    caller: &workflows::Node,
    inputs: Option<&workflows::Node>,
) -> Vec<&'static str> {
    if !standard_lane(file) {
        return Vec::new();
    }
    [
        ("source_sha", "${{ inputs.source_sha }}"),
        ("original_event_name", "${{ inputs.original_event_name }}"),
        (
            "force_hosted",
            "${{ toJSON(fromJson(inputs.lane_plan_json).signals.force_hosted) != 'false' }}",
        ),
        ("use_depot", "${{ inputs.use_depot }}"),
    ]
    .into_iter()
    .filter_map(|(key, expected)| {
        let declared = inputs.is_some_and(|inputs| inputs.get(key).is_some());
        if key == "use_depot" && !declared {
            return None;
        }
        (!declared || workflows::input(caller, key) != Some(expected)).then_some(key)
    })
    .collect()
}

fn entry_event(file: &str) -> Option<&'static str> {
    match file {
        "main_linux.yml" | "main_macos.yml" | "main_quality.yml" | "main_website.yml"
        | "main_windows.yml" => Some("push"),
        "pr_linux.yml" | "pr_macos.yml" | "pr_quality.yml" | "pr_website.yml"
        | "pr_windows.yml" => Some("pull_request"),
        _ => None,
    }
}

fn reviewed_value_path(file: &str, job: &str) -> bool {
    standard_lane(file)
        || (entry_event(file).is_some() && job == "lane")
        || matches!(
            (file, job),
            (
                "ci-pr-canary-lane.yml",
                "ui_artifact" | "hosts" | "native_runtime" | "product"
            ) | ("native-sdk-artifact.yml", "produce_linux_static_abi")
                | (
                    "release.yml",
                    "release_ui" | "build_native_sdk_runtime" | "build_swift_sdk_artifact"
                )
                | ("pr_ci_canary.yml", "canary")
        )
}

/// Main and PR entrypoints bind the protected plan and the original event to
/// each lane; a reusable workflow's own event would be `workflow_call`.
fn entry_forwarding_mismatches(
    file: &str,
    document: &workflows::Node,
    caller: &workflows::Node,
    inputs: Option<&workflows::Node>,
) -> Vec<&'static str> {
    let Some(event) = entry_event(file) else {
        return Vec::new();
    };
    let mut mismatches = Vec::new();
    let triggers = document.get("on");
    if !triggers.is_some_and(|on| {
        on.entries().len() == 1
            && on.get(event).is_some_and(|trigger| {
                event != "push" || trigger.get("branches").is_some_and(only_main)
            })
    }) {
        mismatches.push("trigger");
    }
    for (key, expected) in [
        ("source_sha", "${{ needs.plan.outputs.source_sha }}"),
        ("original_event_name", event),
        ("lane_plan_json", "${{ needs.plan.outputs.lane_plan }}"),
        ("plan_digest", "${{ needs.plan.outputs.plan_digest }}"),
    ] {
        if !inputs.is_some_and(|inputs| inputs.get(key).is_some())
            || workflows::input(caller, key) != Some(expected)
        {
            mismatches.push(key);
        }
    }
    mismatches
}

/// Canary, release, and nested SDK edges have intentionally distinct trust
/// boundaries from the ordinary lane graph.
fn special_forwarding_mismatches(
    file: &str,
    job: &str,
    document: &workflows::Node,
    caller: &workflows::Node,
    inputs: Option<&workflows::Node>,
) -> Vec<&'static str> {
    let expected: &[(&str, &str)] = match (file, job) {
        ("ci-pr-canary-lane.yml", "ui_artifact" | "hosts" | "native_runtime" | "product") => &[
            ("source_sha", "${{ inputs.merge_sha }}"),
            ("original_event_name", "pull_request"),
            ("force_hosted", "true"),
            ("use_depot", "false"),
        ],
        ("native-sdk-artifact.yml", "produce_linux_static_abi") => &[
            ("source_sha", "${{ inputs.source_sha }}"),
            ("original_event_name", "${{ inputs.original_event_name }}"),
            ("force_hosted", "${{ inputs.force_hosted }}"),
            ("use_depot", "${{ inputs.use_depot }}"),
        ],
        ("release.yml", "release_ui" | "build_native_sdk_runtime" | "build_swift_sdk_artifact") => {
            &[
                ("source_sha", "${{ needs.metadata.outputs.source_sha }}"),
                ("original_event_name", "${{ github.event_name }}"),
                ("force_hosted", "true"),
                ("use_depot", "${{ inputs.use_depot == true }}"),
            ]
        }
        ("pr_ci_canary.yml", "canary") => &[
            ("merge_sha", "${{ github.sha }}"),
            ("head_sha", "${{ github.event.pull_request.head.sha }}"),
            ("base_sha", "${{ github.event.pull_request.base.sha }}"),
            ("draft", "${{ github.event.pull_request.draft || false }}"),
            ("pr_number", "${{ github.event.pull_request.number }}"),
        ],
        _ => return Vec::new(),
    };
    let mut mismatches = Vec::new();
    if file == "pr_ci_canary.yml"
        && !document
            .get("on")
            .is_some_and(|on| on.entries().len() == 1 && on.get("pull_request").is_some())
    {
        mismatches.push("trigger");
    }
    for &(key, value) in expected {
        let declared = inputs.is_some_and(|inputs| inputs.get(key).is_some());
        if key == "use_depot" && !declared {
            continue;
        }
        if !declared || workflows::input(caller, key) != Some(value) {
            mismatches.push(key);
        }
    }
    mismatches
}

fn missing_forwarding(paths: &BTreeSet<String>) -> Vec<String> {
    let mut missing = Vec::new();
    for edge in paths {
        let (origin, target) = edge.split_once(" -> ").unwrap();
        let (file, job) = origin.split_once('/').unwrap();
        let target = target.split('@').next().unwrap();
        let called = workflows::document(target);
        let inputs = called
            .get("on")
            .and_then(|on| on.get("workflow_call"))
            .and_then(|call| call.get("inputs"));
        let caller_document = workflows::document(file);
        let caller = workflows::job(&caller_document, job);
        if !reviewed_value_path(file, job) {
            missing.push(format!("{edge}: no value contract"));
        }
        for key in [
            "force_hosted",
            "original_event_name",
            "source_sha",
            "use_depot",
        ] {
            if inputs.is_some_and(|inputs| inputs.get(key).is_some())
                && workflows::input(caller, key).is_none_or(|value| value.trim().is_empty())
            {
                if key == "use_depot"
                    && inputs.is_some_and(|inputs| {
                        allowed_direct_default(
                            edge,
                            &caller_document,
                            inputs,
                            workflows::input(caller, key),
                        )
                    })
                {
                    continue;
                }
                missing.push(format!("{edge}: {key}"));
            }
        }
        missing.extend(
            lane_forwarding_mismatches(file, caller, inputs)
                .into_iter()
                .map(|key| format!("{edge}: incorrect {key}")),
        );
        missing.extend(
            entry_forwarding_mismatches(file, &caller_document, caller, inputs)
                .into_iter()
                .map(|key| format!("{edge}: incorrect {key}")),
        );
        missing.extend(
            special_forwarding_mismatches(file, job, &caller_document, caller, inputs)
                .into_iter()
                .map(|key| format!("{edge}: incorrect {key}")),
        );
    }
    missing
}

#[test]
fn entrypoint_rejects_nonempty_source_event_plan_and_trigger_substitutions() {
    let source = workflows::source("pr_linux.yml");
    let called = workflows::document("ci-linux-lane.yml");
    let inputs = called
        .get("on")
        .and_then(|on| on.get("workflow_call"))
        .and_then(|call| call.get("inputs"));
    let original = workflows::workflow_yaml::parse(&source).unwrap();
    assert!(
        entry_forwarding_mismatches(
            "pr_linux.yml",
            &original,
            workflows::job(&original, "lane"),
            inputs,
        )
        .is_empty()
    );
    for (before, after, expected) in [
        (
            "      source_sha: ${{ needs.plan.outputs.source_sha }}",
            "      source_sha: ${{ github.sha }}",
            "source_sha",
        ),
        (
            "      original_event_name: pull_request",
            "      original_event_name: push",
            "original_event_name",
        ),
        (
            "      plan_digest: ${{ needs.plan.outputs.plan_digest }}",
            "      plan_digest: ${{ github.sha }}",
            "plan_digest",
        ),
        ("  pull_request:\n", "  workflow_dispatch:\n", "trigger"),
    ] {
        assert!(source.contains(before));
        let changed = workflows::workflow_yaml::parse(&source.replacen(before, after, 1)).unwrap();
        let mismatches = entry_forwarding_mismatches(
            "pr_linux.yml",
            &changed,
            workflows::job(&changed, "lane"),
            inputs,
        );
        assert!(mismatches.contains(&expected), "{mismatches:?}");
    }
}

#[test]
fn main_entrypoint_rejects_a_broader_branch_trigger() {
    let source = workflows::source("main_macos.yml");
    let changed = source.replacen("    branches: [main]", "    branches: ['*']", 1);
    assert_ne!(source, changed);
    let changed = workflows::workflow_yaml::parse(&changed).unwrap();
    let called = workflows::document("ci-macos-lane.yml");
    let inputs = called
        .get("on")
        .and_then(|on| on.get("workflow_call"))
        .and_then(|call| call.get("inputs"));
    let mismatches = entry_forwarding_mismatches(
        "main_macos.yml",
        &changed,
        workflows::job(&changed, "lane"),
        inputs,
    );
    assert!(mismatches.contains(&"trigger"));
}

#[test]
fn special_paths_reject_nonempty_identity_and_hosted_override_substitutions() {
    let cases = [
        (
            "ci-pr-canary-lane.yml",
            "ui_artifact",
            "ci-ui-artifact-slice.yml",
            "\n      source_sha: ${{ inputs.merge_sha }}",
            "\n      source_sha: ${{ github.sha }}",
            "source_sha",
        ),
        (
            "ci-pr-canary-lane.yml",
            "hosts",
            "ci-linux-host-slice.yml",
            "\n      use_depot: false",
            "\n      use_depot: true",
            "use_depot",
        ),
        (
            "release.yml",
            "release_ui",
            "ci-ui-artifact-slice.yml",
            "\n      force_hosted: true",
            "\n      force_hosted: false",
            "force_hosted",
        ),
        (
            "native-sdk-artifact.yml",
            "produce_linux_static_abi",
            "static-abi-artifact.yml",
            "\n      source_sha: ${{ inputs.source_sha }}",
            "\n      source_sha: ${{ github.sha }}",
            "source_sha",
        ),
        (
            "pr_ci_canary.yml",
            "canary",
            "ci-pr-canary-lane.yml",
            "\n      head_sha: ${{ github.event.pull_request.head.sha }}",
            "\n      head_sha: ${{ github.sha }}",
            "head_sha",
        ),
    ];
    for (file, job, target, before, after, expected) in cases {
        let source = workflows::source(file);
        assert!(source.contains(before));
        let called = workflows::document(target);
        let inputs = called
            .get("on")
            .and_then(|on| on.get("workflow_call"))
            .and_then(|call| call.get("inputs"));
        let original = workflows::workflow_yaml::parse(&source).unwrap();
        assert!(
            special_forwarding_mismatches(
                file,
                job,
                &original,
                workflows::job(&original, job),
                inputs
            )
            .is_empty()
        );
        let changed = workflows::workflow_yaml::parse(&source.replacen(before, after, 1)).unwrap();
        let mismatches = special_forwarding_mismatches(
            file,
            job,
            &changed,
            workflows::job(&changed, job),
            inputs,
        );
        assert!(
            mismatches.contains(&expected),
            "{file}/{job}: {mismatches:?}"
        );
    }
}

#[test]
fn new_special_edges_require_an_explicit_value_contract() {
    assert!(reviewed_value_path("ci-pr-canary-lane.yml", "hosts"));
    assert!(reviewed_value_path("pr_linux.yml", "lane"));
    assert!(!reviewed_value_path("ci-pr-canary-lane.yml", "new_job"));
    assert!(!reviewed_value_path("pr_linux.yml", "new_job"));
    assert!(!reviewed_value_path("unreviewed.yml", "lane"));
}

#[test]
fn standard_lane_rejects_nonempty_source_event_override_and_depot_substitutions() {
    let source = workflows::source("ci-linux-lane.yml");
    let cases = [
        (
            "ui_artifact",
            "ci-ui-artifact-slice.yml",
            "      source_sha: ${{ inputs.source_sha }}",
            "      source_sha: ${{ github.sha }}",
            "source_sha",
        ),
        (
            "ui_artifact",
            "ci-ui-artifact-slice.yml",
            "      original_event_name: ${{ inputs.original_event_name }}",
            "      original_event_name: ${{ github.event_name }}",
            "original_event_name",
        ),
        (
            "ui_artifact",
            "ci-ui-artifact-slice.yml",
            "      force_hosted: ${{ toJSON(fromJson(inputs.lane_plan_json).signals.force_hosted) != 'false' }}",
            "      force_hosted: false",
            "force_hosted",
        ),
        (
            "static_abi",
            "static-abi-artifact.yml",
            "      use_depot: ${{ inputs.use_depot }}",
            "      use_depot: true",
            "use_depot",
        ),
    ];
    for (job, target, original, replacement, expected) in cases {
        assert!(source.contains(original));
        let changed = source.replacen(original, replacement, 1);
        let caller = workflows::workflow_yaml::parse(&changed).unwrap();
        let called = workflows::document(target);
        let inputs = called
            .get("on")
            .and_then(|on| on.get("workflow_call"))
            .and_then(|call| call.get("inputs"));
        let mismatches =
            lane_forwarding_mismatches("ci-linux-lane.yml", workflows::job(&caller, job), inputs);
        assert!(mismatches.contains(&expected), "{job}: {mismatches:?}");
    }
}

#[test]
fn direct_entry_default_refuses_new_events_or_enabled_defaults() {
    let edge = "main_linux.yml/lane -> ci-linux-lane.yml";
    let caller = workflows::document("main_linux.yml");
    let target = workflows::document("ci-linux-lane.yml");
    let inputs = target
        .get("on")
        .unwrap()
        .get("workflow_call")
        .unwrap()
        .get("inputs")
        .unwrap();
    assert!(allowed_direct_default(edge, &caller, inputs, None));
    assert!(!allowed_direct_default(edge, &caller, inputs, Some("")));
    let source = workflows::source("main_linux.yml");
    let anchor = "  push:\n    branches: [main]";
    assert_eq!(source.matches(anchor).count(), 1);
    let changed_caller = source.replacen(
        anchor,
        "  push:\n    branches: [main]\n  workflow_dispatch:",
        1,
    );
    let changed_caller = workflows::workflow_yaml::parse(&changed_caller).unwrap();
    assert!(!allowed_direct_default(edge, &changed_caller, inputs, None));
    let source = workflows::source("ci-linux-lane.yml");
    let anchor = "      use_depot:\n        description: Opt into the guarded main-only Depot canary.\n        required: false\n        default: false";
    assert_eq!(source.matches(anchor).count(), 1);
    let changed_target = source.replacen(
        anchor,
        "      use_depot:\n        description: Opt into the guarded main-only Depot canary.\n        required: false\n        default: true",
        1,
    );
    let changed_target = workflows::workflow_yaml::parse(&changed_target).unwrap();
    let inputs = changed_target
        .get("on")
        .unwrap()
        .get("workflow_call")
        .unwrap()
        .get("inputs")
        .unwrap();
    assert!(!allowed_direct_default(edge, &caller, inputs, None));
}

#[test]
fn selector_reusable_paths_declare_policy_and_source_inputs() {
    let missing = missing_forwarding(&selector_paths());
    assert!(missing.is_empty(), "missing forwarding: {missing:#?}");
}

#[test]
fn every_reusable_selector_path_is_in_the_source_census() {
    let actual = selector_paths();
    let roster = include_str!("selector_reusable_paths.txt");
    let expected = roster.lines().map(str::to_owned).collect::<BTreeSet<_>>();
    assert_eq!(expected.len(), roster.lines().count(), "duplicate path");
    assert_eq!(actual, expected, "new selector paths require review");
    let documentation =
        std::fs::read_to_string(support::root().join("ci/RUNNER_SELECTION_ACCEPTANCE.md")).unwrap();
    assert!(documentation.contains(&format!("tracks {} static call edges", actual.len())));
}
