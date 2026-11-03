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

fn allowed_direct_default(edge: &str, caller: &workflows::Node, inputs: &workflows::Node) -> bool {
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
    let only_main = |branches: &workflows::Node| match branches {
        workflows::Node::Scalar(value) => value == "main",
        workflows::Node::Seq(values) => values.len() == 1 && values[0].text() == Some("main"),
        workflows::Node::Map(_) => false,
    };
    triggers.entries().len() == 1
        && (event != "push" || trigger.get("branches").is_some_and(only_main))
        && workflows::text(input, "type") == Some("boolean")
        && workflows::text(input, "default") == Some("false")
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
                        allowed_direct_default(edge, &caller_document, inputs)
                    })
                {
                    continue;
                }
                missing.push(format!("{edge}: {key}"));
            }
        }
    }
    missing
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
    assert!(allowed_direct_default(edge, &caller, inputs));
    let source = workflows::source("main_linux.yml");
    let anchor = "  push:\n    branches: [main]";
    assert_eq!(source.matches(anchor).count(), 1);
    let changed_caller = source.replacen(
        anchor,
        "  push:\n    branches: [main]\n  workflow_dispatch:",
        1,
    );
    let changed_caller = workflows::workflow_yaml::parse(&changed_caller).unwrap();
    assert!(!allowed_direct_default(edge, &changed_caller, inputs));
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
    assert!(!allowed_direct_default(edge, &caller, inputs));
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
}
