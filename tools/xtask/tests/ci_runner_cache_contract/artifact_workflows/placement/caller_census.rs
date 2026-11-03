//! Inventory direct selector callers without claiming complete case coverage.

use super::super as workflows;
use super::support;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

type Caller = (String, String, String);

fn is_selector(value: &str) -> bool {
    value == "./.github/actions/select-ci-runners"
        || value.contains("/.github/actions/select-ci-runners@")
}

fn inspect_steps(file: &str, job: &str, steps: &workflows::Node, found: &mut Vec<Caller>) {
    let workflows::Node::Seq(steps) = steps else {
        panic!("{file}/{job}: steps must be a sequence")
    };
    for step in steps {
        if !workflows::text(step, "uses").is_some_and(is_selector) {
            continue;
        }
        let id = workflows::text(step, "id").expect("selector step must have an id");
        let inputs = step.get("with").expect("selector inputs must be declared");
        for key in [
            "event_name",
            "original_event_name",
            "repository",
            "head_repository",
            "head_sha",
            "ref",
            "depot_main_enabled",
            "depot_pr_enabled",
            "force_hosted",
        ] {
            assert!(
                inputs.get(key).and_then(workflows::Node::text).is_some(),
                "{file}/{job}/{id}: missing {key}"
            );
        }
        found.push((file.into(), job.into(), id.into()));
    }
}

fn direct_workflow_callers(root: &Path) -> Vec<Caller> {
    let mut found = Vec::new();
    for entry in fs::read_dir(root.join(".github/workflows")).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if !matches!(
            path.extension().and_then(|ext| ext.to_str()),
            Some("yml" | "yaml")
        ) {
            continue;
        }
        let source = fs::read_to_string(&path).unwrap();
        if !source.contains("select-ci-runners") {
            continue;
        }
        let file = path.file_name().unwrap().to_str().unwrap();
        let document = workflows::workflow_yaml::parse(&source).unwrap();
        if let Some(jobs) = document.get("jobs") {
            for (job, value) in jobs.entries() {
                if let Some(steps) = value.get("steps") {
                    inspect_steps(file, job, steps, &mut found);
                }
            }
        }
    }
    found
}

fn assert_no_uninventoried_composite_callers(root: &Path) {
    for entry in fs::read_dir(root.join(".github/actions")).unwrap() {
        let path = entry.unwrap().path().join("action.yml");
        if !path.is_file() {
            continue;
        }
        let source = fs::read_to_string(&path).unwrap();
        if !source.contains("select-ci-runners") {
            continue;
        }
        let document = workflows::workflow_yaml::parse(&source).unwrap();
        if let Some(steps) = document.get("runs").and_then(|runs| runs.get("steps")) {
            let mut found = Vec::new();
            inspect_steps(path.to_str().unwrap(), "runs", steps, &mut found);
            assert!(
                found.is_empty(),
                "extend the census for composite selector callers: {found:?}"
            );
        }
    }
}

fn declared_callers(document: &str) -> BTreeMap<Caller, &str> {
    let section = document
        .split_once("<!-- selector-census:start -->")
        .and_then(|(_, tail)| tail.split_once("<!-- selector-census:end -->"))
        .map(|(section, _)| section)
        .expect("selector census markers");
    let mut declared = BTreeMap::new();
    for line in section.lines().filter(|line| line.starts_with("| `")) {
        let fields = line
            .trim_matches('|')
            .split('|')
            .map(str::trim)
            .collect::<Vec<_>>();
        assert_eq!(fields.len(), 4, "invalid selector census row");
        let caller = (
            fields[0].trim_matches('`').to_owned(),
            fields[1].trim_matches('`').to_owned(),
            fields[2].trim_matches('`').to_owned(),
        );
        assert!(
            ["partial", "open"].contains(&fields[3]),
            "invalid coverage status"
        );
        assert!(
            declared.insert(caller, fields[3]).is_none(),
            "duplicate selector census row"
        );
    }
    declared
}

#[test]
fn every_direct_selector_caller_declares_inputs_and_appears_in_the_census() {
    let root = support::root();
    let document = fs::read_to_string(root.join("ci/RUNNER_SELECTION_ACCEPTANCE.md")).unwrap();
    let declared = declared_callers(&document);
    let mut actual = direct_workflow_callers(&root);
    actual.sort();
    let mut deduplicated = actual.clone();
    deduplicated.dedup();
    assert_eq!(actual, deduplicated, "duplicate selector step identity");
    assert_eq!(actual, declared.keys().cloned().collect::<Vec<_>>());
    assert_no_uninventoried_composite_callers(&root);

    let partial = declared
        .values()
        .filter(|status| **status == "partial")
        .count();
    let open = declared
        .values()
        .filter(|status| **status == "open")
        .count();
    assert!(document.contains(&format!(
        "Current census: {} direct selector steps; {partial} partial and {open} open.",
        declared.len()
    )));
}
