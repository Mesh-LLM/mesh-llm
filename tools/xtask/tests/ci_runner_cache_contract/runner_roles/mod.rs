//! Every runner role has one explicit provider and preparation decision.
mod checks;
mod derive;
mod matrix;

use crate::{support, workflow_yaml::Node};
use matrix::{Decision, HostedFallback, Matrix, Role};
use std::{collections::BTreeMap, fs};

fn checked_in_matrix() -> Matrix {
    let text = fs::read_to_string(support::root().join("ci/runner-roles.json")).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn role<'a>(matrix: &'a mut Matrix, id: &str) -> &'a mut Role {
    matrix.roles.iter_mut().find(|r| r.id == id).unwrap()
}

fn assert_rejected(matrix: &Matrix, inventory: &derive::Inventory, expected: &str) {
    let errors = checks::violations(matrix, inventory);
    assert!(
        errors.iter().any(|e| e.contains(expected)),
        "expected a violation containing {expected:?}, got {errors:#?}"
    );
}

#[test]
fn every_runner_role_matches_the_checked_in_workflows() {
    let inventory = derive::inventory(&derive::checked_in_workflows());
    let errors = checks::violations(&checked_in_matrix(), &inventory);
    assert!(errors.is_empty(), "{errors:#?}");
}

#[test]
fn only_the_existing_trusted_main_allowlist_is_depot_eligible() {
    let matrix = checked_in_matrix();
    let main_depot: Vec<&str> = matrix
        .roles
        .iter()
        .filter(|r| r.main == Decision::DepotGated)
        .map(|r| r.id.as_str())
        .collect();
    assert_eq!(
        main_depot,
        [
            "quality",
            "linux-host",
            "linux-runtime-gpu",
            "static-abi",
            "kotlin-native-sdk"
        ]
    );
    for id in [
        "rust-tests",
        "linux-runtime-cpu-cache-producer",
        "macos",
        "windows",
        "website",
    ] {
        let row = matrix.roles.iter().find(|r| r.id == id).unwrap();
        assert_eq!(row.main, Decision::NoDepot, "{id}");
    }
}

#[test]
fn full_release_linux_legs_stay_hosted_through_required_operator_inputs() {
    let matrix = checked_in_matrix();
    let release = matrix
        .roles
        .iter()
        .find(|r| r.id == "release-publication")
        .unwrap();
    assert_eq!(release.hosted_fallback, HostedFallback::OperatorInput);
    let required = BTreeMap::from([("force_hosted_runners".to_owned(), true)]);
    assert_eq!(release.hosted_required_inputs, required);
    let inventory = derive::inventory(&derive::checked_in_workflows());
    let declared = &inventory.dispatch_inputs["release.yml"];
    for name in required.keys() {
        assert_eq!(
            declared.get(name).map(String::as_str),
            Some("boolean"),
            "{name}"
        );
    }
    let preview = rolling_preview_release_inputs();
    for (name, value) in &required {
        assert_eq!(preview.get(name), Some(&value.to_string()), "{name}");
    }
}

fn rolling_preview_release_inputs() -> BTreeMap<String, String> {
    let text =
        fs::read_to_string(support::root().join(".github/workflows/rolling-preview.yml")).unwrap();
    let workflow = crate::workflow_yaml::parse(&text).unwrap();
    let mut found = BTreeMap::new();
    for (_, job) in workflow.get("jobs").unwrap().entries() {
        let Some(Node::Seq(steps)) = job.get("steps") else {
            continue;
        };
        for run in steps
            .iter()
            .filter_map(|s| s.get("run").and_then(Node::text))
        {
            if !run.contains("gh workflow run release.yml") {
                continue;
            }
            let mut words = run.split_whitespace();
            while let Some(word) = words.next() {
                if word != "-f" {
                    continue;
                }
                let (key, value) = words.next().unwrap().split_once('=').unwrap();
                found.insert(key.to_owned(), value.to_owned());
            }
        }
    }
    found
}

#[test]
fn a_runner_selecting_job_without_a_row_is_rejected() {
    let inventory = derive::inventory(&derive::checked_in_workflows());
    let mut matrix = checked_in_matrix();
    role(&mut matrix, "quality")
        .selectors
        .retain(|s| s != "ci-quality-slice.yml#runner_policy/policy");
    assert_rejected(
        &matrix,
        &inventory,
        "selector ci-quality-slice.yml#runner_policy/policy has no runner role row",
    );
    let mut matrix = checked_in_matrix();
    role(&mut matrix, "depot-diagnostics")
        .jobs
        .retain(|j| j != "depot-registry-canary.yml#pull");
    assert_rejected(
        &matrix,
        &inventory,
        "runner-selecting job depot-registry-canary.yml#pull has no runner role row",
    );
}

#[test]
fn a_row_that_disagrees_with_its_selector_inputs_is_rejected() {
    let inventory = derive::inventory(&derive::checked_in_workflows());
    let mut matrix = checked_in_matrix();
    role(&mut matrix, "rust-tests").main = Decision::DepotGated;
    assert_rejected(
        &matrix,
        &inventory,
        "selector ci-rust-tests-slice.yml#runner_policy/policy decides main NoDepot",
    );
    let mut matrix = checked_in_matrix();
    role(&mut matrix, "linux-host").main = Decision::NoDepot;
    assert_rejected(
        &matrix,
        &inventory,
        "selector ci-linux-host-slice.yml#runner_policy/policy decides main DepotGated",
    );
}

#[test]
fn an_unsupported_provider_profile_combination_is_rejected() {
    let mut inventory = derive::inventory(&derive::checked_in_workflows());
    // Put the hosted-bare native runtime legs back on a selected label.
    inventory
        .jobs
        .get_mut("release.yml#build_native_runtime")
        .unwrap()
        .selectors
        .insert(
            "release.yml#metadata/runners".to_owned(),
            ["runner_8".to_owned()].into(),
        );
    let mut matrix = checked_in_matrix();
    let release = role(&mut matrix, "release-publication");
    release.hosted_fallback = HostedFallback::Selector;
    release.hosted_required_inputs.clear();
    assert_rejected(
        &matrix,
        &inventory,
        "release.yml#build_native_runtime: hosted-bare preparation is reachable",
    );
}

#[test]
fn a_new_depot_label_or_profile_needs_a_row_change() {
    let inventory = derive::inventory(&derive::checked_in_workflows());
    let mut matrix = checked_in_matrix();
    role(&mut matrix, "macos").labels.remove("depot-macos-15");
    assert_rejected(&matrix, &inventory, "row macos: workflows place on labels");
    let mut matrix = checked_in_matrix();
    role(&mut matrix, "website")
        .preparation_profiles
        .insert(matrix::Profile::HostedBare);
    assert_rejected(&matrix, &inventory, "row website: workflows prepare with");
}

#[test]
fn readiness_is_active_only_for_rows_on_depot_for_main() {
    let inventory = derive::inventory(&derive::checked_in_workflows());
    let mut matrix = checked_in_matrix();
    role(&mut matrix, "windows").readiness.state = matrix::ReadinessState::Active;
    assert_rejected(
        &matrix,
        &inventory,
        "row windows: only rows that run on Depot on main are active",
    );
}

#[test]
fn quoted_expression_values_exclude_comparison_operands() {
    let values =
        derive::quoted_literals("${{ inputs.kind == 'swift' && 'macos-15' || 'ubuntu-24.04' }}");
    assert_eq!(values, ["macos-15", "ubuntu-24.04"]);
}
