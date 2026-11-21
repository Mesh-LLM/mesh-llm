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
    let text =
        fs::read_to_string(support::root().join(".github/workflows/rolling-preview.yml")).unwrap();
    let dispatches = release_dispatch_inputs(&text);
    assert!(
        !dispatches.is_empty(),
        "rolling preview dispatches no release"
    );
    for preview in &dispatches {
        for (name, value) in &required {
            assert_eq!(preview.get(name), Some(&value.to_string()), "{name}");
        }
    }
}

/// The `-f`/`-F` inputs of each `gh workflow run release.yml` in a workflow.
fn release_dispatch_inputs(text: &str) -> Vec<BTreeMap<String, String>> {
    let workflow = crate::workflow_yaml::parse(text).unwrap();
    let mut dispatches = Vec::new();
    for (_, job) in workflow.get("jobs").unwrap().entries() {
        let Some(Node::Seq(steps)) = job.get("steps") else {
            continue;
        };
        for run in steps
            .iter()
            .filter_map(|s| s.get("run").and_then(Node::text))
        {
            for dispatch in run.split("gh workflow run release.yml").skip(1) {
                let mut found = BTreeMap::new();
                let mut words = dispatch.split_whitespace();
                while let Some(word) = words.next() {
                    if !["-f", "--raw-field", "-F", "--field"].contains(&word) {
                        continue;
                    }
                    let (key, value) = words.next().unwrap().split_once('=').unwrap();
                    found.insert(key.to_owned(), value.to_owned());
                }
                dispatches.push(found);
            }
        }
    }
    dispatches
}

#[test]
fn each_release_dispatch_carries_its_own_inputs() {
    let dispatches = release_dispatch_inputs(
        "jobs:
  preview:
    runs-on: ubuntu-24.04
    steps:
      - run: |
          gh workflow run release.yml --raw-field force_hosted_runners=true
          gh workflow run release.yml -f preview=true
",
    );
    let hosted = |d: &BTreeMap<String, String>| d.get("force_hosted_runners").cloned();
    assert_eq!(
        dispatches.iter().map(hosted).collect::<Vec<_>>(),
        [Some("true".to_owned()), None]
    );
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
fn a_consumed_selector_output_that_is_not_a_runner_is_rejected() {
    let mut inventory = derive::inventory(&derive::checked_in_workflows());
    inventory
        .jobs
        .get_mut("ci-quality-slice.yml#rust_fmt")
        .unwrap()
        .selectors
        // Its runs-on forwards a non-runner output instead of runner_4.
        .insert(
            "ci-quality-slice.yml#runner_policy/policy".to_owned(),
            ["depot_enabled".to_owned()].into(),
        );
    assert_rejected(
        &checked_in_matrix(),
        &inventory,
        "ci-quality-slice.yml#rust_fmt: runs on ci-quality-slice.yml#runner_policy/policy: depot_enabled, which is not a selector runner output",
    );
}

#[test]
fn ci_controller_preparation_on_a_non_linux_runner_is_rejected() {
    let inventory = derive::inventory(&[workflow(
        "mac.yml",
        "on: pull_request
jobs:
  plan:
    runs-on: macos-15
    steps:
      - uses: ./.github/actions/prepare-ci-controller
",
    )]);
    assert_rejected(
        &checked_in_matrix(),
        &inventory,
        "mac.yml#plan: ci-controller preparation is reachable",
    );
}

#[test]
fn preinstalled_preparation_for_another_os_is_rejected() {
    let inventory = derive::inventory(&[workflow(
        "lab.yml",
        "on: workflow_dispatch
jobs:
  build:
    runs-on: [self-hosted, Linux, X64]
    steps:
      - uses: ./.github/actions/prepare-automation
        with:
          runner-profile: preinstalled-macos
",
    )]);
    let mut matrix = checked_in_matrix();
    role(&mut matrix, "self-hosted-lab")
        .jobs
        .push("lab.yml#build".to_owned());
    assert_rejected(
        &matrix,
        &inventory,
        "lab.yml#build: preinstalled preparation for another OS is reachable",
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

#[test]
fn the_github_hosted_guard_must_hold_on_every_path() {
    assert!(derive::requires_github_hosted(
        "runner.os == 'Linux' && job.container.id == '' && runner.environment == 'github-hosted' && env.MESH_LLM_AUTOMATION_BIN == ''"
    ));
    assert!(!derive::requires_github_hosted(
        "runner.environment == 'github-hosted' || inputs.use_depot"
    ));
    assert!(!derive::requires_github_hosted(""));
}

#[test]
fn hosted_bare_is_guarded_only_by_the_branch_that_selects_it() {
    let checked_in = derive::guarded_literals(
        "${{ runner.os == 'Linux' && job.container.id == '' && runner.environment == 'github-hosted' && 'hosted-bare' || 'image' }}",
        derive::GITHUB_HOSTED_GUARD,
    );
    assert_eq!(
        checked_in,
        [
            ("hosted-bare".to_owned(), true),
            ("image".to_owned(), false)
        ]
    );
    let fallback = derive::guarded_literals(
        "${{ runner.environment == 'github-hosted' && 'image' || 'hosted-bare' }}",
        derive::GITHUB_HOSTED_GUARD,
    );
    assert_eq!(
        fallback,
        [
            ("image".to_owned(), true),
            ("hosted-bare".to_owned(), false)
        ]
    );
}

fn workflow(name: &str, text: &str) -> derive::Workflow {
    derive::Workflow {
        name: name.to_owned(),
        document: crate::workflow_yaml::parse(text).unwrap(),
    }
}

#[test]
fn a_duplicate_row_id_is_rejected() {
    let inventory = derive::inventory(&derive::checked_in_workflows());
    let mut matrix = checked_in_matrix();
    let mut copy = role(&mut matrix, "website").clone();
    copy.id = "quality".to_owned();
    matrix.roles.push(copy);
    assert_rejected(&matrix, &inventory, "row id quality is not unique");
}

#[test]
fn a_fixed_depot_job_in_a_selector_row_must_be_manual_dispatch_only() {
    let mut inventory = derive::inventory(&derive::checked_in_workflows());
    let job = derive::Job {
        workflow: "main_quality.yml".to_owned(),
        labels: ["depot-ubuntu-24.04-4".to_owned()].into(),
        ..derive::Job::default()
    };
    inventory
        .jobs
        .insert("main_quality.yml#fixed".to_owned(), job);
    let mut matrix = checked_in_matrix();
    role(&mut matrix, "quality")
        .jobs
        .push("main_quality.yml#fixed".to_owned());
    assert_rejected(
        &matrix,
        &inventory,
        "row quality: fixed Depot labels in main_quality.yml must be manual-dispatch only",
    );
}

#[test]
fn hosted_bare_on_an_unresolved_runner_is_rejected() {
    let mut inventory = derive::inventory(&derive::checked_in_workflows());
    let job = derive::Job {
        workflow: "payload.yml".to_owned(),
        dynamic: vec!["${{ github.event.client_payload.runner }}".to_owned()],
        profiles: [matrix::Profile::HostedBare].into(),
        ..derive::Job::default()
    };
    inventory.jobs.insert("payload.yml#run".to_owned(), job);
    assert_rejected(
        &checked_in_matrix(),
        &inventory,
        "payload.yml#run: hosted-bare preparation runs on unresolved runner",
    );
}

#[test]
fn unguarded_hosted_bare_on_a_non_linux_runner_is_rejected() {
    let inventory = derive::inventory(&[workflow(
        "os.yml",
        "on: workflow_dispatch
jobs:
  bare:
    runs-on: macos-15
    steps:
      - uses: ./.github/actions/prepare-automation
        with:
          runner-profile: hosted-bare
  linux_only:
    runs-on: macos-15
    steps:
      - uses: ./.github/actions/prepare-automation
        with:
          runner-profile: ${{ runner.os == 'Linux' && 'hosted-bare' || 'image' }}
",
    )]);
    let errors = checks::violations(&checked_in_matrix(), &inventory);
    assert!(
        errors
            .iter()
            .any(|e| e.starts_with("os.yml#bare: hosted-bare") && e.contains("requires Linux")),
        "{errors:#?}"
    );
    assert!(
        !errors.iter().any(|e| e.starts_with("os.yml#linux_only")),
        "{errors:#?}"
    );
}

#[test]
fn operator_input_fallback_does_not_exempt_a_depot_reachable_hosted_bare_job() {
    let mut inventory = derive::inventory(&derive::checked_in_workflows());
    inventory
        .jobs
        .get_mut("release.yml#build_native_runtime")
        .unwrap()
        .selectors
        .insert(
            "release.yml#metadata/runners".to_owned(),
            ["runner_8".to_owned()].into(),
        );
    let matrix = checked_in_matrix();
    let release = matrix
        .roles
        .iter()
        .find(|r| r.id == "release-publication")
        .unwrap();
    assert_eq!(release.hosted_fallback, HostedFallback::OperatorInput);
    assert_rejected(
        &matrix,
        &inventory,
        "release.yml#build_native_runtime: hosted-bare preparation is reachable",
    );
}

#[test]
fn scalar_and_sequence_triggers_are_derived() {
    let jobs = "jobs:
  run:
    runs-on: ubuntu-24.04
    steps:
      - run: true
";
    let inventory = derive::inventory(&[
        workflow("scalar.yml", &format!("on: workflow_dispatch\n{jobs}")),
        workflow(
            "sequence.yml",
            &format!("on: [push, workflow_dispatch]\n{jobs}"),
        ),
    ]);
    let names = |list: &[&str]| list.iter().map(|s| (*s).to_owned()).collect();
    assert_eq!(
        inventory.triggers["scalar.yml"],
        names(&["workflow_dispatch"])
    );
    assert_eq!(
        inventory.triggers["sequence.yml"],
        names(&["push", "workflow_dispatch"])
    );
}

#[test]
fn an_unrecognized_runner_expression_needs_a_row() {
    let inventory = derive::inventory(&[workflow(
        "payload.yml",
        "on: repository_dispatch
jobs:
  run:
    runs-on: ${{ github.event.client_payload.runner }}
    steps:
      - run: true
",
    )]);
    assert_rejected(
        &checked_in_matrix(),
        &inventory,
        "runner-selecting job payload.yml#run has no runner role row",
    );
}

#[test]
fn the_manual_opt_in_must_be_the_operator_input() {
    let mut inventory = derive::inventory(&derive::checked_in_workflows());
    let key = "ci-quality-slice.yml#runner_policy/policy";
    let selector = inventory.selectors.get_mut(key).unwrap();
    selector
        .inputs
        .insert("manual_use_depot".to_owned(), "true".to_owned());
    assert_rejected(
        &checked_in_matrix(),
        &inventory,
        "unrecognized main selector inputs",
    );
}

#[test]
fn an_invalid_force_hosted_value_is_rejected() {
    let mut inventory = derive::inventory(&derive::checked_in_workflows());
    let key = "ci-quality-slice.yml#runner_policy/policy";
    let selector = inventory.selectors.get_mut(key).unwrap();
    selector
        .inputs
        .insert("force_hosted".to_owned(), "tru".to_owned());
    assert_rejected(
        &checked_in_matrix(),
        &inventory,
        "unrecognized force_hosted input tru",
    );
}

#[test]
fn a_fixed_depot_branch_of_a_selector_job_must_be_manual_dispatch_only() {
    let mut inventory = derive::inventory(&derive::checked_in_workflows());
    let job = inventory
        .jobs
        .values_mut()
        .find(|j| {
            j.selectors
                .contains_key("ci-quality-slice.yml#runner_policy/policy")
        })
        .unwrap();
    job.labels.insert("depot-ubuntu-24.04-4".to_owned());
    assert_rejected(
        &checked_in_matrix(),
        &inventory,
        "row quality: fixed Depot labels in ci-quality-slice.yml must be manual-dispatch only",
    );
}

#[test]
fn preinstalled_preparation_on_a_fixed_hosted_runner_is_rejected() {
    let inventory = derive::inventory(&[workflow(
        "hosted.yml",
        "on: workflow_dispatch
jobs:
  run:
    runs-on: ubuntu-24.04
    steps:
      - uses: ./.github/actions/prepare-automation
        with:
          runner-profile: preinstalled-linux
",
    )]);
    assert_rejected(
        &checked_in_matrix(),
        &inventory,
        "hosted.yml#run: preinstalled preparation is reachable",
    );
}

#[test]
fn expression_items_of_a_runs_on_sequence_are_resolved() {
    let inventory = derive::inventory(&[workflow(
        "gpu.yml",
        "on: workflow_dispatch
jobs:
  run:
    runs-on: [self-hosted, \"${{ matrix.accelerator }}\"]
    strategy:
      matrix:
        accelerator: [gpu-nvidia]
    steps:
      - run: true
",
    )]);
    let labels = &inventory.jobs["gpu.yml#run"].labels;
    let names = ["gpu-nvidia", "self-hosted"].map(str::to_owned);
    assert_eq!(*labels, names.into());
}

#[test]
fn an_unresolved_runner_branch_needs_a_row_beside_a_hosted_branch() {
    let inventory = derive::inventory(&[workflow(
        "payload.yml",
        "on: [push, repository_dispatch]
jobs:
  run:
    runs-on: ${{ github.event_name == 'push' && 'ubuntu-24.04' || github.event.client_payload.runner }}
    steps:
      - run: true
",
    )]);
    assert_rejected(
        &checked_in_matrix(),
        &inventory,
        "runner-selecting job payload.yml#run has no runner role row",
    );
}

fn wrapper_action(root: &std::path::Path, name: &str, file: &str, uses: &str) {
    let directory = root.join(".github/actions").join(name);
    fs::create_dir_all(&directory).unwrap();
    let body = format!("runs:\n  using: composite\n  steps:\n    - uses: {uses}\n");
    fs::write(directory.join(file), body).unwrap();
}

const DEEP_WRAPPER_JOB: &str = "on: workflow_dispatch
jobs:
  bare:
    runs-on: macos-15
    steps:
      - uses: ./.github/actions/wrap-1
";

#[test]
fn hosted_bare_behind_deep_and_action_yaml_wrappers_is_derived() {
    let root = tempfile::tempdir().unwrap();
    wrapper_action(
        root.path(),
        "wrap-1",
        "action.yml",
        "./.github/actions/wrap-2",
    );
    wrapper_action(
        root.path(),
        "wrap-2",
        "action.yml",
        "./.github/actions/wrap-3",
    );
    wrapper_action(
        root.path(),
        "wrap-3",
        "action.yml",
        "./.github/actions/wrap-4",
    );
    let prepare =
        "./.github/actions/prepare-automation\n      with:\n        runner-profile: hosted-bare";
    wrapper_action(root.path(), "wrap-4", "action.yaml", prepare);
    let workflows = [workflow("deep.yml", DEEP_WRAPPER_JOB)];
    let inventory = derive::inventory_in(&workflows, root.path());
    assert_rejected(
        &checked_in_matrix(),
        &inventory,
        "deep.yml#bare: hosted-bare preparation without a runner.os == 'Linux' guard",
    );
}

#[test]
#[should_panic(expected = "local action wrap-2 has no action.yml or action.yaml")]
fn a_missing_local_action_is_rejected() {
    let root = tempfile::tempdir().unwrap();
    wrapper_action(
        root.path(),
        "wrap-1",
        "action.yml",
        "./.github/actions/wrap-2",
    );
    derive::inventory_in(&[workflow("deep.yml", DEEP_WRAPPER_JOB)], root.path());
}

#[test]
#[should_panic(expected = "local action cycle")]
fn a_local_action_cycle_is_rejected() {
    let root = tempfile::tempdir().unwrap();
    wrapper_action(
        root.path(),
        "wrap-1",
        "action.yml",
        "./.github/actions/wrap-1",
    );
    derive::inventory_in(&[workflow("deep.yml", DEEP_WRAPPER_JOB)], root.path());
}

#[test]
#[should_panic(expected = "has unresolved branch inputs.runner_profile")]
fn a_runner_profile_with_an_unresolved_fallback_is_rejected() {
    derive::inventory(&[workflow(
        "forward.yml",
        "on: workflow_call
jobs:
  run:
    runs-on: ubuntu-24.04
    steps:
      - uses: ./.github/actions/prepare-automation
        with:
          runner-profile: ${{ inputs.use_image && 'image' || inputs.runner_profile }}
",
    )]);
}

#[test]
#[should_panic(expected = "names no literal profile")]
fn a_forwarded_runner_profile_expression_is_rejected() {
    derive::inventory(&[workflow(
        "forward.yml",
        "on: workflow_call
jobs:
  run:
    runs-on: ubuntu-24.04
    steps:
      - uses: ./.github/actions/prepare-automation
        with:
          runner-profile: ${{ inputs.runner_profile }}
",
    )]);
}
