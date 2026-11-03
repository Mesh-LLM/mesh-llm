//! Exercise the protected PR canary's source matrix and hosted propagation.

use super::super as workflows;
use super::{assert_wiring_mutation_rejected, plan, support};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::process::Command;

const CALLERS: &[(&str, &str)] = &[
    ("ui_artifact", "ci-ui-artifact-slice.yml"),
    ("hosts", "ci-linux-host-slice.yml"),
    ("native_runtime", "ci-linux-runtime-slice.yml"),
    ("product", "ci-linux-product-slice.yml"),
];

fn canary_callers_force_hosted(workflow: &workflows::Node) -> bool {
    CALLERS.iter().all(|(name, target)| {
        let caller = workflows::job(workflow, name);
        workflows::text(caller, "uses") == Some(format!("./.github/workflows/{target}").as_str())
            && workflows::input(caller, "force_hosted") == Some("true")
            && workflows::input(caller, "original_event_name") == Some("pull_request")
            && workflows::input(caller, "source_sha") == Some("${{ inputs.merge_sha }}")
    })
}

fn canary_matrices(workflow: &workflows::Node) -> BTreeMap<String, Value> {
    let step = workflows::steps(workflows::job(workflow, "plan"))
        .iter()
        .find(|step| workflows::text(step, "id") == Some("source_matrix"))
        .unwrap();
    let temp = tempfile::tempdir().unwrap();
    let output = temp.path().join("outputs");
    let result = Command::new("bash")
        .args(["-c", workflows::text(step, "run").unwrap()])
        .current_dir(support::root())
        .env("GITHUB_OUTPUT", &output)
        .output()
        .unwrap();
    assert!(result.status.success(), "{result:?}");
    std::fs::read_to_string(output)
        .unwrap()
        .lines()
        .map(|line| {
            let (key, value) = line.split_once('=').unwrap();
            (key.to_owned(), serde_json::from_str(value).unwrap())
        })
        .collect()
}

#[test]
fn canary_cpu_matrix_and_nested_callers_remain_source_bound_and_hosted() {
    let lane = workflows::document("ci-pr-canary-lane.yml");
    let plan_step = workflows::steps(workflows::job(&lane, "plan"))
        .iter()
        .find(|step| workflows::text(step, "id") == Some("plan"))
        .unwrap();
    assert_eq!(
        workflows::text(plan_step, "uses"),
        Some("./.github/actions/plan-ci")
    );
    for (key, value) in [
        ("event_name", "pull_request"),
        ("source_sha", "${{ inputs.merge_sha }}"),
        ("base_sha", "${{ inputs.base_sha }}"),
        ("draft", "${{ inputs.draft }}"),
    ] {
        assert_eq!(workflows::input(plan_step, key), Some(value), "{key}");
    }
    for profile in ["pr-draft", "pr-ready"] {
        let planned = plan(
            profile,
            "pull_request",
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "mesh/crates/mesh-llm/src/main.rs",
        );
        assert_eq!(
            planned["source_sha"],
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        );
    }

    let matrices = canary_matrices(&lane);
    let catalog: Value =
        serde_json::from_slice(&std::fs::read(support::root().join("ci/slices.yml")).unwrap())
            .unwrap();
    let rows = catalog["runtime_rows"].as_array().unwrap();
    let cpu = rows
        .iter()
        .filter(|row| row["id"] == "linux-cpu")
        .collect::<Vec<_>>();
    assert_eq!(cpu.len(), 1);
    assert_eq!(matrices["runtime_matrix"], json!([cpu[0]]));
    assert_eq!(
        matrices["host_matrix"],
        json!([{"id": "linux-cpu", "platform": "linux", "architecture": "amd64"}])
    );
    assert_eq!(
        matrices["product_matrix"],
        json!([{"id": "linux-cpu", "platform": "linux", "architecture": "amd64", "backend": "cpu"}])
    );
    assert!(canary_callers_force_hosted(&lane));
    assert_wiring_mutation_rejected(
        "ci-pr-canary-lane.yml",
        "      original_event_name: pull_request\n      force_hosted: true\n      hosts_matrix:",
        "      original_event_name: pull_request\n      force_hosted: false\n      hosts_matrix:",
        canary_callers_force_hosted,
    );
    for (caller, input, output) in [
        ("hosts", "hosts_matrix", "host_matrix"),
        ("native_runtime", "runtime_matrix", "runtime_matrix"),
        ("product", "runtime_matrix", "product_matrix"),
    ] {
        assert_eq!(
            workflows::input(workflows::job(&lane, caller), input),
            Some(format!("${{{{ needs.plan.outputs.{output} }}}}").as_str())
        );
    }
    let artifact = "ci-pr-canary-ui-${{ github.run_id }}";
    assert_eq!(
        workflows::input(workflows::job(&lane, "ui_artifact"), "artifact_name"),
        Some(artifact)
    );
    assert_eq!(
        workflows::input(workflows::job(&lane, "hosts"), "ui_artifact_name"),
        Some(artifact)
    );

    for (caller, file) in CALLERS {
        let slice = workflows::document(file);
        let selectors = workflows::steps(workflows::job(&slice, "runner_policy"))
            .iter()
            .filter(|step| {
                workflows::text(step, "uses") == Some("./.github/actions/select-ci-runners")
            })
            .collect::<Vec<_>>();
        assert!(!selectors.is_empty(), "{caller}");
        for selector in selectors {
            let cpu = workflows::text(selector, "id") == Some("cpu_policy");
            assert_eq!(
                workflows::input(selector, "force_hosted"),
                Some(if cpu {
                    "true"
                } else {
                    "${{ inputs.force_hosted }}"
                })
            );
            assert_eq!(
                workflows::input(selector, "original_event_name"),
                Some("${{ inputs.original_event_name }}")
            );
            for head in ["Mesh-LLM/mesh-llm", "fork/mesh-llm"] {
                let fixture = support::Fixture::new();
                let result = fixture.selector(&[
                    ("INPUT_EVENT_NAME", "pull_request"),
                    ("INPUT_ORIGINAL_EVENT_NAME", "pull_request"),
                    ("INPUT_HEAD_REPOSITORY", head),
                    ("INPUT_REF", "refs/pull/12/merge"),
                    ("INPUT_DEPOT_PR_ENABLED", "true"),
                    ("INPUT_PR_CANARY_REF", "refs/pull/12/merge"),
                    ("FIXTURE_DATE", "2026-09-13"),
                    ("INPUT_FORCE_HOSTED", "true"),
                ]);
                assert!(result.status.success(), "{caller}: {result:?}");
                let selected = fixture.outputs();
                assert_eq!(selected["depot_enabled"], "false", "{caller}/{head}");
                assert_eq!(selected["runner_8"], "ubuntu-24.04");
                assert_eq!(selected["allow_depot_remote_cache"], "false");
                assert_eq!(selected["allow_native_github_cache"], "true");
                fixture.0.close().unwrap();
            }
        }
    }
    // Removing the forwarded override would make this synthetic trusted PR
    // eligible for the selector's historical, bounded Depot exception.
    let fixture = support::Fixture::new();
    let result = fixture.selector(&[
        ("INPUT_DEPOT_PR_ENABLED", "true"),
        ("FIXTURE_DATE", "2026-09-13"),
        ("INPUT_FORCE_HOSTED", "false"),
    ]);
    assert!(result.status.success());
    assert_eq!(fixture.outputs()["depot_enabled"], "true");
    fixture.0.close().unwrap();
}
