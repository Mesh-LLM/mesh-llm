//! Rust-only graph and digest contracts: each one mutates a temporary copy of
//! a checked-in workflow, or the digest-bound plan, and expects exactly one
//! named defect. The unmodified inputs pass in `happy.rs`.

use crate::support::{
    Call, Golden, Scratch, TestResult, assert_output, expected_jobs, lane_workflow, needs,
    validate_entrypoints,
};
use serde_json::Value;

const PR_ONLY: &str = "${{ inputs.original_event_name == 'pull_request' }}";

fn planned(case: &str, lane: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    Ok(expected_jobs()?[case][lane]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect())
}

/// `control-main` plans every job of every lane, so each graph edge is live.
fn graph_rejects(lane: &str, edits: &[(&str, &str)], message: &str) -> TestResult {
    let scratch = Scratch::new("graph")?;
    let workflow = scratch.mutated(&lane_workflow(lane), edits)?;
    let golden = Golden::load("control-main")?;
    let jobs = planned("control-main", lane)?;
    let jobs = jobs.iter().map(String::as_str).collect::<Vec<_>>();
    let output = Call::lane(&golden.lane_plan(lane), &needs(lane, &jobs))
        .with("--workflow", &workflow)
        .run_rust_only()?;
    assert_output(&output, 2, &format!("ERROR: {message}\n"));
    Ok(())
}

#[test]
fn migration_ci_graph_consumer_without_declared_need_is_rejected() -> TestResult {
    // Given: runtime_product still gates on hosts but no longer needs it.
    graph_rejects(
        "linux",
        &[(
            "needs: [hosts, native_runtimes]",
            "needs: [native_runtimes]",
        )],
        // When/Then: the undeclared read is a disconnected consumer.
        "job 'runtime_product' reads needs.hosts without declaring it",
    )
}

#[test]
fn migration_ci_graph_artifact_producer_off_needs_path_is_rejected() -> TestResult {
    // Given: hosts still downloads the console artifact but lost its edge.
    graph_rejects(
        "linux",
        &[(
            "needs: [ui_artifact]\n    if: ${{ !cancelled() && needs.ui_artifact.result == 'success' && ",
            "if: ${{ !cancelled() && ",
        )],
        // When/Then: the producer is unreachable through needs.
        "consumer 'hosts' input ui_artifact_name: producer 'ui_artifact' is not on its needs path",
    )
}

#[test]
fn migration_ci_graph_unplanned_producer_is_rejected() -> TestResult {
    // Given: the legacy Windows fixture that plans products without hosts,
    // which the legacy result check accepted.
    let plan = r#"{"lane":"windows","required":true,"required_slices":["runtime-product"],"matrices":{"hosts":[],"runtime_products":[{"id":"windows-cpu"}],"platform_checks":[]}}"#;
    let jobs = needs("windows", &["native_runtimes", "runtime_product"]);
    // When: the lane graph is consulted.
    let output = Call::lane(plan, &jobs)
        .with("--workflow", &lane_workflow("windows"))
        .run_rust_only()?;
    // Then: the planned consumer's producer is not planned.
    assert_output(
        &output,
        2,
        "ERROR: planned job 'runtime_product' needs producer 'hosts', which is not planned\n",
    );
    Ok(())
}

#[test]
fn migration_ci_graph_planned_job_outside_static_graph_is_rejected() -> TestResult {
    // Given: the runner-contract job was renamed out of the static graph.
    graph_rejects(
        "quality",
        &[
            (
                "needs: [quality, runner_contract]",
                "needs: [quality, runner_contracts]",
            ),
            ("  runner_contract:\n", "  runner_contracts:\n"),
        ],
        // When/Then: the planned ID has no job to report it.
        "planned job 'runner_contract' is not in the lane workflow graph",
    )
}

#[test]
fn migration_ci_graph_summary_must_need_every_job_and_survive_failure() -> TestResult {
    // Given/When/Then: a summary that forgets a slice call.
    graph_rejects(
        "linux",
        &[("      - sdk\n      - product_smoke\n", "      - sdk\n")],
        "summary does not need lane job 'product_smoke'",
    )?;
    // Given/When/Then: a summary that also runs on cancellation.
    graph_rejects(
        "website",
        &[(
            "needs: [web]\n    if: ${{ !cancelled() }}",
            "needs: [web]\n    if: ${{ always() }}",
        )],
        "summary must run with if: ${{ !cancelled() }}",
    )
}

#[test]
fn migration_ci_graph_lane_identity_must_match_workflow() -> TestResult {
    // Given: the quality projection handed to the Linux lane.
    let golden = Golden::load("control-main")?;
    let output = Call::lane(
        &golden.lane_plan("quality"),
        &needs("quality", &["quality", "runner_contract"]),
    )
    .with("--workflow", &lane_workflow("linux"))
    .run_rust_only()?;
    // When/Then: the summary identity disagrees.
    assert_output(
        &output,
        2,
        "ERROR: lane workflow summary is 'CI / Linux', expected 'CI / Quality'\n",
    );
    Ok(())
}

#[test]
fn migration_ci_graph_failfast_cancellation_and_trust_policy() -> TestResult {
    // Given/When/Then: fail-fast fixed on for every profile.
    graph_rejects(
        "linux",
        &[(&format!("fail_fast: {PR_ONLY}"), "fail_fast: true")],
        "job 'rust_tests' fail_fast must be enabled only for pull requests",
    )?;
    // Given/When/Then: main runs cancelled by a newer revision.
    graph_rejects(
        "macos",
        &[(
            &format!("cancel-in-progress: {PR_ONLY}"),
            "cancel-in-progress: true",
        )],
        "lane concurrency must cancel only superseded pull requests",
    )?;
    // Given/When/Then: the HF credential reaches pull request runs.
    graph_rejects(
        "linux",
        &[(
            "HF_TOKEN: ${{ inputs.original_event_name == 'push' && secrets.HF_TOKEN || '' }}",
            "HF_TOKEN: ${{ secrets.HF_TOKEN }}",
        )],
        "job 'sdk' passes secret HF_TOKEN outside trusted push runs",
    )
}

#[test]
fn migration_ci_graph_mismatched_digest_and_projection_are_rejected() -> TestResult {
    let golden = Golden::load("control-main")?;
    let plan = golden.output("plan_json");
    let jobs = planned("control-main", "windows")?;
    let jobs = jobs.iter().map(String::as_str).collect::<Vec<_>>();
    let lane = golden.lane_plan("windows");
    let run = |lane_plan: &str, digest: &str, canonical: &str| {
        Call::lane(lane_plan, &needs("windows", &jobs))
            .with_text("--plan-digest", digest)
            .with_text("--canonical-plan", canonical)
            .run_rust_only()
    };
    // Given/When/Then: a digest that does not hash the canonical plan.
    let wrong = format!("{}0", &golden.output("plan_digest")[..63]);
    assert_output(
        &run(&lane, &wrong, plan)?,
        2,
        "ERROR: plan digest does not match the canonical plan\n",
    );
    // Given/When/Then: a projection whose slices drifted from the plan.
    let drifted = lane.replacen("\"required_slices\":[", "\"required_slices\":[\"web\",", 1);
    assert_output(
        &run(&drifted, golden.output("plan_digest"), plan)?,
        2,
        "ERROR: lane plan required_slices does not match the canonical plan\n",
    );
    // Given/When/Then: a projected row the canonical plan never selected.
    let invented = lane.replacen("\"id\":\"windows-cpu\"", "\"id\":\"windows-invented\"", 1);
    assert_output(
        &run(&invented, golden.output("plan_digest"), plan)?,
        2,
        "ERROR: lane plan matrix runtime_products row 'windows-invented' is not in the canonical plan\n",
    );
    Ok(())
}

fn entrypoints_reject(file: &str, edits: &[(&str, &str)], message: &str) -> TestResult {
    let scratch = Scratch::new("entry")?;
    let workflows = scratch.entrypoints(file, edits)?;
    let output = validate_entrypoints(&workflows)?;
    assert_output(&output, 2, &format!("ERROR: {file}: {message}\n"));
    Ok(())
}

#[test]
fn migration_ci_graph_pr_entrypoints_stay_native_and_unfiltered() -> TestResult {
    // Given/When/Then: a path filter that could leave the check absent.
    entrypoints_reject(
        "pr_linux.yml",
        &[(
            "types: [opened, synchronize, reopened, ready_for_review]",
            "types: [opened, synchronize, reopened, ready_for_review]\n    paths: ['crates/**']",
        )],
        "pull_request trigger must not filter paths",
    )?;
    // Given/When/Then: superseded PR runs that keep running.
    entrypoints_reject(
        "pr_macos.yml",
        &[("cancel-in-progress: true", "cancel-in-progress: false")],
        "concurrency must cancel superseded runs of the same PR",
    )?;
    // Given/When/Then: the Linux entry projecting the macOS lane.
    entrypoints_reject(
        "pr_linux.yml",
        &[(
            "steps.plan.outputs.linux_lane_plan",
            "steps.plan.outputs.macos_lane_plan",
        )],
        "plan output lane_plan must be ${{ steps.plan.outputs.linux_lane_plan }}",
    )?;
    // Given/When/Then: a PR lane handed a repository secret.
    entrypoints_reject(
        "pr_linux.yml",
        &[(
            "      supersession_key: pr-${{ github.event.pull_request.number }}\n",
            "      supersession_key: pr-${{ github.event.pull_request.number }}\n    secrets: inherit\n",
        )],
        "PR lane job must not pass secrets",
    )?;
    // Given/When/Then: a renamed stable required check.
    entrypoints_reject(
        "pr_windows.yml",
        &[("name: PR / Windows", "name: PR / Win")],
        "required job must be named 'PR / Windows'",
    )
}

#[test]
fn migration_ci_graph_main_and_controller_policy() -> TestResult {
    // Given/When/Then: main revisions cancelling one another.
    entrypoints_reject(
        "main_quality.yml",
        &[(
            "\njobs:\n",
            "\nconcurrency:\n  group: main\n  cancel-in-progress: true\n\njobs:\n",
        )],
        "main entrypoints must not declare concurrency",
    )?;
    // Given/When/Then: the manual controller reacting to pushes.
    entrypoints_reject(
        "ci-control.yml",
        &[(
            "on:\n  workflow_dispatch:\n",
            "on:\n  push:\n  workflow_dispatch:\n",
        )],
        "controller must be triggered only by workflow_dispatch",
    )?;
    // Given/When/Then: the controller dispatching without its digest check.
    entrypoints_reject(
        "ci-control.yml",
        &[("digest !== process.env.PLAN_DIGEST", "false")],
        "controller must verify the plan digest before dispatch",
    )
}
