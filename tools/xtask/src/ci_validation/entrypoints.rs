use crate::command::{DynResult, ensure_contains, ensure_not_contains};

pub(super) fn check_workflow_invariants(
    release_workflow: &str,
    pr_workflows: &[(&str, String)],
    main_workflows: &[(&str, String)],
    website_pages: &str,
) -> DynResult<()> {
    for (text, needle, context) in [
        (
            release_workflow,
            "compose_windows_gpu:",
            "release Windows GPU composition",
        ),
        (
            release_workflow,
            "publish_crates_preflight:",
            "release crates.io preflight",
        ),
        (
            website_pages,
            "name: Public Website Deploy",
            "public website deploy workflow",
        ),
        (
            website_pages,
            "branches: [main]",
            "public website main trigger",
        ),
    ] {
        ensure_contains(text, needle, context)?;
    }

    for (lane, workflow) in pr_workflows {
        ensure_contains(workflow, "pull_request:", &format!("PR {lane} trigger"))?;
        ensure_contains(
            workflow,
            &format!("uses: Mesh-LLM/mesh-llm/.github/workflows/ci-{lane}-lane.yml@main"),
            &format!("PR protected native {lane} lane call"),
        )?;
        ensure_contains(
            workflow,
            "needs: [plan, lane]",
            &format!("PR {lane} required job"),
        )?;
        ensure_not_contains(
            workflow,
            "pull_request_target",
            &format!("PR {lane} trust boundary"),
        )?;
        ensure_not_contains(workflow, "secrets:", &format!("PR {lane} secret boundary"))?;
    }
    for (lane, workflow) in main_workflows {
        ensure_contains(
            workflow,
            "push:\n    branches: [main]",
            &format!("main {lane} trigger"),
        )?;
        ensure_contains(
            workflow,
            &format!("uses: ./.github/workflows/ci-{lane}-lane.yml"),
            &format!("main same-commit {lane} lane call"),
        )?;
        ensure_contains(
            workflow,
            "needs: [plan, lane]",
            &format!("main {lane} required job"),
        )?;
        ensure_not_contains(
            workflow,
            "createWorkflowDispatch",
            &format!("main {lane} native visibility"),
        )?;
        ensure_not_contains(
            workflow,
            "concurrency:",
            &format!("main {lane} exhaustive evidence"),
        )?;
    }
    Ok(())
}

pub(super) fn check_orchestrator_invariants(
    controller: &str,
    pr_workflows: &[(&str, String)],
    main_workflows: &[(&str, String)],
    lanes: &[(&str, String)],
    compute_changes: &str,
) -> DynResult<()> {
    ensure_contains(
        controller,
        "name: CI · Manual Full",
        "manual-full workflow identity",
    )?;
    ensure_contains(
        controller,
        "uses: ./.github/actions/plan-ci",
        "controller canonical planner call",
    )?;
    ensure_contains(
        controller,
        "github.rest.actions.createWorkflowDispatch",
        "controller native lane dispatch",
    )?;
    ensure_contains(controller, "workflow_dispatch:", "manual-full trigger")?;
    ensure_not_contains(controller, "workflow_run:", "manual controller trigger")?;
    ensure_not_contains(controller, "\n  push:\n", "manual controller push trigger")?;
    let lane_workflow = |name: &str| {
        lanes
            .iter()
            .find_map(|(lane, workflow)| (*lane == name).then_some(workflow.as_str()))
            .unwrap_or("")
    };
    for lane in ["quality", "website", "linux", "macos", "windows"] {
        let pr_workflow = pr_workflows
            .iter()
            .find_map(|(name, workflow)| (*name == lane).then_some(workflow.as_str()))
            .unwrap_or("");
        ensure_contains(
            pr_workflow,
            &format!("uses: Mesh-LLM/mesh-llm/.github/workflows/ci-{lane}-lane.yml@main"),
            &format!("native PR {lane} lane call"),
        )?;
        let main_workflow = main_workflows
            .iter()
            .find_map(|(name, workflow)| (*name == lane).then_some(workflow.as_str()))
            .unwrap_or("");
        ensure_contains(
            main_workflow,
            &format!("uses: ./.github/workflows/ci-{lane}-lane.yml"),
            &format!("native main {lane} lane call"),
        )?;
        ensure_contains(
            lane_workflow(lane),
            "workflow_call:",
            &format!("{lane} reusable lane trigger"),
        )?;
    }
    ensure_contains(
        lane_workflow("quality"),
        "uses: ./.github/workflows/ci-quality-slice.yml",
        "quality lane slice call",
    )?;
    for lane in ["linux", "macos", "windows"] {
        for component in ["host", "runtime", "product"] {
            ensure_contains(
                lane_workflow(lane),
                &format!("uses: ./.github/workflows/ci-{lane}-{component}-slice.yml"),
                &format!("{lane} lane {component} call"),
            )?;
        }
        for legacy in ["ci-host-slice.yml", "ci-runtime-product-slice.yml"] {
            ensure_not_contains(
                lane_workflow(lane),
                legacy,
                &format!("{lane} lane cross-platform placeholder graph"),
            )?;
        }
    }
    for (lane, component) in [
        ("linux", "product-smoke"),
        ("linux", "sdk"),
        ("macos", "product-smoke"),
        ("macos", "sdk"),
    ] {
        ensure_contains(
            lane_workflow(lane),
            &format!("uses: ./.github/workflows/ci-{lane}-{component}-slice.yml"),
            &format!("{lane} lane {component} call"),
        )?;
    }
    ensure_contains(
        controller,
        "name: 'CI Required'",
        "stable dispatched CI required check",
    )?;
    ensure_contains(
        compute_changes,
        "changed_files:",
        "changed-file planner input",
    )?;
    ensure_contains(
        compute_changes,
        "affected_crates:",
        "affected-crate planner input",
    )?;
    Ok(())
}
