//! Bind direct PR and main entrypoints to the planner's source and lane output.
//! This checks the checked-in wiring, not a GitHub run or authenticated identity.
use super::super as workflows;
use workflows::Node;

fn by_id<'a>(plan: &'a Node, id: &str) -> &'a Node {
    let matches = workflows::steps(plan)
        .iter()
        .filter(|step| workflows::text(step, "id") == Some(id))
        .collect::<Vec<_>>();
    assert_eq!(matches.len(), 1, "one {id} step is required");
    matches[0]
}

fn mismatches(document: &Node, lane: &str, event: &str) -> Vec<&'static str> {
    let plan = workflows::job(document, "plan");
    let outputs = plan.get("outputs").unwrap();
    let identity = by_id(plan, "identity").get("env").unwrap();
    let changes = by_id(plan, "changes");
    let planner = by_id(plan, "plan");
    let lane_job = workflows::job(document, "lane");
    let (base, head, draft) = if event == "push" {
        ("${{ github.event.before }}", "${{ github.sha }}", "false")
    } else {
        (
            "${{ github.event.pull_request.base.sha }}",
            "${{ github.event.pull_request.head.sha }}",
            "${{ github.event.pull_request.draft || false }}",
        )
    };
    let planned_base = if event == "push" {
        "${{ steps.identity.outputs.base_sha }}"
    } else {
        base
    };
    let mut errors = Vec::new();
    for (key, expected) in [("BASE_SHA", base), ("SOURCE_SHA", head)] {
        if workflows::text(identity, key) != Some(expected) {
            errors.push("event identity");
        }
    }
    for (key, expected) in [
        ("event_name", event),
        ("base_sha", planned_base),
        ("head_sha", "${{ steps.identity.outputs.source_sha }}"),
    ] {
        if workflows::input(changes, key) != Some(expected) {
            errors.push("change inputs");
        }
    }
    if workflows::text(changes, "uses") != Some("./.github/actions/compute-changes") {
        errors.push("change action");
    }
    for (key, expected) in [
        ("event_name", event),
        ("source_sha", "${{ steps.identity.outputs.source_sha }}"),
        ("base_sha", planned_base),
        ("draft", draft),
        (
            "changed_files",
            "${{ steps.change_limit.outputs.force_all == 'true' && '__force_all__' || steps.changes.outputs.changed_files }}",
        ),
        (
            "affected_crates",
            "${{ steps.changes.outputs.affected_crates }}",
        ),
    ] {
        if workflows::input(planner, key) != Some(expected) {
            errors.push("planner inputs");
        }
    }
    if workflows::text(planner, "uses") != Some("./.github/actions/plan-ci") {
        errors.push("planner action");
    }
    for (key, expected) in [
        (
            "source_sha",
            "${{ steps.identity.outputs.source_sha }}".to_owned(),
        ),
        (
            "plan_digest",
            "${{ steps.plan.outputs.plan_digest }}".to_owned(),
        ),
        (
            "lane_plan",
            format!("${{{{ steps.plan.outputs.{lane}_lane_plan }}}}"),
        ),
    ] {
        if workflows::text(outputs, key) != Some(expected.as_str()) {
            errors.push("plan outputs");
        }
    }
    if workflows::text(lane_job, "if")
        != Some(
            "${{ needs.plan.result == 'success' && fromJson(needs.plan.outputs.lane_plan).required }}",
        )
    {
        errors.push("lane admission");
    }
    if !lane_job
        .get("needs")
        .is_some_and(|needs| needs.list().contains(&"plan"))
    {
        errors.push("lane dependency");
    }
    errors
}

#[test]
fn all_direct_entrypoints_bind_event_identity_and_correct_lane_projection() {
    for (prefix, event) in [("main", "push"), ("pr", "pull_request")] {
        for lane in ["quality", "website", "linux", "macos", "windows"] {
            let file = format!("{prefix}_{lane}.yml");
            let errors = mismatches(&workflows::document(&file), lane, event);
            assert!(errors.is_empty(), "{file}: {errors:?}");
        }
    }
}

#[test]
fn entrypoint_check_rejects_wrong_head_base_and_lane_projection() {
    let source = workflows::source("pr_linux.yml");
    for (before, after, expected) in [
        (
            "SOURCE_SHA: ${{ github.event.pull_request.head.sha }}",
            "SOURCE_SHA: ${{ github.sha }}",
            "event identity",
        ),
        (
            "base_sha: ${{ github.event.pull_request.base.sha }}",
            "base_sha: ${{ github.sha }}",
            "change inputs",
        ),
        (
            "lane_plan: ${{ steps.plan.outputs.linux_lane_plan }}",
            "lane_plan: ${{ steps.plan.outputs.windows_lane_plan }}",
            "plan outputs",
        ),
    ] {
        assert_eq!(
            source.matches(before).count(),
            if expected == "change inputs" { 2 } else { 1 }
        );
        let changed = source.replacen(before, after, 1);
        let document = workflows::workflow_yaml::parse(&changed).unwrap();
        assert!(mismatches(&document, "linux", "pull_request").contains(&expected));
    }
}
