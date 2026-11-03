//! Planner and selector projections for Quality and its sentinel.

use super::super as workflows;

use super::{plan, support};

#[test]
fn quality_callers_follow_planned_policy_and_keep_the_sentinel_separate() {
    let base = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    let main = plan("main", "push", base, "mesh/crates/mesh-llm/src/main.rs");
    let control = plan("main", "push", base, ".github/workflows/main_linux.yml");
    let pr = plan(
        "pr-ready",
        "pull_request",
        base,
        "mesh/crates/mesh-llm/src/main.rs",
    );
    let manual = plan("manual-full", "workflow_dispatch", base, "README.md");
    assert_eq!(main["signals"]["force_hosted"], false);
    assert_eq!(control["signals"]["force_hosted"], true);
    assert!(
        main["required_slices"]
            .as_array()
            .unwrap()
            .iter()
            .any(|slice| slice == "quality")
    );
    assert!(!main["matrices"]["clippy"].as_array().unwrap().is_empty());
    assert_quality_caller_wiring();
    assert_quality_selection_cases(&main, &control, &pr, &manual);
    assert_quality_sentinel_cases();
}

fn assert_quality_caller_wiring() {
    let lane = workflows::document("ci-quality-lane.yml");
    let caller = workflows::job(&lane, "quality");
    for (key, expected) in [
        (
            "force_hosted",
            "${{ toJSON(fromJson(inputs.lane_plan_json).signals.force_hosted) != 'false' }}",
        ),
        (
            "clippy_matrix",
            "${{ toJson(fromJson(inputs.lane_plan_json).matrices.clippy) }}",
        ),
        ("source_sha", "${{ inputs.source_sha }}"),
        ("original_event_name", "${{ inputs.original_event_name }}"),
        ("use_depot", "${{ inputs.use_depot }}"),
    ] {
        assert_eq!(workflows::input(caller, key), Some(expected), "{key}");
    }
    let slice = workflows::document("ci-quality-slice.yml");
    let policy = workflows::job(&slice, "runner_policy");
    let normal = workflows::steps(policy)
        .iter()
        .find(|step| workflows::text(step, "id") == Some("policy"))
        .unwrap();
    let sentinel = workflows::steps(policy)
        .iter()
        .find(|step| workflows::text(step, "id") == Some("sentinel_policy"))
        .unwrap();
    for selector in [normal, sentinel] {
        assert_eq!(
            workflows::text(selector, "uses"),
            Some("./.github/actions/select-ci-runners")
        );
        assert_eq!(
            workflows::input(selector, "force_hosted"),
            Some("${{ inputs.force_hosted }}")
        );
    }
    assert_eq!(
        workflows::input(normal, "manual_use_depot"),
        Some("${{ inputs.use_depot }}")
    );
    for (key, expected) in [
        ("depot_main_enabled", "false"),
        ("depot_pr_enabled", "false"),
        ("manual_use_depot", "false"),
        ("pr_canary_ref", "${{ vars.DEPOT_PR_SENTINEL_REF }}"),
    ] {
        assert_eq!(workflows::input(sentinel, key), Some(expected), "{key}");
    }
    for (job, output) in [
        ("quality_contracts", "runner_4"),
        ("rust_clippy", "runner_8"),
    ] {
        assert_eq!(
            workflows::text(policy.get("outputs").unwrap(), output),
            Some(format!("${{{{ steps.policy.outputs.{output} }}}}").as_str())
        );
        assert_eq!(
            workflows::text(workflows::job(&slice, job), "runs-on"),
            Some(format!("${{{{ needs.runner_policy.outputs.{output} }}}}").as_str())
        );
    }
    let sentinel_job = workflows::job(&slice, "authority_sentinel");
    let guard = workflows::text(sentinel_job, "if").unwrap();
    for condition in [
        "authority_sentinel_depot_enabled",
        "inputs.original_event_name == 'pull_request'",
        "github.event_name == 'pull_request'",
        "github.ref == vars.DEPOT_PR_SENTINEL_REF",
    ] {
        assert!(guard.contains(condition), "{condition}");
    }
}

fn assert_quality_selection_cases(
    main: &serde_json::Value,
    control: &serde_json::Value,
    pr: &serde_json::Value,
    manual: &serde_json::Value,
) {
    for (name, plan, event, origin, head_repo, reference, date, depot) in [
        (
            "main",
            main,
            "push",
            "",
            "Mesh-LLM/mesh-llm",
            "refs/heads/main",
            "2026-11-03",
            true,
        ),
        (
            "control",
            control,
            "push",
            "",
            "Mesh-LLM/mesh-llm",
            "refs/heads/main",
            "2026-11-03",
            false,
        ),
        (
            "expired PR",
            pr,
            "pull_request",
            "",
            "Mesh-LLM/mesh-llm",
            "refs/pull/12/merge",
            "2026-09-14",
            false,
        ),
        (
            "fork",
            pr,
            "pull_request",
            "",
            "fork/mesh-llm",
            "refs/pull/12/merge",
            "2026-09-13",
            false,
        ),
        (
            "PR dispatch",
            manual,
            "workflow_dispatch",
            "pull_request",
            "Mesh-LLM/mesh-llm",
            "refs/heads/main",
            "2026-09-13",
            false,
        ),
    ] {
        let fixture = support::Fixture::new();
        let force = plan["signals"]["force_hosted"].as_bool().unwrap();
        let result = fixture.selector(&[
            ("INPUT_EVENT_NAME", event),
            ("INPUT_ORIGINAL_EVENT_NAME", origin),
            ("INPUT_HEAD_REPOSITORY", head_repo),
            ("INPUT_REF", reference),
            ("FIXTURE_DATE", date),
            ("INPUT_DEPOT_MAIN_ENABLED", "true"),
            ("INPUT_DEPOT_PR_ENABLED", "true"),
            ("INPUT_MANUAL_USE_DEPOT", "true"),
            ("INPUT_FORCE_HOSTED", if force { "true" } else { "false" }),
        ]);
        assert!(result.status.success(), "{name}: {result:?}");
        let selected = fixture.outputs();
        assert_eq!(selected["depot_enabled"], depot.to_string(), "{name}");
        assert_eq!(
            selected["runner_4"],
            if depot {
                "depot-ubuntu-24.04-4"
            } else {
                "ubuntu-24.04"
            },
            "{name}"
        );
        assert_eq!(
            selected["runner_8"],
            if depot {
                "depot-ubuntu-24.04-8"
            } else {
                "ubuntu-24.04"
            },
            "{name}"
        );
        assert_eq!(selected["allow_depot_remote_cache"], "false", "{name}");
        fixture.0.close().unwrap();
    }
}

fn assert_quality_sentinel_cases() {
    // The exact-ref sentinel is an isolated historical fixture, not a live
    // activation. Both ordinary gates and manual intent remain disabled.
    for (canary, expected) in [
        ("", "false"),
        ("refs/pull/12/merge", "true"),
        ("refs/pull/13/merge", "false"),
    ] {
        let fixture = support::Fixture::new();
        let result = fixture.selector(&[
            ("FIXTURE_DATE", "2026-11-03"),
            ("INPUT_DEPOT_MAIN_ENABLED", "false"),
            ("INPUT_DEPOT_PR_ENABLED", "false"),
            ("INPUT_MANUAL_USE_DEPOT", "false"),
            ("INPUT_PR_CANARY_REF", canary),
        ]);
        assert!(result.status.success(), "{result:?}");
        let selected = fixture.outputs();
        assert_eq!(selected["depot_enabled"], expected);
        assert_eq!(selected["allow_depot_remote_cache"], "false");
        fixture.0.close().unwrap();
    }
}
