//! Run each direct entrypoint's actual required-result guard with local inputs.
//! These cases do not establish a hosted check, policy activation or job identity.
use super::super as workflows;
use crate::support::Fixture;
use std::process::Command;

#[test]
fn entrypoint_guards_refuse_invalid_plans_and_incomplete_required_lanes() {
    let cases = [
        (
            "required succeeds",
            "{\"required\":true}",
            "success",
            "success",
            true,
        ),
        (
            "required skipped",
            "{\"required\":true}",
            "success",
            "skipped",
            false,
        ),
        (
            "required failed",
            "{\"required\":true}",
            "success",
            "failure",
            false,
        ),
        (
            "required cancelled",
            "{\"required\":true}",
            "success",
            "cancelled",
            false,
        ),
        (
            "unneeded skipped",
            "{\"required\":false}",
            "success",
            "skipped",
            true,
        ),
        (
            "unneeded ran",
            "{\"required\":false}",
            "success",
            "success",
            false,
        ),
        (
            "failed plan",
            "{\"required\":false}",
            "failure",
            "skipped",
            false,
        ),
        ("missing flag", "{}", "success", "skipped", false),
        (
            "string flag",
            "{\"required\":\"false\"}",
            "success",
            "skipped",
            false,
        ),
        (
            "null flag",
            "{\"required\":null}",
            "success",
            "skipped",
            false,
        ),
        (
            "duplicate flag",
            "{\"required\":true,\"required\":false}",
            "success",
            "skipped",
            false,
        ),
        (
            "escaped duplicate flag",
            "{\"required\":true,\"requ\\u0069red\":false}",
            "success",
            "skipped",
            false,
        ),
        ("array", "[]", "success", "skipped", false),
        (
            "multiple plans",
            "{\"required\":false}\n{\"required\":false}",
            "success",
            "skipped",
            false,
        ),
        ("malformed", "{", "success", "skipped", false),
    ];
    for prefix in ["main", "pr"] {
        for lane in ["quality", "website", "linux", "macos", "windows"] {
            let file = format!("{prefix}_{lane}.yml");
            let document = workflows::document(&file);
            let required = workflows::job(&document, "required");
            assert_eq!(
                required.get("needs").unwrap().list(),
                ["plan", "lane"],
                "{file}"
            );
            assert_eq!(
                workflows::text(required, "if"),
                Some("${{ !cancelled() }}"),
                "{file}"
            );
            let steps = workflows::steps(required);
            assert_eq!(steps.len(), 1, "{file}");
            let step = &steps[0];
            let environment = step.get("env").unwrap();
            for (key, value) in [
                ("LANE_PLAN", "${{ needs.plan.outputs.lane_plan }}"),
                ("LANE_RESULT", "${{ needs.lane.result }}"),
                ("PLAN_RESULT", "${{ needs.plan.result }}"),
                ("PLAN_DIGEST", "${{ needs.plan.outputs.plan_digest }}"),
            ] {
                assert_eq!(
                    workflows::text(environment, key),
                    Some(value),
                    "{file}/{key}"
                );
            }
            let script = workflows::text(step, "run").unwrap();
            for (case, plan, plan_result, lane_result, expected) in cases {
                let fixture = Fixture::new();
                let summary = fixture.path().join("summary");
                let mut command = Command::new("bash");
                command
                    .current_dir(fixture.path())
                    .env("LANE_PLAN", plan)
                    .env("PLAN_RESULT", plan_result)
                    .env("LANE_RESULT", lane_result)
                    .env("PLAN_DIGEST", "fixture")
                    .env("GITHUB_STEP_SUMMARY", &summary)
                    .args(["-c", script]);
                let result = fixture.run(command);
                assert_eq!(
                    result.status.success(),
                    expected,
                    "{file}/{case}: {result:?}"
                );
                assert_eq!(summary.exists(), expected, "{file}/{case}");
            }
        }
    }
}
