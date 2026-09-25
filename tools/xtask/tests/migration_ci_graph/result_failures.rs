//! Lane result rules: each rejected input keeps the legacy `ERROR:` message
//! and status 2 (verified side by side when the legacy interpreter is set).
//! Each case has exactly one offending job because the legacy script walks
//! its planned jobs in Python set order.

use crate::support::{Call, TestResult, assert_output, text};

const QUALITY: &str =
    r#"{"lane":"quality","required":true,"required_slices":["quality"],"matrices":{}}"#;
const LINUX: &str = r#"{"lane":"linux","required":true,"required_slices":["runtime-product"],"matrices":{"hosts":[{"id":"linux-amd64-host"}],"runtime_products":[{"id":"linux-cpu"}],"rust_tests":[],"smoke":[],"sdk":[]}}"#;

fn linux_needs(runtime_product: &str) -> String {
    format!(
        r#"{{"ui_artifact":{{"result":"success"}},"static_abi":{{"result":"skipped"}},"rust_tests":{{"result":"skipped"}},"hosts":{{"result":"success"}},"native_runtimes":{{"result":"success"}},"runtime_product":{runtime_product},"kotlin_sdk_input":{{"result":"skipped"}},"sdk":{{"result":"skipped"}},"product_smoke":{{"result":"skipped"}}}}"#
    )
}

fn rejects(lane_plan: &str, needs: &str, message: &str) -> TestResult {
    let output = Call::lane(lane_plan, needs).run()?;
    assert_output(&output, 2, &format!("ERROR: {message}\n"));
    Ok(())
}

#[test]
fn migration_ci_graph_planned_failure_is_rejected() -> TestResult {
    // Given/When/Then: a planned consumer that failed.
    rejects(
        LINUX,
        &linux_needs(r#"{"result":"failure"}"#),
        "planned job 'runtime_product' finished with 'failure'",
    )
}

#[test]
fn migration_ci_graph_planned_cancellation_is_rejected() -> TestResult {
    // Given/When/Then: a planned consumer cancelled by fail-fast or supersession.
    rejects(
        LINUX,
        &linux_needs(r#"{"result":"cancelled"}"#),
        "planned job 'runtime_product' finished with 'cancelled'",
    )
}

#[test]
fn migration_ci_graph_selected_skip_is_rejected() -> TestResult {
    // Given/When/Then: a planned consumer that skipped (missing producer).
    rejects(
        LINUX,
        &linux_needs(r#"{"result":"skipped"}"#),
        "planned job 'runtime_product' finished with 'skipped'",
    )
}

#[test]
fn migration_ci_graph_unknown_and_absent_results_are_rejected() -> TestResult {
    // Given: an unknown planned result, a malformed need and an absent need.
    // When/Then: each fails with the legacy repr of the observed result.
    rejects(
        QUALITY,
        r#"{"quality":{"result":"neutral"}}"#,
        "planned job 'quality' finished with 'neutral'",
    )?;
    rejects(
        QUALITY,
        r#"{"quality":null}"#,
        "planned job 'quality' finished with None",
    )?;
    rejects(QUALITY, "{}", "planned job 'quality' finished with None")?;
    rejects(
        QUALITY,
        r#"{"quality":{"result":"success"},"x\n":{"result":1.5}}"#,
        "lane job 'x\\n' finished with 1.5",
    )
}

#[test]
fn migration_ci_graph_unplanned_job_must_skip() -> TestResult {
    // Given/When/Then: an unplanned job that ran anyway.
    rejects(
        QUALITY,
        r#"{"quality":{"result":"success"},"runner_contract":{"result":"success"}}"#,
        "lane job 'runner_contract' finished with 'success'",
    )
}

#[test]
fn migration_ci_graph_required_lane_needs_planned_jobs() -> TestResult {
    // Given/When/Then: a lane claiming to be required with nothing planned.
    rejects(
        r#"{"lane":"windows","required":true,"required_slices":[],"matrices":{"hosts":[],"runtime_products":[],"platform_checks":[]}}"#,
        "{}",
        "required lane has no planned jobs",
    )
}

#[test]
fn migration_ci_graph_malformed_plans_keep_legacy_diagnostics() -> TestResult {
    // Given/When/Then: each malformed projection names its legacy defect.
    for (plan, message) in [
        ("[]", "lane plan must be a JSON object"),
        (r#"{"required":1}"#, "lane plan required must be a boolean"),
        (
            r#"{"required":true}"#,
            "lane plan needs lane and required_slices",
        ),
        (
            r#"{"required":true,"lane":"xé","required_slices":[]}"#,
            "unknown CI lane 'xé'",
        ),
        (
            r#"{"required":true,"lane":"linux","required_slices":[],"matrices":[]}"#,
            "lane plan matrices must be an object",
        ),
        (
            r#"{"required":true,"lane":"linux","required_slices":[],"matrices":{"hosts":null}}"#,
            "lane plan matrix 'hosts' must be an array",
        ),
        (
            r#"{"required":true,"lane":"macos","required_slices":[],"matrices":{"hosts":[1]}}"#,
            "lane plan matrix hosts[0] needs an ID",
        ),
    ] {
        rejects(plan, "{}", message)?;
    }
    rejects(QUALITY, "[]", "needs must be a JSON object")
}

#[test]
fn migration_ci_graph_duplicate_row_id_is_rejected() -> TestResult {
    // Given: a projection whose host matrix repeats a row ID, which the
    // legacy set silently collapsed (Rust-only: the planner never emits it).
    let plan = r#"{"lane":"linux","required":true,"required_slices":[],"matrices":{"hosts":[{"id":"a"},{"id":"a"}]}}"#;
    let needs = r#"{"ui_artifact":{"result":"success"},"hosts":{"result":"success"}}"#;
    // When: it is validated.
    let output = Call::lane(plan, needs).run_rust_only()?;
    // Then: the duplicate fails the lane.
    assert_output(
        &output,
        2,
        "ERROR: lane plan matrix hosts contains duplicate ID 'a'\n",
    );
    Ok(())
}

#[test]
fn migration_ci_graph_invalid_json_keeps_prefix_and_status() -> TestResult {
    // Given/When: an unparsable lane plan (serde_json wording follows the
    // prefix instead of Python's JSONDecodeError text).
    let output = Call::lane("{", "{}").run_rust_only()?;
    // Then: the legacy prefix and status.
    assert_eq!(output.status.code(), Some(2));
    assert!(
        text(&output.stderr).starts_with("ERROR: lane plan is not valid JSON: "),
        "{}",
        text(&output.stderr)
    );
    Ok(())
}

#[test]
fn migration_ci_graph_argument_errors_match_argparse() -> TestResult {
    // Given/When/Then: argparse's usage line, program-prefixed error and 2.
    let usage = "usage: validate-ci-lane-results.py [-h] --lane-plan LANE_PLAN --needs NEEDS\n";
    for (args, error) in [
        (
            &[][..],
            "the following arguments are required: --lane-plan, --needs",
        ),
        (
            &["--bogus"][..],
            "the following arguments are required: --lane-plan, --needs",
        ),
        (
            &["--needs", "{}"][..],
            "the following arguments are required: --lane-plan",
        ),
        (
            &["--lane-plan"][..],
            "argument --lane-plan: expected one argument",
        ),
        (
            &["--lane-plan", "{}", "extra", "--needs", "{}"][..],
            "unrecognized arguments: extra",
        ),
    ] {
        let output = Call::raw(args).run()?;
        assert_output(
            &output,
            2,
            &format!("{usage}validate-ci-lane-results.py: error: {error}\n"),
        );
    }
    Ok(())
}
