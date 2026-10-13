//! `ci-ops collect-metrics` GitHub collection cases (`--workflow` run
//! lists, `--run-id` exact runs, job pagination, and every `gh` failure
//! mode), run through the stub harness in `ci_metrics_github_stub.rs`.

use crate::ci_metrics::{MARKDOWN_OUTPUT, OUTPUT, RAW_OUTPUT};
use crate::ci_metrics_github_stub::{
    Reply, failing, gh_case, job, json_reply, run_object, text_reply,
};
use crate::support::TestResult;
use serde_json::{Value, json};

fn jobs_page(jobs: Vec<Value>, total: Option<Value>) -> Reply {
    let mut page = json!({"jobs": jobs});
    if let Some(total) = total {
        page["total_count"] = total;
    }
    json_reply(&page)
}

fn key(text: &str) -> String {
    text.to_owned()
}

fn many_jobs(run: i64, count: i64) -> Vec<Value> {
    (0..count)
        .map(|index| job(run * 1000 + index, "matrix", 20 + index % 30))
        .collect()
}

#[test]
fn migration_ci_operations_ci_metrics_github_workflow_list() -> TestResult {
    let replies = vec![
        (
            key("run_list"),
            json_reply(&json!([
                run_object(301, "success"),
                run_object(302, "failure")
            ])),
        ),
        // A full page without a usable total asks for the next page.
        (key("jobs_301_1"), jobs_page(many_jobs(301, 100), None)),
        (
            key("jobs_301_2"),
            jobs_page(vec![job(1, "smoke", 30)], Some(json!("101"))),
        ),
        // A full page whose total_count is reached stops.
        (
            key("jobs_302_1"),
            jobs_page(many_jobs(302, 100), Some(json!(100))),
        ),
    ];
    let observed = gh_case(
        "github_workflow_list",
        &[
            "--repo",
            "Example/repo",
            "--workflow",
            "PR Builds",
            "--limit",
            "2",
            "--status",
            "completed",
            "--branch",
            "main",
            "--event",
            "push",
            "--created",
            ">=2026-07-01",
            "--label",
            "provider=stub",
            "--json-out",
            OUTPUT,
            "--markdown-out",
            MARKDOWN_OUTPUT,
        ],
        &replies,
        true,
    )?;
    assert_eq!(observed.code, 0, "{}", observed.stderr);
    assert_eq!(observed.gh_argv.len(), 4);
    assert_eq!(observed.gh_argv[0][..2], ["run", "list"]);
    let report: Value = serde_json::from_str(observed.output.as_deref().ok_or("no report")?)?;
    assert_eq!(report["source"]["kind"], "github");
    assert_eq!(report["source"]["description"], "Example/repo:PR Builds");
    Ok(())
}

#[test]
fn migration_ci_operations_ci_metrics_github_default_filters() -> TestResult {
    let replies = vec![
        (
            key("run_list"),
            json_reply(&json!([run_object(401, "success")])),
        ),
        (
            key("jobs_401_1"),
            jobs_page(vec![job(4011, "build", 20)], Some(json!(true))),
        ),
    ];
    let observed = gh_case(
        "github_status_all_raw",
        &[
            "--workflow",
            "ci.yml",
            "--status",
            "all",
            "--raw-out",
            RAW_OUTPUT,
            "--json-out",
            OUTPUT,
        ],
        &replies,
        true,
    )?;
    assert!(!observed.gh_argv[0].contains(&"--status".to_owned()));
    gh_case(
        "github_default_stdout",
        &["--workflow", "ci.yml"],
        &replies,
        true,
    )?;
    Ok(())
}

#[test]
fn migration_ci_operations_ci_metrics_github_exact_runs() -> TestResult {
    let replies = vec![
        (key("run_view_7"), json_reply(&run_object(7, "success"))),
        (
            key("jobs_7_1"),
            jobs_page(
                vec![job(71, "build", 20), job(72, "test", 25)],
                Some(json!(2)),
            ),
        ),
        (
            key("run_view_8"),
            json_reply(
                &json!({"databaseId": 8, "jobs": "old", "status": "completed", "conclusion": "success", "createdAt": "2026-07-01T00:00:00Z", "updatedAt": "2026-07-01T00:05:00Z"}),
            ),
        ),
        (key("jobs_8_1"), jobs_page(vec![job(81, "build", 40)], None)),
    ];
    let observed = gh_case(
        "github_exact_runs",
        &[
            "--run-id",
            "7",
            "--run-id",
            "8",
            "--repo",
            "Example/repo",
            "--json-out",
            OUTPUT,
            "--raw-out",
            RAW_OUTPUT,
        ],
        &replies,
        true,
    )?;
    assert_eq!(observed.code, 0, "{}", observed.stderr);
    let report: Value = serde_json::from_str(observed.output.as_deref().ok_or("no report")?)?;
    assert_eq!(report["source"]["description"], "Example/repo:7,8");
    assert_eq!(report["source"]["workflow"], Value::Null);
    Ok(())
}

#[test]
fn migration_ci_operations_ci_metrics_github_failures() -> TestResult {
    let list = || {
        (
            key("run_list"),
            json_reply(&json!([run_object(501, "success")])),
        )
    };
    let cases: Vec<(&str, Vec<(String, Reply)>)> = vec![
        (
            "github_error_exit_stderr",
            vec![(
                key("run_list"),
                failing(1, "ignored", "  HTTP 404: Not Found\n"),
            )],
        ),
        (
            "github_error_exit_stdout",
            vec![(key("run_list"), failing(4, "  auth required \n", " \n"))],
        ),
        (
            "github_error_exit_silent",
            vec![(key("run_list"), failing(2, "", ""))],
        ),
        (
            "github_error_invalid_json",
            vec![(key("run_list"), text_reply("[{\"databaseId\": 1,"))],
        ),
        (
            "github_error_list_not_array",
            vec![(key("run_list"), json_reply(&json!({"runs": []})))],
        ),
        (
            "github_error_invalid_run",
            vec![(key("run_list"), json_reply(&json!([{"databaseId": "x"}])))],
        ),
        (
            "github_error_jobs_shape",
            vec![
                list(),
                (key("jobs_501_1"), json_reply(&json!({"jobs": {}}))),
            ],
        ),
        (
            "github_error_jobs_exit",
            vec![
                list(),
                (key("jobs_501_1"), failing(1, "", "rate limited\n")),
            ],
        ),
        (
            "github_empty_run_list",
            vec![(key("run_list"), text_reply("[]\n"))],
        ),
    ];
    for (name, replies) in cases {
        let observed = gh_case(
            name,
            &["--workflow", "PR Builds", "--json-out", OUTPUT],
            &replies,
            true,
        )?;
        assert_ne!(observed.code, 0, "{name}");
        assert_eq!(observed.output, None, "{name}");
    }
    let view = vec![(key("run_view_9"), json_reply(&json!([1])))];
    gh_case(
        "github_error_run_view_shape",
        &["--run-id", "9"],
        &view,
        true,
    )?;
    Ok(())
}

#[test]
fn migration_ci_operations_ci_metrics_github_missing_gh() -> TestResult {
    let observed = gh_case(
        "github_error_missing_gh",
        &["--run-id", "3", "--json-out", OUTPUT],
        &[],
        false,
    )?;
    assert_eq!(observed.code, 2);
    assert_eq!(
        observed.stderr,
        "ci metrics error: gh is required for live collection\n"
    );
    assert!(observed.gh_argv.is_empty());
    Ok(())
}
