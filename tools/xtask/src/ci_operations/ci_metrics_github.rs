//! Live GitHub collection of `scripts/collect-ci-metrics.py`: `gh_json`,
//! `fetch_jobs`, `fetch_exact_run` and `fetch_runs`, plus the `github`
//! report source `main` builds for `--workflow`/`--run-id`. The `gh` CLI is
//! found on `PATH` and invoked with the legacy argv; the process call sits
//! behind [`Gh`] so the collection logic does not own process spawning.

use super::ci_metrics_github_process;
use crate::ci_operations::build_cache_tree::io_text;
use crate::ci_operations::ci_metrics_argv::Args;
use crate::ci_operations::ci_metrics_normalize::{Failure, Outcome};
use crate::ci_operations::ci_metrics_value::{Value, display, object, parse};
use crate::process::Cancellation;
use crate::repository::text::strip;
use std::path::Path;

/// `RUN_FIELDS`: the `gh run list/view --json` field list.
const RUN_FIELDS: &str = "databaseId,attempt,workflowName,displayTitle,event,status,conclusion,\
createdAt,startedAt,updatedAt,url,headSha,headBranch";
const COMPAT_RUN_FIELDS: &str = "databaseId,workflowName,displayTitle,event,status,conclusion,\
createdAt,startedAt,updatedAt,url,headSha,headBranch";

/// A finished `gh` invocation, with text-mode (universal newline) streams.
pub(crate) struct GhOutput {
    pub(crate) success: bool,
    pub(crate) stdout: String,
    pub(crate) stderr: String,
}

/// Runs `gh <arguments>`; the error is the spawn failure.
pub(crate) trait Gh {
    fn run(&mut self, arguments: &[String]) -> std::io::Result<GhOutput>;
}

/// The `gh` executable found on `PATH`.
pub(crate) struct GhCli {
    cancellation: Cancellation,
}

impl GhCli {
    pub(crate) fn new(cancellation: &Cancellation) -> Self {
        Self {
            cancellation: cancellation.clone(),
        }
    }
}

pub(super) fn text_mode(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes)
        .replace("\r\n", "\n")
        .replace('\r', "\n")
}

impl Gh for GhCli {
    fn run(&mut self, arguments: &[String]) -> std::io::Result<GhOutput> {
        ci_metrics_github_process::run(arguments, &self.cancellation)
    }
}

fn runtime(message: String) -> Failure {
    Failure::Reported(message)
}

/// `gh_json(arguments)`.
fn gh_json(gh: &mut dyn Gh, arguments: &[String]) -> Outcome<Value> {
    let command = format!("gh {}", arguments.join(" "));
    let output = gh.run(arguments).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            runtime("gh is required for live collection".to_owned())
        } else {
            runtime(io_text(&error, Path::new("gh")))
        }
    })?;
    if !output.success {
        let detail = [strip(&output.stderr), strip(&output.stdout)]
            .into_iter()
            .find(|text| !text.is_empty())
            .unwrap_or("unknown error");
        return Err(runtime(format!("{command} failed: {detail}")));
    }
    parse(output.stdout.as_bytes()).map_err(|_| runtime(format!("{command} returned invalid JSON")))
}

fn owned(parts: &[&str]) -> Vec<String> {
    parts.iter().map(|part| (*part).to_owned()).collect()
}

fn python_int(value: Option<&Value>) -> Option<Value> {
    value
        .filter(|value| matches!(value, Value::Int(number) if u64::try_from(*number).is_ok()))
        .cloned()
}

fn at_least(count: usize, total: &Value) -> bool {
    let count = i128::try_from(count).unwrap_or(i128::MAX);
    match total {
        Value::Int(total) => count >= *total,
        _ => false,
    }
}

/// `fetch_jobs(repository, run_id)`; `run_id` is already `str()`-formatted.
fn fetch_jobs(gh: &mut dyn Gh, repository: &str, run_id: &str) -> Outcome<Vec<Value>> {
    let mut jobs: Vec<Value> = Vec::new();
    let mut page = 1_u64;
    loop {
        let mut arguments = owned(&["api", "--method", "GET"]);
        arguments.push(format!("repos/{repository}/actions/runs/{run_id}/jobs"));
        arguments.extend(owned(&["-f", "filter=latest", "-f", "per_page=100", "-f"]));
        arguments.push(format!("page={page}"));
        let response = gh_json(gh, &arguments)?;
        let Some(Value::Array(page_jobs)) = response.get("jobs") else {
            return Err(runtime(format!("invalid jobs response for run {run_id}")));
        };
        let short_page = page_jobs.len() < 100;
        jobs.extend(page_jobs.iter().cloned());
        let total = python_int(response.get("total_count"));
        if short_page || total.is_some_and(|total| at_least(jobs.len(), &total)) {
            return Ok(jobs);
        }
        page += 1;
    }
}

/// `run["jobs"] = jobs`: an existing key keeps its position.
fn set_jobs(run: &mut Value, jobs: Vec<Value>) {
    if let Value::Object(entries) = run {
        match entries.iter_mut().find(|(key, _)| key == "jobs") {
            Some(entry) => entry.1 = Value::Array(jobs),
            None => entries.push(("jobs".to_owned(), Value::Array(jobs))),
        }
    }
}

// Some supported `gh` versions cannot expose `attempt` through `--json`.
// Retry only that explicit unsupported-field error, never a permission or
// transport failure. The REST run read supplies and binds the real attempt.
fn run_json(gh: &mut dyn Gh, arguments: &[String]) -> Outcome<(Value, bool)> {
    let unsupported = format!(
        "gh {} failed: Unknown JSON field: \"attempt\"",
        arguments.join(" ")
    );
    match gh_json(gh, arguments) {
        Ok(value) => Ok((value, false)),
        Err(Failure::Reported(message)) if message.starts_with(&unsupported) => {
            let mut compatible = arguments.to_vec();
            let field = compatible
                .iter_mut()
                .find(|value| value.as_str() == RUN_FIELDS)
                .ok_or_else(|| runtime("missing run field list for compatibility retry".into()))?;
            *field = COMPAT_RUN_FIELDS.to_owned();
            Ok((gh_json(gh, &compatible)?, true))
        }
        Err(error) => Err(error),
    }
}

#[derive(PartialEq, Eq)]
struct AttemptIdentity {
    attempt: i128,
    head: String,
    updated: String,
}

fn attempt_identity(
    gh: &mut dyn Gh,
    repository: &str,
    id: &Value,
    run: &Value,
) -> Outcome<AttemptIdentity> {
    let id_text = display(id);
    let mut arguments = owned(&["api", "--method", "GET"]);
    arguments.push(format!("repos/{repository}/actions/runs/{id_text}"));
    let response = gh_json(gh, &arguments)?;
    let Value::Int(expected_id) = id else {
        return Err(runtime("invalid run identity".into()));
    };
    if !matches!(response.get("id"), Some(Value::Int(actual)) if actual == expected_id) {
        return Err(runtime(format!(
            "run {id_text} identity changed during collection"
        )));
    }
    let Some(Value::Int(attempt)) = response.get("run_attempt") else {
        return Err(runtime(format!("run {id_text} has no valid API attempt")));
    };
    if *attempt <= 0 {
        return Err(runtime(format!("run {id_text} has no valid API attempt")));
    }
    let (Some(Value::Str(head)), Some(Value::Str(updated))) =
        (response.get("head_sha"), response.get("updated_at"))
    else {
        return Err(runtime(format!("run {id_text} API identity is incomplete")));
    };
    if head.is_empty()
        || updated.is_empty()
        || !matches!(run.get("headSha"), Some(Value::Str(actual)) if actual == head)
        || !matches!(run.get("updatedAt"), Some(Value::Str(actual)) if actual == updated)
    {
        return Err(runtime(format!("run {id_text} changed during collection")));
    }
    Ok(AttemptIdentity {
        attempt: *attempt,
        head: head.clone(),
        updated: updated.clone(),
    })
}

fn attach_jobs(
    gh: &mut dyn Gh,
    repository: &str,
    run: &mut Value,
    compatible: bool,
) -> Outcome<()> {
    let Some(id) = python_int(run.get("databaseId")) else {
        return Err(runtime("run contains an invalid run identity".into()));
    };
    if compatible && run.get("attempt").is_some() {
        return Err(runtime(
            "compatibility response unexpectedly included an attempt".into(),
        ));
    }
    let before = compatible
        .then(|| attempt_identity(gh, repository, &id, run))
        .transpose()?;
    let jobs = fetch_jobs(gh, repository, &display(&id))?;
    if let Some(before) = before {
        let after = attempt_identity(gh, repository, &id, run)?;
        if before != after {
            return Err(runtime(format!(
                "run {} changed during collection",
                display(&id)
            )));
        }
        let Value::Object(entries) = run else {
            return Err(runtime("invalid run response".into()));
        };
        entries.push(("attempt".to_owned(), Value::Int(before.attempt)));
    }
    set_jobs(run, jobs);
    Ok(())
}

/// `fetch_exact_run(repository, run_id)`.
fn fetch_exact_run(gh: &mut dyn Gh, repository: &str, run_id: i64) -> Outcome<Value> {
    let id = run_id.to_string();
    let arguments = owned(&[
        "run", "view", &id, "--repo", repository, "--json", RUN_FIELDS,
    ]);
    let (mut run, compatible) = run_json(gh, &arguments)?;
    if !matches!(run, Value::Object(_)) {
        return Err(runtime(format!("invalid run response for {id}")));
    }
    if !matches!(run.get("databaseId"), Some(Value::Int(actual)) if *actual == i128::from(run_id)) {
        return Err(runtime(format!("invalid run response for {id}")));
    }
    attach_jobs(gh, repository, &mut run, compatible)?;
    Ok(run)
}

fn list_arguments(args: &Args) -> Vec<String> {
    let limit = args.limit.to_string();
    let workflow = args.workflow.as_deref().unwrap_or_default();
    let mut command = owned(&[
        "run",
        "list",
        "--repo",
        &args.repo,
        "--workflow",
        workflow,
        "--limit",
        &limit,
        "--json",
        RUN_FIELDS,
    ]);
    let status = Some(args.status.as_str()).filter(|status| *status != "all");
    for (flag, value) in [
        ("--status", status),
        ("--branch", args.branch.as_deref()),
        ("--event", args.event.as_deref()),
        ("--created", args.created.as_deref()),
    ] {
        if let Some(value) = value.filter(|value| !value.is_empty()) {
            command.extend(owned(&[flag, value]));
        }
    }
    command
}

/// `fetch_runs(args)`.
pub(crate) fn fetch_runs(gh: &mut dyn Gh, args: &Args) -> Outcome<Vec<Value>> {
    if !args.run_id.is_empty() {
        return args
            .run_id
            .iter()
            .map(|run_id| fetch_exact_run(gh, &args.repo, *run_id))
            .collect();
    }
    let (value, compatible) = run_json(gh, &list_arguments(args))?;
    let Value::Array(mut runs) = value else {
        return Err(runtime("invalid run list response".to_owned()));
    };
    for run in &mut runs {
        if python_int(run.get("databaseId")).is_none() {
            return Err(runtime("run list contains an invalid run".to_owned()));
        }
        attach_jobs(gh, &args.repo, run, compatible)?;
    }
    Ok(runs)
}

/// The `github` report source `main` records for live collection.
pub(crate) fn github_source(args: &Args) -> Value {
    let target = if args.run_id.is_empty() {
        args.workflow.clone().unwrap_or_else(|| "None".to_owned())
    } else {
        let ids: Vec<String> = args.run_id.iter().map(i64::to_string).collect();
        ids.join(",")
    };
    object([
        ("kind", Value::text("github")),
        (
            "description",
            Value::text(&format!("{}:{target}", args.repo)),
        ),
        ("repository", Value::text(&args.repo)),
        ("workflow", Value::opt_text(args.workflow.as_deref())),
    ])
}

#[cfg(test)]
mod compatibility_tests {
    use super::*;
    use std::collections::VecDeque;

    struct Scripted {
        replies: VecDeque<GhOutput>,
        calls: Vec<Vec<String>>,
    }

    impl Gh for Scripted {
        fn run(&mut self, arguments: &[String]) -> std::io::Result<GhOutput> {
            self.calls.push(arguments.to_vec());
            self.replies
                .pop_front()
                .ok_or_else(|| std::io::Error::other("unexpected API call"))
        }
    }

    fn response(success: bool, stdout: &str, stderr: &str) -> GhOutput {
        GhOutput {
            success,
            stdout: stdout.into(),
            stderr: stderr.into(),
        }
    }

    const RUN: &str = r#"{"databaseId":7,"headSha":"abc","updatedAt":"2026-01-01T00:00:00Z"}"#;
    const BEFORE: &str =
        r#"{"id":7,"head_sha":"abc","updated_at":"2026-01-01T00:00:00Z","run_attempt":2}"#;
    const JOBS: &str = r#"{"total_count":1,"jobs":[{"id":11}]}"#;

    fn exact(after: &str) -> Scripted {
        Scripted {
            replies: VecDeque::from([
                response(false, "", "Unknown JSON field: \"attempt\""),
                response(true, RUN, ""),
                response(true, BEFORE, ""),
                response(true, JOBS, ""),
                response(true, after, ""),
            ]),
            calls: Vec::new(),
        }
    }

    #[test]
    fn older_gh_uses_verified_api_attempt_without_fabricating_it() {
        let mut gh = exact(BEFORE);
        let run = fetch_exact_run(&mut gh, "owner/repo", 7).unwrap();
        assert!(matches!(run.get("attempt"), Some(Value::Int(2))));
        assert!(matches!(run.get("jobs"), Some(Value::Array(jobs)) if jobs.len() == 1));
        assert_eq!(gh.calls.len(), 5);
        assert!(gh.calls[0].iter().any(|value| value == RUN_FIELDS));
        assert!(gh.calls[1].iter().any(|value| value == COMPAT_RUN_FIELDS));
        assert_eq!(gh.calls[2][3], "repos/owner/repo/actions/runs/7");
        assert_eq!(gh.calls[3][3], "repos/owner/repo/actions/runs/7/jobs");
        assert_eq!(gh.calls[4][3], "repos/owner/repo/actions/runs/7");
    }

    #[test]
    fn older_gh_workflow_list_binds_each_api_attempt() {
        let mut gh = Scripted {
            replies: VecDeque::from([
                response(false, "", "Unknown JSON field: \"attempt\""),
                response(true, &format!("[{RUN}]"), ""),
                response(true, BEFORE, ""),
                response(true, JOBS, ""),
                response(true, BEFORE, ""),
            ]),
            calls: Vec::new(),
        };
        let args = Args {
            workflow: Some("ci.yml".into()),
            repo: "owner/repo".into(),
            ..Args::default()
        };
        let runs = fetch_runs(&mut gh, &args).unwrap();
        assert_eq!(runs.len(), 1);
        assert!(matches!(runs[0].get("attempt"), Some(Value::Int(2))));
        assert!(matches!(runs[0].get("jobs"), Some(Value::Array(jobs)) if jobs.len() == 1));
        assert_eq!(gh.calls.len(), 5);
        assert!(gh.calls[0].iter().any(|value| value == RUN_FIELDS));
        assert!(gh.calls[1].iter().any(|value| value == COMPAT_RUN_FIELDS));
    }

    #[test]
    fn incompatible_attempt_or_head_changes_refuse_a_mixed_report() {
        for after in [
            r#"{"id":7,"head_sha":"abc","updated_at":"2026-01-01T00:00:00Z","run_attempt":3}"#,
            r#"{"id":7,"head_sha":"other","updated_at":"2026-01-01T00:00:00Z","run_attempt":2}"#,
            r#"{"id":7,"head_sha":"abc","updated_at":"2026-01-01T00:00:00Z","run_attempt":0}"#,
        ] {
            let mut gh = exact(after);
            assert!(fetch_exact_run(&mut gh, "owner/repo", 7).is_err());
        }
    }

    #[test]
    fn authorization_failure_never_triggers_compatibility_retry() {
        let mut gh = Scripted {
            replies: VecDeque::from([response(false, "", "HTTP 403: Resource not accessible")]),
            calls: Vec::new(),
        };
        assert!(fetch_exact_run(&mut gh, "owner/repo", 7).is_err());
        assert_eq!(gh.calls.len(), 1);
    }

    #[test]
    fn compatibility_response_cannot_smuggle_an_unverified_attempt() {
        let mut gh = Scripted {
            replies: VecDeque::from([
                response(false, "", "Unknown JSON field: \"attempt\""),
                response(
                    true,
                    r#"{"databaseId":7,"headSha":"abc","updatedAt":"2026-01-01T00:00:00Z","attempt":99}"#,
                    "",
                ),
            ]),
            calls: Vec::new(),
        };
        assert!(fetch_exact_run(&mut gh, "owner/repo", 7).is_err());
        assert_eq!(gh.calls.len(), 2);
    }
}
