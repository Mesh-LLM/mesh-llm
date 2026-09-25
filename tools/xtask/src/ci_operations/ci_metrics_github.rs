//! Live GitHub collection of `scripts/collect-ci-metrics.py`: `gh_json`,
//! `fetch_jobs`, `fetch_exact_run` and `fetch_runs`, plus the `github`
//! report source `main` builds for `--workflow`/`--run-id`. The `gh` CLI is
//! found on `PATH` and invoked with the legacy argv; the process call sits
//! behind [`Gh`] so the collection logic does not own process spawning.

use crate::ci_operations::build_cache_tree::io_text;
use crate::ci_operations::ci_metrics_argv::Args;
use crate::ci_operations::ci_metrics_normalize::{Failure, Outcome};
use crate::ci_operations::ci_metrics_value::{Value, display, object, parse};
use crate::repository::python_text::strip;
use std::path::Path;
use std::process::{Command, Stdio};

/// `RUN_FIELDS`: the `gh run list/view --json` field list.
const RUN_FIELDS: &str = "databaseId,attempt,workflowName,displayTitle,event,status,conclusion,\
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
pub(crate) struct GhCli;

fn text_mode(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes)
        .replace("\r\n", "\n")
        .replace('\r', "\n")
}

impl Gh for GhCli {
    fn run(&mut self, arguments: &[String]) -> std::io::Result<GhOutput> {
        let output = Command::new("gh")
            .args(arguments)
            .stdin(Stdio::inherit())
            .output()?;
        Ok(GhOutput {
            success: output.status.success(),
            stdout: text_mode(&output.stdout),
            stderr: text_mode(&output.stderr),
        })
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

/// `isinstance(value, int)`: `bool` is an `int` subclass.
fn python_int(value: Option<&Value>) -> Option<Value> {
    value
        .filter(|value| matches!(value, Value::Int(_) | Value::BigInt(_) | Value::Bool(_)))
        .cloned()
}

fn at_least(count: usize, total: &Value) -> bool {
    let count = i128::try_from(count).unwrap_or(i128::MAX);
    match total {
        Value::Int(total) => count >= *total,
        Value::Bool(flag) => count >= i128::from(*flag),
        // Only a huge positive (or negative) integer fails to fit `i128`.
        Value::BigInt(text) => text.starts_with('-'),
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

/// `fetch_exact_run(repository, run_id)`.
fn fetch_exact_run(gh: &mut dyn Gh, repository: &str, run_id: i64) -> Outcome<Value> {
    let id = run_id.to_string();
    let arguments = owned(&[
        "run", "view", &id, "--repo", repository, "--json", RUN_FIELDS,
    ]);
    let mut run = gh_json(gh, &arguments)?;
    if !matches!(run, Value::Object(_)) {
        return Err(runtime(format!("invalid run response for {id}")));
    }
    let jobs = fetch_jobs(gh, repository, &id)?;
    set_jobs(&mut run, jobs);
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
    let Value::Array(mut runs) = gh_json(gh, &list_arguments(args))? else {
        return Err(runtime("invalid run list response".to_owned()));
    };
    for run in &mut runs {
        let Some(run_id) = python_int(run.get("databaseId")) else {
            return Err(runtime("run list contains an invalid run".to_owned()));
        };
        let jobs = fetch_jobs(gh, &args.repo, &display(&run_id))?;
        set_jobs(run, jobs);
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
