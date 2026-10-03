//! `ci-ops collect-metrics`: the Rust owner of the offline JSON core of
//! `scripts/collect-ci-metrics.py`. Loads saved run JSON (`--input`),
//! normalizes it, analyzes timing, queue and runner metrics, and writes
//! `json.dumps(report, indent=2, sort_keys=True)` (plus `--raw-out`) and the
//! Markdown summary (`--markdown-out`, or stdout without any output path).
//! `--compare-input` analyzes a baseline file and adds the report's
//! `comparison`. Without `--input`, runs come from GitHub through the `gh`
//! CLI (`ci_metrics_github`).

use super::ci_metrics_transport;
use crate::ci_operations::build_cache_tree::io_text;
use crate::ci_operations::ci_metrics_argv::{Args, parse};
use crate::ci_operations::ci_metrics_compare::compare_reports;
use crate::ci_operations::ci_metrics_github::{GhCli, fetch_runs, github_source};
use crate::ci_operations::ci_metrics_input::{labels, load_runs};
use crate::ci_operations::ci_metrics_markdown::render_markdown;
use crate::ci_operations::ci_metrics_normalize::{Failure, Outcome, normalize_run};
use crate::ci_operations::ci_metrics_report::{Request, analyze, validate_finite};
use crate::ci_operations::ci_metrics_value::{Value, dumps, object};
use crate::ci_plan::catalog::python_path_display;
use crate::command_interrupt::Interrupt;
use crate::process::Cancellation;
use crate::repository::check_report::CheckReport;
use std::io::Read as _;
use std::path::Path;

pub(crate) fn run(args: &[String]) -> CheckReport {
    let args = match parse(args) {
        Ok(args) => args,
        Err(report) => return report,
    };
    let mut report = CheckReport::default();
    match collect_owned(&args, &mut report) {
        Ok(()) => {}
        Err(Failure::Reported(message)) => {
            report
                .stderr
                .push_str(&format!("ci metrics error: {message}\n"));
            report.code = 2;
        }
        Err(Failure::Uncaught(message)) => {
            report.stderr.push_str(&format!("{message}\n"));
            report.code = 1;
        }
    }
    report
}

fn collect_owned(args: &Args, out: &mut CheckReport) -> Outcome<()> {
    if args.input.as_deref() == Some("-") || args.compare_input.as_deref() == Some("-") {
        return collect(args, out, &Cancellation::default());
    }
    let interrupt = Interrupt::install().map_err(|error| Failure::Reported(error.to_string()))?;
    let result = collect(args, out, &interrupt.cancellation());
    interrupt
        .finish()
        .map_err(|error| Failure::Reported(error.to_string()))?;
    result
}

fn truthy(value: Option<&String>) -> Option<&str> {
    value.map(String::as_str).filter(|text| !text.is_empty())
}

fn file_source(path: &str) -> Value {
    object([
        ("kind", Value::text("file")),
        ("description", Value::text(path)),
    ])
}

fn analyze_runs(
    args: &Args,
    raw_runs: &[Value],
    source: Value,
    labels: impl FnOnce() -> Outcome<Vec<(String, Value)>>,
) -> Outcome<Value> {
    let runs = raw_runs
        .iter()
        .map(normalize_run)
        .collect::<Outcome<Vec<_>>>()?;
    let request = Request {
        status: &args.status,
        top: request_top(args),
        source,
        labels: labels()?,
    };
    analyze(&runs, request)
}

/// `--input` runs with a `file` source, or live GitHub runs.
fn primary_runs(args: &Args, cancellation: &Cancellation) -> Outcome<(Vec<Value>, Value)> {
    match truthy(args.input.as_ref()) {
        Some(path) => Ok((
            load_runs(&read_input(path, cancellation)?)?,
            file_source(path),
        )),
        None => Ok((
            fetch_runs(&mut GhCli::new(cancellation), args)?,
            github_source(args),
        )),
    }
}

fn collect(args: &Args, out: &mut CheckReport, cancellation: &Cancellation) -> Outcome<()> {
    let (raw_runs, source) = primary_runs(args, cancellation)?;
    let mut report = analyze_runs(args, &raw_runs, source, || labels(&args.label))?;
    if let Some(path) = truthy(args.compare_input.as_ref()) {
        let cohort = || Ok(vec![("cohort".to_owned(), Value::text("baseline"))]);
        let baseline_runs = load_runs(&read_input(path, cancellation)?)?;
        let baseline = analyze_runs(args, &baseline_runs, file_source(path), cohort)?;
        let comparison = compare_reports(&baseline, &report);
        if let Value::Object(entries) = &mut report {
            entries.push(("comparison".to_owned(), comparison));
        }
    }
    validate_finite(&report)?;
    // Rendering also validates report field types before any output is published.
    let summary = render_markdown(&report, request_top(args))?;
    ci_metrics_transport::check(cancellation)?;
    if let Some(path) = truthy(args.raw_out.as_ref()) {
        let raw = object([
            ("schema_version", Value::Int(1)),
            ("runs", Value::Array(raw_runs)),
        ]);
        write(out, path, &(dumps(&raw, false) + "\n"))?;
    }
    if let Some(path) = truthy(args.json_out.as_ref()) {
        write(out, path, &(dumps(&report, true) + "\n"))?;
    }
    let markdown_out = truthy(args.markdown_out.as_ref());
    if let Some(path) = markdown_out {
        write(out, path, &summary)?;
    }
    if truthy(args.json_out.as_ref()).is_none() && markdown_out.is_none() {
        out.stdout.push_str(&summary);
    }
    Ok(())
}

fn request_top(args: &Args) -> usize {
    usize::try_from(args.top).unwrap_or(usize::MAX)
}

fn os_error(error: &std::io::Error, shown: &str) -> Failure {
    Failure::Reported(io_text(error, Path::new(shown)))
}

/// `open(path, encoding="utf-8")` or `sys.stdin`, read whole.
fn read_input(path: &str, cancellation: &Cancellation) -> Outcome<Vec<u8>> {
    let mut bytes = Vec::new();
    if path == "-" {
        std::io::stdin()
            .lock()
            .read_to_end(&mut bytes)
            .map_err(|error| os_error(&error, "<stdin>"))?;
        return Ok(bytes);
    }
    ci_metrics_transport::read_file(Path::new(path), cancellation)
}

/// `write(path, content)`: `-` is stdout; otherwise create the parents
/// (`Path.mkdir(parents=True, exist_ok=True)`) and write UTF-8 text.
fn write(out: &mut CheckReport, path: &str, content: &str) -> Outcome<()> {
    if path == "-" {
        out.stdout.push_str(content);
        return Ok(());
    }
    let shown = python_path_display(Path::new(path));
    if let Some(parent) = parent_of(&shown) {
        make_dirs(&parent)?;
    }
    std::fs::write(&shown, content).map_err(|error| os_error(&error, &shown))
}

/// `PurePath.parent` of a normalized path string; `None` at a root.
fn parent_of(shown: &str) -> Option<String> {
    match shown.rsplit_once('/') {
        Some(("", _)) if shown != "/" => Some("/".to_owned()),
        Some((parent, _)) if !parent.is_empty() => Some(parent.to_owned()),
        _ if shown != "." && shown != "/" => Some(".".to_owned()),
        _ => None,
    }
}

fn make_dirs(path: &str) -> Outcome<()> {
    match std::fs::create_dir(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            if let Some(parent) = parent_of(path).filter(|parent| parent != path) {
                make_dirs(&parent)?;
                return make_dirs_once(path);
            }
            Err(os_error(&error, path))
        }
        Err(error) if Path::new(path).is_dir() => {
            let _exists = error;
            Ok(())
        }
        Err(error) => Err(os_error(&error, path)),
    }
}

fn make_dirs_once(path: &str) -> Outcome<()> {
    match std::fs::create_dir(path) {
        Err(error) if !Path::new(path).is_dir() => Err(os_error(&error, path)),
        _ => Ok(()),
    }
}
