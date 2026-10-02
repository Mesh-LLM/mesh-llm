//! `release notes-classify`: `scripts/release-notes-classify.py`. Reads the
//! release body's pull requests, the range's squash-merge commits (through
//! [`ReleaseHost`]) and optional `--links` records, and writes the
//! deterministic Keep a Changelog plan.

use crate::ci_operations::ci_metrics_value::{Value, dumps, parse as parse_json};
use crate::prepared_input::text_io::decode_utf8;
use crate::release::classify_argv::{Invocation, PlanArgs, parse};
use crate::release::classify_rules::build_plan;
use crate::release::command_failure::{Uncaught, argv_repr, called_process_error};
use crate::release::link_body::{canonical, entry_pr};
use crate::release::link_commits::{pr_suffix, trailers};
use crate::release::link_host::{Exit, ReleaseHost, text_streams};
use crate::repository::check_report::CheckReport;
use crate::repository::text::strip;
use std::collections::HashMap;
use std::path::Path;

pub(crate) fn run(args: &[String], host: &mut dyn ReleaseHost) -> CheckReport {
    let outcome = match parse(args) {
        Err(report) => return report,
        Ok(Invocation::HasEntries(body)) => read_body_prs(&body).map(|prs| CheckReport {
            code: i32::from(prs.is_empty()),
            ..CheckReport::default()
        }),
        Ok(Invocation::Plan(plan)) => classify(&plan, host),
    };
    outcome.unwrap_or_else(|uncaught| uncaught.report(String::new()))
}

fn classify(args: &PlanArgs, host: &mut dyn ReleaseHost) -> Result<CheckReport, Uncaught> {
    let prs = read_body_prs(&args.body)?;
    if prs.is_empty() {
        return Ok(CheckReport::failure(
            String::new(),
            "error: no PR entries found in the release body\n".to_owned(),
        ));
    }
    let mut commits = read_commits(host, args)?;
    if let Some(links) = &args.links {
        for (pr, record) in load_links(links)? {
            commits.entry(pr).or_insert(record);
        }
    }
    let (plan, unclassified) = build_plan(&prs, &commits, &args.version, &args.date)?;
    let text = format!("{}\n", dumps(&plan, false));
    std::fs::write(&args.out, text).map_err(|error| Uncaught::os(Path::new(&args.out), &error))?;
    let total = prs.len();
    Ok(CheckReport::success(format!(
        "classified {}/{total} entries deterministically ({unclassified} in 'Other changes')\n",
        total - unclassified
    )))
}

fn read_text(path: &str) -> Result<String, Uncaught> {
    let bytes = std::fs::read(path).map_err(|error| Uncaught::os(Path::new(path), &error))?;
    let text = decode_utf8(bytes).map_err(Uncaught::decode)?;
    Ok(text.replace("\r\n", "\n").replace('\r', "\n"))
}

/// `read_body_prs(path)`: text-mode lines up to the release tail.
fn read_body_prs(path: &str) -> Result<Vec<String>, Uncaught> {
    let text = read_text(path)?;
    let mut prs = Vec::new();
    for line in text.split_inclusive('\n') {
        if line.starts_with("## New Contributors") || line.starts_with("**Full Changelog**") {
            break;
        }
        if let Some(pr) = entry_pr(line.trim_end_matches('\n')) {
            prs.push(pr);
        }
    }
    Ok(prs)
}

/// `git log` argv after the program name.
fn log_args(range: &str) -> Vec<String> {
    ["log", "--format=%s%x1f%b%x1e", range]
        .map(str::to_owned)
        .to_vec()
}

/// `read_commits(git_range, repo_root)`: PR number -> commit record.
fn read_commits(
    host: &mut dyn ReleaseHost,
    args: &PlanArgs,
) -> Result<HashMap<String, Value>, Uncaught> {
    let git_args = log_args(&args.range);
    let cwd = args.repo_root.as_deref().map(Path::new);
    let output = host
        .git(&git_args, cwd)
        .map_err(|failure| Uncaught::os(&failure.filename, &failure.error))?;
    let failure = |code, signal| {
        let argv = argv_repr("git", &git_args);
        Uncaught::new(
            "subprocess.CalledProcessError",
            called_process_error(&argv, code, signal),
        )
    };
    let (stdout, _) = text_streams(&output).map_err(Uncaught::decode)?;
    match output.exit {
        Exit::Code(0) | Exit::TimedOut => Ok(parse_log(&stdout)),
        Exit::Code(code) => Err(failure(Some(code), None)),
        Exit::Signal(signal) => Err(failure(None, Some(signal))),
    }
}

fn parse_log(stdout: &str) -> HashMap<String, Value> {
    let mut commits = HashMap::new();
    for record in stdout.split('\u{1e}') {
        if strip(record).is_empty() {
            continue;
        }
        let record = record.trim_matches('\n');
        let (subject, body) = record.split_once('\u{1f}').unwrap_or((record, ""));
        let subject = strip(subject);
        // `PR_SUFFIX_RE.search`: only the digits of a trailing `(#N)` matter.
        let Some((_, digits)) = pr_suffix(subject) else {
            continue;
        };
        let trailers = trailers(body)
            .into_iter()
            .map(|(key, value)| (key, Value::Str(value)))
            .collect();
        let record = Value::Object(vec![
            ("subject".to_owned(), Value::text(subject)),
            ("trailers".to_owned(), Value::Object(trailers)),
        ]);
        commits.insert(canonical(digits), record);
    }
    commits
}

/// `load_links(path)`: `{int(pr): record}` in file order.
fn load_links(path: &str) -> Result<Vec<(String, Value)>, Uncaught> {
    let bytes = std::fs::read(path).map_err(|error| Uncaught::os(Path::new(path), &error))?;
    let value = parse_json(&bytes).map_err(|message| Uncaught::new("links", message))?;
    let Value::Object(entries) = value else {
        return Err(Uncaught::new(
            "links",
            "release links must be an object keyed by pull-request numbers".into(),
        ));
    };
    let mut links: Vec<(String, Value)> = Vec::new();
    for (pr, record) in entries {
        if pr.is_empty()
            || !pr.bytes().all(|byte| byte.is_ascii_digit())
            || pr.bytes().all(|byte| byte == b'0')
        {
            return Err(Uncaught::new(
                "links",
                "release link key must be a positive decimal pull-request number".into(),
            ));
        }
        let key = canonical(&pr);
        match links.iter_mut().find(|(seen, _)| *seen == key) {
            Some(entry) => entry.1 = record,
            None => links.push((key, record)),
        }
    }
    Ok(links)
}
