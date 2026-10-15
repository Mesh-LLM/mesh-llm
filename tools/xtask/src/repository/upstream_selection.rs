//! repository llama-upstream-selection: choose the llama.cpp upstream commit
//! the canary should pin.
//!
//! The canary used to pin the tip of the upstream default branch. When that tip
//! is red in its own upstream CI, or still running, the run spends a whole
//! repair and certification cycle on a change upstream has already rejected.
//! This owner walks the branch history from its newest commit downwards and
//! returns the newest commit whose own GitHub check runs all completed without
//! a failure: the most recent upstream change that passes CI.
//!
//! The current pin is the floor. The walk stops at the pin, so the command can
//! only move it forward, and a window with no newer passing commit keeps the pin
//! unchanged instead of failing the run. An explicit request is admitted exactly
//! like the branch tip: a requested commit that does not pass CI is replaced by
//! the newest passing commit at or below it, and the substitution is reported.

mod gh;

use crate::automation::command_interrupt::Interrupt;
use crate::command::DynResult;
use crate::repository::check_args::{Grammar, ParsedArgs};
use gh::{Gh, GhTransaction};
use serde::Deserialize;
use std::{collections::BTreeMap, fs::OpenOptions, io::Write, path::Path, time::Duration};

const GRAMMAR: Grammar = Grammar {
    usage: "repository llama-upstream-selection --pin SHA [--requested SHA] [--upstream-repository OWNER/NAME] [--branch NAME] [--max-history N] [--max-candidates N] [--retry-delay-ms N] [--github-output PATH --summary PATH] [--timeout-secs SECONDS]",
    values: &[
        "--pin",
        "--requested",
        "--upstream-repository",
        "--branch",
        "--max-history",
        "--max-candidates",
        "--retry-delay-ms",
        "--github-output",
        "--summary",
        "--timeout-secs",
    ],
    flags: &[],
};

/// The upstream repository whose branch history is walked.
const DEFAULT_UPSTREAM_REPOSITORY: &str = "ggml-org/llama.cpp";
const DEFAULT_BRANCH: &str = "master";
/// The pin and an explicit request must be findable this far back for their
/// boundary to be provable; the candidate walk is bounded separately.
const DEFAULT_MAX_HISTORY: usize = 300;
const MAX_HISTORY_CEILING: usize = 1000;
/// Every examined candidate costs at least one GitHub API request.
const DEFAULT_MAX_CANDIDATES: usize = 60;
const MAX_CANDIDATES_CEILING: usize = 200;
const DEFAULT_RETRY_DELAY_MS: u64 = 2_000;
const MAX_RETRY_DELAY_MS: u64 = 30_000;
const API_ATTEMPTS: u32 = 3;
const PAGE_SIZE: usize = 100;
const MAX_CHECK_RUN_PAGES: usize = 5;
const DEFAULT_TIMEOUT_SECONDS: u64 = 900;
const MIN_TIMEOUT_SECONDS: u64 = 30;
const MAX_TIMEOUT_SECONDS: u64 = 3600;
/// Evidence table rows; the complete walk still reaches the summary as counts.
const MAX_EVIDENCE_ROWS: usize = 20;

/// Only these conclusions are evidence that a commit passed its own CI:
/// skipped and neutral are ordinary for path-filtered lanes.
const PASSING_CONCLUSIONS: [&str; 3] = ["success", "skipped", "neutral"];

/// A commit below the branch tip passed; the requested commit's own verdict is
/// reported separately when it did not.
const OUTCOME_WALK: &str = "walk";
/// The requested upstream commit itself passed.
const OUTCOME_REQUESTED: &str = "requested";
/// Nothing newer than the pin passed, so the pin is kept.
const OUTCOME_PIN: &str = "pin";

struct Request {
    upstream_repository: String,
    branch: String,
    pin: String,
    requested: Option<String>,
    max_history: usize,
    max_candidates: usize,
    retry_delay: Duration,
}

impl Request {
    fn commits_path(&self) -> String {
        format!("repos/{}/commits", self.upstream_repository)
    }

    fn check_runs_path(&self, sha: &str) -> String {
        format!("repos/{}/commits/{sha}/check-runs", self.upstream_repository)
    }
}

/// One branch commit, newest first.
struct CommitRow {
    sha: String,
    date: String,
    subject: String,
}

/// One examined candidate and why it was skipped, or that it passed.
#[derive(Debug)]
struct Candidate {
    sha: String,
    date: String,
    subject: String,
    checks: usize,
    pending: usize,
    failing: Vec<String>,
}

impl Candidate {
    fn passing(&self) -> bool {
        self.checks > 0 && self.pending == 0 && self.failing.is_empty()
    }
}

#[derive(Debug)]
struct Selection {
    sha: String,
    outcome: &'static str,
    requested: Option<String>,
    examined: usize,
    candidates: Vec<Candidate>,
}

#[derive(Deserialize)]
struct CommitBody {
    message: Option<String>,
    committer: Option<CommitIdentity>,
}

#[derive(Deserialize)]
struct CommitIdentity {
    date: Option<String>,
}

#[derive(Deserialize)]
struct CommitEntry {
    sha: String,
    commit: CommitBody,
}

#[derive(Deserialize)]
struct CheckRunsPage {
    total_count: Option<u64>,
    check_runs: Vec<CheckRun>,
}

#[derive(Deserialize, Clone)]
struct CheckRun {
    id: u64,
    name: String,
    status: String,
    conclusion: Option<String>,
    started_at: Option<String>,
    app: Option<CheckRunApp>,
}

#[derive(Deserialize, Clone)]
struct CheckRunApp {
    id: Option<u64>,
    slug: Option<String>,
}

pub(crate) fn run(args: &[String]) -> DynResult<()> {
    let parsed = match GRAMMAR.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report.emit(),
    };
    if !parsed.positionals.is_empty() {
        return Err("llama-upstream-selection accepts named inputs only".into());
    }
    let request = request(&parsed)?;
    let timeout = seconds(&parsed)?;
    let interrupt = Interrupt::install()?;
    let transaction = GhTransaction::new(Duration::from_secs(timeout), interrupt.cancellation())?;
    let result = select(&transaction, &request);
    interrupt.finish()?;
    let selection = result?;
    if !sha(&selection.sha) {
        return Err(format!(
            "selected upstream commit is not a lowercase SHA: {}",
            selection.sha
        )
        .into());
    }
    publish(
        &selection,
        &request,
        parsed.last("--github-output"),
        parsed.last("--summary"),
    )?;
    println!("{}", selection.sha);
    Ok(())
}

fn request(parsed: &ParsedArgs) -> Result<Request, String> {
    let pin = parsed.last("--pin").ok_or("the current pin is required")?;
    if !sha(pin) {
        return Err(format!("--pin must be a lowercase 40-character SHA: {pin}"));
    }
    let requested = parsed
        .last("--requested")
        .filter(|value| !value.is_empty() && *value != "latest")
        .map(str::to_owned);
    if let Some(requested) = requested.as_deref() {
        if !sha(requested) {
            return Err(format!(
                "--requested must be a lowercase 40-character SHA: {requested}"
            ));
        }
    }
    let upstream_repository = parsed
        .last("--upstream-repository")
        .unwrap_or(DEFAULT_UPSTREAM_REPOSITORY);
    if !repository(upstream_repository) {
        return Err(format!(
            "--upstream-repository must be OWNER/NAME: {upstream_repository}"
        ));
    }
    let branch = parsed.last("--branch").unwrap_or(DEFAULT_BRANCH);
    if branch.is_empty() || branch.starts_with('-') || branch.chars().any(char::is_whitespace) {
        return Err(format!("--branch must be a branch name: {branch}"));
    }
    Ok(Request {
        upstream_repository: upstream_repository.to_owned(),
        branch: branch.to_owned(),
        pin: pin.to_owned(),
        requested,
        max_history: bounded(
            parsed,
            "--max-history",
            DEFAULT_MAX_HISTORY,
            MAX_HISTORY_CEILING,
        )?,
        max_candidates: bounded(
            parsed,
            "--max-candidates",
            DEFAULT_MAX_CANDIDATES,
            MAX_CANDIDATES_CEILING,
        )?,
        retry_delay: Duration::from_millis(
            parsed
                .last("--retry-delay-ms")
                .map(|value| value.parse::<u64>())
                .transpose()
                .map_err(|_| "--retry-delay-ms must be a non-negative integer".to_owned())?
                .unwrap_or(DEFAULT_RETRY_DELAY_MS)
                .min(MAX_RETRY_DELAY_MS),
        ),
    })
}

fn bounded(
    parsed: &ParsedArgs,
    name: &str,
    default: usize,
    ceiling: usize,
) -> Result<usize, String> {
    match parsed.last(name) {
        Some(value) => match value.parse::<usize>() {
            Ok(value) if (1..=ceiling).contains(&value) => Ok(value),
            _ => Err(format!("{name} must be 1..{ceiling}")),
        },
        None => Ok(default),
    }
}

fn seconds(parsed: &ParsedArgs) -> Result<u64, String> {
    match parsed.last("--timeout-secs") {
        Some(value) => match value.parse::<u64>() {
            Ok(value) if (MIN_TIMEOUT_SECONDS..=MAX_TIMEOUT_SECONDS).contains(&value) => Ok(value),
            _ => Err(format!(
                "--timeout-secs must be {MIN_TIMEOUT_SECONDS}..{MAX_TIMEOUT_SECONDS}"
            )),
        },
        None => Ok(DEFAULT_TIMEOUT_SECONDS),
    }
}

fn sha(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn repository(value: &str) -> bool {
    let Some((owner, name)) = value.split_once('/') else {
        return false;
    };
    !owner.is_empty()
        && !name.is_empty()
        && !name.contains('/')
        && value.chars().all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.' | '/')
        })
}

fn select(gh: &dyn Gh, request: &Request) -> Result<Selection, String> {
    let history = branch_history(gh, request)?;
    let pin_index = history.iter().position(|row| row.sha == request.pin);
    let start = match request.requested.as_deref() {
        Some(requested) => history
            .iter()
            .position(|row| row.sha == requested)
            .ok_or(format!(
                "requested upstream commit {requested} is not within the newest {} commits of {}/{}",
                request.max_history, request.upstream_repository, request.branch
            ))?,
        None => 0,
    };
    // Only an explicit request can be below the pin: the branch tip is always
    // at or above it, and the window below already stops at the pin.
    if request.requested.is_some() {
        if let Some(pin_index) = pin_index {
            if pin_index <= start {
                return Err(format!(
                    "requested upstream commit is at or below the current pin {}",
                    request.pin
                ));
            }
        }
    }
    let end = pin_index.unwrap_or(history.len()).min(history.len());
    let window = if end > start { &history[start..end] } else { &[] };
    let mut candidates = Vec::new();
    for row in window.iter().take(request.max_candidates) {
        let candidate = assess(gh, request, row)?;
        let passed = candidate.passing();
        candidates.push(candidate);
        if passed {
            return Ok(Selection {
                sha: row.sha.clone(),
                // The requested commit passing is a distinct outcome from a
                // substitution below it; both are still a walk result.
                outcome: match request.requested.as_deref() {
                    Some(requested) if requested == row.sha => OUTCOME_REQUESTED,
                    _ => OUTCOME_WALK,
                },
                requested: request.requested.clone(),
                examined: candidates.len(),
                candidates,
            });
        }
    }
    let examined = candidates.len();
    Ok(Selection {
        sha: request.pin.clone(),
        outcome: OUTCOME_PIN,
        requested: request.requested.clone(),
        examined,
        candidates,
    })
}

fn branch_history(gh: &dyn Gh, request: &Request) -> Result<Vec<CommitRow>, String> {
    let mut rows: Vec<CommitRow> = Vec::new();
    let pages = request.max_history.div_ceil(PAGE_SIZE).max(1);
    for page in 1..=pages {
        let body = api(
            gh,
            request,
            &request.commits_path(),
            &[
                ("sha", request.branch.clone()),
                ("per_page", PAGE_SIZE.to_string()),
                ("page", page.to_string()),
            ],
        )?;
        let entries: Vec<CommitEntry> = serde_json::from_str(&body).map_err(|error| {
            format!(
                "invalid commit list for {}/{}: {error}",
                request.upstream_repository, request.branch
            )
        })?;
        let short = entries.len() < PAGE_SIZE;
        for entry in entries {
            if !sha(&entry.sha) {
                return Err(format!(
                    "commit list for {}/{} carries an invalid SHA: {}",
                    request.upstream_repository, request.branch, entry.sha
                ));
            }
            rows.push(CommitRow {
                sha: entry.sha,
                date: entry
                    .commit
                    .committer
                    .and_then(|committer| committer.date)
                    .unwrap_or_default(),
                subject: subject(entry.commit.message.as_deref().unwrap_or_default()),
            });
        }
        if short || rows.len() >= request.max_history {
            break;
        }
    }
    rows.truncate(request.max_history);
    if rows.is_empty() {
        return Err(format!(
            "no commits reported for {}/{}",
            request.upstream_repository, request.branch
        ));
    }
    Ok(rows)
}

fn assess(gh: &dyn Gh, request: &Request, row: &CommitRow) -> Result<Candidate, String> {
    let runs = check_runs(gh, request, &row.sha)?;
    let mut latest: BTreeMap<(String, String), CheckRun> = BTreeMap::new();
    for run in runs {
        let app = run
            .app
            .as_ref()
            .map(|app| {
                app.slug
                    .clone()
                    .unwrap_or_else(|| app.id.map(|id| id.to_string()).unwrap_or_default())
            })
            .unwrap_or_default();
        let key = (app, run.name.clone());
        match latest.get(&key) {
            Some(existing) if !newer(&run, existing) => {}
            _ => {
                latest.insert(key, run);
            }
        }
    }
    let mut pending = 0;
    let mut failing = Vec::new();
    for run in latest.values() {
        if run.status != "completed" {
            pending += 1;
            continue;
        }
        match run.conclusion.as_deref() {
            Some(conclusion) if PASSING_CONCLUSIONS.contains(&conclusion) => {}
            Some(conclusion) => failing.push(format!("{} ({conclusion})", run.name)),
            None => failing.push(format!("{} (no conclusion)", run.name)),
        }
    }
    failing.sort();
    Ok(Candidate {
        sha: row.sha.clone(),
        date: row.date.clone(),
        subject: row.subject.clone(),
        checks: latest.len(),
        pending,
        failing,
    })
}

fn check_runs(gh: &dyn Gh, request: &Request, sha: &str) -> Result<Vec<CheckRun>, String> {
    let mut runs = Vec::new();
    for page in 1..=MAX_CHECK_RUN_PAGES {
        let body = api(
            gh,
            request,
            &request.check_runs_path(sha),
            &[
                ("filter", "all".to_owned()),
                ("per_page", PAGE_SIZE.to_string()),
                ("page", page.to_string()),
            ],
        )?;
        let parsed: CheckRunsPage = serde_json::from_str(&body)
            .map_err(|error| format!("invalid check runs for {sha}: {error}"))?;
        let total = parsed.total_count.unwrap_or(0) as usize;
        let short = parsed.check_runs.len() < PAGE_SIZE;
        runs.extend(parsed.check_runs);
        if short || runs.len() >= total {
            break;
        }
    }
    Ok(runs)
}

/// The most recent run for one app and check name decides, exactly as GitHub
/// reports a re-run.
fn newer(candidate: &CheckRun, existing: &CheckRun) -> bool {
    match (&candidate.started_at, &existing.started_at) {
        (Some(candidate), Some(existing)) if candidate != existing => candidate > existing,
        (Some(_), None) => true,
        (None, Some(_)) => false,
        _ => candidate.id > existing.id,
    }
}

fn api(
    gh: &dyn Gh,
    request: &Request,
    path: &str,
    query: &[(&str, String)],
) -> Result<String, String> {
    let mut attempt = 1;
    loop {
        match gh.api(path, query) {
            Ok(body) => return Ok(body),
            Err(_) if attempt < API_ATTEMPTS => {
                if !request.retry_delay.is_zero() {
                    std::thread::sleep(request.retry_delay * attempt);
                }
                attempt += 1;
            }
            Err(error) => return Err(format!("{error} (after {attempt} attempts)")),
        }
    }
}

fn subject(message: &str) -> String {
    let line = message.lines().next().unwrap_or_default().trim();
    line.chars().take(80).collect()
}

fn append(path: &str, bytes: &[u8]) -> DynResult<()> {
    let path = Path::new(path);
    if std::fs::symlink_metadata(path).is_ok_and(|meta| !meta.file_type().is_file()) {
        return Err("GitHub evidence destination must be a regular file".into());
    }
    let mut options = OpenOptions::new();
    options.create(true).append(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let mut output = options.open(path)?;
    if !output.metadata()?.is_file() {
        return Err("GitHub evidence destination must be regular".into());
    }
    output.write_all(bytes)?;
    Ok(())
}

fn publish(
    selection: &Selection,
    request: &Request,
    output: Option<&str>,
    summary_path: Option<&str>,
) -> DynResult<()> {
    match (output, summary_path) {
        (Some(output), Some(summary_path)) => {
            let values = format!(
                "upstream_selection={}\nupstream_examined={}\n",
                selection.outcome, selection.examined
            );
            append(output, values.as_bytes())?;
            append(summary_path, summary(selection, request).as_bytes())?;
        }
        (None, None) => (),
        _ => return Err("GitHub output and summary must be provided together".into()),
    }
    Ok(())
}

fn summary(selection: &Selection, request: &Request) -> String {
    let mut summary = String::from("### llama.cpp upstream selection\n\n");
    summary.push_str(&format!(
        "- upstream: {}/{} (newest first)\n- current pin: {}\n",
        sanitize(&request.upstream_repository),
        sanitize(&request.branch),
        selection.sha_text(request)
    ));
    match selection.outcome {
        OUTCOME_PIN => summary.push_str(&format!(
            "- selected: {} (unchanged: no newer commit passes its own CI)\n",
            request.pin
        )),
        OUTCOME_REQUESTED => summary.push_str(&format!(
            "- selected: {} (the requested upstream commit)\n",
            selection.sha
        )),
        _ => summary.push_str(&format!(
            "- selected: {} ({})\n",
            selection.sha,
            selection
                .candidates
                .last()
                .map(|candidate| format!("{} {}", candidate.date, candidate.subject))
                .unwrap_or_default()
        )),
    }
    if let Some(requested) = selection.requested.as_deref() {
        if selection.outcome != OUTCOME_REQUESTED {
            summary.push_str(&format!(
                "- requested: {requested} did not pass its own CI; the newest passing commit at or below it was selected\n"
            ));
        }
    }
    let pending = selection
        .candidates
        .iter()
        .filter(|candidate| candidate.pending > 0)
        .count();
    let failing = selection
        .candidates
        .iter()
        .filter(|candidate| !candidate.failing.is_empty())
        .count();
    let unchecked = selection
        .candidates
        .iter()
        .filter(|candidate| candidate.checks == 0)
        .count();
    summary.push_str(&format!(
        "- walked: {} of at most {} commits ({} with failing checks, {} still running, {} reporting no checks)\n",
        selection.examined, request.max_candidates, failing, pending, unchecked
    ));
    summary.push_str(
        "\nA commit passes only when every check run it reports is completed and concluded success, skipped or neutral; failures, cancellations, timeouts and still-running checks all reject it.\n\n",
    );
    if selection.candidates.is_empty() {
        summary.push_str(
            "No candidate was examined: the current pin is already the newest commit.\n",
        );
        return summary;
    }
    summary.push_str("| commit | committed | checks | running | failing | subject |\n");
    summary.push_str("| --- | --- | --- | --- | --- | --- |\n");
    for candidate in selection.candidates.iter().take(MAX_EVIDENCE_ROWS) {
        summary.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} |\n",
            candidate.sha,
            sanitize(&candidate.date),
            candidate.checks,
            candidate.pending,
            if candidate.failing.is_empty() {
                "-".to_owned()
            } else {
                sanitize(&candidate.failing.join(", "))
            },
            sanitize(&candidate.subject)
        ));
    }
    if selection.candidates.len() > MAX_EVIDENCE_ROWS {
        summary.push_str(&format!(
            "\n{} further commits were examined and skipped.\n",
            selection.candidates.len() - MAX_EVIDENCE_ROWS
        ));
    }
    summary
}

impl Selection {
    /// The pin value, shown as the selected commit only when it was kept.
    fn sha_text(&self, request: &Request) -> String {
        match self.outcome {
            OUTCOME_PIN => format!("{} (kept)", request.pin),
            _ => self.sha.clone(),
        }
    }
}

/// Summary cells are one line and never carry table delimiters.
fn sanitize(value: &str) -> String {
    value
        .chars()
        .map(|character| match character {
            '\n' | '\r' | '|' => ' ',
            _ => character,
        })
        .collect::<String>()
        .trim()
        .to_owned()
}

#[cfg(test)]
#[path = "upstream_selection/tests.rs"]
mod tests;
