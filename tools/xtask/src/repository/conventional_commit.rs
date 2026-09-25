//! `repository conventional-commits`: the Rust owner of
//! `scripts/check-conventional-commit.py`. Validates one message file, one
//! `--message`, or every non-merge commit in a `--range`.

mod subject;
mod trailers;

use crate::command::DynResult;
use crate::repository::check_args::{Grammar, ParsedArgs};
use crate::repository::check_report::CheckReport;
use crate::repository::python_text::{splitlines, strip};
use std::path::Path;
use std::process::Command;

const GRAMMAR: Grammar = Grammar {
    usage: "cargo xtool repository conventional-commits [--message MESSAGE] [--range RANGE] [--trailers-only] [file]",
    values: &["--message", "--range"],
    flags: &["--trailers-only"],
};

const GUIDANCE: &str = "\nConventional Commits: https://www.conventionalcommits.org/en/v1.0.0/\n\
The type decides which release-notes section the change lands in.\n\
Add a 'Release-Notes: <Section>' trailer to override, or\n\
'BREAKING CHANGE: <what>' for a breaking change.\n\
\nCatch this at commit time instead of in CI:  just hooks-install\n\
Agent, bot, and relay attribution trailers are not kept in this\n\
history. GitHub re-adds them when squashing a PR whose commits carry\n\
them, so remove them from the branch commits.\n\
Bypass once with --no-verify if you know the commit is not a release entry.\n";

enum Source<'a> {
    File(&'a str),
    Message(&'a str),
    Range(&'a str),
}

pub(crate) fn run(cwd: &Path, args: &[String]) -> DynResult<()> {
    let report = match GRAMMAR.parse(args).and_then(|parsed| check(cwd, &parsed)) {
        Ok(report) | Err(report) => report,
    };
    report.emit()
}

fn source(parsed: &ParsedArgs) -> Result<Source<'_>, CheckReport> {
    if parsed.positionals.len() > 1 {
        let extra = parsed.positionals[1..].join(" ");
        return Err(GRAMMAR.error(&format!("unrecognized arguments: {extra}")));
    }
    let chosen = [
        parsed.positionals.first().map(|file| Source::File(file)),
        parsed.last("--message").map(Source::Message),
        parsed.last("--range").map(Source::Range),
    ];
    let mut chosen = chosen.into_iter().flatten();
    match (chosen.next(), chosen.next()) {
        (Some(source), None) => Ok(source),
        (None, _) => Err(GRAMMAR.error("one of the arguments file --message --range is required")),
        (Some(_), Some(_)) => {
            Err(GRAMMAR.error("only one of file, --message and --range is allowed"))
        }
    }
}

fn check(cwd: &Path, parsed: &ParsedArgs) -> Result<CheckReport, CheckReport> {
    let trailers_only = parsed.flag("--trailers-only");
    let lines = match source(parsed)? {
        Source::Range(range) => return check_range(cwd, range, trailers_only),
        Source::Message(message) if strip(message).is_empty() => Vec::new(),
        Source::Message(message) => splitlines(message).into_iter().map(str::to_owned).collect(),
        Source::File(file) => read_message_file(&cwd.join(file))?,
    };
    let stderr = review(&lines, trailers_only);
    Ok(finish(stderr))
}

fn read_message_file(path: &Path) -> Result<Vec<String>, CheckReport> {
    let text = std::fs::read_to_string(path).map_err(|error| {
        CheckReport::failure(
            String::new(),
            format!(
                "error: cannot read commit message {}: {error}\n",
                path.display()
            ),
        )
    })?;
    Ok(splitlines(&text)
        .into_iter()
        .filter(|line| !line.starts_with('#'))
        .map(str::to_owned)
        .collect())
}

/// The rejection report for one message, or an empty string when accepted.
fn review(lines: &[String], trailers_only: bool) -> String {
    let subject = lines.first().map_or("", String::as_str);
    let body = lines.iter().skip(1).map(String::as_str);
    let problems = if trailers_only {
        trailers::check_trailers(lines.iter().map(String::as_str))
    } else {
        let mut problems = subject::check_subject(subject);
        problems.extend(trailers::check_trailers(body));
        problems
    };
    if problems.is_empty() {
        return String::new();
    }
    let mut report = format!("commit message rejected: {subject}\n");
    for problem in problems {
        report.push_str(&format!("  {problem}\n"));
    }
    report + GUIDANCE
}

fn finish(stderr: String) -> CheckReport {
    if stderr.is_empty() {
        CheckReport::success(String::new())
    } else {
        CheckReport::failure(String::new(), stderr)
    }
}

/// Every record from `git log --no-merges`, newest first; blank lines are
/// dropped before the first line is taken as the subject.
fn check_range(cwd: &Path, range: &str, trailers_only: bool) -> Result<CheckReport, CheckReport> {
    let output = Command::new("git")
        .current_dir(cwd)
        .args(["log", "--format=%B%x1e", "--no-merges", range])
        .output()
        .map_err(|error| {
            CheckReport::failure(
                String::new(),
                format!("error: cannot run git log: {error}\n"),
            )
        })?;
    if !output.status.success() {
        return Err(CheckReport::failure(
            String::new(),
            format!(
                "error: git log --format=%B%x1e --no-merges {range} failed: {}\n",
                String::from_utf8_lossy(&output.stderr).trim_end()
            ),
        ));
    }
    let log = String::from_utf8_lossy(&output.stdout);
    let stderr = log
        .split('\u{1e}')
        .map(|record| {
            splitlines(record.trim_matches('\n'))
                .into_iter()
                .filter(|line| !strip(line).is_empty())
                .map(str::to_owned)
                .collect::<Vec<_>>()
        })
        .filter(|lines| !lines.is_empty())
        .map(|lines| review(&lines, trailers_only))
        .collect::<String>();
    Ok(finish(stderr))
}
