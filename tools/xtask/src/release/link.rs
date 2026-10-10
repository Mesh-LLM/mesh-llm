//! `release notes-link`: `scripts/release-notes-link.py`. Pairs each commit
//! of a range with its pull request, credits pull requests GitHub's body
//! left out, and writes the augmented body plus the API-linked records.

use crate::ci_operations::ci_metrics_value::Value;
use crate::prepared_input::text_io::decode_utf8;
use crate::release::command_failure::{Uncaught, argv_repr, called_process_error};
use crate::release::link_argv::{Args, parse};
use crate::release::link_body::{Body, augment, canonical, dump_links, entry_line, read_body};
use crate::release::link_commits::{
    Commit, commit_links, log_args, parse_log, pr_suffix, without_suffix,
};
use crate::release::link_gh::Gh;
use crate::release::link_host::{Exit, ReleaseHost, text_streams};
use crate::repository::check_report::CheckReport;
use std::path::Path;

pub(crate) fn run(args: &[String], host: &mut dyn ReleaseHost) -> CheckReport {
    let args = match parse(args) {
        Ok(args) => args,
        Err(report) => return report,
    };
    match link(&args, host) {
        Ok(report) => report,
        Err((stderr, uncaught)) => uncaught.report(stderr),
    }
}

type Failed = (String, Uncaught);

fn link(args: &Args, host: &mut dyn ReleaseHost) -> Result<CheckReport, Failed> {
    let fail = |uncaught| (String::new(), uncaught);
    let body = read_body(&read_text(Path::new(&args.body)).map_err(fail)?);
    let mut commits = read_commits(host, args).map_err(fail)?;
    let mut gh = Gh::new(host, args.api_budget);
    let outcome = recover(&mut commits, &body, &args.repo, &mut gh);
    let (insertions, trailing) = match outcome {
        Ok(recovered) => recovered,
        Err(uncaught) => return Err((gh.stderr, uncaught)),
    };
    let linked = commits.iter().filter(|commit| commit.linked_by_api).count();
    let recovered = insertions
        .iter()
        .map(|(_, lines)| lines.len())
        .sum::<usize>()
        + trailing.len();
    let written = write(&args.out_body, &augment(&body, &insertions, &trailing))
        .and_then(|()| write(&args.out_links, &dump_links(&commit_links(&commits))));
    let mut stderr = std::mem::take(&mut gh.stderr);
    if let Err(uncaught) = written {
        return Err((stderr, uncaught));
    }
    let stdout = format!(
        "linked {linked} commit(s) to a pull request through the API; \
         recovered {recovered} entry(ies) GitHub did not credit ({} published)\n",
        body.credited.len()
    );
    if gh.failures > 0 {
        stderr.push_str(&format!(
            "release-notes-link: {} API call(s) failed\n",
            gh.failures
        ));
    }
    Ok(CheckReport {
        stdout,
        stderr,
        code: 0,
    })
}

type Recovered = (Vec<(String, Vec<String>)>, Vec<String>);

fn recover(
    commits: &mut [Commit],
    body: &Body,
    repo: &str,
    gh: &mut Gh<'_>,
) -> Result<Recovered, Uncaught> {
    resolve_pull_requests(commits, repo, gh)?;
    recover_entries(commits, &body.credited, repo, gh)
}

fn read_text(path: &Path) -> Result<String, Uncaught> {
    let bytes = std::fs::read(path).map_err(|error| Uncaught::os(path, &error))?;
    decode_utf8(bytes).map_err(Uncaught::decode)
}

fn write(path: &str, text: &str) -> Result<(), Uncaught> {
    std::fs::write(path, text).map_err(|error| Uncaught::os(Path::new(path), &error))
}

/// `read_commits(git_range, repo_root)` with `check=True`.
fn read_commits(host: &mut dyn ReleaseHost, args: &Args) -> Result<Vec<Commit>, Uncaught> {
    let git_args = log_args(&args.range);
    let cwd = args.repo_root.as_deref().map(Path::new);
    let output = host
        .git(&git_args, cwd)
        .map_err(|failure| Uncaught::os(&failure.filename, &failure.error))?;
    let (stdout, _) = text_streams(&output).map_err(Uncaught::decode)?;
    let argv = argv_repr("git", &git_args);
    let failure = |code, signal| {
        Uncaught::new(
            "subprocess.CalledProcessError",
            called_process_error(&argv, code, signal),
        )
    };
    match output.exit {
        Exit::Code(0) | Exit::TimedOut => Ok(parse_log(&stdout)),
        Exit::Code(code) => Err(failure(Some(code), None)),
        Exit::Signal(signal) => Err(failure(None, Some(signal))),
    }
}

/// `resolve_pull_requests`: the suffix is free; others cost one API call.
fn resolve_pull_requests(
    commits: &mut [Commit],
    repo: &str,
    gh: &mut Gh<'_>,
) -> Result<(), Uncaught> {
    for commit in commits.iter_mut() {
        if let Some((_, digits)) = pr_suffix(&commit.subject) {
            commit.pr = Some(canonical(digits));
            continue;
        }
        commit.pr = None;
        let endpoint = format!("repos/{repo}/commits/{}/pulls", commit.sha);
        let call = ["api", &endpoint, "--jq", "[.[].number]"].map(str::to_owned);
        if let Some(numbers) = gh.json(&call) {
            let Value::Array(items) = numbers else {
                return Err(Uncaught::new(
                    "pull_request",
                    "pull-request lookup must return an array of integer numbers".into(),
                ));
            };
            let numbers = items
                .iter()
                .map(pull_request_number)
                .collect::<Result<Vec<_>, _>>()?;
            if let Some(number) = numbers.first() {
                commit.pr = Some(number.clone());
                commit.linked_by_api = true;
            }
        }
    }
    Ok(())
}

fn pull_request_number(value: &Value) -> Result<String, Uncaught> {
    match value {
        Value::Int(number) if u64::try_from(*number).is_ok_and(|number| number > 0) => {
            Ok(number.to_string())
        }
        Value::BigInt(number) if number.parse::<u64>().is_ok_and(|number| number > 0) => {
            Ok(number.clone())
        }
        _ => Err(Uncaught::new(
            "pull_request",
            "pull-request number must be a positive u64 integer".into(),
        )),
    }
}

/// `recover_entries`: uncredited pull requests go beneath the next
/// credited one in commit order, or at the end when none follows.
fn recover_entries(
    commits: &[Commit],
    credited: &[String],
    repo: &str,
    gh: &mut Gh<'_>,
) -> Result<Recovered, Uncaught> {
    let mut seen: Vec<&str> = Vec::new();
    let mut insertions: Vec<(String, Vec<String>)> = Vec::new();
    let mut pending: Vec<String> = Vec::new();
    for commit in commits {
        let Some(pr) = commit.pr.as_deref() else {
            continue;
        };
        if seen.contains(&pr) {
            continue;
        }
        seen.push(pr);
        if credited.iter().any(|known| known == pr) {
            if !pending.is_empty() {
                match insertions.iter_mut().find(|(carrier, _)| carrier == pr) {
                    Some((_, lines)) => lines.append(&mut pending),
                    None => insertions.push((pr.to_owned(), std::mem::take(&mut pending))),
                }
            }
            continue;
        }
        if let Some(line) = build_entry(pr, commit, repo, gh)? {
            pending.push(line);
        }
    }
    Ok((insertions, pending))
}

fn optional_text<'a>(record: &'a Value, key: &str) -> Result<Option<&'a str>, Uncaught> {
    match record.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Str(text)) => Ok((!text.is_empty()).then_some(text.as_str())),
        Some(_) => Err(Uncaught::new(
            "pull_request",
            format!("pull-request {key} must be a string or null"),
        )),
    }
}

/// `build_entry`: the entry GitHub would have published, or `None` (with a
/// note on stderr) when the pull request has no author to credit.
fn build_entry(
    pr: &str,
    commit: &Commit,
    repo: &str,
    gh: &mut Gh<'_>,
) -> Result<Option<String>, Uncaught> {
    let call = ["pr", "view", pr, "--repo", repo, "--json", "title,author"].map(str::to_owned);
    let details = gh.json(&call).unwrap_or(Value::Object(Vec::new()));
    if !matches!(details, Value::Object(_)) {
        return Err(Uncaught::new(
            "pull_request",
            "pull-request details must be an object".into(),
        ));
    }
    let author = match details.get("author") {
        None | Some(Value::Null) => None,
        Some(author @ Value::Object(_)) => optional_text(author, "login")?,
        Some(_) => {
            return Err(Uncaught::new(
                "pull_request",
                "pull-request author must be an object or null".into(),
            ));
        }
    };
    let Some(author) = author else {
        gh.stderr.push_str(&format!(
            "release-notes-link: cannot credit #{pr} without its author; skipping it\n"
        ));
        return Ok(None);
    };
    let title =
        optional_text(&details, "title")?.unwrap_or_else(|| without_suffix(&commit.subject));
    Ok(Some(entry_line(repo, pr, title, author)))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn api_pull_request_numbers_require_positive_integer_identity() {
        assert_eq!(pull_request_number(&Value::Int(7)).unwrap(), "7");
        assert_eq!(
            pull_request_number(&Value::BigInt(u64::MAX.to_string())).unwrap(),
            u64::MAX.to_string()
        );
        for value in [
            Value::Bool(true),
            Value::Int(0),
            Value::Int(-1),
            Value::Int(i128::MAX),
            Value::Float(7.0),
            Value::Float(f64::NAN),
            Value::text("7"),
            Value::Null,
        ] {
            assert!(pull_request_number(&value).is_err());
        }
    }
    #[test]
    fn api_author_and_title_fields_are_optional_strings_without_coercion() {
        let record = |value| Value::Object(vec![("login".into(), value)]);
        assert_eq!(optional_text(&record(Value::Null), "login").unwrap(), None);
        assert_eq!(
            optional_text(&record(Value::text("")), "login").unwrap(),
            None
        );
        assert_eq!(
            optional_text(&record(Value::text("writer")), "login").unwrap(),
            Some("writer")
        );
        for value in [
            Value::Bool(false),
            Value::Int(12),
            Value::Float(3.5),
            Value::Object(vec![]),
        ] {
            assert!(optional_text(&record(value), "login").is_err());
        }
    }
}
