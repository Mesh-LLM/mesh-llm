//! `release notes-link`: `scripts/release-notes-link.py`. Pairs each commit
//! of a range with its pull request, credits pull requests GitHub's body
//! left out, and writes the augmented body plus the API-linked records.

use crate::ci_operations::ci_metrics_int::python_int_text;
use crate::ci_operations::ci_metrics_value::{Value, display};
use crate::prepared_input::python_io::decode_utf8;
use crate::release::link_argv::{Args, parse};
use crate::release::link_body::{Body, augment, canonical, dump_links, entry_line, read_body};
use crate::release::link_commits::{
    Commit, commit_links, log_args, parse_log, pr_suffix, without_suffix,
};
use crate::release::link_gh::{Gh, truthy, type_name};
use crate::release::link_host::{Exit, ReleaseHost, text_streams};
use crate::release::python_failure::{Uncaught, argv_repr, called_process_error};
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
        if let Some(numbers) = gh.json(&call)?.filter(truthy) {
            commit.pr = Some(python_int(&first_item(&numbers)?)?);
            commit.linked_by_api = true;
        }
    }
    Ok(())
}

/// `numbers[0]` of a truthy JSON value.
fn first_item(numbers: &Value) -> Result<Value, Uncaught> {
    match numbers {
        Value::Array(items) => Ok(items.first().cloned().unwrap_or(Value::Null)),
        Value::Str(text) => Ok(Value::Str(text.chars().take(1).collect())),
        // JSON object keys are strings, so the int key `0` is never present.
        Value::Object(_) => Err(Uncaught::new("KeyError", "0".to_owned())),
        other => Err(Uncaught::new(
            "TypeError",
            format!("'{}' object is not subscriptable", type_name(other)),
        )),
    }
}

/// `int(value)` as canonical decimal text.
fn python_int(value: &Value) -> Result<String, Uncaught> {
    match value {
        Value::Bool(flag) => Ok(u8::from(*flag).to_string()),
        Value::Int(int) => Ok(int.to_string()),
        Value::BigInt(text) => Ok(text.clone()),
        Value::Float(float) if float.is_nan() => Err(Uncaught::new(
            "ValueError",
            "cannot convert float NaN to integer".to_owned(),
        )),
        Value::Float(float) if float.is_infinite() => Err(Uncaught::new(
            "OverflowError",
            "cannot convert float infinity to integer".to_owned(),
        )),
        // `int(float)` truncates toward zero; `{:.0}` prints it exactly.
        Value::Float(float) => Ok(format!("{:.0}", float.trunc() + 0.0)),
        Value::Str(text) => python_int_text(text).ok_or_else(|| {
            Uncaught::new(
                "ValueError",
                format!(
                    "invalid literal for int() with base 10: {}",
                    crate::repository::python_text::repr(text)
                ),
            )
        }),
        other => Err(Uncaught::new(
            "TypeError",
            format!(
                "int() argument must be a string, a bytes-like object or a real number, not '{}'",
                type_name(other)
            ),
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

/// `mapping.get(key)` with the `AttributeError` of a non-dict.
fn get(value: &Value, key: &str) -> Result<Value, Uncaught> {
    match value {
        Value::Object(_) => Ok(value.get(key).cloned().unwrap_or(Value::Null)),
        other => Err(Uncaught::new(
            "AttributeError",
            format!("'{}' object has no attribute 'get'", type_name(other)),
        )),
    }
}

fn or_empty(value: Value) -> Value {
    if truthy(&value) {
        value
    } else {
        Value::Object(Vec::new())
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
    let details = or_empty(gh.json(&call)?.unwrap_or(Value::Null));
    let author = get(&or_empty(get(&details, "author")?), "login")?;
    if !truthy(&author) {
        gh.stderr.push_str(&format!(
            "release-notes-link: cannot credit #{pr} without its author; skipping it\n"
        ));
        return Ok(None);
    }
    let title = get(&details, "title")?;
    let title = if truthy(&title) {
        display(&title)
    } else {
        without_suffix(&commit.subject).to_owned()
    };
    Ok(Some(entry_line(repo, pr, &title, &display(&author))))
}
