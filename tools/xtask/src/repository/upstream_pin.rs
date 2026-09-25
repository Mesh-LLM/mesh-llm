//! `repository llama-upstream-pin`: the Rust owner of
//! `scripts/check-llama-upstream-pin.py`. Rejects PRs that move the pinned
//! llama.cpp revision backwards or to a divergent history, comparing the PR
//! head against its merge base with the target.

mod git;
mod upstream_mirror;

use crate::command::DynResult;
use crate::repository::check_args::Grammar;
use crate::repository::check_report::CheckReport;
use crate::repository::python_text::{repr, strip};
use git::PinGuardError;
use std::path::{Path, PathBuf};
use upstream_mirror::UpstreamMirror;

const PIN_PATH: &str = "third_party/llama.cpp/upstream.txt";
const DEFAULT_UPSTREAM_URL: &str = "https://github.com/ggml-org/llama.cpp.git";

const GRAMMAR: Grammar = Grammar {
    usage: "cargo xtool repository llama-upstream-pin [--repository REPOSITORY] [--upstream-url UPSTREAM_URL] base_revision head_revision",
    values: &["--repository", "--upstream-url"],
    flags: &[],
};

struct PinRequest<'a> {
    repository: PathBuf,
    base: &'a str,
    head: &'a str,
    upstream_url: &'a str,
}

/// `default_repository` supplies the checkout when `--repository` is absent.
pub(crate) fn run(
    args: &[String],
    default_repository: impl FnOnce() -> DynResult<PathBuf>,
) -> DynResult<()> {
    let parsed = match GRAMMAR.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report.emit(),
    };
    let [base, head] = parsed.positionals.as_slice() else {
        let message = match parsed.positionals.len() {
            0 => "the following arguments are required: base_revision, head_revision".to_owned(),
            1 => "the following arguments are required: head_revision".to_owned(),
            _ => format!(
                "unrecognized arguments: {}",
                parsed.positionals[2..].join(" ")
            ),
        };
        return GRAMMAR.error(&message).emit();
    };
    let repository = match parsed.last("--repository") {
        Some(path) => std::path::absolute(path)?,
        None => default_repository()?,
    };
    let request = PinRequest {
        repository,
        base,
        head,
        upstream_url: parsed
            .last("--upstream-url")
            .unwrap_or(DEFAULT_UPSTREAM_URL),
    };
    let mut stdout = String::new();
    let report = match check_pin(&request, &mut stdout) {
        Ok(()) => CheckReport::success(stdout),
        Err(PinGuardError(message)) => CheckReport::failure(stdout, format!("ERROR: {message}\n")),
    };
    report.emit()
}

fn is_sha(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate(value: &str, label: &str) -> Result<(), PinGuardError> {
    if is_sha(value) {
        return Ok(());
    }
    Err(PinGuardError(format!(
        "{label} must be a lowercase 40-character SHA: {}",
        repr(value)
    )))
}

fn check_pin(request: &PinRequest<'_>, stdout: &mut String) -> Result<(), PinGuardError> {
    validate(request.base, "base revision")?;
    validate(request.head, "head revision")?;
    let repository = request.repository.as_path();
    for revision in [request.base, request.head] {
        let object = format!("{revision}^{{commit}}");
        if !git::run(repository, &["cat-file", "-e", &object], None)?.success() {
            return Err(PinGuardError(format!(
                "repository is missing {revision}; cannot inspect PR pin"
            )));
        }
    }
    let merge_base = git::run(
        repository,
        &["merge-base", request.base, request.head],
        None,
    )?;
    if !merge_base.success() {
        return Err(PinGuardError(format!(
            "cannot determine the PR merge base: {}",
            merge_base.detail()
        )));
    }
    let merge_base = strip(&merge_base.stdout).to_owned();
    if !is_sha(&merge_base) {
        return Err(PinGuardError(format!(
            "git returned an invalid PR merge base: {}",
            repr(&merge_base)
        )));
    }
    let base_pin = read_pin(repository, &merge_base, "PR merge-base")?;
    let proposed_pin = read_pin(repository, request.head, "head")?;
    stdout.push_str(&format!(
        "PR merge-base commit:   {merge_base}\nmerge-base llama.cpp pin: {base_pin}\nproposed llama.cpp pin: {proposed_pin}\n"
    ));
    if base_pin == proposed_pin {
        stdout.push_str("llama.cpp upstream pin is unchanged\n");
        return Ok(());
    }
    let mirror = UpstreamMirror::fetch(request.upstream_url, &base_pin, &proposed_pin)?;
    compare_pins(&mirror.repository(), &base_pin, &proposed_pin)?;
    stdout.push_str("llama.cpp upstream pin moves forward\n");
    Ok(())
}

/// Forward only when the base pin is an ancestor of the proposed pin and not
/// the reverse; `merge-base --is-ancestor` exits 1 for "no".
fn compare_pins(upstream: &Path, base_pin: &str, proposed_pin: &str) -> Result<(), PinGuardError> {
    let is_ancestor = |ancestor: &str, descendant: &str| -> Result<bool, PinGuardError> {
        let output = git::run(
            upstream,
            &["merge-base", "--is-ancestor", ancestor, descendant],
            None,
        )?;
        match output.code {
            Some(0) => Ok(true),
            Some(1) => Ok(false),
            _ => Err(PinGuardError(format!(
                "cannot compare llama.cpp upstream pins: {}",
                output.detail()
            ))),
        }
    };
    if is_ancestor(proposed_pin, base_pin)? {
        return Err(PinGuardError(format!(
            "PR moves {PIN_PATH} backward: {proposed_pin} is an ancestor of the base pin {base_pin}"
        )));
    }
    if is_ancestor(base_pin, proposed_pin)? {
        return Ok(());
    }
    Err(PinGuardError(
        "PR proposes a divergent llama.cpp upstream history; the guard cannot prove a forward pin update".to_owned(),
    ))
}

/// The pin must be one regular `100644` blob whose trimmed text is a SHA;
/// symlinks are rejected before their target is ever read.
fn read_pin(repository: &Path, revision: &str, label: &str) -> Result<String, PinGuardError> {
    let tree = git::checked(repository, &["ls-tree", "-z", revision, "--", PIN_PATH])?;
    let entries = tree
        .stdout
        .split('\0')
        .filter(|entry| !entry.is_empty())
        .collect::<Vec<_>>();
    let [entry] = entries.as_slice() else {
        return Err(PinGuardError(format!(
            "{label} commit {revision} must contain exactly one {PIN_PATH} entry"
        )));
    };
    let (metadata, path) = entry.split_once('\t').unwrap_or((entry, ""));
    let parts = metadata.split_whitespace().collect::<Vec<_>>();
    if path != PIN_PATH || !matches!(parts.as_slice(), ["100644", "blob", _]) {
        return Err(PinGuardError(format!(
            "{label} commit {revision} {PIN_PATH} must be a regular 100644 blob"
        )));
    }
    let shown = git::checked(repository, &["show", &format!("{revision}:{PIN_PATH}")])?;
    let pin = strip(&shown.stdout);
    if !is_sha(pin) {
        return Err(PinGuardError(format!(
            "{label} commit {revision} has an invalid {PIN_PATH} value: {}",
            repr(pin)
        )));
    }
    Ok(pin.to_owned())
}
