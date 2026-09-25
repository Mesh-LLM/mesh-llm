//! `workflow_jobs` and the image-binding census of `check` in
//! `scripts/runner-image-identity.py`: every literal runner image in every
//! workflow must be a registered binding with the catalogued digest.

use crate::ci_operations::identity_text::REPOSITORY;
use crate::ci_operations::python_access::{Outcome, require};
use crate::ci_operations::workflow_text::{blank_to_eol, line_starts};
use crate::ci_plan::catalog::{os_error_text, python_path_display};
use crate::repository::python_text::splitlines;
use std::collections::BTreeMap;
use std::path::Path;

/// Jobs of one workflow in declaration order: `(name, body)`.
pub(crate) type Jobs = Vec<(String, String)>;

pub(crate) fn read_text(path: &Path) -> Outcome<String> {
    let bytes =
        std::fs::read(path).map_err(|error| os_error_text(&error, &python_path_display(path)))?;
    String::from_utf8(bytes).map_err(|error| {
        let start = error.utf8_error().valid_up_to();
        format!(
            "'utf-8' codec can't decode byte 0x{:02x} in position {start}: invalid start byte",
            error.as_bytes()[start]
        )
    })
}

fn declaration_allowed(line: &str) -> bool {
    [
        (6, "image:"),
        (12, "runner_image:"),
        (10, "allow_trusted_seed:"),
    ]
    .iter()
    .any(|(width, head)| {
        let spaces = line.len() - line.trim_start_matches(' ').len();
        spaces == *width
            && line[spaces..].strip_prefix(head).is_some_and(|tail| {
                let value = tail.trim_start_matches([' ', '\t']);
                value.len() < tail.len()
                    && value.chars().next().is_some_and(|ch| !ch.is_whitespace())
            })
    })
}

fn is_job_header(line: &str) -> Option<&str> {
    let name = line.strip_prefix("  ")?;
    let colon = name.find(':')?;
    let (ident, tail) = (&name[..colon], &name[colon + 1..]);
    let mut chars = ident.chars();
    let first_ok = chars
        .next()
        .is_some_and(|ch| ch.is_ascii_alphabetic() || ch == '_');
    let rest_ok = chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '-');
    (first_ok && rest_ok && blank_to_eol(tail, 0).is_some()).then_some(ident)
}

fn guard_lines(text: &str) -> Vec<&str> {
    line_starts(text)
        .into_iter()
        .map(|start| {
            &text[start
                ..text[start..]
                    .find('\n')
                    .map_or(text.len(), |end| start + end)]
        })
        .filter(|line| {
            line.strip_prefix("          allow_trusted_seed:")
                .is_some_and(|tail| tail.contains(REPOSITORY))
        })
        .collect()
}

pub(crate) fn workflow_jobs(path: &Path, seed_guard_job: Option<&str>) -> Outcome<Jobs> {
    let text = read_text(path)?;
    let name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default();
    for (number, line) in splitlines(&text).into_iter().enumerate() {
        if line.contains(REPOSITORY)
            && !crate::repository::python_text::strip(line).starts_with('#')
        {
            require(declaration_allowed(line), || {
                format!(
                    "{name}:{}: unsupported runner image reference declaration",
                    number + 1
                )
            })?;
        }
    }
    let heads: Vec<usize> = line_starts(&text)
        .into_iter()
        .filter(|start| {
            text[*start..]
                .strip_prefix("jobs:")
                .is_some_and(|tail| blank_to_eol(tail, 0).is_some())
        })
        .collect();
    require(heads.len() == 1, || {
        format!("{name}: expected one block-style jobs mapping")
    })?;
    let after = &text[heads[0]..];
    let after = &after[5..];
    let body_start = blank_to_eol(after, 0).unwrap_or(0);
    let body = &after[body_start..];
    let body = line_starts(body)
        .into_iter()
        .find(|start| {
            body[*start..]
                .chars()
                .next()
                .is_some_and(|ch| !crate::repository::python_text::is_space(ch))
        })
        .map_or(body, |stop| &body[..stop]);
    let headers: Vec<(usize, usize, &str)> = line_starts(body)
        .into_iter()
        .filter_map(|start| {
            let line = &body[start..];
            let ident = is_job_header(line)?;
            let tail = &line[2 + ident.len() + 1..];
            let end = blank_to_eol(tail, 0)? + start + 2 + ident.len() + 1;
            Some((start, end, ident))
        })
        .collect();
    let mut jobs: Jobs = Vec::new();
    for (index, (_, end, ident)) in headers.iter().enumerate() {
        require(!jobs.iter().any(|(known, _)| known == ident), || {
            format!("{name}: duplicate job {ident}")
        })?;
        let stop = headers.get(index + 1).map_or(body.len(), |next| next.0);
        jobs.push(((*ident).to_owned(), body[(*end).min(stop)..stop].to_owned()));
    }
    require(!jobs.is_empty(), || {
        format!("{name}: unsupported jobs mapping")
    })?;
    let scoped = seed_guard_job
        .and_then(|job| jobs.iter().find(|(known, _)| known == job))
        .map_or_else(Vec::new, |(_, body)| guard_lines(body));
    require(guard_lines(&text) == scoped, || {
        format!("{name}: seed guard image references are limited to the runtime consumer job")
    })?;
    Ok(jobs)
}

pub(crate) fn job<'a>(jobs: &'a Jobs, name: &str) -> Option<&'a str> {
    jobs.iter()
        .find(|(known, _)| known == name)
        .map(|(_, body)| body.as_str())
}

/// All workflows by file name (sorted, like `sorted(Path.iterdir())`).
pub(crate) type Workflows = BTreeMap<String, Jobs>;
