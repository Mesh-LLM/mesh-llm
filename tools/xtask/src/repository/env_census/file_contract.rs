//! Strict per-file contract: build-script, deferred, bootstrap and serial-test
//! mutation rules, plus the stale-TODO sweep (`check_file` in the legacy
//! script).

use super::registry::{DEFERRED_FILES, SERIAL_TEST_HELPERS, SYNCHRONOUS_BOOTSTRAP_FILES, TODO};
use super::source_scan::{
    is_serial_attribute, mutation_lines, nearest_function, preceding_comment_block,
};
use crate::command::DynResult;
use crate::repository::python_text::{split_whitespace, splitlines};
use std::path::Path;

/// The file-wide classification that selects which rule applies.
enum Role {
    BuildScript,
    Deferred,
    Bootstrap(&'static str),
    Test { has_test_module: bool },
}

struct Site<'a> {
    location: String,
    function: Option<(usize, &'a str)>,
    nearby: String,
}

pub(super) fn check_file(root: &Path, relative: &str) -> DynResult<Vec<String>> {
    let path = root.join(relative);
    if !path.is_file() {
        return Ok(vec![format!("{relative}: audited source file is missing")]);
    }
    let text = std::fs::read_to_string(&path)?;
    let lines = splitlines(&text);
    let role = role(relative, &lines);
    let mut errors = Vec::new();
    for line_index in mutation_lines(&lines) {
        let function = nearest_function(&lines, line_index);
        let site = Site {
            location: format!(
                "{relative}:{} ({})",
                line_index + 1,
                function.map_or("<module>", |(_, name)| name)
            ),
            function,
            nearby: preceding_comment_block(&lines, line_index),
        };
        errors.extend(check_site(&role, relative, &lines, &site));
    }
    if matches!(role, Role::Bootstrap(_) | Role::Test { .. }) {
        errors.extend(
            lines
                .iter()
                .enumerate()
                .filter(|(_, line)| line.contains(TODO))
                .map(|(index, _)| {
                    format!(
                        "{relative}:{}: stale environment audit TODO remains",
                        index + 1
                    )
                }),
        );
    }
    Ok(errors)
}

fn role(relative: &str, lines: &[&str]) -> Role {
    let name = relative.rsplit('/').next().unwrap_or(relative);
    if name == "build.rs" {
        return Role::BuildScript;
    }
    if DEFERRED_FILES.contains(&relative) {
        return Role::Deferred;
    }
    if let Some((_, function)) = SYNCHRONOUS_BOOTSTRAP_FILES
        .iter()
        .find(|(file, _)| *file == relative)
    {
        return Role::Bootstrap(function);
    }
    Role::Test {
        has_test_module: lines.iter().any(|line| line.contains("#[cfg(test)]"))
            || relative.contains("/tests/")
            || name == "tests.rs"
            || name.ends_with("_tests.rs"),
    }
}

fn check_site(role: &Role, relative: &str, lines: &[&str], site: &Site<'_>) -> Vec<String> {
    let at = &site.location;
    let nearby = &site.nearby;
    match role {
        Role::BuildScript => (!nearby.contains("SAFETY:")
            || !nearby.to_lowercase().contains("build script"))
        .then(|| {
            format!("{at}: build-script environment mutation needs a build-script SAFETY comment")
        })
        .into_iter()
        .collect(),
        Role::Deferred => (!nearby.contains("SAFETY:") || !nearby.contains(TODO))
            .then(|| {
                format!(
                    "{at}: deferred runtime mutation needs adjacent SAFETY and audit TODO comments"
                )
            })
            .into_iter()
            .collect(),
        Role::Bootstrap(expected) => check_bootstrap(at, expected, site),
        Role::Test {
            has_test_module: false,
        } => {
            vec![format!(
                "{at}: audited mutation is outside a recognized test module"
            )]
        }
        Role::Test {
            has_test_module: true,
        } => {
            let mut errors = Vec::new();
            if !nearby.contains("SAFETY:") {
                errors.push(format!(
                    "{at}: test environment mutation needs a SAFETY comment"
                ));
            }
            if !serial_contract(relative, lines, site) {
                errors.push(format!(
                    "{at}: test environment mutation is not covered by #[serial]"
                ));
            }
            errors
        }
    }
}

fn check_bootstrap(at: &str, expected: &str, site: &Site<'_>) -> Vec<String> {
    let mut errors = Vec::new();
    if site.function.map_or("<module>", |(_, name)| name) != expected {
        errors.push(format!(
            "{at}: bootstrap mutation must remain in {expected}"
        ));
    }
    let normalized = split_whitespace(&site.nearby).collect::<Vec<_>>().join(" ");
    if !["SAFETY:", "single-threaded", "Tokio runtime"]
        .iter()
        .all(|marker| normalized.contains(marker))
    {
        errors.push(format!(
            "{at}: bootstrap mutation needs an adjacent SAFETY comment with the single-threaded pre-Tokio ordering guarantee"
        ));
    }
    errors
}

/// A `#[serial]` attribute within eight lines above the function, or a listed
/// helper whose adjacent comment mentions serial callers.
fn serial_contract(relative: &str, lines: &[&str], site: &Site<'_>) -> bool {
    let Some((function_index, function_name)) = site.function else {
        return false;
    };
    if lines[function_index.saturating_sub(8)..function_index]
        .iter()
        .any(|line| is_serial_attribute(line))
    {
        return true;
    }
    SERIAL_TEST_HELPERS
        .iter()
        .find(|(file, _)| *file == relative)
        .is_some_and(|(_, helpers)| helpers.contains(&function_name))
        && site.nearby.to_lowercase().contains("serial")
}
