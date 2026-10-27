//! Contract: the platform slice filters planned unit-test owners against the
//! checked-out workspace before the platform test loops consume them (#2151).
//!
//! The `platform_checks` unit rows resolve their package lists through the
//! protected `resolve-cargo-packages` action. That resolver passes absent *unmapped*
//! names through unchanged, so a fixed-list member that has since been removed from
//! the workspace would reach the test loops and abort the run with
//! "package ID specification <crate> did not match any packages" instead of a
//! documented skip. The quality and rust-tests slices already resolve their planned
//! batches against the checked-out workspace's `cargo metadata` members; the
//! platform slice must do the same, so the test commands consume a
//! membership-filtered list rather than the raw resolver output.

use crate::ci_validation::lane_results::workflow_yaml::{self, Node};
use crate::command::DynResult;
use std::fs;
use std::path::Path;

const PLATFORM_SLICE: &str = ".github/workflows/ci-platform-checks-slice.yml";
const FILTER_NAME_MARKER: &str = "against the checked-out workspace";
const JOB: &str = "platform_checks";
const MACOS_TEST_STEP: &str = "Run macOS unit tests";
const WINDOWS_TEST_STEP: &str = "Run Windows unit tests";
const MACOS_TEST_ENVS: [&str; 1] = ["TEST_CRATES"];
const WINDOWS_TEST_ENVS: [&str; 3] = [
    "TEST_CRATES",
    "WINDOWS_TEST_CRATES",
    "WINDOWS_DYNAMIC_TEST_CRATES",
];

/// Enforce the contract against the checked-out repository.
pub(super) fn check(repo_root: &Path) -> DynResult<()> {
    let source = fs::read_to_string(repo_root.join(PLATFORM_SLICE))?;
    check_source(&source)
}

fn check_source(source: &str) -> DynResult<()> {
    let document =
        workflow_yaml::parse(source).map_err(|error| format!("parse {PLATFORM_SLICE}: {error}"))?;
    let job = document
        .get("jobs")
        .and_then(|jobs| jobs.get(JOB))
        .ok_or_else(|| format!("{PLATFORM_SLICE}: missing jobs.{JOB}"))?;
    let steps = match job
        .get("steps")
        .ok_or_else(|| format!("{PLATFORM_SLICE}: jobs.{JOB} has no steps"))?
    {
        Node::Seq(items) => items,
        _ => return Err(format!("{PLATFORM_SLICE}: jobs.{JOB}.steps is not a sequence").into()),
    };

    let filtered_ids = filtered_step_ids(steps);
    if filtered_ids.is_empty() {
        return Err(format!(
            "{PLATFORM_SLICE}: the {JOB} unit rows must resolve their planned unit-test \
             owners against the checked-out workspace (a step named \
             '{FILTER_NAME_MARKER}') before the test loops consume them, but no such step \
             exists; removing a crate from a fixed list would abort the run with a \
             missing-package failure"
        )
        .into());
    }

    consume_filtered(steps, MACOS_TEST_STEP, &MACOS_TEST_ENVS, &filtered_ids)?;
    consume_filtered(steps, WINDOWS_TEST_STEP, &WINDOWS_TEST_ENVS, &filtered_ids)?;
    Ok(())
}

/// Step ids of the membership-filter steps, identified by their name.
fn filtered_step_ids(steps: &[Node]) -> Vec<String> {
    steps
        .iter()
        .filter(|step| {
            step.get("name")
                .and_then(Node::text)
                .is_some_and(|name| name.contains(FILTER_NAME_MARKER))
        })
        .filter_map(|step| step.get("id").and_then(Node::text).map(str::to_owned))
        .collect()
}

/// Assert each `env` of the named test step references a filtered step's output.
fn consume_filtered(
    steps: &[Node],
    step_name: &str,
    env_keys: &[&str],
    filtered_ids: &[String],
) -> DynResult<()> {
    let step = steps
        .iter()
        .find(|candidate| candidate.get("name").and_then(Node::text) == Some(step_name))
        .ok_or_else(|| format!("{PLATFORM_SLICE}: missing step '{step_name}'"))?;

    for key in env_keys {
        let value = step
            .get("env")
            .and_then(|env| env.get(key))
            .and_then(Node::text)
            .map(str::to_owned)
            .ok_or_else(|| format!("{PLATFORM_SLICE}: step '{step_name}' env {key} missing"))?;
        let referenced = referenced_step_id(&value).ok_or_else(|| {
            format!(
                "{PLATFORM_SLICE}: step '{step_name}' env {key} ({value}) must reference \
                 steps.<id>.outputs.<name>"
            )
        })?;
        if !filtered_ids.contains(&referenced) {
            return Err(format!(
                "{PLATFORM_SLICE}: step '{step_name}' env {key} consumes resolver output \
                 '{referenced}' directly; it must consume the membership-filtered list so a \
                 removed crate is skipped with a warning, not a missing-package failure"
            )
            .into());
        }
    }
    Ok(())
}

/// Extract the referenced step id from a `steps.<id>.outputs.<name>` expression.
fn referenced_step_id(value: &str) -> Option<String> {
    let inner = value.trim();
    let inner = inner
        .strip_prefix("${{")
        .and_then(|rest| rest.strip_suffix("}}"))
        .unwrap_or(inner)
        .trim();
    inner
        .strip_prefix("steps.")
        .and_then(|rest| rest.split_once(".outputs."))
        .map(|(step_id, _)| step_id.trim().to_owned())
        .filter(|step_id| !step_id.is_empty())
}

#[cfg(test)]
mod tests {
    use super::{check_source, PLATFORM_SLICE};
    use crate::repo_consistency::repo_root;
    use std::fs;

    #[test]
    fn platform_test_loops_consume_membership_filtered_lists() {
        let root = repo_root().expect("resolve the repository root");
        let source =
            fs::read_to_string(root.join(PLATFORM_SLICE)).expect("read the platform slice");
        check_source(&source).expect(
            "the platform slice must filter planned unit-test owners against the checked-out \
             workspace before the test loops consume them",
        );
    }
}
