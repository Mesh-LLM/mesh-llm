//! The Windows ABI warmer is the only job that saves those caches on main, so
//! every input of their key has to wake it: an input it does not watch changes
//! the key on main with no warmed entry behind it, and the next Windows runtime
//! run of every PR builds the native tree cold.
use super::{Node, workflow_yaml};
use crate::ci_operations::cache_inputs::Inventory;
use crate::command::DynResult;
use std::collections::BTreeSet;
use std::path::Path;

const WARMER: &str = ".github/workflows/windows-warm-caches.yml";

pub(super) fn check(root: &Path) -> DynResult<()> {
    let document = workflow_yaml::parse(&std::fs::read_to_string(root.join(WARMER))?)?;
    let missing = unwatched(&Inventory::WindowsAbi.inputs(), &push_paths(&document)?)?;
    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{WARMER}: on.push.paths must cover every Windows ABI cache key input; missing {}",
            missing.join(", ")
        )
        .into())
    }
}

fn push_paths(document: &Node) -> DynResult<Vec<String>> {
    let Some(Node::Seq(paths)) = document
        .get("on")
        .and_then(|on| on.get("push"))
        .and_then(|push| push.get("paths"))
    else {
        return Err(format!("{WARMER}: on.push.paths is not a list").into());
    };
    paths
        .iter()
        .map(|path| {
            path.text()
                .map(str::to_owned)
                .ok_or_else(|| format!("{WARMER}: on.push.paths holds a non-scalar entry").into())
        })
        .collect()
}

/// Key inputs no push path covers, judged conservatively: a path covers an
/// input it equals, or that sits under its literal directory when the path is
/// `<dir>/**` (`scripts/**` covers `scripts/lib/lld.sh` and
/// `tools/xtask/src/**` covers `tools/xtask/src/native_policy/**`). Any other
/// glob covers only its own text, since matching one pattern's text against
/// another says nothing about the files GitHub filters. Ordered `!` exclusions
/// are refused rather than evaluated.
fn unwatched(inputs: &BTreeSet<String>, watched: &[String]) -> DynResult<Vec<String>> {
    if let Some(exclusion) = watched.iter().find(|path| path.starts_with('!')) {
        return Err(format!(
            "{WARMER}: on.push.paths exclusion {exclusion:?} is not supported by the key coverage check"
        )
        .into());
    }
    Ok(inputs
        .iter()
        .filter(|input| !watched.iter().any(|path| covers(path, input)))
        .cloned()
        .collect())
}

fn covers(path: &str, input: &str) -> bool {
    path == input
        || path.strip_suffix("**").is_some_and(|directory| {
            directory.ends_with('/')
                && !directory.contains(['*', '?', '[', ']', '!', '{', '}'])
                && input.starts_with(directory)
        })
}

#[cfg(test)]
#[path = "windows_warmer_paths_tests.rs"]
mod tests;
