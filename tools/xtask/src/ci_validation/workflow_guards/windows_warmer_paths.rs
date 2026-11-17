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

/// Key inputs no push path covers. A path covers an input it equals or
/// matches as a glob (`scripts/**` covers `scripts/lib/lld.sh` and
/// `tools/xtask/src/**` covers `tools/xtask/src/native_policy/**`).
fn unwatched(inputs: &BTreeSet<String>, watched: &[String]) -> DynResult<Vec<String>> {
    let patterns = watched
        .iter()
        .map(|path| glob::Pattern::new(path))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(inputs
        .iter()
        .filter(|input| !patterns.iter().any(|pattern| pattern.matches(input)))
        .cloned()
        .collect())
}

#[cfg(test)]
#[path = "windows_warmer_paths_tests.rs"]
mod tests;
