//! Scripts reachable from jobs that restore product automation run the restored
//! executable; `cargo xtool` stays only as the fallback when none was restored.
use super::{Node, field, handoffs as h};
use crate::command::DynResult;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

const RESTORE: &str = "./.github/actions/restore-automation";
const BIN: &str = "MESH_LLM_AUTOMATION_BIN";

pub(super) fn check(root: &Path, workflows: &BTreeMap<String, Node>) -> DynResult<()> {
    let mut failures = Vec::new();
    for script in reachable_scripts(root, workflows)? {
        let source = std::fs::read_to_string(root.join(&script))?;
        for line in direct_calls(&source) {
            failures.push(format!("{}:{line}", script.display()));
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "restored automation consumers reach direct cargo xtool: {}",
            failures.join(", ")
        )
        .into())
    }
}

/// Every checkout script named by a restoring job, its local actions, or a reached script.
fn reachable_scripts(
    root: &Path,
    workflows: &BTreeMap<String, Node>,
) -> DynResult<BTreeSet<PathBuf>> {
    let mut index = BTreeMap::<String, Vec<PathBuf>>::new();
    index_scripts(root, Path::new(""), &mut index)?;
    let mut pending = Vec::new();
    let mut actions = BTreeSet::new();
    for workflow in workflows.values() {
        for (_, job) in h::member(workflow, "jobs")?.entries() {
            let Ok(steps) = h::steps(job) else { continue };
            if !steps
                .iter()
                .any(|step| field(step, "uses") == Some(RESTORE))
            {
                continue;
            }
            for step in steps {
                pending.extend(script_names(field(step, "run").unwrap_or("")));
                if let Some(action) = field(step, "uses").and_then(local_action) {
                    pending.extend(action_scripts(root, action, &mut actions)?);
                }
            }
        }
    }
    let mut reached = BTreeSet::new();
    while let Some(name) = pending.pop() {
        for script in index.get(&name).into_iter().flatten() {
            if reached.insert(script.clone()) {
                pending.extend(script_names(&std::fs::read_to_string(root.join(script))?));
            }
        }
    }
    Ok(reached)
}

fn local_action(uses: &str) -> Option<&str> {
    uses.strip_prefix("./.github/actions/")
        .filter(|name| *name != "restore-automation")
}

fn action_scripts(root: &Path, name: &str, seen: &mut BTreeSet<String>) -> DynResult<Vec<String>> {
    if !seen.insert(name.to_owned()) {
        return Ok(Vec::new());
    }
    let source = std::fs::read_to_string(root.join(format!(".github/actions/{name}/action.yml")))?;
    let mut names = script_names(&source);
    for line in source.lines() {
        if let Some(nested) = line.trim().strip_prefix("uses: ").and_then(local_action) {
            names.extend(action_scripts(root, nested, seen)?);
        }
    }
    Ok(names)
}

fn script_names(text: &str) -> Vec<String> {
    text.split(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-')))
        .filter(|token| token.len() > 3 && token.ends_with(".sh"))
        .map(str::to_owned)
        .collect()
}

fn index_scripts(
    root: &Path,
    relative: &Path,
    index: &mut BTreeMap<String, Vec<PathBuf>>,
) -> DynResult<()> {
    for entry in std::fs::read_dir(root.join(relative))? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        let path = relative.join(&name);
        if entry.file_type()?.is_dir() {
            if !name.starts_with('.') && !matches!(name.as_str(), "target" | "node_modules") {
                index_scripts(root, &path, index)?;
            }
        } else if name.ends_with(".sh") {
            index.entry(name).or_default().push(path);
        }
    }
    Ok(())
}

struct Branch {
    selects_restored: bool,
    fallback: bool,
}

/// One-based lines that invoke `cargo xtool` outside a restored-executable fallback.
fn direct_calls(source: &str) -> Vec<usize> {
    let default_overridden = source.contains(&format!("automation=(\"${BIN}\")"));
    let mut branches: Vec<Branch> = Vec::new();
    let mut direct = Vec::new();
    for (index, line) in source.lines().enumerate() {
        let line = line.trim();
        if line.starts_with('#') {
            continue;
        }
        if line.starts_with("if ") && !line.ends_with("fi") {
            branches.push(Branch {
                selects_restored: line.contains(BIN),
                fallback: false,
            });
        } else if line.starts_with("elif ") || line == "else" {
            if let Some(branch) = branches.last_mut() {
                branch.fallback = true;
            }
        } else if line == "fi" {
            branches.pop();
        }
        if !line.contains("cargo xtool") {
            continue;
        }
        let inline_fallback = line.contains(BIN) && line.contains("else");
        let overridden_default = line == "automation=(cargo xtool)" && default_overridden;
        let in_fallback = branches
            .iter()
            .rev()
            .find(|branch| branch.selects_restored)
            .is_some_and(|branch| branch.fallback);
        if !(inline_fallback || overridden_default || in_fallback) {
            direct.push(index + 1);
        }
    }
    direct
}

#[cfg(test)]
#[path = "restored_scripts_tests.rs"]
mod tests;
