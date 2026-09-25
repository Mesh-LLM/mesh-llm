use super::just_recipes;
use super::scan::Candidate;
use crate::command::DynResult;
use std::collections::{BTreeSet, VecDeque};
use std::fs;
use std::path::Path;

const LANES: [&str; 5] = ["quality", "website", "linux", "macos", "windows"];

pub(super) fn required_roots(root: &Path, paths: &[String]) -> DynResult<Vec<String>> {
    let known = paths.iter().map(String::as_str).collect::<BTreeSet<_>>();
    let mut roots = vec!["Justfile".to_owned(), "just/ci.just".to_owned()];
    for lane in LANES {
        for (prefix, event) in [("pr", "pull_request:"), ("main", "push:")] {
            let path = format!(".github/workflows/{prefix}_{lane}.yml");
            if !known.contains(path.as_str())
                || !fs::read_to_string(root.join(&path))?.contains(event)
            {
                return Err(
                    format!("automation inventory: missing required {event} root {path}").into(),
                );
            }
            roots.push(path);
        }
    }
    Ok(roots)
}

fn references(line: &str) -> impl Iterator<Item = &str> {
    line.split(|ch: char| !(ch.is_ascii_alphanumeric() || matches!(ch, '/' | '.' | '_' | '-')))
        .filter(|word| {
            (word.starts_with("scripts/")
                || word.starts_with("just/")
                || word.starts_with("evals/")
                || word.starts_with("tools/")
                || word.starts_with(".github/"))
                && [".py", ".sh", ".ps1", ".just", ".yml", ".yaml"]
                    .iter()
                    .any(|extension| word.ends_with(extension))
        })
}

fn action_reference(line: &str) -> Option<&str> {
    line.trim()
        .strip_prefix("uses: ./.github/actions/")
        .or_else(|| line.trim().strip_prefix("- uses: ./.github/actions/"))
        .filter(|path| !path.is_empty() && !path.contains(char::is_whitespace))
}

fn workflow_reference(line: &str) -> Option<&str> {
    line.trim()
        .strip_prefix("uses: ./.github/workflows/")
        .or_else(|| line.trim().strip_prefix("- uses: ./.github/workflows/"))
        .filter(|path| {
            !path.is_empty()
                && !path.contains(char::is_whitespace)
                && (path.ends_with(".yml") || path.ends_with(".yaml"))
        })
}

pub(super) fn check_required_closure(
    root: &Path,
    paths: &[String],
    observed: &[Candidate],
    validated: &BTreeSet<String>,
    roots: &[&str],
) -> DynResult<()> {
    let known = paths.iter().map(String::as_str).collect::<BTreeSet<_>>();
    let mut queue = roots
        .iter()
        .map(|root| (*root).to_owned())
        .collect::<VecDeque<_>>();
    let mut visited = BTreeSet::new();
    let mut owned_children = BTreeSet::new();
    let mut recipes = BTreeSet::new();
    if roots.contains(&"Justfile") {
        recipes.insert("default".to_owned());
    }
    if roots.contains(&"just/ci.just") {
        recipes.extend(["ci-validate".to_owned(), "test-all".to_owned()]);
    }
    while let Some(path) = queue.pop_front() {
        if !known.contains(path.as_str()) || !root.join(&path).is_file() {
            return Err(format!("automation inventory: missing required source {path}").into());
        }
        if !visited.insert(path.clone()) {
            continue;
        }
        if path.ends_with(".py") && !owned_children.contains(&path) {
            return Err(
                format!("automation inventory: unowned required Python child {path}").into(),
            );
        }
        let text = fs::read_to_string(root.join(&path))?;
        if text.lines().next().is_some_and(|line| {
            line.starts_with("#!") && (line.contains("python") || line.contains("/env py"))
        }) && !path.ends_with(".py")
            && !observed.iter().any(|row| {
                row.path == path
                    && row.executable
                    && row.source_block == text.lines().next().unwrap_or("")
                    && validated.contains(&row.id)
            })
        {
            return Err(format!("automation inventory: required Python shebang in {path}").into());
        }
        for line in text
            .lines()
            .filter(|line| !line.trim_start().starts_with('#'))
        {
            if (path.ends_with(".yml") || path.ends_with(".yaml"))
                && !line.trim_start().starts_with("echo ")
                && !line.trim_start().starts_with("#")
            {
                let command = line
                    .trim()
                    .trim_start_matches("run: ")
                    .trim_start_matches('@');
                if let Some(recipe) = command
                    .strip_prefix("just ")
                    .and_then(|rest| rest.split_whitespace().next())
                    && recipe
                        .chars()
                        .all(|ch| ch.is_ascii_alphanumeric() || ch == '-')
                {
                    recipes.insert(recipe.to_owned());
                }
            }
            if path == "scripts/manage-build-cache.py" {
                if line.contains("[\"just\", \"cache-cargo-metadata\"]") {
                    recipes.insert("cache-cargo-metadata".to_owned());
                }
                if line.contains("[\"just\", \"cache-cargo-clean\"]") {
                    recipes.insert("cache-cargo-clean".to_owned());
                }
            }
            if let Some(action) = action_reference(line) {
                queue.push_back(format!(".github/actions/{action}/action.yml"));
            }
            if let Some(workflow) = workflow_reference(line) {
                queue.push_back(format!(".github/workflows/{workflow}"));
            }
            for child in references(line) {
                let source_edge = observed
                    .iter()
                    .find(|row| row.path == path && row.source_block == line.trim());
                if source_edge.is_some_and(|row| !row.executable && validated.contains(&row.id)) {
                    continue;
                }
                if !known.contains(child) {
                    return Err(format!(
                    "automation inventory: required closure child {child} from {path} is absent from source roster"
                    )
                    .into());
                }
                if child.ends_with(".py")
                    && source_edge.is_some_and(|row| row.executable && validated.contains(&row.id))
                {
                    owned_children.insert(child.to_owned());
                }
                queue.push_back(child.to_owned());
            }
        }
        if let Some(candidate) = observed
            .iter()
            .find(|row| row.path == path && row.executable && !validated.contains(&row.id))
        {
            return Err(format!(
                "automation inventory: unowned required closure execution in {path} [{}] targets {}; record its exact source ownership before adding this child",
                candidate.id,
                references(&candidate.source_block).next().unwrap_or("inline Python")
            )
            .into());
        }
    }
    just_recipes::check_recipe_children(root, &recipes)?;
    Ok(())
}
