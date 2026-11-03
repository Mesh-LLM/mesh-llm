//! Source-only reverse reachability for checked-in reusable workflow calls.

use super::{Node, document, support, text};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;

fn target(uses: &str) -> Option<(&str, Option<&str>)> {
    if let Some(path) = uses.strip_prefix("./.github/workflows/") {
        return Some((path, None));
    }
    let prefix = "mesh-llm/mesh-llm/.github/workflows/";
    if uses
        .get(..prefix.len())
        .is_some_and(|value| value.eq_ignore_ascii_case(prefix))
    {
        let (path, revision) = uses[prefix.len()..]
            .split_once('@')
            .expect("repository workflow revision");
        return Some((path, Some(revision)));
    }
    None
}

fn paths(workflows: &BTreeMap<String, Node>, seeds: &BTreeSet<String>) -> BTreeSet<String> {
    let mut affected = seeds.clone();
    loop {
        let before = affected.len();
        for (file, doc) in workflows {
            for (_, definition) in doc.get("jobs").unwrap().entries() {
                if text(definition, "uses")
                    .and_then(target)
                    .is_some_and(|(path, _)| affected.contains(path))
                {
                    affected.insert(file.clone());
                }
            }
        }
        if affected.len() == before {
            break;
        }
    }
    let mut found = BTreeSet::new();
    for (file, doc) in workflows {
        for (name, definition) in doc.get("jobs").unwrap().entries() {
            let Some((path, revision)) = text(definition, "uses").and_then(target) else {
                continue;
            };
            if affected.contains(path) {
                let suffix = revision.map_or_else(String::new, |value| format!("@{value}"));
                assert!(found.insert(format!("{file}/{name} -> {path}{suffix}")));
            }
        }
    }
    found
}

/// List checked-in call edges leading to `seeds`; protected refs are source
/// declarations, not proof that the checked-out revision matches `@main`.
pub(super) fn callers(seeds: &BTreeSet<String>) -> BTreeSet<String> {
    let mut workflows = BTreeMap::new();
    for entry in fs::read_dir(support::root().join(".github/workflows")).unwrap() {
        let file = entry.unwrap().file_name().into_string().unwrap();
        if file.ends_with(".yml") || file.ends_with(".yaml") {
            workflows.insert(file.clone(), document(&file));
        }
    }
    paths(&workflows, seeds)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transitive_local_and_protected_calls_are_counted_without_external_repositories() {
        let parse = |source| super::super::workflow_yaml::parse(source).unwrap();
        let workflows = BTreeMap::from([
            (
                "slice.yml".into(),
                parse("jobs:\n  selector:\n    runs-on: ubuntu-latest\n"),
            ),
            (
                "lane.yml".into(),
                parse("jobs:\n  build:\n    uses: ./.github/workflows/slice.yml\n"),
            ),
            (
                "entry.yml".into(),
                parse(
                    "jobs:\n  lane:\n    uses: Mesh-LLM/mesh-llm/.github/workflows/lane.yml@main\n",
                ),
            ),
            (
                "outer.yml".into(),
                parse("jobs:\n  entry:\n    uses: ./.github/workflows/entry.yml\n"),
            ),
            (
                "other.yml".into(),
                parse("jobs:\n  build:\n    uses: other/repo/.github/workflows/slice.yml@main\n"),
            ),
        ]);
        assert_eq!(
            paths(&workflows, &BTreeSet::from(["slice.yml".into()])),
            BTreeSet::from([
                "entry.yml/lane -> lane.yml@main".into(),
                "lane.yml/build -> slice.yml".into(),
                "outer.yml/entry -> entry.yml".into(),
            ])
        );
    }
}
