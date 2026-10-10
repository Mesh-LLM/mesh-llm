//! Crate ownership and reverse-dependency closure over `cargo metadata
//! --no-deps`, mirroring the jq/Bash pipeline in `scripts/affected-crates.sh`.

use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// A Cargo failure that makes the legacy script fail open, carrying the exit
/// status its `WARNING` line reports.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct FailOpen(pub(super) i32);

/// jq's exit status for a parse or type error on the metadata document.
const JQ_ERROR: i32 = 5;

/// Package names, workspace-relative directories and reverse dependencies.
pub(super) struct CrateGraph {
    /// Relative directory per package name; a later duplicate name wins.
    dirs: BTreeMap<String, String>,
    /// Dependents per package in metadata declaration order, duplicates kept.
    dependents: BTreeMap<String, Vec<String>>,
}

impl CrateGraph {
    /// Builds the graph from raw metadata stdout. Non-JSON or a non-object
    /// document fails open like `jq -r .workspace_root`; a missing or
    /// malformed package list yields an empty graph, as jq's errors inside
    /// the process substitutions only truncate the legacy maps.
    pub(super) fn parse(stdout: &[u8]) -> Result<Self, FailOpen> {
        let document: Value = serde_json::from_slice(stdout).map_err(|_| FailOpen(JQ_ERROR))?;
        let root = match &document {
            Value::Object(object) => object.get("workspace_root").map_or_else(
                || "null".to_owned(),
                |root| {
                    root.as_str()
                        .map_or_else(|| root.to_string(), str::to_owned)
                },
            ),
            Value::Null => "null".to_owned(),
            _ => return Err(FailOpen(JQ_ERROR)),
        };
        let packages = document
            .get("packages")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or_default();
        let mut dirs = BTreeMap::new();
        for package in packages {
            if let (Some(name), Some(manifest)) = (
                package.get("name").and_then(Value::as_str),
                package.get("manifest_path").and_then(Value::as_str),
            ) {
                dirs.insert(name.to_owned(), relative_dir(&root, manifest));
            }
        }
        let mut dependents: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for package in packages {
            let Some(node) = package.get("name").and_then(Value::as_str) else {
                continue;
            };
            let dependencies = package.get("dependencies").and_then(Value::as_array);
            for dependency in dependencies.map(Vec::as_slice).unwrap_or_default() {
                let Some(name) = dependency.get("name").and_then(Value::as_str) else {
                    continue;
                };
                if !node.is_empty() && dirs.contains_key(name) {
                    dependents
                        .entry(name.to_owned())
                        .or_default()
                        .push(node.to_owned());
                }
            }
        }
        Ok(Self { dirs, dependents })
    }

    /// The owning crate of `file`: the longest non-root directory prefix.
    pub(super) fn owner(&self, file: &str) -> Option<&str> {
        self.dirs
            .iter()
            .filter(|(_, dir)| {
                !dir.is_empty()
                    && (file == dir.as_str()
                        || file
                            .strip_prefix(dir.as_str())
                            .is_some_and(|rest| rest.starts_with('/')))
            })
            .max_by_key(|(_, dir)| dir.len())
            .map(|(name, _)| name.as_str())
    }

    /// Breadth-first closure over dependents, starting from `seeds` in order.
    pub(super) fn closure(&self, seeds: &[String]) -> Vec<String> {
        let mut queue = seeds.iter().cloned().collect::<VecDeque<_>>();
        let mut visited = BTreeSet::new();
        let mut affected = Vec::new();
        while let Some(current) = queue.pop_front() {
            if !visited.insert(current.clone()) {
                continue;
            }
            for dependent in self.dependents.get(&current).into_iter().flatten() {
                if !visited.contains(dependent) {
                    queue.push_back(dependent.clone());
                }
            }
            affected.push(current);
        }
        affected
    }
}

/// `${manifest_path%/Cargo.toml}` made relative to the workspace root.
fn relative_dir(root: &str, manifest: &str) -> String {
    let dir = manifest.strip_suffix("/Cargo.toml").unwrap_or(manifest);
    if dir == root {
        return String::new();
    }
    dir.strip_prefix(root)
        .and_then(|rest| rest.strip_prefix('/'))
        .unwrap_or(dir)
        .to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph() -> CrateGraph {
        let metadata = serde_json::json!({
            "workspace_root": "/ws",
            "packages": [
                {"name": "a", "manifest_path": "/ws/crates/a/Cargo.toml", "dependencies": []},
                {"name": "b", "manifest_path": "/ws/crates/b/Cargo.toml",
                 "dependencies": [{"name": "a"}, {"name": "serde"}, {"name": "a"}]},
                {"name": "c", "manifest_path": "/ws/crates/a/c/Cargo.toml",
                 "dependencies": [{"name": "b"}]},
                {"name": "root", "manifest_path": "/ws/Cargo.toml"}
            ]
        });
        CrateGraph::parse(metadata.to_string().as_bytes()).expect("valid metadata")
    }

    #[test]
    fn migration_repository_affected_owner_is_longest_prefix() {
        let graph = graph();
        assert_eq!(graph.owner("crates/a/c/src/lib.rs"), Some("c"));
        assert_eq!(graph.owner("crates/a/src/lib.rs"), Some("a"));
        assert_eq!(graph.owner("crates/a"), Some("a"));
        assert_eq!(graph.owner("crates/ab/src/lib.rs"), None);
        assert_eq!(graph.owner("Cargo.toml"), None);
    }

    #[test]
    fn migration_repository_affected_closure_is_breadth_first() {
        let graph = graph();
        assert_eq!(graph.closure(&["a".to_owned()]), ["a", "b", "c"]);
        assert_eq!(
            graph.closure(&["c".to_owned(), "a".to_owned()]),
            ["c", "a", "b"]
        );
    }

    #[test]
    fn migration_repository_affected_malformed_metadata_classes() {
        assert_eq!(CrateGraph::parse(b"not json").err(), Some(FailOpen(5)));
        assert_eq!(CrateGraph::parse(b"[1]").err(), Some(FailOpen(5)));
        let empty = CrateGraph::parse(b"{}").expect("object without packages");
        assert!(empty.closure(&[]).is_empty());
        assert_eq!(empty.owner("crates/a/src/lib.rs"), None);
    }
}
