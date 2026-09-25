use crate::command::DynResult;
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::PathBuf;

/// Workspace-local packages the bootstrap tool may compile. Every other
/// workspace crate is product, native-runtime, or UI code and would pull
/// llama.cpp or UI preparation into automation bootstrap.
const BOOTSTRAP_LOCAL_PACKAGES: [&str; 2] = ["xtask", "mesh-llm-release-footer"];

#[derive(Deserialize)]
pub(super) struct ResolvedMetadata {
    packages: Vec<Package>,
    resolve: Option<Resolve>,
    pub(super) target_directory: PathBuf,
}

#[derive(Deserialize)]
struct Package {
    id: String,
    name: String,
    source: Option<String>,
    links: Option<String>,
}

#[derive(Deserialize)]
struct Resolve {
    nodes: Vec<Node>,
}

#[derive(Deserialize)]
struct Node {
    id: String,
    deps: Vec<NodeDep>,
}

#[derive(Deserialize)]
struct NodeDep {
    pkg: String,
    dep_kinds: Vec<DepKind>,
}

#[derive(Deserialize)]
struct DepKind {
    kind: Option<String>,
}

impl NodeDep {
    /// Normal and build-script edges are compiled by `cargo build`; dev-only
    /// edges are not part of the bootstrap binary.
    fn compiled_for_binary(&self) -> bool {
        self.dep_kinds
            .iter()
            .any(|dep_kind| dep_kind.kind.as_deref() != Some("dev"))
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum BoundaryViolation {
    WorkspaceCrate { package: String },
    NativeLink { package: String, links: String },
}

impl fmt::Display for BoundaryViolation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WorkspaceCrate { package } => {
                write!(formatter, "product/native/UI workspace crate `{package}`")
            }
            Self::NativeLink { package, links } => {
                write!(formatter, "native library link `{links}` via `{package}`")
            }
        }
    }
}

/// Returns every boundary violation in the compiled dependency closure of
/// `tool_package`, sorted for stable diagnostics.
pub(super) fn violations(
    metadata: &ResolvedMetadata,
    tool_package: &str,
) -> DynResult<Vec<BoundaryViolation>> {
    let packages = metadata
        .packages
        .iter()
        .map(|package| (package.id.as_str(), package))
        .collect::<BTreeMap<_, _>>();
    let root = metadata
        .packages
        .iter()
        .find(|package| package.name == tool_package && package.source.is_none())
        .ok_or_else(|| format!("cargo metadata has no workspace package `{tool_package}`"))?;
    let nodes = metadata
        .resolve
        .as_ref()
        .ok_or("cargo metadata has no dependency resolution")?
        .nodes
        .iter()
        .map(|node| (node.id.as_str(), node))
        .collect::<BTreeMap<_, _>>();

    let mut reached = BTreeSet::from([root.id.as_str()]);
    let mut pending = vec![root.id.as_str()];
    while let Some(id) = pending.pop() {
        let node = nodes
            .get(id)
            .ok_or_else(|| format!("cargo metadata has no resolve node for `{id}`"))?;
        for dep in node.deps.iter().filter(|dep| dep.compiled_for_binary()) {
            if reached.insert(dep.pkg.as_str()) {
                pending.push(dep.pkg.as_str());
            }
        }
    }

    let mut violations = BTreeSet::new();
    for id in reached {
        let package = packages
            .get(id)
            .ok_or_else(|| format!("cargo metadata has no package for `{id}`"))?;
        if package.source.is_none() && !BOOTSTRAP_LOCAL_PACKAGES.contains(&package.name.as_str()) {
            violations.insert(BoundaryViolation::WorkspaceCrate {
                package: package.name.clone(),
            });
        }
        if let Some(links) = &package.links {
            violations.insert(BoundaryViolation::NativeLink {
                package: package.name.clone(),
                links: links.clone(),
            });
        }
    }
    Ok(violations.into_iter().collect())
}

pub(super) fn check(metadata: &ResolvedMetadata, tool_package: &str) -> DynResult<()> {
    let violations = violations(metadata, tool_package)?;
    if violations.is_empty() {
        return Ok(());
    }
    let listed = violations
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("; ");
    Err(format!(
        "automation bootstrap: `{tool_package}` must build without product, native, or UI crates, but its dependency closure contains: {listed}"
    )
    .into())
}
