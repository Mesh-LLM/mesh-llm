//! Workspace packages, directly changed crates and the affected-crate scope.
//! Cargo discovery and the reverse-dependency closure always run in the
//! protected checkout (`root`), never in the manifest root.

use crate::ci_plan::catalog::stripped;
use crate::ci_plan::diagnostics::{
    PlanError, PlanResult, fail, nonempty_string, repr, repr_list, string_list,
};
use crate::ci_plan::document::Json;
use crate::ci_plan::request::Profile;
use serde_json::Value;
use std::collections::BTreeSet;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

pub(super) struct Package {
    pub(super) name: String,
    pub(super) path: String,
}

/// Input packages when given, otherwise `cargo metadata --no-deps`.
pub(super) fn packages(root: &Path, raw: Option<&Json>) -> PlanResult<Vec<Package>> {
    match raw {
        Some(raw) => declared_packages(raw),
        None => metadata_packages(root),
    }
}

fn declared_packages(raw: &Json) -> PlanResult<Vec<Package>> {
    let Some(items) = raw.as_array() else {
        return fail("workspace_packages must be an array");
    };
    let mut packages: Vec<Package> = Vec::with_capacity(items.len());
    for (index, item) in items.iter().enumerate() {
        if item.as_object().is_none() {
            return fail(format!("workspace_packages[{index}] must be an object"));
        }
        let name = nonempty_string(
            item.get("name"),
            &format!("workspace_packages[{index}].name"),
        )?;
        let path = nonempty_string(
            item.get("path"),
            &format!("workspace_packages[{index}].path"),
        )?;
        if packages.iter().any(|package| package.name == name) {
            return fail(format!(
                "workspace_packages contains duplicate {}",
                repr(&name)
            ));
        }
        let path = path.trim_end_matches('/').to_owned();
        packages.push(Package { name, path });
    }
    Ok(packages)
}

/// `Path.resolve()`: the canonical path when it exists, else the lexical one.
fn resolved(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

fn metadata_packages(root: &Path) -> PlanResult<Vec<Package>> {
    let output = Command::new("cargo")
        .args(["metadata", "--format-version=1", "--no-deps"])
        .current_dir(root)
        .stdin(Stdio::null())
        .output()
        .map_err(|error| PlanError(format!("cargo metadata failed: {error}")))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return fail(format!("cargo metadata failed: {}", stripped(&stderr)));
    }
    let metadata: Value = serde_json::from_slice(&output.stdout)
        .map_err(|error| PlanError(format!("cargo metadata emitted invalid JSON: {error}")))?;
    let members = metadata["workspace_members"]
        .as_array()
        .map(|ids| {
            ids.iter()
                .filter_map(Value::as_str)
                .collect::<BTreeSet<_>>()
        })
        .unwrap_or_default();
    let workspace_root = metadata["workspace_root"]
        .as_str()
        .map_or_else(|| resolved(root), |text| resolved(Path::new(text)));
    let mut packages = Vec::new();
    for package in metadata["packages"].as_array().into_iter().flatten() {
        if !package["id"]
            .as_str()
            .is_some_and(|id| members.contains(id))
        {
            continue;
        }
        let (Some(name), Some(manifest)) =
            (package["name"].as_str(), package["manifest_path"].as_str())
        else {
            return fail("cargo metadata emitted a malformed package");
        };
        let manifest = resolved(Path::new(manifest));
        let directory = manifest.parent().unwrap_or(Path::new(""));
        let relative = directory.strip_prefix(&workspace_root).map_err(|_| {
            PlanError(format!(
                "{} is not in the subpath of {}",
                directory.display(),
                workspace_root.display()
            ))
        })?;
        let path = relative.to_string_lossy().into_owned();
        packages.push(Package {
            name: name.to_owned(),
            path: if path.is_empty() {
                ".".to_owned()
            } else {
                path
            },
        });
    }
    if packages.is_empty() {
        return fail("cargo metadata returned no workspace packages");
    }
    Ok(packages)
}

/// Packages owning at least one changed path, in workspace order.
pub(super) fn direct_crates(changed_files: &[String], packages: &[Package]) -> Vec<String> {
    packages
        .iter()
        .filter(|package| package.path != ".")
        .filter(|package| {
            let prefix = format!("{}/", package.path.trim_end_matches('/'));
            changed_files
                .iter()
                .any(|path| *path == package.path || path.starts_with(&prefix))
        })
        .map(|package| package.name.clone())
        .collect()
}

/// What the affected-crate scope is computed from.
pub(super) struct Scope<'a> {
    pub(super) root: &'a Path,
    pub(super) changed_files: &'a [String],
    pub(super) packages: &'a [Package],
    pub(super) profile: Profile,
}

/// Every crate for exhaustive profiles, a non-empty input list as given, or
/// else the reverse-dependency closure; always reported in workspace order.
pub(super) fn affected_crates(scope: &Scope<'_>, raw: Option<&Json>) -> PlanResult<Vec<String>> {
    let workspace = scope
        .packages
        .iter()
        .map(|package| package.name.as_str())
        .collect::<Vec<_>>();
    let affected = if scope.profile.exhaustive() {
        workspace.iter().map(|name| (*name).to_owned()).collect()
    } else {
        match raw {
            Some(raw) if *raw != Json::Array(Vec::new()) => {
                string_list(Some(raw), "affected_crates")?
            }
            _ => reverse_dependency_closure(scope.root, scope.changed_files)?,
        }
    };
    let known = workspace.iter().copied().collect::<BTreeSet<_>>();
    let unknown = affected
        .iter()
        .filter(|name| !known.contains(name.as_str()))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    if !unknown.is_empty() {
        return fail(format!(
            "affected_crates contains non-workspace crates: {}",
            repr_list(&unknown)
        ));
    }
    let selected = affected.iter().map(String::as_str).collect::<BTreeSet<_>>();
    Ok(workspace
        .into_iter()
        .filter(|name| selected.contains(name))
        .map(str::to_owned)
        .collect())
}

/// `repository affected-crates --stdin` (the Rust owner of
/// `scripts/affected-crates.sh`), run from the protected checkout.
fn reverse_dependency_closure(root: &Path, changed_files: &[String]) -> PlanResult<Vec<String>> {
    let failure = |detail: String| PlanError(format!("affected-crates.sh failed: {detail}"));
    let program = std::env::current_exe().map_err(|error| failure(error.to_string()))?;
    let mut child = Command::new(program)
        .args(["repository", "affected-crates", "--stdin"])
        .current_dir(root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| failure(error.to_string()))?;
    let input = format!("{}\n", changed_files.join("\n"));
    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(input.as_bytes())
            .map_err(|error| failure(error.to_string()))?;
    }
    let output = child
        .wait_with_output()
        .map_err(|error| failure(error.to_string()))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(failure(stripped(&stderr).to_owned()));
    }
    let document = Json::parse(&output.stdout)
        .map_err(|error| PlanError(format!("affected-crates.sh emitted invalid JSON: {error}")))?;
    if document.as_object().is_none() {
        return fail("affected-crates.sh emitted invalid JSON: not an object");
    }
    string_list(document.get("affected"), "affected_crates")
}
