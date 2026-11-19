//! Manifest edges are checked without Cargo or source-language parsing.
use super::{Inventory, PACKAGE_TOOL, XTASK};
use crate::command::DynResult;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

pub(super) fn check(root: &Path, inventory: Inventory) -> DynResult<()> {
    let packages: Vec<&str> = match inventory {
        Inventory::HostedXtask => XTASK.to_vec(),
        Inventory::HostedController => vec!["tools/ci-controller"],
        Inventory::LinuxRuntime => XTASK.iter().chain(PACKAGE_TOOL).copied().collect(),
        Inventory::WindowsAbi => XTASK.to_vec(),
    };
    let admitted: BTreeSet<_> = packages
        .iter()
        .map(|p| root.join(p).canonicalize())
        .collect::<Result<_, _>>()?;
    let workspace: toml::Value =
        toml::from_str(&std::fs::read_to_string(root.join("Cargo.toml"))?)?;
    for (registry, patches) in workspace
        .get("patch")
        .and_then(toml::Value::as_table)
        .into_iter()
        .flatten()
    {
        for (name, patch) in patches.as_table().into_iter().flatten() {
            if let Some(path) = patch.get("path").and_then(toml::Value::as_str) {
                return Err(format!(
                    "root [patch.{registry}] local override {name} ({path}) is unsupported: cache closure cannot prove registry/transitive selection or patched feature closure without Cargo metadata"
                )
                .into());
            }
        }
    }
    for package in packages {
        let directory = root.join(package);
        let manifest: toml::Value =
            toml::from_str(&std::fs::read_to_string(directory.join("Cargo.toml"))?)?;
        let build = manifest.get("package").and_then(|p| p.get("build"));
        let build_path = build.and_then(toml::Value::as_str).unwrap_or("build.rs");
        if build.and_then(toml::Value::as_bool) != Some(false)
            && directory.join(build_path).is_file()
            && !inventory
                .inputs()
                .contains(&format!("{package}/{build_path}"))
        {
            return Err(
                format!("{package}: cache inventory omits build script {build_path}").into(),
            );
        }
        check_tables(&directory, &manifest, &workspace, &admitted)?;
    }
    Ok(())
}

fn check_tables(
    directory: &Path,
    manifest: &toml::Value,
    workspace: &toml::Value,
    admitted: &BTreeSet<PathBuf>,
) -> DynResult<()> {
    check_feature(manifest, "default", workspace, &mut BTreeSet::new())?;
    for (name, dependency) in production_dependencies(manifest) {
        let inherited = dependency.get("workspace").and_then(toml::Value::as_bool) == Some(true);
        let source = if inherited {
            workspace
                .get("workspace")
                .and_then(|v| v.get("dependencies"))
                .and_then(|v| v.get(name))
                .ok_or_else(|| format!("missing workspace dependency {name}"))?
        } else {
            dependency
        };
        // Activated optional local dependencies were refused using the full manifest above.
        if dependency.get("optional").and_then(toml::Value::as_bool) == Some(true) {
            continue;
        }
        if let Some(path) = source.get("path").and_then(toml::Value::as_str) {
            let base = if inherited {
                workspace_root(directory)?
            } else {
                directory.to_path_buf()
            };
            let resolved = base.join(path).canonicalize()?;
            if !admitted.contains(&resolved) {
                return Err(format!(
                    "{}: cache inventory omits dependency {name} ({})",
                    directory.display(),
                    resolved.display()
                )
                .into());
            }
            for features in dependency
                .get("features")
                .and_then(toml::Value::as_array)
                .into_iter()
                .chain(
                    inherited
                        .then(|| source.get("features").and_then(toml::Value::as_array))
                        .flatten(),
                )
            {
                let child: toml::Value =
                    toml::from_str(&std::fs::read_to_string(resolved.join("Cargo.toml"))?)?;
                for feature in features.iter().filter_map(toml::Value::as_str) {
                    check_feature(&child, feature, workspace, &mut BTreeSet::new())?;
                }
            }
        }
    }
    Ok(())
}

fn production_dependencies(
    manifest: &toml::Value,
) -> impl Iterator<Item = (&String, &toml::Value)> {
    std::iter::once(manifest)
        .chain(
            manifest
                .get("target")
                .and_then(toml::Value::as_table)
                .into_iter()
                .flat_map(|targets| targets.values()),
        )
        .flat_map(|table| {
            ["dependencies", "build-dependencies"]
                .into_iter()
                .filter_map(move |kind| table.get(kind).and_then(toml::Value::as_table))
                .flat_map(|dependencies| dependencies.iter())
        })
}

fn workspace_root(directory: &Path) -> DynResult<PathBuf> {
    for parent in directory.ancestors().skip(1) {
        let path = parent.join("Cargo.toml");
        if path.is_file() {
            let value: toml::Value = toml::from_str(&std::fs::read_to_string(path)?)?;
            if value.get("workspace").is_some() {
                return Ok(parent.to_path_buf());
            }
        }
    }
    Err("workspace root missing".into())
}

fn check_feature(
    manifest: &toml::Value,
    feature: &str,
    workspace: &toml::Value,
    visited: &mut BTreeSet<String>,
) -> DynResult<()> {
    if !visited.insert(feature.into()) {
        return Ok(());
    }
    let values = manifest
        .get("features")
        .and_then(|v| v.get(feature))
        .and_then(toml::Value::as_array);
    // Optional dependencies also define implicit features without a [features] entry.
    for value in values
        .into_iter()
        .flatten()
        .filter_map(toml::Value::as_str)
        .chain(values.is_none().then_some(feature))
    {
        let dependency_name = value
            .strip_prefix("dep:")
            .unwrap_or(value)
            .split('/')
            .next()
            .unwrap_or(value)
            .trim_end_matches('?');
        for (name, dependency) in
            production_dependencies(manifest).filter(|(name, _)| name.as_str() == dependency_name)
        {
            let source = if dependency.get("workspace").and_then(toml::Value::as_bool) == Some(true)
            {
                workspace
                    .get("workspace")
                    .and_then(|v| v.get("dependencies"))
                    .and_then(|v| v.get(name))
                    .ok_or_else(|| format!("missing workspace dependency {name}"))?
            } else {
                dependency
            };
            if source.get("path").is_some() {
                if value.contains('/') {
                    return Err(format!("feature {feature} forwards local dependency feature {value}; declare its cache closure").into());
                }
                if dependency.get("optional").and_then(toml::Value::as_bool) == Some(true) {
                    return Err(format!("feature {feature} activates optional local dependency {name}; declare its cache closure").into());
                }
            }
        }
        check_feature(manifest, value, workspace, visited)?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "cache_inputs_closure_tests.rs"]
mod tests;
