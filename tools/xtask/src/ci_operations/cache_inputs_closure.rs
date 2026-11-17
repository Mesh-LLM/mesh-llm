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
    for kind in ["dependencies", "build-dependencies"] {
        if let Some(table) = manifest.get(kind).and_then(toml::Value::as_table) {
            for (name, dependency) in table {
                let inherited =
                    dependency.get("workspace").and_then(toml::Value::as_bool) == Some(true);
                let source = if inherited {
                    workspace
                        .get("workspace")
                        .and_then(|v| v.get("dependencies"))
                        .and_then(|v| v.get(name))
                        .ok_or_else(|| format!("missing workspace dependency {name}"))?
                } else {
                    dependency
                };
                // Optional dependencies are feature-controlled. Refuse newly activated local
                // features below rather than treating every optional tool as compiled.
                if source.get("path").is_some()
                    && dependency.get("optional").and_then(toml::Value::as_bool) == Some(true)
                {
                    check_optional(manifest, name)?;
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
                    if let Some(features) =
                        dependency.get("features").and_then(toml::Value::as_array)
                    {
                        let child: toml::Value =
                            toml::from_str(&std::fs::read_to_string(resolved.join("Cargo.toml"))?)?;
                        for feature in features.iter().filter_map(toml::Value::as_str) {
                            check_feature(&child, feature, &mut BTreeSet::new())?;
                        }
                    }
                }
            }
        }
    }
    if let Some(targets) = manifest.get("target").and_then(toml::Value::as_table) {
        for target in targets.values() {
            check_tables(directory, target, workspace, admitted)?;
        }
    }
    Ok(())
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

fn check_optional(manifest: &toml::Value, name: &str) -> DynResult<()> {
    let Some(features) = manifest.get("features").and_then(toml::Value::as_table) else {
        return Ok(());
    };
    let mut pending = vec!["default"];
    let mut visited = BTreeSet::new();
    while let Some(feature) = pending.pop() {
        if !visited.insert(feature) {
            continue;
        }
        if let Some(values) = features.get(feature).and_then(toml::Value::as_array) {
            for value in values.iter().filter_map(toml::Value::as_str) {
                if value == name
                    || value == format!("dep:{name}")
                    || value.starts_with(&format!("{name}/"))
                {
                    return Err(format!("default feature activates optional dependency {name}; declare its cache closure").into());
                }
                pending.push(value);
            }
        }
    }
    Ok(())
}

fn check_feature(
    manifest: &toml::Value,
    feature: &str,
    visited: &mut BTreeSet<String>,
) -> DynResult<()> {
    if !visited.insert(feature.into()) {
        return Ok(());
    }
    let Some(values) = manifest
        .get("features")
        .and_then(|v| v.get(feature))
        .and_then(toml::Value::as_array)
    else {
        return Ok(());
    };
    for value in values.iter().filter_map(toml::Value::as_str) {
        let dependency = value
            .strip_prefix("dep:")
            .unwrap_or(value)
            .split('/')
            .next()
            .unwrap_or(value);
        if manifest
            .get("dependencies")
            .and_then(|v| v.get(dependency))
            .is_some_and(|dep| {
                dep.get("path").is_some()
                    && dep.get("optional").and_then(toml::Value::as_bool) == Some(true)
            })
        {
            return Err(format!("feature {feature} activates optional local dependency {dependency}; declare its cache closure").into());
        }
        check_feature(manifest, value, visited)?;
    }
    Ok(())
}
