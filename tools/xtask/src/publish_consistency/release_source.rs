//! Publication checks for immutable source data, independent of controller policy.
use super::{
    CargoMetadata, DynResult, check_publish_crate_dependencies, check_publish_crate_metadata,
    check_publish_literal_includes, publish_order, workspace_packages_by_dir,
    workspace_packages_by_name,
};
use std::collections::BTreeSet;
use std::path::Path;

pub(crate) fn check(root: &Path, input: &[u8], roster: &[String]) -> DynResult<()> {
    let metadata: CargoMetadata = serde_json::from_slice(input)?;
    let root = root.canonicalize()?;
    let members: BTreeSet<_> = metadata.workspace_members.iter().cloned().collect();
    let packages = workspace_packages_by_name(&metadata, &members);
    let directories = workspace_packages_by_dir(&metadata, &members)?;
    let order = publish_order(roster)?;

    for name in roster {
        let package = packages
            .get(name)
            .ok_or_else(|| format!("missing selected package {name}"))?;
        let manifest = package.manifest_path.canonicalize()?;
        if !manifest.starts_with(&root)
            || manifest.file_name().is_none_or(|name| name != "Cargo.toml")
            || !manifest.is_file()
        {
            return Err(format!("{name}: manifest is outside the selected source").into());
        }
    }
    check_publish_crate_metadata(&root, roster, &packages)?;
    check_publish_crate_dependencies(&order, &packages, &directories)?;
    check_publish_literal_includes(roster, &packages)?;
    check_selected_catalogs(&root, &packages)?;
    Ok(())
}

fn catalog_path(
    root: &Path,
    package: &super::CargoPackage,
    leaf: &str,
) -> DynResult<std::path::PathBuf> {
    let manifest = package.manifest_path.canonicalize()?;
    if !manifest.starts_with(root)
        || manifest.file_name().is_none_or(|name| name != "Cargo.toml")
        || !manifest.is_file()
    {
        return Err("catalog package manifest is outside selected source".into());
    }
    let directory = manifest
        .parent()
        .ok_or("catalog package directory missing")?;
    let path = directory.join(leaf).canonicalize()?;
    if !path.starts_with(directory) || !path.is_file() {
        return Err("catalog is outside its admitted package directory".into());
    }
    Ok(path)
}

fn check_selected_catalogs(
    root: &Path,
    packages: &std::collections::BTreeMap<String, &super::CargoPackage>,
) -> DynResult<()> {
    let client = packages
        .get("mesh-llm-client")
        .ok_or("missing admitted catalog client package")?;
    let node = packages
        .get("mesh-llm-node")
        .ok_or("missing admitted catalog node package")?;
    let client = std::fs::read_to_string(catalog_path(root, client, "src/models/catalog.json")?)?;
    let node = std::fs::read_to_string(catalog_path(root, node, "src/catalog.json")?)?;
    crate::command::ensure_eq(
        &client,
        &node,
        "selected mesh-llm-node packaged catalog copy",
    )
}

#[cfg(test)]
#[path = "release_source_tests.rs"]
mod tests;
