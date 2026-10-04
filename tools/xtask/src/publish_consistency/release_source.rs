//! Publication checks for immutable source data, independent of controller policy.
use super::{
    CargoMetadata, DynResult, check_publish_catalog_sync, check_publish_crate_dependencies,
    check_publish_crate_metadata, check_publish_literal_includes, publish_order,
    workspace_packages_by_dir, workspace_packages_by_name,
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
    check_publish_catalog_sync(&root)?;
    Ok(())
}
