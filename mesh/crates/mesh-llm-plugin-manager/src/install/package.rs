//! The package-validation boundary before an install replaces executable files.

use anyhow::{Result, bail};

use super::ResolvedInstallSource;
use crate::{
    archive::{ExtractedPluginArchive, stage_plugin_archive},
    compatibility::{self as compat, HostCompatibility},
    github::GitHubReleaseAsset,
    install::{InstallOutcome, PluginInstallOptions},
    plugin_toml::PluginToml,
    store::InstalledPluginMetadata,
    target::{ArchiveExt, PluginTarget},
};

/// Every install checks the archive's effective host bounds before committing
/// it, so catalog metadata cannot hide an incompatible packaged release.
pub(super) fn extract_for_install(
    archive_path: &std::path::Path,
    archive_ext: ArchiveExt,
    name: &str,
    options: &PluginInstallOptions,
    catalog_compatibility: Option<&HostCompatibility>,
    host_version: &str,
) -> Result<ExtractedPluginArchive> {
    let staged = stage_plugin_archive(
        archive_path,
        archive_ext,
        options.target.executable_suffix(),
        name,
        &options.install_root,
    )?;
    let compatibility = compatibility_from_package(staged.plugin_toml(), catalog_compatibility);
    if let compat::Verdict::Incompatible(reason) =
        compat::evaluate(compatibility.as_ref(), host_version)
    {
        bail!("plugin '{name}' does not support MeshLLM {host_version}: {reason}");
    }
    staged.commit()
}

/// A same-tag fast path cannot turn a known incompatible install into success.
pub(super) fn unchanged_install(
    current: Option<&InstalledPluginMetadata>,
    release_tag: &str,
    host_version: &str,
) -> Result<Option<InstallOutcome>> {
    let Some(current) = current.filter(|current| current.installed_version == release_tag) else {
        return Ok(None);
    };
    if let compat::Verdict::Incompatible(reason) =
        compat::evaluate(current.host_compatibility.as_ref(), host_version)
    {
        bail!(
            "plugin '{}' is unchanged and incompatible: {reason}",
            current.name
        );
    }
    Ok(Some(InstallOutcome {
        metadata: current.clone(),
        changed: false,
    }))
}

pub(super) fn build_installed_metadata(
    resolved: &ResolvedInstallSource,
    release_tag: &str,
    asset: &GitHubReleaseAsset,
    target: &PluginTarget,
    extracted: ExtractedPluginArchive,
    current: Option<&InstalledPluginMetadata>,
    verified_sha256: Option<String>,
) -> InstalledPluginMetadata {
    InstalledPluginMetadata {
        name: resolved.plugin_name.clone(),
        source_repository: resolved.source.url(),
        installed_version: release_tag.to_string(),
        target_triple: target.triple().to_string(),
        downloaded_asset_name: asset.name.clone(),
        install_path: extracted.install_path,
        enabled: current.map(|metadata| metadata.enabled).unwrap_or(true),
        default_managed: false,
        host_compatibility: compatibility_from_package(
            &extracted.plugin_toml,
            resolved.compatibility.as_ref(),
        ),
        allowed_surfaces: retained_surfaces(current),
        manifest: extracted.manifest,
        // Negotiation and status describe the replaced executable, not this one.
        last_protocol_version: None,
        last_status: None,
        last_error: None,
        verified_sha256,
    }
}

/// Replacing a release never widens the operator's retained surface grants.
pub(super) fn retained_surfaces(current: Option<&InstalledPluginMetadata>) -> Option<Vec<String>> {
    current.and_then(|metadata| match &metadata.allowed_surfaces {
        Some(allowed) => Some(allowed.clone()),
        None if metadata.default_managed => {
            Some(crate::legacy_defaults::migrated_allows_keys(&metadata.name))
        }
        None => None,
    })
}

/// Package declarations take precedence over catalog declarations, just as
/// they do in the installed record used by startup and later host upgrades.
pub(super) fn compatibility_from_package(
    plugin_toml: &PluginToml,
    catalog: Option<&HostCompatibility>,
) -> Option<HostCompatibility> {
    if plugin_toml.min_host_version.is_some() || plugin_toml.max_host_version.is_some() {
        return Some(HostCompatibility {
            min: plugin_toml.min_host_version.clone(),
            max: plugin_toml.max_host_version.clone(),
        });
    }
    catalog.cloned()
}

#[cfg(test)]
#[path = "package_tests.rs"]
mod tests;
