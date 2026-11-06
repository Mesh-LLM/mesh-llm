use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

use crate::{
    source_ref::{PluginVersion, is_valid_name},
    target::PluginTarget,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PluginAsset {
    pub name: String,
    pub kind: AssetMatchKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AssetMatchKind {
    Versioned,
    StableAlias,
}

pub fn select_plugin_asset(
    plugin_name: &str,
    version: Option<&PluginVersion>,
    target: &PluginTarget,
    assets: &[String],
) -> Result<PluginAsset> {
    if !is_valid_name(plugin_name) {
        bail!("invalid plugin name for release asset matching: {plugin_name}");
    }

    for candidate in asset_candidates(plugin_name, version, target) {
        if assets.iter().any(|asset| asset == &candidate.name) {
            return Ok(candidate);
        }
    }

    bail!(
        "no compatible release asset for plugin '{}' and target {}",
        plugin_name,
        target.triple()
    )
}

/// Infers the plugin name from a release whose assets are not named after the
/// repository, e.g. `Mesh-LLM/ln-payments-plugin` shipping
/// `pay-lightning-v0.1.0-<target>.tar.gz`. Only versioned assets for this
/// exact release and target count, and exactly one distinct valid name must
/// remain; anything else is ambiguous and returns `None`. The packaged
/// manifest still has to declare the inferred name at extraction time.
pub fn infer_plugin_asset(
    version: &PluginVersion,
    target: &PluginTarget,
    assets: &[String],
) -> Option<(String, PluginAsset)> {
    let mut found: Option<(String, PluginAsset)> = None;
    for segment in version.matching_segments() {
        let suffix = format!("-{}-{}.{}", segment, target.triple(), target.archive_ext());
        for asset in assets {
            let Some(name) = asset.strip_suffix(&suffix) else {
                continue;
            };
            if !is_valid_name(name) {
                continue;
            }
            match &found {
                Some((existing, _)) if existing == name => {}
                Some(_) => return None,
                None => {
                    found = Some((
                        name.to_string(),
                        PluginAsset {
                            name: asset.clone(),
                            kind: AssetMatchKind::Versioned,
                        },
                    ))
                }
            }
        }
    }
    found
}

pub fn asset_candidates(
    plugin_name: &str,
    version: Option<&PluginVersion>,
    target: &PluginTarget,
) -> Vec<PluginAsset> {
    let mut candidates = Vec::new();
    if let Some(version) = version {
        for segment in version.matching_segments() {
            push_unique(
                &mut candidates,
                PluginAsset {
                    name: format!(
                        "{}-{}-{}.{}",
                        plugin_name,
                        segment,
                        target.triple(),
                        target.archive_ext()
                    ),
                    kind: AssetMatchKind::Versioned,
                },
            );
        }
    }

    push_unique(
        &mut candidates,
        PluginAsset {
            name: format!(
                "{}-{}.{}",
                plugin_name,
                target.triple(),
                target.archive_ext()
            ),
            kind: AssetMatchKind::StableAlias,
        },
    );
    candidates
}

fn push_unique(candidates: &mut Vec<PluginAsset>, candidate: PluginAsset) {
    if candidates
        .iter()
        .all(|existing| existing.name != candidate.name)
    {
        candidates.push(candidate);
    }
}

#[cfg(test)]
mod tests {
    use crate::{PluginVersion, target::PluginTarget};

    use super::*;

    #[test]
    fn prefers_exact_versioned_asset() {
        let target = PluginTarget::from_os_arch("macos", "aarch64").unwrap();
        let version = PluginVersion::new("v1.1.0").unwrap();
        let assets = vec![
            "cool-plugin-aarch64-apple-darwin.tar.gz".to_string(),
            "cool-plugin-v1.1.0-aarch64-apple-darwin.tar.gz".to_string(),
        ];
        let selected =
            select_plugin_asset("cool-plugin", Some(&version), &target, &assets).unwrap();
        assert_eq!(selected.kind, AssetMatchKind::Versioned);
        assert_eq!(
            selected.name,
            "cool-plugin-v1.1.0-aarch64-apple-darwin.tar.gz"
        );
    }

    #[test]
    fn accepts_version_without_v_when_requested_version_has_v() {
        let target = PluginTarget::from_os_arch("linux", "x86_64").unwrap();
        let version = PluginVersion::new("v1.1.0").unwrap();
        let assets = vec!["cool-plugin-1.1.0-x86_64-unknown-linux-gnu.tar.gz".to_string()];
        let selected =
            select_plugin_asset("cool-plugin", Some(&version), &target, &assets).unwrap();
        assert_eq!(selected.kind, AssetMatchKind::Versioned);
        assert_eq!(
            selected.name,
            "cool-plugin-1.1.0-x86_64-unknown-linux-gnu.tar.gz"
        );
    }

    #[test]
    fn falls_back_to_stable_alias() {
        let target = PluginTarget::from_os_arch("windows", "x86_64").unwrap();
        let version = PluginVersion::new("1.1.0").unwrap();
        let assets = vec!["cool-plugin-x86_64-pc-windows-msvc.zip".to_string()];
        let selected =
            select_plugin_asset("cool-plugin", Some(&version), &target, &assets).unwrap();
        assert_eq!(selected.kind, AssetMatchKind::StableAlias);
    }

    #[test]
    fn infers_plugin_name_when_assets_differ_from_repo_name() {
        let target = PluginTarget::from_os_arch("windows", "x86_64").unwrap();
        let version = PluginVersion::new("v0.1.1").unwrap();
        let assets = vec![
            "pay-lightning-v0.1.1-aarch64-apple-darwin.tar.gz".to_string(),
            "pay-lightning-v0.1.1-x86_64-pc-windows-msvc.zip".to_string(),
            "pay-lightning-v0.1.1-x86_64-pc-windows-msvc.zip.sha256".to_string(),
        ];
        assert!(
            select_plugin_asset("ln-payments-plugin", Some(&version), &target, &assets).is_err()
        );
        let (name, asset) = infer_plugin_asset(&version, &target, &assets).unwrap();
        assert_eq!(name, "pay-lightning");
        assert_eq!(
            asset.name,
            "pay-lightning-v0.1.1-x86_64-pc-windows-msvc.zip"
        );
    }

    #[test]
    fn inference_refuses_ambiguous_or_other_version_assets() {
        let target = PluginTarget::from_os_arch("linux", "x86_64").unwrap();
        let version = PluginVersion::new("v0.1.1").unwrap();
        let two = vec![
            "a-v0.1.1-x86_64-unknown-linux-gnu.tar.gz".to_string(),
            "b-v0.1.1-x86_64-unknown-linux-gnu.tar.gz".to_string(),
        ];
        assert!(infer_plugin_asset(&version, &target, &two).is_none());
        let stale = vec!["a-v0.1.0-x86_64-unknown-linux-gnu.tar.gz".to_string()];
        assert!(infer_plugin_asset(&version, &target, &stale).is_none());
    }
}
