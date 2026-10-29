//! Requirements between installed plugins, declared by `requires` in a
//! plugin's `plugin.toml` and recorded on its install metadata.
//!
//! Install fetches a missing requirement from the catalog; delete refuses to
//! orphan one; a node start and `plugins list` report a requirement that is
//! missing or disabled. The host never starts, orders or links plugins for
//! each other: how two plugins talk stays an operator-authorized binding.

use std::collections::BTreeSet;

use anyhow::{Result, bail};

use crate::store::{InstalledPluginMetadata, PluginStore};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissingRequirement {
    /// The installed, enabled plugin that declared the requirement.
    pub plugin: String,
    pub required: String,
    pub reason: MissingRequirementReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissingRequirementReason {
    NotInstalled,
    Disabled,
}

impl MissingRequirement {
    pub fn describe(&self) -> String {
        match self.reason {
            MissingRequirementReason::NotInstalled => format!(
                "{} requires {}, which is not installed; run `mesh-llm plugins install {}`",
                self.plugin, self.required, self.required
            ),
            MissingRequirementReason::Disabled => format!(
                "{} requires {}, which is disabled; run `mesh-llm plugins enable {}`",
                self.plugin, self.required, self.required
            ),
        }
    }
}

/// Requirements of the enabled installed plugins that are not met by another
/// enabled installed plugin. A disabled plugin's own requirements are not
/// checked: nothing of it runs.
pub fn missing_requirements(installed: &[InstalledPluginMetadata]) -> Vec<MissingRequirement> {
    let mut missing = Vec::new();
    for plugin in installed.iter().filter(|plugin| plugin.enabled) {
        for required in &plugin.requires {
            let reason = match installed.iter().find(|other| &other.name == required) {
                None => MissingRequirementReason::NotInstalled,
                Some(other) if !other.enabled => MissingRequirementReason::Disabled,
                Some(_) => continue,
            };
            missing.push(MissingRequirement {
                plugin: plugin.name.clone(),
                required: required.clone(),
                reason,
            });
        }
    }
    missing
}

/// Names of the installed plugins (enabled or not) whose `requires` names
/// `name`, in store order.
pub fn required_by(installed: &[InstalledPluginMetadata], name: &str) -> Vec<String> {
    installed
        .iter()
        .filter(|plugin| plugin.requires.iter().any(|required| required == name))
        .map(|plugin| plugin.name.clone())
        .collect()
}

/// Refuse to delete or disable `name` while an installed plugin requires it,
/// so an install never silently loses something it declared it needs.
pub fn ensure_not_required(store: &PluginStore, name: &str, action: &str) -> Result<()> {
    let dependents = required_by(&store.list()?, name);
    if dependents.is_empty() {
        return Ok(());
    }
    bail!(
        "cannot {action} {name}: required by {}; {action} {} first",
        dependents.join(", "),
        if dependents.len() == 1 { "it" } else { "them" }
    )
}

/// The requirements declared by `plugin` that no installed plugin provides,
/// each listed once, in declaration order. Install resolves these from the
/// catalog; a requirement that is installed but disabled is left alone (the
/// operator turned it off) and surfaces through `missing_requirements`.
pub fn uninstalled_requirements(
    store: &PluginStore,
    plugin: &InstalledPluginMetadata,
) -> Result<Vec<String>> {
    let installed: BTreeSet<String> = store
        .list()?
        .into_iter()
        .map(|installed| installed.name)
        .collect();
    let mut pending = Vec::new();
    for required in &plugin.requires {
        if !installed.contains(required) && !pending.contains(required) {
            pending.push(required.clone());
        }
    }
    Ok(pending)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    fn plugin(name: &str, enabled: bool, requires: &[&str]) -> InstalledPluginMetadata {
        InstalledPluginMetadata {
            name: name.to_string(),
            source_repository: format!("https://github.com/Mesh-LLM/{name}"),
            installed_version: "v0.1.0".to_string(),
            target_triple: "aarch64-apple-darwin".to_string(),
            downloaded_asset_name: format!("{name}-0.1.0-aarch64-apple-darwin.tar.gz"),
            install_path: PathBuf::from(format!("/plugins/installed/{name}")),
            enabled,
            default_managed: false,
            requires: requires.iter().map(|value| value.to_string()).collect(),
            manifest: None,
            last_protocol_version: None,
            last_status: None,
            last_error: None,
        }
    }

    #[test]
    fn a_met_requirement_is_not_reported() {
        let installed = [
            plugin("lexe-wallet", true, &["pay-lightning"]),
            plugin("pay-lightning", true, &[]),
        ];
        assert!(missing_requirements(&installed).is_empty());
    }

    #[test]
    fn missing_and_disabled_requirements_are_reported_with_their_reason() {
        let installed = [
            plugin("lexe-wallet", true, &["pay-lightning", "capsules"]),
            plugin("capsules", false, &[]),
        ];
        let missing = missing_requirements(&installed);
        assert_eq!(
            missing,
            vec![
                MissingRequirement {
                    plugin: "lexe-wallet".into(),
                    required: "pay-lightning".into(),
                    reason: MissingRequirementReason::NotInstalled,
                },
                MissingRequirement {
                    plugin: "lexe-wallet".into(),
                    required: "capsules".into(),
                    reason: MissingRequirementReason::Disabled,
                },
            ]
        );
        assert!(
            missing[0]
                .describe()
                .contains("plugins install pay-lightning")
        );
        assert!(missing[1].describe().contains("plugins enable capsules"));
    }

    #[test]
    fn a_disabled_plugin_does_not_report_its_own_requirements() {
        let installed = [plugin("lexe-wallet", false, &["pay-lightning"])];
        assert!(missing_requirements(&installed).is_empty());
    }

    #[test]
    fn required_by_lists_dependents_whether_or_not_they_are_enabled() {
        let installed = [
            plugin("lexe-wallet", true, &["pay-lightning"]),
            plugin("nwc-wallet", false, &["pay-lightning"]),
            plugin("pay-lightning", true, &[]),
        ];
        assert_eq!(
            required_by(&installed, "pay-lightning"),
            vec!["lexe-wallet".to_string(), "nwc-wallet".to_string()]
        );
        assert!(required_by(&installed, "lexe-wallet").is_empty());
    }

    #[test]
    fn deleting_a_required_plugin_is_refused_until_its_dependents_go() {
        let temp = tempfile::tempdir().unwrap();
        let store = PluginStore::new(temp.path());
        store
            .save(&plugin("lexe-wallet", true, &["pay-lightning"]))
            .unwrap();
        store.save(&plugin("pay-lightning", true, &[])).unwrap();

        let error = ensure_not_required(&store, "pay-lightning", "delete").unwrap_err();
        assert_eq!(
            error.to_string(),
            "cannot delete pay-lightning: required by lexe-wallet; delete it first"
        );
        assert!(ensure_not_required(&store, "lexe-wallet", "delete").is_ok());

        store.delete("lexe-wallet").unwrap();
        assert!(ensure_not_required(&store, "pay-lightning", "delete").is_ok());
    }

    #[test]
    fn uninstalled_requirements_skips_installed_ones_even_when_disabled() {
        let temp = tempfile::tempdir().unwrap();
        let store = PluginStore::new(temp.path());
        store.save(&plugin("capsules", false, &[])).unwrap();
        let wallet = plugin("lexe-wallet", true, &["pay-lightning", "capsules"]);
        assert_eq!(
            uninstalled_requirements(&store, &wallet).unwrap(),
            vec!["pay-lightning".to_string()]
        );
    }
}
