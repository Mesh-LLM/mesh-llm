//! Requirements between installed plugins, declared by `requires` in a
//! plugin's `plugin.toml` and recorded on its install metadata.
//!
//! Install fetches a missing requirement from the catalog; delete refuses to
//! orphan one; a node start and `plugins list` report a requirement that is
//! missing or disabled. The host never starts, orders or links plugins for
//! each other: how two plugins talk stays an operator-authorized binding.

use std::collections::{BTreeMap, BTreeSet};

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

/// Refuse a package before it is put in place when its `requires` would close
/// a cycle through the installed plugins: some installed plugin that `name`
/// would require already requires `name`, directly or through others. Checked
/// on the staged package, so a refused install or update leaves the installed
/// executable and metadata as they were.
pub fn ensure_no_cycle_through_installed(
    store: &PluginStore,
    name: &str,
    requires: &[String],
) -> Result<()> {
    let installed: BTreeMap<String, Vec<String>> = store
        .list()?
        .into_iter()
        .map(|plugin| (plugin.name, plugin.requires))
        .collect();
    // Each visited plugin's dependent on the path from `name`, so the cycle
    // can be spelled out.
    let mut parent: BTreeMap<String, String> = BTreeMap::new();
    let mut pending: Vec<(String, String)> = requires
        .iter()
        .rev()
        .map(|required| (name.to_string(), required.clone()))
        .collect();
    while let Some((dependent, required)) = pending.pop() {
        if required == name {
            // `dependent` back to `name` through `parent`, then forward.
            let mut path = Vec::new();
            let mut current = Some(dependent.as_str());
            while let Some(node) = current {
                path.push(node.to_string());
                current = parent.get(node).map(String::as_str);
            }
            path.reverse();
            path.push(name.to_string());
            bail!(
                "plugin requirements form a cycle: {}; {name} was not installed",
                path.join(" requires ")
            );
        }
        if parent.contains_key(&required) {
            continue;
        }
        parent.insert(required.clone(), dependent);
        let Some(next) = installed.get(&required) else {
            continue;
        };
        for next in next.iter().rev() {
            pending.push((required.clone(), next.clone()));
        }
    }
    Ok(())
}

/// Installs one missing requirement: `required`, needed by `dependent`.
/// Returns the metadata the install recorded.
pub trait RequirementInstaller {
    fn install(
        &mut self,
        dependent: String,
        required: String,
    ) -> impl std::future::Future<Output = Result<InstalledPluginMetadata>>;
}

/// Walk `root`'s requirement graph and install every plugin that is not
/// installed, via `install(dependent, required)`. Installed plugins are walked
/// through, never reinstalled or re-enabled, so a retry after a transitive
/// failure (A→B installed, B→C failed) reaches C. A requirement that closes a
/// cycle back to an ancestor is refused: the plugin that declared the closing
/// edge is removed again if this walk installed it, so no cycle is persisted.
/// The first failure stops the walk; everything installed before it stays.
pub async fn install_missing(
    store: &PluginStore,
    root: &InstalledPluginMetadata,
    installer: &mut impl RequirementInstaller,
) -> Result<()> {
    let mut parent: BTreeMap<String, String> = BTreeMap::new();
    let mut visited: BTreeSet<String> = BTreeSet::from([root.name.clone()]);
    let mut pending: Vec<(String, String)> = root
        .requires
        .iter()
        .rev()
        .map(|required| (root.name.clone(), required.clone()))
        .collect();
    while let Some((dependent, required)) = pending.pop() {
        if is_ancestor(&parent, &dependent, &required) {
            bail!(
                "plugin requirements form a cycle: {dependent} requires {required}, which requires {dependent} in turn"
            );
        }
        if !visited.insert(required.clone()) {
            continue;
        }
        parent.insert(required.clone(), dependent.clone());
        let (metadata, fresh) = match store.load_optional(&required)? {
            Some(existing) => (existing, false),
            None => (
                installer
                    .install(dependent.clone(), required.clone())
                    .await?,
                true,
            ),
        };
        if let Some(back) = metadata
            .requires
            .iter()
            .find(|next| is_ancestor(&parent, &metadata.name, next))
        {
            if fresh {
                store.delete(&metadata.name)?;
            }
            bail!(
                "plugin requirements form a cycle: {} requires {back}, which requires {} in turn{}",
                metadata.name,
                metadata.name,
                if fresh { "; it was not installed" } else { "" }
            );
        }
        for next in metadata.requires.iter().rev() {
            pending.push((metadata.name.clone(), next.clone()));
        }
    }
    Ok(())
}

/// Whether `candidate` is `name` itself or one of its ancestors in `parent`.
fn is_ancestor(parent: &BTreeMap<String, String>, name: &str, candidate: &str) -> bool {
    let mut current = Some(name);
    while let Some(node) = current {
        if node == candidate {
            return true;
        }
        current = parent.get(node).map(String::as_str);
    }
    false
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

    fn store_with(dir: &std::path::Path, plugins: &[InstalledPluginMetadata]) -> PluginStore {
        let store = PluginStore::new(dir);
        for plugin in plugins {
            store.save(plugin).unwrap();
        }
        store
    }

    /// Test installer: records each request, saves `plugin(required, requires_for(required))`
    /// into the store, or fails for names in `fail`.
    struct FakeInstaller {
        root: PathBuf,
        requires_for: fn(&str) -> Vec<&'static str>,
        fail: &'static [&'static str],
        installed: Vec<String>,
    }

    impl RequirementInstaller for FakeInstaller {
        async fn install(
            &mut self,
            dependent: String,
            required: String,
        ) -> Result<InstalledPluginMetadata> {
            self.installed.push(format!("{dependent}->{required}"));
            if self.fail.contains(&required.as_str()) {
                bail!("{required} is not in the catalog");
            }
            let metadata = plugin(&required, true, &(self.requires_for)(&required));
            PluginStore::new(&self.root).save(&metadata)?;
            Ok(metadata)
        }
    }

    fn fake(
        root: &std::path::Path,
        requires_for: fn(&str) -> Vec<&'static str>,
        fail: &'static [&'static str],
    ) -> FakeInstaller {
        FakeInstaller {
            root: root.to_path_buf(),
            requires_for,
            fail,
            installed: Vec::new(),
        }
    }

    #[tokio::test]
    async fn retry_walks_through_an_installed_requirement_to_the_one_still_missing() {
        let temp = tempfile::tempdir().unwrap();
        let store = store_with(
            temp.path(),
            &[plugin("a", true, &["b"]), plugin("b", true, &["c"])],
        );
        let mut installer = fake(temp.path(), |_| vec![], &[]);
        install_missing(&store, &plugin("a", true, &["b"]), &mut installer)
            .await
            .unwrap();
        assert_eq!(installer.installed, vec!["b->c"]);
        assert!(store.load_optional("c").unwrap().is_some());
    }

    #[tokio::test]
    async fn a_failure_stops_the_walk_and_keeps_what_was_installed() {
        let temp = tempfile::tempdir().unwrap();
        let store = store_with(temp.path(), &[plugin("a", true, &["b"])]);
        let mut installer = fake(
            temp.path(),
            |name| if name == "b" { vec!["c"] } else { vec![] },
            &["c"],
        );
        let error = install_missing(&store, &plugin("a", true, &["b"]), &mut installer)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("not in the catalog"), "{error}");
        assert!(store.load_optional("b").unwrap().is_some());
        assert!(store.load_optional("c").unwrap().is_none());
    }

    #[tokio::test]
    async fn a_cycle_is_refused_and_the_closing_plugin_is_not_kept() {
        let temp = tempfile::tempdir().unwrap();
        let store = store_with(temp.path(), &[plugin("a", true, &["b"])]);
        let mut installer = fake(temp.path(), |_| vec!["a"], &[]);
        let error = install_missing(&store, &plugin("a", true, &["b"]), &mut installer)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("cycle"), "{error}");
        assert!(store.load_optional("b").unwrap().is_none());
        assert!(ensure_not_required(&store, "a", "delete").is_ok());
    }

    #[tokio::test]
    async fn a_shared_requirement_is_installed_once() {
        let temp = tempfile::tempdir().unwrap();
        let store = store_with(temp.path(), &[plugin("a", true, &["b", "c"])]);
        let mut installer = fake(
            temp.path(),
            |name| if name == "b" { vec!["c"] } else { vec![] },
            &[],
        );
        install_missing(&store, &plugin("a", true, &["b", "c"]), &mut installer)
            .await
            .unwrap();
        assert_eq!(installer.installed, vec!["a->b", "b->c"]);
    }

    #[test]
    fn a_candidate_whose_requirements_lead_back_to_it_is_refused_before_it_lands() {
        // Installed: b requires c, c requires a (a never landed). A candidate
        // a requiring b would close a→b→c→a.
        let temp = tempfile::tempdir().unwrap();
        let store = store_with(
            temp.path(),
            &[plugin("b", true, &["c"]), plugin("c", false, &["a"])],
        );
        let error = ensure_no_cycle_through_installed(&store, "a", &["b".to_string()]).unwrap_err();
        assert_eq!(
            error.to_string(),
            "plugin requirements form a cycle: a requires b requires c requires a; a was not installed"
        );
        // Requiring c directly closes a→c→a the same way.
        assert!(ensure_no_cycle_through_installed(&store, "a", &["c".to_string()]).is_err());
    }

    #[test]
    fn a_candidate_is_accepted_when_nothing_installed_requires_it() {
        let temp = tempfile::tempdir().unwrap();
        let store = store_with(
            temp.path(),
            &[plugin("b", true, &["c"]), plugin("c", true, &[])],
        );
        ensure_no_cycle_through_installed(&store, "a", &["b".to_string(), "d".to_string()])
            .unwrap();
        // An update that keeps a's own old entry in the store is judged on the
        // candidate's requirements, not the stored ones.
        store.save(&plugin("a", true, &["c"])).unwrap();
        ensure_no_cycle_through_installed(&store, "a", &["b".to_string()]).unwrap();
    }

    #[test]
    fn a_met_requirement_is_not_reported() {
        let installed = [
            plugin("demo-app", true, &["toolkit"]),
            plugin("toolkit", true, &[]),
        ];
        assert!(missing_requirements(&installed).is_empty());
    }

    #[test]
    fn missing_and_disabled_requirements_are_reported_with_their_reason() {
        let installed = [
            plugin("demo-app", true, &["toolkit", "capsules"]),
            plugin("capsules", false, &[]),
        ];
        let missing = missing_requirements(&installed);
        assert_eq!(
            missing,
            vec![
                MissingRequirement {
                    plugin: "demo-app".into(),
                    required: "toolkit".into(),
                    reason: MissingRequirementReason::NotInstalled,
                },
                MissingRequirement {
                    plugin: "demo-app".into(),
                    required: "capsules".into(),
                    reason: MissingRequirementReason::Disabled,
                },
            ]
        );
        assert!(missing[0].describe().contains("plugins install toolkit"));
        assert!(missing[1].describe().contains("plugins enable capsules"));
    }

    #[test]
    fn a_disabled_plugin_does_not_report_its_own_requirements() {
        let installed = [plugin("demo-app", false, &["toolkit"])];
        assert!(missing_requirements(&installed).is_empty());
    }

    #[test]
    fn required_by_lists_dependents_whether_or_not_they_are_enabled() {
        let installed = [
            plugin("demo-app", true, &["toolkit"]),
            plugin("other-app", false, &["toolkit"]),
            plugin("toolkit", true, &[]),
        ];
        assert_eq!(
            required_by(&installed, "toolkit"),
            vec!["demo-app".to_string(), "other-app".to_string()]
        );
        assert!(required_by(&installed, "demo-app").is_empty());
    }

    #[test]
    fn deleting_a_required_plugin_is_refused_until_its_dependents_go() {
        let temp = tempfile::tempdir().unwrap();
        let store = PluginStore::new(temp.path());
        store.save(&plugin("demo-app", true, &["toolkit"])).unwrap();
        store.save(&plugin("toolkit", true, &[])).unwrap();

        let error = ensure_not_required(&store, "toolkit", "delete").unwrap_err();
        assert_eq!(
            error.to_string(),
            "cannot delete toolkit: required by demo-app; delete it first"
        );
        assert!(ensure_not_required(&store, "demo-app", "delete").is_ok());

        store.delete("demo-app").unwrap();
        assert!(ensure_not_required(&store, "toolkit", "delete").is_ok());
    }

    #[test]
    fn uninstalled_requirements_skips_installed_ones_even_when_disabled() {
        let temp = tempfile::tempdir().unwrap();
        let store = PluginStore::new(temp.path());
        store.save(&plugin("capsules", false, &[])).unwrap();
        let app = plugin("demo-app", true, &["toolkit", "capsules"]);
        assert_eq!(
            uninstalled_requirements(&store, &app).unwrap(),
            vec!["toolkit".to_string()]
        );
    }
}
