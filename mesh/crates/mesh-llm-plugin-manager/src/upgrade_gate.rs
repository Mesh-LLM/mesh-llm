//! The one pre-activation plugin gate every host upgrade path runs.
//!
//! `mesh-llm update`, the automatic update at node startup, and the bootstrap
//! scripts (through `mesh-llm setup`) all replace the host binary, so they all
//! owe the installed plugins the same treatment: check each against the
//! target host version from one catalog snapshot, repair known
//! incompatibilities from verified catalog pins, and refuse to proceed when a
//! plugin's compatibility cannot be established or a repair fails. Callers
//! differ only in how lines are rendered and in what a refusal means for
//! them, so this module reports through callbacks and returns the verdict.

use std::time::Duration;

use anyhow::{Result, bail};

use crate::catalog::PluginCatalog;
use crate::install::{PluginInstallOptions, PluginProgressReporter};
use crate::store::PluginStore;
use crate::upgrade::{PluginUpgradeVerdict, plan_plugin_upgrades, repair_incompatible_plugins};

/// One catalog snapshot for one operation: the same fetched copy backs set
/// resolution, upgrade planning, and installation so membership cannot shift
/// mid-operation. The request carries a timeout so a stalled catalog server
/// cannot hang a set command, setup's optional phase, or an upgrade.
pub async fn fetch_catalog(options: &PluginInstallOptions) -> Result<PluginCatalog> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()?;
    let sets_url = crate::sets_url_for(&options.catalog_url);
    PluginCatalog::fetch_with_sets(&client, &options.catalog_url, sets_url.as_deref()).await
}

/// Check every installed plugin against `target_version` and stage required
/// repairs, before the new host is activated.
///
/// - The catalog is fetched once and every decision reads that snapshot.
/// - With no catalog, compatibility must be established locally or the
///   gate refuses with an explanation.
/// - A plugin that needs repair and has a verified compatible catalog
///   release is updated now (settings, data, and enabled state preserved);
///   a failed repair is reported and refused.
///
/// `Ok(())` means the host may be activated. `Err` means it must not be, and
/// every blocker has already been passed to `out`. A machine with no plugin
/// store has nothing installed and passes.
pub async fn check_and_repair_installed_plugins(
    target_version: &str,
    plugin_protocol_version: u32,
    out: &mut dyn FnMut(&str),
    progress: &mut impl PluginProgressReporter,
) -> Result<()> {
    let options = match PluginInstallOptions::from_env() {
        Ok(options) => options,
        Err(error) => {
            out(&format!(
                "ℹ️  Installed plugin check skipped: no plugin store ({error:#})."
            ));
            return Ok(());
        }
    };
    check_and_repair_with_options(
        &options,
        target_version,
        plugin_protocol_version,
        out,
        progress,
    )
    .await
}

pub async fn check_and_repair_with_options(
    options: &PluginInstallOptions,
    target_version: &str,
    plugin_protocol_version: u32,
    out: &mut dyn FnMut(&str),
    progress: &mut impl PluginProgressReporter,
) -> Result<()> {
    let store = PluginStore::new(&options.store_root);
    if let Err(error) = store.migrate_legacy_defaults() {
        out(&format!(
            "⚠️  Legacy default plugin records were not migrated: {error:#}"
        ));
    }
    if store.list()?.is_empty() {
        out("ℹ️  No installed plugins to check.");
        return Ok(());
    }

    let catalog = fetch_catalog(options).await.ok();
    let plan = plan_plugin_upgrades(
        &store,
        catalog.as_ref(),
        options.target.triple(),
        target_version,
        plugin_protocol_version,
    )?;

    if !plan.catalog_available {
        out("⚠️  Plugin catalog unavailable; checking installed plugins locally.");
    }
    for check in &plan.checks {
        if let PluginUpgradeVerdict::Unknown { reason } = &check.verdict {
            out(&format!(
                "⚠️  {} {}: compatibility with MeshLLM {target_version} is unknown ({reason}).",
                check.name, check.installed_version
            ));
        }
    }
    let repairs = plan.required_repairs();
    if repairs.is_empty() {
        let compatible = plan
            .checks
            .iter()
            .filter(|check| !matches!(check.verdict, PluginUpgradeVerdict::Unknown { .. }))
            .count();
        out(&format!(
            "ℹ️  Checking installed plugins: {compatible} compatible; 0 require updates."
        ));
        if plan.has_unestablished_compatibility() {
            bail!(
                "compatibility with MeshLLM {target_version} could not be established for \
                 every installed plugin and no required repair is possible. Retry when the \
                 plugin catalog is reachable."
            );
        }
        return Ok(());
    }

    let mut blocking: Vec<String> = Vec::new();
    // Report every plugin's state before repairing anything: each required
    // update and each blocker is named, whatever happens next.
    for check in &repairs {
        match &check.verdict {
            PluginUpgradeVerdict::Repair { to, reason } => {
                out(&format!(
                    "⚠️  {} {} is incompatible with MeshLLM {target_version} ({reason}).",
                    check.name, check.installed_version
                ));
                out(&format!(
                    "ℹ️  Required plugin update: {} {} -> {to}.",
                    check.name, check.installed_version
                ));
            }
            PluginUpgradeVerdict::RepairBlocked { reason } => {
                out(&format!(
                    "⚠️  {} {} is incompatible with MeshLLM {target_version} ({reason}).",
                    check.name, check.installed_version
                ));
                blocking.push(format!("{}: {reason}", check.name));
            }
            _ => {}
        }
    }
    // Attempt every repair once — but only when nothing already blocks the
    // upgrade: a repair is chosen for the target host and can exclude the
    // host that keeps running, so landing repairs the upgrade will not
    // activate leaves those plugins stopped for no gain.
    if blocking.is_empty() {
        let outcome = repair_incompatible_plugins(&plan, target_version, options, progress).await?;
        for (name, outcome) in &outcome.repaired {
            let state = if outcome.metadata.enabled {
                "settings and data preserved"
            } else {
                "remains disabled"
            };
            out(&format!(
                "✅ Updated {name} to {}; {state}.",
                outcome.metadata.installed_version
            ));
        }
        for (name, reason) in &outcome.failed {
            blocking.push(format!("{name}: {reason}"));
        }
    }
    if !blocking.is_empty() {
        for blocker in &blocking {
            out(&format!("❌ Required plugin repair failed: {blocker}"));
        }
        bail!(
            "one or more installed plugins are incompatible with MeshLLM {target_version} and \
             could not be repaired"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compatibility::HostCompatibility;
    use crate::store::InstalledPluginMetadata;
    use crate::target::PluginTarget;

    const PROTOCOL: u32 = 3;

    fn options(root: &std::path::Path) -> PluginInstallOptions {
        PluginInstallOptions {
            store_root: root.to_path_buf(),
            install_root: root.join("installed"),
            // Nothing listens here, so the catalog is unavailable and the
            // gate must decide from what the installed release recorded.
            catalog_url: "http://127.0.0.1:1/plugins.jsonl".into(),
            target: PluginTarget::current().expect("supported test platform"),
            host_version: "0.1.0".into(),
        }
    }

    fn save_plugin(
        store: &PluginStore,
        options: &PluginInstallOptions,
        compatibility: HostCompatibility,
    ) {
        store
            .save(&InstalledPluginMetadata {
                name: "lexe".into(),
                source_repository: "https://github.com/mesh-llm/lexe".into(),
                installed_version: "v1.0.0".into(),
                target_triple: options.target.triple().into(),
                downloaded_asset_name: "lexe.tar.gz".into(),
                install_path: store.root().join("lexe"),
                enabled: true,
                default_managed: false,
                host_compatibility: Some(compatibility),
                allowed_surfaces: None,
                manifest: None,
                last_protocol_version: Some(PROTOCOL),
                last_status: None,
                last_error: None,
                verified_sha256: None,
            })
            .unwrap();
    }

    async fn run_gate(options: &PluginInstallOptions, target: &str) -> (Result<()>, Vec<String>) {
        let mut lines = Vec::new();
        let result = check_and_repair_with_options(
            options,
            target,
            PROTOCOL,
            &mut |line| lines.push(line.to_string()),
            &mut |_| {},
        )
        .await;
        (result, lines)
    }

    #[tokio::test]
    async fn the_gate_refuses_a_host_a_plugin_cannot_run_on_and_leaves_it_unchanged() {
        let temp = tempfile::tempdir().unwrap();
        let options = options(temp.path());
        let store = PluginStore::new(&options.store_root);
        save_plugin(
            &store,
            &options,
            HostCompatibility {
                min: Some("0.1.0".into()),
                max: Some("0.5.0".into()),
            },
        );

        let (result, lines) = run_gate(&options, "0.81.0").await;

        let error = result.expect_err("an incompatible plugin with no repair must refuse");
        assert!(
            error.to_string().contains("could not be repaired"),
            "{error:#}"
        );
        assert!(
            lines
                .iter()
                .any(|line| line.contains("lexe v1.0.0 is incompatible")),
            "{lines:?}"
        );
        let after = store.list().unwrap();
        assert_eq!(after.len(), 1);
        assert_eq!(after[0].installed_version, "v1.0.0");
        assert!(after[0].enabled);
    }

    #[tokio::test]
    async fn the_gate_passes_when_every_installed_plugin_supports_the_target_host() {
        let temp = tempfile::tempdir().unwrap();
        let options = options(temp.path());
        let store = PluginStore::new(&options.store_root);
        save_plugin(
            &store,
            &options,
            HostCompatibility {
                min: Some("0.1.0".into()),
                max: None,
            },
        );

        let (result, lines) = run_gate(&options, "0.81.0").await;

        result.unwrap_or_else(|error| panic!("{error:#}; {lines:?}"));
    }

    #[tokio::test]
    async fn the_gate_passes_with_no_installed_plugins() {
        let temp = tempfile::tempdir().unwrap();
        let options = options(temp.path());

        let (result, lines) = run_gate(&options, "0.81.0").await;

        result.unwrap();
        assert_eq!(lines, ["ℹ️  No installed plugins to check."]);
    }
}
