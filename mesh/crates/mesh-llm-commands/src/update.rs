//! `mesh-llm update`: upgrade the host and take care of installed plugins.
//!
//! Every managed upgrade path checks **all** installed plugins against the
//! target host version — set membership never filters the inventory. Known
//! incompatibilities are repaired automatically from verified catalog pins
//! before the new host is activated; a repair that cannot be made stops the
//! upgrade rather than activating a known-incompatible plugin. Compatible
//! plugins stay at their versions, with attended notices for available
//! updates. Optional plugin sets install only when explicitly selected or
//! accepted at an attended prompt; unattended paths install nothing extra.

use anyhow::{Result, bail};
use mesh_llm_cli::{BinaryFlavor, Cli, Command};
use mesh_llm_plugin_manager::{
    PluginInstallOptions, PluginProgressEvent, PluginProgressReporter, PluginStore,
    PluginUpgradeVerdict, default_plugins_opted_out, plan_plugin_upgrades,
    repair_incompatible_plugins,
};
use mesh_llm_system::{autoupdate, backend};
use std::io::Write;

use crate::plugin_sets::{
    attended, fetch_catalog, optional_plugin_plan, render_update_notices, run_optional_plugin_phase,
};

pub async fn run_update(cli: &Cli) -> Result<()> {
    let (requested_version, flavor, detect_flavor, plugin_set, skip_plugin_install, no_interactive) =
        match &cli.command {
            Some(Command::Update {
                version,
                flavor,
                detect_flavor,
                plugin_set,
                skip_plugin_install,
                no_interactive,
                no_default_plugins,
            }) => (
                version.as_deref(),
                binary_flavor_to_backend(*flavor),
                *detect_flavor,
                plugin_set.as_deref(),
                *skip_plugin_install || *no_default_plugins,
                *no_interactive,
            ),
            _ => (None, None, false, None, false, false),
        };

    // Preflight the host update first: the install target, release, bundle
    // asset, and writability are all checked before anything is touched, so
    // plugin repairs never run against a host install that would fail on a
    // condition already knowable now.
    let prepared = autoupdate::prepare_update_command(autoupdate::UpdateCommandOptions {
        flavor,
        detect_flavor,
        requested_version,
        current_version: mesh_llm_build_info::BUILD_VERSION,
    })
    .await?;

    // With no newer release the running version stays the target: the same
    // check still repairs plugins incompatible with the host that keeps
    // running, and the optional phase proceeds.
    let target_version = prepared
        .as_ref()
        .map(|prepared| prepared.release_version().to_string())
        .unwrap_or_else(|| mesh_llm_build_info::RELEASE_VERSION.to_string());

    run_plugin_upgrade_checks(&target_version).await?;

    let is_attended = attended(no_interactive);
    let plan = optional_plugin_plan(
        plugin_set,
        skip_plugin_install,
        default_plugins_opted_out(),
        is_attended,
    );
    // Windows exits during installer handoff, so report notices and any
    // unfinished optional work before handing control to the installer.
    if is_attended && let Some(notices) = update_notices(&target_version).await {
        let mut err = mesh_llm_events::console_err();
        for line in notices {
            let _ = writeln!(err, "{line}");
        }
    }
    if cfg!(windows) && prepared.is_some() {
        let mut err = mesh_llm_events::console_err();
        for line in crate::plugin_sets::handoff_optional_plugin_lines(&plan) {
            let _ = writeln!(err, "{line}");
        }
    }
    if let Some(prepared) = prepared.as_ref() {
        autoupdate::apply_prepared_update(prepared, mesh_llm_build_info::BUILD_VERSION).await?;
    }

    // The host upgrade landed. Optional plugin installation never fails it:
    // warn, guide a retry, and report the installation as incomplete.
    match run_optional_plugin_phase(&plan, &target_version, &mut |set| {
        crate::plugin_sets::prompt_install_set(set)
    })
    .await
    {
        Ok(_) => {}
        // An unknown explicitly selected set is a usage error; the host
        // upgrade already succeeded, so say what happened instead of
        // discarding it.
        Err(error) => {
            let mut err = mesh_llm_events::console_err();
            let _ = writeln!(err, "⚠️  {error:#}");
        }
    }
    Ok(())
}

/// Check every installed plugin against the target host version and stage
/// required repairs, before the new host is activated.
///
/// - The catalog is fetched once and every decision reads that snapshot.
/// - With no catalog, compatibility must be established locally or the
///   upgrade stays unapplied with an explanation.
/// - A plugin that needs repair and has a verified compatible catalog
///   release is updated now (settings, data, and enabled state preserved);
///   a failed repair stops the upgrade before the host is replaced.
async fn run_plugin_upgrade_checks(target_version: &str) -> Result<()> {
    let options = match PluginInstallOptions::from_env() {
        Ok(options) => options,
        Err(error) => {
            // No plugin store on this machine (or an unsupported platform):
            // nothing is installed, so nothing can block the upgrade.
            let mut err = mesh_llm_events::console_err();
            let _ = writeln!(
                err,
                "ℹ️  Installed plugin check skipped: no plugin store ({error:#})."
            );
            return Ok(());
        }
    };
    let store = PluginStore::new(&options.store_root);
    if let Err(error) = store.migrate_legacy_defaults() {
        let mut err = mesh_llm_events::console_err();
        let _ = writeln!(
            err,
            "⚠️  Legacy default plugin records were not migrated: {error:#}"
        );
    }
    let installed = store.list()?;
    let mut err = mesh_llm_events::console_err();
    if installed.is_empty() {
        let _ = writeln!(err, "ℹ️  No installed plugins to check.");
        return Ok(());
    }

    let catalog = fetch_catalog(&options).await.ok();
    let plan = plan_plugin_upgrades(
        &store,
        catalog.as_ref(),
        options.target.triple(),
        target_version,
        mesh_llm_plugin::PROTOCOL_VERSION,
    )?;

    if !plan.catalog_available {
        let _ = writeln!(
            err,
            "⚠️  Plugin catalog unavailable; checking installed plugins locally."
        );
    }
    for check in &plan.checks {
        if let PluginUpgradeVerdict::Unknown { reason } = &check.verdict {
            let _ = writeln!(
                err,
                "⚠️  {} {}: compatibility with MeshLLM {target_version} is unknown ({reason}).",
                check.name, check.installed_version
            );
        }
    }
    let repairs = plan.required_repairs();
    if repairs.is_empty() {
        let compatible = plan
            .checks
            .iter()
            .filter(|check| !matches!(check.verdict, PluginUpgradeVerdict::Unknown { .. }))
            .count();
        let _ = writeln!(
            err,
            "ℹ️  Checking installed plugins: {compatible} compatible; 0 require updates."
        );
        if plan.has_unestablished_compatibility() {
            bail!(
                "Upgrade not applied: compatibility with MeshLLM {target_version} could not be \
                 established for every installed plugin and no required repair is possible. \
                 Retry when the plugin catalog is reachable."
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
                let _ = writeln!(
                    err,
                    "⚠️  {} {} is incompatible with MeshLLM {target_version} ({reason}).",
                    check.name, check.installed_version
                );
                let _ = writeln!(
                    err,
                    "ℹ️  Required plugin update: {} {} -> {to}.",
                    check.name, check.installed_version
                );
            }
            PluginUpgradeVerdict::RepairBlocked { reason } => {
                let _ = writeln!(
                    err,
                    "⚠️  {} {} is incompatible with MeshLLM {target_version} ({reason}).",
                    check.name, check.installed_version
                );
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
        let repair_progress = &mut UpdatePluginProgress::default();
        let repairs_outcome =
            repair_incompatible_plugins(&plan, target_version, &options, repair_progress).await?;
        let mut err = mesh_llm_events::console_err();
        for (name, outcome) in &repairs_outcome.repaired {
            let state = if outcome.metadata.enabled {
                "settings and data preserved"
            } else {
                "remains disabled"
            };
            let _ = writeln!(
                err,
                "✅ Updated {name} to {}; {state}.",
                outcome.metadata.installed_version
            );
        }
        for (name, reason) in &repairs_outcome.failed {
            blocking.push(format!("{name}: {reason}"));
        }
    }
    if !blocking.is_empty() {
        let mut err = mesh_llm_events::console_err();
        for blocker in &blocking {
            let _ = writeln!(err, "❌ Required plugin repair failed: {blocker}");
        }
        let _ = writeln!(
            err,
            "ℹ️  Upgrade not applied. MeshLLM stays at {}. Completed plugin updates remain \
             installed and may require the target host; restore their previous releases before \
             running the old host.",
            mesh_llm_build_info::BUILD_VERSION
        );
        let _ = writeln!(err, "Retry: mesh-llm update");
        bail!(
            "one or more installed plugins are incompatible with MeshLLM {target_version} and could not be repaired"
        );
    }
    Ok(())
}

/// Update notices for the attended report, resolved after required repairs
/// and before a Windows installer handoff can exit this process.
async fn update_notices(target_version: &str) -> Option<Vec<String>> {
    let options = PluginInstallOptions::from_env().ok()?;
    let catalog = fetch_catalog(&options).await.ok()?;
    let store = PluginStore::new(&options.store_root);
    let plan = plan_plugin_upgrades(
        &store,
        Some(&catalog),
        options.target.triple(),
        target_version,
        mesh_llm_plugin::PROTOCOL_VERSION,
    )
    .ok()?;
    let lines = render_update_notices(&plan);
    (!lines.is_empty()).then_some(lines)
}

fn binary_flavor_to_backend(flavor: Option<BinaryFlavor>) -> Option<backend::BinaryFlavor> {
    flavor.map(|flavor| match flavor {
        BinaryFlavor::Cpu => backend::BinaryFlavor::Cpu,
        BinaryFlavor::Cuda => backend::BinaryFlavor::Cuda,
        BinaryFlavor::Rocm => backend::BinaryFlavor::Rocm,
        BinaryFlavor::Vulkan => backend::BinaryFlavor::Vulkan,
        BinaryFlavor::Metal => backend::BinaryFlavor::Metal,
    })
}

/// The progress rendering for automatic repairs.
#[derive(Default)]
struct UpdatePluginProgress {
    inner: crate::plugin::CliPluginProgress,
}

impl PluginProgressReporter for UpdatePluginProgress {
    fn report(&mut self, event: PluginProgressEvent) {
        match event {
            PluginProgressEvent::Warning { message } => {
                let mut err = mesh_llm_events::console_err();
                let _ = writeln!(err, "⚠️  {message}");
            }
            other => self.inner.report(other),
        }
    }
}
