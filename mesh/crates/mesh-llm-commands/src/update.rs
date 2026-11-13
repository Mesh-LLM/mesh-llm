//! `mesh-llm update`: upgrade the host and take care of installed plugins.
//!
//! The explicit update command checks **all** installed plugins against the
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
    default_plugins_opted_out, plan_plugin_upgrades,
    upgrade_gate::check_and_repair_installed_plugins,
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
    validate_plugin_set_handoff(plugin_set, cfg!(windows) && prepared.is_some())?;

    // With no newer release the running version stays the target: the same
    // check still repairs plugins incompatible with the host that keeps
    // running, and the optional phase proceeds.
    let target_version = prepared
        .as_ref()
        .map(|prepared| prepared.release_version().to_string())
        .unwrap_or_else(|| mesh_llm_build_info::RELEASE_VERSION.to_string());

    let is_attended = attended(no_interactive);
    let plan = optional_plugin_plan(
        plugin_set,
        skip_plugin_install,
        default_plugins_opted_out(),
        is_attended,
    );
    apply_host_after_plugin_gate(&target_version, async || {
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
        Ok(())
    })
    .await?;

    // The host upgrade landed. Optional plugin installation never fails it:
    // warn, guide a retry, and report the installation as incomplete.
    match run_optional_plugin_phase(&plan, &target_version, &mut |set| {
        crate::plugin_sets::prompt_install_set(set)
    })
    .await
    {
        Ok(_) => {}
        // Preserve the completed host result even if optional work fails.
        Err(error) => {
            let mut err = mesh_llm_events::console_err();
            let _ = writeln!(err, "⚠️  {error:#}");
        }
    }
    Ok(())
}

/// Run the shared plugin gate, rendering to the console: the check and
/// repairs every host upgrade owes installed plugins (see
/// [`mesh_llm_plugin_manager::upgrade_gate`]). `mesh-llm update` and
/// `mesh-llm setup` (the bootstrap scripts' upgrade path) both call it.
pub(crate) async fn check_installed_plugins_for_host(target_version: &str) -> Result<()> {
    let mut progress = UpdatePluginProgress::default();
    check_and_repair_installed_plugins(
        target_version,
        mesh_llm_plugin::PROTOCOL_VERSION,
        &mut |line| {
            let mut err = mesh_llm_events::console_err();
            let _ = writeln!(err, "{line}");
        },
        &mut progress,
    )
    .await
}

/// The gate before the new host is activated: a refusal leaves the running
/// host in place.
async fn run_plugin_upgrade_checks(target_version: &str) -> Result<()> {
    if let Err(error) = check_installed_plugins_for_host(target_version).await {
        let mut err = mesh_llm_events::console_err();
        let _ = writeln!(
            err,
            "ℹ️  Upgrade not applied. MeshLLM stays at {}. Completed plugin updates remain \
             installed and may require the target host; restore their previous releases before \
             running the old host.",
            mesh_llm_build_info::BUILD_VERSION
        );
        let _ = writeln!(err, "Retry: mesh-llm update");
        bail!("Upgrade not applied: {error:#}");
    }
    Ok(())
}

/// Replace the host only after the plugin gate passes. `run_update` reaches
/// `apply` only through here, so the gate runs on every update.
async fn apply_host_after_plugin_gate(
    target_version: &str,
    apply: impl AsyncFnOnce() -> Result<()>,
) -> Result<()> {
    run_plugin_upgrade_checks(target_version).await?;
    apply().await
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

/// Installer handoff exits this process before it can honor a selected set.
/// Reject that combination before any plugin repair or host replacement.
fn validate_plugin_set_handoff(plugin_set: Option<&str>, windows_handoff: bool) -> Result<()> {
    if windows_handoff && let Some(name) = plugin_set {
        bail!(
            "Windows host installer handoff cannot complete --plugin-set {name}. \
             Run the host update without --plugin-set, then run \
             `mesh-llm plugins sets install {name}` after the installer finishes. \
             No plugin repairs or host update were applied."
        );
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::{apply_host_after_plugin_gate, validate_plugin_set_handoff};
    use crate::plugin_upgrade_test_support::{with_plugin_store, write_plugin};

    #[test]
    fn an_explicit_set_cannot_be_discarded_by_windows_installer_handoff() {
        let error = validate_plugin_set_handoff(Some("extra"), true).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("mesh-llm plugins sets install extra")
        );
        assert!(
            error
                .to_string()
                .contains("No plugin repairs or host update were applied")
        );
        assert!(validate_plugin_set_handoff(None, true).is_ok());
        assert!(validate_plugin_set_handoff(Some("extra"), false).is_ok());
    }

    #[test]
    #[serial_test::serial]
    fn the_host_is_not_replaced_while_an_installed_plugin_cannot_run_on_it() {
        with_plugin_store(|store| {
            write_plugin(store, "0.1.0", Some("0.5.0"));
            let applied = std::cell::Cell::new(false);
            let result =
                tokio::runtime::Runtime::new()
                    .unwrap()
                    .block_on(apply_host_after_plugin_gate(
                        mesh_llm_build_info::RELEASE_VERSION,
                        async || {
                            applied.set(true);
                            Ok(())
                        },
                    ));
            let error = result.expect_err("an unrepairable plugin must refuse the upgrade");
            assert!(
                error.to_string().contains("Upgrade not applied"),
                "{error:#}"
            );
            assert!(!applied.get(), "host replaced despite a refused gate");
        });
    }

    #[test]
    #[serial_test::serial]
    fn the_host_is_replaced_when_installed_plugins_pass_the_gate() {
        with_plugin_store(|store| {
            write_plugin(store, "0.1.0", None);
            let applied = std::cell::Cell::new(false);
            tokio::runtime::Runtime::new()
                .unwrap()
                .block_on(apply_host_after_plugin_gate(
                    mesh_llm_build_info::RELEASE_VERSION,
                    async || {
                        applied.set(true);
                        Ok(())
                    },
                ))
                .unwrap();
            assert!(applied.get());
        });
    }
}
