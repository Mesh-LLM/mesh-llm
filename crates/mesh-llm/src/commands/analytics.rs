//! `mesh-llm analytics` — inspect and change anonymous usage reporting.
//!
//! The opt-out is written to the config file rather than held in memory, so
//! it survives restarts and upgrades. Reporting is the shipped default, which
//! makes a durable, discoverable off switch part of the bargain.

use anyhow::{Context, Result};
use mesh_llm_analytics::{ConsentInputs, Disposition};
use mesh_llm_cli::AnalyticsCommand;
use mesh_llm_config::{ConfigStore, config_path, load_config};
use serde_json::json;
use std::path::Path;

pub(crate) fn dispatch_analytics_command(
    command: &AnalyticsCommand,
    config_override: Option<&Path>,
) -> Result<()> {
    match command {
        AnalyticsCommand::Status { json } => run_status(config_override, *json),
        AnalyticsCommand::Enable => set_enabled(config_override, true),
        AnalyticsCommand::Disable => set_enabled(config_override, false),
    }
}

/// Read `[analytics] enabled` without failing on an unreadable config.
///
/// A broken config must not make it impossible to find out what is being
/// reported, so an unreadable file is treated as "unstated".
fn configured_enabled(config_override: Option<&Path>) -> Option<bool> {
    load_config(config_override)
        .ok()
        .and_then(|config| config.analytics.enabled)
}

fn run_status(config_override: Option<&Path>, json: bool) -> Result<()> {
    let inputs = ConsentInputs::from_env(configured_enabled(config_override));
    let disposition = inputs.resolve();
    let install_id = mesh_llm_analytics::state_dir()
        .ok()
        .and_then(|dir| mesh_llm_analytics::load_or_create(&dir).ok());
    let path = config_path(config_override)?;

    if json {
        let payload = json!({
            "enabled": disposition.is_enabled(),
            "reason": disposition.explain(),
            "install_id": install_id.as_ref().map(mesh_llm_analytics::InstallId::as_str),
            "endpoint": disposition
                .is_enabled()
                .then(mesh_llm_analytics::ingestion_host),
            "config_path": path.display().to_string(),
        });
        println!("{}", serde_json::to_string_pretty(&payload)?);
        return Ok(());
    }

    println!(
        "Anonymous usage reporting: {}",
        if disposition.is_enabled() {
            "on"
        } else {
            "off"
        }
    );
    println!("  Reason:     {}", disposition.explain());
    if let Some(install_id) = install_id.as_ref() {
        println!("  Install ID: {}", install_id.as_str());
    }
    if disposition.is_enabled() {
        println!("  Endpoint:   {}", mesh_llm_analytics::ingestion_host());
    }
    println!("  Config:     {}", path.display());
    println!("\n{}", mesh_llm_analytics::NOTICE);

    if matches!(disposition, Disposition::DisabledNoKey) {
        println!(
            "\nThis build has no analytics key, so it reports nothing regardless of settings."
        );
    }
    Ok(())
}

fn set_enabled(config_override: Option<&Path>, enabled: bool) -> Result<()> {
    let path = config_path(config_override)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create config directory {}", parent.display()))?;
    }

    let store = ConfigStore::open(&path);
    // `edit_preserving` keeps comments and key order in a hand-edited config.
    store
        .edit_preserving(|document| {
            // Create `[analytics]` as a real section rather than letting
            // toml_edit infer an inline `analytics = { enabled = false }`,
            // which is valid but reads badly in a file people hand-edit.
            let analytics = document
                .entry("analytics")
                .or_insert_with(|| toml_edit::Item::Table(toml_edit::Table::new()));
            if analytics.as_table().is_none() {
                *analytics = toml_edit::Item::Table(toml_edit::Table::new());
            }
            analytics["enabled"] = toml_edit::value(enabled);
            Ok(())
        })
        .with_context(|| format!("failed to update {}", path.display()))?;

    if enabled {
        println!(
            "Anonymous usage reporting enabled. Written to {}",
            path.display()
        );
    } else {
        println!(
            "Anonymous usage reporting disabled. Written to {}",
            path.display()
        );
        println!("No further usage data will be sent from this machine.");
    }
    Ok(())
}

#[cfg(test)]
#[path = "analytics/tests.rs"]
mod tests;
