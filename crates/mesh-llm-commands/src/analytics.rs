//! `mesh-llm analytics` — inspect and change anonymous usage reporting.
//!
//! The opt-out is written to the config file rather than held in memory, so
//! it survives restarts and upgrades. Reporting is the shipped default, which
//! makes a durable, discoverable off switch part of the bargain.

use anyhow::{Context, Result};
use mesh_llm_analytics::{ConfigPreference, ConsentInputs, Disposition};
use mesh_llm_cli::AnalyticsCommand;
use mesh_llm_config::{ConfigStore, config_path, load_config};
use mesh_llm_events::{console_out, machine_out};
use serde_json::json;
use std::io::Write;
use std::path::Path;

pub fn dispatch_analytics_command(
    command: &AnalyticsCommand,
    config_override: Option<&Path>,
) -> Result<()> {
    match command {
        AnalyticsCommand::Status { json } => run_status(config_override, *json),
        AnalyticsCommand::Enable => set_enabled(config_override, true),
        AnalyticsCommand::Disable => set_enabled(config_override, false),
    }
}

/// Read `[analytics] enabled`, distinguishing "absent" from "unreadable".
///
/// A privacy control fails closed: an unreadable config may be hiding an
/// opt-out, and one mistyped key anywhere fails the whole document, so it
/// must not resolve to "no preference stated".
pub fn config_preference(config_override: Option<&Path>) -> ConfigPreference {
    match load_config(config_override) {
        Ok(config) => match config.analytics.enabled {
            Some(enabled) => ConfigPreference::Stated(enabled),
            None => ConfigPreference::Unstated,
        },
        Err(_) => ConfigPreference::Unreadable,
    }
}

fn run_status(config_override: Option<&Path>, json: bool) -> Result<()> {
    let inputs = ConsentInputs::from_env(config_preference(config_override));
    let disposition = inputs.resolve();
    // Read-only: asking what is collected must not create the identifier,
    // which would consume the first-run signal so the real first run never
    // reports as one.
    let install_id = mesh_llm_analytics::state_dir()
        .ok()
        .and_then(|dir| mesh_llm_analytics::load(&dir));
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
        // The document `--json` was invoked to produce: `machine_out`, which
        // is never suppressed, and never `println!` — a raw print would land
        // on stdout alongside an installed JSON sink's own output.
        writeln!(machine_out(), "{}", serde_json::to_string_pretty(&payload)?)?;
        return Ok(());
    }

    let mut out = console_out();
    writeln!(
        out,
        "Anonymous usage reporting: {}",
        if disposition.is_enabled() {
            "on"
        } else {
            "off"
        }
    )?;
    writeln!(out, "  Reason:     {}", disposition.explain())?;
    if let Some(install_id) = install_id.as_ref() {
        writeln!(out, "  Install ID: {}", install_id.as_str())?;
    }
    if disposition.is_enabled() {
        writeln!(
            out,
            "  Endpoint:   {}",
            mesh_llm_analytics::ingestion_host()
        )?;
    }
    writeln!(out, "  Config:     {}", path.display())?;
    writeln!(out, "\n{}", mesh_llm_analytics::NOTICE)?;

    if matches!(disposition, Disposition::DisabledNoKey) {
        writeln!(
            out,
            "\nThis build has no analytics key, so it reports nothing regardless of settings."
        )?;
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

    let mut out = console_out();
    if enabled {
        writeln!(
            out,
            "Anonymous usage reporting enabled. Written to {}",
            path.display()
        )?;
    } else {
        writeln!(
            out,
            "Anonymous usage reporting disabled. Written to {}",
            path.display()
        )?;
        // The setting is read at process start, so a node already serving
        // keeps its reporter until it restarts. Claiming otherwise would be
        // the one false statement in a feature whose whole case rests on its
        // claims being checkable.
        writeln!(out, "Newly started mesh-llm processes will send nothing.")?;
        writeln!(
            out,
            "A node that is already running keeps reporting until it restarts."
        )?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "analytics/tests.rs"]
mod tests;
