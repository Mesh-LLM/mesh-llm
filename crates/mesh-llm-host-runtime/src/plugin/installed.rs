use super::config::{ExternalPluginSpec, PluginConfigEntry};
use super::startup::PluginStartupOptions;
use super::{PluginSummary, PluginWebUiState, derive_plugin_web_ui_state};
use anyhow::{Context, Result, bail};
use mesh_llm_plugin_manager::{InstalledPluginMetadata, PluginStore, default_store_root};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub(crate) enum ConfiguredExternalPlugin {
    Active(ExternalPluginSpec),
    Inactive(PluginSummary),
}

pub(crate) fn configured_disabled_installed_plugin_summary(
    entry: &PluginConfigEntry,
) -> Option<PluginSummary> {
    let metadata = installed_plugin_metadata_for_name(&entry.name)
        .ok()
        .flatten()?;
    metadata
        .manifest
        .as_ref()
        .and_then(|manifest| manifest.web_ui.as_ref())?;
    Some(PluginSummary {
        name: entry.name.clone(),
        kind: "installed".to_string(),
        enabled: false,
        status: "disabled".to_string(),
        pid: None,
        version: Some(metadata.installed_version.clone()),
        capabilities: Vec::new(),
        command: Some(installed_plugin_command(&metadata).display().to_string()),
        args: entry.args.clone(),
        tools: Vec::new(),
        manifest: None,
        web_ui: derive_plugin_web_ui_state(super::PluginWebUiStateInput {
            plugin_name: &entry.name,
            live_manifest: None,
            installed_metadata: Some(&metadata),
            web_ui_enabled: entry.web_ui_enabled,
            runtime_available: false,
            runtime_unavailable_reason: Some("plugin process is disabled"),
        }),
        startup: None,
        error: None,
    })
}

pub(crate) fn configured_external_plugin_spec(
    entry: &PluginConfigEntry,
) -> Result<ConfiguredExternalPlugin> {
    let startup = PluginStartupOptions::from_config(&entry.startup);
    let command = entry
        .command
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string);

    let command = match command {
        Some(command) => command,
        None => match installed_plugin_command_for_name(&entry.name) {
            Ok(command) => command,
            Err(error) if startup.optional => {
                return Ok(ConfiguredExternalPlugin::Inactive(
                    optional_configured_plugin_summary(entry, &startup, error),
                ));
            }
            Err(error) => return Err(error),
        },
    };

    Ok(ConfiguredExternalPlugin::Active(ExternalPluginSpec {
        name: entry.name.clone(),
        command,
        args: entry.args.clone(),
        url: entry.url.clone(),
        env: BTreeMap::new(),
        startup,
        web_ui_enabled: entry.web_ui_enabled,
        installed_metadata: installed_plugin_metadata_for_name(&entry.name)
            .ok()
            .flatten(),
    }))
}

pub(crate) fn append_installed_plugins(
    externals: &mut Vec<ExternalPluginSpec>,
    inactive: &mut Vec<PluginSummary>,
    names: &mut BTreeMap<String, ()>,
) {
    #[cfg(test)]
    if std::env::var_os("MESH_LLM_PLUGIN_DIR").is_none() {
        return;
    }

    let Ok(root) = default_store_root() else {
        return;
    };
    let store = PluginStore::new(root);
    let installed = match store.list() {
        Ok(installed) => installed,
        Err(error) => {
            inactive.push(installed_store_error_summary(error));
            return;
        }
    };

    for metadata in installed {
        if names.contains_key(&metadata.name) {
            continue;
        }
        names.insert(metadata.name.clone(), ());
        if !metadata.enabled {
            inactive.push(disabled_installed_plugin_summary(&metadata));
            continue;
        }
        let command = installed_plugin_command(&metadata);
        if !command.exists() {
            inactive.push(missing_installed_plugin_summary(&metadata, &command));
            continue;
        }
        externals.push(installed_plugin_spec(&metadata));
    }
}

fn installed_plugin_command_for_name(name: &str) -> Result<String> {
    let root = default_store_root().context("Cannot determine plugin install root")?;
    let store = PluginStore::new(root);
    let metadata = store
        .load_optional(name)?
        .with_context(|| {
            format!(
                "Plugin '{name}' is external. Run `mesh-llm plugins install {name}` or set `command` to the plugin binary."
            )
        })?;
    let command = installed_plugin_command(&metadata);
    if !command.exists() {
        bail!(
            "Plugin '{}' is installed but its executable is missing: {}",
            name,
            command.display()
        );
    }
    Ok(command.display().to_string())
}

fn installed_plugin_metadata_for_name(name: &str) -> Result<Option<InstalledPluginMetadata>> {
    let root = default_store_root().context("Cannot determine plugin install root")?;
    PluginStore::new(root).load_optional(name)
}

fn installed_plugin_spec(metadata: &InstalledPluginMetadata) -> ExternalPluginSpec {
    ExternalPluginSpec {
        name: metadata.name.clone(),
        command: installed_plugin_command(metadata).display().to_string(),
        args: Vec::new(),
        url: None,
        env: BTreeMap::new(),
        startup: PluginStartupOptions::default(),
        web_ui_enabled: None,
        installed_metadata: Some(metadata.clone()),
    }
}

fn optional_configured_plugin_summary(
    entry: &PluginConfigEntry,
    startup: &PluginStartupOptions,
    error: anyhow::Error,
) -> PluginSummary {
    PluginSummary {
        name: entry.name.clone(),
        kind: "external".to_string(),
        enabled: true,
        status: "missing".to_string(),
        pid: None,
        version: None,
        capabilities: Vec::new(),
        command: entry.command.clone(),
        args: entry.args.clone(),
        tools: Vec::new(),
        manifest: None,
        web_ui: derive_plugin_web_ui_state(super::PluginWebUiStateInput {
            plugin_name: &entry.name,
            live_manifest: None,
            installed_metadata: None,
            web_ui_enabled: entry.web_ui_enabled,
            runtime_available: false,
            runtime_unavailable_reason: Some("optional plugin is not loaded"),
        }),
        startup: Some(startup.summary()),
        error: Some(format!("optional plugin not loaded: {error}")),
    }
}

fn installed_plugin_command(metadata: &InstalledPluginMetadata) -> PathBuf {
    metadata.executable_path()
}

fn disabled_installed_plugin_summary(metadata: &InstalledPluginMetadata) -> PluginSummary {
    installed_plugin_summary(metadata, "disabled", metadata.last_error.clone())
}

fn missing_installed_plugin_summary(
    metadata: &InstalledPluginMetadata,
    command: &Path,
) -> PluginSummary {
    installed_plugin_summary(
        metadata,
        "error",
        Some(format!(
            "installed plugin executable is missing: {}",
            command.display()
        )),
    )
}

fn installed_store_error_summary(error: anyhow::Error) -> PluginSummary {
    PluginSummary {
        name: "installed-plugins".to_string(),
        kind: "installed".to_string(),
        enabled: false,
        status: "error".to_string(),
        pid: None,
        version: None,
        capabilities: Vec::new(),
        command: None,
        args: Vec::new(),
        tools: Vec::new(),
        manifest: None,
        web_ui: PluginWebUiState::default(),
        startup: None,
        error: Some(error.to_string()),
    }
}

fn installed_plugin_summary(
    metadata: &InstalledPluginMetadata,
    status: &str,
    error: Option<String>,
) -> PluginSummary {
    PluginSummary {
        name: metadata.name.clone(),
        kind: "installed".to_string(),
        enabled: metadata.enabled,
        status: status.to_string(),
        pid: None,
        version: Some(metadata.installed_version.clone()),
        capabilities: Vec::new(),
        command: Some(installed_plugin_command(metadata).display().to_string()),
        args: Vec::new(),
        tools: Vec::new(),
        manifest: None,
        web_ui: derive_plugin_web_ui_state(super::PluginWebUiStateInput {
            plugin_name: &metadata.name,
            live_manifest: None,
            installed_metadata: Some(metadata),
            web_ui_enabled: None,
            runtime_available: false,
            runtime_unavailable_reason: Some("plugin process is not running"),
        }),
        startup: None,
        error,
    }
}
