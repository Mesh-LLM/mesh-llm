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
    Some(disabled_entry_summary(entry, &metadata))
}

fn disabled_entry_summary(
    entry: &PluginConfigEntry,
    metadata: &InstalledPluginMetadata,
) -> PluginSummary {
    PluginSummary {
        name: entry.name.clone(),
        kind: "installed".to_string(),
        enabled: false,
        status: "disabled".to_string(),
        pid: None,
        version: Some(metadata.installed_version.clone()),
        capabilities: Vec::new(),
        command: Some(installed_plugin_command(metadata).display().to_string()),
        args: entry.args.clone(),
        tools: Vec::new(),
        manifest: None,
        web_ui: derive_plugin_web_ui_state(super::PluginWebUiStateInput {
            plugin_name: &entry.name,
            live_manifest: None,
            installed_metadata: Some(metadata),
            web_ui_enabled: entry.web_ui_enabled,
            web_ui_primary_tab: entry.web_ui_primary_tab_preference(),
            runtime_available: false,
            runtime_unavailable_reason: Some("plugin process is disabled"),
        }),
        startup: None,
        error: None,
    }
}

pub(crate) fn configured_external_plugin_spec(
    entry: &PluginConfigEntry,
) -> Result<ConfiguredExternalPlugin> {
    let url = entry.url.as_deref().map(str::trim);
    if url == Some("") {
        bail!("plugin URL must not be empty");
    }
    if url.is_some_and(|url| url::Url::parse(url).is_ok_and(|url| url.scheme() == "tcp")) {
        anyhow::bail!(
            "tcp:// plugin control is unsupported because it has no authenticated capability handshake"
        );
    }
    let command = entry
        .command
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    let mut startup = PluginStartupOptions::from_config(&entry.startup);
    // Package assets and identity still belong to an explicitly selected
    // installed executable, but never to an unrelated same-name command.
    let installed_metadata = installed_plugin_metadata_for_name(&entry.name)
        .ok()
        .flatten()
        .filter(|metadata| {
            command
                .as_deref()
                .is_none_or(|command| Path::new(command) == metadata.executable_path())
        });
    if command.is_none()
        && installed_metadata
            .as_ref()
            .is_some_and(|metadata| metadata.default_managed)
    {
        startup.optional = true;
    }

    // An entry without its own command runs the store's copy, so the store's
    // enabled flag (`mesh-llm plugins disable`) decides whether it starts.
    if command.is_none()
        && let Some(metadata) = installed_metadata
            .as_ref()
            .filter(|metadata| !metadata.enabled)
    {
        return Ok(ConfiguredExternalPlugin::Inactive(disabled_entry_summary(
            entry, metadata,
        )));
    }

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
        openai_exchange_grant: entry.openai_exchange_grant.clone().map(Box::new),
        name: entry.name.clone(),
        command,
        args: entry.args.clone(),
        url: url.map(str::to_owned),
        env: BTreeMap::new(),
        startup,
        web_ui_enabled: entry.web_ui_enabled,
        web_ui_primary_tab: entry.web_ui_primary_tab,
        endpoint_models: entry.models.clone(),
        installed_metadata,
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
        openai_exchange_grant: None,
        name: metadata.name.clone(),
        command: installed_plugin_command(metadata).display().to_string(),
        args: Vec::new(),
        url: None,
        env: BTreeMap::new(),
        startup: PluginStartupOptions {
            optional: metadata.default_managed,
            ..PluginStartupOptions::default()
        },
        web_ui_enabled: None,
        web_ui_primary_tab: None,
        endpoint_models: None,
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
            web_ui_primary_tab: entry.web_ui_primary_tab_preference(),
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
            web_ui_primary_tab: false,
            runtime_available: false,
            runtime_unavailable_reason: Some("plugin process is not running"),
        }),
        startup: None,
        error,
    }
}

#[cfg(test)]
#[path = "installed_tests.rs"]
mod tests;
