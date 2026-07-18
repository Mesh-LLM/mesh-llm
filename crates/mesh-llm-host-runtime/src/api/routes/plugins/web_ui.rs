use std::{
    collections::BTreeMap,
    path::{Component, Path},
};

use mesh_llm_plugin_manager::{PluginStore, default_store_root};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::net::TcpStream;

use crate::{
    api::{
        MeshApi,
        http::{respond_bytes_cached, respond_error, respond_json},
    },
    plugin::{PluginWebUiState, PluginWebUiStateKind},
};

use super::{PluginWebUiSubroute, parse_plugin_web_ui_route};

pub(super) async fn handle_metadata(
    stream: &mut TcpStream,
    state: &MeshApi,
    path: &str,
) -> anyhow::Result<()> {
    let Some((plugin_name, PluginWebUiSubroute::Metadata)) = parse_plugin_web_ui_route(path) else {
        return respond_error(stream, 404, "Not found").await;
    };
    let plugin_manager = state.inner.lock().await.plugin_manager.clone();
    match plugin_manager.web_ui_state(plugin_name).await {
        Ok(web_ui) => respond_json(stream, 200, &web_ui).await?,
        Err(error) => respond_error(stream, 404, &error.to_string()).await?,
    }
    Ok(())
}

#[derive(Debug, Deserialize)]
struct WebUiEnabledRequest {
    enabled: bool,
}

pub(super) async fn handle_enabled(
    stream: &mut TcpStream,
    state: &MeshApi,
    path: &str,
    body: &str,
) -> anyhow::Result<()> {
    let Some((plugin_name, PluginWebUiSubroute::Enabled)) = parse_plugin_web_ui_route(path) else {
        return respond_error(stream, 404, "Not found").await;
    };
    let request: WebUiEnabledRequest = match serde_json::from_str(body) {
        Ok(request) => request,
        Err(_) => return respond_error(stream, 400, "Invalid JSON body").await,
    };
    let plugin_manager = state.inner.lock().await.plugin_manager.clone();
    let current = match plugin_manager.web_ui_state(plugin_name).await {
        Ok(web_ui) => web_ui,
        Err(error) => return respond_error(stream, 404, &error.to_string()).await,
    };
    if !current.declared {
        return respond_error(stream, 400, "Plugin does not declare a web UI").await;
    }
    let web_ui = match plugin_manager
        .set_web_ui_enabled(plugin_name, request.enabled)
        .await
    {
        Ok(web_ui) => web_ui,
        Err(error) => return respond_error(stream, 404, &error.to_string()).await,
    };
    if let Err(error) = state
        .capture_node
        .set_plugin_web_ui_enabled(plugin_name, request.enabled)
        .await
    {
        if let Err(rollback_error) = plugin_manager
            .set_web_ui_enabled(plugin_name, current.enabled)
            .await
        {
            return respond_error(
                stream,
                500,
                &format!("{error}; failed to roll back runtime web UI state: {rollback_error}"),
            )
            .await;
        }
        return respond_error(stream, 500, &error.to_string()).await;
    }
    respond_json(stream, 200, &web_ui).await?;
    Ok(())
}

pub(super) async fn handle_asset(
    stream: &mut TcpStream,
    state: &MeshApi,
    path: &str,
) -> anyhow::Result<()> {
    let Some((plugin_name, asset_path)) = parse_asset_path(path) else {
        respond_error(stream, 404, "Not found").await?;
        return Ok(());
    };
    let plugin_manager = state.inner.lock().await.plugin_manager.clone();
    let web_ui = match plugin_manager.web_ui_state(plugin_name).await {
        Ok(web_ui) => web_ui,
        Err(_) => {
            respond_error(stream, 404, "Not found").await?;
            return Ok(());
        }
    };
    match web_ui.state {
        PluginWebUiStateKind::Ready => {
            serve_ready_asset(stream, &plugin_manager, plugin_name, asset_path, &web_ui).await?
        }
        PluginWebUiStateKind::Invalid | PluginWebUiStateKind::PluginNotRunning => {
            let reason = web_ui
                .unavailable_reason
                .as_deref()
                .unwrap_or("plugin web UI is unavailable");
            respond_error(stream, 409, reason).await?;
        }
        PluginWebUiStateKind::None | PluginWebUiStateKind::Disabled => {
            respond_error(stream, 404, "Not found").await?;
        }
    }
    Ok(())
}

async fn serve_ready_asset(
    stream: &mut TcpStream,
    plugin_manager: &crate::plugin::PluginManager,
    plugin_name: &str,
    asset_path: &str,
    web_ui: &PluginWebUiState,
) -> anyhow::Result<()> {
    if web_ui.asset_base_url.is_none() {
        respond_error(stream, 409, "plugin web UI asset root is unavailable").await?;
        return Ok(());
    }
    let Some(rel) = clean_asset_path(asset_path) else {
        respond_error(stream, 404, "Not found").await?;
        return Ok(());
    };
    let Some(root) = plugin_manager.web_ui_asset_root(plugin_name).await? else {
        respond_error(stream, 409, "plugin web UI asset root is unavailable").await?;
        return Ok(());
    };
    let Some(full_path) = canonical_asset_path(&root, &rel).await else {
        respond_error(stream, 404, "Not found").await?;
        return Ok(());
    };
    match tokio::fs::read(&full_path).await {
        Ok(contents) => {
            respond_bytes_cached(
                stream,
                200,
                "OK",
                mesh_llm_ui::content_type(&rel),
                "no-cache",
                &contents,
            )
            .await?;
        }
        Err(_) => respond_error(stream, 404, "Not found").await?,
    }
    Ok(())
}

fn parse_asset_path(path: &str) -> Option<(&str, &str)> {
    match parse_plugin_web_ui_route(path)? {
        (plugin_name, PluginWebUiSubroute::Asset(asset_path)) => Some((plugin_name, asset_path)),
        _ => None,
    }
}

#[derive(Debug, Deserialize)]
struct WebUiConfigMutationRequest {
    #[serde(default)]
    plugin: Option<String>,
    #[serde(default)]
    settings: BTreeMap<String, Value>,
    #[serde(default)]
    unset: Vec<String>,
}

#[derive(Debug, Serialize)]
struct PluginWebUiConfigResponse {
    plugin: String,
    settings: BTreeMap<String, toml::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    schema: Option<Value>,
}

pub(super) async fn handle_config(
    stream: &mut TcpStream,
    state: &MeshApi,
    method: &str,
    path: &str,
    body: &str,
) -> anyhow::Result<()> {
    let Some((plugin_name, PluginWebUiSubroute::Config)) = parse_plugin_web_ui_route(path) else {
        return respond_error(stream, 404, "Not found").await;
    };
    let plugin_manager = state.inner.lock().await.plugin_manager.clone();
    if let Err(error) = plugin_manager.web_ui_state(plugin_name).await {
        return respond_error(stream, 404, &error.to_string()).await;
    }

    match method {
        "GET" => {
            let response = config_response(state, plugin_name).await;
            respond_json(stream, 200, &response).await
        }
        "PATCH" => {
            let request = match parse_config_mutation(plugin_name, body) {
                Ok(request) => request,
                Err(error) => return respond_error(stream, 400, &error).await,
            };
            match state
                .capture_node
                .patch_plugin_settings(plugin_name, request.0, request.1)
                .await
            {
                Ok(settings) => {
                    let response = PluginWebUiConfigResponse {
                        plugin: plugin_name.to_string(),
                        settings,
                        schema: installed_config_schema_json(plugin_name).await,
                    };
                    respond_json(stream, 200, &response).await
                }
                Err(error) => respond_error(stream, 422, &error.to_string()).await,
            }
        }
        _ => respond_error(stream, 405, "Method not allowed").await,
    }
}

async fn config_response(state: &MeshApi, plugin_name: &str) -> PluginWebUiConfigResponse {
    PluginWebUiConfigResponse {
        plugin: plugin_name.to_string(),
        settings: state.capture_node.plugin_settings(plugin_name).await,
        schema: installed_config_schema_json(plugin_name).await,
    }
}

fn parse_config_mutation(
    plugin_name: &str,
    body: &str,
) -> Result<(BTreeMap<String, toml::Value>, Vec<String>), String> {
    let request: WebUiConfigMutationRequest =
        serde_json::from_str(body).map_err(|_| "Invalid JSON body".to_string())?;
    if let Some(request_plugin) = request.plugin.as_deref()
        && request_plugin != plugin_name
    {
        return Err(format!(
            "Config mutation plugin '{request_plugin}' does not match mounted plugin '{plugin_name}'"
        ));
    }

    let mut settings = BTreeMap::new();
    for (key, value) in request.settings {
        validate_setting_key(&key)?;
        if value.is_null() {
            return Err(format!(
                "Plugin setting '{key}' cannot be null; use unset instead"
            ));
        }
        let value = toml::Value::try_from(value).map_err(|error| {
            format!("Plugin setting '{key}' has unsupported value type: {error}")
        })?;
        settings.insert(key, value);
    }
    for key in &request.unset {
        validate_setting_key(key)?;
    }
    Ok((settings, request.unset))
}

fn validate_setting_key(key: &str) -> Result<(), String> {
    let trimmed = key.trim();
    if trimmed.is_empty() {
        return Err("Plugin setting key cannot be empty".into());
    }
    if matches!(
        trimmed,
        "enabled" | "web_ui_enabled" | "command" | "args" | "url" | "startup"
    ) {
        return Err(format!(
            "Plugin setting key '{trimmed}' targets a host-owned field"
        ));
    }
    if !trimmed
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '-')
    {
        return Err(format!(
            "Plugin setting key '{trimmed}' must contain only ASCII letters, numbers, '-' or '_'"
        ));
    }
    Ok(())
}

async fn installed_config_schema_json(plugin_name: &str) -> Option<Value> {
    let plugin_name = plugin_name.to_string();
    tokio::task::spawn_blocking(move || {
        let root = default_store_root().ok()?;
        let metadata = PluginStore::new(root).load_optional(&plugin_name).ok()??;
        let schema = metadata.manifest?.config_schema?;
        serde_json::to_value(schema).ok()
    })
    .await
    .ok()
    .flatten()
}

fn clean_asset_path(path: &str) -> Option<String> {
    let decoded = urlencoding::decode(path).ok()?;
    if decoded.contains("..") {
        return None;
    }
    let rel = decoded.trim_start_matches('/');
    if rel.is_empty() || rel.starts_with('.') || Path::new(rel).is_absolute() {
        return None;
    }
    if Path::new(rel)
        .components()
        .any(|component| match component {
            Component::Normal(name) => name.to_string_lossy().starts_with('.'),
            Component::CurDir
            | Component::ParentDir
            | Component::RootDir
            | Component::Prefix(_) => true,
        })
    {
        return None;
    }
    Some(rel.to_string())
}

async fn canonical_asset_path(root: &Path, rel: &str) -> Option<std::path::PathBuf> {
    let root = tokio::fs::canonicalize(root).await.ok()?;
    let full_path = tokio::fs::canonicalize(root.join(rel)).await.ok()?;
    if !full_path.starts_with(&root) || !tokio::fs::metadata(&full_path).await.ok()?.is_file() {
        return None;
    }
    Some(full_path)
}
