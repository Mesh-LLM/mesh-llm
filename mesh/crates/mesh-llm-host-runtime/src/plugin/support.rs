use super::{ToolSummary, proto};
use anyhow::{Context, Result};
use rmcp::model::ServerConfig;
use serde::Serialize;

pub(crate) fn serialize_params<T: Serialize>(params: T) -> Result<String> {
    serde_json::to_string(&params).context("serialize plugin RPC params")
}

pub(crate) fn parse_optional_json(raw: &str) -> Result<Option<serde_json::Value>> {
    if raw.trim().is_empty() {
        Ok(None)
    } else {
        Ok(Some(
            serde_json::from_str(raw).context("parse plugin JSON payload")?,
        ))
    }
}

/// A plugin answered a host request with a JSON-RPC error. The code tells a
/// route whether the caller did something wrong or the plugin did, so it has
/// to survive the trip out of the plugin layer as a value, not as text.
#[derive(Debug)]
pub(crate) struct PluginRpcFailure {
    pub(crate) code: i32,
}

impl std::fmt::Display for PluginRpcFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "plugin JSON-RPC error {}", self.code)
    }
}

impl std::error::Error for PluginRpcFailure {}

/// The plugin did not answer within its deadline. Unlike a transport failure
/// the request did reach the plugin, so sending it again would start the same
/// work a second time in a fresh process.
#[derive(Debug)]
pub(crate) struct PluginTimeout {
    pub(crate) plugin_name: String,
}

impl std::fmt::Display for PluginTimeout {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "Plugin '{}' timed out", self.plugin_name)
    }
}

impl std::error::Error for PluginTimeout {}

pub(crate) fn plugin_error(
    plugin_name: &str,
    method: &str,
    err: &proto::ErrorResponse,
) -> anyhow::Error {
    let detail = if err.data_json.trim().is_empty() {
        format!(
            "Plugin '{}' failed '{}' (code {}): {}",
            plugin_name, method, err.code, err.message
        )
    } else {
        format!(
            "Plugin '{}' failed '{}' (code {}): {} ({})",
            plugin_name, method, err.code, err.message, err.data_json
        )
    };
    anyhow::Error::new(PluginRpcFailure { code: err.code }).context(detail)
}

pub(crate) fn summarize_capabilities(server_info: &ServerConfig, extra: &[String]) -> Vec<String> {
    let mut capabilities = extra.to_vec();
    let caps = &server_info.capabilities;
    if caps.tools.is_some() {
        capabilities.push("mcp:tools".into());
    }
    if caps.prompts.is_some() {
        capabilities.push("mcp:prompts".into());
    }
    if caps.resources.is_some() {
        capabilities.push("mcp:resources".into());
    }
    if caps.completions.is_some() {
        capabilities.push("mcp:completions".into());
    }
    if caps.logging.is_some() {
        capabilities.push("mcp:logging".into());
    }
    if caps.supports_tasks() {
        capabilities.push("mcp:tasks".into());
    }
    if let Some(extensions) = &caps.extensions {
        for key in extensions.keys() {
            capabilities.push(format!("mcp:extension:{key}"));
        }
    }
    capabilities.sort();
    capabilities.dedup();
    capabilities
}

pub(crate) fn format_args_for_log(args: &[String]) -> String {
    if args.is_empty() {
        "[]".to_string()
    } else {
        format!("[{}]", args.join(", "))
    }
}

pub(crate) fn format_slice_for_log(values: &[String]) -> String {
    if values.is_empty() {
        "[]".to_string()
    } else {
        format!("[{}]", values.join(", "))
    }
}

pub(crate) fn format_tool_names_for_log(tools: &[ToolSummary]) -> String {
    let names = tools
        .iter()
        .map(|tool| tool.name.clone())
        .collect::<Vec<_>>();
    format_slice_for_log(&names)
}

#[cfg(test)]
mod tests {
    use super::summarize_capabilities;
    use rmcp::model::{ServerCapabilities, ServerConfig};

    #[test]
    fn task_capability_is_detected_from_rmcp_extension() {
        let server_info = ServerConfig::new(ServerCapabilities::builder().enable_tasks().build());

        let capabilities = summarize_capabilities(&server_info, &[]);

        assert!(capabilities.iter().any(|entry| entry == "mcp:tasks"));
    }
}
