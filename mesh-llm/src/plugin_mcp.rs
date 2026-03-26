use anyhow::Result;
use rmcp::{
    model::{
        CallToolResult, Content, Implementation, ListToolsResult, ServerCapabilities, ServerInfo,
        Tool,
    },
    service::{RequestContext, RoleServer},
    transport::io::stdio,
    ErrorData, ServerHandler, ServiceExt,
};
use std::collections::BTreeMap;
use std::sync::Arc;

use crate::plugin::{self, PluginManager};

#[derive(Clone)]
struct PluginToolRef {
    plugin_name: String,
    tool_name: String,
    tool: Tool,
}

#[derive(Clone)]
pub struct PluginMcpServer {
    plugin_manager: PluginManager,
}

impl PluginMcpServer {
    pub fn new(plugin_manager: PluginManager) -> Self {
        Self { plugin_manager }
    }

    async fn discover_tools(&self) -> Result<BTreeMap<String, PluginToolRef>, ErrorData> {
        let mut tools = BTreeMap::new();

        for plugin in self.plugin_manager.list().await {
            if !plugin.enabled || plugin.status != "running" {
                continue;
            }
            for tool_desc in plugin.tools {
                for mcp_name in tool_aliases(&plugin.name, &tool_desc.name) {
                    let schema = parse_input_schema(&tool_desc.input_schema_json);
                    tools.insert(
                        mcp_name.clone(),
                        PluginToolRef {
                            plugin_name: plugin.name.clone(),
                            tool_name: tool_desc.name.clone(),
                            tool: Tool::new(
                                mcp_name,
                                tool_desc.description.clone(),
                                Arc::new(schema),
                            ),
                        },
                    );
                }
            }
        }

        Ok(tools)
    }
}

impl ServerHandler for PluginMcpServer {
    async fn list_tools(
        &self,
        _request: Option<rmcp::model::PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        let tools = self
            .discover_tools()
            .await?
            .into_values()
            .map(|entry| entry.tool)
            .collect();
        Ok(ListToolsResult {
            tools,
            meta: None,
            next_cursor: None,
        })
    }

    fn get_tool(&self, _name: &str) -> Option<Tool> {
        None
    }

    async fn call_tool(
        &self,
        request: rmcp::model::CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, ErrorData> {
        let tools = self.discover_tools().await?;
        let Some(tool_ref) = tools.get(request.name.as_ref()) else {
            return Err(ErrorData::invalid_params(
                format!("Unknown MCP tool '{}'", request.name),
                None,
            ));
        };

        let arguments = request
            .arguments
            .map(serde_json::Value::Object)
            .unwrap_or_else(|| serde_json::json!({}));
        let resp = self
            .plugin_manager
            .call_tool(
                &tool_ref.plugin_name,
                &tool_ref.tool_name,
                &arguments.to_string(),
            )
            .await
            .map_err(internal_error)?;

        if resp.is_error {
            return Ok(CallToolResult::error(vec![Content::text(render_content(
                &resp.content_json,
            ))]));
        }

        match serde_json::from_str::<serde_json::Value>(&resp.content_json) {
            Ok(value) => Ok(CallToolResult::structured(value)),
            Err(_) => Ok(CallToolResult::success(vec![Content::text(resp.content_json)])),
        }
    }

    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(
                Implementation::new("mesh-plugins", env!("CARGO_PKG_VERSION"))
                    .with_title("Mesh Plugin MCP")
                    .with_description(
                        "Automatically re-exposes mesh-llm plugins as MCP tools.",
                    ),
            )
            .with_instructions(
                "Every running plugin tool is exposed as an MCP tool named <plugin>.<tool>. \
                 Blackboard also keeps the legacy aliases blackboard_feed, blackboard_search, and blackboard_post. \
                 Tool calls go directly through mesh-llm to the plugin protobuf transport.",
            )
    }
}

pub async fn run_mcp_server(plugin_manager: PluginManager) -> Result<()> {
    let server = PluginMcpServer::new(plugin_manager);
    let transport = stdio();
    server.serve(transport).await?.waiting().await?;
    Ok(())
}

fn parse_input_schema(input_schema_json: &str) -> serde_json::Map<String, serde_json::Value> {
    serde_json::from_str::<serde_json::Value>(input_schema_json)
        .ok()
        .and_then(|value| value.as_object().cloned())
        .unwrap_or_else(|| {
            serde_json::json!({
                "type": "object",
                "additionalProperties": true
            })
            .as_object()
            .cloned()
            .unwrap()
        })
}

fn internal_error(err: impl std::fmt::Display) -> ErrorData {
    ErrorData::internal_error(err.to_string(), None)
}

fn tool_aliases(plugin_name: &str, tool_name: &str) -> Vec<String> {
    let canonical = format!("{plugin_name}.{tool_name}");
    let mut names = vec![canonical];
    if plugin_name == plugin::BLACKBOARD_PLUGIN_ID {
        names.push(format!("blackboard_{tool_name}"));
    }
    names
}

fn render_content(content_json: &str) -> String {
    serde_json::from_str::<serde_json::Value>(content_json)
        .map(|value| {
            serde_json::to_string_pretty(&value).unwrap_or_else(|_| content_json.to_string())
        })
        .unwrap_or_else(|_| content_json.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_valid_schema() {
        let schema =
            parse_input_schema("{\"type\":\"object\",\"properties\":{\"x\":{\"type\":\"string\"}}}");
        assert_eq!(schema.get("type").and_then(|v| v.as_str()), Some("object"));
    }

    #[test]
    fn falls_back_for_invalid_schema() {
        let schema = parse_input_schema("not-json");
        assert_eq!(schema.get("type").and_then(|v| v.as_str()), Some("object"));
        assert_eq!(
            schema.get("additionalProperties").and_then(|v| v.as_bool()),
            Some(true)
        );
    }

    #[test]
    fn blackboard_aliases_include_legacy_names() {
        let aliases = tool_aliases("blackboard", "feed");
        assert_eq!(aliases, vec!["blackboard.feed", "blackboard_feed"]);
    }

    #[test]
    fn non_blackboard_aliases_only_use_canonical_name() {
        let aliases = tool_aliases("demo", "echo");
        assert_eq!(aliases, vec!["demo.echo"]);
    }
}
