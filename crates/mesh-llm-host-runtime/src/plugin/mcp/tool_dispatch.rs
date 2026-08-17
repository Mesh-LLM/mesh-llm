use anyhow::Result;
use rmcp::{
    ErrorData, RoleServer, ServerHandler,
    model::{
        CallToolRequestParams, CallToolResult, CancelTaskParams, CancelTaskResult,
        CompleteRequestParams, CompleteResult, CustomNotification, CustomRequest,
        GetPromptRequestParams, GetPromptResult, GetTaskInfoParams, GetTaskPayloadResult,
        GetTaskResult, GetTaskResultParams, Implementation, ListPromptsResult,
        ListResourceTemplatesResult, ListResourcesResult, ListTasksResult, ListToolsResult,
        PaginatedRequestParams, ReadResourceRequestParams, ReadResourceResult, ServerCapabilities,
        ServerInfo, SetLevelRequestParams, SubscribeRequestParams, UnsubscribeRequestParams,
    },
    service::{NotificationContext, RequestContext},
};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;
use std::sync::Arc;

use crate::plugin::{self, stapler};

use super::external_mcp::{ExternalMcpClient, ExternalMcpEndpoint};
use super::{PluginMcpServer, internal_error, try_plugins};

#[derive(Clone)]
enum ToolTarget {
    Plugin {
        plugin_name: String,
        tool_name: String,
    },
    External {
        endpoint: ExternalMcpEndpoint,
        tool_name: String,
    },
}

#[derive(Clone)]
struct ToolRef {
    target: ToolTarget,
    tool: rmcp::model::Tool,
}

fn normalize_tool_schema(mut tool: rmcp::model::Tool) -> rmcp::model::Tool {
    tool.input_schema = Arc::new(normalize_input_schema((*tool.input_schema).clone()));
    if tool
        .output_schema
        .as_deref()
        .is_some_and(|schema| schema.get("type").and_then(Value::as_str) != Some("object"))
    {
        tool.output_schema = None;
    }
    tool
}

fn normalize_input_schema(
    mut schema: serde_json::Map<String, serde_json::Value>,
) -> serde_json::Map<String, serde_json::Value> {
    if schema.get("type").and_then(Value::as_str) == Some("object") {
        return schema;
    }
    if schema.contains_key("properties") {
        schema.insert("type".to_string(), serde_json::json!("object"));
        return schema;
    }
    serde_json::json!({
        "type": "object",
        "additionalProperties": true,
    })
    .as_object()
    .cloned()
    .expect("object schema")
}

#[derive(Clone)]
enum PromptTarget {
    Plugin {
        plugin_name: String,
        prompt_name: String,
    },
    External {
        endpoint: ExternalMcpEndpoint,
        prompt_name: String,
    },
}

#[derive(Clone)]
struct PromptRef {
    target: PromptTarget,
}

#[derive(Clone)]
enum ResourceTarget {
    Plugin {
        plugin_name: String,
        resource_uri: String,
    },
    External {
        endpoint: ExternalMcpEndpoint,
        original_uri: String,
    },
}

#[derive(Clone)]
struct ResourceRef {
    target: ResourceTarget,
}

impl PluginMcpServer {
    async fn active_external_mcp_endpoints(&self) -> Result<Vec<ExternalMcpEndpoint>, ErrorData> {
        let passive_endpoint_summaries = self
            .plugin_manager
            .endpoints()
            .await
            .map_err(internal_error)?;
        let endpoints = passive_endpoint_summaries
            .into_iter()
            .filter_map(ExternalMcpEndpoint::from_summary)
            .collect::<Vec<_>>();
        let active_keys = endpoints
            .iter()
            .map(|endpoint| endpoint.key.clone())
            .collect::<BTreeSet<_>>();
        self.external_mcp.retain_active(&active_keys).await;
        Ok(endpoints)
    }

    async fn collect_external_items<T, Fetch, Fut>(
        &self,
        client_skip_message: &'static str,
        list_fail_message: &'static str,
        mut fetch: Fetch,
    ) -> Result<Vec<(ExternalMcpEndpoint, Vec<T>)>, ErrorData>
    where
        Fetch: FnMut(Arc<ExternalMcpClient>) -> Fut,
        Fut: Future<Output = Result<Vec<T>>>,
    {
        let mut items = Vec::new();
        for endpoint in self.active_external_mcp_endpoints().await? {
            if let Some(item) = self
                .collect_external_items_for_endpoint(
                    endpoint,
                    client_skip_message,
                    list_fail_message,
                    &mut fetch,
                )
                .await
            {
                items.push(item);
            }
        }
        Ok(items)
    }

    async fn collect_external_items_for_endpoint<T, Fetch, Fut>(
        &self,
        endpoint: ExternalMcpEndpoint,
        client_skip_message: &'static str,
        list_fail_message: &'static str,
        fetch: &mut Fetch,
    ) -> Option<(ExternalMcpEndpoint, Vec<T>)>
    where
        Fetch: FnMut(Arc<ExternalMcpClient>) -> Fut,
        Fut: Future<Output = Result<Vec<T>>>,
    {
        let client = match self.external_mcp.client_for(&endpoint).await {
            Ok(client) => client,
            Err(err) => {
                tracing::warn!(
                    plugin = %endpoint.plugin_name,
                    endpoint = %endpoint.endpoint_id,
                    error = %err,
                    "{client_skip_message}"
                );
                return None;
            }
        };
        let listed = match fetch(client).await {
            Ok(listed) => listed,
            Err(err) => {
                tracing::warn!(
                    plugin = %endpoint.plugin_name,
                    endpoint = %endpoint.endpoint_id,
                    error = %err,
                    "{list_fail_message}"
                );
                return None;
            }
        };
        Some((endpoint, listed))
    }

    async fn discover_tools(&self) -> Result<BTreeMap<String, ToolRef>, ErrorData> {
        let mut tools = BTreeMap::new();
        for (plugin_name, manifest) in self.plugin_manifests().await? {
            if manifest.operations.is_empty() {
                continue;
            }
            for operation in &manifest.operations {
                let raw_name = operation.name.clone();
                for mcp_name in tool_aliases(&plugin_name, &raw_name) {
                    tools.insert(
                        mcp_name.clone(),
                        ToolRef {
                            target: ToolTarget::Plugin {
                                plugin_name: plugin_name.clone(),
                                tool_name: raw_name.clone(),
                            },
                            tool: normalize_tool_schema(stapler::operation(mcp_name, operation)),
                        },
                    );
                }
            }
        }
        for (endpoint, listed) in self
            .collect_external_items(
                "Skipping external MCP endpoint during tool discovery",
                "Failed to list tools from external MCP endpoint",
                |client| async move {
                    client
                        .peer
                        .list_all_tools()
                        .await
                        .map_err(anyhow::Error::from)
                },
            )
            .await?
        {
            for tool in listed {
                let raw_name = tool.name.to_string();
                let canonical_name = endpoint.canonical_name(&raw_name);
                let mut namespaced = normalize_tool_schema(tool.clone());
                namespaced.name = canonical_name.clone().into();
                tools.insert(
                    canonical_name,
                    ToolRef {
                        target: ToolTarget::External {
                            endpoint: endpoint.clone(),
                            tool_name: raw_name,
                        },
                        tool: namespaced,
                    },
                );
            }
        }
        Ok(tools)
    }

    async fn discover_prompts(&self) -> Result<BTreeMap<String, PromptRef>, ErrorData> {
        let mut prompts = BTreeMap::new();
        for (plugin_name, manifest) in self.plugin_manifests().await? {
            if manifest.prompts.is_empty() {
                continue;
            }
            for prompt in &manifest.prompts {
                prompts.insert(
                    canonical_name(&plugin_name, &prompt.name),
                    PromptRef {
                        target: PromptTarget::Plugin {
                            plugin_name: plugin_name.clone(),
                            prompt_name: prompt.name.clone(),
                        },
                    },
                );
            }
        }
        for (endpoint, listed) in self
            .collect_external_items(
                "Skipping external MCP endpoint during prompt discovery",
                "Failed to list prompts from external MCP endpoint",
                |client| async move {
                    client
                        .peer
                        .list_all_prompts()
                        .await
                        .map_err(anyhow::Error::from)
                },
            )
            .await?
        {
            for prompt in listed {
                prompts.insert(
                    endpoint.canonical_name(&prompt.name),
                    PromptRef {
                        target: PromptTarget::External {
                            endpoint: endpoint.clone(),
                            prompt_name: prompt.name,
                        },
                    },
                );
            }
        }
        Ok(prompts)
    }

    async fn discover_resources(&self) -> Result<BTreeMap<String, ResourceRef>, ErrorData> {
        let mut resources = BTreeMap::new();
        for (plugin_name, manifest) in self.plugin_manifests().await? {
            if manifest.resources.is_empty() {
                continue;
            }
            for resource in manifest.resources {
                resources.insert(
                    resource.uri.clone(),
                    ResourceRef {
                        target: ResourceTarget::Plugin {
                            plugin_name: plugin_name.clone(),
                            resource_uri: resource.uri,
                        },
                    },
                );
            }
        }
        for (endpoint, listed) in self
            .collect_external_items(
                "Skipping external MCP endpoint during resource discovery",
                "Failed to list resources from external MCP endpoint",
                |client| async move {
                    client
                        .peer
                        .list_all_resources()
                        .await
                        .map_err(anyhow::Error::from)
                },
            )
            .await?
        {
            for resource in listed {
                resources.insert(
                    endpoint.canonical_resource_uri(&resource.raw.uri),
                    ResourceRef {
                        target: ResourceTarget::External {
                            endpoint: endpoint.clone(),
                            original_uri: resource.raw.uri,
                        },
                    },
                );
            }
        }
        Ok(resources)
    }
}

impl ServerHandler for PluginMcpServer {
    async fn initialize(
        &self,
        request: rmcp::model::InitializeRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<ServerInfo, ErrorData> {
        if context.peer.peer_info().is_none() {
            context.peer.set_peer_info(request);
        }
        self.refresh_peer(context.peer.clone()).await;
        Ok(self.get_info())
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        self.refresh_peer(context.peer.clone()).await;
        Ok(ListToolsResult {
            tools: self
                .discover_tools()
                .await?
                .into_values()
                .map(|entry| entry.tool)
                .collect(),
            meta: None,
            next_cursor: None,
        })
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, ErrorData> {
        self.refresh_peer(context.peer.clone()).await;
        let tools = self.discover_tools().await?;
        let Some(tool_ref) = tools.get(request.name.as_ref()) else {
            return Err(ErrorData::invalid_params(
                format!("Unknown MCP tool '{}'", request.name),
                None,
            ));
        };
        match &tool_ref.target {
            ToolTarget::Plugin {
                plugin_name,
                tool_name,
            } => {
                let arguments = request
                    .arguments
                    .map(Value::Object)
                    .unwrap_or_else(|| serde_json::json!({}));
                let result = self
                    .plugin_manager
                    .invoke_operation_without_timeout(
                        plugin_name,
                        tool_name,
                        &arguments.to_string(),
                    )
                    .await
                    .map_err(internal_error)?;
                Ok(operation_result_to_call_tool_result(result))
            }
            ToolTarget::External {
                endpoint,
                tool_name,
            } => {
                let client = self.external_mcp.client_for(endpoint).await?;
                let mut params = CallToolRequestParams::new(tool_name.clone());
                if let Some(arguments) = request.arguments {
                    params = params.with_arguments(arguments);
                }
                if let Some(task) = request.task {
                    params = params.with_task(task);
                }
                if let Some(meta) = request.meta {
                    params.meta = Some(meta);
                }
                client.peer.call_tool(params).await.map_err(internal_error)
            }
        }
    }

    async fn list_prompts(
        &self,
        _request: Option<PaginatedRequestParams>,
        context: RequestContext<RoleServer>,
    ) -> Result<ListPromptsResult, ErrorData> {
        self.refresh_peer(context.peer.clone()).await;
        let mut prompts = Vec::new();
        for (plugin_name, manifest) in self.plugin_manifests().await? {
            if manifest.prompts.is_empty() {
                continue;
            }
            prompts.extend(manifest.prompts.into_iter().map(|prompt| {
                stapler::prompt(canonical_name(&plugin_name, &prompt.name), &prompt)
            }));
        }
        for (endpoint, listed) in self
            .collect_external_items(
                "Skipping external MCP endpoint during prompt listing",
                "Failed to list prompts from external MCP endpoint",
                |client| async move {
                    client
                        .peer
                        .list_all_prompts()
                        .await
                        .map_err(anyhow::Error::from)
                },
            )
            .await?
        {
            prompts.extend(listed.into_iter().map(|mut prompt| {
                prompt.name = endpoint.canonical_name(&prompt.name);
                prompt
            }));
        }
        Ok(ListPromptsResult {
            prompts,
            meta: None,
            next_cursor: None,
        })
    }

    async fn get_prompt(
        &self,
        request: GetPromptRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<GetPromptResult, ErrorData> {
        self.refresh_peer(context.peer.clone()).await;
        let prompts = self.discover_prompts().await?;
        let Some(entry) = prompts.get(request.name.as_str()) else {
            return Err(ErrorData::invalid_params(
                format!("Unknown MCP prompt '{}'", request.name),
                None,
            ));
        };

        match &entry.target {
            PromptTarget::Plugin {
                plugin_name,
                prompt_name,
            } => {
                let mut params = GetPromptRequestParams::new(prompt_name.clone());
                if let Some(arguments) = request.arguments {
                    params = params.with_arguments(arguments);
                }
                if let Some(meta) = request.meta {
                    params.meta = Some(meta);
                }

                self.plugin_manager
                    .get_prompt(plugin_name, prompt_name, params)
                    .await
                    .map_err(internal_error)
            }
            PromptTarget::External {
                endpoint,
                prompt_name,
            } => {
                let client = self.external_mcp.client_for(endpoint).await?;
                let mut params = GetPromptRequestParams::new(prompt_name.clone());
                if let Some(arguments) = request.arguments {
                    params = params.with_arguments(arguments);
                }
                if let Some(meta) = request.meta {
                    params.meta = Some(meta);
                }
                client.peer.get_prompt(params).await.map_err(internal_error)
            }
        }
    }

    async fn list_resources(
        &self,
        _request: Option<PaginatedRequestParams>,
        context: RequestContext<RoleServer>,
    ) -> Result<ListResourcesResult, ErrorData> {
        self.refresh_peer(context.peer.clone()).await;
        let mut resources = Vec::new();
        for (_, manifest) in self.plugin_manifests().await? {
            if manifest.resources.is_empty() {
                continue;
            }
            resources.extend(manifest.resources.iter().map(stapler::resource));
        }
        for (endpoint, listed) in self
            .collect_external_items(
                "Skipping external MCP endpoint during resource listing",
                "Failed to list resources from external MCP endpoint",
                |client| async move {
                    client
                        .peer
                        .list_all_resources()
                        .await
                        .map_err(anyhow::Error::from)
                },
            )
            .await?
        {
            resources.extend(listed.into_iter().map(|mut resource| {
                resource.raw.name = endpoint.canonical_name(&resource.raw.name);
                resource.raw.uri = endpoint.canonical_resource_uri(&resource.raw.uri);
                resource
            }));
        }
        Ok(ListResourcesResult {
            resources,
            meta: None,
            next_cursor: None,
        })
    }

    async fn list_resource_templates(
        &self,
        _request: Option<PaginatedRequestParams>,
        context: RequestContext<RoleServer>,
    ) -> Result<ListResourceTemplatesResult, ErrorData> {
        self.refresh_peer(context.peer.clone()).await;
        let mut resource_templates = Vec::new();
        for (_, manifest) in self.plugin_manifests().await? {
            if manifest.resource_templates.is_empty() {
                continue;
            }
            resource_templates.extend(
                manifest
                    .resource_templates
                    .iter()
                    .map(stapler::resource_template),
            );
        }
        for (endpoint, listed) in self
            .collect_external_items(
                "Skipping external MCP endpoint during resource template listing",
                "Failed to list resource templates from external MCP endpoint",
                |client| async move {
                    client
                        .peer
                        .list_all_resource_templates()
                        .await
                        .map_err(anyhow::Error::from)
                },
            )
            .await?
        {
            resource_templates.extend(listed.into_iter().map(|mut template| {
                template.raw.name = endpoint.canonical_name(&template.raw.name);
                template.raw.uri_template =
                    endpoint.canonical_resource_template_uri(&template.raw.uri_template);
                template
            }));
        }
        Ok(ListResourceTemplatesResult {
            resource_templates,
            meta: None,
            next_cursor: None,
        })
    }

    async fn read_resource(
        &self,
        request: ReadResourceRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<ReadResourceResult, ErrorData> {
        self.refresh_peer(context.peer.clone()).await;
        if let Some(resource_ref) = self.discover_resources().await?.get(&request.uri).cloned() {
            match resource_ref.target {
                ResourceTarget::Plugin {
                    plugin_name,
                    resource_uri,
                } => {
                    let mut params = ReadResourceRequestParams::new(resource_uri);
                    if let Some(meta) = request.meta {
                        params.meta = Some(meta);
                    }
                    return self
                        .plugin_manager
                        .read_resource(&plugin_name, &request.uri, params)
                        .await
                        .map_err(internal_error);
                }
                ResourceTarget::External {
                    endpoint,
                    original_uri,
                } => {
                    let client = self.external_mcp.client_for(&endpoint).await?;
                    let mut params = ReadResourceRequestParams::new(original_uri);
                    if let Some(meta) = request.meta {
                        params.meta = Some(meta);
                    }
                    let mut result = client
                        .peer
                        .read_resource(params)
                        .await
                        .map_err(internal_error)?;
                    for content in &mut result.contents {
                        match content {
                            rmcp::model::ResourceContents::TextResourceContents { uri, .. }
                            | rmcp::model::ResourceContents::BlobResourceContents { uri, .. } => {
                                *uri = request.uri.clone();
                            }
                        }
                    }
                    return Ok(result);
                }
            }
        }
        try_plugins(&self.plugin_manager, "resources/read", request).await
    }

    async fn subscribe(
        &self,
        request: SubscribeRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<(), ErrorData> {
        self.refresh_peer(context.peer.clone()).await;
        if let Some(resource_ref) = self.discover_resources().await?.get(&request.uri).cloned() {
            match resource_ref.target {
                ResourceTarget::Plugin { .. } => {}
                ResourceTarget::External {
                    endpoint,
                    original_uri,
                } => {
                    let client = self.external_mcp.client_for(&endpoint).await?;
                    let mut params = SubscribeRequestParams::new(original_uri);
                    if let Some(meta) = request.meta {
                        params.meta = Some(meta);
                    }
                    return client.peer.subscribe(params).await.map_err(internal_error);
                }
            }
        }
        try_plugins::<(), _>(&self.plugin_manager, "resources/subscribe", request).await
    }

    async fn unsubscribe(
        &self,
        request: UnsubscribeRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<(), ErrorData> {
        self.refresh_peer(context.peer.clone()).await;
        if let Some(resource_ref) = self.discover_resources().await?.get(&request.uri).cloned() {
            match resource_ref.target {
                ResourceTarget::Plugin { .. } => {}
                ResourceTarget::External {
                    endpoint,
                    original_uri,
                } => {
                    let client = self.external_mcp.client_for(&endpoint).await?;
                    let mut params = UnsubscribeRequestParams::new(original_uri);
                    if let Some(meta) = request.meta {
                        params.meta = Some(meta);
                    }
                    return client
                        .peer
                        .unsubscribe(params)
                        .await
                        .map_err(internal_error);
                }
            }
        }
        try_plugins::<(), _>(&self.plugin_manager, "resources/unsubscribe", request).await
    }

    async fn complete(
        &self,
        mut request: CompleteRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CompleteResult, ErrorData> {
        self.refresh_peer(context.peer.clone()).await;
        if let Some(name) = request.r#ref.as_prompt_name() {
            let prompts = self.discover_prompts().await?;
            let Some(entry) = prompts.get(name) else {
                return Err(ErrorData::invalid_params(
                    format!("Unknown MCP prompt reference '{}'", name),
                    None,
                ));
            };
            match &entry.target {
                PromptTarget::Plugin {
                    plugin_name,
                    prompt_name,
                } => {
                    if let rmcp::model::Reference::Prompt(prompt) = &mut request.r#ref {
                        prompt.name = prompt_name.clone();
                    }
                    return self
                        .plugin_manager
                        .complete(plugin_name, prompt_name, request)
                        .await
                        .map_err(internal_error);
                }
                PromptTarget::External {
                    endpoint,
                    prompt_name,
                } => {
                    if let rmcp::model::Reference::Prompt(prompt) = &mut request.r#ref {
                        prompt.name = prompt_name.clone();
                    }
                    let client = self.external_mcp.client_for(endpoint).await?;
                    return client.peer.complete(request).await.map_err(internal_error);
                }
            }
        }
        try_plugins(&self.plugin_manager, "completion/complete", request).await
    }

    async fn set_level(
        &self,
        request: SetLevelRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<(), ErrorData> {
        self.refresh_peer(context.peer.clone()).await;
        let mut first_error = None;
        for (plugin_name, server_info) in self.plugin_manager.list_server_infos().await {
            if server_info.capabilities.logging.is_none() {
                continue;
            }
            if let Err(err) = self
                .plugin_manager
                .mcp_request::<(), _>(&plugin_name, "logging/setLevel", request.clone())
                .await
            {
                first_error.get_or_insert(err);
            }
        }
        if let Some(err) = first_error {
            Err(internal_error(err))
        } else {
            Ok(())
        }
    }

    async fn list_tasks(
        &self,
        _request: Option<PaginatedRequestParams>,
        context: RequestContext<RoleServer>,
    ) -> Result<ListTasksResult, ErrorData> {
        self.refresh_peer(context.peer.clone()).await;
        let mut tasks = Vec::new();
        for (plugin_name, server_info) in self.plugin_manager.list_server_infos().await {
            if server_info.capabilities.tasks.is_none() {
                continue;
            }
            let result: ListTasksResult = self
                .plugin_manager
                .mcp_request(
                    &plugin_name,
                    "tasks/list",
                    Option::<PaginatedRequestParams>::None,
                )
                .await
                .map_err(internal_error)?;
            tasks.extend(result.tasks);
        }
        Ok(ListTasksResult::new(tasks))
    }

    async fn get_task_info(
        &self,
        request: GetTaskInfoParams,
        context: RequestContext<RoleServer>,
    ) -> Result<GetTaskResult, ErrorData> {
        self.refresh_peer(context.peer.clone()).await;
        try_plugins(&self.plugin_manager, "tasks/get", request).await
    }

    async fn get_task_result(
        &self,
        request: GetTaskResultParams,
        context: RequestContext<RoleServer>,
    ) -> Result<GetTaskPayloadResult, ErrorData> {
        self.refresh_peer(context.peer.clone()).await;
        try_plugins(&self.plugin_manager, "tasks/result", request).await
    }

    async fn cancel_task(
        &self,
        request: CancelTaskParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CancelTaskResult, ErrorData> {
        self.refresh_peer(context.peer.clone()).await;
        try_plugins(&self.plugin_manager, "tasks/cancel", request).await
    }

    async fn on_cancelled(
        &self,
        notification: rmcp::model::CancelledNotificationParam,
        context: NotificationContext<RoleServer>,
    ) {
        self.refresh_peer(context.peer.clone()).await;
        self.broadcast_notification("notifications/cancelled", notification)
            .await;
    }

    async fn on_progress(
        &self,
        notification: rmcp::model::ProgressNotificationParam,
        context: NotificationContext<RoleServer>,
    ) {
        self.refresh_peer(context.peer.clone()).await;
        self.broadcast_notification("notifications/progress", notification)
            .await;
    }

    async fn on_initialized(&self, context: NotificationContext<RoleServer>) {
        self.refresh_peer(context.peer.clone()).await;
        self.broadcast_notification("notifications/initialized", serde_json::json!({}))
            .await;
    }

    async fn on_roots_list_changed(&self, context: NotificationContext<RoleServer>) {
        self.refresh_peer(context.peer.clone()).await;
        self.broadcast_notification("notifications/roots/list_changed", serde_json::json!({}))
            .await;
    }

    async fn on_custom_notification(
        &self,
        notification: CustomNotification,
        context: NotificationContext<RoleServer>,
    ) {
        self.refresh_peer(context.peer.clone()).await;
        self.broadcast_notification(
            &notification.method,
            notification.params.unwrap_or(serde_json::Value::Null),
        )
        .await;
    }

    async fn on_custom_request(
        &self,
        request: CustomRequest,
        context: RequestContext<RoleServer>,
    ) -> Result<rmcp::model::CustomResult, ErrorData> {
        self.refresh_peer(context.peer.clone()).await;
        try_plugins(
            &self.plugin_manager,
            &request.method,
            request.params.unwrap_or(serde_json::Value::Null),
        )
        .await
    }

    #[allow(deprecated)]
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(
            ServerCapabilities::builder()
                .enable_completions()
                .enable_prompts()
                .enable_prompts_list_changed()
                .enable_resources()
                .enable_resources_list_changed()
                .enable_resources_subscribe()
                .enable_tools()
                .enable_tool_list_changed()
                .enable_tasks()
                .build(),
        )
        .with_server_info(
            Implementation::new("mesh-plugins", crate::BUILD_VERSION)
                .with_title("Mesh Plugin MCP")
                .with_description(
                    "Re-exposes mesh-llm plugins as a single MCP server with the standard MCP surface.",
                ),
        )
        .with_instructions(
            "Running plugins are aggregated into one MCP server. Tool and prompt names are namespaced as <plugin>.<name> to avoid collisions.",
        )
    }
}

fn operation_result_to_call_tool_result(result: plugin::ToolCallResult) -> CallToolResult {
    let mut call_result = match serde_json::from_str::<Value>(&result.content_json) {
        Ok(value) => CallToolResult::structured(value),
        Err(_) => CallToolResult::success(vec![rmcp::model::Content::text(result.content_json)]),
    };
    call_result.is_error = Some(result.is_error);
    call_result
}

fn tool_aliases(plugin_name: &str, tool_name: &str) -> Vec<String> {
    vec![canonical_name(plugin_name, tool_name)]
}

fn canonical_name(plugin_name: &str, local_name: &str) -> String {
    format!("{plugin_name}.{local_name}")
}

#[cfg(test)]
mod tests {
    #[cfg(unix)]
    use super::super::external_mcp::spawn_fake_external_unix_endpoint;
    use super::super::external_mcp::{
        fake_external_client, spawn_fake_external_http_endpoint, spawn_fake_external_tcp_endpoint,
    };
    use super::super::{ActiveBridge, proto_error};
    use super::*;
    use crate::plugin::{PluginEndpointSummary, PluginManager, PluginRpcBridge, RpcResult};
    use rmcp::model::Tool;
    use serde_json::json;

    #[test]
    fn normalize_tool_schema_makes_empty_input_schema_object() {
        let mut tool = Tool::new("bad", "Bad schema", Arc::new(Default::default()));
        tool.output_schema = Some(Arc::new(
            json!({
                "type": "array",
                "items": { "type": "string" }
            })
            .as_object()
            .cloned()
            .unwrap(),
        ));

        let normalized = normalize_tool_schema(tool);

        assert_eq!(
            normalized.input_schema.get("type").and_then(Value::as_str),
            Some("object")
        );
        assert_eq!(
            normalized
                .input_schema
                .get("additionalProperties")
                .and_then(Value::as_bool),
            Some(true)
        );
        assert!(
            normalized.output_schema.is_none(),
            "non-object output schemas should be omitted for strict MCP clients"
        );
    }

    struct NoopBridge;

    impl PluginRpcBridge for NoopBridge {
        fn handle_request(
            &self,
            _plugin_name: String,
            _method: String,
            _params_json: String,
        ) -> crate::plugin::BridgeFuture<Result<RpcResult, plugin::proto::ErrorResponse>> {
            Box::pin(async move { Err(proto_error::internal("unexpected test bridge request")) })
        }

        fn handle_notification(
            &self,
            _plugin_name: String,
            _method: String,
            _params_json: String,
        ) -> crate::plugin::BridgeFuture<()> {
            Box::pin(async {})
        }
    }

    async fn test_server_with_external_endpoint() -> PluginMcpServer {
        let plugin_manager = PluginManager::for_test_bridge(&[], Arc::new(NoopBridge));
        plugin_manager
            .set_test_endpoints(vec![PluginEndpointSummary {
                plugin_name: "adapter".into(),
                plugin_status: "running".into(),
                endpoint_id: "notes".into(),
                state: "healthy".into(),
                available: true,
                kind: "mcp".into(),
                transport_kind: "stdio".into(),
                protocol: None,
                address: Some("fake-external".into()),
                args: Vec::new(),
                namespace: Some("notes".into()),
                supports_streaming: false,
                managed_by_plugin: false,
                detail: None,
                models: Vec::new(),
            }])
            .await;
        let server = PluginMcpServer::new(plugin_manager, ActiveBridge::default());
        server
            .external_mcp
            .register_test_client("adapter:notes", fake_external_client().await)
            .await;
        server
    }

    #[tokio::test]
    async fn external_mcp_endpoint_is_aggregated_into_discovery() {
        let server = test_server_with_external_endpoint().await;

        let tools = server.discover_tools().await.unwrap();
        assert!(tools.contains_key("adapter.notes.echo"));

        let prompts = server.discover_prompts().await.unwrap();
        assert!(prompts.contains_key("adapter.notes.brief"));

        let resources = server.discover_resources().await.unwrap();
        assert!(resources.contains_key("mesh-mcp://adapter/notes/resource/note%3A%2F%2Fone"));

        let endpoint = server
            .active_external_mcp_endpoints()
            .await
            .unwrap()
            .remove(0);
        let client = server.external_mcp.client_for(&endpoint).await.unwrap();
        let result = client
            .peer
            .call_tool(
                CallToolRequestParams::new("echo").with_arguments(
                    serde_json::json!({ "message": "hello" })
                        .as_object()
                        .cloned()
                        .unwrap(),
                ),
            )
            .await
            .unwrap();
        assert_eq!(
            result.structured_content,
            Some(json!({"echo": "hello", "tool": "echo"}))
        );
    }

    #[tokio::test]
    async fn unavailable_external_mcp_endpoint_is_skipped_from_discovery() {
        let plugin_manager = PluginManager::for_test_bridge(&[], Arc::new(NoopBridge));
        plugin_manager
            .set_test_endpoints(vec![PluginEndpointSummary {
                plugin_name: "adapter".into(),
                plugin_status: "running".into(),
                endpoint_id: "notes".into(),
                state: "unhealthy".into(),
                available: false,
                kind: "mcp".into(),
                transport_kind: "stdio".into(),
                protocol: None,
                address: Some("fake-external".into()),
                args: Vec::new(),
                namespace: Some("notes".into()),
                supports_streaming: false,
                managed_by_plugin: false,
                detail: Some("warming".into()),
                models: Vec::new(),
            }])
            .await;
        let server = PluginMcpServer::new(plugin_manager, ActiveBridge::default());

        let tools = server.discover_tools().await.unwrap();
        assert!(!tools.contains_key("adapter.notes.echo"));
    }

    #[tokio::test]
    async fn tcp_external_mcp_endpoint_is_aggregated() {
        let address = spawn_fake_external_tcp_endpoint().await;
        let plugin_manager = PluginManager::for_test_bridge(&[], Arc::new(NoopBridge));
        plugin_manager
            .set_test_endpoints(vec![PluginEndpointSummary {
                plugin_name: "adapter".into(),
                plugin_status: "running".into(),
                endpoint_id: "notes".into(),
                state: "healthy".into(),
                available: true,
                kind: "mcp".into(),
                transport_kind: "tcp".into(),
                protocol: None,
                address: Some(address),
                args: Vec::new(),
                namespace: Some("notes".into()),
                supports_streaming: false,
                managed_by_plugin: false,
                detail: None,
                models: Vec::new(),
            }])
            .await;
        let server = PluginMcpServer::new(plugin_manager, ActiveBridge::default());
        let tools = server.discover_tools().await.unwrap();
        assert!(tools.contains_key("adapter.notes.echo"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn unix_socket_external_mcp_endpoint_is_aggregated() {
        let path = spawn_fake_external_unix_endpoint().await;
        let plugin_manager = PluginManager::for_test_bridge(&[], Arc::new(NoopBridge));
        plugin_manager
            .set_test_endpoints(vec![PluginEndpointSummary {
                plugin_name: "adapter".into(),
                plugin_status: "running".into(),
                endpoint_id: "notes".into(),
                state: "healthy".into(),
                available: true,
                kind: "mcp".into(),
                transport_kind: "unix_socket".into(),
                protocol: None,
                address: Some(path.display().to_string()),
                args: Vec::new(),
                namespace: Some("notes".into()),
                supports_streaming: false,
                managed_by_plugin: false,
                detail: None,
                models: Vec::new(),
            }])
            .await;
        let server = PluginMcpServer::new(plugin_manager, ActiveBridge::default());
        let tools = server.discover_tools().await.unwrap();
        assert!(tools.contains_key("adapter.notes.echo"));
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn http_external_mcp_endpoint_is_aggregated() {
        let uri = spawn_fake_external_http_endpoint().await;
        let plugin_manager = PluginManager::for_test_bridge(&[], Arc::new(NoopBridge));
        plugin_manager
            .set_test_endpoints(vec![PluginEndpointSummary {
                plugin_name: "adapter".into(),
                plugin_status: "running".into(),
                endpoint_id: "remote".into(),
                state: "healthy".into(),
                available: true,
                kind: "mcp".into(),
                transport_kind: "http".into(),
                protocol: Some("streamable_http".into()),
                address: Some(uri),
                args: Vec::new(),
                namespace: Some("remote".into()),
                supports_streaming: true,
                managed_by_plugin: false,
                detail: None,
                models: Vec::new(),
            }])
            .await;
        let server = PluginMcpServer::new(plugin_manager, ActiveBridge::default());
        let tools = server.discover_tools().await.unwrap();
        assert!(tools.contains_key("adapter.remote.echo"));
        let prompts = server.discover_prompts().await.unwrap();
        assert!(prompts.contains_key("adapter.remote.brief"));
    }
}
