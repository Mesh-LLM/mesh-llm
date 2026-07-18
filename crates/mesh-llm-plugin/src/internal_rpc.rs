use anyhow::Result;
use rmcp::model::{CallToolResult, ListToolsResult, ServerInfo};
use std::collections::BTreeMap;
use std::sync::Arc;

use crate::{
    context::PluginContext,
    error::{PluginResult, PluginRpcResult},
    helpers::{ToolCallRequest, ToolRouter},
    proto,
    runtime::{
        HealthFuture, Plugin, PluginInitializeRequest, PluginMetadata, PluginStartupPolicy,
        RpcMethodFuture, RpcMethodHandler,
    },
    simple_plugin::SimplePlugin,
};

pub struct InternalRpcPluginBuilder {
    plugin: SimplePlugin,
    rpc_handlers: BTreeMap<String, RpcMethodHandler>,
}

impl InternalRpcPluginBuilder {
    pub fn new(metadata: PluginMetadata) -> Self {
        Self {
            plugin: SimplePlugin::new(metadata),
            rpc_handlers: BTreeMap::new(),
        }
    }

    pub fn with_capabilities(mut self, capabilities: Vec<String>) -> Self {
        self.plugin = self.plugin.with_capabilities(capabilities);
        self
    }

    pub fn with_manifest(mut self, manifest: proto::PluginManifest) -> Self {
        self.plugin = self.plugin.with_manifest(manifest);
        self
    }

    pub fn with_startup_policy(mut self, startup_policy: PluginStartupPolicy) -> Self {
        self.plugin = self.plugin.with_startup_policy(startup_policy);
        self
    }

    pub fn with_operation_router(mut self, router: ToolRouter) -> Self {
        self.plugin = self.plugin.with_operation_router(router);
        self
    }

    pub fn with_health<F>(mut self, handler: F) -> Self
    where
        F: for<'a, 'ctx> Fn(&'a mut PluginContext<'ctx>) -> HealthFuture<'a>
            + Send
            + Sync
            + 'static,
    {
        self.plugin = self.plugin.with_health(handler);
        self
    }

    pub fn rpc_method<F>(mut self, method: impl Into<String>, handler: F) -> Self
    where
        F: for<'a, 'ctx> Fn(proto::RpcRequest, &'a mut PluginContext<'ctx>) -> RpcMethodFuture<'a>
            + Send
            + Sync
            + 'static,
    {
        self.rpc_handlers.insert(method.into(), Arc::new(handler));
        self
    }

    pub fn build(self) -> InternalRpcPlugin {
        InternalRpcPlugin {
            plugin: self.plugin,
            rpc_handlers: self.rpc_handlers,
        }
    }
}

#[derive(Clone)]
pub struct InternalRpcPlugin {
    plugin: SimplePlugin,
    rpc_handlers: BTreeMap<String, RpcMethodHandler>,
}

#[crate::async_trait]
impl Plugin for InternalRpcPlugin {
    fn plugin_id(&self) -> &str {
        self.plugin.plugin_id()
    }

    fn plugin_version(&self) -> String {
        self.plugin.plugin_version()
    }

    fn server_info(&self) -> ServerInfo {
        self.plugin.server_info()
    }

    fn capabilities(&self) -> Vec<String> {
        self.plugin.capabilities()
    }

    fn manifest(&self) -> Option<proto::PluginManifest> {
        self.plugin.manifest()
    }

    async fn initialize(
        &mut self,
        request: PluginInitializeRequest,
        context: &mut PluginContext<'_>,
    ) -> PluginResult<()> {
        self.plugin.initialize(request, context).await
    }

    async fn on_initialized(&mut self, context: &mut PluginContext<'_>) -> Result<()> {
        <SimplePlugin as Plugin>::on_initialized(&mut self.plugin, context).await
    }

    async fn health(&mut self, context: &mut PluginContext<'_>) -> Result<String> {
        self.plugin.health(context).await
    }

    async fn list_tools(
        &mut self,
        context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<ListToolsResult>> {
        self.plugin.list_tools(context).await
    }

    async fn call_tool(
        &mut self,
        request: ToolCallRequest,
        context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<CallToolResult>> {
        self.plugin.call_tool(request, context).await
    }

    async fn handle_rpc(
        &mut self,
        request: proto::RpcRequest,
        context: &mut PluginContext<'_>,
    ) -> PluginRpcResult {
        if let Some(handler) = self.rpc_handlers.get(&request.method).cloned() {
            return handler(request, context).await;
        }
        self.plugin.handle_rpc(request, context).await
    }
}
