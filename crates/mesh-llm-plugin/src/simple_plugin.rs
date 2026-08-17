use anyhow::Result;
use rmcp::model::{
    CallToolResult, CancelTaskParams, CancelTaskResult, CompleteRequestParams, CompleteResult,
    GetPromptRequestParams, GetPromptResult, GetTaskInfoParams, GetTaskPayloadResult,
    GetTaskResult, GetTaskResultParams, ListPromptsResult, ListResourceTemplatesResult,
    ListResourcesResult, ListTasksResult, ListToolsResult, PaginatedRequestParams,
    ReadResourceRequestParams, ReadResourceResult, ServerInfo, SetLevelRequestParams,
    SubscribeRequestParams, UnsubscribeRequestParams,
};
use std::sync::Arc;

use crate::{
    context::PluginContext,
    error::{PluginError, PluginResult},
    helpers::{
        CompletionRouter, PromptRouter, ResourceRouter, TaskRouter, ToolCallRequest, ToolRouter,
    },
    proto,
    runtime::{
        BulkHandler, CancelStreamHandler, ChannelHandler, CloseStreamHandler, HealthFuture,
        HealthHandler, InitFuture, InitHandler, InitializeFuture, InitializeHandler,
        MeshEventHandler, MeshVisibility, OpenStreamFuture, OpenStreamHandler, Plugin,
        PluginInitializeRequest, PluginMetadata, PluginStartupPolicy, SetLogLevelFuture,
        SetLogLevelHandler, StreamErrorHandler, SubscribeFuture, SubscribeHandler,
        UnsubscribeHandler,
    },
};

#[derive(Clone)]
pub struct SimplePlugin {
    metadata: PluginMetadata,
    operation_router: Option<ToolRouter>,
    prompt_router: Option<PromptRouter>,
    resource_router: Option<ResourceRouter>,
    completion_router: Option<CompletionRouter>,
    task_router: Option<TaskRouter>,
    initialize_handler: Option<InitializeHandler>,
    on_initialized: Option<InitHandler>,
    health_handler: Option<HealthHandler>,
    subscribe_handler: Option<SubscribeHandler>,
    unsubscribe_handler: Option<UnsubscribeHandler>,
    set_log_level_handler: Option<SetLogLevelHandler>,
    channel_handler: Option<ChannelHandler>,
    bulk_handler: Option<BulkHandler>,
    mesh_event_handler: Option<MeshEventHandler>,
    open_stream_handler: Option<OpenStreamHandler>,
    cancel_stream_handler: Option<CancelStreamHandler>,
    close_stream_handler: Option<CloseStreamHandler>,
    stream_error_handler: Option<StreamErrorHandler>,
}

impl SimplePlugin {
    pub fn new(metadata: PluginMetadata) -> Self {
        Self {
            metadata,
            operation_router: None,
            prompt_router: None,
            resource_router: None,
            completion_router: None,
            task_router: None,
            initialize_handler: None,
            on_initialized: None,
            health_handler: None,
            subscribe_handler: None,
            unsubscribe_handler: None,
            set_log_level_handler: None,
            channel_handler: None,
            bulk_handler: None,
            mesh_event_handler: None,
            open_stream_handler: None,
            cancel_stream_handler: None,
            close_stream_handler: None,
            stream_error_handler: None,
        }
    }

    pub fn with_capabilities(mut self, capabilities: Vec<String>) -> Self {
        self.metadata = self.metadata.with_capabilities(capabilities);
        self
    }

    pub fn with_manifest(mut self, manifest: proto::PluginManifest) -> Self {
        self.metadata = self.metadata.with_manifest(manifest);
        self
    }

    pub fn with_startup_policy(mut self, startup_policy: PluginStartupPolicy) -> Self {
        self.metadata = self.metadata.with_startup_policy(startup_policy);
        self
    }

    pub fn with_operation_router(mut self, router: ToolRouter) -> Self {
        self.operation_router = Some(router);
        self
    }

    pub fn extend_operation_router(mut self, router: ToolRouter) -> Self {
        match &mut self.operation_router {
            Some(existing) => existing.extend(router),
            None => self.operation_router = Some(router),
        }
        self
    }

    pub fn with_prompt_router(mut self, router: PromptRouter) -> Self {
        self.prompt_router = Some(router);
        self
    }

    pub fn with_resource_router(mut self, router: ResourceRouter) -> Self {
        self.resource_router = Some(router);
        self
    }

    pub fn with_completion_router(mut self, router: CompletionRouter) -> Self {
        self.completion_router = Some(router);
        self
    }

    pub fn with_task_router(mut self, router: TaskRouter) -> Self {
        self.task_router = Some(router);
        self
    }

    pub fn on_initialize<F>(mut self, handler: F) -> Self
    where
        F: for<'a, 'ctx> Fn(
                PluginInitializeRequest,
                &'a mut PluginContext<'ctx>,
            ) -> InitializeFuture<'a>
            + Send
            + Sync
            + 'static,
    {
        self.initialize_handler = Some(Arc::new(handler));
        self
    }

    pub fn on_initialized<F>(mut self, handler: F) -> Self
    where
        F: for<'a, 'ctx> Fn(&'a mut PluginContext<'ctx>) -> InitFuture<'a> + Send + Sync + 'static,
    {
        self.on_initialized = Some(Arc::new(handler));
        self
    }

    pub fn with_health<F>(mut self, handler: F) -> Self
    where
        F: for<'a, 'ctx> Fn(&'a mut PluginContext<'ctx>) -> HealthFuture<'a>
            + Send
            + Sync
            + 'static,
    {
        self.health_handler = Some(Arc::new(handler));
        self
    }

    pub fn with_subscribe_resource<F>(mut self, handler: F) -> Self
    where
        F: for<'a, 'ctx> Fn(
                SubscribeRequestParams,
                &'a mut PluginContext<'ctx>,
            ) -> SubscribeFuture<'a>
            + Send
            + Sync
            + 'static,
    {
        self.subscribe_handler = Some(Arc::new(handler));
        self
    }

    pub fn with_unsubscribe_resource<F>(mut self, handler: F) -> Self
    where
        F: for<'a, 'ctx> Fn(
                UnsubscribeRequestParams,
                &'a mut PluginContext<'ctx>,
            ) -> SubscribeFuture<'a>
            + Send
            + Sync
            + 'static,
    {
        self.unsubscribe_handler = Some(Arc::new(handler));
        self
    }

    pub fn with_set_log_level<F>(mut self, handler: F) -> Self
    where
        F: for<'a, 'ctx> Fn(
                SetLevelRequestParams,
                &'a mut PluginContext<'ctx>,
            ) -> SetLogLevelFuture<'a>
            + Send
            + Sync
            + 'static,
    {
        self.set_log_level_handler = Some(Arc::new(handler));
        self
    }

    pub fn on_channel_message<F>(mut self, handler: F) -> Self
    where
        F: for<'a, 'ctx> Fn(proto::ChannelMessage, &'a mut PluginContext<'ctx>) -> InitFuture<'a>
            + Send
            + Sync
            + 'static,
    {
        self.channel_handler = Some(Arc::new(handler));
        self
    }

    pub fn on_bulk_transfer_message<F>(mut self, handler: F) -> Self
    where
        F: for<'a, 'ctx> Fn(
                proto::BulkTransferMessage,
                &'a mut PluginContext<'ctx>,
            ) -> InitFuture<'a>
            + Send
            + Sync
            + 'static,
    {
        self.bulk_handler = Some(Arc::new(handler));
        self
    }

    pub fn on_mesh_event<F>(mut self, handler: F) -> Self
    where
        F: for<'a, 'ctx> Fn(proto::MeshEvent, &'a mut PluginContext<'ctx>) -> InitFuture<'a>
            + Send
            + Sync
            + 'static,
    {
        self.mesh_event_handler = Some(Arc::new(handler));
        self
    }

    pub fn on_open_stream<F>(mut self, handler: F) -> Self
    where
        F: for<'a, 'ctx> Fn(
                proto::OpenStreamRequest,
                &'a mut PluginContext<'ctx>,
            ) -> OpenStreamFuture<'a>
            + Send
            + Sync
            + 'static,
    {
        self.open_stream_handler = Some(Arc::new(handler));
        self
    }

    pub fn on_cancel_stream<F>(mut self, handler: F) -> Self
    where
        F: for<'a, 'ctx> Fn(
                proto::CancelStreamNotification,
                &'a mut PluginContext<'ctx>,
            ) -> InitFuture<'a>
            + Send
            + Sync
            + 'static,
    {
        self.cancel_stream_handler = Some(Arc::new(handler));
        self
    }

    pub fn on_close_stream<F>(mut self, handler: F) -> Self
    where
        F: for<'a, 'ctx> Fn(
                proto::CloseStreamNotification,
                &'a mut PluginContext<'ctx>,
            ) -> InitFuture<'a>
            + Send
            + Sync
            + 'static,
    {
        self.close_stream_handler = Some(Arc::new(handler));
        self
    }

    pub fn on_stream_error<F>(mut self, handler: F) -> Self
    where
        F: for<'a, 'ctx> Fn(proto::StreamError, &'a mut PluginContext<'ctx>) -> InitFuture<'a>
            + Send
            + Sync
            + 'static,
    {
        self.stream_error_handler = Some(Arc::new(handler));
        self
    }
}

#[crate::async_trait]
impl Plugin for SimplePlugin {
    fn plugin_id(&self) -> &str {
        &self.metadata.plugin_id
    }

    fn plugin_version(&self) -> String {
        self.metadata.plugin_version.clone()
    }

    fn server_info(&self) -> ServerInfo {
        self.metadata.server_info.clone()
    }

    fn capabilities(&self) -> Vec<String> {
        self.metadata.capabilities.clone()
    }

    fn manifest(&self) -> Option<proto::PluginManifest> {
        self.metadata.manifest.clone()
    }

    async fn initialize(
        &mut self,
        request: PluginInitializeRequest,
        context: &mut PluginContext<'_>,
    ) -> PluginResult<()> {
        match self.metadata.startup_policy {
            PluginStartupPolicy::Any => {}
            PluginStartupPolicy::PrivateMeshOnly
                if request.mesh_visibility != MeshVisibility::Private =>
            {
                return Err(PluginError::startup_disabled(format!(
                    "Plugin '{}' requires a private mesh",
                    self.metadata.plugin_id
                )));
            }
            PluginStartupPolicy::PublicMeshOnly
                if request.mesh_visibility != MeshVisibility::Public =>
            {
                return Err(PluginError::startup_disabled(format!(
                    "Plugin '{}' requires a public mesh",
                    self.metadata.plugin_id
                )));
            }
            PluginStartupPolicy::PrivateMeshOnly | PluginStartupPolicy::PublicMeshOnly => {}
        }
        match &self.initialize_handler {
            Some(handler) => handler(request, context).await,
            None => Ok(()),
        }
    }

    async fn on_initialized(&mut self, context: &mut PluginContext<'_>) -> Result<()> {
        match &self.on_initialized {
            Some(handler) => handler(context).await,
            None => Ok(()),
        }
    }

    async fn health(&mut self, context: &mut PluginContext<'_>) -> Result<String> {
        match &self.health_handler {
            Some(handler) => handler(context).await,
            None => Ok("ok".into()),
        }
    }

    async fn list_tools(
        &mut self,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<ListToolsResult>> {
        Ok(self
            .operation_router
            .as_ref()
            .map(|router| router.list_tools_result()))
    }

    async fn call_tool(
        &mut self,
        request: ToolCallRequest,
        context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<CallToolResult>> {
        match &self.operation_router {
            Some(router) => Ok(Some(router.call(request, context).await?)),
            None => Ok(None),
        }
    }

    async fn list_prompts(
        &mut self,
        _request: Option<PaginatedRequestParams>,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<ListPromptsResult>> {
        Ok(self
            .prompt_router
            .as_ref()
            .map(|router| router.list_prompts_result()))
    }

    async fn get_prompt(
        &mut self,
        request: GetPromptRequestParams,
        context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<GetPromptResult>> {
        match &self.prompt_router {
            Some(router) => Ok(Some(router.get(request, context).await?)),
            None => Ok(None),
        }
    }

    async fn list_resources(
        &mut self,
        _request: Option<PaginatedRequestParams>,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<ListResourcesResult>> {
        Ok(self
            .resource_router
            .as_ref()
            .map(|router| router.list_resources_result()))
    }

    async fn read_resource(
        &mut self,
        request: ReadResourceRequestParams,
        context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<ReadResourceResult>> {
        match &self.resource_router {
            Some(router) => Ok(Some(router.read(request, context).await?)),
            None => Ok(None),
        }
    }

    async fn list_resource_templates(
        &mut self,
        _request: Option<PaginatedRequestParams>,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<ListResourceTemplatesResult>> {
        Ok(self
            .resource_router
            .as_ref()
            .map(|router| router.list_resource_templates_result()))
    }

    async fn subscribe_resource(
        &mut self,
        request: SubscribeRequestParams,
        context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<()>> {
        match &self.subscribe_handler {
            Some(handler) => Ok(Some(handler(request, context).await?)),
            None => Ok(None),
        }
    }

    async fn unsubscribe_resource(
        &mut self,
        request: UnsubscribeRequestParams,
        context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<()>> {
        match &self.unsubscribe_handler {
            Some(handler) => Ok(Some(handler(request, context).await?)),
            None => Ok(None),
        }
    }

    async fn complete(
        &mut self,
        request: CompleteRequestParams,
        context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<CompleteResult>> {
        match &self.completion_router {
            Some(router) => Ok(Some(router.complete(request, context).await?)),
            None => Ok(None),
        }
    }

    async fn set_log_level(
        &mut self,
        request: SetLevelRequestParams,
        context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<()>> {
        match &self.set_log_level_handler {
            Some(handler) => Ok(Some(handler(request, context).await?)),
            None => Ok(None),
        }
    }

    async fn list_tasks(
        &mut self,
        request: Option<PaginatedRequestParams>,
        context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<ListTasksResult>> {
        match &self.task_router {
            Some(router) => router.list_tasks(request, context).await,
            None => Ok(None),
        }
    }

    async fn get_task_info(
        &mut self,
        request: GetTaskInfoParams,
        context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<GetTaskResult>> {
        match &self.task_router {
            Some(router) => router.get_task_info(request, context).await,
            None => Ok(None),
        }
    }

    async fn get_task_result(
        &mut self,
        request: GetTaskResultParams,
        context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<GetTaskPayloadResult>> {
        match &self.task_router {
            Some(router) => router.get_task_result(request, context).await,
            None => Ok(None),
        }
    }

    async fn cancel_task(
        &mut self,
        request: CancelTaskParams,
        context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<CancelTaskResult>> {
        match &self.task_router {
            Some(router) => router.cancel_task(request, context).await,
            None => Ok(None),
        }
    }

    async fn on_channel_message(
        &mut self,
        message: proto::ChannelMessage,
        context: &mut PluginContext<'_>,
    ) -> Result<()> {
        match &self.channel_handler {
            Some(handler) => handler(message, context).await,
            None => Ok(()),
        }
    }

    async fn on_bulk_transfer_message(
        &mut self,
        message: proto::BulkTransferMessage,
        context: &mut PluginContext<'_>,
    ) -> Result<()> {
        match &self.bulk_handler {
            Some(handler) => handler(message, context).await,
            None => Ok(()),
        }
    }

    async fn on_mesh_event(
        &mut self,
        event: proto::MeshEvent,
        context: &mut PluginContext<'_>,
    ) -> Result<()> {
        match &self.mesh_event_handler {
            Some(handler) => handler(event, context).await,
            None => Ok(()),
        }
    }

    async fn open_stream(
        &mut self,
        request: proto::OpenStreamRequest,
        context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<proto::OpenStreamResponse>> {
        match &self.open_stream_handler {
            Some(handler) => handler(request, context).await,
            None => Ok(None),
        }
    }

    async fn on_cancel_stream(
        &mut self,
        notification: proto::CancelStreamNotification,
        context: &mut PluginContext<'_>,
    ) -> Result<()> {
        match &self.cancel_stream_handler {
            Some(handler) => handler(notification, context).await,
            None => Ok(()),
        }
    }

    async fn on_close_stream(
        &mut self,
        notification: proto::CloseStreamNotification,
        context: &mut PluginContext<'_>,
    ) -> Result<()> {
        match &self.close_stream_handler {
            Some(handler) => handler(notification, context).await,
            None => Ok(()),
        }
    }

    async fn on_stream_error(
        &mut self,
        error: proto::StreamError,
        context: &mut PluginContext<'_>,
    ) -> Result<()> {
        match &self.stream_error_handler {
            Some(handler) => handler(error, context).await,
            None => Ok(()),
        }
    }
}
