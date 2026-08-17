use anyhow::{Result, bail};
use rmcp::model::{
    CallToolResult, CancelTaskParams, CancelTaskResult, CompleteRequestParams, CompleteResult,
    GetPromptRequestParams, GetPromptResult, GetTaskInfoParams, GetTaskPayloadResult,
    GetTaskResult, GetTaskResultParams, ListPromptsResult, ListResourceTemplatesResult,
    ListResourcesResult, ListTasksResult, ListToolsResult, PaginatedRequestParams,
    ReadResourceRequestParams, ReadResourceResult, ServerInfo, SetLevelRequestParams,
    SubscribeRequestParams, UnsubscribeRequestParams,
};
use serde::de::DeserializeOwned;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use tokio::sync::{RwLock, mpsc, watch};

use crate::{
    PROTOCOL_VERSION,
    context::{
        PendingHostResponses, PluginContext, drain_pending_host_responses,
        remove_pending_host_response,
    },
    error::{PluginError, PluginResult, PluginRpcResult},
    helpers::{
        ToolCallRequest, json_response, parse_get_prompt_request, parse_read_resource_request,
        parse_rpc_params, parse_tool_call_request,
    },
    io::{
        LocalReadHalf, LocalStream, LocalWriteHalf, connect_from_env, read_envelope_from,
        write_envelope_to,
    },
    proto,
};
use serde::{Deserialize, Serialize};

pub use crate::internal_rpc::{InternalRpcPlugin, InternalRpcPluginBuilder};
pub use crate::simple_plugin::SimplePlugin;

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MeshVisibility {
    #[default]
    Private,
    Public,
}

impl MeshVisibility {
    fn from_proto(value: i32) -> Self {
        match proto::MeshVisibility::try_from(value).unwrap_or(proto::MeshVisibility::Unspecified) {
            proto::MeshVisibility::Public => Self::Public,
            _ => Self::Private,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PluginStartupPolicy {
    #[default]
    Any,
    PrivateMeshOnly,
    PublicMeshOnly,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PluginInitializeRequest {
    pub host_protocol_version: u32,
    pub host_version: String,
    pub host_info_json: String,
    pub mesh_visibility: MeshVisibility,
}

impl From<proto::InitializeRequest> for PluginInitializeRequest {
    fn from(value: proto::InitializeRequest) -> Self {
        Self {
            host_protocol_version: value.host_protocol_version,
            host_version: value.host_version,
            host_info_json: value.host_info_json,
            mesh_visibility: MeshVisibility::from_proto(value.mesh_visibility),
        }
    }
}

#[derive(Clone)]
pub struct PluginMetadata {
    pub(super) plugin_id: String,
    pub(super) plugin_version: String,
    pub(super) server_info: ServerInfo,
    pub(super) capabilities: Vec<String>,
    pub(super) manifest: Option<proto::PluginManifest>,
    pub(super) startup_policy: PluginStartupPolicy,
}

impl PluginMetadata {
    pub fn new(
        plugin_id: impl Into<String>,
        plugin_version: impl Into<String>,
        server_info: ServerInfo,
    ) -> Self {
        Self {
            plugin_id: plugin_id.into(),
            plugin_version: plugin_version.into(),
            server_info,
            capabilities: Vec::new(),
            manifest: None,
            startup_policy: PluginStartupPolicy::Any,
        }
    }

    pub fn with_capabilities(mut self, capabilities: Vec<String>) -> Self {
        self.capabilities = capabilities;
        self
    }

    pub fn with_manifest(mut self, manifest: proto::PluginManifest) -> Self {
        self.manifest = Some(manifest);
        self
    }

    pub fn with_startup_policy(mut self, startup_policy: PluginStartupPolicy) -> Self {
        self.startup_policy = startup_policy;
        self
    }
}

pub(super) type InitializeFuture<'a> = Pin<Box<dyn Future<Output = PluginResult<()>> + Send + 'a>>;
pub(super) type InitFuture<'a> = Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;
pub(super) type HealthFuture<'a> = Pin<Box<dyn Future<Output = Result<String>> + Send + 'a>>;
pub(super) type OpenStreamFuture<'a> =
    Pin<Box<dyn Future<Output = PluginResult<Option<proto::OpenStreamResponse>>> + Send + 'a>>;
pub(super) type RpcMethodFuture<'a> = Pin<Box<dyn Future<Output = PluginRpcResult> + Send + 'a>>;
pub(super) type SubscribeFuture<'a> = Pin<Box<dyn Future<Output = PluginResult<()>> + Send + 'a>>;
pub(super) type SetLogLevelFuture<'a> = Pin<Box<dyn Future<Output = PluginResult<()>> + Send + 'a>>;

pub(super) type InitializeHandler = Arc<
    dyn for<'a, 'ctx> Fn(
            PluginInitializeRequest,
            &'a mut PluginContext<'ctx>,
        ) -> InitializeFuture<'a>
        + Send
        + Sync,
>;
pub(super) type InitHandler =
    Arc<dyn for<'a, 'ctx> Fn(&'a mut PluginContext<'ctx>) -> InitFuture<'a> + Send + Sync>;
pub(super) type HealthHandler =
    Arc<dyn for<'a, 'ctx> Fn(&'a mut PluginContext<'ctx>) -> HealthFuture<'a> + Send + Sync>;
pub(super) type SubscribeHandler = Arc<
    dyn for<'a, 'ctx> Fn(SubscribeRequestParams, &'a mut PluginContext<'ctx>) -> SubscribeFuture<'a>
        + Send
        + Sync,
>;
pub(super) type UnsubscribeHandler = Arc<
    dyn for<'a, 'ctx> Fn(
            UnsubscribeRequestParams,
            &'a mut PluginContext<'ctx>,
        ) -> SubscribeFuture<'a>
        + Send
        + Sync,
>;
pub(super) type SetLogLevelHandler = Arc<
    dyn for<'a, 'ctx> Fn(
            SetLevelRequestParams,
            &'a mut PluginContext<'ctx>,
        ) -> SetLogLevelFuture<'a>
        + Send
        + Sync,
>;
pub(super) type ChannelHandler = Arc<
    dyn for<'a, 'ctx> Fn(proto::ChannelMessage, &'a mut PluginContext<'ctx>) -> InitFuture<'a>
        + Send
        + Sync,
>;
pub(super) type BulkHandler = Arc<
    dyn for<'a, 'ctx> Fn(proto::BulkTransferMessage, &'a mut PluginContext<'ctx>) -> InitFuture<'a>
        + Send
        + Sync,
>;
pub(super) type MeshEventHandler = Arc<
    dyn for<'a, 'ctx> Fn(proto::MeshEvent, &'a mut PluginContext<'ctx>) -> InitFuture<'a>
        + Send
        + Sync,
>;
pub(super) type RpcMethodHandler = Arc<
    dyn for<'a, 'ctx> Fn(proto::RpcRequest, &'a mut PluginContext<'ctx>) -> RpcMethodFuture<'a>
        + Send
        + Sync,
>;
pub(super) type OpenStreamHandler = Arc<
    dyn for<'a, 'ctx> Fn(
            proto::OpenStreamRequest,
            &'a mut PluginContext<'ctx>,
        ) -> OpenStreamFuture<'a>
        + Send
        + Sync,
>;
pub(super) type CancelStreamHandler = Arc<
    dyn for<'a, 'ctx> Fn(
            proto::CancelStreamNotification,
            &'a mut PluginContext<'ctx>,
        ) -> InitFuture<'a>
        + Send
        + Sync,
>;
pub(super) type CloseStreamHandler = Arc<
    dyn for<'a, 'ctx> Fn(
            proto::CloseStreamNotification,
            &'a mut PluginContext<'ctx>,
        ) -> InitFuture<'a>
        + Send
        + Sync,
>;
pub(super) type StreamErrorHandler = Arc<
    dyn for<'a, 'ctx> Fn(proto::StreamError, &'a mut PluginContext<'ctx>) -> InitFuture<'a>
        + Send
        + Sync,
>;

#[crate::async_trait]
pub trait Plugin: Send {
    fn plugin_id(&self) -> &str;
    fn plugin_version(&self) -> String;
    fn server_info(&self) -> ServerInfo;

    fn capabilities(&self) -> Vec<String> {
        Vec::new()
    }

    fn manifest(&self) -> Option<proto::PluginManifest> {
        None
    }

    async fn initialize(
        &mut self,
        _request: PluginInitializeRequest,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<()> {
        Ok(())
    }

    async fn on_initialized(&mut self, _context: &mut PluginContext<'_>) -> Result<()> {
        Ok(())
    }

    async fn health(&mut self, _context: &mut PluginContext<'_>) -> Result<String> {
        Ok("ok".into())
    }

    async fn list_tools(
        &mut self,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<ListToolsResult>> {
        Ok(None)
    }

    async fn call_tool(
        &mut self,
        _request: ToolCallRequest,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<CallToolResult>> {
        Ok(None)
    }

    async fn list_prompts(
        &mut self,
        _request: Option<PaginatedRequestParams>,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<ListPromptsResult>> {
        Ok(None)
    }

    async fn get_prompt(
        &mut self,
        _request: GetPromptRequestParams,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<GetPromptResult>> {
        Ok(None)
    }

    async fn list_resources(
        &mut self,
        _request: Option<PaginatedRequestParams>,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<ListResourcesResult>> {
        Ok(None)
    }

    async fn read_resource(
        &mut self,
        _request: ReadResourceRequestParams,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<ReadResourceResult>> {
        Ok(None)
    }

    async fn list_resource_templates(
        &mut self,
        _request: Option<PaginatedRequestParams>,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<ListResourceTemplatesResult>> {
        Ok(None)
    }

    async fn subscribe_resource(
        &mut self,
        _request: SubscribeRequestParams,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<()>> {
        Ok(None)
    }

    async fn unsubscribe_resource(
        &mut self,
        _request: UnsubscribeRequestParams,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<()>> {
        Ok(None)
    }

    async fn complete(
        &mut self,
        _request: CompleteRequestParams,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<CompleteResult>> {
        Ok(None)
    }

    async fn set_log_level(
        &mut self,
        _request: SetLevelRequestParams,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<()>> {
        Ok(None)
    }

    async fn list_tasks(
        &mut self,
        _request: Option<PaginatedRequestParams>,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<ListTasksResult>> {
        Ok(None)
    }

    async fn get_task_info(
        &mut self,
        _request: GetTaskInfoParams,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<GetTaskResult>> {
        Ok(None)
    }

    async fn get_task_result(
        &mut self,
        _request: GetTaskResultParams,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<GetTaskPayloadResult>> {
        Ok(None)
    }

    async fn cancel_task(
        &mut self,
        _request: CancelTaskParams,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<CancelTaskResult>> {
        Ok(None)
    }

    async fn invoke_service(
        &mut self,
        request: proto::InvokeServiceRequest,
        context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<proto::InvokeServiceResponse>> {
        match proto::ServiceKind::try_from(request.kind).unwrap_or(proto::ServiceKind::Unspecified)
        {
            proto::ServiceKind::Operation => {
                let arguments = parse_service_input::<serde_json::Value>(&request.input_json)?;
                let tool_request = ToolCallRequest {
                    name: request.service_name,
                    arguments,
                };
                match self.call_tool(tool_request, context).await? {
                    Some(result) => Ok(Some(proto::InvokeServiceResponse {
                        output_json: normalize_call_tool_output(&result)?,
                        is_error: result.is_error.unwrap_or(false),
                    })),
                    None => Ok(None),
                }
            }
            proto::ServiceKind::Prompt => {
                let params = parse_service_input::<GetPromptRequestParams>(&request.input_json)?;
                match self.get_prompt(params, context).await? {
                    Some(result) => Ok(Some(proto::InvokeServiceResponse {
                        output_json: serialize_service_output(&result)?,
                        is_error: false,
                    })),
                    None => Ok(None),
                }
            }
            proto::ServiceKind::Resource => {
                let params = parse_service_input::<ReadResourceRequestParams>(&request.input_json)?;
                match self.read_resource(params, context).await? {
                    Some(result) => Ok(Some(proto::InvokeServiceResponse {
                        output_json: serialize_service_output(&result)?,
                        is_error: false,
                    })),
                    None => Ok(None),
                }
            }
            proto::ServiceKind::Completion => {
                let params = parse_service_input::<CompleteRequestParams>(&request.input_json)?;
                match self.complete(params, context).await? {
                    Some(result) => Ok(Some(proto::InvokeServiceResponse {
                        output_json: serialize_service_output(&result)?,
                        is_error: false,
                    })),
                    None => Ok(None),
                }
            }
            proto::ServiceKind::Unspecified => Err(PluginError::invalid_request(
                "Service invocation kind is required",
            )),
        }
    }

    async fn handle_rpc(
        &mut self,
        request: proto::RpcRequest,
        context: &mut PluginContext<'_>,
    ) -> PluginRpcResult {
        match request.method.as_str() {
            "tools/list" => match self.list_tools(context).await? {
                Some(result) => json_response(&result),
                None => Err(PluginError::method_not_found(
                    "Unsupported MCP method 'tools/list'",
                )),
            },
            "tools/call" => {
                let tool_call = parse_tool_call_request(&request)?;
                match self.call_tool(tool_call, context).await? {
                    Some(result) => json_response(&result),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'tools/call'",
                    )),
                }
            }
            "prompts/list" => {
                let params: Option<PaginatedRequestParams> = parse_rpc_params(&request)?;
                match self.list_prompts(params, context).await? {
                    Some(result) => json_response(&result),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'prompts/list'",
                    )),
                }
            }
            "prompts/get" => {
                let params = parse_get_prompt_request(&request)?;
                match self.get_prompt(params, context).await? {
                    Some(result) => json_response(&result),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'prompts/get'",
                    )),
                }
            }
            "resources/list" => {
                let params: Option<PaginatedRequestParams> = parse_rpc_params(&request)?;
                match self.list_resources(params, context).await? {
                    Some(result) => json_response(&result),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'resources/list'",
                    )),
                }
            }
            "resources/read" => {
                let params = parse_read_resource_request(&request)?;
                match self.read_resource(params, context).await? {
                    Some(result) => json_response(&result),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'resources/read'",
                    )),
                }
            }
            "resources/templates/list" => {
                let params: Option<PaginatedRequestParams> = parse_rpc_params(&request)?;
                match self.list_resource_templates(params, context).await? {
                    Some(result) => json_response(&result),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'resources/templates/list'",
                    )),
                }
            }
            "resources/subscribe" => {
                let params: SubscribeRequestParams = parse_rpc_params(&request)?;
                match self.subscribe_resource(params, context).await? {
                    Some(()) => json_response(&serde_json::json!({})),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'resources/subscribe'",
                    )),
                }
            }
            "resources/unsubscribe" => {
                let params: UnsubscribeRequestParams = parse_rpc_params(&request)?;
                match self.unsubscribe_resource(params, context).await? {
                    Some(()) => json_response(&serde_json::json!({})),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'resources/unsubscribe'",
                    )),
                }
            }
            "completion/complete" => {
                let params: CompleteRequestParams = parse_rpc_params(&request)?;
                match self.complete(params, context).await? {
                    Some(result) => json_response(&result),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'completion/complete'",
                    )),
                }
            }
            "logging/setLevel" => {
                let params: SetLevelRequestParams = parse_rpc_params(&request)?;
                match self.set_log_level(params, context).await? {
                    Some(()) => json_response(&serde_json::json!({})),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'logging/setLevel'",
                    )),
                }
            }
            "tasks/list" => {
                let params: Option<PaginatedRequestParams> = parse_rpc_params(&request)?;
                match self.list_tasks(params, context).await? {
                    Some(result) => json_response(&result),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'tasks/list'",
                    )),
                }
            }
            "tasks/get" => {
                let params: GetTaskInfoParams = parse_rpc_params(&request)?;
                match self.get_task_info(params, context).await? {
                    Some(result) => json_response(&result),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'tasks/get'",
                    )),
                }
            }
            "tasks/result" => {
                let params: GetTaskResultParams = parse_rpc_params(&request)?;
                match self.get_task_result(params, context).await? {
                    Some(result) => json_response(&result),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'tasks/result'",
                    )),
                }
            }
            "tasks/cancel" => {
                let params: CancelTaskParams = parse_rpc_params(&request)?;
                match self.cancel_task(params, context).await? {
                    Some(result) => json_response(&result),
                    None => Err(PluginError::method_not_found(
                        "Unsupported MCP method 'tasks/cancel'",
                    )),
                }
            }
            _ => Err(PluginError::method_not_found(format!(
                "Unsupported MCP method '{}'",
                request.method
            ))),
        }
    }

    async fn on_rpc_notification(
        &mut self,
        _notification: proto::RpcNotification,
        _context: &mut PluginContext<'_>,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_channel_message(
        &mut self,
        _message: proto::ChannelMessage,
        _context: &mut PluginContext<'_>,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_bulk_transfer_message(
        &mut self,
        _message: proto::BulkTransferMessage,
        _context: &mut PluginContext<'_>,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_mesh_event(
        &mut self,
        _event: proto::MeshEvent,
        _context: &mut PluginContext<'_>,
    ) -> Result<()> {
        Ok(())
    }

    async fn open_stream(
        &mut self,
        _request: proto::OpenStreamRequest,
        _context: &mut PluginContext<'_>,
    ) -> PluginResult<Option<proto::OpenStreamResponse>> {
        Ok(None)
    }

    async fn on_cancel_stream(
        &mut self,
        _notification: proto::CancelStreamNotification,
        _context: &mut PluginContext<'_>,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_close_stream(
        &mut self,
        _notification: proto::CloseStreamNotification,
        _context: &mut PluginContext<'_>,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_stream_error(
        &mut self,
        _error: proto::StreamError,
        _context: &mut PluginContext<'_>,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_host_error(
        &mut self,
        error: proto::ErrorResponse,
        _context: &mut PluginContext<'_>,
    ) -> Result<()> {
        bail!("host error: {}", error.message)
    }
}

pub struct PluginRuntime;

struct RuntimeState<P> {
    plugin: Arc<RwLock<P>>,
    plugin_id: String,
    outbound_tx: mpsc::Sender<proto::Envelope>,
    pending_host_responses: PendingHostResponses,
}

struct OrderedPayload {
    request_id: u64,
    payload: proto::envelope::Payload,
}

impl PluginRuntime {
    pub async fn run<P: Plugin + Clone + Sync + 'static>(plugin: P) -> Result<()> {
        let stream = connect_from_env().await?;
        Self::run_with_stream(plugin, stream).await
    }

    pub async fn run_with_stream<P: Plugin + Clone + Sync + 'static>(
        plugin: P,
        stream: LocalStream,
    ) -> Result<()> {
        let plugin_id = plugin.plugin_id().to_string();
        let (read, write) = stream.into_split();
        let (outbound_tx, outbound_rx) = mpsc::channel(256);
        let (ordered_tx, ordered_rx) = mpsc::channel(256);
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let state = Arc::new(RuntimeState {
            plugin: Arc::new(RwLock::new(plugin)),
            plugin_id,
            outbound_tx,
            pending_host_responses: Arc::new(Mutex::new(HashMap::new())),
        });
        let mut writer = tokio::spawn(Self::write_loop(
            write,
            outbound_rx,
            state.pending_host_responses.clone(),
            shutdown_tx.clone(),
            shutdown_rx.clone(),
        ));
        let ordered_handlers = tokio::spawn(Self::ordered_handler_loop(
            state.clone(),
            ordered_rx,
            shutdown_rx,
        ));

        let read_result = Self::read_loop(state.clone(), read, ordered_tx);
        tokio::pin!(read_result);

        let result = tokio::select! {
            read_result = &mut read_result => read_result,
            writer_result = &mut writer => match writer_result {
                Ok(Ok(())) => Ok(()),
                Ok(Err(err)) => Err(err),
                Err(err) => Err(err.into()),
            },
        };

        let shutdown_reason = match &result {
            Ok(()) => "plugin host connection is closed".to_string(),
            Err(err) => format!("plugin host connection is closed: {err}"),
        };
        Self::shutdown_runtime(&shutdown_tx, &state.pending_host_responses, shutdown_reason);
        if !writer.is_finished() {
            let _ = writer.await;
        }
        if !ordered_handlers.is_finished() {
            ordered_handlers.abort();
        }

        match ordered_handlers.await {
            Ok(()) => result,
            Err(err) if err.is_cancelled() => result,
            Err(err) if result.is_ok() => Err(err.into()),
            Err(_) => result,
        }
    }

    fn shutdown_runtime(
        shutdown_tx: &watch::Sender<bool>,
        pending_host_responses: &PendingHostResponses,
        reason: String,
    ) {
        let _ = shutdown_tx.send(true);
        Self::fail_pending_host_responses(pending_host_responses, reason);
    }

    fn fail_pending_host_responses(pending_host_responses: &PendingHostResponses, reason: String) {
        for sender in drain_pending_host_responses(pending_host_responses) {
            let _ = sender.send(Err(anyhow::anyhow!(reason.clone())));
        }
    }

    async fn ordered_handler_loop<P: Plugin + Clone + Sync + 'static>(
        state: Arc<RuntimeState<P>>,
        mut ordered_rx: mpsc::Receiver<OrderedPayload>,
        mut shutdown_rx: watch::Receiver<bool>,
    ) {
        loop {
            tokio::select! {
                _ = shutdown_rx.changed() => {
                    if *shutdown_rx.borrow() {
                        break;
                    }
                }
                payload = ordered_rx.recv() => {
                    let Some(payload) = payload else {
                        break;
                    };
                    let _ = Self::handle_payload(state.clone(), payload.request_id, payload.payload).await;
                }
            }
        }
    }

    async fn read_loop<P: Plugin + Clone + Sync + 'static>(
        state: Arc<RuntimeState<P>>,
        mut read: LocalReadHalf,
        ordered_tx: mpsc::Sender<OrderedPayload>,
    ) -> Result<()> {
        loop {
            let envelope = read_envelope_from(&mut *read).await?;
            if Self::complete_pending_host_response(&state, &envelope).await {
                continue;
            }
            if !Self::handle_envelope(state.clone(), &ordered_tx, envelope).await? {
                break;
            }
        }
        Ok(())
    }

    async fn write_loop(
        mut write: LocalWriteHalf,
        mut outbound_rx: mpsc::Receiver<proto::Envelope>,
        pending_host_responses: PendingHostResponses,
        shutdown_tx: watch::Sender<bool>,
        mut shutdown_rx: watch::Receiver<bool>,
    ) -> Result<()> {
        loop {
            tokio::select! {
                _ = shutdown_rx.changed() => {
                    if *shutdown_rx.borrow() {
                        Self::drain_outbound_before_shutdown(&mut write, &mut outbound_rx).await?;
                        return Ok(());
                    }
                }
                envelope = outbound_rx.recv() => {
                    let Some(envelope) = envelope else {
                        return Ok(());
                    };
                    if let Err(err) = write_envelope_to(&mut *write, &envelope).await {
                        let reason = format!("plugin host write failed: {err}");
                        Self::shutdown_runtime(&shutdown_tx, &pending_host_responses, reason);
                        return Err(err);
                    }
                }
            }
        }
    }

    async fn drain_outbound_before_shutdown(
        write: &mut LocalWriteHalf,
        outbound_rx: &mut mpsc::Receiver<proto::Envelope>,
    ) -> Result<()> {
        while let Ok(envelope) = outbound_rx.try_recv() {
            write_envelope_to(&mut **write, &envelope).await?;
        }
        Ok(())
    }

    async fn complete_pending_host_response<P: Plugin + Clone + Sync>(
        state: &RuntimeState<P>,
        envelope: &proto::Envelope,
    ) -> bool {
        let Some(sender) =
            remove_pending_host_response(&state.pending_host_responses, envelope.request_id)
        else {
            return false;
        };
        let _ = sender.send(Ok(envelope.clone()));
        true
    }

    async fn handle_envelope<P: Plugin + Clone + Sync + 'static>(
        state: Arc<RuntimeState<P>>,
        ordered_tx: &mpsc::Sender<OrderedPayload>,
        envelope: proto::Envelope,
    ) -> Result<bool> {
        let request_id = envelope.request_id;
        let Some(payload) = envelope.payload else {
            return Ok(true);
        };

        match payload {
            proto::envelope::Payload::InitializeRequest(request) => {
                Self::handle_initialize(state, request_id, request).await
            }
            proto::envelope::Payload::HealthRequest(_) => Self::handle_health(state, request_id),
            proto::envelope::Payload::ShutdownRequest(_) => {
                Self::write_payload(
                    &state,
                    request_id,
                    proto::envelope::Payload::ShutdownResponse(proto::ShutdownResponse {}),
                )
                .await?;
                Ok(false)
            }
            payload if Self::is_ordered_payload(&payload) => {
                Self::enqueue_ordered_payload(ordered_tx, request_id, payload).await
            }
            payload => Self::spawn_payload_handler(state, request_id, payload),
        }
    }

    fn is_ordered_payload(payload: &proto::envelope::Payload) -> bool {
        matches!(
            payload,
            proto::envelope::Payload::RpcNotification(_)
                | proto::envelope::Payload::ChannelMessage(_)
                | proto::envelope::Payload::BulkTransferMessage(_)
                | proto::envelope::Payload::MeshEvent(_)
                | proto::envelope::Payload::CancelStreamNotification(_)
                | proto::envelope::Payload::CloseStreamNotification(_)
                | proto::envelope::Payload::StreamError(_)
                | proto::envelope::Payload::ErrorResponse(_)
        )
    }

    async fn enqueue_ordered_payload(
        ordered_tx: &mpsc::Sender<OrderedPayload>,
        request_id: u64,
        payload: proto::envelope::Payload,
    ) -> Result<bool> {
        ordered_tx
            .send(OrderedPayload {
                request_id,
                payload,
            })
            .await
            .map_err(|_| anyhow::anyhow!("plugin ordered handler is closed"))?;
        Ok(true)
    }

    fn spawn_payload_handler<P: Plugin + Clone + Sync + 'static>(
        state: Arc<RuntimeState<P>>,
        request_id: u64,
        payload: proto::envelope::Payload,
    ) -> Result<bool> {
        tokio::spawn(async move {
            let _ = Self::handle_payload(state, request_id, payload).await;
        });
        Ok(true)
    }

    async fn handle_payload<P: Plugin + Clone + Sync>(
        state: Arc<RuntimeState<P>>,
        request_id: u64,
        payload: proto::envelope::Payload,
    ) -> Result<()> {
        match payload {
            proto::envelope::Payload::RpcRequest(request) => {
                let payload = Self::rpc_payload(&state, request).await;
                Self::write_payload(&state, request_id, payload).await
            }
            proto::envelope::Payload::InvokeServiceRequest(request) => {
                let payload = Self::invoke_service_payload(&state, request).await;
                Self::write_payload(&state, request_id, payload).await
            }
            proto::envelope::Payload::OpenStreamRequest(request) => {
                let payload = Self::open_stream_payload(&state, request).await;
                Self::write_payload(&state, request_id, payload).await
            }
            proto::envelope::Payload::RpcNotification(notification) => {
                Self::handle_rpc_notification(&state, notification)
                    .await
                    .map(|_| ())
            }
            proto::envelope::Payload::ChannelMessage(message) => {
                Self::handle_channel_message(&state, message)
                    .await
                    .map(|_| ())
            }
            proto::envelope::Payload::BulkTransferMessage(message) => {
                Self::handle_bulk_transfer_message(&state, message)
                    .await
                    .map(|_| ())
            }
            proto::envelope::Payload::MeshEvent(event) => {
                Self::handle_mesh_event(&state, event).await.map(|_| ())
            }
            proto::envelope::Payload::CancelStreamNotification(notification) => {
                Self::handle_cancel_stream(&state, notification)
                    .await
                    .map(|_| ())
            }
            proto::envelope::Payload::CloseStreamNotification(notification) => {
                Self::handle_close_stream(&state, notification)
                    .await
                    .map(|_| ())
            }
            proto::envelope::Payload::StreamError(error) => {
                Self::handle_stream_error(&state, error).await.map(|_| ())
            }
            proto::envelope::Payload::ErrorResponse(error) => {
                Self::handle_host_error(&state, error).await.map(|_| ())
            }
            _ => Ok(()),
        }?;
        Ok(())
    }

    async fn handle_initialize<P: Plugin + Clone + Sync>(
        state: Arc<RuntimeState<P>>,
        request_id: u64,
        request: proto::InitializeRequest,
    ) -> Result<bool> {
        let mut plugin = state.plugin.write().await;
        let mut context = Self::context(&state);
        let init_result = plugin
            .initialize(PluginInitializeRequest::from(request), &mut context)
            .await;
        if let Err(err) = init_result {
            Self::write_payload(
                &state,
                request_id,
                proto::envelope::Payload::ErrorResponse(err.into_error_response()),
            )
            .await?;
            return Ok(false);
        }

        Self::write_payload(
            &state,
            request_id,
            proto::envelope::Payload::InitializeResponse(proto::InitializeResponse {
                plugin_id: state.plugin_id.clone(),
                plugin_protocol_version: PROTOCOL_VERSION,
                plugin_version: plugin.plugin_version(),
                server_info_json: serde_json::to_string(&plugin.server_info())?,
                capabilities: plugin.capabilities(),
                manifest: plugin.manifest(),
            }),
        )
        .await?;

        let mut context = Self::context(&state);
        plugin.on_initialized(&mut context).await?;
        Ok(true)
    }

    fn handle_health<P: Plugin + Clone + Sync + 'static>(
        state: Arc<RuntimeState<P>>,
        request_id: u64,
    ) -> Result<bool> {
        tokio::spawn(async move {
            let mut plugin = Self::plugin_for_request(&state).await;
            let mut context = Self::context(&state);
            let payload = match plugin.health(&mut context).await {
                Ok(detail) => proto::envelope::Payload::HealthResponse(proto::HealthResponse {
                    status: proto::health_response::Status::Ok as i32,
                    detail,
                }),
                Err(err) => proto::envelope::Payload::ErrorResponse(
                    PluginError::internal(format!("health check failed: {err}"))
                        .into_error_response(),
                ),
            };
            let _ = Self::write_payload(&state, request_id, payload).await;
        });
        Ok(true)
    }

    async fn plugin_for_request<P: Plugin + Clone + Sync>(state: &RuntimeState<P>) -> P {
        state.plugin.read().await.clone()
    }

    async fn rpc_payload<P: Plugin + Clone + Sync>(
        state: &RuntimeState<P>,
        request: proto::RpcRequest,
    ) -> proto::envelope::Payload {
        let mut plugin = Self::plugin_for_request(state).await;
        let mut context = Self::context(state);
        match plugin.handle_rpc(request, &mut context).await {
            Ok(payload) => payload,
            Err(err) => proto::envelope::Payload::ErrorResponse(err.into_error_response()),
        }
    }

    async fn invoke_service_payload<P: Plugin + Clone + Sync>(
        state: &RuntimeState<P>,
        request: proto::InvokeServiceRequest,
    ) -> proto::envelope::Payload {
        let mut plugin = Self::plugin_for_request(state).await;
        let mut context = Self::context(state);
        match plugin.invoke_service(request, &mut context).await {
            Ok(Some(response)) => proto::envelope::Payload::InvokeServiceResponse(response),
            Ok(None) => proto::envelope::Payload::ErrorResponse(
                PluginError::method_not_found("Unsupported service invocation")
                    .into_error_response(),
            ),
            Err(err) => proto::envelope::Payload::ErrorResponse(err.into_error_response()),
        }
    }

    async fn open_stream_payload<P: Plugin + Clone + Sync>(
        state: &RuntimeState<P>,
        request: proto::OpenStreamRequest,
    ) -> proto::envelope::Payload {
        let mut plugin = Self::plugin_for_request(state).await;
        let mut context = Self::context(state);
        match plugin.open_stream(request, &mut context).await {
            Ok(Some(response)) => proto::envelope::Payload::OpenStreamResponse(response),
            Ok(None) => proto::envelope::Payload::ErrorResponse(
                PluginError::method_not_found("Unsupported stream control message 'open_stream'")
                    .into_error_response(),
            ),
            Err(err) => proto::envelope::Payload::ErrorResponse(err.into_error_response()),
        }
    }

    async fn handle_rpc_notification<P: Plugin + Clone + Sync>(
        state: &RuntimeState<P>,
        notification: proto::RpcNotification,
    ) -> Result<bool> {
        let mut plugin = Self::plugin_for_request(state).await;
        let mut context = Self::context(state);
        plugin
            .on_rpc_notification(notification, &mut context)
            .await?;
        Ok(true)
    }

    async fn handle_channel_message<P: Plugin + Clone + Sync>(
        state: &RuntimeState<P>,
        message: proto::ChannelMessage,
    ) -> Result<bool> {
        let mut plugin = Self::plugin_for_request(state).await;
        let mut context = Self::context(state);
        plugin.on_channel_message(message, &mut context).await?;
        Ok(true)
    }

    async fn handle_bulk_transfer_message<P: Plugin + Clone + Sync>(
        state: &RuntimeState<P>,
        message: proto::BulkTransferMessage,
    ) -> Result<bool> {
        let mut plugin = Self::plugin_for_request(state).await;
        let mut context = Self::context(state);
        plugin
            .on_bulk_transfer_message(message, &mut context)
            .await?;
        Ok(true)
    }

    async fn handle_mesh_event<P: Plugin + Clone + Sync>(
        state: &RuntimeState<P>,
        event: proto::MeshEvent,
    ) -> Result<bool> {
        let mut plugin = Self::plugin_for_request(state).await;
        let mut context = Self::context(state);
        plugin.on_mesh_event(event, &mut context).await?;
        Ok(true)
    }

    async fn handle_cancel_stream<P: Plugin + Clone + Sync>(
        state: &RuntimeState<P>,
        notification: proto::CancelStreamNotification,
    ) -> Result<bool> {
        let mut plugin = Self::plugin_for_request(state).await;
        let mut context = Self::context(state);
        plugin.on_cancel_stream(notification, &mut context).await?;
        Ok(true)
    }

    async fn handle_close_stream<P: Plugin + Clone + Sync>(
        state: &RuntimeState<P>,
        notification: proto::CloseStreamNotification,
    ) -> Result<bool> {
        let mut plugin = Self::plugin_for_request(state).await;
        let mut context = Self::context(state);
        plugin.on_close_stream(notification, &mut context).await?;
        Ok(true)
    }

    async fn handle_stream_error<P: Plugin + Clone + Sync>(
        state: &RuntimeState<P>,
        error: proto::StreamError,
    ) -> Result<bool> {
        let mut plugin = Self::plugin_for_request(state).await;
        let mut context = Self::context(state);
        plugin.on_stream_error(error, &mut context).await?;
        Ok(true)
    }

    async fn handle_host_error<P: Plugin + Clone + Sync>(
        state: &RuntimeState<P>,
        error: proto::ErrorResponse,
    ) -> Result<bool> {
        let mut plugin = Self::plugin_for_request(state).await;
        let mut context = Self::context(state);
        plugin.on_host_error(error, &mut context).await?;
        Ok(true)
    }

    fn context<P: Plugin + Clone + Sync>(state: &RuntimeState<P>) -> PluginContext<'static> {
        PluginContext::new(
            state.plugin_id.clone(),
            state.outbound_tx.clone(),
            state.pending_host_responses.clone(),
        )
    }

    async fn write_payload<P: Plugin + Clone + Sync>(
        state: &RuntimeState<P>,
        request_id: u64,
        payload: proto::envelope::Payload,
    ) -> Result<()> {
        state
            .outbound_tx
            .send(proto::Envelope {
                protocol_version: PROTOCOL_VERSION,
                plugin_id: state.plugin_id.clone(),
                request_id,
                payload: Some(payload),
            })
            .await
            .map_err(|_| anyhow::anyhow!("plugin host connection is closed"))
    }
}

fn parse_service_input<T: DeserializeOwned>(input_json: &str) -> PluginResult<T> {
    let input = if input_json.trim().is_empty() {
        "null"
    } else {
        input_json
    };
    serde_json::from_str(input)
        .map_err(|err| PluginError::invalid_params(format!("Invalid service input JSON: {err}")))
}

fn serialize_service_output<T: Serialize>(value: &T) -> PluginResult<String> {
    serde_json::to_string(value)
        .map_err(|err| PluginError::internal(format!("Serialize service output: {err}")))
}

fn normalize_call_tool_output(result: &CallToolResult) -> PluginResult<String> {
    if let Some(value) = &result.structured_content {
        return serialize_service_output(value);
    }
    if let Some(text) = result.content.first().and_then(|content| content.as_text()) {
        return Ok(text.text.clone());
    }
    serialize_service_output(&result.content)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{mcp, plugin, plugin_server_info};
    use crate::{read_envelope, write_envelope};
    use rmcp::model::{
        ArgumentInfo, PromptMessage, PromptMessageContent, PromptMessageRole, Reference,
    };
    use serde_json::json;
    use tokio::sync::{Barrier, Notify};
    use tokio::time::{Duration, timeout};

    #[derive(Clone, Debug, Default, Deserialize, Serialize, schemars::JsonSchema)]
    struct DemoArgs {
        #[serde(default)]
        message: String,
    }

    fn test_context() -> PluginContext<'static> {
        let (outbound_tx, _outbound_rx) = mpsc::channel(8);
        PluginContext::new(
            "demo".into(),
            outbound_tx,
            Arc::new(Mutex::new(HashMap::new())),
        )
    }

    fn test_channel_message(message_kind: &str) -> proto::ChannelMessage {
        proto::ChannelMessage {
            channel: "events".into(),
            source_peer_id: "peer-a".into(),
            target_peer_id: "peer-b".into(),
            content_type: "text/plain".into(),
            body: Vec::new(),
            message_kind: message_kind.into(),
            correlation_id: String::new(),
            metadata_json: String::new(),
        }
    }

    #[tokio::test]
    async fn invoke_service_dispatches_operation_prompt_resource_and_completion() {
        let mut plugin = plugin! {
            metadata: PluginMetadata::new(
                "demo",
                "1.0.0",
                plugin_server_info("demo", "1.0.0", "Demo", "Demo plugin", None::<String>),
            ),
            mcp: [
                mcp::tool("echo")
                    .description("Echo input")
                    .input::<DemoArgs>()
                    .handle(|args, _context| Box::pin(async move {
                        Ok(json!({ "echo": args.message }))
                    })),
                mcp::resource("demo://state")
                    .name("State")
                    .handle(|request, _context| Box::pin(async move {
                        Ok(crate::read_resource_result(vec![
                            rmcp::model::ResourceContents::text("state", request.uri),
                        ]))
                    })),
                mcp::prompt("brief")
                    .description("Brief")
                    .handle(|request, _context| Box::pin(async move {
                        Ok(crate::get_prompt_result(vec![PromptMessage::new(
                            PromptMessageRole::User,
                            PromptMessageContent::text(format!("brief:{}", request.name)),
                        )]))
                    })),
                mcp::completion("prompt.brief.topic")
                    .handle(|_request, _context| Box::pin(async move {
                        crate::complete_result(vec!["alpha".into()])
                    })),
            ],
        };

        let mut context = test_context();

        let op = plugin
            .invoke_service(
                proto::InvokeServiceRequest {
                    kind: proto::ServiceKind::Operation as i32,
                    service_name: "echo".into(),
                    input_json: json!({ "message": "hello" }).to_string(),
                },
                &mut context,
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&op.output_json).unwrap(),
            json!({"echo": "hello"})
        );

        let prompt = plugin
            .invoke_service(
                proto::InvokeServiceRequest {
                    kind: proto::ServiceKind::Prompt as i32,
                    service_name: "brief".into(),
                    input_json: serde_json::to_string(&GetPromptRequestParams::new("brief"))
                        .unwrap(),
                },
                &mut context,
            )
            .await
            .unwrap()
            .unwrap();
        let prompt_result: GetPromptResult = serde_json::from_str(&prompt.output_json).unwrap();
        assert_eq!(prompt_result.messages.len(), 1);

        let resource = plugin
            .invoke_service(
                proto::InvokeServiceRequest {
                    kind: proto::ServiceKind::Resource as i32,
                    service_name: "demo://state".into(),
                    input_json: serde_json::to_string(&ReadResourceRequestParams::new(
                        "demo://state",
                    ))
                    .unwrap(),
                },
                &mut context,
            )
            .await
            .unwrap()
            .unwrap();
        let resource_result: ReadResourceResult =
            serde_json::from_str(&resource.output_json).unwrap();
        assert_eq!(resource_result.contents.len(), 1);

        let completion = plugin
            .invoke_service(
                proto::InvokeServiceRequest {
                    kind: proto::ServiceKind::Completion as i32,
                    service_name: "brief".into(),
                    input_json: serde_json::to_string(&CompleteRequestParams::new(
                        Reference::for_prompt("brief"),
                        ArgumentInfo {
                            name: "topic".into(),
                            value: "a".into(),
                        },
                    ))
                    .unwrap(),
                },
                &mut context,
            )
            .await
            .unwrap()
            .unwrap();
        let completion_result: CompleteResult =
            serde_json::from_str(&completion.output_json).unwrap();
        assert_eq!(
            completion_result.completion.values,
            vec![String::from("alpha")]
        );
    }

    #[tokio::test]
    async fn health_request_returns_while_operation_is_running() {
        let started = Arc::new(Notify::new());
        let started_for_tool = started.clone();
        let plugin = plugin! {
            metadata: PluginMetadata::new(
                "demo",
                "1.0.0",
                plugin_server_info("demo", "1.0.0", "Demo", "Demo plugin", None::<String>),
            ),
            mcp: [
                mcp::tool("slow")
                    .description("Slow operation")
                    .input::<DemoArgs>()
                    .handle(move |_args, _context| {
                        let started = started_for_tool.clone();
                        Box::pin(async move {
                            started.notify_one();
                            tokio::time::sleep(Duration::from_millis(300)).await;
                            Ok(json!({ "done": true }))
                        })
                    }),
            ],
        };

        #[cfg(unix)]
        let (plugin_stream, host_stream) = tokio::net::UnixStream::pair().unwrap();
        #[cfg(not(unix))]
        panic!("runtime stream tests are only implemented for unix");

        let runtime = tokio::spawn(PluginRuntime::run_with_stream(
            plugin,
            LocalStream::Unix(plugin_stream),
        ));
        let mut host_stream = LocalStream::Unix(host_stream);
        write_envelope(
            &mut host_stream,
            &proto::Envelope {
                protocol_version: PROTOCOL_VERSION,
                plugin_id: "demo".into(),
                request_id: 1,
                payload: Some(proto::envelope::Payload::InvokeServiceRequest(
                    proto::InvokeServiceRequest {
                        kind: proto::ServiceKind::Operation as i32,
                        service_name: "slow".into(),
                        input_json: "{}".into(),
                    },
                )),
            },
        )
        .await
        .unwrap();

        started.notified().await;
        write_envelope(
            &mut host_stream,
            &proto::Envelope {
                protocol_version: PROTOCOL_VERSION,
                plugin_id: "demo".into(),
                request_id: 2,
                payload: Some(proto::envelope::Payload::HealthRequest(
                    proto::HealthRequest {},
                )),
            },
        )
        .await
        .unwrap();

        let health = timeout(Duration::from_millis(150), read_envelope(&mut host_stream))
            .await
            .expect("health response should not wait for slow operation")
            .unwrap();
        assert_eq!(health.request_id, 2);
        assert!(matches!(
            health.payload,
            Some(proto::envelope::Payload::HealthResponse(_))
        ));

        let invoke = timeout(Duration::from_secs(1), read_envelope(&mut host_stream))
            .await
            .expect("slow operation should still complete")
            .unwrap();
        assert_eq!(invoke.request_id, 1);

        runtime.abort();
    }

    #[tokio::test]
    async fn invokes_service_requests_concurrently() {
        let barrier = Arc::new(Barrier::new(2));
        let barrier_for_tool = barrier.clone();
        let plugin = plugin! {
            metadata: PluginMetadata::new(
                "demo",
                "1.0.0",
                plugin_server_info("demo", "1.0.0", "Demo", "Demo plugin", None::<String>),
            ),
            mcp: [
                mcp::tool("barrier")
                    .description("Waits for another request")
                    .input::<DemoArgs>()
                    .handle(move |_args, _context| {
                        let barrier = barrier_for_tool.clone();
                        Box::pin(async move {
                            barrier.wait().await;
                            Ok(json!({ "done": true }))
                        })
                    }),
            ],
        };

        #[cfg(unix)]
        let (plugin_stream, host_stream) = tokio::net::UnixStream::pair().unwrap();
        #[cfg(not(unix))]
        panic!("runtime stream tests are only implemented for unix");

        let runtime = tokio::spawn(PluginRuntime::run_with_stream(
            plugin,
            LocalStream::Unix(plugin_stream),
        ));
        let mut host_stream = LocalStream::Unix(host_stream);

        for request_id in [1, 2] {
            write_envelope(
                &mut host_stream,
                &proto::Envelope {
                    protocol_version: PROTOCOL_VERSION,
                    plugin_id: "demo".into(),
                    request_id,
                    payload: Some(proto::envelope::Payload::InvokeServiceRequest(
                        proto::InvokeServiceRequest {
                            kind: proto::ServiceKind::Operation as i32,
                            service_name: "barrier".into(),
                            input_json: "{}".into(),
                        },
                    )),
                },
            )
            .await
            .unwrap();
        }

        let first = timeout(Duration::from_secs(1), read_envelope(&mut host_stream))
            .await
            .expect("first invoke response should not block behind another request")
            .unwrap();
        let second = timeout(Duration::from_secs(1), read_envelope(&mut host_stream))
            .await
            .expect("second invoke response should not block behind another request")
            .unwrap();
        let mut request_ids = vec![first.request_id, second.request_id];
        request_ids.sort_unstable();
        assert_eq!(request_ids, vec![1, 2]);

        runtime.abort();
    }

    #[tokio::test]
    async fn dropped_open_mesh_stream_request_removes_pending_response() {
        let (outbound_tx, mut outbound_rx) = mpsc::channel(8);
        let pending_host_responses = Arc::new(Mutex::new(HashMap::new()));
        let mut context =
            PluginContext::new("demo".into(), outbound_tx, pending_host_responses.clone());

        let request = tokio::spawn(async move {
            let _ = context
                .open_mesh_stream(proto::OpenMeshStreamRequest::default())
                .await;
        });

        let outbound = outbound_rx
            .recv()
            .await
            .expect("request should be sent before awaiting host response");
        assert_ne!(outbound.request_id, 0);
        assert_eq!(
            pending_host_responses
                .lock()
                .expect("pending map should not be poisoned")
                .len(),
            1
        );

        request.abort();
        let _ = request.await;

        assert!(
            pending_host_responses
                .lock()
                .expect("pending map should not be poisoned")
                .is_empty()
        );
    }

    #[tokio::test]
    async fn ordered_notifications_do_not_overtake_each_other() {
        let first_started = Arc::new(Notify::new());
        let release_first = Arc::new(Notify::new());
        let handled = Arc::new(Mutex::new(Vec::<String>::new()));
        let first_started_for_handler = first_started.clone();
        let release_first_for_handler = release_first.clone();
        let handled_for_handler = handled.clone();
        let plugin = SimplePlugin::new(PluginMetadata::new(
            "demo",
            "1.0.0",
            plugin_server_info("demo", "1.0.0", "Demo", "Demo plugin", None::<String>),
        ))
        .on_channel_message(move |message, _context| {
            let first_started = first_started_for_handler.clone();
            let release_first = release_first_for_handler.clone();
            let handled = handled_for_handler.clone();
            Box::pin(async move {
                if message.message_kind == "first" {
                    first_started.notify_one();
                    release_first.notified().await;
                }
                handled
                    .lock()
                    .expect("handled list should not be poisoned")
                    .push(message.message_kind);
                Ok(())
            })
        });

        #[cfg(unix)]
        let (plugin_stream, host_stream) = tokio::net::UnixStream::pair().unwrap();
        #[cfg(not(unix))]
        panic!("runtime stream tests are only implemented for unix");

        let runtime = tokio::spawn(PluginRuntime::run_with_stream(
            plugin,
            LocalStream::Unix(plugin_stream),
        ));
        let mut host_stream = LocalStream::Unix(host_stream);

        for (request_id, message_kind) in [(1, "first"), (2, "second")] {
            write_envelope(
                &mut host_stream,
                &proto::Envelope {
                    protocol_version: PROTOCOL_VERSION,
                    plugin_id: "demo".into(),
                    request_id,
                    payload: Some(proto::envelope::Payload::ChannelMessage(
                        test_channel_message(message_kind),
                    )),
                },
            )
            .await
            .unwrap();
        }

        first_started.notified().await;
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(
            handled
                .lock()
                .expect("handled list should not be poisoned")
                .is_empty(),
            "second message should wait behind the first ordered handler"
        );

        release_first.notify_one();
        timeout(Duration::from_secs(1), async {
            loop {
                if handled
                    .lock()
                    .expect("handled list should not be poisoned")
                    .len()
                    == 2
                {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("ordered handlers should finish");

        assert_eq!(
            *handled.lock().expect("handled list should not be poisoned"),
            vec![String::from("first"), String::from("second")]
        );

        runtime.abort();
    }
}
