//! Shared runtime and protocol helpers for mesh-llm plugins.
//!
//! For simple tools-only plugins, implement [`Plugin::list_tools`] and
//! [`Plugin::call_tool`] instead of overriding raw RPC dispatch. Override
//! [`Plugin::handle_rpc`] only when you need methods beyond the standard
//! `tools/list` and `tools/call` flow.

use anyhow::{Context, Result, bail};
use prost::Message;
use rmcp::model::{
    CallToolResult, Content, ErrorCode, Implementation, ListToolsResult, ServerCapabilities,
    ServerInfo, Tool,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde::de::DeserializeOwned;
use serde_json::json;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

pub use async_trait::async_trait;

#[allow(dead_code)]
pub mod proto {
    include!(concat!(env!("OUT_DIR"), "/meshllm.plugin.v1.rs"));
}

pub const PROTOCOL_VERSION: u32 = 1;

fn default_arguments() -> serde_json::Value {
    json!({})
}

pub enum LocalStream {
    #[cfg(unix)]
    Unix(tokio::net::UnixStream),
    #[cfg(windows)]
    PipeClient(tokio::net::windows::named_pipe::NamedPipeClient),
}

pub struct PluginContext<'a> {
    stream: &'a mut LocalStream,
    plugin_id: &'a str,
}

impl<'a> PluginContext<'a> {
    pub async fn send_channel(&mut self, message: proto::ChannelMessage) -> Result<()> {
        self.send_channel_message(message).await
    }

    pub async fn send_channel_message(&mut self, message: proto::ChannelMessage) -> Result<()> {
        send_channel_message(self.stream, self.plugin_id, message).await
    }

    pub async fn send_bulk(&mut self, message: proto::BulkTransferMessage) -> Result<()> {
        self.send_bulk_transfer_message(message).await
    }

    pub async fn send_bulk_transfer_message(
        &mut self,
        message: proto::BulkTransferMessage,
    ) -> Result<()> {
        send_bulk_transfer_message(self.stream, self.plugin_id, message).await
    }

    pub async fn notify_host<P>(&mut self, method: &str, params: P) -> Result<()>
    where
        P: Serialize,
    {
        write_envelope(
            self.stream,
            &proto::Envelope {
                protocol_version: PROTOCOL_VERSION,
                plugin_id: self.plugin_id.to_string(),
                request_id: 0,
                payload: Some(proto::envelope::Payload::RpcNotification(
                    proto::RpcNotification {
                        method: method.to_string(),
                        params_json: serde_json::to_string(&params)?,
                    },
                )),
            },
        )
        .await
    }
}

#[derive(Debug, Clone)]
pub struct PluginError {
    pub code: i32,
    pub message: String,
    pub data_json: String,
}

impl PluginError {
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self {
            code: ErrorCode::INVALID_REQUEST.0,
            message: message.into(),
            data_json: String::new(),
        }
    }

    pub fn method_not_found(message: impl Into<String>) -> Self {
        Self {
            code: ErrorCode::METHOD_NOT_FOUND.0,
            message: message.into(),
            data_json: String::new(),
        }
    }

    pub fn invalid_params(message: impl Into<String>) -> Self {
        Self {
            code: ErrorCode::INVALID_PARAMS.0,
            message: message.into(),
            data_json: String::new(),
        }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            code: ErrorCode::INTERNAL_ERROR.0,
            message: message.into(),
            data_json: String::new(),
        }
    }

    fn into_error_response(self) -> proto::ErrorResponse {
        proto::ErrorResponse {
            code: self.code,
            message: self.message,
            data_json: self.data_json,
        }
    }
}

impl std::fmt::Display for PluginError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for PluginError {}

impl From<anyhow::Error> for PluginError {
    fn from(value: anyhow::Error) -> Self {
        Self::internal(value.to_string())
    }
}

pub type PluginResult<T> = std::result::Result<T, PluginError>;
pub type PluginRpcResult = PluginResult<proto::envelope::Payload>;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolCallRequest {
    pub name: String,
    #[serde(default = "default_arguments")]
    pub arguments: serde_json::Value,
}

impl ToolCallRequest {
    pub fn arguments<T: DeserializeOwned>(&self) -> PluginResult<T> {
        serde_json::from_value(self.arguments.clone()).map_err(|err| {
            PluginError::invalid_params(format!(
                "Invalid arguments for tool '{}': {err}",
                self.name
            ))
        })
    }

    pub fn arguments_or_default<T>(&self) -> PluginResult<T>
    where
        T: DeserializeOwned + Default,
    {
        self.arguments()
    }
}

pub fn json_string<T: Serialize>(value: &T) -> PluginResult<String> {
    serde_json::to_string(value).map_err(|err| PluginError::internal(err.to_string()))
}

pub fn json_bytes<T: Serialize>(value: &T) -> PluginResult<Vec<u8>> {
    serde_json::to_vec(value).map_err(|err| PluginError::internal(err.to_string()))
}

pub fn structured_tool_result<T: Serialize>(value: T) -> PluginResult<CallToolResult> {
    let value = serde_json::to_value(value).map_err(|err| PluginError::internal(err.to_string()))?;
    Ok(CallToolResult::structured(value))
}

pub fn tool_error(message: impl Into<String>) -> CallToolResult {
    CallToolResult::error(vec![Content::text(message.into())])
}

pub fn list_tools(tools: Vec<Tool>) -> ListToolsResult {
    ListToolsResult {
        tools,
        meta: None,
        next_cursor: None,
    }
}

pub fn plugin_server_info(
    implementation_name: impl Into<String>,
    implementation_version: impl Into<String>,
    title: impl Into<String>,
    description: impl Into<String>,
    instructions: Option<impl Into<String>>,
) -> ServerInfo {
    let info = ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
        .with_server_info(
            Implementation::new(implementation_name, implementation_version)
                .with_title(title)
                .with_description(description),
        );
    match instructions {
        Some(instructions) => info.with_instructions(instructions.into()),
        None => info,
    }
}

pub fn empty_object_schema() -> serde_json::Map<String, serde_json::Value> {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false
    })
    .as_object()
    .cloned()
    .unwrap()
}

pub fn json_schema_for<T: JsonSchema>() -> serde_json::Map<String, serde_json::Value> {
    serde_json::to_value(schemars::schema_for!(T))
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

pub fn tool_with_schema(
    name: impl Into<String>,
    description: impl Into<String>,
    schema: serde_json::Map<String, serde_json::Value>,
) -> Tool {
    Tool::new(name.into(), description.into(), Arc::new(schema))
}

pub fn json_schema_tool<T: JsonSchema>(
    name: impl Into<String>,
    description: impl Into<String>,
) -> Tool {
    tool_with_schema(name, description, json_schema_for::<T>())
}

pub fn channel_message(
    channel: impl Into<String>,
    target_peer_id: impl Into<String>,
    content_type: impl Into<String>,
    body: Vec<u8>,
    message_kind: impl Into<String>,
) -> proto::ChannelMessage {
    proto::ChannelMessage {
        channel: channel.into(),
        source_peer_id: String::new(),
        target_peer_id: target_peer_id.into(),
        content_type: content_type.into(),
        body,
        message_kind: message_kind.into(),
        correlation_id: String::new(),
        metadata_json: String::new(),
    }
}

pub fn json_channel_message<T: Serialize>(
    channel: impl Into<String>,
    target_peer_id: impl Into<String>,
    message_kind: impl Into<String>,
    payload: &T,
) -> PluginResult<proto::ChannelMessage> {
    Ok(channel_message(
        channel,
        target_peer_id,
        "application/json",
        json_bytes(payload)?,
        message_kind,
    ))
}

pub fn bulk_transfer_message(
    kind: i32,
    channel: impl Into<String>,
    target_peer_id: impl Into<String>,
    content_type: impl Into<String>,
    total_bytes: u64,
    offset: u64,
    body: Vec<u8>,
    final_chunk: bool,
) -> proto::BulkTransferMessage {
    proto::BulkTransferMessage {
        kind,
        transfer_id: String::new(),
        channel: channel.into(),
        source_peer_id: String::new(),
        target_peer_id: target_peer_id.into(),
        content_type: content_type.into(),
        correlation_id: String::new(),
        metadata_json: String::new(),
        total_bytes,
        offset,
        body,
        final_chunk,
    }
}

pub fn json_response<T: Serialize>(value: &T) -> PluginRpcResult {
    Ok(proto::envelope::Payload::RpcResponse(proto::RpcResponse {
        result_json: serde_json::to_string(value)
            .map_err(|err| PluginError::internal(err.to_string()))?,
    }))
}

pub fn parse_rpc_params<T: DeserializeOwned>(request: &proto::RpcRequest) -> Result<T, PluginError> {
    serde_json::from_str(&request.params_json)
        .map_err(|err| PluginError::invalid_params(format!("Invalid params for '{}': {err}", request.method)))
}

pub fn parse_tool_call_request(request: &proto::RpcRequest) -> PluginResult<ToolCallRequest> {
    parse_rpc_params(request)
}

pub fn parse_optional_json(raw: &str) -> Option<serde_json::Value> {
    if raw.trim().is_empty() {
        None
    } else {
        serde_json::from_str(raw).ok()
    }
}

#[async_trait]
pub trait Plugin: Send {
    fn plugin_id(&self) -> &str;
    fn plugin_version(&self) -> String;
    fn server_info(&self) -> ServerInfo;

    fn capabilities(&self) -> Vec<String> {
        Vec::new()
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

    async fn on_host_error(
        &mut self,
        error: proto::ErrorResponse,
        _context: &mut PluginContext<'_>,
    ) -> Result<()> {
        bail!("host error: {}", error.message)
    }
}

pub struct PluginRuntime;

impl PluginRuntime {
    pub async fn run<P: Plugin>(plugin: P) -> Result<()> {
        let stream = connect_from_env().await?;
        Self::run_with_stream(plugin, stream).await
    }

    pub async fn run_with_stream<P: Plugin>(mut plugin: P, mut stream: LocalStream) -> Result<()> {
        loop {
            let envelope = read_envelope(&mut stream).await?;
            let request_id = envelope.request_id;
            let plugin_id = plugin.plugin_id().to_string();

            match envelope.payload {
                Some(proto::envelope::Payload::InitializeRequest(_)) => {
                    let response = proto::Envelope {
                        protocol_version: PROTOCOL_VERSION,
                        plugin_id: plugin_id.clone(),
                        request_id,
                        payload: Some(proto::envelope::Payload::InitializeResponse(
                            proto::InitializeResponse {
                                plugin_id: plugin_id.clone(),
                                plugin_protocol_version: PROTOCOL_VERSION,
                                plugin_version: plugin.plugin_version(),
                                server_info_json: serde_json::to_string(&plugin.server_info())?,
                                capabilities: plugin.capabilities(),
                            },
                        )),
                    };
                    write_envelope(&mut stream, &response).await?;
                    let mut context = PluginContext {
                        stream: &mut stream,
                        plugin_id: &plugin_id,
                    };
                    plugin.on_initialized(&mut context).await?;
                }
                Some(proto::envelope::Payload::HealthRequest(_)) => {
                    let detail = {
                        let mut context = PluginContext {
                            stream: &mut stream,
                            plugin_id: &plugin_id,
                        };
                        plugin.health(&mut context).await?
                    };
                    let response = proto::Envelope {
                        protocol_version: PROTOCOL_VERSION,
                        plugin_id: plugin_id.clone(),
                        request_id,
                        payload: Some(proto::envelope::Payload::HealthResponse(
                            proto::HealthResponse {
                                status: proto::health_response::Status::Ok as i32,
                                detail,
                            },
                        )),
                    };
                    write_envelope(&mut stream, &response).await?;
                }
                Some(proto::envelope::Payload::ShutdownRequest(_)) => {
                    let response = proto::Envelope {
                        protocol_version: PROTOCOL_VERSION,
                        plugin_id: plugin_id.clone(),
                        request_id,
                        payload: Some(proto::envelope::Payload::ShutdownResponse(
                            proto::ShutdownResponse {},
                        )),
                    };
                    write_envelope(&mut stream, &response).await?;
                    break;
                }
                Some(proto::envelope::Payload::RpcRequest(request)) => {
                    let payload = {
                        let mut context = PluginContext {
                            stream: &mut stream,
                            plugin_id: &plugin_id,
                        };
                        match plugin.handle_rpc(request, &mut context).await {
                            Ok(payload) => payload,
                            Err(err) => {
                                proto::envelope::Payload::ErrorResponse(err.into_error_response())
                            }
                        }
                    };
                    write_envelope(
                        &mut stream,
                        &proto::Envelope {
                            protocol_version: PROTOCOL_VERSION,
                            plugin_id: plugin_id.clone(),
                            request_id,
                            payload: Some(payload),
                        },
                    )
                    .await?;
                }
                Some(proto::envelope::Payload::RpcNotification(notification)) => {
                    let mut context = PluginContext {
                        stream: &mut stream,
                        plugin_id: &plugin_id,
                    };
                    plugin.on_rpc_notification(notification, &mut context).await?;
                }
                Some(proto::envelope::Payload::ChannelMessage(message)) => {
                    let mut context = PluginContext {
                        stream: &mut stream,
                        plugin_id: &plugin_id,
                    };
                    plugin.on_channel_message(message, &mut context).await?;
                }
                Some(proto::envelope::Payload::BulkTransferMessage(message)) => {
                    let mut context = PluginContext {
                        stream: &mut stream,
                        plugin_id: &plugin_id,
                    };
                    plugin.on_bulk_transfer_message(message, &mut context).await?;
                }
                Some(proto::envelope::Payload::MeshEvent(event)) => {
                    let mut context = PluginContext {
                        stream: &mut stream,
                        plugin_id: &plugin_id,
                    };
                    plugin.on_mesh_event(event, &mut context).await?;
                }
                Some(proto::envelope::Payload::ErrorResponse(error)) => {
                    let mut context = PluginContext {
                        stream: &mut stream,
                        plugin_id: &plugin_id,
                    };
                    plugin.on_host_error(error, &mut context).await?;
                }
                _ => {}
            }
        }

        Ok(())
    }
}

impl LocalStream {
    async fn write_all(&mut self, bytes: &[u8]) -> Result<()> {
        match self {
            #[cfg(unix)]
            LocalStream::Unix(stream) => stream.write_all(bytes).await?,
            #[cfg(windows)]
            LocalStream::PipeClient(stream) => stream.write_all(bytes).await?,
        }
        Ok(())
    }

    async fn read_exact(&mut self, bytes: &mut [u8]) -> Result<()> {
        match self {
            #[cfg(unix)]
            LocalStream::Unix(stream) => {
                let _ = stream.read_exact(bytes).await?;
            }
            #[cfg(windows)]
            LocalStream::PipeClient(stream) => {
                let _ = stream.read_exact(bytes).await?;
            }
        }
        Ok(())
    }
}

pub async fn connect_from_env() -> Result<LocalStream> {
    let endpoint = std::env::var("MESH_LLM_PLUGIN_ENDPOINT")
        .context("MESH_LLM_PLUGIN_ENDPOINT is not set for plugin process")?;
    let transport =
        std::env::var("MESH_LLM_PLUGIN_TRANSPORT").unwrap_or_else(|_| default_transport().into());

    match transport.as_str() {
        #[cfg(unix)]
        "unix" => Ok(LocalStream::Unix(
            tokio::net::UnixStream::connect(&endpoint).await?,
        )),
        #[cfg(windows)]
        "pipe" => Ok(LocalStream::PipeClient(
            tokio::net::windows::named_pipe::ClientOptions::new().open(&endpoint)?,
        )),
        _ => bail!("Unsupported plugin transport '{transport}'"),
    }
}

pub async fn write_envelope(stream: &mut LocalStream, envelope: &proto::Envelope) -> Result<()> {
    let mut body = Vec::new();
    envelope.encode(&mut body)?;
    stream.write_all(&(body.len() as u32).to_le_bytes()).await?;
    stream.write_all(&body).await?;
    Ok(())
}

pub async fn read_envelope(stream: &mut LocalStream) -> Result<proto::Envelope> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf).await?;
    let len = u32::from_le_bytes(len_buf) as usize;
    if len > 16 * 1024 * 1024 {
        bail!("Plugin frame too large");
    }
    let mut body = vec![0u8; len];
    stream.read_exact(&mut body).await?;
    Ok(proto::Envelope::decode(body.as_slice())?)
}

pub async fn send_channel_message(
    stream: &mut LocalStream,
    plugin_id: &str,
    message: proto::ChannelMessage,
) -> Result<()> {
    write_envelope(
        stream,
        &proto::Envelope {
            protocol_version: PROTOCOL_VERSION,
            plugin_id: plugin_id.to_string(),
            request_id: 0,
            payload: Some(proto::envelope::Payload::ChannelMessage(message)),
        },
    )
    .await
}

pub async fn send_bulk_transfer_message(
    stream: &mut LocalStream,
    plugin_id: &str,
    message: proto::BulkTransferMessage,
) -> Result<()> {
    write_envelope(
        stream,
        &proto::Envelope {
            protocol_version: PROTOCOL_VERSION,
            plugin_id: plugin_id.to_string(),
            request_id: 0,
            payload: Some(proto::envelope::Payload::BulkTransferMessage(message)),
        },
    )
    .await
}

fn default_transport() -> &'static str {
    #[cfg(unix)]
    {
        "unix"
    }
    #[cfg(windows)]
    {
        "pipe"
    }
}
