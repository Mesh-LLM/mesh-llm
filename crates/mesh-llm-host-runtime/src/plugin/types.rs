use super::proto;
use super::startup::PluginStartupSummary;
use super::{PluginWebUiManifestOverview, PluginWebUiState};
use serde::Serialize;
use std::future::Future;
use std::pin::Pin;
use tokio::sync::oneshot;

#[derive(Debug)]
pub enum PluginMeshEvent {
    Channel {
        plugin_id: String,
        message: proto::ChannelMessage,
    },
    BulkTransfer {
        plugin_id: String,
        message: proto::BulkTransferMessage,
    },
    OpenStream {
        plugin_id: String,
        request: proto::OpenMeshStreamRequest,
        response_tx: oneshot::Sender<Result<proto::OpenMeshStreamResponse, proto::ErrorResponse>>,
    },
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct ToolSummary {
    pub name: String,
    pub description: String,
    pub input_schema_json: String,
}

#[derive(Clone, Debug)]
pub struct ToolCallResult {
    pub content_json: String,
    pub is_error: bool,
}

#[derive(Clone, Debug)]
pub struct RpcResult {
    pub result_json: String,
}

pub(crate) type BridgeFuture<T> = Pin<Box<dyn Future<Output = T> + Send>>;

pub trait PluginRpcBridge: Send + Sync {
    fn handle_request(
        &self,
        plugin_name: String,
        method: String,
        params_json: String,
    ) -> BridgeFuture<Result<RpcResult, proto::ErrorResponse>>;

    fn handle_notification(
        &self,
        plugin_name: String,
        method: String,
        params_json: String,
    ) -> BridgeFuture<()>;
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct PluginSummary {
    pub name: String,
    pub kind: String,
    pub enabled: bool,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pid: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub capabilities: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub args: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub tools: Vec<ToolSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub manifest: Option<PluginManifestOverview>,
    #[serde(default)]
    pub web_ui: PluginWebUiState,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub startup: Option<PluginStartupSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct PluginManifestOverview {
    pub operations: usize,
    pub resources: usize,
    pub resource_templates: usize,
    pub prompts: usize,
    pub completions: usize,
    pub http_bindings: usize,
    pub endpoints: usize,
    pub mesh_channels: usize,
    pub mesh_event_subscriptions: usize,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub capabilities: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub web_ui: Option<PluginWebUiManifestOverview>,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct PluginEndpointSummary {
    pub plugin_name: String,
    pub plugin_status: String,
    pub endpoint_id: String,
    pub state: String,
    pub available: bool,
    pub kind: String,
    pub transport_kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub protocol: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub address: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub args: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
    pub supports_streaming: bool,
    pub managed_by_plugin: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub models: Vec<String>,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct PluginCapabilityProvider {
    pub capability: String,
    pub plugin_name: String,
    pub plugin_status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub endpoint_id: Option<String>,
    pub available: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Clone, Debug)]
pub struct InferenceEndpointRoute {
    pub plugin_name: String,
    pub endpoint_id: String,
    pub address: String,
    pub models: Vec<String>,
}
