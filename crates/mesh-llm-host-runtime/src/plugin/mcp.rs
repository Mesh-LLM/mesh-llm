use anyhow::{Context, Result, anyhow};
use rmcp::{
    ErrorData, RoleServer,
    model::{
        ClientResult, CreateElicitationRequest, CreateElicitationRequestParams,
        CreateMessageRequest, CreateMessageRequestParams, CustomNotification, CustomRequest,
        ErrorCode, ListRootsRequest, LoggingMessageNotification, LoggingMessageNotificationParam,
        PingRequest, ResourceUpdatedNotificationParam, ServerNotification, ServerRequest,
    },
    service::Peer,
    transport::streamable_http_server::{
        StreamableHttpService, session::local::LocalSessionManager,
    },
};
use serde::Serialize;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::plugin::{self, PluginManager, PluginRpcBridge, RpcResult};

use axum::Router;

mod external_mcp;
mod tool_dispatch;

use external_mcp::ExternalMcpPool;

#[derive(Clone, Default)]
struct ActiveBridge {
    peer: Arc<Mutex<Option<Peer<RoleServer>>>>,
}

impl ActiveBridge {
    async fn set_peer(&self, peer: Peer<RoleServer>) {
        *self.peer.lock().await = Some(peer);
    }

    async fn current_peer(&self) -> Result<Peer<RoleServer>, plugin::proto::ErrorResponse> {
        self.peer
            .lock()
            .await
            .clone()
            .ok_or_else(|| proto_error::internal("No active MCP client session"))
    }
}

impl PluginRpcBridge for ActiveBridge {
    #[allow(deprecated)]
    fn handle_request(
        &self,
        _plugin_name: String,
        method: String,
        params_json: String,
    ) -> crate::plugin::BridgeFuture<Result<RpcResult, plugin::proto::ErrorResponse>> {
        let this = self.clone();
        Box::pin(async move {
            let peer: Peer<RoleServer> = this.current_peer().await?;
            let params = parse_optional_value(&params_json)?;
            let result_json = match method.as_str() {
                "ping" => {
                    let result: ClientResult = peer
                        .send_request(ServerRequest::PingRequest(PingRequest::default()))
                        .await
                        .map_err(proto_error::from_service)?;
                    match result {
                        ClientResult::EmptyResult(result) => to_json_string(&result),
                        _ => Err(proto_error::internal("unexpected ping response")),
                    }
                }
                "roots/list" => {
                    let result: ClientResult = peer
                        .send_request(ServerRequest::ListRootsRequest(ListRootsRequest::default()))
                        .await
                        .map_err(proto_error::from_service)?;
                    match result {
                        ClientResult::ListRootsResult(result) => to_json_string(&result),
                        _ => Err(proto_error::internal("unexpected roots/list response")),
                    }
                }
                "sampling/createMessage" => {
                    let params =
                        deserialize_required::<CreateMessageRequestParams>(params, &method)?;
                    if (params.tools.is_some() || params.tool_choice.is_some())
                        && !peer.supports_sampling_tools()
                    {
                        return Err(proto_error::invalid_params(
                            "tools or toolChoice provided but client does not support sampling tools capability",
                        ));
                    }
                    params.validate().map_err(proto_error::invalid_params)?;
                    let result: ClientResult = peer
                        .send_request(ServerRequest::CreateMessageRequest(
                            CreateMessageRequest::new(params),
                        ))
                        .await
                        .map_err(proto_error::from_service)?;
                    match result {
                        ClientResult::CreateMessageResult(result) => to_json_string(&result),
                        _ => Err(proto_error::internal("unexpected sampling response")),
                    }
                }
                "elicitation/create" => {
                    let params =
                        deserialize_required::<CreateElicitationRequestParams>(params, &method)?;
                    let result: ClientResult = peer
                        .send_request(ServerRequest::CreateElicitationRequest(
                            CreateElicitationRequest::new(params),
                        ))
                        .await
                        .map_err(proto_error::from_service)?;
                    match result {
                        ClientResult::CreateElicitationResult(result) => to_json_string(&result),
                        _ => Err(proto_error::internal("unexpected elicitation response")),
                    }
                }
                _ => {
                    let result: ClientResult = peer
                        .send_request(ServerRequest::CustomRequest(CustomRequest::new(
                            method.clone(),
                            params,
                        )))
                        .await
                        .map_err(proto_error::from_service)?;
                    match result {
                        ClientResult::CustomResult(result) => to_json_string(&result),
                        _ => Err(proto_error::internal("unexpected custom response")),
                    }
                }
            }
            .map_err(|mut err| {
                err.message = format!("bridge request '{method}': {}", err.message);
                err
            })?;

            Ok(RpcResult { result_json })
        })
    }

    #[allow(deprecated)]
    fn handle_notification(
        &self,
        _plugin_name: String,
        method: String,
        params_json: String,
    ) -> crate::plugin::BridgeFuture<()> {
        let this = self.clone();
        Box::pin(async move {
            let Ok(peer): Result<Peer<RoleServer>, _> = this.current_peer().await else {
                return;
            };
            let params = match parse_optional_value(&params_json) {
                Ok(params) => params,
                Err(_) => return,
            };

            match method.as_str() {
                "notifications/resources/updated" => {
                    if let Ok(params) =
                        deserialize_required::<ResourceUpdatedNotificationParam>(params, &method)
                    {
                        let _ = peer.notify_resource_updated(params).await;
                    }
                }
                "notifications/resources/list_changed" => {
                    let _ = peer.notify_resource_list_changed().await;
                }
                "notifications/tools/list_changed" => {
                    let _ = peer.notify_tool_list_changed().await;
                }
                "notifications/prompts/list_changed" => {
                    let _ = peer.notify_prompt_list_changed().await;
                }
                "notifications/message" => {
                    if let Ok(params) =
                        deserialize_required::<LoggingMessageNotificationParam>(params, &method)
                    {
                        let _ = peer
                            .send_notification(ServerNotification::LoggingMessageNotification(
                                LoggingMessageNotification::new(params),
                            ))
                            .await;
                    }
                }
                _ => {
                    let _ = peer
                        .send_notification(ServerNotification::CustomNotification(
                            CustomNotification::new(method, params),
                        ))
                        .await;
                }
            }
        })
    }
}

#[derive(Clone)]
pub struct PluginMcpServer {
    plugin_manager: PluginManager,
    bridge: ActiveBridge,
    external_mcp: ExternalMcpPool,
}

impl PluginMcpServer {
    fn new(plugin_manager: PluginManager, bridge: ActiveBridge) -> Self {
        Self {
            plugin_manager,
            bridge,
            external_mcp: ExternalMcpPool::default(),
        }
    }

    async fn plugin_manifests(
        &self,
    ) -> Result<Vec<(String, plugin::proto::PluginManifest)>, ErrorData> {
        let mut manifests = Vec::new();
        for (plugin_name, _) in self.plugin_manager.list_server_infos().await {
            let manifest = self
                .plugin_manager
                .manifest(&plugin_name)
                .await
                .map_err(internal_error)?;
            if let Some(manifest) = manifest {
                manifests.push((plugin_name, manifest));
            }
        }
        Ok(manifests)
    }

    async fn refresh_peer(&self, peer: Peer<RoleServer>) {
        self.bridge.set_peer(peer).await;
    }

    async fn broadcast_notification<P>(&self, method: &str, params: P)
    where
        P: Serialize + Clone,
    {
        for (plugin_name, _) in self.plugin_manager.list_server_infos().await {
            let _ = self
                .plugin_manager
                .mcp_notify(&plugin_name, method, params.clone())
                .await;
        }
    }
}

#[derive(Clone)]
pub(crate) struct PluginMcpHttpEndpoint {
    plugin_manager: PluginManager,
    bridge: ActiveBridge,
    session_manager: Arc<LocalSessionManager>,
}

impl PluginMcpHttpEndpoint {
    pub(crate) fn new(plugin_manager: PluginManager) -> Self {
        Self {
            plugin_manager,
            bridge: ActiveBridge::default(),
            session_manager: Arc::new(LocalSessionManager::default()),
        }
    }

    pub(crate) async fn handle(
        &self,
        request: http::Request<http_body_util::Full<bytes::Bytes>>,
    ) -> http::Response<http_body_util::combinators::BoxBody<bytes::Bytes, std::convert::Infallible>>
    {
        self.plugin_manager
            .set_rpc_bridge(Some(Arc::new(self.bridge.clone())))
            .await;

        let plugin_manager = self.plugin_manager.clone();
        let bridge = self.bridge.clone();
        let service = StreamableHttpService::new(
            move || Ok(PluginMcpServer::new(plugin_manager.clone(), bridge.clone())),
            self.session_manager.clone(),
            Default::default(),
        );
        service.handle(request).await
    }
}

fn internal_error(err: impl std::fmt::Display) -> ErrorData {
    ErrorData::internal_error(err.to_string(), None)
}

fn to_json_string<T: Serialize>(value: &T) -> Result<String, plugin::proto::ErrorResponse> {
    serde_json::to_string(value).map_err(|err| proto_error::from_anyhow(err.into()))
}

fn parse_optional_value(
    raw: &str,
) -> Result<Option<serde_json::Value>, plugin::proto::ErrorResponse> {
    plugin::parse_optional_json(raw).map_err(proto_error::from_anyhow)
}

fn deserialize_required<T: serde::de::DeserializeOwned>(
    value: Option<serde_json::Value>,
    method: &str,
) -> Result<T, plugin::proto::ErrorResponse> {
    let value = value.unwrap_or(serde_json::Value::Null);
    serde_json::from_value(value).map_err(|err| plugin::proto::ErrorResponse {
        code: ErrorCode::INVALID_PARAMS.0,
        message: format!("Invalid params for '{method}': {err}"),
        data_json: String::new(),
    })
}

async fn try_plugins<T, P>(
    plugin_manager: &PluginManager,
    method: &str,
    params: P,
) -> Result<T, ErrorData>
where
    T: serde::de::DeserializeOwned,
    P: Serialize + Clone,
{
    let mut last_error = None;
    for (plugin_name, _) in plugin_manager.list_server_infos().await {
        match plugin_manager
            .mcp_request::<T, _>(&plugin_name, method, params.clone())
            .await
        {
            Ok(value) => return Ok(value),
            Err(err) => last_error = Some(err),
        }
    }
    Err(internal_error(
        last_error.unwrap_or_else(|| anyhow!("No plugin handled '{method}'")),
    ))
}

mod proto_error {
    use anyhow::Error;
    use rmcp::{ServiceError, model::ErrorCode};

    pub fn from_anyhow(err: Error) -> crate::plugin::proto::ErrorResponse {
        crate::plugin::proto::ErrorResponse {
            code: ErrorCode::INTERNAL_ERROR.0,
            message: err.to_string(),
            data_json: String::new(),
        }
    }

    pub fn from_service(err: ServiceError) -> crate::plugin::proto::ErrorResponse {
        crate::plugin::proto::ErrorResponse {
            code: ErrorCode::INTERNAL_ERROR.0,
            message: err.to_string(),
            data_json: String::new(),
        }
    }

    pub fn internal(message: impl Into<String>) -> crate::plugin::proto::ErrorResponse {
        crate::plugin::proto::ErrorResponse {
            code: ErrorCode::INTERNAL_ERROR.0,
            message: message.into(),
            data_json: String::new(),
        }
    }

    pub fn invalid_params(message: impl Into<String>) -> crate::plugin::proto::ErrorResponse {
        crate::plugin::proto::ErrorResponse {
            code: ErrorCode::INVALID_PARAMS.0,
            message: message.into(),
            data_json: String::new(),
        }
    }
}

#[allow(dead_code)]
pub(crate) async fn run_mcp_server(plugin_manager: PluginManager) -> Result<()> {
    use rmcp::transport::streamable_http_server::{
        StreamableHttpService, session::local::LocalSessionManager,
    };

    let service = StreamableHttpService::new(
        move || {
            Ok(PluginMcpServer::new(
                plugin_manager.clone(),
                Default::default(),
            ))
        },
        Arc::new(LocalSessionManager::default()),
        Default::default(),
    );
    let router = Router::new().nest_service("/mcp", service);

    let bind_addr = std::env::var("MESH_MCP_PORT")
        .ok()
        .and_then(|p| p.parse::<u16>().ok())
        .unwrap_or(3040);

    let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{bind_addr}"))
        .await
        .context("failed to bind MCP server address")?;
    let addr = listener.local_addr()?;
    tracing::info!(%addr, "MCP plugin server listening");

    axum::serve(listener, router)
        .await
        .context("MCP server exited")
}
