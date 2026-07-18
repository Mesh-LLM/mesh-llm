use anyhow::{Context, Result, anyhow};
use rmcp::{
    ErrorData, RoleClient, ServiceExt,
    service::{Peer, RunningService},
    transport::{StreamableHttpClientTransport, TokioChildProcess},
};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use tokio::net::TcpStream;
#[cfg(unix)]
use tokio::net::UnixStream;
use tokio::process::Command;
use tokio::sync::Mutex;

use crate::plugin::PluginEndpointSummary;

use super::internal_error;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum ExternalMcpTransport {
    Stdio { command: String, args: Vec<String> },
    Http { uri: String },
    Tcp { address: String },
    UnixSocket { path: String },
    NamedPipe { name: String },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct ExternalMcpEndpoint {
    pub(super) key: String,
    pub(super) plugin_name: String,
    pub(super) endpoint_id: String,
    pub(super) transport: ExternalMcpTransport,
    pub(super) namespace_prefix: String,
}

impl ExternalMcpEndpoint {
    pub(super) fn from_summary(summary: PluginEndpointSummary) -> Option<Self> {
        if !summary.available || summary.kind != "mcp" {
            return None;
        }
        let local_namespace = summary
            .namespace
            .unwrap_or_else(|| summary.endpoint_id.clone());
        let plugin_name = summary.plugin_name;
        let transport = match summary.transport_kind.as_str() {
            "stdio" => ExternalMcpTransport::Stdio {
                command: summary.address?,
                args: summary.args,
            },
            "http" => ExternalMcpTransport::Http {
                uri: summary.address?,
            },
            "tcp" => ExternalMcpTransport::Tcp {
                address: summary.address?,
            },
            "unix_socket" => ExternalMcpTransport::UnixSocket {
                path: summary.address?,
            },
            "named_pipe" => ExternalMcpTransport::NamedPipe {
                name: summary.address?,
            },
            _ => return None,
        };
        Some(Self {
            key: format!("{}:{}", plugin_name, summary.endpoint_id),
            plugin_name: plugin_name.clone(),
            endpoint_id: summary.endpoint_id,
            transport,
            namespace_prefix: format!("{}.{}", plugin_name, local_namespace),
        })
    }

    pub(super) fn canonical_name(&self, local_name: &str) -> String {
        format!("{}.{}", self.namespace_prefix, local_name)
    }

    pub(super) fn canonical_resource_uri(&self, original_uri: &str) -> String {
        format!(
            "mesh-mcp://{}/{}/resource/{}",
            self.plugin_name,
            self.endpoint_id,
            urlencoding::encode(original_uri)
        )
    }

    pub(super) fn canonical_resource_template_uri(&self, original_uri_template: &str) -> String {
        format!(
            "mesh-mcp://{}/{}/template/{}",
            self.plugin_name,
            self.endpoint_id,
            urlencoding::encode(original_uri_template)
        )
    }

    pub(super) fn transport_label(&self) -> String {
        match &self.transport {
            ExternalMcpTransport::Stdio { command, .. } => command.clone(),
            ExternalMcpTransport::Http { uri } => uri.clone(),
            ExternalMcpTransport::Tcp { address } => address.clone(),
            ExternalMcpTransport::UnixSocket { path } => path.clone(),
            ExternalMcpTransport::NamedPipe { name } => name.clone(),
        }
    }
}

#[derive(Clone)]
pub(super) struct ExternalMcpClient {
    pub(super) peer: Peer<RoleClient>,
    pub(super) running: Arc<Mutex<RunningService<RoleClient, ()>>>,
}

impl ExternalMcpClient {
    async fn connect(endpoint: &ExternalMcpEndpoint) -> Result<Self> {
        let running = match &endpoint.transport {
            ExternalMcpTransport::Stdio { command, args } => {
                let mut child = Command::new(command);
                child.args(args);
                let transport = TokioChildProcess::new(child).with_context(|| {
                    format!(
                        "Spawn external MCP endpoint '{}:{}' with command '{}'",
                        endpoint.plugin_name, endpoint.endpoint_id, command
                    )
                })?;
                ().serve(transport).await.map_err(anyhow::Error::from)
            }
            ExternalMcpTransport::Http { uri } => {
                let transport = StreamableHttpClientTransport::from_uri(uri.clone());
                ().serve(transport).await.map_err(anyhow::Error::from)
            }
            ExternalMcpTransport::Tcp { address } => {
                let stream = TcpStream::connect(address).await.with_context(|| {
                    format!(
                        "Connect TCP external MCP endpoint '{}:{}' at '{}'",
                        endpoint.plugin_name, endpoint.endpoint_id, address
                    )
                })?;
                ().serve(stream).await.map_err(anyhow::Error::from)
            }
            ExternalMcpTransport::UnixSocket { path } => {
                #[cfg(unix)]
                {
                    let stream = UnixStream::connect(path).await.with_context(|| {
                        format!(
                            "Connect Unix socket MCP endpoint '{}:{}' at '{}'",
                            endpoint.plugin_name, endpoint.endpoint_id, path
                        )
                    })?;
                    ().serve(stream).await.map_err(anyhow::Error::from)
                }
                #[cfg(not(unix))]
                {
                    let _ = path;
                    Err(anyhow!(
                        "Unix socket MCP endpoint '{}:{}' is unsupported on this platform",
                        endpoint.plugin_name,
                        endpoint.endpoint_id
                    ))
                }
            }
            ExternalMcpTransport::NamedPipe { name } => {
                #[cfg(windows)]
                {
                    let client = tokio::net::windows::named_pipe::ClientOptions::new()
                        .open(name)
                        .with_context(|| {
                            format!(
                                "Connect named pipe MCP endpoint '{}:{}' at '{}'",
                                endpoint.plugin_name, endpoint.endpoint_id, name
                            )
                        })?;
                    ().serve(client).await.map_err(anyhow::Error::from)
                }
                #[cfg(not(windows))]
                {
                    let _ = name;
                    Err(anyhow!(
                        "Named pipe MCP endpoint '{}:{}' is unsupported on this platform",
                        endpoint.plugin_name,
                        endpoint.endpoint_id
                    ))
                }
            }
        }
        .with_context(|| {
            format!(
                "Connect to external MCP endpoint '{}:{}' via '{}'",
                endpoint.plugin_name,
                endpoint.endpoint_id,
                endpoint.transport_label()
            )
        })?;
        let peer = running.peer().clone();
        Ok(Self {
            peer,
            running: Arc::new(Mutex::new(running)),
        })
    }

    async fn is_closed(&self) -> bool {
        self.running.lock().await.is_closed()
    }
}

#[derive(Clone, Default)]
pub(super) struct ExternalMcpPool {
    clients: Arc<Mutex<BTreeMap<String, Arc<ExternalMcpClient>>>>,
    #[cfg(test)]
    test_clients: Arc<Mutex<BTreeMap<String, Arc<ExternalMcpClient>>>>,
}

impl ExternalMcpPool {
    pub(super) async fn retain_active(&self, active_keys: &BTreeSet<String>) {
        let mut clients = self.clients.lock().await;
        clients.retain(|key, _| active_keys.contains(key));
        #[cfg(test)]
        {
            let mut test_clients = self.test_clients.lock().await;
            test_clients.retain(|key, _| active_keys.contains(key));
        }
    }

    pub(super) async fn client_for(
        &self,
        endpoint: &ExternalMcpEndpoint,
    ) -> Result<Arc<ExternalMcpClient>, ErrorData> {
        #[cfg(test)]
        if let Some(client) = self.test_clients.lock().await.get(&endpoint.key).cloned() {
            return Ok(client);
        }

        if let Some(client) = self.clients.lock().await.get(&endpoint.key).cloned() {
            if !client.is_closed().await {
                return Ok(client);
            }
            self.clients.lock().await.remove(&endpoint.key);
        }

        let client = Arc::new(
            ExternalMcpClient::connect(endpoint)
                .await
                .map_err(internal_error)?,
        );
        self.clients
            .lock()
            .await
            .insert(endpoint.key.clone(), client.clone());
        Ok(client)
    }

    #[cfg(test)]
    pub(super) async fn register_test_client(
        &self,
        endpoint_key: &str,
        client: Arc<ExternalMcpClient>,
    ) {
        self.test_clients
            .lock()
            .await
            .insert(endpoint_key.to_string(), client);
    }
}

#[cfg(test)]
use axum::Router;
#[cfg(test)]
use rmcp::model::{
    AnnotateAble, CallToolRequestParams, CallToolResult, GetPromptRequestParams, GetPromptResult,
    Implementation, ListPromptsResult, ListResourceTemplatesResult, ListResourcesResult,
    ListToolsResult, PaginatedRequestParams, Prompt, PromptMessage, PromptMessageContent,
    PromptMessageRole, RawResource, RawResourceTemplate, ReadResourceRequestParams,
    ReadResourceResult, ResourceContents, ServerCapabilities, ServerInfo, Tool,
};
#[cfg(test)]
use rmcp::service::RequestContext;
#[cfg(test)]
use rmcp::transport::streamable_http_server::{
    StreamableHttpService, session::local::LocalSessionManager,
};
#[cfg(test)]
use rmcp::{RoleServer, ServerHandler};
#[cfg(test)]
use serde_json::json;
#[cfg(test)]
use std::path::PathBuf;

#[cfg(test)]
struct FakeExternalMcpServer;

#[cfg(test)]
impl ServerHandler for FakeExternalMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(
            ServerCapabilities::builder()
                .enable_tools()
                .enable_prompts()
                .enable_resources()
                .build(),
        )
        .with_server_info(Implementation::new("fake-external", "test"))
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        Ok(ListToolsResult::with_all_items(vec![Tool::new(
            "echo",
            "Echo a message",
            Arc::new(
                serde_json::json!({
                    "type": "object",
                    "properties": {
                        "message": { "type": "string" }
                    }
                })
                .as_object()
                .cloned()
                .unwrap(),
            ),
        )]))
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, ErrorData> {
        let message = request
            .arguments
            .as_ref()
            .and_then(|args| args.get("message"))
            .and_then(|value| value.as_str())
            .unwrap_or("missing");
        Ok(CallToolResult::structured(json!({
            "echo": message,
            "tool": request.name.to_string(),
        })))
    }

    async fn list_prompts(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListPromptsResult, ErrorData> {
        Ok(ListPromptsResult::with_all_items(vec![Prompt::new(
            "brief",
            Some("Write a short brief"),
            None::<Vec<_>>,
        )]))
    }

    async fn get_prompt(
        &self,
        request: GetPromptRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<GetPromptResult, ErrorData> {
        Ok(GetPromptResult::new(vec![PromptMessage::new(
            PromptMessageRole::User,
            PromptMessageContent::text(format!("Prompt: {}", request.name)),
        )])
        .with_description("External prompt"))
    }

    async fn list_resources(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListResourcesResult, ErrorData> {
        Ok(ListResourcesResult::with_all_items(vec![
            RawResource::new("note://one", "First Note")
                .with_description("External note")
                .no_annotation(),
        ]))
    }

    async fn list_resource_templates(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListResourceTemplatesResult, ErrorData> {
        Ok(ListResourceTemplatesResult::with_all_items(vec![
            RawResourceTemplate::new("note://{id}", "Note Template").no_annotation(),
        ]))
    }

    async fn read_resource(
        &self,
        request: ReadResourceRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<ReadResourceResult, ErrorData> {
        Ok(ReadResourceResult::new(vec![ResourceContents::text(
            format!("resource:{}", request.uri),
            request.uri,
        )]))
    }
}

#[cfg(test)]
pub(super) async fn fake_external_client() -> Arc<ExternalMcpClient> {
    let (client_stream, server_stream) = tokio::io::duplex(16 * 1024);
    tokio::spawn(async move {
        let _ = FakeExternalMcpServer
            .serve(server_stream)
            .await
            .unwrap()
            .waiting()
            .await;
    });
    let running = ().serve(client_stream).await.unwrap();
    Arc::new(ExternalMcpClient {
        peer: running.peer().clone(),
        running: Arc::new(Mutex::new(running)),
    })
}

#[cfg(test)]
pub(super) async fn spawn_fake_external_tcp_endpoint() -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap().to_string();
    tokio::spawn(async move {
        let Ok((stream, _)) = listener.accept().await else {
            return;
        };
        let _ = FakeExternalMcpServer
            .serve(stream)
            .await
            .unwrap()
            .waiting()
            .await;
    });
    address
}

#[cfg(test)]
pub(super) async fn spawn_fake_external_http_endpoint() -> String {
    let service: StreamableHttpService<FakeExternalMcpServer, LocalSessionManager> =
        StreamableHttpService::new(
            || Ok(FakeExternalMcpServer),
            Default::default(),
            Default::default(),
        );
    let router = Router::new().nest_service("/mcp", service);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, router).await;
    });
    format!("http://{address}/mcp")
}

#[cfg(unix)]
#[cfg(test)]
pub(super) async fn spawn_fake_external_unix_endpoint() -> PathBuf {
    let path = std::env::temp_dir().join(format!("mesh-llm-mcp-{}.sock", rand::random::<u64>()));
    let _ = std::fs::remove_file(&path);
    let listener = tokio::net::UnixListener::bind(&path).unwrap();
    tokio::spawn(async move {
        let Ok((stream, _)) = listener.accept().await else {
            return;
        };
        let _ = FakeExternalMcpServer
            .serve(stream)
            .await
            .unwrap()
            .waiting()
            .await;
    });
    path
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn external_endpoint_namespaces_tools_under_plugin_and_endpoint_namespace() {
        let endpoint = ExternalMcpEndpoint {
            key: "adapter:notes".into(),
            plugin_name: "adapter".into(),
            endpoint_id: "notes".into(),
            transport: ExternalMcpTransport::Stdio {
                command: "fake".into(),
                args: Vec::new(),
            },
            namespace_prefix: "adapter.notes".into(),
        };
        assert_eq!(endpoint.canonical_name("echo"), "adapter.notes.echo");
        assert_eq!(
            endpoint.canonical_resource_uri("note://one"),
            "mesh-mcp://adapter/notes/resource/note%3A%2F%2Fone"
        );
    }

    #[test]
    fn http_external_mcp_endpoint_summary_is_recognized() {
        let endpoint = ExternalMcpEndpoint::from_summary(PluginEndpointSummary {
            plugin_name: "adapter".into(),
            plugin_status: "running".into(),
            endpoint_id: "remote".into(),
            state: "healthy".into(),
            available: true,
            kind: "mcp".into(),
            transport_kind: "http".into(),
            protocol: Some("streamable_http".into()),
            address: Some("http://127.0.0.1:9000/mcp".into()),
            args: Vec::new(),
            namespace: Some("remote".into()),
            supports_streaming: true,
            managed_by_plugin: false,
            detail: None,
            models: Vec::new(),
        })
        .expect("http endpoint");
        assert_eq!(endpoint.canonical_name("echo"), "adapter.remote.echo");
        assert_eq!(
            endpoint.transport,
            ExternalMcpTransport::Http {
                uri: "http://127.0.0.1:9000/mcp".into()
            }
        );
    }
}
