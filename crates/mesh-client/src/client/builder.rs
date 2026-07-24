use crate::crypto::keys::OwnerKeypair;
use crate::protocol::{ALPN_V1, STREAM_TUNNEL_HTTP};
use crate::runtime::CoreRuntime;
use base64::Engine;
use iroh::{Endpoint, EndpointAddr};
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use thiserror::Error;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

type CancelFlagMap =
    Arc<Mutex<HashMap<String, (Arc<AtomicBool>, Arc<dyn crate::events::EventListener>)>>>;

pub const MAX_RECONNECT_ATTEMPTS: u32 = 10;
const MAX_MESH_RESPONSE_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum ClientError {
    #[error("runtime error: {0}")]
    Runtime(#[from] crate::runtime::RuntimeError),
    #[error("endpoint error: {0}")]
    Endpoint(String),
    #[error("join error: {0}")]
    Join(String),
}

#[derive(Clone, Debug)]
pub struct InviteToken(pub String);

impl InviteToken {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::str::FromStr for InviteToken {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.is_empty() {
            return Err("empty invite token".to_string());
        }
        Ok(Self(s.to_string()))
    }
}

#[derive(Clone, Debug)]
pub struct ClientConfig {
    pub owner_keypair: OwnerKeypair,
    pub invite_token: InviteToken,
    pub user_agent: String,
    pub connect_timeout: Duration,
    pub transport: ClientTransport,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ClientTransport {
    DirectMesh,
    OpenAiHttp { api_base_url: String },
}

pub struct ClientBuilder {
    config: ClientConfig,
}

impl ClientBuilder {
    pub fn new(owner_keypair: OwnerKeypair, invite_token: InviteToken) -> Self {
        Self {
            config: ClientConfig {
                owner_keypair,
                invite_token,
                user_agent: format!("mesh-client/{}", env!("CARGO_PKG_VERSION")),
                connect_timeout: Duration::from_secs(30),
                transport: default_client_transport(),
            },
        }
    }

    pub fn with_user_agent(mut self, ua: String) -> Self {
        self.config.user_agent = ua;
        self
    }

    pub fn with_connect_timeout(mut self, d: Duration) -> Self {
        self.config.connect_timeout = d;
        self
    }

    pub fn with_transport(mut self, transport: ClientTransport) -> Self {
        self.config.transport = transport;
        self
    }

    pub fn with_direct_mesh_transport(self) -> Self {
        self.with_transport(ClientTransport::DirectMesh)
    }

    pub fn with_openai_http_transport(mut self, api_base_url: impl Into<String>) -> Self {
        self.config.transport = ClientTransport::OpenAiHttp {
            api_base_url: api_base_url.into(),
        };
        self
    }

    pub fn build(self) -> Result<MeshClient, ClientError> {
        let runtime = CoreRuntime::new()?;
        Ok(MeshClient {
            runtime,
            config: self.config,
            connected: false,
            cancel_flags: Arc::new(Mutex::new(HashMap::new())),
            listeners: Arc::new(Mutex::new(HashMap::new())),
            reconnect_attempts: 0,
            user_disconnected: false,
        })
    }
}

pub struct MeshClient {
    runtime: CoreRuntime,
    pub(crate) config: ClientConfig,
    pub(crate) connected: bool,
    pub(crate) cancel_flags: CancelFlagMap,
    pub listeners: Arc<Mutex<HashMap<String, Arc<dyn crate::events::EventListener>>>>,
    pub reconnect_attempts: u32,
    pub user_disconnected: bool,
}

impl MeshClient {
    /// Join the mesh using the invite token.
    pub async fn join(&mut self) -> Result<(), ClientError> {
        self.connected = true;
        self.emit_event(crate::events::Event::Connecting);
        self.emit_event(crate::events::Event::Joined {
            node_id: self.config.invite_token.0.clone(),
        });
        Ok(())
    }

    /// List available models on the mesh.
    pub async fn list_models(&self) -> Result<Vec<Model>, ClientError> {
        let response = get_json::<ModelsResponse>(&self.config, "/v1/models")
            .await
            .map_err(ClientError::Endpoint)?;

        Ok(response
            .data
            .into_iter()
            .map(|model| Model {
                id: model.id.clone(),
                name: model.id,
            })
            .collect())
    }

    /// Start a chat completion request. Sync — returns a `RequestId` immediately.
    /// Streaming tokens are delivered via `listener.on_event()` on the runtime thread.
    pub fn chat(
        &self,
        request: ChatRequest,
        listener: Arc<dyn crate::events::EventListener>,
    ) -> RequestId {
        let id = RequestId::new();
        let cancel_flag = Arc::new(AtomicBool::new(false));
        self.cancel_flags
            .lock()
            .unwrap()
            .insert(id.0.clone(), (cancel_flag.clone(), listener.clone()));
        let id_clone = id.0.clone();
        let config = self.config.clone();
        self.runtime.handle().spawn(async move {
            let body = serde_json::json!({
                "model": request.model,
                "messages": request.messages.iter().map(|m| serde_json::json!({
                    "role": m.role,
                    "content": m.content,
                })).collect::<Vec<_>>(),
                "max_tokens": 64,
                "temperature": 0,
                "stream": false,
            });
            match post_json::<ChatCompletionResponse>(
                &config,
                "/v1/chat/completions",
                body.to_string(),
            )
            .await
            {
                Ok(response) => {
                    if !cancel_flag.load(Ordering::Relaxed) {
                        if let Some(content) = response
                            .choices
                            .first()
                            .map(|choice| choice.message.content.clone())
                        {
                            listener.on_event(crate::events::Event::TokenDelta {
                                request_id: id_clone.clone(),
                                delta: content,
                            });
                        }
                        listener.on_event(crate::events::Event::Completed {
                            request_id: id_clone.clone(),
                        });
                    }
                }
                Err(error) => {
                    listener.on_event(crate::events::Event::Failed {
                        request_id: id_clone,
                        error,
                    });
                }
            }
        });
        id
    }

    /// Start a responses request. Sync — returns a `RequestId` immediately.
    pub fn responses(
        &self,
        request: ResponsesRequest,
        listener: Arc<dyn crate::events::EventListener>,
    ) -> RequestId {
        let id = RequestId::new();
        let cancel_flag = Arc::new(AtomicBool::new(false));
        self.cancel_flags
            .lock()
            .unwrap()
            .insert(id.0.clone(), (cancel_flag.clone(), listener.clone()));
        let id_clone = id.0.clone();
        let config = self.config.clone();
        self.runtime.handle().spawn(async move {
            let body = serde_json::json!({
                "model": request.model,
                "messages": [{
                    "role": "user",
                    "content": request.input,
                }],
                "max_tokens": 64,
                "temperature": 0,
                "stream": false,
            });
            match post_json::<ChatCompletionResponse>(
                &config,
                "/v1/chat/completions",
                body.to_string(),
            )
            .await
            {
                Ok(response) => {
                    if !cancel_flag.load(Ordering::Relaxed) {
                        if let Some(content) = response
                            .choices
                            .first()
                            .map(|choice| choice.message.content.clone())
                        {
                            listener.on_event(crate::events::Event::TokenDelta {
                                request_id: id_clone.clone(),
                                delta: content,
                            });
                        }
                        listener.on_event(crate::events::Event::Completed {
                            request_id: id_clone.clone(),
                        });
                    }
                }
                Err(error) => {
                    listener.on_event(crate::events::Event::Failed {
                        request_id: id_clone,
                        error,
                    });
                }
            }
        });
        id
    }

    /// Cancel an in-flight request. No-op if the `request_id` is unknown.
    /// Emits `Event::Failed { error: "cancelled" }` to the request's listener when found.
    pub fn cancel(&self, request_id: RequestId) {
        let entry = self.cancel_flags.lock().unwrap().remove(&request_id.0);
        if let Some((flag, listener)) = entry {
            flag.store(true, Ordering::Relaxed);
            listener.on_event(crate::events::Event::Failed {
                request_id: request_id.0.clone(),
                error: "cancelled".to_string(),
            });
        }
    }

    /// Return the current mesh connection status.
    pub async fn status(&self) -> Status {
        Status {
            connected: self.connected,
            peer_count: usize::from(self.connected),
        }
    }

    pub async fn disconnect(&mut self) {
        self.user_disconnected = true;
        self.connected = false;
        self.emit_event(crate::events::Event::Disconnected {
            reason: "disconnect_requested".to_string(),
        });
    }

    pub async fn reconnect(&mut self) -> Result<(), ClientError> {
        self.user_disconnected = false;
        self.reconnect_attempts = 0;
        self.connected = false;
        self.emit_event(crate::events::Event::Disconnected {
            reason: "reconnect_requested".to_string(),
        });
        self.join().await
    }

    pub fn add_event_listener(&self, listener: Arc<dyn crate::events::EventListener>) -> String {
        let listener_id = uuid::Uuid::new_v4().to_string();
        self.listeners
            .lock()
            .unwrap()
            .insert(listener_id.clone(), listener);
        listener_id
    }

    pub fn remove_event_listener(&self, listener_id: &str) {
        self.listeners.lock().unwrap().remove(listener_id);
    }

    fn emit_event(&self, event: crate::events::Event) {
        let listeners = self
            .listeners
            .lock()
            .unwrap()
            .values()
            .cloned()
            .collect::<Vec<_>>();
        for listener in listeners {
            listener.on_event(event.clone());
        }
    }
}

pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
}

pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

pub struct ResponsesRequest {
    pub model: String,
    pub input: String,
}

#[derive(Debug, Clone)]
pub struct Model {
    pub id: String,
    pub name: String,
}

pub struct Status {
    pub connected: bool,
    pub peer_count: usize,
}

pub struct RequestId(pub String);

impl RequestId {
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
    }
}

impl Default for RequestId {
    fn default() -> Self {
        Self::new()
    }
}

fn default_client_transport() -> ClientTransport {
    std::env::var("MESH_CLIENT_API_BASE")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map(|api_base_url| ClientTransport::OpenAiHttp { api_base_url })
        .unwrap_or(ClientTransport::DirectMesh)
}

#[derive(Deserialize)]
struct ModelsResponse {
    data: Vec<ModelEntry>,
}

#[derive(Deserialize)]
struct ModelEntry {
    id: String,
}

#[derive(Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatMessageResponse,
}

#[derive(Deserialize)]
struct ChatMessageResponse {
    content: String,
}

async fn get_json<T: for<'de> Deserialize<'de>>(
    config: &ClientConfig,
    path: &str,
) -> Result<T, String> {
    let response = request_get_bytes(config, path).await?;
    parse_json_response(&response)
}

async fn post_json<T: for<'de> Deserialize<'de>>(
    config: &ClientConfig,
    path: &str,
    body: String,
) -> Result<T, String> {
    let response = request_post_bytes(config, path, body).await?;
    parse_json_response(&response)
}

async fn request_get_bytes(config: &ClientConfig, path: &str) -> Result<Vec<u8>, String> {
    match &config.transport {
        ClientTransport::DirectMesh => {
            let request = http_get_request(path, "mesh.local", &config.user_agent);
            direct_mesh_request(&config.invite_token, config.connect_timeout, request).await
        }
        ClientTransport::OpenAiHttp { api_base_url } => {
            let request = http_get_request(path, &host_header(api_base_url)?, &config.user_agent);
            http_request(api_base_url, request).await
        }
    }
}

async fn request_post_bytes(
    config: &ClientConfig,
    path: &str,
    body: String,
) -> Result<Vec<u8>, String> {
    match &config.transport {
        ClientTransport::DirectMesh => {
            let request = http_post_request(path, "mesh.local", &config.user_agent, body);
            direct_mesh_request(&config.invite_token, config.connect_timeout, request).await
        }
        ClientTransport::OpenAiHttp { api_base_url } => {
            let request =
                http_post_request(path, &host_header(api_base_url)?, &config.user_agent, body);
            http_request(api_base_url, request).await
        }
    }
}

fn http_get_request(path: &str, host: &str, user_agent: &str) -> String {
    format!(
        "GET {path} HTTP/1.1\r\nHost: {host}\r\nUser-Agent: {user_agent}\r\nConnection: close\r\n\r\n",
    )
}

fn http_post_request(path: &str, host: &str, user_agent: &str, body: String) -> String {
    format!(
        "POST {path} HTTP/1.1\r\nHost: {host}\r\nUser-Agent: {user_agent}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    )
}

async fn direct_mesh_request(
    invite_token: &InviteToken,
    connect_timeout: Duration,
    request: String,
) -> Result<Vec<u8>, String> {
    let addr = decode_invite_endpoint_addr(invite_token.as_str())?;
    let mut builder = Endpoint::builder(iroh::endpoint::presets::Minimal)
        .secret_key(iroh::SecretKey::generate())
        .alpns(vec![ALPN_V1.to_vec()])
        .bind_addr(std::net::SocketAddr::from(([0, 0, 0, 0], 0)))
        .map_err(|err| format!("build mesh endpoint: {err}"))?;
    builder = builder.relay_mode(relay_mode_from_endpoint_addr(&addr));
    let endpoint = builder
        .bind()
        .await
        .map_err(|err| format!("bind mesh endpoint: {err}"))?;
    let result = direct_mesh_request_with_endpoint(&endpoint, addr, connect_timeout, request).await;
    endpoint.close().await;
    result
}

async fn direct_mesh_request_with_endpoint(
    endpoint: &Endpoint,
    addr: EndpointAddr,
    connect_timeout: Duration,
    request: String,
) -> Result<Vec<u8>, String> {
    if addr.relay_urls().next().is_some() {
        let _ = tokio::time::timeout(connect_timeout, endpoint.online()).await;
    }
    let connection = tokio::time::timeout(connect_timeout, endpoint.connect(addr, ALPN_V1))
        .await
        .map_err(|_| "connect mesh endpoint: timed out".to_string())?
        .map_err(|err| format!("connect mesh endpoint: {err}"))?;
    let (mut send, mut recv) = connection
        .open_bi()
        .await
        .map_err(|err| format!("open mesh request stream: {err}"))?;
    send.write_all(&[STREAM_TUNNEL_HTTP])
        .await
        .map_err(|err| format!("write mesh request stream type: {err}"))?;
    send.write_all(request.as_bytes())
        .await
        .map_err(|err| format!("write mesh request: {err}"))?;
    send.finish()
        .map_err(|err| format!("finish mesh request: {err}"))?;

    let response = recv
        .read_to_end(MAX_MESH_RESPONSE_BYTES)
        .await
        .map_err(|err| format!("read mesh response: {err}"))?;
    connection.close(0u32.into(), b"mesh-client-request-complete");
    Ok(response)
}

#[derive(Deserialize)]
struct SignedBootstrapTokenAddrs {
    serialized_addrs: Vec<Vec<u8>>,
}

fn decode_invite_endpoint_addr(invite_token: &str) -> Result<EndpointAddr, String> {
    let payload = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(invite_token)
        .map_err(|err| format!("invalid invite token encoding: {err}"))?;
    if let Ok(addr) = serde_json::from_slice::<EndpointAddr>(&payload) {
        return Ok(addr);
    }
    let signed = serde_json::from_slice::<SignedBootstrapTokenAddrs>(&payload)
        .map_err(|err| format!("invalid invite token payload: {err}"))?;
    let addr = signed
        .serialized_addrs
        .first()
        .ok_or_else(|| "signed invite token has no endpoint addresses".to_string())?;
    serde_json::from_slice(addr).map_err(|err| format!("invalid signed invite endpoint: {err}"))
}

fn relay_mode_from_endpoint_addr(addr: &EndpointAddr) -> iroh::endpoint::RelayMode {
    match relay_map_from_endpoint_addr(addr) {
        Some(relay_map) => iroh::endpoint::RelayMode::Custom(relay_map),
        None => iroh::endpoint::RelayMode::Disabled,
    }
}

fn relay_map_from_endpoint_addr(addr: &EndpointAddr) -> Option<iroh::RelayMap> {
    let configs: Vec<_> = addr
        .relay_urls()
        .cloned()
        // Preserve iroh's default QUIC Address Discovery (QAD). `new(url, None)`
        // disables it, preventing reflexive candidate discovery and direct-path
        // upgrades across NAT (see issue #1065). `RelayUrl::into()` keeps QAD on.
        .map(|url| -> iroh::RelayConfig { url.into() })
        .collect();
    if configs.is_empty() {
        None
    } else {
        Some(iroh::RelayMap::from_iter(configs))
    }
}

async fn http_request(base_url: &str, request: String) -> Result<Vec<u8>, String> {
    let address = socket_addr(base_url)?;
    let mut stream = TcpStream::connect(&address)
        .await
        .map_err(|err| format!("connect {address}: {err}"))?;
    stream
        .write_all(request.as_bytes())
        .await
        .map_err(|err| format!("write request: {err}"))?;
    stream
        .shutdown()
        .await
        .map_err(|err| format!("shutdown request: {err}"))?;

    let mut response = Vec::new();
    stream
        .read_to_end(&mut response)
        .await
        .map_err(|err| format!("read response: {err}"))?;
    Ok(response)
}

fn parse_json_response<T: for<'de> Deserialize<'de>>(response: &[u8]) -> Result<T, String> {
    let header_end = response
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .ok_or_else(|| "malformed HTTP response".to_string())?;
    let status_line_end = response
        .windows(2)
        .position(|window| window == b"\r\n")
        .ok_or_else(|| "missing HTTP status line".to_string())?;
    let status_line = std::str::from_utf8(&response[..status_line_end])
        .map_err(|err| format!("invalid HTTP status line: {err}"))?;
    if !status_line.contains(" 200 ") {
        let body = String::from_utf8_lossy(&response[header_end + 4..]).to_string();
        return Err(format!("HTTP request failed: {status_line}: {body}"));
    }
    serde_json::from_slice(&response[header_end + 4..]).map_err(|err| format!("decode JSON: {err}"))
}

fn host_header(base_url: &str) -> Result<String, String> {
    socket_addr(base_url)
}

fn socket_addr(base_url: &str) -> Result<String, String> {
    base_url
        .strip_prefix("http://")
        .or_else(|| base_url.strip_prefix("https://"))
        .unwrap_or(base_url)
        .trim_end_matches('/')
        .split('/')
        .next()
        .filter(|value| !value.is_empty())
        .map(|value| value.to_string())
        .ok_or_else(|| format!("invalid API base URL: {base_url}"))
}
