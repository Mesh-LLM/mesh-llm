use anyhow::{anyhow, bail, Context, Result};
use prost::Message;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::{Child, Command};
use tokio::sync::{mpsc, oneshot, Mutex};

pub const BLACKBOARD_PLUGIN_ID: &str = "blackboard";
const PROTOCOL_VERSION: u32 = 1;
const CONNECT_TIMEOUT_SECS: u64 = 10;
const REQUEST_TIMEOUT_SECS: u64 = 30;

#[allow(dead_code)]
pub mod proto {
    include!(concat!(env!("OUT_DIR"), "/meshllm.plugin.v1.rs"));
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct MeshConfig {
    #[serde(rename = "plugin", default)]
    pub plugins: Vec<PluginConfigEntry>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct PluginConfigEntry {
    pub name: String,
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub command: Option<String>,
    #[serde(default)]
    pub args: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct ResolvedPlugins {
    pub externals: Vec<ExternalPluginSpec>,
}

#[derive(Clone, Debug)]
pub struct ExternalPluginSpec {
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct PluginChannelEvent {
    pub plugin_id: String,
    pub message: proto::ChannelMessage,
}

#[derive(Clone, Debug, Serialize)]
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

#[derive(Clone, Debug, Serialize)]
pub struct PluginSummary {
    pub name: String,
    pub kind: String,
    pub enabled: bool,
    pub status: String,
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
    pub error: Option<String>,
}

#[derive(Clone)]
pub struct PluginManager {
    inner: Arc<PluginManagerInner>,
}

struct PluginManagerInner {
    plugins: BTreeMap<String, ExternalPlugin>,
}

struct ExternalPlugin {
    summary: PluginSummary,
    _child: Mutex<Child>,
    outbound_tx: mpsc::Sender<proto::Envelope>,
    pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Result<proto::Envelope>>>>>,
    next_request_id: AtomicU64,
}

enum LocalStream {
    #[cfg(unix)]
    Unix(tokio::net::UnixStream),
    #[cfg(windows)]
    PipeServer(tokio::net::windows::named_pipe::NamedPipeServer),
    #[cfg(windows)]
    PipeClient(tokio::net::windows::named_pipe::NamedPipeClient),
}

enum LocalListener {
    #[cfg(unix)]
    Unix(tokio::net::UnixListener, PathBuf),
    #[cfg(windows)]
    Pipe(String, tokio::net::windows::named_pipe::NamedPipeServer),
}

pub fn config_path(override_path: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = override_path {
        return Ok(path.to_path_buf());
    }
    if let Ok(path) = std::env::var("MESH_LLM_CONFIG") {
        return Ok(PathBuf::from(path));
    }
    let home = dirs::home_dir().context("Cannot determine home directory")?;
    Ok(home.join(".mesh-llm").join("config.toml"))
}

pub fn load_config(override_path: Option<&Path>) -> Result<MeshConfig> {
    let path = config_path(override_path)?;
    if !path.exists() {
        return Ok(MeshConfig::default());
    }
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("Failed to read config {}", path.display()))?;
    toml::from_str(&raw).with_context(|| format!("Failed to parse config {}", path.display()))
}

pub fn resolve_plugins(config: &MeshConfig) -> Result<ResolvedPlugins> {
    let mut externals = Vec::new();
    let mut names = BTreeMap::<String, ()>::new();
    let mut blackboard_enabled = true;

    for entry in &config.plugins {
        if names.insert(entry.name.clone(), ()).is_some() {
            bail!("Duplicate plugin entry '{}'", entry.name);
        }
        let enabled = entry.enabled.unwrap_or(true);
        if entry.name == BLACKBOARD_PLUGIN_ID {
            if entry.command.is_some() || !entry.args.is_empty() {
                bail!(
                    "Plugin '{}' is served by mesh-llm itself; only `enabled` may be set",
                    BLACKBOARD_PLUGIN_ID
                );
            }
            blackboard_enabled = enabled;
            continue;
        }
        if !enabled {
            continue;
        }
        let command = entry
            .command
            .clone()
            .with_context(|| format!("Plugin '{}' is enabled but missing command", entry.name))?;
        externals.push(ExternalPluginSpec {
            name: entry.name.clone(),
            command,
            args: entry.args.clone(),
        });
    }

    if blackboard_enabled {
        externals.insert(0, blackboard_plugin_spec()?);
    }

    Ok(ResolvedPlugins { externals })
}

pub fn blackboard_plugin_spec() -> Result<ExternalPluginSpec> {
    let command = std::env::current_exe()
        .context("Cannot determine mesh-llm executable path")?
        .display()
        .to_string();
    Ok(ExternalPluginSpec {
        name: BLACKBOARD_PLUGIN_ID.to_string(),
        command,
        args: vec!["--plugin".into(), BLACKBOARD_PLUGIN_ID.into()],
    })
}

impl PluginManager {
    pub async fn start(
        specs: &ResolvedPlugins,
        mesh_tx: mpsc::Sender<PluginChannelEvent>,
    ) -> Result<Self> {
        let mut plugins = BTreeMap::new();
        for spec in &specs.externals {
            let plugin = ExternalPlugin::spawn(spec, mesh_tx.clone()).await?;
            plugins.insert(spec.name.clone(), plugin);
        }
        Ok(Self {
            inner: Arc::new(PluginManagerInner { plugins }),
        })
    }

    pub async fn list(&self) -> Vec<PluginSummary> {
        self.inner
            .plugins
            .values()
            .map(|plugin| plugin.summary.clone())
            .collect()
    }

    pub fn is_enabled(&self, name: &str) -> bool {
        self.inner
            .plugins
            .get(name)
            .map(|plugin| plugin.summary.enabled && plugin.summary.status == "running")
            .unwrap_or(false)
    }

    pub async fn tools(&self, name: &str) -> Result<Vec<ToolSummary>> {
        let plugin = self
            .inner
            .plugins
            .get(name)
            .with_context(|| format!("Unknown plugin '{name}'"))?;
        Ok(plugin.summary.tools.clone())
    }

    pub async fn call_tool(
        &self,
        plugin_name: &str,
        tool_name: &str,
        arguments_json: &str,
    ) -> Result<ToolCallResult> {
        let plugin = self
            .inner
            .plugins
            .get(plugin_name)
            .with_context(|| format!("Unknown plugin '{plugin_name}'"))?;
        plugin.call_tool(tool_name, arguments_json).await
    }

    pub async fn dispatch_channel_message(&self, event: PluginChannelEvent) -> Result<()> {
        let Some(plugin) = self.inner.plugins.get(&event.plugin_id) else {
            tracing::debug!(
                "Dropping channel message for unloaded plugin '{}'",
                event.plugin_id
            );
            return Ok(());
        };
        plugin.send_channel_message(event.message).await
    }
}

impl ExternalPlugin {
    async fn spawn(spec: &ExternalPluginSpec, mesh_tx: mpsc::Sender<PluginChannelEvent>) -> Result<Self> {
        let listener = bind_local_listener(&spec.name).await?;
        let endpoint = listener.endpoint();
        let transport = listener.transport_name();

        let mut child = Command::new(&spec.command);
        child.args(&spec.args);
        child.env("MESH_LLM_PLUGIN_ENDPOINT", &endpoint);
        child.env("MESH_LLM_PLUGIN_TRANSPORT", transport);
        child.env("MESH_LLM_PLUGIN_NAME", &spec.name);
        child.stdin(std::process::Stdio::null());
        child.stdout(std::process::Stdio::null());
        child.stderr(std::process::Stdio::inherit());
        child.kill_on_drop(true);

        let child = child.spawn().with_context(|| {
            format!(
                "Failed to launch plugin '{}' via {}",
                spec.name, spec.command
            )
        })?;

        let stream = tokio::time::timeout(
            std::time::Duration::from_secs(CONNECT_TIMEOUT_SECS),
            listener.accept(),
        )
        .await
        .with_context(|| format!("Timed out waiting for plugin '{}'", spec.name))??;

        let (outbound_tx, outbound_rx) = mpsc::channel(256);
        let pending = Arc::new(Mutex::new(HashMap::new()));
        tokio::spawn(connection_loop(
            stream,
            outbound_rx,
            pending.clone(),
            mesh_tx,
            spec.name.clone(),
        ));

        let mut plugin = Self {
            summary: PluginSummary {
                name: spec.name.clone(),
                kind: "external".into(),
                enabled: true,
                status: "starting".into(),
                version: None,
                capabilities: Vec::new(),
                command: Some(spec.command.clone()),
                args: spec.args.clone(),
                tools: Vec::new(),
                error: None,
            },
            _child: Mutex::new(child),
            outbound_tx,
            pending,
            next_request_id: AtomicU64::new(1),
        };

        let response = plugin
            .request(proto::envelope::Payload::InitializeRequest(
                proto::InitializeRequest {
                    host_protocol_version: PROTOCOL_VERSION,
                    host_version: crate::VERSION.to_string(),
                    requested_capabilities: Vec::new(),
                },
            ))
            .await?;

        let init = match response.payload {
            Some(proto::envelope::Payload::InitializeResponse(resp)) => resp,
            Some(proto::envelope::Payload::ErrorResponse(err)) => {
                bail!("Plugin '{}' rejected initialize: {}", spec.name, err.message)
            }
            _ => bail!("Plugin '{}' returned an unexpected initialize payload", spec.name),
        };

        if init.plugin_id != spec.name {
            bail!(
                "Plugin '{}' identified itself as '{}'",
                spec.name,
                init.plugin_id
            );
        }
        if init.plugin_protocol_version != PROTOCOL_VERSION {
            bail!(
                "Plugin '{}' uses protocol {}, host uses {}",
                spec.name,
                init.plugin_protocol_version,
                PROTOCOL_VERSION
            );
        }

        plugin.summary.status = "running".into();
        plugin.summary.version = Some(init.plugin_version);
        plugin.summary.capabilities = init.capabilities;
        plugin.summary.tools = init
            .tool_schemas
            .into_iter()
            .map(|tool| ToolSummary {
                name: tool.name,
                description: tool.description,
                input_schema_json: tool.input_schema_json,
            })
            .collect();

        Ok(plugin)
    }

    async fn call_tool(&self, tool_name: &str, arguments_json: &str) -> Result<ToolCallResult> {
        let response = self
            .request(proto::envelope::Payload::ToolCallRequest(
                proto::ToolCallRequest {
                    name: tool_name.to_string(),
                    arguments_json: arguments_json.to_string(),
                },
            ))
            .await?;
        match response.payload {
            Some(proto::envelope::Payload::ToolCallResponse(resp)) => Ok(ToolCallResult {
                content_json: resp.content_json,
                is_error: resp.is_error,
            }),
            Some(proto::envelope::Payload::ErrorResponse(err)) => {
                bail!("Plugin tool call failed: {}", err.message)
            }
            _ => bail!(
                "Plugin '{}' returned an unexpected tool payload",
                self.summary.name
            ),
        }
    }

    async fn send_channel_message(&self, message: proto::ChannelMessage) -> Result<()> {
        self.outbound_tx
            .send(proto::Envelope {
                protocol_version: PROTOCOL_VERSION,
                plugin_id: self.summary.name.clone(),
                request_id: 0,
                payload: Some(proto::envelope::Payload::ChannelMessage(message)),
            })
            .await
            .map_err(|_| anyhow!("Plugin '{}' is not accepting messages", self.summary.name))
    }

    async fn request(&self, payload: proto::envelope::Payload) -> Result<proto::Envelope> {
        let request_id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        self.pending.lock().await.insert(request_id, tx);

        let envelope = proto::Envelope {
            protocol_version: PROTOCOL_VERSION,
            plugin_id: self.summary.name.clone(),
            request_id,
            payload: Some(payload),
        };

        if let Err(_send_err) = self.outbound_tx.send(envelope).await {
            self.pending.lock().await.remove(&request_id);
            bail!("Plugin '{}' is not accepting requests", self.summary.name);
        }

        match tokio::time::timeout(std::time::Duration::from_secs(REQUEST_TIMEOUT_SECS), rx).await
        {
            Ok(Ok(resp)) => resp,
            Ok(Err(_)) => bail!("Plugin '{}' dropped the response channel", self.summary.name),
            Err(_) => {
                self.pending.lock().await.remove(&request_id);
                bail!("Plugin '{}' timed out", self.summary.name)
            }
        }
    }
}

async fn connection_loop(
    mut stream: LocalStream,
    mut outbound_rx: mpsc::Receiver<proto::Envelope>,
    pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Result<proto::Envelope>>>>>,
    mesh_tx: mpsc::Sender<PluginChannelEvent>,
    plugin_name: String,
) {
    let result: Result<()> = async {
        loop {
            tokio::select! {
                maybe_outbound = outbound_rx.recv() => {
                    let Some(envelope) = maybe_outbound else {
                        break;
                    };
                    write_envelope(&mut stream, &envelope).await?;
                }
                inbound = read_envelope(&mut stream) => {
                    let envelope = inbound?;
                    let request_id = envelope.request_id;
                    let plugin_id_from_env = envelope.plugin_id.clone();
                    let payload = envelope.payload.clone();
                    match payload {
                        Some(proto::envelope::Payload::ChannelMessage(message)) => {
                            let plugin_id = if plugin_id_from_env.is_empty() {
                                plugin_name.clone()
                            } else {
                                plugin_id_from_env
                            };
                            let _ = mesh_tx.send(PluginChannelEvent { plugin_id, message }).await;
                        }
                        _ => {
                            let responder = pending.lock().await.remove(&request_id);
                            if let Some(responder) = responder {
                                let _ = responder.send(Ok(envelope));
                            } else {
                                tracing::debug!(
                                    "Plugin '{}' sent an unsolicited response id={}",
                                    plugin_name,
                                    request_id
                                );
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
    .await;

    if let Err(err) = result {
        tracing::debug!("Plugin '{}' connection closed: {err}", plugin_name);
    }

    let mut pending = pending.lock().await;
    for (_, responder) in pending.drain() {
        let _ = responder.send(Err(anyhow!("Plugin '{}' disconnected", plugin_name)));
    }
}

impl LocalListener {
    async fn accept(self) -> Result<LocalStream> {
        match self {
            #[cfg(unix)]
            LocalListener::Unix(listener, path) => {
                let (stream, _) = listener.accept().await?;
                let _ = std::fs::remove_file(path);
                Ok(LocalStream::Unix(stream))
            }
            #[cfg(windows)]
            LocalListener::Pipe(_name, server) => {
                server.connect().await?;
                Ok(LocalStream::PipeServer(server))
            }
        }
    }

    fn endpoint(&self) -> String {
        match self {
            #[cfg(unix)]
            LocalListener::Unix(_, path) => path.display().to_string(),
            #[cfg(windows)]
            LocalListener::Pipe(name, _) => name.clone(),
        }
    }

    fn transport_name(&self) -> &'static str {
        #[cfg(unix)]
        {
            "unix"
        }
        #[cfg(windows)]
        {
            "pipe"
        }
    }
}

impl LocalStream {
    async fn write_all(&mut self, bytes: &[u8]) -> Result<()> {
        match self {
            #[cfg(unix)]
            LocalStream::Unix(stream) => stream.write_all(bytes).await?,
            #[cfg(windows)]
            LocalStream::PipeServer(stream) => stream.write_all(bytes).await?,
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
            LocalStream::PipeServer(stream) => {
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

async fn bind_local_listener(name: &str) -> Result<LocalListener> {
    #[cfg(unix)]
    {
        let dir = runtime_dir()?;
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("Failed to create plugin runtime dir {}", dir.display()))?;
        let path = dir.join(format!("{name}.sock"));
        if path.exists() {
            let _ = std::fs::remove_file(&path);
        }
        let listener = tokio::net::UnixListener::bind(&path)
            .with_context(|| format!("Failed to bind plugin socket {}", path.display()))?;
        return Ok(LocalListener::Unix(listener, path));
    }
    #[cfg(windows)]
    {
        let endpoint = format!(r"\\.\pipe\mesh-llm-{name}");
        let server = tokio::net::windows::named_pipe::ServerOptions::new()
            .create(&endpoint)
            .with_context(|| format!("Failed to create plugin pipe {endpoint}"))?;
        return Ok(LocalListener::Pipe(endpoint, server));
    }
}

fn runtime_dir() -> Result<PathBuf> {
    let home = dirs::home_dir().context("Cannot determine home directory")?;
    Ok(home.join(".mesh-llm").join("run").join("plugins"))
}

pub async fn run_plugin_process(name: String) -> Result<()> {
    let endpoint = std::env::var("MESH_LLM_PLUGIN_ENDPOINT")
        .context("MESH_LLM_PLUGIN_ENDPOINT is not set for plugin process")?;
    let transport =
        std::env::var("MESH_LLM_PLUGIN_TRANSPORT").unwrap_or_else(|_| default_transport().into());
    let stream = connect_to_host(&endpoint, &transport).await?;

    match name.as_str() {
        BLACKBOARD_PLUGIN_ID => run_blackboard_plugin(name, stream).await,
        _ => bail!("Unknown built-in plugin '{}'", name),
    }
}

async fn run_blackboard_plugin(name: String, mut stream: LocalStream) -> Result<()> {
    use crate::blackboard::{
        BlackboardItem, BlackboardMessage, BlackboardStore, FeedRequest, PostRequest, SearchRequest,
        BLACKBOARD_CHANNEL,
    };

    let store = BlackboardStore::new(true);
    loop {
        let envelope = read_envelope(&mut stream).await?;
        match envelope.payload {
            Some(proto::envelope::Payload::InitializeRequest(_)) => {
                let response = proto::Envelope {
                    protocol_version: PROTOCOL_VERSION,
                    plugin_id: name.clone(),
                    request_id: envelope.request_id,
                    payload: Some(proto::envelope::Payload::InitializeResponse(
                        proto::InitializeResponse {
                            plugin_id: name.clone(),
                            plugin_protocol_version: PROTOCOL_VERSION,
                            plugin_version: crate::VERSION.to_string(),
                            capabilities: vec![
                                "channel:blackboard".into(),
                                "mcp:blackboard".into(),
                            ],
                            tool_schemas: vec![
                                proto::ToolSchema {
                                    name: "feed".into(),
                                    description: "Read the recent blackboard feed.".into(),
                                    input_schema_json: serde_json::json!({
                                        "type": "object",
                                        "properties": {
                                            "since": {"type": "integer"},
                                            "from": {"type": "string"},
                                            "limit": {"type": "integer"}
                                        }
                                    })
                                    .to_string(),
                                },
                                proto::ToolSchema {
                                    name: "search".into(),
                                    description: "Search blackboard messages.".into(),
                                    input_schema_json: serde_json::json!({
                                        "type": "object",
                                        "properties": {
                                            "query": {"type": "string"},
                                            "since": {"type": "integer"},
                                            "limit": {"type": "integer"}
                                        },
                                        "required": ["query"]
                                    })
                                    .to_string(),
                                },
                                proto::ToolSchema {
                                    name: "post".into(),
                                    description: "Post a blackboard message.".into(),
                                    input_schema_json: serde_json::json!({
                                        "type": "object",
                                        "properties": {
                                            "text": {"type": "string"},
                                            "from": {"type": "string"},
                                            "peer_id": {"type": "string"}
                                        },
                                        "required": ["text"]
                                    })
                                    .to_string(),
                                },
                            ],
                        },
                    )),
                };
                write_envelope(&mut stream, &response).await?;
                send_plugin_channel_message(
                    &mut stream,
                    &name,
                    proto::ChannelMessage {
                        channel: BLACKBOARD_CHANNEL.to_string(),
                        source_peer_id: String::new(),
                        target_peer_id: String::new(),
                        content_type: "application/json".into(),
                        body: serde_json::to_vec(&BlackboardMessage::SyncRequest)?,
                    },
                )
                .await?;
            }
            Some(proto::envelope::Payload::HealthRequest(_)) => {
                let response = proto::Envelope {
                    protocol_version: PROTOCOL_VERSION,
                    plugin_id: name.clone(),
                    request_id: envelope.request_id,
                    payload: Some(proto::envelope::Payload::HealthResponse(
                        proto::HealthResponse {
                            status: proto::health_response::Status::Ok as i32,
                            detail: "ok".into(),
                        },
                    )),
                };
                write_envelope(&mut stream, &response).await?;
            }
            Some(proto::envelope::Payload::ShutdownRequest(_)) => {
                let response = proto::Envelope {
                    protocol_version: PROTOCOL_VERSION,
                    plugin_id: name.clone(),
                    request_id: envelope.request_id,
                    payload: Some(proto::envelope::Payload::ShutdownResponse(
                        proto::ShutdownResponse {},
                    )),
                };
                write_envelope(&mut stream, &response).await?;
                break;
            }
            Some(proto::envelope::Payload::ToolCallRequest(call)) => {
                let payload = match call.name.as_str() {
                    "feed" => {
                        let request = serde_json::from_str::<FeedRequest>(&call.arguments_json)
                            .unwrap_or_default();
                        let items = store.feed(request.since, request.from.as_deref(), request.limit).await;
                        proto::envelope::Payload::ToolCallResponse(proto::ToolCallResponse {
                            content_json: serde_json::to_string(&items)?,
                            is_error: false,
                        })
                    }
                    "search" => {
                        match serde_json::from_str::<SearchRequest>(&call.arguments_json) {
                            Ok(request) => {
                                let mut items = store.search(&request.query, request.since).await;
                                items.truncate(request.limit.max(1));
                                proto::envelope::Payload::ToolCallResponse(
                                    proto::ToolCallResponse {
                                        content_json: serde_json::to_string(&items)?,
                                        is_error: false,
                                    },
                                )
                            }
                            Err(err) => {
                                proto::envelope::Payload::ErrorResponse(proto::ErrorResponse {
                                    code: proto::error_response::Code::InvalidRequest as i32,
                                    message: format!("Invalid search arguments: {err}"),
                                })
                            }
                        }
                    }
                    "post" => {
                        match serde_json::from_str::<PostRequest>(&call.arguments_json) {
                            Ok(request) => {
                                let from = if request.from.trim().is_empty() {
                                    "mcp".to_string()
                                } else {
                                    request.from
                                };
                                let peer_id = if request.peer_id.trim().is_empty() {
                                    "mcp".to_string()
                                } else {
                                    request.peer_id
                                };
                                let item = BlackboardItem::new(from, peer_id, request.text);
                                match store.post(item).await {
                                    Ok(posted) => {
                                        send_plugin_channel_message(
                                            &mut stream,
                                            &name,
                                            proto::ChannelMessage {
                                                channel: BLACKBOARD_CHANNEL.to_string(),
                                                source_peer_id: String::new(),
                                                target_peer_id: String::new(),
                                                content_type: "application/json".into(),
                                                body: serde_json::to_vec(&BlackboardMessage::Post(
                                                    posted.clone(),
                                                ))?,
                                            },
                                        )
                                        .await?;
                                        proto::envelope::Payload::ToolCallResponse(
                                            proto::ToolCallResponse {
                                                content_json: serde_json::to_string(&posted)?,
                                                is_error: false,
                                            },
                                        )
                                    }
                                    Err(reason) => {
                                        proto::envelope::Payload::ErrorResponse(
                                            proto::ErrorResponse {
                                                code: proto::error_response::Code::InvalidRequest
                                                    as i32,
                                                message: reason,
                                            },
                                        )
                                    }
                                }
                            }
                            Err(err) => {
                                proto::envelope::Payload::ErrorResponse(proto::ErrorResponse {
                                    code: proto::error_response::Code::InvalidRequest as i32,
                                    message: format!("Invalid post arguments: {err}"),
                                })
                            }
                        }
                    }
                    _ => proto::envelope::Payload::ErrorResponse(proto::ErrorResponse {
                        code: proto::error_response::Code::UnsupportedCapability as i32,
                        message: format!("Unknown tool '{}'", call.name),
                    }),
                };
                let response = proto::Envelope {
                    protocol_version: PROTOCOL_VERSION,
                    plugin_id: name.clone(),
                    request_id: envelope.request_id,
                    payload: Some(payload),
                };
                write_envelope(&mut stream, &response).await?;
            }
            Some(proto::envelope::Payload::ChannelMessage(message)) => {
                if message.channel != BLACKBOARD_CHANNEL {
                    continue;
                }
                let payload: BlackboardMessage = serde_json::from_slice(&message.body)?;
                match payload {
                    BlackboardMessage::Post(item) => {
                        let _ = store.insert(item).await;
                    }
                    BlackboardMessage::SyncRequest => {
                        let ids = store.ids().await;
                        let response = BlackboardMessage::SyncDigest(ids);
                        send_plugin_channel_message(
                            &mut stream,
                            &name,
                            proto::ChannelMessage {
                                channel: BLACKBOARD_CHANNEL.to_string(),
                                source_peer_id: String::new(),
                                target_peer_id: message.source_peer_id,
                                content_type: "application/json".into(),
                                body: serde_json::to_vec(&response)?,
                            },
                        )
                        .await?;
                    }
                    BlackboardMessage::SyncDigest(ids) => {
                        let our_ids = store.ids().await;
                        let missing: Vec<u64> = ids
                            .into_iter()
                            .filter(|id| !our_ids.contains(id))
                            .collect();
                        if !missing.is_empty() {
                            send_plugin_channel_message(
                                &mut stream,
                                &name,
                                proto::ChannelMessage {
                                    channel: BLACKBOARD_CHANNEL.to_string(),
                                    source_peer_id: String::new(),
                                    target_peer_id: message.source_peer_id,
                                    content_type: "application/json".into(),
                                    body: serde_json::to_vec(&BlackboardMessage::FetchRequest(
                                        missing,
                                    ))?,
                                },
                            )
                            .await?;
                        }
                    }
                    BlackboardMessage::FetchRequest(ids) => {
                        let items = store.get_by_ids(&ids).await;
                        send_plugin_channel_message(
                            &mut stream,
                            &name,
                            proto::ChannelMessage {
                                channel: BLACKBOARD_CHANNEL.to_string(),
                                source_peer_id: String::new(),
                                target_peer_id: message.source_peer_id,
                                content_type: "application/json".into(),
                                body: serde_json::to_vec(&BlackboardMessage::FetchResponse(
                                    items,
                                ))?,
                            },
                        )
                        .await?;
                    }
                    BlackboardMessage::FetchResponse(items) => {
                        for item in items {
                            let _ = store.insert(item).await;
                        }
                    }
                }
            }
            _ => {
                let response = proto::Envelope {
                    protocol_version: PROTOCOL_VERSION,
                    plugin_id: name.clone(),
                    request_id: envelope.request_id,
                    payload: Some(proto::envelope::Payload::ErrorResponse(
                        proto::ErrorResponse {
                            code: proto::error_response::Code::InvalidRequest as i32,
                            message: "Unsupported request".into(),
                        },
                    )),
                };
                write_envelope(&mut stream, &response).await?;
            }
        }
    }

    Ok(())
}

async fn send_plugin_channel_message(
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

async fn connect_to_host(endpoint: &str, transport: &str) -> Result<LocalStream> {
    match transport {
        #[cfg(unix)]
        "unix" => Ok(LocalStream::Unix(tokio::net::UnixStream::connect(endpoint).await?)),
        #[cfg(windows)]
        "pipe" => Ok(LocalStream::PipeClient(
            tokio::net::windows::named_pipe::ClientOptions::new().open(endpoint)?,
        )),
        _ => bail!("Unsupported plugin transport '{transport}'"),
    }
}

async fn write_envelope(stream: &mut LocalStream, envelope: &proto::Envelope) -> Result<()> {
    let mut body = Vec::new();
    envelope.encode(&mut body)?;
    stream.write_all(&(body.len() as u32).to_le_bytes()).await?;
    stream.write_all(&body).await?;
    Ok(())
}

async fn read_envelope(stream: &mut LocalStream) -> Result<proto::Envelope> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_default_blackboard_plugin() {
        let resolved = resolve_plugins(&MeshConfig::default()).unwrap();
        assert_eq!(resolved.externals.len(), 1);
        assert_eq!(resolved.externals[0].name, BLACKBOARD_PLUGIN_ID);
    }

    #[test]
    fn blackboard_can_be_disabled() {
        let config = MeshConfig {
            plugins: vec![PluginConfigEntry {
                name: BLACKBOARD_PLUGIN_ID.into(),
                enabled: Some(false),
                command: None,
                args: Vec::new(),
            }],
        };
        let resolved = resolve_plugins(&config).unwrap();
        assert!(resolved.externals.is_empty());
    }

    #[test]
    fn resolves_external_plugin() {
        let config = MeshConfig {
            plugins: vec![PluginConfigEntry {
                name: "demo".into(),
                enabled: Some(true),
                command: Some("/tmp/demo".into()),
                args: vec!["--flag".into()],
            }],
        };
        let resolved = resolve_plugins(&config).unwrap();
        assert_eq!(resolved.externals.len(), 2);
        assert_eq!(resolved.externals[1].name, "demo");
    }
}
