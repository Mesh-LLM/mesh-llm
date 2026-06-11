#![forbid(unsafe_code)]

use clap::ValueEnum;
use serde_json::Value;
use std::future::Future;
use std::io::{self, IsTerminal};
use std::pin::Pin;
use std::sync::{Arc, OnceLock, RwLock};

pub mod terminal_progress;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
pub enum LogFormat {
    #[default]
    Pretty,
    Json,
}

#[allow(dead_code)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RuntimeStatus {
    NotReady,
    Starting,
    Loading,
    Ready,
    ShuttingDown,
    Stopped,
    Exited,
    Warning,
    Error,
}

impl RuntimeStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            RuntimeStatus::NotReady => "NOT READY",
            RuntimeStatus::Starting => "starting",
            RuntimeStatus::Loading => "loading",
            RuntimeStatus::Ready => "ready",
            RuntimeStatus::ShuttingDown => "shutting down",
            RuntimeStatus::Stopped => "stopped",
            RuntimeStatus::Exited => "exited",
            RuntimeStatus::Warning => "warning",
            RuntimeStatus::Error => "error",
        }
    }
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConsoleSessionMode {
    InteractiveDashboard,
    Fallback,
    None,
}

#[allow(dead_code)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DashboardProcessRow {
    pub name: String,
    pub backend: String,
    pub status: RuntimeStatus,
    pub port: u16,
    pub pid: u32,
}

#[allow(dead_code)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DashboardEndpointRow {
    pub label: String,
    pub status: RuntimeStatus,
    pub url: String,
    pub port: u16,
    pub pid: Option<u32>,
}

#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq)]
pub struct DashboardModelRow {
    pub name: String,
    pub role: Option<String>,
    pub status: RuntimeStatus,
    pub port: Option<u16>,
    pub device: Option<String>,
    pub slots: Option<usize>,
    pub quantization: Option<String>,
    pub ctx_size: Option<u32>,
    pub ctx_used_tokens: Option<u64>,
    pub lanes: Option<Vec<DashboardModelLane>>,
    pub file_size_gb: Option<f64>,
}

#[allow(dead_code)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DashboardModelLane {
    pub index: usize,
    pub active: bool,
}

#[allow(dead_code)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DashboardAcceptedRequestBucket {
    pub second_offset: u32,
    pub accepted_count: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ModelProgressStatus {
    Ensuring,
    Downloading,
    Ready,
}

impl ModelProgressStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            ModelProgressStatus::Ensuring => "ensuring",
            ModelProgressStatus::Downloading => "downloading",
            ModelProgressStatus::Ready => "ready",
        }
    }
}

#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq)]
pub struct DashboardSnapshot {
    pub llama_process_rows: Vec<DashboardProcessRow>,
    pub webserver_rows: Vec<DashboardEndpointRow>,
    pub loaded_model_rows: Vec<DashboardModelRow>,
    pub current_inflight_requests: u64,
    pub accepted_request_buckets: Vec<DashboardAcceptedRequestBucket>,
    pub latency_samples_ms: Vec<u64>,
}

impl Default for DashboardSnapshot {
    fn default() -> Self {
        Self {
            llama_process_rows: Vec::new(),
            webserver_rows: Vec::new(),
            loaded_model_rows: Vec::new(),
            current_inflight_requests: 0,
            accepted_request_buckets: (0..30)
                .map(|second_offset| DashboardAcceptedRequestBucket {
                    second_offset,
                    accepted_count: 0,
                })
                .collect(),
            latency_samples_ms: Vec::new(),
        }
    }
}

#[allow(dead_code)]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DashboardLaunchPlan {
    pub llama_process_rows: Vec<DashboardProcessRow>,
    pub webserver_rows: Vec<DashboardEndpointRow>,
    pub loaded_model_rows: Vec<DashboardModelRow>,
}

#[allow(dead_code)]
pub type DashboardSnapshotFuture<'a> = Pin<Box<dyn Future<Output = DashboardSnapshot> + Send + 'a>>;

#[allow(dead_code)]
pub trait DashboardSnapshotProvider: Send + Sync {
    fn snapshot(&self) -> DashboardSnapshotFuture<'_>;
}

pub type OutputSinkFuture<'a, T> = Pin<Box<dyn Future<Output = io::Result<T>> + Send + 'a>>;

pub trait OutputSink: Send + Sync {
    fn emit_event(&self, event: OutputEvent) -> io::Result<()>;

    fn schedule_ready_prompt(&self) -> io::Result<()> {
        Ok(())
    }

    fn write_ready_prompt(&self) -> io::Result<()> {
        Ok(())
    }

    fn ready_prompt_active(&self) -> bool {
        false
    }

    fn flush(&self) -> OutputSinkFuture<'_, ()> {
        Box::pin(async { Ok(()) })
    }

    fn mode(&self) -> LogFormat {
        LogFormat::Pretty
    }

    fn console_session_mode(&self) -> Option<ConsoleSessionMode> {
        None
    }

    fn register_dashboard_snapshot_provider(&self, _provider: Arc<dyn DashboardSnapshotProvider>) {}

    fn enter_tui(&self) -> OutputSinkFuture<'_, ()> {
        Box::pin(async { Ok(()) })
    }

    fn exit_tui(&self) -> OutputSinkFuture<'_, ()> {
        Box::pin(async { Ok(()) })
    }

    fn dispatch_tui_event(&self, _event: TuiEvent) -> OutputSinkFuture<'_, TuiControlFlow> {
        Box::pin(async { Ok(TuiControlFlow::Continue) })
    }

    fn render_tui_if_dirty(&self) -> OutputSinkFuture<'_, bool> {
        Box::pin(async { Ok(false) })
    }

    fn force_restore_tui_terminal(&self) -> io::Result<()> {
        Ok(())
    }
}

static OUTPUT_SINK: OnceLock<RwLock<Option<Arc<dyn OutputSink>>>> = OnceLock::new();

fn output_sink_slot() -> &'static RwLock<Option<Arc<dyn OutputSink>>> {
    OUTPUT_SINK.get_or_init(|| RwLock::new(None))
}

pub fn set_output_sink(sink: Arc<dyn OutputSink>) {
    if let Ok(mut slot) = output_sink_slot().write() {
        *slot = Some(sink);
    }
}

pub fn clear_output_sink() {
    if let Ok(mut slot) = output_sink_slot().write() {
        *slot = None;
    }
}

pub fn output_sink() -> Option<Arc<dyn OutputSink>> {
    output_sink_slot()
        .read()
        .ok()
        .and_then(|slot| slot.as_ref().cloned())
}

pub fn emit_event(event: OutputEvent) -> io::Result<()> {
    match output_sink() {
        Some(sink) => sink.emit_event(event),
        None => Ok(()),
    }
}

pub async fn flush_output() -> io::Result<()> {
    match output_sink() {
        Some(sink) => sink.flush().await,
        None => Ok(()),
    }
}

pub fn schedule_ready_prompt() -> io::Result<()> {
    match output_sink() {
        Some(sink) => sink.schedule_ready_prompt(),
        None => Ok(()),
    }
}

pub fn json_mode_enabled() -> bool {
    output_sink().is_some_and(|sink| matches!(sink.mode(), LogFormat::Json))
}

pub fn interactive_tui_active() -> bool {
    output_sink().is_some_and(|sink| {
        matches!(sink.mode(), LogFormat::Pretty)
            && matches!(
                sink.console_session_mode(),
                Some(ConsoleSessionMode::InteractiveDashboard)
            )
    })
}

pub fn current_console_session_mode() -> ConsoleSessionMode {
    console_session_mode(
        std::io::stdin().is_terminal(),
        std::io::stderr().is_terminal(),
    )
}

pub fn console_session_mode(stdin_is_tty: bool, stderr_is_tty: bool) -> ConsoleSessionMode {
    console_session_mode_for_term(
        stdin_is_tty,
        stderr_is_tty,
        std::env::var("TERM").ok().as_deref(),
    )
}

pub fn console_session_mode_for_term(
    stdin_is_tty: bool,
    stderr_is_tty: bool,
    term: Option<&str>,
) -> ConsoleSessionMode {
    if stdin_is_tty && stderr_is_tty && terminal_supports_dashboard(term) {
        ConsoleSessionMode::InteractiveDashboard
    } else {
        ConsoleSessionMode::Fallback
    }
}

fn terminal_supports_dashboard(term: Option<&str>) -> bool {
    match term.map(str::trim).filter(|term| !term.is_empty()) {
        Some(term) => term != "dumb",
        None => false,
    }
}

pub fn sort_dashboard_endpoint_rows(rows: &mut [DashboardEndpointRow]) {
    rows.sort_by(|left, right| {
        dashboard_endpoint_sort_bucket(left)
            .cmp(&dashboard_endpoint_sort_bucket(right))
            .then_with(|| left.label.cmp(&right.label))
    });
}

fn dashboard_endpoint_sort_bucket(row: &DashboardEndpointRow) -> u8 {
    if row.label.starts_with("Plugin: ") {
        1
    } else {
        0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TuiKeyEvent {
    Tab,
    BackTab,
    Backspace,
    Enter,
    Escape,
    Left,
    Right,
    Up,
    Down,
    PageUp,
    PageDown,
    Interrupt,
    Char(char),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TuiEvent {
    Key(TuiKeyEvent),
    Resize { columns: u16, rows: u16 },
    MouseDown { column: u16, row: u16 },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TuiControlFlow {
    Continue,
    Quit,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OutputLevel {
    Debug,
    Info,
    Warn,
    Error,
    Fatal,
}

impl OutputLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            OutputLevel::Debug => "debug",
            OutputLevel::Info => "info",
            OutputLevel::Warn => "warn",
            OutputLevel::Error => "error",
            OutputLevel::Fatal => "fatal",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LlamaInstanceKind {
    LlamaServer,
}

impl LlamaInstanceKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            LlamaInstanceKind::LlamaServer => "llama-server",
        }
    }

    pub fn sort_key(&self) -> u8 {
        match self {
            LlamaInstanceKind::LlamaServer => 0,
        }
    }
}

#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq)]
pub enum OutputEvent {
    Info {
        message: String,
        context: Option<String>,
    },
    Startup {
        version: String,
        message: Option<String>,
    },
    LaunchPlan {
        plan: DashboardLaunchPlan,
    },
    NodeIdentity {
        node_id: String,
        mesh_id: Option<String>,
    },
    InviteToken {
        token: String,
        mesh_id: String,
        mesh_name: Option<String>,
    },
    DiscoveryStarting {
        source: String,
    },
    MeshFound {
        mesh: String,
        peers: usize,
        region: Option<String>,
    },
    DiscoveryJoined {
        mesh: String,
    },
    DiscoveryFailed {
        message: String,
        detail: Option<String>,
    },
    WaitingForPeers {
        detail: Option<String>,
    },
    PassiveMode {
        role: String,
        status: RuntimeStatus,
        capacity_gb: Option<f64>,
        models_on_disk: Option<Vec<String>>,
        detail: Option<String>,
    },
    PeerJoined {
        peer_id: String,
        label: Option<String>,
    },
    PeerLeft {
        peer_id: String,
        reason: Option<String>,
    },
    ModelQueued {
        model: String,
    },
    ModelLoading {
        model: String,
        source: Option<String>,
    },
    ModelLoaded {
        model: String,
        bytes: Option<u64>,
    },
    ModelUnloading {
        model: String,
    },
    ModelUnloaded {
        model: String,
    },
    HostElected {
        model: String,
        host: String,
        role: Option<String>,
        capacity_gb: Option<f64>,
    },
    RpcServerStarting {
        port: u16,
        device: String,
        log_path: Option<String>,
    },
    RpcReady {
        port: u16,
        device: String,
        log_path: Option<String>,
    },
    RpcStartupFailed {
        port: u16,
        device: String,
        log_path: Option<String>,
        detail: String,
    },
    LlamaStarting {
        model: Option<String>,
        http_port: u16,
        ctx_size: Option<u32>,
        log_path: Option<String>,
    },
    LlamaReady {
        model: Option<String>,
        port: u16,
        ctx_size: Option<u32>,
        log_path: Option<String>,
    },
    LlamaStartupFailed {
        model: Option<String>,
        http_port: u16,
        ctx_size: Option<u32>,
        log_path: Option<String>,
        detail: String,
    },
    ModelReady {
        model: String,
        internal_port: Option<u16>,
        role: Option<String>,
    },
    MultiModelMode {
        count: usize,
        models: Vec<String>,
    },
    WebserverStarting {
        url: String,
    },
    WebserverReady {
        url: String,
    },
    ApiStarting {
        url: String,
    },
    ApiReady {
        url: String,
    },
    RuntimeReady {
        api_url: String,
        console_url: Option<String>,
        api_port: u16,
        console_port: Option<u16>,
        models_count: Option<usize>,
        pi_command: Option<String>,
        goose_command: Option<String>,
    },
    ModelDownloadProgress {
        label: String,
        file: Option<String>,
        downloaded_bytes: Option<u64>,
        total_bytes: Option<u64>,
        status: ModelProgressStatus,
    },
    RequestRouted {
        model: String,
        target: String,
    },
    Warning {
        message: String,
        context: Option<String>,
    },
    Error {
        message: String,
        context: Option<String>,
    },
    Fatal {
        message: String,
        context: Option<String>,
    },
    ShutdownRequested {
        signal: &'static str,
    },
    Shutdown {
        reason: Option<String>,
    },
    LlamaNativeLog {
        message: String,
        category: &'static str,
        params: Vec<(String, Value)>,
    },
}

impl OutputEvent {
    pub fn event_name(&self) -> &'static str {
        match self {
            OutputEvent::Info { .. } => "info",
            OutputEvent::Startup { .. } => "startup",
            OutputEvent::LaunchPlan { .. } => "launch_plan",
            OutputEvent::NodeIdentity { .. } => "node_identity",
            OutputEvent::InviteToken { .. } => "invite_token",
            OutputEvent::DiscoveryStarting { .. } => "discovery_starting",
            OutputEvent::MeshFound { .. } => "mesh_found",
            OutputEvent::DiscoveryJoined { .. } => "discovery_joined",
            OutputEvent::DiscoveryFailed { .. } => "discovery_failed",
            OutputEvent::WaitingForPeers { .. } => "waiting_for_peers",
            OutputEvent::PassiveMode { .. } => "passive_mode",
            OutputEvent::PeerJoined { .. } => "peer_joined",
            OutputEvent::PeerLeft { .. } => "peer_left",
            OutputEvent::ModelQueued { .. } => "model_queued",
            OutputEvent::ModelLoading { .. } => "model_loading",
            OutputEvent::ModelLoaded { .. } => "model_loaded",
            OutputEvent::ModelUnloading { .. } => "model_unloading",
            OutputEvent::ModelUnloaded { .. } => "model_unloaded",
            OutputEvent::HostElected { .. } => "host_elected",
            OutputEvent::RpcServerStarting { .. } => "rpc_server_starting",
            OutputEvent::RpcReady { .. } => "rpc_ready",
            OutputEvent::RpcStartupFailed { .. } => "rpc_startup_failed",
            OutputEvent::LlamaStarting { .. } => "llama_starting",
            OutputEvent::LlamaReady { .. } => "llama_ready",
            OutputEvent::LlamaStartupFailed { .. } => "llama_startup_failed",
            OutputEvent::ModelReady { .. } => "model_ready",
            OutputEvent::MultiModelMode { .. } => "multi_model_mode",
            OutputEvent::WebserverStarting { .. } => "webserver_starting",
            OutputEvent::WebserverReady { .. } => "webserver_ready",
            OutputEvent::ApiStarting { .. } => "api_starting",
            OutputEvent::ApiReady { .. } => "api_ready",
            OutputEvent::RuntimeReady { .. } => "ready",
            OutputEvent::ModelDownloadProgress { .. } => "model_download_progress",
            OutputEvent::RequestRouted { .. } => "request_routed",
            OutputEvent::Warning { .. } => "warning",
            OutputEvent::Error { .. } => "error",
            OutputEvent::Fatal { .. } => "fatal",
            OutputEvent::ShutdownRequested { signal } => signal,
            OutputEvent::Shutdown { .. } => "shutdown",
            OutputEvent::LlamaNativeLog { category, .. } => category,
        }
    }

    pub fn level(&self) -> OutputLevel {
        match self {
            OutputEvent::RpcStartupFailed { .. } | OutputEvent::LlamaStartupFailed { .. } => {
                OutputLevel::Error
            }
            OutputEvent::LlamaNativeLog { .. } => OutputLevel::Debug,
            OutputEvent::Warning { .. } => OutputLevel::Warn,
            OutputEvent::Error { .. } => OutputLevel::Error,
            OutputEvent::Fatal { .. } => OutputLevel::Fatal,
            _ => OutputLevel::Info,
        }
    }

    pub fn message(&self) -> String {
        match self {
            OutputEvent::Info { message, .. } => message.clone(),
            OutputEvent::Startup { message, .. } => message
                .clone()
                .unwrap_or_else(|| "mesh-llm starting".to_string()),
            OutputEvent::LaunchPlan { plan } => format!(
                "startup plan ready ({} process(es), {} endpoint(s), {} model(s))",
                plan.llama_process_rows.len(),
                plan.webserver_rows.len(),
                plan.loaded_model_rows.len()
            ),
            OutputEvent::NodeIdentity { node_id, mesh_id } => match mesh_id {
                Some(mesh_id) => format!("node {node_id} joined mesh {mesh_id}"),
                None => format!("node {node_id} initialized"),
            },
            OutputEvent::InviteToken {
                mesh_id, mesh_name, ..
            } => {
                let mesh_label = format_invite_mesh_label(mesh_name.as_deref(), mesh_id);
                format!("invite token ready for mesh {mesh_label}")
            }
            OutputEvent::DiscoveryStarting { source } => format!("discovering mesh via {source}"),
            OutputEvent::MeshFound { mesh, peers, .. } => {
                format!("discovered mesh {mesh} ({peers} peer(s))")
            }
            OutputEvent::DiscoveryJoined { mesh } => format!("joined mesh {mesh}"),
            OutputEvent::DiscoveryFailed { message, detail } => match detail {
                Some(detail) => format!("{message}: {detail}"),
                None => message.clone(),
            },
            OutputEvent::WaitingForPeers { detail } => detail
                .clone()
                .unwrap_or_else(|| "waiting for peers".to_string()),
            OutputEvent::PassiveMode {
                role,
                status,
                capacity_gb,
                models_on_disk,
                detail,
            } => {
                let mut line = detail
                    .clone()
                    .unwrap_or_else(|| format!("{role} {}", status.as_str()));
                if let Some(capacity_gb) = capacity_gb {
                    line.push_str(&format!(" ({capacity_gb:.1}GB capacity)"));
                }
                if let Some(models_on_disk) = models_on_disk
                    && !models_on_disk.is_empty()
                {
                    line.push_str(&format!(" models={}", models_on_disk.join(", ")));
                }
                line
            }
            OutputEvent::PeerJoined { peer_id, .. } => format!("peer {peer_id} joined"),
            OutputEvent::PeerLeft { peer_id, .. } => format!("peer {peer_id} left"),
            OutputEvent::ModelQueued { model } => format!("queued model {model}"),
            OutputEvent::ModelLoading { model, .. } => format!("loading model {model}"),
            OutputEvent::ModelLoaded { model, .. } => format!("loaded model {model}"),
            OutputEvent::ModelUnloading { model } => format!("unloading model {model}"),
            OutputEvent::ModelUnloaded { model } => format!("unloaded model {model}"),
            OutputEvent::HostElected {
                model, host, role, ..
            } => match role {
                Some(role) => format!("{model} elected {host} as {role}"),
                None => format!("{model} elected {host} as host"),
            },
            OutputEvent::RpcServerStarting { port, log_path, .. } => {
                let msg = format!("rpc-server starting on port {port}");
                append_log_path(msg, log_path)
            }
            OutputEvent::RpcReady { port, log_path, .. } => {
                let msg = format!("rpc-server ready on port {port}");
                append_log_path(msg, log_path)
            }
            OutputEvent::RpcStartupFailed {
                port,
                detail,
                log_path,
                ..
            } => {
                let msg = format!("rpc-server failed to start on port {port}: {detail}");
                append_log_path(msg, log_path)
            }
            OutputEvent::LlamaStarting {
                http_port,
                log_path,
                ..
            } => {
                let msg = format!("llama-server starting on port {http_port}");
                append_log_path(msg, log_path)
            }
            OutputEvent::LlamaReady { port, log_path, .. } => {
                let msg = format!("llama-server ready on port {port}");
                append_log_path(msg, log_path)
            }
            OutputEvent::LlamaStartupFailed {
                model,
                http_port,
                detail,
                log_path,
                ..
            } => {
                let msg = match model {
                    Some(model) => {
                        format!(
                            "llama-server failed to start for {model} on port {http_port}: {detail}"
                        )
                    }
                    None => format!("llama-server failed to start on port {http_port}: {detail}"),
                };
                append_log_path(msg, log_path)
            }
            OutputEvent::ModelReady {
                model,
                internal_port,
                ..
            } => match internal_port {
                Some(port) => format!("model {model} ready on port {port}"),
                None => format!("model {model} ready"),
            },
            OutputEvent::WebserverStarting { url } => format!("web console starting at {url}"),
            OutputEvent::WebserverReady { url } => format!("web console ready at {url}"),
            OutputEvent::ApiStarting { url } => format!("api starting at {url}"),
            OutputEvent::ApiReady { url } => format!("api ready at {url}"),
            OutputEvent::RuntimeReady { .. } => "mesh-llm runtime ready".to_string(),
            OutputEvent::ModelDownloadProgress {
                label,
                file,
                downloaded_bytes,
                total_bytes,
                status,
            } => format_model_download_progress_message(
                label,
                file.as_deref(),
                *downloaded_bytes,
                *total_bytes,
                status,
            ),
            OutputEvent::MultiModelMode { count, models } => {
                if models.is_empty() {
                    format!("Multi-model mode: {count} model(s)")
                } else {
                    format!("Multi-model mode: {count} model(s): {}", models.join(", "))
                }
            }
            OutputEvent::RequestRouted { model, target } => {
                format!("routed request for {model} to {target}")
            }
            OutputEvent::Warning { message, .. } => message.clone(),
            OutputEvent::Error { message, .. } => message.clone(),
            OutputEvent::Fatal { message, .. } => message.clone(),
            OutputEvent::ShutdownRequested { signal } => format!("shutdown requested ({signal})"),
            OutputEvent::Shutdown { reason } => reason
                .clone()
                .unwrap_or_else(|| "mesh-llm shutting down".to_string()),
            OutputEvent::LlamaNativeLog { message, .. } => message.clone(),
        }
    }
}

fn append_log_path(message: String, log_path: &Option<String>) -> String {
    if let Some(path) = log_path {
        format!("{message}\n  ↳ log={path}")
    } else {
        message
    }
}

fn format_invite_mesh_label(mesh_name: Option<&str>, mesh_id: &str) -> String {
    match mesh_name.map(str::trim).filter(|name| !name.is_empty()) {
        Some(name) => format!("{name} ({mesh_id})"),
        None => mesh_id.to_string(),
    }
}

pub fn format_model_download_progress_message(
    label: &str,
    file: Option<&str>,
    downloaded_bytes: Option<u64>,
    total_bytes: Option<u64>,
    status: &ModelProgressStatus,
) -> String {
    let target = file.unwrap_or(label);
    if let Some(package) = label.strip_prefix("layer package ") {
        return match status {
            ModelProgressStatus::Ensuring => {
                format!("ensuring layer package artifact {target} for {package}")
            }
            ModelProgressStatus::Downloading => match (downloaded_bytes, total_bytes) {
                (Some(downloaded), Some(total)) if total > 0 => format!(
                    "downloading layer package artifact {target} for {package} {}/{}",
                    format_display_bytes(downloaded),
                    format_display_bytes(total)
                ),
                (Some(downloaded), _) if downloaded > 0 => format!(
                    "downloading layer package artifact {target} for {package} {}",
                    format_display_bytes(downloaded)
                ),
                _ => format!("downloading layer package artifact {target} for {package}"),
            },
            ModelProgressStatus::Ready => match total_bytes {
                Some(total) if total > 0 => format!(
                    "layer package artifact {target} ready for {package} ({})",
                    format_display_bytes(total)
                ),
                _ => format!("layer package artifact {target} ready for {package}"),
            },
        };
    }
    match status {
        ModelProgressStatus::Ensuring => format!("ensuring model {target}"),
        ModelProgressStatus::Downloading => match (downloaded_bytes, total_bytes) {
            (Some(downloaded), Some(total)) if total > 0 => format!(
                "downloading model {target} {}/{}",
                format_display_bytes(downloaded),
                format_display_bytes(total)
            ),
            (Some(downloaded), _) if downloaded > 0 => {
                format!(
                    "downloading model {target} {}",
                    format_display_bytes(downloaded)
                )
            }
            _ => format!("downloading model {target}"),
        },
        ModelProgressStatus::Ready => match total_bytes {
            Some(total) if total > 0 => {
                format!("model {target} ready ({})", format_display_bytes(total))
            }
            _ => format!("model {target} ready"),
        },
    }
}

fn format_display_bytes(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1}GB", bytes as f64 / 1e9)
    } else if bytes >= 1_000_000 {
        format!("{:.0}MB", bytes as f64 / 1e6)
    } else if bytes >= 1_000 {
        format!("{:.0}KB", bytes as f64 / 1e3)
    } else {
        format!("{bytes}B")
    }
}
