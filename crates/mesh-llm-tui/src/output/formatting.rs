use super::{
    ConsoleSessionMode, DashboardAction, DashboardLayoutState, DashboardSnapshot,
    DashboardSnapshotProvider, DashboardState, LogFormat, ModelProgressStatus, OutputEvent,
    OutputSink, OutputSinkFuture, PRETTY_TUI_JOIN_TOKEN_PANEL_HEIGHT,
    PRETTY_TUI_MIN_DASHBOARD_WIDTH, PRETTY_TUI_REDRAW_INTERVAL, PRETTY_TUI_SNAPSHOT_INTERVAL,
    RuntimeStatus, TuiControlFlow, TuiEvent, TuiTerminal, draw_tui_dashboard_with_terminal,
    format_invite_mesh_label, render_dashboard_text, strip_leading_severity_icon,
};
use chrono::{SecondsFormat, Utc};
use crossterm::{
    cursor::{Hide, MoveTo, Show},
    execute,
    terminal::{Clear, ClearType, EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode},
};
use ratatui::{Terminal, backend::CrosstermBackend};
use serde_json::{Map, Value, json};
use std::io::{self, Write};
use std::sync::{
    Arc, OnceLock, RwLock,
    atomic::{AtomicBool, Ordering},
};
use tokio::time::{self, Instant, MissedTickBehavior};

pub(in crate::output) trait OutputEventPresentation {
    fn pretty_text(&self) -> String;
    fn summary_line(&self) -> String;
    fn json_fields(&self) -> Map<String, Value>;
    fn passive_mode_summary(
        role: &str,
        status: &RuntimeStatus,
        capacity_gb: Option<f64>,
        models_on_disk: Option<&[String]>,
        detail: Option<&str>,
    ) -> String;
    fn host_elected_summary(
        model: &str,
        host: &str,
        role: Option<&str>,
        capacity_gb: Option<f64>,
    ) -> String;
    fn model_loaded_summary(model: &str, bytes: Option<u64>) -> String;
    fn llama_starting_summary(model: Option<&str>, http_port: u16, ctx_size: Option<u32>)
    -> String;
    fn contextual_summary(context: Option<&str>, message: &str) -> String;
}

impl OutputEventPresentation for OutputEvent {
    fn pretty_text(&self) -> String {
        match self {
            OutputEvent::LlamaNativeLog {
                message, params, ..
            } => format_message_with_params(message, params),
            _ => self.summary_line(),
        }
    }

    fn summary_line(&self) -> String {
        match self {
            OutputEvent::Info { message, context } => match context {
                Some(context) => format!("{context}: {message}"),
                None => message.clone(),
            },
            OutputEvent::DiscoveryStarting { source } => {
                format!("🔍 discovering mesh via {source}")
            }
            OutputEvent::LaunchPlan { plan } => format!(
                "📋 startup plan ready: {} process(es), {} endpoint(s), {} model(s)",
                plan.llama_process_rows.len(),
                plan.webserver_rows.len(),
                plan.loaded_model_rows.len()
            ),
            OutputEvent::MeshFound {
                mesh,
                peers,
                region,
            } => match region {
                Some(region) => {
                    format!("📡 discovered mesh {mesh} ({peers} peer(s)) region={region}")
                }
                None => format!("📡 discovered mesh {mesh} ({peers} peer(s))"),
            },
            OutputEvent::DiscoveryJoined { mesh } => format!("✅ joined mesh {mesh}"),
            OutputEvent::DiscoveryFailed { message, detail } => match detail {
                Some(detail) => format!("⚠️ {message}: {detail}"),
                None => format!("⚠️ {message}"),
            },
            OutputEvent::InviteToken {
                token,
                mesh_id,
                mesh_name,
            } => {
                let mesh_label = format_invite_mesh_label(mesh_name.as_deref(), mesh_id);
                format!("📡 Invite created for mesh {mesh_label}: {token}")
            }
            OutputEvent::WaitingForPeers { detail } => detail
                .clone()
                .map(|detail| format!("⏳ {detail}"))
                .unwrap_or_else(|| "⏳ Waiting for peers...".to_string()),
            OutputEvent::PassiveMode {
                role,
                status,
                capacity_gb,
                models_on_disk,
                detail,
            } => Self::passive_mode_summary(
                role,
                status,
                *capacity_gb,
                models_on_disk.as_deref(),
                detail.as_deref(),
            ),
            OutputEvent::HostElected {
                model,
                host,
                role,
                capacity_gb,
            } => Self::host_elected_summary(model, host, role.as_deref(), *capacity_gb),
            OutputEvent::PeerJoined { peer_id, label } => match label {
                Some(label) => format!("🤝 Peer joined: {label} ({peer_id})"),
                None => format!("🤝 Peer joined: {peer_id}"),
            },
            OutputEvent::PeerLeft { peer_id, reason } => match reason {
                Some(reason) => format!("👋 Peer left: {peer_id} ({reason})"),
                None => format!("👋 Peer left: {peer_id}"),
            },
            OutputEvent::ModelLoaded { model, bytes } => Self::model_loaded_summary(model, *bytes),
            OutputEvent::ModelUnloading { model } => format!("📤 Unloading model: {model}"),
            OutputEvent::ModelUnloaded { model } => format!("✅ Model unloaded: {model}"),
            OutputEvent::RpcServerStarting { port, device, .. } => {
                format!("🧵 rpc-server starting: port={port} device={device}")
            }
            OutputEvent::RpcStartupFailed {
                port,
                device,
                detail,
                ..
            } => {
                format!("❌ rpc-server failed: port={port} device={device} {detail}")
            }
            OutputEvent::LlamaStarting {
                model,
                http_port,
                ctx_size,
                ..
            } => Self::llama_starting_summary(model.as_deref(), *http_port, *ctx_size),
            OutputEvent::LlamaReady { model, port, .. } => match model {
                Some(model) => format!("✅ {model} ready on internal port {port}"),
                None => format!("✅ llama-server ready on port {port}"),
            },
            OutputEvent::LlamaStartupFailed {
                model,
                http_port,
                detail,
                ..
            } => match model {
                Some(model) => {
                    format!("❌ {model} failed to start on port {http_port}: {detail}")
                }
                None => format!("❌ llama-server failed to start on port {http_port}: {detail}"),
            },
            OutputEvent::RuntimeReady { models_count, .. } => match models_count {
                Some(count) => format!("✅ Mesh runtime ready ({count} model(s))"),
                None => "✅ Mesh runtime ready".to_string(),
            },
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
            OutputEvent::Error { context, message }
            | OutputEvent::Warning { message, context }
            | OutputEvent::Fatal { message, context } => {
                Self::contextual_summary(context.as_deref(), message)
            }
            OutputEvent::LlamaNativeLog { message, .. } => message.clone(),
            _ => self.message().to_string(),
        }
    }

    fn passive_mode_summary(
        role: &str,
        status: &RuntimeStatus,
        capacity_gb: Option<f64>,
        models_on_disk: Option<&[String]>,
        detail: Option<&str>,
    ) -> String {
        let prefix = if role == "client" { "📡" } else { "💤" };
        let mut line = match status {
            RuntimeStatus::Ready => format!("{prefix} {role} ready"),
            _ => format!(
                "{prefix} {}",
                detail
                    .map(str::to_string)
                    .unwrap_or_else(|| format_role_active(role))
            ),
        };
        if let Some(capacity_gb) = capacity_gb {
            line.push_str(&format!(" ({capacity_gb:.1}GB capacity)"));
        }
        append_models_on_disk(&mut line, models_on_disk);
        line
    }

    fn host_elected_summary(
        model: &str,
        host: &str,
        role: Option<&str>,
        capacity_gb: Option<f64>,
    ) -> String {
        match (role, capacity_gb) {
            (Some(role), Some(capacity)) => {
                format!("🗳 {model} elected {host} as {role} ({capacity:.1}GB capacity)")
            }
            (Some(role), None) => format!("🗳 {model} elected {host} as {role}"),
            (None, Some(capacity)) => {
                format!("🗳 {model} elected {host} ({capacity:.1}GB capacity)")
            }
            (None, None) => format!("🗳 {model} elected {host}"),
        }
    }

    fn model_loaded_summary(model: &str, bytes: Option<u64>) -> String {
        let mut line = format!("📦 Model loaded: {model}");
        if let Some(bytes) = bytes {
            line.push_str(&format!(" ({})", format_model_size(bytes)));
        }
        line
    }

    fn llama_starting_summary(
        model: Option<&str>,
        http_port: u16,
        ctx_size: Option<u32>,
    ) -> String {
        let mut line = format!("🦙 llama-server starting: port={http_port}");
        if let Some(model) = model {
            line.push_str(&format!(" model={model}"));
        }
        if let Some(ctx_size) = ctx_size {
            line.push_str(&format!(" ctx={ctx_size}"));
        }
        line
    }

    fn contextual_summary(context: Option<&str>, message: &str) -> String {
        let message = strip_leading_severity_icon(message);
        match context {
            Some(context) => format!("{context}: {message}"),
            None => message.to_string(),
        }
    }

    fn json_fields(&self) -> Map<String, Value> {
        let value = match self {
            OutputEvent::Info { message, context } => {
                json!({ "message": message, "context": context })
            }
            OutputEvent::Startup { version, .. } => json!({ "version": version }),
            OutputEvent::LaunchPlan { plan } => json!({
                "llama_process_count": plan.llama_process_rows.len(),
                "webserver_count": plan.webserver_rows.len(),
                "loaded_model_count": plan.loaded_model_rows.len(),
            }),
            OutputEvent::NodeIdentity { node_id, mesh_id } => {
                json!({ "node_id": node_id, "mesh_id": mesh_id })
            }
            OutputEvent::InviteToken {
                token,
                mesh_id,
                mesh_name,
            } => {
                json!({ "token": token, "mesh_id": mesh_id, "mesh_name": mesh_name })
            }
            OutputEvent::DiscoveryStarting { source } => json!({ "source": source }),
            OutputEvent::MeshFound {
                mesh,
                peers,
                region,
            } => json!({ "mesh": mesh, "peers": peers, "region": region }),
            OutputEvent::DiscoveryJoined { mesh } => json!({ "mesh": mesh }),
            OutputEvent::DiscoveryFailed { message, detail } => {
                json!({ "message": message, "detail": detail })
            }
            OutputEvent::WaitingForPeers { detail } => json!({ "detail": detail }),
            OutputEvent::PassiveMode {
                role,
                status,
                capacity_gb,
                models_on_disk,
                detail,
            } => json!({
                "role": role,
                "status": status.as_str(),
                "capacity_gb": capacity_gb,
                "models_on_disk": models_on_disk,
                "detail": detail,
            }),
            OutputEvent::PeerJoined { peer_id, label } => {
                json!({ "peer_id": peer_id, "label": label })
            }
            OutputEvent::PeerLeft { peer_id, reason } => {
                json!({ "peer_id": peer_id, "reason": reason })
            }
            OutputEvent::ModelQueued { model } => json!({ "model": model }),
            OutputEvent::ModelLoading { model, source } => {
                json!({ "model": model, "source": source })
            }
            OutputEvent::ModelLoaded { model, bytes } => json!({
                "model": model,
                "bytes": bytes,
            }),
            OutputEvent::ModelUnloading { model } => json!({ "model": model }),
            OutputEvent::ModelUnloaded { model } => json!({ "model": model }),
            OutputEvent::HostElected {
                model,
                host,
                role,
                capacity_gb,
            } => json!({ "model": model, "host": host, "role": role, "capacity_gb": capacity_gb }),
            OutputEvent::RpcServerStarting {
                port,
                device,
                log_path,
            }
            | OutputEvent::RpcReady {
                port,
                device,
                log_path,
            } => json!({ "port": port, "device": device, "log_path": log_path }),
            OutputEvent::RpcStartupFailed {
                port,
                device,
                log_path,
                detail,
            } => json!({
                "port": port,
                "device": device,
                "log_path": log_path,
                "detail": detail,
            }),
            OutputEvent::LlamaStarting {
                model,
                http_port,
                ctx_size,
                log_path,
            } => json!({
                "model": model,
                "http_port": http_port,
                "ctx_size": ctx_size,
                "log_path": log_path,
            }),
            OutputEvent::LlamaReady {
                model,
                port,
                ctx_size,
                log_path,
            } => json!({
                "model": model,
                "port": port,
                "ctx_size": ctx_size,
                "log_path": log_path,
            }),
            OutputEvent::LlamaStartupFailed {
                model,
                http_port,
                ctx_size,
                log_path,
                detail,
            } => json!({
                "model": model,
                "http_port": http_port,
                "ctx_size": ctx_size,
                "log_path": log_path,
                "detail": detail,
            }),
            OutputEvent::ModelReady {
                model,
                internal_port,
                role,
            } => json!({
                "model": model,
                "port": internal_port,
                "internal_port": internal_port,
                "role": role,
            }),
            OutputEvent::MultiModelMode { count, models } => {
                json!({ "count": count, "models": models })
            }
            OutputEvent::WebserverStarting { url }
            | OutputEvent::WebserverReady { url }
            | OutputEvent::ApiStarting { url }
            | OutputEvent::ApiReady { url } => json!({ "url": url }),
            OutputEvent::RuntimeReady {
                api_url,
                console_url,
                api_port,
                console_port,
                models_count,
                pi_command,
                goose_command,
            } => json!({
                "api_url": api_url,
                "console_url": console_url,
                "api_port": api_port,
                "console_port": console_port,
                "models_count": models_count,
                "pi_command": pi_command,
                "goose_command": goose_command,
            }),
            OutputEvent::ModelDownloadProgress {
                label,
                file,
                downloaded_bytes,
                total_bytes,
                status,
            } => json!({
                "label": label,
                "file": file,
                "downloaded_bytes": downloaded_bytes,
                "total_bytes": total_bytes,
                "status": status.as_str(),
            }),
            OutputEvent::RequestRouted { model, target } => {
                json!({ "model": model, "target": target })
            }
            OutputEvent::Warning { message, context } => {
                json!({ "warning": message, "context": context })
            }
            OutputEvent::Error { message, context } => {
                classified_error_json("error", message, context.as_deref())
            }
            OutputEvent::Fatal { message, context } => {
                classified_error_json("fatal", message, context.as_deref())
            }
            OutputEvent::ShutdownRequested { signal } => json!({ "signal": signal }),
            OutputEvent::Shutdown { reason } => json!({ "reason": reason }),
            OutputEvent::LlamaNativeLog { params, .. } => {
                let mut map = Map::new();
                for (key, value) in params {
                    map.insert(key.clone(), value.clone());
                }
                Value::Object(map)
            }
        };

        match value {
            Value::Object(map) => map,
            _ => Map::new(),
        }
    }
}

pub(super) fn classified_error_json(field: &str, message: &str, context: Option<&str>) -> Value {
    json!({
        field: message,
        "context": context,
        "error_type": classify_error_type(message, context),
    })
}

pub(super) fn format_message_with_params(message: &str, params: &[(String, Value)]) -> String {
    if params.is_empty() {
        return message.to_string();
    }
    let mut rendered = message.to_string();
    for (key, value) in params {
        rendered.push_str("\n  ↳ ");
        rendered.push_str(key);
        rendered.push('=');
        rendered.push_str(&format_json_scalar(value));
    }
    rendered
}

pub(super) fn format_json_scalar(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(v) => v.to_string(),
        Value::Number(v) => v.to_string(),
        Value::String(v) => v.clone(),
        _ => value.to_string(),
    }
}

pub(in crate::output) fn format_model_download_progress_message(
    label: &str,
    file: Option<&str>,
    downloaded_bytes: Option<u64>,
    total_bytes: Option<u64>,
    status: &ModelProgressStatus,
) -> String {
    let target = file.unwrap_or(label);
    if let Some(model) = label.strip_prefix("parts::") {
        return match status {
            ModelProgressStatus::Ensuring => format!("ensuring model parts for {model}"),
            ModelProgressStatus::Downloading => match (downloaded_bytes, total_bytes) {
                (Some(completed), Some(total)) if total > 0 => {
                    format!("downloading model parts for {model} {completed}/{total}")
                }
                _ => format!("downloading model parts for {model}"),
            },
            ModelProgressStatus::Ready => format!("model parts ready for {model}"),
        };
    }
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

pub(super) fn format_display_bytes(bytes: u64) -> String {
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

pub(super) fn format_role_active(role: &str) -> String {
    format!("{role} active")
}

pub(super) fn append_models_on_disk(line: &mut String, models_on_disk: Option<&[String]>) {
    let Some(models_on_disk) = models_on_disk else {
        return;
    };
    if !models_on_disk.is_empty() {
        line.push_str(&format!(" models={}", models_on_disk.join(", ")));
    }
}

pub(super) fn format_model_size(bytes: u64) -> String {
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

pub trait Formatter: Send {
    fn format(&mut self, event: &OutputEvent) -> io::Result<String>;
}

#[derive(Default)]
pub struct DashboardFormatter {
    pub(in crate::output) state: DashboardState,
}

impl Formatter for DashboardFormatter {
    fn format(&mut self, event: &OutputEvent) -> io::Result<String> {
        self.state
            .reduce(DashboardAction::OutputEvent(event.clone()));
        Ok(render_dashboard_text(&self.state))
    }
}

#[derive(Default)]
pub struct InteractiveDashboardFormatter {
    pub(in crate::output) state: DashboardState,
    pub(in crate::output) terminal: Option<TuiTerminal>,
    pub(in crate::output) terminal_active: bool,
    pub(in crate::output) tui_entered: Arc<AtomicBool>,
    pub(in crate::output) panic_restored: Arc<AtomicBool>,
    pub(in crate::output) dirty: bool,
}

impl InteractiveDashboardFormatter {
    pub(super) fn with_tui_state(
        tui_entered: Arc<AtomicBool>,
        panic_restored: Arc<AtomicBool>,
    ) -> Self {
        Self {
            tui_entered,
            panic_restored,
            ..Self::default()
        }
    }

    #[cfg(test)]
    pub(super) fn tui_entered(&self) -> bool {
        self.tui_entered.load(Ordering::Acquire)
    }

    pub(super) fn panic_restored(&self) -> bool {
        self.panic_restored.load(Ordering::Acquire)
    }

    pub(super) fn mark_panic_restored(&mut self) {
        self.terminal = None;
        self.terminal_active = false;
        self.dirty = false;
        self.tui_entered.store(false, Ordering::Release);
        self.panic_restored.store(true, Ordering::Release);
    }

    pub(super) fn handle_output_event(
        &mut self,
        event: &OutputEvent,
    ) -> io::Result<Option<String>> {
        if self.panic_restored() {
            return Ok(None);
        }
        self.state
            .reduce(DashboardAction::OutputEvent(event.clone()));
        if self.terminal_active {
            self.dirty = true;
            Ok(None)
        } else {
            Ok(Some(format!("{}\n", event.pretty_text())))
        }
    }

    pub(super) fn handle_snapshot(&mut self, snapshot: DashboardSnapshot) {
        if self.panic_restored() {
            return;
        }
        self.state
            .reduce(DashboardAction::SnapshotUpdated(snapshot));
        if self.terminal_active {
            self.dirty = true;
        }
    }

    pub(super) fn handle_tui_event(&mut self, event: TuiEvent) -> TuiControlFlow {
        if self.panic_restored() {
            return TuiControlFlow::Continue;
        }
        let control = self.state.apply_tui_event(event);
        if self.terminal_active {
            self.dirty = true;
        }
        control
    }

    pub(super) fn enter_terminal(&mut self) -> io::Result<()> {
        if self.panic_restored() {
            return Ok(());
        }
        if self.terminal_active {
            return Ok(());
        }
        write_tui_enter()?;
        self.mark_terminal_escape_written();
        let backend = CrosstermBackend::new(io::stderr());
        let mut terminal = Terminal::new(backend).map_err(io::Error::other)?;
        terminal.hide_cursor().map_err(io::Error::other)?;
        self.terminal = Some(terminal);
        Ok(())
    }

    pub(super) fn mark_terminal_escape_written(&mut self) {
        // From this point on, a later setup failure still needs normal TUI
        // cleanup: the terminal may already be in alternate-screen/raw-input
        // state even if ratatui terminal construction or cursor hiding fails.
        self.terminal_active = true;
        self.tui_entered.store(true, Ordering::Release);
        self.dirty = true;
    }

    pub(super) fn exit_terminal(&mut self) -> io::Result<()> {
        if !self.terminal_active {
            return Ok(());
        }
        if let Some(mut terminal) = self.terminal.take() {
            terminal.show_cursor().map_err(io::Error::other)?;
        }
        self.terminal_active = false;
        self.dirty = false;
        let result = write_tui_exit();
        if result.is_ok() {
            self.tui_entered.store(false, Ordering::Release);
        }
        result
    }

    pub(in crate::output) fn render_if_dirty(&mut self) -> io::Result<bool> {
        if self.panic_restored() {
            return Ok(false);
        }
        if self
            .state
            .clear_expired_join_token_copy_status(Instant::now())
            && self.terminal_active
        {
            self.dirty = true;
        }
        if !self.terminal_active || !self.dirty {
            return Ok(false);
        }
        let (columns, rows) = crossterm::terminal::size().unwrap_or((120, 40));
        self.state
            .reduce(DashboardAction::Resize(dashboard_layout_for_terminal_size(
                columns, rows,
            )));
        let terminal = self.terminal.as_mut().ok_or_else(|| {
            io::Error::other("pretty TUI terminal missing while terminal mode is active")
        })?;
        draw_tui_dashboard_with_terminal(terminal, &self.state)?;
        self.dirty = false;
        Ok(true)
    }
}

impl Formatter for InteractiveDashboardFormatter {
    fn format(&mut self, event: &OutputEvent) -> io::Result<String> {
        Ok(self.handle_output_event(event)?.unwrap_or_default())
    }
}

pub struct JsonFormatter;

impl Formatter for JsonFormatter {
    fn format(&mut self, event: &OutputEvent) -> io::Result<String> {
        let mut record = Map::new();
        record.insert(
            "timestamp".to_string(),
            Value::String(Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true)),
        );
        record.insert(
            "level".to_string(),
            Value::String(event.level().as_str().to_string()),
        );
        record.insert(
            "event".to_string(),
            Value::String(event.event_name().to_string()),
        );
        record.extend(event.json_fields());
        record.insert("message".to_string(), Value::String(event.message()));
        serde_json::to_string(&Value::Object(record))
            .map(|line| format!("{line}\n"))
            .map_err(io::Error::other)
    }
}

pub struct PrettyFormatter;

impl Formatter for PrettyFormatter {
    fn format(&mut self, event: &OutputEvent) -> io::Result<String> {
        Ok(format!("{}\n", event.pretty_text()))
    }
}

pub(super) enum FormatterSelection {
    InteractiveDashboard(InteractiveDashboardFormatter),
    DashboardFallback(DashboardFormatter),
    Plain(PrettyFormatter),
    Json(JsonFormatter),
}

impl FormatterSelection {
    #[cfg(test)]
    pub(super) fn kind(&self) -> &'static str {
        match self {
            Self::InteractiveDashboard(_) => "interactive_dashboard",
            Self::DashboardFallback(_) => "pretty_fallback",
            Self::Plain(_) => "plain",
            Self::Json(_) => "json",
        }
    }

    fn mode(&self) -> LogFormat {
        match self {
            Self::InteractiveDashboard(_) | Self::DashboardFallback(_) | Self::Plain(_) => {
                LogFormat::Pretty
            }
            Self::Json(_) => LogFormat::Json,
        }
    }

    pub(super) fn is_interactive_dashboard(&self) -> bool {
        matches!(self, Self::InteractiveDashboard(_))
    }

    pub(super) fn handle_output_event(&mut self, event: &OutputEvent) -> io::Result<()> {
        match self {
            Self::InteractiveDashboard(formatter) => {
                if let Some(rendered) = formatter.handle_output_event(event)? {
                    write_rendered_output(LogFormat::Pretty, &rendered)?;
                }
                Ok(())
            }
            _ => {
                let rendered = self.format(event)?;
                write_rendered_output(self.mode(), &rendered)
            }
        }
    }

    fn enter_tui(&mut self) -> io::Result<()> {
        match self {
            Self::InteractiveDashboard(formatter) => formatter.enter_terminal(),
            _ => Ok(()),
        }
    }

    fn exit_tui(&mut self) -> io::Result<()> {
        match self {
            Self::InteractiveDashboard(formatter) => formatter.exit_terminal(),
            _ => Ok(()),
        }
    }

    pub(super) fn handle_tui_event(&mut self, event: TuiEvent) -> TuiControlFlow {
        match self {
            Self::InteractiveDashboard(formatter) => formatter.handle_tui_event(event),
            _ => TuiControlFlow::Continue,
        }
    }

    pub(super) fn handle_tui_snapshot(&mut self, snapshot: DashboardSnapshot) {
        if let Self::InteractiveDashboard(formatter) = self {
            formatter.handle_snapshot(snapshot);
        }
    }

    pub(super) fn mark_panic_restored(&mut self) {
        if let Self::InteractiveDashboard(formatter) = self {
            formatter.mark_panic_restored();
        }
    }

    fn render_interactive_if_dirty(&mut self) -> io::Result<bool> {
        match self {
            Self::InteractiveDashboard(formatter) => formatter.render_if_dirty(),
            _ => Ok(false),
        }
    }

    pub(super) fn writes_ready_prompt(&self) -> bool {
        matches!(self, Self::DashboardFallback(_))
    }
}

impl Formatter for FormatterSelection {
    fn format(&mut self, event: &OutputEvent) -> io::Result<String> {
        match self {
            Self::InteractiveDashboard(formatter) => formatter.format(event),
            Self::DashboardFallback(formatter) => formatter.format(event),
            Self::Plain(formatter) => formatter.format(event),
            Self::Json(formatter) => formatter.format(event),
        }
    }
}

#[cfg(test)]
pub(in crate::output) fn select_formatter(
    mode: LogFormat,
    console_session_mode: ConsoleSessionMode,
) -> FormatterSelection {
    select_formatter_with_tui_state(
        mode,
        console_session_mode,
        Arc::new(AtomicBool::new(false)),
        Arc::new(AtomicBool::new(false)),
    )
}

pub(in crate::output) fn select_formatter_with_tui_state(
    mode: LogFormat,
    console_session_mode: ConsoleSessionMode,
    tui_entered: Arc<AtomicBool>,
    panic_restored: Arc<AtomicBool>,
) -> FormatterSelection {
    match mode {
        LogFormat::Pretty => match console_session_mode {
            ConsoleSessionMode::InteractiveDashboard => FormatterSelection::InteractiveDashboard(
                InteractiveDashboardFormatter::with_tui_state(tui_entered, panic_restored),
            ),
            ConsoleSessionMode::Fallback => {
                FormatterSelection::DashboardFallback(DashboardFormatter::default())
            }
            ConsoleSessionMode::None => FormatterSelection::Plain(PrettyFormatter),
        },
        LogFormat::Json => FormatterSelection::Json(JsonFormatter),
    }
}

pub(super) struct OutputManagerState {
    pub(in crate::output) tx: tokio::sync::mpsc::UnboundedSender<OutputCommand>,
    pub(in crate::output) ready_prompt_active: Arc<AtomicBool>,
    pub(in crate::output) tui_entered: Arc<AtomicBool>,
    pub(in crate::output) panic_restored: Arc<AtomicBool>,
    pub(in crate::output) mode: LogFormat,
    pub(in crate::output) console_session_mode: Option<ConsoleSessionMode>,
    pub(in crate::output) dashboard_snapshot_provider:
        Arc<RwLock<Option<Arc<dyn DashboardSnapshotProvider>>>>,
}

pub struct OutputManager {
    pub(in crate::output) state: RwLock<OutputManagerState>,
}

pub(super) struct OutputManagerSink {
    pub(in crate::output) output_manager: &'static OutputManager,
}

impl OutputManagerSink {
    pub(super) fn new(output_manager: &'static OutputManager) -> Self {
        Self { output_manager }
    }
}

impl OutputSink for OutputManagerSink {
    fn emit_event(&self, event: OutputEvent) -> io::Result<()> {
        self.output_manager.emit_event(event)
    }

    fn schedule_ready_prompt(&self) -> io::Result<()> {
        self.output_manager.schedule_ready_prompt()
    }

    fn write_ready_prompt(&self) -> io::Result<()> {
        self.output_manager.write_ready_prompt()
    }

    fn ready_prompt_active(&self) -> bool {
        self.output_manager.ready_prompt_active()
    }

    fn flush(&self) -> OutputSinkFuture<'_, ()> {
        Box::pin(self.output_manager.flush())
    }

    fn mode(&self) -> LogFormat {
        self.output_manager.mode()
    }

    fn console_session_mode(&self) -> Option<ConsoleSessionMode> {
        self.output_manager.console_session_mode()
    }

    fn register_dashboard_snapshot_provider(&self, provider: Arc<dyn DashboardSnapshotProvider>) {
        self.output_manager
            .register_dashboard_snapshot_provider(provider);
    }

    fn enter_tui(&self) -> OutputSinkFuture<'_, ()> {
        Box::pin(self.output_manager.enter_tui())
    }

    fn exit_tui(&self) -> OutputSinkFuture<'_, ()> {
        Box::pin(self.output_manager.exit_tui())
    }

    fn dispatch_tui_event(&self, event: TuiEvent) -> OutputSinkFuture<'_, TuiControlFlow> {
        Box::pin(self.output_manager.dispatch_tui_event(event))
    }

    fn render_tui_if_dirty(&self) -> OutputSinkFuture<'_, bool> {
        Box::pin(self.output_manager.render_tui_if_dirty())
    }

    fn force_restore_tui_terminal(&self) -> io::Result<()> {
        force_restore_tui_terminal()
    }
}

pub(super) enum OutputCommand {
    Event(OutputEvent),
    ActivateReadyPrompt,
    Flush(tokio::sync::oneshot::Sender<io::Result<()>>),
    EnterTui(tokio::sync::oneshot::Sender<io::Result<()>>),
    ExitTui(tokio::sync::oneshot::Sender<io::Result<()>>),
    TuiEvent {
        event: TuiEvent,
        response: tokio::sync::oneshot::Sender<io::Result<TuiControlFlow>>,
    },
    RenderTui(tokio::sync::oneshot::Sender<io::Result<bool>>),
    PanicRestored,
}

pub(crate) static GLOBAL_OUTPUT_MANAGER: OnceLock<OutputManager> = OnceLock::new();

impl OutputManager {
    pub fn init_global(
        mode: LogFormat,
        console_session_mode: ConsoleSessionMode,
    ) -> &'static OutputManager {
        let output_manager = if let Some(output_manager) = GLOBAL_OUTPUT_MANAGER.get() {
            output_manager.reset(mode, console_session_mode);
            output_manager
        } else {
            GLOBAL_OUTPUT_MANAGER.get_or_init(|| Self::new(mode, console_session_mode))
        };
        mesh_llm_events::set_output_sink(Arc::new(OutputManagerSink::new(output_manager)));
        output_manager
    }

    pub fn global() -> &'static OutputManager {
        GLOBAL_OUTPUT_MANAGER
            .get()
            .expect("OutputManager::init_global must be called before OutputManager::global")
    }

    pub(super) fn new(mode: LogFormat, console_session_mode: ConsoleSessionMode) -> Self {
        Self {
            state: RwLock::new(Self::spawn_state(mode, console_session_mode)),
        }
    }

    pub(super) fn reset(&self, mode: LogFormat, console_session_mode: ConsoleSessionMode) {
        match self.state.write() {
            Ok(mut state) => {
                *state = Self::spawn_state(mode, console_session_mode);
            }
            Err(err) => {
                tracing::warn!("output manager state lock poisoned during reset: {err}");
            }
        }
    }

    fn spawn_state(
        mode: LogFormat,
        console_session_mode: ConsoleSessionMode,
    ) -> OutputManagerState {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<OutputCommand>();
        let ready_prompt_active = Arc::new(AtomicBool::new(false));
        let tui_entered = Arc::new(AtomicBool::new(false));
        let panic_restored = Arc::new(AtomicBool::new(false));
        let worker_prompt_active = ready_prompt_active.clone();
        let worker_tui_entered = tui_entered.clone();
        let worker_panic_restored = panic_restored.clone();
        let dashboard_snapshot_provider: Arc<RwLock<Option<Arc<dyn DashboardSnapshotProvider>>>> =
            Arc::new(RwLock::new(None));
        let worker_snapshot_provider = dashboard_snapshot_provider.clone();
        tokio::spawn(async move {
            let mut formatter = select_formatter_with_tui_state(
                mode,
                console_session_mode,
                worker_tui_entered,
                worker_panic_restored,
            );
            let mut redraw_tick = time::interval(PRETTY_TUI_REDRAW_INTERVAL);
            redraw_tick.set_missed_tick_behavior(MissedTickBehavior::Skip);
            let mut snapshot_tick = time::interval(PRETTY_TUI_SNAPSHOT_INTERVAL);
            snapshot_tick.set_missed_tick_behavior(MissedTickBehavior::Skip);
            let mut last_snapshot_at = Instant::now() - PRETTY_TUI_SNAPSHOT_INTERVAL;
            loop {
                tokio::select! {
                    maybe_command = rx.recv() => {
                        let Some(command) = maybe_command else {
                            if let Err(err) = formatter.exit_tui() {
                                tracing::warn!("interactive terminal cleanup failed: {err}");
                            }
                            break;
                        };
                        match command {
                            OutputCommand::Event(event) => {
                                if let Err(err) = formatter.handle_output_event(&event) {
                                    tracing::warn!("output write failed: {err}");
                                } else if matches!(mode, LogFormat::Pretty)
                                    && worker_prompt_active.load(Ordering::Acquire)
                                    && formatter.writes_ready_prompt()
                                    && let Err(err) = write_prompt() {
                                        tracing::warn!("interactive prompt write failed: {err}");
                                    }
                            }
                            OutputCommand::ActivateReadyPrompt => {
                                worker_prompt_active.store(true, Ordering::Release);
                                if matches!(mode, LogFormat::Pretty) && formatter.writes_ready_prompt()
                                    && let Err(err) = write_prompt() {
                                        tracing::warn!("interactive prompt write failed: {err}");
                                    }
                            }
                            OutputCommand::Flush(response) => {
                                let flush_result = if formatter.is_interactive_dashboard() {
                                    formatter.render_interactive_if_dirty().map(|_| ())
                                } else {
                                    Ok(())
                                };
                                let _ = response.send(flush_result);
                            }
                            OutputCommand::EnterTui(response) => {
                                let _ = response.send(formatter.enter_tui());
                            }
                            OutputCommand::ExitTui(response) => {
                                let _ = response.send(formatter.exit_tui());
                            }
                            OutputCommand::TuiEvent { event, response } => {
                                let _ = response.send(Ok(formatter.handle_tui_event(event)));
                            }
                            OutputCommand::RenderTui(response) => {
                                let _ = response.send(formatter.render_interactive_if_dirty());
                            }
                            OutputCommand::PanicRestored => {
                                formatter.mark_panic_restored();
                            }
                        }
                    }
                    _ = redraw_tick.tick(), if formatter.is_interactive_dashboard() => {
                        if let Err(err) = formatter.render_interactive_if_dirty() {
                            tracing::warn!("interactive dashboard redraw failed: {err}");
                        }
                    }
                    _ = snapshot_tick.tick(), if formatter.is_interactive_dashboard() => {
                        if last_snapshot_at.elapsed() < PRETTY_TUI_SNAPSHOT_INTERVAL {
                            continue;
                        }
                        let Some(provider) = worker_snapshot_provider
                            .read()
                            .ok()
                            .and_then(|slot| slot.clone()) else {
                            continue;
                        };
                        last_snapshot_at = Instant::now();
                        formatter.handle_tui_snapshot(provider.snapshot().await);
                    }
                }
            }
        });
        OutputManagerState {
            tx,
            ready_prompt_active,
            tui_entered,
            panic_restored,
            mode,
            console_session_mode: matches!(mode, LogFormat::Pretty).then_some(console_session_mode),
            dashboard_snapshot_provider,
        }
    }

    pub(super) fn command_tx(
        &self,
    ) -> io::Result<tokio::sync::mpsc::UnboundedSender<OutputCommand>> {
        self.state
            .read()
            .map(|state| state.tx.clone())
            .map_err(|err| io::Error::other(format!("output manager state lock poisoned: {err}")))
    }

    pub fn emit_event(&self, event: OutputEvent) -> io::Result<()> {
        self.command_tx()?
            .send(OutputCommand::Event(event))
            .map_err(|_| {
                io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "output manager worker unavailable",
                )
            })
    }

    pub fn schedule_ready_prompt(&self) -> io::Result<()> {
        self.command_tx()?
            .send(OutputCommand::ActivateReadyPrompt)
            .map_err(|_| {
                io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "output manager worker unavailable",
                )
            })
    }

    pub fn write_ready_prompt(&self) -> io::Result<()> {
        let (ready_prompt_active, mode, console_session_mode) = self
            .state
            .read()
            .map(|state| {
                (
                    state.ready_prompt_active.clone(),
                    state.mode,
                    state.console_session_mode,
                )
            })
            .map_err(|err| {
                io::Error::other(format!("output manager state lock poisoned: {err}"))
            })?;
        ready_prompt_active.store(true, Ordering::Release);
        if matches!(mode, LogFormat::Pretty)
            && !matches!(
                console_session_mode,
                Some(ConsoleSessionMode::InteractiveDashboard)
            )
        {
            write_prompt()
        } else {
            Ok(())
        }
    }

    pub fn ready_prompt_active(&self) -> bool {
        self.state
            .read()
            .map(|state| state.ready_prompt_active.load(Ordering::Acquire))
            .unwrap_or(false)
    }

    pub async fn flush(&self) -> io::Result<()> {
        let (response_tx, response_rx) = tokio::sync::oneshot::channel();
        self.command_tx()?
            .send(OutputCommand::Flush(response_tx))
            .map_err(|_| {
                io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "output manager worker unavailable",
                )
            })?;
        response_rx.await.map_err(|_| {
            io::Error::new(
                io::ErrorKind::BrokenPipe,
                "output manager worker unavailable",
            )
        })?
    }

    pub fn mode(&self) -> LogFormat {
        self.state
            .read()
            .map(|state| state.mode)
            .unwrap_or(LogFormat::Pretty)
    }

    pub fn console_session_mode(&self) -> Option<ConsoleSessionMode> {
        self.state
            .read()
            .map(|state| state.console_session_mode)
            .unwrap_or(None)
    }

    pub(super) fn tui_entered(&self) -> bool {
        self.state
            .read()
            .map(|state| state.tui_entered.load(Ordering::Acquire))
            .unwrap_or(false)
    }

    pub(super) fn mark_panic_restored(&self) {
        let tx = match self.state.read() {
            Ok(state) => {
                state.panic_restored.store(true, Ordering::Release);
                state.tui_entered.store(false, Ordering::Release);
                state.tx.clone()
            }
            Err(err) => {
                tracing::warn!("output manager state lock poisoned during panic restore: {err}");
                return;
            }
        };
        let _ = tx.send(OutputCommand::PanicRestored);
    }

    pub fn register_dashboard_snapshot_provider(
        &self,
        provider: Arc<dyn DashboardSnapshotProvider>,
    ) {
        let dashboard_snapshot_provider = match self.state.read() {
            Ok(state) if matches!(state.mode, LogFormat::Pretty) => {
                state.dashboard_snapshot_provider.clone()
            }
            _ => return,
        };

        if let Ok(mut slot) = dashboard_snapshot_provider.write() {
            *slot = Some(provider);
        }
    }

    #[allow(dead_code)]
    pub async fn dashboard_snapshot(&self) -> Option<DashboardSnapshot> {
        let dashboard_snapshot_provider = match self.state.read() {
            Ok(state) if matches!(state.mode, LogFormat::Pretty) => {
                state.dashboard_snapshot_provider.clone()
            }
            _ => return None,
        };

        let provider = dashboard_snapshot_provider
            .read()
            .ok()
            .and_then(|slot| slot.clone())?;
        Some(provider.snapshot().await)
    }

    pub async fn enter_tui(&self) -> io::Result<()> {
        let (response_tx, response_rx) = tokio::sync::oneshot::channel();
        self.command_tx()?
            .send(OutputCommand::EnterTui(response_tx))
            .map_err(|_| {
                io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "output manager worker unavailable",
                )
            })?;
        response_rx.await.map_err(|_| {
            io::Error::new(
                io::ErrorKind::BrokenPipe,
                "output manager worker unavailable",
            )
        })?
    }

    pub async fn exit_tui(&self) -> io::Result<()> {
        let (response_tx, response_rx) = tokio::sync::oneshot::channel();
        self.command_tx()?
            .send(OutputCommand::ExitTui(response_tx))
            .map_err(|_| {
                io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "output manager worker unavailable",
                )
            })?;
        response_rx.await.map_err(|_| {
            io::Error::new(
                io::ErrorKind::BrokenPipe,
                "output manager worker unavailable",
            )
        })?
    }

    pub async fn dispatch_tui_event(&self, event: TuiEvent) -> io::Result<TuiControlFlow> {
        let (response_tx, response_rx) = tokio::sync::oneshot::channel();
        self.command_tx()?
            .send(OutputCommand::TuiEvent {
                event,
                response: response_tx,
            })
            .map_err(|_| {
                io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "output manager worker unavailable",
                )
            })?;
        response_rx.await.map_err(|_| {
            io::Error::new(
                io::ErrorKind::BrokenPipe,
                "output manager worker unavailable",
            )
        })?
    }

    pub async fn render_tui_if_dirty(&self) -> io::Result<bool> {
        let (response_tx, response_rx) = tokio::sync::oneshot::channel();
        self.command_tx()?
            .send(OutputCommand::RenderTui(response_tx))
            .map_err(|_| {
                io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "output manager worker unavailable",
                )
            })?;
        response_rx.await.map_err(|_| {
            io::Error::new(
                io::ErrorKind::BrokenPipe,
                "output manager worker unavailable",
            )
        })?
    }
}

pub(super) fn write_rendered_output(mode: LogFormat, rendered: &str) -> io::Result<()> {
    let mut stdout = io::stdout().lock();
    let mut stderr = io::stderr().lock();
    write_rendered_output_to_writers(mode, rendered, &mut stdout, &mut stderr)
}

pub(in crate::output) fn write_rendered_output_to_writers<StdoutWriter, StderrWriter>(
    mode: LogFormat,
    rendered: &str,
    stdout: &mut StdoutWriter,
    stderr: &mut StderrWriter,
) -> io::Result<()>
where
    StdoutWriter: Write,
    StderrWriter: Write,
{
    match mode {
        LogFormat::Pretty => {
            stderr.write_all(rendered.as_bytes())?;
            if !rendered.ends_with('\n') {
                stderr.write_all(b"\n")?;
            }
            stderr.flush()
        }
        LogFormat::Json => {
            stdout.write_all(rendered.as_bytes())?;
            if !rendered.ends_with('\n') {
                stdout.write_all(b"\n")?;
            }
            stdout.flush()
        }
    }
}

pub(super) fn classify_error_type(message: &str, context: Option<&str>) -> &'static str {
    if message.starts_with("GGUF file not found:") {
        "missing_gguf"
    } else if message.starts_with("Failed to bind to port")
        || context
            .map(|value| value.contains("Address already in use"))
            .unwrap_or(false)
    {
        "bind_failed"
    } else {
        "runtime_error"
    }
}

pub(crate) fn write_emergency_event(event: &OutputEvent) -> io::Result<()> {
    let mode = GLOBAL_OUTPUT_MANAGER
        .get()
        .map(OutputManager::mode)
        .unwrap_or(LogFormat::Pretty);
    let rendered = render_emergency_event(mode, event)?;
    write_rendered_output(mode, &rendered)
}

pub(super) fn render_emergency_event(mode: LogFormat, event: &OutputEvent) -> io::Result<String> {
    match mode {
        LogFormat::Pretty => PrettyFormatter.format(event),
        LogFormat::Json => JsonFormatter.format(event),
    }
}

pub fn json_mode_enabled() -> bool {
    GLOBAL_OUTPUT_MANAGER
        .get()
        .map(|output_manager| matches!(output_manager.mode(), LogFormat::Json))
        .unwrap_or(false)
}

pub(super) fn write_prompt() -> io::Result<()> {
    let mut stderr = io::stderr().lock();
    stderr.write_all(b"> ")?;
    stderr.flush()
}

pub(super) fn dashboard_layout_for_terminal_size(columns: u16, rows: u16) -> DashboardLayoutState {
    let footer_rows = 2usize;
    let join_token_rows = usize::from(PRETTY_TUI_JOIN_TOKEN_PANEL_HEIGHT);
    let requests_rows = 6usize;
    let requests_band_rows = requests_rows + 2;
    // Cap the dashboard height so it stays compact while leaving enough
    // room for two full-height loaded model cards.
    let max_dashboard_rows = usize::from(rows).min(33);
    let narrow_width_penalty = usize::from(columns < PRETTY_TUI_MIN_DASHBOARD_WIDTH);
    let main_body_rows = max_dashboard_rows
        .saturating_sub(footer_rows + join_token_rows + requests_band_rows)
        .saturating_sub(narrow_width_penalty)
        .max(5);
    let process_body_rows = main_body_rows.saturating_sub(6).max(2);
    let llama_rows = ((process_body_rows.saturating_add(1)) / 3).max(1);
    let webserver_rows = process_body_rows.saturating_sub(llama_rows).max(1);
    let events_rows = main_body_rows.saturating_sub(2).max(1);
    let models_rows = main_body_rows.saturating_sub(2).max(1);
    DashboardLayoutState::new(
        events_rows,
        llama_rows,
        webserver_rows,
        models_rows,
        requests_rows,
    )
}

pub(super) fn write_tui_enter() -> io::Result<()> {
    let mut stderr = io::stderr().lock();
    write_tui_enter_to_writer(&mut stderr)
}

pub(super) fn write_tui_exit() -> io::Result<()> {
    let mut stderr = io::stderr().lock();
    write_tui_exit_to_writer(&mut stderr)
}

#[cfg(test)]
pub(super) fn write_tui_redraw_start_to_writer<W: Write>(writer: &mut W) -> io::Result<()> {
    execute!(writer, Hide, MoveTo(0, 0)).map_err(io::Error::other)
}

pub fn force_restore_tui_terminal() -> io::Result<()> {
    // Emergency restore path for panic/unwind and failed worker cleanup. This
    // intentionally bypasses the OutputManager so terminal recovery still has a
    // chance if its worker is wedged; SIGKILL cannot be recovered in-process.
    write_tui_exit()
}

pub fn force_restore_tui_after_panic() {
    let Some(output_manager) = GLOBAL_OUTPUT_MANAGER.get() else {
        return;
    };
    if !output_manager.tui_entered() {
        return;
    }

    output_manager.mark_panic_restored();
    let _ = force_restore_tui_terminal();
    let _ = disable_raw_mode();
}

pub(super) fn write_tui_enter_to_writer<W: Write>(writer: &mut W) -> io::Result<()> {
    execute!(
        writer,
        EnterAlternateScreen,
        MoveTo(0, 0),
        Clear(ClearType::All),
        Hide
    )
    .map_err(io::Error::other)
}

pub(super) fn write_tui_exit_to_writer<W: Write>(writer: &mut W) -> io::Result<()> {
    execute!(
        writer,
        Show,
        LeaveAlternateScreen,
        MoveTo(0, 0),
        Clear(ClearType::All)
    )
    .map_err(io::Error::other)
}

#[cfg(test)]
pub(super) fn write_tui_frame_to_writer<W: Write>(
    writer: &mut W,
    rendered: &str,
) -> io::Result<()> {
    execute!(writer, MoveTo(0, 0), Clear(ClearType::All)).map_err(io::Error::other)?;
    writer.write_all(rendered.as_bytes())?;
    if !rendered.ends_with('\n') {
        writer.write_all(b"\n")?;
    }
    writer.flush()
}

pub fn emit_event(event: OutputEvent) -> io::Result<()> {
    match GLOBAL_OUTPUT_MANAGER.get() {
        Some(output_manager) => output_manager.emit_event(event),
        None => Ok(()),
    }
}

pub async fn flush_output() -> io::Result<()> {
    match GLOBAL_OUTPUT_MANAGER.get() {
        Some(output_manager) => output_manager.flush().await,
        None => Ok(()),
    }
}

pub fn interactive_tui_active() -> bool {
    GLOBAL_OUTPUT_MANAGER.get().is_some_and(|output_manager| {
        matches!(output_manager.mode(), LogFormat::Pretty)
            && matches!(
                output_manager.console_session_mode(),
                Some(ConsoleSessionMode::InteractiveDashboard)
            )
    })
}

#[cfg(test)]
impl DashboardState {
    pub fn with_mesh_event_limit(mesh_event_limit: usize) -> Self {
        Self {
            mesh_event_limit: mesh_event_limit.max(1),
            ..Self::default()
        }
    }
}

#[cfg(test)]
impl DashboardFormatter {
    pub fn with_state(state: DashboardState) -> Self {
        Self { state }
    }
}
