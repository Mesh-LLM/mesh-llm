thread_local! {
    static ROUTING_TRACING_STDERR: Cell<bool> = const { Cell::new(false) };
}

use anyhow::Result;
use mesh_llm_events::{
    OutputEvent,
    audit::{AuditLevel, AuditLogFormat, FileAuditSink, FileAuditSinkConfig, set_audit_sink},
    emit_event, flush_output,
};
use std::cell::Cell;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing_subscriber::fmt::MakeWriter;

pub(super) struct MeshTracingStderr;

pub(super) struct MeshTracingStderrWriter {
    level: tracing::Level,
    target: String,
    buffer: Vec<u8>,
}

impl MeshTracingStderrWriter {
    fn new(level: tracing::Level, target: impl Into<String>) -> Self {
        Self {
            level,
            target: target.into(),
            buffer: Vec::new(),
        }
    }

    fn drain_complete_lines(&mut self) -> io::Result<()> {
        while let Some(newline_index) = self.buffer.iter().position(|byte| *byte == b'\n') {
            let line = self.buffer.drain(..=newline_index).collect::<Vec<_>>();
            self.write_line(&line)?;
        }
        Ok(())
    }

    fn drain_remainder(&mut self) -> io::Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }

        let line = std::mem::take(&mut self.buffer);
        self.write_line(&line)
    }

    fn write_line(&self, line: &[u8]) -> io::Result<()> {
        let message = String::from_utf8_lossy(line)
            .trim_end_matches(['\r', '\n'])
            .to_string();
        if message.trim().is_empty() {
            return Ok(());
        }

        if self.should_route_to_dashboard() {
            return self.route_line_to_dashboard(message);
        }

        write_stderr_line(&message)
    }

    fn should_route_to_dashboard(&self) -> bool {
        !self.target.starts_with("mesh_llm_tui::output")
            && !self.target.starts_with("mesh_llm_events")
            && mesh_llm_events::interactive_tui_active()
    }

    fn route_line_to_dashboard(&self, message: String) -> io::Result<()> {
        ROUTING_TRACING_STDERR.with(|routing| {
            if routing.get() {
                return write_stderr_line(&message);
            }

            routing.set(true);
            let dashboard_message = strip_ansi_escape_sequences(&message);
            let event = self.dashboard_event_for_message(&dashboard_message);
            let result =
                mesh_llm_events::emit_event(event).or_else(|_| write_stderr_line(&message));
            routing.set(false);
            result
        })
    }

    fn dashboard_event_for_message(&self, message: &str) -> OutputEvent {
        let (message, context) = normalize_tracing_message(&self.target, message);
        match self.level {
            tracing::Level::ERROR => OutputEvent::Error { message, context },
            tracing::Level::WARN => OutputEvent::Warning { message, context },
            _ => OutputEvent::Info { message, context },
        }
    }
}

pub(super) fn normalize_tracing_message(target: &str, message: &str) -> (String, Option<String>) {
    let message = message.trim().to_string();
    if target.starts_with("noq_proto") {
        return (
            normalize_noq_proto_message(target, &message),
            Some("transport".to_string()),
        );
    }

    (message, Some("stderr".to_string()))
}

pub(super) fn normalize_noq_proto_message(target: &str, message: &str) -> String {
    let without_prefix = message
        .find(target)
        .and_then(|target_index| {
            message[target_index + target.len()..]
                .find(':')
                .map(|colon_index| message[target_index + target.len() + colon_index + 1..].trim())
        })
        .unwrap_or(message)
        .trim();
    format_noq_proto_fields(without_prefix)
}

pub(super) fn format_noq_proto_fields(message: &str) -> String {
    let Some(rest) = message.strip_prefix("err=") else {
        return message.to_string();
    };
    let Some((err, detail)) = rest.split_once(' ') else {
        return message.to_string();
    };
    if detail.trim().is_empty() {
        message.to_string()
    } else {
        format!("{} (err={err})", detail.trim())
    }
}

pub(super) fn strip_ansi_escape_sequences(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch != '\u{1b}' {
            output.push(ch);
            continue;
        }

        if matches!(chars.peek(), Some('[')) {
            chars.next();
            for code in chars.by_ref() {
                if ('@'..='~').contains(&code) {
                    break;
                }
            }
        }
    }

    output
}

impl Write for MeshTracingStderrWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.buffer.extend_from_slice(buf);
        self.drain_complete_lines()?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.drain_remainder()
    }
}

impl Drop for MeshTracingStderrWriter {
    fn drop(&mut self) {
        let _ = self.drain_remainder();
    }
}

impl<'writer> MakeWriter<'writer> for MeshTracingStderr {
    type Writer = MeshTracingStderrWriter;

    fn make_writer(&'writer self) -> Self::Writer {
        MeshTracingStderrWriter::new(tracing::Level::INFO, "tracing")
    }

    fn make_writer_for(&'writer self, meta: &tracing::Metadata<'_>) -> Self::Writer {
        MeshTracingStderrWriter::new(*meta.level(), meta.target())
    }
}

pub(super) fn write_stderr_line(message: &str) -> io::Result<()> {
    let mut stderr = io::stderr().lock();
    stderr.write_all(message.as_bytes())?;
    stderr.write_all(b"\n")?;
    stderr.flush()
}

pub(super) fn configure_skippy_native_logging(runtime_dir: Option<&Path>) -> Option<PathBuf> {
    let Some(runtime_dir) = runtime_dir else {
        suppress_skippy_native_logs(
            "suppressing skippy native logs without an instance runtime directory",
        );
        return None;
    };

    let log_dir = runtime_dir.join("logs");
    if let Err(err) = std::fs::create_dir_all(&log_dir) {
        warn_and_suppress_skippy_native_logs(
            &log_dir,
            &err,
            "failed to create skippy native log directory; suppressing native logs",
        );
        return None;
    }

    let native_log_path = log_dir.join("skippy-native.log");
    if let Err(err) = skippy_runtime::redirect_native_logs_to_file(&native_log_path) {
        warn_and_suppress_skippy_native_logs(
            &native_log_path,
            &err,
            "failed to redirect skippy native logs; suppressing native logs",
        );
        return None;
    }

    tracing::info!(
        path = %native_log_path.display(),
        "redirecting skippy native logs away from stdout"
    );
    Some(native_log_path)
}

pub(super) fn suppress_skippy_native_logs(message: &str) {
    skippy_runtime::suppress_native_logs();
    tracing::debug!("{message}");
}

pub(super) fn warn_and_suppress_skippy_native_logs<E: std::fmt::Display>(
    path: &Path,
    err: &E,
    message: &str,
) {
    tracing::warn!(path = %path.display(), error = %err, "{message}");
    skippy_runtime::suppress_native_logs();
}

pub(super) struct SkippyNativeLogForwardingGuard;

impl Drop for SkippyNativeLogForwardingGuard {
    fn drop(&mut self) {
        skippy_runtime::set_filtered_native_logs_enabled(false);
        skippy_runtime::unregister_filtered_native_logs();
    }
}

pub(super) fn native_log_parser_mode(
    mode: mesh_llm_config::LifecycleLogParserMode,
) -> skippy_runtime::NativeLogParserMode {
    match mode {
        mesh_llm_config::LifecycleLogParserMode::Auto => skippy_runtime::NativeLogParserMode::Auto,
        mesh_llm_config::LifecycleLogParserMode::Enabled => {
            skippy_runtime::NativeLogParserMode::Enabled
        }
        mesh_llm_config::LifecycleLogParserMode::Disabled => {
            skippy_runtime::NativeLogParserMode::Disabled
        }
    }
}

/// Without an installed runtime-scoped reporter (event system off, or the
/// native setter refused it) no structured family can reach the host, so the
/// parser fallback must stay available for every category.
pub(super) fn structured_event_capabilities(
    mut capabilities: skippy_runtime::CapabilityReport,
    event_system_off: bool,
    reporter_installed: bool,
) -> skippy_runtime::CapabilityReport {
    if event_system_off || !reporter_installed {
        capabilities.confirmed &= !skippy_ffi::FEATURE_RUNTIME_EVENT_REPORTER;
    }
    capabilities
}

/// Applies the parser policy and reports capability-probe health as normal
/// warnings, so probe problems stay visible whatever the parser mode is.
pub(super) fn configure_lifecycle_log_parser(
    mode: mesh_llm_config::LifecycleLogParserMode,
    capabilities: &skippy_runtime::CapabilityReport,
) -> skippy_runtime::NativeLogParserPolicy {
    for message in &capabilities.health_messages {
        let _ = emit_event(OutputEvent::Warning {
            message: message.clone(),
            context: Some("native runtime capability probe".to_string()),
        });
    }
    submit_capability_probe_warnings(&capabilities.health_messages);
    let policy =
        skippy_runtime::NativeLogParserPolicy::new(native_log_parser_mode(mode), capabilities);
    skippy_runtime::configure_native_log_parser(policy);
    policy
}

/// Mirror each capability-probe health message into the runtime-event
/// reducer as a `warning_raised` diagnostic, so `runtime_state.node.
/// diagnostics` shows a disabled native family. The probe's messages are
/// static text plus a family name and feature bit, never a path or an
/// identifier. They share one `unsupported_capability` correlation key, so
/// the active-warning view holds the latest probe message.
pub(super) fn submit_capability_probe_warnings(messages: &[String]) {
    use mesh_llm_runtime_event_contracts::{
        DiagnosticEventKind, DiagnosticFact, FactData, HumanSummary, OperationId, OperationScope,
        ReasonCode, RuntimeEventIngress, RuntimeFact,
    };

    let Some(engine) = crate::runtime_events::runtime_event_engine() else {
        return;
    };
    for message in messages {
        let fact = RuntimeFact::Diagnostic(DiagnosticFact::with_data(
            DiagnosticEventKind::WarningRaised,
            FactData {
                reason: Some(ReasonCode::UnsupportedCapability),
                summary: HumanSummary::new(message).ok(),
                ..FactData::default()
            },
        ));
        let _ = engine
            .unreserved_ingress(OperationScope::root_only(OperationId::new()))
            .try_submit(fact);
    }
}

pub(super) fn bridge_skippy_native_logs(
    mut native_log_rx: tokio::sync::mpsc::UnboundedReceiver<skippy_runtime::NativeLogEvent>,
) {
    tokio::spawn(async move {
        while let Some(event) = native_log_rx.recv().await {
            let _ = emit_event(OutputEvent::LlamaNativeLog {
                message: event.message,
                category: event.category,
                params: event.params,
            });
        }
    });
}

pub(super) async fn emit_shutdown(reason: Option<String>) {
    crate::system::backend::mark_runtime_shutting_down();
    let _ = emit_event(OutputEvent::Shutdown { reason });
    let _ = flush_output().await;
}

/// Initialize the audit logging sink based on configuration
pub(super) fn init_audit_logging(
    audit_log_path: Option<PathBuf>,
    audit_log_format: AuditLogFormat,
    audit_log_level: AuditLevel,
    max_file_size: u64,
    max_files: usize,
) -> Result<()> {
    if let Some(path) = audit_log_path {
        let config = FileAuditSinkConfig {
            path,
            max_file_size,
            max_files,
            min_level: audit_log_level,
            format: audit_log_format,
        };
        let sink = FileAuditSink::new(config)?;
        set_audit_sink(Arc::new(sink));
        tracing::info!("Audit logging initialized");
    }
    Ok(())
}

pub(super) fn runtime_tracing_subscriber()
-> Result<impl tracing::Subscriber + Send + Sync + 'static> {
    Ok(tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("mesh_inference=info".parse()?)
                .add_directive("nostr_relay_pool=off".parse()?)
                .add_directive("nostr_sdk=warn".parse()?)
                .add_directive("noq_proto::connection=warn".parse()?)
                .add_directive("skippy_server=warn".parse()?)
                .add_directive("mesh_native_serving_plugin_host=warn".parse()?)
                .add_directive("mesh_llm_runtime_install=warn".parse()?),
        )
        .with_writer(MeshTracingStderr)
        .finish())
}

pub(super) fn init_runtime_tracing() -> Result<()> {
    let subscriber = runtime_tracing_subscriber()?;
    tracing::subscriber::set_global_default(subscriber)
        .map_err(|err| anyhow::anyhow!("install runtime tracing subscriber: {err}"))
}

pub(super) fn init_embedded_runtime_tracing() -> Result<()> {
    let subscriber = runtime_tracing_subscriber()?;
    if let Err(err) = tracing::subscriber::set_global_default(subscriber) {
        // The host already had a subscriber, so this warning has somewhere to
        // go. Writing it raw would put it straight onto the dashboard frame.
        tracing::warn!(
            error = %err,
            "mesh-llm embedded runtime using existing tracing subscriber"
        );
    }
    Ok(())
}

pub(super) fn initialize_runtime_entrypoint() -> Result<()> {
    crate::system::backend::clear_runtime_shutting_down();
    init_runtime_tracing()?;
    Ok(())
}

pub(super) fn initialize_embedded_runtime_entrypoint() -> Result<()> {
    crate::system::backend::clear_runtime_shutting_down();
    init_embedded_runtime_tracing()
}

#[cfg(test)]
mod capability_probe_warning_tests {
    use super::submit_capability_probe_warnings;
    use crate::runtime_event_api::state_projection;
    use crate::runtime_events::engine::RuntimeEventEngine;
    use crate::runtime_events::{clear_runtime_event_engine, install_runtime_event_engine};

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn probe_health_messages_become_active_capability_warnings() {
        clear_runtime_event_engine();
        let engine = RuntimeEventEngine::new();
        install_runtime_event_engine(engine.clone());
        let message = "skippy capability probe: family 'kv_events' advertised feature bit \
                       0x200000000 but a required symbol is missing; disabling this family only";

        submit_capability_probe_warnings(&[message.to_string()]);
        engine.drain();

        let node =
            serde_json::to_value(&state_projection::build(&engine).node).expect("serializable");
        let warnings = node["diagnostics"]["active_warnings"]
            .as_array()
            .expect("active warnings");
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0]["reason_code"], "unsupported_capability");
        assert_eq!(warnings[0]["summary"], message);
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn no_engine_or_no_message_submits_nothing() {
        clear_runtime_event_engine();
        submit_capability_probe_warnings(&["ignored".to_string()]);

        let engine = RuntimeEventEngine::new();
        install_runtime_event_engine(engine.clone());
        submit_capability_probe_warnings(&[]);
        engine.drain();
        assert!(engine.replay().is_empty());
        clear_runtime_event_engine();
    }
}
