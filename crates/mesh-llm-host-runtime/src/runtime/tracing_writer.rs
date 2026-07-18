thread_local! {
    static ROUTING_TRACING_STDERR: Cell<bool> = const { Cell::new(false) };
}

use anyhow::Result;
use mesh_llm_events::{OutputEvent, emit_event, flush_output};
use std::cell::Cell;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
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

pub(super) fn runtime_tracing_subscriber()
-> Result<impl tracing::Subscriber + Send + Sync + 'static> {
    Ok(tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("mesh_inference=info".parse()?)
                .add_directive("nostr_relay_pool=off".parse()?)
                .add_directive("nostr_sdk=warn".parse()?)
                .add_directive("noq_proto::connection=warn".parse()?),
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
        eprintln!(
            "mesh-llm embedded runtime using existing tracing subscriber; could not install mesh-llm subscriber: {err}"
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
