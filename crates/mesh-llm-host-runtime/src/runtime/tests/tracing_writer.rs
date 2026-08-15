use super::*;
use mesh_llm_events::{ConsoleSessionMode, LogFormat, OutputEvent, OutputSink};
use std::io::Write as _;
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct InteractiveTracingSink {
    events: Mutex<Vec<OutputEvent>>,
}

impl OutputSink for InteractiveTracingSink {
    fn emit_event(&self, event: OutputEvent) -> std::io::Result<()> {
        self.events.lock().expect("tracing sink lock").push(event);
        Ok(())
    }

    fn mode(&self) -> LogFormat {
        LogFormat::Pretty
    }

    fn console_session_mode(&self) -> Option<ConsoleSessionMode> {
        Some(ConsoleSessionMode::InteractiveDashboard)
    }
}

struct OutputSinkResetGuard;

impl Drop for OutputSinkResetGuard {
    fn drop(&mut self) {
        mesh_llm_events::clear_output_sink();
    }
}

#[test]
fn noq_proto_tracing_messages_use_transport_context() {
    let message = "2026-06-11T03:49:18.033043Z  WARN noq_proto::connection: err=LastOpenPath failed closing path";

    let (message, context) = normalize_tracing_message("noq_proto::connection", message);

    assert_eq!(message, "failed closing path (err=LastOpenPath)");
    assert_eq!(context.as_deref(), Some("transport"));
}

#[test]
fn routed_tracing_messages_strip_ansi_sequences() {
    let formatted = "\u{1b}[2m2026-06-11T03:49:18.033043Z\u{1b}[0m \u{1b}[33m WARN\u{1b}[0m";

    assert_eq!(
        strip_ansi_escape_sequences(formatted),
        "2026-06-11T03:49:18.033043Z  WARN"
    );
}

#[test]
fn non_proto_tracing_messages_keep_stderr_context() {
    let (message, context) = normalize_tracing_message("mesh_llm::runtime", "runtime warning");

    assert_eq!(message, "runtime warning");
    assert_eq!(context.as_deref(), Some("stderr"));
}

#[test]
#[serial_test::serial]
fn skippy_server_warning_routes_to_dashboard_while_tui_is_active() {
    let sink = Arc::new(InteractiveTracingSink::default());
    let _reset_guard = OutputSinkResetGuard;
    mesh_llm_events::set_output_sink(sink.clone());

    let mut writer = MeshTracingStderrWriter::new(
        tracing::Level::WARN,
        "skippy_server::kv_integration::config",
    );
    assert!(writer.should_route_to_dashboard());
    writer
        .write_all(b"KV disk tier unavailable; continuing without it\n")
        .expect("interactive tracing writer should route successfully");

    let events = sink.events.lock().expect("tracing sink lock");
    let [OutputEvent::Warning { message, context }] = events.as_slice() else {
        panic!("expected one dashboard warning event, got {events:?}");
    };
    assert!(message.contains("KV disk tier unavailable"));
    assert_eq!(context.as_deref(), Some("stderr"));
}
