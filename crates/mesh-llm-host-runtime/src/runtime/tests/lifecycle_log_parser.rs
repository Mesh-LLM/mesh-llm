//! Lifecycle log parser policy and capability-probe warnings at startup.

use super::*;

const FULL_STRUCTURED_COVERAGE: u64 = skippy_ffi::FEATURE_RUNTIME_EVENT_REPORTER
    | skippy_ffi::FEATURE_RUNTIME_EVENTS
    | skippy_ffi::FEATURE_MODEL_LOAD_EVENTS_V2
    | skippy_ffi::FEATURE_KV_EVENTS
    | skippy_ffi::FEATURE_DEVICE_EVENTS;
const NATIVE_LOG_CATEGORIES: [&str; 5] = ["backend", "model", "memory", "kv_cache", "tokenizer"];

#[derive(Default)]
struct RecordingOutputSink {
    events: std::sync::Mutex<Vec<mesh_llm_events::OutputEvent>>,
}

impl mesh_llm_events::OutputSink for RecordingOutputSink {
    fn emit_event(&self, event: mesh_llm_events::OutputEvent) -> std::io::Result<()> {
        self.events
            .lock()
            .expect("recording sink mutex poisoned")
            .push(event);
        Ok(())
    }
}

struct ParserStateReset;

impl Drop for ParserStateReset {
    fn drop(&mut self) {
        mesh_llm_events::clear_output_sink();
        skippy_runtime::set_filtered_native_logs_enabled(false);
    }
}

fn capability_report(confirmed: u64, health: &[&str]) -> skippy_runtime::CapabilityReport {
    skippy_runtime::CapabilityReport {
        confirmed,
        health_messages: health.iter().map(|message| message.to_string()).collect(),
    }
}

#[test]
#[serial_test::serial]
fn parser_auto_with_full_coverage_forwards_nothing() {
    let _reset = ParserStateReset;

    let policy = configure_lifecycle_log_parser(
        mesh_llm_config::LifecycleLogParserMode::Auto,
        &capability_report(FULL_STRUCTURED_COVERAGE, &[]),
    );

    assert!(
        NATIVE_LOG_CATEGORIES
            .iter()
            .all(|category| !policy.forwards(category))
    );
}

#[test]
#[serial_test::serial]
fn parser_auto_keeps_fallback_when_event_system_is_off() {
    let _reset = ParserStateReset;
    let capabilities =
        structured_event_capabilities(capability_report(FULL_STRUCTURED_COVERAGE, &[]), true, true);

    let policy = configure_lifecycle_log_parser(
        mesh_llm_config::LifecycleLogParserMode::Auto,
        &capabilities,
    );

    assert!(
        NATIVE_LOG_CATEGORIES
            .iter()
            .all(|category| policy.forwards(category))
    );
}

#[test]
#[serial_test::serial]
fn parser_auto_keeps_fallback_when_reporter_install_was_refused() {
    let _reset = ParserStateReset;
    let capabilities = structured_event_capabilities(
        capability_report(FULL_STRUCTURED_COVERAGE, &[]),
        false,
        false,
    );

    let policy = configure_lifecycle_log_parser(
        mesh_llm_config::LifecycleLogParserMode::Auto,
        &capabilities,
    );

    assert!(
        NATIVE_LOG_CATEGORIES
            .iter()
            .all(|category| policy.forwards(category))
    );
}

#[test]
#[serial_test::serial]
fn probe_health_messages_reach_output_without_parser() {
    let sink = std::sync::Arc::new(RecordingOutputSink::default());
    let _reset = ParserStateReset;
    mesh_llm_events::set_output_sink(sink.clone());
    let health = "skippy capability probe: family 'kv_events' advertised feature bit 0x200000000 but a required symbol is missing; disabling this family only";

    configure_lifecycle_log_parser(
        mesh_llm_config::LifecycleLogParserMode::Disabled,
        &capability_report(0, &[health]),
    );

    let events = std::mem::take(&mut *sink.events.lock().expect("sink mutex"));
    let warnings: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            mesh_llm_events::OutputEvent::Warning { message, .. } => Some(message.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(warnings, vec![health]);
    assert!(
        events
            .iter()
            .all(|event| !matches!(event, mesh_llm_events::OutputEvent::LlamaNativeLog { .. }))
    );
}
