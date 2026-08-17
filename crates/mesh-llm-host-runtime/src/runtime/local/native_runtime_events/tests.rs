use std::io;
use std::sync::{Arc, Mutex as StdMutex};

use mesh_llm_events::{OutputEvent, OutputSink, clear_output_sink, set_output_sink};

use super::*;

#[derive(Default)]
struct RecordingOutputSink {
    events: StdMutex<Vec<OutputEvent>>,
}

impl RecordingOutputSink {
    fn take_events(&self) -> Vec<OutputEvent> {
        std::mem::take(&mut *self.events.lock().expect("recording sink mutex poisoned"))
    }
}

impl OutputSink for RecordingOutputSink {
    fn emit_event(&self, event: OutputEvent) -> io::Result<()> {
        self.events
            .lock()
            .expect("recording sink mutex poisoned")
            .push(event);
        Ok(())
    }
}

struct OutputSinkResetGuard;

impl Drop for OutputSinkResetGuard {
    fn drop(&mut self) {
        clear_output_sink();
    }
}

#[test]
fn native_model_open_finished_translates_to_info_without_readiness_events() {
    let translated =
        translate_skippy_native_runtime_event_snapshot(SkippyNativeRuntimeEventSnapshot {
            kind: SkippyNativeRuntimeEventKind::ModelOpenFinished,
            sequence: 7,
            status: "Ok",
            emitter: "OpenThread",
            progress_current: 500,
            progress_total: 1000,
            progress_unit: SkippyNativeRuntimeProgressUnit::Steps,
        })
        .expect("finished event should produce output visibility");

    match translated {
        OutputEvent::Info { message, context } => {
            assert!(message.contains("waiting for Rust runtime readiness"));
            assert!(
                context
                    .as_deref()
                    .is_some_and(|value| value.contains("sequence=7"))
            );
        }
        other => panic!("expected info event, got {other:?}"),
    }
}

#[test]
fn native_model_open_progress_translates_to_percentage_visibility() {
    let translated =
        translate_skippy_native_runtime_event_snapshot(SkippyNativeRuntimeEventSnapshot {
            kind: SkippyNativeRuntimeEventKind::ModelOpenProgress,
            sequence: 7,
            status: "Ok",
            emitter: "OpenThread",
            progress_current: 500,
            progress_total: 1000,
            progress_unit: SkippyNativeRuntimeProgressUnit::Steps,
        })
        .expect("progress event should produce output visibility");

    match translated {
        OutputEvent::Info { message, .. } => {
            assert!(message.contains("Opening native model 50%"));
        }
        other => panic!("expected info event, got {other:?}"),
    }
}

#[test]
fn native_model_open_handled_failure_translates_to_warning_without_readiness_events() {
    let translated =
        translate_skippy_native_runtime_event_snapshot(SkippyNativeRuntimeEventSnapshot {
            kind: SkippyNativeRuntimeEventKind::ModelOpenFailedHandled,
            sequence: 8,
            status: "Err",
            emitter: "OpenThread",
            progress_current: 0,
            progress_total: 0,
            progress_unit: SkippyNativeRuntimeProgressUnit::Steps,
        })
        .expect("handled failure should still produce output visibility");

    match translated {
        OutputEvent::Warning { message, context } => {
            assert!(message.contains("handled model-open failure"));
            assert!(
                context
                    .as_deref()
                    .is_some_and(|value| value.contains("sequence=8"))
            );
        }
        other => panic!("expected warning event, got {other:?}"),
    }
}

#[test]
#[serial_test::serial]
fn native_model_open_reporter_emits_visibility_only_events() {
    let sink = Arc::new(RecordingOutputSink::default());
    let _reset_guard = OutputSinkResetGuard;
    set_output_sink(sink.clone());

    let mut reporter =
        skippy_native_model_open_event_reporter("/private/models/model-a.gguf".to_string());
    for kind in [
        SkippyNativeRuntimeEventKind::ModelOpenStarted,
        SkippyNativeRuntimeEventKind::ModelOpenProgress,
        SkippyNativeRuntimeEventKind::ModelOpenFinished,
        SkippyNativeRuntimeEventKind::ModelOpenFailedHandled,
    ] {
        reporter(SkippyNativeRuntimeEvent {
            abi_version: 1,
            category: skippy_runtime::RuntimeEventCategory::ModelOpen,
            kind,
            sequence: 1,
            emitter: skippy_runtime::RuntimeEventEmitterKind::OpenThread,
            timestamp_mono_ns: 10,
            model_id: 11,
            stage_id: 0,
            session_id: 0,
            progress_current: 500,
            progress_total: 1000,
            progress_unit: SkippyNativeRuntimeProgressUnit::Steps,
            failure_code: if kind == SkippyNativeRuntimeEventKind::ModelOpenFailedHandled {
                skippy_runtime::RuntimeEventFailureCode::ModelError
            } else {
                skippy_runtime::RuntimeEventFailureCode::None
            },
            status: skippy_runtime::Status::Ok,
            detail_bytes: b"prompt=private native detail".to_vec(),
        });
    }

    let events = sink.take_events();
    let model_events = events
        .iter()
        .filter(|event| {
            matches!(
                event,
                OutputEvent::Info { .. } | OutputEvent::Warning { .. }
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        model_events.len(),
        4,
        "every native callback should stay visible"
    );
    assert!(model_events.iter().all(|event| {
        matches!(
            event,
            OutputEvent::Info { .. } | OutputEvent::Warning { .. }
        )
    }));
    assert!(events.iter().all(|event| {
        !matches!(
            event,
            OutputEvent::LaunchPlan { .. }
                | OutputEvent::ApiReady { .. }
                | OutputEvent::WebserverReady { .. }
                | OutputEvent::ModelLoading { .. }
                | OutputEvent::ModelLoaded { .. }
                | OutputEvent::ModelReady { .. }
                | OutputEvent::RuntimeReady { .. }
        )
    }));
    let serialized = format!("{events:?}");
    for raw_value in [
        "/private/models/model-a.gguf",
        "prompt=private native detail",
    ] {
        assert!(
            !serialized.contains(raw_value),
            "native presentation must not include {raw_value}"
        );
    }
}

#[test]
fn native_model_open_callbacks_map_only_static_operational_transitions() {
    assert_eq!(
        [
            SkippyNativeRuntimeEventKind::ModelOpenStarted,
            SkippyNativeRuntimeEventKind::ModelOpenProgress,
            SkippyNativeRuntimeEventKind::BackendDeviceSelected,
            SkippyNativeRuntimeEventKind::ModelOpenFinished,
            SkippyNativeRuntimeEventKind::ModelOpenFailedHandled,
        ]
        .into_iter()
        .filter_map(native_skippy_operational_event)
        .collect::<Vec<_>>(),
        vec![
            NativeSkippyOperationalEvent::ModelOpenStarted,
            NativeSkippyOperationalEvent::ModelOpenFinished,
            NativeSkippyOperationalEvent::ModelOpenFailed,
        ]
    );
}

struct FixedStoreClock(&'static str);

impl mesh_llm_log_store::Clock for FixedStoreClock {
    fn now(&self) -> String {
        self.0.to_string()
    }
}

#[tokio::test]
#[serial_test::serial]
async fn native_reporter_keeps_rich_presentation_while_audit_stays_static() {
    let temporary_directory = tempfile::tempdir().expect("temporary logging root");
    let clock: Arc<dyn mesh_llm_log_store::Clock> =
        Arc::new(FixedStoreClock("2026-08-07T12:00:00Z"));
    crate::initialize_logging_foundation_with_store_clock_for_test(
        &mesh_llm_config::LoggingConfig {
            application_state_root: Some(temporary_directory.path().join("logging")),
            ..Default::default()
        },
        clock,
    )
    .await;

    let service = crate::runtime::run_auto::start_run_auto_logging_service()
        .await
        .expect("startable logging service");

    let sink = Arc::new(RecordingOutputSink::default());
    let _reset_guard = OutputSinkResetGuard;
    set_output_sink(sink.clone());

    let mut reporter =
        skippy_native_model_open_event_reporter("/private/models/model-a.gguf".to_string());
    for kind in [
        SkippyNativeRuntimeEventKind::ModelOpenStarted,
        SkippyNativeRuntimeEventKind::ModelOpenProgress,
        SkippyNativeRuntimeEventKind::ModelOpenFinished,
    ] {
        reporter(SkippyNativeRuntimeEvent {
            abi_version: 1,
            category: skippy_runtime::RuntimeEventCategory::ModelOpen,
            kind,
            sequence: 1,
            emitter: skippy_runtime::RuntimeEventEmitterKind::OpenThread,
            timestamp_mono_ns: 10,
            model_id: 11,
            stage_id: 0,
            session_id: 0,
            progress_current: 500,
            progress_total: 1000,
            progress_unit: SkippyNativeRuntimeProgressUnit::Steps,
            failure_code: skippy_runtime::RuntimeEventFailureCode::None,
            status: skippy_runtime::Status::Ok,
            detail_bytes: b"prompt=private native detail".to_vec(),
        });
    }

    let presentation = sink.take_events();
    assert_eq!(presentation.len(), 3);
    let serialized_presentation = format!("{presentation:?}");
    for rich_context in [
        "sequence=1",
        "status=Ok",
        "emitter=OpenThread",
        "Opening native model 50%",
    ] {
        assert!(
            serialized_presentation.contains(rich_context),
            "presentation must keep {rich_context}"
        );
    }

    let audits = service
        .bus_ref()
        .drain()
        .into_iter()
        .map(|entry| {
            let audit: serde_json::Value =
                serde_json::from_str(&entry.payload).expect("audit payload");
            serde_json::json!({
                "kind": "audit",
                "level": audit["severity"],
                "message": audit["code"],
            })
        })
        .filter(|entry| {
            entry["message"]
                .as_str()
                .is_some_and(|code| code.starts_with("skippy_native_"))
        })
        .collect::<Vec<_>>();
    assert_eq!(
        audits,
        vec![
            serde_json::json!({
                "kind": "audit",
                "level": "info",
                "message": "skippy_native_model_open_started",
            }),
            serde_json::json!({
                "kind": "audit",
                "level": "info",
                "message": "skippy_native_model_open_finished",
            }),
        ]
    );
    let serialized_audits = format!("{audits:?}");
    for raw_value in [
        "model-a.gguf",
        "prompt=private native detail",
        "OpenThread",
        "sequence=",
    ] {
        assert!(
            !serialized_audits.contains(raw_value),
            "audit payloads must not include {raw_value}"
        );
    }

    assert!(service.shutdown().await);
}
