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
    let translated = translate_skippy_native_runtime_event_snapshot(
        "model-a",
        SkippyNativeRuntimeEventSnapshot {
            kind: SkippyNativeRuntimeEventKind::ModelOpenFinished,
            sequence: 7,
            status: "Ok",
            emitter: "OpenThread",
            progress_current: 500,
            progress_total: 1000,
            progress_unit: SkippyNativeRuntimeProgressUnit::Steps,
            detail: Some("Metal GPU 0"),
        },
    )
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
    let translated = translate_skippy_native_runtime_event_snapshot(
        "model-a",
        SkippyNativeRuntimeEventSnapshot {
            kind: SkippyNativeRuntimeEventKind::ModelOpenProgress,
            sequence: 7,
            status: "Ok",
            emitter: "OpenThread",
            progress_current: 500,
            progress_total: 1000,
            progress_unit: SkippyNativeRuntimeProgressUnit::Steps,
            detail: Some("Metal GPU 0"),
        },
    )
    .expect("progress event should produce output visibility");

    match translated {
        OutputEvent::Info { message, .. } => {
            assert!(message.contains("Opening model 'model-a' 50%"));
        }
        other => panic!("expected info event, got {other:?}"),
    }
}

#[test]
fn native_model_open_handled_failure_translates_to_warning_without_readiness_events() {
    let translated = translate_skippy_native_runtime_event_snapshot(
        "model-a",
        SkippyNativeRuntimeEventSnapshot {
            kind: SkippyNativeRuntimeEventKind::ModelOpenFailedHandled,
            sequence: 8,
            status: "Err",
            emitter: "OpenThread",
            progress_current: 0,
            progress_total: 0,
            progress_unit: SkippyNativeRuntimeProgressUnit::Steps,
            detail: Some("simulated native error"),
        },
    )
    .expect("handled failure should still produce output visibility");

    match translated {
        OutputEvent::Warning { message, context } => {
            assert!(message.contains("handled model-open failure"));
            assert!(
                context
                    .as_deref()
                    .is_some_and(|value| value.contains("detail=simulated native error"))
            );
        }
        other => panic!("expected warning event, got {other:?}"),
    }
}

#[test]
fn native_model_open_reporter_emits_visibility_only_events() {
    let sink = Arc::new(RecordingOutputSink::default());
    let _reset_guard = OutputSinkResetGuard;
    set_output_sink(sink.clone());

    let mut reporter = skippy_native_model_open_event_reporter("model-a".to_string());
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
            detail_bytes: b"Metal GPU 0".to_vec(),
        });
    }

    let events = sink.take_events();
    assert_eq!(events.len(), 4, "every native callback should stay visible");
    assert!(events.iter().all(|event| {
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
}
