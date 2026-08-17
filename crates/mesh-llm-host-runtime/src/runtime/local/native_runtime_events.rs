use crate::runtime::operational_logging::{
    NativeSkippyOperationalEvent, record_native_skippy_operational_event,
};
use mesh_llm_events::{OutputEvent, emit_event};
use skippy_runtime::{
    RuntimeEvent as SkippyNativeRuntimeEvent, RuntimeEventKind as SkippyNativeRuntimeEventKind,
    RuntimeEventProgressUnit as SkippyNativeRuntimeProgressUnit,
};

fn skippy_native_runtime_event_context(
    sequence: u64,
    status: &str,
    emitter: &str,
) -> Option<String> {
    Some(
        [
            format!("sequence={sequence}"),
            format!("status={status}"),
            format!("emitter={emitter}"),
        ]
        .join(" "),
    )
}

struct SkippyNativeRuntimeEventSnapshot<'a> {
    kind: SkippyNativeRuntimeEventKind,
    sequence: u64,
    status: &'a str,
    emitter: &'a str,
    progress_current: u64,
    progress_total: u64,
    progress_unit: SkippyNativeRuntimeProgressUnit,
}

fn translate_skippy_native_runtime_event_snapshot(
    snapshot: SkippyNativeRuntimeEventSnapshot<'_>,
) -> Option<OutputEvent> {
    let context =
        skippy_native_runtime_event_context(snapshot.sequence, snapshot.status, snapshot.emitter);
    match snapshot.kind {
        SkippyNativeRuntimeEventKind::ModelOpenStarted => Some(OutputEvent::Info {
            message: "Native runtime started opening model".to_string(),
            context,
        }),
        SkippyNativeRuntimeEventKind::ModelOpenProgress => {
            let progress = match (
                snapshot.progress_current,
                snapshot.progress_total,
                snapshot.progress_unit,
            ) {
                (current, total, SkippyNativeRuntimeProgressUnit::Steps) if total > 0 => {
                    format!("{}%", current.saturating_mul(100) / total)
                }
                (current, total, unit) if total > 0 => {
                    format!("{current}/{total} {unit:?}")
                }
                (current, _, unit) => format!("{current} {unit:?}"),
            };
            Some(OutputEvent::Info {
                message: format!("Opening native model {progress}"),
                context,
            })
        }
        SkippyNativeRuntimeEventKind::BackendDeviceSelected => Some(OutputEvent::Info {
            message: "Native runtime selected a backend device".to_string(),
            context,
        }),
        SkippyNativeRuntimeEventKind::ModelOpenFinished => Some(OutputEvent::Info {
            message: "Native runtime finished opening model; waiting for Rust runtime readiness"
                .to_string(),
            context,
        }),
        SkippyNativeRuntimeEventKind::ModelOpenFailedHandled => Some(OutputEvent::Warning {
            message: "Native runtime reported a handled model-open failure".to_string(),
            context,
        }),
        SkippyNativeRuntimeEventKind::Unknown(_) => None,
    }
}

fn translate_skippy_native_runtime_event(event: &SkippyNativeRuntimeEvent) -> Option<OutputEvent> {
    let status = format!("{:?}", event.status);
    let emitter = format!("{:?}", event.emitter);
    translate_skippy_native_runtime_event_snapshot(SkippyNativeRuntimeEventSnapshot {
        kind: event.kind,
        sequence: event.sequence,
        status: &status,
        emitter: &emitter,
        progress_current: event.progress_current,
        progress_total: event.progress_total,
        progress_unit: event.progress_unit,
    })
}

fn native_skippy_operational_event(
    kind: SkippyNativeRuntimeEventKind,
) -> Option<NativeSkippyOperationalEvent> {
    match kind {
        SkippyNativeRuntimeEventKind::ModelOpenStarted => {
            Some(NativeSkippyOperationalEvent::ModelOpenStarted)
        }
        SkippyNativeRuntimeEventKind::ModelOpenFinished => {
            Some(NativeSkippyOperationalEvent::ModelOpenFinished)
        }
        SkippyNativeRuntimeEventKind::ModelOpenFailedHandled => {
            Some(NativeSkippyOperationalEvent::ModelOpenFailed)
        }
        SkippyNativeRuntimeEventKind::ModelOpenProgress
        | SkippyNativeRuntimeEventKind::BackendDeviceSelected
        | SkippyNativeRuntimeEventKind::Unknown(_) => None,
    }
}

fn emit_skippy_native_runtime_event(event: SkippyNativeRuntimeEvent) {
    if let Some(operational_event) = native_skippy_operational_event(event.kind) {
        record_native_skippy_operational_event(operational_event);
    }
    let Some(output_event) = translate_skippy_native_runtime_event(&event) else {
        return;
    };
    let _ = emit_event(output_event);
}

pub(super) fn skippy_native_model_open_event_reporter(
    _model_name: String,
) -> crate::inference::skippy::NativeModelOpenEventReporter {
    Box::new(emit_skippy_native_runtime_event)
}

#[cfg(test)]
mod tests;
