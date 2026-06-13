use mesh_llm_events::{OutputEvent, emit_event};
use skippy_runtime::{
    RuntimeEvent as SkippyNativeRuntimeEvent, RuntimeEventKind as SkippyNativeRuntimeEventKind,
    RuntimeEventProgressUnit as SkippyNativeRuntimeProgressUnit,
};

fn skippy_native_runtime_event_detail(event: &SkippyNativeRuntimeEvent) -> Option<String> {
    let detail = String::from_utf8_lossy(&event.detail_bytes)
        .trim()
        .to_string();
    (!detail.is_empty()).then_some(detail)
}

fn skippy_native_runtime_event_context(
    sequence: u64,
    status: &str,
    emitter: &str,
    detail: Option<&str>,
) -> Option<String> {
    let mut parts = vec![
        format!("sequence={sequence}"),
        format!("status={status}"),
        format!("emitter={emitter}"),
    ];
    if let Some(detail) = detail {
        parts.push(format!("detail={detail}"));
    }
    Some(parts.join(" "))
}

struct SkippyNativeRuntimeEventSnapshot<'a> {
    kind: SkippyNativeRuntimeEventKind,
    sequence: u64,
    status: &'a str,
    emitter: &'a str,
    progress_current: u64,
    progress_total: u64,
    progress_unit: SkippyNativeRuntimeProgressUnit,
    detail: Option<&'a str>,
}

fn translate_skippy_native_runtime_event_snapshot(
    model_name: &str,
    snapshot: SkippyNativeRuntimeEventSnapshot<'_>,
) -> Option<OutputEvent> {
    let context = skippy_native_runtime_event_context(
        snapshot.sequence,
        snapshot.status,
        snapshot.emitter,
        snapshot.detail,
    );
    match snapshot.kind {
        SkippyNativeRuntimeEventKind::ModelOpenStarted => Some(OutputEvent::Info {
            message: format!("Native runtime started opening model '{model_name}'"),
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
                message: format!("Opening model '{model_name}' {progress}"),
                context,
            })
        }
        SkippyNativeRuntimeEventKind::BackendDeviceSelected => Some(OutputEvent::Info {
            message: match snapshot.detail {
                Some(device) => {
                    format!("Native runtime selected backend device for '{model_name}': {device}")
                }
                None => format!("Native runtime selected a backend device for '{model_name}'"),
            },
            context,
        }),
        SkippyNativeRuntimeEventKind::ModelOpenFinished => Some(OutputEvent::Info {
            message: format!(
                "Native runtime finished opening model '{model_name}'; waiting for Rust runtime readiness"
            ),
            context,
        }),
        SkippyNativeRuntimeEventKind::ModelOpenFailedHandled => Some(OutputEvent::Warning {
            message: format!(
                "Native runtime reported a handled model-open failure for '{model_name}'"
            ),
            context,
        }),
        SkippyNativeRuntimeEventKind::Unknown(_) => None,
    }
}

fn translate_skippy_native_runtime_event(
    model_name: &str,
    event: &SkippyNativeRuntimeEvent,
) -> Option<OutputEvent> {
    let detail = skippy_native_runtime_event_detail(event);
    let status = format!("{:?}", event.status);
    let emitter = format!("{:?}", event.emitter);
    translate_skippy_native_runtime_event_snapshot(
        model_name,
        SkippyNativeRuntimeEventSnapshot {
            kind: event.kind,
            sequence: event.sequence,
            status: &status,
            emitter: &emitter,
            progress_current: event.progress_current,
            progress_total: event.progress_total,
            progress_unit: event.progress_unit,
            detail: detail.as_deref(),
        },
    )
}

fn emit_skippy_native_runtime_event(model_name: &str, event: SkippyNativeRuntimeEvent) {
    let Some(output_event) = translate_skippy_native_runtime_event(model_name, &event) else {
        return;
    };
    let _ = emit_event(output_event);
}

pub(super) fn skippy_native_model_open_event_reporter(
    model_name: String,
) -> crate::inference::skippy::NativeModelOpenEventReporter {
    Box::new(move |event| emit_skippy_native_runtime_event(&model_name, event))
}

#[cfg(test)]
mod tests;
