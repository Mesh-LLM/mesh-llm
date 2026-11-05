//! Native ABI call timings as debug spans, separating time spent inside the
//! patched llama.cpp runtime from Skippy's own scheduling and bookkeeping.
//! Each call's span id doubles as its native trace tag, so the native spans
//! and GPU command buffers the call causes land under it.

use std::cell::RefCell;
use std::collections::BTreeMap;

use serde_json::json;
use skippy_runtime::native_timing::{NativeCall, NativeCallObserver, set_native_call_observer};
use skippy_runtime::native_tracing::request_native_tracing;

use super::{Telemetry, gpu_commands, native_spans, trace_context::Frame};

thread_local! {
    /// Calls begun on this thread and not yet observed, innermost last.
    static OPEN_CALLS: RefCell<Vec<Frame>> = const { RefCell::new(Vec::new()) };
}

struct NativeCallSpans(Telemetry);

impl NativeCallObserver for NativeCallSpans {
    fn begin(&self, _symbol: &'static str) -> u64 {
        let frame = Frame::child();
        let tag = frame.tag();
        OPEN_CALLS.with(|open| open.borrow_mut().push(frame));
        tag
    }

    fn observe(&self, call: NativeCall) {
        let frame = OPEN_CALLS
            .with(|open| {
                let mut open = open.borrow_mut();
                let index = open.iter().rposition(|frame| frame.tag() == call.id)?;
                Some(open.remove(index))
            })
            .unwrap_or_else(Frame::child);
        self.0.emit_debug_frame_span(
            &frame,
            &span_name(call.symbol),
            BTreeMap::from([
                ("skippy.native.symbol".to_string(), json!(call.symbol)),
                (
                    "thread.name".to_string(),
                    json!(std::thread::current().name().unwrap_or("unnamed")),
                ),
            ]),
            call.start_unix_nanos,
            call.end_unix_nanos,
        );
        // The call's native spans have all ended; its GPU work arrives later.
        native_spans::emit_pending(&self.0);
        gpu_commands::emit_pending(&self.0);
    }
}

fn span_name(symbol: &str) -> String {
    format!("native.{symbol}")
}

/// Report native call timings, and the native spans and GPU work each call
/// causes, through `telemetry`. The first debug handle with a sink serves the
/// whole process.
pub(super) fn install(telemetry: &Telemetry) {
    if set_native_call_observer(Box::new(NativeCallSpans(telemetry.clone()))) {
        request_native_tracing();
    }
}

#[cfg(test)]
mod tests {
    use super::{NativeCallSpans, span_name};
    use crate::telemetry::{Telemetry, TelemetryLevel, capture_stage_config};
    use skippy_runtime::native_timing::{NativeCall, NativeCallObserver};

    #[test]
    fn spans_are_named_after_the_native_symbol() {
        assert_eq!(
            span_name("skippy_iteration_batch_sampled"),
            "native.skippy_iteration_batch_sampled"
        );
    }

    #[test]
    fn a_call_span_takes_the_id_its_gpu_work_is_tagged_with() {
        let (telemetry, rx) = Telemetry::captured(capture_stage_config(), TelemetryLevel::Debug);
        let observer = NativeCallSpans(telemetry);
        let tag = observer.begin("skippy_iteration_batch_sampled");
        observer.observe(NativeCall {
            symbol: "skippy_iteration_batch_sampled",
            id: tag,
            start_unix_nanos: 10,
            end_unix_nanos: 20,
        });

        let span = rx.try_recv().expect("native span captured");
        assert_eq!(span.event, "native.skippy_iteration_batch_sampled");
        let (context, _) = span.span.expect("an exported span");
        assert_eq!(u64::from_be_bytes(context.span_id), tag);
    }
}
