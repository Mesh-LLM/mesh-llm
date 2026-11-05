//! Native spans as debug spans: the phases of host work inside each native
//! call (llama graph build, encode, synchronize, sampling, Metal library and
//! pipeline compilation), so a native call's span has no unexplained time.
//!
//! A native span keeps the id the runtime gave it, so its children, which
//! end and arrive first, can already name it as their parent. A span with no
//! native parent sits under the call whose tag it carries, or under the
//! process root when it ran before anything was tagged.

use std::collections::BTreeMap;

use serde_json::{Value, json};
use skippy_runtime::native_tracing::spans::{NativeSpan, drain_native_spans};

use super::{
    Telemetry,
    log_events::{self, BufferAllocated, GraphReserved, PipelineCompiled},
    trace_context::{self, Frame},
};

/// Emit every native span queued so far.
pub(super) fn emit_pending(telemetry: &Telemetry) {
    if !telemetry.is_debug_enabled() {
        return;
    }
    drain_native_spans(|span| {
        let frame = frame(&span);
        record_stall(&frame, &span);
        telemetry.emit_debug_frame_span(
            &frame,
            span.name,
            attributes(&span),
            span.start_unix_nanos,
            span.end_unix_nanos,
        );
    });
}

/// Native work that, once the model is serving, stalls the graph behind it
/// also becomes an event on its span.
fn record_stall(frame: &Frame, span: &NativeSpan) {
    if !log_events::serving() || !log_events::enabled() {
        return;
    }
    let detail = span.detail.as_deref().map(str::to_string);
    let duration_ms = span.end_unix_nanos.saturating_sub(span.start_unix_nanos) as f64 / 1e6;
    match span.name {
        "llama.sched_reserve" => {
            log_events::record_under(
                frame,
                &GraphReserved {
                    detail,
                    duration_ms,
                },
            );
        }
        "mtl.pipeline_compile" => {
            log_events::record_under(
                frame,
                &PipelineCompiled {
                    detail,
                    duration_ms,
                },
            );
        }
        "mtl.buffer_alloc" => {
            log_events::record_under(
                frame,
                &BufferAllocated {
                    detail,
                    duration_ms,
                },
            );
        }
        _ => {}
    }
}

/// The span's own id, under its native parent or tagged call.
fn frame(span: &NativeSpan) -> Frame {
    Frame::tagged_under(trace_context::context_for_tag(span.parent), span.id)
}

fn attributes(span: &NativeSpan) -> BTreeMap<String, Value> {
    let mut attrs = BTreeMap::new();
    if let Some(detail) = span.detail.as_deref() {
        attrs.insert("skippy.native.detail".to_string(), json!(detail));
    }
    if let Some((name, value)) = span.value {
        attrs.insert(format!("skippy.native.{name}"), json!(value));
    }
    attrs
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(detail: Option<&str>, value: Option<(&'static str, i64)>) -> NativeSpan {
        NativeSpan {
            id: (1 << 63) | 3,
            parent: 17,
            name: "mtl.pipeline_compile",
            detail: detail.map(Into::into),
            start_unix_nanos: 10,
            end_unix_nanos: 20,
            value,
        }
    }

    #[test]
    fn a_native_span_keeps_its_id_under_the_call_it_names() {
        let call = Frame::child();
        let span = NativeSpan {
            parent: call.tag(),
            ..sample(None, None)
        };
        let native = frame(&span);
        assert_eq!(native.tag(), span.id);
        assert_eq!(native.parent_span_id(), Some(call.context().span_id));

        let child = NativeSpan {
            id: (1 << 63) | 4,
            parent: span.id,
            ..span.clone()
        };
        assert_eq!(
            frame(&child).parent_span_id(),
            Some(native.context().span_id)
        );

        let untagged = NativeSpan { parent: 0, ..span };
        assert_eq!(
            frame(&untagged).parent_span_id(),
            Some(trace_context::process().root().context().span_id)
        );
    }

    #[test]
    fn attributes_carry_the_detail_and_the_measured_value() {
        let attrs = attributes(&sample(Some("kernel_add_fuse_1"), Some(("nodes", 42))));
        assert_eq!(attrs["skippy.native.detail"], json!("kernel_add_fuse_1"));
        assert_eq!(attrs["skippy.native.nodes"], json!(42));
        assert!(attributes(&sample(None, None)).is_empty());
    }
}
