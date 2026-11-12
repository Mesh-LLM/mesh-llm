//! Where telemetry goes. `Telemetry` says what happened, as it happens; each
//! sink decides what to do with it. A sink that only wants finished spans
//! (OTLP) ignores the live calls; one that must act at the moment a span
//! begins on a thread (Tracy, whose zones cannot be backdated) acts in
//! `create_span_with` and `end_span`.

use std::{any::Any, collections::BTreeMap, time::Duration};

use opentelemetry_proto::tonic::metrics::v1::Metric;
use serde_json::Value;

use super::{
    active_span::SpanLevel,
    trace_context::{Frame, SpanContext, SpanEvent},
};

/// A finished span, placed in the process trace.
#[derive(Clone)]
pub(crate) struct SpanRecord {
    pub(crate) context: SpanContext,
    pub(crate) parent: Option<[u8; 8]>,
    pub(crate) name: String,
    pub(crate) attributes: BTreeMap<String, Value>,
    pub(crate) start_time_unix_nanos: u64,
    pub(crate) end_time_unix_nanos: u64,
    pub(crate) links: Vec<SpanContext>,
    pub(crate) events: Vec<SpanEvent>,
}

impl SpanRecord {
    pub(crate) fn of_frame(
        frame: &Frame,
        name: &str,
        attributes: BTreeMap<String, Value>,
        start_time_unix_nanos: u64,
        end_time_unix_nanos: u64,
    ) -> Self {
        Self {
            context: frame.context(),
            parent: frame.parent_span_id(),
            name: name.to_string(),
            attributes,
            start_time_unix_nanos,
            end_time_unix_nanos: end_time_unix_nanos.max(start_time_unix_nanos.saturating_add(1)),
            links: Vec::new(),
            events: Vec::new(),
        }
    }
}

/// A span that has just begun on the calling thread. Its attributes, links
/// and events arrive with the [`SpanRecord`] at `end_span`.
// Read by sinks that act on live spans, which is the Tracy sink.
#[cfg_attr(not(feature = "tracy"), allow(dead_code))]
pub(crate) struct SpanStart {
    pub(crate) name: &'static str,
    pub(crate) level: SpanLevel,
    pub(crate) context: SpanContext,
    pub(crate) parent: Option<[u8; 8]>,
    pub(crate) start_time_unix_nanos: u64,
}

/// What a sink keeps from a span's start to its end. A sink whose state is
/// bound to the starting thread keeps it there, keyed by the span id, since
/// `end_span` may run on another thread.
pub(crate) struct OpenSpan(Option<Box<dyn Any + Send>>);

// Kept by sinks that act on live spans, which is the Tracy sink.
#[cfg_attr(not(feature = "tracy"), allow(dead_code))]
impl OpenSpan {
    pub(crate) const NONE: Self = Self(None);

    pub(crate) fn keep(state: impl Any + Send) -> Self {
        Self(Some(Box::new(state)))
    }

    pub(crate) fn take<T: Any>(self) -> Option<T> {
        self.0
            .and_then(|state| state.downcast::<T>().ok())
            .map(|state| *state)
    }
}

pub(crate) trait TraceSink: Send + Sync {
    /// A span begins now, on this thread. What is returned comes back to
    /// `end_span`.
    fn create_span_with(&self, span: &SpanStart) -> OpenSpan;

    /// The span `create_span_with` saw ends, with everything learnt about it.
    fn end_span(&self, open: OpenSpan, span: &SpanRecord);

    /// The span `create_span_with` saw turned out to do no work and is not
    /// kept; a sink that already shows it closes it here.
    fn discard_span(&self, open: OpenSpan);

    /// A span measured elsewhere, on its own clock: a native call, a GPU
    /// command, a span that ended after its frame was gone.
    fn record_span(&self, span: &SpanRecord);

    /// A point event, as it happens, on the frame it belongs to.
    fn record_event(&self, event: &SpanEvent, frame: &Frame);

    /// One sample of the process: CPU, GPU, memory, scheduler and KV gauges,
    /// taken together at one instant.
    fn record_metrics(&self, metrics: &[Metric]);

    /// Whether events are also wanted folded into their span's record, the
    /// OTLP shape; a sink that keeps its own line per event says no.
    fn folds_events_into_spans(&self) -> bool;

    /// Block until what was queued has left the process, or `timeout` passes.
    fn flush(&self, timeout: Duration);
}

/// Test-only emission shape handed to sibling-module tests.
#[cfg(test)]
pub(crate) struct CapturedEvent {
    pub event: String,
    pub attributes: BTreeMap<String, Value>,
    pub start_time_unix_nanos: u64,
    pub end_time_unix_nanos: u64,
    /// The exported span's identity and parent; `None` for a point event.
    pub span: Option<(SpanContext, Option<[u8; 8]>)>,
}

/// Test-only: hands every emission to an in-process receiver, so tests
/// assert exact emissions and level gating without a collector or stderr.
#[cfg(test)]
pub(crate) struct CaptureSink {
    tx: std::sync::mpsc::SyncSender<CapturedEvent>,
}

#[cfg(test)]
impl CaptureSink {
    pub(crate) fn channel() -> (Self, std::sync::mpsc::Receiver<CapturedEvent>) {
        let (tx, rx) = std::sync::mpsc::sync_channel(256);
        (Self { tx }, rx)
    }
}

#[cfg(test)]
impl TraceSink for CaptureSink {
    fn create_span_with(&self, _: &SpanStart) -> OpenSpan {
        OpenSpan::NONE
    }

    fn end_span(&self, _: OpenSpan, span: &SpanRecord) {
        self.record_span(span);
    }

    fn discard_span(&self, _: OpenSpan) {}

    fn record_span(&self, span: &SpanRecord) {
        let _ = self.tx.try_send(CapturedEvent {
            event: span.name.clone(),
            attributes: span.attributes.clone(),
            start_time_unix_nanos: span.start_time_unix_nanos,
            end_time_unix_nanos: span.end_time_unix_nanos,
            span: Some((span.context, span.parent)),
        });
    }

    fn record_event(&self, event: &SpanEvent, _: &Frame) {
        let _ = self.tx.try_send(CapturedEvent {
            event: event.name.clone(),
            attributes: event.attributes.clone(),
            start_time_unix_nanos: event.time_unix_nanos,
            end_time_unix_nanos: event.time_unix_nanos + 1,
            span: None,
        });
    }

    fn record_metrics(&self, _: &[Metric]) {}

    fn folds_events_into_spans(&self) -> bool {
        false
    }

    fn flush(&self, _: Duration) {}
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::telemetry::{
        Telemetry, TelemetryCounters, TelemetryLevel, active_span::ActiveSpan,
        capture_stage_config, trace_context,
    };

    #[test]
    fn an_open_span_returns_what_the_sink_kept() {
        assert_eq!(OpenSpan::keep(7_u32).take::<u32>(), Some(7));
        assert_eq!(OpenSpan::keep(7_u32).take::<u64>(), None);
        assert_eq!(OpenSpan::NONE.take::<u32>(), None);
    }

    /// A sink that acts at both ends of a live span, as Tracy must.
    #[derive(Default)]
    struct Seen {
        started: Mutex<Vec<(&'static str, SpanLevel, SpanContext, Option<[u8; 8]>)>>,
        ended: Mutex<Vec<(String, SpanContext, Option<u64>)>>,
        discarded: Mutex<Vec<Option<u64>>>,
    }

    struct LiveSink(Arc<Seen>);

    impl TraceSink for LiveSink {
        fn create_span_with(&self, span: &SpanStart) -> OpenSpan {
            self.0
                .started
                .lock()
                .unwrap()
                .push((span.name, span.level, span.context, span.parent));
            OpenSpan::keep(span.start_time_unix_nanos)
        }

        fn end_span(&self, open: OpenSpan, span: &SpanRecord) {
            self.0.ended.lock().unwrap().push((
                span.name.clone(),
                span.context,
                open.take::<u64>(),
            ));
        }

        fn discard_span(&self, open: OpenSpan) {
            self.0.discarded.lock().unwrap().push(open.take::<u64>());
        }

        fn record_span(&self, _: &SpanRecord) {}

        fn record_event(&self, _: &SpanEvent, _: &Frame) {}

        fn record_metrics(&self, _: &[Metric]) {}

        fn folds_events_into_spans(&self) -> bool {
            false
        }

        fn flush(&self, _: Duration) {}
    }

    #[test]
    fn a_live_span_reaches_its_sink_at_both_ends() {
        let seen = Arc::new(Seen::default());
        let telemetry = Telemetry::with_sinks(
            vec![Box::new(LiveSink(seen.clone()))],
            Arc::new(TelemetryCounters::default()),
            Arc::new(capture_stage_config()),
            TelemetryLevel::Summary,
        );
        let root = trace_context::process().root().context();

        let span = ActiveSpan::start(Some(&telemetry), "stage.kept", SpanLevel::Summary);
        let kept = span.frame().context();
        span.end();
        let span = ActiveSpan::start(Some(&telemetry), "stage.dropped", SpanLevel::Summary);
        let dropped = span.frame().context();
        span.discard();
        ActiveSpan::start(Some(&telemetry), "stage.unrecorded", SpanLevel::Debug).end();

        let started = seen.started.lock().unwrap();
        assert_eq!(started.len(), 2, "an unrecorded span never reaches a sink");
        assert_eq!(
            started[0],
            ("stage.kept", SpanLevel::Summary, kept, Some(root.span_id))
        );
        assert_eq!(started[1].2, dropped);
        let ended = seen.ended.lock().unwrap();
        assert_eq!(ended.len(), 1);
        assert_eq!(ended[0].0, "stage.kept");
        assert_eq!(ended[0].1, kept);
        assert!(
            ended[0].2.is_some(),
            "what the sink kept comes back at the end"
        );
        let discarded = seen.discarded.lock().unwrap();
        assert_eq!(discarded.len(), 1);
        assert!(
            discarded[0].is_some(),
            "a discarded span returns what was kept too"
        );
    }
}
