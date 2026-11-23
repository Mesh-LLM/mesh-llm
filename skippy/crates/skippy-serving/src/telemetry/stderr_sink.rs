//! The `SKIPPY_TELEMETRY_STDERR` debug sink: Skippy's own event JSON, one
//! line per span or event, written to stderr by a background thread.

use std::{
    collections::BTreeMap,
    env,
    io::{BufWriter, Write},
    sync::{Arc, OnceLock, atomic::Ordering, mpsc as std_mpsc},
    thread::{self, JoinHandle},
    time::Duration,
};

use opentelemetry_proto::tonic::metrics::v1::Metric;
use serde::Serialize;
use serde_json::Value;

use super::{
    TelemetryCounters,
    sink::{OpenSpan, SpanRecord, SpanStart, TraceSink},
    trace_context::{Frame, SpanEvent},
};

static STDERR_SINK: OnceLock<bool> = OnceLock::new();
/// Whether the stderr debug sink is enabled via `SKIPPY_TELEMETRY_STDERR`.
///
/// Read once: this is consulted on the record and lookup paths, which run per
/// request.
pub(super) fn stderr_sink_enabled() -> bool {
    *STDERR_SINK.get_or_init(|| {
        stderr_sink_enabled_from_value(env::var("SKIPPY_TELEMETRY_STDERR").ok().as_deref())
    })
}

fn stderr_sink_enabled_from_value(value: Option<&str>) -> bool {
    value.is_some_and(|value| {
        !matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "" | "0" | "false" | "no" | "off"
        )
    })
}

#[derive(Serialize)]
pub(super) struct StderrTelemetryEvent {
    pub(super) event: String,
    pub(super) attributes: BTreeMap<String, Value>,
    pub(super) start_time_unix_nanos: u64,
    pub(super) end_time_unix_nanos: u64,
}

pub(super) struct StderrTelemetrySink {
    tx: Option<std_mpsc::SyncSender<StderrTelemetryEvent>>,
    worker: Option<JoinHandle<()>>,
}

impl StderrTelemetrySink {
    fn channel(
        capacity: usize,
    ) -> (
        std_mpsc::SyncSender<StderrTelemetryEvent>,
        std_mpsc::Receiver<StderrTelemetryEvent>,
    ) {
        // Disabled telemetry supplies zero capacity. The stderr override can
        // enable it later, so give that opt-in sink the same bounded default
        // as debug telemetry instead of a one-event queue that loses bursts.
        std_mpsc::sync_channel(if capacity == 0 { 1024 } else { capacity })
    }

    pub(super) fn new(capacity: usize) -> Option<Self> {
        let (tx, rx) = Self::channel(capacity);
        let worker = thread::Builder::new()
            .name("skippy-telemetry-stderr".to_string())
            .spawn(move || stderr_telemetry_loop(rx))
            .ok()?;
        Some(Self {
            tx: Some(tx),
            worker: Some(worker),
        })
    }

    pub(super) fn try_send(&self, event: StderrTelemetryEvent) -> bool {
        self.tx
            .as_ref()
            .is_some_and(|tx| tx.try_send(event).is_ok())
    }
}

impl Drop for StderrTelemetrySink {
    fn drop(&mut self) {
        self.tx.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// The stderr writer as a trace sink: one line per span and per event.
pub(super) struct StderrTraceSink {
    sink: StderrTelemetrySink,
    stats: Arc<TelemetryCounters>,
}

impl StderrTraceSink {
    pub(super) fn enabled(capacity: usize, stats: Arc<TelemetryCounters>) -> Option<Self> {
        stderr_sink_enabled()
            .then(|| StderrTelemetrySink::new(capacity))
            .flatten()
            .map(|sink| Self { sink, stats })
    }

    fn line(
        &self,
        name: &str,
        attributes: &BTreeMap<String, Value>,
        start_time_unix_nanos: u64,
        end_time_unix_nanos: u64,
    ) {
        if !self.sink.try_send(StderrTelemetryEvent {
            event: name.to_string(),
            attributes: attributes.clone(),
            start_time_unix_nanos,
            end_time_unix_nanos,
        }) {
            self.stats.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }
}

impl TraceSink for StderrTraceSink {
    fn create_span_with(&self, _: &SpanStart) -> OpenSpan {
        OpenSpan::NONE
    }

    fn end_span(&self, _: OpenSpan, span: &SpanRecord) {
        self.record_span(span);
    }

    fn discard_span(&self, _: OpenSpan) {}

    fn record_span(&self, span: &SpanRecord) {
        self.line(
            &span.name,
            &span.attributes,
            span.start_time_unix_nanos,
            span.end_time_unix_nanos,
        );
    }

    fn record_event(&self, event: &SpanEvent, _: &Frame) {
        self.line(
            &event.name,
            &event.attributes,
            event.time_unix_nanos,
            event.time_unix_nanos + 1,
        );
    }

    /// Samples are not lines.
    fn record_metrics(&self, _: &[Metric]) {}

    fn folds_events_into_spans(&self) -> bool {
        false
    }

    /// Lines are best effort; shutdown never waited on them.
    fn flush(&self, _: Duration) {}
}

fn stderr_telemetry_loop(rx: std_mpsc::Receiver<StderrTelemetryEvent>) {
    while let Ok(event) = rx.recv() {
        let stderr = std::io::stderr();
        let mut writer = BufWriter::new(stderr.lock());
        if serde_json::to_writer(&mut writer, &event).is_err() || writer.write_all(b"\n").is_err() {
            return;
        }
        while let Ok(event) = rx.try_recv() {
            if serde_json::to_writer(&mut writer, &event).is_err()
                || writer.write_all(b"\n").is_err()
            {
                return;
            }
        }
        if writer.flush().is_err() {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{StderrTelemetryEvent, StderrTelemetrySink, stderr_sink_enabled_from_value};

    #[test]
    fn stderr_queue_buffers_disabled_config_bursts_and_respects_explicit_bounds() {
        for (configured, expected) in [(0, 1024), (1, 1), (8, 8)] {
            let (tx, rx) = StderrTelemetrySink::channel(configured);
            let event = || StderrTelemetryEvent {
                event: "stage.binary_request_summary".into(),
                attributes: Default::default(),
                start_time_unix_nanos: 1,
                end_time_unix_nanos: 2,
            };
            for _ in 0..expected {
                tx.try_send(event()).expect("burst fits configured buffer");
            }
            assert!(matches!(
                tx.try_send(event()),
                Err(std::sync::mpsc::TrySendError::Full(_))
            ));
            assert_eq!(rx.try_iter().count(), expected);
            tx.try_send(event()).expect("drained buffer accepts events");
        }
    }

    #[test]
    fn stderr_sink_false_values_do_not_enable_debug_telemetry() {
        for value in [
            None,
            Some(""),
            Some("0"),
            Some("false"),
            Some("no"),
            Some("off"),
        ] {
            assert!(!stderr_sink_enabled_from_value(value), "value={value:?}");
        }
        for value in [
            Some("1"),
            Some("true"),
            Some("yes"),
            Some("on"),
            Some("debug"),
        ] {
            assert!(stderr_sink_enabled_from_value(value), "value={value:?}");
        }
    }
}
