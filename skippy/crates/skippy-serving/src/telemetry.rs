#[cfg(test)]
use std::sync::mpsc as std_mpsc;
use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use opentelemetry_proto::tonic::metrics::v1::Metric;
use serde::Serialize;
use serde_json::{Value, json};
use skippy_metrics::{attr, metric};
use skippy_protocol::StageConfig;

pub(crate) mod active_span;
mod gpu_commands;
pub(crate) mod log_events;
mod native_calls;
mod native_spans;
mod otlp_collector;
mod otlp_encoding;
mod otlp_file;
mod resource_attrs;
pub(crate) mod resource_metrics;
mod resource_usage;
pub(crate) mod sink;
mod span_components;
mod stderr_sink;
pub(crate) mod trace_context;
#[cfg(feature = "tracy")]
mod tracy_sink;

#[cfg(test)]
pub(crate) use sink::CapturedEvent;
use sink::{OpenSpan, SpanRecord, SpanStart, TraceSink};
use trace_context::{Frame, SpanContext, SpanEvent};

/// How long shutdown waits for queued spans to reach their sinks.
const FLUSH_TIMEOUT: Duration = Duration::from_secs(5);
/// Bound on delivering a span that ends after the process span.
const LATE_SPAN_FLUSH_TIMEOUT: Duration = Duration::from_millis(500);
/// Spans per exported request, for the collector and the file alike.
const EXPORT_BATCH_MAX: usize = 128;

/// The Tracy sink, when this build carries it and the environment asks.
#[cfg(feature = "tracy")]
fn tracy_sink() -> Option<Box<dyn TraceSink>> {
    tracy_sink::TracySink::enabled().map(|sink| Box::new(sink) as Box<dyn TraceSink>)
}

#[cfg(not(feature = "tracy"))]
fn tracy_sink() -> Option<Box<dyn TraceSink>> {
    None
}

/// A local debug sink (stderr, OTLP file or Tracy) acts as a floor on the
/// configured level.
///
/// It never lowers an explicitly configured level, so a collector-backed
/// `Summary` deployment is unaffected by the variable being set.
fn effective_level(configured: TelemetryLevel, local_sink: bool) -> TelemetryLevel {
    if local_sink && configured == TelemetryLevel::Off {
        TelemetryLevel::Debug
    } else {
        configured
    }
}

/// What happened in this stage, told to every sink as it happens. Spans
/// begin and end here; what each sink makes of them is its own business
/// (see [`sink::TraceSink`]).
#[derive(Clone)]
pub struct Telemetry {
    sinks: Arc<[Box<dyn TraceSink>]>,
    stats: Arc<TelemetryCounters>,
    config: Arc<StageConfig>,
    level: TelemetryLevel,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TelemetryLevel {
    Off,
    Summary,
    Debug,
}

#[derive(Default)]
pub(crate) struct TelemetryCounters {
    pub(crate) queued: AtomicU64,
    pub(crate) sent: AtomicU64,
    pub(crate) dropped: AtomicU64,
    pub(crate) export_errors: AtomicU64,
}

#[derive(Clone, Debug, Serialize)]
pub struct TelemetryStats {
    pub queued: u64,
    pub sent: u64,
    pub dropped: u64,
    pub export_errors: u64,
}

/// What every sink keeps for one live span, in sink order.
pub(crate) struct OpenSpans(Vec<OpenSpan>);

impl Telemetry {
    pub fn new(
        endpoint: Option<String>,
        capacity: usize,
        config: StageConfig,
        level: TelemetryLevel,
    ) -> Self {
        // `SKIPPY_TELEMETRY_STDERR` promises a local debug sink that needs no
        // collector. It was checked in `enqueue`, which `emit` never reached
        // when the level was `Off` -- and the embedded runtime defaults to
        // `Off`. So the variable silently did nothing in exactly the
        // single-process case it exists for. Honour it as a level floor, and
        // the existing level checks then work unchanged. The OTLP file sink
        // is the same kind of local opt-in.
        let stats = Arc::new(TelemetryCounters::default());
        let config = Arc::new(config);
        let file = otlp_file::OtlpFileTraceSink::configured(config.clone(), stats.clone());
        let tracy = tracy_sink();
        let level = effective_level(
            level,
            stderr_sink::stderr_sink_enabled() || file.is_some() || tracy.is_some(),
        );
        let metrics_endpoint = endpoint.clone();
        let collector = (level != TelemetryLevel::Off)
            .then_some(endpoint)
            .flatten()
            .map(|endpoint| {
                otlp_collector::OtlpCollectorSink::start(
                    endpoint,
                    capacity,
                    stats.clone(),
                    config.clone(),
                )
            });
        let stderr = stderr_sink::StderrTraceSink::enabled(capacity, stats.clone());
        // Samples are debug telemetry: they go to the metrics file and the
        // collector's metrics service, and to any sink that plots them.
        let metrics = (level == TelemetryLevel::Debug)
            .then(|| {
                resource_metrics::OtlpMetricsSink::configured(
                    config.clone(),
                    std::env::var_os(resource_metrics::METRICS_FILE_ENV)
                        .filter(|path| !path.is_empty()),
                    metrics_endpoint.as_deref(),
                )
            })
            .flatten();
        let sinks: Vec<Box<dyn TraceSink>> = [
            collector.map(|sink| Box::new(sink) as Box<dyn TraceSink>),
            stderr.map(|sink| Box::new(sink) as Box<dyn TraceSink>),
            file.map(|sink| Box::new(sink) as Box<dyn TraceSink>),
            metrics.map(|sink| Box::new(sink) as Box<dyn TraceSink>),
            tracy,
        ]
        .into_iter()
        .flatten()
        .collect();
        let telemetry = Self::with_sinks(sinks, stats, config, level);
        log_events::start(&telemetry, metrics_endpoint.as_deref());
        if telemetry.is_debug_enabled() && telemetry.has_sink() {
            native_calls::install(&telemetry);
            resource_metrics::start(&telemetry);
        }
        telemetry
    }

    fn with_sinks(
        sinks: Vec<Box<dyn TraceSink>>,
        stats: Arc<TelemetryCounters>,
        config: Arc<StageConfig>,
        level: TelemetryLevel,
    ) -> Self {
        Self {
            sinks: sinks.into(),
            stats,
            config,
            level,
        }
    }

    fn has_sink(&self) -> bool {
        !self.sinks.is_empty()
    }

    /// Whether some sink wants events inside their span's record.
    fn has_trace_sink(&self) -> bool {
        self.sinks.iter().any(|sink| sink.folds_events_into_spans())
    }

    /// Whether process samples are worth taking: debug telemetry with a sink.
    pub(crate) fn records_metrics(&self) -> bool {
        self.level == TelemetryLevel::Debug && self.has_sink()
    }

    /// One process sample, to every sink.
    pub(crate) fn record_metrics(&self, metrics: &[Metric]) {
        self.sinks
            .iter()
            .for_each(|sink| sink.record_metrics(metrics));
    }

    /// A live span begins now on this thread; `end_span` takes what comes back.
    pub(crate) fn begin_span(&self, span: &SpanStart) -> OpenSpans {
        OpenSpans(
            self.sinks
                .iter()
                .map(|sink| sink.create_span_with(span))
                .collect(),
        )
    }

    /// The live span `begin_span` saw ends: export it with its frame's events.
    pub(crate) fn end_span(
        &self,
        open: OpenSpans,
        frame: &Frame,
        name: &str,
        mut attributes: BTreeMap<String, Value>,
        start_time_unix_nanos: u64,
        end_time_unix_nanos: u64,
        links: Vec<SpanContext>,
    ) {
        let events = frame.take_events();
        self.add_common_attributes(&mut attributes);
        let mut record = SpanRecord::of_frame(
            frame,
            name,
            attributes,
            start_time_unix_nanos,
            end_time_unix_nanos,
        );
        record.links = links;
        record.events = events;
        self.sinks
            .iter()
            .zip(open.0)
            .for_each(|(sink, open)| sink.end_span(open, &record));
        self.after_delivery();
    }

    /// The live span `begin_span` saw is not kept.
    pub(crate) fn discard_span(&self, open: OpenSpans) {
        self.sinks
            .iter()
            .zip(open.0)
            .for_each(|(sink, open)| sink.discard_span(open));
    }

    /// A point event. Trace sinks record it as an event on the active span;
    /// stderr keeps its standalone line.
    pub fn emit(&self, name: &str, mut attributes: BTreeMap<String, Value>) {
        if self.level == TelemetryLevel::Off {
            return;
        }
        self.add_common_attributes(&mut attributes);
        let frame = trace_context::current_frame_or_root();
        log_events::record_point_event(&frame, name, &attributes);
        self.record_event_on(
            &frame,
            SpanEvent {
                name: name.to_string(),
                time_unix_nanos: now_unix_nanos() as u64,
                attributes,
            },
        );
    }

    /// Tell every sink about `event` as it happens, then fold it into
    /// `frame`'s span for the sinks that read events there. When that span
    /// already ended or is full, they get it as an instant span beneath it.
    pub(crate) fn record_event_on(&self, frame: &Frame, event: SpanEvent) {
        self.sinks
            .iter()
            .for_each(|sink| sink.record_event(&event, frame));
        if !self.has_trace_sink() {
            return;
        }
        if let Err(event) = frame.record_event(event) {
            let record = SpanRecord::of_frame(
                &Frame::child(),
                &event.name,
                event.attributes,
                event.time_unix_nanos,
                event.time_unix_nanos + 1,
            );
            self.sinks
                .iter()
                .filter(|sink| sink.folds_events_into_spans())
                .for_each(|sink| sink.record_span(&record));
        }
    }

    /// A span measured by the caller, placed under the active span.
    pub fn emit_span(
        &self,
        name: &str,
        mut attributes: BTreeMap<String, Value>,
        start_time_unix_nanos: u64,
        end_time_unix_nanos: u64,
    ) {
        if self.level == TelemetryLevel::Off {
            return;
        }
        self.add_common_attributes(&mut attributes);
        self.deliver(SpanRecord::of_frame(
            &Frame::child(),
            name,
            attributes,
            start_time_unix_nanos,
            end_time_unix_nanos,
        ));
    }

    /// Export the span `frame` identifies, with the events recorded on it.
    pub(crate) fn emit_frame_span(
        &self,
        frame: &Frame,
        name: &str,
        mut attributes: BTreeMap<String, Value>,
        start_time_unix_nanos: u64,
        end_time_unix_nanos: u64,
        links: Vec<SpanContext>,
    ) {
        let events = frame.take_events();
        if self.level == TelemetryLevel::Off {
            return;
        }
        self.add_common_attributes(&mut attributes);
        let mut record = SpanRecord::of_frame(
            frame,
            name,
            attributes,
            start_time_unix_nanos,
            end_time_unix_nanos,
        );
        record.links = links;
        record.events = events;
        self.deliver(record);
    }

    pub(crate) fn emit_debug_frame_span(
        &self,
        frame: &Frame,
        name: &str,
        attributes: BTreeMap<String, Value>,
        start_time_unix_nanos: u64,
        end_time_unix_nanos: u64,
    ) {
        if self.level == TelemetryLevel::Debug {
            self.emit_frame_span(
                frame,
                name,
                attributes,
                start_time_unix_nanos,
                end_time_unix_nanos,
                Vec::new(),
            );
        } else {
            frame.take_events();
        }
    }

    /// End the process span and flush queued spans. Shutdown calls this; only
    /// the first call in the process does anything.
    pub fn finish_process(&self) {
        let process = trace_context::process();
        if self.level == TelemetryLevel::Off || !process.begin_finish() {
            return;
        }
        // Spans and GPU records still queued belong to calls that already ended.
        native_spans::emit_pending(self);
        gpu_commands::emit_pending(self);
        self.record_resources("shutdown");
        resource_metrics::finish();
        log_events::finish();
        let mut attributes = lifecycle_attrs(&self.config);
        attributes.insert("process.pid".to_string(), json!(std::process::id()));
        self.emit_frame_span(
            process.root(),
            "skippy.process",
            attributes,
            process.start_unix_nanos(),
            now_unix_nanos() as u64,
            Vec::new(),
        );
        self.flush(FLUSH_TIMEOUT);
    }

    /// Wait, bounded by `timeout` per sink, for queued spans to leave.
    fn flush(&self, timeout: Duration) {
        self.sinks.iter().for_each(|sink| sink.flush(timeout));
    }

    fn add_common_attributes(&self, attributes: &mut BTreeMap<String, Value>) {
        attributes
            .entry(attr::RUN_ID.to_string())
            .or_insert_with(|| json!(self.config.run_id));
        attributes
            .entry(attr::STAGE_ID.to_string())
            .or_insert_with(|| json!(self.config.stage_id));
        attributes.insert(
            metric::OTEL_DROPPED_EVENTS.to_string(),
            json!(self.stats.dropped.load(Ordering::Relaxed)),
        );
        attributes.insert(
            metric::OTEL_EXPORT_ERRORS.to_string(),
            json!(self.stats.export_errors.load(Ordering::Relaxed)),
        );
    }

    fn deliver(&self, record: SpanRecord) {
        self.sinks.iter().for_each(|sink| sink.record_span(&record));
        self.after_delivery();
    }

    /// Components that stop after the process span (the scheduler worker,
    /// say) end their spans during exit; deliver each one immediately
    /// because the process will not wait for the exporters.
    fn after_delivery(&self) {
        if trace_context::process().is_finished() {
            self.flush(LATE_SPAN_FLUSH_TIMEOUT);
        }
    }

    pub fn emit_debug(&self, name: &str, attributes: BTreeMap<String, Value>) {
        if self.level == TelemetryLevel::Debug {
            self.emit(name, attributes);
        }
    }

    pub fn emit_debug_span(
        &self,
        name: &str,
        attributes: BTreeMap<String, Value>,
        start_time_unix_nanos: u64,
        end_time_unix_nanos: u64,
    ) {
        if self.level == TelemetryLevel::Debug {
            self.emit_span(name, attributes, start_time_unix_nanos, end_time_unix_nanos);
        }
    }

    /// A debug span under `parent`, which may live on another thread.
    pub(crate) fn emit_debug_span_under(
        &self,
        parent: SpanContext,
        name: &str,
        mut attributes: BTreeMap<String, Value>,
        start_time_unix_nanos: u64,
        end_time_unix_nanos: u64,
    ) {
        if self.level != TelemetryLevel::Debug {
            return;
        }
        self.add_common_attributes(&mut attributes);
        self.deliver(SpanRecord::of_frame(
            &Frame::under(parent),
            name,
            attributes,
            start_time_unix_nanos,
            end_time_unix_nanos,
        ));
    }

    pub fn is_enabled(&self) -> bool {
        self.level != TelemetryLevel::Off
    }

    pub fn is_debug_enabled(&self) -> bool {
        self.level == TelemetryLevel::Debug
    }

    pub fn stats(&self) -> TelemetryStats {
        TelemetryStats {
            queued: self.stats.queued.load(Ordering::Relaxed),
            sent: self.stats.sent.load(Ordering::Relaxed),
            dropped: self.stats.dropped.load(Ordering::Relaxed),
            export_errors: self.stats.export_errors.load(Ordering::Relaxed),
        }
    }
}

#[cfg(test)]
impl Telemetry {
    /// Test-only: wire a `Telemetry` to an in-process capture receiver.
    ///
    /// Bypasses both the collector channel and the env-gated stderr sink so
    /// tests can assert exact emissions and level gating deterministically,
    /// without subprocess or stderr capture. The returned receiver yields
    /// [`CapturedEvent`]s in emission order; drain it with `try_recv`.
    pub(crate) fn captured(
        config: StageConfig,
        level: TelemetryLevel,
    ) -> (Self, std_mpsc::Receiver<CapturedEvent>) {
        let (sink, rx) = sink::CaptureSink::channel();
        (
            Self::with_sinks(
                vec![Box::new(sink)],
                Arc::new(TelemetryCounters::default()),
                Arc::new(config),
                level,
            ),
            rx,
        )
    }
}

/// Test-only: the smallest stage config a captured `Telemetry` accepts.
#[cfg(test)]
pub(crate) fn capture_stage_config() -> StageConfig {
    serde_json::from_value(json!({
        "run_id": "run",
        "topology_id": "topology",
        "model_id": "org/model:Q4_K_M",
        "stage_id": "stage-0",
        "stage_index": 0,
        "layer_start": 0,
        "layer_end": 4,
        "load_mode": "runtime-slice",
        "execution_contract": "",
        "bind_addr": "127.0.0.1:0",
    }))
    .expect("minimal stage config for telemetry capture")
}

#[cfg(test)]
mod capture_tests {
    use super::{BTreeMap, Telemetry, TelemetryLevel, capture_stage_config as capture_config};
    use serde_json::json;

    #[test]
    fn captured_sink_reports_normal_events_with_common_attributes_at_summary_level() {
        let (telemetry, rx) = Telemetry::captured(capture_config(), TelemetryLevel::Summary);
        telemetry.emit("stage.test_normal", BTreeMap::new());
        telemetry.emit_debug("stage.test_debug", BTreeMap::new());

        let event = rx.try_recv().expect("normal-level event captured");
        assert_eq!(event.event, "stage.test_normal");
        assert_eq!(event.attributes["skippy.run_id"], json!("run"));
        assert_eq!(event.attributes["skippy.stage_id"], json!("stage-0"));
        assert!(rx.try_recv().is_err(), "debug-level event must not fire");
    }

    #[test]
    fn captured_sink_reports_debug_spans_only_at_debug_level() {
        let (telemetry, rx) = Telemetry::captured(capture_config(), TelemetryLevel::Debug);
        telemetry.emit_debug_span("stage.test_span", BTreeMap::new(), 1, 2);

        let event = rx.try_recv().expect("debug span captured at debug level");
        assert_eq!(event.event, "stage.test_span");
        assert_eq!(event.start_time_unix_nanos, 1);
        assert_eq!(event.end_time_unix_nanos, 2);
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn captured_sink_emits_nothing_at_off_level() {
        let (telemetry, rx) = Telemetry::captured(capture_config(), TelemetryLevel::Off);
        telemetry.emit("stage.test_normal", BTreeMap::new());
        telemetry.emit_debug("stage.test_debug", BTreeMap::new());

        assert!(rx.try_recv().is_err(), "off level must suppress everything");
    }
}

pub fn lifecycle_attrs(config: &StageConfig) -> BTreeMap<String, Value> {
    BTreeMap::from([
        (attr::RUN_ID.to_string(), json!(config.run_id)),
        (attr::MODEL_ID.to_string(), json!(config.model_id)),
        (attr::TOPOLOGY_ID.to_string(), json!(config.topology_id)),
        (attr::STAGE_ID.to_string(), json!(config.stage_id)),
        (attr::STAGE_INDEX.to_string(), json!(config.stage_index)),
        (attr::LAYER_START.to_string(), json!(config.layer_start)),
        (attr::LAYER_END.to_string(), json!(config.layer_end)),
        (
            attr::LOAD_MODE.to_string(),
            serde_json::to_value(&config.load_mode).unwrap_or(Value::Null),
        ),
    ])
}

pub fn now_unix_nanos() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before Unix epoch")
        .as_nanos() as i64
}

#[cfg(test)]
mod tests {
    use super::{TelemetryLevel, effective_level};
    /// `SKIPPY_TELEMETRY_STDERR` is the only way to see record/lookup
    /// decisions in a single-process embedded run, and the embedded runtime
    /// defaults to `Off`. If the variable does not raise the level, the sink
    /// is unreachable and the variable is a no-op -- which is what it was.
    #[test]
    fn stderr_sink_raises_an_off_level_so_events_are_emitted() {
        assert_eq!(
            effective_level(TelemetryLevel::Off, true),
            TelemetryLevel::Debug
        );
    }

    #[test]
    fn stderr_sink_never_lowers_a_configured_level() {
        assert_eq!(
            effective_level(TelemetryLevel::Summary, true),
            TelemetryLevel::Summary
        );
        assert_eq!(
            effective_level(TelemetryLevel::Debug, true),
            TelemetryLevel::Debug
        );
    }

    #[test]
    fn level_is_untouched_without_the_stderr_sink() {
        assert_eq!(
            effective_level(TelemetryLevel::Off, false),
            TelemetryLevel::Off
        );
        assert_eq!(
            effective_level(TelemetryLevel::Summary, false),
            TelemetryLevel::Summary
        );
    }
}
