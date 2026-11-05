#[cfg(test)]
use std::sync::mpsc as std_mpsc;
use std::{
    collections::{BTreeMap, VecDeque},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use opentelemetry_proto::tonic::collector::trace::v1::{
    ExportTraceServiceRequest, trace_service_client::TraceServiceClient,
};
use serde::Serialize;
use serde_json::{Value, json};
use skippy_metrics::{attr, metric};
use skippy_protocol::StageConfig;
use tokio::{
    sync::mpsc,
    time::{sleep, timeout},
};

pub(crate) mod active_span;
mod gpu_commands;
pub(crate) mod log_events;
mod native_calls;
mod native_spans;
mod otlp_encoding;
mod otlp_file;
mod resource_attrs;
pub(crate) mod resource_metrics;
mod resource_usage;
mod span_components;
mod stderr_sink;
pub(crate) mod trace_context;

use otlp_encoding::{event_to_span, resource_spans_by_component};
use stderr_sink::{StderrTelemetryEvent, StderrTelemetrySink, stderr_sink_enabled};
use trace_context::{Frame, SpanContext, SpanEvent};

/// How long shutdown waits for queued spans to reach their sinks.
const FLUSH_TIMEOUT: Duration = Duration::from_secs(5);
/// Bound on delivering a span that ends after the process span.
const LATE_SPAN_FLUSH_TIMEOUT: Duration = Duration::from_millis(500);

/// A local debug sink (stderr or OTLP file) acts as a floor on the configured
/// level.
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
const EXPORT_TIMEOUT: Duration = Duration::from_secs(10);
const EXPORT_BATCH_MAX: usize = 128;
const RETRY_BUFFER_CAPACITY: usize = 8192;
const RETRY_BACKOFF: Duration = Duration::from_millis(100);

#[derive(Clone)]
pub struct Telemetry {
    tx: Option<mpsc::Sender<TelemetryEvent>>,
    stderr: Option<Arc<StderrTelemetrySink>>,
    otlp_file: Option<&'static otlp_file::OtlpFileSink>,
    stats: Arc<TelemetryCounters>,
    config: Arc<StageConfig>,
    level: TelemetryLevel,
    /// Test-only in-process capture channel; `None` in production. Sibling
    /// tests receive events here instead of the private stderr sink, so
    /// `StderrTelemetryEvent` and its fields stay module-private.
    #[cfg(test)]
    capture: Option<std_mpsc::SyncSender<CapturedEvent>>,
}

/// Test-only emission shape handed to sibling-module tests. Mirrors
/// `StderrTelemetryEvent` without exposing the production struct or its
/// fields beyond this module.
#[cfg(test)]
pub(crate) struct CapturedEvent {
    pub event: String,
    pub attributes: BTreeMap<String, Value>,
    pub start_time_unix_nanos: u64,
    pub end_time_unix_nanos: u64,
    /// The exported span's identity and parent; `None` for a point event.
    pub span: Option<(SpanContext, Option<[u8; 8]>)>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TelemetryLevel {
    Off,
    Summary,
    Debug,
}

#[derive(Default)]
struct TelemetryCounters {
    queued: AtomicU64,
    sent: AtomicU64,
    dropped: AtomicU64,
    export_errors: AtomicU64,
}

#[derive(Clone, Debug, Serialize)]
pub struct TelemetryStats {
    pub queued: u64,
    pub sent: u64,
    pub dropped: u64,
    pub export_errors: u64,
}

/// One finished span, placed in the process trace.
#[derive(Clone)]
struct TelemetryEvent {
    context: SpanContext,
    parent: Option<[u8; 8]>,
    name: String,
    attributes: BTreeMap<String, Value>,
    start_time_unix_nanos: u64,
    end_time_unix_nanos: u64,
    links: Vec<SpanContext>,
    events: Vec<SpanEvent>,
}

impl TelemetryEvent {
    fn span(
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
        let otlp_file = otlp_file::configured();
        let level = effective_level(level, stderr_sink_enabled() || otlp_file.is_some());
        let stats = Arc::new(TelemetryCounters::default());
        let config = Arc::new(config);
        let metrics_endpoint = endpoint.clone();
        let tx = (level != TelemetryLevel::Off)
            .then_some(endpoint)
            .flatten()
            .map(|endpoint| {
                let (tx, rx) = mpsc::channel(capacity);
                tokio::spawn(telemetry_export_loop(
                    endpoint,
                    rx,
                    stats.clone(),
                    config.clone(),
                ));
                tx
            });

        let telemetry = Self {
            tx,
            stderr: stderr_sink_enabled()
                .then(|| StderrTelemetrySink::new(capacity))
                .flatten()
                .map(Arc::new),
            otlp_file,
            stats,
            config,
            level,
            #[cfg(test)]
            capture: None,
        };
        log_events::start(&telemetry, metrics_endpoint.as_deref());
        if telemetry.is_debug_enabled() && telemetry.has_sink() {
            native_calls::install(&telemetry);
            resource_metrics::start(&telemetry, metrics_endpoint.as_deref());
        }
        telemetry
    }

    fn has_sink(&self) -> bool {
        self.tx.is_some() || self.stderr.is_some() || self.otlp_file.is_some()
    }

    /// A point event. Trace sinks record it as an event on the active span;
    /// stderr keeps its standalone line.
    pub fn emit(&self, name: &str, mut attributes: BTreeMap<String, Value>) {
        if self.level == TelemetryLevel::Off {
            return;
        }
        self.add_common_attributes(&mut attributes);
        let now = now_unix_nanos() as u64;
        #[cfg(test)]
        if self.capture(name, &attributes, now, now + 1, None) {
            return;
        }
        self.send_stderr(name, &attributes, now, now + 1);
        let frame = trace_context::current_frame_or_root();
        log_events::record_point_event(&frame, name, &attributes);
        if !self.has_trace_sink() {
            return;
        }
        let event = SpanEvent {
            name: name.to_string(),
            time_unix_nanos: now,
            attributes,
        };
        if let Err(event) = frame.record_event(event) {
            // The active span already ended or is full: keep the event as an
            // instant span beneath it instead of losing it.
            self.export(TelemetryEvent::span(
                &Frame::child(),
                &event.name,
                event.attributes,
                now,
                now + 1,
            ));
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
        self.deliver(TelemetryEvent::span(
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
        let mut record = TelemetryEvent::span(
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

    /// Wait, bounded by `timeout`, for queued spans to reach their sinks.
    fn flush(&self, timeout: Duration) {
        let deadline = Instant::now() + timeout;
        if let Some(file) = self.otlp_file {
            file.flush(timeout);
        }
        if self.tx.is_some() {
            without_starving_runtime(|| {
                while self.stats.sent.load(Ordering::Relaxed)
                    < self.stats.queued.load(Ordering::Relaxed)
                    && Instant::now() < deadline
                {
                    thread::sleep(Duration::from_millis(20));
                }
            });
        }
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

    fn deliver(&self, record: TelemetryEvent) {
        #[cfg(test)]
        if self.capture(
            &record.name,
            &record.attributes,
            record.start_time_unix_nanos,
            record.end_time_unix_nanos,
            Some((record.context, record.parent)),
        ) {
            return;
        }
        self.send_stderr(
            &record.name,
            &record.attributes,
            record.start_time_unix_nanos,
            record.end_time_unix_nanos,
        );
        self.export(record);
        // Components that stop after the process span (the scheduler worker,
        // say) end their spans during exit; deliver each one immediately
        // because the process will not wait for the exporters.
        if trace_context::process().is_finished() {
            self.flush(LATE_SPAN_FLUSH_TIMEOUT);
        }
    }

    fn has_trace_sink(&self) -> bool {
        self.tx.is_some() || self.otlp_file.is_some()
    }

    fn send_stderr(
        &self,
        name: &str,
        attributes: &BTreeMap<String, Value>,
        start_time_unix_nanos: u64,
        end_time_unix_nanos: u64,
    ) {
        let Some(stderr) = self.stderr.as_ref() else {
            return;
        };
        if !stderr.try_send(StderrTelemetryEvent {
            event: name.to_string(),
            attributes: attributes.clone(),
            start_time_unix_nanos,
            end_time_unix_nanos,
        }) {
            self.stats.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn export(&self, record: TelemetryEvent) {
        if let Some(file) = self.otlp_file
            && !file.try_send(&self.config, event_to_span(&record, &self.stats))
        {
            self.stats.dropped.fetch_add(1, Ordering::Relaxed);
        }
        let Some(tx) = self.tx.as_ref() else {
            return;
        };
        match tx.try_send(record) {
            Ok(()) => {
                self.stats.queued.fetch_add(1, Ordering::Relaxed);
            }
            Err(_) => {
                self.stats.dropped.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Test-only capture takes precedence so sibling tests observe exact
    /// emissions without a collector or the env-gated stderr sink.
    #[cfg(test)]
    fn capture(
        &self,
        name: &str,
        attributes: &BTreeMap<String, Value>,
        start_time_unix_nanos: u64,
        end_time_unix_nanos: u64,
        span: Option<(SpanContext, Option<[u8; 8]>)>,
    ) -> bool {
        let Some(capture) = self.capture.as_ref() else {
            return false;
        };
        let _ = capture.try_send(CapturedEvent {
            event: name.to_string(),
            attributes: attributes.clone(),
            start_time_unix_nanos,
            end_time_unix_nanos,
            span,
        });
        true
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
        self.deliver(TelemetryEvent::span(
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
        let (tx, rx) = std_mpsc::sync_channel(256);
        (
            Self {
                tx: None,
                stderr: None,
                otlp_file: None,
                stats: Arc::new(TelemetryCounters::default()),
                config: Arc::new(config),
                level,
                #[cfg(test)]
                capture: Some(tx),
            },
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

/// Run a blocking `wait` for the gRPC exporter, a task on the caller's
/// runtime. Blocking a runtime worker in place can starve that task: woken
/// from this worker, it sits in the worker's LIFO slot, which other workers
/// never steal. `block_in_place` hands the worker's queue to another thread
/// first. A current-thread runtime cannot run the exporter while blocked, so
/// waiting there would only stall.
fn without_starving_runtime(wait: impl FnOnce()) {
    match tokio::runtime::Handle::try_current().map(|handle| handle.runtime_flavor()) {
        Ok(tokio::runtime::RuntimeFlavor::MultiThread) => tokio::task::block_in_place(wait),
        Ok(_) => {}
        Err(_) => wait(),
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

async fn telemetry_export_loop(
    endpoint: String,
    mut rx: mpsc::Receiver<TelemetryEvent>,
    stats: Arc<TelemetryCounters>,
    config: Arc<StageConfig>,
) {
    let mut client = None;
    let mut retry_buffer = VecDeque::new();

    loop {
        let Some(first_event) = next_export_event(&mut rx, &mut retry_buffer).await else {
            break;
        };
        let mut batch = Vec::with_capacity(EXPORT_BATCH_MAX);
        batch.push(first_event);
        fill_export_batch(&mut rx, &mut retry_buffer, &mut batch);

        let event_count = batch.len() as u64;
        if export_batch(&endpoint, &mut client, &config, &stats, &batch).await {
            stats.sent.fetch_add(event_count, Ordering::Relaxed);
        } else {
            stats
                .export_errors
                .fetch_add(event_count, Ordering::Relaxed);
            let dropped = requeue_failed_batch(&mut retry_buffer, batch);
            if dropped > 0 {
                stats.dropped.fetch_add(dropped as u64, Ordering::Relaxed);
            }
            sleep(RETRY_BACKOFF).await;
        }
    }
}

async fn next_export_event(
    rx: &mut mpsc::Receiver<TelemetryEvent>,
    retry_buffer: &mut VecDeque<TelemetryEvent>,
) -> Option<TelemetryEvent> {
    match retry_buffer.pop_front() {
        Some(event) => Some(event),
        None => rx.recv().await,
    }
}

fn fill_export_batch(
    rx: &mut mpsc::Receiver<TelemetryEvent>,
    retry_buffer: &mut VecDeque<TelemetryEvent>,
    batch: &mut Vec<TelemetryEvent>,
) {
    while batch.len() < EXPORT_BATCH_MAX {
        if let Some(event) = retry_buffer.pop_front() {
            batch.push(event);
            continue;
        }
        match rx.try_recv() {
            Ok(event) => batch.push(event),
            Err(_) => break,
        }
    }
}

async fn export_batch(
    endpoint: &str,
    client: &mut Option<TraceServiceClient<tonic::transport::Channel>>,
    config: &StageConfig,
    stats: &TelemetryCounters,
    events: &[TelemetryEvent],
) -> bool {
    if client.is_none() {
        match timeout(
            EXPORT_TIMEOUT,
            TraceServiceClient::connect(endpoint.to_string()),
        )
        .await
        {
            Ok(Ok(connected)) => {
                *client = Some(connected);
            }
            Ok(Err(_)) | Err(_) => return false,
        }
    }

    let request = ExportTraceServiceRequest {
        resource_spans: resource_spans_by_component(
            config,
            events
                .iter()
                .map(|event| event_to_span(event, stats))
                .collect(),
        ),
    };

    let export = client
        .as_mut()
        .expect("client is connected above")
        .export(request);
    match timeout(EXPORT_TIMEOUT, export).await {
        Ok(Ok(_)) => true,
        Ok(Err(_)) | Err(_) => {
            *client = None;
            false
        }
    }
}

fn requeue_failed_batch(
    retry_buffer: &mut VecDeque<TelemetryEvent>,
    batch: Vec<TelemetryEvent>,
) -> usize {
    let overflow = retry_buffer
        .len()
        .saturating_add(batch.len())
        .saturating_sub(RETRY_BUFFER_CAPACITY);
    for _ in 0..overflow {
        retry_buffer.pop_back();
    }
    for event in batch.into_iter().rev() {
        retry_buffer.push_front(event);
    }
    overflow
}

#[cfg(test)]
mod tests {
    use super::{TelemetryLevel, effective_level, without_starving_runtime};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{Duration, Instant};

    /// Block on a one-worker runtime until a task spawned from that worker
    /// runs, or `patience` passes; true when the task ran.
    fn spawned_task_runs_while_waiting(
        patience: Duration,
        wait: fn(Box<dyn FnOnce() + '_>),
    ) -> bool {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .build()
            .unwrap();
        runtime.block_on(async move {
            tokio::spawn(async move {
                let ran = Arc::new(AtomicBool::new(false));
                let task_ran = Arc::clone(&ran);
                tokio::spawn(async move { task_ran.store(true, Ordering::SeqCst) });
                let deadline = Instant::now() + patience;
                wait(Box::new(|| {
                    while !ran.load(Ordering::SeqCst) && Instant::now() < deadline {
                        std::thread::sleep(Duration::from_millis(1));
                    }
                }));
                ran.load(Ordering::SeqCst)
            })
            .await
            .unwrap()
        })
    }

    #[test]
    fn waiting_on_a_runtime_worker_lets_the_workers_tasks_run() {
        assert!(spawned_task_runs_while_waiting(
            Duration::from_secs(5),
            |wait| without_starving_runtime(wait)
        ));
    }

    #[test]
    fn waiting_in_place_on_a_runtime_worker_starves_its_tasks() {
        // The failure the helper exists for; if this ever passes, tokio has
        // made the LIFO slot stealable.
        assert!(!spawned_task_runs_while_waiting(
            Duration::from_millis(200),
            |wait| wait()
        ));
    }

    #[test]
    fn a_current_thread_runtime_does_not_stall_waiting() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let started = Instant::now();
        runtime.block_on(async {
            without_starving_runtime(|| std::thread::sleep(Duration::from_secs(5)));
        });
        assert!(started.elapsed() < Duration::from_secs(1));
    }

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
