//! Tracy as a trace sink: build with `--features tracy`, set
//! `SKIPPY_TELEMETRY_TRACY=1`, and connect a Tracy profiler from this
//! machine; nothing is recorded until one connects.
//!
//! Tracy stamps time only when it is told, so this sink acts at both ends of
//! a live span. Every span runs on a Tracy fiber, which any thread may resume,
//! so a span may end on a thread other than the one it began on: a request's
//! spans share a lane taken for that request (`request-N`, returned when the
//! request ends) and every other span runs on a lane named after the thread
//! that began it. Metal command execution becomes GPU zones in a context
//! whose clock is Unix nanoseconds, process samples and resource snapshots
//! become plots, a span's own resource window becomes its zone text, and
//! other events become messages.

use std::{
    collections::{BTreeMap, HashMap},
    env,
    ffi::{CStr, CString},
    sync::Mutex,
    thread::{self, ThreadId},
    time::Duration,
};

use opentelemetry_proto::tonic::metrics::v1::{Metric, metric, number_data_point};
use serde_json::Value;
use tracy_client::{Client, GpuContext, GpuContextType, PlotName, sys};

use super::{
    now_unix_nanos,
    sink::{OpenSpan, SpanRecord, SpanStart, TraceSink},
    trace_context::{Frame, SpanEvent},
};

const ENV: &str = "SKIPPY_TELEMETRY_TRACY";
const REQUEST_ROOT: &str = "stage.openai_http_request";
const RESOURCES_EVENT: &str = "skippy.resources";
const RESOURCES_PREFIX: &str = "skippy.resources.";
const KV_CELLS: &str = "skippy.kv.used_cells";
const SPAN_WINDOW_PREFIX: &str = "skippy.span.";
const GPU_EXECUTE_SUFFIX: &str = ".execute";

/// Whether `SKIPPY_TELEMETRY_TRACY` asks for this sink.
pub(super) fn tracy_enabled() -> bool {
    enabled_from_value(env::var(ENV).ok().as_deref())
}

fn enabled_from_value(value: Option<&str>) -> bool {
    value.is_some_and(|value| {
        !matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "" | "0" | "false" | "no" | "off"
        )
    })
}

/// A Tracy fiber. Tracy knows a fiber by its name's address, so the name
/// lives as long as the process.
#[derive(Clone, Copy)]
struct Lane(&'static CStr);

impl Lane {
    fn named(name: String) -> Self {
        Self(Box::leak(
            CString::new(name).unwrap_or_default().into_boxed_c_str(),
        ))
    }

    /// Run `act` with this lane as the calling thread's current fiber.
    fn resume(self, act: impl FnOnce()) {
        // SAFETY: the name outlives the process; enter and leave pair on this thread.
        unsafe { sys::___tracy_fiber_enter(self.0.as_ptr()) };
        act();
        unsafe { sys::___tracy_fiber_leave() };
    }
}

#[derive(Default)]
struct Lanes {
    /// The lane each live span runs on, by span id.
    by_span: HashMap<[u8; 8], Lane>,
    by_thread: HashMap<ThreadId, Lane>,
    free_requests: Vec<Lane>,
    requests_made: usize,
}

impl Lanes {
    /// A child joins its parent's lane; a request root takes a request
    /// lane; anything else runs on the lane of the thread that began it.
    fn assign(&mut self, span: &SpanStart) -> Lane {
        let parent_lane = span
            .parent
            .and_then(|parent| self.by_span.get(&parent).copied());
        let lane = match parent_lane {
            Some(lane) => lane,
            None if span.name == REQUEST_ROOT => self.request_lane(),
            None => self.thread_lane(),
        };
        self.by_span.insert(span.context.span_id, lane);
        lane
    }

    fn request_lane(&mut self) -> Lane {
        self.free_requests.pop().unwrap_or_else(|| {
            let lane = Lane::named(format!("request-{}", self.requests_made));
            self.requests_made += 1;
            lane
        })
    }

    fn thread_lane(&mut self) -> Lane {
        let thread = thread::current();
        *self.by_thread.entry(thread.id()).or_insert_with(|| {
            Lane::named(format!(
                "{} {:?}",
                thread.name().unwrap_or("thread"),
                thread.id()
            ))
        })
    }

    /// Forget a span that ended; a request root hands its lane back.
    fn release(&mut self, span_id: [u8; 8], name: &str) {
        if let Some(lane) = self.by_span.remove(&span_id)
            && name == REQUEST_ROOT
        {
            self.free_requests.push(lane);
        }
    }
}

/// What the sink keeps while a span is live.
struct Open {
    zone: sys::___tracy_c_zone_context,
    lane: Lane,
}

pub(super) struct TracySink {
    client: Client,
    lanes: Mutex<Lanes>,
    plots: Mutex<HashMap<String, PlotName>>,
    /// Metal execution, timed in Unix nanoseconds like every span here.
    gpu: Option<GpuContext>,
}

impl TracySink {
    pub(super) fn enabled() -> Option<Self> {
        tracy_enabled().then(Self::start)
    }

    fn start() -> Self {
        let client = Client::start();
        let gpu = client
            .clone()
            .new_gpu_context(
                Some("metal"),
                GpuContextType::Invalid,
                now_unix_nanos(),
                1.0,
            )
            .ok();
        Self {
            client,
            lanes: Mutex::new(Lanes::default()),
            plots: Mutex::new(HashMap::new()),
            gpu,
        }
    }

    fn lanes(&self) -> std::sync::MutexGuard<'_, Lanes> {
        self.lanes
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn zone_begin(name: &str) -> sys::___tracy_c_zone_context {
        // SAFETY: Tracy copies the location strings; the zone is active (1).
        unsafe {
            let location = sys::___tracy_alloc_srcloc_name(
                0,
                c"".as_ptr(),
                0,
                name.as_ptr().cast(),
                name.len(),
                name.as_ptr().cast(),
                name.len(),
                0,
            );
            sys::___tracy_emit_zone_begin_alloc(location, 1)
        }
    }

    fn zone_end(zone: sys::___tracy_c_zone_context) {
        // SAFETY: `zone` came from `zone_begin` and ends once, on its lane.
        unsafe { sys::___tracy_emit_zone_end(zone) };
    }

    fn zone_text(zone: sys::___tracy_c_zone_context, text: &str) {
        // SAFETY: Tracy copies the text; the zone is still open on its lane.
        unsafe { sys::___tracy_emit_zone_text(zone, text.as_ptr().cast(), text.len()) };
    }

    fn close(&self, open: OpenSpan, span: &SpanRecord) {
        let Some(open) = open.take::<Open>() else {
            return;
        };
        let window = span_window_text(&span.attributes);
        open.lane.resume(|| {
            if let Some(text) = window.as_deref() {
                Self::zone_text(open.zone, text);
            }
            Self::zone_end(open.zone);
        });
        self.lanes().release(span.context.span_id, &span.name);
    }

    fn plot(&self, name: &str, value: f64) {
        let plot = *self
            .plots
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .entry(name.to_string())
            .or_insert_with(|| PlotName::new_leak(name.to_string()));
        self.client.plot(plot, value);
    }

    /// Plot every number under `prefix`, named without it.
    fn plot_numbers(&self, prefix: &str, attributes: &BTreeMap<String, Value>) {
        attributes
            .iter()
            .filter_map(|(key, value)| Some((key.strip_prefix(prefix)?, value.as_f64()?)))
            .for_each(|(name, value)| self.plot(name, value));
    }

    /// A span whose GPU execution window is already known, as a GPU zone.
    fn gpu_zone(&self, span: &SpanRecord) {
        let Some(gpu) = self.gpu.as_ref() else {
            return;
        };
        let Ok(mut zone) = gpu.span_alloc(&span.name, "", "", 0) else {
            return;
        };
        zone.end_zone();
        zone.upload_timestamp_start(span.start_time_unix_nanos as i64);
        zone.upload_timestamp_end(span.end_time_unix_nanos as i64);
    }
}

impl TraceSink for TracySink {
    fn create_span_with(&self, span: &SpanStart) -> OpenSpan {
        let lane = self.lanes().assign(span);
        let mut zone = None;
        lane.resume(|| zone = Some(Self::zone_begin(span.name)));
        zone.map_or(OpenSpan::NONE, |zone| OpenSpan::keep(Open { zone, lane }))
    }

    fn end_span(&self, open: OpenSpan, span: &SpanRecord) {
        self.close(open, span);
        if let Some(cells) = span.attributes.get(KV_CELLS).and_then(Value::as_f64) {
            self.plot("kv.used_cells", cells);
        }
    }

    fn discard_span(&self, open: OpenSpan) {
        if let Some(open) = open.take::<Open>() {
            open.lane.resume(|| Self::zone_end(open.zone));
        }
    }

    /// Only Metal execution is shown after the fact; native call spans are
    /// nested, which a GPU context cannot take out of order.
    fn record_span(&self, span: &SpanRecord) {
        if span.name.ends_with(GPU_EXECUTE_SUFFIX) {
            self.gpu_zone(span);
        }
    }

    fn record_event(&self, event: &SpanEvent, _: &Frame) {
        if event.name == RESOURCES_EVENT {
            self.plot_numbers(RESOURCES_PREFIX, &event.attributes);
        } else {
            self.client.message(&event.name, 0);
        }
    }

    /// Every gauge point becomes a plot, named by the metric and the point's
    /// attributes (`skippy.gpu.busy{device=MTL0}`), one curve per series.
    fn record_metrics(&self, metrics: &[Metric]) {
        metrics
            .iter()
            .flat_map(|metric| {
                gauge_points(metric)
                    .map(move |(series, value)| (plot_name(&metric.name, &series), value))
            })
            .for_each(|(name, value)| self.plot(&name, value));
    }

    fn folds_events_into_spans(&self) -> bool {
        false
    }

    /// Tracy streams as it goes; there is no queue to wait on.
    fn flush(&self, _: Duration) {}
}

/// A span's own resource window (`skippy.span.*`), as one line of zone text.
fn span_window_text(attributes: &BTreeMap<String, Value>) -> Option<String> {
    let parts = attributes
        .iter()
        .filter_map(|(key, value)| Some((key.strip_prefix(SPAN_WINDOW_PREFIX)?, value.as_f64()?)))
        .map(|(name, value)| format!("{name} {value:.2}"))
        .collect::<Vec<_>>();
    (!parts.is_empty()).then(|| parts.join("  "))
}

/// The number points of a gauge, each with its attributes as `key=value`.
fn gauge_points(metric: &Metric) -> impl Iterator<Item = (Vec<String>, f64)> + '_ {
    let points = match metric.data.as_ref() {
        Some(metric::Data::Gauge(gauge)) => gauge.data_points.as_slice(),
        _ => &[],
    };
    points.iter().filter_map(|point| {
        let value = match point.value.as_ref()? {
            number_data_point::Value::AsDouble(value) => *value,
            number_data_point::Value::AsInt(value) => *value as f64,
        };
        let series = point
            .attributes
            .iter()
            .map(|attribute| {
                let value = attribute
                    .value
                    .as_ref()
                    .and_then(|value| value.value.as_ref())
                    .map_or_else(String::new, |value| match value {
                        opentelemetry_proto::tonic::common::v1::any_value::Value::StringValue(
                            text,
                        ) => text.clone(),
                        other => format!("{other:?}"),
                    });
                format!("{}={value}", attribute.key)
            })
            .collect();
        Some((series, value))
    })
}

fn plot_name(metric: &str, series: &[String]) -> String {
    if series.is_empty() {
        metric.to_string()
    } else {
        format!("{metric}{{{}}}", series.join(","))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use opentelemetry_proto::tonic::{
        common::v1::{AnyValue, KeyValue, any_value},
        metrics::v1::{Gauge, NumberDataPoint},
    };
    use serde_json::json;

    use super::*;
    use crate::telemetry::{
        Telemetry, TelemetryCounters, TelemetryLevel,
        active_span::{ActiveSpan, SpanLevel},
        capture_stage_config,
        trace_context::SpanContext,
    };

    fn start(name: &'static str, span_id: u8, parent: Option<u8>) -> SpanStart {
        SpanStart {
            name,
            level: SpanLevel::Summary,
            context: SpanContext {
                trace_id: [1; 16],
                span_id: [span_id; 8],
            },
            parent: parent.map(|parent| [parent; 8]),
            start_time_unix_nanos: 1,
        }
    }

    #[test]
    fn a_request_and_its_children_share_a_lane_that_is_reused_after_it_ends() {
        let mut lanes = Lanes::default();
        let request = lanes.assign(&start(REQUEST_ROOT, 1, None));
        let child = lanes.assign(&start("stage.openai_prefill", 2, Some(1)));
        let worker = lanes.assign(&start("stage.scheduler_feature_iteration", 3, None));
        assert!(std::ptr::eq(request.0, child.0));
        assert!(!std::ptr::eq(request.0, worker.0));
        assert_eq!(lanes.requests_made, 1);

        lanes.release([2; 8], "stage.openai_prefill");
        lanes.release([1; 8], REQUEST_ROOT);
        let next = lanes.assign(&start(REQUEST_ROOT, 4, None));
        assert!(
            std::ptr::eq(request.0, next.0),
            "a freed request lane is taken again"
        );
        assert_eq!(lanes.requests_made, 1);
    }

    #[test]
    fn two_live_requests_take_two_lanes() {
        let mut lanes = Lanes::default();
        let first = lanes.assign(&start(REQUEST_ROOT, 1, None));
        let second = lanes.assign(&start(REQUEST_ROOT, 2, None));
        assert!(!std::ptr::eq(first.0, second.0));
        assert_eq!(lanes.requests_made, 2);
    }

    /// The real client, with no profiler attached: every call must be safe
    /// to make, from any thread, in any order the span API allows.
    #[test]
    fn spans_begin_and_end_on_any_thread_through_the_real_client() {
        let telemetry = Telemetry::with_sinks(
            vec![Box::new(TracySink::start())],
            Arc::new(TelemetryCounters::default()),
            Arc::new(capture_stage_config()),
            TelemetryLevel::Debug,
        );
        let request = ActiveSpan::start(Some(&telemetry), REQUEST_ROOT, SpanLevel::Summary);
        let child = {
            let _request = request.enter();
            ActiveSpan::start(Some(&telemetry), "stage.openai_prefill", SpanLevel::Debug)
        };
        let worker = ActiveSpan::start_under_process(
            Some(&telemetry),
            "stage.scheduler_feature_iteration",
            SpanLevel::Debug,
        );
        telemetry.emit(
            RESOURCES_EVENT,
            BTreeMap::from([(
                format!("{RESOURCES_PREFIX}process.footprint_bytes"),
                json!(1_u64 << 30),
            )]),
        );
        telemetry.emit("skippy.kv.lookup", BTreeMap::new());
        telemetry.emit_debug_span("mtl.execute", BTreeMap::new(), 10, 20);
        let mut worker = worker;
        worker.extend_attributes([(KV_CELLS.to_string(), json!(512))]);
        worker.end();
        thread::spawn(move || {
            child.end();
            request.end();
        })
        .join()
        .expect("spans end on another thread");
        ActiveSpan::start(Some(&telemetry), "stage.nothing", SpanLevel::Summary).discard();
        assert!(telemetry.sinks.len() == 1);
    }

    fn point(value: f64, attributes: &[(&str, &str)]) -> NumberDataPoint {
        NumberDataPoint {
            attributes: attributes
                .iter()
                .map(|(key, value)| KeyValue {
                    key: (*key).to_string(),
                    value: Some(AnyValue {
                        value: Some(any_value::Value::StringValue((*value).to_string())),
                    }),
                    ..Default::default()
                })
                .collect(),
            value: Some(number_data_point::Value::AsDouble(value)),
            ..Default::default()
        }
    }

    #[test]
    fn a_gauge_point_is_a_plot_named_by_its_series() {
        let metric = Metric {
            name: "skippy.gpu.busy".to_string(),
            data: Some(metric::Data::Gauge(Gauge {
                data_points: vec![point(0.5, &[("device", "MTL0")]), point(0.25, &[])],
            })),
            ..Default::default()
        };
        let plots = gauge_points(&metric)
            .map(|(series, value)| (plot_name(&metric.name, &series), value))
            .collect::<Vec<_>>();
        assert_eq!(
            plots,
            vec![
                ("skippy.gpu.busy{device=MTL0}".to_string(), 0.5),
                ("skippy.gpu.busy".to_string(), 0.25),
            ]
        );
    }

    #[test]
    fn a_span_window_becomes_one_line_of_text() {
        let attributes = BTreeMap::from([
            ("skippy.span.cpu_ms".to_string(), json!(1.25)),
            ("skippy.span.gpu_execute_ms".to_string(), json!(8.0)),
            ("skippy.other".to_string(), json!(3)),
        ]);
        assert_eq!(
            span_window_text(&attributes).as_deref(),
            Some("cpu_ms 1.25  gpu_execute_ms 8.00")
        );
        assert_eq!(span_window_text(&BTreeMap::new()), None);
    }

    #[test]
    fn the_variable_must_be_set_and_truthy() {
        assert!(enabled_from_value(Some("1")));
        assert!(enabled_from_value(Some("yes")));
        assert!(!enabled_from_value(Some("0")));
        assert!(!enabled_from_value(Some(" off ")));
        assert!(!enabled_from_value(Some("")));
        assert!(!enabled_from_value(None));
    }
}
