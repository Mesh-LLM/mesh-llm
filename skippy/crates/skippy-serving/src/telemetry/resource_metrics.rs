//! Resource gauges for debugging throughput: CPU, GPU and memory, sampled on
//! a timer and exported as OTLP metrics, plus snapshots attached to the trace
//! at chosen points (model ready, server start, each request, shutdown).
//!
//! The sampler runs only with debug telemetry and a metrics destination: the
//! OTLP endpoint traces already go to, and the JSON-lines file named by
//! `SKIPPY_TELEMETRY_OTLP_METRICS_FILE`. GPU gauges come from the command
//! buffer timings native tracing reports, and the device's own allocation
//! counters; runtime gauges come from counters the scheduler bumps as it
//! works, which cost one relaxed atomic each.

use std::{
    collections::BTreeMap,
    env,
    sync::{
        Mutex, OnceLock, PoisonError,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use opentelemetry_proto::tonic::metrics::v1::Metric;
use serde_json::{Value, json};
use skippy_runtime::{BackendDeviceType, backend_devices_untraced};

use super::{
    Telemetry, TelemetryLevel, now_unix_nanos,
    resource_usage::{Paging, ProcessUsage, SystemMemory, process_usage, system_memory},
    trace_context::{Frame, SpanEvent},
};

pub(crate) mod gpu_op_classes;
pub(crate) mod kv_tiers;
mod otlp_metrics;

use gpu_op_classes::{GraphShape, OpClassCounters};
use kv_tiers::KvTiers;
use otlp_metrics::{MetricSinks, gauge, point};

/// Where sampled metrics are appended, one OTLP/JSON request per line.
pub(super) const METRICS_FILE_ENV: &str = "SKIPPY_TELEMETRY_OTLP_METRICS_FILE";
/// Sampling period in milliseconds; 250 when unset.
pub(super) const METRICS_INTERVAL_ENV: &str = "SKIPPY_TELEMETRY_METRICS_INTERVAL_MS";
const DEFAULT_INTERVAL: Duration = Duration::from_millis(250);
const MIN_INTERVAL: Duration = Duration::from_millis(20);
/// Marks a gauge with no reading yet.
const UNSET: u64 = u64::MAX;

static GPU_EXECUTE_NANOS: AtomicU64 = AtomicU64::new(0);
static GPU_QUEUE_NANOS: AtomicU64 = AtomicU64::new(0);
static GPU_COMMANDS: AtomicU64 = AtomicU64::new(0);
static RUNTIME_BUSY_NANOS: AtomicU64 = AtomicU64::new(0);
static DECODE_TOKENS: AtomicU64 = AtomicU64::new(0);
static SCHEDULER_RUNNING: AtomicU64 = AtomicU64::new(UNSET);
static SCHEDULER_WAITING: AtomicU64 = AtomicU64::new(UNSET);
static SCHEDULER_BATCH: AtomicU64 = AtomicU64::new(UNSET);
static KV_USED_CELLS: AtomicU64 = AtomicU64::new(UNSET);

/// One GPU command buffer: how long the GPU ran it, and how long it waited
/// between being committed and starting.
pub(crate) fn record_gpu_command(execute: Duration, queue: Duration) {
    GPU_EXECUTE_NANOS.fetch_add(nanos(execute), Ordering::Relaxed);
    GPU_QUEUE_NANOS.fetch_add(nanos(queue), Ordering::Relaxed);
    GPU_COMMANDS.fetch_add(1, Ordering::Relaxed);
}

/// Time the scheduler held the runtime doing native work.
pub(crate) fn record_runtime_busy(elapsed: Duration) {
    RUNTIME_BUSY_NANOS.fetch_add(nanos(elapsed), Ordering::Relaxed);
}

pub(crate) fn record_decode_tokens(tokens: u64) {
    DECODE_TOKENS.fetch_add(tokens, Ordering::Relaxed);
}

/// The scheduler's queue as of its latest step.
pub(crate) fn record_scheduler(running: usize, waiting: usize, batch: usize) {
    SCHEDULER_RUNNING.store(running as u64, Ordering::Relaxed);
    SCHEDULER_WAITING.store(waiting as u64, Ordering::Relaxed);
    SCHEDULER_BATCH.store(batch as u64, Ordering::Relaxed);
}

pub(crate) fn record_kv_used_cells(cells: u64) {
    KV_USED_CELLS.store(cells, Ordering::Relaxed);
}

fn nanos(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

fn current(gauge: &AtomicU64) -> Option<u64> {
    let value = gauge.load(Ordering::Relaxed);
    (value != UNSET).then_some(value)
}

/// Monotonic counters at one instant.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Counters {
    gpu_execute_nanos: u64,
    gpu_queue_nanos: u64,
    gpu_commands: u64,
    runtime_busy_nanos: u64,
    decode_tokens: u64,
}

impl Counters {
    fn now() -> Self {
        Self {
            gpu_execute_nanos: GPU_EXECUTE_NANOS.load(Ordering::Relaxed),
            gpu_queue_nanos: GPU_QUEUE_NANOS.load(Ordering::Relaxed),
            gpu_commands: GPU_COMMANDS.load(Ordering::Relaxed),
            runtime_busy_nanos: RUNTIME_BUSY_NANOS.load(Ordering::Relaxed),
            decode_tokens: DECODE_TOKENS.load(Ordering::Relaxed),
        }
    }
}

/// Everything rates are computed from: two of these make one sample.
#[derive(Clone, Copy, Debug)]
struct Reading {
    at: Instant,
    process: Option<ProcessUsage>,
    paging: Option<Paging>,
    counters: Counters,
    op_classes: OpClassCounters,
    kv: KvTiers,
}

impl Reading {
    fn take(system: Option<&SystemMemory>) -> Self {
        Self {
            at: Instant::now(),
            process: process_usage(),
            paging: system.and_then(|system| system.paging),
            counters: Counters::now(),
            op_classes: OpClassCounters::now(),
            kv: KvTiers::now(),
        }
    }
}

/// Levels read at the sample instant.
#[derive(Clone, Debug, Default)]
struct Levels {
    system: Option<SystemMemory>,
    gpu_memory: Vec<GpuMemory>,
    running: Option<u64>,
    waiting: Option<u64>,
    batch: Option<u64>,
    kv_used_cells: Option<u64>,
    graph_shape: GraphShape,
    telemetry_dropped: u64,
    native_spans_dropped: u64,
    gpu_records_dropped: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct GpuMemory {
    device: String,
    used_bytes: u64,
    limit_bytes: u64,
}

/// GPU memory per device, once native tracing shows the runtime is loaded:
/// before that a device query could be the first native call.
fn gpu_memory() -> Vec<GpuMemory> {
    if !skippy_runtime::native_tracing::native_tracing_active() {
        return Vec::new();
    }
    backend_devices_untraced()
        .unwrap_or_default()
        .into_iter()
        .filter(|device| {
            matches!(
                device.device_type,
                BackendDeviceType::Gpu | BackendDeviceType::IntegratedGpu
            )
        })
        .map(|device| GpuMemory {
            used_bytes: device.memory_total.saturating_sub(device.memory_free),
            limit_bytes: device.memory_total,
            device: device.name,
        })
        .collect()
}

fn levels(telemetry: &Telemetry, system: Option<SystemMemory>) -> Levels {
    use skippy_runtime::native_tracing::{gpu_commands, spans};
    Levels {
        system,
        gpu_memory: gpu_memory(),
        running: current(&SCHEDULER_RUNNING),
        waiting: current(&SCHEDULER_WAITING),
        batch: current(&SCHEDULER_BATCH),
        kv_used_cells: current(&KV_USED_CELLS),
        graph_shape: GraphShape::take(),
        telemetry_dropped: telemetry.stats().dropped,
        native_spans_dropped: spans::dropped_native_spans(),
        gpu_records_dropped: gpu_commands::dropped_gpu_command_timings(),
    }
}

/// The gauges for the interval from `before` to `after`, stamped `time`.
fn gauges(before: &Reading, after: &Reading, levels: &Levels, time: u64) -> Vec<Metric> {
    let seconds = after.at.duration_since(before.at).as_secs_f64();
    if seconds <= 0.0 {
        return Vec::new();
    }
    let mut metrics = Vec::new();
    process_gauges(&mut metrics, before, after, seconds, time);
    system_gauges(&mut metrics, before, after, levels, seconds, time);
    gpu_gauges(&mut metrics, before, after, levels, seconds, time);
    gpu_op_classes::gauges(
        &mut metrics,
        &before.op_classes,
        &after.op_classes,
        levels.graph_shape,
        seconds,
        time,
    );
    runtime_gauges(&mut metrics, before, after, levels, seconds, time);
    kv_tiers::gauges(
        &mut metrics,
        &before.kv,
        &after.kv,
        levels.kv_used_cells,
        seconds,
        time,
    );
    metrics
}

fn rate(before: u64, after: u64, seconds: f64) -> f64 {
    after.saturating_sub(before) as f64 / seconds
}

fn process_gauges(
    metrics: &mut Vec<Metric>,
    before: &Reading,
    after: &Reading,
    seconds: f64,
    time: u64,
) {
    let (Some(was), Some(now)) = (before.process, after.process) else {
        return;
    };
    let busy = |was: Duration, now: Duration| now.saturating_sub(was).as_secs_f64() / seconds;
    metrics.push(gauge(
        "process.cpu.utilization",
        "1",
        "CPU time per second of wall time; 1 is one core",
        vec![
            point(time, busy(was.user, now.user), &[("cpu.mode", "user")]),
            point(
                time,
                busy(was.system, now.system),
                &[("cpu.mode", "system")],
            ),
        ],
    ));
    metrics.push(gauge(
        "process.memory.usage",
        "By",
        "Resident memory",
        vec![point(time, now.resident_bytes as f64, &[])],
    ));
    metrics.push(gauge(
        "skippy.process.memory.footprint",
        "By",
        "Memory charged to the process, Metal buffers included on macOS",
        vec![point(time, now.footprint_bytes as f64, &[])],
    ));
    if let Some(wired) = now.wired_bytes {
        metrics.push(gauge(
            "skippy.process.memory.wired",
            "By",
            "Wired (unpageable) memory",
            vec![point(time, wired as f64, &[])],
        ));
    }
    metrics.push(gauge(
        "skippy.process.pageins",
        "{page}/s",
        "Pages read in from disk, such as mapped weights",
        vec![point(time, rate(was.pageins, now.pageins, seconds), &[])],
    ));
    if let (Some(was), Some(now)) = (was.disk_read_bytes, now.disk_read_bytes) {
        metrics.push(gauge(
            "skippy.process.disk.read",
            "By/s",
            "Bytes read from disk",
            vec![point(time, rate(was, now, seconds), &[])],
        ));
    }
}

fn system_gauges(
    metrics: &mut Vec<Metric>,
    before: &Reading,
    after: &Reading,
    levels: &Levels,
    seconds: f64,
    time: u64,
) {
    let Some(system) = levels.system.as_ref() else {
        return;
    };
    metrics.push(gauge(
        "system.memory.usage",
        "By",
        "System memory by state",
        system
            .states
            .iter()
            .map(|(state, bytes)| point(time, *bytes as f64, &[("system.memory.state", state)]))
            .collect(),
    ));
    metrics.push(gauge(
        "system.memory.limit",
        "By",
        "Installed memory",
        vec![point(time, system.total_bytes as f64, &[])],
    ));
    metrics.push(gauge(
        "skippy.system.swap.used",
        "By",
        "Swap in use",
        vec![point(time, system.swap_used_bytes as f64, &[])],
    ));
    if let Some(level) = system.pressure_level {
        metrics.push(gauge(
            "skippy.system.memory.pressure",
            "1",
            "macOS memory pressure: 1 normal, 2 warning, 4 critical",
            vec![point(time, f64::from(level), &[])],
        ));
    }
    if let (Some(was), Some(now)) = (before.paging, after.paging) {
        let pages = |was: u64, now: u64, direction: &str| {
            point(time, rate(was, now, seconds), &[("direction", direction)])
        };
        metrics.push(gauge(
            "skippy.system.paging",
            "{page}/s",
            "System-wide paging and swapping",
            vec![
                pages(was.pageins, now.pageins, "pagein"),
                pages(was.pageouts, now.pageouts, "pageout"),
                pages(was.swapins, now.swapins, "swapin"),
                pages(was.swapouts, now.swapouts, "swapout"),
            ],
        ));
    }
}

fn gpu_gauges(
    metrics: &mut Vec<Metric>,
    before: &Reading,
    after: &Reading,
    levels: &Levels,
    seconds: f64,
    time: u64,
) {
    let (was, now) = (before.counters, after.counters);
    let commands = now.gpu_commands.saturating_sub(was.gpu_commands);
    let execute = now.gpu_execute_nanos.saturating_sub(was.gpu_execute_nanos);
    metrics.push(gauge(
        "skippy.gpu.busy",
        "1",
        "Fraction of wall time the GPU spent running this process's command buffers",
        vec![point(time, execute as f64 / 1e9 / seconds, &[])],
    ));
    metrics.push(gauge(
        "skippy.gpu.command_buffers",
        "{command_buffer}/s",
        "Command buffers completed",
        vec![point(time, commands as f64 / seconds, &[])],
    ));
    if commands > 0 {
        let queue = now.gpu_queue_nanos.saturating_sub(was.gpu_queue_nanos);
        metrics.push(gauge(
            "skippy.gpu.queue_wait",
            "ms",
            "Mean wait between committing a command buffer and the GPU starting it",
            vec![point(time, queue as f64 / 1e6 / commands as f64, &[])],
        ));
    }
    if levels.gpu_memory.is_empty() {
        return;
    }
    let by_device = |value: fn(&GpuMemory) -> u64| {
        levels
            .gpu_memory
            .iter()
            .map(|memory| {
                point(
                    time,
                    value(memory) as f64,
                    &[("device", memory.device.as_str())],
                )
            })
            .collect()
    };
    metrics.push(gauge(
        "skippy.gpu.memory.usage",
        "By",
        "Memory the GPU device has allocated",
        by_device(|memory| memory.used_bytes),
    ));
    metrics.push(gauge(
        "skippy.gpu.memory.limit",
        "By",
        "The device's recommended working set",
        by_device(|memory| memory.limit_bytes),
    ));
}

fn runtime_gauges(
    metrics: &mut Vec<Metric>,
    before: &Reading,
    after: &Reading,
    levels: &Levels,
    seconds: f64,
    time: u64,
) {
    let (was, now) = (before.counters, after.counters);
    metrics.push(gauge(
        "skippy.runtime.busy",
        "1",
        "Fraction of wall time the scheduler held the runtime for native work",
        vec![point(
            time,
            now.runtime_busy_nanos
                .saturating_sub(was.runtime_busy_nanos) as f64
                / 1e9
                / seconds,
            &[],
        )],
    ));
    metrics.push(gauge(
        "skippy.decode.tokens",
        "{token}/s",
        "Decode tokens produced",
        vec![point(
            time,
            rate(was.decode_tokens, now.decode_tokens, seconds),
            &[],
        )],
    ));
    let levels_now = [
        (
            "skippy.scheduler.running",
            "{sequence}",
            "Sequences running",
            levels.running,
        ),
        (
            "skippy.scheduler.waiting",
            "{sequence}",
            "Sequences waiting",
            levels.waiting,
        ),
        (
            "skippy.scheduler.batch_size",
            "{sequence}",
            "Sequences in the latest step",
            levels.batch,
        ),
        (
            "skippy.kv.used_cells",
            "{cell}",
            "KV cells in use",
            levels.kv_used_cells,
        ),
    ];
    for (name, unit, description, value) in levels_now {
        if let Some(value) = value {
            metrics.push(gauge(
                name,
                unit,
                description,
                vec![point(time, value as f64, &[])],
            ));
        }
    }
    metrics.push(gauge(
        "skippy.telemetry.dropped",
        "{item}",
        "Telemetry lost to full queues since start",
        vec![
            point(
                time,
                levels.telemetry_dropped as f64,
                &[("source", "spans")],
            ),
            point(
                time,
                levels.native_spans_dropped as f64,
                &[("source", "native_spans")],
            ),
            point(
                time,
                levels.gpu_records_dropped as f64,
                &[("source", "gpu_commands")],
            ),
        ],
    ));
}

/// The running sampler: its sinks and the reading the next sample starts at.
struct Sampler {
    telemetry: Telemetry,
    sinks: MetricSinks,
    last: Mutex<Reading>,
    stopped: AtomicBool,
}

impl Sampler {
    fn tick(&self) {
        let system = system_memory();
        let mut last = self.last.lock().unwrap_or_else(PoisonError::into_inner);
        let now = Reading::take(system.as_ref());
        let time = u64::try_from(now_unix_nanos()).unwrap_or_default();
        let metrics = gauges(&last, &now, &levels(&self.telemetry, system), time);
        *last = now;
        drop(last);
        if !metrics.is_empty() {
            self.sinks.send(&self.telemetry.config, metrics);
        }
    }
}

static SAMPLER: OnceLock<Sampler> = OnceLock::new();

fn interval_from_value(value: Option<&str>) -> Duration {
    value
        .and_then(|value| value.trim().parse::<u64>().ok())
        .map_or(DEFAULT_INTERVAL, |millis| {
            Duration::from_millis(millis).max(MIN_INTERVAL)
        })
}

/// Start sampling for the process, if there is anywhere to send it. The first
/// debug handle with a destination serves the whole process.
pub(super) fn start(telemetry: &Telemetry, endpoint: Option<&str>) {
    let Some(sinks) = MetricSinks::open(
        env::var_os(METRICS_FILE_ENV).filter(|path| !path.is_empty()),
        endpoint,
    ) else {
        return;
    };
    let sampler = Sampler {
        telemetry: telemetry.clone(),
        sinks,
        last: Mutex::new(Reading::take(None)),
        stopped: AtomicBool::new(false),
    };
    if SAMPLER.set(sampler).is_err() {
        return;
    }
    let interval = interval_from_value(env::var(METRICS_INTERVAL_ENV).ok().as_deref());
    let spawned = thread::Builder::new()
        .name("skippy-metrics".to_string())
        .spawn(move || {
            let Some(sampler) = SAMPLER.get() else {
                return;
            };
            while !sampler.stopped.load(Ordering::Acquire) {
                thread::sleep(interval);
                sampler.tick();
            }
        });
    if let Err(error) = spawned {
        tracing::warn!("telemetry metrics sampler did not start: {error}");
    }
}

/// Take a last sample and stop; called as the process span ends.
pub(super) fn finish() {
    if let Some(sampler) = SAMPLER.get()
        && !sampler.stopped.swap(true, Ordering::AcqRel)
    {
        sampler.tick();
        sampler.sinks.flush();
    }
}

/// Resource readings for a span event, keyed `skippy.resources.*`.
fn snapshot(point: &str) -> BTreeMap<String, Value> {
    let key = |name: &str| format!("skippy.resources.{name}");
    let mut attrs = BTreeMap::from([(key("point"), json!(point))]);
    if let Some(process) = process_usage() {
        attrs.extend([
            (
                key("process.footprint_bytes"),
                json!(process.footprint_bytes),
            ),
            (key("process.resident_bytes"), json!(process.resident_bytes)),
            (key("process.cpu_user_s"), json!(process.user.as_secs_f64())),
            (
                key("process.cpu_system_s"),
                json!(process.system.as_secs_f64()),
            ),
            (key("process.pageins"), json!(process.pageins)),
        ]);
        if let Some(wired) = process.wired_bytes {
            attrs.insert(key("process.wired_bytes"), json!(wired));
        }
    }
    if let Some(system) = system_memory() {
        attrs.insert(key("system.total_bytes"), json!(system.total_bytes));
        attrs.insert(key("system.swap_used_bytes"), json!(system.swap_used_bytes));
        for (state, bytes) in &system.states {
            attrs.insert(key(&format!("system.{state}_bytes")), json!(bytes));
        }
        if let Some(level) = system.pressure_level {
            attrs.insert(key("system.pressure_level"), json!(level));
        }
    }
    for memory in gpu_memory() {
        attrs.insert(
            key(&format!("gpu.{}.used_bytes", memory.device)),
            json!(memory.used_bytes),
        );
        attrs.insert(
            key(&format!("gpu.{}.limit_bytes", memory.device)),
            json!(memory.limit_bytes),
        );
    }
    let gauges = [
        ("scheduler.running", current(&SCHEDULER_RUNNING)),
        ("scheduler.waiting", current(&SCHEDULER_WAITING)),
        ("kv.used_cells", current(&KV_USED_CELLS)),
    ];
    for (name, value) in gauges {
        if let Some(value) = value {
            attrs.insert(key(name), json!(value));
        }
    }
    kv_tiers::snapshot(&mut attrs, key);
    attrs
}

impl Telemetry {
    /// Record a resource snapshot, named `point`, on the active span.
    pub(crate) fn record_resources(&self, point: &str) {
        if self.level == TelemetryLevel::Debug {
            self.emit("skippy.resources", snapshot(point));
        }
    }

    /// Record a resource snapshot on `frame`, which need not be active.
    pub(crate) fn record_resources_on(&self, frame: &Frame, point: &str) {
        if self.level != TelemetryLevel::Debug || !self.has_trace_sink() {
            return;
        }
        let mut attributes = snapshot(point);
        self.add_common_attributes(&mut attributes);
        let event = SpanEvent {
            name: "skippy.resources".to_string(),
            time_unix_nanos: u64::try_from(now_unix_nanos()).unwrap_or_default(),
            attributes,
        };
        if let Err(event) = frame.record_event(event) {
            self.emit(&event.name, event.attributes);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opentelemetry_proto::tonic::metrics::v1::{metric::Data, number_data_point};

    fn reading(at: Instant, cpu_ms: u64, counters: Counters) -> Reading {
        Reading {
            at,
            process: Some(ProcessUsage {
                user: Duration::from_millis(cpu_ms),
                system: Duration::from_millis(cpu_ms / 2),
                resident_bytes: 1 << 30,
                footprint_bytes: 2 << 30,
                wired_bytes: Some(1 << 20),
                pageins: 100,
                disk_read_bytes: None,
            }),
            paging: None,
            counters,
            op_classes: OpClassCounters::default(),
            kv: KvTiers::default(),
        }
    }

    fn values(metrics: &[Metric], name: &str) -> Vec<f64> {
        metrics
            .iter()
            .find(|metric| metric.name == name)
            .map(|metric| match metric.data.as_ref() {
                Some(Data::Gauge(gauge)) => gauge
                    .data_points
                    .iter()
                    .map(|point| match point.value {
                        Some(number_data_point::Value::AsDouble(value)) => value,
                        _ => f64::NAN,
                    })
                    .collect(),
                _ => Vec::new(),
            })
            .unwrap_or_default()
    }

    #[test]
    fn rates_come_from_the_interval_between_readings() {
        let start = Instant::now();
        let before = reading(start, 1_000, Counters::default());
        let after = reading(
            start + Duration::from_millis(500),
            1_400,
            Counters {
                gpu_execute_nanos: 250_000_000,
                gpu_queue_nanos: 2_000_000,
                gpu_commands: 4,
                runtime_busy_nanos: 400_000_000,
                decode_tokens: 40,
            },
        );
        let levels = Levels {
            running: Some(4),
            kv_used_cells: Some(2048),
            ..Levels::default()
        };
        let metrics = gauges(&before, &after, &levels, 7);

        let close = |got: &[f64], want: &[f64]| {
            assert_eq!(got.len(), want.len(), "{got:?} vs {want:?}");
            for (got, want) in got.iter().zip(want) {
                assert!((got - want).abs() < 1e-9, "{got} vs {want}");
            }
        };
        close(&values(&metrics, "process.cpu.utilization"), &[0.8, 0.4]);
        close(&values(&metrics, "skippy.gpu.busy"), &[0.5]);
        close(&values(&metrics, "skippy.gpu.command_buffers"), &[8.0]);
        close(&values(&metrics, "skippy.gpu.queue_wait"), &[0.5]);
        close(&values(&metrics, "skippy.runtime.busy"), &[0.8]);
        close(&values(&metrics, "skippy.decode.tokens"), &[80.0]);
        close(&values(&metrics, "skippy.scheduler.running"), &[4.0]);
        close(&values(&metrics, "skippy.kv.used_cells"), &[2048.0]);
        assert!(values(&metrics, "skippy.scheduler.waiting").is_empty());
        assert!(values(&metrics, "skippy.gpu.memory.usage").is_empty());
    }

    #[test]
    fn the_sampling_interval_has_a_default_and_a_floor() {
        assert_eq!(interval_from_value(None), DEFAULT_INTERVAL);
        assert_eq!(interval_from_value(Some("x")), DEFAULT_INTERVAL);
        assert_eq!(interval_from_value(Some(" 1000 ")), Duration::from_secs(1));
        assert_eq!(interval_from_value(Some("1")), MIN_INTERVAL);
    }

    #[test]
    fn a_snapshot_names_its_point_and_this_process() {
        let attrs = snapshot("test_point");
        assert_eq!(attrs["skippy.resources.point"], json!("test_point"));
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        assert!(attrs.contains_key("skippy.resources.process.resident_bytes"));
    }
}
