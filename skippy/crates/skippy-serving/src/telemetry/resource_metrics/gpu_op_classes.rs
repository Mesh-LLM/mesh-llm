//! GPU work by op class: the bytes each class reads per second (an estimate
//! of where memory traffic goes), and with op profiling, which splits graphs
//! so each command buffer holds one class, the GPU time each class takes.
//! Shape gauges give the largest value in each interval: the KV span one
//! attention query addressed (layers differ: sliding-window and full
//! attention, so the last command's value alone would be noise), and the
//! recurrent states one step gathered and carried back unchanged.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use opentelemetry_proto::tonic::metrics::v1::Metric;
use skippy_runtime::native_tracing::gpu_commands::{GpuOpClass, GpuOpClasses};

use super::{
    nanos,
    otlp_metrics::{gauge, point},
    rate,
};

const CLASSES: usize = GpuOpClass::ALL.len();

static INPUT_BYTES: [AtomicU64; CLASSES] = [const { AtomicU64::new(0) }; CLASSES];
static PROFILED_EXECUTE_NANOS: [AtomicU64; CLASSES] = [const { AtomicU64::new(0) }; CLASSES];
/// Largest since the last sample; 0 is none.
static ATTENTION_KV_CELLS: AtomicU64 = AtomicU64::new(0);
static ATTENTION_QUERIES: AtomicU64 = AtomicU64::new(0);
static RECURRENT_STATES: AtomicU64 = AtomicU64::new(0);
static RECURRENT_STATES_CARRIED: AtomicU64 = AtomicU64::new(0);

fn index(class: GpuOpClass) -> usize {
    match class {
        GpuOpClass::Attention => 0,
        GpuOpClass::Matmul => 1,
        GpuOpClass::Recurrent => 2,
        GpuOpClass::KvWrite => 3,
        GpuOpClass::Other => 4,
    }
}

/// One command buffer's classes and GPU execution time.
pub(crate) fn record(classes: &GpuOpClasses, execute: Duration) {
    for class in GpuOpClass::ALL {
        INPUT_BYTES[index(class)].fetch_add(classes.bytes.get(class), Ordering::Relaxed);
    }
    if classes.split_by_class
        && let Some(class) = classes.single
    {
        PROFILED_EXECUTE_NANOS[index(class)].fetch_add(nanos(execute), Ordering::Relaxed);
    }
    if classes.nodes.attention > 0 && classes.attention_kv_cells > 0 {
        ATTENTION_KV_CELLS.fetch_max(u64::from(classes.attention_kv_cells), Ordering::Relaxed);
        ATTENTION_QUERIES.fetch_max(u64::from(classes.attention_queries), Ordering::Relaxed);
    }
    if let Some(states) = classes.recurrent_states {
        RECURRENT_STATES.fetch_max(u64::from(states.gathered), Ordering::Relaxed);
        RECURRENT_STATES_CARRIED.fetch_max(u64::from(states.carried), Ordering::Relaxed);
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct OpClassCounters {
    input_bytes: [u64; CLASSES],
    profiled_execute_nanos: [u64; CLASSES],
}

impl OpClassCounters {
    pub(super) fn now() -> Self {
        Self {
            input_bytes: INPUT_BYTES
                .each_ref()
                .map(|bytes| bytes.load(Ordering::Relaxed)),
            profiled_execute_nanos: PROFILED_EXECUTE_NANOS
                .each_ref()
                .map(|nanos| nanos.load(Ordering::Relaxed)),
        }
    }
}

/// The largest graph shape since the last sample: KV cells one attention
/// query addressed, query rows one attention node processed, and recurrent
/// states one step gathered and, of those, carried back unchanged.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct GraphShape {
    kv_cells: Option<u64>,
    queries: Option<u64>,
    recurrent: Option<RecurrentShape>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RecurrentShape {
    states: u64,
    /// 0 is a real value: every gathered state belonged to the batch.
    carried: u64,
}

impl GraphShape {
    /// Takes the interval's maxima and starts the next interval.
    pub(super) fn take() -> Self {
        let take =
            |max: &AtomicU64| Some(max.swap(0, Ordering::Relaxed)).filter(|&value| value > 0);
        let carried = RECURRENT_STATES_CARRIED.swap(0, Ordering::Relaxed);
        Self {
            kv_cells: take(&ATTENTION_KV_CELLS),
            queries: take(&ATTENTION_QUERIES),
            recurrent: take(&RECURRENT_STATES).map(|states| RecurrentShape { states, carried }),
        }
    }
}

/// Per-class gauges for the interval from `was` to `now`.
pub(super) fn gauges(
    metrics: &mut Vec<Metric>,
    was: &OpClassCounters,
    now: &OpClassCounters,
    shape: GraphShape,
    seconds: f64,
    time: u64,
) {
    let by_class = |values: &dyn Fn(usize) -> f64| {
        GpuOpClass::ALL
            .into_iter()
            .map(|class| point(time, values(index(class)), &[("op_class", class.name())]))
            .collect::<Vec<_>>()
    };
    if now.input_bytes.iter().any(|&bytes| bytes > 0) {
        metrics.push(gauge(
            "skippy.gpu.input_bytes",
            "By/s",
            "Bytes GPU commands read, by op_class: gathers and row writes count their rows, attention the K/V rows it addresses",
            by_class(&|i| rate(was.input_bytes[i], now.input_bytes[i], seconds)),
        ));
    }
    let profiled =
        (0..CLASSES).any(|i| now.profiled_execute_nanos[i] > was.profiled_execute_nanos[i]);
    if profiled {
        metrics.push(gauge(
            "skippy.gpu.class_busy",
            "1",
            "GPU execution time per wall second, by op_class, from graphs split by class (GGML_METAL_OP_PROFILE)",
            by_class(&|i| {
                now.profiled_execute_nanos[i].saturating_sub(was.profiled_execute_nanos[i]) as f64
                    / 1e9
                    / seconds
            }),
        ));
    }
    shape_gauges(metrics, shape, time);
}

fn shape_gauges(metrics: &mut Vec<Metric>, shape: GraphShape, time: u64) {
    if let Some(cells) = shape.kv_cells {
        metrics.push(gauge(
            "skippy.attention.kv_cells",
            "{cell}",
            "The most KV cells one attention query addressed in the interval: its row list, or the whole K/V span",
            vec![point(time, cells as f64, &[])],
        ));
    }
    if let Some(queries) = shape.queries {
        metrics.push(gauge(
            "skippy.attention.queries",
            "{query}",
            "The most query rows one attention node processed in the interval",
            vec![point(time, queries as f64, &[])],
        ));
    }
    if let Some(recurrent) = shape.recurrent {
        metrics.push(gauge(
            "skippy.recurrent.states",
            "{state}",
            "The most recurrent states one step gathered in the interval: its sequences' and every state between them",
            vec![point(time, recurrent.states as f64, &[])],
        ));
        metrics.push(gauge(
            "skippy.recurrent.states_carried",
            "{state}",
            "The most gathered recurrent states one step copied back unchanged in the interval",
            vec![point(time, recurrent.carried as f64, &[])],
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use skippy_runtime::native_tracing::gpu_commands::PerOpClass;

    fn values(metrics: &[Metric], name: &str) -> Vec<f64> {
        use opentelemetry_proto::tonic::metrics::v1::{metric::Data, number_data_point};
        metrics
            .iter()
            .find(|metric| metric.name == name)
            .and_then(|metric| match metric.data.as_ref() {
                Some(Data::Gauge(gauge)) => Some(
                    gauge
                        .data_points
                        .iter()
                        .map(|point| match point.value {
                            Some(number_data_point::Value::AsDouble(value)) => value,
                            _ => f64::NAN,
                        })
                        .collect(),
                ),
                _ => None,
            })
            .unwrap_or_default()
    }

    #[test]
    fn bytes_are_rates_and_profiled_time_is_a_share_of_wall_time() {
        let was = OpClassCounters::default();
        let now = OpClassCounters {
            input_bytes: [100, 400, 0, 0, 50],
            profiled_execute_nanos: [250_000_000, 0, 0, 0, 0],
        };
        let shape = GraphShape {
            kv_cells: Some(4096),
            queries: Some(4),
            recurrent: Some(RecurrentShape {
                states: 9,
                carried: 0,
            }),
        };
        let mut metrics = Vec::new();
        gauges(&mut metrics, &was, &now, shape, 0.5, 1);
        assert_eq!(
            values(&metrics, "skippy.gpu.input_bytes"),
            [200.0, 800.0, 0.0, 0.0, 100.0]
        );
        assert_eq!(values(&metrics, "skippy.gpu.class_busy")[0], 0.5);
        assert_eq!(values(&metrics, "skippy.attention.kv_cells"), [4096.0]);
        assert_eq!(values(&metrics, "skippy.recurrent.states"), [9.0]);
        assert_eq!(
            values(&metrics, "skippy.recurrent.states_carried"),
            [0.0],
            "nothing carried is a reading, not an absence"
        );

        let mut unprofiled = Vec::new();
        gauges(&mut unprofiled, &now, &now, GraphShape::default(), 0.5, 1);
        assert!(values(&unprofiled, "skippy.gpu.class_busy").is_empty());
        assert!(values(&unprofiled, "skippy.attention.kv_cells").is_empty());
        assert!(values(&unprofiled, "skippy.recurrent.states_carried").is_empty());
    }

    #[test]
    fn classes_index_their_own_slot() {
        let classes = GpuOpClasses {
            single: Some(GpuOpClass::Matmul),
            layer: None,
            split_by_class: true,
            attention_kv_cells: 0,
            attention_queries: 0,
            attention_row_list_nodes: 0,
            attention_flash_nodes: 0,
            nodes: PerOpClass::default(),
            bytes: PerOpClass {
                matmul: 7,
                ..PerOpClass::default()
            },
            recurrent_states: None,
        };
        let indices = GpuOpClass::ALL.map(index);
        assert_eq!(indices, [0, 1, 2, 3, 4]);
        assert_eq!(classes.bytes.get(GpuOpClass::ALL[indices[1]]), 7);
    }
}
