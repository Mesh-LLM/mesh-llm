//! GPU command buffers as debug spans, each under the native call that
//! submitted it. A command buffer's span runs from the start of its CPU
//! encode to the end of its GPU execution, split into `encode`, `queue`
//! (committed, waiting for the GPU) and `execute` children, which a graph
//! split by op class keeps as attributes instead; `idle` spans
//! mark the gaps where the GPU waited for the host. Runtimes that classify
//! graph nodes add what each command did: attention, mat-muls, recurrent
//! state, KV writes or other work, with the bytes each class's inputs span.

use std::collections::BTreeMap;
use std::time::Duration;

use serde_json::{Value, json};
use skippy_runtime::native_tracing::gpu_commands::{
    GpuCommandTiming, GpuOpClass, GpuOpClasses, drain_gpu_command_timings,
};

use super::{
    Telemetry, resource_metrics,
    trace_context::{self, Frame},
};

mod attention_path;
mod idle_gaps;

/// Emit every GPU record queued so far.
pub(super) fn emit_pending(telemetry: &Telemetry) {
    if !telemetry.is_debug_enabled() {
        return;
    }
    drain_gpu_command_timings(|timing| emit(telemetry, &timing));
}

fn emit(telemetry: &Telemetry, timing: &GpuCommandTiming) {
    record_busy(timing);
    let prefix = span_prefix(timing.backend);
    let frame = Frame::under(trace_context::context_for_tag(timing.tag));
    idle_gaps::emit(telemetry, &prefix, frame.context(), timing);
    attention_path::record(&frame, timing);
    let (start, end) = span_window(timing);
    telemetry.emit_debug_frame_span(
        &frame,
        &format!("{prefix}.command_buffer"),
        attributes(timing),
        start,
        end,
    );
    // A graph split by op class encodes hundreds of command buffers; their
    // phases stay attributes so the exporter's queue keeps every command.
    if split_by_class(timing) {
        return;
    }
    for (phase, start, end) in phases(timing) {
        telemetry.emit_debug_span_under(
            frame.context(),
            &format!("{prefix}.{phase}"),
            BTreeMap::new(),
            start,
            end,
        );
    }
}

/// Feed the GPU busy and queue-wait gauges.
fn record_busy(timing: &GpuCommandTiming) {
    let between = |start: u64, end: u64| {
        Duration::from_nanos(if start > 0 && end > start {
            end - start
        } else {
            0
        })
    };
    let execute = between(
        timing.execute_start_unix_nanos,
        timing.execute_end_unix_nanos,
    );
    resource_metrics::record_gpu_command(
        execute,
        between(
            timing.encode_end_unix_nanos,
            timing.execute_start_unix_nanos,
        ),
    );
    if let Some(classes) = timing.op_classes.as_ref() {
        resource_metrics::gpu_op_classes::record(classes, execute);
    }
}

fn split_by_class(timing: &GpuCommandTiming) -> bool {
    timing
        .op_classes
        .as_ref()
        .is_some_and(|classes| classes.split_by_class)
}

/// `mtl` for Metal's `MTL` registry; other backends get their own prefix.
fn span_prefix(backend: &str) -> String {
    backend
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect()
}

/// Encode start to execute end, using whichever instants were observed.
fn span_window(timing: &GpuCommandTiming) -> (u64, u64) {
    let instants = [
        timing.encode_start_unix_nanos,
        timing.encode_end_unix_nanos,
        timing.execute_start_unix_nanos,
        timing.execute_end_unix_nanos,
    ];
    let observed = instants.iter().copied().filter(|&instant| instant > 0);
    let start = observed.clone().min().unwrap_or(0);
    (start, observed.max().unwrap_or(start))
}

/// The observed, non-empty phases of one command buffer, in order.
fn phases(timing: &GpuCommandTiming) -> Vec<(&'static str, u64, u64)> {
    [
        (
            "encode",
            timing.encode_start_unix_nanos,
            timing.encode_end_unix_nanos,
        ),
        (
            "queue",
            timing.encode_end_unix_nanos,
            timing.execute_start_unix_nanos,
        ),
        (
            "execute",
            timing.execute_start_unix_nanos,
            timing.execute_end_unix_nanos,
        ),
    ]
    .into_iter()
    .filter(|&(_, start, end)| start > 0 && end > start)
    .collect()
}

fn millis_between(start: u64, end: u64) -> Option<f64> {
    (start > 0 && end > 0).then(|| (end as f64 - start as f64) / 1_000_000.0)
}

fn attributes(timing: &GpuCommandTiming) -> BTreeMap<String, Value> {
    let mut attrs = BTreeMap::from([
        ("skippy.gpu.backend".to_string(), json!(timing.backend)),
        (
            "skippy.gpu.context_id".to_string(),
            json!(timing.context_id),
        ),
        (
            "skippy.gpu.compute_index".to_string(),
            json!(timing.compute_index),
        ),
        (
            "skippy.gpu.command_index".to_string(),
            json!(timing.command_index),
        ),
        (
            "skippy.gpu.command_count".to_string(),
            json!(timing.command_count),
        ),
        (
            "skippy.gpu.node_count".to_string(),
            json!(timing.node_count),
        ),
    ]);
    let durations = [
        (
            "skippy.gpu.encode_ms",
            millis_between(timing.encode_start_unix_nanos, timing.encode_end_unix_nanos),
        ),
        (
            "skippy.gpu.queue_ms",
            millis_between(
                timing.encode_end_unix_nanos,
                timing.execute_start_unix_nanos,
            ),
        ),
        (
            "skippy.gpu.execute_ms",
            millis_between(
                timing.execute_start_unix_nanos,
                timing.execute_end_unix_nanos,
            ),
        ),
    ];
    for (key, value) in durations {
        if let Some(value) = value {
            attrs.insert(key.to_string(), json!(value));
        }
    }
    if timing.status != 0 {
        attrs.insert("skippy.gpu.status".to_string(), json!(timing.status));
    }
    if let Some(classes) = timing.op_classes.as_ref() {
        insert_op_classes(&mut attrs, classes);
    }
    attrs
}

/// `skippy.gpu.op_class` and, for every class the command holds, its node
/// count and the bytes its inputs span; attention adds its KV shape.
fn insert_op_classes(attrs: &mut BTreeMap<String, Value>, classes: &GpuOpClasses) {
    attrs.insert(
        "skippy.gpu.op_class".to_string(),
        json!(classes.single.map_or("mixed", GpuOpClass::name)),
    );
    if let Some(layer) = classes.layer {
        attrs.insert("skippy.gpu.layer".to_string(), json!(layer));
    }
    if classes.split_by_class {
        attrs.insert("skippy.gpu.split_by_class".to_string(), json!(true));
    }
    for class in GpuOpClass::ALL {
        let nodes = classes.nodes.get(class);
        if nodes == 0 {
            continue;
        }
        attrs.insert(format!("skippy.gpu.nodes.{}", class.name()), json!(nodes));
        attrs.insert(
            format!("skippy.gpu.input_bytes.{}", class.name()),
            json!(classes.bytes.get(class)),
        );
    }
    if classes.nodes.attention > 0 && classes.attention_kv_cells > 0 {
        attrs.extend([
            (
                "skippy.attention.kv_cells".to_string(),
                json!(classes.attention_kv_cells),
            ),
            (
                "skippy.attention.queries".to_string(),
                json!(classes.attention_queries),
            ),
            (
                "skippy.attention.row_list_nodes".to_string(),
                json!(classes.attention_row_list_nodes),
            ),
            (
                "skippy.attention.flash_nodes".to_string(),
                json!(classes.attention_flash_nodes),
            ),
        ]);
    }
    if let Some(path) = classes.attention_path() {
        attrs.insert("skippy.attention.path".to_string(), json!(path.name()));
    }
    if let Some(states) = classes
        .recurrent_states
        .filter(|states| states.gathered > 0)
    {
        attrs.extend([
            (
                "skippy.recurrent.states".to_string(),
                json!(states.gathered),
            ),
            (
                "skippy.recurrent.states_carried".to_string(),
                json!(states.carried),
            ),
        ]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::telemetry::{TelemetryLevel, capture_stage_config};
    use skippy_runtime::native_tracing::gpu_commands::{PerOpClass, RecurrentStates};

    fn timing() -> GpuCommandTiming {
        GpuCommandTiming {
            tag: 0,
            backend: "MTL",
            context_id: 1,
            compute_index: 12,
            command_index: 0,
            command_count: 3,
            node_count: 64,
            status: 0,
            encode_start_unix_nanos: 1_000_000,
            encode_end_unix_nanos: 3_000_000,
            execute_start_unix_nanos: 3_500_000,
            execute_end_unix_nanos: 9_500_000,
            op_classes: None,
        }
    }

    #[test]
    fn a_classified_command_says_what_its_nodes_did() {
        let timing = GpuCommandTiming {
            op_classes: Some(GpuOpClasses {
                single: None,
                layer: Some(5),
                split_by_class: false,
                attention_kv_cells: 4096,
                attention_queries: 4,
                attention_row_list_nodes: 0,
                attention_flash_nodes: 1,
                nodes: PerOpClass {
                    attention: 1,
                    matmul: 3,
                    ..PerOpClass::default()
                },
                bytes: PerOpClass {
                    attention: 1 << 20,
                    matmul: 1 << 24,
                    ..PerOpClass::default()
                },
                recurrent_states: Some(RecurrentStates::default()),
            }),
            ..timing()
        };
        let attrs = attributes(&timing);
        assert_eq!(attrs["skippy.gpu.op_class"], json!("mixed"));
        assert_eq!(attrs["skippy.gpu.layer"], json!(5));
        assert_eq!(attrs["skippy.gpu.nodes.matmul"], json!(3));
        assert_eq!(attrs["skippy.gpu.input_bytes.attention"], json!(1 << 20));
        assert_eq!(attrs["skippy.attention.kv_cells"], json!(4096));
        assert_eq!(attrs["skippy.attention.row_list_nodes"], json!(0));
        assert_eq!(attrs["skippy.attention.path"], json!("flash"));
        assert!(!attrs.contains_key("skippy.gpu.nodes.recurrent"));
        assert!(!attrs.contains_key("skippy.gpu.split_by_class"));
        assert!(!attrs.contains_key("skippy.recurrent.states"));
    }

    #[test]
    fn a_command_with_recurrent_state_says_how_many_states_it_moved() {
        let mut timing = timing();
        timing.op_classes = Some(GpuOpClasses {
            single: Some(GpuOpClass::Recurrent),
            layer: Some(1),
            split_by_class: false,
            attention_kv_cells: 0,
            attention_queries: 0,
            attention_row_list_nodes: 0,
            attention_flash_nodes: 0,
            nodes: PerOpClass::default(),
            bytes: PerOpClass::default(),
            recurrent_states: Some(RecurrentStates {
                gathered: 9,
                carried: 5,
            }),
        });
        let attrs = attributes(&timing);
        assert_eq!(attrs["skippy.recurrent.states"], json!(9));
        assert_eq!(attrs["skippy.recurrent.states_carried"], json!(5));
    }

    #[test]
    fn spans_are_named_after_the_backend() {
        assert_eq!(span_prefix("MTL"), "mtl");
        assert_eq!(span_prefix("Vulkan 1"), "vulkan_1");
    }

    #[test]
    fn a_command_buffer_covers_its_encode_queue_and_execution() {
        let timing = timing();
        assert_eq!(span_window(&timing), (1_000_000, 9_500_000));
        assert_eq!(
            phases(&timing),
            vec![
                ("encode", 1_000_000, 3_000_000),
                ("queue", 3_000_000, 3_500_000),
                ("execute", 3_500_000, 9_500_000),
            ]
        );
        let attrs = attributes(&timing);
        assert_eq!(attrs["skippy.gpu.encode_ms"], json!(2.0));
        assert_eq!(attrs["skippy.gpu.queue_ms"], json!(0.5));
        assert_eq!(attrs["skippy.gpu.execute_ms"], json!(6.0));
        assert_eq!(attrs["skippy.gpu.compute_index"], json!(12));
        assert!(!attrs.contains_key("skippy.gpu.status"));
    }

    #[test]
    fn a_failed_buffer_keeps_only_its_encode() {
        let timing = GpuCommandTiming {
            status: 5,
            execute_start_unix_nanos: 0,
            execute_end_unix_nanos: 0,
            ..timing()
        };
        assert_eq!(span_window(&timing), (1_000_000, 3_000_000));
        assert_eq!(phases(&timing), vec![("encode", 1_000_000, 3_000_000)]);
        let attrs = attributes(&timing);
        assert_eq!(attrs["skippy.gpu.status"], json!(5));
        assert!(!attrs.contains_key("skippy.gpu.execute_ms"));
    }

    #[test]
    fn phases_nest_under_the_command_buffer_under_the_call() {
        let (telemetry, rx) = Telemetry::captured(capture_stage_config(), TelemetryLevel::Debug);
        let call = Frame::child();
        emit(
            &telemetry,
            &GpuCommandTiming {
                tag: call.tag(),
                ..timing()
            },
        );
        let spans = rx
            .try_iter()
            .map(|event| (event.event, event.span.expect("a span")))
            .collect::<Vec<_>>();
        let names = spans
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            names,
            [
                "mtl.command_buffer",
                "mtl.encode",
                "mtl.queue",
                "mtl.execute"
            ]
        );
        let (buffer, buffer_parent) = spans[0].1;
        assert_eq!(buffer_parent, Some(call.context().span_id));
        for (_, (_, parent)) in &spans[1..] {
            assert_eq!(*parent, Some(buffer.span_id));
        }
    }

    #[test]
    fn a_command_split_by_class_keeps_its_phases_as_attributes() {
        let (telemetry, rx) = Telemetry::captured(capture_stage_config(), TelemetryLevel::Debug);
        emit(
            &telemetry,
            &GpuCommandTiming {
                op_classes: Some(GpuOpClasses {
                    single: Some(GpuOpClass::Matmul),
                    layer: Some(3),
                    split_by_class: true,
                    attention_kv_cells: 0,
                    attention_queries: 0,
                    attention_row_list_nodes: 0,
                    attention_flash_nodes: 0,
                    nodes: PerOpClass {
                        matmul: 2,
                        ..PerOpClass::default()
                    },
                    bytes: PerOpClass {
                        matmul: 1 << 20,
                        ..PerOpClass::default()
                    },
                    recurrent_states: None,
                }),
                ..timing()
            },
        );
        let events = rx.try_iter().collect::<Vec<_>>();
        assert_eq!(
            events
                .iter()
                .map(|event| event.event.as_str())
                .collect::<Vec<_>>(),
            ["mtl.command_buffer"]
        );
        assert_eq!(events[0].attributes["skippy.gpu.execute_ms"], json!(6.0));
    }
}
