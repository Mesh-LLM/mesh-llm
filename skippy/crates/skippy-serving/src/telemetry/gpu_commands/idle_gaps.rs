//! The GPU's idle stretches between command buffers, as `<backend>.idle`
//! spans under the call whose work ended the wait. An idle stretch inside
//! one graph compute means the host encoded slower than the GPU ran; one
//! between graph computes is host work between steps, such as sampling,
//! scheduling or building the next graph.

use std::collections::BTreeMap;
use std::sync::{Mutex, PoisonError};

use serde_json::{Value, json};
use skippy_runtime::native_tracing::gpu_commands::GpuCommandTiming;

use super::super::{Telemetry, trace_context::SpanContext};

/// Shorter gaps are command scheduling, not the GPU waiting for the host.
const MIN_IDLE_NANOS: u64 = 50_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LastCommand {
    backend: &'static str,
    context_id: u32,
    compute_index: u32,
    execute_end: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct IdleGap {
    start: u64,
    end: u64,
    within_graph: bool,
    previous_compute_index: u32,
}

/// The latest execution end on each backend. Every context on a backend
/// shares its device, so the device is idle only between all of them.
#[derive(Debug, Default)]
struct IdleTracker {
    last: Vec<LastCommand>,
}

impl IdleTracker {
    /// The idle stretch that ended when `timing` started executing, if long
    /// enough to report. Records arrive as the device completes them.
    fn observe(&mut self, timing: &GpuCommandTiming) -> Option<IdleGap> {
        let (start, end) = (
            timing.execute_start_unix_nanos,
            timing.execute_end_unix_nanos,
        );
        if start == 0 || end < start {
            return None;
        }
        let this = LastCommand {
            backend: timing.backend,
            context_id: timing.context_id,
            compute_index: timing.compute_index,
            execute_end: end,
        };
        let Some(last) = self
            .last
            .iter_mut()
            .find(|last| last.backend == timing.backend)
        else {
            self.last.push(this);
            return None;
        };
        let previous = *last;
        if end > previous.execute_end {
            *last = this;
        }
        (start >= previous.execute_end.saturating_add(MIN_IDLE_NANOS)).then_some(IdleGap {
            start: previous.execute_end,
            end: start,
            within_graph: previous.context_id == timing.context_id
                && previous.compute_index == timing.compute_index,
            previous_compute_index: previous.compute_index,
        })
    }
}

static TRACKER: Mutex<IdleTracker> = Mutex::new(IdleTracker { last: Vec::new() });

/// Emit the idle span `timing` ended, under `parent`.
pub(super) fn emit(
    telemetry: &Telemetry,
    prefix: &str,
    parent: SpanContext,
    timing: &GpuCommandTiming,
) {
    let gap = TRACKER
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .observe(timing);
    if let Some(gap) = gap {
        telemetry.emit_debug_span_under(
            parent,
            &format!("{prefix}.idle"),
            attributes(&gap, timing),
            gap.start,
            gap.end,
        );
    }
}

fn attributes(gap: &IdleGap, timing: &GpuCommandTiming) -> BTreeMap<String, Value> {
    BTreeMap::from([
        (
            "skippy.gpu.idle_ms".to_string(),
            json!((gap.end - gap.start) as f64 / 1e6),
        ),
        (
            "skippy.gpu.idle_kind".to_string(),
            json!(if gap.within_graph {
                "within_graph"
            } else {
                "between_graphs"
            }),
        ),
        (
            "skippy.gpu.compute_index".to_string(),
            json!(timing.compute_index),
        ),
        (
            "skippy.gpu.previous_compute_index".to_string(),
            json!(gap.previous_compute_index),
        ),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn command(compute_index: u32, start: u64, end: u64) -> GpuCommandTiming {
        GpuCommandTiming {
            tag: 0,
            backend: "MTL",
            context_id: 1,
            compute_index,
            command_index: 0,
            command_count: 2,
            node_count: 10,
            status: 0,
            encode_start_unix_nanos: 0,
            encode_end_unix_nanos: 0,
            execute_start_unix_nanos: start,
            execute_end_unix_nanos: end,
            op_classes: None,
        }
    }

    #[test]
    fn gaps_between_and_within_graphs_are_told_apart() {
        let mut tracker = IdleTracker::default();
        assert_eq!(tracker.observe(&command(1, 1_000_000, 2_000_000)), None);
        let within = tracker
            .observe(&command(1, 2_400_000, 3_000_000))
            .expect("within");
        assert!(within.within_graph);
        assert_eq!((within.start, within.end), (2_000_000, 2_400_000));
        let between = tracker
            .observe(&command(2, 5_000_000, 6_000_000))
            .expect("between");
        assert!(!between.within_graph);
        assert_eq!(between.previous_compute_index, 1);
    }

    #[test]
    fn short_and_overlapping_gaps_are_not_idle() {
        let mut tracker = IdleTracker::default();
        tracker.observe(&command(1, 1_000_000, 2_000_000));
        assert_eq!(tracker.observe(&command(1, 2_010_000, 3_000_000)), None);
        assert_eq!(tracker.observe(&command(1, 2_500_000, 2_900_000)), None);
        assert_eq!(tracker.observe(&command(1, 0, 0)), None);
    }
}
