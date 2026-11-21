//! Events from what GPU commands report about their graph nodes:
//! `skippy.attention.path` when a backend's attention is first seen and
//! whenever it switches between row lists, flash attention over the whole
//! K/V span and unfused attention, and `skippy.gpu.op_profiling` once per
//! backend that splits graphs by op class.

use std::sync::{Mutex, PoisonError};

use skippy_runtime::native_tracing::gpu_commands::{AttentionPath, GpuCommandTiming};

use super::super::{
    log_events::{self, AttentionPathSeen, GpuOpProfiling},
    trace_context::Frame,
};

#[derive(Debug, Default)]
struct PathTracker {
    /// Each backend's latest attention path.
    attention: Vec<(&'static str, AttentionPath)>,
    profiling: Vec<&'static str>,
}

impl PathTracker {
    fn observe(
        &mut self,
        timing: &GpuCommandTiming,
    ) -> (Option<AttentionPathSeen>, Option<GpuOpProfiling>) {
        let Some(classes) = timing.op_classes else {
            return (None, None);
        };
        let profiling =
            (classes.split_by_class && !self.profiling.contains(&timing.backend)).then(|| {
                self.profiling.push(timing.backend);
                GpuOpProfiling {
                    backend: timing.backend,
                }
            });
        let Some(path) = classes
            .attention_path()
            .filter(|_| classes.attention_kv_cells > 0)
        else {
            return (None, profiling);
        };
        let changed = match self
            .attention
            .iter_mut()
            .find(|(backend, _)| *backend == timing.backend)
        {
            Some((_, seen)) if *seen == path => false,
            Some((_, seen)) => {
                *seen = path;
                true
            }
            None => {
                self.attention.push((timing.backend, path));
                true
            }
        };
        let seen = AttentionPathSeen {
            backend: timing.backend,
            path,
            kv_cells: classes.attention_kv_cells,
            queries: classes.attention_queries,
        };
        (changed.then_some(seen), profiling)
    }
}

static TRACKER: Mutex<PathTracker> = Mutex::new(PathTracker {
    attention: Vec::new(),
    profiling: Vec::new(),
});

/// Record the events `timing` implies, under `frame`: the call that
/// submitted its work.
pub(super) fn record(frame: &Frame, timing: &GpuCommandTiming) {
    if !log_events::enabled() {
        return;
    }
    let (path, profiling) = TRACKER
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .observe(timing);
    if let Some(profiling) = profiling {
        log_events::record_under(frame, &profiling);
    }
    if let Some(path) = path {
        log_events::record_under(frame, &path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use skippy_runtime::native_tracing::gpu_commands::{GpuOpClass, GpuOpClasses, PerOpClass};

    fn command(
        row_list_nodes: u32,
        flash_nodes: u32,
        kv_cells: u32,
        split: bool,
    ) -> GpuCommandTiming {
        GpuCommandTiming {
            tag: 0,
            backend: "MTL",
            context_id: 1,
            compute_index: 1,
            command_index: 0,
            command_count: 1,
            node_count: 4,
            status: 0,
            encode_start_unix_nanos: 0,
            encode_end_unix_nanos: 0,
            execute_start_unix_nanos: 1,
            execute_end_unix_nanos: 2,
            op_classes: Some(GpuOpClasses {
                single: Some(GpuOpClass::Attention),
                layer: Some(3),
                split_by_class: split,
                attention_kv_cells: kv_cells,
                attention_queries: 4,
                attention_row_list_nodes: row_list_nodes,
                attention_flash_nodes: flash_nodes,
                nodes: PerOpClass {
                    attention: 1,
                    ..PerOpClass::default()
                },
                bytes: PerOpClass::default(),
                recurrent_states: None,
            }),
        }
    }

    #[test]
    fn the_path_is_reported_when_first_seen_and_when_it_switches() {
        let mut tracker = PathTracker::default();
        let (first, _) = tracker.observe(&command(1, 1, 512, false));
        assert_eq!(first.map(|seen| seen.path), Some(AttentionPath::RowLists));
        assert_eq!(tracker.observe(&command(1, 1, 600, false)).0, None);
        let (flash, _) = tracker.observe(&command(0, 1, 4096, false));
        assert_eq!(
            flash.map(|seen| (seen.path, seen.kv_cells)),
            Some((AttentionPath::Flash, 4096))
        );
        let (unfused, _) = tracker.observe(&command(0, 0, 4096, false));
        assert_eq!(unfused.map(|seen| seen.path), Some(AttentionPath::Unfused));
        assert_eq!(tracker.observe(&command(0, 0, 0, false)).0, None);
    }

    #[test]
    fn profiling_is_reported_once_per_backend() {
        let mut tracker = PathTracker::default();
        assert!(tracker.observe(&command(1, 1, 512, true)).1.is_some());
        assert!(tracker.observe(&command(1, 1, 512, true)).1.is_none());
        assert!(tracker.observe(&command(1, 1, 512, false)).1.is_none());
    }
}
