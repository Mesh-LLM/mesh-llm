//! Debug telemetry for executed scheduler iterations: one span per step, with
//! the shared KV pool occupancy observed under its lease and links to the
//! requests it served.

use super::*;
use crate::telemetry::trace_context::SpanContext;
use serde_json::Value;

const KV_USED_CELLS: &str = "skippy.kv.used_cells";

/// The span and evidence for one executed planned iteration.
pub(super) struct StepTiming {
    pub(super) span: ActiveSpan,
    pub(super) execution_elapsed: Duration,
    pub(super) kv_used_cells: Option<u64>,
    pub(super) links: Vec<SpanContext>,
}

impl SchedulerWorker {
    pub(super) fn emit_step_telemetry(
        &mut self,
        step: &skippy_scheduler::IterationTelemetry,
        timing: StepTiming,
    ) {
        let StepTiming {
            mut span,
            execution_elapsed,
            kv_used_cells,
            links,
        } = timing;
        if self.telemetry.is_none() {
            return;
        }
        let elapsed = span.elapsed();
        let lifecycle_counters = (
            step.finished,
            step.failed,
            step.cancelled,
            step.rejected_overload,
        );
        let lifecycle_changed = lifecycle_counters != self.last_emitted_lifecycle_counters;
        if lifecycle_changed {
            self.last_emitted_lifecycle_counters = lifecycle_counters;
        }
        let mut attrs = BTreeMap::from([
            (
                "skippy.scheduler.iteration".to_string(),
                json!(step.iteration),
            ),
            (
                "skippy.scheduler.running".to_string(),
                json!(step.active_sequences),
            ),
            (
                "skippy.scheduler.waiting".to_string(),
                json!(step.waiting_sequences),
            ),
            (
                "skippy.scheduler.admitted".to_string(),
                json!(step.admitted),
            ),
            (
                "skippy.scheduler.preempted".to_string(),
                json!(step.preempted),
            ),
            (
                "skippy.scheduler.prefill_tokens".to_string(),
                json!(step.prefill_tokens),
            ),
            (
                "skippy.scheduler.recompute_tokens".to_string(),
                json!(step.recompute_tokens),
            ),
            (
                "skippy.scheduler.decode_tokens".to_string(),
                json!(step.decode_tokens),
            ),
            (
                "skippy.scheduler.prefix_hits".to_string(),
                json!(step.prefix_hits),
            ),
            (
                "skippy.scheduler.prefix_misses".to_string(),
                json!(step.prefix_misses),
            ),
            (
                "skippy.scheduler.step_ms".to_string(),
                json!(elapsed.as_secs_f64() * 1_000.0),
            ),
            (
                "skippy.scheduler.execution_ms".to_string(),
                json!(execution_elapsed.as_secs_f64() * 1_000.0),
            ),
            (
                "skippy.scheduler.overhead_ms".to_string(),
                json!(elapsed.saturating_sub(execution_elapsed).as_secs_f64() * 1_000.0),
            ),
        ]);
        if lifecycle_changed {
            attrs.extend([
                (
                    "skippy.scheduler.finished".to_string(),
                    json!(step.finished),
                ),
                ("skippy.scheduler.failed".to_string(), json!(step.failed)),
                (
                    "skippy.scheduler.cancelled".to_string(),
                    json!(step.cancelled),
                ),
                (
                    "skippy.scheduler.rejected_overload".to_string(),
                    json!(step.rejected_overload),
                ),
            ]);
        }
        for (name, used) in &step.component_used_bytes {
            attrs.insert(
                format!("skippy.scheduler.component.{name}.used_bytes"),
                json!(used),
            );
        }
        for (name, available) in &step.component_available_bytes {
            attrs.insert(
                format!("skippy.scheduler.component.{name}.available_bytes"),
                json!(available),
            );
            let used = step
                .component_used_bytes
                .iter()
                .find_map(|(used_name, used)| (used_name == name).then_some(*used))
                .unwrap_or(0);
            attrs.insert(
                format!("skippy.scheduler.component.{name}.free_bytes"),
                json!(available.saturating_sub(used)),
            );
        }
        insert_kv_used_cells(&mut attrs, kv_used_cells);
        links.into_iter().for_each(|context| span.link(context));
        span.extend_attributes(attrs);
        span.end();
    }
}

/// Attention cells in use across the shared KV pool, read under the caller's
/// runtime lease only when debug telemetry will record it.
pub(super) fn sample_kv_used_cells(
    telemetry: Option<&Telemetry>,
    runtime: &mut RuntimeState,
    session_id: &str,
) -> Option<u64> {
    if !telemetry.is_some_and(Telemetry::is_debug_enabled) {
        return None;
    }
    crate::telemetry::resource_metrics::kv_tiers::record_kv_pool_cells(u64::from(
        runtime.kv_pool_tokens(),
    ));
    runtime.memory_used_cells(session_id).ok()
}

pub(super) fn insert_kv_used_cells(attrs: &mut BTreeMap<String, Value>, used: Option<u64>) {
    if let Some(used) = used {
        crate::telemetry::resource_metrics::record_kv_used_cells(used);
        attrs.insert(KV_USED_CELLS.to_string(), json!(used));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::telemetry::TelemetryLevel;

    fn stage_config() -> StageConfig {
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
        .expect("minimal stage config")
    }

    fn worker_with(telemetry: Telemetry) -> SchedulerWorker {
        let (_commands, receiver) = std_mpsc::channel();
        SchedulerWorker {
            compute_meter: Arc::default(),
            runtime: Arc::new(Mutex::new(RuntimeState::new_modelless_for_test(1))),
            scheduler: Scheduler::new(build_scheduler_config(1, 64, 0, Some(8), Some(8), 8, false)),
            requests: BTreeMap::new(),
            direct_iterations: VecDeque::new(),
            cache_runtime_queue: CacheRuntimeQueue::new(CACHE_AGING_COST_PER_TURN, true),
            commands: receiver,
            kv_capacity_tokens: 64,
            max_direct_batch_size: 1,
            direct_group_batch_size: 1,
            batch_policy: crate::frontend::iteration_scheduler::batch_policy::default_policy(),
            max_direct_iteration_tokens: MAX_NATIVE_ITERATION_TOKENS,
            max_commands_per_turn: 8,
            iteration_interval: Duration::ZERO,
            active_runtime_sessions: 0,
            cache_wave_sessions: 0,
            direct_wave_full: false,
            telemetry: Some(telemetry),
            last_served_direct: false,
            last_served_cache_runtime: false,
            last_emitted_lifecycle_counters: (0, 0, 0, 0),
        }
    }

    #[test]
    fn planned_iteration_is_a_span_over_the_whole_step() {
        let (telemetry, rx) = Telemetry::captured(stage_config(), TelemetryLevel::Debug);
        let mut worker = worker_with(telemetry);
        let span = ActiveSpan::start(
            worker.telemetry.as_ref(),
            "stage.scheduler_iteration",
            SpanLevel::Debug,
        );
        thread::sleep(Duration::from_millis(2));

        worker.emit_step_telemetry(
            &skippy_scheduler::IterationTelemetry::default(),
            StepTiming {
                span,
                execution_elapsed: Duration::from_millis(1),
                kv_used_cells: Some(512),
                links: Vec::new(),
            },
        );

        let event = rx.try_recv().expect("iteration span captured");
        assert_eq!(event.event, "stage.scheduler_iteration");
        assert!(event.end_time_unix_nanos - event.start_time_unix_nanos >= 2_000_000);
        assert_eq!(event.attributes["skippy.kv.used_cells"], json!(512));
    }

    #[test]
    fn idle_iterations_are_not_exported() {
        let (telemetry, rx) = Telemetry::captured(stage_config(), TelemetryLevel::Debug);
        ActiveSpan::start(
            Some(&telemetry),
            "stage.scheduler_iteration",
            SpanLevel::Debug,
        )
        .discard();
        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn kv_used_cells_is_recorded_only_when_observed() {
        let mut attrs = BTreeMap::new();
        insert_kv_used_cells(&mut attrs, None);
        assert!(attrs.is_empty());
        insert_kv_used_cells(&mut attrs, Some(23_499));
        assert_eq!(attrs["skippy.kv.used_cells"], json!(23_499));
    }
}
