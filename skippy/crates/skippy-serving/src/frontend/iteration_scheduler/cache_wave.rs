//! Cooperative restore waves count usable token state, not allocated sampling lanes.
use super::*;

impl SchedulerWorker {
    /// Refresh only while cache work needs wave admission. A sampling setup may
    /// allocate a lane before its required restore/prefill has been scheduled.
    pub(super) fn refresh_cache_wave_sessions(&mut self) {
        if !self.cache_runtime_queue.has_wave_aware_operations() {
            return;
        }
        self.cache_wave_sessions = lock_runtime(&self.runtime).nonempty_active_session_count();
        if self.cache_wave_sessions < self.max_direct_batch_size {
            self.direct_wave_full = false;
        }
    }

    pub(super) fn should_defer_cache_runtime(&self) -> bool {
        self.cache_runtime_queue.has_wave_aware_operations()
            && self.direct_wave_full
            && self.cache_wave_sessions >= self.max_direct_batch_size
            && self.direct_iterations.is_empty()
            && self.requests.is_empty()
            && !self.cache_runtime_queue.is_empty()
    }

    pub(super) fn run_work_turn(&mut self) {
        let has_cache_runtime = !self.cache_runtime_queue.is_empty();
        let has_wave_aware_cache_runtime = self.cache_runtime_queue.has_wave_aware_operations();
        let has_direct = !self.direct_iterations.is_empty();
        let has_planned = !self.requests.is_empty();
        let has_iteration = has_planned || has_direct;
        if has_wave_aware_cache_runtime
            && has_direct
            && self.cache_wave_sessions >= self.max_direct_batch_size
        {
            self.direct_wave_full = true;
        }
        let suppress_cache_runtime = should_suppress_cache_runtime(
            has_wave_aware_cache_runtime,
            self.direct_wave_full,
            self.cache_wave_sessions,
            self.max_direct_batch_size,
        );
        let fill_cache_wave = should_fill_cache_runtime_wave(
            has_direct,
            has_wave_aware_cache_runtime,
            self.cache_wave_sessions,
            self.max_direct_batch_size,
        );
        let serve_cache_runtime = !suppress_cache_runtime
            && (fill_cache_wave
                || should_serve_cache_runtime(
                    has_cache_runtime,
                    has_iteration,
                    self.last_served_cache_runtime,
                ));
        self.cache_runtime_queue.advance_turn();
        if serve_cache_runtime {
            self.last_served_cache_runtime = true;
            self.run_cache_runtime_operation();
        } else {
            self.last_served_cache_runtime = false;
            self.run_iteration();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    fn worker(runtime: Arc<Mutex<RuntimeState>>, executable: usize) -> SchedulerWorker {
        let (_commands, receiver) = std_mpsc::channel();
        SchedulerWorker {
            compute_meter: Arc::default(),
            runtime,
            scheduler: Scheduler::new(build_scheduler_config(
                4,
                8192,
                0,
                Some(32),
                Some(32),
                8,
                false,
            )),
            requests: BTreeMap::new(),
            direct_iterations: VecDeque::new(),
            cache_runtime_queue: CacheRuntimeQueue::new(CACHE_AGING_COST_PER_TURN, true),
            commands: receiver,
            kv_capacity_tokens: 8192,
            max_direct_batch_size: 4,
            direct_group_batch_size: 4,
            max_direct_iteration_tokens: 32,
            max_commands_per_turn: 8,
            iteration_interval: Duration::ZERO,
            active_runtime_sessions: 4,
            cache_wave_sessions: executable,
            direct_wave_full: true,
            telemetry: None,
            last_served_direct: true,
            last_served_cache_runtime: false,
            last_emitted_lifecycle_counters: (0, 0, 0, 0),
        }
    }

    fn queue_restore(worker: &mut SchedulerWorker, operation: RuntimeOperation) {
        worker.cache_runtime_queue.enqueue_with_policy(
            operation,
            skippy_scheduler::CacheAffinity::default(),
            Arc::from([1, 2, 3]),
            0,
            CacheSchedulingPolicy::Cooperative,
            None,
        );
        for index in 0..3 {
            let mut direct = super::super::tests::direct_iteration(&format!("decode-{index}"), 1);
            // If restore is incorrectly suppressed, draining this test batch
            // must fail safely rather than touching a modelless runtime.
            direct.deadline = Some(Instant::now());
            worker.direct_iterations.push_back(direct);
        }
    }

    #[test]
    fn fourth_allocated_lane_does_not_suppress_its_required_restore() {
        let runtime = Arc::new(Mutex::new(RuntimeState::new_modelless_for_test(4)));
        let mut worker = worker(runtime, 3);
        let restored = Arc::new(AtomicBool::new(false));
        let observed = restored.clone();
        queue_restore(
            &mut worker,
            RuntimeOperation {
                label: "fourth-required-restore",
                control: None,
                run: Box::new(move |_| {
                    observed.store(true, Ordering::Release);
                    Duration::ZERO
                }),
            },
        );
        assert!(!worker.should_defer_cache_runtime());
        worker.run_work_turn();
        assert!(
            restored.load(Ordering::Acquire),
            "three executable lanes must allow the fourth restore"
        );
        assert_eq!(
            worker.direct_iterations.len(),
            3,
            "restore must run before the existing wave drains"
        );
        assert!(worker.cache_runtime_queue.is_empty());
    }

    #[test]
    fn four_executable_lanes_keep_the_existing_wave_protection() {
        let runtime = Arc::new(Mutex::new(RuntimeState::new_modelless_for_test(4)));
        let mut worker = worker(runtime, 4);
        let restored = Arc::new(AtomicBool::new(false));
        let observed = restored.clone();
        queue_restore(
            &mut worker,
            RuntimeOperation {
                label: "next-wave-restore",
                control: None,
                run: Box::new(move |_| {
                    observed.store(true, Ordering::Release);
                    Duration::ZERO
                }),
            },
        );
        worker.run_work_turn();
        assert!(!restored.load(Ordering::Acquire));
        assert!(!worker.cache_runtime_queue.is_empty());
        assert!(worker.should_defer_cache_runtime());
    }

    /// Partial prefill is already direct iteration work. A full nonempty wave
    /// may suppress the next request's restore without blocking this suffix.
    fn partial_prefill_progresses_inside_full_wave(
        worker: &mut SchedulerWorker,
        runtime: &Arc<Mutex<RuntimeState>>,
    ) -> anyhow::Result<()> {
        worker.direct_iterations.clear();
        let mut suffix = super::super::tests::direct_iteration("sampling-only", 1);
        suffix.token_ids = vec![4];
        suffix.sample_last = false;
        worker.direct_iterations.push_back(suffix);
        let next_restore = Arc::new(AtomicBool::new(false));
        let observed = next_restore.clone();
        worker.cache_runtime_queue.enqueue_with_policy(
            RuntimeOperation {
                label: "next-wave-required-restore",
                control: None,
                run: Box::new(move |_| {
                    observed.store(true, Ordering::Release);
                    Duration::ZERO
                }),
            },
            skippy_scheduler::CacheAffinity::default(),
            Arc::from([9]),
            0,
            CacheSchedulingPolicy::Cooperative,
            None,
        );
        worker.refresh_cache_wave_sessions();
        worker.run_work_turn();
        assert!(!next_restore.load(Ordering::Acquire));
        assert!(!worker.cache_runtime_queue.is_empty());
        let rt = lock_runtime(runtime);
        assert_eq!(
            rt.canonical_session_position("sampling-only")?,
            4,
            "remaining suffix must progress inside the full wave"
        );
        assert_eq!(rt.active_session_count(), 4);
        for index in 0..3 {
            assert_eq!(
                rt.canonical_session_position(&format!("decode-{index}"))?,
                3
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL and _LAYERS; actual loaded checkpoint runtime"]
    fn sampling_only_fourth_native_lane_restores_before_other_lanes_are_released()
    -> anyhow::Result<()> {
        skippy_runtime::test_support::require_native_runtime();
        let config = StageConfig {
            model_path: Some(env::var("SKIPPY_RECURRENT_CACHE_TEST_MODEL")?),
            layer_end: env::var("SKIPPY_RECURRENT_CACHE_TEST_MODEL_LAYERS")?.parse()?,
            ctx_size: 2048,
            lane_count: 4,
            n_batch: Some(32),
            n_ubatch: Some(32),
            n_gpu_layers: env::var("SKIPPY_RECURRENT_CACHE_TEST_GPU_LAYERS")
                .ok()
                .map(|v| v.parse())
                .transpose()?
                .unwrap_or(0),
            flash_attn_type: skippy_protocol::FlashAttentionType::Enabled,
            ..Default::default()
        };
        let runtime = crate::runtime_state::load_runtime(&config)?.expect("native runtime loaded");
        assert!(
            crate::runtime_state::loaded_memory_cache_capabilities(Some(&runtime))
                .unwrap()
                .resident_checkpoint
        );
        {
            let mut rt = lock_runtime(&runtime);
            for index in 0..3 {
                rt.iteration_batch_sampled(&[RuntimeIterationBatchRequest {
                    session_id: &format!("decode-{index}"),
                    token_ids: &[1, 2, 3],
                    positions: &[],
                    sampling: None,
                    input: None,
                    sample_last: false,
                    phase: IterationBatchPhase::Prefill,
                }])?;
            }
            rt.save_resident_prefix("decode-0", 16, 3)?;
            rt.configure_chat_sampling("sampling-only", r#"{"grammar":""}"#, 4, None)?;
            assert_eq!(
                rt.active_session_count(),
                4,
                "sampling reserves a real execution lane"
            );
            assert_eq!(rt.nonempty_active_session_count(), 3);
            assert_eq!(rt.canonical_session_position("sampling-only")?, 0);
        }
        let mut worker = worker(runtime.clone(), 4);
        let (operation, result) = runtime_operation("fourth-required-native-restore", |rt| {
            rt.restore_resident_prefix("sampling-only", 16, &[1, 2, 3])
                .map_err(openai_backend_error)?;
            Ok((
                rt.active_session_count(),
                rt.canonical_session_position("sampling-only")
                    .map_err(openai_backend_error)?,
            ))
        });
        queue_restore(&mut worker, operation);
        worker.refresh_cache_wave_sessions();
        assert_eq!(
            worker.active_runtime_sessions, 4,
            "physical admission remains unchanged"
        );
        assert_eq!(worker.cache_wave_sessions, 3);
        assert!(!worker.should_defer_cache_runtime());
        worker.run_work_turn();
        let restored = result.recv_timeout(Duration::from_secs(2))??;
        assert_eq!(
            restored.value,
            (4, 3),
            "restore must retain all existing lane owners"
        );
        assert_eq!(worker.direct_iterations.len(), 3);
        assert_eq!(lock_runtime(&runtime).nonempty_active_session_count(), 4);
        partial_prefill_progresses_inside_full_wave(&mut worker, &runtime)?;
        lock_runtime(&runtime).drop_resident_prefix_sequence("sampling-only", 16)?;
        Ok(())
    }
}
