use super::super::*;

fn worker(commands: std_mpsc::Receiver<SchedulerCommand>) -> SchedulerWorker {
    SchedulerWorker {
        compute_meter: Arc::default(),
        // Zero native lanes make planned runtime admission fail in Rust,
        // before the modelless fixture could dereference a native handle.
        runtime: Arc::new(Mutex::new(RuntimeState::new_modelless_for_test(0))),
        scheduler: Scheduler::new(build_scheduler_config(4, 64, 0, Some(8), Some(8), 8, false)),
        requests: BTreeMap::new(),
        direct_iterations: VecDeque::new(),
        cache_runtime_queue: CacheRuntimeQueue::new(CACHE_AGING_COST_PER_TURN, true),
        commands,
        kv_capacity_tokens: 64,
        max_direct_batch_size: 4,
        direct_group_batch_size: 4,
        max_direct_iteration_tokens: MAX_NATIVE_ITERATION_TOKENS,
        // Leave the sentinel command for coalescing or the next idle turn.
        max_commands_per_turn: 0,
        iteration_interval: Duration::ZERO,
        active_runtime_sessions: 4,
        cache_wave_sessions: 0,
        direct_wave_full: false,
        telemetry: None,
        last_served_direct: true,
        last_served_cache_runtime: false,
        last_emitted_lifecycle_counters: (0, 0, 0, 0),
    }
}

#[test]
fn active_sessions_do_not_start_an_empty_direct_batch() {
    for active in [0, 1, 4, 16, 32] {
        for limit in [0, 1, 4, 16] {
            assert_eq!(direct_coalesce_target(active, 0, limit), 0);
        }
    }
}

#[test]
fn nonempty_coalescing_still_tracks_sessions_queue_and_group_limit() {
    for (active, queued, limit, expected) in [
        (1, 1, 16, 1),
        (16, 1, 16, 16),
        (32, 4, 16, 16),
        (0, 4, 16, 4),
        (4, 1, 2, 2),
    ] {
        assert_eq!(direct_coalesce_target(active, queued, limit), expected);
    }
}

#[test]
fn nonempty_direct_batch_collects_queued_commands_to_its_target() {
    let (commands, receiver) = std_mpsc::channel();
    let mut worker = worker(receiver);
    worker.active_runtime_sessions = 2;
    worker
        .direct_iterations
        .push_back(tests::direct_iteration("first", 1));
    commands
        .send(SchedulerCommand::ExecuteIteration(Box::new(
            tests::direct_iteration("second", 1),
        )))
        .unwrap();

    assert!(worker.coalesce_direct_iterations());
    assert_eq!(worker.direct_iterations.len(), 2);
    assert_eq!(worker.direct_iterations[1].session_id, "second");
}

#[test]
fn planned_work_reaches_runtime_admission_without_coalescing_an_empty_direct_queue() {
    let (commands, receiver) = std_mpsc::channel();
    let mut worker = worker(receiver);
    let (reply, events) = std_mpsc::channel();
    worker.submit(ScheduledRequest {
        id: "planned".into(),
        prompt_tokens: vec![1, 2],
        max_tokens: 1,
        sampling: None,
        chat_sampling_metadata: None,
        generated_tokens: Vec::new(),
        runtime_configured: false,
        retain_runtime: false,
        reply,
    });
    // An erroneous empty-queue coalesce would consume Shutdown and fail the
    // planned request. Correct scheduling leaves it for the next idle turn.
    commands.send(SchedulerCommand::Shutdown).unwrap();

    worker.run_loop();

    let SchedulerEvent::Error(error) = events.try_recv().unwrap() else {
        panic!("planned work must reach the fixture's runtime admission guard");
    };
    assert!(
        error
            .to_string()
            .contains("prepare scheduler session planned: all execution lanes are busy")
    );
    assert!(worker.requests.is_empty());
    assert!(worker.direct_iterations.is_empty());
    assert!(!worker.last_served_direct);
}
