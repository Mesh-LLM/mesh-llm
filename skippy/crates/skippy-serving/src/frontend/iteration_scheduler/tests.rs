use super::*;
use crate::frontend::local_generation::CaptureTaskOutstandingGuard;

/// A direct iteration on this session panics inside the worker's batch run.
pub(super) const PANICKING_DIRECT_SESSION: &str = "panicking-direct-iteration";

fn direct_iteration(session_id: &str, token_count: usize) -> DirectIteration {
    let (reply, _result) = std_mpsc::sync_channel(1);
    DirectIteration {
        session_id: session_id.to_string(),
        target_token_count: None,
        token_ids: vec![1; token_count],
        positions: Vec::new(),
        sampling: None,
        input: None,
        sample_last: true,
        phase: IterationBatchPhase::Prefill,
        deadline: None,
        cancellation: None,
        enqueued_at: Instant::now(),
        reply: reply.into(),
    }
}

/// An exporting stage must fit one native microbatch, while a stage that
/// exports nothing (terminal or unsplit) keeps its whole-iteration budget.
#[test]
fn activation_export_stages_cap_iteration_tokens_at_the_native_microbatch() {
    assert_eq!(
        iteration_token_budget(true, 2048, None),
        usize::try_from(skippy_runtime::LLAMA_SERVER_DEFAULT_N_UBATCH).unwrap(),
        "an exporting stage without an explicit n_ubatch caps at the default microbatch"
    );
    assert_eq!(
        iteration_token_budget(true, 2048, Some(512)),
        512,
        "an exporting stage caps at its configured microbatch"
    );
    assert_eq!(
        iteration_token_budget(true, 128, Some(512)),
        128,
        "the microbatch cap never raises the stage budget"
    );
    assert_eq!(
        iteration_token_budget(false, 2048, Some(512)),
        2048,
        "a stage that exports nothing keeps the full iteration budget"
    );
}

#[test]
fn planned_prefills_share_the_export_microbatch_budget() {
    for (exports, expected_tokens) in [(true, 512), (false, 1024)] {
        let config = build_scheduler_config(4, 4096, 0, Some(2048), Some(512), 8, exports);
        let mut scheduler = Scheduler::new(config);
        for index in 0..4 {
            scheduler
                .submit(Sequence::new(
                    format!("request-{index}"),
                    vec![1; 256],
                    1,
                    None,
                    0,
                ))
                .unwrap();
        }
        let plan = scheduler.plan_iteration();
        assert_eq!(
            plan.work
                .iter()
                .map(|work| work.tokens.len())
                .sum::<usize>(),
            expected_tokens
        );
        assert_eq!(plan.work.len(), expected_tokens / 256);
    }
}

#[test]
fn expired_direct_iteration_behind_blocked_worker_never_reaches_native_runtime() {
    let runtime = Arc::new(Mutex::new(RuntimeState::new_modelless_for_test(1)));
    let (commands, receiver) = std_mpsc::sync_channel(8);
    let worker = thread::spawn(move || {
        SchedulerWorker {
            compute_meter: std::sync::Arc::default(),
            runtime,
            scheduler: Scheduler::new(build_scheduler_config(1, 64, 0, Some(8), Some(8), 8, false)),
            requests: BTreeMap::new(),
            direct_iterations: VecDeque::new(),
            cache_runtime_queue: CacheRuntimeQueue::new(CACHE_AGING_COST_PER_TURN, true),
            commands: receiver,
            kv_capacity_tokens: 64,
            max_direct_batch_size: 1,
            direct_group_batch_size: 1,
            max_direct_iteration_tokens: MAX_NATIVE_ITERATION_TOKENS,
            max_commands_per_turn: 8,
            iteration_interval: Duration::ZERO,
            active_runtime_sessions: 0,
            direct_wave_full: false,
            telemetry: None,
            last_served_direct: false,
            last_served_cache_runtime: false,
            last_emitted_lifecycle_counters: (0, 0, 0, 0),
        }
        .run();
    });
    let (worker_blocked, worker_blocked_rx) = std_mpsc::sync_channel(0);
    let (release_worker, release_worker_rx) = std_mpsc::sync_channel(0);
    commands
        .send(SchedulerCommand::ExecuteRuntime(RuntimeOperation {
            label: "deadline-test-gate",
            control: None,
            run: Box::new(move |_| {
                worker_blocked.send(()).unwrap();
                release_worker_rx.recv().unwrap();
                Duration::ZERO
            }),
        }))
        .unwrap();
    worker_blocked_rx.recv().unwrap();

    let deadline = Instant::now() + Duration::from_millis(10);
    let (reply, result) = std_mpsc::sync_channel(1);
    let mut request = direct_iteration("expired-deferred-suffix", 1);
    request.deadline = Some(deadline);
    request.reply = reply.into();
    commands
        .send(SchedulerCommand::ExecuteIteration(Box::new(request)))
        .unwrap();
    thread::sleep(
        deadline
            .saturating_duration_since(Instant::now())
            .saturating_add(Duration::from_millis(1)),
    );
    release_worker.send(()).unwrap();

    let error = result
        .recv_timeout(Duration::from_secs(1))
        .expect("blocked direct iteration returned")
        .unwrap_err();
    assert!(error.to_string().contains("deadline exceeded"));
    commands.send(SchedulerCommand::Shutdown).unwrap();
    worker.join().unwrap();
}

#[test]
fn direct_iteration_rechecks_deadline_after_worker_reply() {
    let (commands, receiver) = std_mpsc::sync_channel(1);
    let scheduler = IterationScheduler {
        shared: Arc::new(IterationSchedulerShared {
            commands,
            max_direct_iteration_tokens: MAX_NATIVE_ITERATION_TOKENS,
            owner_count: AtomicUsize::new(1),
            worker: Mutex::new(None),
        }),
    };
    let deadline = Instant::now() + Duration::from_millis(10);
    let caller = thread::spawn(move || {
        let channel = scheduler.direct_iteration_channel();
        scheduler.execute_iteration_on(
            &channel,
            "late-worker-reply",
            &[1],
            &[],
            None,
            false,
            IterationBatchPhase::Prefill,
            Some(deadline),
            None,
        )
    });

    let SchedulerCommand::ExecuteIteration(request) = receiver.recv().unwrap() else {
        panic!("expected direct iteration command");
    };
    thread::sleep(
        deadline
            .saturating_duration_since(Instant::now())
            .saturating_add(Duration::from_millis(1)),
    );
    request
        .reply
        .send(Err(InferenceError::backend("late worker result")));

    let error = caller.join().unwrap().unwrap_err();
    assert!(error.to_string().contains("deadline exceeded"));
}

#[test]
fn queued_direct_iteration_carries_cancellation_state() {
    let cancellation = skippy_inference_api::CancellationToken::new();
    let mut request = direct_iteration("cancelled-deferred-suffix", 1);
    request.cancellation = Some(cancellation.clone());
    cancellation.cancel();

    let error = request.ensure_active().unwrap_err();
    assert!(error.to_string().contains("request cancelled"));
}

#[test]
fn continuous_batching_controls_multi_request_direct_iterations() {
    let mut enabled = VecDeque::from([
        direct_iteration("session-a", 1),
        direct_iteration("session-b", 1),
    ]);
    let mut disabled = VecDeque::from([
        direct_iteration("session-a", 1),
        direct_iteration("session-b", 1),
    ]);

    let enabled_batch = take_direct_iteration_batch(
        &mut enabled,
        effective_scheduler_lane_count(2, false, true),
        2,
    );
    let disabled_batch = take_direct_iteration_batch(
        &mut disabled,
        effective_scheduler_lane_count(2, false, false),
        2,
    );

    assert_eq!(enabled_batch.len(), 2);
    assert_eq!(disabled_batch.len(), 1);
    assert_eq!(disabled.len(), 1);
}

#[test]
fn direct_coalescing_tracks_active_sessions_without_penalizing_singletons() {
    assert_eq!(direct_coalesce_target(1, 1, 16), 1);
    assert_eq!(direct_coalesce_target(16, 1, 16), 16);
    assert_eq!(direct_coalesce_target(32, 4, 16), 16);
    assert_eq!(direct_coalesce_target(0, 4, 16), 4);
}

#[test]
fn server_scheduler_config_uses_runtime_lanes_and_native_batch_limits() {
    let config = build_scheduler_config(32, 131_072, 1024, Some(4096), Some(128), 64, false);
    assert_eq!(config.max_active_sequences, 32);
    assert_eq!(config.max_waiting_sequences, 64);
    assert_eq!(config.max_tokens_per_iteration, 2048);
    assert_eq!(config.prefill_chunk_tokens, 128);
    assert_eq!(config.max_consecutive_prefill_iterations, 1);
    assert!(!config.mixed_prefill_decode);
    assert_eq!(config.memory_components[0].capacity_bytes, 131_072);
    assert_eq!(config.memory_components[1].bytes_per_sequence, 1024);
    assert_eq!(config.memory_components[1].capacity_bytes, 65_536);

    let non_recurrent = build_scheduler_config(32, 131_072, 0, Some(4096), Some(128), 64, false);
    assert!(non_recurrent.mixed_prefill_decode);
}

#[test]
fn server_scheduler_worker_batches_and_completes_default_generations() {
    let runtime = Arc::new(Mutex::new(RuntimeState::new_modelless_for_test(2)));
    let (_commands, receiver) = std_mpsc::channel();
    let mut worker = SchedulerWorker {
        compute_meter: std::sync::Arc::default(),
        runtime,
        scheduler: Scheduler::new(build_scheduler_config(2, 64, 0, Some(8), Some(8), 8, false)),
        requests: BTreeMap::new(),
        direct_iterations: VecDeque::new(),
        cache_runtime_queue: CacheRuntimeQueue::new(CACHE_AGING_COST_PER_TURN, true),
        commands: receiver,
        kv_capacity_tokens: 64,
        max_direct_batch_size: 2,
        direct_group_batch_size: 2,
        max_direct_iteration_tokens: MAX_NATIVE_ITERATION_TOKENS,
        max_commands_per_turn: 8,
        iteration_interval: Duration::ZERO,
        active_runtime_sessions: 0,
        direct_wave_full: false,
        telemetry: None,
        last_served_direct: false,
        last_served_cache_runtime: false,
        last_emitted_lifecycle_counters: (0, 0, 0, 0),
    };
    let (reply_a, events_a) = std_mpsc::channel();
    let (reply_b, events_b) = std_mpsc::channel();
    worker.submit(ScheduledRequest {
        id: "a".into(),
        prompt_tokens: vec![1, 2],
        max_tokens: 1,
        sampling: None,
        chat_sampling_metadata: None,
        generated_tokens: Vec::new(),
        runtime_configured: false,
        retain_runtime: false,
        reply: reply_a,
    });
    worker.submit(ScheduledRequest {
        id: "b".into(),
        prompt_tokens: vec![3, 4],
        max_tokens: 1,
        sampling: None,
        chat_sampling_metadata: None,
        generated_tokens: Vec::new(),
        runtime_configured: false,
        retain_runtime: false,
        reply: reply_b,
    });

    let receive = |events: std_mpsc::Receiver<SchedulerEvent>| {
        thread::spawn(move || {
            let mut tokens = Vec::new();
            loop {
                match events.recv().unwrap() {
                    SchedulerEvent::Token { token, ack } => {
                        tokens.push(token);
                        // The terminal token is followed by Complete and
                        // may close its request before this consumer runs.
                        // A late terminal acknowledgement is therefore a
                        // valid disconnect, not a scheduler failure.
                        let _ = ack.send(TokenControl::Continue);
                    }
                    SchedulerEvent::Complete => return tokens,
                    SchedulerEvent::Error(error) => panic!("scheduler failed: {error}"),
                }
            }
        })
    };
    let consumer_a = receive(events_a);
    let consumer_b = receive(events_b);
    let plan = worker.scheduler.plan_iteration();
    assert_eq!(plan.work.len(), 2);
    assert!(plan.work.iter().all(|work| work.sample_last));

    let predictions = [
        IterationPrediction {
            work_index: 0,
            token: 10,
        },
        IterationPrediction {
            work_index: 1,
            token: 20,
        },
    ];
    let step = worker.scheduler.complete_iteration(&plan, &predictions);
    worker.finish_iteration(&plan, &predictions);
    assert_eq!(step.admitted, 2);

    assert_eq!(consumer_a.join().unwrap(), vec![10]);
    assert_eq!(consumer_b.join().unwrap(), vec![20]);
    assert!(worker.requests.is_empty());
    assert_eq!(worker.scheduler.metrics().finished, 2);
}

#[test]
fn feature_driver_iterations_enforce_the_native_batch_shape() {
    assert!(validate_direct_iteration(&[1], &[], 512).is_ok());
    assert!(validate_direct_iteration(&[1, 2], &[0, 1], 512).is_ok());
    assert!(validate_direct_iteration(&[], &[], 512).is_err());
    assert!(validate_direct_iteration(&[1, 2], &[0, 1, 2], 512).is_err());
    assert!(validate_direct_iteration(&vec![1; 513], &[], 512).is_err());
    assert!(
        validate_direct_iteration(&vec![1; MAX_NATIVE_ITERATION_TOKENS + 1], &[], usize::MAX)
            .is_err()
    );
}

#[test]
fn direct_iteration_batch_defers_duplicate_sessions() {
    let mut queue = VecDeque::from([
        direct_iteration("same", 1),
        direct_iteration("same", 1),
        direct_iteration("other", 1),
    ]);

    let batch = take_direct_iteration_batch(&mut queue, 3, 8);

    assert_eq!(
        batch
            .iter()
            .map(|request| request.session_id.as_str())
            .collect::<Vec<_>>(),
        ["same", "other"]
    );
    assert_eq!(queue.len(), 1);
    assert_eq!(queue.front().unwrap().session_id, "same");
}

#[test]
fn direct_prefill_wave_never_exceeds_configured_native_batch_tokens() {
    let mut queue = (0..8)
        .map(|index| direct_iteration(&format!("session-{index}"), 128))
        .collect::<VecDeque<_>>();

    let batch = take_direct_iteration_batch(&mut queue, 8, 512);

    assert_eq!(batch.len(), 4);
    assert_eq!(
        batch
            .iter()
            .map(|request| request.token_ids.len())
            .sum::<usize>(),
        512
    );
    assert_eq!(queue.len(), 4);
}

#[test]
fn direct_wave_packs_decode_rows_queued_behind_an_oversized_prefill_chunk() {
    let mut queue = VecDeque::from([
        direct_iteration("prefill-a", 384),
        direct_iteration("prefill-b", 256),
        direct_iteration("decode-a", 1),
        direct_iteration("decode-b", 1),
        direct_iteration("late-arrival", 1),
    ]);

    let batch = take_direct_iteration_batch(&mut queue, 4, 512);

    assert_eq!(
        batch
            .iter()
            .map(|request| request.session_id.as_str())
            .collect::<Vec<_>>(),
        ["prefill-a", "decode-a", "decode-b", "late-arrival"]
    );
    assert_eq!(queue.len(), 1);
    assert_eq!(queue.front().unwrap().session_id, "prefill-b");
}

#[test]
fn deferred_prefill_chunk_stays_ahead_of_requests_it_did_not_reach() {
    let mut queue = VecDeque::from([
        direct_iteration("prefill-a", 384),
        direct_iteration("prefill-b", 256),
        direct_iteration("decode-a", 1),
        direct_iteration("decode-b", 1),
    ]);

    let batch = take_direct_iteration_batch(&mut queue, 2, 512);

    assert_eq!(batch.len(), 2);
    assert_eq!(
        queue
            .iter()
            .map(|request| request.session_id.as_str())
            .collect::<Vec<_>>(),
        ["prefill-b", "decode-b"]
    );
}

#[test]
fn deferred_prefill_chunk_is_not_overtaken_by_its_own_session() {
    let mut queue = VecDeque::from([
        direct_iteration("prefill-a", 384),
        direct_iteration("same", 256),
        direct_iteration("same", 1),
        direct_iteration("decode-a", 1),
    ]);

    let batch = take_direct_iteration_batch(&mut queue, 4, 512);

    assert_eq!(
        batch
            .iter()
            .map(|request| request.session_id.as_str())
            .collect::<Vec<_>>(),
        ["prefill-a", "decode-a"]
    );
    assert_eq!(
        queue
            .iter()
            .map(|request| request.token_ids.len())
            .collect::<Vec<_>>(),
        [256, 1]
    );
}

#[test]
fn token_control_is_applied_without_blocking_the_scheduler_iteration() {
    let runtime = Arc::new(Mutex::new(RuntimeState::new_modelless_for_test(1)));
    let (_commands, receiver) = std_mpsc::channel();
    let mut worker = SchedulerWorker {
        compute_meter: std::sync::Arc::default(),
        runtime,
        scheduler: Scheduler::new(build_scheduler_config(1, 64, 0, Some(8), Some(8), 8, false)),
        requests: BTreeMap::new(),
        direct_iterations: VecDeque::new(),
        cache_runtime_queue: CacheRuntimeQueue::new(CACHE_AGING_COST_PER_TURN, true),
        commands: receiver,
        kv_capacity_tokens: 64,
        max_direct_batch_size: 1,
        direct_group_batch_size: 1,
        max_direct_iteration_tokens: MAX_NATIVE_ITERATION_TOKENS,
        max_commands_per_turn: 8,
        iteration_interval: Duration::ZERO,
        active_runtime_sessions: 0,
        direct_wave_full: false,
        telemetry: None,
        last_served_direct: false,
        last_served_cache_runtime: false,
        last_emitted_lifecycle_counters: (0, 0, 0, 0),
    };
    let (reply, events) = std_mpsc::channel();
    worker.submit(ScheduledRequest {
        id: "slow-consumer".into(),
        prompt_tokens: vec![1],
        max_tokens: 2,
        sampling: None,
        chat_sampling_metadata: None,
        generated_tokens: Vec::new(),
        runtime_configured: false,
        retain_runtime: false,
        reply,
    });
    let plan = worker.scheduler.plan_iteration();
    let predictions = [IterationPrediction {
        work_index: 0,
        token: 10,
    }];
    worker.scheduler.complete_iteration(&plan, &predictions);

    worker.finish_iteration(&plan, &predictions);

    let SchedulerEvent::Token { ack, .. } = events.recv().unwrap() else {
        panic!("expected token event");
    };
    assert!(worker.requests.contains_key("slow-consumer"));
    ack.send(TokenControl::Stop).unwrap();
    worker.apply_pending_controls();
    assert!(!worker.requests.contains_key("slow-consumer"));
}

#[test]
fn resumed_request_cancellation_leaves_runtime_for_caller_cleanup() {
    let runtime = Arc::new(Mutex::new(RuntimeState::new_modelless_for_test(1)));
    runtime
        .lock()
        .unwrap()
        .track_session_tokens_for_test("resumed", 1);
    let (_commands, receiver) = std_mpsc::channel();
    let mut worker = SchedulerWorker {
        compute_meter: std::sync::Arc::default(),
        runtime: Arc::clone(&runtime),
        scheduler: Scheduler::new(build_scheduler_config(1, 64, 0, Some(8), Some(8), 8, false)),
        requests: BTreeMap::new(),
        direct_iterations: VecDeque::new(),
        cache_runtime_queue: CacheRuntimeQueue::new(CACHE_AGING_COST_PER_TURN, true),
        commands: receiver,
        kv_capacity_tokens: 64,
        max_direct_batch_size: 1,
        direct_group_batch_size: 1,
        max_direct_iteration_tokens: MAX_NATIVE_ITERATION_TOKENS,
        max_commands_per_turn: 8,
        iteration_interval: Duration::ZERO,
        active_runtime_sessions: 0,
        direct_wave_full: false,
        telemetry: None,
        last_served_direct: false,
        last_served_cache_runtime: false,
        last_emitted_lifecycle_counters: (0, 0, 0, 0),
    };
    worker
        .scheduler
        .submit(
            Sequence::new("resumed".to_string(), vec![1], 2, None, 0)
                .with_prefilled_generation(vec![7]),
        )
        .unwrap();
    let (reply, _events) = std_mpsc::channel();
    worker.requests.insert(
        "resumed".to_string(),
        RequestState {
            reply,
            pending_controls: VecDeque::new(),
            sampling: None,
            chat_sampling_metadata: None,
            prompt_token_count: 1,
            runtime_configured: true,
            retain_runtime: true,
        },
    );

    worker.cancel("resumed");

    assert_eq!(
        runtime.lock().unwrap().session_stats().tracked_token_counts,
        1
    );
    assert!(!worker.requests.contains_key("resumed"));
    assert!(worker.scheduler.sequence("resumed").is_none());
}

#[test]
fn feature_runtime_operations_execute_on_the_scheduler_worker() {
    let runtime = Arc::new(Mutex::new(RuntimeState::new_modelless_for_test(3)));
    let (commands, receiver) = std_mpsc::sync_channel(8);
    let worker = thread::spawn(move || {
        SchedulerWorker {
            compute_meter: std::sync::Arc::default(),
            runtime,
            scheduler: Scheduler::new(build_scheduler_config(3, 64, 0, Some(8), Some(8), 8, false)),
            requests: BTreeMap::new(),
            direct_iterations: VecDeque::new(),
            cache_runtime_queue: CacheRuntimeQueue::new(CACHE_AGING_COST_PER_TURN, true),
            commands: receiver,
            kv_capacity_tokens: 64,
            max_direct_batch_size: 3,
            direct_group_batch_size: 3,
            max_direct_iteration_tokens: MAX_NATIVE_ITERATION_TOKENS,
            max_commands_per_turn: 8,
            iteration_interval: Duration::ZERO,
            active_runtime_sessions: 0,
            direct_wave_full: false,
            telemetry: None,
            last_served_direct: false,
            last_served_cache_runtime: false,
            last_emitted_lifecycle_counters: (0, 0, 0, 0),
        }
        .run();
    });
    let scheduler = IterationScheduler {
        shared: Arc::new(IterationSchedulerShared {
            commands,
            max_direct_iteration_tokens: MAX_NATIVE_ITERATION_TOKENS,
            owner_count: AtomicUsize::new(1),
            worker: Mutex::new(Some(worker)),
        }),
    };

    let outcome = scheduler
        .execute_runtime_timed("test-feature-runtime", |runtime| Ok(runtime.lane_count()))
        .unwrap();
    assert_eq!(outcome.value, 3);
    assert!(outcome.queue_wait_ms >= 0.0);
    assert!(outcome.runtime_lock_wait_ms >= 0.0);
    assert!(outcome.runtime_lock_hold_ms >= 0.0);
}

#[test]
fn detached_runtime_operation_returns_before_work_completes() {
    let runtime = Arc::new(Mutex::new(RuntimeState::new_modelless_for_test(3)));
    let (commands, receiver) = std_mpsc::sync_channel(8);
    let worker = thread::spawn(move || {
        SchedulerWorker {
            compute_meter: std::sync::Arc::default(),
            runtime,
            scheduler: Scheduler::new(build_scheduler_config(3, 64, 0, Some(8), Some(8), 8, false)),
            requests: BTreeMap::new(),
            direct_iterations: VecDeque::new(),
            cache_runtime_queue: CacheRuntimeQueue::new(CACHE_AGING_COST_PER_TURN, true),
            commands: receiver,
            kv_capacity_tokens: 64,
            max_direct_batch_size: 3,
            direct_group_batch_size: 3,
            max_direct_iteration_tokens: MAX_NATIVE_ITERATION_TOKENS,
            max_commands_per_turn: 8,
            iteration_interval: Duration::ZERO,
            active_runtime_sessions: 0,
            direct_wave_full: false,
            telemetry: None,
            last_served_direct: false,
            last_served_cache_runtime: false,
            last_emitted_lifecycle_counters: (0, 0, 0, 0),
        }
        .run();
    });
    let scheduler = IterationScheduler {
        shared: Arc::new(IterationSchedulerShared {
            commands,
            max_direct_iteration_tokens: MAX_NATIVE_ITERATION_TOKENS,
            owner_count: AtomicUsize::new(1),
            worker: Mutex::new(Some(worker)),
        }),
    };
    let (started_tx, started_rx) = std_mpsc::sync_channel(1);
    let (release_tx, release_rx) = std_mpsc::sync_channel(1);

    scheduler
        .execute_runtime_detached("detached-test", move |_| {
            started_tx.send(()).unwrap();
            release_rx.recv().unwrap();
        })
        .unwrap();

    started_rx.recv_timeout(Duration::from_secs(1)).unwrap();
    release_tx.send(()).unwrap();
    scheduler
        .execute_runtime("detached-test-barrier", |_| Ok(()))
        .unwrap();
}

#[test]
fn cancelled_cache_runtime_is_rejected_before_runtime_or_user_work() {
    let runtime = Arc::new(Mutex::new(RuntimeState::new_modelless_for_test(1)));
    let cancellation = skippy_inference_api::CancellationToken::new();
    let executions = Arc::new(AtomicUsize::new(0));
    let worker_executions = executions.clone();
    let (operation, result, _) = cache_runtime_operation(
        "cancelled-cache",
        "request-7".to_string(),
        Instant::now() + Duration::from_secs(1),
        Some(&cancellation),
        move |_, _| {
            worker_executions.fetch_add(1, Ordering::Relaxed);
            Ok(())
        },
    );

    cancellation.cancel();
    (operation.run)(&runtime);

    let error = result.recv().unwrap().unwrap_err();
    assert!(error.to_string().contains("request-7 was cancelled"));
    assert_eq!(executions.load(Ordering::Relaxed), 0);
    assert_eq!(runtime.lock().unwrap().active_session_count(), 0);
}

#[test]
fn expired_cache_runtime_is_rejected_before_runtime_or_user_work() {
    let runtime = Arc::new(Mutex::new(RuntimeState::new_modelless_for_test(1)));
    let executions = Arc::new(AtomicUsize::new(0));
    let worker_executions = executions.clone();
    let (operation, result, _) = cache_runtime_operation(
        "expired-cache",
        "request-8".to_string(),
        Instant::now(),
        None,
        move |_, _| {
            worker_executions.fetch_add(1, Ordering::Relaxed);
            Ok(())
        },
    );

    (operation.run)(&runtime);

    let error = result.recv().unwrap().unwrap_err();
    assert!(
        error
            .to_string()
            .contains("request-8 exceeded its deadline")
    );
    assert_eq!(executions.load(Ordering::Relaxed), 0);
    assert_eq!(runtime.lock().unwrap().active_session_count(), 0);
}

#[test]
fn in_flight_cache_runtime_observes_cancellation_at_checkpoint() {
    let runtime = Arc::new(Mutex::new(RuntimeState::new_modelless_for_test(1)));
    let cancellation = skippy_inference_api::CancellationToken::new();
    let executions = Arc::new(AtomicUsize::new(0));
    let worker_executions = executions.clone();
    let (started, started_rx) = std_mpsc::sync_channel(0);
    let (cancelled, cancelled_rx) = std_mpsc::sync_channel(0);
    let (operation, result, _) = cache_runtime_operation(
        "in-flight-cancelled-cache",
        "request-9".to_string(),
        Instant::now() + Duration::from_secs(1),
        Some(&cancellation),
        move |_, control| {
            worker_executions.fetch_add(1, Ordering::Relaxed);
            started.send(()).unwrap();
            cancelled_rx.recv().unwrap();
            control.ensure_active()?;
            worker_executions.fetch_add(1, Ordering::Relaxed);
            Ok(())
        },
    );
    let canceller = thread::spawn(move || {
        started_rx.recv().unwrap();
        cancellation.cancel();
        cancelled.send(()).unwrap();
    });

    (operation.run)(&runtime);
    canceller.join().unwrap();

    let error = result.recv().unwrap().unwrap_err();
    assert!(error.to_string().contains("request-9 was cancelled"));
    assert_eq!(executions.load(Ordering::Relaxed), 1);
}

#[test]
fn full_direct_wave_suppresses_cache_runtime_while_direct_queue_is_temporarily_empty() {
    let runtime = Arc::new(Mutex::new(RuntimeState::new_modelless_for_test(1)));
    let (_commands, receiver) = std_mpsc::channel();
    let (selected, selected_rx) = std_mpsc::channel();
    let mut worker = SchedulerWorker {
        compute_meter: std::sync::Arc::default(),
        runtime,
        scheduler: Scheduler::new(build_scheduler_config(1, 64, 0, Some(8), Some(8), 8, false)),
        requests: BTreeMap::new(),
        direct_iterations: VecDeque::new(),
        cache_runtime_queue: CacheRuntimeQueue::new(CACHE_AGING_COST_PER_TURN, true),
        commands: receiver,
        kv_capacity_tokens: 64,
        max_direct_batch_size: 1,
        direct_group_batch_size: 1,
        max_direct_iteration_tokens: MAX_NATIVE_ITERATION_TOKENS,
        max_commands_per_turn: 8,
        iteration_interval: Duration::ZERO,
        active_runtime_sessions: 1,
        direct_wave_full: true,
        telemetry: None,
        last_served_direct: true,
        last_served_cache_runtime: false,
        last_emitted_lifecycle_counters: (0, 0, 0, 0),
    };
    worker.cache_runtime_queue.enqueue_with_payload(
        RuntimeOperation {
            label: "full-wave-cache",
            control: None,
            run: Box::new(move |_| {
                selected.send(()).unwrap();
                Duration::ZERO
            }),
        },
        skippy_scheduler::CacheAffinity::default(),
        Arc::from([1]),
        0,
        StagePrefixCachePayload::KvRecurrent,
        None,
    );

    worker.run_work_turn();

    assert!(selected_rx.try_recv().is_err());
    assert!(!worker.cache_runtime_queue.is_empty());
}

#[test]
fn resident_kv_does_not_engage_direct_wave_gate() {
    let runtime = Arc::new(Mutex::new(RuntimeState::new_modelless_for_test(1)));
    let (_commands, receiver) = std_mpsc::channel();
    let (selected, selected_rx) = std_mpsc::channel();
    let mut worker = SchedulerWorker {
        compute_meter: std::sync::Arc::default(),
        runtime,
        scheduler: Scheduler::new(build_scheduler_config(1, 64, 0, Some(8), Some(8), 8, false)),
        requests: BTreeMap::new(),
        direct_iterations: VecDeque::new(),
        cache_runtime_queue: CacheRuntimeQueue::new(CACHE_AGING_COST_PER_TURN, true),
        commands: receiver,
        kv_capacity_tokens: 64,
        max_direct_batch_size: 1,
        direct_group_batch_size: 1,
        max_direct_iteration_tokens: MAX_NATIVE_ITERATION_TOKENS,
        max_commands_per_turn: 8,
        iteration_interval: Duration::ZERO,
        active_runtime_sessions: 1,
        direct_wave_full: true,
        telemetry: None,
        last_served_direct: true,
        last_served_cache_runtime: false,
        last_emitted_lifecycle_counters: (0, 0, 0, 0),
    };
    worker.cache_runtime_queue.enqueue(
        RuntimeOperation {
            label: "resident-cache",
            control: None,
            run: Box::new(move |_| {
                selected.send(()).unwrap();
                Duration::ZERO
            }),
        },
        skippy_scheduler::CacheAffinity::default(),
        Arc::from([1]),
        0,
        None,
    );

    worker.run_work_turn();

    assert!(selected_rx.try_recv().is_ok());
    assert!(worker.cache_runtime_queue.is_empty());
}

#[test]
fn safe_mode_parser_is_explicit_and_case_insensitive() {
    for enabled in ["1", "true", "TRUE", "yes", "on"] {
        assert!(scheduler_safe_mode_from_value(Some(enabled)));
    }
    for disabled in ["0", "false", "off", "", "invalid"] {
        assert!(!scheduler_safe_mode_from_value(Some(disabled)));
    }
    assert!(!scheduler_safe_mode_from_value(None));
}

#[test]
fn direct_and_planned_work_alternate_without_starvation() {
    assert!(should_serve_direct(true, true, false));
    assert!(!should_serve_direct(true, true, true));
    assert!(should_serve_direct(true, false, true));
    assert!(!should_serve_direct(false, true, false));
    assert!(!should_serve_direct(false, false, false));
}

#[test]
fn bounded_command_queue_fails_closed_with_overload() {
    let (commands, receiver) = std_mpsc::sync_channel(1);
    commands
        .send(SchedulerCommand::Cancel("occupy-queue".into()))
        .unwrap();
    let scheduler = IterationScheduler {
        shared: Arc::new(IterationSchedulerShared {
            commands,
            max_direct_iteration_tokens: MAX_NATIVE_ITERATION_TOKENS,
            owner_count: AtomicUsize::new(1),
            worker: Mutex::new(None),
        }),
    };

    let error = scheduler
        .enqueue_command(SchedulerCommand::Cancel("rejected".into()))
        .unwrap_err();
    assert!(error.to_string().contains("generation queue is full"));

    receiver.try_recv().unwrap();
    drop(scheduler);
    assert!(matches!(
        receiver.try_recv(),
        Ok(SchedulerCommand::Shutdown)
    ));
}

#[test]
fn worker_panic_is_contained_and_fails_active_requests() {
    let runtime = Arc::new(Mutex::new(RuntimeState::new_modelless_for_test(1)));
    let (commands, receiver) = std_mpsc::sync_channel(8);
    let worker = thread::spawn(move || {
        SchedulerWorker {
            compute_meter: std::sync::Arc::default(),
            runtime,
            scheduler: Scheduler::new(build_scheduler_config(1, 64, 0, Some(8), Some(8), 8, false)),
            requests: BTreeMap::new(),
            direct_iterations: VecDeque::new(),
            cache_runtime_queue: CacheRuntimeQueue::new(CACHE_AGING_COST_PER_TURN, true),
            commands: receiver,
            kv_capacity_tokens: 64,
            max_direct_batch_size: 1,
            direct_group_batch_size: 1,
            max_direct_iteration_tokens: MAX_NATIVE_ITERATION_TOKENS,
            max_commands_per_turn: 8,
            iteration_interval: Duration::ZERO,
            active_runtime_sessions: 0,
            direct_wave_full: false,
            telemetry: None,
            last_served_direct: false,
            last_served_cache_runtime: false,
            last_emitted_lifecycle_counters: (0, 0, 0, 0),
        }
        .run();
    });
    let (worker_blocked, worker_blocked_rx) = std_mpsc::sync_channel(0);
    let (release_worker, release_worker_rx) = std_mpsc::sync_channel(0);
    commands
        .send(SchedulerCommand::ExecuteRuntime(RuntimeOperation {
            label: "panic-test-gate",
            control: None,
            run: Box::new(move |_| {
                worker_blocked.send(()).unwrap();
                release_worker_rx.recv().unwrap();
                Duration::ZERO
            }),
        }))
        .unwrap();
    worker_blocked_rx.recv().unwrap();

    let (reply, events) = std_mpsc::channel();
    commands
        .send(SchedulerCommand::Submit(ScheduledRequest {
            id: "panic-contained".into(),
            prompt_tokens: vec![1],
            max_tokens: 1,
            sampling: None,
            chat_sampling_metadata: None,
            generated_tokens: Vec::new(),
            runtime_configured: false,
            retain_runtime: false,
            reply,
        }))
        .unwrap();
    commands
        .send(SchedulerCommand::ExecuteRuntime(RuntimeOperation {
            label: "panic-test",
            control: None,
            run: Box::new(|_| -> Duration { panic!("injected scheduler worker panic") }),
        }))
        .unwrap();
    release_worker.send(()).unwrap();

    let SchedulerEvent::Error(error) = events.recv().unwrap() else {
        panic!("expected contained worker panic to fail the request");
    };
    assert!(error.to_string().contains("worker panicked"));
    // The worker survives the panic, so it stops only when told to.
    commands.send(SchedulerCommand::Shutdown).unwrap();
    worker.join().unwrap();
}

/// Exercises the REAL test-only CaptureTaskOutstandingGuard through the real
/// channel/command types: the unit is accounted at guard construction —
/// before submission — and released exactly once by its Drop, whether the
/// detached operation runs, is rejected on a full queue, or sits queued and
/// is dropped unexecuted at shutdown.
#[test]
fn detached_capture_unit_releases_on_completion_rejection_and_shutdown_drop() {
    let counter = Arc::new(AtomicUsize::new(0));

    // Rejection on a full queue: try_send returns the command inside the
    // error, and dropping that error drops the owned operation and its guard.
    let (commands, _receiver) = std_mpsc::sync_channel(1);
    commands
        .send(SchedulerCommand::ExecuteRuntime(capture_operation(
            &counter,
            "capture-unit-queued",
        )))
        .unwrap();
    let rejected = commands.try_send(SchedulerCommand::ExecuteRuntime(capture_operation(
        &counter,
        "capture-unit-rejected",
    )));
    match rejected {
        Err(std_mpsc::TrySendError::Full(command)) => drop(command),
        Err(std_mpsc::TrySendError::Disconnected(_)) => {
            panic!("scheduler queue disconnected unexpectedly")
        }
        Ok(()) => panic!("second command was unexpectedly accepted on a full queue"),
    }
    assert_eq!(
        counter.load(Ordering::Acquire),
        1,
        "only the queued operation holds a unit; the rejected one was released by its drop"
    );

    // Shutdown with the queued operation never executed: dropping the queue
    // (as fail_queued/queue teardown does) drops the operation and its guard.
    drop(commands);
    drop(_receiver);
    assert_eq!(
        counter.load(Ordering::Acquire),
        0,
        "drop-before-run releases the unit exactly once"
    );

    // Run path: executing the operation releases the unit at completion.
    let counter = Arc::new(AtomicUsize::new(0));
    let (commands, receiver) = std_mpsc::sync_channel(1);
    let runtime = Arc::new(Mutex::new(RuntimeState::new_modelless_for_test(1)));
    let worker = thread::spawn(move || {
        SchedulerWorker {
            runtime,
            compute_meter: std::sync::Arc::default(),
            scheduler: Scheduler::new(build_scheduler_config(1, 64, 0, Some(8), Some(8), 8, false)),
            requests: BTreeMap::new(),
            direct_iterations: VecDeque::new(),
            cache_runtime_queue: CacheRuntimeQueue::new(CACHE_AGING_COST_PER_TURN, true),
            commands: receiver,
            kv_capacity_tokens: 64,
            max_direct_batch_size: 1,
            direct_group_batch_size: 1,
            max_direct_iteration_tokens: MAX_NATIVE_ITERATION_TOKENS,
            max_commands_per_turn: 8,
            iteration_interval: Duration::ZERO,
            active_runtime_sessions: 0,
            direct_wave_full: false,
            telemetry: None,
            last_served_direct: false,
            last_served_cache_runtime: false,
            last_emitted_lifecycle_counters: (0, 0, 0, 0),
        }
        .run();
    });
    let (ran, ran_rx) = std_mpsc::sync_channel(0);
    let run_counter = counter.clone();
    let run_guard = CaptureTaskOutstandingGuard::new(run_counter);
    commands
        .send(SchedulerCommand::ExecuteRuntime(RuntimeOperation {
            label: "capture-unit-run",
            control: None,
            run: Box::new(move |_| {
                let _guard = run_guard;
                ran.send(()).unwrap();
                Duration::ZERO
            }),
        }))
        .unwrap();
    ran_rx.recv().unwrap();
    commands.send(SchedulerCommand::Shutdown).unwrap();
    worker.join().unwrap();
    assert_eq!(
        counter.load(Ordering::Acquire),
        0,
        "run path releases the unit at completion"
    );
}

fn capture_operation(counter: &Arc<AtomicUsize>, label: &'static str) -> RuntimeOperation {
    let guard = CaptureTaskOutstandingGuard::new(counter.clone());
    RuntimeOperation {
        label,
        control: None,
        run: Box::new(move |_| {
            let _guard = guard;
            Duration::ZERO
        }),
    }
}

#[test]
fn runtime_operation_succeeds_after_a_panic_poisons_the_runtime_lock() {
    let runtime = Arc::new(Mutex::new(RuntimeState::new_modelless_for_test(1)));
    let poisoner = Arc::clone(&runtime);
    let _ = thread::spawn(move || {
        let mut runtime = poisoner.lock().unwrap();
        runtime.track_session_tokens_for_test("panicked", 3);
        panic!("request panicked while holding the runtime lock");
    })
    .join();
    assert!(runtime.is_poisoned(), "precondition: runtime is poisoned");

    let (operation, result) = runtime_operation("after-panic", |runtime| {
        Ok(runtime.session_stats().tracked_token_counts)
    });
    (operation.run)(&runtime);
    let outcome = result
        .recv()
        .unwrap()
        .expect("the next request must not fail on a poisoned lock");
    assert_eq!(
        outcome.value, 0,
        "the panicked request's lane state is reset"
    );
    assert!(!runtime.is_poisoned());
}

#[test]
fn scheduler_worker_keeps_serving_after_an_operation_panics() {
    // A panic inside a runtime operation must not end the worker; otherwise
    // every later request finds a disconnected command channel and never
    // reaches the runtime lock's recovery.
    let runtime = Arc::new(Mutex::new(RuntimeState::new_modelless_for_test(1)));
    let (commands, receiver) = std_mpsc::sync_channel(8);
    let worker = thread::spawn(move || {
        SchedulerWorker {
            compute_meter: std::sync::Arc::default(),
            runtime,
            scheduler: Scheduler::new(build_scheduler_config(1, 64, 0, Some(8), Some(8), 8, false)),
            requests: BTreeMap::new(),
            direct_iterations: VecDeque::new(),
            cache_runtime_queue: CacheRuntimeQueue::new(CACHE_AGING_COST_PER_TURN, true),
            commands: receiver,
            kv_capacity_tokens: 64,
            max_direct_batch_size: 1,
            direct_group_batch_size: 1,
            max_direct_iteration_tokens: MAX_NATIVE_ITERATION_TOKENS,
            max_commands_per_turn: 8,
            iteration_interval: Duration::ZERO,
            active_runtime_sessions: 0,
            direct_wave_full: false,
            telemetry: None,
            last_served_direct: false,
            last_served_cache_runtime: false,
            last_emitted_lifecycle_counters: (0, 0, 0, 0),
        }
        .run();
    });

    let (panicking, panicking_result) = runtime_operation("panics", |runtime| {
        if runtime.lane_count() > 0 {
            panic!("runtime operation panicked");
        }
        Ok(0usize)
    });
    commands
        .send(SchedulerCommand::ExecuteRuntime(panicking))
        .unwrap();
    assert!(
        panicking_result
            .recv_timeout(Duration::from_secs(5))
            .map_or(true, |outcome| outcome.is_err()),
        "the panicking operation must not report success"
    );

    let (next, next_result) =
        runtime_operation("after-panic", |runtime| Ok(runtime.active_session_count()));
    commands
        .send(SchedulerCommand::ExecuteRuntime(next))
        .expect("the worker must still accept commands after a panic");
    let outcome = next_result
        .recv_timeout(Duration::from_secs(5))
        .expect("the worker must still run operations after a panic")
        .expect("the operation after the panic succeeds");
    assert_eq!(outcome.value, 0);

    commands.send(SchedulerCommand::Shutdown).unwrap();
    worker.join().unwrap();
}

#[test]
fn direct_iterations_in_a_panicking_batch_get_an_error() {
    // The waiter keeps its own reply sender, so a batch dropped by a panic
    // must still answer each request; otherwise its waiter blocks forever.
    let runtime = Arc::new(Mutex::new(RuntimeState::new_modelless_for_test(1)));
    let (commands, receiver) = std_mpsc::sync_channel(8);
    let worker = thread::spawn(move || {
        SchedulerWorker {
            compute_meter: std::sync::Arc::default(),
            runtime,
            scheduler: Scheduler::new(build_scheduler_config(1, 64, 0, Some(8), Some(8), 8, false)),
            requests: BTreeMap::new(),
            direct_iterations: VecDeque::new(),
            cache_runtime_queue: CacheRuntimeQueue::new(CACHE_AGING_COST_PER_TURN, true),
            commands: receiver,
            kv_capacity_tokens: 64,
            max_direct_batch_size: 1,
            direct_group_batch_size: 1,
            max_direct_iteration_tokens: MAX_NATIVE_ITERATION_TOKENS,
            max_commands_per_turn: 8,
            iteration_interval: Duration::ZERO,
            active_runtime_sessions: 0,
            direct_wave_full: false,
            telemetry: None,
            last_served_direct: false,
            last_served_cache_runtime: false,
            last_emitted_lifecycle_counters: (0, 0, 0, 0),
        }
        .run();
    });

    let mut request = direct_iteration(PANICKING_DIRECT_SESSION, 1);
    let (reply, result) = std_mpsc::sync_channel(1);
    let waiter_reply = reply.clone();
    request.reply = reply.into();
    commands
        .send(SchedulerCommand::ExecuteIteration(Box::new(request)))
        .unwrap();
    let outcome = result
        .recv_timeout(Duration::from_secs(5))
        .expect("a request in a panicking batch must get an answer");
    assert!(outcome.is_err());
    drop(waiter_reply);

    commands.send(SchedulerCommand::Shutdown).unwrap();
    worker.join().unwrap();
}
