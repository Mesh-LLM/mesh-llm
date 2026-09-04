use super::*;

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
        reply,
    }
}

#[test]
fn expired_direct_iteration_behind_blocked_worker_never_reaches_native_runtime() {
    let runtime = Arc::new(Mutex::new(RuntimeState::new_modelless_for_test(1)));
    let (commands, receiver) = std_mpsc::sync_channel(8);
    let worker = thread::spawn(move || {
        SchedulerWorker {
            runtime,
            scheduler: Scheduler::new(build_scheduler_config(1, 64, 0, Some(8), Some(8), 8)),
            requests: BTreeMap::new(),
            direct_iterations: VecDeque::new(),
            cache_runtime_queue: CacheRuntimeQueue::new(CACHE_AGING_COST_PER_TURN, true),
            commands: receiver,
            kv_capacity_tokens: 64,
            max_direct_batch_size: 1,
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
            }),
        }))
        .unwrap();
    worker_blocked_rx.recv().unwrap();

    let deadline = Instant::now() + Duration::from_millis(10);
    let (reply, result) = std_mpsc::sync_channel(1);
    let mut request = direct_iteration("expired-deferred-suffix", 1);
    request.deadline = Some(deadline);
    request.reply = reply;
    commands
        .send(SchedulerCommand::ExecuteIteration(request))
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
        .send(Err(OpenAiError::backend("late worker result")))
        .unwrap();

    let error = caller.join().unwrap().unwrap_err();
    assert!(error.to_string().contains("deadline exceeded"));
}

#[test]
fn queued_direct_iteration_carries_cancellation_state() {
    let cancellation = openai_frontend::CancellationToken::new();
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
    let config = build_scheduler_config(32, 131_072, 1024, Some(4096), Some(128), 64);
    assert_eq!(config.max_active_sequences, 32);
    assert_eq!(config.max_waiting_sequences, 64);
    assert_eq!(config.max_tokens_per_iteration, 2048);
    assert_eq!(config.prefill_chunk_tokens, 128);
    assert_eq!(config.max_consecutive_prefill_iterations, 1);
    assert!(!config.mixed_prefill_decode);
    assert_eq!(config.memory_components[0].capacity_bytes, 131_072);
    assert_eq!(config.memory_components[1].bytes_per_sequence, 1024);
    assert_eq!(config.memory_components[1].capacity_bytes, 65_536);

    let non_recurrent = build_scheduler_config(32, 131_072, 0, Some(4096), Some(128), 64);
    assert!(non_recurrent.mixed_prefill_decode);
}

#[test]
fn server_scheduler_worker_batches_and_completes_default_generations() {
    let runtime = Arc::new(Mutex::new(RuntimeState::new_modelless_for_test(2)));
    let (_commands, receiver) = std_mpsc::channel();
    let mut worker = SchedulerWorker {
        runtime,
        scheduler: Scheduler::new(build_scheduler_config(2, 64, 0, Some(8), Some(8), 8)),
        requests: BTreeMap::new(),
        direct_iterations: VecDeque::new(),
        cache_runtime_queue: CacheRuntimeQueue::new(CACHE_AGING_COST_PER_TURN, true),
        commands: receiver,
        kv_capacity_tokens: 64,
        max_direct_batch_size: 2,
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
fn token_control_is_applied_without_blocking_the_scheduler_iteration() {
    let runtime = Arc::new(Mutex::new(RuntimeState::new_modelless_for_test(1)));
    let (_commands, receiver) = std_mpsc::channel();
    let mut worker = SchedulerWorker {
        runtime,
        scheduler: Scheduler::new(build_scheduler_config(1, 64, 0, Some(8), Some(8), 8)),
        requests: BTreeMap::new(),
        direct_iterations: VecDeque::new(),
        cache_runtime_queue: CacheRuntimeQueue::new(CACHE_AGING_COST_PER_TURN, true),
        commands: receiver,
        kv_capacity_tokens: 64,
        max_direct_batch_size: 1,
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
        runtime: Arc::clone(&runtime),
        scheduler: Scheduler::new(build_scheduler_config(1, 64, 0, Some(8), Some(8), 8)),
        requests: BTreeMap::new(),
        direct_iterations: VecDeque::new(),
        cache_runtime_queue: CacheRuntimeQueue::new(CACHE_AGING_COST_PER_TURN, true),
        commands: receiver,
        kv_capacity_tokens: 64,
        max_direct_batch_size: 1,
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
            runtime,
            scheduler: Scheduler::new(build_scheduler_config(3, 64, 0, Some(8), Some(8), 8)),
            requests: BTreeMap::new(),
            direct_iterations: VecDeque::new(),
            cache_runtime_queue: CacheRuntimeQueue::new(CACHE_AGING_COST_PER_TURN, true),
            commands: receiver,
            kv_capacity_tokens: 64,
            max_direct_batch_size: 3,
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
            runtime,
            scheduler: Scheduler::new(build_scheduler_config(3, 64, 0, Some(8), Some(8), 8)),
            requests: BTreeMap::new(),
            direct_iterations: VecDeque::new(),
            cache_runtime_queue: CacheRuntimeQueue::new(CACHE_AGING_COST_PER_TURN, true),
            commands: receiver,
            kv_capacity_tokens: 64,
            max_direct_batch_size: 3,
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
    let cancellation = openai_frontend::CancellationToken::new();
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
    let cancellation = openai_frontend::CancellationToken::new();
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
        runtime,
        scheduler: Scheduler::new(build_scheduler_config(1, 64, 0, Some(8), Some(8), 8)),
        requests: BTreeMap::new(),
        direct_iterations: VecDeque::new(),
        cache_runtime_queue: CacheRuntimeQueue::new(CACHE_AGING_COST_PER_TURN, true),
        commands: receiver,
        kv_capacity_tokens: 64,
        max_direct_batch_size: 1,
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
        runtime,
        scheduler: Scheduler::new(build_scheduler_config(1, 64, 0, Some(8), Some(8), 8)),
        requests: BTreeMap::new(),
        direct_iterations: VecDeque::new(),
        cache_runtime_queue: CacheRuntimeQueue::new(CACHE_AGING_COST_PER_TURN, true),
        commands: receiver,
        kv_capacity_tokens: 64,
        max_direct_batch_size: 1,
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
            runtime,
            scheduler: Scheduler::new(build_scheduler_config(1, 64, 0, Some(8), Some(8), 8)),
            requests: BTreeMap::new(),
            direct_iterations: VecDeque::new(),
            cache_runtime_queue: CacheRuntimeQueue::new(CACHE_AGING_COST_PER_TURN, true),
            commands: receiver,
            kv_capacity_tokens: 64,
            max_direct_batch_size: 1,
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
            run: Box::new(|_| panic!("injected scheduler worker panic")),
        }))
        .unwrap();
    release_worker.send(()).unwrap();

    let SchedulerEvent::Error(error) = events.recv().unwrap() else {
        panic!("expected contained worker panic to fail the request");
    };
    assert!(error.to_string().contains("worker panicked"));
    worker.join().unwrap();
}
