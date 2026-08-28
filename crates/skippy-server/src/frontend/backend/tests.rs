use super::*;
use openai_frontend::ChatCompletionRequest;
use openai_frontend::FinishReason;
use serde_json::json;
use tokio::runtime::Runtime;

/// A disabled telemetry sink for `StreamEventSender` construction in tests.
///
/// `TelemetryLevel::Off` makes `emit` a no-op, so these tests exercise the
/// stall/drop control flow without needing a collector; the sink only has to
/// be a valid handle.
fn test_telemetry() -> crate::telemetry::Telemetry {
    let config: skippy_protocol::StageConfig = serde_json::from_value(json!({
        "run_id": "run",
        "topology_id": "topology",
        "model_id": "org/model:Q4_K_M",
        "stage_id": "stage-0",
        "stage_index": 0,
        "layer_start": 0,
        "layer_end": 4,
        "load_mode": "runtime-slice",
        "bind_addr": "127.0.0.1:0",
    }))
    .expect("minimal stage config for telemetry");
    crate::telemetry::Telemetry::new(None, 1, config, crate::telemetry::TelemetryLevel::Off)
}

fn trusted_ids(session_id: &str) -> OpenAiGenerationIds {
    OpenAiGenerationIds::new_with_trust(OpenAiCacheHints::default(), Some(session_id), true)
}

fn trusted_session_key(session_id: &str) -> String {
    trusted_generation_session_key(&trusted_ids(session_id)).expect("trusted session key")
}

fn admission_controller(
    generation_concurrency: usize,
    generation_queue_limit: usize,
) -> GenerationAdmissionController {
    GenerationAdmissionController {
        generation_limit: Arc::new(Semaphore::new(generation_concurrency)),
        generation_queue_depth: Arc::new(AtomicUsize::new(0)),
        generation_queue_limit,
        generation_service_estimator: Arc::new(GenerationServiceEstimator::new(
            generation_concurrency,
        )),
        generation_session_locks: Arc::new(Mutex::new(BTreeMap::new())),
    }
}

fn result_error<T>(result: OpenAiResult<T>) -> OpenAiError {
    match result {
        Ok(_) => panic!("expected generation admission to fail"),
        Err(error) => error,
    }
}

#[tokio::test]
async fn predicted_wait_rejection_preserves_queue_capacity() {
    let controller = admission_controller(1, 2);
    let work = GenerationAdmissionWork::new(100, 100);
    controller
        .generation_service_estimator
        .observe_completed(work, 100.0, 100.0);
    let active = controller
        .acquire_work(
            &trusted_ids("agent-1"),
            &openai_frontend::CancellationToken::new(),
            Duration::from_secs(1),
            work,
        )
        .await
        .expect("active request admission");

    let error = result_error(
        controller
            .acquire_work(
                &trusted_ids("agent-2"),
                &openai_frontend::CancellationToken::new(),
                Duration::from_millis(199),
                work,
            )
            .await,
    );

    assert!(
        error
            .body()
            .error
            .message
            .contains("predicted generation wait")
    );
    assert_eq!(controller.generation_queue_depth.load(Ordering::Acquire), 0);
    drop(active);
    assert_eq!(controller.generation_limit.available_permits(), 1);
}

#[test]
fn session_registry_counts_live_leases_and_cleans_replaced_entries() {
    let registry = Arc::new(Mutex::new(BTreeMap::new()));
    let first = GenerationSessionPermit::new(registry.clone(), "agent-1".to_owned())
        .expect("first session lease");
    let second = GenerationSessionPermit::new(registry.clone(), "agent-1".to_owned())
        .expect("second session lease");

    {
        let locks = registry.lock().expect("session registry lock");
        let entry = locks.get("agent-1").expect("shared session entry");
        assert_eq!(entry.users.load(Ordering::Acquire), 2);
    }

    drop(first);
    {
        let locks = registry.lock().expect("session registry lock");
        let entry = locks.get("agent-1").expect("live session entry");
        assert_eq!(entry.users.load(Ordering::Acquire), 1);
    }

    drop(second);
    assert!(registry.lock().expect("session registry lock").is_empty());

    let replacement = GenerationSessionPermit::new(registry.clone(), "agent-1".to_owned())
        .expect("replacement session lease");
    assert_eq!(registry.lock().expect("session registry lock").len(), 1);
    drop(replacement);
    assert!(registry.lock().expect("session registry lock").is_empty());
}

#[tokio::test]
async fn same_trusted_session_serializes_without_consuming_global_queue_capacity() {
    let controller = admission_controller(1, 1);
    let session_key = trusted_session_key("agent-1");
    let first_cancellation = openai_frontend::CancellationToken::new();
    let first = controller
        .acquire(
            &trusted_ids("agent-1"),
            &first_cancellation,
            Duration::from_secs(1),
        )
        .await
        .expect("first session admission");

    let second_controller = controller.clone();
    let second_cancellation = openai_frontend::CancellationToken::new();
    let waiter_cancellation = second_cancellation.clone();
    let waiter = tokio::spawn(async move {
        second_controller
            .acquire(
                &trusted_ids("agent-1"),
                &waiter_cancellation,
                Duration::from_secs(1),
            )
            .await
    });

    tokio::time::timeout(Duration::from_millis(100), async {
        loop {
            let users = controller
                .generation_session_locks
                .lock()
                .expect("session registry lock")
                .get(&session_key)
                .map_or(0, |entry| entry.users.load(Ordering::Acquire));
            if users == 2 {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("second turn registered its session wait");

    assert!(!waiter.is_finished());
    assert_eq!(
        controller.generation_queue_depth.load(Ordering::Acquire),
        0,
        "session contention must not consume global queue capacity"
    );

    second_cancellation.cancel();
    let error = result_error(
        tokio::time::timeout(Duration::from_millis(100), waiter)
            .await
            .expect("cancelled session waiter returned")
            .expect("session waiter task completed"),
    );
    assert!(error.body().error.message.contains("request cancelled"));
    assert_eq!(controller.generation_queue_depth.load(Ordering::Acquire), 0);

    drop(first);
    assert_eq!(controller.generation_limit.available_permits(), 1);
    assert!(
        controller
            .generation_session_locks
            .lock()
            .expect("session registry lock")
            .is_empty()
    );
}

#[tokio::test]
async fn same_trusted_session_acquires_only_after_the_first_turn_releases() {
    let controller = admission_controller(1, 1);
    let first_cancellation = openai_frontend::CancellationToken::new();
    let first = controller
        .acquire(
            &trusted_ids("agent-1"),
            &first_cancellation,
            Duration::from_secs(1),
        )
        .await
        .expect("first session admission");

    let second_controller = controller.clone();
    let second = tokio::spawn(async move {
        second_controller
            .acquire(
                &trusted_ids("agent-1"),
                &openai_frontend::CancellationToken::new(),
                Duration::from_secs(1),
            )
            .await
    });

    tokio::time::sleep(Duration::from_millis(5)).await;
    assert!(!second.is_finished());
    assert_eq!(controller.generation_queue_depth.load(Ordering::Acquire), 0);

    drop(first);
    let second = tokio::time::timeout(Duration::from_millis(100), second)
        .await
        .expect("second turn acquired after first released")
        .expect("second turn task completed")
        .expect("second session admission");
    assert_eq!(controller.generation_limit.available_permits(), 0);
    drop(second);
    assert_eq!(controller.generation_limit.available_permits(), 1);
}

#[tokio::test]
async fn session_and_global_admission_share_one_absolute_deadline() {
    let controller = admission_controller(1, 1);
    let first_cancellation = openai_frontend::CancellationToken::new();
    let (global_permit, session_permit) = controller
        .acquire(
            &trusted_ids("agent-1"),
            &first_cancellation,
            Duration::from_secs(1),
        )
        .await
        .expect("first session admission");
    let session_permit = session_permit.expect("trusted session permit");
    let release_session = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(140)).await;
        drop(session_permit);
    });
    let started = Instant::now();

    let error = result_error(
        controller
            .acquire(
                &trusted_ids("agent-1"),
                &openai_frontend::CancellationToken::new(),
                Duration::from_millis(200),
            )
            .await,
    );

    assert!(error.body().error.message.contains("timed out waiting"));
    assert!(
        started.elapsed() < Duration::from_millis(300),
        "global-lane waiting must not restart the request admission timeout"
    );
    assert_eq!(controller.generation_queue_depth.load(Ordering::Acquire), 0);
    release_session
        .await
        .expect("session release task completed");
    drop(global_permit);
}

#[tokio::test]
async fn unrelated_session_is_not_starved_by_a_same_session_waiter() {
    let controller = admission_controller(2, 2);
    let session_key = trusted_session_key("agent-1");
    let first_cancellation = openai_frontend::CancellationToken::new();
    let first = controller
        .acquire(
            &trusted_ids("agent-1"),
            &first_cancellation,
            Duration::from_secs(1),
        )
        .await
        .expect("first session admission");

    let duplicate_controller = controller.clone();
    let duplicate_cancellation = openai_frontend::CancellationToken::new();
    let waiter_cancellation = duplicate_cancellation.clone();
    let duplicate = tokio::spawn(async move {
        duplicate_controller
            .acquire(
                &trusted_ids("agent-1"),
                &waiter_cancellation,
                Duration::from_secs(1),
            )
            .await
    });

    tokio::time::timeout(Duration::from_millis(100), async {
        loop {
            let users = controller
                .generation_session_locks
                .lock()
                .expect("session registry lock")
                .get(&session_key)
                .map_or(0, |entry| entry.users.load(Ordering::Acquire));
            if users == 2 {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("duplicate turn registered its session wait");

    assert_eq!(controller.generation_limit.available_permits(), 1);
    assert_eq!(controller.generation_queue_depth.load(Ordering::Acquire), 0);

    let unrelated = controller
        .acquire(
            &trusted_ids("agent-2"),
            &openai_frontend::CancellationToken::new(),
            Duration::from_millis(100),
        )
        .await
        .expect("unrelated session used the free global lane");
    assert_eq!(controller.generation_limit.available_permits(), 0);
    assert!(!duplicate.is_finished());

    duplicate_cancellation.cancel();
    let duplicate_error = result_error(
        tokio::time::timeout(Duration::from_millis(100), duplicate)
            .await
            .expect("duplicate waiter cancelled")
            .expect("duplicate waiter task completed"),
    );
    assert!(
        duplicate_error
            .body()
            .error
            .message
            .contains("request cancelled")
    );
    drop((first, unrelated));
    assert_eq!(controller.generation_limit.available_permits(), 2);
}

#[tokio::test]
async fn same_session_waiter_does_not_reserve_the_only_global_queue_slot() {
    let controller = admission_controller(1, 1);
    let session_key = trusted_session_key("agent-1");
    let first = controller
        .acquire(
            &trusted_ids("agent-1"),
            &openai_frontend::CancellationToken::new(),
            Duration::from_secs(1),
        )
        .await
        .expect("first session admission");

    let duplicate_controller = controller.clone();
    let duplicate_cancellation = openai_frontend::CancellationToken::new();
    let waiter_cancellation = duplicate_cancellation.clone();
    let duplicate = tokio::spawn(async move {
        duplicate_controller
            .acquire(
                &trusted_ids("agent-1"),
                &waiter_cancellation,
                Duration::from_secs(1),
            )
            .await
    });
    tokio::time::timeout(Duration::from_millis(100), async {
        loop {
            let users = controller
                .generation_session_locks
                .lock()
                .expect("session registry lock")
                .get(&session_key)
                .map_or(0, |entry| entry.users.load(Ordering::Acquire));
            if users == 2 {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("duplicate turn registered its session wait");
    assert_eq!(controller.generation_queue_depth.load(Ordering::Acquire), 0);

    let unrelated_controller = controller.clone();
    let unrelated = tokio::spawn(async move {
        unrelated_controller
            .acquire(
                &trusted_ids("agent-2"),
                &openai_frontend::CancellationToken::new(),
                Duration::from_secs(1),
            )
            .await
    });
    tokio::time::timeout(Duration::from_millis(100), async {
        while controller.generation_queue_depth.load(Ordering::Acquire) != 1 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("unrelated turn reserved the only global queue slot");
    assert!(!unrelated.is_finished());

    duplicate_cancellation.cancel();
    let duplicate_error = result_error(
        tokio::time::timeout(Duration::from_millis(100), duplicate)
            .await
            .expect("duplicate waiter cancelled")
            .expect("duplicate waiter task completed"),
    );
    assert_eq!(duplicate_error.status().as_u16(), 499);

    drop(first);
    let unrelated = tokio::time::timeout(Duration::from_millis(100), unrelated)
        .await
        .expect("unrelated turn acquired the released lane")
        .expect("unrelated waiter task completed")
        .expect("unrelated session admission");
    assert_eq!(controller.generation_queue_depth.load(Ordering::Acquire), 0);
    drop(unrelated);
}

#[tokio::test]
async fn different_trusted_sessions_can_hold_generation_lanes_concurrently() {
    let controller = admission_controller(2, 2);
    let first_cancellation = openai_frontend::CancellationToken::new();
    let second_cancellation = openai_frontend::CancellationToken::new();
    let first_ids = trusted_ids("agent-1");
    let second_ids = trusted_ids("agent-2");

    let (first, second) = tokio::join!(
        controller.acquire(&first_ids, &first_cancellation, Duration::from_secs(1)),
        controller.acquire(&second_ids, &second_cancellation, Duration::from_secs(1)),
    );
    let first = first.expect("first session admission");
    let second = second.expect("second session admission");

    assert_eq!(controller.generation_limit.available_permits(), 0);
    assert_eq!(
        controller
            .generation_session_locks
            .lock()
            .expect("session registry lock")
            .len(),
        2
    );

    drop((first, second));
    assert_eq!(controller.generation_limit.available_permits(), 2);
    assert!(
        controller
            .generation_session_locks
            .lock()
            .expect("session registry lock")
            .is_empty()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn blocking_worker_holds_global_and_session_permits_until_work_finishes() {
    let controller = admission_controller(1, 1);
    let session_key = trusted_session_key("agent-1");
    let cancellation = openai_frontend::CancellationToken::new();
    let (global_permit, session_permit) = controller
        .acquire(
            &trusted_ids("agent-1"),
            &cancellation,
            Duration::from_secs(1),
        )
        .await
        .expect("worker admission");
    let worker_context = OpenAiRequestContext::new();
    let worker_started = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let release_worker = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let started = worker_started.clone();
    let release = release_worker.clone();

    let worker = tokio::spawn(run_blocking_generation_worker(
        global_permit,
        worker_context,
        move |_| {
            let _session_permit = session_permit;
            started.store(true, Ordering::Release);
            while !release.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
        },
    ));

    tokio::time::timeout(Duration::from_millis(100), async {
        while !worker_started.load(Ordering::Acquire) {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("blocking generation worker started");

    assert_eq!(controller.generation_limit.available_permits(), 0);
    let entry = controller
        .generation_session_locks
        .lock()
        .expect("session registry lock")
        .get(&session_key)
        .expect("worker retains session entry")
        .semaphore
        .clone();
    assert_eq!(entry.available_permits(), 0);

    release_worker.store(true, Ordering::Release);
    tokio::time::timeout(Duration::from_millis(100), worker)
        .await
        .expect("blocking generation worker completed")
        .expect("worker task completed")
        .expect("blocking worker joined");
    assert_eq!(controller.generation_limit.available_permits(), 1);
    assert!(
        controller
            .generation_session_locks
            .lock()
            .expect("session registry lock")
            .is_empty()
    );
}

#[test]
fn untrusted_conversation_affinity_bypasses_session_registry() {
    let registry = Arc::new(Mutex::new(BTreeMap::new()));
    let untrusted = OpenAiGenerationIds::new_with_trust(
        OpenAiCacheHints::default(),
        Some("conversation-7"),
        false,
    );
    assert!(trusted_generation_session_key(&untrusted).is_none());
    assert!(registry.lock().expect("session registry lock").is_empty());

    let trusted = trusted_ids("agent-7");
    let key = trusted_generation_session_key(&trusted).expect("trusted session key");
    assert_eq!(key, trusted.session_id_string());
    let _permit =
        GenerationSessionPermit::new(registry.clone(), key).expect("trusted session lease");
    assert_eq!(registry.lock().expect("session registry lock").len(), 1);
}

#[test]
fn direct_backend_calls_ignore_spoofed_request_trust_metadata() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "capture-model",
        "messages": [{"role": "user", "content": "hello"}],
        "mesh_internal_agent_session_id": "spoofed-session",
        "mesh_internal_agent_session_source": "x-litellm-session-id",
        "mesh_internal_agent_session_trusted": true
    }))
    .expect("request with spoofed metadata");
    let context = OpenAiRequestContext::new();
    let ids = generation_ids(
        OpenAiCacheHints::from_chat_request(&request),
        request.agent_session(),
        &context,
    );

    assert_eq!(ids.agent_session_id.as_deref(), Some("spoofed-session"));
    assert!(!ids.agent_session_trusted);
    assert!(trusted_generation_session_key(&ids).is_none());
}

#[test]
fn internal_stream_usage_observation_preserves_client_wire_preference() {
    let direct = OpenAiRequestContext::new();
    assert!(!should_emit_stream_usage(false, &direct));
    assert!(should_emit_stream_usage(true, &direct));

    let observed = OpenAiRequestContext::new().with_stream_usage_observation();
    assert!(should_emit_stream_usage(false, &observed));
}

/// Reproduces the orphaned-generation report: a client can vanish (dropped
/// connection, or one that hasn't been noticed yet -- e.g. behind a proxy
/// that doesn't propagate the close) leaving the SSE receiver alive but
/// permanently undrained. `StreamEventSender::send` must not let that pin
/// the generation worker, and the execution lane it holds, forever: once the
/// request is cancelled it must give up promptly even though the channel
/// stays full and the receiver is never dropped.
///
/// This runs the send on its own thread and waits for a result over a
/// bounded `recv_timeout` rather than joining directly, so a regression back
/// to an unconditional blocking send fails this test instead of hanging the
/// suite. It uses the real `STREAM_SEND_STALL_TIMEOUT`, so cancellation --
/// not the stall timeout -- must be what ends the wait.
#[test]
fn stalled_receiver_does_not_pin_the_generation_worker_forever() {
    let (tx, rx) = mpsc::channel(1);
    tx.try_send(Ok(GenerationStreamEvent::Delta("first".to_owned())))
        .expect("channel has room for the first event");
    let context = OpenAiRequestContext::new();
    let rt = Runtime::new().expect("tokio runtime for stall test");
    let sender = StreamEventSender::new(
        tx,
        rt.handle().clone(),
        STREAM_SEND_STALL_TIMEOUT,
        "test-request".to_owned(),
        test_telemetry(),
    );

    let sender_context = context.clone();
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let result = sender.send(
            Ok(GenerationStreamEvent::Delta("second".to_owned())),
            &sender_context,
        );
        // Keep `rx` alive without draining it until after the send settles,
        // so a fix that works only because the channel closed doesn't pass.
        drop(rx);
        let _ = done_tx.send(result.is_err());
    });

    // Give the sender thread a chance to observe the full channel before
    // cancelling -- simulating cancellation arriving (e.g. from a
    // connection-drop observer) after the worker is already stuck sending.
    std::thread::sleep(Duration::from_millis(50));
    context.cancel();

    let cancelled = done_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("a stalled send must be interrupted by cancellation, not block forever");
    assert!(cancelled, "cancelled send must return an error");
}

/// Covers the case the report actually flagged as unproven: nothing ever
/// calls `cancel()` -- the connection-drop observer (`CancelOnDropSseStream`)
/// simply never fires, e.g. because the client vanished behind a proxy that
/// kept the socket to mesh-llm open. A stalled, never-dropped, never-drained
/// receiver must still cause the send to give up and self-cancel once it has
/// been full for the (here, injected and short) stall timeout, so the lane
/// isn't held indefinitely.
#[test]
fn stalled_receiver_self_cancels_after_the_stall_timeout_with_no_external_cancel() {
    let (tx, rx) = mpsc::channel(1);
    tx.try_send(Ok(GenerationStreamEvent::Delta("first".to_owned())))
        .expect("channel has room for the first event");
    let context = OpenAiRequestContext::new();
    let rt = Runtime::new().expect("tokio runtime for stall test");
    let sender = StreamEventSender::new(
        tx,
        rt.handle().clone(),
        Duration::from_millis(50),
        "test-request".to_owned(),
        test_telemetry(),
    );

    let result = sender.send(
        Ok(GenerationStreamEvent::Delta("second".to_owned())),
        &context,
    );

    assert!(
        result.is_err(),
        "a send stalled past the timeout must fail rather than hang"
    );
    assert!(
        context.is_cancelled(),
        "a self-detected stall must cancel the request so the lane is freed"
    );
    drop(rx);
}

/// Red->green for the swallowed-terminal-frame defect: on the pre-fix code,
/// the `run_generation_stream` cancellation branch checked
/// `context.is_cancelled()` before sending, so an already-cancelled request
/// caused the cancellation error frame -- and, by the same shape, the
/// `parser.finish` error frame and the outer generation error frame -- to be
/// silently dropped instead of enqueued. That flips
/// `stream_lifecycle`'s terminal classification: without the `Err` frame,
/// `drop_outcome()` falls through to `StreamDropOutcome::Cancelled` instead
/// of the `BackendError`/`StreamTerminal` path `lifecycle.failed(error)`
/// drives. `send_terminal` must deliver the frame to a receiver that is
/// merely cancelled but still alive and draining, while `send` (used only
/// for in-flight events) must still refuse to send once cancelled.
#[test]
fn terminal_frames_are_delivered_after_the_request_is_cancelled() {
    let (tx, mut rx) = mpsc::channel(4);
    let context = OpenAiRequestContext::new();
    context.cancel();
    let rt = Runtime::new().expect("tokio runtime for terminal-delivery test");
    let sender = StreamEventSender::new(
        tx,
        rt.handle().clone(),
        STREAM_SEND_STALL_TIMEOUT,
        "test-request".to_owned(),
        test_telemetry(),
    );

    sender
        .send_terminal(Ok(GenerationStreamEvent::Done(FinishReason::Stop)))
        .expect("terminal frames must still reach a live, cancelled-but-draining receiver");

    let received = rx
        .try_recv()
        .expect("the terminal frame must be enqueued, not silently swallowed");
    assert!(matches!(
        received,
        Ok(GenerationStreamEvent::Done(FinishReason::Stop))
    ));

    let send_result = sender.send(
        Ok(GenerationStreamEvent::Delta("late".to_owned())),
        &context,
    );
    assert!(
        send_result.is_err(),
        "the cancellation check is bypassed only for terminal frames, not in-flight ones"
    );
}

/// Bounds the double-wait hazard: once an in-flight send has already proven
/// the receiver unreachable (stalled past the timeout, here injected short),
/// a subsequent terminal send must not wait out the same stall timeout a
/// second time -- that would double the execution lane's hold to
/// `2 * stall_timeout` and defeat the point of freeing it promptly.
#[test]
fn terminal_frames_are_dropped_once_the_receiver_is_proven_unreachable() {
    let (tx, rx) = mpsc::channel(1);
    tx.try_send(Ok(GenerationStreamEvent::Delta("first".to_owned())))
        .expect("channel has room for the first event");
    let context = OpenAiRequestContext::new();
    let rt = Runtime::new().expect("tokio runtime for double-wait test");
    // Inject a generous stall timeout so the short-circuit assertion has a wide
    // margin on a loaded CI runner: a terminal send that (wrongly) waited out
    // the stall again would take at least `stall_timeout`, while the correct
    // short-circuit is one atomic load. Deriving the bound from the timeout
    // instead of a fixed wall-clock number keeps the two coupled.
    let stall_timeout = Duration::from_millis(500);
    let sender = StreamEventSender::new(
        tx,
        rt.handle().clone(),
        stall_timeout,
        "test-request".to_owned(),
        test_telemetry(),
    );

    let stalled = sender.send(
        Ok(GenerationStreamEvent::Delta("second".to_owned())),
        &context,
    );
    assert!(
        stalled.is_err(),
        "the in-flight send must self-cancel once the receiver proves unreachable"
    );

    let started = Instant::now();
    let terminal = sender.send_terminal(Ok(GenerationStreamEvent::Done(FinishReason::Stop)));
    let elapsed = started.elapsed();

    assert!(
        terminal.is_err(),
        "a proven-unreachable receiver must not be handed a terminal frame either"
    );
    // The short-circuit must complete in a small fraction of the injected
    // stall timeout; a second wait would consume at least the whole timeout.
    assert!(
        elapsed < stall_timeout / 5,
        "terminal send must short-circuit instead of waiting out the stall timeout again, took {elapsed:?} (timeout {stall_timeout:?})"
    );
    drop(rx);
}
