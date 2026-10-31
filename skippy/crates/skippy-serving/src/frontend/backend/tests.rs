mod admission;

use super::*;
use crate::frontend::EmbeddedOpenAiRequestDefaults;
use crate::frontend::SpeculativeDecodeConfig;
use crate::frontend::admission::GenerationTokenBudget;
use crate::frontend::generation::InferenceBackendMode;
use crate::frontend::iteration_scheduler::IterationScheduler;
use crate::frontend::prefill::PrefillChunkPolicy;
use crate::runtime_state::RuntimeState;
use futures_util::StreamExt;
use serde_json::json;
use skippy_inference_api::ChatCompletionChunk;
use skippy_inference_api::ChatCompletionRequest;
use skippy_inference_api::ChatCompletionResponse;
use skippy_inference_api::ChatHookOutcome;
use skippy_inference_api::FinishReason;
use skippy_inference_api::InferenceHookPolicy;
use skippy_inference_api::Usage;
use skippy_inference_api::set_chat_skippy_hooks_enabled;
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
        "execution_contract": "",
        "bind_addr": "127.0.0.1:0",
    }))
    .expect("minimal stage config for telemetry");
    crate::telemetry::Telemetry::new(None, 1, config, crate::telemetry::TelemetryLevel::Off)
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
    let context = InferenceRequestContext::new();
    let ids = generation_ids(
        InferenceCacheHints::from_chat_request(&request),
        request.agent_session(),
        &context,
    );

    assert_eq!(ids.agent_session_id.as_deref(), Some("spoofed-session"));
    assert!(!ids.agent_session_trusted);
    assert!(trusted_generation_session_key(&ids).is_none());
}

#[test]
fn internal_stream_usage_observation_preserves_client_wire_preference() {
    let direct = InferenceRequestContext::new();
    assert!(!should_emit_stream_usage(false, &direct));
    assert!(should_emit_stream_usage(true, &direct));

    let observed = InferenceRequestContext::new().with_stream_usage_observation();
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
    let context = InferenceRequestContext::new();
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
    let context = InferenceRequestContext::new();
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
    let context = InferenceRequestContext::new();
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

/// Once streaming has committed HTTP 200, a native generation failure cannot
/// be converted into a new HTTP status. It must remain an `Err` item on the
/// backend stream so `skippy-inference-api` can frame an explicit SSE error event
/// before `[DONE]` instead of making the response look like a zero-token
/// success.
#[test]
fn backend_errors_are_delivered_as_terminal_stream_frames() {
    let (tx, mut rx) = mpsc::channel(4);
    let rt = Runtime::new().expect("tokio runtime for terminal-error test");
    let sender = StreamEventSender::new(
        tx,
        rt.handle().clone(),
        STREAM_SEND_STALL_TIMEOUT,
        "test-request".to_owned(),
        test_telemetry(),
    );

    sender
        .send_terminal(Err(InferenceError::backend(
            "native decode failed to find a memory slot",
        )))
        .expect("backend failure must reach a live stream receiver");

    let error = match rx.try_recv().expect("backend failure must be enqueued") {
        Ok(_) => panic!("terminal item must remain an error"),
        Err(error) => error,
    };
    assert!(
        error
            .to_string()
            .contains("native decode failed to find a memory slot")
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
    let context = InferenceRequestContext::new();
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

// --- Terminal-hook lifecycle wiring (mesh1437 production wiring) ---
//
// `chat_completion_with_hooks`/`chat_completion_stream_with_hooks` are unit
// tested directly with a fake `dispatch` closure rather than through the
// full `chat_completion_with_context`/`chat_completion_stream` trait methods:
// real generation needs a loaded GGUF (see `recurrent_test_backend` in
// `local_generation/tests.rs`, gated on `SKIPPY_RECURRENT_CACHE_TEST_MODEL`),
// but the hook lifecycle itself never touches `self.runtime` — it only reads
// `self.hook_policy` — so it's fully exercisable on a modelless backend.

pub(super) fn hooks_test_backend(
    hook_policy: Option<Arc<dyn InferenceHookPolicy>>,
) -> StageOpenAiBackend {
    let config: skippy_protocol::StageConfig = serde_json::from_value(json!({
        "run_id": "hooks-test",
        "topology_id": "hooks-test",
        "model_id": "hooks-test-model",
        "stage_id": "stage-0",
        "stage_index": 0,
        "layer_start": 0,
        "layer_end": 1,
        "load_mode": "runtime-slice",
        "execution_contract": "",
        "bind_addr": "127.0.0.1:0",
    }))
    .expect("minimal stage config for hook lifecycle tests");
    let runtime = Arc::new(Mutex::new(RuntimeState::new_modelless_for_test(1)));
    let telemetry = crate::telemetry::Telemetry::new(
        None,
        1,
        config.clone(),
        crate::telemetry::TelemetryLevel::Off,
    );
    let iteration_scheduler =
        IterationScheduler::new(runtime.clone(), &config, 1, true, None, telemetry.clone())
            .expect("iteration scheduler for hook lifecycle tests");
    StageOpenAiBackend {
        runtime: runtime.clone(),
        workload: Default::default(),
        config: config.clone(),
        telemetry,
        model_id: "hooks-test-model".to_string(),
        default_max_tokens: 16,
        request_defaults: EmbeddedOpenAiRequestDefaults::default(),
        thinking: None,
        ctx_size: 128,
        mode: InferenceBackendMode::LocalRuntime,
        draft: None,
        speculative_window: 0,
        adaptive_speculative_window: false,
        ngram_max: 0,
        speculative: SpeculativeDecodeConfig::default(),
        speculation_governor: None,
        runahead_governor: None,
        generation_limit: Arc::new(GenerationConcurrencyController::fixed(1)),
        generation_queue_depth: Arc::new(AtomicUsize::new(0)),
        generation_queue_limit: 1,
        generation_admission_timeout: Duration::from_secs(10),
        generation_service_estimator: Arc::new(GenerationServiceEstimator::new(1)),
        generation_session_locks: Arc::new(Mutex::new(BTreeMap::new())),
        generation_token_budget: Arc::new(GenerationTokenBudget::new(128)),
        hook_policy,
        generation_receipt: None,
        generation_lifecycle: None,
        linear_proposal_ingress: None,
        kv: None,
        iteration_scheduler,
    }
}

/// Construct a stage-zero embedded topology for non-chat admission tests.
fn embedded_non_chat_test_mode(config: skippy_protocol::StageConfig) -> InferenceBackendMode {
    InferenceBackendMode::EmbeddedStageZero {
        config,
        prefill_chunk_policy: PrefillChunkPolicy::Fixed { chunk_size: 64 },
        activation_width: 0,
        downstream_wire_condition: crate::binary_transport::WireCondition::new(0.0, None)
            .expect("unconditioned test wire"),
        prefill_reply_credit_limit: 0,
        lane_pool: None,
        prediction_returns: None,
    }
}

#[test]
/// Allow non-chat execution for a complete, unsplit embedded model.
fn embedded_stage_zero_admits_unsplit_local_non_chat_topology() {
    let mut backend = hooks_test_backend(None);
    backend.mode = embedded_non_chat_test_mode(backend.config.clone());

    assert!(backend.has_unsplit_full_model_topology());
}

#[test]
/// Reject non-chat execution with incomplete layer, tensor, or expert ownership.
fn non_chat_topology_guard_rejects_staged_and_filtered_models() {
    let mut backend = hooks_test_backend(None);
    let full_config = backend.config.clone();
    let mut cases = Vec::new();

    let mut downstream = full_config.clone();
    downstream.downstream = Some(skippy_protocol::PeerConfig {
        persistent_lane_marker: false,
        stage_id: "stage-1".to_string(),
        stage_index: 1,
        endpoint: "127.0.0.1:0".to_string(),
    });
    cases.push(downstream);

    let mut upstream = full_config.clone();
    upstream.upstream = Some(skippy_protocol::PeerConfig {
        persistent_lane_marker: false,
        stage_id: "stage-previous".to_string(),
        stage_index: 0,
        endpoint: "127.0.0.1:0".to_string(),
    });
    cases.push(upstream);

    let mut filtered = full_config.clone();
    filtered.resident_tensor_names = vec!["block.0".to_string()];
    cases.push(filtered);

    let mut partial = full_config.clone();
    partial.layer_start = 1;
    cases.push(partial);

    let mut non_first_stage = full_config.clone();
    non_first_stage.stage_index = 1;
    cases.push(non_first_stage);

    let mut empty = full_config.clone();
    empty.layer_end = 0;
    cases.push(empty);

    for config in cases {
        backend.mode = embedded_non_chat_test_mode(config.clone());
        assert!(!backend.has_unsplit_full_model_topology(), "{config:?}");
        backend.mode = InferenceBackendMode::LocalRuntime;
        backend.config = config;
        assert!(!backend.has_unsplit_full_model_topology());
        backend.config = full_config.clone();
    }
}

fn mesh_hooks_request(model: &str) -> ChatCompletionRequest {
    let mut request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
    }))
    .expect("minimal chat completion request");
    set_chat_skippy_hooks_enabled(&mut request, true);
    request
}

#[derive(Debug, Clone, PartialEq)]
enum HookTerminalRecord {
    Success { model: String },
    Error { status: u16, message: String },
    Denied { status: u16, reason: String },
    Cancelled,
    StreamCompleted,
}

#[derive(Default)]
struct RecordingHookPolicy {
    deny: bool,
    hang_before_dispatch: bool,
    terminals: Mutex<Vec<HookTerminalRecord>>,
    terminal_exchange_ids: Mutex<Vec<String>>,
}

#[async_trait]
impl InferenceHookPolicy for RecordingHookPolicy {
    async fn before_chat_completion(
        &self,
        _request: &mut ChatCompletionRequest,
    ) -> InferenceResult<ChatHookOutcome> {
        if self.hang_before_dispatch {
            std::future::pending::<()>().await;
        }
        if self.deny {
            return Err(InferenceError::invalid_request("denied by policy"));
        }
        Ok(ChatHookOutcome::none())
    }

    async fn on_chat_completion_terminal(
        &self,
        _request: &ChatCompletionRequest,
        exchange_id: &str,
        outcome: &ChatCompletionOutcome<'_>,
    ) {
        self.terminal_exchange_ids
            .lock()
            .unwrap()
            .push(exchange_id.to_string());
        let record = match outcome {
            ChatCompletionOutcome::Success { response } => HookTerminalRecord::Success {
                model: response.model.clone(),
            },
            ChatCompletionOutcome::Error { status, message } => HookTerminalRecord::Error {
                status: *status,
                message: (*message).to_string(),
            },
            ChatCompletionOutcome::Denied { status, reason } => HookTerminalRecord::Denied {
                status: *status,
                reason: (*reason).to_string(),
            },
            ChatCompletionOutcome::Cancelled => HookTerminalRecord::Cancelled,
            ChatCompletionOutcome::StreamCompleted => HookTerminalRecord::StreamCompleted,
            other => {
                unreachable!("unhandled ChatCompletionOutcome variant in test fixture: {other:?}")
            }
        };
        self.terminals.lock().unwrap().push(record);
    }
}

/// Terminal delivery for a dropped/streamed exchange fires from a detached
/// spawned task (see `TerminalGuard::drop`/`fire_detached`), so it lands
/// sometime after the driving future is aborted or the stream stops
/// yielding items, not synchronously at that instant.
async fn wait_for_hook_terminal(policy: &RecordingHookPolicy) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(1);
    loop {
        if !policy.terminals.lock().unwrap().is_empty() {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "terminal event never fired"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

#[tokio::test]
async fn stage_backend_success_fires_terminal_exactly_once() {
    let policy = Arc::new(RecordingHookPolicy::default());
    let backend = hooks_test_backend(Some(policy.clone()));
    let request = mesh_hooks_request("hooks-test-model");

    let response = backend
        .chat_completion_with_hooks(request, |request| async move {
            Ok(ChatCompletionResponse::new(
                request.model,
                "ok",
                Usage::new(1, 1),
            ))
        })
        .await
        .expect("fake dispatch succeeds");
    assert_eq!(response.model, "hooks-test-model");

    let terminals = policy.terminals.lock().unwrap();
    assert_eq!(
        terminals.as_slice(),
        [HookTerminalRecord::Success {
            model: "hooks-test-model".to_string()
        }]
    );
}

/// the `exchange_id` a successful chat completion
/// response carries must be the exact same value the terminal hook observed
/// for this exchange — the shared join key that the Logs record and any other
/// consumer of the exchange events correlate on.
#[tokio::test]
async fn stage_backend_success_attaches_the_same_exchange_id_the_terminal_hook_observed() {
    let policy = Arc::new(RecordingHookPolicy::default());
    let backend = hooks_test_backend(Some(policy.clone()));
    let request = mesh_hooks_request("hooks-test-model");

    let response = backend
        .chat_completion_with_hooks(request, |request| async move {
            Ok(ChatCompletionResponse::new(
                request.model,
                "ok",
                Usage::new(1, 1),
            ))
        })
        .await
        .expect("fake dispatch succeeds");

    let terminal_exchange_ids = policy.terminal_exchange_ids.lock().unwrap();
    assert_eq!(terminal_exchange_ids.len(), 1);
    let observed_exchange_id = &terminal_exchange_ids[0];
    assert!(!observed_exchange_id.is_empty());
    assert_eq!(
        response.exchange_id.as_deref(),
        Some(observed_exchange_id.as_str())
    );
}

/// A chat completion whose request never enables mesh hooks never arms a
/// `TerminalGuard` — there is no tracked exchange, so the
/// response must carry no join-key rather than a dangling one.
#[tokio::test]
async fn stage_backend_success_without_mesh_hooks_carries_no_exchange_id() {
    let policy = Arc::new(RecordingHookPolicy::default());
    let backend = hooks_test_backend(Some(policy.clone()));
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "hooks-test-model",
        "messages": [{"role": "user", "content": "hi"}],
    }))
    .expect("minimal chat completion request");

    let response = backend
        .chat_completion_with_hooks(request, |request| async move {
            Ok(ChatCompletionResponse::new(
                request.model,
                "ok",
                Usage::new(1, 1),
            ))
        })
        .await
        .expect("fake dispatch succeeds");

    assert!(policy.terminal_exchange_ids.lock().unwrap().is_empty());
    assert_eq!(response.exchange_id, None);
}

#[tokio::test]
async fn stage_backend_dispatch_error_fires_terminal_exactly_once() {
    let policy = Arc::new(RecordingHookPolicy::default());
    let backend = hooks_test_backend(Some(policy.clone()));
    let request = mesh_hooks_request("hooks-test-model");

    let error = backend
        .chat_completion_with_hooks(request, |_request| async move {
            Err(InferenceError::backend("upstream exploded"))
        })
        .await
        .expect_err("fake dispatch fails");
    assert_eq!(error.status().as_u16(), 502);

    let terminals = policy.terminals.lock().unwrap();
    assert_eq!(terminals.len(), 1);
    assert!(matches!(
        &terminals[0],
        HookTerminalRecord::Error { status: 502, message }
            if message.contains("upstream exploded")
    ));
}

#[tokio::test]
async fn stage_backend_denied_request_never_dispatches_and_fires_terminal_exactly_once() {
    let policy = Arc::new(RecordingHookPolicy {
        deny: true,
        ..RecordingHookPolicy::default()
    });
    let backend = hooks_test_backend(Some(policy.clone()));
    let request = mesh_hooks_request("hooks-test-model");
    let dispatched = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let dispatched_flag = dispatched.clone();

    let error = backend
        .chat_completion_with_hooks(request, move |request| {
            dispatched_flag.store(true, Ordering::Release);
            async move {
                Ok(ChatCompletionResponse::new(
                    request.model,
                    "ok",
                    Usage::new(0, 0),
                ))
            }
        })
        .await
        .expect_err("policy denies the request");
    assert_eq!(error.status().as_u16(), 400);
    assert!(
        !dispatched.load(Ordering::Acquire),
        "a denied request must never reach dispatch"
    );

    let terminals = policy.terminals.lock().unwrap();
    assert_eq!(terminals.len(), 1);
    assert!(matches!(
        &terminals[0],
        HookTerminalRecord::Denied { status: 400, reason }
            if reason.contains("denied by policy")
    ));
}

#[tokio::test]
async fn stage_backend_dropped_during_admission_hook_fires_exactly_one_cancelled_terminal() {
    let policy = Arc::new(RecordingHookPolicy {
        hang_before_dispatch: true,
        ..RecordingHookPolicy::default()
    });
    let backend = hooks_test_backend(Some(policy.clone()));
    let request = mesh_hooks_request("hooks-test-model");

    let handle = tokio::spawn(async move {
        backend
            .chat_completion_with_hooks(request, |request| async move {
                Ok(ChatCompletionResponse::new(
                    request.model,
                    "ok",
                    Usage::new(0, 0),
                ))
            })
            .await
    });

    // Let the task run until it's parked in `before_chat_completion`, then
    // cancel it the way an outer timeout or client disconnect would.
    tokio::task::yield_now().await;
    handle.abort();
    let _ = handle.await;
    wait_for_hook_terminal(&policy).await;

    let terminals = policy.terminals.lock().unwrap();
    assert_eq!(terminals.as_slice(), [HookTerminalRecord::Cancelled]);
}

#[tokio::test]
async fn stage_backend_stream_that_ends_normally_fires_stream_completed_terminal_exactly_once() {
    let policy = Arc::new(RecordingHookPolicy::default());
    let backend = hooks_test_backend(Some(policy.clone()));
    let request = mesh_hooks_request("hooks-test-model");
    let context = InferenceRequestContext::new();

    let mut stream = backend
        .chat_completion_stream_with_hooks(request, &context, |request| async move {
            Ok(Box::pin(futures_util::stream::iter(vec![
                Ok(ChatCompletionChunk::delta(request.model.clone(), "hi")),
                Ok(ChatCompletionChunk::done(request.model)),
            ])) as ChatCompletionStream)
        })
        .await
        .expect("stream created");
    while stream
        .next()
        .await
        .transpose()
        .expect("no chunk errors")
        .is_some()
    {}
    wait_for_hook_terminal(&policy).await;

    let terminals = policy.terminals.lock().unwrap();
    assert_eq!(terminals.as_slice(), [HookTerminalRecord::StreamCompleted]);
    let exchange_ids = policy.terminal_exchange_ids.lock().unwrap();
    assert_eq!(
        context.exchange_id().as_deref(),
        exchange_ids.first().map(String::as_str)
    );
}

/// The explicit case this wiring exists for: a client disconnects (or an
/// outer timeout fires) after a stream has already delivered a chunk but
/// before it ends on its own. Without `TerminalGuardedChatStream` wired into
/// `StageOpenAiBackend`, this exchange got zero terminal events.
#[tokio::test]
async fn stage_backend_stream_dropped_mid_stream_fires_exactly_one_cancelled_terminal() {
    let policy = Arc::new(RecordingHookPolicy::default());
    let backend = hooks_test_backend(Some(policy.clone()));
    let request = mesh_hooks_request("hooks-test-model");
    let context = InferenceRequestContext::new();

    let mut stream = backend
        .chat_completion_stream_with_hooks(request, &context, |request| async move {
            let first = ChatCompletionChunk::delta(request.model, "partial");
            Ok(Box::pin(
                futures_util::stream::once(async move { Ok(first) })
                    .chain(futures_util::stream::pending()),
            ) as ChatCompletionStream)
        })
        .await
        .expect("stream created");
    let first = stream.next().await;
    assert!(matches!(first, Some(Ok(_))), "first chunk should flow");
    drop(stream);
    wait_for_hook_terminal(&policy).await;

    let terminals = policy.terminals.lock().unwrap();
    assert_eq!(terminals.as_slice(), [HookTerminalRecord::Cancelled]);
}

#[tokio::test]
async fn stage_backend_stream_error_chunk_fires_error_terminal_exactly_once() {
    let policy = Arc::new(RecordingHookPolicy::default());
    let backend = hooks_test_backend(Some(policy.clone()));
    let request = mesh_hooks_request("hooks-test-model");
    let context = InferenceRequestContext::new();

    let mut stream = backend
        .chat_completion_stream_with_hooks(request, &context, |request| async move {
            Ok(Box::pin(futures_util::stream::iter(vec![
                Ok(ChatCompletionChunk::delta(request.model, "hi")),
                Err(InferenceError::backend("upstream exploded")),
            ])) as ChatCompletionStream)
        })
        .await
        .expect("stream created");
    while let Some(item) = stream.next().await {
        let _ = item;
    }
    wait_for_hook_terminal(&policy).await;

    let terminals = policy.terminals.lock().unwrap();
    assert_eq!(terminals.len(), 1);
    assert!(matches!(
        &terminals[0],
        HookTerminalRecord::Error { status: 502, message }
            if message.contains("upstream exploded")
    ));
}

#[tokio::test]
async fn stage_backend_stream_denied_never_dispatches_and_fires_terminal_exactly_once() {
    let policy = Arc::new(RecordingHookPolicy {
        deny: true,
        ..RecordingHookPolicy::default()
    });
    let backend = hooks_test_backend(Some(policy.clone()));
    let request = mesh_hooks_request("hooks-test-model");
    let context = InferenceRequestContext::new();

    let error = match backend
        .chat_completion_stream_with_hooks(request, &context, |_request| async move {
            panic!("a denied request must never reach dispatch")
        })
        .await
    {
        Ok(_) => panic!("policy denies the request"),
        Err(error) => error,
    };
    assert_eq!(error.status().as_u16(), 400);

    let terminals = policy.terminals.lock().unwrap();
    assert_eq!(terminals.len(), 1);
    assert!(matches!(
        &terminals[0],
        HookTerminalRecord::Denied { status: 400, reason }
            if reason.contains("denied by policy")
    ));
}

#[test]
fn generation_ids_carries_the_frontend_request_id_byte_equal_to_the_context() {
    let request_id = skippy_inference_api::parse_request_id("c0a801ef-2a39-4f52-99f5-bdc849127cde")
        .expect("test UUID should parse");
    let context = InferenceRequestContext::with_request_id(request_id);
    let ids = generation_ids(InferenceCacheHints::default(), None, &context);
    assert_eq!(
        ids.frontend_request_id,
        Some(request_id.as_uuid().into_bytes())
    );
}

#[test]
fn generation_ids_leaves_frontend_request_id_absent_for_a_non_frontend_context() {
    let context = InferenceRequestContext::new();
    let ids = generation_ids(InferenceCacheHints::default(), None, &context);
    assert_eq!(ids.frontend_request_id, None);
}

fn identity_config() -> skippy_protocol::StageConfig {
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
    .expect("minimal stage config for identity attrs")
}

/// The frontend UUID join: when a generation entered through the OpenAI HTTP
/// boundary, the identity attrs carry `openai.frontend_request_id` (UUID
/// string) next to the native numeric ids, so the same event joins both.
#[test]
fn openai_identity_attrs_carry_the_frontend_request_uuid_when_present() {
    let expected = uuid::Uuid::new_v4();
    let ids = InferenceGenerationIds::new_with_trust(
        InferenceCacheHints::default(),
        None,
        false,
        Some(*expected.as_bytes()),
    );

    let attrs = openai_identity_attrs(&identity_config(), "test-backend", &ids);

    let parsed = uuid::Uuid::parse_str(attrs["openai.frontend_request_id"].as_str().unwrap())
        .expect("frontend request id must be a UUID string");
    assert_eq!(parsed, expected);
    assert_eq!(attrs["llama_stage.openai_backend"], json!("test-backend"));
    assert!(!attrs["skippy.request_id"].as_str().unwrap().is_empty());
    assert!(!attrs["skippy.session_id"].as_str().unwrap().is_empty());
    assert_eq!(attrs["skippy.run_id"], json!("run"));
    assert_eq!(attrs["skippy.stage_id"], json!("stage-0"));
}

/// Non-frontend callers have no frontend UUID; the key must stay absent so
/// log consumers can distinguish HTTP-admitted generations from embedded
/// ones.
#[test]
fn openai_identity_attrs_omit_the_frontend_request_uuid_without_one() {
    let ids =
        InferenceGenerationIds::new_with_trust(InferenceCacheHints::default(), None, false, None);

    let attrs = openai_identity_attrs(&identity_config(), "test-backend", &ids);

    assert!(!attrs.contains_key("openai.frontend_request_id"));
    assert!(!attrs["skippy.request_id"].as_str().unwrap().is_empty());
}

/// The embedded sync-forward seam emits exactly one downstream forward-error
/// event with the OpenAI request identity (numeric + frontend UUID) and the
/// intended downstream identity, and propagates the original io error
/// unchanged.
#[test]
fn embedded_forward_seam_emits_error_event_on_write_failure() {
    let mut config = identity_config();
    config.downstream = Some(skippy_protocol::PeerConfig {
        persistent_lane_marker: false,
        stage_id: "stage-2".to_string(),
        stage_index: 2,
        endpoint: "10.0.0.3:50052".to_string(),
    });
    let expected = uuid::Uuid::new_v4();
    let ids = InferenceGenerationIds::new_with_trust(
        InferenceCacheHints::default(),
        None,
        false,
        Some(*expected.as_bytes()),
    );
    let (telemetry, rx) = crate::telemetry::Telemetry::captured(
        config.clone(),
        crate::telemetry::TelemetryLevel::Summary,
    );

    let result = write_downstream_or_emit_forward_error(&telemetry, &config, &ids, 1234, || {
        Err(std::io::Error::new(std::io::ErrorKind::BrokenPipe, "boom"))
    });

    let error = result.unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::BrokenPipe);
    assert_eq!(error.to_string(), "boom");

    let event = rx
        .try_recv()
        .expect("exactly one forward error event on failure");
    assert_eq!(event.event, "stage.binary_downstream_forward_error");
    assert_eq!(
        event.attributes["skippy.request_id"],
        json!(ids.request_id_string())
    );
    assert_eq!(
        event.attributes["skippy.session_id"],
        json!(ids.session_id_string())
    );
    let parsed = uuid::Uuid::parse_str(
        event.attributes["openai.frontend_request_id"]
            .as_str()
            .unwrap(),
    )
    .expect("frontend uuid present on the embedded failure event");
    assert_eq!(parsed, expected);
    assert_eq!(
        event.attributes["llama_stage.downstream_stage_id"],
        json!("stage-2")
    );
    assert_eq!(
        event.attributes["llama_stage.downstream_endpoint"],
        json!("10.0.0.3:50052")
    );
    assert!(
        event.attributes["llama_stage.error"]
            .as_str()
            .unwrap()
            .contains("boom")
    );
    assert_eq!(event.start_time_unix_nanos, 1234);
    assert!(rx.try_recv().is_err(), "no additional events on failure");
}

#[test]
fn embedded_forward_seam_emits_nothing_on_success() {
    let mut config = identity_config();
    config.downstream = Some(skippy_protocol::PeerConfig {
        persistent_lane_marker: false,
        stage_id: "stage-2".to_string(),
        stage_index: 2,
        endpoint: "10.0.0.3:50052".to_string(),
    });
    let ids =
        InferenceGenerationIds::new_with_trust(InferenceCacheHints::default(), None, false, None);
    let (telemetry, rx) = crate::telemetry::Telemetry::captured(
        config.clone(),
        crate::telemetry::TelemetryLevel::Summary,
    );

    let result = write_downstream_or_emit_forward_error(&telemetry, &config, &ids, 1234, || Ok(()));

    assert!(result.is_ok());
    assert!(rx.try_recv().is_err(), "success must not emit");
}

/// The live embedded A→B route's lane checkout is a production connect
/// boundary: a refused downstream connection emits exactly one
/// `stage.binary_downstream_connect_error` carrying the request identity and
/// the intended downstream, and the checkout still fails with an OpenAI
/// backend error.
#[test]
fn lane_checkout_connect_failure_emits_connect_event() {
    // Controlled refusal: hold a kernel-assigned loopback port bound
    // WITHOUT listening for the whole test. The live binding keeps the
    // port reserved across connect_binary_downstream's retry window (no
    // other process can take it), and a bound-but-not-listening socket
    // refuses every connection by construction.
    let refusal_port = socket2::Socket::new(
        socket2::Domain::IPV4,
        socket2::Type::STREAM,
        Some(socket2::Protocol::TCP),
    )
    .unwrap();
    refusal_port
        .bind(&socket2::SockAddr::from(
            "127.0.0.1:0".parse::<std::net::SocketAddr>().unwrap(),
        ))
        .unwrap();
    let mut config = identity_config();
    config.downstream = Some(skippy_protocol::PeerConfig {
        persistent_lane_marker: false,
        stage_id: "stage-2".to_string(),
        stage_index: 2,
        endpoint: refusal_port
            .local_addr()
            .unwrap()
            .as_socket()
            .unwrap()
            .to_string(),
    });
    let expected = uuid::Uuid::new_v4();
    let ids = InferenceGenerationIds::new_with_trust(
        InferenceCacheHints::default(),
        None,
        false,
        Some(*expected.as_bytes()),
    );
    let (telemetry, rx) = crate::telemetry::Telemetry::captured(
        config.clone(),
        crate::telemetry::TelemetryLevel::Summary,
    );

    let pool = crate::frontend::generation::PersistentStageLanePool {
        config: config.clone(),
        timeout_secs: 1,
        telemetry,
        lanes: std::sync::Mutex::new(Vec::new()),
        prefill_transport: std::sync::Mutex::new(
            crate::frontend::generation::PrefillTransportEstimate::default(),
        ),
        next_lane_id: std::sync::atomic::AtomicU64::new(0),
        capacity: 1,
    };

    let result = pool.checkout(&ids);

    // PersistentStageLane has no Debug impl; extract the error without
    // widening production traits for this test.
    let error = match result {
        Ok(lane) => panic!(
            "checkout must fail against the refused endpoint, got lane {}",
            lane.id
        ),
        Err(error) => error,
    };
    assert!(
        error.to_string().to_lowercase().contains("connect")
            || error.to_string().to_lowercase().contains("refused")
            || error.to_string().to_lowercase().contains("handshake")
            || error.to_string().to_lowercase().contains("downstream"),
        "checkout surfaces an OpenAI backend error: {error}"
    );

    let event = rx
        .try_recv()
        .expect("exactly one connect event on checkout failure");
    assert_eq!(event.event, "stage.binary_downstream_connect_error");
    assert_eq!(
        event.attributes["skippy.request_id"],
        json!(ids.request_id_string())
    );
    assert_eq!(
        event.attributes["skippy.session_id"],
        json!(ids.session_id_string())
    );
    let parsed = uuid::Uuid::parse_str(
        event.attributes["openai.frontend_request_id"]
            .as_str()
            .unwrap(),
    )
    .expect("frontend uuid present on the connect failure event");
    assert_eq!(parsed, expected);
    assert_eq!(event.attributes["skippy.run_id"], json!("run"));
    assert_eq!(event.attributes["skippy.topology_id"], json!("topology"));
    assert_eq!(
        event.attributes["llama_stage.downstream_stage_id"],
        json!("stage-2")
    );
    assert_eq!(
        event.attributes["llama_stage.downstream_stage_index"],
        json!(2)
    );
    assert_eq!(
        event.attributes["llama_stage.downstream_endpoint"],
        json!(config.downstream.as_ref().unwrap().endpoint.as_str())
    );
    assert!(
        !event.attributes["llama_stage.error"]
            .as_str()
            .unwrap()
            .is_empty()
    );
    assert!(rx.try_recv().is_err(), "no additional events on failure");
}
