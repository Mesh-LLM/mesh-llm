mod completion_exchange;
mod exchange_hooks;
mod inference;
mod prepared_exchange;
#[cfg(test)]
mod prepared_exchange_tests;
use crate::frontend::admission::GenerationTokenBudget;
use crate::frontend::admission::GenerationTokenBudgetRequest;
use crate::frontend::admission::GenerationTokenReservation;
use crate::frontend::generation::ChatOutputStreamParser;
use crate::frontend::generation::GeneratedText;
use crate::frontend::generation::GenerationActiveWorkReservation;
use crate::frontend::generation::GenerationAdmissionClaim;
use crate::frontend::generation::GenerationAdmissionScheduling;
use crate::frontend::generation::GenerationAdmissionWork;
use crate::frontend::generation::GenerationConcurrencyController;
use crate::frontend::generation::GenerationConcurrencyDecision;
use crate::frontend::generation::GenerationConcurrencyObservation;
use crate::frontend::generation::GenerationConcurrencyPermit;
use crate::frontend::generation::GenerationServiceEstimator;
use crate::frontend::generation::GenerationSessionLockEntry;
use crate::frontend::generation::GenerationStream;
use crate::frontend::generation::GenerationStreamEvent;
use crate::frontend::generation::GenerationTokenLimit;
use crate::frontend::generation::InferenceBackendMode;
use crate::frontend::generation::InferenceCacheHints;
use crate::frontend::generation::InferenceGenerationIds;
use crate::frontend::generation::PhaseTimer;
use crate::frontend::generation::PreparedGenerationPrompt;
use crate::frontend::generation::PreparedTextPrompt;
use crate::frontend::generation::StageOpenAiBackend;
use crate::frontend::generation::apply_reasoning_visibility;
use crate::frontend::generation::chat_output_parser_required;
use crate::frontend::generation::completion_response_from_generated_text;
use crate::frontend::generation::ensure_requested_model;
use crate::frontend::generation::generation_event_to_completion_chunk;
use crate::frontend::generation::generation_event_to_inference_event;
use crate::frontend::generation::generation_predicted_wait_error;
use crate::frontend::generation::generation_queue_timeout_error;
use crate::frontend::generation::generation_response_from_generated_text;
use crate::frontend::generation::template_exposes_reasoning;
use crate::frontend::request::{
    apply_chat_request_defaults, apply_completion_request_defaults, chat_sampling_config,
    chat_template_options, completion_sampling_config, ensure_chat_runtime_features_supported,
    ensure_completion_runtime_features_supported, resolve_chat_request_defaults,
    resolve_completion_request_defaults, sampling_config,
};
use crate::runtime_state::RuntimeSessionStats;
use crate::runtime_state::panic_recovery::lock_runtime;
use crate::telemetry::Telemetry;
use crate::telemetry::lifecycle_attrs;
use crate::telemetry::now_unix_nanos;
use async_trait::async_trait;
use futures_util::StreamExt;
use futures_util::stream;
use prepared_exchange::PreparedExchangeAdmission;
use serde_json::Value;
use serde_json::json;
use skippy_inference_api::AudioFormat;
use skippy_inference_api::AudioResponse;
use skippy_inference_api::AudioSpeechRequest;
use skippy_inference_api::AudioTranscriptionRequest;
use skippy_inference_api::AudioTranscriptionResponse;
use skippy_inference_api::ChatCompletionOutcome;
use skippy_inference_api::ChatExchangeRoute;
use skippy_inference_api::CompletionRequest;
use skippy_inference_api::CompletionResponse;
use skippy_inference_api::CompletionStream;
use skippy_inference_api::Embedding;
use skippy_inference_api::EmbeddingInput;
use skippy_inference_api::EmbeddingResponse;
use skippy_inference_api::EmbeddingsRequest;
use skippy_inference_api::InferenceBackend;
use skippy_inference_api::InferenceError;
use skippy_inference_api::InferenceRequestContext;
use skippy_inference_api::InferenceResult;
use skippy_inference_api::ModelObject;
use skippy_inference_api::RerankRequest;
use skippy_inference_api::RerankResponse;
use skippy_inference_api::RerankResult;
use skippy_inference_api::SystemOneRequest;
use skippy_inference_api::SystemOneResponse;
use skippy_inference_api::TerminalGuard;
use skippy_inference_api::TerminalGuardedChatStream;
use skippy_inference_api::apply_chat_hook_outcome;
use skippy_inference_api::capsule_id_is_valid;
use skippy_inference_api::chat_skippy_hooks_enabled;
use skippy_inference_api::generation::GenerationEventStream;
use skippy_inference_api::generation::GenerationRequest;
use skippy_inference_api::generation::GenerationResponse;
use skippy_metrics::attr as attr_key;
use skippy_protocol::StageConfig;
use skippy_runtime::{
    MediaInput, ModelWorkload, SamplingConfig, SpeechOutputFormat, SpeechSynthesisConfig,
    WorkloadInfo,
};
use std::collections::BTreeMap;
use std::io;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Duration;
use std::time::Instant;
use tokio::sync::OwnedSemaphorePermit;
use tokio::sync::Semaphore;
use tokio::sync::TryAcquireError;
use tokio::sync::mpsc;
use tokio::sync::mpsc::error::TrySendError;
use tokio::task;

mod non_chat;

fn request_cancelled_error() -> InferenceError {
    InferenceError::cancelled("request cancelled")
}

/// How long a full SSE event channel is treated as merely backed up before
/// the generation worker gives up on it and frees its execution lane.
///
/// Deliberately its own value rather than an alias of
/// the configured generation admission timeout: admission queueing and stream-stall
/// tolerance are unrelated policies, and retuning one must not silently
/// retune the other. It bounds a single send, not a whole generation.
const STREAM_SEND_STALL_TIMEOUT: Duration = Duration::from_secs(10);

/// Forwards generation events to the SSE channel without letting a consumer
/// that has stopped draining pin the generation worker, and the execution
/// lane it holds, forever.
///
/// `mpsc::Sender::blocking_send` waits indefinitely for buffer space. A
/// consumer that stops draining without the channel ever being dropped --
/// a client gone before the server's own disconnect detection notices, e.g.
/// behind a proxy that doesn't propagate the close -- can then pin the
/// generation worker, and the execution lane it holds, forever even after
/// the request is cancelled. This races each send against cancellation and
/// against `stall_timeout` via `tokio::select!` instead of polling, so a
/// consumer draining normally is woken the instant space appears rather than
/// after up to a 20 ms poll tick.
struct StreamEventSender {
    tx: mpsc::Sender<InferenceResult<GenerationStreamEvent>>,
    runtime: tokio::runtime::Handle,
    stall_timeout: Duration,
    /// Request identifier carried for diagnostics so a stalled or dropped
    /// consumer can be attributed to the exact request that held (and then
    /// freed) an execution lane. `send_terminal` has no request context of
    /// its own, so the id is captured once at construction.
    request_id: String,
    /// Set once the receiver is gone or has stayed full past the stall
    /// timeout. Nothing further can reach the client, so later frames are
    /// dropped rather than waited on again.
    receiver_unreachable: AtomicBool,
    /// Structured telemetry sink for the stall/drop diagnostics. `skippy-serving`
    /// routes operator-facing signal through `Telemetry`, not a logging facade,
    /// so a freed execution lane is correlated by `request_id` and observable
    /// without scraping stderr.
    telemetry: Telemetry,
}

impl StreamEventSender {
    fn new(
        tx: mpsc::Sender<InferenceResult<GenerationStreamEvent>>,
        runtime: tokio::runtime::Handle,
        stall_timeout: Duration,
        request_id: String,
        telemetry: Telemetry,
    ) -> Self {
        Self {
            tx,
            runtime,
            stall_timeout,
            request_id,
            receiver_unreachable: AtomicBool::new(false),
            telemetry,
        }
    }

    /// Emit a structured "execution lane freed" event when a consumer is found
    /// gone or stalled. `outcome` names the failure (dropped vs stalled) and
    /// `frame_kind` names which send path hit it (an in-flight event vs a
    /// terminal frame), so an operator can tell a client cancellation apart
    /// from a stalled consumer pinning a lane without log scraping.
    fn emit_lane_freed(&self, outcome: &str, frame_kind: &str) {
        let mut attrs = BTreeMap::new();
        attrs.insert(attr_key::REQUEST_ID.to_string(), json!(self.request_id));
        attrs.insert("skippy.stream.outcome".to_string(), json!(outcome));
        attrs.insert("skippy.stream.frame_kind".to_string(), json!(frame_kind));
        attrs.insert(
            "skippy.stream.stall_timeout_ms".to_string(),
            json!(self.stall_timeout.as_millis() as u64),
        );
        self.telemetry.emit("stage.openai_stream_lane_freed", attrs);
    }

    /// Mark the receiver unreachable and free the request's execution lane.
    /// Called once nothing further sent on this channel could possibly reach
    /// the client: the receiver dropped, or it stayed full past
    /// `stall_timeout`.
    fn mark_receiver_unreachable(&self, context: &InferenceRequestContext) {
        self.receiver_unreachable.store(true, Ordering::Release);
        context.cancel();
    }

    /// Send one in-flight event (from the `on_text_chunk` callback). Checks
    /// cancellation first so an already-cancelled request returns
    /// immediately and deterministically, matching the pre-existing
    /// early-return semantics.
    fn send(
        &self,
        event: InferenceResult<GenerationStreamEvent>,
        context: &InferenceRequestContext,
    ) -> Result<(), InferenceError> {
        if self.receiver_unreachable.load(Ordering::Acquire) {
            return Err(InferenceError::backend("stream receiver unreachable"));
        }
        if context.is_cancelled() {
            return Err(request_cancelled_error());
        }
        let event = match self.tx.try_send(event) {
            Ok(()) => return Ok(()),
            Err(TrySendError::Full(event)) => event,
            Err(TrySendError::Closed(_)) => {
                self.emit_lane_freed("receiver_dropped", "in_flight");
                self.mark_receiver_unreachable(context);
                return Err(InferenceError::backend("stream receiver dropped"));
            }
        };
        let cancellation = context.cancellation_token();
        // `tokio::time::sleep` needs an entered runtime the instant it is
        // called, not just when polled, so it must be constructed inside the
        // `block_on`-driven future rather than before it.
        self.runtime.block_on(async {
            let send = self.tx.send(event);
            let sleep = tokio::time::sleep(self.stall_timeout);
            tokio::select! {
                biased;
                () = cancellation.cancelled() => Err(request_cancelled_error()),
                result = send => match result {
                    Ok(()) => Ok(()),
                    Err(_) => {
                        self.emit_lane_freed("receiver_dropped", "in_flight");
                        self.mark_receiver_unreachable(context);
                        Err(InferenceError::backend("stream receiver dropped"))
                    }
                },
                () = sleep => {
                    self.emit_lane_freed("receiver_stalled", "in_flight");
                    self.mark_receiver_unreachable(context);
                    Err(InferenceError::backend(
                        "stream receiver stalled without draining",
                    ))
                }
            }
        })
    }

    /// Send one terminal frame -- everything emitted after generation
    /// finishes: the cancellation error, a parser-finish error, the backend
    /// error, usage, and `Done`.
    ///
    /// Deliberately does **not** consult cancellation. These are exactly the
    /// frames the frontend lifecycle needs when the request *was* cancelled:
    /// on `main`, `blocking_send` delivered them unconditionally. Skipping
    /// them for a cancelled-but-still-live receiver would flip
    /// `stream_lifecycle`'s terminal classification -- an `Err` frame drives
    /// `lifecycle.failed(error)`, which marks `backend_error` and yields
    /// `StreamDropOutcome::BackendError`/`StreamTerminal`; without it,
    /// `drop_outcome()` falls through to `StreamDropOutcome::Cancelled`
    /// instead.
    ///
    /// It does still refuse to wait on a receiver already proven
    /// unreachable: the in-flight send that got us here may already have
    /// waited out `stall_timeout` once, and waiting again would double the
    /// execution lane's hold to `2 * stall_timeout`.
    fn send_terminal(
        &self,
        event: InferenceResult<GenerationStreamEvent>,
    ) -> Result<(), InferenceError> {
        if self.receiver_unreachable.load(Ordering::Acquire) {
            return Err(InferenceError::backend("stream receiver unreachable"));
        }
        // See the matching comment in `send`: the sleep future must be
        // constructed inside the entered runtime `block_on` provides.
        self.runtime.block_on(async {
            let send = self.tx.send(event);
            let sleep = tokio::time::sleep(self.stall_timeout);
            tokio::select! {
                result = send => match result {
                    Ok(()) => Ok(()),
                    Err(_) => {
                        self.emit_lane_freed("receiver_dropped", "terminal");
                        self.receiver_unreachable.store(true, Ordering::Release);
                        Err(InferenceError::backend("stream receiver dropped"))
                    }
                },
                () = sleep => {
                    self.emit_lane_freed("receiver_stalled", "terminal");
                    self.receiver_unreachable.store(true, Ordering::Release);
                    Err(InferenceError::backend(
                        "stream receiver stalled without draining",
                    ))
                }
            }
        })
    }
}

fn should_emit_stream_usage(
    request_include_usage: bool,
    context: &InferenceRequestContext,
) -> bool {
    request_include_usage || context.observes_stream_usage()
}

pub(in crate::frontend) struct GenerationSessionPermit {
    registry: Arc<Mutex<BTreeMap<String, Arc<GenerationSessionLockEntry>>>>,
    key: String,
    entry: Arc<GenerationSessionLockEntry>,
    permit: Option<OwnedSemaphorePermit>,
}

impl GenerationSessionPermit {
    fn new(
        registry: Arc<Mutex<BTreeMap<String, Arc<GenerationSessionLockEntry>>>>,
        key: String,
    ) -> InferenceResult<Self> {
        let entry = {
            let mut locks = registry
                .lock()
                .map_err(|_| InferenceError::backend("generation session lock map poisoned"))?;
            let entry = locks
                .entry(key.clone())
                .or_insert_with(|| {
                    Arc::new(GenerationSessionLockEntry {
                        semaphore: Arc::new(Semaphore::new(1)),
                        users: AtomicUsize::new(0),
                    })
                })
                .clone();
            // Lookup and lease registration share the registry mutex with
            // cleanup, so a dropping lease cannot remove and replace this
            // entry between those two operations.
            entry.users.fetch_add(1, Ordering::AcqRel);
            entry
        };
        Ok(Self {
            registry,
            key,
            entry,
            permit: None,
        })
    }

    fn try_acquire(&mut self) -> InferenceResult<bool> {
        match self.entry.semaphore.clone().try_acquire_owned() {
            Ok(permit) => {
                self.permit = Some(permit);
                Ok(true)
            }
            Err(TryAcquireError::NoPermits) => Ok(false),
            Err(TryAcquireError::Closed) => {
                Err(InferenceError::backend("generation session lock closed"))
            }
        }
    }

    async fn acquire_until(
        mut self,
        deadline: Option<Instant>,
        admission_timeout: Duration,
        cancellation: &skippy_inference_api::CancellationToken,
    ) -> InferenceResult<Self> {
        let permit = if let Some(deadline) = deadline {
            let acquire = tokio::time::timeout_at(
                tokio::time::Instant::from_std(deadline),
                self.entry.semaphore.clone().acquire_owned(),
            );
            tokio::select! {
                result = acquire => result
                    .map_err(|_| generation_queue_timeout_error(admission_timeout))?
                    .map_err(|_| InferenceError::backend("generation session lock closed"))?,
                () = cancellation.cancelled() => return Err(request_cancelled_error()),
            }
        } else {
            tokio::select! {
                result = self.entry.semaphore.clone().acquire_owned() => result
                    .map_err(|_| InferenceError::backend("generation session lock closed"))?,
                () = cancellation.cancelled() => return Err(request_cancelled_error()),
            }
        };
        if cancellation.is_cancelled() {
            return Err(request_cancelled_error());
        }
        self.permit = Some(permit);
        Ok(self)
    }
}

impl Drop for GenerationSessionPermit {
    fn drop(&mut self) {
        self.permit.take();
        let Ok(mut locks) = self.registry.lock() else {
            return;
        };
        if self.entry.users.fetch_sub(1, Ordering::AcqRel) == 1
            && locks
                .get(&self.key)
                .is_some_and(|entry| Arc::ptr_eq(entry, &self.entry))
        {
            locks.remove(&self.key);
        }
    }
}

fn trusted_generation_session_key(ids: &InferenceGenerationIds) -> Option<String> {
    ids.agent_session_trusted.then(|| ids.session_id_string())
}

#[derive(Clone)]
struct GenerationAdmissionController {
    generation_limit: Arc<GenerationConcurrencyController>,
    generation_queue_depth: Arc<AtomicUsize>,
    generation_queue_limit: usize,
    generation_service_estimator: Arc<GenerationServiceEstimator>,
    generation_session_locks: Arc<Mutex<BTreeMap<String, Arc<GenerationSessionLockEntry>>>>,
    generation_token_budget: Arc<GenerationTokenBudget>,
}

impl GenerationAdmissionController {
    fn for_backend(backend: &StageOpenAiBackend) -> Self {
        Self {
            generation_limit: backend.generation_limit.clone(),
            generation_queue_depth: backend.generation_queue_depth.clone(),
            generation_queue_limit: backend.generation_queue_limit,
            generation_service_estimator: backend.generation_service_estimator.clone(),
            generation_session_locks: backend.generation_session_locks.clone(),
            generation_token_budget: backend.generation_token_budget.clone(),
        }
    }

    #[cfg(test)]
    async fn acquire(
        &self,
        ids: &InferenceGenerationIds,
        cancellation: &skippy_inference_api::CancellationToken,
        admission_timeout: Duration,
    ) -> InferenceResult<(GenerationAdmissionPermit, Option<GenerationSessionPermit>)> {
        self.acquire_work(
            ids,
            cancellation,
            admission_timeout,
            GenerationAdmissionWork::default(),
        )
        .await
    }

    #[cfg(test)]
    async fn acquire_work(
        &self,
        ids: &InferenceGenerationIds,
        cancellation: &skippy_inference_api::CancellationToken,
        admission_timeout: Duration,
        work: GenerationAdmissionWork,
    ) -> InferenceResult<(GenerationAdmissionPermit, Option<GenerationSessionPermit>)> {
        self.acquire_scheduled_work(
            ids,
            cancellation,
            admission_timeout,
            work,
            GenerationAdmissionScheduling::default(),
        )
        .await
    }

    async fn acquire_scheduled_work(
        &self,
        ids: &InferenceGenerationIds,
        cancellation: &skippy_inference_api::CancellationToken,
        admission_timeout: Duration,
        work: GenerationAdmissionWork,
        scheduling: GenerationAdmissionScheduling,
    ) -> InferenceResult<(GenerationAdmissionPermit, Option<GenerationSessionPermit>)> {
        let deadline = if admission_timeout.is_zero() {
            None
        } else {
            Some(
                Instant::now()
                    .checked_add(admission_timeout)
                    .ok_or_else(|| {
                        InferenceError::backend("generation admission deadline overflow")
                    })?,
            )
        };
        let session_permit = self
            .acquire_session_until(ids, deadline, admission_timeout, cancellation)
            .await?;

        if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
            return Err(generation_queue_timeout_error(admission_timeout));
        }
        let generation_permit = self
            .acquire_generation_permit_until(
                deadline,
                admission_timeout,
                cancellation,
                work,
                scheduling,
            )
            .await?;
        if cancellation.is_cancelled() {
            return Err(request_cancelled_error());
        }
        Ok((generation_permit, session_permit))
    }

    async fn acquire_session_until(
        &self,
        ids: &InferenceGenerationIds,
        deadline: Option<Instant>,
        admission_timeout: Duration,
        cancellation: &skippy_inference_api::CancellationToken,
    ) -> InferenceResult<Option<GenerationSessionPermit>> {
        let Some(session_key) = trusted_generation_session_key(ids) else {
            return Ok(None);
        };
        if cancellation.is_cancelled() {
            return Err(request_cancelled_error());
        }
        let mut session =
            GenerationSessionPermit::new(self.generation_session_locks.clone(), session_key)?;
        if session.try_acquire()? {
            return Ok(Some(session));
        }
        session
            .acquire_until(deadline, admission_timeout, cancellation)
            .await
            .map(Some)
    }

    async fn acquire_generation_permit_until(
        &self,
        deadline: Option<Instant>,
        admission_timeout: Duration,
        cancellation: &skippy_inference_api::CancellationToken,
        work: GenerationAdmissionWork,
        scheduling: GenerationAdmissionScheduling,
    ) -> InferenceResult<GenerationAdmissionPermit> {
        if cancellation.is_cancelled() {
            return Err(request_cancelled_error());
        }
        let claim = self.generation_limit.admission_queue().claim_or_enqueue(
            self.generation_limit.semaphore(),
            self.generation_token_budget.clone(),
            GenerationTokenBudgetRequest::new(
                usize::try_from(work.prompt_tokens).unwrap_or(usize::MAX),
                u32::try_from(work.decode_tokens).unwrap_or(u32::MAX),
            ),
            scheduling,
            self.generation_queue_depth.clone(),
            self.generation_queue_limit,
        )?;
        match claim {
            GenerationAdmissionClaim::Acquired(resources) => Ok(GenerationAdmissionPermit {
                _lane: self.generation_limit.wrap_permit(resources.lane),
                token_budget: resources.token_budget,
                _active_work: self.generation_service_estimator.start_active(work),
                predicted_wait_ms: Some(0.0),
                demand_epoch: self.generation_limit.demand_epoch(),
                queued_at_start: self.generation_queue_depth.load(Ordering::Acquire) > 0,
                waited_for_lane: false,
                at_capacity_at_start: self.generation_limit.is_at_capacity(),
                started_at: Instant::now(),
            }),
            GenerationAdmissionClaim::Queued(lease) => {
                self.generation_limit.note_queued_demand();
                self.generation_service_estimator
                    .set_concurrency(self.generation_limit.current_limit());
                let predicted_wait_ms = self.generation_service_estimator.predicted_wait_ms();
                let queued_work = self
                    .generation_service_estimator
                    .reserve_queued(work, admission_timeout)
                    .map_err(|predicted_wait_ms| {
                        generation_predicted_wait_error(predicted_wait_ms, admission_timeout)
                    })?;
                let resources = lease
                    .acquire(
                        self.generation_limit.semaphore(),
                        admission_timeout,
                        deadline,
                        cancellation,
                    )
                    .await?;
                Ok(GenerationAdmissionPermit {
                    _lane: self.generation_limit.wrap_permit(resources.lane),
                    token_budget: resources.token_budget,
                    _active_work: queued_work.promote(),
                    predicted_wait_ms,
                    demand_epoch: self.generation_limit.demand_epoch(),
                    queued_at_start: self.generation_queue_depth.load(Ordering::Acquire) > 0,
                    waited_for_lane: true,
                    at_capacity_at_start: self.generation_limit.is_at_capacity(),
                    started_at: Instant::now(),
                })
            }
        }
    }
}

pub(in crate::frontend) struct GenerationAdmissionPermit {
    token_budget: GenerationTokenReservation,
    _lane: GenerationConcurrencyPermit,
    _active_work: GenerationActiveWorkReservation,
    predicted_wait_ms: Option<f64>,
    demand_epoch: u64,
    queued_at_start: bool,
    waited_for_lane: bool,
    at_capacity_at_start: bool,
    started_at: Instant,
}

impl GenerationAdmissionPermit {
    fn token_budget_stats(&self) -> (usize, usize) {
        (
            self.token_budget.tokens(),
            self.token_budget.active_tokens_after_reservation(),
        )
    }

    fn demand_observation(&self) -> GenerationDemandObservation {
        GenerationDemandObservation {
            demand_epoch: self.demand_epoch,
            queued_at_start: self.queued_at_start,
            waited_for_lane: self.waited_for_lane,
            at_capacity_at_start: self.at_capacity_at_start,
            started_at: self.started_at,
        }
    }
}

#[derive(Clone, Copy)]
struct GenerationDemandObservation {
    demand_epoch: u64,
    queued_at_start: bool,
    waited_for_lane: bool,
    at_capacity_at_start: bool,
    started_at: Instant,
}

impl StageOpenAiBackend {
    fn observe_generation_completed(
        &self,
        output: &GeneratedText,
        demand: GenerationDemandObservation,
    ) {
        let work = GenerationAdmissionWork::new(
            usize::try_from(output.prompt_tokens).unwrap_or(usize::MAX),
            output.completion_tokens,
        );
        self.generation_service_estimator.observe_completed(
            work,
            output.prompt_ms,
            output.predicted_ms,
        );
        let completed_tokens = work.prompt_tokens.saturating_add(work.decode_tokens);
        let executed_tokens =
            u64::from(output.suffix_prefill_tokens).saturating_add(work.decode_tokens);
        let saturated = demand.waited_for_lane
            || self.generation_limit.was_saturated_since(
                demand.demand_epoch,
                demand.queued_at_start,
                self.generation_queue_depth.load(Ordering::Acquire) > 0,
            );
        let Some(decision) =
            self.generation_limit
                .observe_completed(GenerationConcurrencyObservation {
                    completed_tokens,
                    executed_tokens,
                    latency_ms: output.prompt_ms + output.predicted_ms,
                    saturated,
                    at_capacity: demand.at_capacity_at_start,
                    started_at: demand.started_at,
                })
        else {
            return;
        };
        self.emit_generation_concurrency_decision(decision);
    }

    fn observe_generation_failed(&self) {
        if let Some(decision) = self.generation_limit.observe_failed() {
            self.emit_generation_concurrency_decision(decision);
        }
    }

    fn emit_generation_concurrency_decision(&self, decision: GenerationConcurrencyDecision) {
        self.generation_service_estimator
            .set_concurrency(decision.current_limit);
        let mut attrs = lifecycle_attrs(&self.config);
        attrs.insert(
            "llama_stage.generation_concurrency_action".to_string(),
            json!(decision.action),
        );
        attrs.insert(
            "llama_stage.generation_concurrency_reason".to_string(),
            json!(decision.reason),
        );
        attrs.insert(
            "llama_stage.generation_concurrency_previous".to_string(),
            json!(decision.previous_limit),
        );
        attrs.insert(
            "llama_stage.generation_concurrency_current".to_string(),
            json!(decision.current_limit),
        );
        attrs.insert(
            "llama_stage.generation_concurrency_ceiling".to_string(),
            json!(self.generation_limit.hard_limit()),
        );
        if let Some(throughput) = decision.throughput_tokens_per_second {
            attrs.insert(
                "llama_stage.generation_throughput_tokens_per_second".to_string(),
                json!(throughput),
            );
        }
        if let Some(p95_latency_ms) = decision.p95_latency_ms {
            attrs.insert(
                "llama_stage.generation_p95_latency_ms".to_string(),
                json!(p95_latency_ms),
            );
        }
        if let Some(p95_hardware_ms_per_token) = decision.p95_hardware_ms_per_token {
            attrs.insert(
                "llama_stage.generation_p95_hardware_ms_per_token".to_string(),
                json!(p95_hardware_ms_per_token),
            );
        }
        if let Some(improvement) = decision.throughput_improvement {
            attrs.insert(
                "llama_stage.generation_throughput_improvement".to_string(),
                json!(improvement),
            );
        }
        if let Some(ratio) = decision.p95_latency_ratio {
            attrs.insert(
                "llama_stage.generation_p95_latency_ratio".to_string(),
                json!(ratio),
            );
        }
        if let Some(ratio) = decision.hardware_pressure_ratio {
            attrs.insert(
                "llama_stage.generation_hardware_pressure_ratio".to_string(),
                json!(ratio),
            );
        }
        if let Some(observed_requests) = decision.observed_requests {
            attrs.insert(
                "llama_stage.generation_observed_requests".to_string(),
                json!(observed_requests),
            );
        }
        if let Some(saturated_requests) = decision.saturated_requests {
            attrs.insert(
                "llama_stage.generation_saturated_requests".to_string(),
                json!(saturated_requests),
            );
        }
        self.telemetry
            .emit("stage.openai_generation_concurrency_adapt", attrs);
    }
}

fn insert_generation_admission_attrs(
    attrs: &mut BTreeMap<String, Value>,
    permit: &GenerationAdmissionPermit,
    queue_depth: usize,
    queue_capacity: usize,
) {
    attrs.insert(
        "llama_stage.generation_queue_depth".to_string(),
        json!(queue_depth),
    );
    attrs.insert(
        "skippy.scheduler.admission_waiting".to_string(),
        json!(queue_depth),
    );
    attrs.insert(
        "llama_stage.generation_queue_capacity".to_string(),
        json!(queue_capacity),
    );
    if let Some(predicted_wait_ms) = permit.predicted_wait_ms {
        attrs.insert(
            "llama_stage.generation_predicted_wait_ms".to_string(),
            json!(predicted_wait_ms),
        );
    }
}

/// Shared OpenAI request-identity attribute set: lifecycle identity, the
/// native numeric session/request ids, and the frontend request UUID when the
/// generation entered through the OpenAI HTTP boundary.
///
/// Emitted by `stage.openai_request_summary` (success path, normal level) and
/// by the debug-level `stage.openai_generation_admit` phase, which is the
/// pre-failure event carrying this identity: on failed requests the success
/// summary is absent, so a normal-level UUID join before failure does not
/// exist without debug telemetry.
fn openai_identity_attrs(
    config: &StageConfig,
    backend_label: &str,
    ids: &InferenceGenerationIds,
) -> BTreeMap<String, Value> {
    let mut attrs = lifecycle_attrs(config);
    attrs.insert(
        attr_key::SESSION_ID.to_string(),
        json!(ids.session_id_string()),
    );
    attrs.insert(
        attr_key::REQUEST_ID.to_string(),
        json!(ids.request_id_string()),
    );
    attrs.insert(
        "llama_stage.openai_backend".to_string(),
        json!(backend_label),
    );
    if let Some(frid) = ids.frontend_request_id {
        attrs.insert(
            "openai.frontend_request_id".to_string(),
            json!(uuid::Uuid::from_bytes(frid).to_string()),
        );
    }
    if let Some(cache_key) = ids.cache.prompt_cache_key.as_deref() {
        attrs.insert("openai.prompt_cache_key".to_string(), json!(cache_key));
    }
    if let Some(retention) = ids.cache.prompt_cache_retention.as_deref() {
        attrs.insert(
            "openai.prompt_cache_retention".to_string(),
            json!(retention),
        );
    }
    attrs
}

/// Embedded-route flavor of the downstream failure contract: the OpenAI
/// request identity (native numeric ids + frontend UUID) instead of binary
/// wire ids, plus the same intended-downstream identity and error chain.
pub(super) fn openai_downstream_error_attrs(
    config: &StageConfig,
    ids: &InferenceGenerationIds,
    error: &str,
) -> BTreeMap<String, Value> {
    let mut attrs =
        openai_identity_attrs(config, InferenceBackendMode::EMBEDDED_STAGE_ZERO_LABEL, ids);
    if let Some(downstream) = &config.downstream {
        crate::binary_transport::binary_messaging::insert_downstream_identity(
            &mut attrs, downstream,
        );
    }
    attrs.insert("llama_stage.error".to_string(), json!(error));
    attrs
}

/// Embedded sync-forward boundary: run `write`, and on failure emit
/// `stage.binary_downstream_forward_error` exactly once with the OpenAI
/// request identity and the intended downstream identity before propagating
/// the original `io::Error` unchanged. Reply reads are NOT routed through
/// this boundary — they are a separate failure class, never labeled as
/// connect or forward errors.
pub(super) fn write_downstream_or_emit_forward_error(
    telemetry: &Telemetry,
    config: &StageConfig,
    ids: &InferenceGenerationIds,
    write_start_unix_nanos: u64,
    write: impl FnOnce() -> io::Result<()>,
) -> io::Result<()> {
    write().inspect_err(|error| {
        telemetry.emit_span(
            crate::binary_transport::binary_messaging::STAGE_BINARY_DOWNSTREAM_FORWARD_ERROR,
            openai_downstream_error_attrs(config, ids, &error.to_string()),
            write_start_unix_nanos,
            now_unix_nanos() as u64,
        );
    })
}

fn generation_ids(
    cache: InferenceCacheHints,
    agent_session_id: Option<&str>,
    context: &InferenceRequestContext,
) -> InferenceGenerationIds {
    let frontend_request_id = context
        .request_id()
        .map(|request_id| request_id.as_uuid().into_bytes());
    InferenceGenerationIds::new_with_trust(
        cache,
        agent_session_id,
        context.has_trusted_agent_session(),
        frontend_request_id,
    )
}

pub(in crate::frontend) async fn run_blocking_generation_worker<T, F, P>(
    permit: P,
    context: InferenceRequestContext,
    work: F,
) -> Result<T, task::JoinError>
where
    T: Send + 'static,
    F: FnOnce(skippy_inference_api::CancellationToken) -> T + Send + 'static,
    P: Send + 'static,
{
    task::spawn_blocking(move || {
        let _permit = permit;
        work(context.cancellation_token())
    })
    .await
}

impl StageOpenAiBackend {
    /// Admit generation work subject to concurrency, queue, token-budget and cancellation gates.
    pub(in crate::frontend) async fn acquire_generation_admission(
        &self,
        ids: &InferenceGenerationIds,
        cancellation: &skippy_inference_api::CancellationToken,
        work: GenerationAdmissionWork,
        scheduling: GenerationAdmissionScheduling,
    ) -> InferenceResult<(GenerationAdmissionPermit, Option<GenerationSessionPermit>)> {
        let result = GenerationAdmissionController::for_backend(self)
            .acquire_scheduled_work(
                ids,
                cancellation,
                self.generation_admission_timeout,
                work,
                scheduling,
            )
            .await;
        if let Err(error) = &result {
            let mut attrs = self.openai_attrs(ids);
            attrs.insert(
                "llama_stage.generation_queue_depth".to_string(),
                json!(self.generation_queue_depth.load(Ordering::Acquire)),
            );
            attrs.insert(
                "llama_stage.generation_queue_capacity".to_string(),
                json!(self.generation_queue_limit),
            );
            attrs.insert(
                "llama_stage.generation_admission_status".to_string(),
                json!("rejected"),
            );
            attrs.insert(
                "llama_stage.generation_admission_error".to_string(),
                json!(error.body().error.message),
            );
            if let Some(predicted_wait_ms) = self.generation_service_estimator.predicted_wait_ms() {
                attrs.insert(
                    "llama_stage.generation_predicted_wait_ms".to_string(),
                    json!(predicted_wait_ms),
                );
            }
            self.telemetry
                .emit("stage.openai_generation_admission_rejected", attrs);
        }
        result
    }

    fn generation_admission_scheduling(
        &self,
        prepared_text: Option<&PreparedTextPrompt>,
        ids: &InferenceGenerationIds,
    ) -> GenerationAdmissionScheduling {
        let Some(prepared) = prepared_text else {
            return GenerationAdmissionScheduling::default();
        };
        let prompt_tokens = Arc::<[i32]>::from(prepared.token_ids.clone());
        let Some(kv) = self.kv.as_ref() else {
            return GenerationAdmissionScheduling::new(
                prompt_tokens,
                Arc::new(skippy_scheduler::CacheAffinity::default),
            );
        };
        let prefill_tokens = prepared
            .token_ids
            .get(..prepared.token_ids.len().saturating_sub(1))
            .unwrap_or_default();
        if prefill_tokens.is_empty() {
            return GenerationAdmissionScheduling::new(
                prompt_tokens,
                Arc::new(skippy_scheduler::CacheAffinity::default),
            );
        }
        let base = self.local_kv_message_base(&ids.session_label, ids);
        let identities = Arc::from(kv.lookup_identities(&self.config, &base, 0, prefill_tokens));
        let kv = Arc::clone(kv);
        let config = self.config.clone();
        GenerationAdmissionScheduling::new(
            prompt_tokens,
            Arc::new(move || kv.peek_cache_affinity(&config, &identities)),
        )
    }

    fn generation_admission_work(
        &self,
        prompt: &PreparedGenerationPrompt,
        max_tokens: GenerationTokenLimit,
        prepared_text: Option<&PreparedTextPrompt>,
    ) -> InferenceResult<GenerationAdmissionWork> {
        if let Some(prepared) = prepared_text {
            return Ok(GenerationAdmissionWork::new(
                prepared.token_ids.len(),
                prepared.max_tokens,
            ));
        }
        let estimated_prompt_tokens = prompt.text.len().div_ceil(4).max(1);
        let decode_tokens = max_tokens.resolve(estimated_prompt_tokens, self.ctx_size)?;
        Ok(GenerationAdmissionWork::new(
            estimated_prompt_tokens,
            decode_tokens,
        ))
    }

    pub(super) fn openai_attrs(&self, ids: &InferenceGenerationIds) -> BTreeMap<String, Value> {
        openai_identity_attrs(&self.config, self.mode.label(), ids)
    }

    pub(super) fn insert_runtime_session_stats(
        attrs: &mut BTreeMap<String, Value>,
        prefix: &str,
        stats: &RuntimeSessionStats,
    ) {
        attrs.insert(
            format!("{prefix}.active_sessions"),
            json!(stats.active_sessions),
        );
        attrs.insert(
            format!("{prefix}.idle_sessions"),
            json!(stats.idle_sessions),
        );
        attrs.insert(
            format!("{prefix}.idle_resident_prefixes"),
            json!(stats.idle_resident_prefixes),
        );
        attrs.insert(
            format!("{prefix}.tracked_token_counts"),
            json!(stats.tracked_token_counts),
        );
    }

    pub(super) fn emit_openai_phase(
        &self,
        name: &str,
        timer: PhaseTimer,
        mut attrs: BTreeMap<String, Value>,
    ) -> f64 {
        let elapsed_ms = timer.elapsed_ms();
        attrs.insert("llama_stage.elapsed_ms".to_string(), json!(elapsed_ms));
        let end = now_unix_nanos() as u64;
        self.telemetry
            .emit_debug_span(name, attrs, timer.start_unix_nanos, end);
        elapsed_ms
    }

    pub(super) fn emit_openai_summary(
        &self,
        name: &str,
        timer: PhaseTimer,
        mut attrs: BTreeMap<String, Value>,
    ) -> f64 {
        let elapsed_ms = timer.elapsed_ms();
        attrs.insert("llama_stage.elapsed_ms".to_string(), json!(elapsed_ms));
        let end = now_unix_nanos() as u64;
        self.telemetry
            .emit_span(name, attrs, timer.start_unix_nanos, end);
        elapsed_ms
    }

    pub(super) fn ensure_model(&self, requested: &str) -> InferenceResult<()> {
        ensure_requested_model(&self.model_id, requested)
    }

    #[allow(clippy::too_many_arguments)]
    async fn run_generation(
        &self,
        prompt: PreparedGenerationPrompt,
        max_tokens: GenerationTokenLimit,
        stop: Option<skippy_inference_api::StopSequence>,
        sampling: SamplingConfig,
        hook_request: Option<GenerationRequest>,
        context: InferenceRequestContext,
        ids: InferenceGenerationIds,
    ) -> InferenceResult<GeneratedText> {
        let (prompt, prepared_text) = if prompt.has_media() {
            (prompt, None)
        } else {
            let backend = self.clone();
            let ids_for_tokenize = ids.clone();
            task::spawn_blocking(move || {
                let prepared =
                    backend.prepare_text_prompt(&prompt, max_tokens, &ids_for_tokenize)?;
                Ok::<_, InferenceError>((prompt, Some(prepared)))
            })
            .await
            .map_err(|error| {
                InferenceError::backend(format!("prompt tokenization task failed: {error}"))
            })??
        };
        let admission_work =
            self.generation_admission_work(&prompt, max_tokens, prepared_text.as_ref())?;
        let admission_scheduling =
            self.generation_admission_scheduling(prepared_text.as_ref(), &ids);
        let admit_timer = PhaseTimer::start();
        let cancellation = context.cancellation_token();
        let (permit, session_permit) = self
            .acquire_generation_admission(&ids, &cancellation, admission_work, admission_scheduling)
            .await?;
        let admission_wait_ms = admit_timer.elapsed_ms();
        let mut admit_attrs = self.openai_attrs(&ids);
        admit_attrs.insert(
            "llama_stage.openai_phase".to_string(),
            json!("generation_admit"),
        );
        insert_generation_admission_attrs(
            &mut admit_attrs,
            &permit,
            self.generation_queue_depth.load(Ordering::Acquire),
            self.generation_queue_limit,
        );
        self.emit_openai_phase("stage.openai_generation_admit", admit_timer, admit_attrs);
        let demand = permit.demand_observation();
        let token_budget_stats = permit.token_budget_stats();
        let backend = self.clone();
        let hook_runtime = Some(tokio::runtime::Handle::current());
        let worker_context = context.clone();
        let mut result =
            run_blocking_generation_worker(permit, worker_context.clone(), move |token| {
                let _session_permit = session_permit;
                let output = backend.generate_text(
                    prompt,
                    max_tokens,
                    prepared_text,
                    token_budget_stats,
                    stop.as_ref(),
                    sampling,
                    hook_request,
                    hook_runtime,
                    Some(&token),
                    ids,
                    |_| Ok(()),
                );
                if worker_context.is_cancelled() {
                    Err(request_cancelled_error())
                } else {
                    output
                }
            })
            .await
            .map_err(|error| InferenceError::backend(format!("generation task failed: {error}")))?;
        if let Ok(output) = &mut result {
            output.queue_wait_ms = admission_wait_ms;
        }
        if context.is_cancelled() {
            Err(request_cancelled_error())
        } else {
            if let Ok(output) = &result {
                self.observe_generation_completed(output, demand);
            } else {
                self.observe_generation_failed();
            }
            result
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn run_generation_stream(
        &self,
        prompt: PreparedGenerationPrompt,
        max_tokens: GenerationTokenLimit,
        stop: Option<skippy_inference_api::StopSequence>,
        sampling: SamplingConfig,
        include_usage: bool,
        hook_request: Option<GenerationRequest>,
        parse_chat_output: bool,
        emit_reasoning: bool,
        context: InferenceRequestContext,
        ids: InferenceGenerationIds,
    ) -> InferenceResult<GenerationStream> {
        let (prompt, prepared_text) = if prompt.has_media() {
            (prompt, None)
        } else {
            let backend = self.clone();
            let ids_for_tokenize = ids.clone();
            task::spawn_blocking(move || {
                let prepared =
                    backend.prepare_text_prompt(&prompt, max_tokens, &ids_for_tokenize)?;
                Ok::<_, InferenceError>((prompt, Some(prepared)))
            })
            .await
            .map_err(|error| {
                InferenceError::backend(format!("prompt tokenization task failed: {error}"))
            })??
        };
        let admit_timer = PhaseTimer::start();
        let admission_work =
            self.generation_admission_work(&prompt, max_tokens, prepared_text.as_ref())?;
        let admission_scheduling =
            self.generation_admission_scheduling(prepared_text.as_ref(), &ids);
        let cancellation = context.cancellation_token();
        let (permit, session_permit) = self
            .acquire_generation_admission(&ids, &cancellation, admission_work, admission_scheduling)
            .await?;
        let admission_wait_ms = admit_timer.elapsed_ms();
        let mut admit_attrs = self.openai_attrs(&ids);
        admit_attrs.insert(
            "llama_stage.openai_phase".to_string(),
            json!("generation_admit"),
        );
        insert_generation_admission_attrs(
            &mut admit_attrs,
            &permit,
            self.generation_queue_depth.load(Ordering::Acquire),
            self.generation_queue_limit,
        );
        self.emit_openai_phase("stage.openai_generation_admit", admit_timer, admit_attrs);
        let demand = permit.demand_observation();
        let token_budget_stats = permit.token_budget_stats();
        let backend = self.clone();
        let chat_parse_metadata = prompt.chat_parse_metadata.clone();
        let (tx, rx) = mpsc::channel(16);
        let hook_runtime = Some(tokio::runtime::Handle::current());
        let sender = StreamEventSender::new(
            tx,
            tokio::runtime::Handle::current(),
            STREAM_SEND_STALL_TIMEOUT,
            ids.request_id_string(),
            self.telemetry.clone(),
        );
        let generation_request_needed = self.hook_policy.is_some()
            || hook_request
                .as_ref()
                .is_some_and(crate::frontend::generation::tool_calls_requested);
        let mut hook_request = hook_request;
        let parser_request = if parse_chat_output {
            if generation_request_needed {
                hook_request.clone()
            } else {
                hook_request.take()
            }
        } else {
            None
        };
        let mut chat_stream_parser =
            if let (Some(request), Some(metadata)) = (parser_request, chat_parse_metadata) {
                Some(ChatOutputStreamParser::new(
                    backend.clone(),
                    request,
                    metadata,
                    emit_reasoning,
                )?)
            } else {
                None
            };
        task::spawn_blocking(move || {
            let _session_permit = session_permit;
            let _permit = permit;
            let result = backend.generate_text(
                prompt,
                max_tokens,
                prepared_text,
                token_budget_stats,
                stop.as_ref(),
                sampling,
                hook_request,
                hook_runtime,
                Some(&context.cancellation_token()),
                ids,
                |chunk| {
                    if context.is_cancelled() {
                        return Err(InferenceError::backend("stream receiver cancelled"));
                    }
                    let events = if let Some(parser) = chat_stream_parser.as_mut() {
                        parser.push_delta(chunk)?
                    } else {
                        vec![GenerationStreamEvent::Delta(chunk.to_string())]
                    };
                    for event in events {
                        sender.send(Ok(event), &context)?;
                    }
                    Ok(())
                },
            );
            if context.is_cancelled() {
                let _ = sender.send_terminal(Err(request_cancelled_error()));
                return;
            }
            match result {
                Ok(mut output) => {
                    output.queue_wait_ms = admission_wait_ms;
                    backend.observe_generation_completed(&output, demand);
                    let finish_reason = if let Some(parser) = chat_stream_parser.as_mut() {
                        match parser.finish(&output.text, output.finish_reason) {
                            Ok((events, reason)) => {
                                for event in events {
                                    if sender.send_terminal(Ok(event)).is_err() {
                                        return;
                                    }
                                }
                                reason
                            }
                            Err(error) => {
                                let _ = sender.send_terminal(Err(error));
                                return;
                            }
                        }
                    } else {
                        output.finish_reason
                    };
                    if should_emit_stream_usage(include_usage, &context)
                        && sender
                            .send_terminal(Ok(GenerationStreamEvent::Usage(
                                output.usage(),
                                output.timings(),
                            )))
                            .is_err()
                    {
                        return;
                    }
                    let _ = sender.send_terminal(Ok(GenerationStreamEvent::Done(finish_reason)));
                }
                Err(error) => {
                    backend.observe_generation_failed();
                    let _ = sender.send_terminal(Err(error));
                }
            }
        });
        Ok(Box::pin(stream::unfold(rx, |mut rx| async {
            rx.recv().await.map(|item| (item, rx))
        })))
    }
}

#[cfg(test)]
mod tests;
