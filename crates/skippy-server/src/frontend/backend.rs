use crate::frontend::generation::ChatOutputStreamParser;
use crate::frontend::generation::GENERATION_ADMISSION_TIMEOUT;
use crate::frontend::generation::GeneratedText;
use crate::frontend::generation::GenerationSessionLockEntry;
use crate::frontend::generation::GenerationStream;
use crate::frontend::generation::GenerationStreamEvent;
use crate::frontend::generation::GenerationTokenLimit;
use crate::frontend::generation::OpenAiCacheHints;
use crate::frontend::generation::OpenAiGenerationIds;
use crate::frontend::generation::PhaseTimer;
use crate::frontend::generation::PreparedGenerationPrompt;
use crate::frontend::generation::StageOpenAiBackend;
use crate::frontend::generation::acquire_generation_permit_with_queue_reservation;
use crate::frontend::generation::apply_reasoning_visibility;
use crate::frontend::generation::chat_output_parser_required;
use crate::frontend::generation::chat_response_from_generated_text;
use crate::frontend::generation::completion_response_from_generated_text;
use crate::frontend::generation::ensure_requested_model;
use crate::frontend::generation::generation_event_to_chat_chunk;
use crate::frontend::generation::generation_event_to_completion_chunk;
use crate::frontend::generation::generation_queue_full_error;
use crate::frontend::generation::generation_queue_timeout_error;
use crate::frontend::generation::reserve_generation_queue;
use crate::frontend::generation::template_exposes_reasoning;
use crate::frontend::request::{
    apply_chat_request_defaults, apply_completion_request_defaults, chat_sampling_config,
    chat_template_options, completion_sampling_config, ensure_chat_runtime_features_supported,
    ensure_completion_runtime_features_supported,
};
use crate::runtime_state::RuntimeSessionStats;
use crate::telemetry::Telemetry;
use crate::telemetry::lifecycle_attrs;
use crate::telemetry::now_unix_nanos;
use async_trait::async_trait;
use futures_util::StreamExt;
use futures_util::stream;
use openai_frontend::ChatCompletionRequest;
use openai_frontend::ChatCompletionResponse;
use openai_frontend::ChatCompletionStream;
use openai_frontend::CompletionRequest;
use openai_frontend::CompletionResponse;
use openai_frontend::CompletionStream;
use openai_frontend::ModelObject;
use openai_frontend::OpenAiBackend;
use openai_frontend::OpenAiError;
use openai_frontend::OpenAiRequestContext;
use openai_frontend::OpenAiResult;
use openai_frontend::apply_chat_hook_outcome;
use openai_frontend::chat_mesh_hooks_enabled;
use serde_json::Value;
use serde_json::json;
use skippy_metrics::attr as attr_key;
use skippy_runtime::SamplingConfig;
use std::collections::BTreeMap;
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
use tokio::task;

fn request_cancelled_error() -> OpenAiError {
    OpenAiError::cancelled("request cancelled")
}

/// How long a full SSE event channel is treated as merely backed up before
/// the generation worker gives up on it and frees its execution lane.
///
/// Deliberately its own value rather than an alias of
/// `GENERATION_ADMISSION_TIMEOUT`: admission queueing and stream-stall
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
    tx: mpsc::Sender<OpenAiResult<GenerationStreamEvent>>,
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
    /// Structured telemetry sink for the stall/drop diagnostics. `skippy-server`
    /// routes operator-facing signal through `Telemetry`, not a logging facade,
    /// so a freed execution lane is correlated by `request_id` and observable
    /// without scraping stderr.
    telemetry: Telemetry,
}

impl StreamEventSender {
    fn new(
        tx: mpsc::Sender<OpenAiResult<GenerationStreamEvent>>,
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
    fn mark_receiver_unreachable(&self, context: &OpenAiRequestContext) {
        self.receiver_unreachable.store(true, Ordering::Release);
        context.cancel();
    }

    /// Send one in-flight event (from the `on_text_chunk` callback). Checks
    /// cancellation first so an already-cancelled request returns
    /// immediately and deterministically, matching the pre-existing
    /// early-return semantics.
    fn send(
        &self,
        event: OpenAiResult<GenerationStreamEvent>,
        context: &OpenAiRequestContext,
    ) -> Result<(), OpenAiError> {
        if self.receiver_unreachable.load(Ordering::Acquire) {
            return Err(OpenAiError::backend("stream receiver unreachable"));
        }
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
                        Err(OpenAiError::backend("stream receiver dropped"))
                    }
                },
                () = sleep => {
                    self.emit_lane_freed("receiver_stalled", "in_flight");
                    self.mark_receiver_unreachable(context);
                    Err(OpenAiError::backend(
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
    fn send_terminal(&self, event: OpenAiResult<GenerationStreamEvent>) -> Result<(), OpenAiError> {
        if self.receiver_unreachable.load(Ordering::Acquire) {
            return Err(OpenAiError::backend("stream receiver unreachable"));
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
                        Err(OpenAiError::backend("stream receiver dropped"))
                    }
                },
                () = sleep => {
                    self.emit_lane_freed("receiver_stalled", "terminal");
                    self.receiver_unreachable.store(true, Ordering::Release);
                    Err(OpenAiError::backend(
                        "stream receiver stalled without draining",
                    ))
                }
            }
        })
    }
}

fn should_emit_stream_usage(request_include_usage: bool, context: &OpenAiRequestContext) -> bool {
    request_include_usage || context.observes_stream_usage()
}

struct GenerationSessionPermit {
    registry: Arc<Mutex<BTreeMap<String, Arc<GenerationSessionLockEntry>>>>,
    key: String,
    entry: Arc<GenerationSessionLockEntry>,
    permit: Option<OwnedSemaphorePermit>,
}

impl GenerationSessionPermit {
    fn new(
        registry: Arc<Mutex<BTreeMap<String, Arc<GenerationSessionLockEntry>>>>,
        key: String,
    ) -> OpenAiResult<Self> {
        let entry = {
            let mut locks = registry
                .lock()
                .map_err(|_| OpenAiError::backend("generation session lock map poisoned"))?;
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

    fn try_acquire(&mut self) -> OpenAiResult<bool> {
        match self.entry.semaphore.clone().try_acquire_owned() {
            Ok(permit) => {
                self.permit = Some(permit);
                Ok(true)
            }
            Err(TryAcquireError::NoPermits) => Ok(false),
            Err(TryAcquireError::Closed) => {
                Err(OpenAiError::backend("generation session lock closed"))
            }
        }
    }

    async fn acquire_until(
        mut self,
        deadline: Instant,
        admission_timeout: Duration,
        cancellation: &openai_frontend::CancellationToken,
    ) -> OpenAiResult<Self> {
        let acquire = tokio::time::timeout_at(
            tokio::time::Instant::from_std(deadline),
            self.entry.semaphore.clone().acquire_owned(),
        );
        let permit = tokio::select! {
            result = acquire => result
                .map_err(|_| generation_queue_timeout_error(admission_timeout))?
                .map_err(|_| OpenAiError::backend("generation session lock closed"))?,
            () = cancellation.cancelled() => return Err(request_cancelled_error()),
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

fn trusted_generation_session_key(ids: &OpenAiGenerationIds) -> Option<String> {
    ids.agent_session_trusted.then(|| ids.session_id_string())
}

#[derive(Clone)]
struct GenerationAdmissionController {
    generation_limit: Arc<Semaphore>,
    generation_queue_depth: Arc<AtomicUsize>,
    generation_queue_limit: usize,
    generation_session_locks: Arc<Mutex<BTreeMap<String, Arc<GenerationSessionLockEntry>>>>,
}

impl GenerationAdmissionController {
    fn for_backend(backend: &StageOpenAiBackend) -> Self {
        Self {
            generation_limit: backend.generation_limit.clone(),
            generation_queue_depth: backend.generation_queue_depth.clone(),
            generation_queue_limit: backend.generation_queue_limit,
            generation_session_locks: backend.generation_session_locks.clone(),
        }
    }

    async fn acquire(
        &self,
        ids: &OpenAiGenerationIds,
        cancellation: &openai_frontend::CancellationToken,
        admission_timeout: Duration,
    ) -> OpenAiResult<(OwnedSemaphorePermit, Option<GenerationSessionPermit>)> {
        let deadline = Instant::now()
            .checked_add(admission_timeout)
            .ok_or_else(|| OpenAiError::backend("generation admission deadline overflow"))?;
        let session_permit = self
            .acquire_session_until(ids, deadline, admission_timeout, cancellation)
            .await?;

        if Instant::now() >= deadline {
            return Err(generation_queue_timeout_error(admission_timeout));
        }
        let generation_permit = self
            .acquire_generation_permit_until(deadline, admission_timeout, cancellation)
            .await?;
        if cancellation.is_cancelled() {
            return Err(request_cancelled_error());
        }
        Ok((generation_permit, session_permit))
    }

    async fn acquire_session_until(
        &self,
        ids: &OpenAiGenerationIds,
        deadline: Instant,
        admission_timeout: Duration,
        cancellation: &openai_frontend::CancellationToken,
    ) -> OpenAiResult<Option<GenerationSessionPermit>> {
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
        deadline: Instant,
        admission_timeout: Duration,
        cancellation: &openai_frontend::CancellationToken,
    ) -> OpenAiResult<OwnedSemaphorePermit> {
        if cancellation.is_cancelled() {
            return Err(request_cancelled_error());
        }
        match self.generation_limit.clone().try_acquire_owned() {
            Ok(permit) => return Ok(permit),
            Err(TryAcquireError::Closed) => {
                return Err(OpenAiError::backend("generation lanes closed"));
            }
            Err(TryAcquireError::NoPermits) => {}
        }
        let reservation = reserve_generation_queue(
            self.generation_queue_depth.clone(),
            self.generation_queue_limit,
        )
        .ok_or_else(generation_queue_full_error)?;
        acquire_generation_permit_with_queue_reservation(
            self.generation_limit.clone(),
            reservation,
            admission_timeout,
            deadline,
            cancellation,
        )
        .await
    }
}

fn generation_ids(
    cache: OpenAiCacheHints,
    agent_session_id: Option<&str>,
    context: &OpenAiRequestContext,
) -> OpenAiGenerationIds {
    OpenAiGenerationIds::new_with_trust(
        cache,
        agent_session_id,
        context.has_trusted_agent_session(),
    )
}

pub(in crate::frontend) async fn run_blocking_generation_worker<T, F>(
    permit: OwnedSemaphorePermit,
    context: OpenAiRequestContext,
    work: F,
) -> Result<T, task::JoinError>
where
    T: Send + 'static,
    F: FnOnce(openai_frontend::CancellationToken) -> T + Send + 'static,
{
    task::spawn_blocking(move || {
        let _permit = permit;
        work(context.cancellation_token())
    })
    .await
}

#[async_trait]
impl OpenAiBackend for StageOpenAiBackend {
    async fn models(&self) -> OpenAiResult<Vec<ModelObject>> {
        Ok(vec![ModelObject::new(self.model_id.clone())])
    }

    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> OpenAiResult<ChatCompletionResponse> {
        self.chat_completion_with_context(request, OpenAiRequestContext::new())
            .await
    }

    async fn chat_completion_with_context(
        &self,
        mut request: ChatCompletionRequest,
        context: OpenAiRequestContext,
    ) -> OpenAiResult<ChatCompletionResponse> {
        let ids = generation_ids(
            OpenAiCacheHints::from_chat_request(&request),
            request.agent_session(),
            &context,
        );
        let request_timer = PhaseTimer::start();
        self.apply_before_chat_hooks(&mut request).await?;
        self.ensure_model(&request.model)?;
        apply_chat_request_defaults(&mut request, &self.request_defaults)?;
        ensure_chat_runtime_features_supported(&request)?;
        let sampling = chat_sampling_config(&request)?;
        let template_options = chat_template_options(&request, &self.request_defaults)?;
        let parse_chat_output = chat_output_parser_required(&request, &template_options);
        let template_timer = PhaseTimer::start();
        let prompt = self.prepare_chat_prompt(&request, template_options.clone())?;
        let mut template_attrs = self.openai_attrs(&ids);
        template_attrs.insert(
            "llama_stage.openai_operation".to_string(),
            json!("chat_completion"),
        );
        template_attrs.insert(
            "llama_stage.chat_message_count".to_string(),
            json!(request.messages.len()),
        );
        template_attrs.insert(
            "llama_stage.prompt_chars".to_string(),
            json!(prompt.text.len()),
        );
        template_attrs.insert(
            "llama_stage.media_item_count".to_string(),
            json!(prompt.media.len()),
        );
        self.emit_openai_phase("stage.openai_chat_template", template_timer, template_attrs);
        let max_tokens = GenerationTokenLimit::from_request(
            request.effective_max_tokens(),
            self.default_max_tokens,
        );
        let chat_parse_metadata = prompt.chat_parse_metadata.clone();
        let output = self
            .run_generation(
                prompt,
                max_tokens,
                request.stop.clone(),
                sampling,
                Some(request.clone()),
                context,
                ids.clone(),
            )
            .await?;
        let response_timer = PhaseTimer::start();
        let parsed_message = if parse_chat_output {
            self.parse_chat_output(
                &output.text,
                &request,
                chat_parse_metadata.as_deref(),
                false,
            )?
        } else {
            None
        };
        let parsed_message = apply_reasoning_visibility(parsed_message, &template_options);
        let response =
            chat_response_from_generated_text(request.model.clone(), &output, parsed_message);
        let mut response_attrs = self.openai_attrs(&ids);
        response_attrs.insert(
            "llama_stage.openai_operation".to_string(),
            json!("chat_completion"),
        );
        response_attrs.insert(
            "llama_stage.prompt_token_count".to_string(),
            json!(output.prompt_tokens),
        );
        response_attrs.insert(
            "llama_stage.completion_token_count".to_string(),
            json!(output.completion_tokens),
        );
        self.emit_openai_phase(
            "stage.openai_response_build",
            response_timer,
            response_attrs,
        );
        let mut summary_attrs = self.openai_attrs(&ids);
        summary_attrs.insert(
            "llama_stage.openai_operation".to_string(),
            json!("chat_completion"),
        );
        summary_attrs.insert("llama_stage.status".to_string(), json!("ok"));
        summary_attrs.insert(
            "llama_stage.prompt_token_count".to_string(),
            json!(output.prompt_tokens),
        );
        summary_attrs.insert(
            "llama_stage.completion_token_count".to_string(),
            json!(output.completion_tokens),
        );
        self.emit_openai_summary("stage.openai_request_summary", request_timer, summary_attrs);
        Ok(response)
    }

    async fn chat_completion_stream(
        &self,
        mut request: ChatCompletionRequest,
        context: OpenAiRequestContext,
    ) -> OpenAiResult<ChatCompletionStream> {
        let ids = generation_ids(
            OpenAiCacheHints::from_chat_request(&request),
            request.agent_session(),
            &context,
        );
        self.apply_before_chat_hooks(&mut request).await?;
        self.ensure_model(&request.model)?;
        apply_chat_request_defaults(&mut request, &self.request_defaults)?;
        ensure_chat_runtime_features_supported(&request)?;
        let sampling = chat_sampling_config(&request)?;
        let include_usage = request.include_usage();
        let template_options = chat_template_options(&request, &self.request_defaults)?;
        let parse_chat_output = chat_output_parser_required(&request, &template_options);
        let emit_reasoning = template_exposes_reasoning(&template_options);
        let template_timer = PhaseTimer::start();
        let prompt = self.prepare_chat_prompt(&request, template_options)?;
        let mut template_attrs = self.openai_attrs(&ids);
        template_attrs.insert(
            "llama_stage.openai_operation".to_string(),
            json!("chat_completion_stream"),
        );
        template_attrs.insert(
            "llama_stage.chat_message_count".to_string(),
            json!(request.messages.len()),
        );
        template_attrs.insert(
            "llama_stage.prompt_chars".to_string(),
            json!(prompt.text.len()),
        );
        template_attrs.insert(
            "llama_stage.media_item_count".to_string(),
            json!(prompt.media.len()),
        );
        self.emit_openai_phase("stage.openai_chat_template", template_timer, template_attrs);
        let max_tokens = GenerationTokenLimit::from_request(
            request.effective_max_tokens(),
            self.default_max_tokens,
        );
        let model = request.model.clone();
        let stream = self
            .run_generation_stream(
                prompt,
                max_tokens,
                request.stop.clone(),
                sampling,
                include_usage,
                Some(request.clone()),
                parse_chat_output,
                emit_reasoning,
                context,
                ids,
            )
            .await?;
        Ok(Box::pin(stream.map(move |event| {
            generation_event_to_chat_chunk(event, &model)
        })))
    }

    async fn completion(&self, request: CompletionRequest) -> OpenAiResult<CompletionResponse> {
        self.completion_with_context(request, OpenAiRequestContext::new())
            .await
    }

    async fn completion_with_context(
        &self,
        mut request: CompletionRequest,
        context: OpenAiRequestContext,
    ) -> OpenAiResult<CompletionResponse> {
        let ids = generation_ids(
            OpenAiCacheHints::from_completion_request(&request),
            request.agent_session(),
            &context,
        );
        let request_timer = PhaseTimer::start();
        self.ensure_model(&request.model)?;
        apply_completion_request_defaults(&mut request, &self.request_defaults);
        ensure_completion_runtime_features_supported(&request)?;
        let sampling = completion_sampling_config(&request)?;
        let max_tokens =
            GenerationTokenLimit::from_request(request.max_tokens, self.default_max_tokens);
        let prompt_timer = PhaseTimer::start();
        let prompt = PreparedGenerationPrompt::text(request.prompt.text_lossy());
        let mut prompt_attrs = self.openai_attrs(&ids);
        prompt_attrs.insert(
            "llama_stage.openai_operation".to_string(),
            json!("completion"),
        );
        prompt_attrs.insert(
            "llama_stage.prompt_chars".to_string(),
            json!(prompt.text.len()),
        );
        self.emit_openai_phase("stage.openai_prompt_prepare", prompt_timer, prompt_attrs);
        let output = self
            .run_generation(
                prompt,
                max_tokens,
                request.stop.clone(),
                sampling,
                None,
                context,
                ids.clone(),
            )
            .await?;
        let response_timer = PhaseTimer::start();
        let response = completion_response_from_generated_text(request.model, &output);
        let mut response_attrs = self.openai_attrs(&ids);
        response_attrs.insert(
            "llama_stage.openai_operation".to_string(),
            json!("completion"),
        );
        response_attrs.insert(
            "llama_stage.prompt_token_count".to_string(),
            json!(output.prompt_tokens),
        );
        response_attrs.insert(
            "llama_stage.completion_token_count".to_string(),
            json!(output.completion_tokens),
        );
        self.emit_openai_phase(
            "stage.openai_response_build",
            response_timer,
            response_attrs,
        );
        let mut summary_attrs = self.openai_attrs(&ids);
        summary_attrs.insert(
            "llama_stage.openai_operation".to_string(),
            json!("completion"),
        );
        summary_attrs.insert("llama_stage.status".to_string(), json!("ok"));
        summary_attrs.insert(
            "llama_stage.prompt_token_count".to_string(),
            json!(output.prompt_tokens),
        );
        summary_attrs.insert(
            "llama_stage.completion_token_count".to_string(),
            json!(output.completion_tokens),
        );
        self.emit_openai_summary("stage.openai_request_summary", request_timer, summary_attrs);
        Ok(response)
    }

    async fn completion_stream(
        &self,
        mut request: CompletionRequest,
        context: OpenAiRequestContext,
    ) -> OpenAiResult<CompletionStream> {
        let ids = generation_ids(
            OpenAiCacheHints::from_completion_request(&request),
            request.agent_session(),
            &context,
        );
        self.ensure_model(&request.model)?;
        apply_completion_request_defaults(&mut request, &self.request_defaults);
        ensure_completion_runtime_features_supported(&request)?;
        let sampling = completion_sampling_config(&request)?;
        let include_usage = request.include_usage();
        let max_tokens =
            GenerationTokenLimit::from_request(request.max_tokens, self.default_max_tokens);
        let model = request.model.clone();
        let prompt_timer = PhaseTimer::start();
        let prompt = PreparedGenerationPrompt::text(request.prompt.text_lossy());
        let mut prompt_attrs = self.openai_attrs(&ids);
        prompt_attrs.insert(
            "llama_stage.openai_operation".to_string(),
            json!("completion_stream"),
        );
        prompt_attrs.insert(
            "llama_stage.prompt_chars".to_string(),
            json!(prompt.text.len()),
        );
        self.emit_openai_phase("stage.openai_prompt_prepare", prompt_timer, prompt_attrs);
        let stream = self
            .run_generation_stream(
                prompt,
                max_tokens,
                request.stop.clone(),
                sampling,
                include_usage,
                None,
                false,
                false,
                context,
                ids,
            )
            .await?;
        Ok(Box::pin(stream.map(move |event| {
            generation_event_to_completion_chunk(event, &model)
        })))
    }
}

impl StageOpenAiBackend {
    async fn acquire_generation_admission(
        &self,
        ids: &OpenAiGenerationIds,
        cancellation: &openai_frontend::CancellationToken,
    ) -> OpenAiResult<(OwnedSemaphorePermit, Option<GenerationSessionPermit>)> {
        GenerationAdmissionController::for_backend(self)
            .acquire(ids, cancellation, GENERATION_ADMISSION_TIMEOUT)
            .await
    }

    pub(super) fn openai_attrs(&self, ids: &OpenAiGenerationIds) -> BTreeMap<String, Value> {
        let mut attrs = lifecycle_attrs(&self.config);
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
            json!(self.mode.label()),
        );
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

    pub(super) fn ensure_model(&self, requested: &str) -> OpenAiResult<()> {
        ensure_requested_model(&self.model_id, requested)
    }

    async fn apply_before_chat_hooks(
        &self,
        request: &mut ChatCompletionRequest,
    ) -> OpenAiResult<()> {
        let Some(hooks) = self.hook_policy.as_ref() else {
            return Ok(());
        };
        if !chat_mesh_hooks_enabled(request) {
            return Ok(());
        }
        let outcome = hooks.before_chat_completion(request).await?;
        apply_chat_hook_outcome(request, &outcome);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    async fn run_generation(
        &self,
        prompt: PreparedGenerationPrompt,
        max_tokens: GenerationTokenLimit,
        stop: Option<openai_frontend::StopSequence>,
        sampling: SamplingConfig,
        hook_request: Option<ChatCompletionRequest>,
        context: OpenAiRequestContext,
        ids: OpenAiGenerationIds,
    ) -> OpenAiResult<GeneratedText> {
        let admit_timer = PhaseTimer::start();
        let cancellation = context.cancellation_token();
        let (permit, session_permit) = self
            .acquire_generation_admission(&ids, &cancellation)
            .await?;
        let mut admit_attrs = self.openai_attrs(&ids);
        admit_attrs.insert(
            "llama_stage.openai_phase".to_string(),
            json!("generation_admit"),
        );
        self.emit_openai_phase("stage.openai_generation_admit", admit_timer, admit_attrs);
        let backend = self.clone();
        let hook_runtime = Some(tokio::runtime::Handle::current());
        let worker_context = context.clone();
        let result = run_blocking_generation_worker(permit, worker_context.clone(), move |token| {
            let _session_permit = session_permit;
            let output = backend.generate_text(
                prompt,
                max_tokens,
                None,
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
        .map_err(|error| OpenAiError::backend(format!("generation task failed: {error}")))?;
        if context.is_cancelled() {
            Err(request_cancelled_error())
        } else {
            result
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn run_generation_stream(
        &self,
        prompt: PreparedGenerationPrompt,
        max_tokens: GenerationTokenLimit,
        stop: Option<openai_frontend::StopSequence>,
        sampling: SamplingConfig,
        include_usage: bool,
        hook_request: Option<ChatCompletionRequest>,
        parse_chat_output: bool,
        emit_reasoning: bool,
        context: OpenAiRequestContext,
        ids: OpenAiGenerationIds,
    ) -> OpenAiResult<GenerationStream> {
        let prepared_text = if prompt.has_media() {
            None
        } else {
            let backend = self.clone();
            let prompt_for_tokenize = prompt.clone();
            let ids_for_tokenize = ids.clone();
            Some(
                task::spawn_blocking(move || {
                    backend.prepare_text_prompt(&prompt_for_tokenize, max_tokens, &ids_for_tokenize)
                })
                .await
                .map_err(|error| {
                    OpenAiError::backend(format!("prompt tokenization task failed: {error}"))
                })??,
            )
        };
        let admit_timer = PhaseTimer::start();
        let cancellation = context.cancellation_token();
        let (permit, session_permit) = self
            .acquire_generation_admission(&ids, &cancellation)
            .await?;
        let mut admit_attrs = self.openai_attrs(&ids);
        admit_attrs.insert(
            "llama_stage.openai_phase".to_string(),
            json!("generation_admit"),
        );
        self.emit_openai_phase("stage.openai_generation_admit", admit_timer, admit_attrs);
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
        let mut chat_stream_parser = if let (true, Some(request), Some(metadata)) =
            (parse_chat_output, hook_request.clone(), chat_parse_metadata)
        {
            Some(ChatOutputStreamParser::new(
                backend.clone(),
                request,
                metadata,
                emit_reasoning,
            ))
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
                stop.as_ref(),
                sampling,
                hook_request,
                hook_runtime,
                Some(&context.cancellation_token()),
                ids,
                |chunk| {
                    if context.is_cancelled() {
                        return Err(OpenAiError::backend("stream receiver cancelled"));
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
                Ok(output) => {
                    let finish_reason = if let Some(parser) = chat_stream_parser.as_mut() {
                        match parser.finish(&output.text) {
                            Ok(events) => {
                                for event in events {
                                    if sender.send_terminal(Ok(event)).is_err() {
                                        return;
                                    }
                                }
                                parser.finish_reason(output.finish_reason)
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
                            .send_terminal(Ok(GenerationStreamEvent::Usage(output.usage())))
                            .is_err()
                    {
                        return;
                    }
                    let _ = sender.send_terminal(Ok(GenerationStreamEvent::Done(finish_reason)));
                }
                Err(error) => {
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
