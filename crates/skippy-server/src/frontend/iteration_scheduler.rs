mod cache_runtime;

use self::cache_runtime::{
    CacheAffinityRefresh, CacheRuntimeQueue, CacheRuntimeTelemetry, should_serve_cache_runtime,
};
use crate::frontend::admission::DECODE_BATCH_HEADROOM_TOKENS;
use crate::frontend::generation::TokenControl;
use crate::frontend::generation::generation_queue_full_error;
use crate::frontend::util::openai_backend_error;
use crate::runtime_state::{RuntimeIterationBatchRequest, RuntimeSessionAlignStats, RuntimeState};
use crate::telemetry::Telemetry;
use openai_frontend::{OpenAiError, OpenAiResult};
use serde_json::json;
use skippy_protocol::StageConfig;
use skippy_runtime::{ActivationFrame, IterationBatchPhase, SamplingConfig};
use skippy_scheduler::{
    AdmissionError, CacheAffinity, IterationPhase, IterationPrediction, MemoryComponent, Scheduler,
    SchedulerConfig, Sequence,
};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::env;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc as std_mpsc;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const MAX_NATIVE_ITERATION_TOKENS: usize = 2048;
const CANCELLATION_POLL_INTERVAL: Duration = Duration::from_millis(10);
const MIN_COMMAND_QUEUE_CAPACITY: usize = 8;
const MAX_COMMANDS_PER_TURN: usize = 64;
const CACHE_AGING_COST_PER_TURN: u64 = 4_096;
const SAFE_MODE_ENV: &str = "SKIPPY_ITERATION_SCHEDULER_SAFE_MODE";

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(super) struct ScheduledGenerationStats {
    pub(super) prompt_ms: f64,
    pub(super) predicted_ms: f64,
}

pub(crate) struct IterationScheduler {
    shared: Arc<IterationSchedulerShared>,
}

pub(super) struct ScheduledGenerationRequest<'a> {
    pub(super) id: &'a str,
    pub(super) prompt_tokens: &'a [i32],
    pub(super) max_tokens: u32,
    pub(super) sampling: Option<&'a SamplingConfig>,
    pub(super) chat_sampling_metadata: Option<&'a str>,
    pub(super) cancellation: Option<&'a openai_frontend::CancellationToken>,
}

#[derive(Debug, Clone)]
pub(crate) struct SchedulerIterationOutcome {
    pub(crate) predicted: i32,
    pub(crate) output: ActivationFrame,
    pub(crate) batch_size: usize,
    pub(crate) batch_wait_ms: f64,
    pub(crate) runtime_lock_wait_ms: f64,
    pub(crate) runtime_lock_hold_ms: f64,
    pub(crate) session_alignment: Option<RuntimeSessionAlignStats>,
}

#[derive(Debug)]
pub(crate) struct SchedulerRuntimeOutcome<T> {
    pub(crate) value: T,
    pub(crate) queue_wait_ms: f64,
    pub(crate) runtime_lock_wait_ms: f64,
    pub(crate) runtime_lock_hold_ms: f64,
}

struct IterationSchedulerShared {
    commands: std_mpsc::SyncSender<SchedulerCommand>,
    owner_count: AtomicUsize,
    worker: Mutex<Option<JoinHandle<()>>>,
}

struct ScheduledRequest {
    id: String,
    prompt_tokens: Vec<i32>,
    max_tokens: u32,
    sampling: Option<SamplingConfig>,
    chat_sampling_metadata: Option<String>,
    reply: std_mpsc::Sender<SchedulerEvent>,
}

struct DirectIteration {
    session_id: String,
    target_token_count: Option<u64>,
    token_ids: Vec<i32>,
    positions: Vec<i32>,
    sampling: Option<SamplingConfig>,
    input: Option<ActivationFrame>,
    sample_last: bool,
    phase: IterationBatchPhase,
    enqueued_at: Instant,
    reply: std_mpsc::SyncSender<OpenAiResult<SchedulerIterationOutcome>>,
}

type RuntimeOperationFn = Box<dyn FnOnce(&Arc<Mutex<RuntimeState>>) + Send>;
type RuntimeSetupOutcome = (Vec<String>, Vec<(String, OpenAiError)>);

struct RuntimeOperation {
    label: &'static str,
    run: RuntimeOperationFn,
}

fn runtime_operation<T>(
    label: &'static str,
    operation: impl FnOnce(&mut RuntimeState) -> OpenAiResult<T> + Send + 'static,
) -> (
    RuntimeOperation,
    std_mpsc::Receiver<OpenAiResult<SchedulerRuntimeOutcome<T>>>,
)
where
    T: Send + 'static,
{
    let (reply, result) = std_mpsc::sync_channel(1);
    let enqueued_at = Instant::now();
    let operation = RuntimeOperation {
        label,
        run: Box::new(move |runtime: &Arc<Mutex<RuntimeState>>| {
            let queue_wait_ms = enqueued_at.elapsed().as_secs_f64() * 1_000.0;
            let lock_started = Instant::now();
            let outcome = runtime
                .lock()
                .map_err(|_| OpenAiError::backend("runtime lock poisoned"))
                .and_then(|mut runtime| {
                    let runtime_lock_wait_ms = lock_started.elapsed().as_secs_f64() * 1_000.0;
                    let hold_started = Instant::now();
                    operation(&mut runtime).map(|value| SchedulerRuntimeOutcome {
                        value,
                        queue_wait_ms,
                        runtime_lock_wait_ms,
                        runtime_lock_hold_ms: hold_started.elapsed().as_secs_f64() * 1_000.0,
                    })
                });
            let _ = reply.send(outcome);
        }),
    };
    (operation, result)
}

enum SchedulerCommand {
    Submit(ScheduledRequest),
    ExecuteIteration(DirectIteration),
    ExecuteRuntime(RuntimeOperation),
    ExecuteCacheAwareRuntime(
        RuntimeOperation,
        CacheAffinity,
        Arc<[i32]>,
        u64,
        Option<CacheAffinityRefresh>,
    ),
    Cancel(String),
    Shutdown,
}

enum SchedulerEvent {
    Token {
        token: i32,
        ack: std_mpsc::SyncSender<TokenControl>,
    },
    Complete,
    Error(OpenAiError),
}

struct RequestState {
    reply: std_mpsc::Sender<SchedulerEvent>,
    pending_controls: VecDeque<std_mpsc::Receiver<TokenControl>>,
    sampling: Option<SamplingConfig>,
    chat_sampling_metadata: Option<String>,
    prompt_token_count: usize,
    runtime_configured: bool,
}

struct SchedulerWorker {
    runtime: Arc<Mutex<RuntimeState>>,
    scheduler: Scheduler,
    requests: BTreeMap<String, RequestState>,
    direct_iterations: VecDeque<DirectIteration>,
    cache_runtime_queue: CacheRuntimeQueue,
    commands: std_mpsc::Receiver<SchedulerCommand>,
    kv_capacity_tokens: usize,
    max_direct_batch_size: usize,
    max_commands_per_turn: usize,
    iteration_interval: Duration,
    telemetry: Option<Telemetry>,
    last_served_direct: bool,
    last_served_cache_runtime: bool,
    last_emitted_lifecycle_counters: (u64, u64, u64, u64),
}

impl IterationScheduler {
    pub(crate) fn new(
        runtime: Arc<Mutex<RuntimeState>>,
        config: &StageConfig,
        queue_capacity: usize,
        telemetry: Telemetry,
    ) -> OpenAiResult<Self> {
        let (lane_count, kv_pool_tokens) = {
            let runtime = runtime
                .lock()
                .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
            (
                runtime.lane_count() as usize,
                runtime.kv_pool_tokens() as usize,
            )
        };
        let safe_mode = scheduler_safe_mode_from_value(env::var(SAFE_MODE_ENV).ok().as_deref());
        let scheduler_lane_count = if safe_mode { 1 } else { lane_count };
        let scheduler_config = build_scheduler_config(
            scheduler_lane_count,
            kv_pool_tokens,
            stage_recurrent_bytes_per_native_sequence(config),
            config.n_batch,
            config.n_ubatch,
            queue_capacity,
        );
        let iteration_interval = scheduler_config.iteration_interval;
        let max_consecutive_prefill_iterations =
            scheduler_config.max_consecutive_prefill_iterations;
        let mixed_prefill_decode = scheduler_config.mixed_prefill_decode;
        let cache_runtime_queue = CacheRuntimeQueue::new(
            scheduler_config.cache_aging_cost_per_iteration,
            scheduler_config.group_waiting_prefixes,
        );
        let kv_capacity_tokens = scheduler_config
            .memory_components
            .first()
            .and_then(|component| usize::try_from(component.capacity_bytes).ok())
            .unwrap_or(1);
        let scheduler = Scheduler::new(scheduler_config);
        let command_queue_capacity = queue_capacity
            .saturating_add(scheduler_lane_count)
            .max(MIN_COMMAND_QUEUE_CAPACITY);
        telemetry.emit(
            "stage.scheduler_start",
            BTreeMap::from([
                ("skippy.scheduler.safe_mode".to_string(), json!(safe_mode)),
                (
                    "skippy.scheduler.command_queue_capacity".to_string(),
                    json!(command_queue_capacity),
                ),
                (
                    "skippy.scheduler.max_active_sequences".to_string(),
                    json!(scheduler_lane_count),
                ),
                (
                    "skippy.scheduler.max_consecutive_prefill_iterations".to_string(),
                    json!(max_consecutive_prefill_iterations),
                ),
                (
                    "skippy.scheduler.mixed_prefill_decode".to_string(),
                    json!(mixed_prefill_decode),
                ),
                (
                    "skippy.scheduler.cache_policy".to_string(),
                    json!("weighted_lpm_aging_dfs_waiting_prefix"),
                ),
            ]),
        );
        let (commands, receiver) = std_mpsc::sync_channel(command_queue_capacity);
        let worker = thread::Builder::new()
            .name("skippy-iteration-scheduler".to_string())
            .spawn(move || {
                SchedulerWorker {
                    runtime,
                    scheduler,
                    requests: BTreeMap::new(),
                    direct_iterations: VecDeque::new(),
                    cache_runtime_queue,
                    commands: receiver,
                    kv_capacity_tokens,
                    max_direct_batch_size: scheduler_lane_count.max(1),
                    max_commands_per_turn: command_queue_capacity.min(MAX_COMMANDS_PER_TURN),
                    iteration_interval,
                    telemetry: Some(telemetry),
                    last_served_direct: false,
                    last_served_cache_runtime: false,
                    last_emitted_lifecycle_counters: (0, 0, 0, 0),
                }
                .run();
            })
            .map_err(|error| {
                OpenAiError::backend(format!("spawn iteration scheduler worker: {error}"))
            })?;
        Ok(Self {
            shared: Arc::new(IterationSchedulerShared {
                commands,
                owner_count: AtomicUsize::new(1),
                worker: Mutex::new(Some(worker)),
            }),
        })
    }

    pub(super) fn generate(
        &self,
        request: ScheduledGenerationRequest<'_>,
        mut on_token: impl FnMut(i32) -> OpenAiResult<TokenControl>,
    ) -> OpenAiResult<ScheduledGenerationStats> {
        let started = Instant::now();
        let (reply, events) = std_mpsc::channel();
        self.enqueue_command(SchedulerCommand::Submit(ScheduledRequest {
            id: request.id.to_string(),
            prompt_tokens: request.prompt_tokens.to_vec(),
            max_tokens: request.max_tokens,
            sampling: request.sampling.cloned(),
            chat_sampling_metadata: request.chat_sampling_metadata.map(str::to_string),
            reply,
        }))?;

        let mut first_token_at = None;
        loop {
            if request
                .cancellation
                .is_some_and(openai_frontend::CancellationToken::is_cancelled)
            {
                let _ = self
                    .shared
                    .commands
                    .try_send(SchedulerCommand::Cancel(request.id.to_string()));
                return Err(OpenAiError::backend("request cancelled"));
            }
            match events.recv_timeout(CANCELLATION_POLL_INTERVAL) {
                Ok(SchedulerEvent::Token { token, ack }) => {
                    first_token_at.get_or_insert_with(Instant::now);
                    match on_token(token) {
                        Ok(control) => {
                            let _ = ack.send(control);
                        }
                        Err(error) => {
                            let _ = ack.send(TokenControl::Stop);
                            let _ = self
                                .shared
                                .commands
                                .try_send(SchedulerCommand::Cancel(request.id.to_string()));
                            return Err(error);
                        }
                    }
                }
                Ok(SchedulerEvent::Complete) => {
                    let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
                    let prompt_ms = first_token_at
                        .map(|first| first.duration_since(started).as_secs_f64() * 1_000.0)
                        .unwrap_or(elapsed_ms);
                    return Ok(ScheduledGenerationStats {
                        prompt_ms,
                        predicted_ms: (elapsed_ms - prompt_ms).max(0.0),
                    });
                }
                Ok(SchedulerEvent::Error(error)) => return Err(error),
                Err(std_mpsc::RecvTimeoutError::Timeout) => {}
                Err(std_mpsc::RecvTimeoutError::Disconnected) => {
                    return Err(OpenAiError::backend(
                        "iteration scheduler stopped before generation completed",
                    ));
                }
            }
        }
    }

    pub(super) fn execute_iteration(
        &self,
        session_id: &str,
        token_ids: &[i32],
        positions: &[i32],
        sampling: Option<&SamplingConfig>,
        sample_last: bool,
        phase: IterationBatchPhase,
    ) -> OpenAiResult<SchedulerIterationOutcome> {
        self.execute_direct_iteration(
            session_id,
            None,
            token_ids,
            positions,
            sampling,
            None,
            sample_last,
            phase,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_frame_iteration(
        &self,
        session_id: &str,
        target_token_count: u64,
        token_ids: &[i32],
        positions: &[i32],
        sampling: Option<&SamplingConfig>,
        input: Option<ActivationFrame>,
        sample_last: bool,
    ) -> OpenAiResult<SchedulerIterationOutcome> {
        self.execute_direct_iteration(
            session_id,
            Some(target_token_count),
            token_ids,
            positions,
            sampling,
            input,
            sample_last,
            IterationBatchPhase::Decode,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_direct_iteration(
        &self,
        session_id: &str,
        target_token_count: Option<u64>,
        token_ids: &[i32],
        positions: &[i32],
        sampling: Option<&SamplingConfig>,
        input: Option<ActivationFrame>,
        sample_last: bool,
        phase: IterationBatchPhase,
    ) -> OpenAiResult<SchedulerIterationOutcome> {
        validate_direct_iteration(token_ids, positions)?;
        let (reply, result) = std_mpsc::sync_channel(1);
        self.enqueue_command(SchedulerCommand::ExecuteIteration(DirectIteration {
            session_id: session_id.to_string(),
            target_token_count,
            token_ids: token_ids.to_vec(),
            positions: positions.to_vec(),
            sampling: sampling.cloned(),
            input,
            sample_last,
            phase,
            enqueued_at: Instant::now(),
            reply,
        }))?;
        result.recv().map_err(|error| {
            OpenAiError::backend(format!("iteration scheduler stopped: {error}"))
        })?
    }

    /// Runs an operation on the scheduler worker so speculative verification,
    /// checkpoint repair, and other strategy-specific runtime work cannot race
    /// scheduler-owned mixed iterations.
    pub(crate) fn execute_runtime<T>(
        &self,
        label: &'static str,
        operation: impl FnOnce(&mut RuntimeState) -> OpenAiResult<T> + Send + 'static,
    ) -> OpenAiResult<T>
    where
        T: Send + 'static,
    {
        self.execute_runtime_timed(label, operation)
            .map(|outcome| outcome.value)
    }

    pub(crate) fn execute_runtime_timed<T>(
        &self,
        label: &'static str,
        operation: impl FnOnce(&mut RuntimeState) -> OpenAiResult<T> + Send + 'static,
    ) -> OpenAiResult<SchedulerRuntimeOutcome<T>>
    where
        T: Send + 'static,
    {
        let (operation, result) = runtime_operation(label, operation);
        self.enqueue_command(SchedulerCommand::ExecuteRuntime(operation))?;
        result.recv().map_err(|error| {
            OpenAiError::backend(format!("iteration scheduler stopped: {error}"))
        })?
    }

    /// Queue cache restore/prefill work by stage-local radix affinity while
    /// retaining aging and decode-turn fairness.
    pub(crate) fn execute_cache_aware_runtime_timed<T>(
        &self,
        label: &'static str,
        affinity: CacheAffinity,
        prompt_tokens: Arc<[i32]>,
        priority: u64,
        refresh_affinity: Option<CacheAffinityRefresh>,
        operation: impl FnOnce(&mut RuntimeState) -> OpenAiResult<T> + Send + 'static,
    ) -> OpenAiResult<SchedulerRuntimeOutcome<T>>
    where
        T: Send + 'static,
    {
        let (runtime_operation, result) = runtime_operation(label, operation);
        self.enqueue_command(SchedulerCommand::ExecuteCacheAwareRuntime(
            runtime_operation,
            affinity,
            prompt_tokens,
            priority,
            refresh_affinity,
        ))?;
        result.recv().map_err(|error| {
            OpenAiError::backend(format!("iteration scheduler stopped: {error}"))
        })?
    }

    fn enqueue_command(&self, command: SchedulerCommand) -> OpenAiResult<()> {
        self.shared
            .commands
            .try_send(command)
            .map_err(|error| match error {
                std_mpsc::TrySendError::Full(_) => generation_queue_full_error(),
                std_mpsc::TrySendError::Disconnected(_) => {
                    OpenAiError::backend("iteration scheduler stopped")
                }
            })
    }
}

impl Clone for IterationScheduler {
    fn clone(&self) -> Self {
        self.shared.owner_count.fetch_add(1, Ordering::Relaxed);
        Self {
            shared: Arc::clone(&self.shared),
        }
    }
}

impl Drop for IterationScheduler {
    fn drop(&mut self) {
        if self.shared.owner_count.fetch_sub(1, Ordering::AcqRel) != 1 {
            return;
        }
        let _ = self.shared.commands.send(SchedulerCommand::Shutdown);
        let worker = match self.shared.worker.lock() {
            Ok(mut worker) => worker.take(),
            Err(poisoned) => poisoned.into_inner().take(),
        };
        if let Some(worker) = worker {
            let _ = worker.join();
        }
    }
}

impl SchedulerWorker {
    fn run(mut self) {
        let outcome = catch_unwind(AssertUnwindSafe(|| self.run_loop()));
        if outcome.is_err() {
            let error = OpenAiError::backend("iteration scheduler worker panicked");
            if let Some(telemetry) = self.telemetry.as_ref() {
                telemetry.emit(
                    "stage.scheduler_worker_panic",
                    BTreeMap::from([(
                        "skippy.scheduler.failure_contained".to_string(),
                        json!(true),
                    )]),
                );
            }
            self.fail_all(error.clone());
            self.fail_queued(error);
        }
    }

    fn run_loop(&mut self) {
        loop {
            if self.requests.is_empty()
                && self.direct_iterations.is_empty()
                && self.cache_runtime_queue.is_empty()
            {
                match self.commands.recv() {
                    Ok(SchedulerCommand::Shutdown) | Err(_) => break,
                    Ok(command) => self.handle_command(command),
                }
            }
            for _ in 0..self.max_commands_per_turn {
                match self.commands.try_recv() {
                    Ok(SchedulerCommand::Shutdown) => {
                        self.fail_all(OpenAiError::backend("iteration scheduler stopped"));
                        return;
                    }
                    Ok(command) => self.handle_command(command),
                    Err(std_mpsc::TryRecvError::Empty) => break,
                    Err(std_mpsc::TryRecvError::Disconnected) => return,
                }
            }
            if self.requests.is_empty()
                && self.direct_iterations.is_empty()
                && self.cache_runtime_queue.is_empty()
            {
                continue;
            }
            self.run_work_turn();
            if !self.iteration_interval.is_zero() {
                match self.commands.recv_timeout(self.iteration_interval) {
                    Ok(SchedulerCommand::Shutdown) => {
                        self.fail_all(OpenAiError::backend("iteration scheduler stopped"));
                        return;
                    }
                    Ok(command) => self.handle_command(command),
                    Err(std_mpsc::RecvTimeoutError::Timeout) => {}
                    Err(std_mpsc::RecvTimeoutError::Disconnected) => return,
                }
            }
        }
    }

    fn handle_command(&mut self, command: SchedulerCommand) {
        match command {
            SchedulerCommand::Submit(request) => self.submit(request),
            SchedulerCommand::ExecuteIteration(request) => {
                self.direct_iterations.push_back(request);
            }
            SchedulerCommand::ExecuteRuntime(operation) => {
                self.run_runtime_operation(operation, None);
            }
            SchedulerCommand::ExecuteCacheAwareRuntime(
                operation,
                affinity,
                prompt_tokens,
                priority,
                refresh_affinity,
            ) => {
                self.cache_runtime_queue.enqueue(
                    operation,
                    affinity,
                    prompt_tokens,
                    priority,
                    refresh_affinity,
                );
            }
            SchedulerCommand::Cancel(id) => self.cancel(&id),
            SchedulerCommand::Shutdown => {}
        }
    }

    fn run_work_turn(&mut self) {
        let has_cache_runtime = !self.cache_runtime_queue.is_empty();
        let has_iteration = !self.requests.is_empty() || !self.direct_iterations.is_empty();
        let serve_cache_runtime = should_serve_cache_runtime(
            has_cache_runtime,
            has_iteration,
            self.last_served_cache_runtime,
        );
        self.cache_runtime_queue.advance_turn();
        if serve_cache_runtime {
            self.last_served_cache_runtime = true;
            self.run_cache_runtime_operation();
        } else {
            self.last_served_cache_runtime = false;
            self.run_iteration();
        }
    }

    fn run_cache_runtime_operation(&mut self) {
        let Some((queued, telemetry)) = self.cache_runtime_queue.pop_next() else {
            return;
        };
        self.run_runtime_operation(queued.operation, Some(telemetry));
    }

    fn run_runtime_operation(
        &self,
        operation: RuntimeOperation,
        cache: Option<CacheRuntimeTelemetry>,
    ) {
        let started = Instant::now();
        let label = operation.label;
        (operation.run)(&self.runtime);
        if let Some(telemetry) = self.telemetry.as_ref() {
            let mut attrs = BTreeMap::from([
                ("skippy.scheduler.operation".to_string(), json!(label)),
                (
                    "skippy.scheduler.operation_ms".to_string(),
                    json!(started.elapsed().as_secs_f64() * 1_000.0),
                ),
            ]);
            if let Some(cache) = cache {
                attrs.insert(
                    "skippy.scheduler.cache_matched_tokens".to_string(),
                    json!(cache.matched_tokens),
                );
                attrs.insert(
                    "skippy.scheduler.cache_saved_cost".to_string(),
                    json!(cache.saved_cost),
                );
                attrs.insert(
                    "skippy.scheduler.cache_age_turns".to_string(),
                    json!(cache.age_turns),
                );
                attrs.insert(
                    "skippy.scheduler.cache_stage_hits".to_string(),
                    json!(cache.stage_hits),
                );
                attrs.insert(
                    "skippy.scheduler.cache_epoch".to_string(),
                    json!(cache.cache_epoch),
                );
                attrs.insert(
                    "skippy.scheduler.cache_stale_fallback".to_string(),
                    json!(cache.stale_affinity_fallback),
                );
            }
            telemetry.emit_debug("stage.scheduler_feature_runtime", attrs);
        }
    }

    fn submit(&mut self, request: ScheduledRequest) {
        let admission_tokens = request
            .prompt_tokens
            .len()
            .saturating_add(
                usize::try_from(request.max_tokens)
                    .unwrap_or(usize::MAX)
                    .min(DECODE_BATCH_HEADROOM_TOKENS),
            )
            .min(self.kv_capacity_tokens);
        let sequence = Sequence::new(
            request.id.clone(),
            request.prompt_tokens.clone(),
            request.max_tokens,
            request.sampling.clone(),
            0,
        )
        .with_admission_tokens(admission_tokens);
        if let Err(error) = self.scheduler.submit(sequence) {
            let error = match error {
                AdmissionError::QueueFull { .. } => generation_queue_full_error(),
                AdmissionError::DuplicateSequence(_) | AdmissionError::EmptyPrompt => {
                    OpenAiError::invalid_request(error.to_string())
                }
            };
            let _ = request.reply.send(SchedulerEvent::Error(error));
            return;
        }
        self.requests.insert(
            request.id,
            RequestState {
                reply: request.reply,
                pending_controls: VecDeque::new(),
                sampling: request.sampling,
                chat_sampling_metadata: request.chat_sampling_metadata,
                prompt_token_count: request.prompt_tokens.len(),
                runtime_configured: false,
            },
        );
    }

    fn cancel(&mut self, id: &str) {
        self.scheduler.cancel(id);
        self.drop_runtime_sessions([id]);
        self.requests.remove(id);
    }

    fn fail_request(&mut self, id: &str, error: OpenAiError) {
        self.scheduler.cancel(id);
        self.drop_runtime_sessions([id]);
        if let Some(request) = self.requests.remove(id) {
            let _ = request.reply.send(SchedulerEvent::Error(error));
        }
    }

    fn apply_pending_controls(&mut self) {
        let mut stopped = Vec::new();
        for (id, request) in &mut self.requests {
            while let Some(control) = request.pending_controls.front() {
                match control.try_recv() {
                    Ok(TokenControl::Continue) => {
                        request.pending_controls.pop_front();
                    }
                    Ok(TokenControl::Stop) | Err(std_mpsc::TryRecvError::Disconnected) => {
                        stopped.push(id.clone());
                        break;
                    }
                    Err(std_mpsc::TryRecvError::Empty) => break,
                }
            }
        }
        for id in stopped {
            self.scheduler.cancel(&id);
            self.drop_runtime_sessions([id.as_str()]);
            if let Some(request) = self.requests.remove(&id) {
                let _ = request.reply.send(SchedulerEvent::Complete);
            }
        }
    }

    fn run_iteration(&mut self) {
        self.apply_pending_controls();
        let has_direct = !self.direct_iterations.is_empty();
        let has_planned = !self.requests.is_empty();

        // Decide which queue owns this turn before planning scheduler work:
        // plan_iteration mutates sequence cursors and must only be called when
        // the resulting plan will actually execute.
        let serve_direct = should_serve_direct(has_direct, has_planned, self.last_served_direct);

        if serve_direct {
            self.last_served_direct = true;
            self.run_direct_iteration_batch();
            return;
        }

        let iteration_started = Instant::now();
        let mut plan = self.scheduler.plan_iteration();
        if plan.work.is_empty() {
            return;
        }

        self.last_served_direct = false;
        let mut setup_ids = BTreeSet::new();
        let mut setup = Vec::new();
        for work in &plan.work {
            let Some(request) = self.requests.get_mut(&work.sequence_id) else {
                continue;
            };
            if !request.runtime_configured && setup_ids.insert(work.sequence_id.clone()) {
                setup.push((
                    work.sequence_id.clone(),
                    request.chat_sampling_metadata.clone(),
                    request.sampling.clone(),
                    request.prompt_token_count,
                ));
            }
        }
        let (configured, setup_failures) = match self.prepare_runtime_sessions(&setup) {
            Ok(result) => result,
            Err(error) => {
                self.fail_all(error);
                return;
            }
        };
        for id in configured {
            if let Some(request) = self.requests.get_mut(&id) {
                request.runtime_configured = true;
            }
        }
        for (id, error) in setup_failures {
            self.fail_request(&id, error);
        }
        plan.work
            .retain(|work| self.requests.contains_key(&work.sequence_id));
        plan.token_count = plan.work.iter().map(|work| work.tokens.len()).sum();
        if plan.work.is_empty() {
            return;
        }

        let execution_started = Instant::now();
        let result = self.execute_plan(&plan);
        let execution_elapsed = execution_started.elapsed();
        let predicted = match result {
            Ok(predicted) => predicted,
            Err(error) => {
                self.fail_all(error);
                return;
            }
        };
        self.scheduler
            .observe_iteration_duration(&plan, execution_elapsed);
        let step = self.scheduler.complete_iteration(&plan, &predicted);
        self.finish_iteration(&plan, &predicted);
        self.emit_step_telemetry(&step, iteration_started.elapsed());
    }

    fn run_direct_iteration_batch(&mut self) {
        let batch = take_direct_iteration_batch(
            &mut self.direct_iterations,
            self.max_direct_batch_size,
            MAX_NATIVE_ITERATION_TOKENS,
        );
        debug_assert!(!batch.is_empty(), "validated direct queue must yield work");

        let lock_started = Instant::now();
        let mut runtime = match self.runtime.lock() {
            Ok(runtime) => runtime,
            Err(_) => {
                for request in batch {
                    let _ = request
                        .reply
                        .send(Err(OpenAiError::backend("runtime lock poisoned")));
                }
                return;
            }
        };
        let runtime_lock_wait_ms = lock_started.elapsed().as_secs_f64() * 1_000.0;
        let hold_started = Instant::now();
        let mut runnable = Vec::with_capacity(batch.len());
        for request in batch {
            let alignment = match request.target_token_count {
                Some(target_token_count) => runtime
                    .align_session_to_token_count_if_ahead(&request.session_id, target_token_count),
                None => Ok(None),
            };
            match alignment {
                Ok(alignment) => runnable.push((request, alignment)),
                Err(error) => {
                    let _ = request.reply.send(Err(openai_backend_error(error)));
                }
            }
        }
        if runnable.is_empty() {
            return;
        }
        let batch_size = runnable.len();
        let token_count = runnable
            .iter()
            .map(|(request, _)| request.token_ids.len())
            .sum::<usize>();
        let batch_wait_ms = runnable
            .iter()
            .map(|(request, _)| request.enqueued_at.elapsed().as_secs_f64() * 1_000.0)
            .collect::<Vec<_>>();
        let requests = runnable
            .iter()
            .map(|(request, _)| RuntimeIterationBatchRequest {
                session_id: &request.session_id,
                token_ids: &request.token_ids,
                positions: &request.positions,
                sampling: request.sampling.as_ref(),
                input: request.input.as_ref(),
                sample_last: request.sample_last,
                phase: request.phase,
            })
            .collect::<Vec<_>>();
        let result = runtime
            .iteration_batch_sampled(&requests)
            .map_err(openai_backend_error);
        let runtime_lock_hold_ms = hold_started.elapsed().as_secs_f64() * 1_000.0;
        drop(runtime);
        if let Some(telemetry) = self.telemetry.as_ref() {
            telemetry.emit_debug(
                "stage.scheduler_feature_iteration",
                BTreeMap::from([
                    ("skippy.scheduler.batch_size".to_string(), json!(batch_size)),
                    (
                        "skippy.scheduler.token_count".to_string(),
                        json!(token_count),
                    ),
                    (
                        "skippy.scheduler.batch_wait_max_ms".to_string(),
                        json!(batch_wait_ms.iter().copied().fold(0.0_f64, f64::max)),
                    ),
                    (
                        "skippy.scheduler.runtime_lock_wait_ms".to_string(),
                        json!(runtime_lock_wait_ms),
                    ),
                    (
                        "skippy.scheduler.runtime_lock_hold_ms".to_string(),
                        json!(runtime_lock_hold_ms),
                    ),
                ]),
            );
        }

        match result {
            Ok(outputs) => {
                if outputs.request_outputs.len() != runnable.len() {
                    let error = OpenAiError::backend(format!(
                        "scheduler iteration returned {} outputs for {} requests",
                        outputs.request_outputs.len(),
                        runnable.len()
                    ));
                    for (request, _) in runnable {
                        let _ = request.reply.send(Err(error.clone()));
                    }
                    return;
                }
                let mut predicted = vec![None; runnable.len()];
                for sample in outputs.samples {
                    let Some(slot) = predicted.get_mut(sample.request_index) else {
                        let error = OpenAiError::backend(format!(
                            "scheduler iteration sample references request {}, but only {} requests ran",
                            sample.request_index,
                            runnable.len()
                        ));
                        for (request, _) in runnable {
                            let _ = request.reply.send(Err(error.clone()));
                        }
                        return;
                    };
                    if slot.replace(sample.predicted_token).is_some() {
                        let error = OpenAiError::backend(format!(
                            "scheduler iteration returned duplicate sample for request {}",
                            sample.request_index
                        ));
                        for (request, _) in runnable {
                            let _ = request.reply.send(Err(error.clone()));
                        }
                        return;
                    }
                }
                for ((((request, session_alignment), output), predicted), batch_wait_ms) in runnable
                    .into_iter()
                    .zip(outputs.request_outputs)
                    .zip(predicted)
                    .zip(batch_wait_ms)
                {
                    let _ = request.reply.send(Ok(SchedulerIterationOutcome {
                        predicted: predicted.unwrap_or(-1),
                        output,
                        batch_size,
                        batch_wait_ms,
                        runtime_lock_wait_ms,
                        runtime_lock_hold_ms,
                        session_alignment,
                    }));
                }
            }
            Err(error) => {
                for (request, _) in runnable {
                    let _ = request.reply.send(Err(error.clone()));
                }
            }
        }
    }

    fn finish_iteration(
        &mut self,
        plan: &skippy_scheduler::IterationPlan,
        predicted: &[IterationPrediction],
    ) {
        let mut stopped = BTreeSet::new();
        let mut missing_predictions = BTreeSet::new();
        let predicted_by_work = predicted
            .iter()
            .map(|prediction| (prediction.work_index, prediction.token))
            .collect::<BTreeMap<_, _>>();
        for (index, work) in plan.work.iter().enumerate() {
            if !work.sample_last {
                continue;
            }
            let Some(token) = predicted_by_work.get(&index).copied() else {
                missing_predictions.insert(work.sequence_id.clone());
                continue;
            };
            if token < 0 {
                continue;
            }
            let Some(request) = self.requests.get_mut(&work.sequence_id) else {
                continue;
            };
            let (ack, control) = std_mpsc::sync_channel(1);
            if request
                .reply
                .send(SchedulerEvent::Token { token, ack })
                .is_err()
            {
                stopped.insert(work.sequence_id.clone());
                continue;
            }
            request.pending_controls.push_back(control);
        }

        for id in &missing_predictions {
            self.fail_request(
                id,
                OpenAiError::backend(format!(
                    "scheduler iteration returned no prediction for {id}"
                )),
            );
        }

        for id in &stopped {
            self.scheduler.cancel(id);
        }

        let terminal = self
            .requests
            .keys()
            .filter(|id| self.scheduler.sequence(id).is_none())
            .cloned()
            .collect::<Vec<_>>();
        self.drop_runtime_sessions(terminal.iter().map(String::as_str));
        for id in terminal {
            if let Some(request) = self.requests.remove(&id) {
                let _ = request.reply.send(SchedulerEvent::Complete);
            }
        }
    }

    fn emit_step_telemetry(
        &mut self,
        step: &skippy_scheduler::IterationTelemetry,
        elapsed: Duration,
    ) {
        if self.telemetry.is_none() {
            return;
        }
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
        let telemetry = self
            .telemetry
            .as_ref()
            .expect("telemetry presence checked above");
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
        telemetry.emit_debug("stage.scheduler_iteration", attrs);
    }

    fn prepare_runtime_sessions(
        &self,
        setup: &[(String, Option<String>, Option<SamplingConfig>, usize)],
    ) -> OpenAiResult<RuntimeSetupOutcome> {
        let mut runtime = self
            .runtime
            .lock()
            .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
        let mut configured = Vec::new();
        let mut failures = Vec::new();
        for (id, metadata, sampling, prompt_token_count) in setup {
            let result = runtime.ensure_session_active(id).and_then(|_| {
                if let Some(metadata) = metadata {
                    runtime.configure_chat_sampling(
                        id,
                        metadata,
                        *prompt_token_count as u64,
                        sampling.as_ref(),
                    )
                } else {
                    Ok(())
                }
            });
            match result {
                Ok(()) => configured.push(id.clone()),
                Err(error) => {
                    let _ = runtime.drop_session_timed(id);
                    failures.push((
                        id.clone(),
                        OpenAiError::backend(format!("prepare scheduler session {id}: {error:#}")),
                    ));
                }
            }
        }
        Ok((configured, failures))
    }

    fn execute_plan(
        &self,
        plan: &skippy_scheduler::IterationPlan,
    ) -> OpenAiResult<Vec<IterationPrediction>> {
        let mut runtime = self
            .runtime
            .lock()
            .map_err(|_| OpenAiError::backend("runtime lock poisoned"))?;
        let requests = plan
            .work
            .iter()
            .map(|work| RuntimeIterationBatchRequest {
                session_id: &work.sequence_id,
                token_ids: &work.tokens,
                // Text-only serving leaves positions to the native runtime so
                // model-specific widths (for example Qwen mRoPE's four
                // position dimensions) are expanded correctly.
                positions: &[],
                sampling: work.sampling.as_ref(),
                input: None,
                sample_last: work.sample_last,
                phase: match work.phase {
                    IterationPhase::Decode => IterationBatchPhase::Decode,
                    IterationPhase::Prefill | IterationPhase::Recompute => {
                        IterationBatchPhase::Prefill
                    }
                },
            })
            .collect::<Vec<_>>();
        runtime
            .iteration_batch_sampled(&requests)
            .map(|outputs| {
                outputs
                    .samples
                    .into_iter()
                    .map(|sample| IterationPrediction {
                        work_index: sample.request_index,
                        token: sample.predicted_token,
                    })
                    .collect()
            })
            .map_err(openai_backend_error)
    }

    fn fail_all(&mut self, error: OpenAiError) {
        let ids = self.requests.keys().cloned().collect::<Vec<_>>();
        for id in &ids {
            self.scheduler.cancel(id);
        }
        self.drop_runtime_sessions(ids.iter().map(String::as_str));
        for id in ids {
            if let Some(request) = self.requests.remove(&id) {
                let _ = request.reply.send(SchedulerEvent::Error(error.clone()));
            }
        }
        for request in self.direct_iterations.drain(..) {
            let _ = request.reply.send(Err(error.clone()));
        }
    }

    fn fail_queued(&mut self, error: OpenAiError) {
        while let Ok(command) = self.commands.try_recv() {
            match command {
                SchedulerCommand::Submit(request) => {
                    let _ = request.reply.send(SchedulerEvent::Error(error.clone()));
                }
                SchedulerCommand::ExecuteIteration(request) => {
                    let _ = request.reply.send(Err(error.clone()));
                }
                SchedulerCommand::ExecuteRuntime(_)
                | SchedulerCommand::ExecuteCacheAwareRuntime(_, _, _, _, _)
                | SchedulerCommand::Cancel(_)
                | SchedulerCommand::Shutdown => {}
            }
        }
    }

    fn drop_runtime_sessions<'a>(&self, ids: impl IntoIterator<Item = &'a str>) {
        if let Ok(mut runtime) = self.runtime.lock() {
            for id in ids {
                let _ = runtime.drop_session_timed(id);
            }
        }
    }
}

fn should_serve_direct(has_direct: bool, has_planned: bool, last_served_direct: bool) -> bool {
    if has_direct && has_planned {
        !last_served_direct
    } else {
        has_direct
    }
}

fn scheduler_safe_mode_from_value(value: Option<&str>) -> bool {
    value.is_some_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn take_direct_iteration_batch(
    queue: &mut VecDeque<DirectIteration>,
    max_batch_size: usize,
    mut token_budget: usize,
) -> Vec<DirectIteration> {
    let mut batch = Vec::new();
    let mut batched_sessions = BTreeSet::new();
    let mut deferred = VecDeque::new();
    let queued = queue.len();
    for _ in 0..queued {
        if batch.len() >= max_batch_size {
            break;
        }
        let Some(request) = queue.pop_front() else {
            break;
        };
        if batched_sessions.contains(&request.session_id) {
            deferred.push_back(request);
            continue;
        }
        if request.token_ids.len() > token_budget {
            deferred.push_back(request);
            break;
        }
        token_budget = token_budget.saturating_sub(request.token_ids.len());
        batched_sessions.insert(request.session_id.clone());
        batch.push(request);
        if token_budget == 0 {
            break;
        }
    }
    deferred.append(queue);
    *queue = deferred;
    batch
}

fn validate_direct_iteration(token_ids: &[i32], positions: &[i32]) -> OpenAiResult<()> {
    if token_ids.is_empty() {
        return Err(OpenAiError::invalid_request(
            "scheduler iteration requires at least one token",
        ));
    }
    if token_ids.len() > MAX_NATIVE_ITERATION_TOKENS {
        return Err(OpenAiError::invalid_request(format!(
            "scheduler iteration exceeds the {MAX_NATIVE_ITERATION_TOKENS}-token limit"
        )));
    }
    if !positions.is_empty() && !positions.len().is_multiple_of(token_ids.len()) {
        return Err(OpenAiError::invalid_request(
            "scheduler iteration positions must be empty or token-major",
        ));
    }
    Ok(())
}

fn build_scheduler_config(
    lane_count: usize,
    kv_pool_tokens: usize,
    recurrent_bytes_per_sequence: u64,
    n_batch: Option<u32>,
    n_ubatch: Option<u32>,
    queue_capacity: usize,
) -> SchedulerConfig {
    let max_tokens_per_iteration = usize::try_from(n_batch.unwrap_or(2048))
        .unwrap_or(MAX_NATIVE_ITERATION_TOKENS)
        .clamp(1, MAX_NATIVE_ITERATION_TOKENS);
    let prefill_chunk_tokens = usize::try_from(n_ubatch.unwrap_or(256))
        .unwrap_or(max_tokens_per_iteration)
        .clamp(1, max_tokens_per_iteration);
    let kv_pool_tokens = kv_pool_tokens.max(1);
    let mut memory_components = vec![MemoryComponent {
        name: "kv-cells".to_string(),
        capacity_bytes: u64::try_from(kv_pool_tokens).unwrap_or(u64::MAX),
        resident_bytes: 0,
        bytes_per_token: 1,
        bytes_per_sequence: 0,
    }];
    if recurrent_bytes_per_sequence > 0 {
        // Native recurrent contexts reserve two sequence slots per configured
        // lane and three state planes per native sequence. The per-sequence
        // value already includes the three planes; capacity reflects the
        // native two-slots-per-lane allocation.
        memory_components.push(MemoryComponent {
            name: "recurrent-state".to_string(),
            capacity_bytes: recurrent_bytes_per_sequence
                .saturating_mul(u64::try_from(lane_count).unwrap_or(u64::MAX))
                .saturating_mul(2),
            resident_bytes: 0,
            bytes_per_token: 0,
            bytes_per_sequence: recurrent_bytes_per_sequence,
        });
    }
    SchedulerConfig {
        max_active_sequences: lane_count.max(1),
        reserved_sequence_ids: 0,
        max_waiting_sequences: queue_capacity.max(1),
        max_tokens_per_iteration,
        prefill_chunk_tokens,
        // Recurrent models can become substantially slower when several
        // independent prompt rows share one native prefill call. Preserve
        // decode batching, but serialize recurrent prefills through the same
        // scheduler worker.
        max_prefill_sequences_per_iteration: if recurrent_bytes_per_sequence > 0 {
            1
        } else {
            usize::MAX
        },
        // Retained for scheduler-lab compatibility; production uses the mixed
        // path below when the model does not carry recurrent state.
        max_consecutive_prefill_iterations: 1,
        // A live mixed probe permanently raised subsequent decode-step cost by
        // 9-20% on the Studio54 Qwen3.5 recurrent trace even after admission
        // fell back. Keep phase-homogeneous execution for recurrent models.
        mixed_prefill_decode: recurrent_bytes_per_sequence == 0,
        cache_aging_cost_per_iteration: CACHE_AGING_COST_PER_TURN,
        group_waiting_prefixes: true,
        // Native execution already provides a collection window: requests
        // submitted while one mixed batch is running are drained before the
        // next plan. An additional fixed sleep is pure N=1 latency and makes
        // every generated token pay scheduler overhead.
        iteration_interval: Duration::ZERO,
        memory_components,
    }
    .normalized()
}

fn stage_recurrent_bytes_per_native_sequence(config: &StageConfig) -> u64 {
    let model_path = config
        .materialized_path
        .as_deref()
        .or(config.model_path.as_deref())
        .or(config.source_model_path.as_deref());
    let Some(meta) =
        model_path.and_then(|path| model_artifact::gguf::scan_gguf_compact_meta(Path::new(path)))
    else {
        return 0;
    };
    let start = usize::try_from(config.layer_start).unwrap_or(usize::MAX);
    let end = usize::try_from(config.layer_end).unwrap_or(usize::MAX);
    meta.recurrent_bytes_per_native_sequence_by_layer()
        .get(start..end)
        .unwrap_or_default()
        .iter()
        .copied()
        .fold(0u64, u64::saturating_add)
}

#[cfg(test)]
mod tests {
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
            enqueued_at: Instant::now(),
            reply,
        }
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
            max_commands_per_turn: 8,
            iteration_interval: Duration::ZERO,
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
            reply: reply_a,
        });
        worker.submit(ScheduledRequest {
            id: "b".into(),
            prompt_tokens: vec![3, 4],
            max_tokens: 1,
            sampling: None,
            chat_sampling_metadata: None,
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
        assert!(validate_direct_iteration(&[1], &[]).is_ok());
        assert!(validate_direct_iteration(&[1, 2], &[0, 1]).is_ok());
        assert!(validate_direct_iteration(&[], &[]).is_err());
        assert!(validate_direct_iteration(&[1, 2], &[0, 1, 2]).is_err());
        assert!(validate_direct_iteration(&vec![1; MAX_NATIVE_ITERATION_TOKENS + 1], &[]).is_err());
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
            max_commands_per_turn: 8,
            iteration_interval: Duration::ZERO,
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
                max_commands_per_turn: 8,
                iteration_interval: Duration::ZERO,
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
                max_commands_per_turn: 8,
                iteration_interval: Duration::ZERO,
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
                reply,
            }))
            .unwrap();
        commands
            .send(SchedulerCommand::ExecuteRuntime(RuntimeOperation {
                label: "panic-test",
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
}
