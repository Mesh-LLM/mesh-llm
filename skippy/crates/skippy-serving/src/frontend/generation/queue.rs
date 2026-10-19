use crate::frontend::admission::GenerationTokenBudget;
use crate::frontend::admission::GenerationTokenBudgetRequest;
use crate::frontend::admission::GenerationTokenReservation;
use crate::frontend::generation::CONTEXT_BUDGET_MAX_TOKENS;
use crate::frontend::generation::GENERATION_RETRY_AFTER_SECS;
use crate::frontend::generation::PhaseTimer;
use crate::frontend::util::context_budget_completion_tokens;
use crate::runtime_state::RuntimeState;
use crate::runtime_state::panic_recovery::lock_runtime;
use crate::telemetry::Telemetry;
use crate::telemetry::lifecycle_attrs;
use crate::telemetry::now_unix_nanos;
use anyhow::Result;
use anyhow::bail;
use axum::http::StatusCode;
use serde_json::json;
use skippy_inference_api::CancellationToken;
use skippy_inference_api::InferenceError;
use skippy_inference_api::InferenceErrorKind;
use skippy_inference_api::InferenceResult;
use skippy_protocol::StageConfig;
use skippy_scheduler::{CacheAffinity, CacheAwareCandidate, order_cache_aware_candidates};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Duration;
use std::time::Instant;
use tokio::sync::Notify;
use tokio::sync::OwnedSemaphorePermit;
use tokio::sync::Semaphore;

const ADMISSION_CACHE_AGING_COST_PER_TURN: u64 = 4_096;

/// How many admissions the oldest waiter may be bypassed before the queue stops
/// admitting anyone else and reserves the pool for it.
///
/// `selected_waiter` normally ranks only waiters that fit the *currently* free
/// KV budget, so a request larger than the free budget is never a candidate. A
/// continuous stream of smaller requests that each fit can therefore bypass a
/// large waiter forever — unbounded when the admission timeout is zero and there
/// is no deadline to break the wait. Once the oldest waiter has been passed over
/// this many turns (each turn is one completed admission) and still cannot fit,
/// admission is held so active reservations drain toward it. Every queued
/// request fits an empty pool (`claim_or_enqueue` rejects impossible requests up
/// front), so draining always makes progress. The bound trades a little
/// throughput and cache locality for a hard ceiling on bypasses before drain
/// reservation begins.
pub(in crate::frontend) const ADMISSION_STARVATION_BOUND_TURNS: u64 = 32;

pub(in crate::frontend) type GenerationCacheAffinityRefresh =
    Arc<dyn Fn() -> CacheAffinity + Send + Sync>;

#[derive(Clone)]
pub(in crate::frontend) struct GenerationAdmissionScheduling {
    prompt_tokens: Arc<[i32]>,
    refresh_affinity: GenerationCacheAffinityRefresh,
}

impl GenerationAdmissionScheduling {
    pub(in crate::frontend) fn new(
        prompt_tokens: Arc<[i32]>,
        refresh_affinity: GenerationCacheAffinityRefresh,
    ) -> Self {
        Self {
            prompt_tokens,
            refresh_affinity,
        }
    }
}

impl Default for GenerationAdmissionScheduling {
    fn default() -> Self {
        Self {
            prompt_tokens: Arc::from([]),
            refresh_affinity: Arc::new(CacheAffinity::default),
        }
    }
}

#[derive(Clone)]
struct GenerationAdmissionWaiter {
    scheduling: GenerationAdmissionScheduling,
    token_budget_request: GenerationTokenBudgetRequest,
    enqueued_turn: u64,
    order: u64,
}

#[derive(Default)]
struct GenerationAdmissionQueueState {
    turn: u64,
    next_id: u64,
    selected_id: Option<u64>,
    waiters: BTreeMap<u64, GenerationAdmissionWaiter>,
}

/// Scheduler-owned waiting room in front of the finite native lane pool.
///
/// Tokenized prompts enter this queue before acquiring a generation lane, so
/// cache affinity and waiting-prefix locality remain visible across the whole
/// offered backlog instead of only the currently running tranche.
pub(in crate::frontend) struct GenerationAdmissionQueue {
    state: Mutex<GenerationAdmissionQueueState>,
    election: Mutex<()>,
    changed: Notify,
}

impl GenerationAdmissionQueue {
    pub(in crate::frontend) fn new() -> Self {
        Self {
            state: Mutex::new(GenerationAdmissionQueueState::default()),
            election: Mutex::new(()),
            changed: Notify::new(),
        }
    }

    pub(in crate::frontend) fn claim_or_enqueue(
        self: &Arc<Self>,
        generation_limit: Arc<Semaphore>,
        token_budget: Arc<GenerationTokenBudget>,
        token_budget_request: GenerationTokenBudgetRequest,
        scheduling: GenerationAdmissionScheduling,
        generation_queue_depth: Arc<AtomicUsize>,
        generation_queue_limit: usize,
    ) -> InferenceResult<GenerationAdmissionClaim> {
        // Validate an impossible request before queue capacity or current load
        // can disguise it as transient overload.
        let _ = token_budget.can_reserve_now(token_budget_request)?;
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.waiters.is_empty()
            && let Some(resources) = try_acquire_generation_resources(
                generation_limit.clone(),
                &token_budget,
                token_budget_request,
            )?
        {
            return Ok(GenerationAdmissionClaim::Acquired(resources));
        }
        let reservation = reserve_generation_queue(generation_queue_depth, generation_queue_limit)
            .ok_or_else(generation_queue_full_error)?;
        let id = state.next_id;
        state.next_id = state.next_id.saturating_add(1);
        let turn = state.turn;
        // Claiming a lane or entering the waiting room is one queue-locked operation. A lane that
        // becomes free after the failed fast claim cannot be taken by a newer arrival before this
        // waiter is visible.
        state.selected_id = None;
        state.waiters.insert(
            id,
            GenerationAdmissionWaiter {
                scheduling,
                token_budget_request,
                enqueued_turn: turn,
                order: id,
            },
        );
        drop(state);
        self.changed.notify_waiters();
        Ok(GenerationAdmissionClaim::Queued(
            GenerationAdmissionQueueLease {
                queue: Arc::clone(self),
                token_budget,
                token_budget_request,
                id,
                reservation: Some(reservation),
            },
        ))
    }

    pub(in crate::frontend) fn notify_lane_available(&self) {
        self.changed.notify_waiters();
    }

    fn selected_waiter(
        &self,
        token_budget: &GenerationTokenBudget,
    ) -> InferenceResult<Option<u64>> {
        // Notify wakes the whole waiting set. Serialize and memoize one
        // election per available lane so N waiters do not each refresh and
        // sort the same N affinities.
        let _election = self
            .election
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // Refreshing affinity consults the prefix cache and can be more
        // expensive than queue bookkeeping. Snapshot under the queue lock,
        // then release it before crossing that subsystem boundary.
        let (turn, waiters) = {
            let state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(selected_id) = state.selected_id {
                return Ok(Some(selected_id));
            }
            (
                state.turn,
                state
                    .waiters
                    .iter()
                    .map(|(id, waiter)| (*id, waiter.clone()))
                    .collect::<Vec<_>>(),
            )
        };
        // Anti-starvation head-of-line reservation. The oldest waiter (smallest
        // enqueued turn, then smallest order) is the one at risk: once it has
        // been bypassed ADMISSION_STARVATION_BOUND_TURNS times, reserve the pool
        // for it instead of letting a steady stream of smaller, currently-fitting
        // requests keep passing it. If it now fits, hand it the lane directly so a
        // higher cache-value newcomer cannot re-occupy the drained pool ahead of
        // it; otherwise select nobody so active reservations keep draining.
        if let Some((senior_id, senior)) = waiters
            .iter()
            .min_by_key(|(_, waiter)| (waiter.enqueued_turn, waiter.order))
            && turn.saturating_sub(senior.enqueued_turn) >= ADMISSION_STARVATION_BOUND_TURNS
        {
            let senior_id = *senior_id;
            if !token_budget.can_reserve_now(senior.token_budget_request)? {
                return Ok(None);
            }
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if state.selected_id.is_none() && state.waiters.contains_key(&senior_id) {
                state.selected_id = Some(senior_id);
            }
            return Ok(state.selected_id);
        }
        let mut eligible_waiters = Vec::with_capacity(waiters.len());
        for (id, waiter) in waiters {
            if token_budget.can_reserve_now(waiter.token_budget_request)? {
                eligible_waiters.push((id, waiter));
            }
        }
        let affinities = eligible_waiters
            .iter()
            .map(|(_, waiter)| (waiter.scheduling.refresh_affinity)())
            .collect::<Vec<_>>();
        let selected_id = order_cache_aware_candidates(
            eligible_waiters
                .iter()
                .map(|(_, waiter)| waiter)
                .zip(affinities.iter())
                .enumerate()
                .map(|(index, (waiter, affinity))| CacheAwareCandidate {
                    index,
                    priority: 0,
                    affinity,
                    prompt_tokens: &waiter.scheduling.prompt_tokens,
                    enqueued_turn: waiter.enqueued_turn,
                    order: waiter.order,
                }),
            turn,
            ADMISSION_CACHE_AGING_COST_PER_TURN,
            true,
        )
        .first()
        .and_then(|index| eligible_waiters.get(*index).map(|(id, _)| id))
        .copied();
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.selected_id.is_none()
            && selected_id.is_some_and(|id| state.waiters.contains_key(&id))
        {
            state.selected_id = selected_id;
        }
        Ok(state.selected_id)
    }

    fn remove(&self, id: u64) -> bool {
        let removed = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let removed = state.waiters.remove(&id).is_some();
            if state.selected_id == Some(id) {
                state.selected_id = None;
            }
            removed
        };
        if removed {
            self.changed.notify_waiters();
        }
        removed
    }

    fn complete_selection(&self, id: u64) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.waiters.remove(&id).is_some() {
            state.turn = state.turn.saturating_add(1);
        }
        if state.selected_id == Some(id) {
            state.selected_id = None;
        }
    }
}

pub(in crate::frontend) enum GenerationAdmissionClaim {
    Acquired(GenerationAdmissionResources),
    Queued(GenerationAdmissionQueueLease),
}

pub(in crate::frontend) struct GenerationAdmissionResources {
    pub(in crate::frontend) lane: OwnedSemaphorePermit,
    pub(in crate::frontend) token_budget: GenerationTokenReservation,
}

pub(in crate::frontend) struct GenerationAdmissionQueueLease {
    queue: Arc<GenerationAdmissionQueue>,
    token_budget: Arc<GenerationTokenBudget>,
    token_budget_request: GenerationTokenBudgetRequest,
    id: u64,
    reservation: Option<GenerationQueueReservation>,
}

impl GenerationAdmissionQueueLease {
    pub(in crate::frontend) async fn acquire(
        mut self,
        generation_limit: Arc<Semaphore>,
        admission_timeout: Duration,
        deadline: Option<Instant>,
        cancellation: &CancellationToken,
    ) -> InferenceResult<GenerationAdmissionResources> {
        loop {
            if cancellation.is_cancelled() {
                return Err(InferenceError::cancelled("request cancelled"));
            }
            if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                return Err(generation_queue_timeout_error(admission_timeout));
            }
            let notified = self.queue.changed.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            let token_released = self.token_budget.release_notification().notified();
            tokio::pin!(token_released);
            token_released.as_mut().enable();
            if self.queue.selected_waiter(&self.token_budget)? == Some(self.id)
                && let Some(resources) = try_acquire_generation_resources(
                    generation_limit.clone(),
                    &self.token_budget,
                    self.token_budget_request,
                )?
            {
                self.queue.complete_selection(self.id);
                self.reservation.take();
                self.queue.changed.notify_waiters();
                return Ok(resources);
            }
            if let Some(deadline) = deadline {
                let timeout = tokio::time::sleep_until(tokio::time::Instant::from_std(deadline));
                tokio::select! {
                    () = &mut notified => {}
                    () = &mut token_released => {}
                    () = timeout => return Err(generation_queue_timeout_error(admission_timeout)),
                    () = cancellation.cancelled() => {
                        return Err(InferenceError::cancelled("request cancelled"));
                    }
                }
            } else {
                tokio::select! {
                    () = &mut notified => {}
                    () = &mut token_released => {}
                    () = cancellation.cancelled() => {
                        return Err(InferenceError::cancelled("request cancelled"));
                    }
                }
            }
        }
    }
}

fn try_acquire_generation_resources(
    generation_limit: Arc<Semaphore>,
    token_budget: &Arc<GenerationTokenBudget>,
    token_budget_request: GenerationTokenBudgetRequest,
) -> InferenceResult<Option<GenerationAdmissionResources>> {
    let lane = match generation_limit.try_acquire_owned() {
        Ok(lane) => lane,
        Err(tokio::sync::TryAcquireError::NoPermits) => return Ok(None),
        Err(tokio::sync::TryAcquireError::Closed) => {
            return Err(generation_lanes_busy_error());
        }
    };
    let Some(token_budget) = token_budget.try_reserve(token_budget_request)? else {
        // This is the raw semaphore permit rather than the controller wrapper,
        // so releasing an unsuccessful partial claim does not wake the same
        // waiter into a hot loop. A real KV release provides the wakeup.
        drop(lane);
        return Ok(None);
    };
    Ok(Some(GenerationAdmissionResources { lane, token_budget }))
}

impl Drop for GenerationAdmissionQueueLease {
    fn drop(&mut self) {
        if self.reservation.is_some() {
            self.queue.remove(self.id);
        }
    }
}

#[cfg(test)]
use tokio::sync::TryAcquireError;

pub(in crate::frontend) struct GenerationQueueReservation {
    pub(in crate::frontend) depth: Arc<AtomicUsize>,
}

impl Drop for GenerationQueueReservation {
    fn drop(&mut self) {
        self.depth.fetch_sub(1, Ordering::AcqRel);
    }
}

#[cfg(test)]
pub(in crate::frontend) async fn acquire_generation_permit_with_queue_reservation(
    generation_limit: Arc<Semaphore>,
    reservation: GenerationQueueReservation,
    admission_timeout: Duration,
    deadline: Instant,
    cancellation: &CancellationToken,
) -> InferenceResult<OwnedSemaphorePermit> {
    if cancellation.is_cancelled() {
        return Err(InferenceError::cancelled("request cancelled"));
    }

    let timeout = tokio::time::timeout_at(
        tokio::time::Instant::from_std(deadline),
        generation_limit.acquire_owned(),
    );
    tokio::select! {
        result = timeout => {
            drop(reservation);
            match result {
                Ok(Ok(permit)) if cancellation.is_cancelled() => {
                    drop(permit);
                    Err(InferenceError::cancelled("request cancelled"))
                }
                Ok(Ok(permit)) => Ok(permit),
                Ok(Err(_)) => Err(generation_lanes_busy_error()),
                Err(_) => Err(generation_queue_timeout_error(admission_timeout)),
            }
        }
        () = cancellation.cancelled() => {
            drop(reservation);
            Err(InferenceError::cancelled("request cancelled"))
        }
    }
}

#[cfg(test)]
pub(in crate::frontend) async fn acquire_generation_permit_with_queue(
    generation_limit: Arc<Semaphore>,
    generation_queue_depth: Arc<AtomicUsize>,
    generation_queue_limit: usize,
    admission_timeout: Duration,
) -> InferenceResult<OwnedSemaphorePermit> {
    let deadline = Instant::now() + admission_timeout;
    match generation_limit.clone().try_acquire_owned() {
        Ok(permit) => return Ok(permit),
        Err(TryAcquireError::Closed) => return Err(generation_lanes_busy_error()),
        Err(TryAcquireError::NoPermits) => {}
    }

    let queue_reservation =
        reserve_generation_queue(generation_queue_depth, generation_queue_limit)
            .ok_or_else(generation_queue_full_error)?;
    let cancellation = CancellationToken::new();
    acquire_generation_permit_with_queue_reservation(
        generation_limit,
        queue_reservation,
        admission_timeout,
        deadline,
        &cancellation,
    )
    .await
}

pub(in crate::frontend) fn reserve_generation_queue(
    generation_queue_depth: Arc<AtomicUsize>,
    generation_queue_limit: usize,
) -> Option<GenerationQueueReservation> {
    let mut current = generation_queue_depth.load(Ordering::Acquire);
    loop {
        if current >= generation_queue_limit {
            return None;
        }
        match generation_queue_depth.compare_exchange_weak(
            current,
            current + 1,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => {
                return Some(GenerationQueueReservation {
                    depth: generation_queue_depth,
                });
            }
            Err(next) => current = next,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::frontend) enum GenerationTokenLimit {
    /// Client sent a concrete `max_tokens`. Must fit in the context
    /// window; otherwise return a context_length_exceeded error so the
    /// client knows their request couldn't be honored as-asked.
    Explicit(u32),
    /// Caller didn't send `max_tokens`, but the server has a configured
    /// default cap. Clamp down to whatever fits in the remaining
    /// context budget rather than rejecting — the client didn't ask
    /// for the specific number, the server picked it.
    Default(u32),
    /// Caller didn't send `max_tokens` and the server is configured
    /// with [`CONTEXT_BUDGET_MAX_TOKENS`] (opt-in unbounded). Use the
    /// entire remaining context window.
    ContextBudget,
}

impl GenerationTokenLimit {
    pub(in crate::frontend) fn from_request(
        requested: Option<u32>,
        default_max_tokens: u32,
    ) -> Self {
        match requested {
            Some(max_tokens) => Self::Explicit(max_tokens),
            None if default_max_tokens == CONTEXT_BUDGET_MAX_TOKENS => Self::ContextBudget,
            None => Self::Default(default_max_tokens),
        }
    }

    pub(in crate::frontend) fn resolve(
        self,
        prompt_token_count: usize,
        ctx_size: usize,
    ) -> InferenceResult<u32> {
        match self {
            Self::Explicit(max_tokens) => {
                // Client-asserted ceiling. `max_tokens` is an upper bound on
                // the reply, not a reservation the server must be able to
                // honour in full: OpenAI-compatible clients routinely send a
                // very large value to mean "no limit" (Buzz Agent Desktop
                // defaults to 65,536). Clamp to the remaining context and let
                // generation stop with `finish_reason: "length"`. A prompt that
                // overflows the window on its own is still a real error.
                //
                // Rejecting a ceiling that wouldn't fit made routing and the
                // backend disagree about a request the server could serve
                // comfortably — issue #1350.
                let remaining = context_budget_completion_tokens(prompt_token_count, ctx_size)?;
                Ok(remaining.min(max_tokens))
            }
            Self::Default(default_max_tokens) => {
                // Server-picked default. Always clamp to the remaining
                // context budget. If the prompt already exceeds the
                // window, surface that as a real error — but never
                // reject just because our default wouldn't fit.
                let remaining = context_budget_completion_tokens(prompt_token_count, ctx_size)?;
                Ok(remaining.min(default_max_tokens))
            }
            Self::ContextBudget => context_budget_completion_tokens(prompt_token_count, ctx_size),
        }
    }
}

pub(in crate::frontend) fn prewarm_generation_sessions(
    runtime: &Arc<Mutex<RuntimeState>>,
    generation_concurrency: usize,
    telemetry: &Telemetry,
    config: &StageConfig,
    event_name: &'static str,
) -> Result<()> {
    let timer = PhaseTimer::start();
    let mut runtime = lock_runtime(runtime);
    let generation_graph_warmed = runtime.warmup_generation_graph()?;
    let sessions = runtime.prewarm_idle_sessions(generation_concurrency)?;
    let mut attrs = lifecycle_attrs(config);
    attrs.insert(
        "llama_stage.generation_concurrency".to_string(),
        json!(generation_concurrency),
    );
    attrs.insert(
        "llama_stage.lane_count".to_string(),
        json!(sessions.lane_count),
    );
    attrs.insert(
        "llama_stage.runtime_sessions_active".to_string(),
        json!(sessions.active_sessions),
    );
    attrs.insert(
        "llama_stage.runtime_sessions_idle".to_string(),
        json!(sessions.idle_sessions),
    );
    attrs.insert(
        "llama_stage.generation_graph_warmed".to_string(),
        json!(generation_graph_warmed),
    );
    attrs.insert(
        "llama_stage.elapsed_ms".to_string(),
        json!(timer.elapsed_ms()),
    );
    telemetry.emit_span(
        event_name,
        attrs,
        timer.start_unix_nanos,
        now_unix_nanos() as u64,
    );
    Ok(())
}

pub(in crate::frontend) fn ensure_generation_concurrency_fits_lanes(
    generation_concurrency: usize,
    lane_count: u32,
    flag_name: &str,
) -> Result<()> {
    let lane_count = usize::try_from(lane_count).unwrap_or(usize::MAX);
    if generation_concurrency > lane_count {
        bail!(
            "{flag_name} ({generation_concurrency}) cannot exceed configured lane_count ({lane_count})"
        );
    }
    Ok(())
}

pub(in crate::frontend) fn generation_lanes_busy_error() -> InferenceError {
    InferenceError::from_kind(
        StatusCode::TOO_MANY_REQUESTS,
        InferenceErrorKind::RateLimit,
        "all execution lanes are busy",
    )
    .with_retry_after_secs(GENERATION_RETRY_AFTER_SECS)
}

pub(in crate::frontend) fn generation_queue_full_error() -> InferenceError {
    InferenceError::from_kind(
        StatusCode::TOO_MANY_REQUESTS,
        InferenceErrorKind::RateLimit,
        "generation queue is full; retry later",
    )
    .with_retry_after_secs(GENERATION_RETRY_AFTER_SECS)
}

pub(in crate::frontend) fn generation_queue_timeout_error(timeout: Duration) -> InferenceError {
    InferenceError::from_kind(
        StatusCode::TOO_MANY_REQUESTS,
        InferenceErrorKind::RateLimit,
        format!(
            "timed out waiting for an execution lane after {} seconds",
            timeout.as_secs()
        ),
    )
    .with_retry_after_secs(GENERATION_RETRY_AFTER_SECS)
}
