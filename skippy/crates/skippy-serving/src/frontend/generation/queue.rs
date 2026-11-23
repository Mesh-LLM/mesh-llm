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
use skippy_inference_api::InferenceRequestTier;
use skippy_inference_api::InferenceResult;
use skippy_protocol::StageConfig;
use skippy_scheduler::{
    CacheAffinity, CacheAwareCandidate, PriorityPassovers, order_cache_aware_candidates,
};
use std::collections::BTreeMap;
use std::collections::BTreeSet;
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
/// reservation begins. Higher tiers are not held: the bound caps bypasses by
/// the waiter's own and lower tiers.
pub(in crate::frontend) const ADMISSION_STARVATION_BOUND_TURNS: u64 = 32;

pub(in crate::frontend) type GenerationCacheAffinityRefresh =
    Arc<dyn Fn() -> CacheAffinity + Send + Sync>;

#[derive(Clone)]
pub(in crate::frontend) struct GenerationAdmissionScheduling {
    prompt_tokens: Arc<[i32]>,
    refresh_affinity: GenerationCacheAffinityRefresh,
    tier: InferenceRequestTier,
}

impl GenerationAdmissionScheduling {
    pub(in crate::frontend) fn new(
        prompt_tokens: Arc<[i32]>,
        refresh_affinity: GenerationCacheAffinityRefresh,
    ) -> Self {
        Self {
            prompt_tokens,
            refresh_affinity,
            tier: InferenceRequestTier::default(),
        }
    }

    /// Rank this work at `tier`: higher tiers are admitted first, and lower
    /// tiers never take a higher tier's room in the waiting queue.
    pub(in crate::frontend) fn with_tier(mut self, tier: InferenceRequestTier) -> Self {
        self.tier = tier;
        self
    }

    pub(in crate::frontend) fn tier(&self) -> InferenceRequestTier {
        self.tier
    }
}

impl Default for GenerationAdmissionScheduling {
    fn default() -> Self {
        Self {
            prompt_tokens: Arc::from([]),
            refresh_affinity: Arc::new(CacheAffinity::default),
            tier: InferenceRequestTier::default(),
        }
    }
}

#[derive(Clone)]
struct GenerationAdmissionWaiter {
    scheduling: GenerationAdmissionScheduling,
    token_budget_request: GenerationTokenBudgetRequest,
    enqueued_turn: u64,
    order: u64,
    /// [`PriorityPassovers::mark`] when the waiter entered the room.
    passover_mark: u64,
}

/// A waiter as one election sees it.
struct WaiterSnapshot {
    id: u64,
    waiter: GenerationAdmissionWaiter,
    /// How many higher-tier waiters were admitted while it waited.
    passed_over: u64,
}

/// What the anti-starvation head-of-line reservation asks of the next
/// election.
enum SeniorHold {
    /// Hand the lane to this senior waiter, which fits now, so a higher
    /// cache-value newcomer cannot re-occupy the drained pool ahead of it.
    Select(u64),
    /// A senior waiter of this tier does not fit yet: only higher tiers may
    /// pass it, so the pool drains for it rather than letting a steady
    /// stream of smaller requests of its own or a lower tier keep passing it.
    HoldAtOrBelow(InferenceRequestTier),
    None,
}

/// Each tier's oldest waiter (smallest enqueued turn, then smallest order)
/// is the one at risk in that tier. From the highest tier down, the first
/// that has been bypassed `ADMISSION_STARVATION_BOUND_TURNS` times gets the
/// reservation. A higher tier is never held for a lower one: a senior is
/// handed the lane only when no higher-tier waiter fits, and a lower tier's
/// old waiter cannot remove a higher tier's protection. Across tiers, aging
/// protects only a lower-tier waiter that fits: one that does not fit waits
/// until higher-tier load leaves it room, or until its deadline.
fn senior_hold(
    waiters: &[WaiterSnapshot],
    turn: u64,
    token_budget: &GenerationTokenBudget,
) -> InferenceResult<SeniorHold> {
    let mut seniors = BTreeMap::<InferenceRequestTier, &WaiterSnapshot>::new();
    for snapshot in waiters {
        let senior = seniors
            .entry(snapshot.waiter.scheduling.tier)
            .or_insert(snapshot);
        let age =
            |snapshot: &WaiterSnapshot| (snapshot.waiter.enqueued_turn, snapshot.waiter.order);
        if age(snapshot) < age(senior) {
            *senior = snapshot;
        }
    }
    let mut higher_tier_fits = false;
    for (tier, senior) in seniors.into_iter().rev() {
        if turn.saturating_sub(senior.waiter.enqueued_turn) >= ADMISSION_STARVATION_BOUND_TURNS {
            if !token_budget.can_reserve_now(senior.waiter.token_budget_request)? {
                return Ok(SeniorHold::HoldAtOrBelow(tier));
            }
            if !higher_tier_fits {
                return Ok(SeniorHold::Select(senior.id));
            }
        }
        higher_tier_fits =
            higher_tier_fits || tier_has_fitting_waiter(waiters, tier, token_budget)?;
    }
    Ok(SeniorHold::None)
}

fn tier_has_fitting_waiter(
    waiters: &[WaiterSnapshot],
    tier: InferenceRequestTier,
    token_budget: &GenerationTokenBudget,
) -> InferenceResult<bool> {
    for snapshot in waiters {
        if snapshot.waiter.scheduling.tier == tier
            && token_budget.can_reserve_now(snapshot.waiter.token_budget_request)?
        {
            return Ok(true);
        }
    }
    Ok(false)
}

#[derive(Default)]
struct GenerationAdmissionQueueState {
    turn: u64,
    /// Admissions by tier priority, which age the waiters they pass over.
    passovers: PriorityPassovers,
    next_id: u64,
    /// Advances whenever the waiting set changes, so an election run on an
    /// older snapshot is not published.
    revision: u64,
    selected_id: Option<u64>,
    /// The published winner's tier and the waiters that fit when it won.
    selection_basis: Option<SelectionBasis>,
    waiters: BTreeMap<u64, GenerationAdmissionWaiter>,
}

struct SelectionBasis {
    tier: InferenceRequestTier,
    fit: BTreeSet<u64>,
}

/// One election's view of the waiting room.
struct ElectionSnapshot {
    revision: u64,
    turn: u64,
    waiters: Vec<WaiterSnapshot>,
}

impl GenerationAdmissionQueueState {
    /// A published winner is stale once a higher-tier waiter that did not
    /// fit at its election fits now: a release can make room for it.
    fn selection_is_stale(&self, token_budget: &GenerationTokenBudget) -> InferenceResult<bool> {
        let Some(basis) = self.selection_basis.as_ref() else {
            return Ok(false);
        };
        for (id, waiter) in &self.waiters {
            if waiter.scheduling.tier > basis.tier
                && !basis.fit.contains(id)
                && token_budget.can_reserve_now(waiter.token_budget_request)?
            {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn clear_selection(&mut self) {
        self.selected_id = None;
        self.selection_basis = None;
    }

    fn snapshot(&self) -> ElectionSnapshot {
        ElectionSnapshot {
            revision: self.revision,
            turn: self.turn,
            waiters: self
                .waiters
                .iter()
                .map(|(id, waiter)| WaiterSnapshot {
                    id: *id,
                    waiter: waiter.clone(),
                    passed_over: self
                        .passovers
                        .passed_over(waiter.scheduling.tier.priority(), waiter.passover_mark),
                })
                .collect(),
        }
    }
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
        let reservation = tier_has_queue_room(&state, scheduling.tier, generation_queue_limit)
            .then(|| reserve_generation_queue(generation_queue_depth, usize::MAX))
            .flatten()
            .ok_or_else(generation_queue_full_error)?;
        let id = state.next_id;
        state.next_id = state.next_id.saturating_add(1);
        let turn = state.turn;
        // Claiming a lane or entering the waiting room is one queue-locked operation. A lane that
        // becomes free after the failed fast claim cannot be taken by a newer arrival before this
        // waiter is visible.
        state.clear_selection();
        state.revision = state.revision.wrapping_add(1);
        let passover_mark = state.passovers.mark(scheduling.tier.priority());
        state.waiters.insert(
            id,
            GenerationAdmissionWaiter {
                scheduling,
                token_budget_request,
                enqueued_turn: turn,
                order: id,
                passover_mark,
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

    /// The number of waiters and the published winner.
    #[cfg(test)]
    pub(in crate::frontend) fn election_for_test(&self) -> (usize, Option<u64>) {
        let state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        (state.waiters.len(), state.selected_id)
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
        loop {
            // Refreshing affinity consults the prefix cache and can be more
            // expensive than queue bookkeeping. Snapshot under the queue
            // lock, then release it before crossing that subsystem boundary.
            let snapshot = {
                let mut state = self
                    .state
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if let Some(selected_id) = state.selected_id {
                    if !state.selection_is_stale(token_budget)? {
                        return Ok(Some(selected_id));
                    }
                    state.clear_selection();
                }
                state.snapshot()
            };
            let fit = fitting_waiters(&snapshot.waiters, token_budget)?;
            let winner = elect(&snapshot, &fit, token_budget)?;
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            // A waiter that arrived or left during the election was not
            // ranked: elect again from the current set.
            if state.revision != snapshot.revision {
                continue;
            }
            if let Some((id, tier)) = winner {
                state.selected_id = Some(id);
                state.selection_basis = Some(SelectionBasis { tier, fit });
            }
            return Ok(state.selected_id);
        }
    }

    fn remove(&self, id: u64) -> bool {
        let removed = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let removed = state.waiters.remove(&id).is_some();
            if removed {
                state.revision = state.revision.wrapping_add(1);
            }
            if state.selected_id == Some(id) {
                state.clear_selection();
            }
            removed
        };
        if removed {
            self.changed.notify_waiters();
        }
        removed
    }

    /// Take a lane and its tokens for waiter `id` if it is the current
    /// winner. Confirming the election and claiming happen under one lock,
    /// so a waiter that arrives after the election is not passed by it.
    fn claim_if_selected(
        &self,
        id: u64,
        generation_limit: Arc<Semaphore>,
        token_budget: &Arc<GenerationTokenBudget>,
        token_budget_request: GenerationTokenBudgetRequest,
    ) -> InferenceResult<Option<GenerationAdmissionResources>> {
        if self.selected_waiter(token_budget)? != Some(id) {
            return Ok(None);
        }
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.selected_id != Some(id) {
            return Ok(None);
        }
        if state.selection_is_stale(token_budget)? {
            // Elect again, this waiter included.
            state.clear_selection();
            drop(state);
            self.changed.notify_waiters();
            return Ok(None);
        }
        let Some(resources) =
            try_acquire_generation_resources(generation_limit, token_budget, token_budget_request)?
        else {
            return Ok(None);
        };
        Self::complete_selection(&mut state, id);
        drop(state);
        self.changed.notify_waiters();
        Ok(Some(resources))
    }

    fn complete_selection(state: &mut GenerationAdmissionQueueState, id: u64) {
        if let Some(waiter) = state.waiters.remove(&id) {
            state.revision = state.revision.wrapping_add(1);
            state.turn = state.turn.saturating_add(1);
            state
                .passovers
                .record_selection(waiter.scheduling.tier.priority());
        }
        if state.selected_id == Some(id) {
            state.clear_selection();
        }
    }
}

/// The waiters whose token request fits the budget now.
fn fitting_waiters(
    waiters: &[WaiterSnapshot],
    token_budget: &GenerationTokenBudget,
) -> InferenceResult<BTreeSet<u64>> {
    let mut fit = BTreeSet::new();
    for snapshot in waiters {
        if token_budget.can_reserve_now(snapshot.waiter.token_budget_request)? {
            fit.insert(snapshot.id);
        }
    }
    Ok(fit)
}

/// The waiter the next free lane goes to, with its tier.
fn elect(
    snapshot: &ElectionSnapshot,
    fit: &BTreeSet<u64>,
    token_budget: &GenerationTokenBudget,
) -> InferenceResult<Option<(u64, InferenceRequestTier)>> {
    let tier_of = |id: u64| {
        snapshot
            .waiters
            .iter()
            .find(|candidate| candidate.id == id)
            .map(|candidate| (id, candidate.waiter.scheduling.tier))
    };
    let held_for = match senior_hold(&snapshot.waiters, snapshot.turn, token_budget)? {
        SeniorHold::Select(senior_id) => return Ok(tier_of(senior_id)),
        SeniorHold::HoldAtOrBelow(tier) => Some(tier),
        SeniorHold::None => None,
    };
    let eligible_waiters = snapshot
        .waiters
        .iter()
        .filter(|candidate| {
            fit.contains(&candidate.id)
                && held_for.is_none_or(|tier| candidate.waiter.scheduling.tier > tier)
        })
        .collect::<Vec<_>>();
    let affinities = eligible_waiters
        .iter()
        .map(|candidate| (candidate.waiter.scheduling.refresh_affinity)())
        .collect::<Vec<_>>();
    let winner = order_cache_aware_candidates(
        eligible_waiters
            .iter()
            .zip(affinities.iter())
            .enumerate()
            .map(|(index, (candidate, affinity))| CacheAwareCandidate {
                index,
                priority: candidate.waiter.scheduling.tier.priority(),
                affinity,
                prompt_tokens: &candidate.waiter.scheduling.prompt_tokens,
                enqueued_turn: candidate.waiter.enqueued_turn,
                order: candidate.waiter.order,
                passed_over: candidate.passed_over,
            }),
        snapshot.turn,
        ADMISSION_CACHE_AGING_COST_PER_TURN,
        true,
    )
    .first()
    .and_then(|index| eligible_waiters.get(*index).map(|candidate| candidate.id));
    Ok(winner.and_then(tier_of))
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
            if let Some(resources) = self.queue.claim_if_selected(
                self.id,
                generation_limit.clone(),
                &self.token_budget,
                self.token_budget_request,
            )? {
                self.reservation.take();
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

/// Whether a request at `tier` may join the waiting room. Only waiters at the
/// same or a higher tier count against the limit, so lower-tier traffic cannot
/// fill the room ahead of a higher tier. The room holds at most one limit per
/// tier.
fn tier_has_queue_room(
    state: &GenerationAdmissionQueueState,
    tier: InferenceRequestTier,
    generation_queue_limit: usize,
) -> bool {
    state
        .waiters
        .values()
        .filter(|waiter| waiter.scheduling.tier >= tier)
        .count()
        < generation_queue_limit
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
    // The one-token warmup above does not reserve the graphs real traffic
    // uses: the first full prefill ubatch and the first concurrent decode
    // wave each grow the reservation mid-request. Warm the serving shapes
    // now so no request pays the graph growth.
    let warmup_prefill_tokens = config
        .n_ubatch
        .or(config.n_batch)
        .unwrap_or(512)
        .min(config.ctx_size.saturating_sub(64)) as usize;
    runtime.warmup_serving_shapes(warmup_prefill_tokens)?;
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
            "{flag_name} ({generation_concurrency}) cannot exceed effective lane_count ({lane_count})"
        );
    }
    Ok(())
}

/// Lane bound OpenAI generation admission must measure against.
///
/// The loaded runtime's lane count is the effective bound. Native serializes
/// encoder-decoder work onto one lane, so `RuntimeState::lane_count()` can be
/// below `StageConfig::lane_count`; admission measured against the configured
/// value starts a frontend that queues generations the runtime can never serve.
pub(in crate::frontend) fn generation_admission_lane_count(
    runtime: &Arc<Mutex<RuntimeState>>,
) -> u32 {
    lock_runtime(runtime).lane_count()
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
