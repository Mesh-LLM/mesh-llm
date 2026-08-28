use crate::frontend::generation::CONTEXT_BUDGET_MAX_TOKENS;
use crate::frontend::generation::GENERATION_RETRY_AFTER_SECS;
use crate::frontend::generation::PhaseTimer;
use crate::frontend::util::context_budget_completion_tokens;
use crate::runtime_state::RuntimeState;
use crate::telemetry::Telemetry;
use crate::telemetry::lifecycle_attrs;
use crate::telemetry::now_unix_nanos;
use anyhow::Result;
use anyhow::anyhow;
use anyhow::bail;
use axum::http::StatusCode;
use openai_frontend::CancellationToken;
use openai_frontend::OpenAiError;
use openai_frontend::OpenAiErrorKind;
use openai_frontend::OpenAiResult;
use serde_json::json;
use skippy_protocol::StageConfig;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Duration;
use std::time::Instant;
use tokio::sync::OwnedSemaphorePermit;
use tokio::sync::Semaphore;

const SERVICE_RATE_EWMA_ALPHA: f64 = 0.25;
const SERVICE_RATE_WINDOW: usize = 64;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(in crate::frontend) struct GenerationAdmissionWork {
    pub(in crate::frontend) prompt_tokens: u64,
    pub(in crate::frontend) decode_tokens: u64,
}

impl GenerationAdmissionWork {
    pub(in crate::frontend) fn new(prompt_tokens: usize, decode_tokens: u32) -> Self {
        Self {
            prompt_tokens: u64::try_from(prompt_tokens).unwrap_or(u64::MAX),
            decode_tokens: u64::from(decode_tokens),
        }
    }
}

#[derive(Default)]
struct GenerationServiceState {
    active: GenerationAdmissionWork,
    queued: GenerationAdmissionWork,
    prefill_ms_per_token_ewma: Option<f64>,
    decode_ms_per_token_ewma: Option<f64>,
    prefill_ms_per_token_samples: VecDeque<f64>,
    decode_ms_per_token_samples: VecDeque<f64>,
}

pub(in crate::frontend) struct GenerationServiceEstimator {
    concurrency: usize,
    state: Mutex<GenerationServiceState>,
}

impl GenerationServiceEstimator {
    pub(in crate::frontend) fn new(concurrency: usize) -> Self {
        Self {
            concurrency: concurrency.max(1),
            state: Mutex::new(GenerationServiceState::default()),
        }
    }

    pub(in crate::frontend) fn predicted_wait_ms(&self) -> Option<f64> {
        let state = self.state.lock().ok()?;
        predicted_wait_ms_for_state(&state, self.concurrency)
    }

    pub(in crate::frontend) fn reserve_queued(
        self: &Arc<Self>,
        work: GenerationAdmissionWork,
        admission_timeout: Duration,
    ) -> Result<GenerationQueuedWorkReservation, f64> {
        let Ok(mut state) = self.state.lock() else {
            return Ok(GenerationQueuedWorkReservation {
                estimator: self.clone(),
                work,
                queued: false,
            });
        };
        if let Some(wait_ms) = predicted_wait_ms_for_state(&state, self.concurrency)
            && wait_ms > admission_timeout.as_secs_f64() * 1_000.0
        {
            return Err(wait_ms);
        }
        state.queued = add_work(state.queued, work);
        Ok(GenerationQueuedWorkReservation {
            estimator: self.clone(),
            work,
            queued: true,
        })
    }

    pub(in crate::frontend) fn start_active(
        self: &Arc<Self>,
        work: GenerationAdmissionWork,
    ) -> GenerationActiveWorkReservation {
        if let Ok(mut state) = self.state.lock() {
            state.active = add_work(state.active, work);
        }
        GenerationActiveWorkReservation {
            estimator: self.clone(),
            work,
        }
    }

    pub(in crate::frontend) fn observe_completed(
        &self,
        work: GenerationAdmissionWork,
        prompt_ms: f64,
        decode_ms: f64,
    ) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        if work.prompt_tokens > 0 && prompt_ms.is_finite() && prompt_ms > 0.0 {
            let sample = prompt_ms / work.prompt_tokens as f64;
            state.prefill_ms_per_token_ewma =
                Some(update_ewma(state.prefill_ms_per_token_ewma, sample));
            push_sample(&mut state.prefill_ms_per_token_samples, sample);
        }
        if work.decode_tokens > 0 && decode_ms.is_finite() && decode_ms > 0.0 {
            let sample = decode_ms / work.decode_tokens as f64;
            state.decode_ms_per_token_ewma =
                Some(update_ewma(state.decode_ms_per_token_ewma, sample));
            push_sample(&mut state.decode_ms_per_token_samples, sample);
        }
    }
}

fn predicted_wait_ms_for_state(state: &GenerationServiceState, concurrency: usize) -> Option<f64> {
    let prompt_ms_per_token = conservative_ms_per_token(
        state.prefill_ms_per_token_ewma,
        &state.prefill_ms_per_token_samples,
    );
    let decode_ms_per_token = conservative_ms_per_token(
        state.decode_ms_per_token_ewma,
        &state.decode_ms_per_token_samples,
    );
    if prompt_ms_per_token.is_none() && decode_ms_per_token.is_none() {
        return None;
    }
    let work = add_work(state.active, state.queued);
    let prompt_ms = prompt_ms_per_token.unwrap_or(0.0) * work.prompt_tokens as f64;
    let decode_ms = decode_ms_per_token.unwrap_or(0.0) * work.decode_tokens as f64;
    Some((prompt_ms + decode_ms) / concurrency.max(1) as f64)
}

pub(in crate::frontend) struct GenerationQueuedWorkReservation {
    estimator: Arc<GenerationServiceEstimator>,
    work: GenerationAdmissionWork,
    queued: bool,
}

impl GenerationQueuedWorkReservation {
    pub(in crate::frontend) fn promote(mut self) -> GenerationActiveWorkReservation {
        if let Ok(mut state) = self.estimator.state.lock() {
            state.queued = subtract_work(state.queued, self.work);
            state.active = add_work(state.active, self.work);
        }
        self.queued = false;
        GenerationActiveWorkReservation {
            estimator: self.estimator.clone(),
            work: self.work,
        }
    }
}

impl Drop for GenerationQueuedWorkReservation {
    fn drop(&mut self) {
        if self.queued
            && let Ok(mut state) = self.estimator.state.lock()
        {
            state.queued = subtract_work(state.queued, self.work);
        }
    }
}

pub(in crate::frontend) struct GenerationActiveWorkReservation {
    estimator: Arc<GenerationServiceEstimator>,
    work: GenerationAdmissionWork,
}

impl Drop for GenerationActiveWorkReservation {
    fn drop(&mut self) {
        if let Ok(mut state) = self.estimator.state.lock() {
            state.active = subtract_work(state.active, self.work);
        }
    }
}

fn add_work(
    left: GenerationAdmissionWork,
    right: GenerationAdmissionWork,
) -> GenerationAdmissionWork {
    GenerationAdmissionWork {
        prompt_tokens: left.prompt_tokens.saturating_add(right.prompt_tokens),
        decode_tokens: left.decode_tokens.saturating_add(right.decode_tokens),
    }
}

fn subtract_work(
    left: GenerationAdmissionWork,
    right: GenerationAdmissionWork,
) -> GenerationAdmissionWork {
    GenerationAdmissionWork {
        prompt_tokens: left.prompt_tokens.saturating_sub(right.prompt_tokens),
        decode_tokens: left.decode_tokens.saturating_sub(right.decode_tokens),
    }
}

fn update_ewma(previous: Option<f64>, sample: f64) -> f64 {
    previous.map_or(sample, |previous| {
        previous * (1.0 - SERVICE_RATE_EWMA_ALPHA) + sample * SERVICE_RATE_EWMA_ALPHA
    })
}

fn push_sample(samples: &mut VecDeque<f64>, sample: f64) {
    if samples.len() == SERVICE_RATE_WINDOW {
        samples.pop_front();
    }
    samples.push_back(sample);
}

fn conservative_ms_per_token(ewma: Option<f64>, samples: &VecDeque<f64>) -> Option<f64> {
    let mut sorted = samples.iter().copied().collect::<Vec<_>>();
    sorted.sort_by(f64::total_cmp);
    let p95 = if sorted.is_empty() {
        None
    } else {
        let index = (sorted.len() * 95).div_ceil(100).saturating_sub(1);
        sorted.get(index).copied()
    };
    match (ewma, p95) {
        (Some(ewma), Some(p95)) => Some(ewma.max(p95)),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
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

pub(in crate::frontend) async fn acquire_generation_permit_with_queue_reservation(
    generation_limit: Arc<Semaphore>,
    reservation: GenerationQueueReservation,
    admission_timeout: Duration,
    deadline: Instant,
    cancellation: &CancellationToken,
) -> OpenAiResult<OwnedSemaphorePermit> {
    if cancellation.is_cancelled() {
        return Err(OpenAiError::cancelled("request cancelled"));
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
                    Err(OpenAiError::cancelled("request cancelled"))
                }
                Ok(Ok(permit)) => Ok(permit),
                Ok(Err(_)) => Err(generation_lanes_busy_error()),
                Err(_) => Err(generation_queue_timeout_error(admission_timeout)),
            }
        }
        () = cancellation.cancelled() => {
            drop(reservation);
            Err(OpenAiError::cancelled("request cancelled"))
        }
    }
}

#[cfg(test)]
pub(in crate::frontend) async fn acquire_generation_permit_with_queue(
    generation_limit: Arc<Semaphore>,
    generation_queue_depth: Arc<AtomicUsize>,
    generation_queue_limit: usize,
    admission_timeout: Duration,
) -> OpenAiResult<OwnedSemaphorePermit> {
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
    ) -> OpenAiResult<u32> {
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
    let sessions = runtime
        .lock()
        .map_err(|_| anyhow!("runtime lock poisoned"))?
        .prewarm_idle_sessions(generation_concurrency)?;
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

pub(in crate::frontend) fn generation_lanes_busy_error() -> OpenAiError {
    OpenAiError::from_kind(
        StatusCode::TOO_MANY_REQUESTS,
        OpenAiErrorKind::RateLimit,
        "all execution lanes are busy",
    )
    .with_retry_after_secs(GENERATION_RETRY_AFTER_SECS)
}

pub(in crate::frontend) fn generation_queue_full_error() -> OpenAiError {
    OpenAiError::from_kind(
        StatusCode::TOO_MANY_REQUESTS,
        OpenAiErrorKind::RateLimit,
        "generation queue is full; retry later",
    )
    .with_retry_after_secs(GENERATION_RETRY_AFTER_SECS)
}

pub(in crate::frontend) fn generation_queue_timeout_error(timeout: Duration) -> OpenAiError {
    OpenAiError::from_kind(
        StatusCode::TOO_MANY_REQUESTS,
        OpenAiErrorKind::RateLimit,
        format!(
            "timed out waiting for an execution lane after {} seconds",
            timeout.as_secs()
        ),
    )
    .with_retry_after_secs(GENERATION_RETRY_AFTER_SECS)
}

pub(in crate::frontend) fn generation_predicted_wait_error(
    predicted_wait_ms: f64,
    timeout: Duration,
) -> OpenAiError {
    OpenAiError::from_kind(
        StatusCode::TOO_MANY_REQUESTS,
        OpenAiErrorKind::RateLimit,
        format!(
            "predicted generation wait {:.3} seconds exceeds the {:.3}-second admission timeout",
            predicted_wait_ms / 1_000.0,
            timeout.as_secs_f64(),
        ),
    )
    .with_retry_after_secs(GENERATION_RETRY_AFTER_SECS)
}

#[cfg(test)]
mod service_estimator_tests {
    use super::*;

    #[test]
    fn warm_estimator_rejects_work_beyond_the_wait_slo() {
        let estimator = Arc::new(GenerationServiceEstimator::new(1));
        let work = GenerationAdmissionWork::new(100, 100);
        estimator.observe_completed(work, 100.0, 100.0);
        let active = estimator.start_active(work);

        assert_eq!(estimator.predicted_wait_ms(), Some(200.0));
        assert!(
            estimator
                .reserve_queued(work, Duration::from_millis(199))
                .is_err()
        );
        assert!(
            estimator
                .reserve_queued(work, Duration::from_millis(200))
                .is_ok()
        );

        drop(active);
    }

    #[test]
    fn queued_and_active_work_are_released_by_raii() {
        let estimator = Arc::new(GenerationServiceEstimator::new(2));
        let work = GenerationAdmissionWork::new(80, 20);
        estimator.observe_completed(work, 80.0, 20.0);
        let queued = estimator
            .reserve_queued(work, Duration::from_secs(1))
            .expect("cold queue reservation");
        assert_eq!(estimator.predicted_wait_ms(), Some(50.0));

        let active = queued.promote();
        assert_eq!(estimator.predicted_wait_ms(), Some(50.0));
        drop(active);
        assert_eq!(estimator.predicted_wait_ms(), Some(0.0));
    }

    #[test]
    fn conservative_rate_uses_the_slower_p95_sample() {
        let samples = VecDeque::from([1.0, 1.0, 1.0, 10.0]);
        assert_eq!(conservative_ms_per_token(Some(1.5), &samples), Some(10.0));
    }
}
