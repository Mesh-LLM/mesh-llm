use crate::frontend::generation::CONTEXT_BUDGET_MAX_TOKENS;
use crate::frontend::generation::GENERATION_RETRY_AFTER_SECS;
use crate::frontend::generation::PhaseTimer;
use crate::frontend::util::context_budget_completion_tokens;
use crate::frontend::util::ensure_context_capacity;
use crate::runtime_state::RuntimeState;
use crate::telemetry::Telemetry;
use crate::telemetry::lifecycle_attrs;
use crate::telemetry::now_unix_nanos;
use anyhow::Result;
use anyhow::anyhow;
use anyhow::bail;
use axum::http::StatusCode;
use openai_frontend::OpenAiError;
use openai_frontend::OpenAiErrorKind;
use openai_frontend::OpenAiResult;
use serde_json::json;
use skippy_protocol::StageConfig;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Duration;
use tokio::sync::OwnedSemaphorePermit;
use tokio::sync::Semaphore;
use tokio::sync::TryAcquireError;

pub(in crate::frontend) struct GenerationQueueReservation {
    pub(in crate::frontend) depth: Arc<AtomicUsize>,
}

impl Drop for GenerationQueueReservation {
    fn drop(&mut self) {
        self.depth.fetch_sub(1, Ordering::AcqRel);
    }
}

pub(in crate::frontend) async fn acquire_generation_permit_with_queue(
    generation_limit: Arc<Semaphore>,
    generation_queue_depth: Arc<AtomicUsize>,
    generation_queue_limit: usize,
    admission_timeout: Duration,
) -> OpenAiResult<OwnedSemaphorePermit> {
    match generation_limit.clone().try_acquire_owned() {
        Ok(permit) => return Ok(permit),
        Err(TryAcquireError::Closed) => return Err(generation_lanes_busy_error()),
        Err(TryAcquireError::NoPermits) => {}
    }

    let _queue_reservation =
        reserve_generation_queue(generation_queue_depth, generation_queue_limit)
            .ok_or_else(generation_queue_full_error)?;
    tokio::time::timeout(admission_timeout, generation_limit.acquire_owned())
        .await
        .map_err(|_| generation_queue_timeout_error(admission_timeout))?
        .map_err(|_| generation_lanes_busy_error())
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
                ensure_context_capacity(prompt_token_count, max_tokens, ctx_size)?;
                Ok(max_tokens)
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
