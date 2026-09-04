use super::{
    DEFAULT_MAX_CONSECUTIVE_PREFILL_BATCHES, DIRECT_ARBITRATION_POLICY_ENV,
    DIRECT_ARBITRATION_PREFILL_LIMIT_ENV, DirectIteration, MAX_NATIVE_ITERATION_TOKENS,
};
use openai_frontend::{OpenAiError, OpenAiResult};
use std::collections::{BTreeSet, VecDeque};
use std::env;

/// Outer arbitration policy between the direct queue (prefill chunks and
/// direct decode) and the planned decode queue.
///
/// `alternate` is the historical one-direct/one-planned turn behavior.
/// `budgeted` is vLLM-style step composition: direct decode work is always
/// served before planned work, and direct prefill chunks only run when no
/// direct decode is pending, bounded by a hard prefill-progress limit so
/// planned decode cannot starve behind a long prompt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DirectArbitrationPolicy {
    Alternate,
    Budgeted {
        /// Maximum consecutive direct prefill batches before one planned
        /// decode iteration is forced.
        max_consecutive_prefill_batches: usize,
    },
}

impl DirectArbitrationPolicy {
    pub(super) fn from_env() -> Self {
        match env::var(DIRECT_ARBITRATION_POLICY_ENV)
            .ok()
            .as_deref()
            .map(str::trim)
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("budgeted") => Self::Budgeted {
                max_consecutive_prefill_batches: env::var(DIRECT_ARBITRATION_PREFILL_LIMIT_ENV)
                    .ok()
                    .and_then(|value| value.trim().parse().ok())
                    .unwrap_or(DEFAULT_MAX_CONSECUTIVE_PREFILL_BATCHES),
            },
            _ => Self::Alternate,
        }
    }
}

pub(super) fn should_serve_direct(
    has_direct: bool,
    has_planned: bool,
    last_served_direct: bool,
) -> bool {
    if has_direct && has_planned {
        !last_served_direct
    } else {
        has_direct
    }
}

/// `budgeted` arbitration: direct decode always wins the turn. Direct prefill
/// runs only when no direct decode is queued, and yields to planned decode
/// after `max_consecutive_prefill_batches` consecutive prefill batches.
pub(super) fn should_serve_direct_budgeted(
    direct_decode_pending: bool,
    direct_prefill_pending: bool,
    has_planned: bool,
    consecutive_prefill_batches: usize,
    max_consecutive_prefill_batches: usize,
) -> bool {
    if direct_decode_pending {
        return true;
    }
    if !direct_prefill_pending {
        return false;
    }
    // Prefill is pending and decode (direct or planned) is not: run prefill
    // unless the fairness bound forces a planned turn.
    if has_planned && consecutive_prefill_batches >= max_consecutive_prefill_batches {
        return false;
    }
    true
}

pub(super) fn direct_coalesce_target(
    active_runtime_sessions: usize,
    queued_direct_iterations: usize,
    max_direct_batch_size: usize,
) -> usize {
    active_runtime_sessions
        .max(queued_direct_iterations)
        .min(max_direct_batch_size)
}

pub(super) fn scheduler_safe_mode_from_value(value: Option<&str>) -> bool {
    value.is_some_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

pub(super) const fn effective_scheduler_lane_count(
    lane_count: usize,
    safe_mode: bool,
    continuous_batching: bool,
) -> usize {
    if safe_mode || !continuous_batching {
        1
    } else {
        lane_count
    }
}

pub(super) fn take_direct_iteration_batch(
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

pub(super) fn validate_direct_iteration(
    token_ids: &[i32],
    positions: &[i32],
    max_iteration_tokens: usize,
) -> OpenAiResult<()> {
    if token_ids.is_empty() {
        return Err(OpenAiError::invalid_request(
            "scheduler iteration requires at least one token",
        ));
    }
    let token_limit = max_iteration_tokens.min(MAX_NATIVE_ITERATION_TOKENS);
    if token_ids.len() > token_limit {
        return Err(OpenAiError::invalid_request(format!(
            "scheduler iteration exceeds the {token_limit}-token configured iteration limit"
        )));
    }
    if !positions.is_empty() && !positions.len().is_multiple_of(token_ids.len()) {
        return Err(OpenAiError::invalid_request(
            "scheduler iteration positions must be empty or token-major",
        ));
    }
    Ok(())
}
