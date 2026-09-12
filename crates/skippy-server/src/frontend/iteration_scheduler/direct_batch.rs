use super::{DirectIteration, MAX_NATIVE_ITERATION_TOKENS};
use openai_frontend::{OpenAiError, OpenAiResult};
use std::collections::{BTreeSet, VecDeque};

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
