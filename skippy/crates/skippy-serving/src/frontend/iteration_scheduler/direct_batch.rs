use super::{DirectIteration, MAX_NATIVE_ITERATION_TOKENS};
use skippy_inference_api::{OpenAiError, OpenAiResult};
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

/// Environment switch: split each coalesced decode wave into this many
/// pipeline groups, so a pipelined split keeps more than one batch in flight
/// (group A computes on one stage while group B computes on the next).
pub(super) const PIPELINE_DECODE_GROUPS_ENV: &str = "SKIPPY_PIPELINE_DECODE_GROUPS";

pub(super) fn pipeline_decode_groups_from_value(value: Option<&str>) -> usize {
    value
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|groups| *groups > 0)
        .unwrap_or(1)
}

/// Decode-wave group count, from the environment override or the planned value.
///
/// `throughput.pipeline_decode_groups` is the configured plan. The environment
/// variable stays as a bench and incident-response override that does not need a
/// replan, the same shape as `SKIPPY_KV_CACHE` — so an existing benchmark
/// invocation keeps working now that the setting is configurable. An
/// unparseable or zero environment value falls back to ungrouped rather than to
/// the plan, matching what [`pipeline_decode_groups_from_value`] did when the
/// variable was the only input.
pub(super) fn resolve_pipeline_decode_groups(
    env_value: Option<&str>,
    planned: Option<usize>,
) -> usize {
    match env_value {
        Some(value) => pipeline_decode_groups_from_value(Some(value)),
        None => planned.filter(|groups| *groups > 0).unwrap_or(1),
    }
}

/// Largest decode batch per pipeline group: the lane count divided across
/// groups, rounded up so every lane still fits in one wave of groups.
pub(super) fn pipeline_group_batch_size(max_direct_batch_size: usize, groups: usize) -> usize {
    max_direct_batch_size.div_ceil(groups.max(1)).max(1)
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
    // Sessions that already have a request in this wave or deferred from it.
    // A later request from such a session must wait so it cannot overtake
    // the earlier one.
    let mut claimed_sessions = BTreeSet::new();
    let mut deferred = VecDeque::new();
    let queued = queue.len();
    for _ in 0..queued {
        if batch.len() >= max_batch_size {
            break;
        }
        let Some(request) = queue.pop_front() else {
            break;
        };
        if claimed_sessions.contains(&request.session_id) {
            deferred.push_back(request);
            continue;
        }
        // Keep scanning past a chunk that does not fit so smaller decode rows
        // queued behind a large prefill chunk still join this wave. The
        // deferred chunk keeps its place ahead of later arrivals, and the
        // queue head always fits a fresh budget, so it cannot starve.
        if request.token_ids.len() > token_budget {
            claimed_sessions.insert(request.session_id.clone());
            deferred.push_back(request);
            continue;
        }
        token_budget = token_budget.saturating_sub(request.token_ids.len());
        claimed_sessions.insert(request.session_id.clone());
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

#[cfg(test)]
mod pipeline_group_tests {
    use super::*;

    #[test]
    fn groups_default_to_one_and_ignore_invalid_values() {
        assert_eq!(pipeline_decode_groups_from_value(None), 1);
        assert_eq!(pipeline_decode_groups_from_value(Some("0")), 1);
        assert_eq!(pipeline_decode_groups_from_value(Some("two")), 1);
        assert_eq!(pipeline_decode_groups_from_value(Some(" 2 ")), 2);
    }

    #[test]
    fn group_batch_size_divides_lanes_rounding_up() {
        assert_eq!(pipeline_group_batch_size(4, 1), 4);
        assert_eq!(pipeline_group_batch_size(4, 2), 2);
        assert_eq!(pipeline_group_batch_size(5, 2), 3);
        assert_eq!(pipeline_group_batch_size(1, 4), 1);
    }

    #[test]
    fn coalescing_stops_at_the_group_size() {
        let group = pipeline_group_batch_size(4, 2);
        assert_eq!(direct_coalesce_target(4, 1, group), 2);
    }
}

#[cfg(test)]
mod pipeline_decode_group_tests {
    use super::{pipeline_group_batch_size, resolve_pipeline_decode_groups};

    #[test]
    fn the_planned_value_applies_when_the_environment_is_unset() {
        assert_eq!(resolve_pipeline_decode_groups(None, Some(2)), 2);
    }

    #[test]
    fn ungrouped_is_the_default_with_neither_source() {
        assert_eq!(resolve_pipeline_decode_groups(None, None), 1);
    }

    /// The override exists so a bench can regroup a running configuration
    /// without replanning a topology.
    #[test]
    fn the_environment_override_beats_the_planned_value() {
        assert_eq!(resolve_pipeline_decode_groups(Some("3"), Some(2)), 3);
    }

    /// A set-but-unusable override means ungrouped, not "fall through to the
    /// plan": that is what the variable did when it was the only input, and
    /// silently honouring the plan would hide the operator's typo.
    #[test]
    fn an_unusable_environment_override_does_not_fall_back_to_the_plan() {
        assert_eq!(resolve_pipeline_decode_groups(Some("nonsense"), Some(4)), 1);
        assert_eq!(resolve_pipeline_decode_groups(Some("0"), Some(4)), 1);
        assert_eq!(resolve_pipeline_decode_groups(Some(""), Some(4)), 1);
    }

    /// A zero reaching the resolver from a plan cannot divide a wave.
    #[test]
    fn a_zero_plan_is_treated_as_ungrouped() {
        assert_eq!(resolve_pipeline_decode_groups(None, Some(0)), 1);
    }

    #[test]
    fn grouping_divides_the_lane_count_and_never_reaches_zero() {
        assert_eq!(pipeline_group_batch_size(4, 2), 2);
        assert_eq!(pipeline_group_batch_size(4, 1), 4);
        // Rounded up, so every lane still fits in one wave of groups.
        assert_eq!(pipeline_group_batch_size(5, 2), 3);
        assert_eq!(pipeline_group_batch_size(1, 4), 1);
    }
}
