#[cfg(test)]
use super::DIRECT_ITERATION_COALESCE_WINDOW;
#[cfg(test)]
use super::DirectIteration;
use super::MAX_NATIVE_ITERATION_TOKENS;
#[cfg(test)]
use super::batch_policy::{BatchPlatform, BatchTuning, default_policy, take_batch};
use skippy_inference_api::{InferenceError, InferenceResult};
#[cfg(test)]
use skippy_protocol::BatchConfig;
#[cfg(test)]
use std::collections::VecDeque;
use std::time::Duration;

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
    // Active sessions may belong to the planned scheduler. There is no
    // direct batch to coalesce until its first request is actually queued;
    // waiting here would delay every planned decode by the coalesce window.
    if queued_direct_iterations == 0 {
        return 0;
    }
    active_runtime_sessions
        .max(queued_direct_iterations)
        .min(max_direct_batch_size)
}

/// How long a prefill may wait for company, per prompt token, unless the
/// stage's batch settings say otherwise.
pub(super) const PREFILL_COALESCE_PER_TOKEN: Duration = Duration::from_micros(15);
/// The longest a prefill waits unless the stage configures another ceiling.
pub(super) const PREFILL_COALESCE_WINDOW_MAX: Duration = Duration::from_millis(8);

/// The ceiling on the prefill coalesce window: the stage's
/// `prefill_coalesce_max_ms`, else the scheduler's own.
#[cfg(test)]
pub(super) fn prefill_coalesce_max(configured_ms: Option<u32>) -> Duration {
    BatchTuning::resolve(BatchPlatform::Other, &BatchConfig::default(), configured_ms)
        .prefill_window_max
}

/// The default tuning's coalesce window for a batch that holds a prefill of
/// `tokens`, under the ceiling `max`.
#[cfg(test)]
pub(super) fn prefill_coalesce_window(tokens: usize, max: Duration) -> Duration {
    BatchTuning {
        prefill_window_max: max,
        ..BatchTuning::resolve(BatchPlatform::Other, &BatchConfig::default(), None)
    }
    .prefill_window(tokens)
}

#[cfg(test)]
mod coalescing_tests;
#[cfg(test)]
mod selection_tests;

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

/// A batch chosen by the default policy.
#[cfg(test)]
pub(super) fn take_direct_iteration_batch(
    queue: &mut VecDeque<DirectIteration>,
    max_batch_size: usize,
    token_budget: usize,
) -> Vec<DirectIteration> {
    take_batch(
        queue,
        default_policy().as_ref(),
        0,
        max_batch_size,
        token_budget,
    )
    .0
}

pub(super) fn validate_direct_iteration(
    token_ids: &[i32],
    positions: &[i32],
    max_iteration_tokens: usize,
) -> InferenceResult<()> {
    if token_ids.is_empty() {
        return Err(InferenceError::invalid_request(
            "scheduler iteration requires at least one token",
        ));
    }
    let token_limit = max_iteration_tokens.min(MAX_NATIVE_ITERATION_TOKENS);
    if token_ids.len() > token_limit {
        return Err(InferenceError::invalid_request(format!(
            "scheduler iteration exceeds the {token_limit}-token configured iteration limit"
        )));
    }
    if !positions.is_empty() && !positions.len().is_multiple_of(token_ids.len()) {
        return Err(InferenceError::invalid_request(
            "scheduler iteration positions must be empty or token-major",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod prefill_coalesce_tests {
    use super::*;

    #[test]
    fn a_long_prompt_waits_the_most_and_a_short_one_the_decode_window() {
        let max = prefill_coalesce_max(None);
        assert_eq!(max, PREFILL_COALESCE_WINDOW_MAX);
        assert_eq!(
            prefill_coalesce_window(512, max),
            Duration::from_micros(7_680)
        );
        assert_eq!(prefill_coalesce_window(4096, max), max);
        assert_eq!(
            prefill_coalesce_window(9, max),
            DIRECT_ITERATION_COALESCE_WINDOW
        );
        assert_eq!(prefill_coalesce_window(200, max), Duration::from_millis(3));
    }

    /// The stage's ceiling bounds every window, the decode floor included:
    /// zero means a prefill never waits for company.
    #[test]
    fn a_configured_ceiling_bounds_every_window() {
        let max = prefill_coalesce_max(Some(3));
        assert_eq!(prefill_coalesce_window(4096, max), Duration::from_millis(3));
        assert_eq!(
            prefill_coalesce_window(9, max),
            DIRECT_ITERATION_COALESCE_WINDOW
        );
        let never = prefill_coalesce_max(Some(0));
        assert_eq!(prefill_coalesce_window(4096, never), Duration::ZERO);
        assert_eq!(prefill_coalesce_window(9, never), Duration::ZERO);
        let long = prefill_coalesce_max(Some(50));
        assert_eq!(
            prefill_coalesce_window(4096, long),
            Duration::from_micros(61_440).min(long)
        );
    }
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
