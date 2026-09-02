use super::{DirectIteration, MAX_NATIVE_ITERATION_TOKENS};
use openai_frontend::{OpenAiError, OpenAiResult};
use std::collections::{BTreeSet, VecDeque};
use std::time::Duration;

const DECODE_BURST_TURNS: u32 = 8;
const TIME_DEFICIT_PLANNED_SHARE: u32 = 3;
const TIME_DEFICIT_MAX_PLANNED_TURNS: u32 = 64;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) enum DirectWorkPolicy {
    #[default]
    Alternate,
    DecodeBurst,
    TimeDeficit,
}

impl DirectWorkPolicy {
    pub(super) fn from_value(value: Option<&str>) -> OpenAiResult<Self> {
        match value.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
            Some("decode-burst") => Ok(Self::DecodeBurst),
            Some("time-deficit") => Ok(Self::TimeDeficit),
            Some("alternate") | None => Ok(Self::Alternate),
            Some(value) => Err(OpenAiError::invalid_request(format!(
                "invalid direct work policy {value:?}; expected alternate, decode-burst, or time-deficit"
            ))),
        }
    }

    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::Alternate => "alternate",
            Self::DecodeBurst => "decode-burst",
            Self::TimeDeficit => "time-deficit",
        }
    }
}

#[derive(Debug, Default)]
pub(super) struct DirectWorkArbiter {
    policy: DirectWorkPolicy,
    last_served_direct: bool,
    planned_turns_since_direct: u32,
    planned_time_debt: Duration,
}

impl DirectWorkArbiter {
    pub(super) fn new(policy: DirectWorkPolicy) -> Self {
        Self {
            policy,
            ..Self::default()
        }
    }

    pub(super) fn should_serve_direct(&mut self, has_direct: bool, has_planned: bool) -> bool {
        let serve_direct = match (has_direct, has_planned) {
            (true, true) => match self.policy {
                DirectWorkPolicy::Alternate => !self.last_served_direct,
                DirectWorkPolicy::DecodeBurst => {
                    self.planned_turns_since_direct >= DECODE_BURST_TURNS
                }
                DirectWorkPolicy::TimeDeficit => {
                    self.planned_time_debt.is_zero()
                        || self.planned_turns_since_direct >= TIME_DEFICIT_MAX_PLANNED_TURNS
                }
            },
            (true, false) => true,
            (false, _) => false,
        };

        if serve_direct {
            self.last_served_direct = true;
            self.planned_turns_since_direct = 0;
        } else if has_planned {
            self.last_served_direct = false;
            if has_direct {
                self.planned_turns_since_direct = self.planned_turns_since_direct.saturating_add(1);
            }
        }
        serve_direct
    }

    pub(super) fn observe_direct(&mut self, elapsed: Duration) {
        if self.policy == DirectWorkPolicy::TimeDeficit {
            self.planned_time_debt = self
                .planned_time_debt
                .saturating_add(elapsed.saturating_mul(TIME_DEFICIT_PLANNED_SHARE));
        }
    }

    pub(super) fn observe_planned(&mut self, elapsed: Duration) {
        if self.policy == DirectWorkPolicy::TimeDeficit {
            self.planned_time_debt = self.planned_time_debt.saturating_sub(elapsed);
        }
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

pub(super) fn validate_direct_iteration(token_ids: &[i32], positions: &[i32]) -> OpenAiResult<()> {
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
