//! Where the scheduler's batching decisions are made.
//!
//! A [`BatchPolicy`] answers two questions for the direct queue: how long a
//! batch waits for more requests ([`BatchPolicy::gather`]), and which queued
//! requests it takes ([`BatchPolicy::select`]). The scheduler checks every
//! selection before it runs ([`take_batch`]), so a policy can reorder and
//! hold back work but cannot break a session's ordering, starve the head of
//! the queue, or overfill an iteration.
//!
//! A new algorithm is a new [`BatchPolicyKind`] and an implementation here;
//! [`BatchTuning::resolve`] is the one place a backend gets different values.

use super::direct_batch::{
    PREFILL_COALESCE_PER_TOKEN, PREFILL_COALESCE_WINDOW_MAX, direct_coalesce_target,
};
use super::{DIRECT_ITERATION_COALESCE_WINDOW, DirectIteration};
use skippy_protocol::{BatchConfig, BatchPolicyKind};
use skippy_runtime::{BackendDeviceType, IterationBatchPhase};
use std::collections::{BTreeSet, VecDeque};
use std::time::Duration;

/// The compute backend the stage's batches run on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BatchPlatform {
    Metal,
    Cuda,
    Vulkan,
    Cpu,
    Other,
}

impl BatchPlatform {
    /// From a backend device name as the runtime reports it (`MTL0`, `CUDA1`).
    pub(super) fn from_device_name(name: &str) -> Self {
        let family = name.trim_end_matches(|c: char| c.is_ascii_digit());
        match family.to_ascii_uppercase().as_str() {
            "MTL" | "METAL" => Self::Metal,
            "CUDA" => Self::Cuda,
            "VULKAN" => Self::Vulkan,
            "CPU" => Self::Cpu,
            _ => Self::Other,
        }
    }

    /// The first GPU the runtime reports, else its first device.
    pub(super) fn detect() -> Self {
        let Ok(devices) = skippy_runtime::backend_devices_untraced() else {
            return Self::Other;
        };
        devices
            .iter()
            .find(|device| {
                matches!(
                    device.device_type,
                    BackendDeviceType::Gpu | BackendDeviceType::IntegratedGpu
                )
            })
            .or_else(|| devices.first())
            .map_or(Self::Cpu, |device| Self::from_device_name(&device.name))
    }

    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::Metal => "metal",
            Self::Cuda => "cuda",
            Self::Vulkan => "vulkan",
            Self::Cpu => "cpu",
            Self::Other => "other",
        }
    }
}

/// The values a policy waits by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct BatchTuning {
    /// How long generated tokens wait for the other active requests.
    pub(super) decode_window: Duration,
    /// How long a prompt waits for company, per prompt token. A prefill costs
    /// on the order of a millisecond a token, so a few microseconds a token
    /// buys the chance that requests arriving together prefill as one wave.
    pub(super) prefill_window_per_token: Duration,
    /// The longest a prompt waits. May be shorter than `decode_window`, or zero.
    pub(super) prefill_window_max: Duration,
}

impl BatchTuning {
    /// The policy's values for `platform`, with whatever the stage configures
    /// laid over them.
    pub(super) fn resolve(
        platform: BatchPlatform,
        config: &BatchConfig,
        prefill_coalesce_max_ms: Option<u32>,
    ) -> Self {
        let base = match platform {
            BatchPlatform::Metal
            | BatchPlatform::Cuda
            | BatchPlatform::Vulkan
            | BatchPlatform::Cpu
            | BatchPlatform::Other => Self {
                decode_window: DIRECT_ITERATION_COALESCE_WINDOW,
                prefill_window_per_token: PREFILL_COALESCE_PER_TOKEN,
                prefill_window_max: PREFILL_COALESCE_WINDOW_MAX,
            },
        };
        Self {
            decode_window: config.decode_window_us.map_or(base.decode_window, |us| {
                Duration::from_micros(u64::from(us))
            }),
            prefill_window_per_token: config
                .prefill_window_per_token_us
                .map_or(base.prefill_window_per_token, |us| {
                    Duration::from_micros(u64::from(us))
                }),
            prefill_window_max: prefill_coalesce_max_ms.map_or(base.prefill_window_max, |ms| {
                Duration::from_millis(u64::from(ms))
            }),
        }
    }

    /// The wait for a batch that holds a prefill of `tokens`: a small fraction
    /// of the prefill itself, at least the decode window, never past the cap.
    pub(super) fn prefill_window(&self, tokens: usize) -> Duration {
        (self.prefill_window_per_token * u32::try_from(tokens).unwrap_or(u32::MAX))
            .max(self.decode_window)
            .min(self.prefill_window_max)
    }
}

/// One queued request, as a policy sees it.
#[derive(Clone, Copy, Debug)]
pub(super) struct QueuedRequest<'a> {
    pub(super) session: &'a str,
    pub(super) tokens: usize,
    pub(super) phase: IterationBatchPhase,
}

/// What the scheduler knows when it is about to form a batch. `queued` is in
/// arrival order.
#[derive(Clone, Debug)]
pub(super) struct BatchView<'a> {
    pub(super) queued: Vec<QueuedRequest<'a>>,
    /// Sessions the runtime is serving, queued or not.
    pub(super) active_sessions: usize,
    /// The most requests one batch may hold.
    pub(super) lanes: usize,
    /// The most tokens one batch may hold.
    pub(super) token_budget: usize,
}

impl<'a> BatchView<'a> {
    pub(super) fn of(
        queue: &'a VecDeque<DirectIteration>,
        active_sessions: usize,
        lanes: usize,
        token_budget: usize,
    ) -> Self {
        Self {
            queued: queue
                .iter()
                .map(|request| QueuedRequest {
                    session: &request.session_id,
                    tokens: request.token_ids.len(),
                    phase: request.phase,
                })
                .collect(),
            active_sessions,
            lanes,
            token_budget,
        }
    }
}

/// Wait until `target` requests are queued or `window` has passed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Gather {
    pub(super) target: usize,
    pub(super) window: Duration,
}

/// Positions in [`BatchView::queued`] of the requests to run now.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Selection(pub(super) Vec<usize>);

pub(super) trait BatchPolicy: Send {
    fn kind(&self) -> BatchPolicyKind;
    fn tuning(&self) -> BatchTuning;
    fn gather(&self, view: &BatchView<'_>) -> Gather;
    fn select(&self, view: &BatchView<'_>) -> Selection;
}

/// The policy a stage gets when it configures nothing, on an unnamed backend.
#[cfg(test)]
pub(in crate::frontend::iteration_scheduler) fn default_policy() -> Box<dyn BatchPolicy> {
    batch_policy(
        BatchPolicyKind::default(),
        BatchTuning::resolve(BatchPlatform::Other, &BatchConfig::default(), None),
    )
}

pub(super) fn batch_policy(kind: BatchPolicyKind, tuning: BatchTuning) -> Box<dyn BatchPolicy> {
    match kind {
        BatchPolicyKind::ArrivalOrder => Box::new(ArrivalOrder { tuning }),
    }
}

/// Requests join in the order they arrived. A batch that holds a prompt
/// waits for a full wave, for a window sized to the prompt; a batch of
/// generated tokens waits the decode window for the sessions already active.
pub(super) struct ArrivalOrder {
    pub(super) tuning: BatchTuning,
}

impl BatchPolicy for ArrivalOrder {
    fn kind(&self) -> BatchPolicyKind {
        BatchPolicyKind::ArrivalOrder
    }

    fn tuning(&self) -> BatchTuning {
        self.tuning
    }

    fn gather(&self, view: &BatchView<'_>) -> Gather {
        let prefill_tokens = view
            .queued
            .iter()
            .filter(|request| request.phase == IterationBatchPhase::Prefill)
            .map(|request| request.tokens)
            .max();
        match prefill_tokens {
            Some(tokens) => Gather {
                target: view.lanes,
                window: self.tuning.prefill_window(tokens),
            },
            None => Gather {
                target: direct_coalesce_target(view.active_sessions, view.queued.len(), view.lanes),
                window: self.tuning.decode_window,
            },
        }
    }

    fn select(&self, view: &BatchView<'_>) -> Selection {
        let mut budget = view.token_budget;
        // Sessions with a request taken or passed over: a later request from
        // one must wait, so it cannot overtake the earlier one.
        let mut claimed = BTreeSet::new();
        let mut selected = Vec::new();
        for (position, request) in view.queued.iter().enumerate() {
            if selected.len() >= view.lanes || budget == 0 {
                break;
            }
            if !claimed.insert(request.session) {
                continue;
            }
            // Keep scanning past a chunk that does not fit, so smaller decode
            // rows queued behind a large prefill chunk still join this wave.
            if request.tokens > budget {
                continue;
            }
            budget -= request.tokens;
            selected.push(position);
        }
        Selection(selected)
    }
}

/// Why a policy's selection was not run as given.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BatchRejection {
    Empty,
    NotInQueueOrder,
    OutOfRange,
    TooManyRequests,
    OverTokenBudget,
    SessionTwice,
    OvertakesSession,
    SkipsQueueHead,
}

impl BatchRejection {
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::Empty => "empty",
            Self::NotInQueueOrder => "not_in_queue_order",
            Self::OutOfRange => "out_of_range",
            Self::TooManyRequests => "too_many_requests",
            Self::OverTokenBudget => "over_token_budget",
            Self::SessionTwice => "session_twice",
            Self::OvertakesSession => "overtakes_session",
            Self::SkipsQueueHead => "skips_queue_head",
        }
    }
}

/// The rules every batch keeps, whichever policy chose it.
///
/// The head of the queue always fits a fresh budget (iterations are validated
/// against it when queued), so requiring it keeps every request moving.
pub(super) fn check_selection(
    view: &BatchView<'_>,
    selection: &Selection,
) -> Result<(), BatchRejection> {
    let positions = &selection.0;
    if positions.is_empty() {
        return Err(BatchRejection::Empty);
    }
    if positions.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(BatchRejection::NotInQueueOrder);
    }
    if positions
        .iter()
        .any(|&position| position >= view.queued.len())
    {
        return Err(BatchRejection::OutOfRange);
    }
    if positions[0] != 0 {
        return Err(BatchRejection::SkipsQueueHead);
    }
    if positions.len() > view.lanes {
        return Err(BatchRejection::TooManyRequests);
    }
    let tokens: usize = positions
        .iter()
        .map(|&position| view.queued[position].tokens)
        .sum();
    if tokens > view.token_budget {
        return Err(BatchRejection::OverTokenBudget);
    }
    let mut sessions = BTreeSet::new();
    if !positions
        .iter()
        .all(|&position| sessions.insert(view.queued[position].session))
    {
        return Err(BatchRejection::SessionTwice);
    }
    let overtakes = positions.iter().any(|&position| {
        view.queued[..position]
            .iter()
            .enumerate()
            .any(|(earlier, request)| {
                request.session == view.queued[position].session
                    && positions.binary_search(&earlier).is_err()
            })
    });
    if overtakes {
        return Err(BatchRejection::OvertakesSession);
    }
    Ok(())
}

/// A batch off the queue: the policy's selection when it keeps the rules,
/// otherwise arrival order, with the reason the policy's was set aside.
///
/// Requests left behind keep their order. The batch is ordered by session so
/// arrival timing cannot permute graph rows between otherwise identical waves.
pub(super) fn take_batch(
    queue: &mut VecDeque<DirectIteration>,
    policy: &dyn BatchPolicy,
    active_sessions: usize,
    lanes: usize,
    token_budget: usize,
) -> (Vec<DirectIteration>, Option<BatchRejection>) {
    if queue.is_empty() {
        return (Vec::new(), None);
    }
    let (positions, rejection) = {
        let view = BatchView::of(queue, active_sessions, lanes, token_budget);
        let selection = policy.select(&view);
        match check_selection(&view, &selection) {
            Ok(()) => (selection.0, None),
            Err(rejection) => {
                let fallback = ArrivalOrder {
                    tuning: policy.tuning(),
                }
                .select(&view);
                (fallback.0, Some(rejection))
            }
        }
    };
    let mut batch = Vec::with_capacity(positions.len());
    let mut left = VecDeque::with_capacity(queue.len());
    let mut wanted = positions.iter().copied().peekable();
    for (position, request) in std::mem::take(queue).into_iter().enumerate() {
        if wanted.peek() == Some(&position) {
            wanted.next();
            batch.push(request);
        } else {
            left.push_back(request);
        }
    }
    *queue = left;
    batch.sort_unstable_by(|left, right| left.session_id.cmp(&right.session_id));
    (batch, rejection)
}

#[cfg(test)]
mod tests;
