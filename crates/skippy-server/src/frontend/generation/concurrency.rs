use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use tokio::sync::{OwnedSemaphorePermit, Semaphore, TryAcquireError};

const ADAPTIVE_WINDOW_REQUESTS: usize = 16;
const MIN_THROUGHPUT_IMPROVEMENT: f64 = 0.03;
const MAX_P95_LATENCY_REGRESSION: f64 = 0.10;
const MAX_HARDWARE_PRESSURE_REGRESSION: f64 = 0.10;
const REPROBE_COOLDOWN_WINDOWS: usize = 2;
const DEMAND_IDLE_RESET: Duration = Duration::from_secs(5);

#[derive(Clone, Copy, Debug, PartialEq)]
struct GenerationConcurrencyMetrics {
    throughput_tokens_per_second: f64,
    p95_latency_ms: f64,
    p95_hardware_ms_per_token: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct GenerationConcurrencyComparison {
    throughput_improvement: f64,
    p95_latency_ratio: f64,
    hardware_pressure_ratio: f64,
}

struct GenerationConcurrencyWindow {
    started_at: Option<Instant>,
    completed_requests: usize,
    completed_tokens: u64,
    latency_samples_ms: VecDeque<f64>,
    hardware_ms_per_token_samples: VecDeque<f64>,
}

impl GenerationConcurrencyWindow {
    fn new() -> Self {
        Self {
            started_at: None,
            completed_requests: 0,
            completed_tokens: 0,
            latency_samples_ms: VecDeque::new(),
            hardware_ms_per_token_samples: VecDeque::new(),
        }
    }

    fn reset(&mut self) {
        *self = Self::new();
    }

    fn push(
        &mut self,
        completed_tokens: u64,
        executed_tokens: u64,
        latency_ms: f64,
        started_at: Instant,
    ) {
        self.started_at = Some(
            self.started_at
                .map_or(started_at, |current| current.min(started_at)),
        );
        self.completed_requests = self.completed_requests.saturating_add(1);
        self.completed_tokens = self.completed_tokens.saturating_add(completed_tokens);
        self.latency_samples_ms.push_back(latency_ms);
        self.hardware_ms_per_token_samples
            .push_back(latency_ms / executed_tokens as f64);
    }

    fn metrics(&self, now: Instant) -> Option<GenerationConcurrencyMetrics> {
        let elapsed_seconds = now.duration_since(self.started_at?).as_secs_f64();
        if self.completed_requests < ADAPTIVE_WINDOW_REQUESTS
            || self.completed_tokens == 0
            || !elapsed_seconds.is_finite()
            || elapsed_seconds <= 0.0
        {
            return None;
        }
        Some(GenerationConcurrencyMetrics {
            throughput_tokens_per_second: self.completed_tokens as f64 / elapsed_seconds,
            p95_latency_ms: p95(&self.latency_samples_ms)?,
            p95_hardware_ms_per_token: p95(&self.hardware_ms_per_token_samples)?,
        })
    }
}

fn p95(samples: &VecDeque<f64>) -> Option<f64> {
    let mut sorted = samples.iter().copied().collect::<Vec<_>>();
    sorted.sort_by(f64::total_cmp);
    let index = (sorted.len() * 95).div_ceil(100).saturating_sub(1);
    sorted.get(index).copied()
}

fn compare_metrics(
    baseline: GenerationConcurrencyMetrics,
    current: GenerationConcurrencyMetrics,
) -> GenerationConcurrencyComparison {
    GenerationConcurrencyComparison {
        throughput_improvement: current.throughput_tokens_per_second
            / baseline.throughput_tokens_per_second
            - 1.0,
        p95_latency_ratio: current.p95_latency_ms / baseline.p95_latency_ms,
        hardware_pressure_ratio: current.p95_hardware_ms_per_token
            / baseline.p95_hardware_ms_per_token,
    }
}

#[derive(Clone, Copy)]
struct GenerationConcurrencyTrial {
    committed_limit: usize,
    baseline: GenerationConcurrencyMetrics,
}

struct AdaptiveGenerationConcurrencyState {
    floor_limit: usize,
    committed_limit: usize,
    baseline: Option<GenerationConcurrencyMetrics>,
    trial: Option<GenerationConcurrencyTrial>,
    cooldown_windows: usize,
    last_saturated_at: Option<Instant>,
    observed_requests: usize,
    saturated_requests: usize,
    window: GenerationConcurrencyWindow,
}

#[derive(Default)]
struct PermitRetirementState {
    pending: usize,
}

#[derive(Default)]
struct PermitRetirement {
    state: Mutex<PermitRetirementState>,
}

/// A generation lane whose release can satisfy a pending controller rollback.
pub(in crate::frontend) struct GenerationConcurrencyPermit {
    permit: Option<OwnedSemaphorePermit>,
    retirement: Arc<PermitRetirement>,
}

impl Drop for GenerationConcurrencyPermit {
    fn drop(&mut self) {
        let Some(permit) = self.permit.take() else {
            return;
        };
        let mut retirement = self
            .retirement
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if retirement.pending > 0 {
            retirement.pending -= 1;
            permit.forget();
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(in crate::frontend) struct GenerationConcurrencyDecision {
    pub(in crate::frontend) action: &'static str,
    pub(in crate::frontend) reason: &'static str,
    pub(in crate::frontend) previous_limit: usize,
    pub(in crate::frontend) current_limit: usize,
    pub(in crate::frontend) throughput_tokens_per_second: Option<f64>,
    pub(in crate::frontend) p95_latency_ms: Option<f64>,
    pub(in crate::frontend) p95_hardware_ms_per_token: Option<f64>,
    pub(in crate::frontend) throughput_improvement: Option<f64>,
    pub(in crate::frontend) p95_latency_ratio: Option<f64>,
    pub(in crate::frontend) hardware_pressure_ratio: Option<f64>,
    pub(in crate::frontend) observed_requests: Option<usize>,
    pub(in crate::frontend) saturated_requests: Option<usize>,
}

pub(in crate::frontend) struct GenerationConcurrencyObservation {
    pub(in crate::frontend) completed_tokens: u64,
    pub(in crate::frontend) executed_tokens: u64,
    pub(in crate::frontend) latency_ms: f64,
    pub(in crate::frontend) saturated: bool,
    pub(in crate::frontend) at_capacity: bool,
    pub(in crate::frontend) started_at: Instant,
}

/// Admission gate whose configured maximum remains the KV-lane hard ceiling.
///
/// Fixed mode exposes the full ceiling immediately. Adaptive mode establishes a
/// safe baseline, tentatively probes one additional lane, and commits the probe
/// only when useful-token retirement improves without hardware service-time or
/// p95 latency pressure. A failed probe drains back to the committed limit;
/// steady-state pressure can back off further, and a cooldown periodically
/// re-probes after the workload changes.
pub(in crate::frontend) struct GenerationConcurrencyController {
    semaphore: Arc<Semaphore>,
    retirement: Arc<PermitRetirement>,
    hard_limit: usize,
    current_limit: AtomicUsize,
    demand_epoch: AtomicU64,
    adaptive: Mutex<Option<AdaptiveGenerationConcurrencyState>>,
}

impl GenerationConcurrencyController {
    pub(in crate::frontend) fn fixed(hard_limit: usize) -> Self {
        Self::new(hard_limit, None)
    }

    pub(in crate::frontend) fn adaptive(hard_limit: usize, initial_limit: usize) -> Self {
        Self::new(hard_limit, Some(initial_limit))
    }

    fn new(hard_limit: usize, initial_limit: Option<usize>) -> Self {
        let hard_limit = hard_limit.max(1);
        let adaptive_enabled = initial_limit.is_some();
        let initial_limit = initial_limit.unwrap_or(hard_limit).clamp(1, hard_limit);
        Self {
            semaphore: Arc::new(Semaphore::new(initial_limit)),
            retirement: Arc::new(PermitRetirement::default()),
            hard_limit,
            current_limit: AtomicUsize::new(initial_limit),
            demand_epoch: AtomicU64::new(0),
            adaptive: Mutex::new(
                adaptive_enabled.then_some(AdaptiveGenerationConcurrencyState {
                    floor_limit: 1,
                    committed_limit: initial_limit,
                    baseline: None,
                    trial: None,
                    cooldown_windows: 0,
                    last_saturated_at: None,
                    observed_requests: 0,
                    saturated_requests: 0,
                    window: GenerationConcurrencyWindow::new(),
                }),
            ),
        }
    }

    pub(in crate::frontend) fn try_acquire_owned(
        &self,
    ) -> Result<GenerationConcurrencyPermit, TryAcquireError> {
        self.semaphore
            .clone()
            .try_acquire_owned()
            .map(|permit| self.wrap_permit(permit))
    }

    pub(in crate::frontend) fn semaphore(&self) -> Arc<Semaphore> {
        self.semaphore.clone()
    }

    pub(in crate::frontend) fn wrap_permit(
        &self,
        permit: OwnedSemaphorePermit,
    ) -> GenerationConcurrencyPermit {
        GenerationConcurrencyPermit {
            permit: Some(permit),
            retirement: Arc::clone(&self.retirement),
        }
    }

    #[cfg(test)]
    pub(in crate::frontend) fn available_permits(&self) -> usize {
        self.semaphore.available_permits()
    }

    pub(in crate::frontend) fn current_limit(&self) -> usize {
        self.current_limit.load(Ordering::Acquire)
    }

    pub(in crate::frontend) fn is_at_capacity(&self) -> bool {
        self.semaphore.available_permits() == 0
    }

    pub(in crate::frontend) fn demand_epoch(&self) -> u64 {
        self.demand_epoch.load(Ordering::Acquire)
    }

    pub(in crate::frontend) fn note_queued_demand(&self) {
        self.demand_epoch.fetch_add(1, Ordering::AcqRel);
    }

    pub(in crate::frontend) fn was_saturated_since(
        &self,
        demand_epoch: u64,
        queued_at_start: bool,
        queued_now: bool,
    ) -> bool {
        queued_at_start || queued_now || self.demand_epoch() != demand_epoch
    }

    pub(in crate::frontend) fn hard_limit(&self) -> usize {
        self.hard_limit
    }

    pub(in crate::frontend) fn observe_completed(
        &self,
        observation: GenerationConcurrencyObservation,
    ) -> Option<GenerationConcurrencyDecision> {
        self.observe_completed_at(observation, Instant::now())
    }

    fn observe_completed_at(
        &self,
        observation: GenerationConcurrencyObservation,
        now: Instant,
    ) -> Option<GenerationConcurrencyDecision> {
        let Ok(mut adaptive) = self.adaptive.lock() else {
            return None;
        };
        let state = adaptive.as_mut()?;
        if observation.completed_tokens == 0
            || observation.executed_tokens == 0
            || !observation.latency_ms.is_finite()
            || observation.latency_ms <= 0.0
        {
            state.window.reset();
            return None;
        }
        state.observed_requests = state.observed_requests.saturating_add(1);
        let trial_at_capacity = state.trial.is_some() && observation.at_capacity;
        if !observation.saturated && !trial_at_capacity {
            let demand_expired = state.last_saturated_at.is_some_and(|last_saturated_at| {
                now.saturating_duration_since(last_saturated_at) >= DEMAND_IDLE_RESET
            });
            if demand_expired {
                state.window.reset();
                state.baseline = None;
                state.last_saturated_at = None;
                let decision = state
                    .trial
                    .take()
                    .map(|trial| self.rollback_trial(state, trial, "trial-demand-ended"));
                if let Some(decision) = decision {
                    return Some(take_observation_counts(state, decision));
                }
            }
            if state.observed_requests >= ADAPTIVE_WINDOW_REQUESTS {
                let current_limit = self.current_limit();
                return Some(take_observation_counts(
                    state,
                    empty_decision(
                        "hold",
                        "insufficient-sustained-demand",
                        current_limit,
                        current_limit,
                    ),
                ));
            }
            return None;
        }
        state.last_saturated_at = Some(now);
        state.saturated_requests = state.saturated_requests.saturating_add(1);

        state.window.push(
            observation.completed_tokens,
            observation.executed_tokens,
            observation.latency_ms,
            observation.started_at,
        );
        let Some(metrics) = state.window.metrics(now) else {
            if state.observed_requests < ADAPTIVE_WINDOW_REQUESTS {
                return None;
            }
            let current_limit = self.current_limit();
            return Some(take_observation_counts(
                state,
                empty_decision(
                    "hold",
                    "insufficient-sustained-demand",
                    current_limit,
                    current_limit,
                ),
            ));
        };
        state.window.reset();

        let decision = if let Some(trial) = state.trial.take() {
            self.evaluate_trial(state, trial, metrics)
        } else {
            self.evaluate_committed_window(state, metrics)
        };
        Some(take_observation_counts(state, decision))
    }

    pub(in crate::frontend) fn observe_failed(&self) -> Option<GenerationConcurrencyDecision> {
        let Ok(mut adaptive) = self.adaptive.lock() else {
            return None;
        };
        let state = adaptive.as_mut()?;
        state.window.reset();
        state.baseline = None;
        state.last_saturated_at = None;
        state.observed_requests = 0;
        state.saturated_requests = 0;
        if let Some(trial) = state.trial.take() {
            return Some(self.rollback_trial(state, trial, "generation-failure"));
        }
        let previous_limit = self.current_limit();
        if previous_limit <= state.floor_limit {
            return Some(empty_decision(
                "hold",
                "generation-failure-at-minimum",
                previous_limit,
                previous_limit,
            ));
        }
        let current_limit = previous_limit - 1;
        state.committed_limit = current_limit;
        state.cooldown_windows = REPROBE_COOLDOWN_WINDOWS;
        self.reduce_limit(previous_limit, current_limit);
        Some(empty_decision(
            "backoff",
            "generation-failure",
            previous_limit,
            current_limit,
        ))
    }

    fn evaluate_trial(
        &self,
        state: &mut AdaptiveGenerationConcurrencyState,
        trial: GenerationConcurrencyTrial,
        metrics: GenerationConcurrencyMetrics,
    ) -> GenerationConcurrencyDecision {
        let comparison = compare_metrics(trial.baseline, metrics);
        let throughput_improved = comparison.throughput_improvement >= MIN_THROUGHPUT_IMPROVEMENT;
        let latency_healthy = comparison.p95_latency_ratio <= 1.0 + MAX_P95_LATENCY_REGRESSION;
        let hardware_healthy =
            comparison.hardware_pressure_ratio <= 1.0 + MAX_HARDWARE_PRESSURE_REGRESSION;
        if throughput_improved && latency_healthy && hardware_healthy {
            let current_limit = self.current_limit();
            state.committed_limit = current_limit;
            state.baseline = Some(metrics);
            state.cooldown_windows = 0;
            return metrics_decision(
                "commit",
                "probe-healthy",
                trial.committed_limit,
                current_limit,
                metrics,
                Some(comparison),
            );
        }

        let reason = if !hardware_healthy {
            "hardware-backpressure"
        } else if !latency_healthy {
            "p95-latency-regression"
        } else {
            "retirement-not-improved"
        };
        let mut decision = self.rollback_trial(state, trial, reason);
        apply_metrics(&mut decision, metrics, Some(comparison));
        decision
    }

    fn evaluate_committed_window(
        &self,
        state: &mut AdaptiveGenerationConcurrencyState,
        metrics: GenerationConcurrencyMetrics,
    ) -> GenerationConcurrencyDecision {
        let previous_limit = self.current_limit();
        let comparison = state
            .baseline
            .map(|baseline| compare_metrics(baseline, metrics));
        if comparison.is_some_and(|value| {
            value.hardware_pressure_ratio > 1.0 + MAX_HARDWARE_PRESSURE_REGRESSION
        }) && previous_limit > state.floor_limit
        {
            let current_limit = previous_limit - 1;
            state.committed_limit = current_limit;
            state.baseline = None;
            state.cooldown_windows = REPROBE_COOLDOWN_WINDOWS;
            self.reduce_limit(previous_limit, current_limit);
            return metrics_decision(
                "backoff",
                "hardware-backpressure",
                previous_limit,
                current_limit,
                metrics,
                comparison,
            );
        }

        state.baseline = Some(metrics);
        if state.cooldown_windows > 0 {
            state.cooldown_windows -= 1;
            return metrics_decision(
                "hold",
                "reprobe-cooldown",
                previous_limit,
                previous_limit,
                metrics,
                comparison,
            );
        }
        if previous_limit >= self.hard_limit {
            return metrics_decision(
                "hold",
                "hard-ceiling-reached",
                previous_limit,
                previous_limit,
                metrics,
                comparison,
            );
        }
        if self.pending_retirements() > 0 {
            return metrics_decision(
                "hold",
                "rollback-draining",
                previous_limit,
                previous_limit,
                metrics,
                comparison,
            );
        }

        state.trial = Some(GenerationConcurrencyTrial {
            committed_limit: state.committed_limit,
            baseline: metrics,
        });
        let current_limit = previous_limit + 1;
        self.semaphore.add_permits(1);
        self.current_limit.store(current_limit, Ordering::Release);
        metrics_decision(
            "probe",
            "hardware-headroom",
            previous_limit,
            current_limit,
            metrics,
            comparison,
        )
    }

    fn rollback_trial(
        &self,
        state: &mut AdaptiveGenerationConcurrencyState,
        trial: GenerationConcurrencyTrial,
        reason: &'static str,
    ) -> GenerationConcurrencyDecision {
        let previous_limit = self.current_limit();
        state.committed_limit = trial.committed_limit;
        state.baseline = None;
        state.cooldown_windows = REPROBE_COOLDOWN_WINDOWS;
        self.reduce_limit(previous_limit, trial.committed_limit);
        empty_decision("rollback", reason, previous_limit, trial.committed_limit)
    }

    fn reduce_limit(&self, previous_limit: usize, current_limit: usize) {
        let count = previous_limit.saturating_sub(current_limit);
        let mut retirement = self
            .retirement
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let retired_available = self.semaphore.forget_permits(count);
        retirement.pending = retirement
            .pending
            .saturating_add(count.saturating_sub(retired_available));
        self.current_limit.store(current_limit, Ordering::Release);
    }

    fn pending_retirements(&self) -> usize {
        self.retirement
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pending
    }
}

fn empty_decision(
    action: &'static str,
    reason: &'static str,
    previous_limit: usize,
    current_limit: usize,
) -> GenerationConcurrencyDecision {
    GenerationConcurrencyDecision {
        action,
        reason,
        previous_limit,
        current_limit,
        throughput_tokens_per_second: None,
        p95_latency_ms: None,
        p95_hardware_ms_per_token: None,
        throughput_improvement: None,
        p95_latency_ratio: None,
        hardware_pressure_ratio: None,
        observed_requests: None,
        saturated_requests: None,
    }
}

fn take_observation_counts(
    state: &mut AdaptiveGenerationConcurrencyState,
    mut decision: GenerationConcurrencyDecision,
) -> GenerationConcurrencyDecision {
    decision.observed_requests = Some(state.observed_requests);
    decision.saturated_requests = Some(state.saturated_requests);
    state.observed_requests = 0;
    state.saturated_requests = 0;
    decision
}

fn metrics_decision(
    action: &'static str,
    reason: &'static str,
    previous_limit: usize,
    current_limit: usize,
    metrics: GenerationConcurrencyMetrics,
    comparison: Option<GenerationConcurrencyComparison>,
) -> GenerationConcurrencyDecision {
    let mut decision = empty_decision(action, reason, previous_limit, current_limit);
    apply_metrics(&mut decision, metrics, comparison);
    decision
}

fn apply_metrics(
    decision: &mut GenerationConcurrencyDecision,
    metrics: GenerationConcurrencyMetrics,
    comparison: Option<GenerationConcurrencyComparison>,
) {
    decision.throughput_tokens_per_second = Some(metrics.throughput_tokens_per_second);
    decision.p95_latency_ms = Some(metrics.p95_latency_ms);
    decision.p95_hardware_ms_per_token = Some(metrics.p95_hardware_ms_per_token);
    decision.throughput_improvement = comparison.map(|value| value.throughput_improvement);
    decision.p95_latency_ratio = comparison.map(|value| value.p95_latency_ratio);
    decision.hardware_pressure_ratio = comparison.map(|value| value.hardware_pressure_ratio);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn observation(
        started_at: Instant,
        saturated: bool,
        at_capacity: bool,
        latency_ms: f64,
    ) -> GenerationConcurrencyObservation {
        GenerationConcurrencyObservation {
            completed_tokens: 100,
            executed_tokens: 100,
            latency_ms,
            saturated,
            at_capacity,
            started_at,
        }
    }

    struct ObservationClock {
        now: Instant,
    }

    impl ObservationClock {
        fn new() -> Self {
            Self {
                now: Instant::now(),
            }
        }

        fn observe_window(
            &mut self,
            controller: &GenerationConcurrencyController,
            completed_tokens: u64,
            executed_tokens: u64,
            latency_ms: f64,
            interval_ms: u64,
        ) -> GenerationConcurrencyDecision {
            (0..ADAPTIVE_WINDOW_REQUESTS)
                .find_map(|_| {
                    let started_at = self.now;
                    self.now += Duration::from_millis(interval_ms);
                    controller.observe_completed_at(
                        GenerationConcurrencyObservation {
                            completed_tokens,
                            executed_tokens,
                            latency_ms,
                            saturated: true,
                            at_capacity: false,
                            started_at,
                        },
                        self.now,
                    )
                })
                .expect("window decision")
        }
    }

    #[test]
    fn fixed_controller_exposes_the_hard_ceiling() {
        let controller = GenerationConcurrencyController::fixed(4);
        assert_eq!(controller.current_limit(), 4);
        assert_eq!(controller.hard_limit(), 4);
        assert!(
            controller
                .observe_completed(GenerationConcurrencyObservation {
                    completed_tokens: 8,
                    executed_tokens: 8,
                    latency_ms: 10.0,
                    saturated: true,
                    at_capacity: false,
                    started_at: Instant::now(),
                })
                .is_none()
        );
    }

    #[test]
    fn throughput_window_includes_the_first_requests_service_wall_time() {
        let started_at = Instant::now();
        let mut window = GenerationConcurrencyWindow::new();
        for _ in 0..ADAPTIVE_WINDOW_REQUESTS {
            window.push(100, 100, 100.0, started_at);
        }
        let metrics = window
            .metrics(started_at + Duration::from_secs(2))
            .expect("complete window");
        assert_eq!(metrics.throughput_tokens_per_second, 800.0);
    }

    #[test]
    fn healthy_probe_is_committed() {
        let controller = GenerationConcurrencyController::adaptive(4, 1);
        let mut clock = ObservationClock::new();
        let probe = clock.observe_window(&controller, 100, 100, 100.0, 100);
        assert_eq!(probe.action, "probe");
        assert_eq!(controller.current_limit(), 2);

        let commit = clock.observe_window(&controller, 100, 100, 105.0, 60);
        assert_eq!(commit.action, "commit");
        assert_eq!(controller.current_limit(), 2);
    }

    #[test]
    fn hardware_pressure_rolls_back_the_tentative_lane() {
        let controller = GenerationConcurrencyController::adaptive(4, 1);
        let mut clock = ObservationClock::new();
        let probe = clock.observe_window(&controller, 100, 100, 100.0, 100);
        assert_eq!(probe.action, "probe");

        let rollback = clock.observe_window(&controller, 100, 100, 130.0, 60);
        assert_eq!(rollback.action, "rollback");
        assert_eq!(rollback.reason, "hardware-backpressure");
        assert_eq!(controller.current_limit(), 1);
        assert_eq!(controller.available_permits(), 1);
    }

    #[test]
    fn trial_accepts_full_occupancy_without_queue_waiting() {
        let controller = GenerationConcurrencyController::adaptive(2, 1);
        let mut clock = ObservationClock::new();
        let probe = clock.observe_window(&controller, 100, 100, 100.0, 100);
        assert_eq!(probe.action, "probe");

        let commit = (0..ADAPTIVE_WINDOW_REQUESTS)
            .find_map(|_| {
                clock.now += Duration::from_millis(60);
                controller.observe_completed_at(
                    observation(clock.now - Duration::from_millis(60), false, true, 105.0),
                    clock.now,
                )
            })
            .expect("trial decision");
        assert_eq!(commit.action, "commit");
        assert_eq!(controller.current_limit(), 2);
    }

    #[test]
    fn rollback_drains_an_in_flight_permit() {
        let controller = GenerationConcurrencyController::adaptive(3, 1);
        let mut clock = ObservationClock::new();
        clock.observe_window(&controller, 100, 100, 100.0, 100);
        let first = controller.try_acquire_owned().expect("first trial lane");
        let second = controller.try_acquire_owned().expect("second trial lane");
        assert_eq!(controller.available_permits(), 0);

        let rollback = clock.observe_window(&controller, 100, 100, 130.0, 60);
        assert_eq!(rollback.action, "rollback");
        assert_eq!(controller.current_limit(), 1);
        drop(first);
        assert_eq!(controller.available_permits(), 0);
        drop(second);
        assert_eq!(controller.available_permits(), 1);
    }

    #[test]
    fn failure_rolls_back_a_live_probe() {
        let controller = GenerationConcurrencyController::adaptive(4, 1);
        let mut clock = ObservationClock::new();
        clock.observe_window(&controller, 100, 100, 100.0, 100);
        let decision = controller.observe_failed().expect("failure decision");
        assert_eq!(decision.action, "rollback");
        assert_eq!(decision.reason, "generation-failure");
        assert_eq!(controller.current_limit(), 1);
    }

    #[test]
    fn adaptive_seed_at_the_ceiling_can_back_off_after_failure() {
        let controller = GenerationConcurrencyController::adaptive(4, 4);
        let decision = controller.observe_failed().expect("failure decision");
        assert_eq!(decision.action, "backoff");
        assert_eq!(decision.reason, "generation-failure");
        assert_eq!(controller.current_limit(), 3);
    }

    #[test]
    fn queued_demand_epoch_survives_a_queue_handoff_before_completion() {
        let controller = GenerationConcurrencyController::adaptive(2, 1);
        let demand_epoch = controller.demand_epoch();
        assert!(!controller.was_saturated_since(demand_epoch, false, false));

        controller.note_queued_demand();
        assert!(controller.was_saturated_since(demand_epoch, false, false));
        assert!(controller.was_saturated_since(controller.demand_epoch(), true, false));
    }

    #[test]
    fn controller_reprobes_after_rollback_cooldown() {
        let controller = GenerationConcurrencyController::adaptive(4, 1);
        let mut clock = ObservationClock::new();
        clock.observe_window(&controller, 100, 100, 100.0, 100);
        clock.observe_window(&controller, 100, 100, 130.0, 60);

        let first_hold = clock.observe_window(&controller, 100, 100, 100.0, 100);
        let second_hold = clock.observe_window(&controller, 100, 100, 100.0, 100);
        let reprobe = clock.observe_window(&controller, 100, 100, 100.0, 100);
        assert_eq!(first_hold.reason, "reprobe-cooldown");
        assert_eq!(second_hold.reason, "reprobe-cooldown");
        assert_eq!(reprobe.action, "probe");
        assert_eq!(controller.current_limit(), 2);
    }

    #[test]
    fn idle_timeout_cancels_a_tentative_probe() {
        let controller = GenerationConcurrencyController::adaptive(4, 1);
        let mut clock = ObservationClock::new();
        clock.observe_window(&controller, 100, 100, 100.0, 100);
        clock.now += Duration::from_millis(100);
        assert!(
            controller
                .observe_completed_at(
                    observation(clock.now - Duration::from_millis(100), false, false, 100.0,),
                    clock.now,
                )
                .is_none()
        );
        clock.now += DEMAND_IDLE_RESET;
        let decision = controller
            .observe_completed_at(
                observation(clock.now - Duration::from_millis(100), false, false, 100.0),
                clock.now,
            )
            .expect("trial rollback");
        assert_eq!(decision.action, "rollback");
        assert_eq!(decision.reason, "trial-demand-ended");
        assert_eq!(controller.current_limit(), 1);
    }

    #[test]
    fn one_queue_handoff_gap_does_not_erase_a_saturated_window() {
        let controller = GenerationConcurrencyController::adaptive(2, 1);
        let mut clock = ObservationClock::new();
        for _ in 0..(ADAPTIVE_WINDOW_REQUESTS / 2) {
            clock.now += Duration::from_millis(100);
            assert!(
                controller
                    .observe_completed_at(
                        observation(clock.now - Duration::from_millis(100), true, false, 100.0,),
                        clock.now,
                    )
                    .is_none()
            );
        }
        clock.now += Duration::from_millis(100);
        assert!(
            controller
                .observe_completed_at(
                    observation(clock.now - Duration::from_millis(100), false, false, 100.0,),
                    clock.now,
                )
                .is_none()
        );
        let decision = (0..ADAPTIVE_WINDOW_REQUESTS)
            .find_map(|_| {
                clock.now += Duration::from_millis(100);
                controller
                    .observe_completed_at(
                        observation(clock.now - Duration::from_millis(100), true, false, 100.0),
                        clock.now,
                    )
                    .filter(|decision| decision.action == "probe")
            })
            .expect("probe decision");
        assert_eq!(decision.action, "probe");
        assert_eq!(controller.current_limit(), 2);
    }

    #[test]
    fn insufficient_demand_is_reported_without_opening_a_lane() {
        let controller = GenerationConcurrencyController::adaptive(2, 1);
        let mut clock = ObservationClock::new();
        let decision = (0..ADAPTIVE_WINDOW_REQUESTS)
            .find_map(|_| {
                clock.now += Duration::from_millis(100);
                controller.observe_completed_at(
                    observation(clock.now - Duration::from_millis(100), false, false, 100.0),
                    clock.now,
                )
            })
            .expect("demand observation");
        assert_eq!(decision.action, "hold");
        assert_eq!(decision.reason, "insufficient-sustained-demand");
        assert_eq!(decision.observed_requests, Some(ADAPTIVE_WINDOW_REQUESTS));
        assert_eq!(decision.saturated_requests, Some(0));
        assert_eq!(controller.current_limit(), 1);
    }
}
