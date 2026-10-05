//! Closed-loop layer rebalancing for `--auto-balance` splits.
//!
//! Every token passes through every stage, so over a window each stage does
//! the same logical work and the stage with the most compute-busy time paces
//! the pipeline. A stage's measured rate — the weight bytes it holds divided
//! by its busy time — is the speed the planner balances, which captures what
//! a bandwidth estimate misses (host overheads on the stage-0 node, thermal
//! throttling, other work on the machine).
//!
//! The controller only decides. It proposes a rebalance after a sustained
//! imbalance under real load, then judges the move by the decode throughput
//! that follows, rolling back a move that made things worse. The coordinator
//! owns planning and the drain/re-cut/resume.

use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct AutoBalanceControllerConfig {
    /// Shortest window a decision is made on.
    pub(super) min_window: Duration,
    /// Decode throughput below which the split is treated as idle.
    pub(super) min_decode_tokens_per_second: f64,
    /// Busy-share gap between the busiest and least busy stage that counts as
    /// imbalance, in 0..1 of wall time.
    pub(super) imbalance_threshold: f64,
    /// Consecutive imbalanced windows required before proposing a move.
    pub(super) required_windows: u32,
    /// Quiet period after a move is accepted or rolled back.
    pub(super) cooldown: Duration,
    /// Fractional throughput drop, relative to the pre-move baseline, that
    /// rolls a move back. Set above run-to-run noise.
    pub(super) rollback_margin: f64,
}

impl AutoBalanceControllerConfig {
    /// Defaults, with the window and cooldown overridable for experiments via
    /// `MESH_LLM_AUTO_BALANCE_WINDOW_SECS` and `MESH_LLM_AUTO_BALANCE_COOLDOWN_SECS`.
    pub(super) fn from_env() -> Self {
        let secs = |name: &str| {
            std::env::var(name)
                .ok()
                .and_then(|value| value.trim().parse::<u64>().ok())
                .filter(|secs| *secs > 0)
                .map(Duration::from_secs)
        };
        let defaults = Self::default();
        Self {
            min_window: secs("MESH_LLM_AUTO_BALANCE_WINDOW_SECS").unwrap_or(defaults.min_window),
            cooldown: secs("MESH_LLM_AUTO_BALANCE_COOLDOWN_SECS").unwrap_or(defaults.cooldown),
            ..defaults
        }
    }
}

impl Default for AutoBalanceControllerConfig {
    fn default() -> Self {
        Self {
            min_window: Duration::from_secs(60),
            min_decode_tokens_per_second: 1.0,
            // Sustained runs vary about +/-15% window to window; act only on
            // imbalance and gains well above that.
            imbalance_threshold: 0.20,
            required_windows: 3,
            cooldown: Duration::from_secs(300),
            rollback_margin: 0.10,
        }
    }
}

/// Cumulative counters for one stage at one instant.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct StageBusy {
    pub(super) layer_start: u32,
    pub(super) layer_end: u32,
    /// Weight bytes resident on the stage.
    pub(super) weight_bytes: u64,
    /// Cumulative compute-busy nanoseconds.
    pub(super) busy_nanos: u64,
}

/// One observation of the running split, stage order preserved.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct AutoBalanceSample {
    pub(super) at: Instant,
    pub(super) stages: Vec<StageBusy>,
    /// Cumulative decode tokens produced by the split.
    pub(super) decode_tokens: u64,
}

/// What the measured window says about the split.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct WindowMeasurement {
    pub(super) seconds: f64,
    pub(super) decode_tokens_per_second: f64,
    /// Busy share of wall time per stage.
    pub(super) utilization: Vec<f64>,
    /// Measured decode rate per stage in weight bytes per busy second.
    pub(super) bytes_per_second: Vec<u64>,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum AutoBalanceDecision {
    /// Keep sampling; `why` is for logs.
    Hold { why: &'static str },
    /// Re-plan layer boundaries with these measured per-stage rates.
    Rebalance { measurement: WindowMeasurement },
    /// The last move held up; keep it.
    Accept { baseline: f64, observed: f64 },
    /// The last move cost throughput; restore these boundaries.
    Rollback {
        boundaries: Vec<(u32, u32)>,
        baseline: f64,
        observed: f64,
    },
}

#[derive(Clone, Debug, PartialEq)]
struct Trial {
    baseline_tokens_per_second: f64,
    previous_boundaries: Vec<(u32, u32)>,
    /// Windows skipped because the load was too low to judge the move.
    deferred_windows: u32,
}

/// A trial window must reach this fraction of the baseline decode rate to be
/// judged; below it the window is mostly idle and the verdict would measure
/// the idle share, not the move.
const TRIAL_JUDGE_LOAD_FRACTION: f64 = 0.5;

/// Judge the trial anyway after this many partial-load windows, so a move
/// cannot sit unjudged forever under a permanently reduced load.
const TRIAL_MAX_DEFERRED_WINDOWS: u32 = 3;

#[derive(Debug)]
pub(super) struct AutoBalanceController {
    config: AutoBalanceControllerConfig,
    last: Option<AutoBalanceSample>,
    imbalanced_windows: u32,
    cooldown_until: Option<Instant>,
    trial: Option<Trial>,
}

impl AutoBalanceController {
    pub(super) fn new(config: AutoBalanceControllerConfig) -> Self {
        Self {
            config,
            last: None,
            imbalanced_windows: 0,
            cooldown_until: None,
            trial: None,
        }
    }

    /// Feed the latest cumulative counters and get a decision.
    pub(super) fn observe(&mut self, sample: AutoBalanceSample) -> AutoBalanceDecision {
        let Some(last) = self.last.as_ref() else {
            self.last = Some(sample);
            return AutoBalanceDecision::Hold {
                why: "first sample",
            };
        };
        if !same_shape(last, &sample) {
            // The topology changed underneath us (a move, a replan, a
            // restart): start a fresh window on the new shape. A pending
            // trial was measured on the old shape, so its next window would
            // judge the old move against the new generation and could issue
            // an unrelated rollback. Cancel it and re-learn the new shape.
            self.last = Some(sample);
            self.trial = None;
            self.imbalanced_windows = 0;
            return AutoBalanceDecision::Hold {
                why: "topology changed; new window",
            };
        }
        let elapsed = sample.at.saturating_duration_since(last.at);
        if elapsed < self.config.min_window {
            return AutoBalanceDecision::Hold {
                why: "window not yet full",
            };
        }
        let measurement = measure(last, &sample, elapsed);
        self.last = Some(sample.clone());

        if measurement.decode_tokens_per_second < self.config.min_decode_tokens_per_second {
            self.imbalanced_windows = 0;
            return AutoBalanceDecision::Hold { why: "idle" };
        }

        if let Some(trial) = self.trial.take() {
            let observed = measurement.decode_tokens_per_second;
            let judge_floor = trial.baseline_tokens_per_second * TRIAL_JUDGE_LOAD_FRACTION;
            if observed < judge_floor && trial.deferred_windows < TRIAL_MAX_DEFERRED_WINDOWS {
                // A partially loaded window measures the idle share, not the
                // move: a 21 tok/s baseline judged over 30s idle + 30s busy
                // reads ~10.5 and would roll back a move whose steady state
                // is better. Defer to the next full window.
                self.trial = Some(Trial {
                    deferred_windows: trial.deferred_windows + 1,
                    ..trial
                });
                return AutoBalanceDecision::Hold {
                    why: "trial window partially loaded; deferring judgment",
                };
            }
            self.cooldown_until = Some(sample.at + self.config.cooldown);
            self.imbalanced_windows = 0;
            let floor = trial.baseline_tokens_per_second * (1.0 - self.config.rollback_margin);
            return if observed < floor {
                AutoBalanceDecision::Rollback {
                    boundaries: trial.previous_boundaries,
                    baseline: trial.baseline_tokens_per_second,
                    observed,
                }
            } else {
                AutoBalanceDecision::Accept {
                    baseline: trial.baseline_tokens_per_second,
                    observed,
                }
            };
        }

        if self.cooldown_until.is_some_and(|until| sample.at < until) {
            return AutoBalanceDecision::Hold {
                why: "cooling down after a move",
            };
        }
        if measurement.bytes_per_second.contains(&0) {
            self.imbalanced_windows = 0;
            return AutoBalanceDecision::Hold {
                why: "a stage reported no busy time",
            };
        }
        let (min, max) = utilization_range(&measurement.utilization);
        if max - min < self.config.imbalance_threshold {
            self.imbalanced_windows = 0;
            return AutoBalanceDecision::Hold { why: "balanced" };
        }
        self.imbalanced_windows += 1;
        if self.imbalanced_windows < self.config.required_windows {
            return AutoBalanceDecision::Hold {
                why: "imbalance not yet sustained",
            };
        }
        self.imbalanced_windows = 0;
        AutoBalanceDecision::Rebalance { measurement }
    }

    /// The coordinator applied a move proposed by `Rebalance`. The next full
    /// window is judged against `baseline` and may roll back to `previous`.
    pub(super) fn note_moved(&mut self, baseline: f64, previous: Vec<(u32, u32)>) {
        self.trial = Some(Trial {
            baseline_tokens_per_second: baseline,
            previous_boundaries: previous,
            deferred_windows: 0,
        });
        self.last = None;
    }

    /// The coordinator declined or failed to apply a proposed move.
    pub(super) fn note_not_moved(&mut self, at: Instant) {
        self.cooldown_until = Some(at + self.config.cooldown);
    }

    /// A rollback was applied; wait out the cooldown before trying again.
    pub(super) fn note_rolled_back(&mut self, at: Instant) {
        self.trial = None;
        self.last = None;
        self.cooldown_until = Some(at + self.config.cooldown);
    }
}

/// Move each internal stage boundary at most halfway toward `target`, and
/// always at least one layer when it moves, keeping every stage non-empty.
///
/// The measured-rate model treats a stage's time as proportional to its
/// weight bytes, but part of each step is fixed (output head, sampling,
/// activation I/O). Measured from a far-off cut that error sends a single
/// full jump past the optimum; halving the step converges without
/// oscillating, and each step is re-measured before the next.
pub(super) fn damped_boundaries(current: &[(u32, u32)], target: &[(u32, u32)]) -> Vec<(u32, u32)> {
    if current.len() != target.len() || current.is_empty() {
        return current.to_vec();
    }
    let layer_end = current.last().map(|(_, end)| *end).unwrap_or(0);
    let mut ends = Vec::with_capacity(current.len());
    for (index, ((_, now), (_, goal))) in current.iter().zip(target).enumerate() {
        if index + 1 == current.len() {
            ends.push(layer_end);
            break;
        }
        let delta = i64::from(*goal) - i64::from(*now);
        let step = if delta == 0 {
            0
        } else {
            (delta / 2).signum() * (delta.abs() / 2).max(1)
        };
        let step = if delta != 0 && step == 0 {
            delta.signum()
        } else {
            step
        };
        let lowest = ends.last().map(|end| end + 1).unwrap_or(1);
        let highest = layer_end - (current.len() - 1 - index) as u32;
        ends.push(((i64::from(*now) + step).clamp(i64::from(lowest), i64::from(highest))) as u32);
    }
    let mut start = 0;
    ends.into_iter()
        .map(|end| {
            let range = (start, end);
            start = end;
            range
        })
        .collect()
}

fn same_shape(left: &AutoBalanceSample, right: &AutoBalanceSample) -> bool {
    left.stages.len() == right.stages.len()
        && left
            .stages
            .iter()
            .zip(&right.stages)
            .all(|(a, b)| a.layer_start == b.layer_start && a.layer_end == b.layer_end)
        && right.decode_tokens >= left.decode_tokens
        && left
            .stages
            .iter()
            .zip(&right.stages)
            .all(|(a, b)| b.busy_nanos >= a.busy_nanos)
}

fn measure(
    last: &AutoBalanceSample,
    now: &AutoBalanceSample,
    elapsed: Duration,
) -> WindowMeasurement {
    let seconds = elapsed.as_secs_f64().max(f64::MIN_POSITIVE);
    let decode_tokens_per_second = (now.decode_tokens - last.decode_tokens) as f64 / seconds;
    let (utilization, bytes_per_second) = last
        .stages
        .iter()
        .zip(&now.stages)
        .map(|(before, after)| {
            let busy = after.busy_nanos - before.busy_nanos;
            let utilization = busy as f64 / 1e9 / seconds;
            let rate = if busy == 0 {
                0
            } else {
                (u128::from(after.weight_bytes) * 1_000_000_000 / u128::from(busy))
                    .min(u128::from(u64::MAX)) as u64
            };
            (utilization, rate)
        })
        .unzip();
    WindowMeasurement {
        seconds,
        decode_tokens_per_second,
        utilization,
        bytes_per_second,
    }
}

fn utilization_range(utilization: &[f64]) -> (f64, f64) {
    utilization
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), value| {
            (min.min(*value), max.max(*value))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    const LAYER_BYTES: u64 = 130_000_000;

    fn config() -> AutoBalanceControllerConfig {
        AutoBalanceControllerConfig {
            min_window: Duration::from_secs(60),
            cooldown: Duration::from_secs(300),
            imbalance_threshold: 0.15,
            required_windows: 2,
            ..AutoBalanceControllerConfig::default()
        }
    }

    /// Two stages; `busy_share` is each stage's busy fraction of wall time.
    fn sample(
        at: Instant,
        cut: u32,
        seconds_elapsed: u64,
        busy_share: (f64, f64),
        tokens_per_second: f64,
    ) -> AutoBalanceSample {
        let nanos = |share: f64| (share * seconds_elapsed as f64 * 1e9) as u64;
        AutoBalanceSample {
            at,
            stages: vec![
                StageBusy {
                    layer_start: 0,
                    layer_end: cut,
                    weight_bytes: u64::from(cut) * LAYER_BYTES,
                    busy_nanos: nanos(busy_share.0),
                },
                StageBusy {
                    layer_start: cut,
                    layer_end: 36,
                    weight_bytes: u64::from(36 - cut) * LAYER_BYTES,
                    busy_nanos: nanos(busy_share.1),
                },
            ],
            decode_tokens: (tokens_per_second * seconds_elapsed as f64) as u64,
        }
    }

    #[test]
    fn sustained_imbalance_proposes_a_rebalance_with_measured_rates() {
        // The 18/18 profile: stage 0 (M1) 93% busy, stage 1 (M4) 51% busy.
        let t0 = Instant::now();
        let mut controller = AutoBalanceController::new(config());
        let at = |s| t0 + Duration::from_secs(s);
        assert!(matches!(
            controller.observe(sample(at(0), 18, 0, (0.93, 0.51), 21.0)),
            AutoBalanceDecision::Hold { .. }
        ));
        assert!(matches!(
            controller.observe(sample(at(60), 18, 60, (0.93, 0.51), 21.0)),
            AutoBalanceDecision::Hold {
                why: "imbalance not yet sustained"
            }
        ));
        let AutoBalanceDecision::Rebalance { measurement } =
            controller.observe(sample(at(120), 18, 120, (0.93, 0.51), 21.0))
        else {
            panic!("expected a rebalance after two imbalanced windows");
        };
        // Same weight on both stages, so the rate ratio is the busy ratio.
        let ratio = measurement.bytes_per_second[1] as f64 / measurement.bytes_per_second[0] as f64;
        assert!((ratio - 0.93 / 0.51).abs() < 0.01, "{ratio}");
        assert!((measurement.decode_tokens_per_second - 21.0).abs() < 0.5);
    }

    #[test]
    fn balanced_or_idle_splits_hold() {
        let t0 = Instant::now();
        let mut controller = AutoBalanceController::new(config());
        controller.observe(sample(t0, 12, 0, (0.0, 0.0), 0.0));
        assert_eq!(
            controller.observe(sample(
                t0 + Duration::from_secs(60),
                12,
                60,
                (0.90, 0.85),
                30.0
            )),
            AutoBalanceDecision::Hold { why: "balanced" }
        );
        assert_eq!(
            controller.observe(sample(
                t0 + Duration::from_secs(120),
                12,
                120,
                (0.90, 0.85),
                15.0 * 0.0
            )),
            AutoBalanceDecision::Hold {
                why: "topology changed; new window"
            },
            "counters going backwards restart the window"
        );
        let mut idle = AutoBalanceController::new(config());
        idle.observe(sample(t0, 12, 0, (0.0, 0.0), 0.0));
        assert_eq!(
            idle.observe(sample(
                t0 + Duration::from_secs(60),
                12,
                60,
                (0.01, 0.30),
                0.2
            )),
            AutoBalanceDecision::Hold { why: "idle" }
        );
    }

    #[test]
    fn a_move_that_holds_is_accepted_and_one_that_hurts_rolls_back() {
        let t0 = Instant::now();
        let at = |s| t0 + Duration::from_secs(s);

        let mut good = AutoBalanceController::new(config());
        good.note_moved(21.0, vec![(0, 18), (18, 36)]);
        good.observe(sample(at(0), 12, 0, (0.0, 0.0), 0.0));
        assert!(matches!(
            good.observe(sample(at(60), 12, 60, (0.88, 0.86), 31.0)),
            AutoBalanceDecision::Accept { .. }
        ));
        assert_eq!(
            good.observe(sample(at(120), 12, 120, (0.95, 0.60), 31.0)),
            AutoBalanceDecision::Hold {
                why: "cooling down after a move"
            }
        );

        let mut bad = AutoBalanceController::new(config());
        bad.note_moved(21.0, vec![(0, 18), (18, 36)]);
        bad.observe(sample(at(0), 30, 0, (0.0, 0.0), 0.0));
        assert_eq!(
            bad.observe(sample(at(60), 30, 60, (0.99, 0.10), 12.0)),
            AutoBalanceDecision::Rollback {
                boundaries: vec![(0, 18), (18, 36)],
                baseline: 21.0,
                observed: 12.0,
            }
        );
    }

    #[test]
    fn damping_moves_halfway_and_at_least_one_layer() {
        // 35/1 toward 12/24 moves to 24/12, then 18/18, then 15/21 ...
        assert_eq!(
            damped_boundaries(&[(0, 35), (35, 36)], &[(0, 12), (12, 36)]),
            vec![(0, 24), (24, 36)]
        );
        // One layer away still moves one layer.
        assert_eq!(
            damped_boundaries(&[(0, 13), (13, 36)], &[(0, 12), (12, 36)]),
            vec![(0, 12), (12, 36)]
        );
        // Already there: no change.
        assert_eq!(
            damped_boundaries(&[(0, 12), (12, 36)], &[(0, 12), (12, 36)]),
            vec![(0, 12), (12, 36)]
        );
    }

    #[test]
    fn damping_keeps_every_stage_non_empty() {
        let damped = damped_boundaries(&[(0, 2), (2, 3), (3, 10)], &[(0, 9), (9, 9), (9, 10)]);
        assert!(damped.iter().all(|(start, end)| end > start), "{damped:?}");
        assert_eq!(damped.last().unwrap().1, 10);
    }

    #[test]
    fn a_short_window_waits() {
        let t0 = Instant::now();
        let mut controller = AutoBalanceController::new(config());
        controller.observe(sample(t0, 18, 0, (0.0, 0.0), 0.0));
        assert_eq!(
            controller.observe(sample(
                t0 + Duration::from_secs(30),
                18,
                30,
                (0.93, 0.51),
                21.0
            )),
            AutoBalanceDecision::Hold {
                why: "window not yet full"
            }
        );
    }

    #[test]
    fn a_partially_loaded_trial_window_defers_judgment() {
        let t0 = Instant::now();
        let at = |s| t0 + Duration::from_secs(s);
        let mut controller = AutoBalanceController::new(config());
        controller.note_moved(21.0, vec![(0, 18), (18, 36)]);
        controller.observe(sample(at(0), 12, 0, (0.0, 0.0), 0.0));
        // Half the window idle: the 21 tok/s baseline judged over 30s idle
        // plus 30s busy reads ~9 tok/s and must not roll back the move.
        assert_eq!(
            controller.observe(sample(at(60), 12, 60, (0.90, 0.30), 9.0)),
            AutoBalanceDecision::Hold {
                why: "trial window partially loaded; deferring judgment"
            }
        );
        // The next window at full load judges the move on its merits.
        assert!(matches!(
            controller.observe(sample(at(120), 12, 120, (0.90, 0.30), 31.0)),
            AutoBalanceDecision::Accept { .. }
        ));
    }

    #[test]
    fn a_topology_change_mid_trial_cancels_the_trial() {
        let t0 = Instant::now();
        let at = |s| t0 + Duration::from_secs(s);
        let mut controller = AutoBalanceController::new(config());
        controller.note_moved(21.0, vec![(0, 18), (18, 36)]);
        // First post-move sample starts the trial window on the new shape.
        controller.observe(sample(at(0), 30, 0, (0.0, 0.0), 0.0));
        // An unrelated replan changes the topology while the trial is open.
        assert_eq!(
            controller.observe(sample(at(60), 24, 60, (0.90, 0.30), 4.0)),
            AutoBalanceDecision::Hold {
                why: "topology changed; new window"
            }
        );
        // The next window is judged on the new generation, not against the
        // cancelled trial: a poor measurement must not roll back to the old
        // boundaries of a move that no longer exists.
        assert_eq!(
            controller.observe(sample(at(120), 24, 120, (0.90, 0.30), 5.0)),
            AutoBalanceDecision::Hold {
                why: "imbalance not yet sustained"
            }
        );
    }
}
