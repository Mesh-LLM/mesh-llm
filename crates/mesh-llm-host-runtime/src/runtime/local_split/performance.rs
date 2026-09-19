//! Closed-loop layer rebalancing for `--performance-aware` splits.
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
pub(super) struct PerformanceControllerConfig {
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

impl PerformanceControllerConfig {
    /// Defaults, with the window and cooldown overridable for experiments via
    /// `MESH_LLM_PERFORMANCE_WINDOW_SECS` and `MESH_LLM_PERFORMANCE_COOLDOWN_SECS`.
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
            min_window: secs("MESH_LLM_PERFORMANCE_WINDOW_SECS").unwrap_or(defaults.min_window),
            cooldown: secs("MESH_LLM_PERFORMANCE_COOLDOWN_SECS").unwrap_or(defaults.cooldown),
            ..defaults
        }
    }
}

impl Default for PerformanceControllerConfig {
    fn default() -> Self {
        Self {
            min_window: Duration::from_secs(60),
            min_decode_tokens_per_second: 1.0,
            imbalance_threshold: 0.15,
            required_windows: 2,
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
pub(super) struct PerformanceSample {
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
pub(super) enum PerformanceDecision {
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
}

#[derive(Debug)]
pub(super) struct PerformanceController {
    config: PerformanceControllerConfig,
    last: Option<PerformanceSample>,
    imbalanced_windows: u32,
    cooldown_until: Option<Instant>,
    trial: Option<Trial>,
}

impl PerformanceController {
    pub(super) fn new(config: PerformanceControllerConfig) -> Self {
        Self {
            config,
            last: None,
            imbalanced_windows: 0,
            cooldown_until: None,
            trial: None,
        }
    }

    /// Feed the latest cumulative counters and get a decision.
    pub(super) fn observe(&mut self, sample: PerformanceSample) -> PerformanceDecision {
        let Some(last) = self.last.as_ref() else {
            self.last = Some(sample);
            return PerformanceDecision::Hold {
                why: "first sample",
            };
        };
        if !same_shape(last, &sample) {
            // The topology changed underneath us (a move, a replan, a
            // restart): start a fresh window on the new shape.
            self.last = Some(sample);
            return PerformanceDecision::Hold {
                why: "topology changed; new window",
            };
        }
        let elapsed = sample.at.saturating_duration_since(last.at);
        if elapsed < self.config.min_window {
            return PerformanceDecision::Hold {
                why: "window not yet full",
            };
        }
        let measurement = measure(last, &sample, elapsed);
        self.last = Some(sample.clone());

        if measurement.decode_tokens_per_second < self.config.min_decode_tokens_per_second {
            self.imbalanced_windows = 0;
            return PerformanceDecision::Hold { why: "idle" };
        }

        if let Some(trial) = self.trial.take() {
            self.cooldown_until = Some(sample.at + self.config.cooldown);
            self.imbalanced_windows = 0;
            let observed = measurement.decode_tokens_per_second;
            let floor = trial.baseline_tokens_per_second * (1.0 - self.config.rollback_margin);
            return if observed < floor {
                PerformanceDecision::Rollback {
                    boundaries: trial.previous_boundaries,
                    baseline: trial.baseline_tokens_per_second,
                    observed,
                }
            } else {
                PerformanceDecision::Accept {
                    baseline: trial.baseline_tokens_per_second,
                    observed,
                }
            };
        }

        if self.cooldown_until.is_some_and(|until| sample.at < until) {
            return PerformanceDecision::Hold {
                why: "cooling down after a move",
            };
        }
        if measurement.bytes_per_second.contains(&0) {
            self.imbalanced_windows = 0;
            return PerformanceDecision::Hold {
                why: "a stage reported no busy time",
            };
        }
        let (min, max) = utilization_range(&measurement.utilization);
        if max - min < self.config.imbalance_threshold {
            self.imbalanced_windows = 0;
            return PerformanceDecision::Hold { why: "balanced" };
        }
        self.imbalanced_windows += 1;
        if self.imbalanced_windows < self.config.required_windows {
            return PerformanceDecision::Hold {
                why: "imbalance not yet sustained",
            };
        }
        self.imbalanced_windows = 0;
        PerformanceDecision::Rebalance { measurement }
    }

    /// The coordinator applied a move proposed by `Rebalance`. The next full
    /// window is judged against `baseline` and may roll back to `previous`.
    pub(super) fn note_moved(&mut self, baseline: f64, previous: Vec<(u32, u32)>) {
        self.trial = Some(Trial {
            baseline_tokens_per_second: baseline,
            previous_boundaries: previous,
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

fn same_shape(left: &PerformanceSample, right: &PerformanceSample) -> bool {
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
    last: &PerformanceSample,
    now: &PerformanceSample,
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

    fn config() -> PerformanceControllerConfig {
        PerformanceControllerConfig {
            min_window: Duration::from_secs(60),
            cooldown: Duration::from_secs(300),
            ..PerformanceControllerConfig::default()
        }
    }

    /// Two stages; `busy_share` is each stage's busy fraction of wall time.
    fn sample(
        at: Instant,
        cut: u32,
        seconds_elapsed: u64,
        busy_share: (f64, f64),
        tokens_per_second: f64,
    ) -> PerformanceSample {
        let nanos = |share: f64| (share * seconds_elapsed as f64 * 1e9) as u64;
        PerformanceSample {
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
        let mut controller = PerformanceController::new(config());
        let at = |s| t0 + Duration::from_secs(s);
        assert!(matches!(
            controller.observe(sample(at(0), 18, 0, (0.93, 0.51), 21.0)),
            PerformanceDecision::Hold { .. }
        ));
        assert!(matches!(
            controller.observe(sample(at(60), 18, 60, (0.93, 0.51), 21.0)),
            PerformanceDecision::Hold {
                why: "imbalance not yet sustained"
            }
        ));
        let PerformanceDecision::Rebalance { measurement } =
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
        let mut controller = PerformanceController::new(config());
        controller.observe(sample(t0, 12, 0, (0.0, 0.0), 0.0));
        assert_eq!(
            controller.observe(sample(
                t0 + Duration::from_secs(60),
                12,
                60,
                (0.90, 0.85),
                30.0
            )),
            PerformanceDecision::Hold { why: "balanced" }
        );
        assert_eq!(
            controller.observe(sample(
                t0 + Duration::from_secs(120),
                12,
                120,
                (0.90, 0.85),
                15.0 * 0.0
            )),
            PerformanceDecision::Hold {
                why: "topology changed; new window"
            },
            "counters going backwards restart the window"
        );
        let mut idle = PerformanceController::new(config());
        idle.observe(sample(t0, 12, 0, (0.0, 0.0), 0.0));
        assert_eq!(
            idle.observe(sample(
                t0 + Duration::from_secs(60),
                12,
                60,
                (0.01, 0.30),
                0.2
            )),
            PerformanceDecision::Hold { why: "idle" }
        );
    }

    #[test]
    fn a_move_that_holds_is_accepted_and_one_that_hurts_rolls_back() {
        let t0 = Instant::now();
        let at = |s| t0 + Duration::from_secs(s);

        let mut good = PerformanceController::new(config());
        good.note_moved(21.0, vec![(0, 18), (18, 36)]);
        good.observe(sample(at(0), 12, 0, (0.0, 0.0), 0.0));
        assert!(matches!(
            good.observe(sample(at(60), 12, 60, (0.88, 0.86), 31.0)),
            PerformanceDecision::Accept { .. }
        ));
        assert_eq!(
            good.observe(sample(at(120), 12, 120, (0.95, 0.60), 31.0)),
            PerformanceDecision::Hold {
                why: "cooling down after a move"
            }
        );

        let mut bad = PerformanceController::new(config());
        bad.note_moved(21.0, vec![(0, 18), (18, 36)]);
        bad.observe(sample(at(0), 30, 0, (0.0, 0.0), 0.0));
        assert_eq!(
            bad.observe(sample(at(60), 30, 60, (0.99, 0.10), 12.0)),
            PerformanceDecision::Rollback {
                boundaries: vec![(0, 18), (18, 36)],
                baseline: 21.0,
                observed: 12.0,
            }
        );
    }

    #[test]
    fn a_short_window_waits() {
        let t0 = Instant::now();
        let mut controller = PerformanceController::new(config());
        controller.observe(sample(t0, 18, 0, (0.0, 0.0), 0.0));
        assert_eq!(
            controller.observe(sample(
                t0 + Duration::from_secs(30),
                18,
                30,
                (0.93, 0.51),
                21.0
            )),
            PerformanceDecision::Hold {
                why: "window not yet full"
            }
        );
    }
}
