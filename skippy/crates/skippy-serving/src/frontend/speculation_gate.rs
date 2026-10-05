//! Closed-loop speculation gating: measure whether speculation pays, do not predict it.
//!
//! The N-gram proposers pay off on input-grounded output and can lose on
//! freeform prose. Both are measured: a re-emit workload ran +427% over
//! target-only decode, and a freeform workload on a two-node split ran
//! 12.2–12.4 tok/s against 13.8 for plain decode. Nothing in a config file
//! says which regime a deployment is in, and the same deployment can be in
//! both at different times of day.
//!
//! # Why this is a controller and not a formula
//!
//! `WAN_SPLIT_PERF.md` gives a closed form: speculation pays when
//! `C_verify + 2·RTT/k_accepted < C_total + 2·RTT`. Tempting, and not
//! evaluable here. `k_accepted` and the per-window verify cost are both
//! measured, but `C_total` — plain decode time for all layers, under the same
//! load, on the same hardware — is not, and cannot be: while speculation is on
//! no plain decode happens. `C_verify` is not separable either, because the
//! measured verify time includes the round trip the formula is trying to
//! amortise against.
//!
//! So the gate needs a quantity that is unobservable precisely when it would
//! be consulted. Feeding it a guessed `C_total` would be a tuning constant
//! wearing a derivation. Instead this flips the setting and compares observed
//! completion throughput — the thing being optimised — which needs no constant.
//!
//! # The failure mode this must not have
//!
//! `WAN_SPLIT_PERF.md` records a run with `window_shrinks 0` across 231 early
//! rejections: an adaptive policy that never adapted, so it kept proposing
//! deep, kept rejecting, and kept paying three round trips to commit what
//! plain decode commits in one. A gate that cannot be shown to stand down
//! under sustained rejection is that bug again.
//! [`tests::a_losing_configuration_is_stood_down`] is that proof.
//!
//! # Why acceptance does not trigger the trial
//!
//! Acceptance looks like the obvious cheap signal and is not usable as one.
//! The losing run above accepted 660 of 2214 drafts — 29.8%, which is not a
//! low number. Meanwhile #1037 measured `simple` at 53.2% acceptance beating
//! `cache` at 67.1% on throughput, because it drafted more tokens in absolute
//! terms. Acceptance percentage alone does not rank a proposer, so no
//! threshold on it separates the configurations that pay from the ones that
//! do not.
//!
//! So the trial is periodic instead: every cooldown the gate flips the
//! setting, measures one window, and keeps the winner. One window in thirty
//! spent on the other setting is the price of not guessing. Acceptance is
//! still recorded on every window, because it is what makes a verdict
//! interpretable afterwards — it just does not decide when to look.

use std::time::{Duration, Instant};

/// Thresholds for [`SpeculationGate`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SpeculationGateConfig {
    /// Shortest window a decision is made on.
    pub(crate) min_window: Duration,
    /// Completion throughput below which the model is treated as idle. An idle
    /// window measures the idle share, not the setting.
    pub(crate) min_completion_tokens_per_second: f64,
    /// Fractional throughput change needed to call a trial decisive. Must sit
    /// above run-to-run noise, which sustained split runs showed at roughly
    /// ±8% window to window.
    pub(crate) decisive_margin: f64,
    /// Quiet period after a verdict, so the gate cannot oscillate.
    pub(crate) cooldown: Duration,
}

impl Default for SpeculationGateConfig {
    fn default() -> Self {
        Self {
            min_window: Duration::from_secs(60),
            min_completion_tokens_per_second: 1.0,
            // Comfortably above the ~8% noise floor, so a verdict is a signal
            // and not a coin flip.
            decisive_margin: 0.15,
            // Half an hour between trials, against a 60s window: one window in
            // thirty is spent measuring the setting not in use. Short enough to
            // follow a workload that changes through the day, long enough that
            // the measurement is nearly free.
            cooldown: Duration::from_secs(1800),
        }
    }
}

/// Cumulative counters for one model at one instant.
///
/// Cumulative rather than per-window so a dropped or late sample cannot lose
/// tokens: every window is a difference of two totals.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SpeculationSample {
    pub(crate) at: Instant,
    /// Cumulative completion tokens emitted to callers.
    pub(crate) completion_tokens: u64,
    /// Cumulative speculative tokens drafted.
    pub(crate) proposed_tokens: u64,
    /// Cumulative speculative tokens the target accepted.
    pub(crate) accepted_tokens: u64,
    /// Whether speculation was enabled for the window ending at this sample.
    pub(crate) speculating: bool,
}

/// What a measured window says.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct GateWindow {
    pub(crate) seconds: f64,
    pub(crate) completion_tokens_per_second: f64,
    /// `None` when nothing was drafted, which is not the same as 0.0: a
    /// proposer that never fired has no acceptance to judge.
    pub(crate) accept_rate: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum GateDecision {
    /// Keep sampling; `why` is for logs.
    Hold { why: &'static str },
    /// Flip speculation to `enable` and judge the next full window against
    /// `baseline`.
    Trial { enable: bool, baseline: f64 },
    /// The trialled setting won; keep it.
    Keep {
        speculating: bool,
        baseline: f64,
        observed: f64,
    },
    /// The trialled setting lost or was indecisive; put it back.
    Revert {
        enable: bool,
        baseline: f64,
        observed: f64,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Trial {
    /// Throughput of the setting we flipped *away* from.
    baseline: f64,
    /// What the flip set speculation to.
    enabled: bool,
}

/// Decides whether speculation is earning its keep, from observed throughput.
///
/// Only decides. The caller owns the gate state the generation path reads and
/// the telemetry that explains a flip.
#[derive(Debug)]
pub(crate) struct SpeculationGate {
    config: SpeculationGateConfig,
    last: Option<SpeculationSample>,
    cooldown_until: Option<Instant>,
    trial: Option<Trial>,
}

impl SpeculationGate {
    pub(crate) fn new(config: SpeculationGateConfig) -> Self {
        Self {
            config,
            last: None,
            // The first trial waits a full cooldown rather than firing on the
            // first loaded window, so a model that has just come up is not
            // flipped while its caches are still cold.
            cooldown_until: None,
            trial: None,
        }
    }

    /// Feed the latest cumulative counters and get a decision.
    pub(crate) fn observe(&mut self, sample: SpeculationSample) -> GateDecision {
        let Some(last) = self.last else {
            self.last = Some(sample);
            self.cooldown_until = Some(sample.at + self.config.cooldown);
            return GateDecision::Hold {
                why: "first sample",
            };
        };

        // Counters only ever climb, so a drop means the model reloaded and the
        // totals restarted. Differencing across that reads as a huge negative
        // window; re-baseline instead, and drop a trial that was measured
        // against counters which no longer exist.
        if sample.completion_tokens < last.completion_tokens
            || sample.proposed_tokens < last.proposed_tokens
        {
            self.reset(sample);
            return GateDecision::Hold {
                why: "counters restarted; new window",
            };
        }

        let elapsed = sample.at.saturating_duration_since(last.at);
        if elapsed < self.config.min_window {
            return GateDecision::Hold {
                why: "window not yet full",
            };
        }

        let window = measure(&last, &sample, elapsed);
        self.last = Some(sample);

        if window.completion_tokens_per_second < self.config.min_completion_tokens_per_second {
            // An idle window measures idleness. Judging a trial on one would
            // revert a setting for not being used, and starting one would
            // measure noise.
            return GateDecision::Hold { why: "idle" };
        }

        if let Some(trial) = self.trial.take() {
            return self.judge(trial, window, sample.at);
        }

        if let Some(until) = self.cooldown_until
            && sample.at < until
        {
            return GateDecision::Hold { why: "cooldown" };
        }

        // Periodic, because no cheap signal reliably predicts which setting
        // wins — see the module docs on acceptance. Flip whatever is running
        // and measure it against the window just observed.
        let enable = !sample.speculating;
        self.trial = Some(Trial {
            baseline: window.completion_tokens_per_second,
            enabled: enable,
        });
        GateDecision::Trial {
            enable,
            baseline: window.completion_tokens_per_second,
        }
    }

    /// Verdict on a flip, from the window that followed it.
    fn judge(&mut self, trial: Trial, window: GateWindow, at: Instant) -> GateDecision {
        let observed = window.completion_tokens_per_second;
        self.cooldown_until = Some(at + self.config.cooldown);

        // Required to *beat* the baseline by the margin, not merely match it.
        // An indecisive trial reverts: the incumbent is the setting the
        // operator or the strategy chose, so a flip has to earn its place
        // rather than win by being last.
        if observed > trial.baseline * (1.0 + self.config.decisive_margin) {
            GateDecision::Keep {
                speculating: trial.enabled,
                baseline: trial.baseline,
                observed,
            }
        } else {
            GateDecision::Revert {
                enable: !trial.enabled,
                baseline: trial.baseline,
                observed,
            }
        }
    }

    fn reset(&mut self, sample: SpeculationSample) {
        self.last = Some(sample);
        self.trial = None;
        self.cooldown_until = Some(sample.at + self.config.cooldown);
    }
}

/// Environment switch that enables the gate. Default off.
///
/// Off by default because the gate changes what a speculating deployment does
/// over time, and the evidence for its thresholds — the ±8% noise floor and
/// the 15% decisive margin — comes from a different measurement (#1935's
/// rebalance windows) than the one it governs. It wants its own two-box run
/// before it becomes anyone's default.
///
/// The intended destination is `--strategy balanced`, whose composition in
/// #2112 is "package-declared speculation, gated on live break-even".
/// `interactive` should keep speculating ungated: there the operator named a
/// workload, and overriding that from a throughput window would be second-
/// guessing a stated intent rather than filling an unstated gap.
pub(crate) const SPECULATION_GATE_ENV: &str = "SKIPPY_SPECULATION_GATE";

pub(crate) fn speculation_gate_enabled() -> bool {
    std::env::var(SPECULATION_GATE_ENV).is_ok_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

/// Whether a resolved plan speculates at all.
///
/// Governing a plan that does not is worse than pointless: the gate would
/// trial speculation *on*, turning a deliberate `strategy = "disabled"` into
/// something that flips back by itself.
pub(crate) fn speculation_plan_is_active(
    config: &crate::frontend::SpeculativeDecodeConfig,
) -> bool {
    config.ngram.is_some() || config.native_mtp.enabled
}

/// Owns the gate, the counters that feed it, and the switch it throws.
///
/// Shared per model and read on the generation path, so the read is a single
/// relaxed atomic load: the gate changes at most once per window, and a request
/// that straddles a flip is correct either way — speculation is a throughput
/// choice, never a correctness one.
#[derive(Debug)]
pub(crate) struct SpeculationGovernor {
    gate: std::sync::Mutex<SpeculationGate>,
    completion_tokens: std::sync::atomic::AtomicU64,
    proposed_tokens: std::sync::atomic::AtomicU64,
    accepted_tokens: std::sync::atomic::AtomicU64,
    /// The switch the generation path reads.
    speculating: std::sync::atomic::AtomicBool,
}

impl SpeculationGovernor {
    /// `speculating` is what the resolved plan asked for, which is the
    /// incumbent the first trial is measured against.
    pub(crate) fn new(config: SpeculationGateConfig, speculating: bool) -> Self {
        Self {
            gate: std::sync::Mutex::new(SpeculationGate::new(config)),
            completion_tokens: std::sync::atomic::AtomicU64::new(0),
            proposed_tokens: std::sync::atomic::AtomicU64::new(0),
            accepted_tokens: std::sync::atomic::AtomicU64::new(0),
            speculating: std::sync::atomic::AtomicBool::new(speculating),
        }
    }

    /// Whether the generation path should speculate right now.
    pub(crate) fn allows_speculation(&self) -> bool {
        self.speculating.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Fold one finished request into the counters and, if a window has
    /// closed, act on the verdict.
    ///
    /// Returns the decision so the caller can log it; `None` means the gate
    /// was not consulted or had nothing to say.
    pub(crate) fn record(
        &self,
        completion_tokens: u64,
        proposed_tokens: u64,
        accepted_tokens: u64,
    ) -> Option<GateDecision> {
        use std::sync::atomic::Ordering::Relaxed;
        let completion =
            self.completion_tokens.fetch_add(completion_tokens, Relaxed) + completion_tokens;
        let proposed = self.proposed_tokens.fetch_add(proposed_tokens, Relaxed) + proposed_tokens;
        let accepted = self.accepted_tokens.fetch_add(accepted_tokens, Relaxed) + accepted_tokens;

        // A poisoned gate must not take serving down with it: speculation is
        // an optimisation, so losing the controller means losing adaptation,
        // not correctness.
        let mut gate = self.gate.lock().ok()?;
        let decision = gate.observe(SpeculationSample {
            at: Instant::now(),
            completion_tokens: completion,
            proposed_tokens: proposed,
            accepted_tokens: accepted,
            speculating: self.allows_speculation(),
        });
        match decision {
            GateDecision::Hold { .. } => None,
            GateDecision::Trial { enable, .. } | GateDecision::Revert { enable, .. } => {
                self.speculating.store(enable, Relaxed);
                Some(decision)
            }
            GateDecision::Keep { speculating, .. } => {
                self.speculating.store(speculating, Relaxed);
                Some(decision)
            }
        }
    }
}

fn measure(last: &SpeculationSample, now: &SpeculationSample, elapsed: Duration) -> GateWindow {
    let seconds = elapsed.as_secs_f64().max(f64::MIN_POSITIVE);
    let completed = now.completion_tokens.saturating_sub(last.completion_tokens);
    let proposed = now.proposed_tokens.saturating_sub(last.proposed_tokens);
    let accepted = now.accepted_tokens.saturating_sub(last.accepted_tokens);
    GateWindow {
        seconds,
        completion_tokens_per_second: completed as f64 / seconds,
        accept_rate: (proposed > 0).then(|| accepted as f64 / proposed as f64),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const WINDOW_S: u64 = 60;

    fn config() -> SpeculationGateConfig {
        SpeculationGateConfig {
            min_window: Duration::from_secs(WINDOW_S),
            // Two windows of cooldown keeps the tests readable; the shipped
            // default is half an hour.
            cooldown: Duration::from_secs(WINDOW_S * 2),
            ..Default::default()
        }
    }

    /// Cumulative counters advanced one window at a time, which is the shape
    /// the gate actually consumes.
    struct Counters {
        at: Instant,
        completion: u64,
        proposed: u64,
        accepted: u64,
    }

    impl Counters {
        fn new() -> Self {
            Self {
                at: Instant::now(),
                completion: 0,
                proposed: 0,
                accepted: 0,
            }
        }

        fn window(
            &mut self,
            tokens_per_second: f64,
            drafted: u64,
            landed: u64,
            speculating: bool,
        ) -> SpeculationSample {
            self.at += Duration::from_secs(WINDOW_S);
            self.completion += (tokens_per_second * WINDOW_S as f64) as u64;
            self.proposed += drafted;
            self.accepted += landed;
            SpeculationSample {
                at: self.at,
                completion_tokens: self.completion,
                proposed_tokens: self.proposed,
                accepted_tokens: self.accepted,
                speculating,
            }
        }
    }

    /// Run the one window that sits inside the initial cooldown, so the next
    /// observation is the one that trials. With a two-window cooldown and the
    /// first sample landing at one window, exactly one window is held.
    fn settle(gate: &mut SpeculationGate, c: &mut Counters, speculating: bool) {
        let decision = gate.observe(c.window(12.0, 2200, 660, speculating));
        assert_eq!(
            decision,
            GateDecision::Hold { why: "cooldown" },
            "settling should sit inside the initial cooldown"
        );
    }

    #[test]
    fn the_first_sample_only_establishes_a_baseline() {
        let mut gate = SpeculationGate::new(config());
        let mut c = Counters::new();
        assert_eq!(
            gate.observe(c.window(0.0, 0, 0, true)),
            GateDecision::Hold {
                why: "first sample"
            }
        );
    }

    #[test]
    fn a_short_window_is_not_judged() {
        let mut gate = SpeculationGate::new(config());
        let mut c = Counters::new();
        gate.observe(c.window(0.0, 0, 0, true));
        let early = SpeculationSample {
            at: c.at + Duration::from_secs(10),
            completion_tokens: c.completion + 200,
            ..c.window(0.0, 0, 0, true)
        };
        assert_eq!(
            gate.observe(early),
            GateDecision::Hold {
                why: "window not yet full"
            }
        );
    }

    /// A model that has just come up is not flipped while its caches are cold.
    #[test]
    fn the_first_trial_waits_a_full_cooldown() {
        let mut gate = SpeculationGate::new(config());
        let mut c = Counters::new();
        gate.observe(c.window(0.0, 0, 0, true));
        assert_eq!(
            gate.observe(c.window(60.0, 400, 340, true)),
            GateDecision::Hold { why: "cooldown" }
        );
    }

    /// The `window_shrinks 0` bug as a test: a configuration that loses has to
    /// actually be stood down.
    ///
    /// The numbers are the losing run from `WAN_SPLIT_PERF.md` — 660 accepted
    /// of 2200 drafted, 12 tok/s against 17 for plain decode. Note the 29.8%
    /// acceptance: healthy-looking, which is exactly why acceptance is not the
    /// trigger.
    #[test]
    fn a_losing_configuration_is_stood_down() {
        let mut gate = SpeculationGate::new(config());
        let mut c = Counters::new();
        gate.observe(c.window(0.0, 0, 0, true));
        settle(&mut gate, &mut c, true);

        assert_eq!(
            gate.observe(c.window(12.0, 2200, 660, true)),
            GateDecision::Trial {
                enable: false,
                baseline: 12.0,
            }
        );
        assert_eq!(
            gate.observe(c.window(17.0, 0, 0, false)),
            GateDecision::Keep {
                speculating: false,
                baseline: 12.0,
                observed: 17.0,
            }
        );
    }

    /// The mirror case, and what stops the gate being a one-way ratchet.
    #[test]
    fn a_stand_down_that_does_not_pay_is_reverted() {
        let mut gate = SpeculationGate::new(config());
        let mut c = Counters::new();
        gate.observe(c.window(0.0, 0, 0, true));
        settle(&mut gate, &mut c, true);
        gate.observe(c.window(12.0, 2200, 660, true));

        assert_eq!(
            gate.observe(c.window(12.2, 0, 0, false)),
            GateDecision::Revert {
                enable: true,
                baseline: 12.0,
                observed: 12.2,
            }
        );
    }

    /// A workload that suits the proposer keeps it: #1037's re-emit arm ran
    /// several times plain decode, which is decisive by any margin.
    #[test]
    fn a_winning_configuration_is_kept() {
        let mut gate = SpeculationGate::new(config());
        let mut c = Counters::new();
        gate.observe(c.window(0.0, 0, 0, false));
        gate.observe(c.window(20.0, 0, 0, false));
        assert_eq!(
            gate.observe(c.window(20.0, 0, 0, false)),
            GateDecision::Trial {
                enable: true,
                baseline: 20.0,
            }
        );
        assert_eq!(
            gate.observe(c.window(68.0, 425, 361, true)),
            GateDecision::Keep {
                speculating: true,
                baseline: 20.0,
                observed: 68.0,
            }
        );
    }

    /// An improvement inside the noise floor is not a verdict. #1935 measured
    /// roughly ±8% window to window; the margin is 15%.
    #[test]
    fn an_improvement_inside_the_noise_margin_is_not_decisive() {
        let mut gate = SpeculationGate::new(config());
        let mut c = Counters::new();
        gate.observe(c.window(0.0, 0, 0, true));
        settle(&mut gate, &mut c, true);
        gate.observe(c.window(20.0, 2000, 200, true));

        assert_eq!(
            gate.observe(c.window(22.0, 0, 0, false)),
            GateDecision::Revert {
                enable: true,
                baseline: 20.0,
                observed: 22.0,
            }
        );
    }

    #[test]
    fn a_verdict_is_followed_by_a_cooldown() {
        let mut gate = SpeculationGate::new(config());
        let mut c = Counters::new();
        gate.observe(c.window(0.0, 0, 0, true));
        settle(&mut gate, &mut c, true);
        gate.observe(c.window(12.0, 2200, 660, true));
        gate.observe(c.window(17.0, 0, 0, false));

        assert_eq!(
            gate.observe(c.window(17.0, 0, 0, false)),
            GateDecision::Hold { why: "cooldown" }
        );
    }

    /// Standing down must not be permanent: a deployment whose work turns
    /// input-grounded should pick speculation back up on its own.
    #[test]
    fn speculation_is_retried_after_the_cooldown() {
        let mut gate = SpeculationGate::new(config());
        let mut c = Counters::new();
        gate.observe(c.window(0.0, 0, 0, true));
        settle(&mut gate, &mut c, true);
        gate.observe(c.window(12.0, 2200, 660, true));
        gate.observe(c.window(17.0, 0, 0, false));

        // One window inside the post-verdict cooldown, then the gate looks
        // again — and this time the flip is back towards speculation.
        assert_eq!(
            gate.observe(c.window(17.0, 0, 0, false)),
            GateDecision::Hold { why: "cooldown" }
        );
        assert_eq!(
            gate.observe(c.window(17.0, 0, 0, false)),
            GateDecision::Trial {
                enable: true,
                baseline: 17.0,
            }
        );
    }

    #[test]
    fn an_idle_window_is_never_a_verdict() {
        let mut gate = SpeculationGate::new(config());
        let mut c = Counters::new();
        gate.observe(c.window(0.0, 0, 0, true));
        settle(&mut gate, &mut c, true);
        assert_eq!(
            gate.observe(c.window(12.0, 2200, 660, true)),
            GateDecision::Trial {
                enable: false,
                baseline: 12.0,
            }
        );
        // The trial window was idle: judging it would stand a setting down for
        // not being used.
        assert_eq!(
            gate.observe(c.window(0.0, 0, 0, false)),
            GateDecision::Hold { why: "idle" }
        );
    }

    /// A model reload restarts the counters. Differencing across that would
    /// read as an enormous negative window.
    #[test]
    fn restarted_counters_re_baseline_and_cancel_a_trial() {
        let mut gate = SpeculationGate::new(config());
        let mut c = Counters::new();
        gate.observe(c.window(0.0, 0, 0, true));
        settle(&mut gate, &mut c, true);
        assert_eq!(
            gate.observe(c.window(12.0, 2200, 660, true)),
            GateDecision::Trial {
                enable: false,
                baseline: 12.0,
            }
        );

        let reloaded = SpeculationSample {
            at: c.at + Duration::from_secs(WINDOW_S),
            completion_tokens: 0,
            proposed_tokens: 0,
            accepted_tokens: 0,
            speculating: false,
        };
        assert_eq!(
            gate.observe(reloaded),
            GateDecision::Hold {
                why: "counters restarted; new window"
            }
        );

        // The cancelled trial is not judged later against totals that no
        // longer exist, and the fresh baseline starts its own cooldown.
        let mut after = Counters::new();
        after.at = reloaded.at;
        assert_eq!(
            gate.observe(after.window(30.0, 0, 0, false)),
            GateDecision::Hold { why: "cooldown" }
        );
    }

    /// Acceptance is reported per window, not over the model's lifetime: a
    /// long healthy history must not hide a workload that just changed.
    #[test]
    fn acceptance_is_measured_over_the_window_not_the_lifetime() {
        let mut gate = SpeculationGate::new(config());
        let mut c = Counters::new();
        gate.observe(c.window(0.0, 0, 0, true));
        for _ in 0..10 {
            gate.observe(c.window(60.0, 1000, 900, true));
        }
        // Those windows have already produced a verdict or two; take the gate
        // back to a point where the next loaded window trials.
        while !matches!(
            gate.observe(c.window(60.0, 1000, 900, true)),
            GateDecision::Hold { why: "cooldown" }
        ) {}
        // This window drafted 2200 and landed 300; lifetime acceptance would
        // still read about 0.78.
        let decision = gate.observe(c.window(12.0, 2200, 300, true));
        let GateDecision::Trial { baseline, .. } = decision else {
            panic!("expected a trial, got {decision:?}");
        };
        assert!((baseline - 12.0).abs() < 0.001, "baseline {baseline}");
    }

    /// A silent proposer still gets trialled — it costs a lookup per token and
    /// the trial is how we find out whether that lookup is worth anything.
    #[test]
    fn a_silent_proposer_is_still_measured() {
        let mut gate = SpeculationGate::new(config());
        let mut c = Counters::new();
        gate.observe(c.window(0.0, 0, 0, true));
        gate.observe(c.window(15.0, 0, 0, true));
        assert_eq!(
            gate.observe(c.window(15.0, 0, 0, true)),
            GateDecision::Trial {
                enable: false,
                baseline: 15.0,
            }
        );
    }

    #[test]
    fn a_window_with_no_drafts_reports_no_accept_rate() {
        let a = SpeculationSample {
            at: Instant::now(),
            completion_tokens: 0,
            proposed_tokens: 0,
            accepted_tokens: 0,
            speculating: false,
        };
        let b = SpeculationSample {
            at: a.at + Duration::from_secs(60),
            completion_tokens: 600,
            ..a
        };
        let window = measure(&a, &b, Duration::from_secs(60));
        // None, not 0.0: a proposer that never fired has no acceptance to
        // judge, and reporting zero would read as total rejection.
        assert_eq!(window.accept_rate, None);
        assert!((window.completion_tokens_per_second - 10.0).abs() < 0.001);
    }
}
