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

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Thresholds for [`SpeculationGate`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SpeculationGateConfig {
    /// Shortest window a decision is made on.
    pub(crate) min_window: Duration,
    /// Matching requests a window needs before it can be judged. One request's
    /// rate is noise; a verdict on it would be a coin flip.
    ///
    /// Eight rather than three, because the margin below is small enough that
    /// the mean has to be tight. The standard error of a mean falls as
    /// `1/sqrt(n)`: on the measured 0.49% per-request CV, three requests give
    /// 0.28% and eight give 0.17%. Eight is what licenses a 5% margin with two
    /// orders of magnitude to spare rather than one.
    pub(crate) min_requests: u64,
    /// Fractional improvement in mean decode rate needed to call a trial
    /// decisive.
    ///
    /// This was 0.15, justified as "comfortably above the ~8% noise floor".
    /// That floor was measured on **window-to-window wall-clock rates**, and
    /// this gate does not use them: #2260 moved it to a mean of per-request
    /// `predicted_per_second`, precisely because a wall-clock window rate
    /// below saturation measures offered load rather than capacity. The
    /// justification outlived the quantity it was about.
    ///
    /// The noise that matters is the noise in the quantity this gate compares:
    /// a window mean of per-request rates, **within one process**. It never
    /// compares across processes, so cross-run spread does not enter.
    ///
    /// Measured on this gate's own freeform workload, 24 requests in one
    /// process: per-request CV **0.49%**, so the mean of a `min_requests`
    /// window of 8 has a standard error of 0.17%, or 0.34% at two sigma. A 5%
    /// margin is roughly 15x that.
    ///
    /// Cross-formation spread on the same workload is far larger — the same
    /// `plain` arm returned 10.10, 10.15 and 10.97 tok/s in three separate
    /// formations, an 8.7% range, which is where the original "~8% noise
    /// floor" came from. That number is real but belongs to comparing *runs*,
    /// which is a benchmark-harness problem and not this controller's.
    ///
    /// And 0.15 could not catch the regression this gate exists for:
    ///
    /// | case | gain from standing down | decisive at 0.15? |
    /// |---|---|---|
    /// | #1581, the motivating run (12.3 against 13.8 tok/s) | 12.2% | **no** |
    /// | #2112's freeform arm (9.89 against 10.15 tok/s) | 2.6% | **no** |
    ///
    /// A gate that calls its own motivating case indecisive and reverts to
    /// speculating is the "adaptive policy that never adapted" failure in the
    /// module docs above, arriving through the margin instead of the window.
    ///
    /// 0.05 catches #1581 with room and sits well above a window mean's own
    /// noise. The 2.6% case is deliberately still reachable in principle but
    /// not worth chasing: an effect that small is below what a single pair of
    /// windows should be trusted to rank, and the cooldown means a wrong
    /// verdict persists for half an hour.
    pub(crate) decisive_margin: f64,
    /// Quiet period after a verdict, so the gate cannot oscillate.
    pub(crate) cooldown: Duration,
}

/// The gate's settings as an operator states them.
///
/// Separate from [`SpeculationGateConfig`] because that one holds `Duration`s
/// for the controller to compare against, while a config file states seconds.
/// Promoted out of the environment because #2112 workstream 5 puts the gate
/// under the `balanced` strategy, and a strategy composes configuration — an
/// environment-only switch cannot be composed, and a 1800s cooldown nobody can
/// shorten cannot be validated either.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SpeculationGateSettings {
    /// Off by default: the gate costs one window in every `cooldown` measuring
    /// the setting not in use, and that trade wants stating rather than
    /// inheriting.
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_gate_min_window_s")]
    pub min_window_s: u64,
    #[serde(default = "default_gate_min_requests")]
    pub min_requests: u64,
    #[serde(default = "default_gate_decisive_margin")]
    pub decisive_margin: f64,
    #[serde(default = "default_gate_cooldown_s")]
    pub cooldown_s: u64,
}

fn default_gate_min_window_s() -> u64 {
    60
}

fn default_gate_min_requests() -> u64 {
    8
}

fn default_gate_decisive_margin() -> f64 {
    0.05
}

fn default_gate_cooldown_s() -> u64 {
    1800
}

impl Default for SpeculationGateSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            min_window_s: default_gate_min_window_s(),
            min_requests: default_gate_min_requests(),
            decisive_margin: default_gate_decisive_margin(),
            cooldown_s: default_gate_cooldown_s(),
        }
    }
}

impl From<SpeculationGateSettings> for SpeculationGateConfig {
    fn from(settings: SpeculationGateSettings) -> Self {
        let default = Self::default();
        // A zero means "unset" rather than "instant": a zero-length window or a
        // zero-request quorum would judge on a single sample, which is the coin
        // flip `min_requests` exists to prevent.
        Self {
            min_window: if settings.min_window_s == 0 {
                default.min_window
            } else {
                Duration::from_secs(settings.min_window_s)
            },
            min_requests: settings.min_requests.max(1),
            decisive_margin: if settings.decisive_margin > 0.0 {
                settings.decisive_margin
            } else {
                default.decisive_margin
            },
            cooldown: Duration::from_secs(settings.cooldown_s),
        }
    }
}

impl Default for SpeculationGateConfig {
    fn default() -> Self {
        Self {
            min_window: Duration::from_secs(60),
            min_requests: 8,
            decisive_margin: 0.05,
            // Half an hour between trials, against a 60s window: one window in
            // thirty is spent measuring the setting not in use. Short enough to
            // follow a workload that changes through the day, long enough that
            // the measurement is nearly free.
            cooldown: Duration::from_secs(1800),
        }
    }
}

/// One finished request, as the gate needs to see it.
///
/// `decode_tokens_per_second` is the request's **own** decode rate — the
/// engine's `predicted_per_second`, completion tokens over decode time. Not a
/// window rate over wall clock, which is the mistake this type exists to
/// prevent: below saturation a window rate equals the offered load, so it does
/// not move when the setting changes, and a losing configuration would survive
/// every trial. It also mis-attributes a long request's whole output to
/// whichever window it happened to finish in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RequestOutcome {
    pub(crate) decode_tokens_per_second: f64,
    pub(crate) proposed_tokens: u64,
    pub(crate) accepted_tokens: u64,
    /// What this request actually used, which may differ from what the gate
    /// wants now if it finished across a flip.
    pub(crate) speculated: bool,
}

/// What a measured window says.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct GateWindow {
    pub(crate) requests: u64,
    /// Mean of the per-request decode rates in the window.
    pub(crate) mean_decode_tokens_per_second: f64,
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
    /// Mean decode rate of the setting we flipped away from.
    baseline: f64,
    /// What the flip set speculation to.
    enabled: bool,
}

/// Requests accumulated for the setting currently under measurement.
///
/// Per-window rather than cumulative. Cumulative totals would have to be
/// differenced across samples, and a concurrent pair of completions can read
/// those totals out of order — which looks exactly like a counter restart and
/// would drop a trial and push the cooldown out, repeatedly. There is nothing
/// to difference here.
#[derive(Clone, Copy, Debug)]
struct Window {
    opened: Instant,
    /// The setting this window is measuring. A request that used the other
    /// setting straddled a flip and is discarded.
    speculating: bool,
    rate_sum: f64,
    requests: u64,
    proposed: u64,
    accepted: u64,
}

impl Window {
    fn open(at: Instant, speculating: bool) -> Self {
        Self {
            opened: at,
            speculating,
            rate_sum: 0.0,
            requests: 0,
            proposed: 0,
            accepted: 0,
        }
    }

    fn measure(&self) -> GateWindow {
        GateWindow {
            requests: self.requests,
            mean_decode_tokens_per_second: if self.requests == 0 {
                0.0
            } else {
                self.rate_sum / self.requests as f64
            },
            accept_rate: (self.proposed > 0).then(|| self.accepted as f64 / self.proposed as f64),
        }
    }
}

/// Decides whether speculation is earning its keep, from measured decode rate.
///
/// Only decides. The caller owns the switch the generation path reads and the
/// telemetry that explains a flip.
#[derive(Debug)]
pub(crate) struct SpeculationGate {
    config: SpeculationGateConfig,
    window: Option<Window>,
    cooldown_until: Option<Instant>,
    trial: Option<Trial>,
}

impl SpeculationGate {
    pub(crate) fn new(config: SpeculationGateConfig) -> Self {
        Self {
            config,
            window: None,
            cooldown_until: None,
            trial: None,
        }
    }

    /// Fold one finished request in, and decide if its window has closed.
    ///
    /// `speculating` is the gate's current setting, which the caller owns.
    pub(crate) fn observe(
        &mut self,
        at: Instant,
        speculating: bool,
        outcome: RequestOutcome,
    ) -> GateDecision {
        if self.window.is_none() {
            // The first trial waits a full cooldown, so a model that has just
            // come up is not flipped while its caches are still cold.
            self.cooldown_until = Some(at + self.config.cooldown);
            self.window = Some(Window::open(at, speculating));
        }
        let window = self.window.as_mut().expect("opened just above");

        // The setting changed under us (a flip, or a caller that switched for
        // its own reasons): the requests gathered so far measured the old one.
        if window.speculating != speculating {
            *window = Window::open(at, speculating);
            if self.cooldown_until.is_none() {
                self.cooldown_until = Some(at + self.config.cooldown);
            }
            return GateDecision::Hold {
                why: "setting changed; new window",
            };
        }

        if outcome.speculated != speculating {
            // Straddled a flip: its tokens were produced under the other
            // setting, so counting them would blend the two.
            return GateDecision::Hold {
                why: "request straddled a flip; discarded",
            };
        }

        window.rate_sum += outcome.decode_tokens_per_second;
        window.requests += 1;
        window.proposed += outcome.proposed_tokens;
        window.accepted += outcome.accepted_tokens;

        if at.saturating_duration_since(window.opened) < self.config.min_window {
            return GateDecision::Hold {
                why: "window not yet full",
            };
        }
        if window.requests < self.config.min_requests {
            return GateDecision::Hold {
                why: "too few requests to judge",
            };
        }

        let measured = window.measure();
        self.window = Some(Window::open(at, speculating));

        if let Some(trial) = self.trial.take() {
            return self.judge(trial, measured, at);
        }
        if let Some(until) = self.cooldown_until
            && at < until
        {
            return GateDecision::Hold { why: "cooldown" };
        }

        // Periodic, because no cheap signal reliably predicts which setting
        // wins — see the module docs on acceptance.
        let enable = !speculating;
        self.trial = Some(Trial {
            baseline: measured.mean_decode_tokens_per_second,
            enabled: enable,
        });
        GateDecision::Trial {
            enable,
            baseline: measured.mean_decode_tokens_per_second,
        }
    }

    /// Verdict on a flip, from the window that followed it.
    fn judge(&mut self, trial: Trial, window: GateWindow, at: Instant) -> GateDecision {
        let observed = window.mean_decode_tokens_per_second;
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
}

/// Environment switch that enables the gate. Default off.
///
/// Off by default because the gate changes what a speculating deployment does
/// over time. Its thresholds now come from the measurement it governs — #2112's
/// two-box freeform arm and #1581's documented regression — rather than from
/// #1935's rebalance windows, which measured a different quantity on a
/// wall-clock rate this gate no longer uses. Flipping the default on is
/// #2112 workstream 5's last step and wants its own run.
///
/// The intended destination is `--strategy balanced`, whose composition in
/// #2112 is "package-declared speculation, gated on live break-even".
/// `interactive` should keep speculating ungated: there the operator named a
/// workload, and overriding that from a throughput window would be second-
/// guessing a stated intent rather than filling an unstated gap.
pub(crate) const SPECULATION_GATE_ENV: &str = "SKIPPY_SPECULATION_GATE";

fn truthy(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

/// Whether the gate runs, from the configured setting and the environment
/// override.
///
/// The environment is read first and decides on its own, the same shape as
/// `SKIPPY_LAST_STAGE_DECODE_BATCH`: a bench or incident override that needs no
/// replan. A set-but-unparseable value means off rather than deferring to the
/// config, so an operator's typo cannot look like a policy.
pub(crate) fn resolve_speculation_gate_enabled(env_value: Option<&str>, configured: bool) -> bool {
    match env_value {
        Some(value) => truthy(value),
        None => configured,
    }
}

pub(crate) fn speculation_gate_enabled(settings: SpeculationGateSettings) -> bool {
    resolve_speculation_gate_enabled(
        std::env::var(SPECULATION_GATE_ENV).ok().as_deref(),
        settings.enabled,
    )
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

/// Owns the gate and the switch it throws.
///
/// The switch is an atomic because the generation path reads it per request;
/// everything the gate decides from lives under the mutex. An earlier version
/// kept cumulative counters as atomics and read them *before* taking the lock,
/// which let two concurrent completions reach `observe` out of order — the
/// smaller total read as a counter restart, dropping the trial and pushing the
/// cooldown out. Repeatedly, that is the "never adapts" failure these docs set
/// out to prevent. There are no cumulative totals now.
#[derive(Debug)]
pub(crate) struct SpeculationGovernor {
    gate: std::sync::Mutex<SpeculationGate>,
    /// The switch the generation path reads.
    speculating: std::sync::atomic::AtomicBool,
}

impl SpeculationGovernor {
    /// `speculating` is what the resolved plan asked for, which is the
    /// incumbent the first trial is measured against.
    pub(crate) fn new(config: SpeculationGateConfig, speculating: bool) -> Self {
        Self {
            gate: std::sync::Mutex::new(SpeculationGate::new(config)),
            speculating: std::sync::atomic::AtomicBool::new(speculating),
        }
    }

    /// Whether the generation path should speculate right now.
    pub(crate) fn allows_speculation(&self) -> bool {
        self.speculating.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Fold one finished request into the gate and act on any verdict.
    ///
    /// Returns the decision so the caller can log it; `None` means the gate had
    /// nothing to say.
    pub(crate) fn record(&self, outcome: RequestOutcome) -> Option<GateDecision> {
        use std::sync::atomic::Ordering::Relaxed;
        // A poisoned gate must not take serving down with it: speculation is an
        // optimisation, so losing the controller means losing adaptation, not
        // correctness.
        let mut gate = self.gate.lock().ok()?;
        let speculating = self.speculating.load(Relaxed);
        let decision = gate.observe(Instant::now(), speculating, outcome);
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

#[cfg(test)]
mod tests {
    use super::*;

    const WINDOW: Duration = Duration::from_secs(60);

    #[test]
    fn the_environment_overrides_the_configured_gate_either_way() {
        // Same precedence as SKIPPY_LAST_STAGE_DECODE_BATCH: the environment is a
        // bench and incident override that needs no replan, so it decides alone.
        assert!(resolve_speculation_gate_enabled(Some("1"), false));
        assert!(!resolve_speculation_gate_enabled(Some("0"), true));
        // Unset defers to configuration, which is what lets a strategy compose it.
        assert!(resolve_speculation_gate_enabled(None, true));
        assert!(!resolve_speculation_gate_enabled(None, false));
        // A typo must not read as a policy.
        assert!(!resolve_speculation_gate_enabled(Some("ture"), true));
    }

    #[test]
    fn settings_become_durations_and_zero_means_unset() {
        let settings = SpeculationGateSettings {
            enabled: true,
            min_window_s: 5,
            min_requests: 4,
            decisive_margin: 0.2,
            cooldown_s: 30,
        };
        let config = SpeculationGateConfig::from(settings);
        assert_eq!(config.min_window, Duration::from_secs(5));
        assert_eq!(config.min_requests, 4);
        assert_eq!(config.decisive_margin, 0.2);
        assert_eq!(config.cooldown, Duration::from_secs(30));

        // A zero window or quorum would judge on a single sample, which is the
        // coin flip `min_requests` exists to prevent, so it restores the default
        // rather than meaning "instantly".
        let defaults = SpeculationGateConfig::default();
        let zeroed = SpeculationGateConfig::from(SpeculationGateSettings {
            min_window_s: 0,
            min_requests: 0,
            decisive_margin: 0.0,
            ..settings
        });
        assert_eq!(zeroed.min_window, defaults.min_window);
        assert_eq!(zeroed.min_requests, 1);
        assert_eq!(zeroed.decisive_margin, defaults.decisive_margin);

        // A zero cooldown IS meaningful: it is how a validation run reaches a
        // verdict without waiting half an hour for the first trial.
        assert_eq!(
            SpeculationGateConfig::from(SpeculationGateSettings {
                cooldown_s: 0,
                ..settings
            })
            .cooldown,
            Duration::ZERO
        );
    }

    #[test]
    fn the_gate_is_off_unless_asked_for() {
        assert!(!SpeculationGateSettings::default().enabled);
    }

    fn config() -> SpeculationGateConfig {
        SpeculationGateConfig {
            min_window: WINDOW,
            min_requests: 2,
            // Two windows of cooldown keeps the tests readable; the shipped
            // default is half an hour.
            cooldown: Duration::from_secs(120),
            ..Default::default()
        }
    }

    fn outcome(rate: f64, drafted: u64, landed: u64, speculated: bool) -> RequestOutcome {
        RequestOutcome {
            decode_tokens_per_second: rate,
            proposed_tokens: drafted,
            accepted_tokens: landed,
            speculated,
        }
    }

    /// Feed a full window of matching requests and return the last decision.
    ///
    /// Three requests: the first may be absorbed opening a window after a
    /// setting change, leaving two to satisfy `min_requests`, with the last
    /// landing once the window is long enough to judge.
    fn window(
        gate: &mut SpeculationGate,
        at: &mut Instant,
        speculating: bool,
        rate: f64,
        drafted: u64,
        landed: u64,
    ) -> GateDecision {
        let one = outcome(rate, drafted, landed, speculating);
        gate.observe(*at, speculating, one);
        gate.observe(*at, speculating, one);
        *at += WINDOW;
        gate.observe(*at, speculating, one)
    }

    #[test]
    fn a_short_window_is_not_judged() {
        let mut gate = SpeculationGate::new(config());
        let at = Instant::now();
        assert_eq!(
            gate.observe(at, true, outcome(60.0, 400, 340, true)),
            GateDecision::Hold {
                why: "window not yet full"
            }
        );
    }

    /// A long-enough window that is too sparse is still not a verdict: a
    /// couple of rates is noise, and judging on them would be a coin flip.
    #[test]
    fn a_window_with_too_few_requests_is_not_judged() {
        let mut gate = SpeculationGate::new(SpeculationGateConfig {
            min_requests: 5,
            ..config()
        });
        let at = Instant::now();
        gate.observe(at, true, outcome(60.0, 400, 340, true));
        assert_eq!(
            gate.observe(at + WINDOW, true, outcome(60.0, 400, 340, true)),
            GateDecision::Hold {
                why: "too few requests to judge"
            }
        );
    }

    /// A model that has just come up is not flipped while its caches are cold.
    #[test]
    fn the_first_trial_waits_a_full_cooldown() {
        let mut gate = SpeculationGate::new(config());
        let mut at = Instant::now();
        assert_eq!(
            window(&mut gate, &mut at, true, 60.0, 400, 340),
            GateDecision::Hold { why: "cooldown" }
        );
    }

    /// The `window_shrinks 0` bug as a test: a configuration that loses has to
    /// actually be stood down.
    ///
    /// Numbers are the losing run from `WAN_SPLIT_PERF.md` — 660 accepted of
    /// 2200 drafted, 12 tok/s against 17 for plain decode. Note the 29.8%
    /// acceptance: healthy-looking, which is why acceptance is not the trigger.
    #[test]
    fn a_losing_configuration_is_stood_down() {
        let mut gate = SpeculationGate::new(config());
        let mut at = Instant::now();
        window(&mut gate, &mut at, true, 12.0, 2200, 660); // cooldown
        assert_eq!(
            window(&mut gate, &mut at, true, 12.0, 2200, 660),
            GateDecision::Trial {
                enable: false,
                baseline: 12.0
            }
        );
        assert_eq!(
            window(&mut gate, &mut at, false, 17.0, 0, 0),
            GateDecision::Keep {
                speculating: false,
                baseline: 12.0,
                observed: 17.0
            }
        );
    }

    /// The mirror case, and what stops the gate being a one-way ratchet.
    #[test]
    fn a_stand_down_that_does_not_pay_is_reverted() {
        let mut gate = SpeculationGate::new(config());
        let mut at = Instant::now();
        window(&mut gate, &mut at, true, 12.0, 2200, 660);
        window(&mut gate, &mut at, true, 12.0, 2200, 660);
        assert_eq!(
            window(&mut gate, &mut at, false, 12.2, 0, 0),
            GateDecision::Revert {
                enable: true,
                baseline: 12.0,
                observed: 12.2
            }
        );
    }

    /// An improvement inside the noise floor is not a verdict. A window mean of
    /// eight requests carries 0.34% at two sigma on the measured workload; the
    /// margin is 5%, so 2% is comfortably inside it.
    #[test]
    fn an_improvement_inside_the_noise_margin_is_not_decisive() {
        let mut gate = SpeculationGate::new(config());
        let mut at = Instant::now();
        window(&mut gate, &mut at, true, 20.0, 2000, 200);
        window(&mut gate, &mut at, true, 20.0, 2000, 200);
        // 2% — the size of #2112's freeform arm, and under twice the measured
        // noise. Deliberately out of reach: a margin that caught this would
        // latch on coin flips.
        let decision = window(&mut gate, &mut at, false, 20.4, 0, 0);
        assert!(
            matches!(decision, GateDecision::Revert { enable: true, .. }),
            "an improvement inside the noise is not a verdict, got {decision:?}"
        );
    }

    /// THE CASE THIS GATE EXISTS FOR. #1581 measured 12.2-12.4 tok/s
    /// speculating against 13.8 for plain decode on a freeform two-node split.
    ///
    /// At the original 0.15 margin that is a 12.2% improvement from standing
    /// down and therefore *indecisive* — the gate reverted to speculating and
    /// kept the regression. A gate that cannot catch its own motivating run is
    /// the "adaptive policy that never adapted" failure in this module's docs,
    /// reached through the margin rather than the window.
    /// The full trial cycle, replayed at the measured cadence of #2112's
    /// stand-down run: cooldown 300s, min_window 20s, min_requests 3, one
    /// request every 32s, 9.8 tok/s speculating against 10.5 standing down.
    ///
    /// That is a 7.4% improvement — INDECISIVE at a 0.15 margin — so the gate
    /// must trial the flip and then put it back. It does, four requests later.
    ///
    /// This exists because the benchmark run it replays showed speculation
    /// switching off and *staying* off for fourteen requests, which this
    /// controller does not do. Whatever suppressed proposals there, it was not
    /// this; see `runahead_search`'s note on reading proposal counts.
    #[test]
    fn an_indecisive_trial_is_put_back_within_a_few_requests() {
        let mut gate = SpeculationGate::new(SpeculationGateConfig {
            min_window: Duration::from_secs(20),
            min_requests: 3,
            decisive_margin: 0.15,
            cooldown: Duration::from_secs(300),
        });
        let mut at = Instant::now();
        let mut speculating = true;
        let mut flipped_off_at = None;
        let mut restored_at = None;
        for request in 0..24u32 {
            let rate = if speculating { 9.8 } else { 10.5 };
            let decision = gate.observe(
                at,
                speculating,
                RequestOutcome {
                    decode_tokens_per_second: rate,
                    proposed_tokens: if speculating { 78 } else { 0 },
                    accepted_tokens: if speculating { 6 } else { 0 },
                    speculated: speculating,
                },
            );
            match decision {
                GateDecision::Trial { enable, .. } | GateDecision::Revert { enable, .. } => {
                    if !enable && flipped_off_at.is_none() {
                        flipped_off_at = Some(request);
                    } else if enable && flipped_off_at.is_some() && restored_at.is_none() {
                        restored_at = Some(request);
                    }
                    speculating = enable;
                }
                GateDecision::Keep {
                    speculating: kept, ..
                } => speculating = kept,
                GateDecision::Hold { .. } => {}
            }
            at += Duration::from_secs(32);
        }
        let off = flipped_off_at.expect("the gate should trial a flip once the cooldown expires");
        let back = restored_at.expect("an indecisive trial must be put back, not kept");
        assert!(
            back - off <= 6,
            "the flip should be reverted within a few requests, not held: off at {off}, back at {back}"
        );
        assert!(
            speculating,
            "the incumbent setting must be in force at the end"
        );
    }

    #[test]
    fn the_documented_freeform_regression_is_caught() {
        let mut gate = SpeculationGate::new(config());
        let mut at = Instant::now();
        window(&mut gate, &mut at, true, 12.3, 2214, 660);
        window(&mut gate, &mut at, true, 12.3, 2214, 660);
        // Matched on shape, not float equality: the baseline is a mean of
        // per-request rates, so it carries accumulation error (12.3 sums to
        // 12.300000000000002) and the verdict is what this pins.
        let decision = window(&mut gate, &mut at, false, 13.8, 0, 0);
        assert!(
            matches!(
                decision,
                GateDecision::Keep {
                    speculating: false,
                    ..
                }
            ),
            "the documented regression must stand speculation down, got {decision:?}"
        );
    }

    #[test]
    fn a_winning_configuration_is_never_stood_down() {
        // The other direction, which the smaller margin must not break: on an
        // input-grounded workload speculation wins several-fold, so the trial
        // measures far worse and the incumbent has to come straight back.
        let mut gate = SpeculationGate::new(config());
        let mut at = Instant::now();
        window(&mut gate, &mut at, true, 49.6, 2286, 1998);
        window(&mut gate, &mut at, true, 49.6, 2286, 1998);
        let decision = window(&mut gate, &mut at, false, 10.1, 0, 0);
        assert!(
            matches!(decision, GateDecision::Revert { enable: true, .. }),
            "a winning configuration must come straight back, got {decision:?}"
        );
    }

    #[test]
    fn a_verdict_is_followed_by_a_cooldown() {
        let mut gate = SpeculationGate::new(config());
        let mut at = Instant::now();
        window(&mut gate, &mut at, true, 12.0, 2200, 660);
        window(&mut gate, &mut at, true, 12.0, 2200, 660);
        window(&mut gate, &mut at, false, 17.0, 0, 0);
        assert_eq!(
            window(&mut gate, &mut at, false, 17.0, 0, 0),
            GateDecision::Hold { why: "cooldown" }
        );
    }

    /// Standing down must not be permanent: work that turns input-grounded
    /// should pick speculation back up on its own.
    #[test]
    fn speculation_is_retried_after_the_cooldown() {
        let mut gate = SpeculationGate::new(config());
        let mut at = Instant::now();
        window(&mut gate, &mut at, true, 12.0, 2200, 660);
        window(&mut gate, &mut at, true, 12.0, 2200, 660);
        window(&mut gate, &mut at, false, 17.0, 0, 0);
        window(&mut gate, &mut at, false, 17.0, 0, 0); // inside the cooldown
        assert_eq!(
            window(&mut gate, &mut at, false, 17.0, 0, 0),
            GateDecision::Trial {
                enable: true,
                baseline: 17.0
            }
        );
    }

    /// The finding this signal exists for: a request that finished across a
    /// flip produced its tokens under the other setting, so counting it would
    /// blend the two and could manufacture a verdict.
    #[test]
    fn a_request_that_straddled_a_flip_is_discarded() {
        let mut gate = SpeculationGate::new(config());
        let at = Instant::now();
        assert_eq!(
            gate.observe(at, false, outcome(99.0, 500, 450, true)),
            GateDecision::Hold {
                why: "request straddled a flip; discarded"
            }
        );
    }

    /// And a setting change starts a fresh window rather than mixing the
    /// requests gathered under the previous one.
    #[test]
    fn a_setting_change_opens_a_new_window() {
        let mut gate = SpeculationGate::new(config());
        let mut at = Instant::now();
        gate.observe(at, true, outcome(12.0, 2200, 660, true));
        at += WINDOW;
        assert_eq!(
            gate.observe(at, false, outcome(17.0, 0, 0, false)),
            GateDecision::Hold {
                why: "setting changed; new window"
            }
        );
    }

    /// The verdict is a mean of per-request rates, so a slow request and a
    /// fast one do not average into whichever happened to be longer.
    #[test]
    fn the_window_rate_is_the_mean_of_per_request_rates() {
        let mut gate = SpeculationGate::new(config());
        let mut at = Instant::now();
        gate.observe(at, true, outcome(10.0, 100, 50, true));
        gate.observe(at, true, outcome(30.0, 100, 50, true));
        at += WINDOW;
        // Inside the opening cooldown, so no verdict — but the window has
        // closed, and the next one starts clean.
        assert_eq!(
            gate.observe(at, true, outcome(20.0, 100, 50, true)),
            GateDecision::Hold { why: "cooldown" }
        );
        // A second window of three equal rates then trials against that mean,
        // proving the rate is the mean of per-request rates and nothing else.
        assert_eq!(
            window(&mut gate, &mut at, true, 42.0, 100, 50),
            GateDecision::Trial {
                enable: false,
                baseline: 42.0
            }
        );
    }

    /// A silent proposer still gets trialled — it costs a lookup per token and
    /// the trial is how we find out whether that lookup is worth anything.
    #[test]
    fn a_silent_proposer_is_still_measured() {
        let mut gate = SpeculationGate::new(config());
        let mut at = Instant::now();
        window(&mut gate, &mut at, true, 15.0, 0, 0);
        assert_eq!(
            window(&mut gate, &mut at, true, 15.0, 0, 0),
            GateDecision::Trial {
                enable: false,
                baseline: 15.0
            }
        );
    }

    #[test]
    fn a_window_with_no_drafts_reports_no_accept_rate() {
        let w = Window {
            opened: Instant::now(),
            speculating: false,
            rate_sum: 20.0,
            requests: 2,
            proposed: 0,
            accepted: 0,
        };
        let measured = w.measure();
        // None, not 0.0: a proposer that never fired has no acceptance to
        // judge, and reporting zero would read as total rejection.
        assert_eq!(measured.accept_rate, None);
        assert!((measured.mean_decode_tokens_per_second - 10.0).abs() < 0.001);
    }
}
