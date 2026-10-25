//! Search for the speculative run-ahead budget by measurement, not by formula.
//!
//! # Why this is a search and not an expression
//!
//! Mesh-LLM/mesh-llm#2112 proposed deriving the budget as
//! `measured RTT x stage decode rate`. Two-box measurement on a two-node split
//! does not support that. The budget curve, at a fixed 25ms and then 50ms of
//! conditioned one-way delay:
//!
//! | budget | 25ms e2e tok/s | 50ms e2e tok/s |
//! |--------|----------------|----------------|
//! | 0      | 22.51          | 20.71          |
//! | 48     | 20.25          | -              |
//! | 96     | 22.90          | 21.32          |
//! | 192    | 23.72          | 22.06          |
//! | 384    | 22.82          | 21.97          |
//!
//! Three facts come out of that, and together they rule out a formula.
//!
//! **The optimum does not move with RTT.** It sits at 192 at both delays.
//! Doubling the delay left it where it was, so there is no rate to multiply by.
//!
//! **A small budget is worse than none.** 48 tokens - one and a half verify
//! windows - lost 10% against not running ahead at all, a bigger loss than the
//! 2.7% that running ahead costs on a zero-latency link. Enough depth to pay
//! the proposal and round-trip overhead, not enough to fill the pipe. So the
//! search must never *try* a shallow budget: the ladder below skips the region
//! outright rather than discovering it once per deployment.
//!
//! **Past the optimum the budget stops binding.** At 384 the peak in flight
//! was 313 tokens in *both* delay conditions, identically - a structural bound
//! from the window and proposal structure, not a latency-dependent level.
//!
//! What remains latency-dependent is only whether to run ahead at all, and
//! that is a throughput comparison the search already makes when it weighs
//! rung 0 against the others.
//!
//! # Why the ladder is in windows and not tokens
//!
//! 192 tokens was six verify windows on the pair that measured it. The window
//! size is itself a configured, tuned quantity, so carrying the ladder in
//! windows scales it with that knob instead of freezing one lab's token count.
//! It also states the floor in the units the floor actually has: the harmful
//! rung was *one and a half windows*, which is a statement about window
//! structure rather than about 48 tokens.
//!
//! Every point behind those numbers ran on a 35/1 cut, where the last stage
//! holds one layer of 36 and there is very little stage work to overlap with.
//! The signs and the shape should carry; the specific rung that wins should
//! not be assumed to. That is the other reason this searches rather than
//! computes.
//!
//! # Interaction with the speculation gate
//!
//! [`super::speculation_gate`] flips speculation wholesale, and this adjusts
//! its depth. Two controllers over the same setting will mislead each other if
//! they run at once: a window the gate spent with speculation stood down says
//! nothing about a budget, and a budget mid-trial perturbs the rate the gate is
//! judging. So an outcome is only folded in when speculation was actually on
//! *and* the gate was quiescent - see [`RequestOutcome::usable`]. The search
//! starves rather than misreads, which is the safe direction: no verdict leaves
//! the incumbent budget in place.
//!
//! The windowing deliberately mirrors `speculation_gate` rather than sharing
//! code with it. That gate is newly landed and still unvalidated against the
//! freeform regression it exists to catch; factoring the two together now would
//! put a refactor underneath a measurement in flight. Worth revisiting once
//! both are proven.

use std::time::{Duration, Instant};

/// Candidate budgets, in verify windows.
///
/// `0` is fixed-depth admission, the incumbent and the thing every other rung
/// has to beat. The jump from 0 to 3 skips the measured harmful region - at one
/// and a half windows the budget pays for depth it cannot use - so no
/// deployment has to rediscover a 10% loss for itself.
const LADDER_WINDOWS: [usize; 4] = [0, 3, 6, 12];

#[derive(Clone, Copy, Debug)]
pub(crate) struct RunaheadSearchConfig {
    /// Shortest window a rung may be judged on.
    pub(crate) min_window: Duration,
    /// Matching requests a rung needs before it counts. One request's rate is
    /// noise and a verdict on it is a coin flip.
    pub(crate) min_requests: u64,
    /// Fractional improvement in mean decode rate needed to prefer a new rung
    /// over the best one measured this walk.
    ///
    /// Deliberately smaller than `speculation_gate`'s 0.15, and the difference
    /// matters: that gate compares speculation against no speculation, which
    /// on an input-grounded workload is a several-hundred-percent effect. The
    /// rungs here differ by 5-10%, so 0.15 would reject every one of them. On
    /// the measured curve it settles on fixed-depth admission and discards a
    /// real 9.8% win - which is how this constant was first written, and the
    /// bug only showed up when the measured rates were replayed through it
    /// instead of invented ones.
    ///
    /// 0.05 sits above the noise actually observed on this workload rather
    /// than above the +-8% that `speculation_gate` cites for sustained
    /// concurrent windows. Repeating one arm formation-to-formation moved it
    /// 1.6% (31.17 against 31.68 tok/s), and single-stream request spread
    /// inside a run was 0.6% (p50 8069ms against p90 8115ms) because decoding
    /// here is deterministic. So the margin is ~3x the noise and ~half the
    /// effect being resolved.
    pub(crate) decisive_margin: f64,
    /// Quiet period after the ladder has been walked once, before walking it
    /// again. A link changes; a budget that was right an hour ago may not be.
    pub(crate) cooldown: Duration,
}

impl Default for RunaheadSearchConfig {
    fn default() -> Self {
        Self {
            min_window: Duration::from_secs(60),
            min_requests: 3,
            decisive_margin: 0.05,
            cooldown: Duration::from_secs(1800),
        }
    }
}

/// One finished request, as the search needs it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RequestOutcome {
    /// This request's own decode rate - the server's `predicted_per_second`,
    /// not a wall-clock rate over a window.
    ///
    /// Below saturation a window rate measures the offered load rather than
    /// the capacity: it would not move when the budget changed, so a losing
    /// rung would survive every trial by looking unchanged. That mistake is
    /// recorded in `speculation_gate`'s history and is not repeated here.
    pub(crate) decode_tokens_per_second: f64,
    /// Whether speculation was actually running for this request.
    pub(crate) speculated: bool,
    /// Whether the speculation gate was holding rather than trialling.
    pub(crate) gate_quiescent: bool,
}

impl RequestOutcome {
    /// Whether this request can be attributed to the budget in force.
    ///
    /// A request that did not speculate exercised no budget, and one taken
    /// while the gate was trialling has the gate's flip mixed into its rate.
    fn usable(self) -> bool {
        self.speculated && self.gate_quiescent && self.decode_tokens_per_second > 0.0
    }
}

/// A rung's accumulating evidence.
#[derive(Clone, Copy, Debug)]
struct Window {
    opened: Instant,
    rung: usize,
    rate_sum: f64,
    requests: u64,
}

impl Window {
    fn new(rung: usize, at: Instant) -> Self {
        Self {
            opened: at,
            rung,
            rate_sum: 0.0,
            requests: 0,
        }
    }

    fn mean_rate(&self) -> Option<f64> {
        (self.requests > 0).then(|| self.rate_sum / self.requests as f64)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SearchDecision {
    /// Keep sampling; `why` is for logs.
    Hold { why: &'static str },
    /// Move to `rung_windows` and judge it against the incumbent.
    Try { rung_windows: usize },
    /// The ladder is walked; `rung_windows` won and stays.
    Settle { rung_windows: usize },
}

/// Walks the ladder once per cooldown and keeps the best rung it measured.
#[derive(Debug)]
pub(crate) struct RunaheadSearch {
    config: RunaheadSearchConfig,
    /// Index into `LADDER_WINDOWS` currently in force.
    current: usize,
    /// Index being measured, if the ladder is mid-walk.
    probing: Option<usize>,
    /// Best mean rate seen this walk, and the rung that produced it.
    best: Option<(usize, f64)>,
    window: Option<Window>,
    settled_until: Option<Instant>,
}

impl RunaheadSearch {
    pub(crate) fn new(config: RunaheadSearchConfig) -> Self {
        Self {
            config,
            current: 0,
            probing: None,
            best: None,
            window: None,
            settled_until: None,
        }
    }

    /// The budget in force, in tokens, for a given verify-window size.
    ///
    /// Tests only. The serving path reads [`RunaheadGovernor::budget_tokens`],
    /// which answers from an atomic without taking the lock this type sits
    /// behind; the duplication is deliberate so a per-request read never
    /// contends with a verdict being computed.
    #[cfg(test)]
    fn budget_tokens(&self, verify_window_tokens: usize) -> usize {
        LADDER_WINDOWS[self.current].saturating_mul(verify_window_tokens)
    }

    /// Fold in one finished request and decide whether to move.
    pub(crate) fn observe(&mut self, at: Instant, outcome: RequestOutcome) -> SearchDecision {
        if !outcome.usable() {
            // Starve rather than misread. A window that collected nothing
            // cannot produce a verdict, which leaves the incumbent in place.
            return SearchDecision::Hold {
                why: "not attributable to the budget",
            };
        }

        if let Some(until) = self.settled_until {
            if at < until {
                return SearchDecision::Hold { why: "settled" };
            }
            // The cooldown has expired: walk the ladder again from the bottom,
            // because the link may have moved under us.
            self.settled_until = None;
            self.best = None;
            self.probing = Some(0);
            self.window = Some(Window::new(0, at));
            self.current = 0;
            return SearchDecision::Try {
                rung_windows: LADDER_WINDOWS[0],
            };
        }

        let window = self
            .window
            .get_or_insert_with(|| Window::new(self.current, at));

        // A rung change mid-window would mix two settings into one mean.
        if window.rung != self.current {
            *window = Window::new(self.current, at);
        }

        window.rate_sum += outcome.decode_tokens_per_second;
        window.requests += 1;

        if at.saturating_duration_since(window.opened) < self.config.min_window {
            return SearchDecision::Hold {
                why: "window too young",
            };
        }
        if window.requests < self.config.min_requests {
            return SearchDecision::Hold {
                why: "too few requests",
            };
        }

        let Some(mean) = window.mean_rate() else {
            return SearchDecision::Hold { why: "no rate" };
        };
        let rung = window.rung;
        self.window = None;

        match self.best {
            // A later rung must beat the best so far by the margin, so noise
            // cannot ratchet the budget upward one window at a time. The
            // asymmetry favours shallower rungs on a tie, which is the right
            // direction: the measured downside of too much depth is wasted
            // proposals, while the downside of too little is a 10% loss.
            Some((_, best_rate)) if mean <= best_rate * (1.0 + self.config.decisive_margin) => {}
            _ => self.best = Some((rung, mean)),
        }

        let next = rung + 1;
        if next < LADDER_WINDOWS.len() {
            self.probing = Some(next);
            self.current = next;
            return SearchDecision::Try {
                rung_windows: LADDER_WINDOWS[next],
            };
        }

        // Ladder walked. Keep the best rung measured and rest.
        let (winner, _) = self.best.unwrap_or((0, mean));
        self.current = winner;
        self.probing = None;
        self.settled_until = Some(at + self.config.cooldown);
        SearchDecision::Settle {
            rung_windows: LADDER_WINDOWS[winner],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const WINDOW: Duration = Duration::from_secs(60);

    fn config() -> RunaheadSearchConfig {
        RunaheadSearchConfig {
            min_window: WINDOW,
            min_requests: 2,
            decisive_margin: 0.05,
            cooldown: Duration::from_secs(1800),
        }
    }

    fn outcome(rate: f64) -> RequestOutcome {
        RequestOutcome {
            decode_tokens_per_second: rate,
            speculated: true,
            gate_quiescent: true,
        }
    }

    /// Fill one rung's window and return the decision it produced.
    fn walk_rung(search: &mut RunaheadSearch, at: &mut Instant, rate: f64) -> SearchDecision {
        search.observe(*at, outcome(rate));
        *at += WINDOW + Duration::from_secs(1);
        search.observe(*at, outcome(rate))
    }

    #[test]
    fn the_ladder_never_offers_a_shallow_budget() {
        // The measured harmful rung is one and a half verify windows. No rung
        // between fixed depth and three windows exists, so no deployment can
        // be walked into the 10% loss that 48 tokens cost.
        assert_eq!(LADDER_WINDOWS[0], 0);
        assert!(
            LADDER_WINDOWS[1] >= 3,
            "ladder must skip the measured harmful region: {LADDER_WINDOWS:?}"
        );
    }

    #[test]
    fn the_budget_scales_with_the_verify_window() {
        // Six windows was the optimum on the pair that measured it; carrying
        // the ladder in windows means a different window size moves with it
        // rather than freezing one lab's token count.
        let mut search = RunaheadSearch::new(config());
        search.current = 2;
        assert_eq!(LADDER_WINDOWS[2], 6);
        assert_eq!(search.budget_tokens(32), 192);
        assert_eq!(search.budget_tokens(64), 384);
    }

    #[test]
    fn a_request_that_did_not_speculate_is_not_evidence() {
        let mut search = RunaheadSearch::new(config());
        let at = Instant::now();
        let decision = search.observe(
            at,
            RequestOutcome {
                speculated: false,
                ..outcome(50.0)
            },
        );
        assert_eq!(
            decision,
            SearchDecision::Hold {
                why: "not attributable to the budget"
            }
        );
    }

    #[test]
    fn a_request_taken_while_the_gate_trialled_is_not_evidence() {
        // The gate flipping speculation mid-window puts its own effect into
        // the rate this search is attributing to a budget.
        let mut search = RunaheadSearch::new(config());
        let at = Instant::now();
        let decision = search.observe(
            at,
            RequestOutcome {
                gate_quiescent: false,
                ..outcome(50.0)
            },
        );
        assert_eq!(
            decision,
            SearchDecision::Hold {
                why: "not attributable to the budget"
            }
        );
    }

    #[test]
    fn it_finds_the_optimum_of_the_measured_curve() {
        // THE REAL RATES, not invented ones. Median server decode tok/s from
        // the 25ms sweep in #2112: fixed depth, then three, six and twelve
        // verify windows. The margin has to be small enough to resolve a 9.8%
        // win and large enough to reject a 5% one in the wrong direction.
        //
        // Writing this test with a made-up 60 tok/s rate for the winning rung
        // hid a real bug: the margin inherited from `speculation_gate` was
        // 0.15, which rejects every rung of this curve and settles on never
        // running ahead at all.
        let mut search = RunaheadSearch::new(config());
        let mut at = Instant::now();

        assert_eq!(
            walk_rung(&mut search, &mut at, 35.91),
            SearchDecision::Try { rung_windows: 3 }
        );
        assert_eq!(
            walk_rung(&mut search, &mut at, 37.53),
            SearchDecision::Try { rung_windows: 6 }
        );
        assert_eq!(
            walk_rung(&mut search, &mut at, 39.44),
            SearchDecision::Try { rung_windows: 12 }
        );
        assert_eq!(
            walk_rung(&mut search, &mut at, 37.74),
            SearchDecision::Settle { rung_windows: 6 }
        );
        // Six windows at the 32-token verify window those runs used is the 192
        // tokens that measured fastest.
        assert_eq!(search.budget_tokens(32), 192);
    }

    #[test]
    fn fixed_depth_survives_a_ladder_that_never_beats_it_decisively() {
        // The zero-latency case: running ahead costs a little and wins
        // nothing, so the search has to come back to fixed-depth admission
        // rather than keeping whichever rung happened to measure highest.
        // Also the real rates: the 0ms arm of the same sweep, where running
        // ahead costs 4.2% on decode and buys nothing.
        let mut search = RunaheadSearch::new(config());
        let mut at = Instant::now();

        walk_rung(&mut search, &mut at, 53.28);
        walk_rung(&mut search, &mut at, 51.03);
        walk_rung(&mut search, &mut at, 51.50);
        let decision = walk_rung(&mut search, &mut at, 50.90);

        assert_eq!(decision, SearchDecision::Settle { rung_windows: 0 });
        assert_eq!(search.budget_tokens(32), 0);
    }

    #[test]
    fn it_rests_after_walking_and_walks_again_when_the_cooldown_expires() {
        let mut search = RunaheadSearch::new(config());
        let mut at = Instant::now();
        for rate in [50.0, 51.0, 52.0, 51.5] {
            walk_rung(&mut search, &mut at, rate);
        }
        assert_eq!(
            search.observe(at, outcome(50.0)),
            SearchDecision::Hold { why: "settled" }
        );

        // A link changes, so the ladder is walked again rather than trusting a
        // verdict from an hour ago.
        at += Duration::from_secs(1801);
        assert_eq!(
            search.observe(at, outcome(50.0)),
            SearchDecision::Try { rung_windows: 0 }
        );
    }
}

/// Owns the search and the budget it currently asserts.
///
/// Shaped like [`super::speculation_gate::SpeculationGovernor`]: a mutex over
/// the controller, and one atomic the per-request path reads without taking a
/// lock. The budget is stored in *windows* so the token figure can be derived
/// against whatever verify-window size the request is using.
#[derive(Debug)]
pub(crate) struct RunaheadGovernor {
    search: std::sync::Mutex<RunaheadSearch>,
    /// Rungs, not tokens — see the module docs on why the ladder is in windows.
    rung_windows: std::sync::atomic::AtomicUsize,
}

impl RunaheadGovernor {
    pub(crate) fn new(config: RunaheadSearchConfig) -> Self {
        Self {
            search: std::sync::Mutex::new(RunaheadSearch::new(config)),
            // Start at fixed-depth admission. Starting mid-ladder would assert
            // a budget nothing has measured on this deployment, and the
            // measured cost of guessing too shallow is a 10% loss.
            rung_windows: std::sync::atomic::AtomicUsize::new(LADDER_WINDOWS[0]),
        }
    }

    /// The budget to admit with for a request using `verify_window_tokens`.
    pub(crate) fn budget_tokens(&self, verify_window_tokens: usize) -> usize {
        self.rung_windows
            .load(std::sync::atomic::Ordering::Relaxed)
            .saturating_mul(verify_window_tokens)
    }

    /// Fold one finished request in, and move the budget if the search says to.
    ///
    /// Returns the decision for logging; `None` means it had nothing to say.
    pub(crate) fn record(&self, outcome: RequestOutcome) -> Option<SearchDecision> {
        // A poisoned lock must not take serving down: the budget is an
        // optimisation, so losing the controller loses adaptation, not
        // correctness. Same reasoning as the speculation gate.
        let mut search = self.search.lock().ok()?;
        let decision = search.observe(Instant::now(), outcome);
        match decision {
            SearchDecision::Hold { .. } => None,
            SearchDecision::Try { rung_windows } | SearchDecision::Settle { rung_windows } => {
                self.rung_windows
                    .store(rung_windows, std::sync::atomic::Ordering::Relaxed);
                Some(decision)
            }
        }
    }
}

/// Apply the searched budget to a request's plan.
///
/// Sibling of `speculation_after_prefix_restore` and the gate's
/// `gated_speculative`, and borrowed unless something actually changes so the
/// common path allocates nothing. Only `runahead_auto` plans are touched: a
/// stated `runahead_max_tokens`, including a stated zero, is the operator's
/// number and is left exactly alone.
pub(crate) fn runahead_after_search<'a>(
    config: &'a crate::frontend::SpeculativeDecodeConfig,
    governor: Option<&RunaheadGovernor>,
) -> std::borrow::Cow<'a, crate::frontend::SpeculativeDecodeConfig> {
    if !config.verify_window.runahead_auto {
        return std::borrow::Cow::Borrowed(config);
    }
    let Some(governor) = governor else {
        return std::borrow::Cow::Borrowed(config);
    };
    let budget = governor.budget_tokens(config.verify_window.max_tokens);
    if budget == config.verify_window.runahead_max_tokens {
        return std::borrow::Cow::Borrowed(config);
    }
    let mut resolved = config.clone();
    resolved.verify_window.runahead_max_tokens = budget;
    std::borrow::Cow::Owned(resolved)
}

#[cfg(test)]
mod governor_tests {
    use super::*;

    fn plan(auto: bool, stated: usize) -> crate::frontend::SpeculativeDecodeConfig {
        let mut config = crate::frontend::SpeculativeDecodeConfig::default();
        config.verify_window.max_tokens = 32;
        config.verify_window.runahead_auto = auto;
        config.verify_window.runahead_max_tokens = stated;
        config
    }

    #[test]
    fn a_stated_budget_is_never_overridden() {
        // Including a stated zero: "use fixed-depth admission" is an answer,
        // and the search must not treat it as an absence of one.
        let governor = RunaheadGovernor::new(RunaheadSearchConfig::default());
        governor
            .rung_windows
            .store(6, std::sync::atomic::Ordering::Relaxed);

        for stated in [0, 128] {
            let config = plan(false, stated);
            let resolved = runahead_after_search(&config, Some(&governor));
            assert!(matches!(resolved, std::borrow::Cow::Borrowed(_)));
            assert_eq!(resolved.verify_window.runahead_max_tokens, stated);
        }
    }

    #[test]
    fn an_auto_budget_takes_the_searched_rung_in_tokens() {
        let governor = RunaheadGovernor::new(RunaheadSearchConfig::default());
        governor
            .rung_windows
            .store(6, std::sync::atomic::Ordering::Relaxed);
        let config = plan(true, 0);
        let resolved = runahead_after_search(&config, Some(&governor));
        assert_eq!(resolved.verify_window.runahead_max_tokens, 6 * 32);
    }

    #[test]
    fn auto_without_a_governor_stays_at_fixed_depth() {
        // Nothing is searching, so nothing asserts a budget. Falling back to a
        // guess would be the shallow-budget hazard with no measurement behind
        // it.
        let config = plan(true, 0);
        let resolved = runahead_after_search(&config, None);
        assert!(matches!(resolved, std::borrow::Cow::Borrowed(_)));
        assert_eq!(resolved.verify_window.runahead_max_tokens, 0);
    }

    #[test]
    fn the_governor_starts_at_fixed_depth() {
        let governor = RunaheadGovernor::new(RunaheadSearchConfig::default());
        assert_eq!(governor.budget_tokens(32), 0);
    }

    #[test]
    fn a_held_decision_does_not_move_the_budget() {
        let governor = RunaheadGovernor::new(RunaheadSearchConfig::default());
        let decision = governor.record(RequestOutcome {
            decode_tokens_per_second: 40.0,
            speculated: false,
            gate_quiescent: true,
        });
        assert!(decision.is_none());
        assert_eq!(governor.budget_tokens(32), 0);
    }
}
