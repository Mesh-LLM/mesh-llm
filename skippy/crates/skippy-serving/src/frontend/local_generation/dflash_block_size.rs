//! Chooses each DFlash block length from measured cost and acceptance.
//!
//! A fixed block length is right for one machine and one kind of text: long
//! blocks pay where verifying extra rows is cheap and drafts land, short ones
//! where rows are expensive or drafts miss. This picks, per window, the length
//! that maximises expected committed tokens per unit of measured time.
//!
//! Cost is learned per draft model and shared by every request on it, because
//! it is a property of the machine. Acceptance is learned per request, seeded
//! from the shared estimate, because it is a property of the text.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

/// Positions tracked for per-position acceptance; longer blocks reuse the last.
const MAX_TRACKED_POSITIONS: usize = 64;
/// Weight kept per new observation by the decayed estimates.
const DECAY: f64 = 0.97;
/// Prior acceptance and its weight in pseudo-observations.
const PRIOR_ACCEPTANCE: f64 = 0.6;
const PRIOR_WEIGHT: f64 = 2.0;
/// Every this many windows, the choice is nudged to keep the cost fit spread.
const EXPLORE_EVERY: u64 = 8;
const EXPLORE_STEP: usize = 2;
/// Measured serial steps needed before serial decoding may be chosen.
pub(super) const MIN_SERIAL_SAMPLES: f64 = 2.0;
/// Weight, in windows, a request gives the shared acceptance it starts from,
/// so its own text dominates within a few windows.
const SEED_ACCEPTANCE_WEIGHT: f64 = 4.0;

/// Whether DFlash block lengths adapt instead of using the configured maximum.
/// On by default: on every machine measured, adaptive matched or beat the best
/// fixed length. `SKIPPY_DFLASH_ADAPTIVE=0` pins blocks at the maximum.
pub(super) fn adaptive_enabled() -> bool {
    adaptive_enabled_by(std::env::var("SKIPPY_DFLASH_ADAPTIVE").ok().as_deref())
}

fn adaptive_enabled_by(setting: Option<&str>) -> bool {
    setting != Some("0")
}

/// Decayed least-squares fit of `time = intercept + slope * rows`, plus the
/// mean cost of drafting one block and of one measured serial step.
///
/// The serial step is measured rather than read off the fit: extrapolating a
/// fit over verify windows down to one row is noisy, and a low estimate makes
/// serial decoding look free.
#[derive(Clone, Debug, Default)]
pub(super) struct BlockCost {
    weight: f64,
    sum_rows: f64,
    sum_time: f64,
    sum_rows_sq: f64,
    sum_rows_time: f64,
    draft_weight: f64,
    draft_time_sum: f64,
    serial_weight: f64,
    serial_time_sum: f64,
}

impl BlockCost {
    pub(super) fn observe_serial(&mut self, micros: f64) {
        self.serial_weight = self.serial_weight * DECAY + 1.0;
        self.serial_time_sum = self.serial_time_sum * DECAY + micros;
    }

    /// Whether enough serial steps were measured to weigh serial decoding.
    pub(super) fn serial_measured(&self) -> bool {
        self.serial_weight >= MIN_SERIAL_SAMPLES * DECAY.powi(2)
    }

    fn serial_micros(&self) -> Option<f64> {
        self.serial_measured()
            .then(|| self.serial_time_sum / self.serial_weight)
    }

    pub(super) fn observe_verify(&mut self, rows: usize, micros: f64) {
        let rows = rows as f64;
        self.weight = self.weight * DECAY + 1.0;
        self.sum_rows = self.sum_rows * DECAY + rows;
        self.sum_time = self.sum_time * DECAY + micros;
        self.sum_rows_sq = self.sum_rows_sq * DECAY + rows * rows;
        self.sum_rows_time = self.sum_rows_time * DECAY + rows * micros;
    }

    pub(super) fn observe_draft(&mut self, micros: f64) {
        self.draft_weight = self.draft_weight * DECAY + 1.0;
        self.draft_time_sum = self.draft_time_sum * DECAY + micros;
    }

    fn draft_micros(&self) -> f64 {
        if self.draft_weight > 0.0 {
            self.draft_time_sum / self.draft_weight
        } else {
            0.0
        }
    }

    /// Fitted verify time for `rows`, or `None` before two row counts were seen.
    fn verify_micros(&self, rows: usize) -> Option<f64> {
        if self.weight <= 0.0 {
            return None;
        }
        let mean_rows = self.sum_rows / self.weight;
        let mean_time = self.sum_time / self.weight;
        let variance = self.sum_rows_sq / self.weight - mean_rows * mean_rows;
        if variance < 1e-6 {
            return None;
        }
        let covariance = self.sum_rows_time / self.weight - mean_rows * mean_time;
        // Verifying more rows never costs less; a negative slope is noise.
        let slope = (covariance / variance).max(0.0);
        // A verify window always costs at least a fixed launch; a negative
        // intercept is the fit overreaching outside the rows it has seen.
        let intercept = (mean_time - slope * mean_rows).max(0.0);
        Some((intercept + slope * rows as f64).max(1.0))
    }
}

/// Per-position conditional acceptance: the chance draft `i` is accepted given
/// draft `i - 1` was.
#[derive(Clone, Debug)]
pub(super) struct Acceptance {
    accepted: Vec<f64>,
    reached: Vec<f64>,
}

impl Default for Acceptance {
    fn default() -> Self {
        Self {
            accepted: vec![PRIOR_ACCEPTANCE * PRIOR_WEIGHT; MAX_TRACKED_POSITIONS],
            reached: vec![PRIOR_WEIGHT; MAX_TRACKED_POSITIONS],
        }
    }
}

impl Acceptance {
    /// Records a window that drafted `drafts` tokens and accepted the first `accepted`.
    pub(super) fn observe(&mut self, drafts: usize, accepted: usize) {
        let reached = (accepted + 1).min(drafts).min(MAX_TRACKED_POSITIONS);
        for position in 0..reached {
            self.reached[position] = self.reached[position] * DECAY + 1.0;
            let hit = if position < accepted { 1.0 } else { 0.0 };
            self.accepted[position] = self.accepted[position] * DECAY + hit;
        }
    }

    /// The same estimates carrying at most `max_weight` windows of evidence
    /// per position, so a request starting from them adapts to its own text.
    pub(super) fn capped(mut self) -> Self {
        for (accepted, reached) in self.accepted.iter_mut().zip(self.reached.iter_mut()) {
            if *reached > SEED_ACCEPTANCE_WEIGHT {
                let scale = SEED_ACCEPTANCE_WEIGHT / *reached;
                *accepted *= scale;
                *reached = SEED_ACCEPTANCE_WEIGHT;
            }
        }
        self
    }

    fn conditional(&self, position: usize) -> f64 {
        let position = position.min(MAX_TRACKED_POSITIONS - 1);
        (self.accepted[position] / self.reached[position]).clamp(0.0, 1.0)
    }

    /// Expected tokens a block of `drafts` commits: its accepted prefix plus
    /// the one token the target always contributes.
    fn expected_tokens(&self, drafts: usize) -> f64 {
        let mut survive = 1.0;
        let mut expected = 1.0;
        for position in 0..drafts {
            survive *= self.conditional(position);
            expected += survive;
        }
        expected
    }
}

/// Picks the block length with the best expected tokens per microsecond.
/// Returns 0 when a measured serial step beats every block; serial decoding is
/// never chosen before serial steps were measured.
pub(super) fn choose_block(
    cost: &BlockCost,
    acceptance: &Acceptance,
    max_drafts: usize,
    window_index: u64,
) -> usize {
    if max_drafts == 0 {
        return 0;
    }
    // Before the fit has two row counts, alternate between short and full
    // blocks so it gets them.
    if cost.verify_micros(1).is_none() {
        return if window_index.is_multiple_of(2) {
            max_drafts
        } else {
            max_drafts.div_ceil(2)
        };
    }
    let draft = cost.draft_micros();
    let serial_rate = cost.serial_micros().map_or(0.0, |serial| 1.0 / serial);
    let mut best = (0, serial_rate);
    for drafts in 1..=max_drafts {
        let Some(verify) = cost.verify_micros(drafts + 1) else {
            continue;
        };
        let rate = acceptance.expected_tokens(drafts) / (draft + verify);
        if rate > best.1 {
            best = (drafts, rate);
        }
    }
    let chosen = best.0;
    if chosen == 0 || window_index % EXPLORE_EVERY != EXPLORE_EVERY - 1 {
        return chosen;
    }
    // Alternate the nudge's direction so the fit sees rows on both sides.
    if (window_index / EXPLORE_EVERY).is_multiple_of(2) {
        (chosen + EXPLORE_STEP).min(max_drafts)
    } else {
        chosen.saturating_sub(EXPLORE_STEP).max(1)
    }
}

/// Shared per-draft-model estimates, learned across requests.
#[derive(Clone, Debug, Default)]
pub(super) struct SharedBlockModel {
    pub(super) cost: BlockCost,
    pub(super) acceptance: Acceptance,
}

fn registry() -> &'static Mutex<HashMap<PathBuf, SharedBlockModel>> {
    static REGISTRY: OnceLock<Mutex<HashMap<PathBuf, SharedBlockModel>>> = OnceLock::new();
    REGISTRY.get_or_init(Default::default)
}

/// The current shared estimates for a draft model.
pub(super) fn shared_model(draft_model_path: &Path) -> SharedBlockModel {
    registry()
        .lock()
        .map(|models| models.get(draft_model_path).cloned().unwrap_or_default())
        .unwrap_or_default()
}

/// Folds one measured serial step into the shared estimates for a draft model.
pub(super) fn record_shared_serial(draft_model_path: &Path, micros: f64) {
    if let Ok(mut models) = registry().lock() {
        models
            .entry(draft_model_path.to_path_buf())
            .or_default()
            .cost
            .observe_serial(micros);
    }
}

/// Folds one window into the shared estimates for a draft model.
pub(super) fn record_shared(
    draft_model_path: &Path,
    verify_rows: usize,
    verify_micros: f64,
    draft_micros: f64,
    drafts: usize,
    accepted: usize,
) {
    if let Ok(mut models) = registry().lock() {
        let model = models.entry(draft_model_path.to_path_buf()).or_default();
        model.cost.observe_verify(verify_rows, verify_micros);
        model.cost.observe_draft(draft_micros);
        model.acceptance.observe(drafts, accepted);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptive_is_on_unless_turned_off() {
        assert!(adaptive_enabled_by(None));
        assert!(adaptive_enabled_by(Some("1")));
        assert!(!adaptive_enabled_by(Some("0")));
    }

    fn linear_cost(intercept: f64, slope: f64, draft: f64) -> BlockCost {
        let mut cost = BlockCost::default();
        for rows in [1, 3, 5, 8, 12, 16] {
            for _ in 0..4 {
                cost.observe_verify(rows, intercept + slope * rows as f64);
            }
        }
        for _ in 0..4 {
            cost.observe_draft(draft);
        }
        cost
    }

    /// The same conditional acceptance `p` at every position.
    fn uniform_acceptance(p: f64) -> Acceptance {
        Acceptance {
            accepted: vec![p * 100.0; MAX_TRACKED_POSITIONS],
            reached: vec![100.0; MAX_TRACKED_POSITIONS],
        }
    }

    #[test]
    fn fit_recovers_a_linear_cost() {
        let cost = linear_cost(20_000.0, 2_500.0, 3_000.0);
        let fitted = cost.verify_micros(10).unwrap();
        assert!((fitted - 45_000.0).abs() < 1.0, "{fitted}");
        assert!((cost.draft_micros() - 3_000.0).abs() < 1e-6);
    }

    #[test]
    fn expected_tokens_compound_per_position() {
        let mut acceptance = Acceptance {
            accepted: vec![0.0; MAX_TRACKED_POSITIONS],
            reached: vec![1.0; MAX_TRACKED_POSITIONS],
        };
        acceptance.accepted[0] = 0.5;
        acceptance.accepted[1] = 0.5;
        // 1 + 0.5 + 0.25
        assert!((acceptance.expected_tokens(2) - 1.75).abs() < 1e-9);
    }

    #[test]
    fn cheap_rows_and_good_drafts_choose_long_blocks() {
        let cost = linear_cost(25_000.0, 300.0, 1_000.0);
        let acceptance = uniform_acceptance(0.85);
        assert!(choose_block(&cost, &acceptance, 15, 0) >= 10);
    }

    #[test]
    fn expensive_rows_or_poor_drafts_choose_short_blocks() {
        let costly = linear_cost(25_000.0, 6_000.0, 3_000.0);
        let good = uniform_acceptance(0.85);
        let poor = uniform_acceptance(0.35);
        let cheap = linear_cost(25_000.0, 300.0, 1_000.0);
        assert!(choose_block(&costly, &good, 15, 0) <= 6);
        assert!(choose_block(&cheap, &poor, 15, 0) <= 4);
    }

    #[test]
    fn hopeless_drafts_decode_serially_once_serial_steps_are_measured() {
        let mut cost = linear_cost(25_000.0, 4_000.0, 8_000.0);
        let acceptance = uniform_acceptance(0.0);
        assert!(
            choose_block(&cost, &acceptance, 15, 0) > 0,
            "an unmeasured serial step is never chosen"
        );
        cost.observe_serial(29_000.0);
        cost.observe_serial(29_000.0);
        assert_eq!(choose_block(&cost, &acceptance, 15, 0), 0);
    }

    #[test]
    fn a_steep_noisy_fit_does_not_make_serial_look_free() {
        // Verify windows of 5 to 8 rows whose noise suggests a steep slope; the
        // fitted one-row cost must not undercut the measured serial step.
        let mut cost = BlockCost::default();
        for (rows, micros) in [(5, 30_000.0), (8, 60_000.0), (5, 31_000.0), (8, 59_000.0)] {
            cost.observe_verify(rows, micros);
        }
        assert!(cost.verify_micros(1).unwrap() >= 1.0);
        cost.observe_serial(26_000.0);
        cost.observe_serial(26_000.0);
        let acceptance = uniform_acceptance(0.75);
        assert!(choose_block(&cost, &acceptance, 7, 0) > 0);
    }

    #[test]
    fn requests_start_from_lightly_weighted_shared_acceptance() {
        let shared = Acceptance {
            accepted: vec![90.0; MAX_TRACKED_POSITIONS],
            reached: vec![100.0; MAX_TRACKED_POSITIONS],
        };
        let mut seeded = shared.capped();
        assert!((seeded.conditional(0) - 0.9).abs() < 1e-9, "rate is kept");
        // Three misses at the first position should pull it well below the seed.
        for _ in 0..3 {
            seeded.observe(8, 0);
        }
        assert!(seeded.conditional(0) < 0.6, "{}", seeded.conditional(0));
    }

    #[test]
    fn unfitted_cost_alternates_block_lengths() {
        let cost = BlockCost::default();
        let acceptance = Acceptance::default();
        assert_eq!(choose_block(&cost, &acceptance, 15, 0), 15);
        assert_eq!(choose_block(&cost, &acceptance, 15, 1), 8);
    }

    #[test]
    fn exploration_nudges_both_ways() {
        let cost = linear_cost(25_000.0, 2_000.0, 2_000.0);
        let acceptance = uniform_acceptance(0.7);
        let base = choose_block(&cost, &acceptance, 15, 0);
        let up = choose_block(&cost, &acceptance, 15, EXPLORE_EVERY - 1);
        let down = choose_block(&cost, &acceptance, 15, 2 * EXPLORE_EVERY - 1);
        assert_eq!(up, (base + EXPLORE_STEP).min(15));
        assert_eq!(down, base.saturating_sub(EXPLORE_STEP).max(1));
    }

    #[test]
    fn acceptance_counts_the_rejected_position_once() {
        let mut acceptance = Acceptance {
            accepted: vec![0.0; MAX_TRACKED_POSITIONS],
            reached: vec![0.0; MAX_TRACKED_POSITIONS],
        };
        acceptance.observe(8, 2);
        // Positions 0 and 1 accepted, position 2 reached and rejected, later unreached.
        assert_eq!(acceptance.reached[..4], [1.0, 1.0, 1.0, 0.0]);
        assert_eq!(acceptance.accepted[..4], [1.0, 1.0, 0.0, 0.0]);
    }
}
