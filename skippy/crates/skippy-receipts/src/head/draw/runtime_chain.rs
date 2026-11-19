//! The node's sampler chain and indexed draw as the runtime computes them,
//! for the draw's exact replay.
//!
//! llama.cpp filters the candidates in f32, in the order it holds them:
//! token id order until a filter sorts them. Top-p normalizes the softmax
//! over the candidates with an f32 sum, which drops the weights of a long
//! tail that fall under half an ulp of the running sum, so its cut can come
//! before the cut of the same mass in f64. This port follows that
//! arithmetic: `llama_sampler_top_k_impl`, `llama_sampler_top_p_apply`,
//! `llama_sampler_min_p_apply` and `llama_sampler_temp_impl` of
//! `src/llama-sampler.cpp` with `min_keep` 0, as `src/skippy/sampling_config.cpp`
//! builds them, then `skippy_receipt_draw_pick` of
//! `src/skippy/receipts_draw.cpp`. A change of those needs the same change
//! here.
//!
//! Where llama.cpp orders equal logits, it does so in no fixed order; this
//! port puts the lower token id first. The replay's band covers the rest
//! ([`super::judge`]).

use super::super::sampling::Filter;

/// The candidates of the chain, as the runtime holds them.
struct Candidates {
    ids: Vec<u32>,
    logits: Vec<f32>,
    /// A filter sorted them, best first.
    sorted: bool,
}

impl Candidates {
    fn new(logits: &[f64]) -> Self {
        Self {
            ids: (0..logits.len() as u32).collect(),
            // the logits already hold the biases, added in f32
            logits: logits.iter().map(|&l| l as f32).collect(),
            sorted: false,
        }
    }

    fn len(&self) -> usize {
        self.ids.len()
    }

    fn truncate(&mut self, n: usize) {
        self.ids.truncate(n);
        self.logits.truncate(n);
    }

    fn sort(&mut self) {
        if self.sorted {
            return;
        }
        let mut order: Vec<usize> = (0..self.len()).collect();
        order.sort_by(|&a, &b| {
            self.logits[b]
                .total_cmp(&self.logits[a])
                .then(self.ids[a].cmp(&self.ids[b]))
        });
        self.ids = order.iter().map(|&i| self.ids[i]).collect();
        self.logits = order.iter().map(|&i| self.logits[i]).collect();
        self.sorted = true;
    }

    fn top_k(&mut self, k: u32) {
        if k == 0 {
            return;
        }
        let k = (k as usize).min(self.len());
        self.sort();
        self.truncate(k);
    }

    /// The softmax of the candidates in their order, in f32
    /// (`llama_sampler_softmax_impl` without a sort).
    fn softmax(&self) -> Vec<f32> {
        let max = if self.sorted {
            self.logits[0]
        } else {
            self.logits.iter().copied().fold(self.logits[0], f32::max)
        };
        let mut probabilities: Vec<f32> = self.logits.iter().map(|&l| (l - max).exp()).collect();
        let mut total = 0.0f32;
        for &p in &probabilities {
            total += p;
        }
        for p in &mut probabilities {
            *p /= total;
        }
        probabilities
    }

    fn top_p(&mut self, p: f32) {
        if p >= 1.0 || self.len() == 0 {
            return;
        }
        let probabilities = self.softmax();
        // the cumulative sum runs best first; the probabilities follow the
        // candidates through the sort
        let mut order: Vec<usize> = (0..self.len()).collect();
        if !self.sorted {
            order.sort_by(|&a, &b| {
                self.logits[b]
                    .total_cmp(&self.logits[a])
                    .then(self.ids[a].cmp(&self.ids[b]))
            });
        }
        let mut cumulative = 0.0f32;
        let mut last = order.len();
        for (i, &at) in order.iter().enumerate() {
            cumulative += probabilities[at];
            if cumulative >= p {
                last = i + 1;
                break;
            }
        }
        self.ids = order[..last].iter().map(|&i| self.ids[i]).collect();
        self.logits = order[..last].iter().map(|&i| self.logits[i]).collect();
        self.sorted = true;
    }

    fn min_p(&mut self, p: f32) {
        if p <= 0.0 || self.len() == 0 {
            return;
        }
        if !self.sorted {
            let max = self.logits.iter().copied().fold(-f32::MAX, f32::max);
            let least = max + p.ln();
            let kept: Vec<usize> = (0..self.len())
                .filter(|&i| self.logits[i] >= least)
                .collect();
            if !kept.is_empty() {
                self.ids = kept.iter().map(|&i| self.ids[i]).collect();
                self.logits = kept.iter().map(|&i| self.logits[i]).collect();
                return;
            }
        }
        self.sort();
        let least = self.logits[0] + p.ln();
        let kept = 1 + self.logits[1..].iter().take_while(|&&l| l >= least).count();
        self.truncate(kept);
    }

    fn divide(&mut self, temperature: f32) {
        for l in &mut self.logits {
            *l /= temperature;
        }
    }
}

/// What the runtime's chain and draw make of one position.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct RuntimePick {
    /// The token the draw picks.
    pub(super) token: u32,
    /// How many candidates the filters keep.
    pub(super) kept: usize,
}

/// The token the runtime picks at the uniform `u` from `logits`, the
/// node's logits with the turn's biases, by token. `filters` are the
/// chain's filters in order, and `temperature` what the logits are divided
/// by after the last of them.
pub(super) fn pick(logits: &[f64], filters: &[Filter], temperature: f64, u: f64) -> RuntimePick {
    let mut candidates = Candidates::new(logits);
    for filter in filters {
        match *filter {
            Filter::TopK(k) => candidates.top_k(k),
            Filter::TopP(p) => candidates.top_p(p as f32),
            Filter::MinP(p) => candidates.min_p(p as f32),
            Filter::Temperature(t) => candidates.divide(t as f32),
        }
    }
    let kept = candidates.len();
    // each temperature after the last filter divides in turn; their
    // product rounds the same up to the last bit, which the band covers
    if temperature != 1.0 {
        for l in &mut candidates.logits {
            *l = (f64::from(*l) / temperature) as f32;
        }
    }
    RuntimePick {
        token: draw(&candidates, u),
        kept,
    }
}

/// `skippy_receipt_draw_pick`: weights against the best logit in f64, a
/// weight that is not a number counting as 0, and the first candidate in
/// token id order whose running weight passes the uniform times the total.
/// When every weight is 0, the last candidate in token id order.
fn draw(candidates: &Candidates, u: f64) -> u32 {
    let Some(&first) = candidates.logits.first() else {
        return 0;
    };
    let best = candidates.logits.iter().copied().fold(first, f32::max);
    let weights: Vec<f64> = candidates
        .logits
        .iter()
        .map(|&l| (f64::from(l) - f64::from(best)).exp())
        .map(|w| if w.is_finite() { w } else { 0.0 })
        .collect();
    let total: f64 = weights.iter().sum();
    let mut order: Vec<usize> = (0..candidates.len()).collect();
    order.sort_by_key(|&i| candidates.ids[i]);
    let target = u * total;
    let mut running = 0.0;
    for &i in &order {
        running += weights[i];
        if running > target {
            return candidates.ids[i];
        }
    }
    order.last().map_or(0, |&i| candidates.ids[i])
}
