//! Replays the draw of a sampled turn.
//!
//! A turn whose statement names the [`crate::turn::INDEXED_DRAW`] rule drew
//! the token at position `i` of its sequence with [`uniform`]`(seed, i)`:
//! over the candidates its filters kept, in token id order, it took the
//! first whose running softmax weight passed the uniform times the total.
//! The seed, the position and the logits fix the token, so a client that
//! checked the logits finds the token the node had to take.
//!
//! The logits the draw is replayed on may differ from the ones the node
//! drew from, each within a band. The band can move a boundary between two
//! tokens, and it can move a token into or out of the kept candidates. So a
//! token passes when the uniform lies in the widest interval the band
//! allows for it ([`judge`]): the interval's lower end takes every weight
//! below the token at its least and every other weight at its most, and
//! the upper end the reverse. A draw that passes only with the band is
//! counted, since a node can steer the draws that land that close to a
//! boundary and no others.
//!
//! The band is narrow ([`super::DRAW_BAND`]): the draw is replayed on the
//! logits the node opened, which it drew from, so only the last bits of
//! `exp` can differ.
//!
//! The draw without the band is the runtime's own arithmetic
//! ([`runtime_chain`]): its filters work in f32, and an f32 top-p can cut
//! the candidates before an f64 one would. The token that arithmetic picks
//! passes, and the candidates it keeps widen the sets the band allows.

use super::sampling::Filter;

mod runtime_chain;

/// The uniform in `[0, 1)` that picks the token at position `index` of a
/// sequence drawn with `seed`: the 53 high bits of SplitMix64 over the seed
/// and the low 32 bits of the index. The node computes the same value
/// (`skippy_receipt_draw_uniform`).
pub fn uniform(seed: u32, index: u64) -> f64 {
    let mut x = (u64::from(seed) << 32) | (index & 0xFFFF_FFFF);
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    (x >> 11) as f64 / (1u64 << 53) as f64
}

/// The draw a statement names: its seed, and the temperature the logits
/// were divided by before the draw.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IndexedDraw {
    pub seed: u32,
    pub temperature: f64,
}

/// One replayed draw.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DrawVerdict {
    /// The uniform of the draw.
    pub uniform: f64,
    /// The uniforms that pick the sampled token, at the widest the band
    /// allows: from `lower` up to `upper`, which is not included.
    pub lower: f64,
    pub upper: f64,
    /// The uniform picks the sampled token on the checked logits as they
    /// are, with no help from the band.
    pub exact: bool,
    pub pass: bool,
}

/// The candidates in order of their logits, best first.
struct Ranked {
    /// Logits, best first.
    logits: Vec<f64>,
    /// By token, its place in `logits`.
    rank: Vec<u32>,
}

impl Ranked {
    fn new(logits: &[f64]) -> Self {
        let mut order: Vec<u32> = (0..logits.len() as u32).collect();
        order.sort_unstable_by(|&a, &b| {
            logits[b as usize]
                .total_cmp(&logits[a as usize])
                .then(a.cmp(&b))
        });
        let mut rank = vec![0u32; logits.len()];
        for (place, &token) in order.iter().enumerate() {
            rank[token as usize] = place as u32;
        }
        Self {
            logits: order.iter().map(|&t| logits[t as usize]).collect(),
            rank,
        }
    }
}

/// How many of the best `sorted` logits the filters keep, applied in order
/// as the runtime's sampler chain applies them.
fn kept(sorted: &[f64], filters: &[Filter]) -> usize {
    let Some(&best) = sorted.first() else {
        return 0;
    };
    let mut n = sorted.len();
    // what the logits are multiplied by, after the temperatures so far
    let mut scale = 1.0;
    for filter in filters {
        match *filter {
            Filter::TopK(0) => {}
            Filter::TopK(k) => n = n.min(k as usize),
            Filter::TopP(p) if p >= 1.0 => {}
            Filter::TopP(p) => {
                let weight = |l: f64| ((l - best) * scale).exp();
                let total: f64 = sorted[..n].iter().map(|&l| weight(l)).sum();
                let mut running = 0.0;
                for (i, &l) in sorted[..n].iter().enumerate() {
                    running += weight(l);
                    if running >= p * total {
                        n = i + 1;
                        break;
                    }
                }
            }
            Filter::MinP(p) if p <= 0.0 => {}
            Filter::MinP(p) => {
                let least = best * scale + p.ln();
                n = sorted[..n].partition_point(|&l| l * scale >= least).max(1);
            }
            Filter::Temperature(t) => scale /= t,
        }
    }
    n
}

/// Whether the token at `place` can be kept when every logit moves by up to
/// `band`: the logits above it at their least, itself and the rest at
/// their most. Only differences matter, so the logits above move down by
/// twice the band.
fn possibly_kept(ranked: &Ranked, filters: &[Filter], place: usize, band: f64) -> bool {
    let (above, rest) = ranked.logits.split_at(place);
    let token = rest[0];
    let lowered: Vec<f64> = above.iter().map(|l| l - 2.0 * band).collect();
    // the lowered logits that still beat the token stay above it
    let still_above = lowered.partition_point(|&l| l > token);
    let mut moved = Vec::with_capacity(ranked.logits.len());
    moved.extend_from_slice(&lowered[..still_above]);
    merge_descending(&mut moved, &lowered[still_above..], rest);
    // the token comes first among its equals
    let at = moved.partition_point(|&l| l > token);
    at < kept(&moved, filters)
}

/// Whether the token at `place` is kept however the logits move by up to
/// `band`: itself at its least and every other at its most, so it moves
/// down by twice the band.
fn surely_kept(ranked: &Ranked, filters: &[Filter], place: usize, band: f64) -> bool {
    let token = ranked.logits[place] - 2.0 * band;
    let mut moved = Vec::with_capacity(ranked.logits.len());
    moved.extend_from_slice(&ranked.logits[..place]);
    moved.extend_from_slice(&ranked.logits[place + 1..]);
    // the token comes last among its equals
    let at = moved.partition_point(|&l| l >= token);
    moved.insert(at, token);
    at < kept(&moved, filters)
}

/// The first of `places` places where `stops` holds, or `places` when it
/// holds at none. `stops` holds from one place on.
fn first_place(places: usize, stops: impl Fn(usize) -> bool) -> usize {
    let (mut low, mut high) = (0, places);
    while low < high {
        let middle = low + (high - low) / 2;
        if stops(middle) {
            high = middle;
        } else {
            low = middle + 1;
        }
    }
    low
}

/// Appends `a` and `b`, both best first, to `out` in that order.
fn merge_descending(out: &mut Vec<f64>, a: &[f64], b: &[f64]) {
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        if a[i] > b[j] {
            out.push(a[i]);
            i += 1;
        } else {
            out.push(b[j]);
            j += 1;
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
}

/// The weights of a draw, by token, with each token's place by logit.
struct Bound<'a> {
    weights: &'a [f64],
    rank: &'a [u32],
    sampled: usize,
}

impl Bound<'_> {
    /// The share of the weight before `end`, a token id, among all of it:
    /// tokens before `end` count when their place is under `before`, at
    /// `before_factor` of their weight, and the others under `after`, at
    /// `after_factor`.
    fn share(&self, end: usize, before: (usize, f64), after: (usize, f64)) -> f64 {
        let mut head = 0.0;
        let mut tail = 0.0;
        for (token, (&w, &place)) in self.weights.iter().zip(self.rank).enumerate() {
            let place = place as usize;
            if token < end {
                if place < before.0 {
                    head += w * before.1;
                }
            } else if place < after.0 {
                tail += w * after.1;
            }
        }
        if head + tail > 0.0 {
            head / (head + tail)
        } else {
            0.0
        }
    }

    /// The uniforms that pick the sampled token, at the widest: from the
    /// lower end up to the upper, which is not included. The lower end
    /// counts the `sure` best candidates before the token at their weight
    /// divided by `factor`, and the `possible` best from the token on at
    /// their weight times `factor`. The upper end counts the `possible`
    /// best up to the token at their most and the `sure` best after it at
    /// their least.
    fn interval(&self, sure: usize, possible: usize, factor: f64) -> (f64, f64) {
        let lower = self.share(self.sampled, (sure, 1.0 / factor), (possible, factor));
        let upper = self.share(self.sampled + 1, (possible, factor), (sure, 1.0 / factor));
        (lower, upper)
    }
}

/// Replays the draw of token `sampled` at position `index` from `logits`,
/// which already hold the turn's logit biases. `filters` are the turn's
/// candidate filters in order, and `band` the slack of each logit for the
/// node's rounding.
pub(crate) fn judge(
    draw: IndexedDraw,
    filters: &[Filter],
    logits: &[f64],
    sampled: usize,
    index: u64,
    band: f64,
) -> DrawVerdict {
    let uniform = uniform(draw.seed, index);
    let ranked = Ranked::new(logits);
    let n = ranked.logits.len();
    // the temperatures between filters divide in the chain, the rest after
    let within: f64 = filters
        .iter()
        .map(|f| match f {
            Filter::Temperature(t) => *t,
            _ => 1.0,
        })
        .product();
    let runtime = runtime_chain::pick(logits, filters, draw.temperature / within, uniform);
    let kept_here = kept(&ranked.logits, filters);
    // a better logit is kept whenever a worse one is, so each set is the
    // best logits up to one place
    let possible = first_place(n, |place| !possibly_kept(&ranked, filters, place, band))
        .max(kept_here)
        .max(runtime.kept);
    let sure = first_place(n, |place| !surely_kept(&ranked, filters, place, band))
        .min(kept_here)
        .min(runtime.kept);
    let best = ranked.logits.first().copied().unwrap_or(0.0);
    let weights: Vec<f64> = logits
        .iter()
        .map(|l| ((l - best) / draw.temperature).exp())
        .map(|w| if w.is_finite() { w } else { 0.0 })
        .collect();
    let bound = Bound {
        weights: &weights,
        rank: &ranked.rank,
        sampled,
    };
    let place = ranked.rank[sampled] as usize;
    let exact = runtime.token as usize == sampled;
    let (lower, upper) = bound.interval(sure, possible, (band / draw.temperature).exp());
    DrawVerdict {
        uniform,
        lower,
        upper,
        exact,
        pass: exact || (place < possible && (lower..upper).contains(&uniform)),
    }
}

#[cfg(test)]
mod tests;
