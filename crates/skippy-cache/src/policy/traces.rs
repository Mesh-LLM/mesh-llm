//! Deterministic synthetic traces for policy comparison (#1650 first slice).
//! Seeded xorshift so every run replays identically.

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// One cache access in a trace: which entry, and how valuable its reuse is.
#[derive(Debug, Clone, Copy)]
pub struct TraceAccess {
    pub entry: u64,
    pub cold_prefill_cost: f64,
    pub restore_cost: f64,
    pub exclusive_bytes: u64,
}

pub const HOT_ZIPF_ENTRIES: u64 = 64;

/// Zipf-like hotset: entry i is accessed with probability proportional to
/// 1/(i+1), so a small hotset dominates while a long tail streams once.
pub fn zipf_hotset_trace(seed: u64, len: usize) -> Vec<TraceAccess> {
    let mut rng = Rng::new(seed);
    let weights: Vec<f64> = (0..HOT_ZIPF_ENTRIES)
        .map(|i| 1.0 / (i as f64 + 1.0))
        .collect();
    let total: f64 = weights.iter().sum();
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        let pick = rng.next_f64() * total;
        let mut entry = 0;
        let mut acc = 0.0;
        for (i, w) in weights.iter().enumerate() {
            acc += w;
            if pick <= acc {
                entry = i as u64;
                break;
            }
        }
        out.push(access_for(entry, &mut rng));
    }
    out
}

/// Turn growth: each "session" revisits its whole history, extending it by a
/// fresh entry — the realistic agentic chat shape.
pub fn turn_growth_trace(seed: u64, sessions: u64, turns: u64) -> Vec<TraceAccess> {
    let mut rng = Rng::new(seed);
    let mut out = Vec::new();
    for session in 0..sessions {
        for turn in 0..turns {
            for entry in 0..=turn {
                out.push(access_for(session * 1000 + entry, &mut rng));
            }
        }
    }
    out
}

/// One-shot stream: every entry seen exactly once, occasionally interleaved
/// with a hot entry so the policy must not wreck the hotset.
pub fn one_shot_trace(seed: u64, len: usize) -> Vec<TraceAccess> {
    let mut rng = Rng::new(seed);
    let mut out = Vec::with_capacity(len);
    for i in 0..len as u64 {
        if i % 8 == 7 {
            out.push(access_for(0, &mut rng)); // hot anchor
        }
        out.push(access_for(1_000_000 + i, &mut rng));
    }
    out
}

/// Mixed sizes: footprints span three orders of magnitude.
pub fn mixed_size_trace(seed: u64, len: usize) -> Vec<TraceAccess> {
    let mut rng = Rng::new(seed);
    let mut out = Vec::with_capacity(len);
    for i in 0..len as u64 {
        let mut access = access_for(i % 32, &mut rng);
        let class = (i / 32) % 3;
        access.exclusive_bytes = match class {
            0 => 64 << 10,
            1 => 4 << 20,
            _ => 256 << 20,
        };
        out.push(access);
    }
    out
}

fn access_for(entry: u64, rng: &mut Rng) -> TraceAccess {
    let cold = 50.0 + rng.next_f64() * 400.0;
    TraceAccess {
        entry,
        cold_prefill_cost: cold,
        restore_cost: cold * 0.3,
        exclusive_bytes: (1 + rng.next_u64() % 8) << 20,
    }
}
