//! Deterministic draws from a VRF output.
//!
//! A challenge and its checked rows must be reproducible by anyone who
//! holds the VRF proof, so they cannot depend on a random number library
//! whose algorithms may change between versions. A [`DrawStream`] is the
//! BLAKE3 XOF of the VRF output under a label, read eight bytes at a time;
//! [`DrawStream::below`] draws uniformly by rejection.

/// A stream of uniform draws, keyed by a seed and a label.
pub struct DrawStream {
    reader: blake3::OutputReader,
}

impl DrawStream {
    /// `label` separates independent draws from one seed, for example the
    /// challenge and the rows of each weight.
    pub fn new(seed: &[u8], label: &str) -> Self {
        let mut hasher = blake3::Hasher::new_derive_key("mesh-llm receipts draw v1");
        hasher.update(&(label.len() as u64).to_le_bytes());
        hasher.update(label.as_bytes());
        hasher.update(seed);
        Self {
            reader: hasher.finalize_xof(),
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut bytes = [0u8; 8];
        self.reader.fill(&mut bytes);
        u64::from_le_bytes(bytes)
    }

    /// A uniform draw from `0..n`. `n` must be positive.
    pub fn below(&mut self, n: u64) -> u64 {
        assert!(n > 0, "a draw from an empty range");
        // the largest multiple of n that fits, so every residue is equally likely
        let zone = u64::MAX - (u64::MAX - n + 1) % n;
        loop {
            let value = self.next_u64();
            if value <= zone {
                return value % n;
            }
        }
    }

    /// A uniform draw from `range`, which must not be empty.
    pub fn in_range(&mut self, range: std::ops::Range<u32>) -> u32 {
        let span = u64::from(range.end - range.start);
        range.start + u32::try_from(self.below(span)).expect("a draw below a u32 span")
    }

    /// A uniform draw from `0..1`, for an audit-rate decision.
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_same_seed_and_label_give_the_same_draws() {
        let draws = |seed: &[u8], label| {
            let mut s = DrawStream::new(seed, label);
            (0..8).map(|_| s.below(1000)).collect::<Vec<_>>()
        };
        assert_eq!(draws(b"seed", "a"), draws(b"seed", "a"));
        assert_ne!(draws(b"seed", "a"), draws(b"seed", "b"));
        assert_ne!(draws(b"seed", "a"), draws(b"seee", "a"));
    }

    #[test]
    fn draws_stay_in_range_and_cover_it() {
        let mut s = DrawStream::new(b"x", "y");
        let mut seen = [false; 7];
        for _ in 0..500 {
            let v = s.below(7);
            seen[v as usize] = true;
        }
        assert!(seen.iter().all(|&s| s));
        for _ in 0..100 {
            assert!((10..13).contains(&s.in_range(10..13)));
            assert!((0.0..1.0).contains(&s.unit()));
        }
        assert_eq!(s.below(1), 0);
        let _ = s.below(u64::MAX);
    }

    #[test]
    fn the_stream_is_pinned() {
        // a change here changes every challenge and invalidates old evidence
        let mut s = DrawStream::new(&[0u8; 64], "challenge");
        let first = s.next_u64();
        assert_eq!(first, DrawStream::new(&[0u8; 64], "challenge").next_u64());
        assert_eq!(first, 12_899_550_690_765_162_838);
    }
}
