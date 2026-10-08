//! Versioned deterministic paired bootstrap with replacement and bounded work.
use sha2::{Digest, Sha256};

use super::statistics::{Evidence, summarize};
use crate::command::DynResult;

pub(super) const ALGORITHM: &str = "sha256-counter-paired-bootstrap-v1";
const MAX_RESAMPLES: usize = 1_000_000;
const MAX_DRAWS: usize = 100_000_000;

struct Stream {
    seed: [u8; 32],
    counter: u64,
    block: [u8; 32],
    offset: usize,
}

impl Stream {
    fn new(root: u64, group: &str, metric: &str) -> DynResult<Self> {
        let mut hash = Sha256::new();
        hash.update(ALGORITHM.as_bytes());
        hash.update([0]);
        hash.update(root.to_be_bytes());
        for text in [group, metric] {
            let length = u32::try_from(text.len())?;
            hash.update(length.to_be_bytes());
            hash.update(text.as_bytes());
        }
        Ok(Self {
            seed: hash.finalize().into(),
            counter: 0,
            block: [0; 32],
            offset: 32,
        })
    }

    fn next(&mut self) -> DynResult<u64> {
        if self.offset == 32 {
            let mut hash = Sha256::new();
            hash.update(self.seed);
            hash.update(self.counter.to_be_bytes());
            self.block = hash.finalize().into();
            self.counter = self
                .counter
                .checked_add(1)
                .ok_or("bootstrap stream exhausted")?;
            self.offset = 0;
        }
        let bytes = self.block[self.offset..self.offset + 8].try_into()?;
        self.offset += 8;
        Ok(u64::from_be_bytes(bytes))
    }

    fn index(&mut self, count: u64) -> DynResult<usize> {
        // Rejection avoids modulo bias when the population does not divide 2^64.
        let threshold = count.wrapping_neg() % count;
        for _ in 0..128 {
            let value = self.next()?;
            if value >= threshold {
                return Ok(usize::try_from(value % count)?);
            }
        }
        Err("bootstrap index rejection budget exhausted".into())
    }
}

fn admit(values: &[f64], resamples: usize) -> DynResult<()> {
    if values.is_empty()
        || values.iter().any(|v| !v.is_finite())
        || !(2..=MAX_RESAMPLES).contains(&resamples)
        || values
            .len()
            .checked_mul(resamples)
            .is_none_or(|n| n > MAX_DRAWS)
    {
        return Err(
            "bootstrap requires finite pairs, bounded resamples and bounded total work".into(),
        );
    }
    Ok(())
}

fn mean(values: &[f64]) -> DynResult<f64> {
    let mut result = 0.0;
    for (index, value) in values.iter().enumerate() {
        result += (value - result) / (index + 1) as f64;
    }
    if !result.is_finite() {
        return Err("paired bootstrap mean overflow".into());
    }
    Ok(result)
}

pub(super) fn bootstrap(
    values: &[f64],
    resamples: usize,
    root: u64,
    group: &str,
    metric: &str,
) -> DynResult<Evidence> {
    admit(values, resamples)?;
    let original = mean(values)?;
    let count = u64::try_from(values.len())?;
    let mut stream = Stream::new(root, group, metric)?;
    let mut means = Vec::with_capacity(resamples);
    for _ in 0..resamples {
        let mut sample = 0.0;
        for index in 0..values.len() {
            let value = values[stream.index(count)?];
            sample += (value - sample) / (index + 1) as f64;
        }
        if !sample.is_finite() {
            return Err("bootstrap resample mean overflow".into());
        }
        means.push(sample);
    }
    summarize(original, &means)
}

#[cfg(test)]
#[path = "resampling_tests.rs"]
mod tests;
