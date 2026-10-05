//! Cumulative compute-busy time for one stage.
//!
//! Every token of every request passes through every stage of a split, so over
//! the same window each stage does the same logical work and the stage with
//! the most busy time is the one pacing the pipeline. A stage's effective
//! decode rate is the weight bytes it holds divided by its busy time; the
//! ratio between stages is what auto-balance placement balances.
//!
//! The iteration scheduler runs all native compute on one worker thread, so
//! summing the durations it records here gives busy time without double
//! counting batched lanes.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

#[derive(Debug, Default)]
pub struct StageComputeMeter {
    busy_nanos: AtomicU64,
    operations: AtomicU64,
    decode_tokens: AtomicU64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct StageComputeSnapshot {
    /// Total time the stage's runtime spent computing.
    pub busy_nanos: u64,
    /// Runtime operations counted into `busy_nanos`.
    pub operations: u64,
    /// Single-token decode steps run on this stage. Every generated token
    /// takes one decode step on every stage, so on stage 0 this is the
    /// split's decode throughput counter.
    pub decode_tokens: u64,
}

impl StageComputeMeter {
    pub fn record(&self, elapsed: Duration) {
        let nanos = u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX);
        self.busy_nanos.fetch_add(nanos, Ordering::Relaxed);
        self.operations.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_decode_tokens(&self, tokens: u64) {
        self.decode_tokens.fetch_add(tokens, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> StageComputeSnapshot {
        StageComputeSnapshot {
            busy_nanos: self.busy_nanos.load(Ordering::Relaxed),
            operations: self.operations.load(Ordering::Relaxed),
            decode_tokens: self.decode_tokens.load(Ordering::Relaxed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulates_busy_time_and_operations() {
        let meter = StageComputeMeter::default();
        meter.record(Duration::from_millis(3));
        meter.record(Duration::from_millis(4));
        meter.record_decode_tokens(5);

        assert_eq!(
            meter.snapshot(),
            StageComputeSnapshot {
                busy_nanos: 7_000_000,
                operations: 2,
                decode_tokens: 5,
            }
        );
    }
}
