//! Channel-specific monotonic sequence generators for replay channels.
//!
//! Each [`ReplayChannel`] gets its own `Arc<AtomicU64>` counter, producing strictly increasing
//! sequence numbers that survive cloning of the owning guard or service handle.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use mesh_llm_events::logging::replay::{ReplayChannel, ReplaySequence};

/// Monotonic sequence generator for a single replay channel.
#[derive(Debug)]
struct ChannelCounter {
    counter: AtomicU64,
}

impl ChannelCounter {
    fn next(&self) -> u64 {
        self.counter.fetch_add(1, Ordering::Relaxed) + 1 // 1-based sequences
    }
}

/// Per-channel monotonic sequence generator.
///
/// Cloning this type shares the same underlying counters — sequence numbers remain strictly increasing regardless of how many clones exist.
#[derive(Clone, Debug)]
pub struct SequenceGenerators {
    requests: Arc<ChannelCounter>,
    operations: Arc<ChannelCounter>,
    system: Arc<ChannelCounter>,
}

impl SequenceGenerators {
    /// Create a new set of sequence generators (all counters start at 0).
    pub fn new() -> Self {
        Self {
            requests: Arc::new(ChannelCounter {
                counter: AtomicU64::new(0),
            }),
            operations: Arc::new(ChannelCounter {
                counter: AtomicU64::new(0),
            }),
            system: Arc::new(ChannelCounter {
                counter: AtomicU64::new(0),
            }),
        }
    }

    /// Generate the next sequence for the given channel. Returns a 1-based monotonically increasing value per channel.
    pub fn next(&self, channel: ReplayChannel) -> u64 {
        match channel {
            ReplayChannel::Requests => self.requests.next(),
            ReplayChannel::Operations => self.operations.next(),
            ReplayChannel::System => self.system.next(),
        }
    }

    /// Build a [`ReplaySequence`] for the given channel using this generator.
    pub fn next_sequence(&self, channel: ReplayChannel) -> ReplaySequence {
        let seq = self.next(channel);
        ReplaySequence::next(channel, seq)
    }

    /// Current counter value for a channel (for observability).
    #[allow(dead_code)]
    pub fn current(&self, channel: ReplayChannel) -> u64 {
        match channel {
            ReplayChannel::Requests => self.requests.counter.load(Ordering::Relaxed),
            ReplayChannel::Operations => self.operations.counter.load(Ordering::Relaxed),
            ReplayChannel::System => self.system.counter.load(Ordering::Relaxed),
        }
    }

    /// Verify that the generators remain consistent after cloning.
    #[allow(dead_code)]
    pub fn assert_counters_shared(&self, other: &Self) -> bool {
        // Clones share Arc<ChannelCounter> — counters should be at the same address conceptually.
        // We verify by checking both see the same current value before any new next() call.
        for ch in [
            ReplayChannel::Requests,
            ReplayChannel::Operations,
            ReplayChannel::System,
        ] {
            if self.current(ch) != other.current(ch) {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequences_are_strictly_increasing_per_channel() {
        let seq_gen = SequenceGenerators::new();

        for _ in 0..100 {
            let s = seq_gen.next(ReplayChannel::Requests);
            assert!(s > 0); // 1-based
        }

        // Each call returns a higher value than the previous.
        let first = seq_gen.current(ReplayChannel::Requests);
        for _ in 0..5 {
            seq_gen.next(ReplayChannel::Requests);
        }
        assert!(seq_gen.current(ReplayChannel::Requests) > first);

        // Other channels are independent.
        assert_eq!(seq_gen.current(ReplayChannel::Operations), 0);
    }

    #[test]
    fn clone_shares_counters() {
        let seq_gen = SequenceGenerators::new();
        seq_gen.next(ReplayChannel::System);
        assert_eq!(seq_gen.current(ReplayChannel::System), 1);

        let cloned = seq_gen.clone();
        // Both see the same counter state.
        assert_eq!(cloned.current(ReplayChannel::System), 1);

        // Advance via clone → original sees it too.
        cloned.next(ReplayChannel::System);
        assert_eq!(seq_gen.current(ReplayChannel::System), 2);
    }

    #[test]
    fn next_sequence_builds_replay_sequence() {
        let seq_gen = SequenceGenerators::new();
        let seq = seq_gen.next_sequence(ReplayChannel::Operations);
        assert_eq!(seq.channel, ReplayChannel::Operations);
        assert_eq!(seq.sequence, 1);

        let seq2 = seq_gen.next_sequence(ReplayChannel::Operations);
        assert_eq!(seq2.sequence, 2);
    }

    #[test]
    fn channels_are_independent() {
        let seq_gen = SequenceGenerators::new();

        // Advance Requests 5 times.
        for _ in 0..5 {
            seq_gen.next(ReplayChannel::Requests);
        }

        assert_eq!(seq_gen.current(ReplayChannel::Requests), 5);
        assert_eq!(seq_gen.current(ReplayChannel::Operations), 0);
        assert_eq!(seq_gen.current(ReplayChannel::System), 0);
    }
}
