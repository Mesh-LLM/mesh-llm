//! Replay channels and sequencing for ordered event streams.

use serde::{Deserialize, Serialize};

/// Logical channel separating different classes of replayable events.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ReplayChannel {
    /// High-level request lifecycle events (admission → completion/failure).
    Requests,
    /// Internal operational events (attempts, proxy hops).
    Operations,
    /// System-level events (startup, shutdown, health).
    System,
}

/// Ordered position within a replay channel.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct ReplaySequence {
    pub channel: ReplayChannel,
    pub sequence: u64,
}

impl ReplaySequence {
    /// Create a new sequence entry for the given channel.
    pub fn next(channel: ReplayChannel, seq: u64) -> Self {
        Self {
            channel,
            sequence: seq,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_serde_tags() {
        let json = serde_json::to_string(&ReplayChannel::Requests).unwrap();
        assert!(json.contains("requests"));

        let parsed: ReplayChannel = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, ReplayChannel::Requests);
    }

    #[test]
    fn test_sequence_roundtrip() {
        let seq = ReplaySequence {
            channel: ReplayChannel::Operations,
            sequence: 42,
        };
        let json = serde_json::to_string(&seq).unwrap();
        let parsed: ReplaySequence = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.channel, ReplayChannel::Operations);
        assert_eq!(parsed.sequence, 42);
    }

    #[test]
    fn test_all_channel_variants() {
        for ch in [
            ReplayChannel::Requests,
            ReplayChannel::Operations,
            ReplayChannel::System,
        ] {
            let json = serde_json::to_string(&ch).unwrap();
            let parsed: ReplayChannel = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, ch);
        }
    }

    #[test]
    fn test_sequence_next() {
        let seq = ReplaySequence::next(ReplayChannel::System, 7);
        assert_eq!(seq.channel, ReplayChannel::System);
        assert_eq!(seq.sequence, 7);
    }
}
