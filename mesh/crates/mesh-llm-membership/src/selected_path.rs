//! Dependency-neutral split-stage path selection contracts.
//!
//! `SelectedPathObservation` records the path type, RTT, and observed direct
//! remote address of a selected connection path. `SplitStagePathSnapshot` is
//! the reduced two-field snapshot used by serving/readiness surfaces. Both are
//! pure data types; the `iroh::endpoint::Connection` walk that produces a
//! `SelectedPathObservation` stays in the host (`mesh::stage_transport`), since
//! it is a serving-transport effect.

use std::net::SocketAddr;

/// A single observation of the selected path on a mesh control connection.
#[derive(Debug, Clone, Copy)]
pub struct SelectedPathObservation {
    pub path_type: &'static str,
    pub rtt_ms: Option<u32>,
    pub observed_direct_remote_addr: Option<SocketAddr>,
}

/// Reduced path-kind classification for serving/readiness surfaces.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SplitStagePathKind {
    Direct,
    Relay,
    Unknown,
}

/// Snapshot of the selected split-stage path for a peer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SplitStagePathSnapshot {
    pub kind: SplitStagePathKind,
    pub rtt_ms: Option<u32>,
}

impl SplitStagePathSnapshot {
    pub const fn direct(rtt_ms: Option<u32>) -> Self {
        Self {
            kind: SplitStagePathKind::Direct,
            rtt_ms,
        }
    }

    pub const fn relay(rtt_ms: Option<u32>) -> Self {
        Self {
            kind: SplitStagePathKind::Relay,
            rtt_ms,
        }
    }

    pub const fn unknown() -> Self {
        Self {
            kind: SplitStagePathKind::Unknown,
            rtt_ms: None,
        }
    }

    pub const fn with_direct_rtt_fallback(self, fallback_rtt_ms: Option<u32>) -> Self {
        match (self.kind, self.rtt_ms, fallback_rtt_ms) {
            (SplitStagePathKind::Direct, None, Some(rtt_ms)) => Self::direct(Some(rtt_ms)),
            _ => self,
        }
    }

    pub fn with_peer_path_fallback(self, fallback: Option<SelectedPathObservation>) -> Self {
        match (self.kind, fallback) {
            (SplitStagePathKind::Direct, Some(observation)) => {
                self.with_direct_rtt_fallback(observation.rtt_ms)
            }
            (SplitStagePathKind::Unknown, Some(observation)) => {
                split_stage_path_snapshot_from_observation(observation)
            }
            _ => self,
        }
    }

    /// Wire/doctor name for the path kind: "direct" | "relay" | "unknown".
    pub fn kind_name(self) -> &'static str {
        match self.kind {
            SplitStagePathKind::Direct => "direct",
            SplitStagePathKind::Relay => "relay",
            SplitStagePathKind::Unknown => "unknown",
        }
    }
}

pub fn split_stage_path_snapshot_from_observation(
    observation: SelectedPathObservation,
) -> SplitStagePathSnapshot {
    match observation.path_type {
        "direct" => SplitStagePathSnapshot::direct(observation.rtt_ms),
        "relay" => SplitStagePathSnapshot::relay(observation.rtt_ms),
        _ => SplitStagePathSnapshot::unknown(),
    }
}
