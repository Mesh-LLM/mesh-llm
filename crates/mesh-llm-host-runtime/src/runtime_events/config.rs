//! Frozen bounds for the host runtime-event engine.
//!
//! Every numeric limit named in the plan's "Frozen engine bounds" table lives
//! here as the single private configuration owner. `EngineConfig` is exposed
//! to health through `crate::runtime_events::health`. Changing any value
//! requires the plan's inventory amendment procedure.

use std::time::Duration;

/// Bound on host-initiated request-root operations.
///
/// Mirrors `MAX_TRACKED_REQUESTS` in `logging/openai_lifecycle.rs` (currently
/// `1_024`). That constant is private to its owning module and is treated
/// there as a reference pattern, not an editable file, so this value is
/// pinned by literal plus a contract test rather than a live import.
pub const REQUEST_ROOT_BOUND: usize = 1_024;

/// Observed-child operations admitted per request root.
pub const CHILD_MULTIPLIER: usize = 2;

/// Runtime/model lifecycle operations outside the request-root tree.
pub const LIFECYCLE_OPERATION_BOUND: usize = 64;

/// `request_root_bound * (1 + child_multiplier) + lifecycle_operation_bound`.
pub const RESERVATION_TABLE_CAPACITY: usize =
    REQUEST_ROOT_BOUND * (1 + CHILD_MULTIPLIER) + LIFECYCLE_OPERATION_BOUND;

/// State-transition lane depth.
pub const STATE_TRANSITION_LANE_DEPTH: usize = 4_096;

/// Diagnostic lane depth.
pub const DIAGNOSTIC_LANE_DEPTH: usize = 2_048;

/// Wake list depth, equal to the reservation table.
pub const WAKE_LIST_DEPTH: usize = RESERVATION_TABLE_CAPACITY;

/// Replay retention: frame count, age, and byte ceilings (first limit wins).
pub const REPLAY_MAX_FRAMES: usize = 4_096;
pub const REPLAY_MAX_AGE: Duration = Duration::from_secs(300);
pub const REPLAY_MAX_BYTES: usize = 8 * 1024 * 1024;

/// Per-subscriber lag: frame count, age, and byte ceilings (first limit wins).
pub const SUBSCRIBER_LAG_MAX_FRAMES: usize = 1_024;
pub const SUBSCRIBER_LAG_MAX_AGE: Duration = Duration::from_secs(30);
pub const SUBSCRIBER_LAG_MAX_BYTES: usize = 4 * 1024 * 1024;

/// Maximum concurrent v1 subscribers.
pub const MAX_CONCURRENT_SUBSCRIBERS: usize = 32;

/// Reconnect limit: connects per client key per window; client key is peer IP.
pub const RECONNECT_LIMIT_PER_WINDOW: usize = 10;
pub const RECONNECT_WINDOW: Duration = Duration::from_secs(60);

/// SSE keepalive interval.
pub const KEEPALIVE_INTERVAL: Duration = Duration::from_secs(15);

/// Health publish cadence: coalesced, at most one frame per second.
pub const HEALTH_PUBLISH_MIN_INTERVAL: Duration = Duration::from_secs(1);

/// Progress presentation/export interval.
pub const PROGRESS_EXPORT_INTERVAL: Duration = Duration::from_millis(100);

/// Existing `PRETTY_TUI_REDRAW_INTERVAL`.
pub const TUI_RENDER_TICK: Duration = Duration::from_millis(33);

/// Shutdown drain deadline.
pub const SHUTDOWN_DRAIN_DEADLINE: Duration = Duration::from_secs(2);

/// Child-settle grace before root release (task 5,
/// `.omo/plans/event-system-fixes.md`): how long a root whose own terminal
/// has already applied and published may hold its slot open while at
/// least one child is still occupied, before the engine synthesizes a
/// `terminal_not_delivered` for each remaining child and releases the
/// root anyway. Frozen equal to `SHUTDOWN_DRAIN_DEADLINE`.
pub const CHILD_SETTLE_GRACE: Duration = SHUTDOWN_DRAIN_DEADLINE;

/// Callback ingress p99 budget on certification hosts.
pub const CALLBACK_INGRESS_P99_BUDGET: Duration = Duration::from_micros(100);

/// Read-only snapshot of the frozen bounds, exposed through engine health.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EngineConfig {
    pub reservation_table_capacity: usize,
    pub state_transition_lane_depth: usize,
    pub diagnostic_lane_depth: usize,
    pub wake_list_depth: usize,
    pub replay_max_frames: usize,
    pub subscriber_lag_max_frames: usize,
    pub max_concurrent_subscribers: usize,
}

impl EngineConfig {
    pub const FROZEN: Self = Self {
        reservation_table_capacity: RESERVATION_TABLE_CAPACITY,
        state_transition_lane_depth: STATE_TRANSITION_LANE_DEPTH,
        diagnostic_lane_depth: DIAGNOSTIC_LANE_DEPTH,
        wake_list_depth: WAKE_LIST_DEPTH,
        replay_max_frames: REPLAY_MAX_FRAMES,
        subscriber_lag_max_frames: SUBSCRIBER_LAG_MAX_FRAMES,
        max_concurrent_subscribers: MAX_CONCURRENT_SUBSCRIBERS,
    };
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self::FROZEN
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacity_derivation_matches_frozen_formula() {
        assert_eq!(REQUEST_ROOT_BOUND, 1_024);
        assert_eq!(CHILD_MULTIPLIER, 2);
        assert_eq!(LIFECYCLE_OPERATION_BOUND, 64);
        assert_eq!(RESERVATION_TABLE_CAPACITY, 3_136);
    }

    #[test]
    fn wake_list_depth_equals_reservation_table() {
        assert_eq!(WAKE_LIST_DEPTH, RESERVATION_TABLE_CAPACITY);
    }

    #[test]
    fn frozen_config_exposes_all_bounds() {
        let config = EngineConfig::default();
        assert_eq!(config.reservation_table_capacity, 3_136);
        assert_eq!(config.state_transition_lane_depth, 4_096);
        assert_eq!(config.diagnostic_lane_depth, 2_048);
        assert_eq!(config.replay_max_frames, 4_096);
        assert_eq!(config.subscriber_lag_max_frames, 1_024);
        assert_eq!(config.max_concurrent_subscribers, 32);
    }
}
