//! Peer heartbeat, relay recovery and removal policy.
//!
//! Consumes transport observations and returns decisions; the host owns I/O,
//! logging and applying those decisions to live connections.

use iroh::EndpointId;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HeartbeatFailurePolicy {
    pub allow_recent_inbound_grace: bool,
    pub failure_threshold: u32,
}

pub fn heartbeat_failure_policy(is_relay_only: bool) -> HeartbeatFailurePolicy {
    HeartbeatFailurePolicy {
        allow_recent_inbound_grace: true,
        // Relay-only peers are far more prone to transient timeouts.
        // Observed behaviour: a Sydney<->Sydney relay-only path (mini's VPN
        // extension blocking the LAN UDP hole-punch) can spike from 200ms
        // to 10s+ RTT during a single relay hiccup. With 60s heartbeat
        // intervals, two such cycles is ~2min — not enough grace for the
        // public mesh's relay to recover. Five cycles = 5min grace, which
        // covers the typical iroh relay path-renegotiation window.
        //
        // Direct paths stay at 2 — when the LAN/internet path is up at
        // all, two consecutive cycles of silence is a real failure signal.
        failure_threshold: if is_relay_only { 5 } else { 2 },
    }
}

pub const RELAY_HEALTH_CHECK_SECS: u64 = 30;
pub const RELAY_MISSING_GRACE_SECS: u64 = 180;
pub const RELAY_ONLY_RECONNECT_SECS: u64 = 1800;
pub const RELAY_ONLY_DIRECT_RESCUE_SECS: u64 = 60;
pub const RELAY_RECONNECT_COOLDOWN_SECS: u64 = 600;
pub const RELAY_DEGRADED_RTT_MS: u32 = 1500;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SelectedPathKind {
    Direct,
    Relay,
    #[default]
    Unknown,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RelayPathSnapshot {
    pub kind: SelectedPathKind,
    pub rtt_ms: Option<u32>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RelayPeerHealth {
    pub relay_since: Option<std::time::Instant>,
    pub last_reconnect_at: Option<std::time::Instant>,
}

impl RelayPeerHealth {
    pub fn observe(&mut self, snapshot: RelayPathSnapshot, now: std::time::Instant) {
        match snapshot.kind {
            SelectedPathKind::Direct => {
                self.relay_since = None;
            }
            SelectedPathKind::Relay => {
                if self.relay_since.is_none() {
                    self.relay_since = Some(now);
                }
            }
            SelectedPathKind::Unknown => {}
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RelayReconnectReason {
    RelayRttDegraded,
    RelayOnlyTooLong,
}

impl RelayReconnectReason {
    pub fn label(self) -> &'static str {
        match self {
            RelayReconnectReason::RelayRttDegraded => "relay RTT degraded",
            RelayReconnectReason::RelayOnlyTooLong => "relay path aged out",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HomeRelayStatusTransition {
    Missing { missing_secs: u64 },
    Restored,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RelayPeerObservation {
    pub peer_id: EndpointId,
    pub snapshot: RelayPathSnapshot,
    pub has_direct_candidate: bool,
}

#[derive(Default)]
pub struct RelayReconnectController {
    peer_health: HashMap<EndpointId, RelayPeerHealth>,
    relay_missing_since: Option<std::time::Instant>,
    relay_missing_reported: bool,
}

impl RelayReconnectController {
    pub fn observe_home_relay(
        &mut self,
        has_home_relay: bool,
        now: std::time::Instant,
    ) -> Option<HomeRelayStatusTransition> {
        if has_home_relay {
            self.relay_missing_reported = false;
            return self
                .relay_missing_since
                .take()
                .map(|_| HomeRelayStatusTransition::Restored);
        }

        let missing_since = *self.relay_missing_since.get_or_insert(now);
        if self.relay_missing_reported {
            return None;
        }

        let missing_secs = now.duration_since(missing_since).as_secs();
        if missing_secs >= RELAY_MISSING_GRACE_SECS {
            self.relay_missing_reported = true;
            return Some(HomeRelayStatusTransition::Missing { missing_secs });
        }
        None
    }

    pub fn plan_reconnect<I>(
        &mut self,
        observations: I,
        now: std::time::Instant,
        inflight_requests: u64,
        has_home_relay: bool,
    ) -> Option<(EndpointId, RelayReconnectReason)>
    where
        I: IntoIterator<Item = RelayPeerObservation>,
    {
        let mut observations: Vec<RelayPeerObservation> = observations.into_iter().collect();
        observations.sort_by_key(|observation| hex::encode(observation.peer_id.as_bytes()));

        if observations.is_empty() {
            self.peer_health.clear();
            return None;
        }

        let active_peers: std::collections::HashSet<EndpointId> = observations
            .iter()
            .map(|observation| observation.peer_id)
            .collect();
        self.peer_health
            .retain(|peer_id, _| active_peers.contains(peer_id));

        let mut stale_candidate: Option<(EndpointId, RelayReconnectReason)> = None;
        for observation in observations {
            let health = self.peer_health.entry(observation.peer_id).or_default();
            health.observe(observation.snapshot, now);

            let Some(reason) = relay_reconnect_reason(
                health,
                observation.snapshot,
                observation.has_direct_candidate,
                now,
                inflight_requests,
                has_home_relay,
            ) else {
                continue;
            };

            if reason == RelayReconnectReason::RelayRttDegraded {
                return Some((observation.peer_id, reason));
            }
            if stale_candidate.is_none() {
                stale_candidate = Some((observation.peer_id, reason));
            }
        }

        stale_candidate
    }

    pub fn record_reconnect_attempt(
        &mut self,
        peer_id: EndpointId,
        _reason: RelayReconnectReason,
        now: std::time::Instant,
    ) {
        let health = self.peer_health.entry(peer_id).or_default();
        health.last_reconnect_at = Some(now);
    }

    pub fn record_reconnect_result(
        &mut self,
        peer_id: EndpointId,
        succeeded: bool,
        now: std::time::Instant,
    ) {
        if succeeded {
            let health = self.peer_health.entry(peer_id).or_default();
            health.relay_since = Some(now);
        }
    }

    #[cfg(test)]
    pub fn peer_health(&self, peer_id: EndpointId) -> Option<&RelayPeerHealth> {
        self.peer_health.get(&peer_id)
    }
}

/// Classify the advertised paths of an existing connection.
/// An empty path set is lenient while its transport negotiates a path;
/// a missing connection is classified separately by `classify_relay_only_for_policy`.
pub fn is_relay_only_path_set<I: IntoIterator<Item = bool>>(path_is_ip_flags: I) -> bool {
    let mut iter = path_is_ip_flags.into_iter();
    let Some(first) = iter.next() else {
        // No path info at all — be lenient (likely a brand-new or
        // already-failing connection). Treat as relay-only so we don't
        // prematurely declare the peer dead before the path negotiator
        // has had a chance to settle.
        return true;
    };
    !first && !iter.any(|is_ip| is_ip)
}

/// Classify a peer as relay-only for failure-tolerance purposes.
///
/// `had_relay_only_connection` is `Some(true)` when we hold a live
/// `Connection` and `is_relay_only_connection` returned true,
/// `Some(false)` when we hold a Connection with at least one IP path,
/// and `None` when no Connection object is present at all (cleanly
/// closed, QUIC idle-expired, never opened).
///
/// When Connection is gone (`None`) we default to STRICT (not
/// relay-only). The lenient threshold exists to absorb mid-flap path
/// renegotiation, which only happens while iroh still holds the
/// Connection. Once the Connection is gone, a previously-direct peer
/// should not silently inherit the lenient grace and keep stale model
/// routes alive an extra few minutes.
pub fn classify_relay_only_for_policy(had_relay_only_connection: Option<bool>) -> bool {
    had_relay_only_connection.unwrap_or(false)
}

pub fn relay_reconnect_reason(
    health: &RelayPeerHealth,
    snapshot: RelayPathSnapshot,
    has_direct_candidate: bool,
    now: std::time::Instant,
    inflight_requests: u64,
    has_home_relay: bool,
) -> Option<RelayReconnectReason> {
    if inflight_requests > 0 || !has_home_relay {
        return None;
    }
    if health.last_reconnect_at.is_some_and(|last| {
        now.duration_since(last) < std::time::Duration::from_secs(RELAY_RECONNECT_COOLDOWN_SECS)
    }) {
        return None;
    }
    if snapshot.kind != SelectedPathKind::Relay {
        return None;
    }
    if snapshot
        .rtt_ms
        .is_some_and(|rtt_ms| rtt_ms >= RELAY_DEGRADED_RTT_MS)
    {
        return Some(RelayReconnectReason::RelayRttDegraded);
    }
    let relay_only_limit_secs = if has_direct_candidate {
        RELAY_ONLY_DIRECT_RESCUE_SECS
    } else {
        RELAY_ONLY_RECONNECT_SECS
    };
    if health.relay_since.is_some_and(|started| {
        now.duration_since(started) >= std::time::Duration::from_secs(relay_only_limit_secs)
    }) {
        return Some(RelayReconnectReason::RelayOnlyTooLong);
    }
    None
}

pub fn should_remove_connection(
    current_stable_id: Option<usize>,
    closing_stable_id: usize,
) -> bool {
    current_stable_id == Some(closing_stable_id)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PeerDownReportDisposition {
    SuppressReporterCooldown,
    RejectRecentlySeen,
    ProbeReachability,
}

pub fn peer_down_report_disposition(
    reporter_cooled: bool,
    recently_seen: bool,
) -> PeerDownReportDisposition {
    if reporter_cooled {
        PeerDownReportDisposition::SuppressReporterCooldown
    } else if recently_seen {
        PeerDownReportDisposition::RejectRecentlySeen
    } else {
        PeerDownReportDisposition::ProbeReachability
    }
}

/// Applies the reachability-confirmation rule for a `PeerDown` claim.
/// Returns `Some(dead_id)` if `dead_id != self_id` AND `should_remove` is `true` (peer confirmed gone).
/// Returns `None` if `dead_id == self_id` (never self-evict) or `should_remove` is `false` (peer still reachable).
pub fn resolve_peer_down(
    self_id: EndpointId,
    dead_id: EndpointId,
    should_remove: bool,
) -> Option<EndpointId> {
    if dead_id == self_id {
        return None;
    }
    if should_remove { Some(dead_id) } else { None }
}

pub fn default_heartbeat_failure_policy() -> HeartbeatFailurePolicy {
    HeartbeatFailurePolicy {
        allow_recent_inbound_grace: true,
        failure_threshold: 2,
    }
}

#[cfg(test)]
mod tests;
