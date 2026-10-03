use super::*;

fn make_test_endpoint_id(seed: u8) -> EndpointId {
    iroh::SecretKey::from_bytes(&[seed; 32]).public()
}

#[test]
fn relay_health_prefers_direct_paths_and_clears_relay_age() {
    let now = std::time::Instant::now();
    let mut health = RelayPeerHealth::default();
    health.observe(
        RelayPathSnapshot {
            kind: SelectedPathKind::Relay,
            rtt_ms: Some(240),
        },
        now - std::time::Duration::from_secs(RELAY_ONLY_RECONNECT_SECS + 5),
    );
    assert!(
        health.relay_since.is_some(),
        "relay age should start on relay path"
    );

    health.observe(
        RelayPathSnapshot {
            kind: SelectedPathKind::Direct,
            rtt_ms: Some(18),
        },
        now,
    );
    assert!(
        health.relay_since.is_none(),
        "direct path should clear relay-only aging"
    );
}

#[test]
fn relay_health_reconnects_degraded_relay_paths() {
    let now = std::time::Instant::now();
    let mut health = RelayPeerHealth::default();
    health.observe(
        RelayPathSnapshot {
            kind: SelectedPathKind::Relay,
            rtt_ms: Some(RELAY_DEGRADED_RTT_MS + 50),
        },
        now - std::time::Duration::from_secs(30),
    );

    assert_eq!(
        relay_reconnect_reason(
            &health,
            RelayPathSnapshot {
                kind: SelectedPathKind::Relay,
                rtt_ms: Some(RELAY_DEGRADED_RTT_MS + 50),
            },
            false,
            now,
            0,
            true,
        ),
        Some(RelayReconnectReason::RelayRttDegraded)
    );
}

#[test]
fn relay_health_rescues_known_direct_candidate_after_short_grace() {
    let now = std::time::Instant::now();
    let mut health = RelayPeerHealth::default();
    health.observe(
        RelayPathSnapshot {
            kind: SelectedPathKind::Relay,
            rtt_ms: Some(260),
        },
        now - std::time::Duration::from_secs(RELAY_ONLY_DIRECT_RESCUE_SECS + 1),
    );

    assert_eq!(
        relay_reconnect_reason(
            &health,
            RelayPathSnapshot {
                kind: SelectedPathKind::Relay,
                rtt_ms: Some(260),
            },
            true,
            now,
            0,
            true,
        ),
        Some(RelayReconnectReason::RelayOnlyTooLong)
    );
    assert_eq!(
        relay_reconnect_reason(
            &health,
            RelayPathSnapshot {
                kind: SelectedPathKind::Relay,
                rtt_ms: Some(260),
            },
            false,
            now,
            0,
            true,
        ),
        None,
        "relay-only peers without a direct candidate keep the longer fallback grace"
    );
}

#[test]
fn relay_health_reconnects_long_lived_relay_paths() {
    let now = std::time::Instant::now();
    let mut health = RelayPeerHealth::default();
    health.observe(
        RelayPathSnapshot {
            kind: SelectedPathKind::Relay,
            rtt_ms: Some(260),
        },
        now - std::time::Duration::from_secs(RELAY_ONLY_RECONNECT_SECS + 5),
    );

    assert_eq!(
        relay_reconnect_reason(
            &health,
            RelayPathSnapshot {
                kind: SelectedPathKind::Relay,
                rtt_ms: Some(260),
            },
            false,
            now,
            0,
            true,
        ),
        Some(RelayReconnectReason::RelayOnlyTooLong)
    );
}

#[test]
fn relay_health_respects_cooldown_and_inflight_requests() {
    let now = std::time::Instant::now();
    let mut health = RelayPeerHealth::default();
    health.observe(
        RelayPathSnapshot {
            kind: SelectedPathKind::Relay,
            rtt_ms: Some(RELAY_DEGRADED_RTT_MS + 10),
        },
        now - std::time::Duration::from_secs(30),
    );
    health.last_reconnect_at =
        Some(now - std::time::Duration::from_secs(RELAY_RECONNECT_COOLDOWN_SECS - 1));

    assert_eq!(
        relay_reconnect_reason(
            &health,
            RelayPathSnapshot {
                kind: SelectedPathKind::Relay,
                rtt_ms: Some(RELAY_DEGRADED_RTT_MS + 10),
            },
            false,
            now,
            0,
            true,
        ),
        None,
        "cooldown should suppress immediate retry"
    );

    health.last_reconnect_at = None;
    assert_eq!(
        relay_reconnect_reason(
            &health,
            RelayPathSnapshot {
                kind: SelectedPathKind::Relay,
                rtt_ms: Some(RELAY_DEGRADED_RTT_MS + 10),
            },
            false,
            now,
            1,
            true,
        ),
        None,
        "active requests should suppress relay refresh"
    );
    assert_eq!(
        relay_reconnect_reason(
            &health,
            RelayPathSnapshot {
                kind: SelectedPathKind::Relay,
                rtt_ms: Some(RELAY_DEGRADED_RTT_MS + 10),
            },
            false,
            now,
            0,
            false,
        ),
        None,
        "missing home relay should suppress churn"
    );
}

#[test]
fn relay_reconnect_controller_prioritizes_degraded_rtt_over_aged_relay() {
    let now = std::time::Instant::now();
    let degraded_peer = make_test_endpoint_id(21);
    let aged_peer = make_test_endpoint_id(22);
    let mut controller = RelayReconnectController::default();

    let initial = now - std::time::Duration::from_secs(RELAY_ONLY_RECONNECT_SECS + 5);
    assert_eq!(
        controller.plan_reconnect(
            vec![
                RelayPeerObservation {
                    peer_id: aged_peer,
                    snapshot: RelayPathSnapshot {
                        kind: SelectedPathKind::Relay,
                        rtt_ms: Some(250),
                    },
                    has_direct_candidate: false,
                },
                RelayPeerObservation {
                    peer_id: degraded_peer,
                    snapshot: RelayPathSnapshot {
                        kind: SelectedPathKind::Relay,
                        rtt_ms: Some(250),
                    },
                    has_direct_candidate: false,
                },
            ],
            initial,
            0,
            true,
        ),
        None
    );

    assert_eq!(
        controller.plan_reconnect(
            vec![
                RelayPeerObservation {
                    peer_id: aged_peer,
                    snapshot: RelayPathSnapshot {
                        kind: SelectedPathKind::Relay,
                        rtt_ms: Some(250),
                    },
                    has_direct_candidate: false,
                },
                RelayPeerObservation {
                    peer_id: degraded_peer,
                    snapshot: RelayPathSnapshot {
                        kind: SelectedPathKind::Relay,
                        rtt_ms: Some(RELAY_DEGRADED_RTT_MS + 25),
                    },
                    has_direct_candidate: false,
                },
            ],
            now,
            0,
            true,
        ),
        Some((degraded_peer, RelayReconnectReason::RelayRttDegraded)),
        "high relay RTT should refresh before merely aged relay paths"
    );
}

#[test]
fn relay_reconnect_controller_tracks_home_relay_missing_and_restored_once() {
    let now = std::time::Instant::now();
    let mut controller = RelayReconnectController::default();

    assert_eq!(controller.observe_home_relay(true, now), None);
    assert_eq!(controller.observe_home_relay(false, now), None);
    assert_eq!(
        controller.observe_home_relay(
            false,
            now + std::time::Duration::from_secs(RELAY_MISSING_GRACE_SECS - 1),
        ),
        None,
        "home relay warning should wait for the grace period"
    );
    assert_eq!(
        controller.observe_home_relay(
            false,
            now + std::time::Duration::from_secs(RELAY_MISSING_GRACE_SECS + 2),
        ),
        Some(HomeRelayStatusTransition::Missing {
            missing_secs: RELAY_MISSING_GRACE_SECS + 2
        })
    );
    assert_eq!(
        controller.observe_home_relay(
            false,
            now + std::time::Duration::from_secs(RELAY_MISSING_GRACE_SECS + 10),
        ),
        None,
        "missing relay should not log on every monitor tick"
    );
    assert_eq!(
        controller.observe_home_relay(
            true,
            now + std::time::Duration::from_secs(RELAY_MISSING_GRACE_SECS + 20),
        ),
        Some(HomeRelayStatusTransition::Restored)
    );
}

#[test]
fn relay_reconnect_controller_applies_cooldown_after_attempt_and_prunes_gone_peers() {
    let now = std::time::Instant::now();
    let peer = make_test_endpoint_id(23);
    let other_peer = make_test_endpoint_id(24);
    let mut controller = RelayReconnectController::default();

    assert_eq!(
        controller.plan_reconnect(
            vec![RelayPeerObservation {
                peer_id: peer,
                snapshot: RelayPathSnapshot {
                    kind: SelectedPathKind::Relay,
                    rtt_ms: Some(RELAY_DEGRADED_RTT_MS + 10),
                },
                has_direct_candidate: false,
            }],
            now,
            0,
            true,
        ),
        Some((peer, RelayReconnectReason::RelayRttDegraded))
    );

    controller.record_reconnect_attempt(peer, RelayReconnectReason::RelayRttDegraded, now);
    assert_eq!(
        controller.plan_reconnect(
            vec![RelayPeerObservation {
                peer_id: peer,
                snapshot: RelayPathSnapshot {
                    kind: SelectedPathKind::Relay,
                    rtt_ms: Some(RELAY_DEGRADED_RTT_MS + 10),
                },
                has_direct_candidate: false,
            }],
            now + std::time::Duration::from_secs(RELAY_RECONNECT_COOLDOWN_SECS - 1),
            0,
            true,
        ),
        None,
        "attempted reconnects should suppress immediate retry even before the next tick"
    );

    controller.plan_reconnect(
        vec![RelayPeerObservation {
            peer_id: other_peer,
            snapshot: RelayPathSnapshot {
                kind: SelectedPathKind::Direct,
                rtt_ms: Some(15),
            },
            has_direct_candidate: false,
        }],
        now + std::time::Duration::from_secs(RELAY_RECONNECT_COOLDOWN_SECS + 1),
        0,
        true,
    );

    assert!(
        controller.peer_health(peer).is_none(),
        "controller should prune peers that are no longer active"
    );
}

#[test]
fn stale_dispatcher_cannot_remove_replacement_connection() {
    assert!(
        should_remove_connection(Some(7), 7),
        "matching stable id should remove tracked connection"
    );
    assert!(
        !should_remove_connection(Some(8), 7),
        "stale dispatcher must not remove a newer replacement connection"
    );
    assert!(
        !should_remove_connection(None, 7),
        "missing connection slot should be a no-op"
    );
}

#[test]
fn relay_only_peers_get_extra_heartbeat_grace() {
    // Relay-only peers get a higher failure threshold so transient
    // relay path-renegotiation (which can spike RTT to 10s+) doesn't
    // prematurely declare them dead and cause MoA reducer fallback.
    // See heartbeat_failure_policy for the rationale.
    let policy = heartbeat_failure_policy(true);

    assert_eq!(
        policy,
        HeartbeatFailurePolicy {
            allow_recent_inbound_grace: true,
            failure_threshold: 5,
        },
        "relay-only peers must have a noticeably higher grace than direct \
         (60s heartbeats × 5 = 5 min)"
    );
}

#[test]
fn is_relay_only_path_set_classifies_correctly() {
    // Empty path set: be lenient (treat as relay-only). The connection
    // is brand-new or mid-failure; we don't want to declare the peer
    // dead prematurely.
    assert!(
        is_relay_only_path_set(std::iter::empty::<bool>()),
        "empty path set must default to relay-only (lenient)"
    );
    // All paths are non-IP (relay): relay-only.
    assert!(is_relay_only_path_set([false]));
    assert!(is_relay_only_path_set([false, false, false]));
    // Any IP path means NOT relay-only.
    assert!(!is_relay_only_path_set([true]));
    assert!(!is_relay_only_path_set([true, false]));
    assert!(!is_relay_only_path_set([false, true]));
    assert!(!is_relay_only_path_set([true, true, true]));
}

#[test]
fn classify_relay_only_defaults_to_strict_when_no_connection() {
    // No Connection object at all (cleanly closed, QUIC idle-expired,
    // never opened): must default to STRICT, not lenient. Otherwise a
    // previously-direct peer that simply disconnected would silently
    // inherit the 5-min relay grace and keep stale model routes alive
    // an extra 3 min beyond what direct policy intends.
    assert!(
        !classify_relay_only_for_policy(None),
        "no Connection object must default to strict (not relay-only)"
    );
    // With a Connection: pass through whatever is_relay_only_connection
    // observed (i.e., classify by the connection's actual paths).
    assert!(
        classify_relay_only_for_policy(Some(true)),
        "a relay-only connection must keep its lenient classification"
    );
    assert!(
        !classify_relay_only_for_policy(Some(false)),
        "a connection with IP paths must remain strict (direct)"
    );
}

#[test]
fn direct_peers_use_strict_heartbeat_threshold() {
    let policy = heartbeat_failure_policy(false);

    assert_eq!(
        policy.failure_threshold, 2,
        "direct paths stay at 2 misses — when the network is up at all, \
         two consecutive cycles of silence is a real failure signal"
    );
}
