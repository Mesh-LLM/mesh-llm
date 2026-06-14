use super::super::direct_path::{
    DIRECT_PATH_REPAIR_COOLDOWN_SECS, DIRECT_PATH_REPAIR_GRACE_SECS,
    DirectPathMaintenanceController, DirectPathObservation, DirectPathRepairReason,
    endpoint_addr_with_previously_advertised_direct_candidates,
};
use super::super::heartbeat::{RelayPathSnapshot, SelectedPathKind};
use super::make_test_endpoint_id;
use iroh::{EndpointAddr, TransportAddr};

#[test]
fn direct_path_maintenance_requires_candidate_and_grace_period() {
    let now = std::time::Instant::now();
    let peer = make_test_endpoint_id(31);
    let mut controller = DirectPathMaintenanceController::default();
    let relay_observation = DirectPathObservation {
        peer_id: peer,
        snapshot: RelayPathSnapshot {
            kind: SelectedPathKind::Relay,
            rtt_ms: Some(200),
        },
        has_direct_candidate: true,
    };

    assert_eq!(
        controller.plan_request([relay_observation], now, 0),
        None,
        "first non-direct observation starts the grace timer"
    );
    assert_eq!(
        controller.plan_request(
            [relay_observation],
            now + std::time::Duration::from_secs(DIRECT_PATH_REPAIR_GRACE_SECS + 2),
            0,
        ),
        Some((peer, DirectPathRepairReason::RelaySelected))
    );

    let mut no_candidate_controller = DirectPathMaintenanceController::default();
    assert_eq!(
        no_candidate_controller.plan_request(
            [DirectPathObservation {
                has_direct_candidate: false,
                ..relay_observation
            }],
            now + std::time::Duration::from_secs(DIRECT_PATH_REPAIR_GRACE_SECS + 1),
            0,
        ),
        None,
        "without a direct candidate there is nothing useful to request"
    );
}

#[test]
fn direct_path_maintenance_cooldown_and_inflight_suppress_requests() {
    let now = std::time::Instant::now();
    let peer = make_test_endpoint_id(32);
    let mut controller = DirectPathMaintenanceController::default();
    let observation = DirectPathObservation {
        peer_id: peer,
        snapshot: RelayPathSnapshot {
            kind: SelectedPathKind::Unknown,
            rtt_ms: None,
        },
        has_direct_candidate: true,
    };

    assert_eq!(controller.plan_request([observation], now, 1), None);
    assert!(
        controller
            .peer_health(peer)
            .and_then(|health| health.non_direct_since)
            .is_some(),
        "active requests suppress repair but still record path state"
    );

    let ready_at = now + std::time::Duration::from_secs(DIRECT_PATH_REPAIR_GRACE_SECS + 1);
    assert_eq!(
        controller.plan_request([observation], ready_at, 0),
        Some((peer, DirectPathRepairReason::UnknownSelected))
    );
    controller.record_request_attempt(peer, ready_at);
    assert_eq!(
        controller.plan_request(
            [observation],
            ready_at + std::time::Duration::from_secs(DIRECT_PATH_REPAIR_COOLDOWN_SECS - 1),
            0,
        ),
        None,
        "cooldown prevents repeated reverse-dial requests"
    );
}

#[test]
fn direct_path_request_keeps_only_previously_advertised_direct_candidates() {
    let peer_id = make_test_endpoint_id(33);
    let advertised_direct = TransportAddr::Ip("10.0.0.7:47916".parse().unwrap());
    let unadvertised_direct = TransportAddr::Ip("10.0.0.99:47916".parse().unwrap());
    let advertised_relay = TransportAddr::Relay("https://relay.example.com".parse().unwrap());

    let mut advertised = EndpointAddr {
        id: peer_id,
        addrs: Default::default(),
    };
    advertised.addrs.insert(advertised_direct.clone());
    advertised.addrs.insert(advertised_relay.clone());

    let mut requested = EndpointAddr {
        id: peer_id,
        addrs: Default::default(),
    };
    requested.addrs.insert(advertised_direct.clone());
    requested.addrs.insert(unadvertised_direct.clone());
    requested.addrs.insert(advertised_relay.clone());

    let filtered =
        endpoint_addr_with_previously_advertised_direct_candidates(requested, &advertised)
            .expect("the previously advertised direct candidate should be kept");
    assert!(filtered.addrs.contains(&advertised_direct));
    assert!(!filtered.addrs.contains(&unadvertised_direct));
    assert!(!filtered.addrs.contains(&advertised_relay));

    let mut unknown_only = EndpointAddr {
        id: peer_id,
        addrs: Default::default(),
    };
    unknown_only.addrs.insert(unadvertised_direct);
    assert!(
        endpoint_addr_with_previously_advertised_direct_candidates(unknown_only, &advertised)
            .is_none(),
        "requests with only unknown direct candidates must not trigger reverse dials"
    );
}
