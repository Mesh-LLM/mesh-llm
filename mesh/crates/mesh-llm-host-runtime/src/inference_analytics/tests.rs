use super::*;
use mesh_llm_analytics::Value;

#[test]
fn route_outcomes_separate_disconnects_and_broken_streams() {
    assert_eq!(
        RouteOutcome::from(AttemptOutcome::Unavailable).as_str(),
        "unavailable"
    );
    assert_eq!(RouteOutcome::StreamFailed.as_str(), "stream_failed");
    assert_eq!(
        RouteOutcome::ClientDisconnected.as_str(),
        "client_disconnected"
    );
}

#[test]
fn a_peer_gone_by_the_end_of_the_attempt_is_departed_not_unknown() {
    assert_eq!(route_link(None), ("departed", None));
    // Still in the mesh with no selected path: genuinely unknown.
    assert_eq!(route_link(Some((None, Some(9)))), ("unknown", None));
    assert_eq!(
        route_link(Some((Some(path("direct", "192.168.1.20:4000")), Some(2)))),
        ("lan", Some(2))
    );
}

#[test]
fn limiter_weights_the_next_event_by_what_it_dropped() {
    let limiter = RateLimiter::new(2);
    let start = Instant::now();

    assert_eq!(limiter.admit(start), Some(1));
    assert_eq!(limiter.admit(start), Some(1));
    assert_eq!(limiter.admit(start), None);
    assert_eq!(limiter.admit(start), None);
    // The next window's first event stands for itself and the two dropped.
    assert_eq!(limiter.admit(start + RateLimiter::WINDOW), Some(3));
}

fn path(path_type: &'static str, addr: &str) -> SelectedPathObservation {
    SelectedPathObservation {
        path_type,
        rtt_ms: Some(4),
        observed_direct_remote_addr: Some(addr.parse().unwrap()),
    }
}

#[test]
fn links_classify_by_path_and_address() {
    assert_eq!(
        LinkClass::of(Some(&path("direct", "192.168.1.20:4000"))),
        LinkClass::Lan
    );
    assert_eq!(
        LinkClass::of(Some(&path("direct", "[fd00::1]:4000"))),
        LinkClass::Lan
    );
    assert_eq!(
        LinkClass::of(Some(&path("direct", "203.0.113.9:4000"))),
        LinkClass::Wan
    );
    assert_eq!(
        LinkClass::of(Some(&path("relay", "203.0.113.9:4000"))),
        LinkClass::Relay
    );
    assert_eq!(LinkClass::of(None), LinkClass::Unknown);
}

#[test]
fn a_relay_link_reports_its_own_round_trip_not_a_stale_direct_one() {
    let mut relay = path("relay", "203.0.113.9:4000");
    relay.rtt_ms = Some(140);
    assert_eq!(
        link_and_rtt(Some(relay), Some(9)),
        (LinkClass::Relay, Some(140))
    );
    relay.rtt_ms = None;
    assert_eq!(link_and_rtt(Some(relay), Some(9)), (LinkClass::Relay, None));
    // No selected path: a gossip RTT belongs to no link.
    assert_eq!(link_and_rtt(None, Some(9)), (LinkClass::Unknown, None));
    // A direct link prefers the best direct measurement.
    assert_eq!(
        link_and_rtt(Some(path("direct", "192.168.1.20:4000")), Some(2)),
        (LinkClass::Lan, Some(2))
    );
}

#[test]
fn mesh_links_summarize_each_class() {
    let properties = mesh_links_properties(&[
        (LinkClass::Lan, Some(1)),
        (LinkClass::Lan, Some(3)),
        (LinkClass::Lan, Some(2)),
        (LinkClass::Wan, Some(80)),
        (LinkClass::Relay, None),
    ]);

    assert_eq!(properties.get("peers"), Some(&Value::Int(5)));
    assert_eq!(properties.get("lan_peers"), Some(&Value::Int(3)));
    assert_eq!(properties.get("lan_rtt_ms_median"), Some(&Value::Int(2)));
    assert_eq!(properties.get("lan_rtt_ms_max"), Some(&Value::Int(3)));
    assert_eq!(properties.get("wan_rtt_ms_median"), Some(&Value::Int(80)));
    assert_eq!(properties.get("relay_peers"), Some(&Value::Int(1)));
    assert!(properties.get("relay_rtt_ms_median").is_none());
    assert_eq!(properties.get("unknown_peers"), Some(&Value::Int(0)));
}
