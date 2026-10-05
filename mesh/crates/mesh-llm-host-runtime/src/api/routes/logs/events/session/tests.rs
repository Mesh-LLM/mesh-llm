use std::sync::Arc;

use mesh_llm_events::logging::events::LifecycleEvent;
use mesh_llm_events::logging::identifiers::{EventId, RequestId};
use mesh_llm_events::logging::replay::ReplaySequence;

use super::super::protocol::MAX_FRAME_BYTES;
use super::*;
use crate::logging::{
    CallerPathType, LoggingService, ReplayBus, RequestSummaryEntry, RequestSummaryEventSnapshots,
    RequestSummaryMetadata, TerminalOutcome,
};

mod audit;
mod queue;

fn summary(created_at: &str, state: &str, metadata: RequestSummaryMetadata) -> RequestSummaryEntry {
    RequestSummaryEntry {
        request_id: "test-request-summary".into(),
        state: state.into(),
        created_at: created_at.into(),
        terminal_at: None,
        metadata,
    }
}

fn current_snapshots(
    created_at: &str,
    state: &str,
    metadata: RequestSummaryMetadata,
) -> RequestSummaryEventSnapshots {
    let summary = summary(created_at, state, metadata);
    RequestSummaryEventSnapshots::current(&summary)
}

fn terminal_snapshots(
    created_at: &str,
    before_metadata: RequestSummaryMetadata,
    state: &str,
    after_metadata: RequestSummaryMetadata,
) -> RequestSummaryEventSnapshots {
    let before = summary(created_at, "active", before_metadata);
    let mut after = summary(created_at, state, after_metadata);
    after.terminal_at = Some("2026-08-03T00:00:09Z".into());
    RequestSummaryEventSnapshots::terminal(&before, &after)
}

fn entry(bus: &ReplayBus, channel: ReplayChannel, sequence: u64, request: RequestId) {
    let occurred_at = format!("2026-08-03T00:00:0{sequence}Z");
    entry_with_event(
        bus,
        channel,
        sequence,
        request,
        occurred_at,
        LifecycleEvent::Admitted {
            model: None,
            method: None,
        },
    );
}

fn entry_with_metadata(
    bus: &ReplayBus,
    channel: ReplayChannel,
    sequence: u64,
    request: RequestId,
    metadata: RequestSummaryMetadata,
) {
    let occurred_at = format!("2026-08-03T00:00:0{sequence}Z");
    let snapshots = current_snapshots(&occurred_at, "active", metadata);
    entry_with_event_and_snapshots(
        bus,
        channel,
        sequence,
        request,
        occurred_at,
        LifecycleEvent::Admitted {
            model: None,
            method: None,
        },
        snapshots,
    );
}

fn entry_with_event(
    bus: &ReplayBus,
    channel: ReplayChannel,
    sequence: u64,
    request: RequestId,
    occurred_at: String,
    event: LifecycleEvent,
) {
    let summary_snapshots =
        current_snapshots(&occurred_at, "active", RequestSummaryMetadata::default());
    entry_with_event_and_snapshots(
        bus,
        channel,
        sequence,
        request,
        occurred_at,
        event,
        summary_snapshots,
    );
}

fn entry_with_event_and_snapshots(
    bus: &ReplayBus,
    channel: ReplayChannel,
    sequence: u64,
    request: RequestId,
    occurred_at: String,
    event: LifecycleEvent,
    summary_snapshots: RequestSummaryEventSnapshots,
) {
    let envelope = CanonicalEnvelope::new(
        EventId::new(),
        request,
        channel,
        sequence,
        occurred_at,
        event,
    );
    bus.push_replay(
        serde_json::json!({
            "canonical_envelope": envelope,
            "request_summary_snapshots": summary_snapshots,
            "not_public": "/private/operator/path?token=secret"
        })
        .to_string(),
        match channel {
            ReplayChannel::Requests => 0,
            ReplayChannel::Operations => 1,
            ReplayChannel::System => 2,
        },
        ReplaySequence::next(channel, sequence),
    );
}

fn subscription(channels: Vec<ReplayChannel>, cursor: Cursor) -> Subscription {
    Subscription {
        channels,
        filters: Default::default(),
        cursor,
        audit: None,
    }
}

#[test]
fn replay_is_ordered_monotonic_and_hides_raw_payload() {
    let bus = ReplayBus::new(4);
    entry(&bus, ReplayChannel::Requests, 1, RequestId::new());
    entry(&bus, ReplayChannel::Operations, 1, RequestId::new());
    entry(&bus, ReplayChannel::Requests, 2, RequestId::new());

    let frames = replay_frames(
        &bus,
        &subscription(
            vec![ReplayChannel::Requests, ReplayChannel::Operations],
            Cursor::default(),
        ),
        Some("durable-cursor".into()),
    );

    assert_eq!(frames.len(), 3);
    assert!(frames[0].contains("id: v1:1.0.0"));
    assert!(frames[1].contains("id: v1:1.1.0"));
    assert!(frames[2].contains("id: v1:2.1.0"));
    assert!(
        frames
            .iter()
            .all(|frame| !frame.contains("private/operator"))
    );
    assert!(frames.iter().all(|frame| !frame.contains("secret")));
    assert!(
        frames
            .iter()
            .all(|frame| !frame.contains("request_summary_snapshots"))
    );
}

#[test]
fn last_event_id_and_explicit_cursor_deduplicate_replay() {
    let bus = ReplayBus::new(3);
    entry(&bus, ReplayChannel::Requests, 1, RequestId::new());
    entry(&bus, ReplayChannel::Requests, 2, RequestId::new());
    let raw = b"GET /api/logs/events HTTP/1.1\r\nHost: localhost\r\nAccept: text/event-stream\r\nLast-Event-ID: v1:1.0.0\r\n\r\n";
    let subscription = super::super::query::parse_subscription(
        "/api/logs/events?channel=requests&cursor=v1%3A0.0.0",
        raw,
    )
    .expect("reconnect subscription parses");

    let frames = replay_frames(&bus, &subscription, None);
    assert_eq!(frames.len(), 1);
    assert!(frames[0].contains("id: v1:2.0.0"));
}

#[test]
fn evicted_cursor_emits_gap_with_rest_recovery_cursor() {
    let bus = ReplayBus::new(1);
    entry(&bus, ReplayChannel::Requests, 1, RequestId::new());
    entry(&bus, ReplayChannel::Requests, 2, RequestId::new());

    let frames = replay_frames(
        &bus,
        &subscription(vec![ReplayChannel::Requests], Cursor::default()),
        Some("opaque-rest-cursor".into()),
    );
    assert!(frames[0].contains("event: replay_gap"));
    assert!(frames[0].contains("opaque-rest-cursor"));
    assert!(frames[1].contains("id: v1:2.0.0"));
}

#[test]
fn replay_session_acknowledges_gap_without_retained_selected_records() {
    let bus = ReplayBus::new(1);
    entry(&bus, ReplayChannel::Requests, 1, RequestId::new());
    entry(&bus, ReplayChannel::Operations, 1, RequestId::new());
    let mut session = ReplaySession::new(subscription(
        vec![ReplayChannel::Requests],
        Cursor::default(),
    ));

    let first = session.next_frames(&bus, Some("recovery".into()));
    assert_eq!(first.len(), 1);
    assert!(first[0].contains("event: replay_gap"));

    let second = session.next_frames(&bus, Some("recovery".into()));
    assert!(second.is_empty(), "an acknowledged gap must not repeat");
}

#[test]
fn live_update_inspects_only_the_increment_after_a_full_high_capacity_replay() {
    const CAPACITY: usize = 10_000;
    let bus = ReplayBus::new(CAPACITY);
    for sequence in 1..=CAPACITY as u64 {
        entry(&bus, ReplayChannel::Requests, sequence, RequestId::new());
    }
    let mut session = ReplaySession::new(subscription(
        vec![ReplayChannel::Requests],
        Cursor::default(),
    ));
    assert_eq!(session.next_frames(&bus, None).len(), CAPACITY);

    let mut updates = bus.subscribe_updates();
    LIFECYCLE_RECORDS_INSPECTED.with(|count| count.set(0));
    entry(
        &bus,
        ReplayChannel::Requests,
        CAPACITY as u64 + 1,
        RequestId::new(),
    );
    let update = updates.try_recv().expect("live replay update");
    let frames = session.next_update_frames(&bus, &update, None);

    assert_eq!(frames.len(), 1);
    assert!(frames[0].contains("id: v1:10001.0.0"));
    LIFECYCLE_RECORDS_INSPECTED
        .with(|count| assert_eq!(count.get(), 1, "retained replay prefix must not be scanned"));
}

#[test]
fn queued_live_update_already_seen_in_initial_replay_is_deduplicated() {
    let bus = ReplayBus::new(2);
    let mut updates = bus.subscribe_updates();
    entry(&bus, ReplayChannel::Requests, 1, RequestId::new());
    let mut session = ReplaySession::new(subscription(
        vec![ReplayChannel::Requests],
        Cursor::default(),
    ));
    assert_eq!(session.next_frames(&bus, None).len(), 1);

    let update = updates.try_recv().expect("queued replay update");
    LIFECYCLE_RECORDS_INSPECTED.with(|count| count.set(0));
    assert!(
        session.next_update_frames(&bus, &update, None).is_empty(),
        "the update included in the initial snapshot must not be re-emitted"
    );
    LIFECYCLE_RECORDS_INSPECTED.with(|count| assert_eq!(count.get(), 0));
}

#[test]
fn filtered_live_update_advances_the_session_cursor() {
    let bus = ReplayBus::new(3);
    let wanted = RequestId::new();
    let mut selected = subscription(vec![ReplayChannel::Requests], Cursor::default());
    selected
        .filters
        .request_ids
        .insert(wanted.as_uuid().to_string());
    let mut session = ReplaySession::new(selected);
    let mut updates = bus.subscribe_updates();

    entry(&bus, ReplayChannel::Requests, 1, RequestId::new());
    let filtered = updates.try_recv().expect("filtered update");
    assert!(session.next_update_frames(&bus, &filtered, None).is_empty());
    LIFECYCLE_RECORDS_INSPECTED.with(|count| count.set(0));
    assert!(session.next_update_frames(&bus, &filtered, None).is_empty());
    LIFECYCLE_RECORDS_INSPECTED.with(|count| assert_eq!(count.get(), 0));

    entry(&bus, ReplayChannel::Requests, 2, wanted);
    let selected = updates.try_recv().expect("selected update");
    let frames = session.next_update_frames(&bus, &selected, None);
    assert_eq!(frames.len(), 1);
    assert!(frames[0].contains("id: v1:2.0.0"));
}

#[test]
fn lagged_live_receiver_recovers_from_the_bounded_snapshot() {
    let bus = ReplayBus::new(2);
    let mut updates = bus.subscribe_updates();
    for sequence in 1..=3 {
        entry(&bus, ReplayChannel::Requests, sequence, RequestId::new());
    }
    assert!(matches!(
        updates.try_recv(),
        Err(tokio::sync::broadcast::error::TryRecvError::Lagged(1))
    ));

    let mut session = ReplaySession::new(subscription(
        vec![ReplayChannel::Requests],
        Cursor::default(),
    ));
    let frames = session.next_frames(&bus, Some("durable-recovery".into()));
    assert_eq!(frames.len(), 3);
    assert!(frames[0].contains("event: replay_gap"));
    assert!(frames[0].contains("durable-recovery"));
    assert!(frames[1].contains("id: v1:2.0.0"));
    assert!(frames[2].contains("id: v1:3.0.0"));
}

#[test]
fn request_filter_selects_only_the_requested_lifecycle() {
    let bus = ReplayBus::new(3);
    let wanted = RequestId::new();
    entry_with_metadata(
        &bus,
        ReplayChannel::Requests,
        1,
        wanted,
        RequestSummaryMetadata::from_parts(
            Some("chat_completions"),
            Some("Qwen/Qwen3"),
            Some("mesh"),
            Some("skippy"),
        ),
    );
    entry(&bus, ReplayChannel::Requests, 2, RequestId::new());
    let mut subscription = subscription(vec![ReplayChannel::Requests], Cursor::default());
    subscription
        .filters
        .request_ids
        .insert(wanted.as_uuid().to_string());

    let frames = replay_frames(&bus, &subscription, None);
    assert_eq!(frames.len(), 1);
    assert!(frames[0].contains(&wanted.as_uuid().to_string()));
    assert!(frames[0].contains("\"request\":"));
    assert!(frames[0].contains("\"route\":\"chat_completions\""));
    assert!(frames[0].contains("\"model\":\"Qwen/Qwen3\""));
    assert!(frames[0].contains("\"source\":\"active\""));
}

#[test]
fn metadata_filters_match_summary_snapshots_not_event_fields() {
    let bus = ReplayBus::new(4);
    let wanted = RequestId::new();
    entry_with_event_and_snapshots(
        &bus,
        ReplayChannel::Requests,
        1,
        wanted,
        "2026-08-03T00:00:01Z".into(),
        LifecycleEvent::RouteSelected {
            model: Some("event-model-must-not-match".into()),
            provider: Some("event-provider-must-not-match".into()),
            engine: Some("event-engine-must-not-match".into()),
        },
        current_snapshots(
            "2026-08-03T00:00:01Z",
            "active",
            RequestSummaryMetadata::from_parts(
                Some("chat_completions"),
                Some("Qwen/Qwen3"),
                Some("mesh"),
                Some("skippy"),
            ),
        ),
    );
    entry_with_event_and_snapshots(
        &bus,
        ReplayChannel::Requests,
        2,
        RequestId::new(),
        "2026-08-03T00:00:02Z".into(),
        LifecycleEvent::RouteSelected {
            model: Some("Qwen/Qwen2.5".into()),
            provider: Some("mesh".into()),
            engine: Some("skippy".into()),
        },
        current_snapshots(
            "2026-08-03T00:00:02Z",
            "active",
            RequestSummaryMetadata::from_parts(
                Some("completions"),
                Some("Qwen/Qwen2.5"),
                Some("mesh"),
                Some("skippy"),
            ),
        ),
    );
    let subscription = super::super::query::parse_subscription(
        "/api/logs/events?channel=requests&route=chat_completions&filter=model%3AQwen%2FQwen3&provider=mesh&engine=skippy&from=2026-08-03T00%3A00%3A00Z&to=2026-08-03T00%3A00%3A01Z",
        b"GET /api/logs/events HTTP/1.1\r\nHost: localhost\r\nAccept: text/event-stream\r\n\r\n",
    )
    .expect("ledger subscription parses");

    let frames = replay_frames(&bus, &subscription, None);
    assert_eq!(frames.len(), 1);
    assert!(frames[0].contains(&wanted.as_uuid().to_string()));
}

#[test]
fn terminal_after_to_matches_created_within_range() {
    let bus = ReplayBus::new(3);
    let wanted = RequestId::new();
    entry_with_event_and_snapshots(
        &bus,
        ReplayChannel::Requests,
        1,
        wanted,
        "2026-08-03T00:00:05Z".into(),
        LifecycleEvent::Completed {
            status_code: Some(200),
            duration_ms: Some(4),
            usage: None,
        },
        terminal_snapshots(
            "2026-08-03T00:00:01Z",
            RequestSummaryMetadata::default(),
            "completed",
            RequestSummaryMetadata::default(),
        ),
    );
    let subscription = super::super::query::parse_subscription(
        "/api/logs/events?channel=requests&from=2026-08-03T00%3A00%3A00Z&to=2026-08-03T00%3A00%3A02Z&outcome=completed",
        b"GET /api/logs/events HTTP/1.1\r\nHost: localhost\r\nAccept: text/event-stream\r\n\r\n",
    )
    .expect("outcome subscription parses");

    let frames = replay_frames(&bus, &subscription, None);
    assert_eq!(frames.len(), 1);
    assert!(frames[0].contains("id: v1:1.0.0"));
}

#[test]
fn event_inside_range_does_not_match_request_created_outside_range() {
    let bus = ReplayBus::new(3);
    entry_with_event_and_snapshots(
        &bus,
        ReplayChannel::Requests,
        1,
        RequestId::new(),
        "2026-08-03T00:00:01Z".into(),
        LifecycleEvent::RouteSelected {
            model: None,
            provider: None,
            engine: None,
        },
        current_snapshots(
            "2026-08-03T00:00:05Z",
            "active",
            RequestSummaryMetadata::default(),
        ),
    );
    let subscription = super::super::query::parse_subscription(
        "/api/logs/events?channel=requests&from=2026-08-03T00%3A00%3A00Z&to=2026-08-03T00%3A00%3A02Z",
        b"GET /api/logs/events HTTP/1.1\r\nHost: localhost\r\nAccept: text/event-stream\r\n\r\n",
    )
    .expect("time subscription parses");

    assert!(replay_frames(&bus, &subscription, None).is_empty());
}

#[test]
fn terminal_completion_notifies_active_and_completed_memberships() {
    let bus = ReplayBus::new(3);
    let wanted = RequestId::new();
    entry_with_event_and_snapshots(
        &bus,
        ReplayChannel::Requests,
        1,
        wanted,
        "2026-08-03T00:00:02Z".into(),
        LifecycleEvent::Completed {
            status_code: Some(200),
            duration_ms: Some(4),
            usage: None,
        },
        terminal_snapshots(
            "2026-08-03T00:00:01Z",
            RequestSummaryMetadata::default(),
            "completed",
            RequestSummaryMetadata::default(),
        ),
    );

    for outcome in ["active", "completed"] {
        let subscription = super::super::query::parse_subscription(
            &format!("/api/logs/events?channel=requests&outcome={outcome}"),
            b"GET /api/logs/events HTTP/1.1\r\nHost: localhost\r\nAccept: text/event-stream\r\n\r\n",
        )
        .expect("outcome subscription parses");
        let frames = replay_frames(&bus, &subscription, None);
        assert_eq!(frames.len(), 1, "{outcome} sees the terminal transition");
        assert!(frames[0].contains(&wanted.as_uuid().to_string()));
    }
}

#[test]
fn failed_terminal_does_not_match_completed_membership() {
    let bus = ReplayBus::new(3);
    entry_with_event_and_snapshots(
        &bus,
        ReplayChannel::Requests,
        1,
        RequestId::new(),
        "2026-08-03T00:00:02Z".into(),
        LifecycleEvent::Failed {
            error: "bounded_failure".into(),
            status_code: None,
        },
        terminal_snapshots(
            "2026-08-03T00:00:01Z",
            RequestSummaryMetadata::default(),
            "failed",
            RequestSummaryMetadata::default(),
        ),
    );
    let completed = super::super::query::parse_subscription(
        "/api/logs/events?channel=requests&outcome=completed",
        b"GET /api/logs/events HTTP/1.1\r\nHost: localhost\r\nAccept: text/event-stream\r\n\r\n",
    )
    .expect("completed subscription parses");
    let active = super::super::query::parse_subscription(
        "/api/logs/events?channel=requests&outcome=active",
        b"GET /api/logs/events HTTP/1.1\r\nHost: localhost\r\nAccept: text/event-stream\r\n\r\n",
    )
    .expect("active subscription parses");

    assert!(replay_frames(&bus, &completed, None).is_empty());
    assert_eq!(replay_frames(&bus, &active, None).len(), 1);
}

#[test]
fn terminal_replay_uses_enriched_summary_metadata_for_filters() {
    let service = Arc::new(LoggingService::new_disabled(Default::default()));
    let known = RequestId::new();
    let (known_guard, _) = service.register_request_with_metadata(
        known,
        RequestSummaryMetadata::from_parts(Some("chat_completions"), None, None, None),
    );
    service.merge_request_metadata(
        known,
        RequestSummaryMetadata::from_parts(
            None,
            Some("acme/model"),
            Some("mesh"),
            Some("raw_ingress"),
        ),
    );
    service
        .transition_terminal(known, &known_guard, TerminalOutcome::Completed)
        .expect("known request terminalizes");

    let absent = RequestId::new();
    let (absent_guard, _) = service.register_request(absent);
    service
        .transition_terminal(absent, &absent_guard, TerminalOutcome::Completed)
        .expect("metadata-absent request terminalizes");

    let subscription = super::super::query::parse_subscription(
        "/api/logs/events?channel=requests&route=chat_completions&model=acme%2Fmodel&provider=mesh&engine=raw_ingress&outcome=completed",
        b"GET /api/logs/events HTTP/1.1\r\nHost: localhost\r\nAccept: text/event-stream\r\n\r\n",
    )
    .expect("metadata subscription parses");
    let bus = service.bus_ref();
    let frames = replay_frames(bus.as_ref(), &subscription, None);

    assert_eq!(frames.len(), 1);
    assert!(frames[0].contains(&known.as_uuid().to_string()));
    assert!(!frames[0].contains(&absent.as_uuid().to_string()));

    let active_subscription = super::super::query::parse_subscription(
        "/api/logs/events?channel=requests&route=chat_completions&model=acme%2Fmodel&provider=mesh&engine=raw_ingress&outcome=active",
        b"GET /api/logs/events HTTP/1.1\r\nHost: localhost\r\nAccept: text/event-stream\r\n\r\n",
    )
    .expect("active metadata subscription parses");
    let active_frames = replay_frames(bus.as_ref(), &active_subscription, None);

    assert_eq!(active_frames.len(), 1);
    assert!(active_frames[0].contains(&known.as_uuid().to_string()));
    assert!(!active_frames[0].contains(&absent.as_uuid().to_string()));
}

#[test]
fn lifecycle_replay_projects_canonical_caller_attribution() {
    let bus = ReplayBus::new(4);
    let cases = [
        (
            RequestSummaryMetadata::from_parts(None, None, None, None).with_caller_identity(
                Some("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"),
                Some("192.0.2.71:11204"),
                Some(CallerPathType::RemoteQuicHttp),
            ),
            Some("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"),
            Some("192.0.2.71:11204"),
            Some("remote_quic_http"),
        ),
        (
            RequestSummaryMetadata::from_parts(None, None, None, None).with_caller_identity(
                Some("abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"),
                Some("192.0.2.72:11204"),
                Some(CallerPathType::Relay),
            ),
            Some("abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"),
            None,
            Some("relay"),
        ),
        (
            RequestSummaryMetadata::from_parts(None, None, None, None).with_caller_identity(
                Some("fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210"),
                None,
                None,
            ),
            Some("fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210"),
            None,
            None,
        ),
        (RequestSummaryMetadata::default(), None, None, None),
    ];

    for (sequence, (metadata, endpoint_id, caller_addr, path_type)) in cases.into_iter().enumerate()
    {
        entry_with_metadata(
            &bus,
            ReplayChannel::Requests,
            sequence as u64 + 1,
            RequestId::new(),
            metadata,
        );
        let frame = replay_frames(
            &bus,
            &subscription(vec![ReplayChannel::Requests], Cursor::default()),
            None,
        )
        .pop()
        .expect("caller attribution replay frame");
        let data = frame
            .lines()
            .find_map(|line| line.strip_prefix("data: "))
            .and_then(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .expect("lifecycle replay data");
        let request = &data["request"];

        assert_eq!(
            request
                .get("callerEndpointId")
                .and_then(|value| value.as_str()),
            endpoint_id
        );
        assert!(endpoint_id.is_some() || request.get("callerEndpointId").is_none());
        assert_eq!(
            request.get("callerAddr").and_then(|value| value.as_str()),
            caller_addr
        );
        assert!(caller_addr.is_some() || request.get("callerAddr").is_none());
        assert_eq!(
            request
                .get("callerPathType")
                .and_then(|value| value.as_str()),
            path_type
        );
        assert!(path_type.is_some() || request.get("callerPathType").is_none());
    }
}

#[test]
fn lifecycle_replay_sanitizes_deserialized_metadata_at_the_sse_boundary() {
    let bus = ReplayBus::new(4);
    let request_id = RequestId::new();
    let sequence = 1;
    let occurred_at = "2026-08-03T00:00:01Z";
    let envelope = CanonicalEnvelope::new(
        EventId::new(),
        request_id,
        ReplayChannel::Requests,
        sequence,
        occurred_at.into(),
        LifecycleEvent::Admitted {
            model: None,
            method: None,
        },
    );
    let unsafe_route = "/private/route?token=route-secret";
    let unsafe_model = "https://model:secret@example.test/private";
    let unsafe_provider = "Bearer provider-secret";
    let unsafe_engine = r"C:\Users\operator\private-engine";
    let unsafe_endpoint = "https://operator:credential@example.test/private?token=secret";
    let unsafe_addr = "/Users/operator/.ssh/id_ed25519";
    let unsafe_path_type = "/private/caller-path";
    bus.push_replay(
        serde_json::json!({
            "canonical_envelope": envelope,
            "request_summary_snapshots": {
                "after": {
                    "created_at": occurred_at,
                    "state": "active",
                    "terminal_at": null,
                    "metadata": {
                        "route": unsafe_route,
                        "model": unsafe_model,
                        "provider": unsafe_provider,
                        "engine": unsafe_engine,
                        "caller_endpoint_id": unsafe_endpoint,
                        "caller_addr": unsafe_addr,
                        "caller_path_type": "remote_quic_http"
                    }
                }
            }
        })
        .to_string(),
        0,
        ReplaySequence::next(ReplayChannel::Requests, sequence),
    );
    let invalid_path_envelope = CanonicalEnvelope::new(
        EventId::new(),
        RequestId::new(),
        ReplayChannel::Requests,
        2,
        "2026-08-03T00:00:02Z".into(),
        LifecycleEvent::Admitted {
            model: None,
            method: None,
        },
    );
    bus.push_replay(
        serde_json::json!({
            "canonical_envelope": invalid_path_envelope,
            "request_summary_snapshots": {
                "after": {
                    "created_at": "2026-08-03T00:00:02Z",
                    "state": "active",
                    "terminal_at": null,
                    "metadata": {
                        "caller_path_type": unsafe_path_type
                    }
                }
            }
        })
        .to_string(),
        0,
        ReplaySequence::next(ReplayChannel::Requests, 2),
    );

    let frames = replay_frames(
        &bus,
        &subscription(vec![ReplayChannel::Requests], Cursor::default()),
        None,
    );
    let frame = frames
        .iter()
        .find(|frame| frame.contains("remote_quic_http"))
        .expect("sanitized metadata replay frame");
    let data = frame
        .lines()
        .find_map(|line| line.strip_prefix("data: "))
        .and_then(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .expect("lifecycle replay data");
    let request = &data["request"];

    assert_eq!(request["route"], "[REDACTED]");
    assert_eq!(request["model"], "[REDACTED]");
    assert_eq!(request["provider"], "[REDACTED]");
    assert_eq!(request["engine"], "[REDACTED]");
    assert_eq!(request["callerEndpointId"], "[REDACTED]");
    assert_eq!(request["callerAddr"], "[REDACTED]");
    assert_eq!(request["callerPathType"], "remote_quic_http");
    let emitted = frames.join("\n");
    for unsafe_value in [
        unsafe_route,
        unsafe_model,
        unsafe_provider,
        unsafe_engine,
        unsafe_endpoint,
        unsafe_addr,
        unsafe_path_type,
    ] {
        assert!(!emitted.contains(unsafe_value));
    }
}

#[test]
fn heartbeat_is_an_sse_comment() {
    assert_eq!(super::super::heartbeat_frame(), ": keepalive\n\n");
}
