use super::*;
use mesh_llm_log_store::{AuditEntryDetail, AuditEntryFilters, LogStore, RealClock};

fn durable_path_details() -> Vec<AuditEntryDetail> {
    let root = tempfile::tempdir().expect("temporary durable audit store");
    let store = LogStore::open(root.path(), Arc::new(RealClock)).expect("open audit store");
    for (entry_id, subject_id, path_type, remote_addr) in [
        (
            "durable-direct",
            "peer-direct",
            "direct",
            "[2001:0db8:0:0:0:0:0:1]:443",
        ),
        (
            "durable-relay",
            "peer-relay",
            "relay",
            "private-relay.example:443",
        ),
    ] {
        let detail = serde_json::json!({
            "context_version": 1,
            "subject_kind": "mesh_peer",
            "subject_id": subject_id,
            "path_type": path_type,
            "remote_addr": remote_addr,
        });
        store
            .insert_audit_entry(
                entry_id,
                None,
                "2026-01-01T00:00:01Z",
                "mesh",
                "peer_path_observed",
                Some(&detail.to_string()),
            )
            .expect("insert durable audit entry");
    }
    store
        .list_audit_entry_details_after_sequence(0, 2, AuditEntryFilters::default())
        .expect("query durable audit entries")
}

fn audit_subscription(cursor: AuditCursor) -> Subscription {
    Subscription {
        channels: Vec::new(),
        filters: Default::default(),
        cursor: Cursor::default(),
        audit: Some(AuditSelection {
            cursor,
            source: None,
            severity: None,
        }),
    }
}

fn push_audit(bus: &ReplayBus, payload: &str) {
    bus.push_audit_replay(payload.to_string(), 2);
}

fn audit_payload(
    entry_id: &str,
    occurred_at: &str,
    source: &str,
    code: &str,
    severity: Option<&str>,
) -> String {
    let mut obj = serde_json::json!({
        "kind": "audit",
        "entry_id": entry_id,
        "occurred_at": occurred_at,
        "source": source,
        "code": code,
    });
    if let Some(s) = severity {
        obj.as_object_mut()
            .unwrap()
            .insert("severity".into(), serde_json::json!(s));
    }
    obj.to_string()
}

#[test]
fn audit_replay_emits_ordered_frames_with_a1_ids() {
    let bus = ReplayBus::new(4);
    push_audit(
        &bus,
        &audit_payload(
            "id-1",
            "2026-01-01T00:00:01Z",
            "runtime",
            "startup",
            Some("info"),
        ),
    );
    push_audit(
        &bus,
        &audit_payload("id-2", "2026-01-01T00:00:02Z", "mesh", "peer_joined", None),
    );
    push_audit(
        &bus,
        &audit_payload(
            "id-3",
            "2026-01-01T00:00:03Z",
            "cli",
            "command",
            Some("warning"),
        ),
    );

    let frames = replay_frames(&bus, &audit_subscription(AuditCursor(0)), None);
    assert_eq!(frames.len(), 3);
    assert!(frames[0].contains("event: audit_entry"));
    assert!(frames[0].contains("id: a1:1"));
    assert!(frames[1].contains("id: a1:2"));
    assert!(frames[2].contains("id: a1:3"));
    assert!(frames[0].contains("\"entryId\":\"id-1\""));
    assert!(frames[1].contains("\"entryId\":\"id-2\""));
    assert!(frames[2].contains("\"entryId\":\"id-3\""));
}

#[test]
fn audit_replay_session_projects_direct_and_relay_paths_without_raw_fields() {
    let bus = ReplayBus::new(4);
    for (entry_id, path_type, remote_addr) in [
        ("direct-entry", "direct", "192.0.2.44:11204"),
        ("relay-entry", "relay", "private-relay.example:443"),
    ] {
        push_audit(
            &bus,
            &serde_json::json!({
                "kind": "audit",
                "entry_id": entry_id,
                "occurred_at": "2026-01-01T00:00:01Z",
                "source": "mesh",
                "code": "peer_path_observed",
                "context_version": 1,
                "path_type": path_type,
                "remote_addr": remote_addr,
                "canonical_envelope": "must-not-leak",
                "detail_json": "must-not-leak"
            })
            .to_string(),
        );
    }
    let mut session = ReplaySession::new(audit_subscription(AuditCursor(0)));

    let frames = session.next_frames(&bus, None);

    assert_eq!(frames.len(), 2);
    assert!(frames[0].contains("id: a1:1"));
    assert!(frames[0].contains("\"pathType\":\"direct\""));
    assert!(frames[0].contains("\"remoteAddr\":\"192.0.2.44:11204\""));
    assert!(frames[1].contains("id: a1:2"));
    assert!(frames[1].contains("\"pathType\":\"relay\""));
    assert!(!frames[1].contains("remoteAddr"));
    assert!(!frames[1].contains("private-relay.example"));
    assert!(
        frames
            .iter()
            .all(|frame| !frame.contains("canonical_envelope"))
    );
    assert!(frames.iter().all(|frame| !frame.contains("detail_json")));
    assert!(frames.iter().all(|frame| !frame.contains("must-not-leak")));
    assert!(frames.iter().all(|frame| frame.len() <= MAX_FRAME_BYTES));
}

#[test]
fn durable_audit_reconciliation_projects_paths_and_advances_cursor() {
    let records = durable_path_details();
    let direct = records[0].clone();
    let relay = records[1].clone();
    let mut session = ReplaySession::new(audit_subscription(AuditCursor(0)));

    let frames = session.durable_audit_frames(vec![direct.clone(), relay.clone()]);

    assert_eq!(frames.len(), 2);
    assert!(frames[0].contains("id: a1:1"));
    assert!(frames[0].contains("\"pathType\":\"direct\""));
    assert!(frames[0].contains("\"remoteAddr\":\"[2001:db8::1]:443\""));
    assert!(frames[1].contains("id: a1:2"));
    assert!(frames[1].contains("\"pathType\":\"relay\""));
    assert!(!frames[1].contains("remoteAddr"));
    assert!(!frames[1].contains("private-relay.example"));
    assert!(
        frames
            .iter()
            .all(|frame| !frame.contains("canonical_envelope"))
    );
    assert!(frames.iter().all(|frame| !frame.contains("detail_json")));
    assert!(session.durable_audit_frames(vec![direct, relay]).is_empty());
    assert_eq!(
        session.durable_audit_query().map(|(cursor, _)| cursor),
        Some(2)
    );
}

#[test]
fn audit_live_updates_are_incremental_and_advance_filtered_records() {
    let bus = ReplayBus::new(4);
    let mut selected = audit_subscription(AuditCursor(0));
    selected.audit.as_mut().unwrap().source = Some("runtime".into());
    let mut session = ReplaySession::new(selected);
    let mut updates = bus.subscribe_updates();

    push_audit(
        &bus,
        &audit_payload("id-1", "2026-01-01T00:00:01Z", "mesh", "ignored", None),
    );
    let filtered = updates.try_recv().expect("filtered audit update");
    assert!(session.next_update_frames(&bus, &filtered, None).is_empty());
    assert!(session.next_update_frames(&bus, &filtered, None).is_empty());

    push_audit(
        &bus,
        &audit_payload("id-2", "2026-01-01T00:00:02Z", "runtime", "selected", None),
    );
    let update = updates.try_recv().expect("selected audit update");
    let frames = session.next_update_frames(&bus, &update, None);
    assert_eq!(frames.len(), 1);
    assert!(frames[0].contains("id: a1:2"));
    assert!(frames[0].contains("\"entryId\":\"id-2\""));
}

#[test]
fn audit_replay_dedupes_records_at_or_below_cursor() {
    let bus = ReplayBus::new(4);
    push_audit(
        &bus,
        &audit_payload("id-1", "2026-01-01T00:00:01Z", "runtime", "a", None),
    );
    push_audit(
        &bus,
        &audit_payload("id-2", "2026-01-01T00:00:02Z", "runtime", "b", None),
    );
    push_audit(
        &bus,
        &audit_payload("id-3", "2026-01-01T00:00:03Z", "runtime", "c", None),
    );

    let frames = replay_frames(&bus, &audit_subscription(AuditCursor(2)), None);
    assert_eq!(frames.len(), 1);
    assert!(frames[0].contains("id: a1:3"));
    assert!(frames[0].contains("id-3"));
}

#[test]
fn audit_replay_filters_by_source() {
    let bus = ReplayBus::new(4);
    push_audit(
        &bus,
        &audit_payload("id-1", "2026-01-01T00:00:01Z", "mesh", "a", None),
    );
    push_audit(
        &bus,
        &audit_payload("id-2", "2026-01-01T00:00:02Z", "runtime", "b", None),
    );
    push_audit(
        &bus,
        &audit_payload("id-3", "2026-01-01T00:00:03Z", "mesh", "c", None),
    );

    let sel = AuditSelection {
        cursor: AuditCursor(0),
        source: Some("mesh".to_owned()),
        severity: None,
    };
    let sub = Subscription {
        channels: Vec::new(),
        filters: Default::default(),
        cursor: Cursor::default(),
        audit: Some(sel.clone()),
    };
    let frames = replay_frames(&bus, &sub, None);
    assert_eq!(frames.len(), 2);
    assert!(frames[0].contains("id-1"));
    assert!(frames[1].contains("id-3"));
    assert!(!frames[0].contains("id-2"));
}

#[test]
fn audit_replay_filters_by_severity() {
    let bus = ReplayBus::new(4);
    push_audit(
        &bus,
        &audit_payload("id-1", "2026-01-01T00:00:01Z", "runtime", "a", Some("info")),
    );
    push_audit(
        &bus,
        &audit_payload(
            "id-2",
            "2026-01-01T00:00:02Z",
            "runtime",
            "b",
            Some("warning"),
        ),
    );
    push_audit(
        &bus,
        &audit_payload(
            "id-3",
            "2026-01-01T00:00:03Z",
            "runtime",
            "c",
            Some("error"),
        ),
    );

    let sel = AuditSelection {
        cursor: AuditCursor(0),
        source: None,
        severity: Some("warning".to_owned()),
    };
    let sub = Subscription {
        channels: Vec::new(),
        filters: Default::default(),
        cursor: Cursor::default(),
        audit: Some(sel),
    };
    let frames = replay_frames(&bus, &sub, None);
    assert_eq!(frames.len(), 1);
    assert!(frames[0].contains("id-2"));
}

#[test]
fn audit_gap_emits_recovery_endpoint() {
    let bus = ReplayBus::new(1);
    push_audit(
        &bus,
        &audit_payload("id-1", "2026-01-01T00:00:01Z", "runtime", "a", None),
    );
    push_audit(
        &bus,
        &audit_payload("id-2", "2026-01-01T00:00:02Z", "runtime", "b", None),
    );

    let frames = replay_frames(
        &bus,
        &audit_subscription(AuditCursor(0)),
        Some("a1:2".to_owned()),
    );
    assert!(frames[0].contains("event: replay_gap"));
    assert!(frames[0].contains("/api/logs/audit"));
    assert!(frames[0].contains("id: a1:1"));
    assert!(frames[1].contains("event: audit_entry"));
    assert!(frames[1].contains("id: a1:2"));
}

#[test]
fn audit_frames_never_contain_lifecycle_fields() {
    let bus = ReplayBus::new(3);
    push_audit(
        &bus,
        &audit_payload("id-1", "2026-01-01T00:00:01Z", "runtime", "a", None),
    );

    let frames = replay_frames(&bus, &audit_subscription(AuditCursor(0)), None);
    assert_eq!(frames.len(), 1);
    assert!(!frames[0].contains("canonical_envelope"));
    assert!(!frames[0].contains("detail_json"));
    assert!(!frames[0].contains("requestId"));
}

#[test]
fn lifecycle_gap_regression_still_uses_requests_endpoint() {
    let bus = ReplayBus::new(1);
    entry(&bus, ReplayChannel::Requests, 1, RequestId::new());
    entry(&bus, ReplayChannel::Requests, 2, RequestId::new());

    let frames = replay_frames(
        &bus,
        &subscription(vec![ReplayChannel::Requests], Cursor::default()),
        Some("opaque-rest-cursor".into()),
    );
    assert!(frames[0].contains("event: replay_gap"));
    assert!(frames[0].contains("/api/logs/requests"));
    assert!(!frames[0].contains("/api/logs/audit"));
}
