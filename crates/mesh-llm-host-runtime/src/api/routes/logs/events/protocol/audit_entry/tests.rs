use super::*;
use crate::logging::{AuditReplayRecord, BusEntry};
use mesh_llm_log_store::{AuditEntryDetail, AuditEntryFilters, LogStore, RealClock};
use std::sync::Arc;

use super::super::MAX_FRAME_BYTES;

fn audit_record(sequence: u64) -> AuditReplayRecord {
    AuditReplayRecord {
        entry: BusEntry {
            payload: serde_json::json!({
                "kind": "audit",
                "entry_id": "test-entry-id",
                "occurred_at": "2026-01-01T00:00:00Z",
                "source": "runtime",
                "code": "startup_complete",
                "severity": "info",
            })
            .to_string(),
            channel_hint: 2,
        },
        sequence,
        cursor: sequence,
    }
}

fn durable_audit_detail(entry_id: &str, detail: serde_json::Value) -> AuditEntryDetail {
    let root = tempfile::tempdir().expect("temporary durable audit store");
    let store = LogStore::open(root.path(), Arc::new(RealClock)).expect("open audit store");
    store
        .insert_audit_entry(
            entry_id,
            None,
            "2026-01-01T00:00:00Z",
            "cli",
            "command_completed",
            Some(&detail.to_string()),
        )
        .expect("insert durable audit entry");
    store
        .list_audit_entry_details_after_sequence(0, 1, AuditEntryFilters::default())
        .expect("query durable audit entry")
        .into_iter()
        .next()
        .expect("durable audit entry")
}

fn frame_data(frame: &str) -> serde_json::Value {
    frame
        .lines()
        .find_map(|line| line.strip_prefix("data: "))
        .and_then(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .expect("audit frame data")
}

#[test]
fn audit_entry_frame_shape_and_fields() {
    let record = audit_record(7);
    let frame = audit_entry_frame(&record).expect("audit entry frame");

    assert!(frame.contains("event: audit_entry"));
    assert!(frame.contains("id: a1:7"));
    assert!(frame.contains("\"entryId\":\"test-entry-id\""));
    assert!(frame.contains("\"occurredAt\":\"2026-01-01T00:00:00Z\""));
    assert!(frame.contains("\"source\":\"runtime\""));
    assert!(frame.contains("\"code\":\"startup_complete\""));
    assert!(frame.contains("\"severity\":\"info\""));
    assert!(frame.contains("\"sequence\":7"));
    assert!(!frame.contains("canonical_envelope"));
    assert!(!frame.contains("detail_json"));
    assert!(frame.len() <= MAX_FRAME_BYTES);
}

#[test]
fn audit_entry_frame_omits_severity_when_none() {
    let mut record = audit_record(3);
    record.entry.payload = serde_json::json!({
        "kind": "audit",
        "entry_id": "id-3",
        "occurred_at": "2026-01-01T00:00:00Z",
        "source": "cli",
        "code": "command_executed",
    })
    .to_string();
    let frame = audit_entry_frame(&record).expect("audit entry without severity");
    assert!(!frame.contains("severity"));
}

#[test]
fn audit_entry_frame_filters_invalid_typed_context_fields() {
    let mut record = audit_record(4);
    record.entry.payload = serde_json::json!({
        "kind": "audit",
        "entry_id": "id-4",
        "occurred_at": "2026-01-01T00:00:00Z",
        "source": "runtime",
        "code": "startup_complete",
        "context_version": 1,
        "subject_kind": "not_a_subject",
        "reason_code": "NOT VALID",
        "outcome": "completed",
        "numeric_summaries": {
            "bad-key": 0,
            "metric_0": 0,
            "metric_1": 1,
            "metric_2": 2,
            "metric_3": 3,
            "metric_4": 4,
            "metric_5": 5,
            "metric_6": 6,
            "metric_7": 7,
            "metric_8": 8
        }
    })
    .to_string();
    let frame = audit_entry_frame(&record).expect("audit context frame");
    let data = frame_data(&frame);

    assert!(data.get("subjectKind").is_none());
    assert!(data.get("reasonCode").is_none());
    assert_eq!(data["outcome"], "completed");
    assert_eq!(data["numericSummaries"].as_object().unwrap().len(), 8);
    assert!(data["numericSummaries"].get("metric_7").is_some());
    assert!(data["numericSummaries"].get("metric_8").is_none());
    assert!(data["numericSummaries"].get("bad-key").is_none());
}

#[test]
fn live_audit_path_projection_is_privacy_filtered() {
    let cases = [
        (
            1,
            "direct",
            Some("[2001:0db8:0:0:0:0:0:1]:443"),
            Some("direct"),
            Some("[2001:db8::1]:443"),
        ),
        (
            1,
            "relay",
            Some("private-relay.example:443"),
            Some("relay"),
            None,
        ),
        (
            1,
            "direct",
            Some("private-hostname:443"),
            Some("direct"),
            None,
        ),
        (1, "unknown", Some("192.0.2.1:443"), None, None),
        (2, "direct", Some("192.0.2.1:443"), None, None),
    ];

    for (context_version, path_type, remote_addr, expected_path, expected_addr) in cases {
        let mut record = audit_record(20);
        record.entry.payload = serde_json::json!({
            "kind": "audit",
            "entry_id": "mesh-path",
            "occurred_at": "2026-01-01T00:00:00Z",
            "source": "mesh",
            "code": "peer_path_observed",
            "context_version": context_version,
            "path_type": path_type,
            "remote_addr": remote_addr,
            "canonical_envelope": "must-not-leak",
            "detail_json": "must-not-leak"
        })
        .to_string();

        let frame = audit_entry_frame(&record).expect("live audit path frame");
        let data = frame_data(&frame);

        assert_eq!(
            data.get("pathType").and_then(|value| value.as_str()),
            expected_path
        );
        assert_eq!(
            data.get("remoteAddr").and_then(|value| value.as_str()),
            expected_addr
        );
        assert!(data.get("canonical_envelope").is_none());
        assert!(data.get("detail_json").is_none());
    }
}

#[test]
fn audit_entry_frame_rejects_malformed_payload() {
    let mut record = audit_record(1);
    record.entry.payload = "not-json".to_string();
    assert!(audit_entry_frame(&record).is_err());
}

#[test]
fn live_audit_frame_drops_malformed_command_summary() {
    let mut record = audit_record(8);
    record.entry.payload = serde_json::json!({
        "kind": "audit",
        "entry_id": "id-8",
        "occurred_at": "2026-01-01T00:00:00Z",
        "source": "cli",
        "code": "command_completed",
        "context_version": 1,
        "command_summary": "mesh-llm gpus --draft run-benchmark --backend cuda"
    })
    .to_string();

    let frame = audit_entry_frame(&record).expect("live audit frame");
    assert!(!frame.contains("commandSummary"));
}

#[test]
fn live_audit_frame_drops_duplicate_command_summary_flags() {
    let mut record = audit_record(13);
    record.entry.payload = serde_json::json!({
        "kind": "audit",
        "entry_id": "id-13",
        "occurred_at": "2026-01-01T00:00:00Z",
        "source": "cli",
        "code": "command_completed",
        "context_version": 1,
        "command_summary": "mesh-llm models list --json --json"
    })
    .to_string();

    let frame = audit_entry_frame(&record).expect("live audit frame");
    assert!(!frame.contains("commandSummary"));
}

#[test]
fn live_audit_frame_drops_deep_malformed_command_summary() {
    let mut record = audit_record(11);
    record.entry.payload = serde_json::json!({
        "kind": "audit",
        "entry_id": "id-11",
        "occurred_at": "2026-01-01T00:00:00Z",
        "source": "cli",
        "code": "command_completed",
        "context_version": 1,
        "command_summary": "mesh-llm load unload status discover rotate-key setup --port 1234"
    })
    .to_string();

    let frame = audit_entry_frame(&record).expect("live audit frame");
    assert!(!frame.contains("commandSummary"));
}

#[test]
fn live_audit_frame_preserves_valid_command_summary() {
    let mut record = audit_record(10);
    record.entry.payload = serde_json::json!({
        "kind": "audit",
        "entry_id": "id-10",
        "occurred_at": "2026-01-01T00:00:00Z",
        "source": "cli",
        "code": "command_completed",
        "context_version": 1,
        "command_summary": "mesh-llm runtime guardrails --mode metrics --port 41731 --root-relay [REDACTED]"
    })
    .to_string();

    let frame = audit_entry_frame(&record).expect("live audit frame");
    assert!(frame.contains(
        "mesh-llm runtime guardrails --mode metrics --port 41731 --root-relay [REDACTED]"
    ));
}

#[test]
fn durable_audit_frame_drops_malformed_command_summary() {
    let record = durable_audit_detail(
        "id-9",
        serde_json::json!({
            "context_version": 1,
            "command_summary": "mesh-llm gpus --draft run-benchmark --backend cuda",
        }),
    );

    let frame = durable_audit_entry_frame(record).expect("durable audit frame");
    assert!(!frame.contains("commandSummary"));
}

#[test]
fn durable_audit_frame_drops_deep_malformed_command_summary() {
    let record = durable_audit_detail(
        "id-12",
        serde_json::json!({
            "context_version": 1,
            "command_summary": "mesh-llm load unload status discover rotate-key setup --port 1234",
        }),
    );

    let frame = durable_audit_entry_frame(record).expect("durable audit frame");
    assert!(!frame.contains("commandSummary"));
}

#[test]
fn durable_audit_frame_preserves_valid_command_summary() {
    let record = durable_audit_detail(
        "id-11",
        serde_json::json!({
            "context_version": 1,
            "command_summary": "mesh-llm runtime guardrails --mode metrics --port 41731 --root-relay [REDACTED]",
        }),
    );

    let frame = durable_audit_entry_frame(record).expect("durable audit frame");
    assert!(frame.contains(
        "mesh-llm runtime guardrails --mode metrics --port 41731 --root-relay [REDACTED]"
    ));
}

#[test]
fn durable_audit_frame_redacts_unsafe_rest_parity_metadata() {
    let record = durable_audit_detail(
        "id-14",
        serde_json::json!({
            "context_version": 1,
            "subject_id": "https://alice:subject-secret@example.test/model?api_key=subject-query",
            "operation_id": "/Users/alice/private-operation",
            "request_id": "request-7?token=request-secret",
        }),
    );

    let frame = durable_audit_entry_frame(record).expect("durable audit frame");
    let data = frame_data(&frame);

    assert_eq!(data["subjectId"], "[REDACTED]");
    assert_eq!(data["operationId"], "[REDACTED]");
    assert_eq!(data["requestId"], "[REDACTED]");

    let serialized = data.to_string();
    for unsafe_value in [
        "alice",
        "subject-secret",
        "subject-query",
        "/Users/alice/private-operation",
        "request-secret",
    ] {
        assert!(!serialized.contains(unsafe_value));
    }
}
