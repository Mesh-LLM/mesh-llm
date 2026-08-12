//! Cross-module acceptance tests for canonical logging contracts.

use super::envelope::{
    CanonicalEnvelope, CanonicalEnvelopeParseError, SCHEMA_VERSION, UnsupportedSchemaVersion,
};
use super::events::LifecycleEvent;
use super::identifiers::{AttemptId, EventId, RequestId};
use super::lifecycle::{LifecycleGuard, LifecycleState, LifecycleTransitionError};
use super::replay::ReplayChannel;

#[test]
fn test_invalid_lifecycle_transition() {
    let mut guard = LifecycleGuard::active();
    assert!(guard.transition(LifecycleState::Completed).is_ok());
    let error = guard.transition(LifecycleState::Failed).unwrap_err();
    assert!(matches!(
        error,
        LifecycleTransitionError {
            from: LifecycleState::Completed,
            to: LifecycleState::Failed,
        }
    ));
    assert_eq!(guard.state(), LifecycleState::Completed);
}

#[test]
fn test_second_terminal_rejected() {
    let mut guard = LifecycleGuard::active();
    assert!(guard.transition(LifecycleState::Completed).is_ok());
    let error = guard.transition(LifecycleState::Completed).unwrap_err();
    assert!(matches!(
        error,
        LifecycleTransitionError {
            from: LifecycleState::Completed,
            to: LifecycleState::Completed,
        }
    ));
    assert_eq!(guard.state(), LifecycleState::Completed);
}

#[test]
fn test_serde_roundtrip_preserves_fields() {
    let env = CanonicalEnvelope::new(
        EventId::new(),
        RequestId::new(),
        ReplayChannel::Requests,
        7,
        "2025-01-01T00:00:00Z".into(),
        LifecycleEvent::Admitted {
            model: Some("llama-3".into()),
            method: Some("POST".into()),
        },
    );
    let json = serde_json::to_string(&env).unwrap();
    let wire: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(wire["schema_version"], serde_json::json!(SCHEMA_VERSION));
    let parsed: CanonicalEnvelope = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.schema_version, env.schema_version);
    assert_eq!(parsed.channel, ReplayChannel::Requests);
    assert_eq!(parsed.sequence, 7);
    assert_eq!(parsed.event, env.event);
}

#[test]
fn test_attempt_events_use_branded_ids_without_changing_uuid_wire_shape() {
    let uuid = uuid::Uuid::parse_str("4ba6be8e-11c7-4aac-9688-b7bf920d190a").unwrap();
    let attempt_id = AttemptId::from(uuid);
    let events = [
        LifecycleEvent::AttemptStarted {
            attempt_id: Some(attempt_id),
        },
        LifecycleEvent::AttemptCompleted {
            attempt_id: Some(attempt_id),
            status_code: Some(200),
        },
        LifecycleEvent::AttemptFailed {
            attempt_id: Some(attempt_id),
            error: Some("timeout".into()),
        },
    ];

    for event in events {
        let attempt_id = match &event {
            LifecycleEvent::AttemptStarted { attempt_id }
            | LifecycleEvent::AttemptCompleted { attempt_id, .. }
            | LifecycleEvent::AttemptFailed { attempt_id, .. } => *attempt_id,
            _ => unreachable!("all fixtures are attempt events"),
        };
        assert_branded_attempt_id(attempt_id);

        let json = serde_json::to_value(&event).unwrap();
        assert_eq!(json["attempt_id"], serde_json::json!(uuid.to_string()));
        assert_eq!(
            serde_json::from_value::<LifecycleEvent>(json).unwrap(),
            event
        );
    }
}

fn assert_branded_attempt_id(_: Option<AttemptId>) {}

#[test]
fn test_unknown_schema_version_returns_typed_parse_error() {
    let env = CanonicalEnvelope::new(
        EventId::new(),
        RequestId::new(),
        ReplayChannel::Requests,
        7,
        "2025-01-01T00:00:00Z".into(),
        LifecycleEvent::Admitted {
            model: Some("llama-3".into()),
            method: Some("POST".into()),
        },
    );
    for version in [0, SCHEMA_VERSION + 1] {
        let mut value = serde_json::to_value(&env).unwrap();
        value["schema_version"] = serde_json::json!(version);

        let json = serde_json::to_string(&value).unwrap();
        let error = CanonicalEnvelope::from_json_str(&json).unwrap_err();
        match error {
            CanonicalEnvelopeParseError::UnsupportedSchemaVersion(UnsupportedSchemaVersion {
                version: actual_version,
            }) => assert_eq!(actual_version, version),
            other => panic!("expected typed unsupported-version error, got {other:?}"),
        }
    }
}

#[test]
fn test_absent_identity_fields_omitted() {
    let env = CanonicalEnvelope::new(
        EventId::new(),
        RequestId::new(),
        ReplayChannel::System,
        0,
        "2025-01-01T00:00:00Z".into(),
        LifecycleEvent::Completed {
            status_code: Some(200),
            duration_ms: None,
        },
    );
    let json = serde_json::to_string(&env).unwrap();
    assert!(!json.contains("tenant_id"));
    assert!(!json.contains("account_id"));
    assert!(!json.contains("user_id"));
    assert!(!json.contains("\"role\""));
}

#[test]
fn test_no_raw_invite_token_in_serialized_events() {
    // The contract types introduce no token-shaped fields: serialized output
    // never gains keys like "invite" or "token". Caller-supplied error strings
    // pass through verbatim (redaction is a later policy-layer concern).
    let env = CanonicalEnvelope::new(
        EventId::new(),
        RequestId::new(),
        ReplayChannel::Requests,
        1,
        "2025-01-01T00:00:00Z".into(),
        LifecycleEvent::Failed {
            error: "upstream rejected the request".into(),
        },
    );
    let json = serde_json::to_string(&env).unwrap();
    let parsed: CanonicalEnvelope = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.event, env.event);
    // No token-bearing key is introduced by the envelope itself.
    let value: serde_json::Value = serde_json::from_str(&json).unwrap();
    let obj = value.as_object().unwrap();
    assert!(!obj.contains_key("invite"));
    assert!(!obj.contains_key("token"));
    assert!(!obj.contains_key("invite_token"));
}
