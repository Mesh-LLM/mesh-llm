//! Module-wiring tests: connection-shape classification and wire-payload
//! allowlist/required-field properties. Black-box HTTP/SSE byte tests live
//! under `crate::api::tests::runtime_events_v1` (they need the full
//! `MeshApi` + TCP harness already established there).

use mesh_llm_runtime_event_contracts::{
    FactData, FamilyFact, Outcome, ReasonCode, RequestEventKind, RuntimeEventIngress, RuntimeFact,
};

use crate::runtime_events::engine::RuntimeEventEngine;

use super::cursor::Cursor;
use super::frames::{EVENT_PROJECTION_ALLOWLIST, REQUIRED_ENVELOPE_KEYS, STATE_TOP_LEVEL_KEYS};
use super::recovery::{ConnectionShape, GapReason, classify};

fn synthetic_unknown() -> RuntimeFact {
    RuntimeFact::Request(FamilyFact::with_data(
        RequestEventKind::RequestFailed,
        FactData {
            outcome: Some(Outcome::Unknown),
            reason: Some(ReasonCode::TerminalNotDelivered),
            ..FactData::default()
        },
    ))
}

fn terminal_success() -> RuntimeFact {
    RuntimeFact::Request(FamilyFact::with_data(
        RequestEventKind::RequestCompleted,
        FactData {
            outcome: Some(Outcome::Success),
            ..FactData::default()
        },
    ))
}

fn submit_and_drain(engine: &std::sync::Arc<RuntimeEventEngine>, count: usize) {
    for _ in 0..count {
        let reservation = engine
            .reserve_root(
                mesh_llm_runtime_event_contracts::OperationId::new(),
                synthetic_unknown,
            )
            .expect("reserve");
        reservation.ingress().try_submit(terminal_success());
    }
    engine.drain();
}

#[test]
fn no_cursor_classifies_as_no_cursor() {
    let engine = RuntimeEventEngine::new();
    assert!(matches!(
        classify(&engine, None).expect("classify"),
        ConnectionShape::NoCursor
    ));
}

#[test]
fn in_window_cursor_returns_only_frames_strictly_after_it() {
    let engine = RuntimeEventEngine::new();
    submit_and_drain(&engine, 3);

    let cursor = Cursor::new(engine.process_instance(), 0);
    let ConnectionShape::InWindow { frames } = classify(&engine, Some(cursor)).expect("classify")
    else {
        panic!("expected in-window shape");
    };
    let sequences: Vec<u64> = frames.iter().map(|frame| frame.sequence.get()).collect();
    assert_eq!(sequences, vec![1, 2]);
}

#[test]
fn future_sequence_for_current_instance_is_rejected() {
    let engine = RuntimeEventEngine::new();
    submit_and_drain(&engine, 1);

    let cursor = Cursor::new(engine.process_instance(), 5);
    assert!(classify(&engine, Some(cursor)).is_err());
}

#[test]
fn cursor_before_any_event_for_a_fresh_engine_is_in_window_and_empty() {
    let engine = RuntimeEventEngine::new();
    let cursor = Cursor::new(engine.process_instance(), 0);
    let ConnectionShape::InWindow { frames } = classify(&engine, Some(cursor)).expect("classify")
    else {
        panic!("expected in-window shape");
    };
    assert!(frames.is_empty());
}

#[test]
fn stale_instance_is_a_gap() {
    let engine = RuntimeEventEngine::new();
    let foreign = mesh_llm_runtime_event_contracts::ProcessInstanceId::new();
    let cursor = Cursor::new(foreign, 0);
    let ConnectionShape::Gap(gap) = classify(&engine, Some(cursor)).expect("classify") else {
        panic!("expected gap shape");
    };
    assert_eq!(gap.reason, GapReason::StaleInstance);
}

#[test]
fn evicted_after_rebuild_is_a_gap() {
    let engine = RuntimeEventEngine::new();
    submit_and_drain(&engine, 2);
    engine.rebuild();

    let cursor = Cursor::new(engine.process_instance(), 0);
    let ConnectionShape::Gap(gap) = classify(&engine, Some(cursor)).expect("classify") else {
        panic!("expected gap shape");
    };
    assert_eq!(gap.reason, GapReason::Evicted);
}

#[test]
fn cursor_exactly_at_the_frontier_after_rebuild_is_in_window_not_evicted() {
    let engine = RuntimeEventEngine::new();
    submit_and_drain(&engine, 1);
    engine.rebuild();

    // Nothing was missed relative to this cursor even though replay was
    // evicted, so this must resolve as an (empty) in-window connection.
    let cursor = Cursor::new(engine.process_instance(), 0);
    assert!(matches!(
        classify(&engine, Some(cursor)).expect("classify"),
        ConnectionShape::InWindow { .. }
    ));
}

#[test]
fn state_top_level_keys_match_the_frozen_set() {
    let engine = RuntimeEventEngine::new();
    let projection = crate::runtime_event_api::state_projection::build(&engine);
    let value = serde_json::to_value(&projection).expect("serializable");
    let object = value.as_object().expect("state is a JSON object");
    let keys: Vec<&str> = object.keys().map(String::as_str).collect();
    for required in STATE_TOP_LEVEL_KEYS {
        assert!(keys.contains(required), "missing state key {required}");
    }
    assert_eq!(keys.len(), STATE_TOP_LEVEL_KEYS.len());
}

#[test]
fn event_projection_keys_are_a_subset_of_the_allowlist_for_every_submitted_kind() {
    let facts = [terminal_success(), synthetic_unknown()];
    for fact in facts {
        let projection = super::frames::event_projection(&fact);
        let value = serde_json::to_value(&projection).expect("serializable");
        let object = value.as_object().expect("event is a JSON object");
        for key in object.keys() {
            assert!(
                EVENT_PROJECTION_ALLOWLIST.contains(&key.as_str()),
                "key {key} is not in the projected-key allowlist"
            );
        }
    }
}

#[test]
fn envelope_frame_carries_every_required_key() {
    let engine = RuntimeEventEngine::new();
    let cursor = Cursor::new(engine.process_instance(), 0);
    let frame = super::frames::state_frame(&engine, cursor);
    let data_line = frame
        .lines()
        .find_map(|line| line.strip_prefix("data: "))
        .expect("data line present");
    let value: serde_json::Value = serde_json::from_str(data_line).expect("valid JSON");
    let object = value.as_object().expect("envelope is a JSON object");
    for required in REQUIRED_ENVELOPE_KEYS {
        assert!(object.contains_key(*required), "missing {required}");
    }
}

#[test]
fn every_frame_is_exactly_id_event_data_blank_line() {
    let engine = RuntimeEventEngine::new();
    let cursor = Cursor::new(engine.process_instance(), 0);
    let frame = super::frames::health_frame(&engine, cursor);
    let mut lines = frame.split('\n');
    assert!(lines.next().unwrap().starts_with("id: rt1:"));
    assert_eq!(lines.next().unwrap(), "event: runtime_health");
    assert!(lines.next().unwrap().starts_with("data: "));
    assert_eq!(lines.next().unwrap(), "");
    assert_eq!(lines.next(), Some(""));
    assert_eq!(lines.next(), None);
}

#[test]
fn keepalive_frame_has_no_id_or_data() {
    assert_eq!(super::frames::KEEPALIVE_FRAME, ": keepalive\n\n");
}

// ─── Shared Rust/TS fixture round-trip ─────────────────────────────────

const FRAMES_FIXTURE: &str = include_str!(
    "../../../../../mesh-llm-runtime-event-contracts/fixtures/runtime_events_v1/frames.json"
);
const CURSORS_FIXTURE: &str = include_str!(
    "../../../../../mesh-llm-runtime-event-contracts/fixtures/runtime_events_v1/cursors.json"
);
const RECOVERY_FIXTURE: &str = include_str!(
    "../../../../../mesh-llm-runtime-event-contracts/fixtures/runtime_events_v1/recovery.json"
);

#[test]
fn frames_fixture_required_keys_match_the_rust_constants() {
    let fixture: serde_json::Value = serde_json::from_str(FRAMES_FIXTURE).expect("valid JSON");
    let required: Vec<&str> = fixture["required_envelope_keys"]
        .as_array()
        .expect("array")
        .iter()
        .map(|value| value.as_str().expect("string"))
        .collect();
    assert_eq!(required, REQUIRED_ENVELOPE_KEYS);

    let state_keys: Vec<&str> = fixture["state_top_level_keys"]
        .as_array()
        .expect("array")
        .iter()
        .map(|value| value.as_str().expect("string"))
        .collect();
    assert_eq!(state_keys, STATE_TOP_LEVEL_KEYS);

    let allowlist: Vec<&str> = fixture["event_projected_key_allowlist"]
        .as_array()
        .expect("array")
        .iter()
        .map(|value| value.as_str().expect("string"))
        .collect();
    assert_eq!(allowlist, EVENT_PROJECTION_ALLOWLIST);

    assert_eq!(fixture["keepalive_frame"], super::frames::KEEPALIVE_FRAME);
}

#[test]
fn cursors_fixture_examples_parse_exactly_as_declared() {
    let fixture: serde_json::Value = serde_json::from_str(CURSORS_FIXTURE).expect("valid JSON");
    for value in fixture["valid"].as_array().expect("array") {
        let text = value.as_str().expect("string");
        assert!(Cursor::parse(text).is_ok(), "must parse: {text}");
    }
    for value in fixture["invalid"].as_array().expect("array") {
        let text = value.as_str().expect("string");
        assert!(Cursor::parse(text).is_err(), "must reject: {text}");
    }
}

#[test]
fn recovery_fixture_gap_reasons_match_the_rust_enum() {
    let fixture: serde_json::Value = serde_json::from_str(RECOVERY_FIXTURE).expect("valid JSON");
    let reasons = fixture["replay_gap_reasons"].as_object().expect("object");
    assert!(reasons.contains_key(GapReason::StaleInstance.as_str()));
    assert!(reasons.contains_key(GapReason::Evicted.as_str()));
}
