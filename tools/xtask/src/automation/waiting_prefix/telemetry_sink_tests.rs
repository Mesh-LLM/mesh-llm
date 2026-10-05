use super::*;
use serde_json::json;

fn event(attributes: Value) -> Vec<u8> {
    serde_json::to_vec(&json!({"event":"stage.openai_generation_summary","attributes":attributes}))
        .unwrap()
}

#[test]
fn projects_only_numeric_measurements_known_statuses_and_native_request_ids() {
    let raw = event(
        json!({"skippy.kv.status":"hit","skippy.kv.suffix_prefill_tokens":3,
        "skippy.request_id":"42","authorization":"private credential","prompt":"secret prompt",
        "skippy.kv.capacity_status":"private credential"}),
    );
    let projected = measurement(&raw).unwrap().unwrap();
    let text = serde_json::to_string(&projected).unwrap();
    assert_eq!(projected.attributes["skippy.request_id"], "42");
    assert_eq!(projected.attributes["skippy.kv.suffix_prefill_tokens"], 3);
    assert_eq!(projected.attributes["skippy.kv.status"], "hit");
    for forbidden in [
        "authorization",
        "private credential",
        "secret prompt",
        "prompt",
    ] {
        assert!(!text.contains(forbidden), "{forbidden}");
    }
}

#[test]
fn rejects_invalid_measurements_and_ignores_unowned_diagnostics() {
    for attributes in [
        json!({"skippy.request_id":"secret"}),
        json!({"skippy.kv.suffix_prefill_tokens":-1}),
        json!({"skippy.kv.suffix_prefill_tokens":"secret"}),
    ] {
        assert!(measurement(&event(attributes)).is_err());
    }
    assert!(
        measurement(b"unstructured diagnostic token")
            .unwrap()
            .is_none()
    );
    assert!(
        measurement(br#"{"event":"unowned","attributes":{"token":"secret"}}"#)
            .unwrap()
            .is_none()
    );
    assert!(measurement(&vec![b'x'; 8193]).is_err());
}

#[test]
fn writer_publishes_complete_lines_without_diagnostic_suppression() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("measurements.log");
    let mut sink = Sink::create(&path).unwrap();
    sink.observe(&event(json!({"skippy.kv.suffix_prefill_tokens":3})))
        .unwrap();
    sink.tick().unwrap();
    sink.finish().unwrap();
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(bytes.last(), Some(&b'\n'));
    let value: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(value["attributes"]["skippy.kv.suffix_prefill_tokens"], 3);
    assert!(Sink::create(&path).is_err());
}

#[test]
fn callback_queue_overflow_is_a_failure_without_unbounded_retention() {
    let directory = tempfile::tempdir().unwrap();
    let mut sink = Sink::create(&directory.path().join("measurements.log")).unwrap();
    for _ in 0..MAX_PENDING {
        sink.observe(&event(json!({}))).unwrap();
    }
    assert!(sink.observe(&event(json!({}))).is_err());
    assert_eq!(sink.pending.len(), MAX_PENDING);
    assert!(sink.finish().is_err());
}
