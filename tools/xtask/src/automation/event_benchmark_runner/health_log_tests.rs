use super::*;
use serde_json::json;
fn record(message: &str) -> Vec<u8> {
    serde_json::to_vec(&json!({"context":"event_system_health","message":message})).unwrap()
}

#[test]
fn missing_or_unrelated_health_is_unmeasured_without_synthesized_zero_values() {
    for log in [
        b"".as_slice(),
        b"banner\n{broken\n",
        b"{\"context\":\"other\",\"message\":\"dropped_progress=0\"}",
    ] {
        assert_eq!(final_observation(log), Observation::default());
    }
}
#[test]
fn final_valid_health_line_preserves_actual_boolean_and_counter_values() {
    let first = record("version=1 dropped_progress=20 ingress_p99_us=55");
    let last = record(
        "version=1 dropped_progress=1 dropped_diagnostic=0 state_degraded=true rebuild_required=false state_transition_rejected=2 ingress_p99_us=6",
    );
    let log = [first, b"\n{malformed}\n".to_vec(), last].concat();
    let result = final_observation(&log);
    let health = result.health.unwrap();
    assert_eq!(health["dropped_progress"], 1);
    assert_eq!(health["state_degraded"], true);
    assert_eq!(health["rebuild_required"], false);
    assert_eq!(health["state_transition_rejected"], 2);
    assert_eq!(result.ingress_p99_us, Some(6.0));
}
#[test]
fn nullable_or_invalid_p99_does_not_erase_observed_health() {
    for p99 in ["null", "NaN", "true", "-1", "1e10000"] {
        let result = line(&record(&format!("dropped_progress=1 ingress_p99_us={p99}"))).unwrap();
        assert_eq!(result.health.unwrap()["dropped_progress"], 1);
        assert_eq!(result.ingress_p99_us, None);
    }
}
#[test]
fn forward_fields_and_malformed_tokens_do_not_redefine_health_counters() {
    let result=line(&record("dropped_progress=1 malformed bounds.reservation_table_capacity=64 future_field=8 dropped_diagnostic=0")).unwrap();
    let health = result.health.unwrap();
    assert_eq!(health.len(), 2);
    assert!(!health.contains_key("future_field"));
}
#[test]
fn negative_or_wrongly_typed_known_counts_are_not_hidden_as_default_zero() {
    let result = line(&record(
        "cancelled_reservation_rejected=-1 dropped_progress=true",
    ))
    .unwrap();
    let health = result.health.unwrap();
    assert_eq!(health["cancelled_reservation_rejected"], -1);
    assert_eq!(health["dropped_progress"], true);
}
#[test]
fn malformed_envelopes_and_oversized_records_do_not_replace_prior_observation() {
    let prior = record("dropped_progress=2 ingress_p99_us=10");
    let bad = serde_json::to_vec(&json!({"context":"event_system_health","message":7})).unwrap();
    let oversized = record(&"x".repeat(20 * 1024));
    let log = [prior, b"\n".to_vec(), bad, b"\n".to_vec(), oversized].concat();
    assert_eq!(
        final_observation(&log).health.unwrap()["dropped_progress"],
        2
    );
}

#[test]
fn final_health_preserves_terminal_diagnostic_cancellation_counts_and_measured_p99_together() {
    let first = record(
        "terminal_delivery_failed=99 dropped_progress=99 dropped_diagnostic=99 cancelled_reservation_rejected=99 ingress_p99_us=99",
    );
    let final_line = record(
        "terminal_delivery_failed=2 dropped_progress=7 dropped_diagnostic=3 cancelled_reservation_rejected=0 ingress_p99_us=6",
    );
    let result = final_observation(&[first, b"\n".to_vec(), final_line].concat());
    let health = result.health.unwrap();
    assert_eq!(health["terminal_delivery_failed"], 2);
    assert_eq!(health["dropped_progress"], 7);
    assert_eq!(health["dropped_diagnostic"], 3);
    assert_eq!(health["cancelled_reservation_rejected"], 0);
    assert_eq!(result.ingress_p99_us, Some(6.0));
}
