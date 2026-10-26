use super::*;
use serde_json::json;

fn health(value: serde_json::Value) -> Health {
    serde_json::from_value(value).unwrap()
}

#[test]
fn missing_health_is_unavailable_without_fabricated_counter_violations() {
    let evidence = evaluate("production", None, Some(20), Some(20));
    assert!(!evidence.available);
    assert!(evidence.violations.is_empty());
}

#[test]
fn absent_off_mode_engine_is_an_intentional_observation() {
    let evidence = evaluate("off", None, Some(20), Some(20));
    assert!(evidence.available);
    assert!(evidence.violations.is_empty());
}

#[test]
fn measured_critical_drops_and_state_flags_each_block() {
    let health = health(
        json!({"terminal_delivery_failed":1,"state_lane_evictions":1,"state_transition_rejected":1,"state_degraded":true,"rebuild_required":true}),
    );
    let evidence = evaluate("production", Some(&health), None, None);
    assert!(evidence.available);
    assert_eq!(evidence.violations.len(), 5);
}

#[test]
fn progress_and_diagnostic_counts_must_match_each_sides_expectations_exactly() {
    let health = health(json!({"dropped_progress":19,"dropped_diagnostic":20}));
    assert_eq!(
        evaluate("production", Some(&health), Some(20), Some(20))
            .violations
            .len(),
        1
    );
    assert!(
        evaluate("production", Some(&health), Some(19), Some(20))
            .violations
            .is_empty()
    );
    assert_eq!(
        evaluate("production", Some(&health), Some(19), Some(19))
            .violations
            .len(),
        1
    );
}

#[test]
fn cancellation_rejection_is_nonnegative_evidence_and_does_not_imply_state_loss() {
    let health = health(json!({"cancelled_reservation_rejected":u64::MAX}));
    assert!(
        evaluate("production", Some(&health), Some(0), Some(0))
            .violations
            .is_empty()
    );
    for value in [json!(-1), json!(true), json!(1.5), json!("1")] {
        assert!(
            serde_json::from_value::<Health>(json!({"cancelled_reservation_rejected":value}))
                .is_err()
        );
    }
}
