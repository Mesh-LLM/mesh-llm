use mesh_llm_runtime_event_contracts::{OperationId, RuntimeEventIngress, SubmitOutcome};

use super::fixtures::{synthetic_unknown, terminal_success};
use crate::runtime_events::engine::RuntimeEventEngine;

#[test]
fn shutdown_drains_a_small_wake_list_fully_within_budget() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let reservation = engine
        .reserve_root(OperationId::new(), synthetic_unknown)
        .expect("reserve");
    reservation.ingress().try_submit(terminal_success());

    let report = engine.shutdown(None);

    assert_eq!(report.started_with, 1);
    assert_eq!(report.applied, 1);
    assert_eq!(report.remaining_after_deadline, 0);
    assert_eq!(engine.health().snapshot().shutdown_degraded, 0);
    assert!(engine.is_shutting_down());
}

#[test]
fn shutdown_past_its_drain_budget_degrades_the_remainder_instead_of_hanging() {
    let engine = RuntimeEventEngine::with_capacity(4);
    for _ in 0..3 {
        let reservation = engine
            .reserve_root(OperationId::new(), synthetic_unknown)
            .expect("reserve");
        reservation.ingress().try_submit(terminal_success());
    }

    // Simulate the deadline expiring after exactly one drained item,
    // deterministically (no real sleep or clock dependence).
    let report = engine.shutdown(Some(1));

    assert_eq!(report.started_with, 3);
    assert_eq!(report.applied, 1);
    assert_eq!(report.remaining_after_deadline, 2);
    assert_eq!(engine.health().snapshot().shutdown_degraded, 1);
    assert_eq!(engine.health().snapshot().terminal_delivery_failed, 2);
}

#[test]
fn admission_still_succeeds_during_shutdown_because_exhaustion_is_the_only_refusal_signal() {
    let engine = RuntimeEventEngine::with_capacity(4);
    engine.shutdown(Some(0));

    // The plan reserves `RejectedShuttingDown` for producer-side lanes, not
    // for admission itself: admission's only refusal signal is capacity
    // exhaustion, so an operation reserved after shutdown begins can still
    // complete cleanly through the drain seam.
    let reservation = engine
        .reserve_root(OperationId::new(), synthetic_unknown)
        .expect("admission still succeeds under capacity during shutdown");
    assert_eq!(
        reservation.ingress().try_submit(terminal_success()),
        SubmitOutcome::Accepted
    );
}
