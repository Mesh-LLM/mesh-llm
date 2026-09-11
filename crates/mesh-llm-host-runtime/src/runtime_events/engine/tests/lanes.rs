//! Lane behavior, asserted on what publishes rather than on what a
//! producer is told.
//!
//! Coalescing and depth used to be producer-visible: a repeat key came
//! back `Coalesced`, and a full lane came back `DroppedDiagnostic`,
//! because both decisions were made inside the admission gate. They are
//! made in the consumer now, so a producer is simply told its fact was
//! accepted. The behavior itself is unchanged and is asserted here on the
//! published stream, which is what a consumer actually sees.

use mesh_llm_runtime_event_contracts::{
    OperationId, OperationScope, RuntimeEventIngress, SubmitOutcome,
};

use super::fixtures::{diagnostic_fact, progress_fact, state_transition_fact, synthetic_unknown};
use crate::runtime_events::engine::RuntimeEventEngine;
use crate::runtime_events::ingress::NON_TERMINAL_CREDITS;

#[test]
fn progress_on_a_reserved_operation_coalesces_to_the_latest_value() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let reservation = engine
        .reserve_root(OperationId::new(), synthetic_unknown)
        .expect("reserve");

    for _ in 0..5 {
        assert_eq!(
            reservation.ingress().try_submit(progress_fact()),
            SubmitOutcome::Accepted
        );
    }

    engine.drain();
    assert_eq!(
        engine.replay().snapshot().len(),
        1,
        "five progress updates inside one export window publish once"
    );
    assert_eq!(
        engine.health().snapshot().dropped_progress,
        4,
        "the four superseded snapshots are counted, not silently discarded"
    );

    reservation.cancel();
}

#[test]
fn unreserved_progress_is_dropped_and_counted() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let scope = OperationScope::root_only(OperationId::new());

    assert_eq!(
        engine.unreserved_ingress(scope).try_submit(progress_fact()),
        SubmitOutcome::DroppedProgress
    );
    assert_eq!(engine.health().snapshot().dropped_progress, 1);
}

#[test]
fn a_repeated_state_transition_kind_publishes_once_with_the_latest_value() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let scope = OperationScope::root_only(OperationId::new());
    let ingress = engine.unreserved_ingress(scope);

    for _ in 0..3 {
        assert_eq!(
            ingress.try_submit(state_transition_fact()),
            SubmitOutcome::Accepted
        );
    }

    engine.drain();
    assert_eq!(
        engine.replay().snapshot().len(),
        1,
        "the same (scope, kind) coalesces to one published frame"
    );
}

/// Coalescing is per `(scope, kind)`, never globally by kind: two
/// operations reporting the same kind must both publish.
#[test]
fn the_same_kind_from_different_scopes_never_coalesces() {
    let engine = RuntimeEventEngine::with_capacity(4);

    for _ in 0..3 {
        let scope = OperationScope::root_only(OperationId::new());
        assert_eq!(
            engine
                .unreserved_ingress(scope)
                .try_submit(state_transition_fact()),
            SubmitOutcome::Accepted
        );
    }

    engine.drain();
    assert_eq!(engine.replay().snapshot().len(), 3);
}

/// Diagnostics are lossless up to the ring's non-terminal budget, and the
/// first submission past it is refused and counted rather than silently
/// evicting an earlier one.
#[test]
fn diagnostics_past_the_non_terminal_budget_are_dropped_and_counted() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let scope = OperationScope::root_only(OperationId::new());
    let ingress = engine.unreserved_ingress(scope);

    for _ in 0..NON_TERMINAL_CREDITS {
        assert_eq!(
            ingress.try_submit(diagnostic_fact()),
            SubmitOutcome::Accepted
        );
    }

    assert_eq!(
        ingress.try_submit(diagnostic_fact()),
        SubmitOutcome::DroppedDiagnostic
    );
    assert_eq!(engine.health().snapshot().dropped_diagnostic, 1);

    // Draining returns the credits, so the ring recovers rather than
    // staying wedged at capacity.
    engine.drain();
    assert_eq!(
        ingress.try_submit(diagnostic_fact()),
        SubmitOutcome::Accepted
    );
}
