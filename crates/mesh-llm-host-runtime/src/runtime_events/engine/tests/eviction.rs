//! Task 6 defect A and defect C (verifier follow-up on
//! `.omo/plans/event-system-fixes.md` task 6): the reducer's `operations`
//! map must be evicted RELEASE-triggered, not merely size-triggered, and
//! an in-flight (still-occupied) operation must never be evicted by
//! either the release-triggered path or the size-triggered capacity
//! backstop.

use mesh_llm_runtime_event_contracts::{
    OperationId, OperationScope, RuntimeEventIngress, SubmitOutcome,
};

use super::fixtures::{state_transition_fact, synthetic_unknown, terminal_success};
use crate::runtime_events::config::RESERVATION_TABLE_CAPACITY;
use crate::runtime_events::engine::RuntimeEventEngine;

/// Task 6 defect A: the parent commit's `evict_settled_over_capacity` only
/// trimmed the `operations` map once it exceeded
/// `RESERVATION_TABLE_CAPACITY`, so 100 sequential reserve -> terminal ->
/// drain cycles left `operation_count()` growing to 100 and pinning there
/// forever -- never shrinking even though every one of those operations
/// had long since settled and released its reservation. Eviction must
/// instead be tied to the release itself, so the map tracks in-flight
/// operations only.
#[test]
fn operation_count_does_not_grow_across_sequential_completions() {
    let engine = RuntimeEventEngine::new();
    for iteration in 0..100 {
        let reservation = engine
            .reserve_root(OperationId::new(), synthetic_unknown)
            .expect("reserve");
        assert_eq!(
            reservation.ingress().try_submit(terminal_success()),
            SubmitOutcome::Accepted
        );
        engine.drain();

        let tracked = engine.reducer_snapshot().operation_count();
        assert_eq!(
            tracked, 0,
            "iteration {iteration}: a settled root's reservation released \
             in this same drain pass, so the reducer must not still be \
             tracking it; got operation_count() = {tracked}"
        );
    }
}

/// Task 6 defect C: an operation whose reservation is STILL occupied must
/// never be evicted by ANY path -- release-triggered (defect A) or the
/// size-triggered capacity backstop -- even while thousands of sibling
/// operations complete, release, and get evicted around it.
#[test]
fn a_held_reservation_survives_release_triggered_eviction_of_many_completed_siblings() {
    let engine = RuntimeEventEngine::new();
    let held = engine
        .reserve_root(OperationId::new(), synthetic_unknown)
        .expect("reserve");
    let held_scope = held.scope();
    assert_eq!(
        held.ingress().try_submit(state_transition_fact()),
        SubmitOutcome::Accepted,
        "a non-terminal fact keeps the reservation open but still tracks state"
    );
    engine.drain();
    assert!(
        engine.reducer_snapshot().operation(held_scope).is_some(),
        "the held scope's state-transition fact must be tracked"
    );

    for _ in 0..(RESERVATION_TABLE_CAPACITY + 200) {
        let reservation = engine
            .reserve_root(OperationId::new(), synthetic_unknown)
            .expect("reserve");
        assert_eq!(
            reservation.ingress().try_submit(terminal_success()),
            SubmitOutcome::Accepted
        );
        engine.drain();
    }

    let snapshot = engine.reducer_snapshot();
    let state = snapshot
        .operation(held_scope)
        .expect("a still-occupied reservation must never be evicted");
    assert!(
        !state.settled,
        "the held operation must still report unsettled"
    );
    assert_eq!(
        snapshot.operation_count(),
        1,
        "release-triggered eviction must leave the map tracking only the \
         still-occupied operation, not pinned at RESERVATION_TABLE_CAPACITY"
    );

    drop(held);
    engine.drain();
}

/// Task 6-fix "also required" (non-blocking review finding on top of
/// defect A): the settled-only capacity backstop's "nothing left to evict"
/// branch used to be a silent `break` -- unbounded growth with no counter
/// and no log. This drives the backstop's stall condition for real (via
/// unreserved state-transition ingress, which bypasses the reservation
/// table's own admission cap entirely -- the only way to get more than
/// `RESERVATION_TABLE_CAPACITY` permanently-unsettled entries into the
/// reducer through the real engine, since a normal reserved caller can
/// never hold more than `RESERVATION_TABLE_CAPACITY` concurrently-occupied
/// slots) and asserts the stall is now observable as a health counter bump.
#[test]
fn eviction_backstop_stall_is_observable_in_engine_health() {
    let engine = RuntimeEventEngine::new();
    assert_eq!(engine.health().snapshot().reducer_eviction_stalled, 0);

    for _ in 0..(RESERVATION_TABLE_CAPACITY + 64) {
        let scope = OperationScope::root_only(OperationId::new());
        assert_eq!(
            engine
                .unreserved_ingress(scope)
                .try_submit(state_transition_fact()),
            SubmitOutcome::Accepted,
            "a distinct scope's state-transition fact never coalesces with another scope's"
        );
    }
    engine.drain();

    assert!(
        engine.reducer_snapshot().operation_count() > RESERVATION_TABLE_CAPACITY,
        "this test's whole point is to genuinely exceed capacity with nothing settled to evict"
    );
    assert!(
        engine.health().snapshot().reducer_eviction_stalled > 0,
        "the capacity backstop's 'nothing settled left to evict' branch must bump a health \
         counter, not silently break out of its loop"
    );
}
