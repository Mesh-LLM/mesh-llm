use mesh_llm_runtime_event_contracts::{FamilyFact, NativeRuntimeEventKind, Outcome, RuntimeFact};

use super::fixtures::{input, input_with_native, progress_fact, scope, terminal_fact};
use crate::runtime_events::reducer::{ReduceOutcome, ReducerSnapshot, RejectReason, apply};

fn native_fact() -> RuntimeFact {
    RuntimeFact::NativeRuntime(FamilyFact::new(NativeRuntimeEventKind::RuntimeStopped))
}

#[test]
fn duplicate_fact_is_rejected_not_applied_twice() {
    let scope = scope();
    let snapshot = ReducerSnapshot::empty();
    let ReduceOutcome::Applied(snapshot) = apply(&snapshot, input(scope, 0, progress_fact(10)))
    else {
        panic!("first apply must succeed");
    };

    let outcome = apply(&snapshot, input(scope, 0, progress_fact(10)));
    assert!(matches!(
        outcome,
        ReduceOutcome::Rejected(RejectReason::Duplicate)
    ));
}

#[test]
fn stale_progress_does_not_regress_published_state() {
    let scope = scope();
    let snapshot = ReducerSnapshot::empty();
    let ReduceOutcome::Applied(snapshot) = apply(&snapshot, input(scope, 0, progress_fact(80)))
    else {
        panic!("first apply must succeed");
    };

    let outcome = apply(&snapshot, input(scope, 1, progress_fact(50)));
    assert!(matches!(
        outcome,
        ReduceOutcome::Rejected(RejectReason::StaleProgress)
    ));
    assert_eq!(
        snapshot
            .operation(scope)
            .expect("operation state")
            .last_progress_current,
        Some(80),
        "rejected regression must leave the last-valid value untouched"
    );
}

#[test]
fn native_sequence_gap_is_flagged_without_reordering() {
    let scope = scope();
    let snapshot = ReducerSnapshot::empty();
    let ReduceOutcome::Applied(snapshot) =
        apply(&snapshot, input_with_native(scope, 0, 10, native_fact()))
    else {
        panic!("first apply must succeed");
    };

    // Native sequence jumps from 10 to 15 (a gap of 4); ingress sequence is
    // still the next in order (1), so this must still apply.
    let ReduceOutcome::Applied(snapshot) =
        apply(&snapshot, input_with_native(scope, 1, 15, native_fact()))
    else {
        panic!("gapped native sequence must still be accepted");
    };

    let state = snapshot.operation(scope).expect("operation state");
    assert_eq!(state.native_gap_count, 1);
    assert_eq!(
        state.last_native_sequence,
        Some(15),
        "ordering stays ingress-sequence order; native sequence is data only"
    );
}

#[test]
fn mixed_native_and_rust_facts_apply_in_ingress_sequence_order() {
    let scope = scope();
    let snapshot = ReducerSnapshot::empty();
    let ReduceOutcome::Applied(snapshot) = apply(&snapshot, input(scope, 0, native_fact())) else {
        panic!("native-origin fact must apply first");
    };
    let ReduceOutcome::Applied(snapshot) = apply(&snapshot, input(scope, 1, progress_fact(20)))
    else {
        panic!("rust-origin fact must apply second, by ingress sequence");
    };

    let state = snapshot.operation(scope).expect("operation state");
    assert_eq!(state.last_ingress_sequence, 1);
    assert_eq!(state.last_progress_current, Some(20));

    // Same two facts, sequence numbers swapped: a fact bearing the LOWER
    // ingress sequence but arriving SECOND is stale, proving order is
    // ingress-sequence only, never arrival order or origin.
    let out_of_order = apply(&snapshot, input(scope, 0, terminal_fact(Outcome::Success)));
    assert!(matches!(
        out_of_order,
        ReduceOutcome::Rejected(RejectReason::Duplicate)
    ));
}

#[test]
fn wall_clock_skew_never_changes_ordering_or_acceptance() {
    let scope = scope();
    let snapshot = ReducerSnapshot::empty();
    let mut early = input(scope, 0, progress_fact(10));
    early.wall_clock_hint = Some(1_000);
    let ReduceOutcome::Applied(snapshot) = apply(&snapshot, early) else {
        panic!("first apply must succeed");
    };

    // Wall clock goes BACKWARDS relative to the previous fact, but the
    // ingress sequence still advances normally; acceptance must be
    // unaffected because wall_clock_hint is inert data, never consulted.
    let mut later = input(scope, 1, progress_fact(20));
    later.wall_clock_hint = Some(500);
    let outcome = apply(&snapshot, later);
    assert!(matches!(outcome, ReduceOutcome::Applied(_)));
}
