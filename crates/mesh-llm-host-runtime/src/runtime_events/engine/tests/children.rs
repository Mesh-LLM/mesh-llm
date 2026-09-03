use mesh_llm_runtime_event_contracts::{
    ChildOperationId, OperationId, RuntimeEventIngress, SubmitOutcome,
};

use super::fixtures::{synthetic_unknown, terminal_success};
use crate::runtime_events::engine::RuntimeEventEngine;

#[test]
fn outstanding_child_is_force_completed_when_the_root_releases() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let root_id = OperationId::new();
    let root = engine
        .reserve_root(root_id, synthetic_unknown)
        .expect("reserve root");
    let child = engine
        .reserve_child(root_id, ChildOperationId::new(), synthetic_unknown)
        .expect("reserve child");
    let child_ingress = child.ingress();
    std::mem::forget(child); // the child is still "outstanding" in caller code

    root.cancel();

    // The child's slot was force-released with the root; a later submit
    // through it is now late (ID/generation no longer matches).
    assert_eq!(
        child_ingress.try_submit(terminal_success()),
        SubmitOutcome::TerminalDeliveryFailed
    );
    assert!(engine.health().snapshot().terminal_delivery_failed >= 1);
}

#[test]
fn a_child_guard_that_later_drops_after_forced_release_cannot_resurrect_the_freed_slot() {
    let engine = RuntimeEventEngine::with_capacity(2);
    let root_id = OperationId::new();
    let root = engine
        .reserve_root(root_id, synthetic_unknown)
        .expect("reserve root");
    let child = engine
        .reserve_child(root_id, ChildOperationId::new(), synthetic_unknown)
        .expect("reserve child");

    root.cancel(); // force-completes and releases the child's slot too

    // A second, unrelated operation may now legitimately claim the freed
    // slot before the original child guard below ever runs its Drop.
    let unrelated = engine
        .reserve_root(OperationId::new(), synthetic_unknown)
        .expect("slot is free for reuse");

    drop(child); // the late Drop must not touch the slot `unrelated` now owns

    assert_eq!(
        unrelated.ingress().try_submit(terminal_success()),
        SubmitOutcome::Accepted,
        "a late child-guard drop must not have written a stray terminal into the reused slot"
    );
    assert_eq!(engine.drain().applied, 1);
}

#[test]
fn root_release_frees_capacity_consumed_by_its_children() {
    let engine = RuntimeEventEngine::with_capacity(2);
    let root_id = OperationId::new();
    let root = engine
        .reserve_root(root_id, synthetic_unknown)
        .expect("reserve root");
    let child = engine
        .reserve_child(root_id, ChildOperationId::new(), synthetic_unknown)
        .expect("reserve child");
    std::mem::forget(child);

    assert!(
        engine
            .reserve_root(OperationId::new(), synthetic_unknown)
            .is_none()
    );

    root.cancel();

    assert!(
        engine
            .reserve_root(OperationId::new(), synthetic_unknown)
            .is_some(),
        "releasing the root must also reclaim its outstanding child's slot"
    );
}
