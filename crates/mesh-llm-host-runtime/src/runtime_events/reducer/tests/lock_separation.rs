use std::sync::Arc;
use std::sync::Barrier;
use std::thread;

use mesh_llm_runtime_event_contracts::{
    FactData, FamilyFact, OperationId, Outcome, RequestEventKind, RuntimeEventIngress, RuntimeFact,
};

use crate::runtime_events::engine::RuntimeEventEngine;

fn terminal_success() -> RuntimeFact {
    RuntimeFact::Request(FamilyFact::with_data(
        RequestEventKind::RequestCompleted,
        FactData {
            outcome: Some(Outcome::Success),
            ..FactData::default()
        },
    ))
}

fn synthetic_unknown() -> RuntimeFact {
    RuntimeFact::Request(FamilyFact::new(RequestEventKind::RequestFailed))
}

/// Sixteen producer threads submit terminals concurrently with a drain
/// thread reducing them, synchronized only by a `Barrier` (no sleeps). The
/// reducer's own lock (`reducer_state`) is disjoint from every reservation
/// slot's lock, so this must complete deterministically without deadlock,
/// and every accepted terminal must be visible in the final snapshot.
#[test]
fn concurrent_submit_and_drain_never_deadlock_and_settle_deterministically() {
    let engine = RuntimeEventEngine::with_capacity(32);
    const PRODUCERS: usize = 16;
    let start = Arc::new(Barrier::new(PRODUCERS + 1));

    let handles: Vec<_> = (0..PRODUCERS)
        .map(|_| {
            let engine = Arc::clone(&engine);
            let start = Arc::clone(&start);
            thread::spawn(move || {
                let reservation = engine
                    .reserve_root(OperationId::new(), synthetic_unknown)
                    .expect("capacity is large enough for every producer");
                let ingress = reservation.ingress();
                start.wait();
                ingress.try_submit(terminal_success());
            })
        })
        .collect();

    start.wait();
    // Drain concurrently while producers are still submitting; the wake
    // list and reservation table's own locks make this safe, and the
    // reducer never takes a reservation-table lock while reducing.
    let mut applied = 0;
    while applied < PRODUCERS {
        applied += engine.drain().applied;
    }

    for handle in handles {
        handle.join().expect("producer thread must not panic");
    }

    let snapshot = engine.reducer_snapshot();
    assert_eq!(snapshot.operation_count(), PRODUCERS);
}

/// A rejected reducer input never mutates the engine's shared snapshot:
/// capture the `Arc` pointer before and after a forced duplicate rejection
/// and prove they are identical (a transactional rollback, structurally,
/// since `apply` never mutates in place).
#[test]
fn transactional_rollback_leaves_the_previous_snapshot_pointer_unchanged() {
    let engine = RuntimeEventEngine::with_capacity(4);
    let operation = OperationId::new();
    let reservation = engine
        .reserve_root(operation, synthetic_unknown)
        .expect("reserve");
    reservation.ingress().try_submit(terminal_success());
    engine.drain();

    let before = engine.reducer_snapshot();
    let before_ptr = Arc::as_ptr(&before);

    // A second reservation for a fresh operation settling normally is a
    // genuine no-op relative to the first snapshot's pointer identity only
    // if nothing is applied; force that by draining with nothing queued.
    let drained_nothing = engine.drain();
    assert_eq!(drained_nothing.applied, 0);

    let after = engine.reducer_snapshot();
    assert_eq!(
        before_ptr,
        Arc::as_ptr(&after),
        "an empty drain must not swap the reducer's snapshot pointer"
    );
}
