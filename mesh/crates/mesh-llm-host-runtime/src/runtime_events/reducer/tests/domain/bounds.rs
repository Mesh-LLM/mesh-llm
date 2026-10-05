//! Cross-category invariants: operations-map bounds under settled churn,
//! in-flight survival across the capacity sweep, rebuild continuity, and
//! deterministic category output order.

use mesh_llm_runtime_event_contracts::{
    ModelAvailabilityEventKind, OperationScope, Outcome, RequestEventKind, ResourceHealthEventKind,
    RuntimeFact, SessionEventKind, StageTopologyEventKind,
};

use super::super::fixtures::{
    device_fact, input, model_fact, model_load_phase_fact, request_fact, scope as root,
    session_fact, stage_fact, terminal_fact,
};
use crate::runtime_events::config::RESERVATION_TABLE_CAPACITY;
use crate::runtime_events::reducer::{
    ReduceOutcome, ReducerSnapshot, apply,
    rebuild::{self, RebuildOutcome},
};

#[test]
fn settled_operations_are_evicted_to_stay_within_reservation_capacity() {
    let mut snapshot = ReducerSnapshot::empty();
    for sequence in 0..10_000u64 {
        let scope = root();
        let fact = terminal_fact(Outcome::Success);
        let ReduceOutcome::Applied(next) = apply(&snapshot, input(scope, sequence, fact)) else {
            panic!("every distinct root scope must apply");
        };
        snapshot = next;
    }
    assert!(
        snapshot.operation_count() <= RESERVATION_TABLE_CAPACITY,
        "operations map must never exceed the reservation table capacity ({RESERVATION_TABLE_CAPACITY}), got {}",
        snapshot.operation_count()
    );
}

#[test]
fn rebuild_preserves_last_valid_domain_state() {
    let snapshot = ReducerSnapshot::empty();
    let ReduceOutcome::Applied(snapshot) = apply(
        &snapshot,
        input(
            root(),
            0,
            model_fact(ModelAvailabilityEventKind::ModelAvailable, "durable-model"),
        ),
    ) else {
        panic!("model_available must apply");
    };

    let RebuildOutcome::Rebuilt(rebuilt) = rebuild::rebuild(&snapshot, 1) else {
        panic!("rebuild to a higher generation must succeed");
    };
    assert!(
        rebuilt
            .domain()
            .models()
            .iter()
            .any(|model| model.id == "durable-model"),
        "domain state must survive a rebuild, not be discarded"
    );
}

/// Task 6 defect C (verifier follow-up, `.omo/plans/event-system-fixes.md`):
/// the missing safety-critical invariant -- an UNSETTLED (in-flight)
/// operation is never evicted, even once the map is driven well past
/// `RESERVATION_TABLE_CAPACITY` with settled churn. A known subset of
/// scopes receives a non-terminal fact (tracked, but never settled) and
/// is held open across the whole run; the size-triggered capacity sweep
/// must skip them every time and evict progressively newer SETTLED
/// entries instead.
#[test]
fn unsettled_operations_survive_the_capacity_sweep_while_settled_ones_are_evicted() {
    let mut snapshot = ReducerSnapshot::empty();
    let mut sequence = 0u64;

    let in_flight: Vec<OperationScope> = (0..8).map(|_| root()).collect();
    for &scope in &in_flight {
        let ReduceOutcome::Applied(next) = apply(
            &snapshot,
            input(
                scope,
                sequence,
                model_load_phase_fact("in-flight-model", "loading"),
            ),
        ) else {
            panic!("a non-terminal fact must apply for an in-flight scope");
        };
        snapshot = next;
        sequence += 1;
    }

    let first_settled_scope = root();
    let settled_total = RESERVATION_TABLE_CAPACITY + 200;
    for index in 0..settled_total {
        let scope = if index == 0 {
            first_settled_scope
        } else {
            root()
        };
        let ReduceOutcome::Applied(next) = apply(
            &snapshot,
            input(scope, sequence, terminal_fact(Outcome::Success)),
        ) else {
            panic!("every distinct settled root scope must apply");
        };
        snapshot = next;
        sequence += 1;
    }

    assert!(
        snapshot.operation_count() <= RESERVATION_TABLE_CAPACITY,
        "operations map must stay bounded even with an in-flight subset held open, got {}",
        snapshot.operation_count()
    );
    for &scope in &in_flight {
        let state = snapshot.operation(scope).unwrap_or_else(|| {
            panic!("an in-flight (unsettled) operation must never be evicted by the capacity sweep")
        });
        assert!(
            !state.settled,
            "the in-flight scope must still report unsettled"
        );
    }
    assert!(
        snapshot.operation(first_settled_scope).is_none(),
        "the earliest-settled operation must be evicted once the map exceeds capacity"
    );
}

/// Task 6 defect D (verifier follow-up, `.omo/plans/event-system-fixes.md`):
/// `models()`/`stages()`/`requests()`/`devices()` used to return
/// `HashMap` iteration order (nondeterministic per process). The
/// `*_order` `VecDeque`s `DomainState` already maintains for bounded
/// eviction must ALSO drive output order, so repeated reads of the same
/// state are stable and match insertion/touch order.
#[test]
fn category_arrays_preserve_deterministic_insertion_order_not_hashmap_order() {
    let mut snapshot = ReducerSnapshot::empty();
    let facts: Vec<RuntimeFact> = vec![
        model_fact(ModelAvailabilityEventKind::ModelAvailable, "model-a"),
        model_fact(ModelAvailabilityEventKind::ModelAvailable, "model-b"),
        model_fact(ModelAvailabilityEventKind::ModelAvailable, "model-c"),
        model_fact(ModelAvailabilityEventKind::ModelAvailable, "model-d"),
        model_fact(ModelAvailabilityEventKind::ModelAvailable, "model-e"),
        stage_fact(StageTopologyEventKind::StageReady, "stage-a", 0),
        stage_fact(StageTopologyEventKind::StageReady, "stage-b", 1),
        stage_fact(StageTopologyEventKind::StageReady, "stage-c", 2),
        stage_fact(StageTopologyEventKind::StageReady, "stage-d", 3),
        stage_fact(StageTopologyEventKind::StageReady, "stage-e", 4),
        request_fact(RequestEventKind::RequestReceived, "req-a"),
        request_fact(RequestEventKind::RequestReceived, "req-b"),
        request_fact(RequestEventKind::RequestReceived, "req-c"),
        request_fact(RequestEventKind::RequestReceived, "req-d"),
        request_fact(RequestEventKind::RequestReceived, "req-e"),
        device_fact(ResourceHealthEventKind::DeviceReady, "device-a"),
        device_fact(ResourceHealthEventKind::DeviceReady, "device-b"),
        device_fact(ResourceHealthEventKind::DeviceReady, "device-c"),
        device_fact(ResourceHealthEventKind::DeviceReady, "device-d"),
        device_fact(ResourceHealthEventKind::DeviceReady, "device-e"),
        // `sessions.recent` is a `VecDeque` already (not a `HashMap`), but
        // the brief for this defect calls it out explicitly ("must be
        // deterministic too -- check it"), so it is proven here rather than
        // left as an unstated assumption.
        session_fact(SessionEventKind::SessionCreated, "sess-a"),
        session_fact(SessionEventKind::SessionClosed, "sess-a"),
        session_fact(SessionEventKind::SessionCreated, "sess-b"),
        session_fact(SessionEventKind::SessionClosed, "sess-b"),
        session_fact(SessionEventKind::SessionCreated, "sess-c"),
        session_fact(SessionEventKind::SessionClosed, "sess-c"),
        session_fact(SessionEventKind::SessionCreated, "sess-d"),
        session_fact(SessionEventKind::SessionClosed, "sess-d"),
        session_fact(SessionEventKind::SessionCreated, "sess-e"),
        session_fact(SessionEventKind::SessionClosed, "sess-e"),
    ];
    for (sequence, fact) in facts.into_iter().enumerate() {
        let ReduceOutcome::Applied(next) = apply(&snapshot, input(root(), sequence as u64, fact))
        else {
            panic!("every distinct-category fact must apply");
        };
        snapshot = next;
    }

    assert_eq!(
        snapshot
            .domain()
            .models()
            .iter()
            .map(|model| model.id.clone())
            .collect::<Vec<_>>(),
        vec!["model-a", "model-b", "model-c", "model-d", "model-e"],
        "state.models must preserve touch order, not HashMap iteration order"
    );
    assert_eq!(
        snapshot
            .domain()
            .stages()
            .iter()
            .map(|stage| stage.id.clone())
            .collect::<Vec<_>>(),
        vec!["stage-a", "stage-b", "stage-c", "stage-d", "stage-e"],
        "state.stages must preserve touch order, not HashMap iteration order"
    );
    assert_eq!(
        snapshot
            .domain()
            .requests()
            .iter()
            .map(|request| request.id.clone())
            .collect::<Vec<_>>(),
        vec!["req-a", "req-b", "req-c", "req-d", "req-e"],
        "state.requests must preserve touch order, not HashMap iteration order"
    );
    assert_eq!(
        snapshot
            .domain()
            .devices()
            .iter()
            .map(|device| device.id.clone())
            .collect::<Vec<_>>(),
        vec!["device-a", "device-b", "device-c", "device-d", "device-e"],
        "state.devices must preserve touch order, not HashMap iteration order"
    );
    assert_eq!(
        snapshot
            .domain()
            .sessions_recent()
            .iter()
            .map(|entry| entry.id.clone())
            .collect::<Vec<_>>(),
        vec!["sess-a", "sess-b", "sess-c", "sess-d", "sess-e"],
        "sessions.recent must preserve closure order (it already does, via a \
         VecDeque -- proven here rather than left unstated)"
    );
}
