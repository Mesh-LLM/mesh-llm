//! `models` category: availability, load phase, provisional-identity
//! supersession, and non-cascading unload.

use mesh_llm_runtime_event_contracts::{
    ModelAvailabilityEventKind, ModelUnloadingEventKind, RequestEventKind, SessionEventKind,
    StageTopologyEventKind,
};

use super::super::fixtures::{
    apply_all, input, model_fact, model_load_phase_fact, request_fact, scope as root, session_fact,
    stage_fact, unload_fact,
};
use crate::runtime_events::reducer::{ReduceOutcome, ReducerSnapshot, apply};

#[test]
fn a_loaded_model_becomes_available_and_unload_removes_it() {
    let snapshot = ReducerSnapshot::empty();
    let ReduceOutcome::Applied(snapshot) = apply(
        &snapshot,
        input(
            root(),
            0,
            model_fact(ModelAvailabilityEventKind::ModelAvailable, "demo-model"),
        ),
    ) else {
        panic!("model_available must apply");
    };

    let models = snapshot.domain().models();
    let demo = models
        .iter()
        .find(|model| model.id == "demo-model")
        .expect("model must appear in state.models after model_available");
    assert_eq!(demo.availability.as_deref(), Some("available"));

    let ReduceOutcome::Applied(snapshot) = apply(
        &snapshot,
        input(
            root(),
            1,
            unload_fact(ModelUnloadingEventKind::UnloadCompleted, "demo-model"),
        ),
    ) else {
        panic!("unload_completed must apply");
    };
    assert!(
        !snapshot
            .domain()
            .models()
            .iter()
            .any(|model| model.id == "demo-model"),
        "unload must remove the model from state.models"
    );
}

#[test]
fn model_load_phase_changed_carries_the_producer_supplied_phase_name() {
    let snapshot = ReducerSnapshot::empty();
    let ReduceOutcome::Applied(snapshot) = apply(
        &snapshot,
        input(
            root(),
            0,
            model_load_phase_fact("phase-model", "downloading_weights"),
        ),
    ) else {
        panic!("model_load_phase_changed must apply");
    };
    let model = snapshot
        .domain()
        .models()
        .into_iter()
        .find(|model| model.id == "phase-model")
        .expect("model must be tracked");
    assert_eq!(model.load_phase.as_deref(), Some("downloading_weights"));
}

/// F2 fix (event-system-fixes, live-sampling finding): a single load
/// operation's model identity legitimately changes mid-flight (a
/// pre-resolution provisional id, superseded by the resolved canonical id
/// once source resolution completes). Before this fix the provisional
/// row was an ORPHANED PHANTOM -- stuck at whatever `load_phase` its last
/// fact set, forever, because no later fact ever referenced that id again
/// to transition or evict it (reproduced live: a stale
/// `"load_phase":"loading"` row survived 15+ minutes and 3 reconnects).
/// `reconcile_model_root_identity` correlates by the fact's ROOT
/// operation (stable across the whole operation, unlike model_id) and
/// evicts the stale row the moment the SAME root reports a different id.
#[test]
fn a_root_operations_provisional_model_row_is_superseded_not_orphaned_on_resolution() {
    let snapshot = ReducerSnapshot::empty();
    let scope = root();
    let ReduceOutcome::Applied(snapshot) = apply(
        &snapshot,
        input(
            scope,
            0,
            model_load_phase_fact("pending-resolution/op-1", "loading"),
        ),
    ) else {
        panic!("the provisional (pre-resolution) fact must apply");
    };
    assert!(
        snapshot
            .domain()
            .models()
            .iter()
            .any(|model| model.id == "pending-resolution/op-1"),
        "the provisional row must exist right after the pre-resolution fact"
    );

    let ReduceOutcome::Applied(snapshot) = apply(
        &snapshot,
        input(scope, 1, model_load_phase_fact("resolved-model", "loading")),
    ) else {
        panic!("the resolved-identity fact for the SAME root must apply");
    };

    let models = snapshot.domain().models();
    assert!(
        !models
            .iter()
            .any(|model| model.id == "pending-resolution/op-1"),
        "the provisional row must be SUPERSEDED once the SAME root reports \
         its resolved identity, not left an orphaned phantom stuck in \
         \"loading\" forever"
    );
    assert!(
        models.iter().any(|model| model.id == "resolved-model"),
        "the resolved identity's own row must exist after supersession"
    );
}

#[test]
fn unload_does_not_cascade_to_sessions_requests_stages() {
    let loaded = apply_all(
        &ReducerSnapshot::empty(),
        root(),
        0,
        vec![
            model_fact(ModelAvailabilityEventKind::ModelAvailable, "unload-model"),
            stage_fact(StageTopologyEventKind::StageReady, "unload-stage", 0),
            session_fact(SessionEventKind::SessionActive, "unload-session"),
        ],
    );
    let loaded = apply_all(
        &loaded,
        root(),
        3,
        vec![request_fact(
            RequestEventKind::RequestReceived,
            "unload-request",
        )],
    );

    let unloaded = apply_all(
        &loaded,
        root(),
        4,
        vec![unload_fact(
            ModelUnloadingEventKind::UnloadCompleted,
            "unload-model",
        )],
    );

    let domain = unloaded.domain();
    assert!(domain.models().is_empty(), "unload removes the model row");
    assert_eq!(
        domain.stages().len(),
        1,
        "stage rows are owned by topology facts"
    );
    assert_eq!(
        domain.sessions_active_count(),
        1,
        "sessions are owned by session facts"
    );
    assert_eq!(
        domain.requests().len(),
        1,
        "requests are owned by request facts"
    );
}
