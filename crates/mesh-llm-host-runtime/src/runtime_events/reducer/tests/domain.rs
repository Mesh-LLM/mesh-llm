//! Task 6 (`.omo/plans/event-system-fixes.md`, defect D6) acceptance tests:
//! the reducer's `operations` map stays bounded by the reservation table
//! capacity even under sustained settled-operation churn, and bounded
//! per-category domain state (models/stages/sessions/requests/devices/cache)
//! is actually populated from applied facts instead of discarded.

use mesh_llm_runtime_event_contracts::{
    FactData, FamilyFact, KvRuntimeStateEventKind, LogicalModelId, ModelAvailabilityEventKind,
    ModelUnloadingEventKind, OperationId, OperationScope, Outcome, RequestEventKind, RequestId,
    ResourceHealthEventKind, RuntimeFact, ScopeIdentities, SessionEventKind, SessionId,
    StageIdentity, StageTopologyEventKind, StateName, StateTransition,
};

use super::fixtures::{input, terminal_fact};
use crate::runtime_events::config::{REQUEST_ROOT_BOUND, RESERVATION_TABLE_CAPACITY};
use crate::runtime_events::reducer::{
    ReduceOutcome, ReducerSnapshot, apply,
    rebuild::{self, RebuildOutcome},
};

fn root() -> OperationScope {
    OperationScope::root_only(OperationId::new())
}

fn model_fact(kind: ModelAvailabilityEventKind, model_id: &str) -> RuntimeFact {
    RuntimeFact::ModelAvailability(FamilyFact::with_data(
        kind,
        FactData {
            scope: ScopeIdentities {
                model_id: Some(LogicalModelId::new(model_id).expect("valid model id")),
                ..ScopeIdentities::default()
            },
            ..FactData::default()
        },
    ))
}

fn unload_fact(kind: ModelUnloadingEventKind, model_id: &str) -> RuntimeFact {
    RuntimeFact::ModelUnloading(FamilyFact::with_data(
        kind,
        FactData {
            scope: ScopeIdentities {
                model_id: Some(LogicalModelId::new(model_id).expect("valid model id")),
                ..ScopeIdentities::default()
            },
            ..FactData::default()
        },
    ))
}

fn stage_fact(kind: StageTopologyEventKind, stage_id: &str, index: u32) -> RuntimeFact {
    RuntimeFact::StageTopology(FamilyFact::with_data(
        kind,
        FactData {
            scope: ScopeIdentities {
                stage: Some(StageIdentity::new(
                    mesh_llm_runtime_event_contracts::StageId::new(stage_id)
                        .expect("valid stage id"),
                    index,
                )),
                ..ScopeIdentities::default()
            },
            ..FactData::default()
        },
    ))
}

fn session_fact(kind: SessionEventKind, session_id: &str) -> RuntimeFact {
    RuntimeFact::Session(FamilyFact::with_data(
        kind,
        FactData {
            scope: ScopeIdentities {
                session_id: Some(SessionId::new(session_id).expect("valid session id")),
                ..ScopeIdentities::default()
            },
            ..FactData::default()
        },
    ))
}

fn request_fact(kind: RequestEventKind, request_id: &str) -> RuntimeFact {
    RuntimeFact::Request(FamilyFact::with_data(
        kind,
        FactData {
            scope: ScopeIdentities {
                request_id: Some(RequestId::new(request_id).expect("valid request id")),
                ..ScopeIdentities::default()
            },
            ..FactData::default()
        },
    ))
}

fn device_fact(kind: ResourceHealthEventKind, device_id: &str) -> RuntimeFact {
    RuntimeFact::ResourceHealth(FamilyFact::with_data(
        kind,
        FactData {
            scope: ScopeIdentities {
                device_id: Some(
                    mesh_llm_runtime_event_contracts::DeviceId::new(device_id)
                        .expect("valid device id"),
                ),
                ..ScopeIdentities::default()
            },
            ..FactData::default()
        },
    ))
}

fn cache_fact(kind: KvRuntimeStateEventKind) -> RuntimeFact {
    RuntimeFact::KvRuntimeState(FamilyFact::new(kind))
}

fn model_load_phase_fact(model_id: &str, phase: &str) -> RuntimeFact {
    RuntimeFact::ModelLoading(FamilyFact::with_data(
        mesh_llm_runtime_event_contracts::ModelLoadingEventKind::ModelLoadPhaseChanged,
        FactData {
            scope: ScopeIdentities {
                model_id: Some(LogicalModelId::new(model_id).expect("valid model id")),
                ..ScopeIdentities::default()
            },
            state: Some(StateTransition::new(
                None,
                StateName::new(phase).expect("valid state name"),
            )),
            ..FactData::default()
        },
    ))
}

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

#[test]
fn stages_track_latest_topology_state() {
    let snapshot = ReducerSnapshot::empty();
    let ReduceOutcome::Applied(snapshot) = apply(
        &snapshot,
        input(
            root(),
            0,
            stage_fact(StageTopologyEventKind::StageReady, "stage-0", 0),
        ),
    ) else {
        panic!("stage_ready must apply");
    };
    let stage = snapshot
        .domain()
        .stages()
        .into_iter()
        .find(|stage| stage.id == "stage-0")
        .expect("stage must be tracked in state.stages");
    assert_eq!(stage.state.as_deref(), Some("ready"));
}

#[test]
fn sessions_track_active_count_and_bounded_recent() {
    let snapshot = ReducerSnapshot::empty();
    let ReduceOutcome::Applied(snapshot) = apply(
        &snapshot,
        input(
            root(),
            0,
            session_fact(SessionEventKind::SessionCreated, "sess-1"),
        ),
    ) else {
        panic!("session_created must apply");
    };
    assert_eq!(snapshot.domain().sessions_active_count(), 1);

    let ReduceOutcome::Applied(snapshot) = apply(
        &snapshot,
        input(
            root(),
            1,
            session_fact(SessionEventKind::SessionClosed, "sess-1"),
        ),
    ) else {
        panic!("session_closed must apply");
    };
    assert_eq!(
        snapshot.domain().sessions_active_count(),
        0,
        "a closed session must leave the active count"
    );
    let recent = snapshot.domain().sessions_recent();
    assert!(
        recent
            .iter()
            .any(|entry| entry.id == "sess-1" && entry.state == "closed"),
        "a closed session must appear in the bounded recent list"
    );
}

#[test]
fn requests_are_in_flight_only_and_bounded() {
    let mut snapshot = ReducerSnapshot::empty();
    for index in 0..(REQUEST_ROOT_BOUND + 50) {
        let request_id = format!("req-{index}");
        let ReduceOutcome::Applied(next) = apply(
            &snapshot,
            input(
                root(),
                index as u64,
                request_fact(RequestEventKind::RequestReceived, &request_id),
            ),
        ) else {
            panic!("request_received must apply for {request_id}");
        };
        snapshot = next;
    }
    assert!(
        snapshot.domain().requests().len() <= REQUEST_ROOT_BOUND,
        "in-flight requests must stay bounded by REQUEST_ROOT_BOUND"
    );

    let ReduceOutcome::Applied(snapshot) = apply(
        &snapshot,
        input(
            root(),
            (REQUEST_ROOT_BOUND + 50) as u64,
            request_fact(RequestEventKind::RequestCompleted, "req-0"),
        ),
    ) else {
        panic!("request_completed must apply even for an evicted-from-domain request");
    };
    assert!(
        !snapshot
            .domain()
            .requests()
            .iter()
            .any(|request| request.id == "req-0"),
        "a completed request must never appear as in-flight"
    );
}

#[test]
fn devices_track_the_latest_resource_health_state() {
    let snapshot = ReducerSnapshot::empty();
    let ReduceOutcome::Applied(snapshot) = apply(
        &snapshot,
        input(
            root(),
            0,
            device_fact(ResourceHealthEventKind::DeviceReady, "gpu-0"),
        ),
    ) else {
        panic!("device_ready must apply");
    };
    let ReduceOutcome::Applied(snapshot) = apply(
        &snapshot,
        input(
            root(),
            1,
            device_fact(ResourceHealthEventKind::DeviceDegraded, "gpu-0"),
        ),
    ) else {
        panic!("device_degraded must apply");
    };
    let device = snapshot
        .domain()
        .devices()
        .into_iter()
        .find(|device| device.id == "gpu-0")
        .expect("device must be tracked in state.devices");
    assert_eq!(device.state.as_deref(), Some("degraded"));
}

#[test]
fn cache_tracks_the_latest_pressure_and_capacity_signal() {
    let snapshot = ReducerSnapshot::empty();
    let ReduceOutcome::Applied(snapshot) = apply(
        &snapshot,
        input(
            root(),
            0,
            cache_fact(KvRuntimeStateEventKind::CachePressureCrossed),
        ),
    ) else {
        panic!("cache_pressure_crossed must apply");
    };
    let ReduceOutcome::Applied(snapshot) = apply(
        &snapshot,
        input(
            root(),
            1,
            cache_fact(KvRuntimeStateEventKind::ContextExhausted),
        ),
    ) else {
        panic!("context_exhausted must apply");
    };
    let cache = snapshot.domain().cache();
    assert_eq!(cache.pressure.as_deref(), Some("pressure"));
    assert_eq!(cache.capacity_state.as_deref(), Some("exhausted"));
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
