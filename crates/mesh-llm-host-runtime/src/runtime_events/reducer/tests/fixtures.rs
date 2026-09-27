use std::sync::Arc;

use mesh_llm_runtime_event_contracts::{
    DeviceId, FactData, FamilyFact, KvRuntimeStateEventKind, LogicalModelId,
    ModelAvailabilityEventKind, ModelLoadingEventKind, ModelUnloadingEventKind,
    NativeRuntimeEventKind, NumericSummary, NumericSummaryKey, NumericValue, OperationId,
    OperationScope, Outcome, Progress, ProgressUnit, ReasonCode, RequestEventKind, RequestId,
    ResourceHealthEventKind, RuntimeFact, ScopeIdentities, SessionEventKind, SessionId, StageId,
    StageIdentity, StageTopologyEventKind, StateName, StateTransition,
};

use crate::runtime_events::reducer::{ReduceOutcome, ReducerInput, ReducerSnapshot, apply};

/// Drive `facts` through the reducer, each on `scope`, starting at ingress
/// sequence `first_sequence`. Panics if any fact is rejected: every domain
/// test feeds a well-formed sequence.
pub fn apply_all(
    snapshot: &Arc<ReducerSnapshot>,
    scope: OperationScope,
    first_sequence: u64,
    facts: Vec<RuntimeFact>,
) -> Arc<ReducerSnapshot> {
    let mut current = Arc::clone(snapshot);
    for (offset, fact) in (first_sequence..).zip(facts) {
        let ReduceOutcome::Applied(next) = apply(&current, input(scope, offset, fact)) else {
            panic!("fact at sequence {offset} must apply");
        };
        current = next;
    }
    current
}

/// Each fact on its own fresh root scope -- the shape node-wide producers
/// (`unreserved_ingress` with a fresh `OperationId`) actually submit.
pub fn apply_each_on_fresh_root(facts: Vec<RuntimeFact>) -> Arc<ReducerSnapshot> {
    let mut current = ReducerSnapshot::empty();
    for (sequence, fact) in (0u64..).zip(facts) {
        current = apply_all(&current, scope(), sequence, vec![fact]);
    }
    current
}

pub fn summaries(
    pairs: &[(&str, u64)],
) -> mesh_llm_runtime_event_contracts::BoundedNumericSummaries {
    let values = pairs
        .iter()
        .map(|(key, value)| {
            NumericSummary::new(
                NumericSummaryKey::new(key).expect("valid summary key"),
                NumericValue::Unsigned(*value),
            )
        })
        .collect();
    mesh_llm_runtime_event_contracts::BoundedNumericSummaries::new(values)
        .expect("within summary bound")
}

pub fn request_scope(request_id: &str) -> ScopeIdentities {
    ScopeIdentities {
        request_id: Some(RequestId::new(request_id).expect("valid request id")),
        ..ScopeIdentities::default()
    }
}

pub fn tokens(current: u64) -> Option<Progress> {
    Some(Progress::new(current, None, ProgressUnit::Tokens))
}

fn model_scope(model_id: &str) -> ScopeIdentities {
    ScopeIdentities {
        model_id: Some(LogicalModelId::new(model_id).expect("valid model id")),
        ..ScopeIdentities::default()
    }
}

fn scoped(scope: ScopeIdentities) -> FactData {
    FactData {
        scope,
        ..FactData::default()
    }
}

pub fn model_fact(kind: ModelAvailabilityEventKind, model_id: &str) -> RuntimeFact {
    RuntimeFact::ModelAvailability(FamilyFact::with_data(kind, scoped(model_scope(model_id))))
}

pub fn unload_fact(kind: ModelUnloadingEventKind, model_id: &str) -> RuntimeFact {
    RuntimeFact::ModelUnloading(FamilyFact::with_data(kind, scoped(model_scope(model_id))))
}

pub fn model_load_phase_fact(model_id: &str, phase: &str) -> RuntimeFact {
    RuntimeFact::ModelLoading(FamilyFact::with_data(
        ModelLoadingEventKind::ModelLoadPhaseChanged,
        FactData {
            state: Some(StateTransition::new(
                None,
                StateName::new(phase).expect("valid state name"),
            )),
            ..scoped(model_scope(model_id))
        },
    ))
}

pub fn stage_fact(kind: StageTopologyEventKind, stage_id: &str, index: u32) -> RuntimeFact {
    let stage = StageIdentity::new(StageId::new(stage_id).expect("valid stage id"), index);
    RuntimeFact::StageTopology(FamilyFact::with_data(
        kind,
        scoped(ScopeIdentities {
            stage: Some(stage),
            ..ScopeIdentities::default()
        }),
    ))
}

pub fn session_fact(kind: SessionEventKind, session_id: &str) -> RuntimeFact {
    RuntimeFact::Session(FamilyFact::with_data(
        kind,
        scoped(ScopeIdentities {
            session_id: Some(SessionId::new(session_id).expect("valid session id")),
            ..ScopeIdentities::default()
        }),
    ))
}

pub fn request_fact(kind: RequestEventKind, request_id: &str) -> RuntimeFact {
    RuntimeFact::Request(FamilyFact::with_data(
        kind,
        scoped(request_scope(request_id)),
    ))
}

pub fn device_fact(kind: ResourceHealthEventKind, device_id: &str) -> RuntimeFact {
    RuntimeFact::ResourceHealth(FamilyFact::with_data(
        kind,
        scoped(ScopeIdentities {
            device_id: Some(DeviceId::new(device_id).expect("valid device id")),
            ..ScopeIdentities::default()
        }),
    ))
}

pub fn cache_fact(kind: KvRuntimeStateEventKind) -> RuntimeFact {
    RuntimeFact::KvRuntimeState(FamilyFact::new(kind))
}

pub fn scope() -> OperationScope {
    OperationScope::root_only(OperationId::new())
}

pub fn progress_fact(current: u64) -> RuntimeFact {
    let data = FactData {
        progress: Some(Progress::new(current, Some(100), ProgressUnit::Tokens)),
        ..FactData::default()
    };
    RuntimeFact::Request(FamilyFact::with_data(RequestEventKind::RequestQueued, data))
}

pub fn terminal_fact(outcome: Outcome) -> RuntimeFact {
    let data = FactData {
        outcome: Some(outcome),
        ..FactData::default()
    };
    RuntimeFact::Request(FamilyFact::with_data(
        RequestEventKind::RequestCompleted,
        data,
    ))
}

pub fn synthesized_terminal_fact() -> RuntimeFact {
    let data = FactData {
        outcome: Some(Outcome::Unknown),
        reason: Some(ReasonCode::TerminalNotDelivered),
        ..FactData::default()
    };
    RuntimeFact::NativeRuntime(FamilyFact::with_data(
        NativeRuntimeEventKind::RuntimeStopped,
        data,
    ))
}

/// Defaults `reserved: true` (R1 fix, task 6-fix,
/// `.omo/plans/event-system-fixes.md`): every existing reducer-level test
/// using this helper models a scope AS IF it has a live reservation
/// backing it (the pre-R1 default, and what these tests were already
/// exercising) -- use [`input_unreserved`] to explicitly exercise the
/// never-reserved bounded-LRU path instead.
pub fn input(scope: OperationScope, ingress_sequence: u64, fact: RuntimeFact) -> ReducerInput {
    ReducerInput {
        scope,
        ingress_sequence,
        native_sequence: None,
        wall_clock_hint: None,
        synthesized: false,
        reserved: true,
        fact,
    }
}

/// Same as [`input`] but `reserved: false` -- models a fact that arrived
/// through `unreserved_ingress` with no `SlotHandle` ever backing its
/// scope (R1 fix, task 6-fix).
pub fn input_unreserved(
    scope: OperationScope,
    ingress_sequence: u64,
    fact: RuntimeFact,
) -> ReducerInput {
    ReducerInput {
        reserved: false,
        ..input(scope, ingress_sequence, fact)
    }
}

pub fn input_with_native(
    scope: OperationScope,
    ingress_sequence: u64,
    native_sequence: u64,
    fact: RuntimeFact,
) -> ReducerInput {
    ReducerInput {
        native_sequence: Some(native_sequence),
        ..input(scope, ingress_sequence, fact)
    }
}
