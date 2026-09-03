use mesh_llm_runtime_event_contracts::{
    FactData, FamilyFact, NativeRuntimeEventKind, OperationId, OperationScope, Outcome, Progress,
    ProgressUnit, ReasonCode, RequestEventKind, RuntimeFact,
};

use crate::runtime_events::reducer::ReducerInput;

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

pub fn input(scope: OperationScope, ingress_sequence: u64, fact: RuntimeFact) -> ReducerInput {
    ReducerInput {
        scope,
        ingress_sequence,
        native_sequence: None,
        wall_clock_hint: None,
        synthesized: false,
        fact,
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
