use std::sync::Arc;

use mesh_llm_runtime_event_contracts::RuntimeFact;

use super::*;
use crate::runtime::model_lifecycle::LoadOperation;
use crate::runtime_events::{clear_runtime_event_engine, install_runtime_event_engine};

const MODEL: &str = "org/model";

fn install_engine() -> Arc<RuntimeEventEngine> {
    clear_runtime_event_engine();
    let engine = RuntimeEventEngine::new();
    install_runtime_event_engine(engine.clone());
    engine
}

fn published(engine: &RuntimeEventEngine) -> Vec<&'static str> {
    engine.drain();
    engine
        .replay()
        .snapshot()
        .iter()
        .map(|frame| frame.fact.kind_id())
        .collect()
}

fn contradiction_reasons(engine: &RuntimeEventEngine) -> Vec<Option<ReasonCode>> {
    engine.drain();
    engine
        .replay()
        .snapshot()
        .iter()
        .filter_map(|frame| match frame.fact.as_ref() {
            RuntimeFact::Diagnostic(fact)
                if *fact.kind() == DiagnosticEventKind::InvariantProtocolViolation =>
            {
                Some(fact.data().reason.clone())
            }
            _ => None,
        })
        .collect()
}

fn scoped(op: &LoadOperation) -> ModelOpenReconciliation {
    ModelOpenReconciliation {
        model: MODEL.to_string(),
        ingress: op.progress_ingress(),
    }
}

fn observed(saw_finished: bool, saw_failed_handled: bool) -> ModelOpenObservation {
    ModelOpenObservation {
        drained: 1,
        saw_finished,
        saw_failed_handled,
        last_sequence: Some(1),
        ..ModelOpenObservation::default()
    }
}

#[test]
#[serial_test::serial(runtime_event_engine_state)]
fn finished_callback_with_failed_return_submits_a_diagnostic_and_the_terminal_stays_failure() {
    let engine = install_engine();
    let op = LoadOperation::begin(MODEL);

    scoped(&op).reconcile(&observed(true, false), ModelOpenReturn::Failed);
    op.load_failed(MODEL);

    let kinds = published(&engine);
    assert_eq!(
        contradiction_reasons(&engine),
        vec![Some(ReasonCode::InternalRuntimeFailure)]
    );
    assert!(kinds.contains(&"model_load_failed"));
    assert!(!kinds.contains(&"native_model_load_completed"));
    clear_runtime_event_engine();
}

#[test]
#[serial_test::serial(runtime_event_engine_state)]
fn failed_handled_callback_with_successful_return_submits_a_diagnostic_and_the_terminal_stays_success()
 {
    let engine = install_engine();
    let op = LoadOperation::begin(MODEL);

    scoped(&op).reconcile(&observed(false, true), ModelOpenReturn::Succeeded);
    let _availability = op.native_load_completed(MODEL);

    let kinds = published(&engine);
    assert_eq!(contradiction_reasons(&engine).len(), 1);
    assert!(kinds.contains(&"native_model_load_completed"));
    assert!(!kinds.contains(&"model_load_failed"));
    clear_runtime_event_engine();
}

#[test]
#[serial_test::serial(runtime_event_engine_state)]
fn missing_terminal_callback_with_successful_return_submits_no_diagnostic() {
    let engine = install_engine();
    let op = LoadOperation::begin(MODEL);

    scoped(&op).reconcile(&observed(false, false), ModelOpenReturn::Succeeded);

    assert!(contradiction_reasons(&engine).is_empty());
    drop(op);
    clear_runtime_event_engine();
}

#[test]
#[serial_test::serial(runtime_event_engine_state)]
fn contradiction_without_a_load_reservation_uses_an_unreserved_ingress() {
    let engine = install_engine();
    let reconciliation = ModelOpenReconciliation {
        model: MODEL.to_string(),
        ingress: None,
    };

    reconciliation.reconcile(&observed(true, false), ModelOpenReturn::Failed);

    assert_eq!(contradiction_reasons(&engine).len(), 1);
    clear_runtime_event_engine();
}

#[test]
#[serial_test::serial(runtime_event_engine_state)]
fn dropped_and_rejected_records_bump_native_health() {
    let engine = install_engine();
    let reconciliation = ModelOpenReconciliation {
        model: MODEL.to_string(),
        ingress: None,
    };
    let observation = ModelOpenObservation {
        dropped: 4,
        rejected: 1,
        ..ModelOpenObservation::default()
    };

    reconciliation.reconcile(&observation, ModelOpenReturn::Succeeded);

    let health = engine.health().snapshot();
    assert_eq!(health.dropped_native, 4);
    assert_eq!(health.rejected_native, 1);
    assert!(contradiction_reasons(&engine).is_empty());
    clear_runtime_event_engine();
}
