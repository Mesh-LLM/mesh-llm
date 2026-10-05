//! `node.runtime`: native-runtime lifecycle status.

use mesh_llm_runtime_event_contracts::{
    FactData, FamilyFact, NativeRuntimeEventKind, Outcome, ReasonCode, RuntimeFact,
};

use super::super::fixtures::{apply_all, scope};
use crate::runtime_events::reducer::ReducerSnapshot;

fn native(kind: NativeRuntimeEventKind) -> RuntimeFact {
    RuntimeFact::NativeRuntime(FamilyFact::new(kind))
}

fn native_with(kind: NativeRuntimeEventKind, data: FactData) -> RuntimeFact {
    RuntimeFact::NativeRuntime(FamilyFact::with_data(kind, data))
}

fn outcome(outcome: Outcome, reason: Option<ReasonCode>) -> FactData {
    FactData {
        outcome: Some(outcome),
        reason,
        ..FactData::default()
    }
}

#[test]
fn a_fresh_reducer_reports_no_runtime_status() {
    assert!(
        ReducerSnapshot::empty()
            .domain()
            .native_runtime()
            .is_empty()
    );
}

#[test]
fn successful_resolution_ends_initialized_with_the_success_outcome() {
    let snapshot = apply_all(
        &ReducerSnapshot::empty(),
        scope(),
        0,
        vec![
            native(NativeRuntimeEventKind::RuntimeResolutionStarted),
            native(NativeRuntimeEventKind::NativeLibraryLoaded),
            native(NativeRuntimeEventKind::RuntimeInitialized),
            native_with(
                NativeRuntimeEventKind::RuntimeResolutionCompleted,
                outcome(Outcome::Success, None),
            ),
        ],
    );

    let runtime = snapshot.domain().native_runtime();
    assert_eq!(runtime.status, Some("initialized"));
    assert_eq!(runtime.last_outcome, Some("success"));
}

#[test]
fn resolution_completed_without_initialization_reports_resolved() {
    let snapshot = apply_all(
        &ReducerSnapshot::empty(),
        scope(),
        0,
        vec![
            native(NativeRuntimeEventKind::RuntimeResolutionStarted),
            native_with(
                NativeRuntimeEventKind::RuntimeResolutionCompleted,
                outcome(Outcome::Success, None),
            ),
        ],
    );

    assert_eq!(snapshot.domain().native_runtime().status, Some("resolved"));
}

#[test]
fn abi_failure_marks_the_runtime_incompatible_with_its_reason() {
    let snapshot = apply_all(
        &ReducerSnapshot::empty(),
        scope(),
        0,
        vec![
            native(NativeRuntimeEventKind::NativeLibraryLoaded),
            native_with(
                NativeRuntimeEventKind::AbiFeatureCompatibilityFailed,
                FactData {
                    reason: Some(ReasonCode::IncompatibleAbiOrFeatureSet),
                    ..FactData::default()
                },
            ),
        ],
    );

    let runtime = snapshot.domain().native_runtime();
    assert_eq!(runtime.status, Some("abi_incompatible"));
    assert_eq!(runtime.abi_compatible, Some(false));
    assert_eq!(
        runtime.last_reason_code.as_deref(),
        Some("incompatible_abi_or_feature_set")
    );
}

#[test]
fn unavailable_library_reports_unavailable_with_the_failure_reason() {
    let snapshot = apply_all(
        &ReducerSnapshot::empty(),
        scope(),
        0,
        vec![
            native(NativeRuntimeEventKind::RuntimeResolutionStarted),
            native(NativeRuntimeEventKind::NativeLibraryUnavailable),
            native_with(
                NativeRuntimeEventKind::RuntimeResolutionFailed,
                outcome(Outcome::Failure, Some(ReasonCode::MissingArtifact)),
            ),
        ],
    );

    let runtime = snapshot.domain().native_runtime();
    assert_eq!(runtime.status, Some("unavailable"));
    assert_eq!(runtime.last_outcome, Some("failure"));
    assert_eq!(
        runtime.last_reason_code.as_deref(),
        Some("missing_artifact")
    );
}

#[test]
fn a_synthesized_undelivered_terminal_does_not_overwrite_the_status() {
    let snapshot = apply_all(
        &ReducerSnapshot::empty(),
        scope(),
        0,
        vec![
            native(NativeRuntimeEventKind::RuntimeInitialized),
            native_with(
                NativeRuntimeEventKind::RuntimeResolutionFailed,
                outcome(Outcome::Unknown, Some(ReasonCode::TerminalNotDelivered)),
            ),
        ],
    );

    let runtime = snapshot.domain().native_runtime();
    assert_eq!(runtime.status, Some("initialized"));
    assert_eq!(
        runtime.last_reason_code.as_deref(),
        Some("terminal_not_delivered")
    );
}
