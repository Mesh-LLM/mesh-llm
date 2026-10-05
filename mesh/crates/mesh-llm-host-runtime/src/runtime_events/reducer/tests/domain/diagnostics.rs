//! `node.diagnostics`: keyed warnings, cap eviction, degraded toggle,
//! pinned fatal, and counters.

use mesh_llm_runtime_event_contracts::{
    DeviceId, DiagnosticEventKind, FactData, FamilyFact, HumanSummary, Outcome, ReasonCode,
    RuntimeFact, ScopeIdentities, UnknownReasonCode,
};

use super::super::fixtures::apply_each_on_fresh_root;
use crate::runtime_events::reducer::{ACTIVE_WARNING_BOUND, ReducerSnapshot};

fn diagnostic(kind: DiagnosticEventKind, data: FactData) -> RuntimeFact {
    RuntimeFact::Diagnostic(FamilyFact::with_data(kind, data))
}

fn warning_on(device: &str, reason: &str) -> FactData {
    FactData {
        scope: ScopeIdentities {
            device_id: Some(DeviceId::new(device).expect("valid device id")),
            ..ScopeIdentities::default()
        },
        reason: Some(ReasonCode::Unknown(
            UnknownReasonCode::new(reason).expect("valid reason"),
        )),
        ..FactData::default()
    }
}

fn with_summary(data: FactData, summary: &str) -> FactData {
    FactData {
        summary: Some(HumanSummary::new(summary).expect("valid summary")),
        ..data
    }
}

#[test]
fn a_fresh_reducer_reports_no_diagnostics() {
    assert!(ReducerSnapshot::empty().domain().diagnostics().is_empty());
}

#[test]
fn raising_the_same_warning_key_updates_it_in_place() {
    let snapshot = apply_each_on_fresh_root(vec![
        diagnostic(
            DiagnosticEventKind::WarningRaised,
            with_summary(warning_on("gpu-0", "thermal"), "first"),
        ),
        diagnostic(
            DiagnosticEventKind::WarningRaised,
            with_summary(warning_on("gpu-0", "thermal"), "second"),
        ),
    ]);

    let warnings = &snapshot.domain().diagnostics().active_warnings;
    assert_eq!(warnings.len(), 1);
    assert_eq!(warnings[0].summary.as_deref(), Some("second"));
    assert_eq!(warnings[0].reason_code.as_deref(), Some("thermal"));
}

#[test]
fn clearing_removes_only_the_warning_with_the_matching_key() {
    let snapshot = apply_each_on_fresh_root(vec![
        diagnostic(
            DiagnosticEventKind::WarningRaised,
            warning_on("gpu-0", "thermal"),
        ),
        diagnostic(
            DiagnosticEventKind::WarningRaised,
            warning_on("gpu-1", "thermal"),
        ),
        diagnostic(
            DiagnosticEventKind::WarningCleared,
            warning_on("gpu-0", "thermal"),
        ),
    ]);

    let keys: Vec<String> = snapshot
        .domain()
        .diagnostics()
        .active_warnings
        .iter()
        .map(|warning| warning.key.clone())
        .collect();
    assert_eq!(keys, vec!["thermal|device=gpu-1".to_string()]);
}

#[test]
fn warnings_past_the_cap_evict_the_oldest_and_count_it() {
    let overflow = 2;
    let facts = (0..ACTIVE_WARNING_BOUND + overflow)
        .map(|index| {
            diagnostic(
                DiagnosticEventKind::WarningRaised,
                warning_on(&format!("gpu-{index}"), "thermal"),
            )
        })
        .collect();

    let snapshot = apply_each_on_fresh_root(facts);

    let diagnostics = snapshot.domain().diagnostics();
    assert_eq!(diagnostics.active_warnings.len(), ACTIVE_WARNING_BOUND);
    assert_eq!(diagnostics.evicted_warnings, overflow as u64);
    assert_eq!(
        diagnostics.active_warnings[0].key, "thermal|device=gpu-2",
        "the two oldest warnings are the ones evicted"
    );
}

#[test]
fn degraded_toggles_on_enter_and_off_on_exit() {
    let entered = apply_each_on_fresh_root(vec![diagnostic(
        DiagnosticEventKind::DegradedOperationEntered,
        FactData::default(),
    )]);
    assert!(entered.domain().diagnostics().degraded);

    let exited = apply_each_on_fresh_root(vec![
        diagnostic(
            DiagnosticEventKind::DegradedOperationEntered,
            FactData::default(),
        ),
        diagnostic(
            DiagnosticEventKind::DegradedOperationExited,
            FactData::default(),
        ),
    ]);
    assert!(!exited.domain().diagnostics().degraded);
}

#[test]
fn the_first_fatal_is_pinned_and_synthesized_fatals_are_ignored() {
    let fatal = |reason: ReasonCode, outcome: Outcome| {
        diagnostic(
            DiagnosticEventKind::FatalNativeFailure,
            FactData {
                outcome: Some(outcome),
                reason: Some(reason),
                ..FactData::default()
            },
        )
    };
    let snapshot = apply_each_on_fresh_root(vec![
        fatal(ReasonCode::TerminalNotDelivered, Outcome::Unknown),
        fatal(ReasonCode::InternalRuntimeFailure, Outcome::Failure),
        fatal(ReasonCode::OutOfMemory, Outcome::Failure),
    ]);

    let pinned = snapshot
        .domain()
        .diagnostics()
        .fatal
        .clone()
        .expect("a real fatal is pinned");
    assert_eq!(
        pinned.reason_code.as_deref(),
        Some("internal_runtime_failure")
    );
}

#[test]
fn failure_fallback_and_invariant_counters_increment() {
    let snapshot = apply_each_on_fresh_root(vec![
        diagnostic(
            DiagnosticEventKind::RecoverableNativeFailure,
            FactData::default(),
        ),
        diagnostic(
            DiagnosticEventKind::RecoverableNativeFailure,
            FactData::default(),
        ),
        diagnostic(DiagnosticEventKind::FallbackApplied, FactData::default()),
        diagnostic(
            DiagnosticEventKind::InvariantProtocolViolation,
            FactData::default(),
        ),
    ]);

    let diagnostics = snapshot.domain().diagnostics();
    assert_eq!(diagnostics.recoverable_failures, 2);
    assert_eq!(diagnostics.fallbacks, 1);
    assert_eq!(diagnostics.invariant_violations, 1);
}
