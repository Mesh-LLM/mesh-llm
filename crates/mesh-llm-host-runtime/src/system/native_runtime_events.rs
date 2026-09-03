//! Runtime-event producer wiring for the native runtime resolution/load
//! boundary (`.omo/specs/event-system.md` §8.1, plan task 9 line 271's
//! `system/native_runtime.rs`).
//!
//! One reservation per resolution attempt: `RuntimeResolutionStarted` ->
//! (optionally) `NativeLibraryLoaded` -> exactly one terminal
//! (`RuntimeResolutionCompleted` / `RuntimeResolutionFailed`). Best-effort
//! and never blocking, matching `runtime::model_lifecycle::events`'s
//! degrade-on-absent-engine / degrade-on-exhaustion contract.

use mesh_llm_runtime_event_contracts::{
    FactData, NativeRuntimeEventKind, NativeRuntimeFact, OperationId, Outcome, ReasonCode,
    RuntimeEventIngress, RuntimeFact,
};

use crate::runtime_events::engine::OperationReservation;
use crate::runtime_events::runtime_event_engine;

fn submit(reservation: &OperationReservation, kind: NativeRuntimeEventKind, data: FactData) {
    let _ =
        reservation
            .ingress()
            .try_submit(RuntimeFact::NativeRuntime(NativeRuntimeFact::with_data(
                kind, data,
            )));
}

fn synthetic_terminal() -> RuntimeFact {
    RuntimeFact::NativeRuntime(NativeRuntimeFact::with_data(
        NativeRuntimeEventKind::RuntimeResolutionFailed,
        FactData {
            outcome: Some(Outcome::Unknown),
            reason: Some(ReasonCode::TerminalNotDelivered),
            ..FactData::default()
        },
    ))
}

/// One native-runtime resolution attempt. Reserved before any discovery,
/// install, or load work begins.
pub(crate) struct NativeRuntimeResolution {
    root: Option<OperationReservation>,
}

impl NativeRuntimeResolution {
    pub(crate) fn begin() -> Self {
        let Some(engine) = runtime_event_engine() else {
            return Self { root: None };
        };
        let root = engine.reserve_root(OperationId::new(), synthetic_terminal);
        if let Some(root) = &root {
            submit(
                root,
                NativeRuntimeEventKind::RuntimeResolutionStarted,
                FactData::default(),
            );
        }
        Self { root }
    }

    pub(crate) fn library_loaded(&self) {
        if let Some(root) = &self.root {
            submit(
                root,
                NativeRuntimeEventKind::NativeLibraryLoaded,
                FactData::default(),
            );
        }
    }

    /// §8.1 `runtime initialized` -- call once the loaded library is fully
    /// set up and ready to serve (after the runtime-scoped event reporter
    /// install attempt, win or lose). StateTransition class.
    pub(crate) fn initialized(&self) {
        if let Some(root) = &self.root {
            submit(
                root,
                NativeRuntimeEventKind::RuntimeInitialized,
                FactData::default(),
            );
        }
    }

    /// No compatible native library could be found at all (as opposed to
    /// [`Self::not_needed`], where one was already loaded and no
    /// resolution work happened). Resolves with a real
    /// `RuntimeResolutionFailed` terminal, not a no-op release.
    pub(crate) fn unavailable(mut self, reason: ReasonCode) {
        if let Some(root) = &self.root {
            submit(
                root,
                NativeRuntimeEventKind::NativeLibraryUnavailable,
                FactData::default(),
            );
        }
        if let Some(root) = self.root.take() {
            submit(
                &root,
                NativeRuntimeEventKind::RuntimeResolutionFailed,
                FactData {
                    outcome: Some(Outcome::Failure),
                    reason: Some(reason),
                    ..FactData::default()
                },
            );
        }
    }

    /// A compatible runtime is already loaded, so no resolution work
    /// actually happened: release without a terminal rather than reporting
    /// a resolution outcome for work that never ran.
    pub(crate) fn not_needed(mut self) {
        if let Some(root) = self.root.take() {
            root.cancel();
        }
    }

    pub(crate) fn completed(mut self) {
        if let Some(root) = self.root.take() {
            submit(
                &root,
                NativeRuntimeEventKind::RuntimeResolutionCompleted,
                FactData {
                    outcome: Some(Outcome::Success),
                    ..FactData::default()
                },
            );
        }
    }

    pub(crate) fn failed(mut self, reason: ReasonCode) {
        if let Some(root) = self.root.take() {
            submit(
                &root,
                NativeRuntimeEventKind::RuntimeResolutionFailed,
                FactData {
                    outcome: Some(Outcome::Failure),
                    reason: Some(reason),
                    ..FactData::default()
                },
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::NativeRuntimeResolution;
    use crate::runtime_events::engine::RuntimeEventEngine;
    use crate::runtime_events::{clear_runtime_event_engine, install_runtime_event_engine};
    use mesh_llm_runtime_event_contracts::ReasonCode;

    fn install_test_engine() -> std::sync::Arc<RuntimeEventEngine> {
        clear_runtime_event_engine();
        let engine = RuntimeEventEngine::new();
        install_runtime_event_engine(engine.clone());
        engine
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn resolution_reserves_before_completing_with_one_terminal() {
        let engine = install_test_engine();
        let resolution = NativeRuntimeResolution::begin();
        assert_eq!(engine.occupied_count(), 1);
        resolution.library_loaded();
        resolution.initialized();
        resolution.completed();
        engine.drain();
        assert_eq!(engine.occupied_count(), 0);
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn unavailable_reports_a_real_terminal_not_a_no_op() {
        let engine = install_test_engine();
        let resolution = NativeRuntimeResolution::begin();
        assert_eq!(engine.occupied_count(), 1);
        resolution.unavailable(ReasonCode::MissingArtifact);
        engine.drain();
        assert_eq!(engine.occupied_count(), 0);
        let reported = engine.replay().snapshot().into_iter().any(|frame| {
            matches!(
                frame.fact.as_ref(),
                mesh_llm_runtime_event_contracts::RuntimeFact::NativeRuntime(fact)
                    if *fact.kind()
                        == mesh_llm_runtime_event_contracts::NativeRuntimeEventKind::RuntimeResolutionFailed
            )
        });
        assert!(
            reported,
            "unavailable() must submit a real terminal, not release silently like not_needed()"
        );
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn resolution_failure_resolves_with_one_terminal() {
        let engine = install_test_engine();
        let resolution = NativeRuntimeResolution::begin();
        resolution.failed(ReasonCode::MissingArtifact);
        engine.drain();
        assert_eq!(engine.occupied_count(), 0);
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn not_needed_releases_without_a_terminal() {
        let engine = install_test_engine();
        let resolution = NativeRuntimeResolution::begin();
        assert_eq!(engine.occupied_count(), 1);
        resolution.not_needed();
        assert_eq!(engine.occupied_count(), 0);
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn dropping_without_a_terminal_synthesizes_one() {
        let engine = install_test_engine();
        {
            let resolution = NativeRuntimeResolution::begin();
            assert_eq!(engine.occupied_count(), 1);
            drop(resolution);
        }
        engine.drain();
        assert_eq!(engine.occupied_count(), 0);
        clear_runtime_event_engine();
    }

    #[test]
    #[serial_test::serial(runtime_event_engine_state)]
    fn absent_engine_never_panics() {
        clear_runtime_event_engine();
        let resolution = NativeRuntimeResolution::begin();
        resolution.library_loaded();
        resolution.completed();
    }
}
