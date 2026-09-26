//! Post-return reconciliation of one native model open.
//!
//! Runs on the loading thread after the final drain, before the caller
//! settles the load terminal from the native return. It only reports:
//! lost callbacks become engine health, and a terminal callback that
//! disagrees with the return becomes an `InvariantProtocolViolation`
//! diagnostic. The terminal itself is never touched here.

use std::sync::Arc;

use mesh_llm_runtime_event_contracts::{
    DiagnosticEventKind, DiagnosticFact, FactData, HumanSummary, OperationId, OperationScope,
    ReasonCode, RuntimeEventIngress, RuntimeFact,
};

use crate::inference::skippy::{ModelOpenObservation, ModelOpenReturn};
use crate::runtime_events::engine::{RuntimeEventEngine, ScopedIngress};
use crate::runtime_events::runtime_event_engine;

const CONTRADICTION_SUMMARY: &str = "native model-open observation contradicts return";

/// Where reconciliation reports: the load's own progress ingress when the
/// caller holds one, otherwise the process engine through an unreserved
/// ingress.
pub(super) struct ModelOpenReconciliation {
    pub(super) model: String,
    pub(super) ingress: Option<ScopedIngress>,
}

impl ModelOpenReconciliation {
    pub(super) fn reconcile(&self, observation: &ModelOpenObservation, returned: ModelOpenReturn) {
        let Some(engine) = self.engine() else {
            return;
        };
        report_losses(&engine, &self.model, observation);
        if let Some(contradiction) = observation.contradiction(returned) {
            tracing::warn!(
                model = %self.model,
                ?contradiction,
                last_sequence = ?observation.last_sequence,
                "{CONTRADICTION_SUMMARY}"
            );
            self.submit_contradiction(&engine);
        }
    }

    fn engine(&self) -> Option<Arc<RuntimeEventEngine>> {
        self.ingress
            .as_ref()
            .map(|ingress| Arc::clone(ingress.engine()))
            .or_else(runtime_event_engine)
    }

    fn submit_contradiction(&self, engine: &Arc<RuntimeEventEngine>) {
        let fact = contradiction_fact(&self.model);
        let _ = match self.ingress.as_ref() {
            Some(ingress) => ingress.try_submit(fact),
            None => engine
                .unreserved_ingress(OperationScope::root_only(OperationId::new()))
                .try_submit(fact),
        };
    }
}

fn report_losses(engine: &RuntimeEventEngine, model: &str, observation: &ModelOpenObservation) {
    if observation.lost() == 0 {
        return;
    }
    engine.health().bump_dropped_native_by(observation.dropped);
    engine
        .health()
        .bump_rejected_native_by(observation.rejected);
    tracing::warn!(
        model,
        dropped = observation.dropped,
        rejected = observation.rejected,
        "native model-open events were lost before reaching the host"
    );
}

fn contradiction_fact(model: &str) -> RuntimeFact {
    RuntimeFact::Diagnostic(DiagnosticFact::with_data(
        DiagnosticEventKind::InvariantProtocolViolation,
        FactData {
            reason: Some(ReasonCode::InternalRuntimeFailure),
            summary: HumanSummary::new(CONTRADICTION_SUMMARY).ok(),
            ..crate::runtime::model_lifecycle::model_scope(model)
        },
    ))
}

#[cfg(test)]
mod tests;
