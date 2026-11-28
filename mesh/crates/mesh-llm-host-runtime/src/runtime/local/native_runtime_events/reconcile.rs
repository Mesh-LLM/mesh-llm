//! Post-return reconciliation of one native model open.
//!
//! Runs on Skippy's drain thread after the final drain. Skippy joins that
//! thread before the load returns, so this is done before the caller
//! settles the load terminal from the native return. It only reports:
//! lost callbacks become engine health, and a terminal callback that
//! disagrees with the return becomes an `InvariantProtocolViolation`
//! diagnostic. When Skippy delivered no summary, the load's events are
//! incomplete and a `WarningRaised` diagnostic says so. The terminal itself
//! is never touched here.

use std::sync::Arc;

use mesh_llm_runtime_event_contracts::{
    DiagnosticEventKind, DiagnosticFact, FactData, HumanSummary, OperationId, OperationScope,
    ReasonCode, RuntimeEventIngress, RuntimeFact,
};

use crate::inference::skippy::{ModelOpenObservation, ModelOpenReturn};
use crate::runtime_events::engine::{RuntimeEventEngine, ScopedIngress};
use crate::runtime_events::runtime_event_engine;

const CONTRADICTION_SUMMARY: &str = "native model-open observation contradicts return";
const UNOBSERVED_SUMMARY: &str = "native model-open events unavailable for this load";

/// Where reconciliation reports: the load's own progress ingress when the
/// caller holds one, otherwise the process engine through an unreserved
/// ingress.
pub(super) struct ModelOpenReconciliation {
    pub(super) model: String,
    pub(super) ingress: Option<ScopedIngress>,
}

impl ModelOpenReconciliation {
    pub(super) fn reconcile(
        &self,
        observation: Option<&ModelOpenObservation>,
        returned: ModelOpenReturn,
    ) {
        let Some(engine) = self.engine() else {
            return;
        };
        let Some(observation) = observation else {
            tracing::warn!(model = %self.model, "{UNOBSERVED_SUMMARY}");
            self.submit(&engine, unobserved_fact(&self.model));
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
            self.submit(&engine, contradiction_fact(&self.model));
        }
    }

    fn engine(&self) -> Option<Arc<RuntimeEventEngine>> {
        self.ingress
            .as_ref()
            .map(|ingress| Arc::clone(ingress.engine()))
            .or_else(runtime_event_engine)
    }

    fn submit(&self, engine: &Arc<RuntimeEventEngine>, fact: RuntimeFact) {
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

fn unobserved_fact(model: &str) -> RuntimeFact {
    RuntimeFact::Diagnostic(DiagnosticFact::with_data(
        DiagnosticEventKind::WarningRaised,
        FactData {
            summary: HumanSummary::new(UNOBSERVED_SUMMARY).ok(),
            ..crate::runtime::model_lifecycle::model_scope(model)
        },
    ))
}

#[cfg(test)]
mod tests;
