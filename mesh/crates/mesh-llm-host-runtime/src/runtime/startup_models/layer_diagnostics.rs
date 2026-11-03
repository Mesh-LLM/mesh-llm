//! Diagnostics for startup model refs that name a layer package which cannot
//! be resolved.
//!
//! Resolution itself is owned by [`super::resolve_split_layer_package`]. This
//! module owns only what the operator gets to see when that resolution comes up
//! empty, so a node that can never announce split model interest does not read
//! as a healthy idle peer that is merely uninterested.

use mesh_llm_runtime_event_contracts::{
    BoundedNumericSummaries, DiagnosticEventKind, DiagnosticFact, FactData, HumanSummary,
    NumericSummary, NumericSummaryKey, NumericValue, OperationId, OperationScope, ReasonCode,
    RuntimeEventIngress, RuntimeFact, UnknownReasonCode,
};

use crate::runtime::tracing_writer::stable_message_id;
use crate::runtime_events::reducer::WARNING_CORRELATION_KEY;

/// Reason code carried by the active warning for an unresolved model ref.
const UNRESOLVED_MODEL_REF_REASON: &str = "model_ref_unresolvable";

/// Report a model ref that names a split layer package which cannot be
/// resolved, as a `warn` line plus an active runtime warning.
///
/// `reason` explains how far resolution got, so an operator can tell a missing
/// catalog entry apart from an unreachable repository. Refs that were never
/// candidates for a layer package stay silent: they resolve to `None` on every
/// healthy node.
pub(super) fn report_unresolved_layer_package(model_query: &str, reason: &str) {
    if !names_layer_package_repo(model_query) {
        return;
    }

    let message = format!(
        "Unresolved model ref {model_query}: {reason}. Split serving will not announce \
         interest for it, so a coordinator excludes this node as missing_model_interest. \
         Publish a model-package.json for this ref or start a model that has a layer package."
    );
    tracing::warn!("{message}");

    let Some(engine) = crate::runtime_events::runtime_event_engine() else {
        return;
    };
    let fact = RuntimeFact::Diagnostic(DiagnosticFact::with_data(
        DiagnosticEventKind::WarningRaised,
        FactData {
            reason: Some(ReasonCode::Unknown(
                UnknownReasonCode::new(UNRESOLVED_MODEL_REF_REASON)
                    .expect("the reason code is within the contract bound"),
            )),
            summary: HumanSummary::new(&message).ok(),
            numeric_summaries: NumericSummaryKey::new(WARNING_CORRELATION_KEY)
                .ok()
                .map(|key| {
                    NumericSummary::new(key, NumericValue::Unsigned(stable_message_id(&message)))
                })
                .and_then(|summary| BoundedNumericSummaries::new(vec![summary]).ok())
                .unwrap_or_default(),
            ..FactData::default()
        },
    ));
    let _ = engine
        .unreserved_ingress(OperationScope::root_only(OperationId::new()))
        .try_submit(fact);
}

/// Whether the ref names a Hugging Face repository that is meant to hold a
/// split layer package.
///
/// The `-layers` suffix is the package-repo convention, so an ordinary
/// monolithic Hugging Face ref does not turn every healthy startup into a
/// warning.
fn names_layer_package_repo(model_query: &str) -> bool {
    crate::models::remote_catalog::huggingface_repo_ref(model_query)
        .is_some_and(|repo| repo.ends_with("-layers"))
}
