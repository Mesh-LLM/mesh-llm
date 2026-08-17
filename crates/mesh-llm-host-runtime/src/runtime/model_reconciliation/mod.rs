pub(crate) mod intent;
mod planner;
mod state;

pub(crate) use intent::{
    DesiredRuntimeIntent, IntentPersistence, IntentSource, ModelIntent,
    ModelTargetReconciliationAction, ModelTargetReconciliationCandidate,
    ModelTargetReconciliationCapacityState,
};
pub(crate) use planner::{
    ModelTargetReconciliationInput, ModelTargetReconciliationPolicy,
    plan_model_target_reconciliation,
};
pub(crate) use state::{ModelTargetReconciliationState, current_time_secs};

fn model_identity_matches(left: &str, right: &str) -> bool {
    if left == right {
        return true;
    }
    let (Ok(left), Ok(right)) = (
        model_ref::ModelRef::parse(left),
        model_ref::ModelRef::parse(right),
    ) else {
        return false;
    };
    left.repo == right.repo
        && left.selector == right.selector
        && (left.revision == right.revision
            || matches!(
                (left.revision.as_deref(), right.revision.as_deref()),
                (None, Some("main")) | (Some("main"), None)
            ))
}
