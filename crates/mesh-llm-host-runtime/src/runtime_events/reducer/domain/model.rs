//! `models` category: per-model availability, load phase, and last outcome,
//! plus the root-operation -> model-id correlation that supersedes
//! provisional (pre-resolution) rows.

use mesh_llm_runtime_event_contracts::{
    FactData, ModelAvailabilityEventKind, ModelLoadingEventKind, ModelPreparationEventKind,
    ModelUnloadingEventKind, OperationId, OperationScope,
};

use super::bounded::{remove_bounded, touch, touch_root_identity};
use super::{DomainState, model_id, outcome_label};
use crate::runtime_events::config::LIFECYCLE_OPERATION_BOUND;

/// One tracked model's reduced domain view: a `models` category row.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ModelDomainState {
    pub id: String,
    pub availability: Option<String>,
    pub load_phase: Option<String>,
    pub last_outcome: Option<String>,
}

/// F2 fix (event-system-fixes, live-sampling finding): call this
/// immediately before every `touch`+`models.entry(id)` site in the model
/// apply functions below. A load operation's model identity legitimately
/// changes mid-flight (see the `model_root_identity` field doc), so `id`
/// for the SAME `root` can legitimately differ across successive facts.
/// When it does, the row `id` previously occupied is evicted -- superseded
/// rather than left an orphaned phantom -- before `root`'s new identity is
/// recorded and the caller proceeds to touch/create the row for `id`.
fn reconcile_model_root_identity(state: &mut DomainState, root: OperationId, id: &str) {
    if let Some(previous) = state.model_root_identity.get(&root) {
        if previous == id {
            return;
        }
        let previous = previous.clone();
        remove_bounded(&mut state.models_order, &mut state.models, &previous);
    }
    touch_root_identity(
        &mut state.model_root_identity_order,
        &mut state.model_root_identity,
        root,
        LIFECYCLE_OPERATION_BOUND,
    );
    state.model_root_identity.insert(root, id.to_string());
}

/// Reconcile, touch, and return the (possibly fresh) row for `id`.
fn touch_model(
    state: &mut DomainState,
    scope: OperationScope,
    id: String,
) -> &mut ModelDomainState {
    reconcile_model_root_identity(state, scope.root(), &id);
    touch(
        &mut state.models_order,
        &mut state.models,
        &id,
        LIFECYCLE_OPERATION_BOUND,
    );
    state
        .models
        .entry(id.clone())
        .or_insert_with(|| ModelDomainState {
            id,
            ..ModelDomainState::default()
        })
}

fn record_outcome(entry: &mut ModelDomainState, data: &FactData) {
    if let Some(outcome) = data.outcome {
        entry.last_outcome = Some(outcome_label(outcome).to_string());
    }
}

fn preparation_phase_label(kind: ModelPreparationEventKind) -> &'static str {
    use ModelPreparationEventKind::{
        ModelDownloadCancelled, ModelDownloadCompleted, ModelDownloadFailed, ModelDownloadProgress,
        ModelDownloadStarted, ModelPreparationCancelled, ModelPreparationCompleted,
        ModelPreparationFailed, ModelPreparationProgress, ModelPreparationStarted, ModelQueued,
        ModelResolutionCompleted, ModelResolutionFailed, ModelResolutionStarted,
    };
    match kind {
        ModelQueued => "queued",
        ModelResolutionStarted | ModelResolutionCompleted | ModelResolutionFailed => "resolving",
        ModelDownloadStarted
        | ModelDownloadProgress
        | ModelDownloadCompleted
        | ModelDownloadFailed
        | ModelDownloadCancelled => "downloading",
        ModelPreparationStarted
        | ModelPreparationProgress
        | ModelPreparationCompleted
        | ModelPreparationFailed
        | ModelPreparationCancelled => "preparing",
    }
}

pub(super) fn apply_model_preparation(
    state: &mut DomainState,
    scope: OperationScope,
    kind: ModelPreparationEventKind,
    data: &FactData,
) {
    let Some(id) = model_id(data) else {
        return;
    };
    let entry = touch_model(state, scope, id);
    entry.load_phase = Some(preparation_phase_label(kind).to_string());
    record_outcome(entry, data);
}

fn loading_phase_label(kind: ModelLoadingEventKind) -> &'static str {
    use ModelLoadingEventKind::{
        BackendDeviceSelected, ModelLoadCancelled, ModelLoadFailed, ModelLoadPhaseChanged,
        ModelLoadProgress, ModelLoadRequested, ModelLoadStarted, ModelMemoryAllocationSummary,
        ModelMemoryPressure, NativeModelLoadCompleted,
    };
    match kind {
        ModelLoadRequested => "requested",
        ModelLoadStarted
        | ModelLoadPhaseChanged
        | ModelLoadProgress
        | BackendDeviceSelected
        | ModelMemoryAllocationSummary
        | ModelMemoryPressure => "loading",
        NativeModelLoadCompleted => "loaded",
        ModelLoadFailed => "failed",
        ModelLoadCancelled => "cancelled",
    }
}

pub(super) fn apply_model_loading(
    state: &mut DomainState,
    scope: OperationScope,
    kind: ModelLoadingEventKind,
    data: &FactData,
) {
    let Some(id) = model_id(data) else {
        return;
    };
    let entry = touch_model(state, scope, id);
    if kind == ModelLoadingEventKind::ModelLoadPhaseChanged
        && let Some(transition) = &data.state
    {
        entry.load_phase = Some(transition.current.as_str().to_string());
    } else {
        entry.load_phase = Some(loading_phase_label(kind).to_string());
    }
    record_outcome(entry, data);
}

fn availability_label(kind: ModelAvailabilityEventKind) -> Option<&'static str> {
    match kind {
        ModelAvailabilityEventKind::ModelAvailable
        | ModelAvailabilityEventKind::ModelRecoveryCompleted => Some("available"),
        ModelAvailabilityEventKind::ModelDegraded => Some("degraded"),
        ModelAvailabilityEventKind::ModelUnavailable
        | ModelAvailabilityEventKind::ModelRecoveryFailed => Some("unavailable"),
        ModelAvailabilityEventKind::NativeModelLoaded
        | ModelAvailabilityEventKind::RustBackendInitializationStarted
        | ModelAvailabilityEventKind::ModelRecoveryStarted
        | ModelAvailabilityEventKind::ModelCapacityChanged => None,
    }
}

pub(super) fn apply_model_availability(
    state: &mut DomainState,
    scope: OperationScope,
    kind: ModelAvailabilityEventKind,
    data: &FactData,
) {
    let Some(id) = model_id(data) else {
        return;
    };
    let entry = touch_model(state, scope, id);
    if let Some(label) = availability_label(kind) {
        entry.availability = Some(label.to_string());
    }
    record_outcome(entry, data);
}

/// Unload completion removes ONLY the model row. Sessions, requests, and
/// stages each have their own authoritative lifecycle facts; the reducer
/// has no authority to synthesize their terminals from a model unload.
pub(super) fn apply_model_unloading(
    state: &mut DomainState,
    scope: OperationScope,
    kind: ModelUnloadingEventKind,
    data: &FactData,
) {
    let Some(id) = model_id(data) else {
        return;
    };
    match kind {
        ModelUnloadingEventKind::UnloadCompleted | ModelUnloadingEventKind::ForcedUnload => {
            remove_bounded(&mut state.models_order, &mut state.models, &id);
        }
        ModelUnloadingEventKind::UnloadRequested
        | ModelUnloadingEventKind::UnloadStarted
        | ModelUnloadingEventKind::SessionDrainingStarted
        | ModelUnloadingEventKind::SessionDrainingCompleted
        | ModelUnloadingEventKind::UnloadFailed => {
            let entry = touch_model(state, scope, id);
            entry.load_phase = Some("unloading".to_string());
            record_outcome(entry, data);
        }
    }
}
