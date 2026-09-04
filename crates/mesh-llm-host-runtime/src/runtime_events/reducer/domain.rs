//! Bounded per-category domain state, reduced from the same applied facts
//! that update `OperationState`.
//!
//! Defect D6 (`.omo/plans/event-system-fixes.md` task 6): the reducer
//! applied every fact but discarded anything beyond generic operation
//! health (outcome/progress/sequence), so `runtime_state`'s category
//! arrays were always empty. `DomainState` reduces the SAME applied fact
//! into a small, EXPLICITLY bounded per-category view -- never derived
//! from `OperationState`, and never unbounded: every collection here is
//! capped by an EXISTING frozen bound from `runtime_events::config`
//! (`LIFECYCLE_OPERATION_BOUND` for models/stages/sessions/devices,
//! `REQUEST_ROOT_BOUND` for in-flight requests), with oldest-touched
//! eviction when a category would exceed its cap.
//!
//! `sessions` and `cache` are intentionally NOT per-entity collections:
//! the plan text describes `sessions` as "active count and bounded
//! recent" and `cache` as "last known capacity/pressure state" -- an
//! aggregate count plus a bounded FIFO, and a single latest-wins object,
//! respectively.

use std::collections::{HashMap, VecDeque};

use mesh_llm_runtime_event_contracts::{
    FactData, KvRuntimeStateEventKind, ModelAvailabilityEventKind, ModelLoadingEventKind,
    ModelPreparationEventKind, ModelUnloadingEventKind, Outcome, RequestEventKind,
    ResourceHealthEventKind, RuntimeFact, SessionEventKind, StageTopologyEventKind,
};

use crate::runtime_events::config::{LIFECYCLE_OPERATION_BOUND, REQUEST_ROOT_BOUND};

/// One tracked model's reduced domain view: a `models` category row.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ModelDomainState {
    pub id: String,
    pub availability: Option<String>,
    pub load_phase: Option<String>,
    pub last_outcome: Option<String>,
}

/// One tracked stage's reduced domain view: a `stages` category row.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StageDomainState {
    pub id: String,
    pub index: Option<u32>,
    pub state: Option<String>,
    pub last_outcome: Option<String>,
}

/// One recently-settled session, retained after leaving the active set:
/// a `sessions.recent` row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionRecentEntry {
    pub id: String,
    pub state: String,
}

/// One in-flight request's reduced domain view: a `requests` category
/// row. Removed the moment its terminal-class outcome settles --
/// "in-flight only" per the plan text.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RequestDomainState {
    pub id: String,
    pub state: Option<String>,
}

/// One tracked device's reduced domain view: a `devices` category row.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DeviceDomainState {
    pub id: String,
    pub state: Option<String>,
}

/// Last known KV-cache capacity/pressure signal: `cache` is a single
/// latest-wins object, not a per-entity collection.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CacheDomainState {
    pub pressure: Option<String>,
    pub capacity_state: Option<String>,
}

/// Bounded, immutable per-category domain state. Cloned on every
/// transition alongside `OperationState`, matching `ReducerSnapshot`'s own
/// clone-on-write discipline -- never mutated in place.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DomainState {
    models: HashMap<String, ModelDomainState>,
    models_order: VecDeque<String>,
    stages: HashMap<String, StageDomainState>,
    stages_order: VecDeque<String>,
    sessions_active: HashMap<String, String>,
    sessions_order: VecDeque<String>,
    sessions_recent: VecDeque<SessionRecentEntry>,
    requests: HashMap<String, RequestDomainState>,
    requests_order: VecDeque<String>,
    devices: HashMap<String, DeviceDomainState>,
    devices_order: VecDeque<String>,
    cache: CacheDomainState,
}

impl DomainState {
    #[must_use]
    pub fn models(&self) -> Vec<ModelDomainState> {
        self.models.values().cloned().collect()
    }

    #[must_use]
    pub fn stages(&self) -> Vec<StageDomainState> {
        self.stages.values().cloned().collect()
    }

    #[must_use]
    pub fn sessions_active_count(&self) -> usize {
        self.sessions_active.len()
    }

    #[must_use]
    pub fn sessions_recent(&self) -> Vec<SessionRecentEntry> {
        self.sessions_recent.iter().cloned().collect()
    }

    #[must_use]
    pub fn requests(&self) -> Vec<RequestDomainState> {
        self.requests.values().cloned().collect()
    }

    #[must_use]
    pub fn devices(&self) -> Vec<DeviceDomainState> {
        self.devices.values().cloned().collect()
    }

    #[must_use]
    pub fn cache(&self) -> CacheDomainState {
        self.cache.clone()
    }

    /// Whether at least one tracked model currently reports `"available"`.
    /// Used by the `runtime_data` event-cutover shadow comparison (task 6)
    /// as a cheap, real reducer-derived signal to compare against the
    /// legacy `RuntimeStatusSnapshot.llama_ready` field.
    #[must_use]
    pub fn has_available_model(&self) -> bool {
        self.models
            .values()
            .any(|model| model.availability.as_deref() == Some("available"))
    }

    /// The set of every model id currently tracked, for the same shadow
    /// comparison against the legacy local-inventory model-name set.
    #[must_use]
    pub fn model_id_set(&self) -> std::collections::HashSet<String> {
        self.models.keys().cloned().collect()
    }

    /// Reduce `fact` into a fresh `DomainState`. Pure: `self` is never
    /// mutated, matching `ReducerSnapshot::with_operation`'s own
    /// transactional discipline.
    #[must_use]
    pub(super) fn apply_fact(&self, fact: &RuntimeFact) -> Self {
        let mut next = self.clone();
        match fact {
            RuntimeFact::ModelPreparation(f) => {
                apply_model_preparation(&mut next, *f.kind(), f.data());
            }
            RuntimeFact::ModelLoading(f) => apply_model_loading(&mut next, *f.kind(), f.data()),
            RuntimeFact::ModelAvailability(f) => {
                apply_model_availability(&mut next, *f.kind(), f.data());
            }
            RuntimeFact::ModelUnloading(f) => {
                apply_model_unloading(&mut next, *f.kind(), f.data());
            }
            RuntimeFact::StageTopology(f) => apply_stage_topology(&mut next, *f.kind(), f.data()),
            RuntimeFact::Session(f) => apply_session(&mut next, *f.kind(), f.data()),
            RuntimeFact::Request(f) => apply_request(&mut next, *f.kind(), f.data()),
            RuntimeFact::ResourceHealth(f) => {
                apply_resource_health(&mut next, *f.kind(), f.data());
            }
            RuntimeFact::KvRuntimeState(f) => apply_kv_runtime_state(&mut next, *f.kind()),
            // Prefill/Generation/Diagnostic/NodeAvailability/EventSystemHealth
            // facts carry no scope identity this task's frozen category set
            // covers (models/stages/sessions/requests/devices/cache); they
            // are left for a later task's projection, exactly like today's
            // `node` category is populated from health/reducer data other
            // fact families don't touch.
            RuntimeFact::NativeRuntime(_)
            | RuntimeFact::Prefill(_)
            | RuntimeFact::Generation(_)
            | RuntimeFact::Diagnostic(_)
            | RuntimeFact::NodeAvailability(_)
            | RuntimeFact::EventSystemHealth(_) => {}
        }
        next
    }
}

fn model_id(data: &FactData) -> Option<String> {
    data.scope
        .model_id
        .as_ref()
        .map(|id| id.as_str().to_string())
}

fn stage_identity(data: &FactData) -> Option<(String, u32)> {
    data.scope
        .stage
        .as_ref()
        .map(|stage| (stage.id.as_str().to_string(), stage.index))
}

fn session_id(data: &FactData) -> Option<String> {
    data.scope
        .session_id
        .as_ref()
        .map(|id| id.as_str().to_string())
}

fn request_id(data: &FactData) -> Option<String> {
    data.scope
        .request_id
        .as_ref()
        .map(|id| id.as_str().to_string())
}

fn device_id(data: &FactData) -> Option<String> {
    data.scope
        .device_id
        .as_ref()
        .map(|id| id.as_str().to_string())
}

/// Stable, lowercase wire-shaped label -- mirrors the convention
/// `api::routes::runtime_events::frames::outcome_str` already uses for the
/// `runtime_event` projection, so a future wire-pinning task (task 7)
/// sees the same vocabulary in both places.
const fn outcome_label(outcome: Outcome) -> &'static str {
    match outcome {
        Outcome::Success => "success",
        Outcome::Failure => "failure",
        Outcome::Rejected => "rejected",
        Outcome::Cancelled => "cancelled",
        Outcome::Unknown => "unknown",
    }
}

/// Move `id` to the back of `order` (most-recently-touched); if `id` is
/// new and `map` is already at `bound`, evict the single oldest entry
/// first. Keeps every bounded category map's insertion-order eviction
/// deterministic instead of relying on `HashMap`'s unspecified iteration
/// order.
fn touch<V>(order: &mut VecDeque<String>, map: &mut HashMap<String, V>, id: &str, bound: usize) {
    if let Some(position) = order.iter().position(|existing| existing == id) {
        order.remove(position);
    } else if map.len() >= bound
        && let Some(oldest) = order.pop_front()
    {
        map.remove(&oldest);
    }
    order.push_back(id.to_string());
}

fn remove_bounded<V>(order: &mut VecDeque<String>, map: &mut HashMap<String, V>, id: &str) {
    if let Some(position) = order.iter().position(|existing| existing == id) {
        order.remove(position);
    }
    map.remove(id);
}

fn push_recent(recent: &mut VecDeque<SessionRecentEntry>, entry: SessionRecentEntry, bound: usize) {
    if recent.len() >= bound {
        recent.pop_front();
    }
    recent.push_back(entry);
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

fn apply_model_preparation(
    state: &mut DomainState,
    kind: ModelPreparationEventKind,
    data: &FactData,
) {
    let Some(id) = model_id(data) else {
        return;
    };
    touch(
        &mut state.models_order,
        &mut state.models,
        &id,
        LIFECYCLE_OPERATION_BOUND,
    );
    let entry = state
        .models
        .entry(id.clone())
        .or_insert_with(|| ModelDomainState {
            id,
            ..ModelDomainState::default()
        });
    entry.load_phase = Some(preparation_phase_label(kind).to_string());
    if let Some(outcome) = data.outcome {
        entry.last_outcome = Some(outcome_label(outcome).to_string());
    }
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

fn apply_model_loading(state: &mut DomainState, kind: ModelLoadingEventKind, data: &FactData) {
    let Some(id) = model_id(data) else {
        return;
    };
    touch(
        &mut state.models_order,
        &mut state.models,
        &id,
        LIFECYCLE_OPERATION_BOUND,
    );
    let entry = state
        .models
        .entry(id.clone())
        .or_insert_with(|| ModelDomainState {
            id,
            ..ModelDomainState::default()
        });
    if kind == ModelLoadingEventKind::ModelLoadPhaseChanged
        && let Some(transition) = &data.state
    {
        entry.load_phase = Some(transition.current.as_str().to_string());
    } else {
        entry.load_phase = Some(loading_phase_label(kind).to_string());
    }
    if let Some(outcome) = data.outcome {
        entry.last_outcome = Some(outcome_label(outcome).to_string());
    }
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

fn apply_model_availability(
    state: &mut DomainState,
    kind: ModelAvailabilityEventKind,
    data: &FactData,
) {
    let Some(id) = model_id(data) else {
        return;
    };
    touch(
        &mut state.models_order,
        &mut state.models,
        &id,
        LIFECYCLE_OPERATION_BOUND,
    );
    let entry = state
        .models
        .entry(id.clone())
        .or_insert_with(|| ModelDomainState {
            id,
            ..ModelDomainState::default()
        });
    if let Some(label) = availability_label(kind) {
        entry.availability = Some(label.to_string());
    }
    if let Some(outcome) = data.outcome {
        entry.last_outcome = Some(outcome_label(outcome).to_string());
    }
}

fn apply_model_unloading(state: &mut DomainState, kind: ModelUnloadingEventKind, data: &FactData) {
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
            touch(
                &mut state.models_order,
                &mut state.models,
                &id,
                LIFECYCLE_OPERATION_BOUND,
            );
            let entry = state
                .models
                .entry(id.clone())
                .or_insert_with(|| ModelDomainState {
                    id,
                    ..ModelDomainState::default()
                });
            entry.load_phase = Some("unloading".to_string());
            if let Some(outcome) = data.outcome {
                entry.last_outcome = Some(outcome_label(outcome).to_string());
            }
        }
    }
}

fn stage_state_label(kind: StageTopologyEventKind) -> &'static str {
    use StageTopologyEventKind::{
        StageConnectionEstablished, StageConnectionLost, StageConnectionRecovered, StageDegraded,
        StageFailed, StageLoading, StageReady, StageStarting, StageStopped, StageStopping,
        StageUnavailable, TopologyAssembling, TopologyDegraded, TopologyReady, TopologyUnavailable,
    };
    match kind {
        StageStarting => "starting",
        StageLoading => "loading",
        StageReady | StageConnectionEstablished | StageConnectionRecovered => "ready",
        StageDegraded => "degraded",
        StageUnavailable => "unavailable",
        StageStopping => "stopping",
        StageStopped => "stopped",
        StageFailed | StageConnectionLost => "failed",
        TopologyAssembling => "assembling",
        TopologyReady => "topology_ready",
        TopologyDegraded => "topology_degraded",
        TopologyUnavailable => "topology_unavailable",
    }
}

fn apply_stage_topology(state: &mut DomainState, kind: StageTopologyEventKind, data: &FactData) {
    let Some((id, index)) = stage_identity(data) else {
        return;
    };
    touch(
        &mut state.stages_order,
        &mut state.stages,
        &id,
        LIFECYCLE_OPERATION_BOUND,
    );
    let entry = state
        .stages
        .entry(id.clone())
        .or_insert_with(|| StageDomainState {
            id,
            ..StageDomainState::default()
        });
    entry.index = Some(index);
    entry.state = Some(stage_state_label(kind).to_string());
    if let Some(outcome) = data.outcome {
        entry.last_outcome = Some(outcome_label(outcome).to_string());
    }
}

fn session_state_label(kind: SessionEventKind) -> &'static str {
    use SessionEventKind::{
        SessionAbandoned, SessionActive, SessionClosed, SessionCreated, SessionDraining,
        SessionFailed, SessionIdle, SessionReclaimed, SessionRequested, SessionReset,
        SessionRestoredFromCheckpoint, SessionRestoredFromPrefixCache, SessionReusable,
        SessionTrimmed,
    };
    match kind {
        SessionRequested => "requested",
        SessionCreated => "created",
        SessionActive => "active",
        SessionIdle => "idle",
        SessionReusable => "reusable",
        SessionReset => "reset",
        SessionTrimmed => "trimmed",
        SessionRestoredFromPrefixCache | SessionRestoredFromCheckpoint => "restored",
        SessionDraining => "draining",
        SessionClosed => "closed",
        SessionFailed => "failed",
        SessionAbandoned => "abandoned",
        SessionReclaimed => "reclaimed",
    }
}

fn is_terminal_session_kind(kind: SessionEventKind) -> bool {
    matches!(
        kind,
        SessionEventKind::SessionClosed
            | SessionEventKind::SessionFailed
            | SessionEventKind::SessionAbandoned
            | SessionEventKind::SessionReclaimed
    )
}

fn apply_session(state: &mut DomainState, kind: SessionEventKind, data: &FactData) {
    let Some(id) = session_id(data) else {
        return;
    };
    let label = session_state_label(kind);
    if is_terminal_session_kind(kind) {
        if let Some(position) = state
            .sessions_order
            .iter()
            .position(|existing| existing == &id)
        {
            state.sessions_order.remove(position);
        }
        state.sessions_active.remove(&id);
        push_recent(
            &mut state.sessions_recent,
            SessionRecentEntry {
                id,
                state: label.to_string(),
            },
            LIFECYCLE_OPERATION_BOUND,
        );
        return;
    }
    touch(
        &mut state.sessions_order,
        &mut state.sessions_active,
        &id,
        LIFECYCLE_OPERATION_BOUND,
    );
    state.sessions_active.insert(id, label.to_string());
}

fn is_terminal_request_kind(kind: RequestEventKind) -> bool {
    matches!(
        kind,
        RequestEventKind::RequestCompleted
            | RequestEventKind::RequestCancelled
            | RequestEventKind::RequestTimedOut
            | RequestEventKind::RequestFailed
            | RequestEventKind::RequestRejected
    )
}

fn request_state_label(kind: RequestEventKind) -> &'static str {
    match kind {
        RequestEventKind::RequestReceived => "received",
        RequestEventKind::RequestQueued => "queued",
        RequestEventKind::RequestAdmitted => "admitted",
        RequestEventKind::RequestExecutionStarted => "executing",
        // Terminal-class kinds never reach this arm: `apply_request` below
        // removes the entry before calling this function for any of them.
        // The match stays exhaustive so a new `RequestEventKind` variant is
        // a compile error here, not a silent no-op.
        RequestEventKind::RequestRejected
        | RequestEventKind::RequestCompleted
        | RequestEventKind::RequestCancelled
        | RequestEventKind::RequestTimedOut
        | RequestEventKind::RequestFailed => "terminal",
    }
}

fn apply_request(state: &mut DomainState, kind: RequestEventKind, data: &FactData) {
    let Some(id) = request_id(data) else {
        return;
    };
    if is_terminal_request_kind(kind) {
        remove_bounded(&mut state.requests_order, &mut state.requests, &id);
        return;
    }
    touch(
        &mut state.requests_order,
        &mut state.requests,
        &id,
        REQUEST_ROOT_BOUND,
    );
    let label = request_state_label(kind);
    state.requests.insert(
        id.clone(),
        RequestDomainState {
            id,
            state: Some(label.to_string()),
        },
    );
}

fn device_state_label(kind: ResourceHealthEventKind) -> &'static str {
    use ResourceHealthEventKind::{
        BackendFallbackActivated, BackendInitializationCompleted, BackendInitializationFailed,
        BackendInitializationStarted, ComputeFailure, CpuFallbackActivated, DeviceDegraded,
        DeviceLost, DeviceReady, DeviceRecovered, DeviceReset, DeviceSelected, DeviceUnavailable,
        MemoryPressureCleared, MemoryPressureCrossed, OutOfMemoryCondition,
        ResourceAllocationCompleted, ResourceAllocationFailed,
    };
    match kind {
        BackendInitializationStarted => "initializing",
        BackendInitializationCompleted
        | DeviceReady
        | DeviceRecovered
        | ResourceAllocationCompleted
        | MemoryPressureCleared => "ready",
        BackendInitializationFailed | ResourceAllocationFailed | ComputeFailure => "failed",
        DeviceSelected => "selected",
        DeviceDegraded => "degraded",
        DeviceUnavailable => "unavailable",
        MemoryPressureCrossed => "pressure",
        OutOfMemoryCondition => "out_of_memory",
        BackendFallbackActivated | CpuFallbackActivated => "fallback",
        DeviceLost => "lost",
        DeviceReset => "reset",
    }
}

fn apply_resource_health(state: &mut DomainState, kind: ResourceHealthEventKind, data: &FactData) {
    let Some(id) = device_id(data) else {
        return;
    };
    touch(
        &mut state.devices_order,
        &mut state.devices,
        &id,
        LIFECYCLE_OPERATION_BOUND,
    );
    let label = device_state_label(kind);
    let entry = state
        .devices
        .entry(id.clone())
        .or_insert_with(|| DeviceDomainState {
            id,
            ..DeviceDomainState::default()
        });
    entry.state = Some(label.to_string());
}

fn apply_kv_runtime_state(state: &mut DomainState, kind: KvRuntimeStateEventKind) {
    match kind {
        KvRuntimeStateEventKind::CachePressureCrossed => {
            state.cache.pressure = Some("pressure".to_string());
        }
        KvRuntimeStateEventKind::CachePressureCleared => {
            state.cache.pressure = Some("normal".to_string());
        }
        KvRuntimeStateEventKind::ContextCapacityApproachingLimit => {
            state.cache.capacity_state = Some("approaching_limit".to_string());
        }
        KvRuntimeStateEventKind::ContextExhausted => {
            state.cache.capacity_state = Some("exhausted".to_string());
        }
        KvRuntimeStateEventKind::CacheReset => {
            state.cache.pressure = Some("normal".to_string());
            state.cache.capacity_state = Some("reset".to_string());
        }
        KvRuntimeStateEventKind::KvCacheInitializationStarted
        | KvRuntimeStateEventKind::KvCacheInitializationCompleted
        | KvRuntimeStateEventKind::KvCacheInitializationFailed
        | KvRuntimeStateEventKind::CacheLookupHit
        | KvRuntimeStateEventKind::CacheLookupMiss
        | KvRuntimeStateEventKind::CacheLookupPartial
        | KvRuntimeStateEventKind::CacheLookupError
        | KvRuntimeStateEventKind::PrefixRestored
        | KvRuntimeStateEventKind::CheckpointRestored
        | KvRuntimeStateEventKind::CacheRecordCompleted
        | KvRuntimeStateEventKind::CacheRecordFailed
        | KvRuntimeStateEventKind::CacheTrim
        | KvRuntimeStateEventKind::CacheEviction
        | KvRuntimeStateEventKind::RuntimeStateImportCompleted
        | KvRuntimeStateEventKind::RuntimeStateImportFailed
        | KvRuntimeStateEventKind::RuntimeStateExportCompleted
        | KvRuntimeStateEventKind::RuntimeStateExportFailed => {}
    }
}
