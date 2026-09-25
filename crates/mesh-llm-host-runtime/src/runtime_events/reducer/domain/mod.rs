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
//!
//! Each category's row type and apply rules live in its own submodule;
//! this module owns the aggregate, the fact dispatch, the read accessors,
//! and the debug-only 1:1 order/map invariants.

mod bounded;
mod cache;
mod device;
mod diagnostics;
mod event_system;
mod execution;
mod model;
mod node;
mod request;
mod runtime;
mod session;
mod stage;

use std::collections::{HashMap, VecDeque};

use mesh_llm_runtime_event_contracts::{
    FactData, NumericSummary, NumericValue, OperationId, OperationScope, Outcome, ReasonCode,
    RuntimeFact,
};

pub use cache::CacheDomainState;
pub use device::DeviceDomainState;
pub use diagnostics::{ACTIVE_WARNING_BOUND, DiagnosticDomainState, DiagnosticEntry};
pub use event_system::EventSystemHealthDomainState;
pub use execution::{RequestGenerationState, RequestPrefillState};
pub use model::ModelDomainState;
pub use node::{NODE_CAPACITY_KEY_BOUND, NodeAvailabilityDomainState};
pub use request::RequestDomainState;
pub use runtime::NativeRuntimeDomainState;
pub use session::SessionRecentEntry;
pub use stage::StageDomainState;

/// Bounded, immutable per-category domain state. Cloned on every
/// transition alongside `OperationState`, matching `ReducerSnapshot`'s own
/// clone-on-write discipline -- never mutated in place.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DomainState {
    models: HashMap<String, ModelDomainState>,
    models_order: VecDeque<String>,
    /// F2 fix (event-system-fixes, live-sampling finding): a load
    /// operation's model identity legitimately CHANGES mid-flight -- the
    /// facts fired before source resolution completes necessarily carry a
    /// provisional identity (see `model_lifecycle::events::model_scope`'s
    /// doc comment), and every later fact for the SAME root operation
    /// carries the resolved canonical id instead. This tracks each root
    /// operation's CURRENT model id so `model::reconcile_model_root_identity`
    /// can evict the stale provisional row the moment the SAME root
    /// reports a different id, instead of leaving it an orphaned phantom
    /// stuck at whatever `load_phase` its last fact set. Bounded at
    /// `LIFECYCLE_OPERATION_BOUND`, exactly like every sibling map here.
    model_root_identity: HashMap<OperationId, String>,
    model_root_identity_order: VecDeque<OperationId>,
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
    native_runtime: NativeRuntimeDomainState,
    node_availability: NodeAvailabilityDomainState,
    diagnostics: DiagnosticDomainState,
    event_system: EventSystemHealthDomainState,
}

impl DomainState {
    /// Task 6-fix defect D (`.omo/plans/event-system-fixes.md`): emits in
    /// `models_order`'s insertion/touch order, NOT `HashMap` iteration
    /// order (unspecified, randomized per process) -- `touch()` already
    /// maintains this deque for bounded eviction; this is the first reader
    /// that also uses it for output order.
    #[must_use]
    pub fn models(&self) -> Vec<ModelDomainState> {
        ordered(&self.models_order, &self.models)
    }

    /// Deterministic insertion/touch order -- see [`Self::models`].
    #[must_use]
    pub fn stages(&self) -> Vec<StageDomainState> {
        ordered(&self.stages_order, &self.stages)
    }

    #[must_use]
    pub fn sessions_active_count(&self) -> usize {
        self.sessions_active.len()
    }

    #[must_use]
    pub fn sessions_recent(&self) -> Vec<SessionRecentEntry> {
        self.sessions_recent.iter().cloned().collect()
    }

    /// Deterministic insertion/touch order -- see [`Self::models`].
    #[must_use]
    pub fn requests(&self) -> Vec<RequestDomainState> {
        ordered(&self.requests_order, &self.requests)
    }

    /// Deterministic insertion/touch order -- see [`Self::models`].
    #[must_use]
    pub fn devices(&self) -> Vec<DeviceDomainState> {
        ordered(&self.devices_order, &self.devices)
    }

    #[must_use]
    pub fn cache(&self) -> CacheDomainState {
        self.cache.clone()
    }

    #[must_use]
    pub fn native_runtime(&self) -> &NativeRuntimeDomainState {
        &self.native_runtime
    }

    #[must_use]
    pub fn node_availability(&self) -> &NodeAvailabilityDomainState {
        &self.node_availability
    }

    #[must_use]
    pub fn diagnostics(&self) -> &DiagnosticDomainState {
        &self.diagnostics
    }

    #[must_use]
    pub fn event_system(&self) -> &EventSystemHealthDomainState {
        &self.event_system
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
    pub(super) fn apply_fact(&self, scope: OperationScope, fact: &RuntimeFact) -> Self {
        let mut next = self.clone();
        next.dispatch(scope, fact);
        next.debug_assert_invariants();
        next
    }

    fn dispatch(&mut self, scope: OperationScope, fact: &RuntimeFact) {
        match fact {
            RuntimeFact::ModelPreparation(f) => {
                model::apply_model_preparation(self, scope, *f.kind(), f.data());
            }
            RuntimeFact::ModelLoading(f) => {
                model::apply_model_loading(self, scope, *f.kind(), f.data());
            }
            RuntimeFact::ModelAvailability(f) => {
                model::apply_model_availability(self, scope, *f.kind(), f.data());
            }
            RuntimeFact::ModelUnloading(f) => {
                model::apply_model_unloading(self, scope, *f.kind(), f.data());
            }
            RuntimeFact::StageTopology(f) => {
                stage::apply_stage_topology(self, *f.kind(), f.data());
            }
            RuntimeFact::Session(f) => session::apply_session(self, *f.kind(), f.data()),
            RuntimeFact::Request(f) => request::apply_request(self, scope, *f.kind(), f.data()),
            RuntimeFact::ResourceHealth(f) => {
                device::apply_resource_health(self, *f.kind(), f.data());
            }
            RuntimeFact::KvRuntimeState(f) => {
                cache::apply_kv_runtime_state(&mut self.cache, *f.kind());
            }
            RuntimeFact::Prefill(f) => execution::apply_prefill(self, scope, *f.kind(), f.data()),
            RuntimeFact::Generation(f) => {
                execution::apply_generation(self, scope, *f.kind(), f.data());
            }
            other => self.dispatch_node_family(other),
        }
    }

    /// Node-wide families: no per-entity identity, one latest-wins view each.
    fn dispatch_node_family(&mut self, fact: &RuntimeFact) {
        match fact {
            RuntimeFact::NativeRuntime(f) => {
                runtime::apply_native_runtime(&mut self.native_runtime, *f.kind(), f.data());
            }
            RuntimeFact::NodeAvailability(f) => {
                node::apply_node_availability(&mut self.node_availability, *f.kind(), f.data());
            }
            RuntimeFact::Diagnostic(f) => {
                diagnostics::apply_diagnostic(&mut self.diagnostics, *f.kind(), f.data());
            }
            RuntimeFact::EventSystemHealth(f) => {
                event_system::apply_event_system_health(
                    &mut self.event_system,
                    *f.kind(),
                    f.data(),
                );
            }
            RuntimeFact::ModelPreparation(_)
            | RuntimeFact::ModelLoading(_)
            | RuntimeFact::ModelAvailability(_)
            | RuntimeFact::ModelUnloading(_)
            | RuntimeFact::StageTopology(_)
            | RuntimeFact::Session(_)
            | RuntimeFact::Request(_)
            | RuntimeFact::Prefill(_)
            | RuntimeFact::Generation(_)
            | RuntimeFact::KvRuntimeState(_)
            | RuntimeFact::ResourceHealth(_) => {}
        }
    }

    /// Task 6-fix R1 invariant assertion (`.omo/plans/event-system-fixes.md`):
    /// every bounded category's `*_order` deque must stay exactly 1:1
    /// with its map, or `models()`/`stages()`/`requests()`/`devices()`
    /// silently drop or duplicate a row instead of loudly failing.
    /// Compiled out entirely in release builds -- zero hot-path cost.
    fn debug_assert_invariants(&self) {
        debug_assert_eq!(self.models_order.len(), self.models.len());
        debug_assert_eq!(self.stages_order.len(), self.stages.len());
        debug_assert_eq!(self.requests_order.len(), self.requests.len());
        debug_assert_eq!(self.devices_order.len(), self.devices.len());
        debug_assert_eq!(self.sessions_order.len(), self.sessions_active.len());
        debug_assert_eq!(
            self.model_root_identity_order.len(),
            self.model_root_identity.len()
        );
    }
}

fn ordered<V: Clone>(order: &VecDeque<String>, map: &HashMap<String, V>) -> Vec<V> {
    order.iter().filter_map(|id| map.get(id).cloned()).collect()
}

fn reason_label(data: &FactData) -> Option<String> {
    data.reason
        .as_ref()
        .map(crate::runtime_events::presentation::reason_code_str)
}

/// The engine's reservation-drop synthesis (`outcome: unknown`,
/// `reason: terminal_not_delivered`). It settles an operation but reports
/// nothing about the domain, so state views must not treat it as evidence.
fn is_undelivered_terminal(data: &FactData) -> bool {
    matches!(data.reason, Some(ReasonCode::TerminalNotDelivered))
}

/// A numeric summary as a non-negative integer; floats and negatives are
/// not counts and are ignored.
fn summary_value(summary: &NumericSummary) -> Option<u64> {
    match summary.value {
        NumericValue::Unsigned(value) => Some(value),
        NumericValue::Signed(value) => u64::try_from(value).ok(),
        NumericValue::Floating(_) => None,
    }
}

fn unsigned_summary(data: &FactData, key: &str) -> Option<u64> {
    data.numeric_summaries
        .as_slice()
        .iter()
        .find(|summary| summary.key.as_str() == key)
        .and_then(summary_value)
}

fn model_id(data: &FactData) -> Option<String> {
    data.scope
        .model_id
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
