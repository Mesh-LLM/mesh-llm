//! Node-wide `runtime_state.node` sub-objects: native runtime, node
//! availability, diagnostics, and event-system health.
//!
//! Each sub-object is omitted entirely while its reducer view is empty, so
//! a fresh engine's `node` stays exactly
//! `{"rebuild_generation":0,"tracked_operation_count":0}` -- the frozen
//! empty sample in `fixtures/runtime_events_v1/frames.json`.

use std::collections::BTreeMap;

use serde::Serialize;

use crate::runtime_events::reducer::{
    DiagnosticDomainState, DiagnosticEntry, DomainState, EventSystemHealthDomainState,
    NativeRuntimeDomainState, NodeAvailabilityDomainState,
};

#[derive(Debug, Serialize)]
pub(crate) struct NativeRuntimeProjection {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) status: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) abi_compatible: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) last_outcome: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) last_reason_code: Option<String>,
}

impl From<&NativeRuntimeDomainState> for NativeRuntimeProjection {
    fn from(state: &NativeRuntimeDomainState) -> Self {
        Self {
            status: state.status,
            abi_compatible: state.abi_compatible,
            last_outcome: state.last_outcome,
            last_reason_code: state.last_reason_code.clone(),
        }
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct NodeAvailabilityProjection {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) state: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) available_model_count: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) available_stage_count: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) local_capacity_available: Option<bool>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub(crate) capacity: BTreeMap<String, u64>,
    #[serde(skip_serializing_if = "is_zero")]
    pub(crate) capacity_overflow: u64,
}

impl From<&NodeAvailabilityDomainState> for NodeAvailabilityProjection {
    fn from(state: &NodeAvailabilityDomainState) -> Self {
        Self {
            state: state.state,
            available_model_count: state.available_model_count,
            available_stage_count: state.available_stage_count,
            local_capacity_available: state.local_capacity_available,
            capacity: state.capacity.clone(),
            capacity_overflow: state.capacity_overflow,
        }
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct DiagnosticEntryProjection {
    pub(crate) key: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) reason_code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) summary: Option<String>,
}

impl From<&DiagnosticEntry> for DiagnosticEntryProjection {
    fn from(entry: &DiagnosticEntry) -> Self {
        Self {
            key: entry.key.clone(),
            reason_code: entry.reason_code.clone(),
            summary: entry.summary.clone(),
        }
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct DiagnosticsProjection {
    pub(crate) active_warnings: Vec<DiagnosticEntryProjection>,
    pub(crate) evicted_warnings: u64,
    pub(crate) degraded: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) fatal: Option<DiagnosticEntryProjection>,
    pub(crate) recoverable_failures: u64,
    pub(crate) fallbacks: u64,
    pub(crate) invariant_violations: u64,
}

impl From<&DiagnosticDomainState> for DiagnosticsProjection {
    fn from(state: &DiagnosticDomainState) -> Self {
        Self {
            active_warnings: state
                .active_warnings
                .iter()
                .map(DiagnosticEntryProjection::from)
                .collect(),
            evicted_warnings: state.evicted_warnings,
            degraded: state.degraded,
            fatal: state.fatal.as_ref().map(DiagnosticEntryProjection::from),
            recoverable_failures: state.recoverable_failures,
            fallbacks: state.fallbacks,
            invariant_violations: state.invariant_violations,
        }
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct EventSystemProjection {
    pub(crate) counts_by_kind: BTreeMap<&'static str, u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) pressure: Option<String>,
    pub(crate) lagging_subscribers: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) telemetry_exporter: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) last_kind: Option<&'static str>,
}

impl From<&EventSystemHealthDomainState> for EventSystemProjection {
    fn from(state: &EventSystemHealthDomainState) -> Self {
        Self {
            counts_by_kind: state.counts_by_kind.clone(),
            pressure: state.pressure.clone(),
            lagging_subscribers: state.lagging_subscribers,
            telemetry_exporter: state.telemetry_exporter,
            last_kind: state.last_kind,
        }
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct NodeProjection {
    pub(crate) rebuild_generation: u64,
    pub(crate) tracked_operation_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) runtime: Option<NativeRuntimeProjection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) availability: Option<NodeAvailabilityProjection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) diagnostics: Option<DiagnosticsProjection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) event_system: Option<EventSystemProjection>,
}

/// Snapshot-level counters plus the node-wide reducer views.
pub(crate) struct NodeCounters {
    pub(crate) rebuild_generation: u64,
    pub(crate) tracked_operation_count: usize,
}

impl NodeProjection {
    pub(crate) fn build(counters: NodeCounters, domain: &DomainState) -> Self {
        let runtime = domain.native_runtime();
        let availability = domain.node_availability();
        let diagnostics = domain.diagnostics();
        let event_system = domain.event_system();
        Self {
            rebuild_generation: counters.rebuild_generation,
            tracked_operation_count: counters.tracked_operation_count,
            runtime: (!runtime.is_empty()).then(|| runtime.into()),
            availability: (!availability.is_empty()).then(|| availability.into()),
            diagnostics: (!diagnostics.is_empty()).then(|| diagnostics.into()),
            event_system: (!event_system.is_empty()).then(|| event_system.into()),
        }
    }
}

const fn is_zero(value: &u64) -> bool {
    *value == 0
}
