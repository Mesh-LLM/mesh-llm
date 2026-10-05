//! `devices` category: latest resource-health state per device.

use mesh_llm_runtime_event_contracts::{FactData, ResourceHealthEventKind};

use super::DomainState;
use super::bounded::touch;
use crate::runtime_events::config::LIFECYCLE_OPERATION_BOUND;

/// One tracked device's reduced domain view: a `devices` category row.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DeviceDomainState {
    pub id: String,
    pub state: Option<String>,
}

fn device_id(data: &FactData) -> Option<String> {
    data.scope
        .device_id
        .as_ref()
        .map(|id| id.as_str().to_string())
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

pub(super) fn apply_resource_health(
    state: &mut DomainState,
    kind: ResourceHealthEventKind,
    data: &FactData,
) {
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
