//! `devices` and `cache` categories.

use mesh_llm_runtime_event_contracts::{KvRuntimeStateEventKind, ResourceHealthEventKind};

use super::super::fixtures::{cache_fact, device_fact, input, scope as root};
use crate::runtime_events::reducer::{ReduceOutcome, ReducerSnapshot, apply};

#[test]
fn devices_track_the_latest_resource_health_state() {
    let snapshot = ReducerSnapshot::empty();
    let ReduceOutcome::Applied(snapshot) = apply(
        &snapshot,
        input(
            root(),
            0,
            device_fact(ResourceHealthEventKind::DeviceReady, "gpu-0"),
        ),
    ) else {
        panic!("device_ready must apply");
    };
    let ReduceOutcome::Applied(snapshot) = apply(
        &snapshot,
        input(
            root(),
            1,
            device_fact(ResourceHealthEventKind::DeviceDegraded, "gpu-0"),
        ),
    ) else {
        panic!("device_degraded must apply");
    };
    let device = snapshot
        .domain()
        .devices()
        .into_iter()
        .find(|device| device.id == "gpu-0")
        .expect("device must be tracked in state.devices");
    assert_eq!(device.state.as_deref(), Some("degraded"));
}

#[test]
fn cache_tracks_the_latest_pressure_and_capacity_signal() {
    let snapshot = ReducerSnapshot::empty();
    let ReduceOutcome::Applied(snapshot) = apply(
        &snapshot,
        input(
            root(),
            0,
            cache_fact(KvRuntimeStateEventKind::CachePressureCrossed),
        ),
    ) else {
        panic!("cache_pressure_crossed must apply");
    };
    let ReduceOutcome::Applied(snapshot) = apply(
        &snapshot,
        input(
            root(),
            1,
            cache_fact(KvRuntimeStateEventKind::ContextExhausted),
        ),
    ) else {
        panic!("context_exhausted must apply");
    };
    let cache = snapshot.domain().cache();
    assert_eq!(cache.pressure.as_deref(), Some("pressure"));
    assert_eq!(cache.capacity_state.as_deref(), Some("exhausted"));
}
