//! `runtime_state` domain projection.
//!
//! SCOPE LIMIT (see `frames.rs` doc comment): the host engine's
//! `ReducerSnapshot` (task 4) is an operation-health map only — it does not
//! yet retain per-category domain state, so it cannot answer "which
//! operations are models/stages/sessions/requests/devices/cache entries".
//! The six category arrays are therefore structurally present (satisfying
//! the frozen `state` top-level-key requirement) but empty until a later
//! task extends the reducer with per-category projections. `node` is
//! genuinely populated from the reducer/health data that IS available
//! today.

use serde::Serialize;

use crate::runtime_events::engine::RuntimeEventEngine;

#[derive(Debug, Serialize)]
pub(super) struct NodeProjection {
    pub(super) rebuild_generation: u64,
    pub(super) tracked_operation_count: usize,
}

#[derive(Debug, Serialize)]
pub(super) struct StateProjection {
    pub(super) node: NodeProjection,
    pub(super) models: Vec<serde_json::Value>,
    pub(super) stages: Vec<serde_json::Value>,
    pub(super) sessions: Vec<serde_json::Value>,
    pub(super) requests: Vec<serde_json::Value>,
    pub(super) devices: Vec<serde_json::Value>,
    pub(super) cache: Vec<serde_json::Value>,
}

pub(super) fn build(engine: &RuntimeEventEngine) -> StateProjection {
    let snapshot = engine.reducer_snapshot();
    StateProjection {
        node: NodeProjection {
            rebuild_generation: snapshot.rebuild_generation,
            tracked_operation_count: snapshot.operation_count(),
        },
        models: Vec::new(),
        stages: Vec::new(),
        sessions: Vec::new(),
        requests: Vec::new(),
        devices: Vec::new(),
        cache: Vec::new(),
    }
}
