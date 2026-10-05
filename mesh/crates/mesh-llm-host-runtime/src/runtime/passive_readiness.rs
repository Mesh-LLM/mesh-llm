//! Readiness output for nodes with no local startup models.

use crate::mesh::Node;
use mesh_llm_events::{OutputEvent, RuntimeStatus, emit_event};

pub(super) fn emit_passive_ready(is_client: bool, node: &Node, local_models: Vec<String>) {
    let native_worker = !is_client && skippy_runtime::native_runtime_loaded();
    let _ = emit_event(OutputEvent::PassiveMode {
        role: if is_client { "client" } else { "standby" }.to_string(),
        status: RuntimeStatus::Ready,
        capacity_gb: native_worker.then(|| node.vram_bytes() as f64 / 1e9),
        models_on_disk: native_worker.then_some(local_models),
        detail: Some(if is_client {
            "Client daemon ready; local model loading is disabled".to_string()
        } else {
            "Runtime daemon ready; no local models are loaded".to_string()
        }),
    });
    super::record_runtime_operational_event(super::RuntimeOperationalEvent::Ready);
}
