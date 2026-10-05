//! Lightweight management liveness and local readiness summary.
//!
//! `GET /health` is intentionally a liveness endpoint: an answering management
//! process returns HTTP 200 even when it has not joined a mesh or is not
//! currently serving a model. The nested fields are advisory readiness signals
//! for operators and infrastructure that wants more detail without fetching
//! the full `/api/status` payload. Like `/api/status`, it is readable on a
//! remotely bound management API and therefore discloses model names and peer
//! counts. GET probes are read-only observations and never enter the management
//! workload lifecycle ledger.

use super::super::{MeshApi, http::respond_json};
use crate::mesh::NodeRole;
use tokio::net::TcpStream;

use mesh_llm_control_api::health::{
    Connectivity, HealthInput, HealthMode, LocalStageHealth, ProcessHealth, health_mode,
    health_response,
};

pub(super) async fn handle(stream: &mut TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    let input = collect_health_input(state).await;
    respond_json(stream, 200, &health_response(input)).await
}

async fn collect_health_input(state: &MeshApi) -> HealthInput {
    let (node, runtime_status, is_host, is_client, plugin_manager) = {
        let inner = state.inner.lock().await;
        (
            inner.node.clone(),
            inner.runtime_data_collector.runtime_status_snapshot(),
            inner.is_host,
            inner.is_client,
            inner.plugin_manager.clone(),
        )
    };
    let role = node.role().await;
    let mode = health_mode(
        is_host || runtime_status.is_host || matches!(role, NodeRole::Host { .. }),
        is_client || runtime_status.is_client || matches!(role, NodeRole::Client),
    );
    let connectivity = node.connectivity_snapshot().await;
    let (plugin_models, plugin_read_failed) = if matches!(mode, HealthMode::Serving) {
        // This reads the plugin health cache, never probes endpoints.
        match plugin_manager.inference_models().await {
            Ok(models) => (models, false),
            Err(_) => (Vec::new(), true),
        }
    } else {
        (Vec::new(), false)
    };
    let local_stages = if !matches!(mode, HealthMode::Client) {
        // Only cached local-node stages contribute to local readiness.
        node.stage_runtime_statuses()
            .await
            .into_iter()
            .filter(|status| status.node_id == Some(node.id()))
            .map(|status| {
                use crate::inference::skippy::StageRuntimeState;
                LocalStageHealth {
                    model: status.model_id,
                    ready: status.state == StageRuntimeState::Ready,
                    stopped: status.state == StageRuntimeState::Stopped,
                    failed: status.state == StageRuntimeState::Failed,
                }
            })
            .collect()
    } else {
        Vec::new()
    };
    let hosted_models = if matches!(mode, HealthMode::Serving) {
        node.hosted_models().await
    } else {
        Vec::new()
    };
    let declared_work =
        !matches!(mode, HealthMode::Client) && !node.serving_models().await.is_empty();
    HealthInput {
        mode,
        connectivity: Connectivity {
            admitted_peer_count: connectivity.admitted_peer_count,
            connected_peer_count: connectivity.connected_peer_count,
        },
        hosted_models,
        plugin_models,
        plugin_read_failed,
        declared_work,
        processes: runtime_status
            .local_processes
            .iter()
            .map(|process| ProcessHealth {
                model: process.model.clone(),
                state: crate::runtime::runtime_status_from_process_status(&process.state),
            })
            .collect(),
        local_stages,
    }
}
