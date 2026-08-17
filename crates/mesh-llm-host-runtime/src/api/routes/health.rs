//! Lightweight management liveness and local readiness summary.
//!
//! `GET /health` is intentionally a liveness endpoint: an answering management
//! process returns HTTP 200 even when it has not joined a mesh or is not
//! currently serving a model. The nested fields are advisory readiness signals
//! for operators and infrastructure that wants more detail without fetching
//! the full `/api/status` payload.

use super::super::{MeshApi, http::respond_json};
use crate::mesh::NodeRole;
use serde::Serialize;
use tokio::net::TcpStream;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
enum HealthMode {
    Worker,
    Client,
    Serving,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    /// Management process liveness. This remains `ok` for all mesh/serving
    /// states as long as this endpoint can answer.
    status: &'static str,
    mode: HealthMode,
    mesh: MeshHealth,
    serving: ServingHealth,
}

#[derive(Debug, Serialize)]
struct MeshHealth {
    /// `connected` means the node currently has at least one admitted peer
    /// with a live control connection. Membership alone is not connectivity.
    status: &'static str,
    admitted_peer_count: usize,
    connected_peer_count: usize,
}

#[derive(Debug, Serialize)]
struct ServingHealth {
    /// `healthy` means at least one local model is advertised as an active
    /// HTTP serving target. `degraded` and `unhealthy` expose terminal local
    /// failures without changing the liveness response. `starting` is a
    /// declared local workload that has not reached readiness; `idle` is a
    /// serving-capable node without declared local work. Client mode uses
    /// `not_applicable`; workers report local split-stage health here.
    status: &'static str,
    models: Vec<String>,
}

pub(super) async fn handle(stream: &mut TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    respond_json(stream, 200, &health_response(state).await).await
}

async fn health_response(state: &MeshApi) -> HealthResponse {
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
        &role,
        is_host || runtime_status.is_host,
        is_client || runtime_status.is_client,
    );
    let connectivity = node.connectivity_snapshot().await;
    let plugin_models = if matches!(mode, HealthMode::Serving) {
        cached_plugin_models(&plugin_manager).await
    } else {
        Vec::new()
    };
    let local_stage_statuses = if matches!(mode, HealthMode::Worker) {
        // This is deliberately the cached status map. Health probes must not
        // dial stage peers or ask a local runtime to refresh its status.
        node.stage_runtime_statuses().await
    } else {
        Vec::new()
    };
    let (healthy_models, has_work, has_failure) = local_serving_state(
        &node,
        mode,
        &runtime_status.local_processes,
        &local_stage_statuses,
        plugin_models,
    )
    .await;
    let serving_status = serving_status(mode, !healthy_models.is_empty(), has_work, has_failure);

    HealthResponse {
        status: "ok",
        mode,
        mesh: MeshHealth {
            status: if connectivity.connected_peer_count > 0 {
                "connected"
            } else {
                "disconnected"
            },
            admitted_peer_count: connectivity.admitted_peer_count,
            connected_peer_count: connectivity.connected_peer_count,
        },
        serving: ServingHealth {
            status: serving_status,
            models: healthy_models,
        },
    }
}

async fn cached_plugin_models(plugin_manager: &crate::plugin::PluginManager) -> Vec<String> {
    // `inference_models` reads the plugin endpoint health snapshot; it does
    // not probe the endpoint. Keep this route cache-only and fail closed when
    // the cached inventory is unavailable.
    plugin_manager.inference_models().await.unwrap_or_default()
}

fn health_mode(role: &NodeRole, is_host: bool, is_client: bool) -> HealthMode {
    if is_client || matches!(role, NodeRole::Client) {
        HealthMode::Client
    } else if is_host || matches!(role, NodeRole::Host { .. }) {
        HealthMode::Serving
    } else {
        HealthMode::Worker
    }
}

async fn local_serving_state(
    node: &crate::mesh::Node,
    mode: HealthMode,
    local_processes: &[crate::runtime_data::RuntimeProcessSnapshot],
    local_stage_statuses: &[crate::mesh::StageRuntimeStatus],
    plugin_models: Vec<String>,
) -> (Vec<String>, bool, bool) {
    let mut models = Vec::new();
    let mut has_work = false;
    let mut has_failure = false;
    if matches!(mode, HealthMode::Serving) {
        models.extend(node.hosted_models().await);
        models.extend(plugin_models);
        models.extend(
            local_processes
                .iter()
                .filter(|process| matches!(process.state.as_str(), "ready" | "serving"))
                .map(|process| process.model.clone()),
        );
        has_work = !node.serving_models().await.is_empty() || !local_processes.is_empty();
        has_failure = local_processes.iter().any(|process| {
            matches!(
                process.state.as_str(),
                "error" | "exited" | "failed" | "stopped"
            )
        });
    } else if matches!(mode, HealthMode::Worker) {
        let local_node_id = node.id();
        let local_stages = local_stage_statuses
            .iter()
            .filter(|status| status.node_id == Some(local_node_id))
            .collect::<Vec<_>>();
        has_work = local_stages
            .iter()
            .any(|status| status.state != crate::inference::skippy::StageRuntimeState::Stopped)
            || !node.serving_models().await.is_empty();
        has_failure = local_stages
            .iter()
            .any(|status| status.state == crate::inference::skippy::StageRuntimeState::Failed);
        models.extend(
            local_stages
                .into_iter()
                .filter(|status| status.state == crate::inference::skippy::StageRuntimeState::Ready)
                .map(|status| status.model_id.clone()),
        );
    }
    models.retain(|model| !model.trim().is_empty());
    models.sort();
    models.dedup();
    (models, has_work, has_failure)
}

fn serving_status(
    mode: HealthMode,
    has_healthy_models: bool,
    has_work: bool,
    has_failure: bool,
) -> &'static str {
    if matches!(mode, HealthMode::Client) {
        return "not_applicable";
    }
    if has_failure && has_healthy_models {
        "degraded"
    } else if has_failure {
        "unhealthy"
    } else if has_healthy_models {
        "healthy"
    } else if has_work {
        "starting"
    } else {
        "idle"
    }
}

#[cfg(test)]
mod tests {
    use super::{HealthMode, health_mode, serving_status};
    use crate::mesh::NodeRole;

    #[test]
    fn mode_distinguishes_client_serving_and_worker() {
        assert_eq!(
            health_mode(&NodeRole::Client, false, false),
            HealthMode::Client
        );
        assert_eq!(
            health_mode(&NodeRole::Host { http_port: 9337 }, false, false),
            HealthMode::Serving
        );
        assert_eq!(
            health_mode(&NodeRole::Worker, false, false),
            HealthMode::Worker
        );
    }

    #[test]
    fn serving_status_is_not_applicable_for_non_serving_modes() {
        assert_eq!(
            serving_status(HealthMode::Client, true, true, true),
            "not_applicable"
        );
    }

    #[test]
    fn serving_status_distinguishes_healthy_starting_and_idle() {
        assert_eq!(
            serving_status(HealthMode::Serving, true, true, false),
            "healthy"
        );
        assert_eq!(
            serving_status(HealthMode::Serving, false, true, false),
            "starting"
        );
        assert_eq!(
            serving_status(HealthMode::Serving, false, false, false),
            "idle"
        );
        assert_eq!(
            serving_status(HealthMode::Worker, true, true, false),
            "healthy"
        );
        assert_eq!(
            serving_status(HealthMode::Worker, false, true, true),
            "unhealthy"
        );
        assert_eq!(
            serving_status(HealthMode::Serving, true, true, true),
            "degraded"
        );
    }
}
