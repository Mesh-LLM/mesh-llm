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

#[derive(Debug, Default)]
struct CachedPluginModels {
    models: Vec<String>,
    read_failed: bool,
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
        CachedPluginModels::default()
    };
    let local_stage_statuses = if !matches!(mode, HealthMode::Client) {
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
            status: mesh_status(connectivity),
            admitted_peer_count: connectivity.admitted_peer_count,
            connected_peer_count: connectivity.connected_peer_count,
        },
        serving: ServingHealth {
            status: serving_status,
            models: healthy_models,
        },
    }
}

async fn cached_plugin_models(plugin_manager: &crate::plugin::PluginManager) -> CachedPluginModels {
    // `inference_models` reads the plugin endpoint health snapshot; it does
    // not probe the endpoint. Preserve read failures so a plugin-only host is
    // not misreported as idle when its cached inventory is unavailable.
    match plugin_manager.inference_models().await {
        Ok(models) => CachedPluginModels {
            models,
            read_failed: false,
        },
        Err(_) => CachedPluginModels {
            models: Vec::new(),
            read_failed: true,
        },
    }
}

fn mesh_status(connectivity: crate::mesh::MeshConnectivitySnapshot) -> &'static str {
    if connectivity.connected_peer_count > 0 {
        "connected"
    } else if connectivity.admitted_peer_count > 0 {
        "disconnected"
    } else {
        "standalone"
    }
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
    plugin_models: CachedPluginModels,
) -> (Vec<String>, bool, bool) {
    let mut models = Vec::new();
    let mut has_work = false;
    let mut has_failure = false;
    if matches!(mode, HealthMode::Serving) {
        models.extend(node.hosted_models().await);
        models.extend(plugin_models.models);
        let (process_has_work, process_has_failure) =
            append_healthy_process_models(local_processes, &mut models);
        has_work = !node.serving_models().await.is_empty() || process_has_work;
        has_failure = plugin_models.read_failed || process_has_failure;
    }
    if !matches!(mode, HealthMode::Client) {
        let local_node_id = node.id();
        let local_stages = local_stage_statuses
            .iter()
            .filter(|status| status.node_id == Some(local_node_id))
            .collect::<Vec<_>>();
        has_work |= local_stages
            .iter()
            .any(|status| status.state != crate::inference::skippy::StageRuntimeState::Stopped)
            || !node.serving_models().await.is_empty();
        has_failure |= local_stages
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

fn append_healthy_process_models(
    local_processes: &[crate::runtime_data::RuntimeProcessSnapshot],
    models: &mut Vec<String>,
) -> (bool, bool) {
    use mesh_llm_events::RuntimeStatus;

    let mut has_work = false;
    let mut has_failure = false;
    for process in local_processes {
        match crate::runtime::runtime_status_from_process_status(&process.state) {
            RuntimeStatus::Ready => {
                has_work = true;
                models.push(process.model.clone());
            }
            RuntimeStatus::Exited | RuntimeStatus::Error => has_failure = true,
            RuntimeStatus::ShuttingDown | RuntimeStatus::Stopped => {
                // Graceful drain and shutdown are not failures or active work.
            }
            RuntimeStatus::NotReady
            | RuntimeStatus::Starting
            | RuntimeStatus::Loading
            | RuntimeStatus::Warning => has_work = true,
        }
    }
    (has_work, has_failure)
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
    use super::{
        CachedPluginModels, HealthMode, append_healthy_process_models, health_mode,
        local_serving_state, mesh_status, serving_status,
    };
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

    #[test]
    fn process_statuses_use_the_runtime_mapping() {
        let processes = [
            crate::runtime_data::RuntimeProcessSnapshot {
                model: "ready-model".to_string(),
                state: "ready".to_string(),
                ..Default::default()
            },
            crate::runtime_data::RuntimeProcessSnapshot {
                model: "stopped-model".to_string(),
                state: "stopped".to_string(),
                ..Default::default()
            },
        ];
        let mut models = Vec::new();
        assert_eq!(
            append_healthy_process_models(&processes, &mut models),
            (true, false)
        );
        assert_eq!(models, ["ready-model"]);

        let mut models = Vec::new();
        assert_eq!(
            append_healthy_process_models(
                &[crate::runtime_data::RuntimeProcessSnapshot {
                    model: "exited-model".to_string(),
                    state: "exited".to_string(),
                    ..Default::default()
                }],
                &mut models,
            ),
            (false, true)
        );
        assert!(models.is_empty());

        let mut models = Vec::new();
        assert_eq!(
            append_healthy_process_models(
                &[crate::runtime_data::RuntimeProcessSnapshot {
                    model: "draining-model".to_string(),
                    state: "shutting down".to_string(),
                    ..Default::default()
                }],
                &mut models,
            ),
            (false, false)
        );
        assert!(models.is_empty());
    }

    #[test]
    fn mesh_status_distinguishes_standalone_from_disconnected() {
        use crate::mesh::MeshConnectivitySnapshot;

        assert_eq!(
            mesh_status(MeshConnectivitySnapshot::default()),
            "standalone"
        );
        assert_eq!(
            mesh_status(MeshConnectivitySnapshot {
                admitted_peer_count: 1,
                connected_peer_count: 0,
            }),
            "disconnected"
        );
    }

    #[tokio::test]
    async fn plugin_inventory_failure_is_not_idle() {
        let node = crate::mesh::Node::new_for_tests(NodeRole::Host { http_port: 9337 })
            .await
            .unwrap();
        let plugin_models = CachedPluginModels {
            models: Vec::new(),
            read_failed: true,
        };
        let (models, has_work, has_failure) =
            local_serving_state(&node, HealthMode::Serving, &[], &[], plugin_models).await;
        assert!(models.is_empty());
        assert_eq!(
            serving_status(HealthMode::Serving, false, has_work, has_failure),
            "unhealthy"
        );
    }
}
