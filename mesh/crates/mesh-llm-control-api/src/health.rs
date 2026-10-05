//! Liveness and advisory readiness views. Constructed from cached observations;
//! rendering never probes peers, runtimes or plugin endpoints.
use mesh_llm_events::RuntimeStatus;
use serde::Serialize;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum HealthMode {
    Worker,
    Client,
    Serving,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
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

/// Connectivity is distinct from membership: an admitted peer may be disconnected.
#[derive(Clone, Copy, Debug, Default)]
pub struct Connectivity {
    pub admitted_peer_count: usize,
    pub connected_peer_count: usize,
}

/// Host process status mapped by the owning runtime before reaching this view.
#[derive(Clone, Debug)]
pub struct ProcessHealth {
    pub model: String,
    pub state: RuntimeStatus,
}

/// Only stages owned by the current node may be supplied. Remote stage state
/// must not make this node ready or unhealthy.
#[derive(Clone, Debug)]
pub struct LocalStageHealth {
    pub model: String,
    pub ready: bool,
    pub stopped: bool,
    pub failed: bool,
}

/// Explicit cached inputs for the management health view.
#[derive(Debug)]
pub struct HealthInput {
    pub mode: HealthMode,
    pub connectivity: Connectivity,
    pub hosted_models: Vec<String>,
    pub plugin_models: Vec<String>,
    pub plugin_read_failed: bool,
    pub declared_work: bool,
    pub processes: Vec<ProcessHealth>,
    pub local_stages: Vec<LocalStageHealth>,
}

/// Client mode takes precedence when role and startup flags overlap.
pub fn health_mode(is_host: bool, is_client: bool) -> HealthMode {
    if is_client {
        HealthMode::Client
    } else if is_host {
        HealthMode::Serving
    } else {
        HealthMode::Worker
    }
}

/// An answering process is live regardless of readiness; callers return HTTP 200.
pub fn health_response(input: HealthInput) -> HealthResponse {
    let mut models = Vec::new();
    let mut has_work = false;
    let mut has_failure = false;
    if matches!(input.mode, HealthMode::Serving) {
        models.extend(input.hosted_models);
        models.extend(input.plugin_models);
        let (process_work, process_failure) =
            append_healthy_process_models(&input.processes, &mut models);
        has_work = input.declared_work || process_work;
        has_failure = input.plugin_read_failed || process_failure;
    }
    if !matches!(input.mode, HealthMode::Client) {
        has_work |= input.local_stages.iter().any(|stage| !stage.stopped) || input.declared_work;
        has_failure |= input.local_stages.iter().any(|stage| stage.failed);
        models.extend(
            input
                .local_stages
                .into_iter()
                .filter(|stage| stage.ready)
                .map(|stage| stage.model),
        );
    }
    models.retain(|model| !model.trim().is_empty());
    models.sort();
    models.dedup();
    HealthResponse {
        status: "ok",
        mode: input.mode,
        mesh: MeshHealth {
            status: mesh_status(input.connectivity),
            admitted_peer_count: input.connectivity.admitted_peer_count,
            connected_peer_count: input.connectivity.connected_peer_count,
        },
        serving: ServingHealth {
            status: serving_status(input.mode, !models.is_empty(), has_work, has_failure),
            models,
        },
    }
}

fn mesh_status(connectivity: Connectivity) -> &'static str {
    if connectivity.connected_peer_count > 0 {
        "connected"
    } else if connectivity.admitted_peer_count > 0 {
        "disconnected"
    } else {
        "standalone"
    }
}

fn append_healthy_process_models(
    processes: &[ProcessHealth],
    models: &mut Vec<String>,
) -> (bool, bool) {
    let mut has_work = false;
    let mut has_failure = false;
    for process in processes {
        match process.state {
            RuntimeStatus::Ready => {
                has_work = true;
                models.push(process.model.clone());
            }
            RuntimeStatus::Exited | RuntimeStatus::Error => has_failure = true,
            RuntimeStatus::ShuttingDown | RuntimeStatus::Stopped => {}
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
    use super::*;
    fn empty_input(mode: HealthMode) -> HealthInput {
        HealthInput {
            mode,
            connectivity: Connectivity::default(),
            hosted_models: Vec::new(),
            plugin_models: Vec::new(),
            plugin_read_failed: false,
            declared_work: false,
            processes: Vec::new(),
            local_stages: Vec::new(),
        }
    }

    #[test]
    fn mode_distinguishes_client_serving_and_worker() {
        assert_eq!(health_mode(false, false), HealthMode::Worker);
        assert_eq!(health_mode(true, false), HealthMode::Serving);
        assert_eq!(health_mode(false, true), HealthMode::Client);
        assert_eq!(health_mode(true, true), HealthMode::Client);
    }

    #[test]
    fn plugin_inventory_failure_is_not_idle() {
        let mut input = empty_input(HealthMode::Serving);
        input.plugin_read_failed = true;
        let response = health_response(input);
        assert_eq!(response.status, "ok");
        assert_eq!(response.serving.status, "unhealthy");
        assert!(response.serving.models.is_empty());
    }

    #[test]
    fn health_json_preserves_liveness_and_combines_local_readiness() {
        let mut input = empty_input(HealthMode::Serving);
        input.connectivity = Connectivity {
            admitted_peer_count: 2,
            connected_peer_count: 1,
        };
        input.hosted_models = vec!["model-b".into(), "model-a".into(), " ".into()];
        input.plugin_models = vec!["model-a".into()];
        input.local_stages = vec![LocalStageHealth {
            model: "failed".into(),
            ready: false,
            stopped: false,
            failed: true,
        }];
        assert_eq!(
            serde_json::to_value(health_response(input)).unwrap(),
            serde_json::json!({
                "status": "ok", "mode": "serving",
                "mesh": { "status": "connected", "admitted_peer_count": 2, "connected_peer_count": 1 },
                "serving": { "status": "degraded", "models": ["model-a", "model-b"] }
            })
        );
    }

    #[test]
    fn worker_uses_local_stages_and_client_ignores_local_serving_inputs() {
        for (mode, expected) in [
            (HealthMode::Worker, "healthy"),
            (HealthMode::Client, "not_applicable"),
        ] {
            let mut input = empty_input(mode);
            input.hosted_models.push("host-only".into());
            input.plugin_read_failed = true;
            input.local_stages.push(LocalStageHealth {
                model: "stage-model".into(),
                ready: true,
                stopped: false,
                failed: false,
            });
            let response = health_response(input);
            assert_eq!(response.serving.status, expected);
            assert_eq!(
                response.serving.models,
                if mode == HealthMode::Worker {
                    vec!["stage-model".to_string()]
                } else {
                    Vec::new()
                }
            );
        }
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
    fn process_statuses_preserve_ready_failed_and_graceful_shutdown_semantics() {
        let processes = [
            ProcessHealth {
                model: "ready-model".to_string(),
                state: RuntimeStatus::Ready,
            },
            ProcessHealth {
                model: "stopped-model".to_string(),
                state: RuntimeStatus::Stopped,
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
                &[ProcessHealth {
                    model: "exited-model".to_string(),
                    state: RuntimeStatus::Exited,
                }],
                &mut models,
            ),
            (false, true)
        );
        assert!(models.is_empty());

        let mut models = Vec::new();
        assert_eq!(
            append_healthy_process_models(
                &[ProcessHealth {
                    model: "draining-model".to_string(),
                    state: RuntimeStatus::ShuttingDown,
                }],
                &mut models,
            ),
            (false, false)
        );
        assert!(models.is_empty());
    }

    #[test]
    fn mesh_status_distinguishes_standalone_from_disconnected() {
        assert_eq!(mesh_status(Connectivity::default()), "standalone");
        assert_eq!(
            mesh_status(Connectivity {
                admitted_peer_count: 1,
                connected_peer_count: 0,
            }),
            "disconnected"
        );
    }
}
