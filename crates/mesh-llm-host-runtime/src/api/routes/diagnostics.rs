use super::super::{
    MeshApi,
    http::{respond_error, respond_json},
};
use mesh_llm_events::audit::{AuditLogFormat, audit_enabled, audit_sink};
use serde::{Deserialize, Serialize};
use tokio::net::TcpStream;
use url::form_urlencoded;

pub(super) async fn handle(
    stream: &mut TcpStream,
    state: &MeshApi,
    path: &str,
) -> anyhow::Result<()> {
    let route = path.split_once('?').map_or(path, |(route, _)| route);
    if route == "/api/diagnostics/split-readiness" {
        return handle_split_readiness(stream, state, path).await;
    }
    if route == "/api/diagnostics" {
        return handle_general_diagnostics(stream, state).await;
    }
    respond_error(stream, 404, "Not found").await
}

async fn handle_split_readiness(
    stream: &mut TcpStream,
    state: &MeshApi,
    path: &str,
) -> anyhow::Result<()> {
    let model_ref = match split_readiness_model_ref(path) {
        Some(model_ref) => model_ref,
        None => {
            return respond_error(stream, 400, "Missing required 'model_ref' query parameter")
                .await;
        }
    };
    let report = state.split_readiness_report(&model_ref).await;
    respond_json(stream, 200, &report).await
}

async fn handle_general_diagnostics(stream: &mut TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    let inner = state.inner.lock().await;
    let node = &inner.node;

    let peers = node.peers().await;
    let served_models = node.models_being_served().await;
    let available_models = node.available_models().await;
    let hosted_models = node.hosted_models().await;
    let requested_models = node.requested_models().await;

    let peer_infos: Vec<PeerDiagnostics> = peers
        .into_iter()
        .map(|p| PeerDiagnostics {
            peer_id: p.id.fmt_short().to_string(),
            endpoint: format!("{:?}", p.addr),
            last_seen_secs_ago: p.last_seen.elapsed().as_secs(),
            models: p.routable_models(),
        })
        .collect();

    let mesh_id = node.mesh_id().await;

    let diagnostics = DiagnosticsResponse {
        node_id: inner.node.id().fmt_short().to_string(),
        mesh_id: mesh_id.map(|m| m.to_string()),
        mesh_name: inner.mesh_name.clone(),
        mesh_region: inner.mesh_region.clone(),
        runtime_status: get_runtime_status(&inner),
        served_models,
        available_models,
        hosted_models,
        requested_models,
        peer_count: peer_infos.len(),
        peers: peer_infos,
        audit_logging: get_audit_logging_status(&inner),
        timestamp: chrono::Utc::now().to_rfc3339(),
    };

    respond_json(stream, 200, &diagnostics).await
}

fn split_readiness_model_ref(path: &str) -> Option<String> {
    let (_, raw_query) = path.split_once('?')?;
    for (key, value) in form_urlencoded::parse(raw_query.as_bytes()) {
        if matches!(key.as_ref(), "model_ref" | "model") {
            let value = value.trim();
            if !value.is_empty() {
                return Some(value.to_string());
            }
        }
    }
    None
}

fn get_runtime_status(inner: &crate::api::state::ApiInner) -> serde_json::Value {
    serde_json::json!({
        "is_host": inner.is_host,
        "is_client": inner.is_client,
        "headless": inner.headless,
        "llama_ready": inner.llama_ready,
        "listeners_ready": inner.listeners_ready,
        "llama_port": inner.llama_port,
        "api_port": inner.api_port,
    })
}

fn get_audit_logging_status(_inner: &crate::api::state::ApiInner) -> serde_json::Value {
    if !audit_enabled() {
        return serde_json::json!({ "enabled": false });
    }

    let Some(sink) = audit_sink() else {
        return serde_json::json!({
            "enabled": false,
            "degraded": true,
            "reason": "sink_unavailable",
        });
    };
    let format = match sink.format() {
        AuditLogFormat::JsonLines => "json_lines",
    };
    let min_level = sink.min_level().as_str();

    serde_json::json!({
        "enabled": true,
        "format": format,
        "min_level": min_level,
    })
}

#[derive(Serialize, Deserialize)]
struct DiagnosticsResponse {
    node_id: String,
    mesh_id: Option<String>,
    mesh_name: Option<String>,
    mesh_region: Option<String>,
    runtime_status: serde_json::Value,
    served_models: Vec<String>,
    available_models: Vec<String>,
    hosted_models: Vec<String>,
    requested_models: Vec<String>,
    peer_count: usize,
    peers: Vec<PeerDiagnostics>,
    audit_logging: serde_json::Value,
    timestamp: String,
}

#[derive(Serialize, Deserialize)]
struct PeerDiagnostics {
    peer_id: String,
    endpoint: String,
    last_seen_secs_ago: u64,
    models: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::split_readiness_model_ref;

    #[test]
    fn split_readiness_query_accepts_percent_encoded_model_ref() {
        assert_eq!(
            split_readiness_model_ref(
                "/api/diagnostics/split-readiness?model_ref=meshllm%2FQwen3-8B-Q4_K_M-layers"
            ),
            Some("meshllm/Qwen3-8B-Q4_K_M-layers".to_string())
        );
    }

    #[test]
    fn split_readiness_query_rejects_blank_model_ref() {
        assert_eq!(
            split_readiness_model_ref("/api/diagnostics/split-readiness?model_ref=%20"),
            None
        );
    }
}
