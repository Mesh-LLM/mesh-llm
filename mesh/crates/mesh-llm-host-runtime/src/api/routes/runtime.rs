use super::super::{
    MeshApi, RuntimeControlRequest,
    http::{respond_error, respond_json, respond_runtime_error},
    status::decode_runtime_model_path,
};
use crate::mesh::encode_endpoint_addr_token;
use super::control_apply_diagnostics::{
    LocalControlApplyDiagnosticPayload, local_control_apply_diagnostic_payload,
    local_control_apply_diagnostic_payload_from_local,
};
use super::runtime_control_state::collect_runtime_config_control_state_payload;
mod owner_control_credential;
mod plugin_approval;
use super::runtime_load_request::parse_runtime_load_request;
use crate::config_schema::{EngineConfigSchemaDescriptor, export_runtime_config_schema_reference};
use crate::crypto::{
    OwnerKeychainLoadError, keystore_metadata, load_keystore, load_owner_keypair_from_keychain,
};
use crate::plugin::validate_config_diagnostics_with_installed_plugin_schemas;
use futures_util::{FutureExt, StreamExt, stream};
use mesh_client::{
    ClientBuilder, ControlPlaneBootstrapOptions, ControlPlaneClientError, ControlPlaneConnection,
    InviteToken, OwnerControlRemoteError, client::control_plane::OwnerControlScanRefreshResult,
};
use mesh_llm_config::{
    ConfigConditionValue, ConfigControlAvailabilitySource, ConfigDisabledWritePolicy,
    ConfigOptionsSource,
};
use mesh_llm_config::{ConfigDiagnosticSeverity, legacy_validation_error_text};
use mesh_llm_node::serving::{UnloadOptions, UnloadTarget};
use plugin_approval::RawPluginCommandApproval;
use serde::{Deserialize, Serialize};
use skippy_inference_api::GuardrailMode;
use std::collections::BTreeMap;
use std::net::SocketAddr;

use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;
use zeroize::Zeroizing;

const CONTROL_KV_CACHE_MAX_ENDPOINTS: usize = 256;
const CONTROL_KV_CACHE_CONCURRENCY: usize = 8;
const CONTROL_KV_CACHE_BATCH_TIMEOUT_SECS: u64 = 45;
/// A route future that is already on the heap.
///
/// In a debug build, a `match` whose arms each await an `async fn` reserves one
/// stack slot per arm, sized by that arm's future, and the slots are not
/// overlapped: the router's frame is the sum of its routes rather than the
/// largest one. Returning an already-boxed future leaves a pointer in the
/// caller instead. Same shape as `DispatchRequestFn` in the parent module.
/// Measured on #1915: this takes `handle_get` from 332 KiB to under 4 KiB and
/// the whole `/api/status` poll chain from 1904 KiB to 1197 KiB.
type BoxedRouteFuture<'a> =
    std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<()>> + Send + 'a>>;

pub(super) async fn handle(
    stream: &mut TcpStream,
    state: &MeshApi,
    method: &str,
    path_only: &str,
    body: &str,
    raw_request: &[u8],
) -> anyhow::Result<()> {
    if requires_owner_control_credential(method, path_only)
        && !ensure_owner_control_caller(stream, state, raw_request).await?
    {
        return Ok(());
    }
    match method {
        "GET" => handle_get(stream, state, path_only).await,
        "POST" => handle_post(stream, state, path_only, body).await,
        "PUT" => handle_put(stream, state, path_only, body).await,
        "DELETE" => handle_delete(stream, state, path_only, body).await,
        _ => Ok(()),
    }
}

/// Owner-control routes make the host act as the owner on the caller's
/// behalf, so they require the owner-control bearer credential on top of
/// loopback reachability. Reads that expose no secret material (bootstrap,
/// schema, control-state, validate, kv-cache status) stay loopback-only.
fn requires_owner_control_credential(method: &str, path_only: &str) -> bool {
    matches!(
        (method, path_only),
        ("POST", "/api/runtime/control/get-config")
            | ("POST", "/api/runtime/control/scan-refresh")
            | ("POST", "/api/runtime/control/refresh-inventory")
            | ("POST", "/api/runtime/control/apply-config")
            | ("POST", "/api/runtime/control/kv-cache")
            | ("POST", "/api/runtime/control/load-model")
            | ("POST", "/api/runtime/control/unload-model")
            | ("POST", "/api/runtime/control/ensure-model")
            | ("POST", "/api/runtime/control/drain-model")
            | ("POST", "/api/runtime/models")
            | ("POST", "/api/runtime/kv-cache/prune")
            | ("POST", "/api/runtime/mesh-guardrails")
            | ("POST", "/api/runtime/control/fleet")
            | ("POST", "/api/runtime/control/mesh-policy")
            | ("DELETE", "/api/runtime/kv-cache")
        | ("DELETE", "/api/runtime/control/mesh-policy")
    ) || (method == "DELETE"
        && (path_only.starts_with("/api/runtime/models/")
            || path_only.starts_with("/api/runtime/instances/")))
}

async fn handle_get(
    stream: &mut TcpStream,
    state: &MeshApi,
    path_only: &str,
) -> anyhow::Result<()> {
    match path_only {
        "/api/status" => handle_status(stream, state).await,
        "/api/models" => handle_models(stream, state).await,
        "/api/runtime" => handle_runtime_status(stream, state).await,
        "/api/runtime/llama" => handle_runtime_llama(stream, state).await,
        "/api/runtime/events" => handle_runtime_events(stream, state).await,
        "/api/runtime/endpoints" => handle_runtime_endpoints(stream, state).await,
        "/api/runtime/processes" => handle_runtime_processes(stream, state).await,
        "/api/runtime/stages" => handle_runtime_stages(stream, state).await,
        "/api/runtime/config-schema" => handle_runtime_config_schema(stream).await,
        "/api/runtime/config-control-state" => {
            handle_runtime_config_control_state(stream, state).await
        }
        "/api/runtime/kv-cache" => super::kv_cache::handle_status(stream).await,
        "/api/runtime/control-bootstrap" => handle_control_bootstrap(stream, state).await,
        "/api/runtime/control/owner-control-token" => {
            owner_control_credential::reject_token_disclosure(stream).await
        }
        "/api/runtime/intents" => handle_get_intents(stream, state).await,
        "/api/runtime/activity" => super::runtime_activity::handle_get(stream, state).await,
        "/api/runtime/fleet/metrics" => handle_fleet_metrics(stream, state).await,
        "/api/runtime/control/mesh-policy" => handle_get_mesh_policy(stream, state).await,
        "/api/events" => handle_events(stream, state).await,
        _ => Ok(()),
    }
}

async fn handle_post(
    stream: &mut TcpStream,
    state: &MeshApi,
    path_only: &str,
    body: &str,
) -> anyhow::Result<()> {
    match path_only {
        "/api/runtime/control/get-config" => handle_control_get_config(stream, state, body).await,
        "/api/runtime/control/scan-refresh" => {
            handle_control_scan_refresh(stream, state, body).await
        }
        "/api/runtime/control/refresh-inventory" => {
            handle_control_refresh_inventory(stream, state, body).await
        }
        "/api/runtime/control/apply-config" => {
            handle_control_apply_config(stream, state, body).await
        }
        "/api/runtime/control/kv-cache" => handle_control_kv_cache(stream, state, body).await,
        "/api/runtime/control/load-model" => handle_control_load_model(stream, state, body).await,
        "/api/runtime/control/unload-model" => {
            handle_control_unload_model(stream, state, body).await
        }
        "/api/runtime/control/ensure-model" => {
            handle_control_ensure_model(stream, state, body).await
        }
        "/api/runtime/control/drain-model" => handle_control_drain_model(stream, state, body).await,
        "/api/runtime/config/validate" => handle_runtime_config_validate(stream, body).await,
        "/api/runtime/mesh-guardrails" => handle_set_mesh_guardrails(stream, state, body).await,
        "/api/runtime/models" => handle_load_model(stream, state, body).await,
        "/api/runtime/kv-cache/prune" => super::kv_cache::handle_prune(stream, body).await,
        "/api/runtime/control/fleet" => handle_control_fleet(stream, state, body).await,
        "/api/runtime/control/mesh-policy" => handle_set_mesh_policy(stream, state, body).await,
        _ => Ok(()),
    }
}

async fn handle_put(
    stream: &mut TcpStream,
    state: &MeshApi,
    path_only: &str,
    body: &str,
) -> anyhow::Result<()> {
    match path_only {
        "/api/runtime/activity/override" => {
            super::runtime_activity::handle_put(stream, state, body).await
        }
        _ => Ok(()),
    }
}

async fn handle_delete(
    stream: &mut TcpStream,
    state: &MeshApi,
    path_only: &str,
    body: &str,
) -> anyhow::Result<()> {
    match path_only {
        "/api/runtime/activity/override" => {
            super::runtime_activity::handle_delete(stream, state).await
        }
        "/api/runtime/kv-cache" => super::kv_cache::handle_clear(stream, body).await,
        "/api/runtime/control/mesh-policy" => {
            handle_control_clear_mesh_policy(stream, state).await
        }
        p if p.starts_with("/api/runtime/instances/") => {
            handle_unload_instance(stream, state, p).await
        }
        p if p.starts_with("/api/runtime/models/") => handle_unload_model(stream, state, p).await,
        _ => Ok(()),
    }
}

async fn handle_control_clear_mesh_policy(
    stream: &mut TcpStream,
    state: &MeshApi,
) -> anyhow::Result<()> {
    state.policy_store().clear().await;
    respond_json(stream, 200, &serde_json::json!({"cleared": true})).await
}

fn handle_control_bootstrap<'a>(
    stream: &'a mut TcpStream,
    state: &'a MeshApi,
) -> BoxedRouteFuture<'a> {
    Box::pin(handle_control_bootstrap_inner(stream, state))
}

async fn handle_control_bootstrap_inner(
    stream: &mut TcpStream,
    state: &MeshApi,
) -> anyhow::Result<()> {
    if !ensure_loopback_control_caller(stream).await? {
        return Ok(());
    }
    respond_json(stream, 200, &state.control_bootstrap().await).await
}

fn handle_runtime_config_schema<'a>(stream: &'a mut TcpStream) -> BoxedRouteFuture<'a> {
    Box::pin(handle_runtime_config_schema_inner(stream))
}

async fn handle_runtime_config_schema_inner(stream: &mut TcpStream) -> anyhow::Result<()> {
    if !ensure_loopback_control_caller(stream).await? {
        return Ok(());
    }

    match export_runtime_config_schema_reference(std::iter::empty::<EngineConfigSchemaDescriptor>())
    {
        Ok(schema) => respond_json(stream, 200, &schema).await,
        Err(error) => respond_error(stream, 500, &error.to_string()).await,
    }
}

fn handle_runtime_config_control_state<'a>(
    stream: &'a mut TcpStream,
    state: &'a MeshApi,
) -> BoxedRouteFuture<'a> {
    Box::pin(handle_runtime_config_control_state_inner(stream, state))
}

async fn handle_runtime_config_control_state_inner(
    stream: &mut TcpStream,
    state: &MeshApi,
) -> anyhow::Result<()> {
    if !ensure_loopback_control_caller(stream).await? {
        return Ok(());
    }

    respond_json(
        stream,
        200,
        &collect_runtime_config_control_state_payload(state).await,
    )
    .await
}

#[derive(Debug, Deserialize)]
struct ControlEndpointRequest {
    endpoint: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ApplyConfigRequest {
    endpoint: Option<String>,
    expected_revision: u64,
    config: crate::plugin::MeshConfig,
    plugin_command_approvals: Vec<RawPluginCommandApproval>,
}

#[derive(Debug, Deserialize)]
struct RawApplyConfigRequest {
    endpoint: Option<String>,
    expected_revision: u64,
    config: serde_json::Value,
    #[serde(default)]
    plugin_command_approvals: Vec<RawPluginCommandApproval>,
}

#[derive(Debug, Deserialize)]
struct ControlKvCacheRequest {
    endpoints: Vec<String>,
    operation: String,
    target_bytes: Option<u64>,
    model_identity: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ValidateConfigRequest {
    toml: String,
    path: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAiGuardrailsModeRequest {
    mode: String,
}

#[derive(Debug, Deserialize)]
struct ControlLifecycleModelRequest {
    endpoint: Option<String>,
    model: Option<String>,
    instance_id: Option<String>,
    profile: Option<String>,
}

#[derive(Clone, Copy)]
enum ControlLifecycleOperation {
    Load,
    Unload,
    Ensure,
    Drain,
}

struct ControlLifecycleTarget {
    model: String,
    instance_id: Option<String>,
    profile: Option<String>,
}

#[derive(Debug, Serialize)]
struct LocalControlSnapshotPayload {
    node_id: String,
    revision: u64,
    config_hash: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    hostname: Option<String>,
    config: crate::plugin::MeshConfig,
}

#[derive(Debug, Serialize)]
struct LocalControlScanRefreshPayload {
    target_node_id: String,
    disposition: Option<String>,
    inventory: Option<Vec<LocalControlInventoryEntryPayload>>,
}

#[derive(Debug, Serialize)]
struct LocalControlInventoryEntryPayload {
    canonical_model_ref: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    display_name: Option<String>,
    total_size_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<LocalControlModelMetadataPayload>,
}

#[derive(Debug, Serialize)]
struct LocalControlModelMetadataPayload {
    model_key: String,
    context_length: u32,
    vocab_size: u32,
    embedding_size: u32,
    head_count: u32,
    layer_count: u32,
    feed_forward_length: u32,
    key_length: u32,
    value_length: u32,
    architecture: String,
    tokenizer_model_name: String,
    special_tokens: Vec<LocalControlSpecialTokenPayload>,
    rope_scale: f32,
    rope_freq_base: f32,
    is_moe: bool,
    expert_count: u32,
    used_expert_count: u32,
    quantization_type: String,
    kv_head_count: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameter_size: Option<String>,
}

#[derive(Debug, Serialize)]
struct LocalControlSpecialTokenPayload {
    name: String,
    token_id: u32,
}

#[derive(Debug, Serialize)]
struct LocalControlApplyPayload {
    success: bool,
    current_revision: u64,
    config_hash: String,
    apply_mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    diagnostics: Vec<LocalControlApplyDiagnosticPayload>,
}

#[derive(Debug, Serialize)]
struct LocalConfigValidatePayload {
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    diagnostics: Vec<LocalControlApplyDiagnosticPayload>,
}

#[derive(Debug, Serialize)]
struct LocalControlErrorPayload {
    code: String,
    message: String,
    legacy_retry_allowed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    current_revision: Option<u64>,
}

#[derive(Debug, Serialize)]
struct LocalControlKvCachePayload {
    results: Vec<LocalControlKvCacheResult>,
}

#[derive(Debug, Serialize)]
struct LocalControlKvCacheResult {
    endpoint: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    target_node_id: Option<String>,
    operation: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    freed_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    status: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<LocalControlErrorPayload>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct ConfigControlStatePayload {
    #[serde(default)]
    pub(crate) settings: BTreeMap<String, ConfigControlStateEntry>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ConfigControlStateEntry {
    pub(crate) enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) note: Option<String>,
    pub(crate) source: ConfigControlAvailabilitySource,
    pub(crate) write_policy: ConfigDisabledWritePolicy,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) options: Option<Vec<ConfigControlOption>>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ConfigControlOption {
    pub(crate) value: ConfigConditionValue,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) note: Option<String>,
    pub(crate) disabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) reason: Option<String>,
    pub(crate) source: ConfigOptionsSource,
}

async fn handle_control_get_config(
    stream: &mut TcpStream,
    state: &MeshApi,
    body: &str,
) -> anyhow::Result<()> {
    if !ensure_loopback_control_caller(stream).await? {
        return Ok(());
    }
    let request: ControlEndpointRequest = match serde_json::from_str(body) {
        Ok(request) => request,
        Err(_) => return respond_error(stream, 400, "Invalid JSON body").await,
    };
    let endpoint = match required_control_endpoint(request.endpoint) {
        Ok(endpoint) => endpoint,
        Err(error) => return respond_control_error(stream, error).await,
    };
    match connect_owner_control_client(state, &endpoint).await {
        Ok(client) => {
            let result = client.get_config().await;
            client.close().await;
            match result {
                Ok(snapshot) => respond_json(
                    stream,
                    200,
                    &serde_json::json!({ "snapshot": local_control_snapshot_payload(snapshot) }),
                )
                .await,
                Err(error) => respond_control_error(stream, control_error_from_client(error)).await,
            }
        }
        Err(error) => respond_control_error(stream, error).await,
    }
}

async fn handle_control_refresh_inventory(
    stream: &mut TcpStream,
    state: &MeshApi,
    body: &str,
) -> anyhow::Result<()> {
    if !ensure_loopback_control_caller(stream).await? {
        return Ok(());
    }
    let request: ControlEndpointRequest = match serde_json::from_str(body) {
        Ok(request) => request,
        Err(_) => return respond_error(stream, 400, "Invalid JSON body").await,
    };
    let endpoint = match required_control_endpoint(request.endpoint) {
        Ok(endpoint) => endpoint,
        Err(error) => return respond_control_error(stream, error).await,
    };
    match connect_owner_control_client(state, &endpoint).await {
        Ok(client) => {
            let result = client.refresh_inventory().await;
            client.close().await;
            match result {
                Ok(snapshot) => respond_json(
                    stream,
                    200,
                    &serde_json::json!({ "snapshot": local_control_snapshot_payload(snapshot) }),
                )
                .await,
                Err(error) => respond_control_error(stream, control_error_from_client(error)).await,
            }
        }
        Err(error) => respond_control_error(stream, error).await,
    }
}

async fn handle_control_scan_refresh(
    stream: &mut TcpStream,
    state: &MeshApi,
    body: &str,
) -> anyhow::Result<()> {
    if !ensure_loopback_control_caller(stream).await? {
        return Ok(());
    }
    let request: ControlEndpointRequest = match serde_json::from_str(body) {
        Ok(request) => request,
        Err(_) => return respond_error(stream, 400, "Invalid JSON body").await,
    };
    let endpoint = match required_control_endpoint(request.endpoint) {
        Ok(endpoint) => endpoint,
        Err(error) => return respond_control_error(stream, error).await,
    };
    match connect_owner_control_client(state, &endpoint).await {
        Ok(client) => {
            let result = client.scan_refresh().await;
            client.close().await;
            match result {
                Ok(response) => {
                    respond_json(stream, 200, &local_control_scan_refresh_payload(response)).await
                }
                Err(error) => respond_control_error(stream, control_error_from_client(error)).await,
            }
        }
        Err(error) => respond_control_error(stream, error).await,
    }
}

async fn handle_control_apply_config(
    stream: &mut TcpStream,
    state: &MeshApi,
    body: &str,
) -> anyhow::Result<()> {
    if !ensure_loopback_control_caller(stream).await? {
        return Ok(());
    }
    let raw_request: RawApplyConfigRequest = match serde_json::from_str(body) {
        Ok(request) => request,
        Err(_) => return respond_error(stream, 400, "Invalid JSON body").await,
    };
    let raw_config_toml = toml::to_string(&raw_request.config).ok();
    let request = ApplyConfigRequest {
        endpoint: raw_request.endpoint,
        expected_revision: raw_request.expected_revision,
        config: match serde_json::from_value(raw_request.config) {
            Ok(config) => config,
            Err(_) => return respond_error(stream, 400, "Invalid JSON body").await,
        },
        plugin_command_approvals: raw_request.plugin_command_approvals,
    };
    let endpoint = match required_control_endpoint(request.endpoint) {
        Ok(endpoint) => endpoint,
        Err(error) => return respond_control_error(stream, error).await,
    };
    let diagnostics = validate_config_diagnostics_with_installed_plugin_schemas(
        &request.config,
        raw_config_toml.as_deref(),
    );
    if diagnostics
        .iter()
        .any(|diagnostic| diagnostic.severity == ConfigDiagnosticSeverity::Error)
    {
        return respond_json(
            stream,
            200,
            &LocalControlApplyPayload {
                success: false,
                current_revision: request.expected_revision,
                config_hash: hex::encode(crate::protocol::convert::canonical_config_hash(
                    &crate::protocol::convert::mesh_config_to_proto(&request.config),
                )),
                apply_mode: "unspecified".to_string(),
                error: Some(legacy_validation_error_text(&diagnostics)),
                diagnostics: diagnostics
                    .iter()
                    .map(local_control_apply_diagnostic_payload_from_local)
                    .collect(),
            },
        )
        .await;
    }
    match connect_owner_control_client(state, &endpoint).await {
        Ok(client) => {
            let result = client
                .apply_config(
                    request.expected_revision,
                    crate::protocol::convert::mesh_config_to_proto(&request.config),
                    request
                        .plugin_command_approvals
                        .iter()
                        .map(RawPluginCommandApproval::to_proto)
                        .collect(),
                )
                .await;
            client.close().await;
            match result {
                Ok(response) => {
                    respond_json(
                        stream,
                        200,
                        &LocalControlApplyPayload {
                            success: response.success,
                            current_revision: response.current_revision,
                            config_hash: hex::encode(response.config_hash),
                            apply_mode: control_apply_mode_label(response.apply_mode),
                            error: response.error,
                            diagnostics: response
                                .diagnostics
                                .iter()
                                .map(local_control_apply_diagnostic_payload)
                                .collect(),
                        },
                    )
                    .await
                }
                Err(error) => respond_control_error(stream, control_error_from_client(error)).await,
            }
        }
        Err(error) => respond_control_error(stream, error).await,
    }
}

async fn handle_control_kv_cache(
    stream: &mut TcpStream,
    state: &MeshApi,
    body: &str,
) -> anyhow::Result<()> {
    if !ensure_loopback_control_caller(stream).await? {
        return Ok(());
    }
    let request: ControlKvCacheRequest = match serde_json::from_str(body) {
        Ok(request) => request,
        Err(_) => return respond_error(stream, 400, "Invalid JSON body").await,
    };
    if request.endpoints.is_empty() {
        return respond_error(
            stream,
            400,
            "at least one owner-control endpoint is required",
        )
        .await;
    }
    if request.endpoints.len() > CONTROL_KV_CACHE_MAX_ENDPOINTS {
        return respond_error(
            stream,
            400,
            "at most 256 owner-control endpoints are allowed",
        )
        .await;
    }
    let operation = match request.operation.as_str() {
        "status" => mesh_client::proto::node::OwnerControlKvCacheOperation::Status,
        "prune" => mesh_client::proto::node::OwnerControlKvCacheOperation::Prune,
        "clear" => mesh_client::proto::node::OwnerControlKvCacheOperation::Clear,
        _ => return respond_error(stream, 400, "unknown kv-cache operation").await,
    };
    let endpoints = request.endpoints;
    let timeout_endpoints = endpoints.clone();
    let operation_label = request.operation.clone();
    let mut operations = stream::iter(endpoints.into_iter().enumerate())
        .map(|(index, endpoint)| {
            execute_control_kv_cache_target(
                state,
                endpoint,
                request.operation.clone(),
                operation,
                request.target_bytes,
                request.model_identity.clone(),
            )
            .map(move |result| (index, result))
        })
        .buffer_unordered(CONTROL_KV_CACHE_CONCURRENCY);
    let deadline = tokio::time::Instant::now()
        + std::time::Duration::from_secs(CONTROL_KV_CACHE_BATCH_TIMEOUT_SECS);
    let mut indexed_results = Vec::with_capacity(timeout_endpoints.len());
    loop {
        match tokio::time::timeout_at(deadline, operations.next()).await {
            Ok(Some(result)) => indexed_results.push(result),
            Ok(None) => break,
            Err(_) => break,
        }
    }
    drop(operations);
    if indexed_results.len() < timeout_endpoints.len() {
        let completed = indexed_results
            .iter()
            .map(|(index, _)| *index)
            .collect::<std::collections::HashSet<_>>();
        indexed_results.extend(
            timeout_endpoints
                .into_iter()
                .enumerate()
                .filter(|(index, _)| !completed.contains(index))
                .map(|(index, endpoint)| {
                    (
                        index,
                        LocalControlKvCacheResult {
                            endpoint,
                            target_node_id: None,
                            operation: operation_label.clone(),
                            freed_bytes: None,
                            status: None,
                            error: Some(LocalControlErrorPayload {
                                code: "control_timeout".to_string(),
                                message: format!(
                                    "kv-cache batch exceeded its {CONTROL_KV_CACHE_BATCH_TIMEOUT_SECS}-second deadline before a receipt was returned"
                                ),
                                legacy_retry_allowed: false,
                                current_revision: None,
                            }),
                        },
                    )
                }),
        );
    }
    indexed_results.sort_by_key(|(index, _)| *index);
    let results = indexed_results
        .into_iter()
        .map(|(_, result)| result)
        .collect();
    respond_json(stream, 200, &LocalControlKvCachePayload { results }).await
}

async fn execute_control_kv_cache_target(
    state: &MeshApi,
    endpoint: String,
    operation_label: String,
    operation: mesh_client::proto::node::OwnerControlKvCacheOperation,
    target_bytes: Option<u64>,
    model_identity: Option<String>,
) -> LocalControlKvCacheResult {
    let mut result = LocalControlKvCacheResult {
        endpoint: endpoint.clone(),
        target_node_id: None,
        operation: operation_label,
        freed_bytes: None,
        status: None,
        error: None,
    };
    match connect_owner_control_client(state, &endpoint).await {
        Ok(client) => {
            result.target_node_id = Some(hex::encode(client.target_node_id()));
            let response = client
                .kv_cache(operation, target_bytes, model_identity)
                .await;
            client.close().await;
            match response {
                Ok(response) => {
                    result.freed_bytes = response.freed_bytes;
                    match serde_json::from_slice(&response.status_json) {
                        Ok(status) => result.status = Some(status),
                        Err(error) => {
                            result.error = Some(control_error_from_anyhow(anyhow::anyhow!(
                                "invalid kv-cache status from target: {error}"
                            )));
                        }
                    }
                }
                Err(error) => result.error = Some(control_error_from_client(error)),
            }
        }
        Err(error) => result.error = Some(error),
    }
    result
}

async fn handle_control_load_model(
    stream: &mut TcpStream,
    state: &MeshApi,
    body: &str,
) -> anyhow::Result<()> {
    handle_control_lifecycle_model(stream, state, body, ControlLifecycleOperation::Load).await
}

async fn handle_control_unload_model(
    stream: &mut TcpStream,
    state: &MeshApi,
    body: &str,
) -> anyhow::Result<()> {
    handle_control_lifecycle_model(stream, state, body, ControlLifecycleOperation::Unload).await
}

async fn handle_control_ensure_model(
    stream: &mut TcpStream,
    state: &MeshApi,
    body: &str,
) -> anyhow::Result<()> {
    handle_control_lifecycle_model(stream, state, body, ControlLifecycleOperation::Ensure).await
}

async fn handle_control_drain_model(
    stream: &mut TcpStream,
    state: &MeshApi,
    body: &str,
) -> anyhow::Result<()> {
    handle_control_lifecycle_model(stream, state, body, ControlLifecycleOperation::Drain).await
}

async fn handle_control_lifecycle_model(
    stream: &mut TcpStream,
    state: &MeshApi,
    body: &str,
    operation: ControlLifecycleOperation,
) -> anyhow::Result<()> {
    if !ensure_loopback_control_caller(stream).await? {
        return Ok(());
    }
    let request: ControlLifecycleModelRequest = match serde_json::from_str(body) {
        Ok(request) => request,
        Err(_) => return respond_error(stream, 400, "Invalid JSON body").await,
    };
    let endpoint = request.endpoint.clone();
    let target = match validate_control_lifecycle_target(operation, request) {
        Ok(target) => target,
        Err(message) => return respond_error(stream, 400, message).await,
    };
    let endpoint = match required_control_endpoint(endpoint) {
        Ok(endpoint) => endpoint,
        Err(error) => return respond_control_error(stream, error).await,
    };
    match connect_owner_control_client(state, &endpoint).await {
        Ok(client) => {
            let result = invoke_control_lifecycle(&client, operation, target).await;
            client.close().await;
            match result {
                Ok(response) => respond_json(stream, 200, &response).await,
                Err(error) => respond_control_error(stream, control_error_from_client(error)).await,
            }
        }
        Err(error) => respond_control_error(stream, error).await,
    }
}

fn validate_control_lifecycle_target(
    operation: ControlLifecycleOperation,
    request: ControlLifecycleModelRequest,
) -> Result<ControlLifecycleTarget, &'static str> {
    let ControlLifecycleModelRequest {
        model,
        instance_id,
        profile,
        ..
    } = request;
    match operation {
        ControlLifecycleOperation::Load | ControlLifecycleOperation::Ensure => {
            match (model, instance_id) {
                (Some(model), None) if !model.trim().is_empty() => Ok(ControlLifecycleTarget {
                    model,
                    instance_id: None,
                    profile,
                }),
                _ if matches!(operation, ControlLifecycleOperation::Load) => {
                    Err("load-model requires a model reference and does not accept an instance id")
                }
                _ => Err(
                    "ensure-model requires a model reference and does not accept an instance id",
                ),
            }
        }
        ControlLifecycleOperation::Unload | ControlLifecycleOperation::Drain => {
            let target = match (model, instance_id, profile) {
                (Some(model), None, None) if !model.trim().is_empty() => {
                    Some(ControlLifecycleTarget {
                        model,
                        instance_id: None,
                        profile: None,
                    })
                }
                (None, Some(instance_id), None) if !instance_id.trim().is_empty() => {
                    Some(ControlLifecycleTarget {
                        model: String::new(),
                        instance_id: Some(instance_id),
                        profile: None,
                    })
                }
                _ => None,
            };
            target.ok_or({
                if matches!(operation, ControlLifecycleOperation::Unload) {
                    "unload-model requires exactly one model reference or instance id"
                } else {
                    "drain-model requires exactly one model reference or instance id"
                }
            })
        }
    }
}

async fn invoke_control_lifecycle(
    client: &mesh_client::OwnerControlClient,
    operation: ControlLifecycleOperation,
    target: ControlLifecycleTarget,
) -> Result<serde_json::Value, ControlPlaneClientError> {
    let (intent_id, accepted_state, response_target) = match operation {
        ControlLifecycleOperation::Load => {
            let response = client.load_model(target.model, target.profile).await?;
            (response.intent_id, response.accepted_state, response.target)
        }
        ControlLifecycleOperation::Unload => {
            let response = client
                .unload_model(target.model, target.instance_id)
                .await?;
            (response.intent_id, response.accepted_state, response.target)
        }
        ControlLifecycleOperation::Ensure => {
            let response = client.ensure_model(target.model, target.profile).await?;
            (response.intent_id, response.accepted_state, response.target)
        }
        ControlLifecycleOperation::Drain => {
            let response = client.drain_model(target.model, target.instance_id).await?;
            (response.intent_id, response.accepted_state, response.target)
        }
    };
    Ok(serde_json::json!({
        "accepted": true,
        "intent_id": intent_id,
        "accepted_state": accepted_state,
        "model": response_target.as_ref().map(|target| &target.canonical_model_ref),
        "instance_id": response_target.as_ref().and_then(|target| target.instance_id.as_deref()),
    }))
}

async fn handle_runtime_config_validate(stream: &mut TcpStream, body: &str) -> anyhow::Result<()> {
    if !ensure_loopback_control_caller(stream).await? {
        return Ok(());
    }

    let request: ValidateConfigRequest = match serde_json::from_str(body) {
        Ok(request) => request,
        Err(_) => return respond_error(stream, 400, "Invalid JSON body").await,
    };

    let config: crate::plugin::MeshConfig = match toml::from_str(&request.toml) {
        Ok(config) => config,
        Err(error) => {
            return respond_json(
                stream,
                200,
                &LocalConfigValidatePayload {
                    ok: false,
                    path: request.path,
                    error: Some(format!("Invalid config TOML: {error}")),
                    diagnostics: Vec::new(),
                },
            )
            .await;
        }
    };

    let diagnostics =
        validate_config_diagnostics_with_installed_plugin_schemas(&config, Some(&request.toml));
    let ok = diagnostics
        .iter()
        .all(|diagnostic| diagnostic.severity != ConfigDiagnosticSeverity::Error);
    respond_json(
        stream,
        200,
        &LocalConfigValidatePayload {
            ok,
            path: request.path,
            error: None,
            diagnostics: diagnostics
                .iter()
                .map(local_control_apply_diagnostic_payload_from_local)
                .collect(),
        },
    )
    .await
}

async fn connect_owner_control_client(
    state: &MeshApi,
    endpoint: &str,
) -> Result<mesh_client::OwnerControlClient, LocalControlErrorPayload> {
    let owner_key_path = state.owner_key_path().await;
    let owner_keypair =
        load_local_owner_keypair(owner_key_path.as_deref()).map_err(control_error_from_anyhow)?;
    let client = ClientBuilder::new(owner_keypair, InviteToken("local-control".to_string()))
        .build()
        .map_err(|error| control_error_from_anyhow(anyhow::anyhow!(error.to_string())))?;
    let connection = client
        .connect_control_plane(ControlPlaneBootstrapOptions::new().with_control_endpoint(endpoint))
        .await
        .map_err(control_error_from_client)?;
    match connection {
        ControlPlaneConnection::OwnerControl(client) => Ok(*client),
    }
}

fn required_control_endpoint(endpoint: Option<String>) -> Result<String, LocalControlErrorPayload> {
    match endpoint.map(|value| value.trim().to_string()) {
        Some(endpoint) if !endpoint.is_empty() => Ok(endpoint),
        _ => Err(LocalControlErrorPayload {
            code: "control_endpoint_required".to_string(),
            message:
                "owner-control endpoint must be supplied explicitly; no gossip or peer inference is used"
                    .to_string(),
            legacy_retry_allowed: false,
            current_revision: None,
        }),
    }
}

pub(super) async fn ensure_loopback_control_caller(stream: &mut TcpStream) -> anyhow::Result<bool> {
    ensure_loopback_control_caller_for_peer_addr(stream, stream.peer_addr()).await
}

pub(crate) async fn ensure_loopback_control_caller_for_peer_addr(
    stream: &mut TcpStream,
    peer_addr: std::io::Result<std::net::SocketAddr>,
) -> anyhow::Result<bool> {
    match peer_addr {
        Ok(addr) if is_loopback_control_caller(addr) => Ok(true),
        Ok(addr) => {
            tracing::warn!("runtime control: rejected non-loopback caller {addr}");
            respond_json(
                stream,
                403,
                &serde_json::json!({"error": "runtime control endpoints only accept localhost connections"}),
            )
            .await?;
            Ok(false)
        }
        Err(error) => {
            tracing::warn!("runtime control: could not determine caller address: {error}");
            respond_json(
                stream,
                403,
                &serde_json::json!({"error": "runtime control endpoints require a localhost caller"}),
            )
            .await?;
            Ok(false)
        }
    }
}

fn is_loopback_control_caller(addr: SocketAddr) -> bool {
    addr.ip().is_loopback()
}

/// Loopback reachability is not owner identity: an owner-control route also
/// requires the host's owner-control bearer credential. A caller without it
/// is rejected before the host signs any owner-control traffic on its
/// behalf, and the host fails closed when no token is configured.
async fn ensure_owner_control_caller(
    stream: &mut TcpStream,
    state: &MeshApi,
    raw_request: &[u8],
) -> anyhow::Result<bool> {
    if !ensure_loopback_control_caller(stream).await? {
        return Ok(false);
    }
    if !ensure_owner_control_local_user(stream).await? {
        return Ok(false);
    }
    let presented = mesh_llm_control_api::access::request_authorization_bearer(raw_request)
        .map_err(|_| anyhow::anyhow!("invalid owner-control credential header"))?
        .unwrap_or_default();
    if let Some(expected) = state.owner_control_token().await
        && crate::api::owner_control_token::owner_control_tokens_match(&expected, presented)
    {
        return Ok(true);
    }
    tracing::warn!("owner control: rejected loopback caller without owner-control credential");
    respond_json(
        stream,
        401,
        &serde_json::json!({"error": "owner control credential required"}),
    )
    .await?;
    Ok(false)
}

/// Owner-control surfaces the host's owner key material, so loopback
/// reachability is not enough: where the platform exposes the peer process's
/// credentials (Unix-domain sockets), the caller must also run as the OS
/// user that runs the host. Loopback TCP never exposes peer credentials on
/// the supported platforms, so there the bearer credential alone remains the
/// owner-control requirement (see `crate::api::peer_identity`).
async fn ensure_owner_control_local_user(stream: &mut TcpStream) -> anyhow::Result<bool> {
    match crate::api::peer_identity::peer_runs_as_current_user(&*stream) {
        Some(false) => {
            tracing::warn!("owner control: rejected caller running as another OS user");
            respond_json(
                stream,
                403,
                &serde_json::json!({"error": "owner control is only available to the host service user"}),
            )
            .await?;
            Ok(false)
        }
        Some(true) | None => Ok(true),
    }
}

fn local_control_snapshot_payload(
    snapshot: mesh_client::proto::node::OwnerControlConfigSnapshot,
) -> LocalControlSnapshotPayload {
    let mut config = snapshot
        .config
        .as_ref()
        .map(crate::protocol::convert::proto_config_to_mesh)
        .unwrap_or_default();
    crate::protocol::config_redaction::redact_telemetry_headers(&mut config);
    LocalControlSnapshotPayload {
        node_id: hex::encode(snapshot.node_id),
        revision: snapshot.revision,
        config_hash: hex::encode(snapshot.config_hash),
        hostname: snapshot.hostname,
        config,
    }
}

fn local_control_scan_refresh_payload(
    response: OwnerControlScanRefreshResult,
) -> LocalControlScanRefreshPayload {
    let target_node_id = hex::encode(&response.snapshot.node_id);
    let (disposition, inventory) = match response.inventory {
        Some(inventory) => (
            Some(control_scan_disposition_label(inventory.disposition)),
            Some(
                inventory
                    .entries
                    .into_iter()
                    .map(local_control_inventory_entry_payload)
                    .collect(),
            ),
        ),
        None => (None, None),
    };
    LocalControlScanRefreshPayload {
        target_node_id,
        disposition,
        inventory,
    }
}

fn control_scan_disposition_label(value: i32) -> String {
    match mesh_client::proto::node::OwnerControlRefreshInventoryDisposition::try_from(value) {
        Ok(mesh_client::proto::node::OwnerControlRefreshInventoryDisposition::Executed) => {
            "executed"
        }
        Ok(mesh_client::proto::node::OwnerControlRefreshInventoryDisposition::Coalesced) => {
            "coalesced"
        }
        _ => "unspecified",
    }
    .to_string()
}

fn local_control_inventory_entry_payload(
    entry: mesh_client::proto::node::OwnerControlInventoryEntry,
) -> LocalControlInventoryEntryPayload {
    LocalControlInventoryEntryPayload {
        canonical_model_ref: entry.canonical_model_ref,
        display_name: entry.display_name,
        total_size_bytes: entry.total_size_bytes,
        metadata: entry.metadata.map(local_control_model_metadata_payload),
    }
}

fn local_control_model_metadata_payload(
    metadata: mesh_client::proto::node::CompactModelMetadata,
) -> LocalControlModelMetadataPayload {
    LocalControlModelMetadataPayload {
        model_key: metadata.model_key,
        context_length: metadata.context_length,
        vocab_size: metadata.vocab_size,
        embedding_size: metadata.embedding_size,
        head_count: metadata.head_count,
        layer_count: metadata.layer_count,
        feed_forward_length: metadata.feed_forward_length,
        key_length: metadata.key_length,
        value_length: metadata.value_length,
        architecture: metadata.architecture,
        tokenizer_model_name: metadata.tokenizer_model_name,
        special_tokens: metadata
            .special_tokens
            .into_iter()
            .map(|token| LocalControlSpecialTokenPayload {
                name: token.name,
                token_id: token.token_id,
            })
            .collect(),
        rope_scale: metadata.rope_scale,
        rope_freq_base: metadata.rope_freq_base,
        is_moe: metadata.is_moe,
        expert_count: metadata.expert_count,
        used_expert_count: metadata.used_expert_count,
        quantization_type: metadata.quantization_type,
        kv_head_count: metadata.kv_head_count,
        parameter_size: metadata.parameter_size,
    }
}

fn control_apply_mode_label(value: i32) -> String {
    match mesh_client::proto::node::ConfigApplyMode::try_from(value) {
        Ok(mesh_client::proto::node::ConfigApplyMode::Staged) => "staged".to_string(),
        Ok(mesh_client::proto::node::ConfigApplyMode::Live) => "live".to_string(),
        Ok(mesh_client::proto::node::ConfigApplyMode::Noop) => "noop".to_string(),
        Ok(mesh_client::proto::node::ConfigApplyMode::Unspecified) => "unspecified".to_string(),
        _ => "unspecified".to_string(),
    }
}

fn load_local_owner_keypair(
    path: Option<&std::path::Path>,
) -> anyhow::Result<crate::crypto::OwnerKeypair> {
    let path =
        path.ok_or_else(|| anyhow::anyhow!("local owner keystore unavailable for this runtime"))?;
    let info = keystore_metadata(path)?;
    if info.encrypted && std::env::var("MESH_LLM_OWNER_PASSPHRASE").is_err() {
        match load_owner_keypair_from_keychain(path) {
            Ok(keypair) => return Ok(keypair),
            Err(OwnerKeychainLoadError::NoEntry)
            | Err(OwnerKeychainLoadError::Crypto(crate::crypto::CryptoError::DecryptionFailed))
            | Err(OwnerKeychainLoadError::Crypto(
                crate::crypto::CryptoError::KeychainUnavailable { .. },
            ))
            | Err(OwnerKeychainLoadError::Crypto(
                crate::crypto::CryptoError::KeychainAccessDenied { .. },
            )) => {}
            Err(OwnerKeychainLoadError::Crypto(err)) => {
                let error: anyhow::Error = err.into();
                return Err(
                    error.context(format!("Failed to load owner keystore {}", path.display()))
                );
            }
        }
    }
    let passphrase = resolve_owner_passphrase(path)?;
    load_keystore(path, passphrase.as_deref().map(|value| value.as_str()))
        .map_err(Into::into)
        .map_err(|error: anyhow::Error| {
            error.context(format!("Failed to load owner keystore {}", path.display()))
        })
}

fn resolve_owner_passphrase(path: &std::path::Path) -> anyhow::Result<Option<Zeroizing<String>>> {
    if let Ok(passphrase) = std::env::var("MESH_LLM_OWNER_PASSPHRASE") {
        return Ok(Some(Zeroizing::new(passphrase)));
    }
    let info = keystore_metadata(path)?;
    if !info.encrypted {
        return Ok(None);
    }
    Err(crate::crypto::CryptoError::MissingPassphrase.into())
}

fn control_error_from_client(error: ControlPlaneClientError) -> LocalControlErrorPayload {
    match error {
        ControlPlaneClientError::Negotiation(error) => LocalControlErrorPayload {
            code: owner_control_error_code_label(error.code),
            message: error.message,
            legacy_retry_allowed: error.legacy_retry_allowed,
            current_revision: None,
        },
        ControlPlaneClientError::Remote(error) => control_error_from_remote(error),
        ControlPlaneClientError::Transport(message) => LocalControlErrorPayload {
            code: "control_unavailable".to_string(),
            message,
            legacy_retry_allowed: false,
            current_revision: None,
        },
        ControlPlaneClientError::Protocol(message) => LocalControlErrorPayload {
            code: "control_protocol_error".to_string(),
            message,
            legacy_retry_allowed: false,
            current_revision: None,
        },
    }
}

fn control_error_from_remote(error: OwnerControlRemoteError) -> LocalControlErrorPayload {
    LocalControlErrorPayload {
        code: owner_control_error_code_label(error.code),
        message: error.message,
        legacy_retry_allowed: false,
        current_revision: error.current_revision,
    }
}

fn control_error_from_anyhow(error: anyhow::Error) -> LocalControlErrorPayload {
    LocalControlErrorPayload {
        code: "control_unavailable".to_string(),
        message: error.to_string(),
        legacy_retry_allowed: false,
        current_revision: None,
    }
}

fn owner_control_error_code_label(code: mesh_client::proto::node::OwnerControlErrorCode) -> String {
    match code {
        mesh_client::proto::node::OwnerControlErrorCode::Unspecified => "unspecified",
        mesh_client::proto::node::OwnerControlErrorCode::ControlEndpointRequired => {
            "control_endpoint_required"
        }
        mesh_client::proto::node::OwnerControlErrorCode::ControlUnavailable => {
            "control_unavailable"
        }
        mesh_client::proto::node::OwnerControlErrorCode::ControlUnsupported => {
            "control_unsupported"
        }
        mesh_client::proto::node::OwnerControlErrorCode::Unauthorized => "unauthorized",
        mesh_client::proto::node::OwnerControlErrorCode::TargetNodeMismatch => {
            "target_node_mismatch"
        }
        mesh_client::proto::node::OwnerControlErrorCode::RevisionConflict => "revision_conflict",
        mesh_client::proto::node::OwnerControlErrorCode::InvalidHandshake => "invalid_handshake",
        mesh_client::proto::node::OwnerControlErrorCode::LegacyJsonUnsupported => {
            "legacy_json_unsupported"
        }
        mesh_client::proto::node::OwnerControlErrorCode::UnknownCommand => "unknown_command",
        mesh_client::proto::node::OwnerControlErrorCode::BadRequest => "bad_request",
    }
    .to_string()
}

fn control_error_status_code(error: &LocalControlErrorPayload) -> u16 {
    match error.code.as_str() {
        "control_endpoint_required" | "bad_request" | "target_node_mismatch" => 400,
        "unauthorized" => 403,
        "revision_conflict" => 409,
        "control_unavailable" | "control_unsupported" => 503,
        _ => 502,
    }
}

async fn respond_control_error(
    stream: &mut TcpStream,
    error: LocalControlErrorPayload,
) -> anyhow::Result<()> {
    respond_json(
        stream,
        control_error_status_code(&error),
        &serde_json::json!({ "error": error }),
    )
    .await
}

fn handle_runtime_stages<'a>(
    stream: &'a mut TcpStream,
    state: &'a MeshApi,
) -> BoxedRouteFuture<'a> {
    Box::pin(handle_runtime_stages_inner(stream, state))
}

async fn handle_runtime_stages_inner(
    stream: &mut TcpStream,
    state: &MeshApi,
) -> anyhow::Result<()> {
    match tokio::time::timeout(std::time::Duration::from_secs(5), state.runtime_stages()).await {
        Ok(runtime_stages) => respond_json(stream, 200, &runtime_stages).await,
        Err(_) => respond_error(stream, 503, "Runtime stage status temporarily unavailable").await,
    }
}

fn handle_status<'a>(stream: &'a mut TcpStream, state: &'a MeshApi) -> BoxedRouteFuture<'a> {
    Box::pin(handle_status_inner(stream, state))
}

async fn handle_status_inner(stream: &mut TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    match tokio::time::timeout(std::time::Duration::from_secs(5), state.status()).await {
        Ok(status) => respond_json(stream, 200, &status).await,
        Err(_) => respond_error(stream, 503, "Status temporarily unavailable").await,
    }
}

fn handle_models<'a>(stream: &'a mut TcpStream, state: &'a MeshApi) -> BoxedRouteFuture<'a> {
    Box::pin(handle_models_inner(stream, state))
}

async fn handle_models_inner(stream: &mut TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    let mesh_models = state.mesh_models().await;
    respond_json(
        stream,
        200,
        &serde_json::json!({ "mesh_models": mesh_models }),
    )
    .await
}

fn handle_runtime_status<'a>(
    stream: &'a mut TcpStream,
    state: &'a MeshApi,
) -> BoxedRouteFuture<'a> {
    Box::pin(handle_runtime_status_inner(stream, state))
}

async fn handle_runtime_status_inner(
    stream: &mut TcpStream,
    state: &MeshApi,
) -> anyhow::Result<()> {
    match tokio::time::timeout(std::time::Duration::from_secs(5), state.runtime_status()).await {
        Ok(runtime_status) => respond_json(stream, 200, &runtime_status).await,
        Err(_) => respond_error(stream, 503, "Runtime status temporarily unavailable").await,
    }
}

fn handle_runtime_processes<'a>(
    stream: &'a mut TcpStream,
    state: &'a MeshApi,
) -> BoxedRouteFuture<'a> {
    Box::pin(handle_runtime_processes_inner(stream, state))
}

async fn handle_runtime_processes_inner(
    stream: &mut TcpStream,
    state: &MeshApi,
) -> anyhow::Result<()> {
    match tokio::time::timeout(std::time::Duration::from_secs(5), state.runtime_processes()).await {
        Ok(runtime_processes) => respond_json(stream, 200, &runtime_processes).await,
        Err(_) => {
            respond_error(
                stream,
                503,
                "Runtime process status temporarily unavailable",
            )
            .await
        }
    }
}

fn handle_runtime_llama<'a>(stream: &'a mut TcpStream, state: &'a MeshApi) -> BoxedRouteFuture<'a> {
    Box::pin(handle_runtime_llama_inner(stream, state))
}

async fn handle_runtime_llama_inner(stream: &mut TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    match tokio::time::timeout(std::time::Duration::from_secs(5), state.runtime_llama()).await {
        Ok(runtime_llama) => respond_json(stream, 200, &runtime_llama).await,
        Err(_) => {
            respond_error(
                stream,
                503,
                "Runtime llama snapshot temporarily unavailable",
            )
            .await
        }
    }
}

fn handle_runtime_events<'a>(
    stream: &'a mut TcpStream,
    state: &'a MeshApi,
) -> BoxedRouteFuture<'a> {
    Box::pin(handle_runtime_events_inner(stream, state))
}

async fn handle_runtime_events_inner(
    stream: &mut TcpStream,
    state: &MeshApi,
) -> anyhow::Result<()> {
    let request_id = super::super::management_lifecycle::response_request_id_header()
        .map(|request_id| format!("x-request-id: {request_id}\r\n"))
        .unwrap_or_default();
    let header = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\nX-Accel-Buffering: no\r\n{request_id}\r\n"
    );
    stream.write_all(header.as_bytes()).await?;
    super::super::management_lifecycle::record_response_status(200);

    let mut subscription = {
        state
            .inner
            .lock()
            .await
            .runtime_data_collector
            .clone()
            .subscribe()
    };
    let mut last_sent_json = None;

    let runtime_llama = state.runtime_llama().await;
    if let Ok(json) = serde_json::to_string(&runtime_llama) {
        stream
            .write_all(format!("data: {json}\n\n").as_bytes())
            .await?;
        last_sent_json = Some(json);
    }

    loop {
        tokio::select! {
            changed = subscription.changed() => {
                match changed {
                    Ok(()) => {
                        let subscription_state = *subscription.borrow_and_update();
                        if !subscription_state.dirty.contains(crate::runtime_data::RuntimeDataDirty::RUNTIME) {
                            continue;
                        }
                        let runtime_llama = state.runtime_llama().await;
                        let Ok(json) = serde_json::to_string(&runtime_llama) else {
                            continue;
                        };
                        if last_sent_json.as_deref() == Some(json.as_str()) {
                            continue;
                        }
                        if stream.write_all(format!("data: {json}\n\n").as_bytes()).await.is_err() {
                            break;
                        }
                        last_sent_json = Some(json);
                    }
                    Err(_) => break,
                }
            }
            _ = tokio::time::sleep(std::time::Duration::from_secs(15)) => {
                if stream.write_all(b": keepalive\n\n").await.is_err() {
                    break;
                }
            }
        }
    }

    Ok(())
}

fn handle_runtime_endpoints<'a>(
    stream: &'a mut TcpStream,
    state: &'a MeshApi,
) -> BoxedRouteFuture<'a> {
    Box::pin(handle_runtime_endpoints_inner(stream, state))
}

async fn handle_runtime_endpoints_inner(
    stream: &mut TcpStream,
    state: &MeshApi,
) -> anyhow::Result<()> {
    match state.runtime_endpoints().await {
        Ok(endpoints) => {
            let endpoints = endpoints
                .into_iter()
                .map(|endpoint| endpoint.redacted_for_network())
                .collect::<Vec<_>>();
            respond_json(stream, 200, &serde_json::json!({ "endpoints": endpoints })).await
        }
        Err(err) => respond_error(stream, 500, &err.to_string()).await,
    }
}

async fn handle_set_mesh_guardrails(
    stream: &mut TcpStream,
    state: &MeshApi,
    body: &str,
) -> anyhow::Result<()> {
    if !ensure_loopback_control_caller(stream).await? {
        return Ok(());
    }

    let Some(control_tx) = state.inner.lock().await.runtime_control.clone() else {
        return respond_error(stream, 503, "Runtime control unavailable").await;
    };
    let request: OpenAiGuardrailsModeRequest = match serde_json::from_str(body) {
        Ok(request) => request,
        Err(_) => return respond_error(stream, 400, "Invalid JSON body").await,
    };
    let Some(mode) = parse_guardrail_mode(&request.mode) else {
        return respond_error(stream, 400, "Invalid guardrail mode").await;
    };

    let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
    let _ = control_tx.send(RuntimeControlRequest::SetOpenAiGuardrailMode {
        mode,
        resp: resp_tx,
    });
    match resp_rx.await {
        Ok(Ok(updated)) => respond_json(stream, 200, &updated).await?,
        Ok(Err(e)) => respond_runtime_error(stream, &e.to_string()).await?,
        Err(_) => respond_error(stream, 503, "Runtime control unavailable").await?,
    }
    Ok(())
}

fn parse_guardrail_mode(mode: &str) -> Option<GuardrailMode> {
    match mode.trim().to_ascii_lowercase().as_str() {
        "disabled" | "disable" | "off" => Some(GuardrailMode::Disabled),
        "metrics" | "metrics_only" | "metrics-only" => Some(GuardrailMode::MetricsOnly),
        "enforce" | "enforced" => Some(GuardrailMode::Enforce),
        _ => None,
    }
}

async fn handle_load_model(
    stream: &mut TcpStream,
    state: &MeshApi,
    body: &str,
) -> anyhow::Result<()> {
    let Some(control_tx) = state.inner.lock().await.runtime_control.clone() else {
        return respond_error(stream, 503, "Runtime control unavailable").await;
    };

    let request = match parse_runtime_load_request(body) {
        Ok(request) => request,
        Err(error) => return respond_error(stream, 400, error.message()).await,
    };

    let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
    let _ = control_tx.send(RuntimeControlRequest::Load {
        spec: request.spec,
        config_model_id: None,
        profile: request.profile,
        sizing: request.sizing,
        resp: resp_tx,
    });
    match resp_rx.await {
        Ok(Ok(loaded)) => {
            respond_json(
                stream,
                201,
                &serde_json::json!({
                    "loaded": loaded.model,
                    "instance_id": loaded.instance_id,
                }),
            )
            .await?;
        }
        Ok(Err(e)) => {
            respond_runtime_error(stream, &format!("{e:#}")).await?;
        }
        Err(_) => {
            respond_error(stream, 503, "Runtime control unavailable").await?;
        }
    }

    Ok(())
}

async fn handle_unload_model(
    stream: &mut TcpStream,
    state: &MeshApi,
    path: &str,
) -> anyhow::Result<()> {
    let Some(control_tx) = state.inner.lock().await.runtime_control.clone() else {
        return respond_error(stream, 503, "Runtime control unavailable").await;
    };
    let Some(model_name) = decode_runtime_model_path(path, "/api/runtime/models/") else {
        return respond_error(stream, 400, "Missing model path").await;
    };

    let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
    let _ = control_tx.send(RuntimeControlRequest::Unload {
        target: UnloadTarget::Model(model_name.clone()),
        options: UnloadOptions::default(),
        resp: resp_tx,
    });
    match resp_rx.await {
        Ok(Ok(dropped)) => {
            respond_json(
                stream,
                200,
                &serde_json::json!({
                    "dropped": dropped.model,
                    "instance_id": dropped.instance_id,
                }),
            )
            .await?;
        }
        Ok(Err(e)) => {
            respond_runtime_error(stream, &e.to_string()).await?;
        }
        Err(_) => {
            respond_error(stream, 503, "Runtime control unavailable").await?;
        }
    }
    Ok(())
}

async fn handle_unload_instance(
    stream: &mut TcpStream,
    state: &MeshApi,
    path: &str,
) -> anyhow::Result<()> {
    let Some(control_tx) = state.inner.lock().await.runtime_control.clone() else {
        return respond_error(stream, 503, "Runtime control unavailable").await;
    };
    let Some(instance_id) = decode_runtime_model_path(path, "/api/runtime/instances/") else {
        return respond_error(stream, 400, "Missing runtime instance path").await;
    };

    let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
    let _ = control_tx.send(RuntimeControlRequest::Unload {
        target: UnloadTarget::Instance(instance_id.clone()),
        options: UnloadOptions::default(),
        resp: resp_tx,
    });
    match resp_rx.await {
        Ok(Ok(dropped)) => {
            respond_json(
                stream,
                200,
                &serde_json::json!({
                    "dropped": dropped.model,
                    "instance_id": dropped.instance_id,
                }),
            )
            .await?;
        }
        Ok(Err(e)) => {
            respond_runtime_error(stream, &e.to_string()).await?;
        }
        Err(_) => {
            respond_error(stream, 503, "Runtime control unavailable").await?;
        }
    }
    Ok(())
}

fn handle_events<'a>(stream: &'a mut TcpStream, state: &'a MeshApi) -> BoxedRouteFuture<'a> {
    Box::pin(handle_events_inner(stream, state))
}

async fn handle_events_inner(stream: &mut TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    let request_id = super::super::management_lifecycle::response_request_id_header()
        .map(|request_id| format!("x-request-id: {request_id}\r\n"))
        .unwrap_or_default();
    let header = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\nX-Accel-Buffering: no\r\n{request_id}\r\n"
    );
    stream.write_all(header.as_bytes()).await?;
    super::super::management_lifecycle::record_response_status(200);

    let status = state.status().await;
    let mut last_sent_json = None;
    if let Ok(json) = serde_json::to_string(&status) {
        stream
            .write_all(format!("data: {json}\n\n").as_bytes())
            .await?;
        last_sent_json = Some(json);
    }

    let mut subscription = {
        state
            .inner
            .lock()
            .await
            .runtime_data_collector
            .clone()
            .subscribe()
    };

    loop {
        tokio::select! {
            changed = subscription.changed() => {
                match changed {
                    Ok(()) => {
                        let subscription_state = *subscription.borrow_and_update();
                        let interesting = subscription_state.dirty.contains(crate::runtime_data::RuntimeDataDirty::STATUS)
                            || subscription_state.dirty.contains(crate::runtime_data::RuntimeDataDirty::MODELS)
                            || subscription_state.dirty.contains(crate::runtime_data::RuntimeDataDirty::ROUTING)
                            || subscription_state.dirty.contains(crate::runtime_data::RuntimeDataDirty::PROCESSES)
                            || subscription_state.dirty.contains(crate::runtime_data::RuntimeDataDirty::INVENTORY)
                            || subscription_state.dirty.contains(crate::runtime_data::RuntimeDataDirty::PLUGINS);
                        if !interesting {
                            continue;
                        }
                        let status = state.status().await;
                        let Ok(json) = serde_json::to_string(&status) else {
                            continue;
                        };
                        if last_sent_json.as_deref() == Some(json.as_str()) {
                            continue;
                        }
                        if stream.write_all(format!("data: {json}\n\n").as_bytes()).await.is_err() {
                            break;
                        }
                        last_sent_json = Some(json);
                    }
                    Err(_) => break,
                }
            }
            _ = tokio::time::sleep(std::time::Duration::from_secs(15)) => {
                if stream.write_all(b": keepalive\n\n").await.is_err() {
                    break;
                }
            }
        }
    }

    Ok(())
}

// ─── Intent and Activity API handlers ─────────────────────────────────────────

use crate::api::status::{IntentEntry, IntentListPayload};

/// Maximum number of intent entries returned in a single response.
const INTENT_CAP: usize = 256;

fn handle_get_intents<'a>(stream: &'a mut TcpStream, state: &'a MeshApi) -> BoxedRouteFuture<'a> {
    Box::pin(handle_get_intents_inner(stream, state))
}

async fn handle_get_intents_inner(stream: &mut TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    if !ensure_loopback_control_caller(stream).await? {
        return Ok(());
    }

    let intent_history = {
        let inner = state.inner.lock().await;
        inner
            .node
            .runtime_intents
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    };

    let total_count = intent_history.len();
    let truncated = total_count > INTENT_CAP;
    let entries = intent_history
        .into_iter()
        .rev()
        .take(INTENT_CAP)
        .map(|intent| IntentEntry {
            intent_id: intent.intent_id,
            model_ref: intent.canonical_model_ref,
            profile: intent.profile,
            source: intent.source.into(),
            desired_state: intent.desired_state.as_str().to_string(),
            instance_target: intent.instance_target,
            persistence: match intent.persistence {
                crate::runtime::IntentPersistence::Process => "process",
                crate::runtime::IntentPersistence::Session => "session",
                crate::runtime::IntentPersistence::Ephemeral => "ephemeral",
            }
            .to_string(),
            created_at_secs: intent.created_at_secs,
            updated_at_secs: intent.updated_at_secs,
            last_error: intent.last_error,
        })
        .collect();

    respond_json(
        stream,
        200,
        &IntentListPayload {
            intents: entries,
            total_count,
            truncated,
        },
    )
    .await
}

// ─── Fleet control (Phase 1.1) ──────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct FleetControlRequest {
    command: String,
    model: Option<String>,
    instance_id: Option<String>,
    profile: Option<String>,
    #[serde(default)]
    role_filter: Vec<String>,
    #[serde(default)]
    node_filter: Vec<String>,
    #[serde(default)]
    dry_run: bool,
    /// Only target peers with at least this many VRAM bytes.
    min_vram_bytes: Option<u64>,
    /// Only target peers that have this model in available_models or catalog_models.
    model_available: Option<String>,
    /// Only target peers that do NOT have this model in serving_models.
    model_not_serving: Option<String>,
    /// Only target peers whose version string is >= this value (string compare).
    version_min: Option<String>,
    /// Only target peers whose owner_id matches this value exactly.
    owner_id: Option<String>,
}

#[derive(Debug, Serialize)]
struct FleetControlResponse {
    results: Vec<FleetNodeResult>,
    total: usize,
    succeeded: usize,
    failed: usize,
}

#[derive(Debug, Serialize)]
struct FleetNodeResult {
    node_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    hostname: Option<String>,
    success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<LocalControlErrorPayload>,
    duration_ms: u64,
}

#[derive(Debug, Serialize)]
struct FleetDryRunResponse {
    targets: Vec<FleetDryRunTarget>,
    total: usize,
}

#[derive(Debug, Serialize)]
struct FleetDryRunTarget {
    node_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    hostname: Option<String>,
}

const FLEET_CONCURRENCY: usize = 32;

async fn handle_control_fleet(
    stream: &mut TcpStream,
    state: &MeshApi,
    body: &str,
) -> anyhow::Result<()> {
    if !ensure_loopback_control_caller(stream).await? {
        return Ok(());
    }
    let request: FleetControlRequest = match serde_json::from_str(body) {
        Ok(r) => r,
        Err(_) => return respond_error(stream, 400, "Invalid JSON body").await,
    };
    let peers = state.node().await.peers().await;

    // Filter peers
    let mut filtered_peers: Vec<_> = peers
        .into_iter()
        .filter(|peer| {
            // Clients are never valid fleet targets regardless of role_filter.
            if matches!(peer.role, mesh_llm_membership::NodeRole::Client) {
                return false;
            }
            // Apply role_filter if present (Clients already excluded above).
            if !request.role_filter.is_empty() {
                let role_str = match &peer.role {
                    mesh_llm_membership::NodeRole::Worker => "Worker",
                    mesh_llm_membership::NodeRole::Host { .. } => "Host",
                    mesh_llm_membership::NodeRole::Client => "Client",
                };
                if !request.role_filter.iter().any(|r| r == role_str) {
                    return false;
                }
            }
            // Apply node_filter
            if !request.node_filter.is_empty() {
                let node_id_hex = hex::encode(peer.id.as_bytes());
                if !request.node_filter.iter().any(|n| {
                    node_id_hex.starts_with(n.as_str()) || node_id_hex == n.as_str()
                }) {
                    return false;
                }
            }
            true
        })
        .collect();

    // Fine-grained fleet filters
    if let Some(min_vram) = request.min_vram_bytes {
        filtered_peers.retain(|p| p.vram_bytes >= min_vram);
    }
    if let Some(ref model) = request.model_available {
        filtered_peers.retain(|p| p.available_models.iter().any(|m| m == model));
    }
    if let Some(ref model) = request.model_not_serving {
        filtered_peers.retain(|p| !p.serving_models.iter().any(|m| m == model));
    }
    if let Some(ref version) = request.version_min {
        filtered_peers
            .retain(|p| p.version.as_deref().map_or(false, |v| v >= version.as_str()));
    }
    if let Some(ref owner) = request.owner_id {
        filtered_peers.retain(|p| {
            p.owner_summary.owner_id.as_deref() == Some(owner.as_str())
        });
    }

    if request.dry_run {
        let targets: Vec<FleetDryRunTarget> = filtered_peers
            .iter()
            .map(|peer| FleetDryRunTarget {
                node_id: hex::encode(peer.id.as_bytes()),
                hostname: peer.hostname.clone(),
            })
            .collect();
        let total = targets.len();
        return respond_json(stream, 200, &FleetDryRunResponse { targets, total }).await;
    }

    // Parse command into lifecycle operation + target
    let operation = match request.command.as_str() {
        "load-model" => ControlLifecycleOperation::Load,
        "unload-model" => ControlLifecycleOperation::Unload,
        "ensure-model" => ControlLifecycleOperation::Ensure,
        "drain-model" => ControlLifecycleOperation::Drain,
        _ => {
            return respond_error(
                stream,
                400,
                "unknown fleet command; expected load-model, unload-model, ensure-model, or drain-model",
            )
            .await;
        }
    };

    let lifecycle_request = ControlLifecycleModelRequest {
        endpoint: None,
        model: request.model.clone(),
        instance_id: request.instance_id.clone(),
        profile: request.profile.clone(),
    };
    let target = match validate_control_lifecycle_target(operation, lifecycle_request) {
        Ok(t) => t,
        Err(msg) => return respond_error(stream, 400, msg).await,
    };

    // Fan out in parallel with bounded concurrency
    let mut join_set = tokio::task::JoinSet::new();
    let mut pending = 0usize;
    let mut results: Vec<FleetNodeResult> = Vec::with_capacity(filtered_peers.len());
    let total = filtered_peers.len();

    let peers_with_endpoints: Vec<(String, Option<String>, String)> = filtered_peers
        .iter()
        .map(|peer| {
            let node_id = hex::encode(peer.id.as_bytes());
            let hostname = peer.hostname.clone();
            let endpoint = encode_endpoint_addr_token(&peer.addr);
            (node_id, hostname, endpoint)
        })
        .collect();

    // We need to pass state's owner key path; clone what we need
    let owner_key_path = state.owner_key_path().await;

    for (node_id, hostname, endpoint) in peers_with_endpoints {
        let node_id_clone = node_id.clone();
        let hostname_clone = hostname.clone();
        let endpoint_clone = endpoint.clone();
        let owner_key_path_clone = owner_key_path.clone();
        // Clone target fields for each task
        let model = target.model.clone();
        let instance_id = target.instance_id.clone();
        let profile = target.profile.clone();

        join_set.spawn(async move {
            let t0 = std::time::Instant::now();
            let fleet_target = ControlLifecycleTarget {
                model,
                instance_id,
                profile,
            };
            let result = execute_fleet_node_lifecycle(
                owner_key_path_clone,
                endpoint_clone,
                operation,
                fleet_target,
            )
            .await;
            let duration_ms = t0.elapsed().as_millis() as u64;
            match result {
                Ok(_) => FleetNodeResult {
                    node_id: node_id_clone,
                    hostname: hostname_clone,
                    success: true,
                    error: None,
                    duration_ms,
                },
                Err(error) => FleetNodeResult {
                    node_id: node_id_clone,
                    hostname: hostname_clone,
                    success: false,
                    error: Some(error),
                    duration_ms,
                },
            }
        });
        pending += 1;

        // Drain if we hit concurrency limit
        if pending >= FLEET_CONCURRENCY {
            if let Some(Ok(node_result)) = join_set.join_next().await {
                results.push(node_result);
                pending -= 1;
            }
        }
    }
    // Drain remaining
    while let Some(Ok(node_result)) = join_set.join_next().await {
        results.push(node_result);
    }

    let succeeded = results.iter().filter(|r| r.success).count();
    let failed = results.iter().filter(|r| !r.success).count();
    respond_json(
        stream,
        200,
        &FleetControlResponse {
            results,
            total,
            succeeded,
            failed,
        },
    )
    .await
}

async fn execute_fleet_node_lifecycle(
    owner_key_path: Option<std::path::PathBuf>,
    endpoint: String,
    operation: ControlLifecycleOperation,
    target: ControlLifecycleTarget,
) -> Result<serde_json::Value, LocalControlErrorPayload> {
    let owner_keypair =
        load_local_owner_keypair(owner_key_path.as_deref()).map_err(control_error_from_anyhow)?;
    let client = ClientBuilder::new(owner_keypair, InviteToken("local-control".to_string()))
        .build()
        .map_err(|error| control_error_from_anyhow(anyhow::anyhow!(error.to_string())))?;
    let connection = client
        .connect_control_plane(ControlPlaneBootstrapOptions::new().with_control_endpoint(&endpoint))
        .await
        .map_err(control_error_from_client)?;
    match connection {
        ControlPlaneConnection::OwnerControl(owner_client) => {
            let result = invoke_control_lifecycle(&owner_client, operation, target).await;
            owner_client.close().await;
            result.map_err(control_error_from_client)
        }
    }
}

// ─── Fleet metrics (Phase 1.3) ──────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct FleetMetricsResponse {
    nodes: Vec<FleetNodeMetrics>,
    fleet_total_avg_tokens_per_second: f64,
    fleet_cumulative_completion_tokens: u64,
    fleet_cumulative_prompt_tokens: u64,
    model_totals: Vec<FleetModelThroughput>,
}

#[derive(Debug, Serialize)]
struct FleetNodeMetrics {
    node_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    hostname: Option<String>,
    role: String,
    total_avg_tokens_per_second: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    cumulative_completion_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cumulative_prompt_tokens: Option<u64>,
    models: Vec<FleetModelThroughput>,
}

#[derive(Debug, Serialize)]
struct FleetModelThroughput {
    model: String,
    avg_tokens_per_second: f64,
    samples: u64,
}

async fn handle_fleet_metrics(
    stream: &mut TcpStream,
    state: &MeshApi,
) -> anyhow::Result<()> {
    if !ensure_loopback_control_caller(stream).await? {
        return Ok(());
    }
    let peers = state.node().await.peers().await;
    let mut fleet_total_tps = 0f64;
    let mut fleet_cumulative_completion = 0u64;
    let mut fleet_cumulative_prompt = 0u64;
    let mut model_aggregate: std::collections::HashMap<String, (f64, u64)> =
        std::collections::HashMap::new();
    let mut nodes = Vec::with_capacity(peers.len());

    for peer in &peers {
        let role_str = match &peer.role {
            mesh_llm_membership::NodeRole::Worker => "Worker",
            mesh_llm_membership::NodeRole::Host { .. } => "Host",
            mesh_llm_membership::NodeRole::Client => "Client",
        };
        let mut node_total_tps = 0f64;
        let mut node_models = Vec::new();
        for hint in &peer.advertised_model_throughput {
            let tps = hint.avg_tokens_per_second_milli as f64 / 1000.0;
            node_total_tps += tps;
            fleet_total_tps += tps;
            let entry = model_aggregate
                .entry(hint.model_name.clone())
                .or_insert((0.0, 0));
            entry.0 += tps;
            entry.1 += hint.throughput_samples;
            node_models.push(FleetModelThroughput {
                model: hint.model_name.clone(),
                avg_tokens_per_second: tps,
                samples: hint.throughput_samples,
            });
        }
        // cumulative tokens from gossip
        let cumulative_completion = peer.cumulative_completion_tokens;
        let cumulative_prompt = peer.cumulative_prompt_tokens;
        fleet_cumulative_completion += cumulative_completion.unwrap_or(0);
        fleet_cumulative_prompt += cumulative_prompt.unwrap_or(0);

        nodes.push(FleetNodeMetrics {
            node_id: hex::encode(peer.id.as_bytes()),
            hostname: peer.hostname.clone(),
            role: role_str.to_string(),
            total_avg_tokens_per_second: node_total_tps,
            cumulative_completion_tokens: cumulative_completion,
            cumulative_prompt_tokens: cumulative_prompt,
            models: node_models,
        });
    }

    let model_totals = {
        let mut totals: Vec<FleetModelThroughput> = model_aggregate
            .into_iter()
            .map(|(model, (tps, samples))| FleetModelThroughput {
                model,
                avg_tokens_per_second: tps,
                samples,
            })
            .collect();
        totals.sort_by(|a, b| a.model.cmp(&b.model));
        totals
    };

    respond_json(
        stream,
        200,
        &FleetMetricsResponse {
            nodes,
            fleet_total_avg_tokens_per_second: fleet_total_tps,
            fleet_cumulative_completion_tokens: fleet_cumulative_completion,
            fleet_cumulative_prompt_tokens: fleet_cumulative_prompt,
            model_totals,
        },
    )
    .await
}

// ─── Mesh policy (Phase 2.2) ─────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct SetMeshPolicyRequest {
    policy_payload_bytes: String, // base64
    signature: String,            // base64
}

#[derive(Debug, Serialize)]
struct SetMeshPolicyResponse {
    accepted: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    active_gen: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<String>,
}

async fn handle_set_mesh_policy(
    stream: &mut TcpStream,
    state: &MeshApi,
    body: &str,
) -> anyhow::Result<()> {
    if !ensure_loopback_control_caller(stream).await? {
        return Ok(());
    }
    let request: SetMeshPolicyRequest = match serde_json::from_str(body) {
        Ok(r) => r,
        Err(_) => return respond_error(stream, 400, "Invalid JSON body").await,
    };
    use base64::Engine;
    let payload_bytes = match base64::engine::general_purpose::STANDARD
        .decode(&request.policy_payload_bytes)
    {
        Ok(b) => b,
        Err(_) => return respond_error(stream, 400, "Invalid base64 for policy_payload_bytes").await,
    };
    let signature = match base64::engine::general_purpose::STANDARD.decode(&request.signature) {
        Ok(b) => b,
        Err(_) => return respond_error(stream, 400, "Invalid base64 for signature").await,
    };
    let policy = crate::proto::node::SignedMeshPolicy {
        policy_payload_bytes: payload_bytes,
        signature,
    };
    let policy_store = state.policy_store();
    let result = policy_store.update(policy).await;
    let response = match &result {
        crate::policy::PolicyUpdateResult::Accepted { policy_gen } => SetMeshPolicyResponse {
            accepted: true,
            active_gen: Some(*policy_gen),
            reason: None,
        },
        crate::policy::PolicyUpdateResult::Stale { current_gen } => SetMeshPolicyResponse {
            accepted: false,
            active_gen: Some(*current_gen),
            reason: Some(format!("policy_gen <= current gen {current_gen}")),
        },
        crate::policy::PolicyUpdateResult::Invalid(msg) => SetMeshPolicyResponse {
            accepted: false,
            active_gen: None,
            reason: Some(msg.clone()),
        },
    };
    // Phase 3.2: persist accepted policy so it survives a restart.
    if matches!(result, crate::policy::PolicyUpdateResult::Accepted { .. }) {
        let state_dir = state
            .inner
            .lock()
            .await
            .node
            .identity_state_dir
            .clone();
        if let Some(state_dir) = state_dir {
            if let Err(error) = policy_store.persist(&state_dir).await {
                tracing::warn!("fleet policy: persist failed: {error:#}");
            }
        }
    }
    let status = if response.accepted { 200 } else { 409 };
    respond_json(stream, status, &response).await
}

#[derive(Debug, Serialize)]
struct GetMeshPolicyResponse {
    has_policy: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    policy_gen: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    owner_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    issued_at_unix_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    expires_at_unix_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_intents: Option<Vec<GetMeshPolicyIntent>>,
}

#[derive(Debug, Serialize)]
struct GetMeshPolicyIntent {
    canonical_model_ref: String,
    action: String,
}

async fn handle_get_mesh_policy(
    stream: &mut TcpStream,
    state: &MeshApi,
) -> anyhow::Result<()> {
    if !ensure_loopback_control_caller(stream).await? {
        return Ok(());
    }
    let current = state.policy_store().current().await;
    let response = match current {
        None => GetMeshPolicyResponse {
            has_policy: false,
            policy_gen: None,
            owner_id: None,
            issued_at_unix_ms: None,
            expires_at_unix_ms: None,
            model_intents: None,
        },
        Some(stored) => {
            use crate::proto::node::MeshModelIntentAction;
            let intents = stored
                .payload
                .model_intents
                .iter()
                .map(|i| GetMeshPolicyIntent {
                    canonical_model_ref: i.canonical_model_ref.clone(),
                    action: match MeshModelIntentAction::try_from(i.action) {
                        Ok(MeshModelIntentAction::Ensure) => "ensure".to_string(),
                        Ok(MeshModelIntentAction::Drain) => "drain".to_string(),
                        _ => "unspecified".to_string(),
                    },
                })
                .collect();
            GetMeshPolicyResponse {
                has_policy: true,
                policy_gen: Some(stored.payload.policy_gen),
                owner_id: Some(stored.payload.owner_id.clone()),
                issued_at_unix_ms: Some(stored.payload.issued_at_unix_ms),
                expires_at_unix_ms: Some(stored.payload.expires_at_unix_ms),
                model_intents: Some(intents),
            }
        }
    };
    respond_json(stream, 200, &response).await
}

#[cfg(test)]
mod tests {
    use super::{
        GuardrailMode, is_loopback_control_caller, local_control_scan_refresh_payload,
        parse_guardrail_mode,
    };
    use mesh_client::client::control_plane::OwnerControlScanRefreshResult;

    #[test]
    fn loopback_control_caller_accepts_localhost_only() {
        assert!(is_loopback_control_caller(
            "127.0.0.1:3131".parse().unwrap()
        ));
        assert!(is_loopback_control_caller("[::1]:3131".parse().unwrap()));
        assert!(!is_loopback_control_caller(
            "192.0.2.10:3131".parse().unwrap()
        ));
        assert!(!is_loopback_control_caller(
            "[2001:db8::1]:3131".parse().unwrap()
        ));
    }

    #[test]
    fn parse_guardrail_mode_accepts_operator_labels() {
        assert_eq!(
            parse_guardrail_mode("disabled"),
            Some(GuardrailMode::Disabled)
        );
        assert_eq!(parse_guardrail_mode("off"), Some(GuardrailMode::Disabled));
        assert_eq!(
            parse_guardrail_mode("metrics-only"),
            Some(GuardrailMode::MetricsOnly)
        );
        assert_eq!(
            parse_guardrail_mode("enforce"),
            Some(GuardrailMode::Enforce)
        );
    }

    #[test]
    fn parse_guardrail_mode_rejects_unknown_labels() {
        assert_eq!(parse_guardrail_mode(""), None);
        assert_eq!(parse_guardrail_mode("strict"), None);
    }

    #[test]
    fn scan_refresh_payload_preserves_snapshot_only_compatibility() {
        let payload = local_control_scan_refresh_payload(OwnerControlScanRefreshResult {
            snapshot: mesh_client::proto::node::OwnerControlConfigSnapshot {
                node_id: vec![0xab, 0xcd],
                ..Default::default()
            },
            inventory: None,
        });

        assert_eq!(
            serde_json::to_value(payload).unwrap(),
            serde_json::json!({
                "target_node_id": "abcd",
                "disposition": null,
                "inventory": null,
            })
        );
    }

    #[test]
    fn scan_refresh_payload_maps_sorted_inventory_and_metadata_explicitly() {
        let payload = local_control_scan_refresh_payload(OwnerControlScanRefreshResult {
            snapshot: mesh_client::proto::node::OwnerControlConfigSnapshot {
                node_id: vec![1],
                ..Default::default()
            },
            inventory: Some(mesh_client::proto::node::OwnerControlRefreshInventory {
                entries: vec![mesh_client::proto::node::OwnerControlInventoryEntry {
                    canonical_model_ref: "a/model".to_string(),
                    display_name: Some("Model A".to_string()),
                    total_size_bytes: 42,
                    metadata: Some(mesh_client::proto::node::CompactModelMetadata {
                        model_key: "a/model".to_string(),
                        architecture: "llama".to_string(),
                        special_tokens: vec![mesh_client::proto::node::SpecialToken {
                            name: "bos".to_string(),
                            token_id: 1,
                        }],
                        ..Default::default()
                    }),
                }],
                disposition:
                    mesh_client::proto::node::OwnerControlRefreshInventoryDisposition::Executed
                        as i32,
            }),
        });

        let value = serde_json::to_value(payload).unwrap();
        assert_eq!(value["target_node_id"], "01");
        assert_eq!(value["disposition"], "executed");
        assert_eq!(value["inventory"][0]["canonical_model_ref"], "a/model");
        assert_eq!(value["inventory"][0]["metadata"]["architecture"], "llama");
        assert_eq!(
            value["inventory"][0]["metadata"]["special_tokens"][0]["name"],
            "bos"
        );
    }
}
