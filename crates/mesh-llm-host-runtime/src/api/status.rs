//! Public status/model payloads and serialization compatibility anchors.
//!
//! Keep these shapes stable; the API layer and collector tests rely on them.

use super::{RuntimeModelPayload, RuntimeProcessPayload};
use crate::crypto::{OwnershipStatus, OwnershipSummary, ReleaseAttestationSummary};
use crate::mesh::requirements::{MeshRequirementPolicySummary, MeshRequirementRejectionEvent};
use crate::network::{affinity, metrics};
use crate::runtime_data;
use crate::system::hardware::expand_gpu_names;
mod runtime;

pub(crate) use runtime::*;
use serde::Serialize;
use skippy_server::OpenAiGuardrailsStatus;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum NodeState {
    Client,
    #[default]
    Standby,
    Loading,
    Serving,
}

impl NodeState {
    pub(crate) const fn node_status_alias(self) -> &'static str {
        match self {
            Self::Client => "Client",
            Self::Standby => "Standby",
            Self::Loading => "Loading",
            Self::Serving => "Serving",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum WakeableNodeState {
    Sleeping,
    Waking,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct RuntimeStatusPayload {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) backend: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) openai_guardrails: Option<OpenAiGuardrailsPayload>,
    pub(crate) models: Vec<RuntimeModelPayload>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub(crate) stages: Vec<RuntimeStagePayload>,

    // ── Runtime daemon state (backward-compatible optional fields) ────────────
    /// Derived runtime daemon state. Optional for backward compatibility.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) daemon_state: Option<DaemonState>,
    /// Coexistence capability booleans. Optional for backward compatibility.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) capabilities: Option<RuntimeCapabilityFlags>,
    /// Bounded lifecycle instance payloads. Optional for backward compatibility.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub(crate) lifecycle_instances: Vec<LifecycleInstancePayload>,
    /// Intent summary counts and errors. Optional for backward compatibility.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) intent_summary: Option<IntentSummary>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct OpenAiGuardrailsPayload {
    pub(crate) mode: &'static str,
    pub(crate) target: &'static str,
    pub(crate) streaming: &'static str,
    pub(crate) retry_exhaustion: &'static str,
    pub(crate) small_model_policy: &'static str,
    pub(crate) small_param_threshold_b: f32,
    pub(crate) max_tool_retries: u8,
    pub(crate) max_structured_retries: u8,
}

impl From<OpenAiGuardrailsStatus> for OpenAiGuardrailsPayload {
    fn from(value: OpenAiGuardrailsStatus) -> Self {
        Self {
            mode: value.mode,
            target: value.target,
            streaming: value.streaming,
            retry_exhaustion: value.retry_exhaustion,
            small_model_policy: value.small_model_policy,
            small_param_threshold_b: value.small_param_threshold_b,
            max_tool_retries: value.max_tool_retries,
            max_structured_retries: value.max_structured_retries,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct RuntimeStagePayload {
    pub(crate) topology_id: String,
    pub(crate) run_id: String,
    pub(crate) model_id: String,
    pub(crate) backend: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) package_ref: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) manifest_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) source_model_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) source_model_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) source_model_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) materialized_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) materialized_bytes: Option<u64>,
    pub(crate) materialized_pinned: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) projector_path: Option<String>,
    pub(crate) multimodal: bool,
    pub(crate) stage_id: String,
    pub(crate) stage_index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) node_id: Option<String>,
    pub(crate) layer_start: u32,
    pub(crate) layer_end: u32,
    pub(crate) state: &'static str,
    pub(crate) bind_addr: String,
    pub(crate) activation_width: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) selected_device: Option<RuntimeStageDevicePayload>,
    pub(crate) ctx_size: u32,
    pub(crate) lane_count: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) error: Option<String>,
    pub(crate) shutdown_generation: u64,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct RuntimeStageDevicePayload {
    pub(crate) backend_device: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) stable_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) index: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) vram_bytes: Option<u64>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct RuntimeProcessesPayload {
    pub(crate) processes: Vec<RuntimeProcessPayload>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct RuntimeLlamaPayload {
    pub(crate) metrics: RuntimeLlamaMetricsPayload,
    pub(crate) slots: RuntimeLlamaSlotsPayload,
    pub(crate) items: RuntimeLlamaItemsPayload,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub(crate) instances: Vec<RuntimeLlamaInstancePayload>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct RuntimeLlamaInstancePayload {
    pub(crate) instance_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) model: Option<String>,
    pub(crate) metrics: RuntimeLlamaMetricsPayload,
    pub(crate) slots: RuntimeLlamaSlotsPayload,
    pub(crate) items: RuntimeLlamaItemsPayload,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct RuntimeLlamaMetricsPayload {
    pub(crate) status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) last_attempt_unix_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) last_success_unix_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) raw_text: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub(crate) samples: Vec<RuntimeLlamaMetricSamplePayload>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct RuntimeLlamaMetricSamplePayload {
    pub(crate) name: String,
    #[serde(skip_serializing_if = "BTreeMap::is_empty", default)]
    pub(crate) labels: BTreeMap<String, String>,
    pub(crate) value: f64,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct RuntimeLlamaSlotsPayload {
    pub(crate) status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) instance_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) last_attempt_unix_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) last_success_unix_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) error: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub(crate) slots: Vec<RuntimeLlamaSlotPayload>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct RuntimeLlamaSlotPayload {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) id: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) id_task: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) n_ctx: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) speculative: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) is_processing: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) next_token: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) params: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "serde_json::Value::is_null")]
    pub(crate) extra: serde_json::Value,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct RuntimeLlamaItemsPayload {
    pub(crate) metrics: Vec<RuntimeLlamaMetricItemPayload>,
    pub(crate) slots: Vec<RuntimeLlamaSlotItemPayload>,
    pub(crate) slots_total: usize,
    pub(crate) slots_busy: usize,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct RuntimeLlamaMetricItemPayload {
    pub(crate) name: String,
    #[serde(skip_serializing_if = "BTreeMap::is_empty", default)]
    pub(crate) labels: BTreeMap<String, String>,
    pub(crate) value: f64,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct RuntimeLlamaSlotItemPayload {
    pub(crate) index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) id: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) id_task: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) n_ctx: Option<u64>,
    pub(crate) is_processing: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct GpuEntry {
    pub(crate) name: String,
    pub(crate) vram_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) rated_vram_gb: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) reserved_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) allocatable_vram_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) mem_bandwidth_gbps: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) compute_tflops_fp32: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) compute_tflops_fp16: Option<f64>,
}

fn inferred_gpu_name_count(gpu_name: Option<&str>) -> usize {
    let Some(raw) = gpu_name.map(str::trim) else {
        return 0;
    };
    if raw.is_empty() {
        return 0;
    }

    raw.split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.split_once('×')
                .or_else(|| part.split_once('x'))
                .or_else(|| part.split_once('X'))
                .and_then(|(count, _)| count.trim().parse::<usize>().ok())
                .filter(|&count| count > 0)
                .unwrap_or(1)
        })
        .sum()
}

pub(crate) fn build_gpus(
    gpu_name: Option<&str>,
    gpu_vram: Option<&str>,
    gpu_reserved_bytes: Option<&str>,
    gpu_mem_bandwidth: Option<&str>,
    gpu_compute_tflops_fp32: Option<&str>,
    gpu_compute_tflops_fp16: Option<&str>,
) -> Vec<GpuEntry> {
    let vrams: Vec<Option<u64>> = gpu_vram
        .map(|s| s.split(',').map(|v| v.trim().parse::<u64>().ok()).collect())
        .unwrap_or_default();
    let reserved: Vec<Option<u64>> = gpu_reserved_bytes
        .map(|s| s.split(',').map(|v| v.trim().parse::<u64>().ok()).collect())
        .unwrap_or_default();
    let bandwidths: Vec<Option<f64>> = gpu_mem_bandwidth
        .map(|s| s.split(',').map(|v| v.trim().parse::<f64>().ok()).collect())
        .unwrap_or_default();
    let compute_fp32: Vec<Option<f64>> = gpu_compute_tflops_fp32
        .map(|s| s.split(',').map(|v| v.trim().parse::<f64>().ok()).collect())
        .unwrap_or_default();
    let compute_fp16: Vec<Option<f64>> = gpu_compute_tflops_fp16
        .map(|s| s.split(',').map(|v| v.trim().parse::<f64>().ok()).collect())
        .unwrap_or_default();
    let expected_count = [
        vrams.len(),
        reserved.len(),
        bandwidths.len(),
        compute_fp32.len(),
        compute_fp16.len(),
        inferred_gpu_name_count(gpu_name),
    ]
    .into_iter()
    .max()
    .unwrap_or(0);
    let names = expand_gpu_names(gpu_name, expected_count)
        .into_iter()
        .filter(|name| !name.is_empty())
        .collect::<Vec<_>>();
    if names.is_empty() {
        return vec![];
    }
    names
        .into_iter()
        .enumerate()
        .map(|(i, name)| GpuEntry {
            name,
            vram_bytes: vrams.get(i).copied().flatten().unwrap_or(0),
            rated_vram_gb: mesh_llm_system::vram::rated_capacity_gb(
                vrams.get(i).copied().flatten().unwrap_or(0),
            ),
            reserved_bytes: reserved.get(i).copied().flatten(),
            allocatable_vram_bytes: Some(mesh_llm_system::vram::allocatable_bytes(
                vrams.get(i).copied().flatten().unwrap_or(0),
                reserved.get(i).copied().flatten(),
            )),
            mem_bandwidth_gbps: bandwidths.get(i).copied().flatten(),
            compute_tflops_fp32: compute_fp32.get(i).copied().flatten(),
            compute_tflops_fp16: compute_fp16.get(i).copied().flatten(),
        })
        .collect()
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct StatusPayload {
    pub(crate) version: String,
    pub(crate) latest_version: Option<String>,
    pub(crate) node_id: String,
    pub(crate) owner: OwnershipPayload,
    pub(crate) release_attestation: ReleaseAttestationSummary,
    pub(crate) token: String,
    pub(crate) node_state: NodeState,
    pub(crate) node_status: String,
    pub(crate) is_host: bool,
    pub(crate) is_client: bool,
    pub(crate) llama_ready: bool,
    pub(crate) runtime: RuntimeStatusPayload,
    pub(crate) model_name: String,
    pub(crate) models: Vec<String>,
    pub(crate) available_models: Vec<String>,
    pub(crate) requested_models: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub(crate) wanted_model_refs: Vec<String>,
    pub(crate) serving_models: Vec<String>,
    pub(crate) hosted_models: Vec<String>,
    pub(crate) draft_name: Option<String>,
    pub(crate) api_port: u16,
    pub(crate) my_vram_gb: f64,
    pub(crate) model_size_gb: f64,
    pub(crate) peers: Vec<PeerPayload>,
    pub(crate) wakeable_nodes: Vec<WakeableNode>,
    pub(crate) local_instances: Vec<LocalInstance>,
    pub(crate) launch_pi: Option<String>,
    pub(crate) launch_goose: Option<String>,
    pub(crate) inflight_requests: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) mesh_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) mesh_name: Option<String>,
    pub(crate) mesh_discovery_mode: String,
    pub(crate) discovery_scope: String,
    pub(crate) discovery_source: String,
    pub(crate) nostr_discovery: bool,
    /// Best-effort publication state per Issue #240: private | public | publish_failed.
    pub(crate) publication_state: String,
    pub(crate) my_hostname: Option<String>,
    pub(crate) my_is_soc: Option<bool>,
    pub(crate) gpus: Vec<GpuEntry>,
    pub(crate) routing_affinity: affinity::AffinityStatsSnapshot,
    /// Local-only routing outcome and current-node pressure snapshot measured on
    /// this node only; not mesh-wide aggregates.
    pub(crate) routing_metrics: metrics::RoutingMetricsStatusSnapshot,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) first_joined_mesh_ts: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) mesh_requirements: Option<MeshRequirementPolicySummary>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub(crate) recent_mesh_rejections: Vec<MeshRequirementRejectionEvent>,
    /// Local-only logging capability and worker health. Omitted until logging
    /// state has been initialized so older consumers retain their prior shape.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) logging: Option<LoggingStatusPayload>,
}

/// Path-free local logging health. This is deliberately absent from mesh
/// payloads and contains no storage paths, raw identifiers, or backend errors.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct LoggingStatusPayload {
    pub(crate) metadata_available: bool,
    pub(crate) metadata_state: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) schema_version: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) supported_schema_version: Option<u32>,
    pub(crate) capture_mode: &'static str,
    /// Artifact storage availability only; see `artifact_capture_ready` for
    /// whether production request paths are actually wired to capture.
    pub(crate) artifact_capture_available: bool,
    pub(crate) artifact_capture_ready: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) artifact_capture_degradation: Option<&'static str>,
    pub(crate) persistence: LoggingPersistenceStatusPayload,
    pub(crate) cleanup: LoggingCleanupStatusPayload,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct LoggingPersistenceStatusPayload {
    pub(crate) state: &'static str,
    pub(crate) queue_drops: u64,
    pub(crate) failures: u64,
    pub(crate) shutdown_losses: u64,
    pub(crate) outstanding: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct LoggingCleanupStatusPayload {
    pub(crate) state: &'static str,
    pub(crate) shutdown_timeouts: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) last_outcome: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) last_deleted_count: Option<u64>,
}

impl From<crate::logging::LoggingRuntimeStatus> for LoggingStatusPayload {
    fn from(status: crate::logging::LoggingRuntimeStatus) -> Self {
        Self {
            metadata_available: status.metadata_available,
            metadata_state: status.metadata_state,
            schema_version: status.schema_version,
            supported_schema_version: status.supported_schema_version,
            capture_mode: status.capture_mode,
            artifact_capture_available: status.artifact_capture_available,
            artifact_capture_ready: status.artifact_capture_ready,
            artifact_capture_degradation: status.artifact_capture_degradation,
            persistence: LoggingPersistenceStatusPayload {
                state: status.persistence_worker_state,
                queue_drops: status.persistence_queue_drops,
                failures: status.persistence_failures,
                shutdown_losses: status.persistence_shutdown_losses,
                outstanding: status.persistence_outstanding,
            },
            cleanup: LoggingCleanupStatusPayload {
                state: status.cleanup_worker_state,
                shutdown_timeouts: status.cleanup_shutdown_timeouts,
                last_outcome: status.cleanup_last_outcome,
                last_deleted_count: status.cleanup_last_deleted_count,
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct WakeableNode {
    pub(crate) logical_id: String,
    pub(crate) models: Vec<String>,
    pub(crate) vram_gb: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) provider: Option<String>,
    pub(crate) state: WakeableNodeState,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) wake_eta_secs: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct PeerPayload {
    pub(crate) id: String,
    pub(crate) owner: OwnershipPayload,
    pub(crate) release_attestation: ReleaseAttestationSummary,
    pub(crate) role: String,
    pub(crate) state: NodeState,
    pub(crate) models: Vec<String>,
    pub(crate) available_models: Vec<String>,
    pub(crate) requested_models: Vec<String>,
    pub(crate) vram_gb: f64,
    pub(crate) serving_models: Vec<String>,
    pub(crate) hosted_models: Vec<String>,
    pub(crate) hosted_models_known: bool,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub(crate) advertised_model_throughput: Vec<metrics::ModelThroughputHint>,
    pub(crate) version: Option<String>,
    pub(crate) rtt_ms: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) latency_ms: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) latency_source: Option<LatencySource>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) latency_age_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) latency_observer_id: Option<String>,
    pub(crate) hostname: Option<String>,
    pub(crate) is_soc: Option<bool>,
    pub(crate) gpus: Vec<GpuEntry>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) first_joined_mesh_ts: Option<u64>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum LatencySource {
    #[default]
    Direct,
    Estimated,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct OwnershipPayload {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) owner_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) cert_id: Option<String>,
    pub(crate) status: String,
    pub(crate) verified: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) expires_at_unix_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) node_label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) hostname_hint: Option<String>,
}

pub(crate) fn build_ownership_payload(summary: &OwnershipSummary) -> OwnershipPayload {
    OwnershipPayload {
        owner_id: summary.owner_id.clone(),
        cert_id: summary.cert_id.clone(),
        status: match summary.status {
            OwnershipStatus::Verified => "verified",
            OwnershipStatus::Unsigned => "unsigned",
            OwnershipStatus::Expired => "expired",
            OwnershipStatus::InvalidSignature => "invalid_signature",
            OwnershipStatus::MismatchedNodeId => "mismatched_node_id",
            OwnershipStatus::RevokedOwner => "revoked_owner",
            OwnershipStatus::RevokedCert => "revoked_cert",
            OwnershipStatus::RevokedNodeId => "revoked_node_id",
            OwnershipStatus::UnsupportedProtocol => "unsupported_protocol",
            OwnershipStatus::UntrustedOwner => "untrusted_owner",
        }
        .to_string(),
        verified: summary.verified,
        expires_at_unix_ms: summary.expires_at_unix_ms,
        node_label: summary.node_label.clone(),
        hostname_hint: summary.hostname_hint.clone(),
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct LocalInstance {
    pub(crate) pid: u32,
    pub(crate) api_port: Option<u16>,
    pub(crate) version: Option<String>,
    pub(crate) started_at_unix: i64,
    pub(crate) runtime_dir: String,
    pub(crate) is_self: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct MeshModelPayload {
    pub(crate) name: String,
    pub(crate) display_name: String,
    pub(crate) status: String,
    pub(crate) node_count: usize,
    pub(crate) mesh_vram_gb: f64,
    pub(crate) size_gb: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) architecture: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) context_length: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) quantization: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tokenizer: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) layer_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) head_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) embedding_size: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) description: Option<String>,
    pub(crate) multimodal: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) multimodal_status: Option<&'static str>,
    pub(crate) vision: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) vision_status: Option<&'static str>,
    pub(crate) audio: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) audio_status: Option<&'static str>,
    pub(crate) reasoning: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) reasoning_status: Option<&'static str>,
    pub(crate) tool_use: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tool_use_status: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) draft_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) request_count: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) last_active_secs_ago: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) target_rank: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) explicit_interest_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) wanted: Option<bool>,
    /// Local-only per-model routing outcome snapshot measured on the current
    /// node only; not mesh-wide aggregates.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) routing_metrics: Option<metrics::ModelRoutingMetricsSnapshot>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) source_page_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) source_ref: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) source_revision: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) source_file: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub(crate) active_nodes: Vec<String>,
    pub(crate) fit_label: String,
    pub(crate) fit_detail: String,
    pub(crate) download_command: String,
    pub(crate) run_command: String,
    pub(crate) auto_command: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct ModelTargetPayload {
    pub(crate) rank: usize,
    pub(crate) model_ref: String,
    pub(crate) display_name: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub(crate) profile: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) model_name: Option<String>,
    pub(crate) explicit_interest_count: usize,
    pub(crate) request_count: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) last_active_secs_ago: Option<u64>,
    pub(crate) serving_node_count: usize,
    pub(crate) requested: bool,
    pub(crate) wanted: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) wanted_reason: Option<&'static str>,
    pub(crate) capacity_advice: ModelTargetCapacityAdvicePayload,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct ModelTargetCapacityAdvicePayload {
    pub(crate) state: ModelTargetCapacityAdviceState,
    pub(crate) reason: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) required_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) best_single_node_capacity_bytes: Option<u64>,
    pub(crate) aggregate_capacity_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) shortfall_bytes: Option<u64>,
    pub(crate) eligible_node_count: usize,
    pub(crate) missing_capacity_node_count: usize,
    pub(crate) excluded_client_node_count: usize,
    pub(crate) split_capable: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ModelTargetCapacityAdviceState {
    AlreadyServing,
    SingleNodeFit,
    SplitCandidate,
    InsufficientCapacity,
    UnknownModelSize,
    UnknownCapacity,
    NoEligibleHosts,
}

pub(crate) fn build_runtime_status_payload(
    model_name: &str,
    primary_backend: Option<String>,
    openai_guardrails: Option<OpenAiGuardrailsPayload>,
    is_host: bool,
    llama_ready: bool,
    llama_port: Option<u16>,
    mut local_processes: Vec<RuntimeProcessPayload>,
) -> RuntimeStatusPayload {
    local_processes.sort_by(|left, right| {
        (
            left.name.to_lowercase(),
            left.instance_id.as_deref().unwrap_or(""),
            left.port,
        )
            .cmp(&(
                right.name.to_lowercase(),
                right.instance_id.as_deref().unwrap_or(""),
                right.port,
            ))
    });
    let backend = primary_backend.clone();

    let mut models: Vec<RuntimeModelPayload> = local_processes
        .into_iter()
        .map(|process| RuntimeModelPayload {
            name: process.name,
            instance_id: process.instance_id,
            profile: process.profile,
            backend: process.backend,
            status: process.status,
            port: Some(process.port),
            context_length: process.context_length,
        })
        .collect();

    let has_model_process = models.iter().any(|model| model.name == model_name);
    if is_host && !llama_ready && !has_model_process && !model_name.is_empty() {
        models.insert(
            0,
            RuntimeModelPayload {
                name: model_name.to_string(),
                instance_id: None,
                profile: String::new(),
                backend: primary_backend.unwrap_or_else(|| "unknown".into()),
                status: "starting".into(),
                port: llama_port,
                context_length: None,
            },
        );
    }

    RuntimeStatusPayload {
        backend,
        openai_guardrails,
        models,
        stages: vec![],
        daemon_state: None,
        capabilities: None,
        lifecycle_instances: vec![],
        intent_summary: None,
    }
}

pub(crate) fn build_runtime_stage_payloads(
    mut statuses: Vec<crate::mesh::StageRuntimeStatus>,
) -> Vec<RuntimeStagePayload> {
    statuses.sort_by(|left, right| {
        (
            &left.model_id,
            &left.topology_id,
            &left.run_id,
            left.stage_index,
            &left.stage_id,
        )
            .cmp(&(
                &right.model_id,
                &right.topology_id,
                &right.run_id,
                right.stage_index,
                &right.stage_id,
            ))
    });

    statuses
        .into_iter()
        .map(|status| {
            let multimodal = status.projector_path.is_some();
            RuntimeStagePayload {
                topology_id: status.topology_id,
                run_id: status.run_id,
                model_id: status.model_id,
                backend: status.backend,
                package_ref: status.package_ref,
                manifest_sha256: status.manifest_sha256,
                source_model_path: status.source_model_path,
                source_model_sha256: status.source_model_sha256,
                source_model_bytes: status.source_model_bytes,
                materialized_bytes: materialized_stage_bytes(status.materialized_path.as_deref()),
                materialized_path: status.materialized_path,
                materialized_pinned: status.materialized_pinned,
                projector_path: status.projector_path,
                multimodal,
                stage_id: status.stage_id,
                stage_index: status.stage_index,
                node_id: status.node_id.map(|id| id.to_string()),
                layer_start: status.layer_start,
                layer_end: status.layer_end,
                state: runtime_stage_state_label(status.state),
                bind_addr: status.bind_addr,
                activation_width: status.activation_width,
                selected_device: status
                    .selected_device
                    .map(|device| RuntimeStageDevicePayload {
                        backend_device: device.backend_device,
                        stable_id: device.stable_id,
                        index: device.index,
                        vram_bytes: device.vram_bytes,
                    }),
                ctx_size: status.ctx_size,
                lane_count: status.lane_count,
                error: status.error,
                shutdown_generation: status.shutdown_generation,
            }
        })
        .collect()
}

fn materialized_stage_bytes(path: Option<&str>) -> Option<u64> {
    let path = path?;
    let metadata = std::fs::metadata(path).ok()?;
    metadata.is_file().then_some(metadata.len())
}

pub(crate) fn runtime_stage_state_label(
    state: crate::inference::skippy::StageRuntimeState,
) -> &'static str {
    match state {
        crate::inference::skippy::StageRuntimeState::Starting => "starting",
        crate::inference::skippy::StageRuntimeState::Ready => "ready",
        crate::inference::skippy::StageRuntimeState::Stopping => "stopping",
        crate::inference::skippy::StageRuntimeState::Stopped => "stopped",
        crate::inference::skippy::StageRuntimeState::Failed => "failed",
    }
}

pub(super) fn build_runtime_processes_payload(
    mut local_processes: Vec<RuntimeProcessPayload>,
) -> RuntimeProcessesPayload {
    local_processes.sort_by(|left, right| {
        (
            left.name.to_lowercase(),
            left.instance_id.as_deref().unwrap_or(""),
            left.port,
        )
            .cmp(&(
                right.name.to_lowercase(),
                right.instance_id.as_deref().unwrap_or(""),
                right.port,
            ))
    });
    RuntimeProcessesPayload {
        processes: local_processes,
    }
}

pub(crate) fn build_runtime_llama_payload(
    snapshot: runtime_data::RuntimeLlamaRuntimeSnapshot,
    snapshots_by_instance: BTreeMap<String, runtime_data::RuntimeLlamaRuntimeSnapshot>,
) -> RuntimeLlamaPayload {
    let instances = snapshots_by_instance
        .into_iter()
        .map(|(instance_id, snapshot)| {
            let model = snapshot.slots.model.clone();
            let (metrics, slots, items) = build_runtime_llama_snapshot_payload(snapshot);
            RuntimeLlamaInstancePayload {
                instance_id,
                model,
                metrics,
                slots,
                items,
            }
        })
        .collect();
    let (metrics, slots, items) = build_runtime_llama_snapshot_payload(snapshot);
    RuntimeLlamaPayload {
        metrics,
        slots,
        items,
        instances,
    }
}

fn build_runtime_llama_snapshot_payload(
    snapshot: runtime_data::RuntimeLlamaRuntimeSnapshot,
) -> (
    RuntimeLlamaMetricsPayload,
    RuntimeLlamaSlotsPayload,
    RuntimeLlamaItemsPayload,
) {
    (
        RuntimeLlamaMetricsPayload {
            status: runtime_llama_endpoint_status(snapshot.metrics.status),
            last_attempt_unix_ms: snapshot.metrics.last_attempt_unix_ms,
            last_success_unix_ms: snapshot.metrics.last_success_unix_ms,
            error: snapshot.metrics.error,
            raw_text: snapshot.metrics.raw_text,
            samples: snapshot
                .metrics
                .samples
                .into_iter()
                .map(|sample| RuntimeLlamaMetricSamplePayload {
                    name: sample.name,
                    labels: sample.labels,
                    value: sample.value,
                })
                .collect(),
        },
        RuntimeLlamaSlotsPayload {
            status: runtime_llama_endpoint_status(snapshot.slots.status),
            model: snapshot.slots.model,
            instance_id: snapshot.slots.instance_id,
            last_attempt_unix_ms: snapshot.slots.last_attempt_unix_ms,
            last_success_unix_ms: snapshot.slots.last_success_unix_ms,
            error: snapshot.slots.error,
            slots: snapshot
                .slots
                .slots
                .into_iter()
                .map(|slot| RuntimeLlamaSlotPayload {
                    id: slot.id,
                    id_task: slot.id_task,
                    n_ctx: slot.n_ctx,
                    speculative: slot.speculative,
                    is_processing: slot.is_processing,
                    next_token: slot.next_token,
                    params: slot.params,
                    extra: slot.extra,
                })
                .collect(),
        },
        RuntimeLlamaItemsPayload {
            metrics: snapshot
                .items
                .metrics
                .into_iter()
                .map(|item| RuntimeLlamaMetricItemPayload {
                    name: item.name,
                    labels: item.labels,
                    value: item.value,
                })
                .collect(),
            slots: snapshot
                .items
                .slots
                .into_iter()
                .map(|item| RuntimeLlamaSlotItemPayload {
                    index: item.index,
                    id: item.id,
                    id_task: item.id_task,
                    n_ctx: item.n_ctx,
                    is_processing: item.is_processing,
                })
                .collect(),
            slots_total: snapshot.items.slots_total,
            slots_busy: snapshot.items.slots_busy,
        },
    )
}

fn runtime_llama_endpoint_status(status: runtime_data::RuntimeLlamaEndpointStatus) -> &'static str {
    match status {
        runtime_data::RuntimeLlamaEndpointStatus::Ready => "ready",
        runtime_data::RuntimeLlamaEndpointStatus::Unavailable => "unavailable",
    }
}

pub(crate) fn classify_runtime_error(msg: &str) -> u16 {
    if msg.contains("not loaded") {
        404
    } else if msg.contains("already loaded") || msg.contains("multiple loaded instances") {
        409
    } else if msg.contains("fit locally")
        || msg.contains("runtime load only supports")
        || msg.contains("runtime capacity")
    {
        422
    } else {
        400
    }
}

pub(super) fn decode_runtime_model_path(path: &str, prefix: &str) -> Option<String> {
    let raw = path.strip_prefix(prefix)?;
    if raw.is_empty() {
        return None;
    }

    let bytes = raw.as_bytes();
    let mut decoded: Vec<u8> = Vec::with_capacity(raw.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 2 < bytes.len() => {
                let hi = bytes[i + 1] as char;
                let lo = bytes[i + 2] as char;
                let hex = [hi, lo].iter().collect::<String>();
                if let Ok(value) = u8::from_str_radix(&hex, 16) {
                    decoded.push(value);
                    i += 3;
                    continue;
                } else {
                    return None;
                }
            }
            b'+' => decoded.push(b'+'),
            b => decoded.push(b),
        }
        i += 1;
    }
    String::from_utf8(decoded).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ReleaseAttestationSummary;

    fn test_owner_payload() -> OwnershipPayload {
        OwnershipPayload {
            owner_id: None,
            cert_id: None,
            status: "unsigned".to_string(),
            verified: false,
            expires_at_unix_ms: None,
            node_label: None,
            hostname_hint: None,
        }
    }

    fn test_release_attestation_summary() -> ReleaseAttestationSummary {
        ReleaseAttestationSummary::default()
    }

    #[test]
    fn materialized_stage_bytes_reports_existing_file_size() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("stage-0.gguf");
        std::fs::write(&path, b"stage").expect("write materialized stage");

        assert_eq!(
            materialized_stage_bytes(path.to_str()),
            Some(b"stage".len() as u64)
        );
        assert_eq!(materialized_stage_bytes(None), None);
        assert_eq!(
            materialized_stage_bytes(Some("/definitely/not/a/materialized/stage")),
            None
        );
    }

    #[test]
    fn test_peer_payload_serializes_version_field() {
        let peer = PeerPayload {
            id: "test-id".to_string(),
            owner: test_owner_payload(),
            release_attestation: test_release_attestation_summary(),
            role: "Worker".to_string(),
            state: NodeState::Standby,
            models: vec![],
            available_models: vec![],
            requested_models: vec![],
            vram_gb: 8.0,
            serving_models: vec![],
            hosted_models: vec![],
            hosted_models_known: false,
            advertised_model_throughput: vec![],
            version: Some("0.56.0".to_string()),
            rtt_ms: None,
            latency_ms: None,
            latency_source: None,
            latency_age_ms: None,
            latency_observer_id: None,
            hostname: None,
            is_soc: None,
            gpus: vec![],
            first_joined_mesh_ts: None,
        };

        let json = serde_json::to_string(&peer).expect("serialization failed");
        assert!(json.contains("\"version\":\"0.56.0\""));
        assert!(!json.contains("advertised_model_throughput"));
    }

    #[test]
    fn test_peer_payload_serializes_null_version() {
        let peer = PeerPayload {
            id: "test-id".to_string(),
            owner: test_owner_payload(),
            release_attestation: test_release_attestation_summary(),
            role: "Worker".to_string(),
            state: NodeState::Standby,
            models: vec![],
            available_models: vec![],
            requested_models: vec![],
            vram_gb: 8.0,
            serving_models: vec![],
            hosted_models: vec![],
            hosted_models_known: false,
            advertised_model_throughput: vec![],
            version: None,
            rtt_ms: None,
            latency_ms: None,
            latency_source: None,
            latency_age_ms: None,
            latency_observer_id: None,
            hostname: None,
            is_soc: None,
            gpus: vec![],
            first_joined_mesh_ts: None,
        };

        let json = serde_json::to_string(&peer).expect("serialization failed");
        assert!(json.contains("\"version\":null"));
    }

    #[test]
    fn test_status_payload_has_local_instances_field() {
        let instances: Vec<LocalInstance> = vec![];
        let json = serde_json::to_string(&instances).expect("serialization failed");
        assert_eq!(json, "[]");
    }

    #[test]
    fn status_payload_serializes_node_state_and_node_status_alias() {
        let status = StatusPayload {
            version: "0.60.2".to_string(),
            latest_version: None,
            node_id: "node-1".to_string(),
            owner: test_owner_payload(),
            release_attestation: test_release_attestation_summary(),
            token: "token-1".to_string(),
            node_state: NodeState::Loading,
            node_status: NodeState::Loading.node_status_alias().to_string(),
            is_host: true,
            is_client: false,
            llama_ready: false,
            runtime: RuntimeStatusPayload {
                backend: None,
                openai_guardrails: None,
                models: vec![],
                stages: vec![],
                daemon_state: None,
                capabilities: None,
                lifecycle_instances: vec![],
                intent_summary: None,
            },
            model_name: "Qwen".to_string(),
            models: vec![],
            available_models: vec![],
            requested_models: vec![],
            wanted_model_refs: vec![],
            serving_models: vec![],
            hosted_models: vec![],
            draft_name: None,
            api_port: 3131,
            my_vram_gb: 0.0,
            model_size_gb: 0.0,
            peers: vec![],
            wakeable_nodes: vec![],
            local_instances: vec![],
            launch_pi: None,
            launch_goose: None,
            inflight_requests: 0,
            mesh_id: None,
            mesh_name: None,
            mesh_discovery_mode: "nostr".into(),
            discovery_scope: "public".into(),
            discovery_source: "nostr-relay".into(),
            nostr_discovery: false,
            publication_state: "private".into(),
            my_hostname: None,
            my_is_soc: None,
            gpus: vec![],
            routing_affinity: affinity::AffinityStatsSnapshot::default(),
            routing_metrics: metrics::RoutingMetricsStatusSnapshot::default(),
            first_joined_mesh_ts: None,
            mesh_requirements: None,
            recent_mesh_rejections: vec![],
            logging: None,
        };

        let json = serde_json::to_string(&status).expect("serialization failed");
        assert!(json.contains("\"node_state\":\"loading\""));
        assert!(json.contains("\"node_status\":\"Loading\""));
        assert!(json.contains("\"mesh_discovery_mode\":\"nostr\""));
        assert!(json.contains("\"discovery_scope\":\"public\""));
        assert!(json.contains("\"discovery_source\":\"nostr-relay\""));
        assert!(
            !json.contains("\"logging\""),
            "uninitialized logging must preserve the older status shape"
        );
    }

    #[test]
    fn status_payload_keeps_node_status_for_compatibility() {
        let status = StatusPayload {
            version: "0.60.2".to_string(),
            latest_version: None,
            node_id: "node-1".to_string(),
            owner: test_owner_payload(),
            release_attestation: test_release_attestation_summary(),
            token: "token-1".to_string(),
            node_state: NodeState::Serving,
            node_status: NodeState::Serving.node_status_alias().to_string(),
            is_host: true,
            is_client: false,
            llama_ready: true,
            runtime: RuntimeStatusPayload {
                backend: None,
                openai_guardrails: None,
                models: vec![],
                stages: vec![],
                daemon_state: None,
                capabilities: None,
                lifecycle_instances: vec![],
                intent_summary: None,
            },
            model_name: "Qwen".to_string(),
            models: vec!["Qwen".to_string()],
            available_models: vec!["Qwen".to_string()],
            requested_models: vec!["Qwen".to_string()],
            wanted_model_refs: vec![],
            serving_models: vec!["Qwen".to_string()],
            hosted_models: vec!["Qwen".to_string()],
            draft_name: None,
            api_port: 3131,
            my_vram_gb: 24.0,
            model_size_gb: 4.0,
            peers: vec![],
            wakeable_nodes: vec![],
            local_instances: vec![],
            launch_pi: None,
            launch_goose: None,
            inflight_requests: 0,
            mesh_id: None,
            mesh_name: None,
            mesh_discovery_mode: "nostr".into(),
            discovery_scope: "public".into(),
            discovery_source: "nostr-relay".into(),
            nostr_discovery: false,
            publication_state: "private".into(),
            my_hostname: None,
            my_is_soc: None,
            gpus: vec![],
            routing_affinity: affinity::AffinityStatsSnapshot::default(),
            routing_metrics: metrics::RoutingMetricsStatusSnapshot::default(),
            first_joined_mesh_ts: None,
            mesh_requirements: None,
            recent_mesh_rejections: vec![],
            logging: None,
        };

        let json = serde_json::to_string(&status).expect("serialization failed");
        assert!(json.contains("\"node_state\":\"serving\""));
        assert!(json.contains("\"node_status\":\"Serving\""));
    }

    #[test]
    fn status_payload_serializes_wakeable_nodes_separately() {
        let status = StatusPayload {
            version: "0.60.2".to_string(),
            latest_version: None,
            node_id: "node-1".to_string(),
            owner: test_owner_payload(),
            release_attestation: test_release_attestation_summary(),
            token: "token-1".to_string(),
            node_state: NodeState::Standby,
            node_status: NodeState::Standby.node_status_alias().to_string(),
            is_host: false,
            is_client: false,
            llama_ready: false,
            runtime: RuntimeStatusPayload {
                backend: None,
                openai_guardrails: None,
                models: vec![],
                stages: vec![],
                daemon_state: None,
                capabilities: None,
                lifecycle_instances: vec![],
                intent_summary: None,
            },
            model_name: String::new(),
            models: vec![],
            available_models: vec![],
            requested_models: vec![],
            wanted_model_refs: vec![],
            serving_models: vec![],
            hosted_models: vec![],
            draft_name: None,
            api_port: 3131,
            my_vram_gb: 0.0,
            model_size_gb: 0.0,
            peers: vec![],
            wakeable_nodes: vec![WakeableNode {
                logical_id: "provider-node-1".to_string(),
                models: vec!["Qwen".to_string()],
                vram_gb: 24.0,
                provider: Some("fly".to_string()),
                state: WakeableNodeState::Sleeping,
                wake_eta_secs: Some(90),
            }],
            local_instances: vec![],
            launch_pi: None,
            launch_goose: None,
            inflight_requests: 0,
            mesh_id: None,
            mesh_name: None,
            mesh_discovery_mode: "nostr".into(),
            discovery_scope: "public".into(),
            discovery_source: "nostr-relay".into(),
            nostr_discovery: false,
            publication_state: "private".into(),
            my_hostname: None,
            my_is_soc: None,
            gpus: vec![],
            routing_affinity: affinity::AffinityStatsSnapshot::default(),
            routing_metrics: metrics::RoutingMetricsStatusSnapshot::default(),
            first_joined_mesh_ts: None,
            mesh_requirements: None,
            recent_mesh_rejections: vec![],
            logging: None,
        };

        let json = serde_json::to_value(&status).expect("serialization failed");
        assert_eq!(json["peers"], serde_json::json!([]));
        assert_eq!(json["wakeable_nodes"].as_array().map(Vec::len), Some(1));
        assert_eq!(json["wakeable_nodes"][0]["state"], "sleeping");
        assert_eq!(json["wakeable_nodes"][0]["logical_id"], "provider-node-1");
    }

    #[test]
    fn status_payload_defaults_to_empty_wakeable_inventory() {
        let status = StatusPayload {
            version: "0.60.2".to_string(),
            latest_version: None,
            node_id: "node-1".to_string(),
            owner: test_owner_payload(),
            release_attestation: test_release_attestation_summary(),
            token: "token-1".to_string(),
            node_state: NodeState::Standby,
            node_status: NodeState::Standby.node_status_alias().to_string(),
            is_host: false,
            is_client: false,
            llama_ready: false,
            runtime: RuntimeStatusPayload {
                backend: None,
                openai_guardrails: None,
                models: vec![],
                stages: vec![],
                daemon_state: None,
                capabilities: None,
                lifecycle_instances: vec![],
                intent_summary: None,
            },
            model_name: String::new(),
            models: vec![],
            available_models: vec![],
            requested_models: vec![],
            wanted_model_refs: vec![],
            serving_models: vec![],
            hosted_models: vec![],
            draft_name: None,
            api_port: 3131,
            my_vram_gb: 0.0,
            model_size_gb: 0.0,
            peers: vec![],
            wakeable_nodes: vec![],
            local_instances: vec![],
            launch_pi: None,
            launch_goose: None,
            inflight_requests: 0,
            mesh_id: None,
            mesh_name: None,
            mesh_discovery_mode: "nostr".into(),
            discovery_scope: "public".into(),
            discovery_source: "nostr-relay".into(),
            nostr_discovery: false,
            publication_state: "private".into(),
            my_hostname: None,
            my_is_soc: None,
            gpus: vec![],
            routing_affinity: affinity::AffinityStatsSnapshot::default(),
            routing_metrics: metrics::RoutingMetricsStatusSnapshot::default(),
            first_joined_mesh_ts: None,
            mesh_requirements: None,
            recent_mesh_rejections: vec![],
            logging: None,
        };

        let json = serde_json::to_value(&status).expect("serialization failed");
        assert_eq!(json["wakeable_nodes"], serde_json::json!([]));
        assert_eq!(json["peers"], serde_json::json!([]));
    }

    #[test]
    fn peer_status_serializes_state_without_mutating_role() {
        let peer = PeerPayload {
            id: "test-id".to_string(),
            owner: test_owner_payload(),
            release_attestation: test_release_attestation_summary(),
            role: "Host".to_string(),
            state: NodeState::Serving,
            models: vec![],
            available_models: vec![],
            requested_models: vec![],
            vram_gb: 8.0,
            serving_models: vec!["Qwen".to_string()],
            hosted_models: vec!["Qwen".to_string()],
            hosted_models_known: true,
            advertised_model_throughput: vec![],
            version: Some("0.60.2".to_string()),
            rtt_ms: Some(12),
            latency_ms: None,
            latency_source: None,
            latency_age_ms: None,
            latency_observer_id: None,
            hostname: Some("peer.local".to_string()),
            is_soc: Some(false),
            gpus: vec![],
            first_joined_mesh_ts: None,
        };

        let json = serde_json::to_string(&peer).expect("serialization failed");
        assert!(json.contains("\"role\":\"Host\""));
        assert!(json.contains("\"state\":\"serving\""));
    }

    #[test]
    fn runtime_status_payload_serializes_privacy_safe_openai_guardrails() {
        let payload = build_runtime_status_payload(
            "Qwen-Test",
            Some("skippy".to_string()),
            Some(OpenAiGuardrailsPayload::from(OpenAiGuardrailsStatus {
                mode: "metrics",
                target: "skippy",
                streaming: "pass_through",
                retry_exhaustion: "error",
                small_model_policy: "small_models_only",
                small_param_threshold_b: 9.0,
                max_tool_retries: 1,
                max_structured_retries: 2,
            })),
            true,
            true,
            Some(9337),
            vec![],
        );

        let json = serde_json::to_value(payload).expect("serialization failed");
        let guardrails = json["openai_guardrails"]
            .as_object()
            .expect("guardrails should be an object");
        assert_eq!(guardrails.get("mode"), Some(&serde_json::json!("metrics")));
        assert_eq!(guardrails.get("target"), Some(&serde_json::json!("skippy")));
        assert_eq!(
            guardrails.get("streaming"),
            Some(&serde_json::json!("pass_through"))
        );
        assert_eq!(
            guardrails.get("retry_exhaustion"),
            Some(&serde_json::json!("error"))
        );
        assert_eq!(
            guardrails.get("small_model_policy"),
            Some(&serde_json::json!("small_models_only"))
        );
        assert_eq!(
            guardrails.get("small_param_threshold_b"),
            Some(&serde_json::json!(9.0))
        );
        assert_eq!(
            guardrails.get("max_tool_retries"),
            Some(&serde_json::json!(1))
        );
        assert_eq!(
            guardrails.get("max_structured_retries"),
            Some(&serde_json::json!(2))
        );
        assert_eq!(guardrails.len(), 8);
        for forbidden in [
            "prompt",
            "schema",
            "tool_args",
            "tool_names",
            "reserved_tool_prefix",
            "sentinels",
        ] {
            assert!(
                guardrails.get(forbidden).is_none(),
                "privacy-safe status should omit {forbidden}"
            );
        }
    }

    #[test]
    fn runtime_status_payload_uses_disabled_guardrail_mode_label() {
        let payload = build_runtime_status_payload(
            "Qwen-Test",
            Some("skippy".to_string()),
            Some(OpenAiGuardrailsPayload::from(OpenAiGuardrailsStatus {
                mode: "disabled",
                target: "skippy",
                streaming: "pass_through",
                retry_exhaustion: "error",
                small_model_policy: "small_models_only",
                small_param_threshold_b: 9.0,
                max_tool_retries: 1,
                max_structured_retries: 2,
            })),
            true,
            true,
            Some(9337),
            vec![],
        );

        let json = serde_json::to_value(payload).expect("serialization failed");
        assert_eq!(
            json["openai_guardrails"]["mode"],
            serde_json::json!("disabled")
        );
        assert_ne!(json["openai_guardrails"]["mode"], serde_json::json!("off"));
    }

    #[test]
    fn test_local_instance_serializes_is_self() {
        let instance = LocalInstance {
            pid: 1234,
            api_port: Some(3131),
            version: Some("0.56.0".to_string()),
            started_at_unix: 1700000000,
            runtime_dir: "/home/user/.mesh-llm/runtime/1234".to_string(),
            is_self: true,
        };

        let json = serde_json::to_string(&instance).expect("serialization failed");
        assert!(json.contains("\"is_self\":true"));
    }

    #[test]
    fn logging_status_serializes_only_fixed_local_health_values() {
        let cases = [
            (
                "healthy",
                LoggingStatusPayload {
                    metadata_available: true,
                    metadata_state: "ready",
                    schema_version: None,
                    supported_schema_version: None,
                    capture_mode: "redacted_artifacts",
                    artifact_capture_available: true,
                    artifact_capture_ready: true,
                    artifact_capture_degradation: None,
                    persistence: LoggingPersistenceStatusPayload {
                        state: "running",
                        queue_drops: 0,
                        failures: 0,
                        shutdown_losses: 0,
                        outstanding: 1,
                    },
                    cleanup: LoggingCleanupStatusPayload {
                        state: "running",
                        shutdown_timeouts: 0,
                        last_outcome: Some("completed"),
                        last_deleted_count: Some(3),
                    },
                },
            ),
            (
                "disabled",
                LoggingStatusPayload {
                    metadata_available: false,
                    metadata_state: "disabled",
                    schema_version: None,
                    supported_schema_version: None,
                    capture_mode: "unavailable",
                    artifact_capture_available: false,
                    artifact_capture_ready: false,
                    artifact_capture_degradation: None,
                    persistence: LoggingPersistenceStatusPayload {
                        state: "unavailable",
                        queue_drops: 0,
                        failures: 0,
                        shutdown_losses: 0,
                        outstanding: 0,
                    },
                    cleanup: LoggingCleanupStatusPayload {
                        state: "not_started",
                        shutdown_timeouts: 0,
                        last_outcome: None,
                        last_deleted_count: None,
                    },
                },
            ),
            (
                "schema_incompatible",
                LoggingStatusPayload {
                    metadata_available: false,
                    metadata_state: "schema_incompatible",
                    schema_version: Some(2),
                    supported_schema_version: Some(1),
                    capture_mode: "unavailable",
                    artifact_capture_available: false,
                    artifact_capture_ready: false,
                    artifact_capture_degradation: None,
                    persistence: LoggingPersistenceStatusPayload {
                        state: "unavailable",
                        queue_drops: 0,
                        failures: 0,
                        shutdown_losses: 0,
                        outstanding: 0,
                    },
                    cleanup: LoggingCleanupStatusPayload {
                        state: "not_started",
                        shutdown_timeouts: 0,
                        last_outcome: None,
                        last_deleted_count: None,
                    },
                },
            ),
            (
                "artifact_degraded",
                LoggingStatusPayload {
                    metadata_available: true,
                    metadata_state: "ready",
                    schema_version: None,
                    supported_schema_version: None,
                    capture_mode: "redacted_artifacts",
                    artifact_capture_available: false,
                    artifact_capture_ready: false,
                    artifact_capture_degradation: Some(
                        "artifact_capture_disabled_privacy_unavailable",
                    ),
                    persistence: LoggingPersistenceStatusPayload {
                        state: "running",
                        queue_drops: 4,
                        failures: 2,
                        shutdown_losses: 0,
                        outstanding: 0,
                    },
                    cleanup: LoggingCleanupStatusPayload {
                        state: "running",
                        shutdown_timeouts: 0,
                        last_outcome: Some("failed"),
                        last_deleted_count: None,
                    },
                },
            ),
            (
                "shutdown",
                LoggingStatusPayload {
                    metadata_available: true,
                    metadata_state: "ready",
                    schema_version: None,
                    supported_schema_version: None,
                    capture_mode: "redacted_artifacts",
                    artifact_capture_available: true,
                    artifact_capture_ready: true,
                    artifact_capture_degradation: None,
                    persistence: LoggingPersistenceStatusPayload {
                        state: "stopped",
                        queue_drops: 0,
                        failures: 0,
                        shutdown_losses: 2,
                        outstanding: 0,
                    },
                    cleanup: LoggingCleanupStatusPayload {
                        state: "stopped",
                        shutdown_timeouts: 0,
                        last_outcome: Some("completed"),
                        last_deleted_count: Some(0),
                    },
                },
            ),
        ];

        for (name, payload) in cases {
            let json = serde_json::to_string(&payload).expect("serialize logging status");
            assert!(json.contains("\"metadata_available\""), "{name}");
            assert!(json.contains("\"persistence\""), "{name}");
            assert!(json.contains("\"cleanup\""), "{name}");
            for forbidden in [
                "sqlite",
                "database",
                "error_message",
                "path",
                "request_id",
                "token",
                "secret",
                "/private/",
            ] {
                assert!(
                    !json.contains(forbidden),
                    "{name} status leaked forbidden {forbidden}: {json}"
                );
            }
        }
    }

    #[test]
    fn logging_status_maps_runtime_snapshot_without_backend_details() {
        let payload = LoggingStatusPayload::from(crate::logging::LoggingRuntimeStatus {
            metadata_available: true,
            metadata_state: "ready",
            schema_version: None,
            supported_schema_version: None,
            capture_mode: "redacted_artifacts",
            artifact_capture_available: false,
            artifact_capture_ready: false,
            artifact_capture_degradation: Some("artifact_capture_disabled_privacy_unavailable"),
            persistence_worker_state: "stopped",
            persistence_queue_drops: 5,
            persistence_failures: 3,
            persistence_shutdown_losses: 2,
            persistence_outstanding: 0,
            cleanup_worker_state: "stopped",
            cleanup_shutdown_timeouts: 0,
            cleanup_last_outcome: Some("failed"),
            cleanup_last_deleted_count: None,
        });

        let json = serde_json::to_value(payload).expect("serialize runtime status mapping");
        assert_eq!(
            json["artifact_capture_degradation"],
            "artifact_capture_disabled_privacy_unavailable"
        );
        assert_eq!(json["persistence"]["state"], "stopped");
        assert_eq!(json["persistence"]["queue_drops"], 5);
        assert_eq!(json["persistence"]["failures"], 3);
        assert_eq!(json["persistence"]["shutdown_losses"], 2);
        assert_eq!(json["cleanup"]["state"], "stopped");
        assert_eq!(json["cleanup"]["shutdown_timeouts"], 0);
        assert_eq!(json["cleanup"]["last_outcome"], "failed");
        assert!(json["cleanup"].get("last_deleted_count").is_none());
    }
}
