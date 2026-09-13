use super::capacity::runtime_model_required_bytes;
use super::context_planning::{
    RuntimeResourcePlan, RuntimeResourcePlanInput, RuntimeResourcePlanningProfile,
    plan_runtime_resources,
};
use super::split_planning::format_gb;
use crate::api;
use crate::inference::{election, skippy};
use crate::mesh;
use crate::models;
use crate::network::router;
use crate::plugin;
use crate::runtime::survey;
use crate::runtime_data::{
    RuntimeLlamaEndpointStatus, RuntimeLlamaSlotSnapshot, RuntimeLlamaSlotsSnapshot,
};
use anyhow::{Context, Result};
use mesh_llm_events::{OutputEvent, emit_event};
use openai_frontend::OpenAiHookPolicy;
use skippy_protocol::{FlashAttentionType, LoadMode};
use skippy_server::serving_hooks::SharedModelServingHooksFactory;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use super::operational_logging::{
    LocalServingOperationalEvent, record_local_serving_operational_event,
};

mod native_runtime_events;

pub(super) use super::local_package::{
    SPLIT_DEFAULT_MIN_PARTICIPANTS, SplitParticipant, SplitParticipantExclusion,
    runtime_model_planning_bytes, scan_layer_package_metadata,
};
pub(super) use super::local_split::{
    SplitCoordinatorAck, SplitCoordinatorEvent, SplitCoordinatorLocalFallbackEvent,
    SplitCoordinatorReplaceEvent, SplitGenerationCleanup, SplitRuntimeReason, SplitRuntimeStart,
    StartupRuntimePlan, now_unix_nanos, start_runtime_split_model, startup_runtime_plan,
    stop_split_generation_cleanup,
};
pub(super) fn skippy_native_model_open_event_reporter(
    model_name: String,
) -> skippy::NativeModelOpenEventReporter {
    native_runtime_events::skippy_native_model_open_event_reporter(model_name)
}

pub(super) type OpenAiGuardrailPolicyHandle = openai_frontend::GuardrailPolicyHandle;

pub(super) fn openai_guardrail_policy_handle(
    mode: openai_frontend::GuardrailMode,
) -> OpenAiGuardrailPolicyHandle {
    OpenAiGuardrailPolicyHandle::new(openai_frontend::GuardrailPolicy {
        mode,
        ..openai_frontend::GuardrailPolicy::default()
    })
}

pub(super) fn set_openai_guardrail_policy_mode(
    handle: &OpenAiGuardrailPolicyHandle,
    mode: openai_frontend::GuardrailMode,
) {
    handle.set_mode(mode);
}

pub(super) enum RuntimeEvent {
    Exited {
        instance_id: String,
        model: String,
        port: u16,
    },
    ModelTargetReconciliationLoadFinished {
        model_ref: String,
        profile: String,
        result: std::result::Result<api::RuntimeLoadResponse, String>,
    },
    StartupModelLoadFinished {
        model_ref: String,
        profile: String,
        result: std::result::Result<api::RuntimeLoadResponse, String>,
    },
}

pub(super) enum LocalRuntimeBackendHandle {
    Skippy {
        model: skippy::SkippyModelHandle,
        http: skippy::SkippyHttpHandle,
        _death_tx: tokio::sync::oneshot::Sender<()>,
    },
}

pub(super) struct LocalRuntimeModelHandle {
    pub(super) port: u16,
    pub(super) backend: String,
    pub(super) context_length: u32,
    pub(super) slots: usize,
    pub(super) capabilities: models::ModelCapabilities,
    pub(super) inner: LocalRuntimeBackendHandle,
}

impl LocalRuntimeModelHandle {
    pub(super) fn pid(&self) -> u32 {
        match &self.inner {
            LocalRuntimeBackendHandle::Skippy { .. } => std::process::id(),
        }
    }

    pub(super) fn ctx_used_tokens(&self) -> Option<u64> {
        match &self.inner {
            LocalRuntimeBackendHandle::Skippy { model, .. } => {
                Some(model.status().max_session_tokens)
            }
        }
    }

    pub(super) fn openai_guardrails(&self) -> Option<skippy::SkippyOpenAiGuardrailsStatus> {
        match &self.inner {
            LocalRuntimeBackendHandle::Skippy { model, .. } => model.openai_guardrails(),
        }
    }

    pub(super) fn openai_server_status(&self) -> skippy_server::EmbeddedServerStatus {
        match &self.inner {
            LocalRuntimeBackendHandle::Skippy { http, .. } => http.status(),
        }
    }

    pub(super) fn set_openai_guardrail_mode(
        &self,
        mode: openai_frontend::GuardrailMode,
    ) -> Option<skippy::SkippyOpenAiGuardrailsStatus> {
        match &self.inner {
            LocalRuntimeBackendHandle::Skippy { model, .. } => {
                model.set_openai_guardrail_mode(mode)
            }
        }
    }

    pub(super) fn llama_slots_snapshot(
        &self,
        model_name: &str,
        instance_id: Option<&str>,
    ) -> Option<RuntimeLlamaSlotsSnapshot> {
        match &self.inner {
            LocalRuntimeBackendHandle::Skippy { model, .. } => {
                let status = model.status();
                let ctx_size = status.ctx_size as u64;
                let now = current_time_unix_ms();
                // Lane data may be a cached snapshot, so report when it was
                // captured: a wedged runtime then shows a success time that
                // stops advancing instead of looking fresh every tick.
                let captured_unix_ms =
                    unix_nanos_to_unix_ms(status.sessions_captured_at_unix_nanos).unwrap_or(now);
                Some(RuntimeLlamaSlotsSnapshot {
                    status: RuntimeLlamaEndpointStatus::Ready,
                    model: Some(model_name.to_string()),
                    instance_id: instance_id.map(str::to_string),
                    last_attempt_unix_ms: Some(now),
                    last_success_unix_ms: Some(captured_unix_ms),
                    error: None,
                    slots: status
                        .lanes
                        .into_iter()
                        .map(|lane| RuntimeLlamaSlotSnapshot {
                            id: Some(lane.index as u64),
                            id_task: None,
                            n_ctx: Some(ctx_size),
                            speculative: None,
                            is_processing: Some(lane.active),
                            next_token: None,
                            params: None,
                            extra: serde_json::json!({
                                "model": model_name,
                                "lane_index": lane.index,
                                "active": lane.active,
                                "session_id": lane.session_id,
                                "token_count": lane.token_count,
                            }),
                        })
                        .collect(),
                })
            }
        }
    }

    pub(super) async fn shutdown(self) {
        match self.inner {
            LocalRuntimeBackendHandle::Skippy { model, http, .. } => {
                let _ = http.shutdown().await;
                model.shutdown();
            }
        }
    }
}

pub(super) fn current_time_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Convert a unix-nanosecond capture time to unix milliseconds. `None` for a
/// non-positive input, meaning "never captured" rather than the epoch.
fn unix_nanos_to_unix_ms(unix_nanos: i64) -> Option<u64> {
    u64::try_from(unix_nanos)
        .ok()
        .filter(|nanos| *nanos > 0)
        .map(|nanos| nanos / 1_000_000)
}

pub(super) struct ManagedModelController {
    pub(super) model_name: String,
    pub(super) profile: String,
    pub(super) stop_tx: tokio::sync::watch::Sender<bool>,
    pub(super) task: tokio::task::JoinHandle<()>,
    pub(super) lifecycle: std::sync::Arc<tokio::sync::Mutex<super::InstanceLifecycleRecord>>,
    pub(super) port: std::sync::Arc<std::sync::atomic::AtomicU16>,
}

pub(super) struct LocalRuntimeModelStartSpec<'a> {
    pub(super) node: &'a mesh::Node,
    pub(super) mesh_config: &'a plugin::MeshConfig,
    pub(super) config_model_id: Option<&'a str>,
    pub(super) runtime_profile: &'a str,
    pub(super) model_path: &'a Path,
    pub(super) preindexed_split_package: Option<&'a skippy::SkippyPackageIdentity>,
    pub(super) model_bytes: u64,
    pub(super) mmproj_override: Option<&'a Path>,
    pub(super) ctx_size_override: Option<u32>,
    pub(super) pinned_gpu: Option<&'a crate::runtime::StartupPinnedGpuTarget>,
    pub(super) device_override: Option<String>,
    pub(super) capacity_budget_bytes: Option<u64>,
    pub(super) cache_type_k_override: Option<&'a str>,
    pub(super) cache_type_v_override: Option<&'a str>,
    pub(super) n_batch_override: Option<u32>,
    pub(super) n_ubatch_override: Option<u32>,
    pub(super) flash_attention_override: FlashAttentionType,
    pub(super) parallel_override: Option<usize>,
    pub(super) local_source_required: bool,
    pub(super) split_topology_lock: Option<&'a Path>,
    pub(super) planning_profile: RuntimeResourcePlanningProfile,
    pub(super) openai_guardrail_policy: OpenAiGuardrailPolicyHandle,
    pub(super) skippy_telemetry: skippy::SkippyTelemetryOptions,
    pub(super) survey_telemetry: survey::SurveyTelemetry,
}

pub(super) struct LocalOpenAiModelStartSpec<'a> {
    pub(super) mesh_config: &'a plugin::MeshConfig,
    pub(super) config_model_id: Option<&'a str>,
    pub(super) model_path: &'a Path,
    pub(super) model_bytes: u64,
    pub(super) mmproj_override: Option<&'a Path>,
    pub(super) ctx_size_override: Option<u32>,
    pub(super) pinned_gpu: Option<&'a crate::runtime::StartupPinnedGpuTarget>,
    pub(super) device_override: Option<String>,
    pub(super) capacity_budget_bytes: u64,
    pub(super) cache_type_k_override: Option<&'a str>,
    pub(super) cache_type_v_override: Option<&'a str>,
    pub(super) n_batch_override: Option<u32>,
    pub(super) n_ubatch_override: Option<u32>,
    pub(super) flash_attention_override: FlashAttentionType,
    pub(super) parallel_override: Option<usize>,
    pub(super) planning_profile: RuntimeResourcePlanningProfile,
    pub(super) openai_guardrail_policy: OpenAiGuardrailPolicyHandle,
    pub(super) skippy_telemetry: skippy::SkippyTelemetryOptions,
    pub(super) survey_telemetry: survey::SurveyTelemetry,
    pub(super) hook_policy: Option<Arc<dyn OpenAiHookPolicy>>,
    pub(super) serving_hooks_factory: Option<SharedModelServingHooksFactory>,
    pub(super) http_bind_addr: SocketAddr,
}

pub(super) fn resolved_model_name(path: &Path) -> String {
    let stem = path
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    router::strip_split_suffix_owned(&stem)
}

pub(super) fn mmproj_path_for_model(model_name: &str) -> Option<PathBuf> {
    let model_path = models::find_model_path(model_name);
    models::find_mmproj_path(model_name, &model_path)
}

fn pinned_skippy_device(
    gpu: &crate::runtime::StartupPinnedGpuTarget,
) -> skippy::SkippyDeviceDescriptor {
    skippy::SkippyDeviceDescriptor {
        backend_device: gpu.backend_device.clone(),
        stable_id: Some(gpu.stable_id.clone()),
        index: Some(gpu.index),
        vram_bytes: Some(gpu.vram_bytes),
    }
}

pub(super) fn pinned_stage_device(
    gpu: &crate::runtime::StartupPinnedGpuTarget,
) -> skippy_protocol::StageDevice {
    skippy_protocol::StageDevice {
        backend_device: gpu.backend_device.clone(),
        stable_id: Some(gpu.stable_id.clone()),
        index: Some(gpu.index),
        vram_bytes: Some(gpu.vram_bytes),
    }
}

pub(super) fn resolve_local_openai_skippy_config(
    spec: &LocalOpenAiModelStartSpec<'_>,
    model_name: &str,
    model_bytes: u64,
    context_length: u32,
    slots: usize,
    fallback_projector_path: Option<PathBuf>,
    compact_meta: Option<&models::gguf::GgufCompactMeta>,
) -> Result<skippy::ResolvedSkippyConfig> {
    let mut resolved = skippy::resolve_skippy_config_for_selector(
        skippy::SkippyConfigResolveRequest {
            mesh_config: spec.mesh_config,
            model_id: model_name,
            model_path: spec.model_path,
            model_bytes,
            allocatable_memory_bytes: Some(spec.capacity_budget_bytes),
            request_defaults: None,
            package_generation: None,
            compact_meta,
        },
        spec.config_model_id,
    )?;
    resolved.model_id = model_name.to_string();
    resolved.model_fit.ctx_size = context_length;
    resolved.throughput.parallel = slots;
    if let Some(cache_type_k) = spec.cache_type_k_override {
        resolved.model_fit.cache_type_k = cache_type_k.to_string();
    }
    if let Some(cache_type_v) = spec.cache_type_v_override {
        resolved.model_fit.cache_type_v = cache_type_v.to_string();
    }
    if let Some(n_batch) = spec.n_batch_override {
        resolved.model_fit.batch = n_batch;
    }
    if let Some(n_ubatch) = spec.n_ubatch_override {
        resolved.model_fit.ubatch = n_ubatch;
    }
    if spec.flash_attention_override != FlashAttentionType::Auto {
        resolved.model_fit.flash_attention = spec.flash_attention_override;
    }
    if let Some(mmproj_override) = spec.mmproj_override {
        resolved.hardware.projector_path = Some(mmproj_override.to_path_buf());
    } else if resolved.hardware.projector_path.is_none() {
        resolved.hardware.projector_path = fallback_projector_path;
    }
    if let Some(gpu) = spec.pinned_gpu {
        resolved.hardware.device = Some(gpu.backend_device.clone());
    }
    if let Some(device) = &spec.device_override {
        resolved.hardware.device = Some(device.clone());
    }
    Ok(resolved)
}

pub(super) async fn alloc_local_port() -> Result<u16> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let port = listener.local_addr()?.port();
    drop(listener);
    Ok(port)
}

pub(super) fn add_runtime_local_target(
    target_tx: &std::sync::Arc<tokio::sync::watch::Sender<election::ModelTargets>>,
    model_name: &str,
    port: u16,
) {
    let mut targets = target_tx.borrow().clone();
    let entry = targets.targets.entry(model_name.to_string()).or_default();
    let target_was_present = entry
        .iter()
        .any(|target| matches!(target, election::InferenceTarget::Local(local_port) if *local_port == port));
    entry.retain(
        |target| !matches!(target, election::InferenceTarget::Local(local_port) if *local_port == port),
    );
    entry.insert(0, election::InferenceTarget::Local(port));
    target_tx.send_replace(targets);
    if !target_was_present {
        record_local_serving_operational_event(LocalServingOperationalEvent::TargetAdded);
    }
}

pub(super) fn remove_runtime_local_target(
    target_tx: &std::sync::Arc<tokio::sync::watch::Sender<election::ModelTargets>>,
    model_name: &str,
    port: u16,
) {
    let mut targets = target_tx.borrow().clone();
    let mut should_remove_model = false;
    let mut target_was_removed = false;
    if let Some(entry) = targets.targets.get_mut(model_name) {
        let entry_len = entry.len();
        entry.retain(|target| {
            !matches!(target, election::InferenceTarget::Local(local_port) if *local_port == port)
        });
        target_was_removed = entry.len() != entry_len;
        should_remove_model = entry.is_empty();
    }
    if should_remove_model {
        targets.targets.remove(model_name);
    }
    target_tx.send_replace(targets);
    if target_was_removed {
        record_local_serving_operational_event(LocalServingOperationalEvent::TargetRemoved);
    }
}

pub(super) async fn advertise_model_ready(
    node: &mesh::Node,
    primary_model_name: &str,
    model_name: &str,
    profile: &str,
) {
    let mut hosted_models = node.hosted_models().await;
    let public_id = if profile.is_empty() {
        model_name.to_string()
    } else {
        format!("{}#{}", model_name, profile)
    };
    if hosted_models.iter().any(|m| m == &public_id) {
        return;
    }
    hosted_models.push(public_id);
    hosted_models.sort();
    if let Some(pos) = hosted_models.iter().position(|m| m == primary_model_name) {
        let primary = hosted_models.remove(pos);
        hosted_models.insert(0, primary);
    }
    node.set_hosted_models(hosted_models).await;
    node.regossip().await;
    record_local_serving_operational_event(LocalServingOperationalEvent::Ready);
}

pub(super) async fn set_advertised_model_context(
    node: &mesh::Node,
    model_name: &str,
    context_length: Option<u32>,
) {
    node.set_model_runtime_context_length(model_name, context_length)
        .await;
    node.regossip().await;
}

/// Record the SHA-256 of the served GGUF's file bytes on this model's served
/// descriptor. `None` when the file could not be hashed (unreadable) -- left
/// absent, never fabricated. This never touches `identity_hash` (the
/// reference-string hash) -- the two are different facts and neither
/// replaces the other.
///
/// Nothing in this crate's `openai.exchange.v1` terminal event
/// (`OpenAiExchangeEnvelope`, `plugin/openai_exchange.rs`) reads this field
/// yet -- it carries only `exchange_id`/`model`/`status`/capsule-marker
/// fields today, not a served-model identity block. A `ServingProvenance`
/// terminal-event block that does thread a served digest through exists, but
/// only on fork-only demo/ledger-UI lineage (`feat/serving-provenance` and
/// descendants), never merged to `origin/main` and absent from this branch.
/// It also deliberately never crosses the mesh gossip wire (see
/// `protocol/convert.rs`'s `descriptor_identity_to_proto`, which drops it on
/// purpose: a peer could never verify a hash of bytes only this node can
/// read). So today this is Record-only: the digest lands on this node's own
/// served-model descriptor and nothing else -- neither an exchange a peer or
/// plugin observes, nor a peer's gossip view -- reads it.
pub(super) async fn set_local_model_weights_digest(
    node: &mesh::Node,
    model_name: &str,
    generation: u64,
    weights_digest: Option<String>,
) -> bool {
    node.set_served_model_weights_digest_for_generation(model_name, generation, weights_digest)
        .await
}

pub(super) async fn withdraw_advertised_model(node: &mesh::Node, model_name: &str, profile: &str) {
    let mut hosted_models = node.hosted_models().await;
    let public_id = if profile.is_empty() {
        model_name.to_string()
    } else {
        format!("{}#{}", model_name, profile)
    };
    let old_len = hosted_models.len();
    hosted_models.retain(|m| m != &public_id);
    if hosted_models.len() == old_len {
        return;
    }
    node.set_hosted_models(hosted_models).await;
    node.regossip().await;
    record_local_serving_operational_event(LocalServingOperationalEvent::Unavailable);
}

pub(super) async fn add_serving_assignment(
    node: &mesh::Node,
    primary_model_name: &str,
    model_name: &str,
) {
    let mut serving_models = node.serving_models().await;
    if serving_models.iter().any(|m| m == model_name) {
        return;
    }
    serving_models.push(model_name.to_string());
    serving_models.sort();
    if let Some(pos) = serving_models.iter().position(|m| m == primary_model_name) {
        let primary = serving_models.remove(pos);
        serving_models.insert(0, primary);
    }
    node.set_serving_models(serving_models).await;
    if let Some(descriptor) =
        mesh::infer_local_served_model_descriptor(model_name, model_name == primary_model_name)
    {
        node.update_served_model_descriptor(model_name, move |existing| {
            let mut descriptor = descriptor;
            carry_forward_weights_digest(
                &mut descriptor,
                existing.and_then(|existing| existing.identity.weights_digest),
            );
            descriptor
        })
        .await;
    }
    node.regossip().await;
}

/// `upsert_served_model_descriptor` replaces any existing descriptor for a
/// model wholesale (`Node::upsert_served_model_descriptor`). A
/// freshly-inferred descriptor has no knowledge of a `weights_digest`
/// `set_local_model_weights_digest` may have already recorded on some other
/// startup path -- carry it forward here rather than letting a descriptor
/// that never touched the digest silently clobber it back to `None`. A
/// no-op when the new descriptor already carries its own digest.
fn carry_forward_weights_digest(
    descriptor: &mut mesh::ServedModelDescriptor,
    existing_digest: Option<String>,
) {
    if descriptor.identity.weights_digest.is_none() {
        descriptor.identity.weights_digest = existing_digest;
    }
}

/// TOCTOU narrowing for `weights_digest` (CodeRabbit, `runtime/local.rs:625`):
/// `before` is the file's (size, mtime) captured before the digest hash and
/// before `start_local_openai_model` was called; `after` is the same
/// fingerprint taken once that call has already returned `Ok` -- meaning
/// whatever read the native loader performed to actually serve the file has
/// already happened. The digest is trustworthy only if the file was stable
/// across that whole window: unreadable-now, or any change in size or
/// mtime, means a replacement could have raced the load, so the digest must
/// be dropped rather than published as fact. A replacement that preserves
/// both size AND mtime exactly is a documented blind spot (the digest
/// cache's own key has the same one) and is not detected by this check.
fn weights_digest_toctou_recheck_passes(
    before: Option<(u64, u128)>,
    after: Option<(u64, u128)>,
) -> bool {
    after.is_some() && after == before
}

pub(super) async fn set_runtime_verified_served_model_capabilities(
    node: &mesh::Node,
    primary_model_name: &str,
    model_name: &str,
    capabilities: models::ModelCapabilities,
) {
    node.update_served_model_descriptor(model_name, |existing| {
        runtime_verified_served_model_descriptor(
            existing,
            primary_model_name,
            model_name,
            capabilities,
        )
    })
    .await;
}

pub(super) fn runtime_verified_served_model_descriptor(
    existing: Option<mesh::ServedModelDescriptor>,
    primary_model_name: &str,
    model_name: &str,
    capabilities: models::ModelCapabilities,
) -> mesh::ServedModelDescriptor {
    let mut descriptor = existing.unwrap_or_else(|| mesh::ServedModelDescriptor {
        identity: mesh::ServedModelIdentity {
            model_name: model_name.to_string(),
            is_primary: model_name == primary_model_name,
            source_kind: mesh::ModelSourceKind::Unknown,
            local_file_name: Some(format!("{model_name}.gguf")),
            ..Default::default()
        },
        capabilities_known: false,
        capabilities: models::ModelCapabilities::default(),
        topology: None,
        metadata: crate::models::served_model_metadata_for_model(model_name),
    });
    descriptor.identity.model_name = model_name.to_string();
    descriptor.identity.is_primary = model_name == primary_model_name;
    descriptor.capabilities_known = true;
    descriptor.capabilities = capabilities;
    descriptor
}

pub(super) async fn remove_serving_assignment(node: &mesh::Node, model_name: &str) {
    let mut serving_models = node.serving_models().await;
    let old_len = serving_models.len();
    serving_models.retain(|m| m != model_name);
    if serving_models.len() == old_len {
        return;
    }
    node.set_serving_models(serving_models).await;
    node.remove_served_model_descriptor(model_name).await;
    node.regossip().await;
}

pub(super) async fn start_runtime_local_model(
    spec: LocalRuntimeModelStartSpec<'_>,
    runtime_model_name: &str,
) -> Result<(
    String,
    LocalRuntimeModelHandle,
    tokio::sync::oneshot::Receiver<()>,
)> {
    let policy_model_id = spec.config_model_id.unwrap_or(runtime_model_name);
    // Register the effective policy before the first await. Hashing a large
    // strict-local GGUF can take minutes; an inbound legacy stage request must
    // fail closed throughout that indexing window, not only after it.
    skippy::register_local_source_policy(
        policy_model_id,
        spec.runtime_profile,
        spec.local_source_required,
    );
    if spec.local_source_required {
        // Locally-fit strict models are still eligible workers for another
        // node's split topology. Index their complete GGUF before advertising
        // the runtime so inventory can resolve the same content identity.
        if spec.preindexed_split_package.is_none() {
            super::local_package::resolve_split_runtime_package(
                spec.model_path,
                policy_model_id,
                true,
            )
            .await?;
        }
    }
    // Start hashing the GGUF's file bytes now, at load -- the one place this
    // node opens the file for serving -- but detached: a large GGUF's hash
    // can take minutes (Qwen3.8 UD-IQ2_XXS is ~612 GiB, ~5 minutes at
    // 2 GB/s), and the node can serve perfectly well while the digest is
    // still `None`. Awaiting this inline would add that whole cost to every
    // cold start; detaching it lets model startup and the hash run
    // concurrently, and the descriptor is updated once the hash completes
    // (see below, after `start_result` resolves). Cached by (path, size,
    // mtime) and persisted to a sidecar record, so a model already hashed
    // for this exact file state costs nothing on a later restart. A
    // layer-package reference or any other non-file path fails the stat and
    // yields `None` -- honest absence, never a fabricated digest.
    let model_path_for_digest = spec.model_path.to_path_buf();
    let weights_digest_before_state =
        tokio::task::spawn_blocking(move || mesh::file_fingerprint(&model_path_for_digest))
            .await
            .unwrap_or(None);
    let model_path_for_digest = spec.model_path.to_path_buf();
    let weights_digest_handle = tokio::spawn(async move {
        tokio::task::spawn_blocking(move || mesh::weights_digest_for_file(&model_path_for_digest))
            .await
            .unwrap_or_else(|join_err| {
                tracing::warn!(
                    error = %join_err,
                    "weights digest thread panicked; treating as unreadable"
                );
                None
            })
    });

    let local_capacity_bytes = spec
        .capacity_budget_bytes
        .or_else(|| spec.pinned_gpu.map(|gpu| gpu.allocatable_vram_bytes()))
        .unwrap_or_else(|| spec.node.vram_bytes());
    let http_bind_addr = ([127, 0, 0, 1], alloc_local_port().await?).into();
    let hook_policy =
        Some(skippy::MeshAutoHookPolicy::new(spec.node.clone()) as Arc<dyn OpenAiHookPolicy>);
    let start_result = start_local_openai_model(
        LocalOpenAiModelStartSpec {
            mesh_config: spec.mesh_config,
            config_model_id: spec.config_model_id,
            model_path: spec.model_path,
            model_bytes: spec.model_bytes,
            mmproj_override: spec.mmproj_override,
            ctx_size_override: spec.ctx_size_override,
            pinned_gpu: spec.pinned_gpu,
            device_override: spec.device_override,
            capacity_budget_bytes: local_capacity_bytes,
            cache_type_k_override: spec.cache_type_k_override,
            cache_type_v_override: spec.cache_type_v_override,
            n_batch_override: spec.n_batch_override,
            n_ubatch_override: spec.n_ubatch_override,
            flash_attention_override: spec.flash_attention_override,
            parallel_override: spec.parallel_override,
            planning_profile: spec.planning_profile,
            openai_guardrail_policy: spec.openai_guardrail_policy,
            skippy_telemetry: spec.skippy_telemetry,
            survey_telemetry: spec.survey_telemetry,
            hook_policy,
            serving_hooks_factory: None,
            http_bind_addr,
        },
        runtime_model_name,
    )
    .await;

    // Commit the weights digest onto the served-model descriptor only after
    // local startup has actually succeeded. `gossip.rs` includes
    // served_model_descriptors in every announcement, so upserting this
    // before start_local_openai_model resolves would let a failed start
    // (capacity check, model load, HTTP bind) advertise a digest for a model
    // that never started serving. On failure the hash task is left detached
    // rather than aborted: it still populates the persisted cache for a
    // later retry, it just never touches the descriptor.
    if start_result.is_ok() {
        let generation = spec
            .node
            .begin_served_model_generation(runtime_model_name)
            .await;
        let node = spec.node.clone();
        let runtime_model_name = runtime_model_name.to_string();
        let model_path_for_recheck = spec.model_path.to_path_buf();
        tokio::spawn(async move {
            let weights_digest = weights_digest_handle.await.unwrap_or_else(|join_err| {
                tracing::warn!(
                    error = %join_err,
                    "weights digest task panicked; treating as unreadable"
                );
                None
            });
            // TOCTOU narrowing (CodeRabbit, runtime/local.rs:625): the digest
            // above was read via an independent `File::open`, entirely
            // separate from whatever read `StageModel::open` performed to
            // actually serve this file (verified: the native loader never
            // returns loaded bytes across the FFI boundary, so there is no
            // API to derive the digest from that same load -- see
            // `mesh::weights_digest_for_file`'s module doc). This does not
            // close that gap, but it bounds it: `weights_digest_before_state`
            // was captured before this async fn even called
            // `start_local_openai_model`, and by this point that call has
            // already returned `Ok` -- so it has already performed whatever
            // read it is going to perform. If the file's (size, mtime) here
            // still match what was captured before, the file was stable for
            // the entire window that could contain the native loader's read;
            // if they don't, a replacement could have raced it, so the
            // digest is dropped rather than published as fact. A replacement
            // that preserves both size AND mtime exactly is the same
            // documented blind spot the digest cache's own key already has,
            // and remains undetected here too.
            let weights_digest = weights_digest.and_then(|digest| {
                let after_state = mesh::file_fingerprint(&model_path_for_recheck);
                if weights_digest_toctou_recheck_passes(weights_digest_before_state, after_state) {
                    Some(digest)
                } else {
                    tracing::warn!(
                        path = %model_path_for_recheck.display(),
                        "model file's (size, mtime) changed between hashing and start finishing; discarding weights_digest"
                    );
                    None
                }
            });
            if !set_local_model_weights_digest(
                &node,
                &runtime_model_name,
                generation,
                weights_digest,
            )
            .await
            {
                tracing::debug!(
                    model = %runtime_model_name,
                    generation,
                    "discarding weights_digest for a model generation that is no longer active"
                );
            }
        });
    }

    start_result
}

pub(super) async fn start_local_openai_model(
    spec: LocalOpenAiModelStartSpec<'_>,
    runtime_model_name: &str,
) -> Result<(
    String,
    LocalRuntimeModelHandle,
    tokio::sync::oneshot::Receiver<()>,
)> {
    let model_name = runtime_model_name.to_string();
    let package_ref = spec.model_path.to_string_lossy().to_string();
    let package = if skippy::is_layer_package_ref(&package_ref) {
        let package_ref_for_identity = package_ref.clone();
        Some(
            tokio::task::spawn_blocking(move || {
                skippy::identity_from_layer_package(&package_ref_for_identity)
            })
            .await
            .context("join identify skippy layer package task")??,
        )
    } else {
        None
    };
    let total_model_bytes = package
        .as_ref()
        .map(|package| package.source_model_bytes)
        .unwrap_or_else(|| election::total_model_bytes(spec.model_path));
    let my_vram = spec.capacity_budget_bytes;

    // For split/layer-package models, compute the local share of model weights
    // and the layer fraction so the context planner budgets correctly.
    // At planning time the exact layer assignment is not yet known, so we
    // estimate the local fraction from the VRAM ratio: this node's VRAM
    // divided by total mesh VRAM (local + peers).
    // This is the local (solo) load path — the entire model is loaded on
    // this node.  Fractional scaling only applies in the split path
    // (start_runtime_split_model).
    let local_model_bytes = total_model_bytes;
    let local_layer_fraction: Option<f64> = None;

    let required_bytes = runtime_model_required_bytes(local_model_bytes);
    anyhow::ensure!(
        my_vram >= required_bytes,
        "runtime load only supports models that fit locally on this node; model requires {}, local capacity is {}",
        format_gb(required_bytes),
        format_gb(my_vram)
    );

    // Read GGUF metadata first: it carries the model's native context length,
    // head counts, and KV dimensions needed for accurate KV budget planning,
    // and drives the KV-cache compatibility guard below. For layer packages it
    // comes from the shared metadata file inside the package. Runs on a blocking
    // thread because the underlying calls do filesystem I/O (stat, open, read
    // GGUF headers).
    let compact_meta = {
        let package_clone = package.clone();
        let model_path = spec.model_path.to_path_buf();
        tokio::task::spawn_blocking(move || {
            if let Some(ref package) = package_clone {
                scan_layer_package_metadata(package)
            } else {
                models::gguf::scan_gguf_compact_meta(&model_path)
            }
        })
        .await
        .ok()
        .flatten()
    };

    // Guard the size-tiered default against quantised-KV load incompatibilities
    // (Flash Attention off, or a head_dim not divisible by the block size) so
    // planning and the load agree and the context build does not fail. Explicit
    // user overrides below are never guarded — they must fail loudly.
    let kv_cache = skippy::KvCachePolicy::for_model_size(total_model_bytes)
        .guarded_for_model(compact_meta.as_ref());
    let effective_cache_type_k = spec
        .cache_type_k_override
        .unwrap_or(kv_cache.cache_type_k());
    let effective_cache_type_v = spec
        .cache_type_v_override
        .unwrap_or(kv_cache.cache_type_v());
    let kv_cache_quant = models::gguf::GgufKvCacheQuant::from_llama_args(
        effective_cache_type_k,
        effective_cache_type_v,
    )
    .unwrap_or(models::gguf::GgufKvCacheQuant::Q8_0);

    let plan = plan_runtime_resources(RuntimeResourcePlanInput {
        ctx_size_override: spec.ctx_size_override,
        parallel_override: spec.parallel_override,
        model_bytes: local_model_bytes,
        vram_bytes: my_vram,
        metadata: compact_meta.as_ref(),
        kv_cache_quant,
        local_layer_fraction,
        planning_profile: spec.planning_profile,
    });

    if let Some(package) = package {
        start_local_package_v2_model(spec, model_name, package, plan, compact_meta.as_ref()).await
    } else {
        start_local_skippy_model(spec, model_name, plan, compact_meta.as_ref()).await
    }
}

async fn start_local_skippy_model(
    spec: LocalOpenAiModelStartSpec<'_>,
    model_name: String,
    plan: RuntimeResourcePlan,
    compact_meta: Option<&models::gguf::GgufCompactMeta>,
) -> Result<(
    String,
    LocalRuntimeModelHandle,
    tokio::sync::oneshot::Receiver<()>,
)> {
    let context_length = plan.context_length;
    let fallback_projector_path = mmproj_path_for_model(&model_name).filter(|path| path.exists());
    let mut resolved = resolve_local_openai_skippy_config(
        &spec,
        &model_name,
        spec.model_bytes,
        context_length,
        plan.slots,
        fallback_projector_path,
        compact_meta,
    )?;
    resolved.materialize_projector_url().await?;
    tracing::info!(
        model = model_name,
        "KV cache: {} K + {} V, {}K context",
        resolved.model_fit.cache_type_k.to_ascii_uppercase(),
        resolved.model_fit.cache_type_v.to_ascii_uppercase(),
        context_length / 1024,
    );
    let capabilities = models::runtime_verified_model_capabilities(
        &model_name,
        spec.model_path,
        models::runtime_media_capability_evidence(
            resolved
                .hardware
                .projector_path
                .as_deref()
                .map(PathBuf::from),
        )
        .await,
    );
    let embedded_openai = resolved.to_embedded_openai_args(0, false)?;
    let mut options = resolved
        .to_model_load_options(spec.skippy_telemetry.clone())?
        .with_embedded_openai(embedded_openai)
        .with_serving_hooks_factory(spec.serving_hooks_factory.clone())
        .with_openai_guardrails(skippy::skippy_openai_guardrails_for_policy_handle(
            spec.openai_guardrail_policy.clone(),
        ));
    if spec.device_override.is_none()
        && let Some(gpu) = spec.pinned_gpu
    {
        options = options.with_selected_device(pinned_skippy_device(gpu));
    }
    let _ = emit_event(OutputEvent::ModelLoading {
        model: model_name.clone(),
        source: None,
    });
    let hook_policy = spec.hook_policy.clone();
    let reporter_model_name = model_name.clone();
    let guardrail_telemetry = spec.survey_telemetry.clone();
    let skippy_model = tokio::task::spawn_blocking(move || {
        skippy::SkippyModelHandle::load_with_hooks_and_open_events(
            options,
            hook_policy,
            Some(skippy_native_model_open_event_reporter(reporter_model_name)),
            guardrail_telemetry,
        )
    })
    .await
    .context("join load skippy direct GGUF task")??;
    let _ = emit_event(OutputEvent::ModelLoaded {
        model: model_name.clone(),
        bytes: None,
    });
    let http = skippy_model.start_http_on(spec.http_bind_addr)?;
    let (death_tx, death_rx) = tokio::sync::oneshot::channel();

    Ok((
        model_name,
        LocalRuntimeModelHandle {
            port: http.port(),
            backend: "skippy".into(),
            context_length,
            slots: plan.slots,
            capabilities,
            inner: LocalRuntimeBackendHandle::Skippy {
                model: skippy_model,
                http,
                _death_tx: death_tx,
            },
        },
        death_rx,
    ))
}

async fn start_local_package_v2_model(
    spec: LocalOpenAiModelStartSpec<'_>,
    model_name: String,
    package: skippy::SkippyPackageIdentity,
    plan: RuntimeResourcePlan,
    compact_meta: Option<&models::gguf::GgufCompactMeta>,
) -> Result<(
    String,
    LocalRuntimeModelHandle,
    tokio::sync::oneshot::Receiver<()>,
)> {
    let (admitted_model_parts, package_projector_path) = if package.source_files.is_empty() {
        let package_ref = package.package_ref.clone();
        tokio::task::spawn_blocking(move || {
            skippy::resolve_package_v2_full_model_to_local(&package_ref)
        })
        .await
        .context("join resolve complete package-v2 model task")??
    } else {
        (
            std::iter::once(package.source_model_path.clone())
                .chain(
                    package
                        .source_files
                        .iter()
                        .map(|source| source.path.clone()),
                )
                .filter(|path| {
                    path.file_name()
                        .is_none_or(|name| name != "model-package.json")
                })
                .fold(Vec::new(), |mut paths, path| {
                    if !paths.contains(&path) {
                        paths.push(path);
                    }
                    paths
                }),
            None,
        )
    };
    let context_length = plan.context_length;
    let fallback_projector_path = package_projector_path
        .or_else(|| mmproj_path_for_model(&model_name).filter(|path| path.exists()));
    let mut resolved = resolve_local_openai_skippy_config(
        &spec,
        &model_name,
        package.source_model_bytes,
        context_length,
        plan.slots,
        fallback_projector_path,
        compact_meta,
    )?;
    resolved.materialize_projector_url().await?;
    tracing::info!(
        model = model_name,
        "KV cache: {} K + {} V, {}K context",
        resolved.model_fit.cache_type_k.to_ascii_uppercase(),
        resolved.model_fit.cache_type_v.to_ascii_uppercase(),
        context_length / 1024,
    );
    let capabilities = models::runtime_verified_model_capabilities(
        &model_name,
        spec.model_path,
        models::runtime_media_capability_evidence(
            resolved
                .hardware
                .projector_path
                .as_deref()
                .map(PathBuf::from),
        )
        .await,
    );
    let run_id = format!("mesh-skippy-{}", now_unix_nanos());
    let embedded_openai = resolved.to_embedded_openai_args(0, true)?;
    let mut runtime_options = resolved.to_embedded_runtime_options(
        &spec.skippy_telemetry,
        Some(package.clone()),
        LoadMode::RuntimeSlice,
    )?;
    runtime_options.config.run_id = run_id.clone();
    runtime_options.config.topology_id = format!("topology-{run_id}");
    runtime_options.config.model_id = model_name.clone();
    runtime_options.config.package_ref = Some(package.package_ref.clone());
    runtime_options.config.manifest_sha256 = Some(package.manifest_sha256.clone());
    runtime_options.config.source_model_path =
        Some(package.source_model_path.to_string_lossy().into_owned());
    runtime_options.config.source_model_sha256 = Some(package.source_model_sha256.clone());
    runtime_options.config.source_model_bytes = Some(package.source_model_bytes);
    runtime_options.config.model_path =
        Some(package.source_model_path.to_string_lossy().into_owned());
    runtime_options.config.model_part_paths = admitted_model_parts
        .into_iter()
        .map(|path| path.to_string_lossy().into_owned())
        .collect();
    runtime_options.config.stage_id = "stage-0".to_string();
    runtime_options.config.stage_index = 0;
    if resolved.hardware.stage_layer_start.is_none() && resolved.hardware.stage_layer_end.is_none()
    {
        runtime_options.config.layer_start = 0;
        runtime_options.config.layer_end = package.layer_count;
    }
    runtime_options.config.ctx_size = context_length;
    runtime_options.config.lane_count = plan.slots as u32;
    runtime_options.config.filter_tensors_on_load = false;
    if spec.device_override.is_none()
        && let Some(gpu) = spec.pinned_gpu
    {
        runtime_options.config.selected_device = Some(pinned_stage_device(gpu));
    }
    runtime_options.config.load_mode = LoadMode::RuntimeSlice;
    runtime_options.config.bind_addr = "127.0.0.1:0".to_string();
    runtime_options.config.upstream = None;
    runtime_options.config.downstream = None;
    let hook_policy = spec.hook_policy.clone();
    let model_ref = model_name.clone();
    let reporter_model_ref = model_ref.clone();
    let skippy_telemetry = spec.skippy_telemetry.clone();
    let guardrail_telemetry = spec.survey_telemetry.clone();
    let openai_guardrails =
        skippy::skippy_openai_guardrails_for_policy_handle(spec.openai_guardrail_policy.clone());
    let _ = emit_event(OutputEvent::ModelLoading {
        model: model_ref.clone(),
        source: None,
    });
    let handle = tokio::task::spawn_blocking(move || {
        skippy::SkippyModelHandle::load_stage0_runtime_options_with_openai_args_and_open_events(
            runtime_options,
            embedded_openai,
            hook_policy,
            skippy_telemetry,
            Some(skippy_native_model_open_event_reporter(reporter_model_ref)),
            skippy::SkippyOpenAiGuardrailOptions::new(Some(openai_guardrails), guardrail_telemetry),
            spec.serving_hooks_factory,
        )
    })
    .await
    .context("join load skippy package-v2 task")??;
    let _ = emit_event(OutputEvent::ModelLoaded {
        model: model_ref,
        bytes: None,
    });
    let http = handle.start_http_on(spec.http_bind_addr)?;
    let (death_tx, death_rx) = tokio::sync::oneshot::channel();

    Ok((
        model_name,
        LocalRuntimeModelHandle {
            port: http.port(),
            backend: "skippy".into(),
            context_length,
            slots: plan.slots,
            capabilities,
            inner: LocalRuntimeBackendHandle::Skippy {
                model: handle,
                http,
                _death_tx: death_tx,
            },
        },
        death_rx,
    ))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn local_process_payload(
    model_name: &str,
    instance_id: Option<&str>,
    profile: &str,
    backend: &str,
    port: u16,
    pid: u32,
    slots: usize,
    context_length: u32,
) -> api::RuntimeProcessPayload {
    local_process_snapshot(
        model_name,
        instance_id,
        profile,
        backend,
        port,
        pid,
        slots,
        context_length,
    )
    .to_payload()
}

#[allow(clippy::too_many_arguments)]
pub(super) fn local_process_snapshot(
    model_name: &str,
    instance_id: Option<&str>,
    profile: &str,
    backend: &str,
    port: u16,
    pid: u32,
    slots: usize,
    context_length: u32,
) -> crate::runtime_data::RuntimeProcessSnapshot {
    crate::runtime_data::RuntimeProcessSnapshot {
        model: model_name.to_string(),
        instance_id: instance_id.map(str::to_string),
        profile: profile.to_string(),
        backend: backend.into(),
        pid,
        slots,
        port,
        context_length: Some(context_length),
        command: None,
        state: "ready".into(),
        start: None,
        health: Some("ready".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        LocalRuntimeModelStartSpec, RuntimeResourcePlanningProfile, openai_guardrail_policy_handle,
        unix_nanos_to_unix_ms,
    };
    use crate::inference::skippy;
    use crate::mesh;
    use crate::plugin;
    use crate::runtime::survey;
    use skippy_protocol::FlashAttentionType;

    #[test]
    fn unix_nanos_to_unix_ms_converts_a_real_capture_time() {
        assert_eq!(
            unix_nanos_to_unix_ms(1_700_000_000_123_000_000),
            Some(1_700_000_000_123)
        );
    }

    #[test]
    fn unix_nanos_to_unix_ms_treats_non_positive_as_never_captured() {
        // A zero or negative capture time means "never read", not "read at the
        // epoch". Callers must not project 1970 as a success timestamp.
        assert_eq!(unix_nanos_to_unix_ms(0), None);
        assert_eq!(unix_nanos_to_unix_ms(-1), None);
    }

    /// Branch 1: a descriptor already exists for the model; the digest is
    /// recorded on the existing entry (overwrite path).
    #[tokio::test]
    async fn set_local_model_weights_digest_overwrites_existing_descriptor() {
        let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
            .await
            .unwrap();
        let model_name = "test-model-overwrite";

        // Pre-seed a descriptor so the function takes the overwrite branch.
        node.upsert_served_model_descriptor(mesh::ServedModelDescriptor {
            identity: mesh::ServedModelIdentity {
                model_name: model_name.to_string(),
                source_kind: mesh::ModelSourceKind::LocalGguf,
                local_file_name: Some(format!("{model_name}.gguf")),
                ..Default::default()
            },
            capabilities_known: false,
            capabilities: crate::models::ModelCapabilities::default(),
            topology: None,
            metadata: None,
        })
        .await;

        let generation = node.begin_served_model_generation(model_name).await;
        assert!(
            super::set_local_model_weights_digest(
                &node,
                model_name,
                generation,
                Some("sha256:abc123deadbeef".to_string()),
            )
            .await
        );

        let descriptors = node.served_model_descriptors().await;
        let descriptor = descriptors
            .iter()
            .find(|d| d.identity.model_name == model_name)
            .expect("descriptor must exist after set_local_model_weights_digest");
        assert_eq!(
            descriptor.identity.weights_digest.as_deref(),
            Some("sha256:abc123deadbeef"),
            "weights_digest must be recorded on the existing descriptor"
        );
    }

    /// Branch 2: no descriptor exists for the model; the function synthesizes
    /// a minimal one and records the digest on it.
    #[tokio::test]
    async fn set_local_model_weights_digest_synthesizes_descriptor_when_absent() {
        let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
            .await
            .unwrap();
        let model_name = "test-model-synthesize";

        // Confirm no pre-existing descriptor for this name.
        let before = node.served_model_descriptors().await;
        assert!(
            before.iter().all(|d| d.identity.model_name != model_name),
            "test setup: no descriptor should exist yet"
        );

        let generation = node.begin_served_model_generation(model_name).await;
        assert!(
            super::set_local_model_weights_digest(
                &node,
                model_name,
                generation,
                Some("sha256:xyz789feedface".to_string()),
            )
            .await
        );

        let descriptors = node.served_model_descriptors().await;
        let descriptor = descriptors
            .iter()
            .find(|d| d.identity.model_name == model_name)
            .expect("synthesized descriptor must exist after set_local_model_weights_digest");
        assert_eq!(
            descriptor.identity.weights_digest.as_deref(),
            Some("sha256:xyz789feedface"),
            "weights_digest must be present on the synthesized descriptor"
        );
    }

    /// A descriptor synthesized by `set_local_model_weights_digest` (branch 2
    /// above) carries a digest but has no other real identity fields. If a
    /// later real-descriptor registration path (`add_serving_assignment`)
    /// unconditionally upserted its own freshly-inferred descriptor, it would
    /// silently clobber that digest back to `None` -- `carry_forward_weights_digest`
    /// exists to prevent exactly that.
    #[test]
    fn carry_forward_weights_digest_preserves_existing_when_new_descriptor_has_none() {
        let mut descriptor = mesh::ServedModelDescriptor::default();
        super::carry_forward_weights_digest(&mut descriptor, Some("sha256:carried".to_string()));
        assert_eq!(
            descriptor.identity.weights_digest.as_deref(),
            Some("sha256:carried")
        );
    }

    /// A descriptor that already has its own freshly-inferred digest must
    /// win over whatever was previously recorded -- carrying forward is only
    /// a fallback for the "new descriptor knows nothing about it" case.
    #[test]
    fn carry_forward_weights_digest_does_not_override_a_freshly_inferred_digest() {
        let mut descriptor = mesh::ServedModelDescriptor {
            identity: mesh::ServedModelIdentity {
                weights_digest: Some("sha256:fresh".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        super::carry_forward_weights_digest(&mut descriptor, Some("sha256:stale".to_string()));
        assert_eq!(
            descriptor.identity.weights_digest.as_deref(),
            Some("sha256:fresh")
        );
    }

    #[tokio::test]
    async fn stale_generation_cannot_restore_a_removed_descriptor() {
        let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
            .await
            .unwrap();
        let model_name = "test-model-removed-before-digest";
        let generation = node.begin_served_model_generation(model_name).await;
        node.remove_served_model_descriptor(model_name).await;

        assert!(
            !super::set_local_model_weights_digest(
                &node,
                model_name,
                generation,
                Some("sha256:stale".to_string()),
            )
            .await
        );
        assert!(
            node.served_model_descriptors()
                .await
                .iter()
                .all(|descriptor| descriptor.identity.model_name != model_name)
        );
    }

    #[tokio::test]
    async fn stale_generation_cannot_overwrite_a_replacement_descriptor() {
        let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
            .await
            .unwrap();
        let model_name = "test-model-replaced-before-digest";
        let old_generation = node.begin_served_model_generation(model_name).await;
        let current_generation = node.begin_served_model_generation(model_name).await;

        assert!(
            !super::set_local_model_weights_digest(
                &node,
                model_name,
                old_generation,
                Some("sha256:stale".to_string()),
            )
            .await
        );
        assert!(
            super::set_local_model_weights_digest(
                &node,
                model_name,
                current_generation,
                Some("sha256:current".to_string()),
            )
            .await
        );
        let descriptor = node
            .served_model_descriptors()
            .await
            .into_iter()
            .find(|descriptor| descriptor.identity.model_name == model_name)
            .unwrap();
        assert_eq!(
            descriptor.identity.weights_digest.as_deref(),
            Some("sha256:current")
        );
    }

    /// TOCTOU narrowing (CodeRabbit, `runtime/local.rs:625`): the file's
    /// (size, mtime) is unchanged across the window from hashing to the
    /// model finishing its load -- the digest is trustworthy.
    #[test]
    fn toctou_recheck_passes_when_file_state_is_unchanged() {
        assert!(super::weights_digest_toctou_recheck_passes(
            Some((100, 1)),
            Some((100, 1))
        ));
    }

    /// A size or mtime change between hashing and the model finishing its
    /// load means a replacement could have raced the load -- the digest
    /// must be dropped, never published as fact for bytes that might not
    /// have been the ones actually served.
    #[test]
    fn toctou_recheck_fails_when_file_size_or_mtime_changed() {
        assert!(!super::weights_digest_toctou_recheck_passes(
            Some((100, 1)),
            Some((200, 1))
        ));
        assert!(!super::weights_digest_toctou_recheck_passes(
            Some((100, 1)),
            Some((100, 2))
        ));
    }

    /// A file that became unreadable by the time the model finished loading
    /// is at least as suspicious as a changed (size, mtime) -- never treated
    /// as "unchanged" just because there is nothing to compare against.
    #[test]
    fn toctou_recheck_fails_when_file_became_unreadable() {
        assert!(!super::weights_digest_toctou_recheck_passes(
            Some((100, 1)),
            None
        ));
    }

    /// A failed `start_runtime_local_model` must not leave behind a
    /// served-model descriptor advertising a weights digest -- `gossip.rs`
    /// includes `served_model_descriptors` in every announcement, so a
    /// descriptor committed before the model actually started serving would
    /// let a failed launch advertise a digest for a model no one is serving.
    /// The capacity budget is set far below the real on-disk file size so
    /// `start_local_openai_model`'s own vram-fit check fails fast, exercising
    /// the full function without needing a real skippy/native model load.
    #[tokio::test]
    async fn failed_local_model_start_leaves_no_served_model_descriptor() {
        let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
            .await
            .unwrap();
        let model_name = "test-model-failed-start";

        let dir = std::env::temp_dir().join(format!(
            "weights-digest-start-fail-test-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("mk temp dir");
        let model_path = dir.join("oversized.gguf");
        std::fs::write(&model_path, vec![0_u8; 10_000_000]).expect("write temp model file");

        let mesh_config = plugin::MeshConfig::default();
        let spec = LocalRuntimeModelStartSpec {
            node: &node,
            mesh_config: &mesh_config,
            config_model_id: None,
            runtime_profile: "",
            model_path: &model_path,
            preindexed_split_package: None,
            model_bytes: 10_000_000,
            mmproj_override: None,
            ctx_size_override: None,
            pinned_gpu: None,
            device_override: None,
            capacity_budget_bytes: Some(1),
            cache_type_k_override: None,
            cache_type_v_override: None,
            n_batch_override: None,
            n_ubatch_override: None,
            flash_attention_override: FlashAttentionType::Auto,
            parallel_override: None,
            local_source_required: false,
            split_topology_lock: None,
            planning_profile: RuntimeResourcePlanningProfile::DedicatedLocal,
            openai_guardrail_policy: openai_guardrail_policy_handle(
                openai_frontend::GuardrailMode::Disabled,
            ),
            skippy_telemetry: skippy::SkippyTelemetryOptions::off(),
            survey_telemetry: survey::SurveyTelemetry::disabled(),
        };

        let result = super::start_runtime_local_model(spec, model_name).await;
        assert!(
            result.is_err(),
            "test setup: an undersized capacity budget must fail the start"
        );

        let descriptors = node.served_model_descriptors().await;
        assert!(
            descriptors
                .iter()
                .all(|descriptor| descriptor.identity.model_name != model_name),
            "a failed local model start must not leave a served-model descriptor behind"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
