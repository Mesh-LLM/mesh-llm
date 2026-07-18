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
use skippy_protocol::{FlashAttentionType, LoadMode};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

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
                Some(RuntimeLlamaSlotsSnapshot {
                    status: RuntimeLlamaEndpointStatus::Ready,
                    model: Some(model_name.to_string()),
                    instance_id: instance_id.map(str::to_string),
                    last_attempt_unix_ms: Some(now),
                    last_success_unix_ms: Some(now),
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

pub(super) struct ManagedModelController {
    pub(super) model_name: String,
    pub(super) stop_tx: tokio::sync::watch::Sender<bool>,
    pub(super) task: tokio::task::JoinHandle<()>,
}

pub(super) struct LocalRuntimeModelStartSpec<'a> {
    pub(super) node: &'a mesh::Node,
    pub(super) mesh_config: &'a plugin::MeshConfig,
    pub(super) config_model_id: Option<&'a str>,
    pub(super) model_path: &'a Path,
    pub(super) model_bytes: u64,
    pub(super) mmproj_override: Option<&'a Path>,
    pub(super) ctx_size_override: Option<u32>,
    pub(super) pinned_gpu: Option<&'a crate::runtime::StartupPinnedGpuTarget>,
    pub(super) capacity_budget_bytes: Option<u64>,
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

pub(super) fn resolve_runtime_skippy_config(
    spec: &LocalRuntimeModelStartSpec<'_>,
    model_name: &str,
    model_bytes: u64,
    context_length: u32,
    slots: usize,
    fallback_projector_path: Option<PathBuf>,
) -> Result<skippy::ResolvedSkippyConfig> {
    let allocatable_memory_bytes = spec
        .capacity_budget_bytes
        .or_else(|| spec.pinned_gpu.map(|gpu| gpu.allocatable_vram_bytes()));
    let mut resolved = skippy::resolve_skippy_config(skippy::SkippyConfigResolveRequest {
        mesh_config: spec.mesh_config,
        model_id: spec.config_model_id.unwrap_or(model_name),
        model_path: spec.model_path,
        model_bytes,
        allocatable_memory_bytes,
        request_defaults: None,
        package_generation: None,
    })?;
    resolved.model_id = model_name.to_string();
    apply_runtime_skippy_launch_overrides(
        &mut resolved,
        spec,
        context_length,
        slots,
        fallback_projector_path,
    );
    Ok(resolved)
}

fn apply_runtime_skippy_launch_overrides(
    resolved: &mut skippy::ResolvedSkippyConfig,
    spec: &LocalRuntimeModelStartSpec<'_>,
    context_length: u32,
    slots: usize,
    fallback_projector_path: Option<PathBuf>,
) {
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
    entry.retain(
        |target| !matches!(target, election::InferenceTarget::Local(local_port) if *local_port == port),
    );
    entry.insert(0, election::InferenceTarget::Local(port));
    target_tx.send_replace(targets);
}

pub(super) fn remove_runtime_local_target(
    target_tx: &std::sync::Arc<tokio::sync::watch::Sender<election::ModelTargets>>,
    model_name: &str,
    port: u16,
) {
    let mut targets = target_tx.borrow().clone();
    let mut should_remove_model = false;
    if let Some(entry) = targets.targets.get_mut(model_name) {
        entry.retain(|target| {
            !matches!(target, election::InferenceTarget::Local(local_port) if *local_port == port)
        });
        should_remove_model = entry.is_empty();
    }
    if should_remove_model {
        targets.targets.remove(model_name);
    }
    target_tx.send_replace(targets);
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
        node.upsert_served_model_descriptor(descriptor).await;
    }
    node.regossip().await;
}

pub(super) async fn set_runtime_verified_served_model_capabilities(
    node: &mesh::Node,
    primary_model_name: &str,
    model_name: &str,
    capabilities: models::ModelCapabilities,
) {
    let existing = node
        .served_model_descriptors()
        .await
        .into_iter()
        .find(|descriptor| descriptor.identity.model_name == model_name);
    let descriptor = runtime_verified_served_model_descriptor(
        existing,
        primary_model_name,
        model_name,
        capabilities,
    );
    node.upsert_served_model_descriptor(descriptor).await;
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
    let model_name = runtime_model_name.to_string();
    let package_ref = spec.model_path.to_string_lossy().to_string();
    let layer_package = if skippy::is_layer_package_ref(&package_ref) {
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
    let total_model_bytes = layer_package
        .as_ref()
        .map(|package| package.source_model_bytes)
        .unwrap_or_else(|| election::total_model_bytes(spec.model_path));
    let my_vram = spec
        .capacity_budget_bytes
        .or_else(|| spec.pinned_gpu.map(|gpu| gpu.allocatable_vram_bytes()))
        .unwrap_or_else(|| spec.node.vram_bytes());

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

    let kv_cache = skippy::KvCachePolicy::for_model_size(total_model_bytes);
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

    // For layer packages, try to read GGUF metadata from the shared metadata
    // file inside the package.  This carries the model's native context length,
    // head counts, and KV dimensions needed for accurate KV budget planning.
    // Runs on a blocking thread because the underlying calls do filesystem I/O
    // (stat, open, read GGUF headers).
    let compact_meta = {
        let package_clone = layer_package.clone();
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

    if let Some(package) = layer_package {
        start_runtime_layer_package_model(spec, model_name, package, plan).await
    } else {
        start_runtime_skippy_model(spec, model_name, plan).await
    }
}

async fn start_runtime_skippy_model(
    spec: LocalRuntimeModelStartSpec<'_>,
    model_name: String,
    plan: RuntimeResourcePlan,
) -> Result<(
    String,
    LocalRuntimeModelHandle,
    tokio::sync::oneshot::Receiver<()>,
)> {
    let port = alloc_local_port().await?;
    let context_length = plan.context_length;
    let fallback_projector_path = mmproj_path_for_model(&model_name).filter(|path| path.exists());
    let resolved = resolve_runtime_skippy_config(
        &spec,
        &model_name,
        spec.model_bytes,
        context_length,
        plan.slots,
        fallback_projector_path,
    )?;
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
        models::RuntimeMediaCapabilityEvidence {
            vision_projector_loaded: resolved.hardware.projector_path.is_some(),
        },
    );
    let embedded_openai = resolved.to_embedded_openai_args(0, false)?;
    let mut options = resolved
        .to_model_load_options(spec.skippy_telemetry.clone())?
        .with_embedded_openai(embedded_openai)
        .with_openai_guardrails(skippy::skippy_openai_guardrails_for_policy_handle(
            spec.openai_guardrail_policy.clone(),
        ));
    if let Some(gpu) = spec.pinned_gpu {
        options = options.with_selected_device(pinned_skippy_device(gpu));
    }
    let _ = emit_event(OutputEvent::ModelLoading {
        model: model_name.clone(),
        source: None,
    });
    let node_for_hook = spec.node.clone();
    let reporter_model_name = model_name.clone();
    let guardrail_telemetry = spec.survey_telemetry.clone();
    let skippy_model = tokio::task::spawn_blocking(move || {
        skippy::SkippyModelHandle::load_with_hooks_and_open_events(
            options,
            Some(skippy::MeshAutoHookPolicy::new(node_for_hook)),
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
    let http = skippy_model.start_http(port);
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

async fn start_runtime_layer_package_model(
    spec: LocalRuntimeModelStartSpec<'_>,
    model_name: String,
    package: skippy::SkippyPackageIdentity,
    plan: RuntimeResourcePlan,
) -> Result<(
    String,
    LocalRuntimeModelHandle,
    tokio::sync::oneshot::Receiver<()>,
)> {
    let context_length = plan.context_length;
    let fallback_projector_path = mmproj_path_for_model(&model_name).filter(|path| path.exists());
    let resolved = resolve_runtime_skippy_config(
        &spec,
        &model_name,
        package.source_model_bytes,
        context_length,
        plan.slots,
        fallback_projector_path,
    )?;
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
        models::RuntimeMediaCapabilityEvidence {
            vision_projector_loaded: resolved.hardware.projector_path.is_some(),
        },
    );
    let activation_width = skippy_stage_activation_width(package.activation_width, &model_name)?;
    let run_id = format!("mesh-skippy-{}", now_unix_nanos());
    let embedded_openai = resolved.to_embedded_openai_args(activation_width, true)?;
    let mut runtime_options = resolved.to_embedded_runtime_options(
        &spec.skippy_telemetry,
        Some(package.clone()),
        LoadMode::LayerPackage,
    )?;
    runtime_options.config.run_id = run_id.clone();
    runtime_options.config.topology_id = format!("topology-{run_id}");
    runtime_options.config.model_id = model_name.clone();
    runtime_options.config.package_ref = Some(package.package_ref.clone());
    runtime_options.config.manifest_sha256 = Some(package.manifest_sha256.clone());
    runtime_options.config.source_model_path = Some(package.package_ref.clone());
    runtime_options.config.source_model_sha256 = Some(package.source_model_sha256.clone());
    runtime_options.config.source_model_bytes = Some(package.source_model_bytes);
    runtime_options.config.model_path = Some(package.package_ref.clone());
    runtime_options.config.stage_id = "stage-0".to_string();
    runtime_options.config.stage_index = 0;
    if resolved.hardware.stage_layer_start.is_none() && resolved.hardware.stage_layer_end.is_none()
    {
        runtime_options.config.layer_start = 0;
        runtime_options.config.layer_end = package.layer_count;
    }
    runtime_options.config.ctx_size = context_length;
    runtime_options.config.lane_count = plan.slots as u32;
    runtime_options.config.filter_tensors_on_load = true;
    if let Some(gpu) = spec.pinned_gpu {
        runtime_options.config.selected_device = Some(pinned_stage_device(gpu));
    }
    runtime_options.config.load_mode = LoadMode::LayerPackage;
    runtime_options.config.bind_addr = "127.0.0.1:0".to_string();
    runtime_options.config.upstream = None;
    runtime_options.config.downstream = None;
    let node_for_hook = spec.node.clone();
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
            Some(skippy::MeshAutoHookPolicy::new(node_for_hook)),
            skippy_telemetry,
            Some(skippy_native_model_open_event_reporter(reporter_model_ref)),
            skippy::SkippyOpenAiGuardrailOptions::new(Some(openai_guardrails), guardrail_telemetry),
        )
    })
    .await
    .context("join load skippy layer package task")??;
    let _ = emit_event(OutputEvent::ModelLoaded {
        model: model_ref,
        bytes: None,
    });
    let http = handle.start_http(alloc_local_port().await?);
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

pub(super) fn skippy_stage_activation_width(activation_width: u32, model_ref: &str) -> Result<i32> {
    i32::try_from(activation_width).with_context(|| {
        format!(
            "activation width {activation_width} for {model_ref} exceeds skippy stage ABI limit"
        )
    })
}
