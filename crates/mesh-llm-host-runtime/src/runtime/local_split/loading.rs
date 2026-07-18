use super::{
    LocalRuntimeBackendHandle, LocalRuntimeModelHandle, OpenAiGuardrailPolicyHandle,
    RuntimeSliceStagePlan, SplitGenerationCleanup, SplitRuntimeGenerationHandle,
    SplitTopologyGeneration, alloc_local_port, pinned_stage_device, skippy_stage_activation_width,
    split_participant_labels, split_stage_plan_labels, stop_split_generation,
};
use crate::inference::skippy;
use crate::mesh;
use crate::models;
use crate::plugin;
use crate::runtime::local::skippy_native_model_open_event_reporter;
use crate::runtime::local_package::{
    split_node_labels, split_participant_set_hash, split_topology_hash,
};
use crate::runtime::survey;
use anyhow::{Context, Result};
use mesh_llm_events::{OutputEvent, emit_event};
use skippy_protocol::{FlashAttentionType, LoadMode, PeerConfig};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

pub(super) struct SplitGenerationLoadSpec<'a> {
    pub(super) node: &'a mesh::Node,
    pub(super) mesh_config: &'a plugin::MeshConfig,
    pub(super) model_ref: &'a str,
    pub(super) model_path: &'a Path,
    pub(super) package: &'a skippy::SkippyPackageIdentity,
    pub(super) generation: &'a SplitTopologyGeneration,
    pub(super) projector_path: Option<String>,
    pub(super) ctx_size: u32,
    pub(super) pinned_gpu: Option<&'a crate::runtime::StartupPinnedGpuTarget>,
    pub(super) slots: usize,
    pub(super) cache_type_k_override: Option<&'a str>,
    pub(super) cache_type_v_override: Option<&'a str>,
    pub(super) n_batch_override: Option<u32>,
    pub(super) n_ubatch_override: Option<u32>,
    pub(super) flash_attention_override: FlashAttentionType,
    pub(super) openai_guardrail_policy: OpenAiGuardrailPolicyHandle,
    pub(super) skippy_telemetry: skippy::SkippyTelemetryOptions,
    pub(super) survey_telemetry: survey::SurveyTelemetry,
}

pub(super) struct SplitGenerationLoadSettings<'a> {
    pub(super) stage0: &'a RuntimeSliceStagePlan,
    pub(super) runtime_options: skippy_server::EmbeddedRuntimeOptions,
    pub(super) embedded_openai: skippy::ResolvedEmbeddedOpenAiArgs,
    pub(super) load_mode: LoadMode,
    pub(super) activation_width: i32,
    pub(super) activation_wire_dtype: skippy::StageWireDType,
}

pub(super) async fn load_split_runtime_generation(
    spec: SplitGenerationLoadSpec<'_>,
) -> Result<SplitRuntimeGenerationHandle> {
    let mut cleanup_on_error = false;
    let result = Box::pin(load_split_runtime_generation_inner(
        &spec,
        &mut cleanup_on_error,
    ))
    .await;
    if let Err(error) = &result
        && cleanup_on_error
    {
        tracing::warn!(
            model_ref = spec.model_ref,
            topology_id = %spec.generation.topology_id,
            run_id = %spec.generation.run_id,
            generation = spec.generation.generation,
            error = %error,
            "cleaning up split runtime generation after failed load"
        );
        stop_split_generation(spec.node, spec.generation, spec.generation.generation).await;
    }
    result
}

pub(super) async fn load_split_runtime_generation_inner(
    spec: &SplitGenerationLoadSpec<'_>,
    cleanup_on_error: &mut bool,
) -> Result<SplitRuntimeGenerationHandle> {
    let settings = split_generation_load_settings(spec)?;
    anyhow::ensure!(
        settings.stage0.node_id == spec.node.id(),
        "split topology stage 0 moved to {}; local coordinator is {}",
        settings.stage0.node_id.fmt_short(),
        spec.node.id().fmt_short()
    );

    claim_split_coordinator_lease(spec.node, spec.model_ref, spec.package, spec.generation).await?;

    let mut ready_by_stage: HashMap<String, skippy::StageStatusSnapshot> = HashMap::new();
    let mut downstream: Option<skippy::StagePeerDescriptor> = None;

    if settings.load_mode == LoadMode::LayerPackage {
        spec.node
            .record_stage_topology(split_stage_topology_instance(
                &spec.generation.topology_id,
                &spec.generation.run_id,
                spec.model_ref,
                spec.package,
                &spec.generation.stages,
                &ready_by_stage,
            ))
            .await;
    }

    let stage0_return_port = alloc_local_port().await?;
    let stage0_return_endpoint = format!("127.0.0.1:{stage0_return_port}");
    spec.node
        .register_stage_transport_alias(
            &spec.generation.topology_id,
            &spec.generation.run_id,
            &settings.stage0.stage_id,
            stage0_return_endpoint.clone(),
        )
        .await;
    let downstream = Box::pin(load_downstream_split_runtime_stages(
        spec,
        &settings,
        cleanup_on_error,
        &mut ready_by_stage,
        &mut downstream,
        &stage0_return_endpoint,
    ))
    .await?;
    let downstream_endpoint = if downstream.node_id == Some(spec.node.id()) {
        downstream.endpoint
    } else {
        spec.node
            .ensure_stage_transport_bridge(
                downstream
                    .node_id
                    .context("downstream split stage is missing node id")?,
                spec.generation.topology_id.clone(),
                spec.generation.run_id.clone(),
                downstream.stage_id.clone(),
            )
            .await?
    };
    let mut runtime_options = settings.runtime_options.clone();
    runtime_options.config.run_id = spec.generation.run_id.clone();
    runtime_options.config.topology_id = spec.generation.topology_id.clone();
    runtime_options.config.model_id = spec.model_ref.to_string();
    runtime_options.config.package_ref = Some(spec.package.package_ref.clone());
    runtime_options.config.manifest_sha256 = Some(spec.package.manifest_sha256.clone());
    let effective_model_path = stage_load_model_path(
        settings.load_mode.clone(),
        &spec.package.package_ref,
        spec.model_path,
    );
    runtime_options.config.source_model_path = Some(effective_model_path.clone());
    runtime_options.config.source_model_sha256 = Some(spec.package.source_model_sha256.clone());
    runtime_options.config.source_model_bytes = Some(spec.package.source_model_bytes);
    runtime_options.config.materialized_path = None;
    runtime_options.config.materialized_pinned = false;
    runtime_options.config.model_path = Some(effective_model_path);
    if runtime_options.config.projector_path.is_none() {
        runtime_options.config.projector_path = spec.projector_path.clone();
    }
    runtime_options.config.stage_id = settings.stage0.stage_id.clone();
    runtime_options.config.stage_index = settings.stage0.stage_index;
    runtime_options.config.layer_start = settings.stage0.layer_start;
    runtime_options.config.layer_end = settings.stage0.layer_end;
    runtime_options.config.ctx_size = spec.ctx_size;
    runtime_options.config.lane_count = spec.slots as u32;
    runtime_options.config.filter_tensors_on_load = true;
    if let Some(gpu) = spec.pinned_gpu {
        runtime_options.config.selected_device = Some(pinned_stage_device(gpu));
    }
    runtime_options.config.load_mode = settings.load_mode.clone();
    runtime_options.config.bind_addr = stage0_return_endpoint;
    runtime_options.config.upstream = None;
    runtime_options.config.downstream = Some(PeerConfig {
        stage_id: downstream.stage_id,
        stage_index: downstream.stage_index,
        endpoint: downstream_endpoint,
    });
    let vision_projector_loaded = runtime_options.config.projector_path.is_some();
    let node_for_hook = spec.node.clone();
    let model_ref = spec.model_ref.to_string();
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
            settings.embedded_openai.clone(),
            Some(skippy::MeshAutoHookPolicy::new(node_for_hook)),
            skippy_telemetry,
            Some(skippy_native_model_open_event_reporter(reporter_model_ref)),
            skippy::SkippyOpenAiGuardrailOptions::new(Some(openai_guardrails), guardrail_telemetry),
        )
    })
    .await
    .context("join load skippy stage0 config task")??;
    let _ = emit_event(OutputEvent::ModelLoaded {
        model: model_ref,
        bytes: None,
    });
    let http = handle.start_http(alloc_local_port().await?);
    let (death_tx, death_rx) = tokio::sync::oneshot::channel();
    let capabilities = models::runtime_verified_model_capabilities(
        spec.model_ref,
        spec.model_path,
        models::RuntimeMediaCapabilityEvidence {
            vision_projector_loaded,
        },
    );

    spec.node
        .activate_stage_topology(split_stage_topology_instance(
            &spec.generation.topology_id,
            &spec.generation.run_id,
            spec.model_ref,
            spec.package,
            &spec.generation.stages,
            &ready_by_stage,
        ))
        .await;

    Ok(SplitRuntimeGenerationHandle {
        loaded_name: spec.model_ref.to_string(),
        handle: LocalRuntimeModelHandle {
            port: http.port(),
            backend: "skippy".into(),
            context_length: spec.ctx_size,
            slots: spec.slots,
            capabilities,
            inner: LocalRuntimeBackendHandle::Skippy {
                model: handle,
                http,
                _death_tx: death_tx,
            },
        },
        death_rx,
        cleanup: Some(SplitGenerationCleanup {
            generation: spec.generation.clone(),
        }),
        coordinator_rx: None,
        coordinator_task: None,
    })
}

pub(super) async fn load_downstream_split_runtime_stages(
    spec: &SplitGenerationLoadSpec<'_>,
    settings: &SplitGenerationLoadSettings<'_>,
    cleanup_on_error: &mut bool,
    ready_by_stage: &mut HashMap<String, skippy::StageStatusSnapshot>,
    downstream: &mut Option<skippy::StagePeerDescriptor>,
    stage0_return_endpoint: &str,
) -> Result<skippy::StagePeerDescriptor> {
    for stage in spec.generation.stages.iter().skip(1).rev() {
        *cleanup_on_error = true;
        let load = split_runtime_stage_load_request(
            spec,
            settings,
            stage,
            downstream.clone(),
            stage0_return_endpoint,
        );
        prepare_split_stage(spec.node, stage.node_id, load.clone()).await?;
        wait_for_split_stage_source(
            spec.node,
            stage.node_id,
            &load,
            Duration::from_secs(30 * 60),
        )
        .await
        .with_context(|| {
            format!(
                "prepare split stage {} on {}",
                stage.stage_id,
                stage.node_id.fmt_short()
            )
        })?;
        let response = if stage.node_id == spec.node.id() {
            spec.node
                .send_local_stage_control(skippy::StageControlRequest::Load(load))
                .await
        } else {
            spec.node
                .send_stage_control(stage.node_id, skippy::StageControlRequest::Load(load))
                .await
        }
        .with_context(|| {
            format!(
                "load split stage {} on {}",
                stage.stage_id,
                stage.node_id.fmt_short()
            )
        })?;
        let skippy::StageControlResponse::Ready(ready) = response else {
            anyhow::bail!(
                "unexpected status response while loading {}",
                stage.stage_id
            );
        };
        anyhow::ensure!(
            ready.accepted,
            "stage {} rejected load: {}",
            stage.stage_id,
            ready.error.unwrap_or_else(|| "unknown error".to_string())
        );
        *downstream = Some(skippy::StagePeerDescriptor {
            stage_id: stage.stage_id.clone(),
            stage_index: stage.stage_index,
            endpoint: ready.status.bind_addr.clone(),
            node_id: Some(stage.node_id),
        });
        ready_by_stage.insert(stage.stage_id.clone(), ready.status);
    }

    downstream
        .clone()
        .context("split topology missing downstream stage")
}

pub(super) fn split_runtime_stage_load_request(
    spec: &SplitGenerationLoadSpec<'_>,
    settings: &SplitGenerationLoadSettings<'_>,
    stage: &RuntimeSliceStagePlan,
    downstream: Option<skippy::StagePeerDescriptor>,
    stage0_return_endpoint: &str,
) -> skippy::StageLoadRequest {
    let resolved_config = &settings.runtime_options.config;
    let upstream = if downstream.is_none() {
        split_runtime_stage_upstream(spec, stage0_return_endpoint)
    } else {
        None
    };
    skippy::StageLoadRequest {
        topology_id: spec.generation.topology_id.clone(),
        run_id: spec.generation.run_id.clone(),
        model_id: spec.model_ref.to_string(),
        backend: "skippy".to_string(),
        package_ref: spec.package.package_ref.clone(),
        manifest_sha256: spec.package.manifest_sha256.clone(),
        stage_id: stage.stage_id.clone(),
        stage_index: stage.stage_index,
        layer_start: stage.layer_start,
        layer_end: stage.layer_end,
        model_path: Some(stage_load_model_path(
            settings.load_mode.clone(),
            &spec.package.package_ref,
            spec.model_path,
        )),
        source_model_bytes: Some(spec.package.source_model_bytes),
        projector_path: spec.projector_path.clone(),
        selected_device: None,
        bind_addr: "127.0.0.1:0".to_string(),
        activation_width: settings.activation_width,
        wire_dtype: settings.activation_wire_dtype,
        ctx_size: spec.ctx_size,
        lane_count: spec.slots as u32,
        n_batch: resolved_config.n_batch,
        n_ubatch: resolved_config.n_ubatch,
        n_gpu_layers: resolved_config.n_gpu_layers,
        mmap: resolved_config.mmap,
        mlock: resolved_config.mlock,
        cache_type_k: resolved_config.cache_type_k.clone(),
        cache_type_v: resolved_config.cache_type_v.clone(),
        flash_attn_type: resolved_config.flash_attn_type,
        native_mtp_enabled: resolved_config.native_mtp_enabled,
        shutdown_generation: spec.generation.generation,
        coordinator_term: spec.generation.coordinator_term,
        coordinator_id: Some(spec.node.id()),
        lease_until_unix_ms: spec.generation.lease_until_unix_ms,
        load_mode: settings.load_mode.clone(),
        upstream,
        downstream,
    }
}

pub(super) fn split_runtime_stage_upstream(
    spec: &SplitGenerationLoadSpec<'_>,
    stage0_return_endpoint: &str,
) -> Option<skippy::StagePeerDescriptor> {
    let stage0 = spec.generation.stages.first()?;
    Some(skippy::StagePeerDescriptor {
        stage_id: stage0.stage_id.clone(),
        stage_index: stage0.stage_index,
        endpoint: stage0_return_endpoint.to_string(),
        node_id: Some(stage0.node_id),
    })
}

pub(super) fn split_generation_load_settings<'a>(
    spec: &'a SplitGenerationLoadSpec<'_>,
) -> Result<SplitGenerationLoadSettings<'a>> {
    let stage0 = spec
        .generation
        .stages
        .first()
        .context("split topology did not produce stage 0")?;
    let load_mode = split_generation_load_mode(spec.package);
    let activation_width =
        skippy_stage_activation_width(spec.package.activation_width, spec.model_ref)?;
    let mut resolved = skippy::resolve_skippy_config(skippy::SkippyConfigResolveRequest {
        mesh_config: spec.mesh_config,
        model_id: spec.model_ref,
        model_path: spec.model_path,
        model_bytes: spec.package.source_model_bytes,
        allocatable_memory_bytes: spec.pinned_gpu.map(|gpu| gpu.allocatable_vram_bytes()),
        request_defaults: None,
        package_generation: spec.package.generation.as_ref(),
    })?;
    resolved.model_fit.ctx_size = spec.ctx_size;
    resolved.throughput.parallel = spec.slots;
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
    if resolved.hardware.projector_path.is_none() {
        resolved.hardware.projector_path = spec.projector_path.as_ref().map(PathBuf::from);
    }
    if let Some(gpu) = spec.pinned_gpu {
        resolved.hardware.device = Some(gpu.backend_device.clone());
    }
    let embedded_openai = resolved.to_embedded_openai_args(activation_width, true)?;
    let runtime_options = resolved.to_embedded_runtime_options(
        &spec.skippy_telemetry,
        Some(spec.package.clone()),
        load_mode.clone(),
    )?;
    tracing::info!(
        model = spec.model_ref,
        "KV cache: {} K + {} V",
        runtime_options.config.cache_type_k.to_ascii_uppercase(),
        runtime_options.config.cache_type_v.to_ascii_uppercase(),
    );
    Ok(SplitGenerationLoadSettings {
        stage0,
        runtime_options,
        embedded_openai,
        load_mode,
        activation_width,
        activation_wire_dtype: resolved.skippy.activation_wire_dtype,
    })
}

pub(super) fn split_generation_load_mode(package: &skippy::SkippyPackageIdentity) -> LoadMode {
    if skippy::is_layer_package_ref(&package.package_ref) {
        LoadMode::LayerPackage
    } else {
        LoadMode::RuntimeSlice
    }
}

pub(super) async fn claim_split_coordinator_lease(
    node: &mesh::Node,
    model_ref: &str,
    package: &skippy::SkippyPackageIdentity,
    generation: &SplitTopologyGeneration,
) -> Result<()> {
    let claim = split_coordinator_claim(node.id(), model_ref, package, generation);
    let required_accepts = skippy_coordinator::quorum_requirement(generation.stages.len());
    let mut accepted = 0usize;
    let mut accepted_nodes = Vec::new();
    let mut errors = Vec::new();
    tracing::info!(
        model_ref,
        topology_id = generation.topology_id,
        run_id = generation.run_id,
        generation = generation.generation,
        coordinator_term = generation.coordinator_term,
        coordinator = %node.id().fmt_short(),
        planned_stages = generation.stages.len(),
        required_accepts,
        stages = ?split_stage_plan_labels(&generation.stages),
        participants = ?split_participant_labels(&generation.participants),
        "claiming split topology coordinator lease"
    );

    for stage in &generation.stages {
        record_split_coordinator_claim_result(
            model_ref,
            generation,
            stage,
            claim_split_coordinator_stage(node, stage, claim.clone()).await,
            &mut accepted,
            &mut accepted_nodes,
            &mut errors,
        );
    }

    anyhow::ensure!(
        accepted >= required_accepts,
        "coordinator claim for {model_ref} accepted by {accepted}/{} planned stage(s), need {required_accepts}: {}",
        generation.stages.len(),
        errors.join("; ")
    );
    tracing::info!(
        model_ref,
        topology_id = generation.topology_id,
        run_id = generation.run_id,
        generation = generation.generation,
        coordinator_term = generation.coordinator_term,
        accepted,
        required_accepts,
        accepted_nodes = ?split_node_labels(&accepted_nodes),
        "split topology coordinator lease quorum reached"
    );
    Ok(())
}

pub(super) enum SplitCoordinatorClaimResult {
    Accepted,
    Rejected(String),
    Unexpected(Box<skippy::StageControlResponse>),
    Failed(anyhow::Error),
}

pub(super) fn record_split_coordinator_claim_result(
    model_ref: &str,
    generation: &SplitTopologyGeneration,
    stage: &RuntimeSliceStagePlan,
    result: SplitCoordinatorClaimResult,
    accepted: &mut usize,
    accepted_nodes: &mut Vec<iroh::EndpointId>,
    errors: &mut Vec<String>,
) {
    match result {
        SplitCoordinatorClaimResult::Accepted => {
            record_claim_accepted(model_ref, generation, stage, accepted, accepted_nodes)
        }
        SplitCoordinatorClaimResult::Rejected(error) => {
            record_claim_rejected(model_ref, generation, stage, error, errors)
        }
        SplitCoordinatorClaimResult::Unexpected(response) => {
            record_claim_unexpected(model_ref, generation, stage, response, errors)
        }
        SplitCoordinatorClaimResult::Failed(err) => {
            record_claim_failed(model_ref, generation, stage, err, errors)
        }
    }
}

pub(super) fn record_claim_accepted(
    model_ref: &str,
    generation: &SplitTopologyGeneration,
    stage: &RuntimeSliceStagePlan,
    accepted: &mut usize,
    accepted_nodes: &mut Vec<iroh::EndpointId>,
) {
    *accepted += 1;
    accepted_nodes.push(stage.node_id);
    tracing::debug!(
        model_ref,
        topology_id = generation.topology_id,
        generation = generation.generation,
        stage_id = stage.stage_id,
        stage_node = %stage.node_id.fmt_short(),
        "split topology coordinator claim accepted by stage"
    );
}

pub(super) fn record_claim_rejected(
    model_ref: &str,
    generation: &SplitTopologyGeneration,
    stage: &RuntimeSliceStagePlan,
    error: String,
    errors: &mut Vec<String>,
) {
    tracing::warn!(
        model_ref,
        topology_id = generation.topology_id,
        generation = generation.generation,
        stage_id = stage.stage_id,
        stage_node = %stage.node_id.fmt_short(),
        error = %error,
        "split topology coordinator claim rejected by stage"
    );
    errors.push(format!(
        "{} rejected claim: {}",
        stage.node_id.fmt_short(),
        error
    ));
}

pub(super) fn record_claim_unexpected(
    model_ref: &str,
    generation: &SplitTopologyGeneration,
    stage: &RuntimeSliceStagePlan,
    response: Box<skippy::StageControlResponse>,
    errors: &mut Vec<String>,
) {
    tracing::warn!(
        model_ref,
        topology_id = generation.topology_id,
        generation = generation.generation,
        stage_id = stage.stage_id,
        stage_node = %stage.node_id.fmt_short(),
        response = ?response,
        "split topology coordinator claim returned unexpected response"
    );
    errors.push(format!(
        "{} returned unexpected claim response: {response:?}",
        stage.node_id.fmt_short()
    ));
}

pub(super) fn record_claim_failed(
    model_ref: &str,
    generation: &SplitTopologyGeneration,
    stage: &RuntimeSliceStagePlan,
    err: anyhow::Error,
    errors: &mut Vec<String>,
) {
    tracing::warn!(
        model_ref,
        topology_id = generation.topology_id,
        generation = generation.generation,
        stage_id = stage.stage_id,
        stage_node = %stage.node_id.fmt_short(),
        error = %err,
        "split topology coordinator claim failed for stage"
    );
    errors.push(format!(
        "{} claim failed: {err:#}",
        stage.node_id.fmt_short()
    ));
}

pub(super) async fn claim_split_coordinator_stage(
    node: &mesh::Node,
    stage: &RuntimeSliceStagePlan,
    claim: skippy::StageCoordinatorClaim,
) -> SplitCoordinatorClaimResult {
    let request = skippy::StageControlRequest::Claim(claim);
    let response = if stage.node_id == node.id() {
        node.send_local_stage_control(request).await
    } else {
        node.send_stage_control(stage.node_id, request).await
    };
    match response {
        Ok(skippy::StageControlResponse::ClaimAccepted(ack)) if ack.accepted => {
            SplitCoordinatorClaimResult::Accepted
        }
        Ok(skippy::StageControlResponse::ClaimAccepted(ack)) => {
            SplitCoordinatorClaimResult::Rejected(
                ack.error.unwrap_or_else(|| "unknown rejection".to_string()),
            )
        }
        Ok(other) => SplitCoordinatorClaimResult::Unexpected(Box::new(other)),
        Err(err) => SplitCoordinatorClaimResult::Failed(err),
    }
}

pub(super) fn split_coordinator_claim(
    coordinator_id: iroh::EndpointId,
    model_ref: &str,
    package: &skippy::SkippyPackageIdentity,
    generation: &SplitTopologyGeneration,
) -> skippy::StageCoordinatorClaim {
    skippy::StageCoordinatorClaim {
        model_id: model_ref.to_string(),
        package_ref: package.package_ref.clone(),
        manifest_sha256: package.manifest_sha256.clone(),
        topology_id: generation.topology_id.clone(),
        run_id: generation.run_id.clone(),
        coordinator_id: coordinator_id.to_string(),
        coordinator_term: generation.coordinator_term,
        participant_set_hash: split_participant_set_hash(&generation.participants),
        topology_hash: split_topology_hash(&generation.stages),
        lease_until_unix_ms: generation.lease_until_unix_ms,
    }
}

pub(super) fn stage_load_model_path(
    load_mode: LoadMode,
    package_ref: &str,
    model_path: &Path,
) -> String {
    match load_mode {
        LoadMode::LayerPackage => package_ref.to_string(),
        LoadMode::RuntimeSlice | LoadMode::ArtifactSlice => {
            model_path.to_string_lossy().to_string()
        }
    }
}

pub(super) async fn prepare_split_stage(
    node: &mesh::Node,
    stage_node_id: iroh::EndpointId,
    load: skippy::StageLoadRequest,
) -> Result<()> {
    let prepare = skippy::StagePrepareRequest {
        load,
        coordinator_id: Some(node.id()),
    };
    let prepare_stage_id = prepare.load.stage_id.clone();
    let response = if stage_node_id == node.id() {
        node.send_local_stage_control(skippy::StageControlRequest::Prepare(prepare))
            .await
    } else {
        node.send_stage_control(stage_node_id, skippy::StageControlRequest::Prepare(prepare))
            .await
    }
    .with_context(|| stage_control_unreachable_message(&prepare_stage_id, stage_node_id))?;
    let skippy::StageControlResponse::PrepareAccepted(accepted) = response else {
        anyhow::bail!(
            "{}",
            stage_control_unreachable_message(&prepare_stage_id, stage_node_id)
        );
    };
    anyhow::ensure!(
        accepted.accepted,
        "{}",
        stage_source_prepare_failed_message(
            &accepted.status.stage_id,
            &accepted
                .error
                .unwrap_or_else(|| "unknown error".to_string())
        )
    );
    Ok(())
}

pub(super) async fn wait_for_split_stage_source(
    node: &mesh::Node,
    stage_node_id: iroh::EndpointId,
    load: &skippy::StageLoadRequest,
    timeout: Duration,
) -> Result<()> {
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        let inventory = query_stage_inventory(node, stage_node_id, load)
            .await
            .with_context(|| stage_control_unreachable_message(&load.stage_id, stage_node_id))?;
        if split_stage_source_is_ready(&inventory, load) {
            tracing::info!(
                topology_id = %load.topology_id,
                run_id = %load.run_id,
                stage_id = %load.stage_id,
                node = %stage_node_id.fmt_short(),
                "split stage source is available; loading runtime"
            );
            return Ok(());
        }
        if let Some(failed) = inventory.preparing_ranges.iter().find(|status| {
            status.stage_id == load.stage_id
                && matches!(status.state, skippy::StagePreparationState::Failed)
        }) {
            anyhow::bail!(
                "{}",
                stage_source_prepare_failed_message(
                    &load.stage_id,
                    failed.error.as_deref().unwrap_or("unknown error")
                )
            );
        }
        if tokio::time::Instant::now() >= deadline {
            anyhow::bail!(
                "{}",
                stage_source_prepare_timeout_message(&load.stage_id, timeout)
            );
        }
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
}

pub(super) fn stage_control_unreachable_message(
    stage_id: &str,
    stage_node_id: iroh::EndpointId,
) -> String {
    format!(
        "stage_control_unreachable: inventory/control request failed for stage {} on {}",
        stage_id,
        stage_node_id.fmt_short()
    )
}

pub(super) fn stage_source_prepare_failed_message(stage_id: &str, error: &str) -> String {
    format!("stage_source_prepare_failed: stage {stage_id} source prepare failed: {error}")
}

pub(super) fn stage_source_prepare_timeout_message(stage_id: &str, timeout: Duration) -> String {
    format!(
        "stage_source_prepare_timeout: timed out waiting for stage {stage_id} source availability after {timeout:?}"
    )
}

pub(super) fn split_stage_source_is_ready(
    inventory: &skippy::StageLayerInventory,
    load: &skippy::StageLoadRequest,
) -> bool {
    let ready_running_stage = inventory
        .ready_ranges
        .iter()
        .any(|range| split_layer_range_covers(range, load));
    if ready_running_stage {
        return true;
    }
    if load.load_mode != LoadMode::LayerPackage && !skippy::is_layer_package_ref(&load.package_ref)
    {
        return inventory
            .available_ranges
            .iter()
            .any(|range| split_layer_range_covers(range, load));
    }
    inventory.preparing_ranges.iter().any(|status| {
        status.topology_id == load.topology_id
            && status.run_id == load.run_id
            && status.stage_id == load.stage_id
            && status.model_id == load.model_id
            && status.package_ref == load.package_ref
            && status.manifest_sha256 == load.manifest_sha256
            && status.layer_start <= load.layer_start
            && status.layer_end >= load.layer_end
            && matches!(
                status.state,
                skippy::StagePreparationState::Available | skippy::StagePreparationState::Ready
            )
    })
}

pub(super) fn split_layer_range_covers(
    range: &skippy::LayerRange,
    load: &skippy::StageLoadRequest,
) -> bool {
    range.layer_start <= load.layer_start && range.layer_end >= load.layer_end
}

pub(super) async fn query_stage_inventory(
    node: &mesh::Node,
    stage_node_id: iroh::EndpointId,
    load: &skippy::StageLoadRequest,
) -> Result<skippy::StageLayerInventory> {
    let request = skippy::StageInventoryRequest {
        model_id: load.model_id.clone(),
        package_ref: load.package_ref.clone(),
        manifest_sha256: load.manifest_sha256.clone(),
    };
    let response = if stage_node_id == node.id() {
        node.send_local_stage_control(skippy::StageControlRequest::Inventory(request))
            .await
    } else {
        node.send_stage_control(
            stage_node_id,
            skippy::StageControlRequest::Inventory(request),
        )
        .await
    }?;
    let skippy::StageControlResponse::Inventory(inventory) = response else {
        anyhow::bail!("unexpected response while querying stage inventory");
    };
    Ok(inventory)
}

pub(super) fn split_stage_topology_instance(
    topology_id: &str,
    run_id: &str,
    model_ref: &str,
    package: &skippy::SkippyPackageIdentity,
    stages: &[RuntimeSliceStagePlan],
    ready_by_stage: &HashMap<String, skippy::StageStatusSnapshot>,
) -> mesh::StageTopologyInstance {
    mesh::StageTopologyInstance {
        topology_id: topology_id.to_string(),
        run_id: run_id.to_string(),
        model_id: model_ref.to_string(),
        package_ref: package.package_ref.clone(),
        manifest_sha256: package.manifest_sha256.clone(),
        stages: stages
            .iter()
            .map(|stage| mesh::StageAssignment {
                stage_id: stage.stage_id.clone(),
                stage_index: stage.stage_index,
                node_id: stage.node_id,
                layer_start: stage.layer_start,
                layer_end: stage.layer_end,
                endpoint: mesh::StageEndpoint {
                    bind_addr: ready_by_stage
                        .get(&stage.stage_id)
                        .map(|status| status.bind_addr.clone())
                        .unwrap_or_default(),
                },
            })
            .collect(),
    }
}
