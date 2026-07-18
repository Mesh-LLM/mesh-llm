use skippy_protocol::FlashAttentionType;

use super::{
    RunningStage, StageCancelPrepareRequest, StageLoadRequest, StagePreparationState,
    StagePreparationStatus, StageRuntimeState, StageStatusSnapshot, StageStopRequest,
    StageWireDType,
};

pub(super) fn status_from_running(stage: &RunningStage) -> StageStatusSnapshot {
    let server = stage.server.status();
    StageStatusSnapshot {
        topology_id: stage.load.topology_id.clone(),
        run_id: stage.load.run_id.clone(),
        model_id: stage.load.model_id.clone(),
        backend: stage.load.backend.clone(),
        package_ref: Some(stage.load.package_ref.clone()),
        manifest_sha256: Some(stage.load.manifest_sha256.clone()),
        source_model_path: source_model_path(stage),
        source_model_sha256: source_model_sha256(stage),
        source_model_bytes: source_model_bytes(stage),
        materialized_path: materialized_path(stage),
        materialized_pinned: stage.materialized.is_some(),
        projector_path: stage.load.projector_path.clone(),
        stage_id: stage.load.stage_id.clone(),
        stage_index: stage.load.stage_index,
        layer_start: stage.load.layer_start,
        layer_end: stage.load.layer_end,
        state: runtime_state(server.state),
        bind_addr: server.bind_addr.to_string(),
        activation_width: stage.load.activation_width.max(0) as u32,
        wire_dtype: stage.load.wire_dtype,
        selected_device: stage.load.selected_device.clone(),
        ctx_size: stage.load.ctx_size,
        lane_count: stage.load.lane_count,
        n_batch: stage.load.n_batch,
        n_ubatch: stage.load.n_ubatch,
        flash_attn_type: stage.load.flash_attn_type,
        weight_quantization: stage.load.weight_quantization,
        error: server.last_error,
        shutdown_generation: stage.load.shutdown_generation,
        coordinator_term: stage.load.coordinator_term,
        coordinator_id: stage.load.coordinator_id,
        lease_until_unix_ms: stage.load.lease_until_unix_ms,
    }
}

fn runtime_state(state: skippy_server::EmbeddedState) -> StageRuntimeState {
    match state {
        skippy_server::EmbeddedState::Starting => StageRuntimeState::Starting,
        skippy_server::EmbeddedState::Ready => StageRuntimeState::Ready,
        skippy_server::EmbeddedState::Stopping => StageRuntimeState::Stopping,
        skippy_server::EmbeddedState::Stopped => StageRuntimeState::Stopped,
        skippy_server::EmbeddedState::Failed => StageRuntimeState::Failed,
    }
}

fn source_model_path(stage: &RunningStage) -> Option<String> {
    stage
        .materialized
        .as_ref()
        .map(|artifact| artifact.source_model_path.clone())
        .or_else(|| {
            stage
                .package
                .as_ref()
                .map(|package| package.source_model_path.clone())
        })
        .or_else(|| {
            stage
                .mlx_artifact
                .as_ref()
                .map(|_| stage.load.package_ref.clone())
        })
        .or_else(|| stage.load.model_path.clone())
}

fn source_model_sha256(stage: &RunningStage) -> Option<String> {
    stage
        .materialized
        .as_ref()
        .map(|artifact| artifact.source_model_sha256.clone())
        .or_else(|| {
            stage
                .package
                .as_ref()
                .map(|package| package.source_model_sha256.clone())
        })
}

fn source_model_bytes(stage: &RunningStage) -> Option<u64> {
    stage
        .materialized
        .as_ref()
        .and_then(|artifact| artifact.source_model_bytes)
        .or_else(|| {
            stage
                .package
                .as_ref()
                .and_then(|package| package.source_model_bytes)
        })
        .or(stage.load.source_model_bytes)
}

fn materialized_path(stage: &RunningStage) -> Option<String> {
    stage
        .materialized
        .as_ref()
        .map(|artifact| artifact.path.to_string_lossy().into_owned())
        .or_else(|| {
            stage
                .mlx_artifact
                .as_ref()
                .map(|artifact| artifact.path.to_string_lossy().into_owned())
        })
}

pub(super) fn stopped_status(stop: &StageStopRequest) -> StageStatusSnapshot {
    StageStatusSnapshot {
        topology_id: stop.topology_id.clone(),
        run_id: stop.run_id.clone(),
        model_id: String::new(),
        backend: "skippy".to_string(),
        package_ref: None,
        manifest_sha256: None,
        source_model_path: None,
        source_model_sha256: None,
        source_model_bytes: None,
        materialized_path: None,
        materialized_pinned: false,
        projector_path: None,
        stage_id: stop.stage_id.clone(),
        stage_index: 0,
        layer_start: 0,
        layer_end: 0,
        state: StageRuntimeState::Stopped,
        bind_addr: String::new(),
        activation_width: 0,
        wire_dtype: StageWireDType::F32,
        selected_device: None,
        ctx_size: 0,
        lane_count: 0,
        n_batch: None,
        n_ubatch: None,
        flash_attn_type: FlashAttentionType::Auto,
        weight_quantization: super::StageWeightQuantization::Auto,
        error: None,
        shutdown_generation: stop.shutdown_generation,
        coordinator_term: stop.coordinator_term,
        coordinator_id: None,
        lease_until_unix_ms: 0,
    }
}

pub(super) fn failed_status_from_load(
    load: &StageLoadRequest,
    error: String,
) -> StageStatusSnapshot {
    StageStatusSnapshot {
        topology_id: load.topology_id.clone(),
        run_id: load.run_id.clone(),
        model_id: load.model_id.clone(),
        backend: load.backend.clone(),
        package_ref: Some(load.package_ref.clone()),
        manifest_sha256: Some(load.manifest_sha256.clone()),
        source_model_path: load.model_path.clone(),
        source_model_sha256: None,
        source_model_bytes: load.source_model_bytes,
        materialized_path: None,
        materialized_pinned: false,
        projector_path: load.projector_path.clone(),
        stage_id: load.stage_id.clone(),
        stage_index: load.stage_index,
        layer_start: load.layer_start,
        layer_end: load.layer_end,
        state: StageRuntimeState::Failed,
        bind_addr: load.bind_addr.clone(),
        activation_width: load.activation_width.max(0) as u32,
        wire_dtype: load.wire_dtype,
        selected_device: load.selected_device.clone(),
        ctx_size: load.ctx_size,
        lane_count: load.lane_count,
        n_batch: load.n_batch,
        n_ubatch: load.n_ubatch,
        flash_attn_type: load.flash_attn_type,
        weight_quantization: load.weight_quantization,
        error: Some(error),
        shutdown_generation: load.shutdown_generation,
        coordinator_term: load.coordinator_term,
        coordinator_id: load.coordinator_id,
        lease_until_unix_ms: load.lease_until_unix_ms,
    }
}

pub(super) fn preparation_status_from_load(
    load: &StageLoadRequest,
    state: StagePreparationState,
    error: Option<String>,
) -> StagePreparationStatus {
    StagePreparationStatus {
        topology_id: load.topology_id.clone(),
        run_id: load.run_id.clone(),
        model_id: load.model_id.clone(),
        backend: load.backend.clone(),
        package_ref: load.package_ref.clone(),
        manifest_sha256: load.manifest_sha256.clone(),
        stage_id: load.stage_id.clone(),
        stage_index: load.stage_index,
        layer_start: load.layer_start,
        layer_end: load.layer_end,
        weight_quantization: load.weight_quantization,
        state,
        bytes_done: None,
        bytes_total: None,
        bind_addr: None,
        error,
        shutdown_generation: load.shutdown_generation,
        coordinator_term: load.coordinator_term,
        coordinator_id: load.coordinator_id,
        lease_until_unix_ms: load.lease_until_unix_ms,
    }
}

pub(super) fn preparation_status_from_cancel(
    cancel: StageCancelPrepareRequest,
) -> StagePreparationStatus {
    StagePreparationStatus {
        topology_id: cancel.topology_id,
        run_id: cancel.run_id,
        model_id: String::new(),
        backend: "skippy".to_string(),
        package_ref: String::new(),
        manifest_sha256: String::new(),
        stage_id: cancel.stage_id,
        stage_index: 0,
        layer_start: 0,
        layer_end: 0,
        weight_quantization: super::StageWeightQuantization::Auto,
        state: StagePreparationState::Cancelled,
        bytes_done: None,
        bytes_total: None,
        bind_addr: None,
        error: None,
        shutdown_generation: cancel.shutdown_generation,
        coordinator_term: 0,
        coordinator_id: None,
        lease_until_unix_ms: 0,
    }
}
