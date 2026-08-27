mod coordinator;
mod loading;
mod recovery;
#[cfg(test)]
mod test_support;
#[cfg(test)]
mod tests;

use super::capacity::model_fits_runtime_capacity;
use super::local::{
    LocalRuntimeBackendHandle, LocalRuntimeModelHandle, LocalRuntimeModelStartSpec,
    OpenAiGuardrailPolicyHandle, alloc_local_port, mmproj_path_for_model, pinned_stage_device,
    resolved_model_name, skippy_stage_activation_width,
};
use super::local_package::{
    SplitParticipant, SplitParticipantExclusion, SplitParticipantSnapshot,
    collect_split_participants, resolve_split_runtime_package, split_runtime_compact_meta,
    split_runtime_kv_bytes_per_token,
};
use super::split_participant_settle::{wait_for_split_membership, wait_for_split_participants};
use super::split_planning::{
    PlannedRuntimeSliceTopology, RuntimeSliceStagePlan, SplitTopologyResourceInputs,
    plan_locked_runtime_slice_topology_with_resources, plan_runtime_slice_topology_with_resources,
    plan_runtime_slice_topology_with_resources_and_stage0, split_participant_exclusion_labels,
    split_participant_labels, split_participants_for_stages, split_stage_plan_labels,
};
use super::split_topology_lock::{
    load_configured_split_assignments, load_locked_split_assignments,
};
use crate::inference::skippy;
use crate::mesh;
use crate::models;
use anyhow::{Context, Result};
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const SPLIT_PARTICIPANT_STABLE_FOR: Duration = Duration::from_secs(2);
const SPLIT_INITIAL_SHUTDOWN_GENERATION: u64 = 1;
const SPLIT_COORDINATOR_LEASE_SECS: u64 = 4 * 60 * 60;

use coordinator::{
    SplitTopologyCoordinator, SplitTopologyGeneration, spawn_split_topology_coordinator,
    stop_split_generation,
};
use recovery::split_stages_meet_minimum;

fn split_coordinator_lease_until_unix_ms() -> u64 {
    super::local::current_time_unix_ms()
        .saturating_add(SPLIT_COORDINATOR_LEASE_SECS.saturating_mul(1000))
}

pub(super) enum SplitRuntimeStart {
    Started(Box<SplitRuntimeGenerationHandle>),
    Standby { coordinator: iroh::EndpointId },
}

enum CanonicalCoordinatorGate<T> {
    Coordinator(T),
    Standby { coordinator: iroh::EndpointId },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum StartupRuntimePlan {
    Local,
    Split { reason: SplitRuntimeReason },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum SplitRuntimeReason {
    Forced,
    LocalCapacity,
}

pub(super) struct SplitRuntimeGenerationHandle {
    pub(super) loaded_name: String,
    pub(super) handle: LocalRuntimeModelHandle,
    pub(super) death_rx: tokio::sync::oneshot::Receiver<()>,
    pub(super) cleanup: Option<SplitGenerationCleanup>,
    pub(super) coordinator_rx: Option<tokio::sync::mpsc::Receiver<SplitCoordinatorEvent>>,
    pub(super) coordinator_task: Option<tokio::task::JoinHandle<()>>,
}

pub(super) enum SplitCoordinatorEvent {
    Replace(Box<SplitCoordinatorReplaceEvent>),
    LocalFallback(SplitCoordinatorLocalFallbackEvent),
    Withdraw(SplitCoordinatorWithdrawEvent),
}

pub(super) struct SplitCoordinatorReplaceEvent {
    pub(super) reason: &'static str,
    pub(super) generation: u64,
    pub(super) loaded: SplitRuntimeGenerationHandle,
    pub(super) ack: tokio::sync::oneshot::Sender<SplitCoordinatorAck>,
}

pub(super) struct SplitCoordinatorLocalFallbackEvent {
    pub(super) reason: &'static str,
    pub(super) generation: u64,
    pub(super) topology_id: String,
    pub(super) run_id: String,
    pub(super) unavailable_stage_nodes: Vec<iroh::EndpointId>,
    pub(super) ack: tokio::sync::oneshot::Sender<SplitCoordinatorAck>,
}

pub(super) struct SplitCoordinatorWithdrawEvent {
    pub(super) reason: &'static str,
    pub(super) generation: u64,
    pub(super) topology_id: String,
    pub(super) run_id: String,
    pub(super) unavailable_stage_nodes: Vec<iroh::EndpointId>,
    pub(super) ack: tokio::sync::oneshot::Sender<SplitCoordinatorAck>,
}

pub(super) enum SplitCoordinatorAck {
    Accepted,
}

#[derive(Clone, Debug)]
pub(super) struct SplitGenerationCleanup {
    generation: SplitTopologyGeneration,
}

pub(super) async fn stop_split_generation_cleanup(
    node: &mesh::Node,
    cleanup: SplitGenerationCleanup,
    shutdown_generation: u64,
) {
    stop_split_generation(node, &cleanup.generation, shutdown_generation).await;
}

pub(super) fn startup_runtime_plan(
    explicit_split: bool,
    local_vram_bytes: u64,
    model_bytes: u64,
) -> StartupRuntimePlan {
    if explicit_split {
        return StartupRuntimePlan::Split {
            reason: SplitRuntimeReason::Forced,
        };
    }
    if model_fits_runtime_capacity(model_bytes, local_vram_bytes) {
        StartupRuntimePlan::Local
    } else {
        StartupRuntimePlan::Split {
            reason: SplitRuntimeReason::LocalCapacity,
        }
    }
}

pub(super) async fn start_runtime_split_model(
    spec: LocalRuntimeModelStartSpec<'_>,
    model_ref: &str,
) -> Result<SplitRuntimeStart> {
    let coordinator_start = elect_split_start_coordinator(&spec, model_ref).await?;
    let (canonical_coordinator, settled_membership) = match coordinator_start {
        CanonicalCoordinatorGate::Coordinator(membership) => (spec.node.id(), membership),
        CanonicalCoordinatorGate::Standby { coordinator } => {
            return Ok(SplitRuntimeStart::Standby { coordinator });
        }
    };

    let run_id = format!("mesh-split-{}", now_unix_nanos());
    let topology_id = format!("topology-{run_id}");
    let split_setup = prepare_split_runtime_start(
        &spec,
        model_ref,
        &topology_id,
        &settled_membership,
        canonical_coordinator,
        Duration::from_secs(30),
    )
    .await?;
    let SplitRuntimeStartPreparation {
        package,
        participant_snapshot,
        compact_meta,
        kv_bytes_per_token,
        planned_topology,
        topology_locked,
    } = split_setup;
    let stages = planned_topology.stages;
    let planned_participants =
        split_participants_for_stages(&participant_snapshot.participants, &stages);
    anyhow::ensure!(
        split_stages_meet_minimum(&stages),
        "split runtime needs at least two stage participants"
    );
    let stage0 = stages
        .first()
        .context("split topology did not produce stage 0")?;
    anyhow::ensure!(
        stage0.node_id == canonical_coordinator,
        "split topology stage 0 {} does not match canonical coordinator {}",
        stage0.node_id.fmt_short(),
        canonical_coordinator.fmt_short()
    );
    tracing::info!(
        model_ref,
        topology_id,
        run_id,
        context_length = planned_topology.context_length,
        parallel_lanes = planned_topology.slots,
        local_node = %spec.node.id().fmt_short(),
        elected_coordinator = %stage0.node_id.fmt_short(),
        stages = ?split_stage_plan_labels(&stages),
        participants = ?split_participant_labels(&planned_participants),
        excluded = ?split_participant_exclusion_labels(&participant_snapshot.excluded),
        "split topology planned; elected coordinator from stage 0"
    );
    let ctx_size = planned_topology.context_length;
    let slots = planned_topology.slots;
    let projector_path = spec
        .mmproj_override
        .map(Path::to_path_buf)
        .or_else(|| mmproj_path_for_model(&resolved_model_name(spec.model_path)))
        .filter(|path| path.exists())
        .map(|path| path.to_string_lossy().to_string());
    let active = SplitTopologyGeneration::new(
        topology_id.clone(),
        run_id.clone(),
        SPLIT_INITIAL_SHUTDOWN_GENERATION,
        planned_participants,
        stages,
    );
    let mut loaded = load_split_runtime_generation(SplitGenerationLoadSpec {
        node: spec.node,
        mesh_config: spec.mesh_config,
        model_ref,
        model_path: spec.model_path,
        package: &package,
        generation: &active,
        projector_path: projector_path.clone(),
        ctx_size,
        cache_type_k_override: spec.cache_type_k_override,
        cache_type_v_override: spec.cache_type_v_override,
        n_batch_override: spec.n_batch_override,
        n_ubatch_override: spec.n_ubatch_override,
        flash_attention_override: spec.flash_attention_override,
        openai_guardrail_policy: spec.openai_guardrail_policy.clone(),
        pinned_gpu: spec.pinned_gpu,
        slots,
        skippy_telemetry: spec.skippy_telemetry.clone(),
        survey_telemetry: spec.survey_telemetry.clone(),
        serving_hooks_factory: None,
    })
    .await?;
    let (coordinator_tx, coordinator_rx) = tokio::sync::mpsc::channel(1);
    loaded.coordinator_rx = Some(coordinator_rx);
    loaded.coordinator_task = Some(spawn_split_topology_coordinator(SplitTopologyCoordinator {
        node: spec.node.clone(),
        mesh_config: spec.mesh_config.clone(),
        model_name: model_ref.to_string(),
        model_path: spec.model_path.to_path_buf(),
        model_ref: model_ref.to_string(),
        package: package.clone(),
        active,
        projector_path,
        ctx_size,
        topology_resources: SplitTopologyResourceInputs {
            native_context_length: compact_meta.context_length,
            kv_bytes_per_token,
            recurrent_bytes_per_sequence_by_layer: compact_meta
                .recurrent_bytes_per_configured_lane_by_layer(),
            ctx_size_override: spec.ctx_size_override,
            parallel_override: spec.parallel_override,
        },
        cache_type_k_override: spec.cache_type_k_override.map(str::to_string),
        cache_type_v_override: spec.cache_type_v_override.map(str::to_string),
        n_batch_override: spec.n_batch_override,
        n_ubatch_override: spec.n_ubatch_override,
        flash_attention_override: spec.flash_attention_override,
        openai_guardrail_policy: spec.openai_guardrail_policy.clone(),
        pinned_gpu: spec.pinned_gpu.cloned(),
        slots,
        skippy_telemetry: spec.skippy_telemetry.clone(),
        survey_telemetry: spec.survey_telemetry.clone(),
        event_tx: coordinator_tx,
        stage_loss_first_seen: None,
        topology_locked,
        health_interval: loading::configured_stage_lifecycle_intervals(spec.mesh_config, model_ref)
            .health_interval,
    }));

    Ok(SplitRuntimeStart::Started(Box::new(loaded)))
}

struct SplitRuntimeStartPreparation {
    package: skippy::SkippyPackageIdentity,
    participant_snapshot: SplitParticipantSnapshot,
    compact_meta: models::gguf::GgufCompactMeta,
    kv_bytes_per_token: u64,
    planned_topology: PlannedRuntimeSliceTopology,
    topology_locked: bool,
}

async fn prepare_split_runtime_start(
    spec: &LocalRuntimeModelStartSpec<'_>,
    model_ref: &str,
    topology_id: &str,
    settled_membership: &[SplitParticipant],
    canonical_coordinator: iroh::EndpointId,
    timeout: Duration,
) -> Result<SplitRuntimeStartPreparation> {
    let package = resolve_split_runtime_package(spec.model_path, model_ref).await?;
    let participant_snapshot = wait_for_split_participants(
        spec.node,
        model_ref,
        model_ref,
        &package,
        spec.pinned_gpu.map(|gpu| gpu.allocatable_vram_bytes()),
        settled_membership,
        timeout,
    )
    .await?;
    let compact_meta = split_runtime_compact_meta(&package).await?;
    let kv_bytes_per_token = split_runtime_kv_bytes_per_token(
        &package,
        &compact_meta,
        model_ref,
        spec.cache_type_k_override,
        spec.cache_type_v_override,
    )?;
    let resources = SplitTopologyResourceInputs {
        native_context_length: compact_meta.context_length,
        kv_bytes_per_token,
        recurrent_bytes_per_sequence_by_layer: compact_meta
            .recurrent_bytes_per_configured_lane_by_layer(),
        ctx_size_override: spec.ctx_size_override,
        parallel_override: spec.parallel_override,
    };
    let configured_locked_stages = load_configured_split_assignments(
        spec.mesh_config,
        spec.config_model_id,
        spec.node,
        &package,
        &participant_snapshot.participants,
    )
    .await?;
    let topology_locked = configured_locked_stages.is_some() || spec.split_topology_lock.is_some();
    let planned_topology = if let Some(locked_stages) = configured_locked_stages {
        anyhow::ensure!(
            locked_stages
                .first()
                .is_some_and(|stage| stage.node_id == canonical_coordinator),
            "configured topology stage 0 must be canonical coordinator {}",
            canonical_coordinator.fmt_short()
        );
        plan_locked_runtime_slice_topology_with_resources(
            topology_id,
            model_ref,
            &package,
            &participant_snapshot.participants,
            &participant_snapshot.excluded,
            resources,
            &locked_stages,
        )?
    } else if let Some(path) = spec.split_topology_lock {
        let locked_stages = load_locked_split_assignments(
            path,
            spec.node,
            model_ref,
            &package,
            &participant_snapshot.participants,
        )
        .await?;
        anyhow::ensure!(
            locked_stages
                .first()
                .is_some_and(|stage| stage.node_id == canonical_coordinator),
            "split topology lock stage 0 must be canonical coordinator {}",
            canonical_coordinator.fmt_short()
        );
        plan_locked_runtime_slice_topology_with_resources(
            topology_id,
            model_ref,
            &package,
            &participant_snapshot.participants,
            &participant_snapshot.excluded,
            resources,
            &locked_stages,
        )?
    } else {
        plan_runtime_slice_topology_with_resources_and_stage0(
            topology_id,
            model_ref,
            &package,
            &participant_snapshot.participants,
            &participant_snapshot.excluded,
            resources,
            Some(canonical_coordinator),
        )?
    };
    Ok(SplitRuntimeStartPreparation {
        package,
        participant_snapshot,
        compact_meta,
        kv_bytes_per_token,
        planned_topology,
        topology_locked,
    })
}

async fn elect_split_start_coordinator(
    spec: &LocalRuntimeModelStartSpec<'_>,
    model_ref: &str,
) -> Result<CanonicalCoordinatorGate<Vec<SplitParticipant>>> {
    let membership =
        wait_for_split_membership(spec.node, model_ref, model_ref, Duration::from_secs(30)).await?;
    let gate = canonical_coordinator_gate(spec.node.id(), membership.participants)?;
    if let CanonicalCoordinatorGate::Standby { coordinator } = gate {
        tracing::info!(
            model_ref,
            local_node = %spec.node.id().fmt_short(),
            elected_coordinator = %coordinator.fmt_short(),
            "canonical split coordinator is remote; local node entering standby before package planning"
        );
        return Ok(CanonicalCoordinatorGate::Standby { coordinator });
    }
    Ok(gate)
}

fn canonical_coordinator_gate(
    local_node: iroh::EndpointId,
    membership: Vec<SplitParticipant>,
) -> Result<CanonicalCoordinatorGate<Vec<SplitParticipant>>> {
    let coordinator = canonical_split_coordinator(&membership)
        .context("split membership did not produce a canonical coordinator")?;
    if coordinator == local_node {
        Ok(CanonicalCoordinatorGate::Coordinator(membership))
    } else {
        Ok(CanonicalCoordinatorGate::Standby { coordinator })
    }
}

fn canonical_split_coordinator(membership: &[SplitParticipant]) -> Option<iroh::EndpointId> {
    membership
        .iter()
        .max_by(|left, right| {
            left.vram_bytes
                .cmp(&right.vram_bytes)
                .then_with(|| right.node_id.to_string().cmp(&left.node_id.to_string()))
        })
        .map(|participant| participant.node_id)
}

use loading::{SplitGenerationLoadSpec, load_split_runtime_generation};
pub(super) fn now_unix_nanos() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos().min(i64::MAX as u128) as i64)
        .unwrap_or(0)
}
