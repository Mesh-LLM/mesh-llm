use crate::inference::skippy;
use anyhow::{Context, Result};
use skippy_coordinator::topology::{
    LockedTopologyStage, ThroughputEstimate, TopologyNode, TopologyPlan, TopologyPlanningInput,
    TopologyStagePlan, estimate_plan_throughput, minimum_valid_context, plan_locked_topology,
    plan_topology, plan_topology_with_stage0, rebalance_topology,
};
use std::collections::HashMap;

use super::local::{SplitParticipant, SplitParticipantExclusion};
use super::split_topology_lock::LockedSplitStageAssignment;

// Fixed per-node reserve the split planner will not fill with weights or KV.
//
// This is the *context-independent* half of the overhead model. The
// *context-scaled* half — compute-graph buffers and scratch that grow with
// `n_ctx` — is charged inside the topology planner, which bills KV at 100/85
// (see `skippy_coordinator::topology`), holding back 15% of each node's
// post-weight space exactly like the single-node context planner's
// `usable_kv_cache_budget`.
//
// This fixed reserve covers what that KV-scaled term does not: the OpenAI
// frontend, per-session runtime state, and — most importantly — margin between
// the advertised budget and physical memory. On Apple Silicon the advertised
// budget (Metal's `recommendedMaxWorkingSetSize`) can sit near 90% of total
// unified memory, so packing a node to it starves the OS and swaps the whole
// machine (observed: it made split hosts unusable). A flat 1/10 (10%) mirrors
// the single-node fit cushion in `runtime::capacity` (which requires 110% of
// model bytes) and, combined with the topology KV compute reserve, keeps a
// split host healthy. Users who want to push a node harder can raise its share
// with `--max-vram`.
const RUNTIME_NODE_HEADROOM_NUMERATOR: u64 = 1;
const RUNTIME_NODE_HEADROOM_DENOMINATOR: u64 = 10;

// Context-independent floor on that reserve.
//
// The KV-scaled term grows with `n_ctx`, but the compute graph is sized by
// lanes and batch: a four-lane stage allocated five buffers of 229.61 MiB each
// — 1.12 GiB — whether it held 12 layers or 18. A stage holding many layers at
// a modest context is therefore priced almost entirely on weights and KV, and
// the proportional share alone does not cover what the graph will take.
//
// Measured on a 16 GB host: planning admitted a 35-of-36-layer stage at 8.7 GB
// against a 12 GB budget; the process reached 12.0 GB resident, the machine
// fell to 10% free, the node stopped heartbeating and the split lost the stage.
//
// 1 GiB is a floor calibrated at one working point, not a model of the buffers,
// so it is deliberately flat: extrapolating the per-lane figure to large lane
// counts would reserve several GiB on exactly the nodes whose context-scaled
// share is already generous.
const RUNTIME_NODE_HEADROOM_FLOOR_BYTES: u64 = 1024 * 1024 * 1024;
const DEFAULT_TARGET_DECODE_TPOT_MS: u32 = 33;

// KV compute reserve, mirroring `skippy_coordinator::topology`'s
// `KV_COMPUTE_RESERVE_*`. Charging KV at 100/85 holds back 15% of post-weight
// space for llama.cpp compute-graph buffers/scratch. Kept in sync with the
// planner so the `split_capacity_shortfall` diagnostic reports the same
// per-layer cost the real planner uses.
const KV_COMPUTE_RESERVE_NUMERATOR: u128 = 100;
const KV_COMPUTE_RESERVE_DENOMINATOR: u128 = 85;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct SplitTopologyPlanInput {
    pub(super) native_context_length: u32,
    pub(super) layer_count: u32,
    pub(super) model_weight_bytes: u64,
    pub(super) layer_weight_bytes: Vec<u64>,
    pub(super) kv_bytes_per_token: u64,
    pub(super) recurrent_bytes_per_sequence_by_layer: Vec<u64>,
    pub(super) reserved_sequence_ids: usize,
    pub(super) context_length_override: Option<u32>,
    pub(super) parallel_lanes_override: Option<usize>,
    pub(super) target_decode_tpot_ms: Option<u32>,
    pub(super) minimum_nodes: usize,
    pub(super) nodes: Vec<SplitTopologyPlanNode>,
    pub(super) auto_balance: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct SplitTopologyPlanNode {
    pub(super) node_id: String,
    pub(super) detected_vram_bytes: u64,
    pub(super) max_vram_bytes: Option<u64>,
    pub(super) runtime_headroom_bytes: u64,
    pub(super) stage_transfer_latency_ms: Option<u32>,
    pub(super) decode_bytes_per_second: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct SplitTopologyPlan {
    pub(super) context_length: u32,
    pub(super) parallel_lanes: usize,
    pub(super) estimated_decode_network_ms_per_token: Option<u32>,
    pub(super) decode_tpot_target_met: Option<bool>,
    pub(super) stages: Vec<TopologyStagePlan>,
    pub(super) throughput: Option<ThroughputEstimate>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct RuntimeSliceStagePlan {
    pub(super) stage_id: String,
    pub(super) stage_index: u32,
    pub(super) node_id: iroh::EndpointId,
    pub(super) layer_start: u32,
    pub(super) layer_end: u32,
    pub(super) parameter_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct SplitTopologyResourceInputs {
    pub(super) native_context_length: u32,
    pub(super) kv_bytes_per_token: u64,
    pub(super) recurrent_bytes_per_sequence_by_layer: Vec<u64>,
    pub(super) ctx_size_override: Option<u32>,
    pub(super) parallel_override: Option<usize>,
    /// Balance layer boundaries by node decode speed (`--auto-balance`).
    pub(super) auto_balance: bool,
}

/// Per-stage capacity-model inputs resolved for a finished plan: the context
/// and lane shape the topology planner planned for, plus the KV and
/// recurrent-state costs it charged per layer. [`validate_split_capacity`]
/// re-checks every stage against this model, so boundaries moved after
/// planning — the initial-cut override — are judged by the same budget the
/// runtime's stage admissions will charge, not by weight bytes alone.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct SplitCapacityModel {
    kv_bytes_per_token: u64,
    recurrent_bytes_per_sequence_by_layer: Vec<u64>,
    context_length: u32,
    parallel_lanes: usize,
    /// Whether the planner behind the validated plan already budgeted its
    /// stages against VRAM minus [`default_runtime_headroom_bytes`]. The
    /// resource-aware planners do — re-charging the same headroom here is
    /// idempotent for their output and is what judges an override-moved cut.
    /// The test-only package-identity planner budgets against raw VRAM, so
    /// its backstop charges none rather than double-counting.
    budgets_runtime_headroom: bool,
}

impl SplitCapacityModel {
    pub(super) fn new(
        resources: &SplitTopologyResourceInputs,
        context_length: u32,
        parallel_lanes: usize,
    ) -> Self {
        Self {
            kv_bytes_per_token: resources.kv_bytes_per_token,
            recurrent_bytes_per_sequence_by_layer: resources
                .recurrent_bytes_per_sequence_by_layer
                .clone(),
            context_length,
            parallel_lanes,
            budgets_runtime_headroom: true,
        }
    }

    /// Weight-only backstop for planning paths with no context model (the
    /// test-only package-identity planner). Stages are priced from the
    /// ranges they hold against the raw node budget, exactly what that
    /// planner fit them against, with no KV, recurrent, or headroom terms.
    #[cfg(test)]
    pub(super) fn weights_only() -> Self {
        Self {
            kv_bytes_per_token: 0,
            recurrent_bytes_per_sequence_by_layer: Vec::new(),
            context_length: 0,
            parallel_lanes: 1,
            budgets_runtime_headroom: false,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct PlannedRuntimeSliceTopology {
    pub(super) stages: Vec<RuntimeSliceStagePlan>,
    pub(super) context_length: u32,
    pub(super) slots: usize,
}

pub(super) fn plan_split_topology(input: SplitTopologyPlanInput) -> Result<SplitTopologyPlan> {
    let plan =
        plan_topology(&topology_planning_input(input)).context("plan skippy split topology")?;
    Ok(split_topology_plan(plan))
}

fn plan_split_topology_with_stage0(
    input: SplitTopologyPlanInput,
    required_stage0: iroh::EndpointId,
) -> Result<SplitTopologyPlan> {
    let input = topology_planning_input(input);
    let required_stage0 = required_stage0.to_string();
    let plan = plan_topology_with_stage0(&input, &required_stage0)
        .context("plan skippy split topology with canonical stage 0")?;
    Ok(split_topology_plan(plan))
}

fn split_topology_plan(plan: skippy_coordinator::topology::TopologyPlan) -> SplitTopologyPlan {
    SplitTopologyPlan {
        context_length: plan.context_length,
        parallel_lanes: plan.parallel_lanes,
        estimated_decode_network_ms_per_token: plan.estimated_decode_network_ms_per_token,
        decode_tpot_target_met: plan.decode_tpot_target_met,
        stages: plan.stages,
        throughput: plan.throughput,
    }
}

fn topology_planning_input(input: SplitTopologyPlanInput) -> TopologyPlanningInput {
    TopologyPlanningInput {
        native_context_length: input.native_context_length,
        layer_count: input.layer_count,
        model_weight_bytes: input.model_weight_bytes,
        layer_weight_bytes: input.layer_weight_bytes,
        kv_bytes_per_token: input.kv_bytes_per_token,
        recurrent_bytes_per_sequence_by_layer: input.recurrent_bytes_per_sequence_by_layer,
        reserved_sequence_ids: input.reserved_sequence_ids,
        minimum_nodes: input.minimum_nodes,
        nodes: input
            .nodes
            .into_iter()
            .map(|node| TopologyNode {
                node_id: node.node_id,
                detected_vram_bytes: node.detected_vram_bytes,
                max_vram_bytes: node.max_vram_bytes,
                runtime_headroom_bytes: node.runtime_headroom_bytes,
                stage_transfer_latency_ms: node.stage_transfer_latency_ms,
                decode_bytes_per_second: node.decode_bytes_per_second,
            })
            .collect(),
        context_length_override: input.context_length_override,
        parallel_lanes_override: input.parallel_lanes_override,
        target_decode_tpot_ms: input.target_decode_tpot_ms,
        auto_balance: input.auto_balance,
    }
}

pub(super) fn default_runtime_headroom_bytes(vram_bytes: u64) -> u64 {
    let proportional = vram_bytes
        .saturating_mul(RUNTIME_NODE_HEADROOM_NUMERATOR)
        .div_ceil(RUNTIME_NODE_HEADROOM_DENOMINATOR);
    // Never reserve more than the node has: a tiny node keeps the proportional
    // share rather than being planned out of existence by the floor.
    proportional
        .max(RUNTIME_NODE_HEADROOM_FLOOR_BYTES.min(vram_bytes / 2))
        .min(vram_bytes)
}

pub(super) fn split_participants_for_stages(
    participants: &[SplitParticipant],
    stages: &[RuntimeSliceStagePlan],
) -> Vec<SplitParticipant> {
    let participant_by_node = participants
        .iter()
        .copied()
        .map(|participant| (participant.node_id, participant))
        .collect::<HashMap<_, _>>();
    stages
        .iter()
        .filter_map(|stage| participant_by_node.get(&stage.node_id).copied())
        .collect()
}

pub(super) fn plan_runtime_slice_topology_with_resources(
    topology_id: &str,
    model_ref: &str,
    package: &skippy::SkippyPackageIdentity,
    participants: &[SplitParticipant],
    excluded: &[SplitParticipantExclusion],
    resources: SplitTopologyResourceInputs,
) -> Result<PlannedRuntimeSliceTopology> {
    plan_runtime_slice_topology_with_resources_and_stage0(
        topology_id,
        model_ref,
        package,
        participants,
        excluded,
        resources,
        None,
    )
}

pub(super) fn plan_runtime_slice_topology_with_resources_and_stage0(
    topology_id: &str,
    model_ref: &str,
    package: &skippy::SkippyPackageIdentity,
    participants: &[SplitParticipant],
    excluded: &[SplitParticipantExclusion],
    resources: SplitTopologyResourceInputs,
    required_stage0: Option<iroh::EndpointId>,
) -> Result<PlannedRuntimeSliceTopology> {
    tracing::info!(
        topology_id,
        model_ref,
        participants = ?split_participant_labels(participants),
        layer_count = package.layer_count,
        native_context_length = resources.native_context_length,
        "planning resource-aware split runtime topology"
    );

    let participant_by_id = participant_index_by_id(participants);
    let plan_auto_balance = resources.auto_balance;
    let capacity_resources = resources.clone();
    let plan_input = runtime_slice_plan_input(package, participants, resources.clone());
    let plan = plan_runtime_slice_topology_result(
        SplitPlanAttempt {
            topology_id,
            model_ref,
            package,
            participants,
            excluded,
            resources,
        },
        plan_input,
        required_stage0,
    )?;

    let plan_labels = PlannedSliceTopologyLabels {
        context_length: plan.context_length,
        slots: plan.parallel_lanes,
        estimated_decode_network_ms_per_token: plan.estimated_decode_network_ms_per_token,
        decode_tpot_target_met: plan.decode_tpot_target_met,
        auto_balance_applied: plan.throughput.is_some(),
        stage_decode_ms: plan.throughput.as_ref().map(stage_decode_ms_labels),
        stage_idle_pct: plan.throughput.as_ref().map(stage_idle_pct_labels),
    };
    let mut stages = map_runtime_slice_stages(plan.stages, &participant_by_id)?;
    stages.sort_by_key(|stage| stage.stage_index);
    apply_initial_cut_override(&mut stages, package, plan_auto_balance);
    let capacity = SplitCapacityModel::new(
        &capacity_resources,
        plan.context_length,
        plan.parallel_lanes,
    );
    validate_split_capacity(
        model_ref,
        package,
        participants,
        &stages,
        excluded,
        &capacity,
    )?;
    log_planned_slice_topology(
        topology_id,
        model_ref,
        plan_auto_balance,
        plan_labels,
        &stages,
    );
    Ok(PlannedRuntimeSliceTopology {
        stages,
        context_length: plan.context_length,
        slots: plan.parallel_lanes,
    })
}

/// Summary of a finished plan, captured before its stages are mapped so the
/// placement summary can be logged after capacity validation.
struct PlannedSliceTopologyLabels {
    context_length: u32,
    slots: usize,
    estimated_decode_network_ms_per_token: Option<u32>,
    decode_tpot_target_met: Option<bool>,
    auto_balance_applied: bool,
    stage_decode_ms: Option<Vec<String>>,
    stage_idle_pct: Option<Vec<String>>,
}

/// Log the planned placement, calling out an auto-balance request that fell
/// back to the memory-only cut because a placed peer has no measured speed.
fn log_planned_slice_topology(
    topology_id: &str,
    model_ref: &str,
    plan_auto_balance: bool,
    labels: PlannedSliceTopologyLabels,
    stages: &[RuntimeSliceStagePlan],
) {
    let PlannedSliceTopologyLabels {
        context_length,
        slots,
        estimated_decode_network_ms_per_token,
        decode_tpot_target_met,
        auto_balance_applied,
        stage_decode_ms,
        stage_idle_pct,
    } = labels;
    if plan_auto_balance && !auto_balance_applied {
        tracing::warn!(
            model_ref,
            "auto-balance requested but at least one placed peer has no measured decode speed; keeping the memory-only placement"
        );
    }
    tracing::info!(
        topology_id,
        model_ref,
        context_length,
        slots,
        estimated_decode_network_ms_per_token,
        decode_tpot_target_met,
        stages = ?split_stage_plan_labels(stages),
        auto_balance_requested = plan_auto_balance,
        auto_balance_applied,
        stage_decode_ms = ?stage_decode_ms,
        stage_idle_pct = ?stage_idle_pct,
        "planned resource-aware split runtime topology"
    );
}

pub(super) fn plan_locked_runtime_slice_topology_with_resources(
    topology_id: &str,
    model_ref: &str,
    package: &skippy::SkippyPackageIdentity,
    participants: &[SplitParticipant],
    excluded: &[SplitParticipantExclusion],
    resources: SplitTopologyResourceInputs,
    locked_stages: &[LockedSplitStageAssignment],
) -> Result<PlannedRuntimeSliceTopology> {
    tracing::info!(
        topology_id,
        model_ref,
        participants = ?split_participant_labels(participants),
        layer_count = package.layer_count,
        native_context_length = resources.native_context_length,
        "planning locked resource-aware split runtime topology"
    );

    let participant_by_id = participant_index_by_id(participants);
    let locked_stages = locked_stages
        .iter()
        .map(|stage| LockedTopologyStage {
            node_id: stage.node_id.to_string(),
            layer_start: stage.layer_start,
            layer_end: stage.layer_end,
        })
        .collect::<Vec<_>>();
    let capacity_resources = resources.clone();
    let input = runtime_slice_plan_input(package, participants, resources);
    let plan = plan_locked_topology(&topology_planning_input(input), &locked_stages)
        .context("validate locked skippy split topology")?;
    let mut stages = map_runtime_slice_stages(plan.stages, &participant_by_id)?;
    stages.sort_by_key(|stage| stage.stage_index);
    let capacity = SplitCapacityModel::new(
        &capacity_resources,
        plan.context_length,
        plan.parallel_lanes,
    );
    validate_split_capacity(
        model_ref,
        package,
        participants,
        &stages,
        excluded,
        &capacity,
    )?;
    tracing::info!(
        topology_id,
        model_ref,
        context_length = plan.context_length,
        slots = plan.parallel_lanes,
        estimated_decode_network_ms_per_token = plan.estimated_decode_network_ms_per_token,
        decode_tpot_target_met = plan.decode_tpot_target_met,
        stages = ?split_stage_plan_labels(&stages),
        "validated locked split runtime topology"
    );
    Ok(PlannedRuntimeSliceTopology {
        stages,
        context_length: plan.context_length,
        slots: plan.parallel_lanes,
    })
}

struct SplitPlanAttempt<'a> {
    topology_id: &'a str,
    model_ref: &'a str,
    package: &'a skippy::SkippyPackageIdentity,
    participants: &'a [SplitParticipant],
    excluded: &'a [SplitParticipantExclusion],
    resources: SplitTopologyResourceInputs,
}

fn plan_runtime_slice_topology_result(
    attempt: SplitPlanAttempt<'_>,
    plan_input: SplitTopologyPlanInput,
    required_stage0: Option<iroh::EndpointId>,
) -> Result<SplitTopologyPlan> {
    let result = match required_stage0 {
        Some(node_id) => plan_split_topology_with_stage0(plan_input, node_id),
        None => plan_split_topology(plan_input),
    };
    match result {
        Ok(plan) => Ok(plan),
        Err(err) => {
            let reason = split_topology_failure_reason(
                attempt.model_ref,
                attempt.package,
                attempt.participants,
                attempt.excluded,
                attempt.resources,
            );
            tracing::warn!(
                topology_id = attempt.topology_id,
                model_ref = attempt.model_ref,
                error = %err,
                reason = %reason,
                participants = ?split_participant_labels(attempt.participants),
                excluded = ?split_participant_exclusion_labels(attempt.excluded),
                "failed to plan resource-aware split runtime topology"
            );
            Err(err.context(reason))
        }
    }
}

/// A throughput re-cut of a running split, from measured per-node rates.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct MeasuredRebalance {
    pub(super) boundaries: Vec<(iroh::EndpointId, u32, u32)>,
    pub(super) bottleneck_before_nanos: u64,
    pub(super) bottleneck_after_nanos: u64,
}

/// Re-cut `stages` (same nodes, same order) so stage decode times balance
/// under `measured_rates` (weight bytes per busy second, per stage node).
/// `None` when a node has no measurement, no feasible cut exists, or the
/// current cut is already the balanced one.
pub(super) fn measured_rebalance(
    package: &skippy::SkippyPackageIdentity,
    participants: &[SplitParticipant],
    resources: SplitTopologyResourceInputs,
    stages: &[RuntimeSliceStagePlan],
    context_length: u32,
    parallel_lanes: usize,
    measured_rates: &HashMap<iroh::EndpointId, u64>,
) -> Option<MeasuredRebalance> {
    let measured = participants
        .iter()
        .map(|participant| {
            participant.with_decode_speed(measured_rates.get(&participant.node_id).copied())
        })
        .collect::<Vec<_>>();
    let input = topology_planning_input(runtime_slice_plan_input(package, &measured, resources));
    let current = TopologyPlan {
        context_length,
        parallel_lanes,
        stages: stages
            .iter()
            .map(|stage| TopologyStagePlan {
                stage_id: stage.stage_id.clone(),
                stage_index: stage.stage_index,
                node_id: stage.node_id.to_string(),
                layer_start: stage.layer_start,
                layer_end: stage.layer_end,
                parameter_bytes: stage.parameter_bytes,
            })
            .collect(),
        estimated_decode_network_ms_per_token: None,
        decode_tpot_target_met: None,
        throughput: None,
    };
    let before = estimate_plan_throughput(&input, &current)?;
    let rebalanced = rebalance_topology(&input, &current)?;
    let after = rebalanced.throughput.as_ref()?;
    let node_by_id = stages
        .iter()
        .map(|stage| (stage.node_id.to_string(), stage.node_id))
        .collect::<HashMap<_, _>>();
    let boundaries = rebalanced
        .stages
        .iter()
        .map(|stage| {
            node_by_id
                .get(&stage.node_id)
                .map(|node| (*node, stage.layer_start, stage.layer_end))
        })
        .collect::<Option<Vec<_>>>()?;
    Some(MeasuredRebalance {
        boundaries,
        bottleneck_before_nanos: before.bottleneck_decode_nanos,
        bottleneck_after_nanos: after.bottleneck_decode_nanos,
    })
}

/// `stage-N@node:ms` per stage, for plan logs.
pub(super) fn stage_decode_ms_labels(throughput: &ThroughputEstimate) -> Vec<String> {
    throughput
        .stages
        .iter()
        .map(|stage| {
            format!(
                "stage-{}:{}..{}:{:.1}ms",
                stage.stage_index,
                stage.layer_start,
                stage.layer_end,
                stage.decode_nanos as f64 / 1_000_000.0
            )
        })
        .collect()
}

/// Share of each pipeline cycle a stage waits on the bottleneck, as percent.
pub(super) fn stage_idle_pct_labels(throughput: &ThroughputEstimate) -> Vec<String> {
    throughput
        .idle_basis_points()
        .into_iter()
        .enumerate()
        .map(|(index, idle)| format!("stage-{index}:{:.0}%", f64::from(idle) / 100.0))
        .collect()
}

fn participant_index_by_id(participants: &[SplitParticipant]) -> HashMap<String, SplitParticipant> {
    participants
        .iter()
        .copied()
        .map(|participant| (participant.node_id.to_string(), participant))
        .collect()
}

fn runtime_slice_plan_input(
    package: &skippy::SkippyPackageIdentity,
    participants: &[SplitParticipant],
    resources: SplitTopologyResourceInputs,
) -> SplitTopologyPlanInput {
    SplitTopologyPlanInput {
        native_context_length: resources.native_context_length,
        layer_count: package.layer_count,
        model_weight_bytes: package.source_model_bytes,
        layer_weight_bytes: package_layer_weight_bytes(package),
        kv_bytes_per_token: resources.kv_bytes_per_token,
        recurrent_bytes_per_sequence_by_layer: resources.recurrent_bytes_per_sequence_by_layer,
        reserved_sequence_ids: 0,
        context_length_override: resources.ctx_size_override,
        parallel_lanes_override: resources.parallel_override,
        target_decode_tpot_ms: Some(DEFAULT_TARGET_DECODE_TPOT_MS),
        minimum_nodes: super::local::SPLIT_DEFAULT_MIN_PARTICIPANTS,
        nodes: participants
            .iter()
            .map(|participant| SplitTopologyPlanNode {
                node_id: participant.node_id.to_string(),
                detected_vram_bytes: participant.vram_bytes,
                max_vram_bytes: Some(participant.vram_bytes),
                runtime_headroom_bytes: default_runtime_headroom_bytes(participant.vram_bytes),
                stage_transfer_latency_ms: participant.rtt_ms,
                decode_bytes_per_second: participant.decode_bytes_per_second,
            })
            .collect(),
        auto_balance: resources.auto_balance,
    }
}

/// Starting boundaries for an auto-balance acceptance test, read from
/// `MESH_LLM_AUTO_BALANCE_INITIAL_CUT` as the layer index ending each stage but
/// the last (`12` for a two-stage 12/24 cut, `12,24` for three stages).
///
/// The controller can only be exercised from a placement worth correcting, and
/// the planner's own paths do not produce one: speed-balanced placement starts
/// at the answer, and the memory-first cut it replaced starts somewhere fatal —
/// it filled the first node to its budget, chose 35 of 36 layers, and the node
/// died before the first window closed. An explicit cut lets a test start from
/// a placement that is deliberately wrong but survivable.
///
/// Nothing here trusts the value: the plan it produces goes through
/// `validate_split_capacity` like any other, so a cut that does not fit is
/// rejected rather than loaded.
fn initial_cut_override() -> Option<Vec<u32>> {
    parse_initial_cut(&std::env::var("MESH_LLM_AUTO_BALANCE_INITIAL_CUT").ok()?)
}

/// Layer indices ending each stage but the last. Every stage must hold at least
/// one layer, so the boundaries strictly increase and none may be zero.
fn parse_initial_cut(raw: &str) -> Option<Vec<u32>> {
    let boundaries = raw
        .split(',')
        .map(|part| part.trim().parse::<u32>().ok())
        .collect::<Option<Vec<_>>>()?;
    (!boundaries.is_empty()
        && boundaries[0] > 0
        && boundaries.windows(2).all(|pair| pair[0] < pair[1]))
    .then_some(boundaries)
}

/// Repoint an already-planned topology at [`initial_cut_override`]'s
/// boundaries.
///
/// Only for `--auto-balance` plans. The override is an acceptance-test knob for
/// the controller, and the controller only exists on an auto-balancing split;
/// letting the variable move boundaries on an ordinary fixed split would change
/// placement with nothing watching it.
fn apply_initial_cut_override(
    stages: &mut [RuntimeSliceStagePlan],
    package: &skippy::SkippyPackageIdentity,
    auto_balance: bool,
) {
    if !auto_balance {
        return;
    }
    if let Some(boundaries) = initial_cut_override() {
        apply_boundaries(stages, &boundaries, package.layer_count);
        reprice_stages(stages, package);
    }
}

/// Recompute each stage's weight bytes for the range it now holds.
///
/// [`validate_split_capacity`] re-prices a stage from its range before
/// applying the planner's capacity model, so leaving the planner's original
/// totals behind after moving the boundaries would have it approve an
/// override against the weights of a cut that no longer exists — and
/// admissions are built from the new ranges, not the old ones.
fn reprice_stages(stages: &mut [RuntimeSliceStagePlan], package: &skippy::SkippyPackageIdentity) {
    let layer_weights = package_layer_weight_bytes(package);
    if layer_weights.is_empty() {
        // No per-layer detail: fall back to an even share of the model so the
        // capacity check still sees the moved boundaries rather than stale
        // totals.
        let per_layer = package
            .source_model_bytes
            .checked_div(u64::from(package.layer_count).max(1))
            .unwrap_or(0);
        for stage in stages.iter_mut() {
            stage.parameter_bytes =
                u64::from(stage.layer_end - stage.layer_start).saturating_mul(per_layer);
        }
        return;
    }
    for stage in stages.iter_mut() {
        stage.parameter_bytes = layer_weights
            [stage.layer_start as usize..(stage.layer_end as usize).min(layer_weights.len())]
            .iter()
            .fold(0u64, |acc, bytes| acc.saturating_add(*bytes));
    }
}

/// Per-layer weights covering every layer, exactly what the topology planner
/// prices ranges from: the package's per-layer detail when complete,
/// otherwise an even share of the model.
fn planner_layer_weight_bytes(package: &skippy::SkippyPackageIdentity) -> Vec<u64> {
    let layer_weights = package_layer_weight_bytes(package);
    if layer_weights.len() == package.layer_count as usize {
        return layer_weights;
    }
    let per_layer = package
        .source_model_bytes
        .div_ceil(u64::from(package.layer_count.max(1)));
    vec![per_layer; package.layer_count as usize]
}

/// Per-layer recurrent-state bytes, mirroring the topology planner: the
/// declared per-layer costs when complete for every layer, otherwise zero.
fn planner_recurrent_bytes_by_layer(recurrent: &[u64], layer_count: u32) -> Vec<u64> {
    if recurrent.len() == layer_count as usize {
        return recurrent.to_vec();
    }
    vec![0; layer_count as usize]
}

/// What a stage costs under the topology planner's capacity model: the
/// weights of the range it holds, context-scaled KV charged at the compute
/// reserve, and lane-scaled recurrent state. Mirrors
/// `layer_required_bytes` in `skippy_coordinator::topology`, including the
/// 100/85 KV compute-reserve charge from `split_candidate_bytes_per_layer`;
/// KV is a single shared allocation, so the lane count never multiplies it.
fn stage_required_bytes(
    stage: &RuntimeSliceStagePlan,
    layer_weights: &[u64],
    recurrent_by_layer: &[u64],
    kv_per_layer: u64,
    context_length: u32,
    parallel_lanes: usize,
) -> u64 {
    let start = (stage.layer_start as usize).min(layer_weights.len());
    let end = (stage.layer_end as usize).min(layer_weights.len());
    layer_weights[start..end]
        .iter()
        .zip(recurrent_by_layer[start..end].iter())
        .fold(0u128, |total, (weight, recurrent)| {
            let kv_with_compute_reserve = u128::from(kv_per_layer)
                .saturating_mul(u128::from(context_length))
                .saturating_mul(KV_COMPUTE_RESERVE_NUMERATOR)
                .div_ceil(KV_COMPUTE_RESERVE_DENOMINATOR);
            let recurrent = u128::from(*recurrent).saturating_mul(parallel_lanes as u128);
            total
                .saturating_add(u128::from(*weight))
                .saturating_add(kv_with_compute_reserve)
                .saturating_add(recurrent)
        })
        .min(u128::from(u64::MAX)) as u64
}

/// Move the cut to `boundaries`, keeping each stage's node and order. Ignored
/// when the boundaries do not describe this topology, so a stale value in the
/// environment cannot silently produce a different split than it names.
fn apply_boundaries(stages: &mut [RuntimeSliceStagePlan], boundaries: &[u32], layer_count: u32) {
    if boundaries.len() + 1 != stages.len() || boundaries[boundaries.len() - 1] >= layer_count {
        tracing::warn!(
            ?boundaries,
            stages = stages.len(),
            layer_count,
            "ignoring MESH_LLM_AUTO_BALANCE_INITIAL_CUT: it does not describe this topology"
        );
        return;
    }
    let mut start = 0u32;
    for (index, stage) in stages.iter_mut().enumerate() {
        let end = boundaries.get(index).copied().unwrap_or(layer_count);
        stage.layer_start = start;
        stage.layer_end = end;
        start = end;
    }
    tracing::warn!(
        ?boundaries,
        "starting from MESH_LLM_AUTO_BALANCE_INITIAL_CUT instead of planned placement"
    );
}

fn package_layer_weight_bytes(package: &skippy::SkippyPackageIdentity) -> Vec<u64> {
    if package.layer_weight_bytes.len() == package.layer_count as usize {
        return package.layer_weight_bytes.clone();
    }
    Vec::new()
}

fn map_runtime_slice_stages(
    stages: Vec<TopologyStagePlan>,
    participant_by_id: &HashMap<String, SplitParticipant>,
) -> Result<Vec<RuntimeSliceStagePlan>> {
    stages
        .into_iter()
        .map(|stage| {
            let participant = participant_by_id.get(&stage.node_id).ok_or_else(|| {
                anyhow::anyhow!("topology planner returned unknown node {}", stage.node_id)
            })?;
            Ok(RuntimeSliceStagePlan {
                stage_id: stage.stage_id,
                stage_index: stage.stage_index,
                node_id: participant.node_id,
                layer_start: stage.layer_start,
                layer_end: stage.layer_end,
                parameter_bytes: stage.parameter_bytes,
            })
        })
        .collect()
}

fn split_topology_failure_reason(
    model_ref: &str,
    package: &skippy::SkippyPackageIdentity,
    participants: &[SplitParticipant],
    excluded: &[SplitParticipantExclusion],
    resources: SplitTopologyResourceInputs,
) -> String {
    let minimum_context = minimum_valid_context(resources.native_context_length);
    let evaluated_context = resources.ctx_size_override.unwrap_or(minimum_context);
    let evaluated_lanes = resources.parallel_override.unwrap_or(1).max(1);
    let weight_per_layer = package
        .source_model_bytes
        .div_ceil(u64::from(package.layer_count.max(1)));
    let kv_per_layer = resources
        .kv_bytes_per_token
        .div_ceil(u64::from(package.layer_count.max(1)));
    let bytes_per_layer = split_candidate_bytes_per_layer(
        weight_per_layer,
        kv_per_layer,
        evaluated_context,
        evaluated_lanes,
    );
    let total_usable_vram = participants
        .iter()
        .map(|participant| {
            participant
                .vram_bytes
                .saturating_sub(default_runtime_headroom_bytes(participant.vram_bytes))
        })
        .sum::<u64>();
    let max_placeable_layers = participants
        .iter()
        .map(|participant| {
            max_layers_for_participant(
                participant.vram_bytes,
                default_runtime_headroom_bytes(participant.vram_bytes),
                bytes_per_layer,
            )
        })
        .sum::<u64>();
    let estimated_total_bytes = bytes_per_layer.saturating_mul(u64::from(package.layer_count));

    format!(
        "split_capacity_shortfall: unable to plan split topology for {model_ref}: native_context={}, minimum_context={}, evaluated_context={}, evaluated_lanes={}, layer_count={}, estimated_bytes_per_layer={}, estimated_total_bytes={}, total_usable_vram={}, max_placeable_layers_at_evaluated_shape={}/{}; participants [{}]; excluded [{}]",
        resources.native_context_length,
        minimum_context,
        evaluated_context,
        evaluated_lanes,
        package.layer_count,
        format_gb(bytes_per_layer),
        format_gb(estimated_total_bytes),
        format_gb(total_usable_vram),
        max_placeable_layers,
        package.layer_count,
        split_topology_fit_labels(participants, bytes_per_layer).join(", "),
        split_participant_exclusion_labels(excluded).join(", ")
    )
}

fn split_candidate_bytes_per_layer(
    weight_per_layer: u64,
    kv_per_layer: u64,
    context_length: u32,
    _parallel_lanes: usize,
) -> u64 {
    // Mirror of `skippy_coordinator::topology::candidate_bytes_per_layer` so the
    // `split_capacity_shortfall` diagnostic reports the same per-layer cost the
    // real planner uses. KV cache is a single unified allocation shared across
    // all parallel lanes with eviction — lane count does not multiply KV cost.
    // KV is charged at KV_COMPUTE_RESERVE_NUMERATOR/DENOMINATOR (100/85) to hold
    // back 15% of post-weight space for compute-graph buffers/scratch.
    let kv_bytes = u128::from(kv_per_layer).saturating_mul(u128::from(context_length));
    let kv_with_compute_reserve = kv_bytes
        .saturating_mul(KV_COMPUTE_RESERVE_NUMERATOR)
        .div_ceil(KV_COMPUTE_RESERVE_DENOMINATOR);
    let total = u128::from(weight_per_layer).saturating_add(kv_with_compute_reserve);
    total.min(u128::from(u64::MAX)) as u64
}

fn max_layers_for_participant(
    vram_bytes: u64,
    runtime_headroom_bytes: u64,
    bytes_per_layer: u64,
) -> u64 {
    if bytes_per_layer == 0 {
        return 0;
    }
    vram_bytes.saturating_sub(runtime_headroom_bytes) / bytes_per_layer
}

fn split_topology_fit_labels(
    participants: &[SplitParticipant],
    bytes_per_layer: u64,
) -> Vec<String> {
    participants
        .iter()
        .map(|participant| {
            let headroom = default_runtime_headroom_bytes(participant.vram_bytes);
            let usable = participant.vram_bytes.saturating_sub(headroom);
            let max_layers =
                max_layers_for_participant(participant.vram_bytes, headroom, bytes_per_layer);
            format!(
                "{}:budget={} headroom={} usable={} max_layers={}",
                participant.node_id.fmt_short(),
                format_gb(participant.vram_bytes),
                format_gb(headroom),
                format_gb(usable),
                max_layers
            )
        })
        .collect()
}

pub(super) fn split_participant_labels(participants: &[SplitParticipant]) -> Vec<String> {
    participants
        .iter()
        .map(|participant| {
            format!(
                "{}:{} cached={} missing={} rtt={}ms transfer={}",
                participant.node_id.fmt_short(),
                format_gb(participant.vram_bytes),
                format_gb(participant.cached_slice_bytes),
                format_gb(participant.missing_artifact_bytes),
                participant.rtt_ms.unwrap_or_default(),
                participant.artifact_transfer_supported
            )
        })
        .collect()
}

pub(super) fn split_participant_exclusion_labels(
    excluded: &[SplitParticipantExclusion],
) -> Vec<String> {
    excluded
        .iter()
        .map(|exclusion| {
            format!(
                "{}:{}",
                exclusion.node_id.fmt_short(),
                exclusion.reason.as_str()
            )
        })
        .collect()
}

/// Validate a finished split placement against aggregate mesh capacity and
/// the topology planner's per-node capacity model.
///
/// The per-stage check re-runs the planner's own budgeting — usable VRAM
/// after runtime headroom, context-scaled KV charged at the compute reserve,
/// and lane-scaled recurrent state — over the ranges the stages actually
/// hold. For planner-produced placement this is idempotent: the planner never
/// emits a stage exceeding the same budget it planned with. For boundaries
/// moved after planning by `apply_initial_cut_override` it is the only real
/// check, because the planner never saw the overridden cut.
pub(super) fn validate_split_capacity(
    model_ref: &str,
    package: &skippy::SkippyPackageIdentity,
    participants: &[SplitParticipant],
    stages: &[RuntimeSliceStagePlan],
    excluded: &[SplitParticipantExclusion],
    capacity: &SplitCapacityModel,
) -> Result<()> {
    let total_vram_bytes = participants
        .iter()
        .map(|participant| participant.vram_bytes)
        .sum::<u64>();
    // Use raw model weight for aggregate split check — the topology planner
    // already performed detailed per-node budgeting with KV and headroom.
    let required_total_bytes = package.source_model_bytes;
    anyhow::ensure!(
        total_vram_bytes >= required_total_bytes,
        "{}",
        format_aggregate_split_capacity_error(
            model_ref,
            required_total_bytes,
            total_vram_bytes,
            participants,
            excluded
        )
    );

    let vram_by_node = participants
        .iter()
        .map(|participant| (participant.node_id, participant.vram_bytes))
        .collect::<HashMap<_, _>>();
    let layer_weights = planner_layer_weight_bytes(package);
    let recurrent_by_layer = planner_recurrent_bytes_by_layer(
        &capacity.recurrent_bytes_per_sequence_by_layer,
        package.layer_count,
    );
    let kv_per_layer = capacity
        .kv_bytes_per_token
        .div_ceil(u64::from(package.layer_count.max(1)));
    for stage in stages {
        let node_vram = vram_by_node
            .get(&stage.node_id)
            .copied()
            .unwrap_or_default();
        let headroom = if capacity.budgets_runtime_headroom {
            default_runtime_headroom_bytes(node_vram)
        } else {
            0
        };
        let usable_vram_bytes = node_vram.saturating_sub(headroom);
        let required_bytes = stage_required_bytes(
            stage,
            &layer_weights,
            &recurrent_by_layer,
            kv_per_layer,
            capacity.context_length,
            capacity.parallel_lanes,
        );
        anyhow::ensure!(
            usable_vram_bytes >= required_bytes,
            "{} assigned to {} for {model_ref} exceeds node capacity: requires {} against usable {} (node budget {} minus {} runtime headroom) for {} layer(s) of repriced weights plus context-scaled KV at the compute reserve and recurrent state (context {}, {} lane(s))",
            stage.stage_id,
            stage.node_id.fmt_short(),
            format_gb(required_bytes),
            format_gb(usable_vram_bytes),
            format_gb(node_vram),
            format_gb(headroom),
            stage.layer_end.saturating_sub(stage.layer_start),
            capacity.context_length,
            capacity.parallel_lanes,
        );
    }
    Ok(())
}

pub(super) fn format_aggregate_split_capacity_error(
    model_ref: &str,
    required_bytes: u64,
    available_bytes: u64,
    participants: &[SplitParticipant],
    excluded: &[SplitParticipantExclusion],
) -> String {
    SplitCapacityReadinessReport::new(required_bytes, available_bytes, participants, excluded)
        .error_message(model_ref)
}

pub(super) fn format_gb(bytes: u64) -> String {
    format!("{:.1}GB", bytes as f64 / 1e9)
}

pub(super) fn split_stage_plan_labels(stages: &[RuntimeSliceStagePlan]) -> Vec<String> {
    stages
        .iter()
        .map(|stage| {
            format!(
                "{}:{}:{}..{}",
                stage.stage_id,
                stage.node_id.fmt_short(),
                stage.layer_start,
                stage.layer_end
            )
        })
        .collect()
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SplitCapacityReadinessReport {
    required_bytes: u64,
    available_bytes: u64,
    missing_bytes: u64,
    participants: Vec<SplitParticipant>,
    excluded: Vec<SplitParticipantExclusion>,
}

impl SplitCapacityReadinessReport {
    fn new(
        required_bytes: u64,
        available_bytes: u64,
        participants: &[SplitParticipant],
        excluded: &[SplitParticipantExclusion],
    ) -> Self {
        Self {
            required_bytes,
            available_bytes,
            missing_bytes: required_bytes.saturating_sub(available_bytes),
            participants: participants.to_vec(),
            excluded: excluded.to_vec(),
        }
    }

    fn error_message(&self, model_ref: &str) -> String {
        let mut message = format!(
            "split_capacity_shortfall: aggregate split capacity for {model_ref} requires {}, mesh has {} across {} participant(s), short by {}",
            format_gb(self.required_bytes),
            format_gb(self.available_bytes),
            self.participants.len(),
            format_gb(self.missing_bytes)
        );
        if !self.participants.is_empty() {
            message.push_str("; participants [");
            message.push_str(&split_participant_labels(&self.participants).join(", "));
            message.push(']');
        }
        if !self.excluded.is_empty() {
            message.push_str("; excluded [");
            message.push_str(&split_participant_exclusion_labels(&self.excluded).join(", "));
            message.push(']');
        }
        message
    }
}

#[cfg(test)]
mod tests {
    use super::super::local_package::SplitParticipantExclusionReason;
    use super::*;
    use iroh::SecretKey;
    use std::path::PathBuf;

    fn stage(index: u32, seed: u8, layer_start: u32, layer_end: u32) -> RuntimeSliceStagePlan {
        RuntimeSliceStagePlan {
            stage_id: format!("stage-{index}"),
            stage_index: index,
            node_id: make_id(seed),
            layer_start,
            layer_end,
            parameter_bytes: 0,
        }
    }

    #[test]
    fn node_headroom_covers_the_compute_graph_floor() {
        // 12 GB budget: the proportional 10% alone left a stage priced 3.3 GB
        // under what it went on to use.
        for vram_gb in [4u64, 8, 12, 16] {
            let vram = vram_gb * 1024 * 1024 * 1024;
            let headroom = default_runtime_headroom_bytes(vram);
            assert!(
                headroom >= RUNTIME_NODE_HEADROOM_FLOOR_BYTES,
                "{vram_gb} GB node reserved {headroom} B, under the compute-graph floor"
            );
            assert!(
                headroom >= vram / 10,
                "{vram_gb} GB node lost the 10% share"
            );
            assert!(headroom < vram, "{vram_gb} GB node reserved everything");
        }
    }

    #[test]
    fn large_nodes_keep_the_proportional_share() {
        let eighty_gb = 80 * 1024 * 1024 * 1024;
        assert_eq!(default_runtime_headroom_bytes(eighty_gb), eighty_gb / 10);
    }

    #[test]
    fn small_nodes_are_not_reserved_out_of_existence() {
        let one_gb = 1024 * 1024 * 1024;
        assert_eq!(default_runtime_headroom_bytes(one_gb), one_gb / 2);
        assert_eq!(default_runtime_headroom_bytes(0), 0);
    }

    #[test]
    fn an_override_reprices_every_stage_for_its_new_range() {
        // validate_split_capacity prices a stage from parameter_bytes, so a
        // moved boundary with stale weights would be checked against a cut
        // that no longer exists.
        let mut pkg = package(36, 36 * 100);
        pkg.layer_weight_bytes = (0..36).map(|_| 100u64).collect();
        let mut stages = vec![stage(0, 1, 0, 18), stage(1, 2, 18, 36)];
        for s in stages.iter_mut() {
            s.parameter_bytes = 1_800;
        }
        apply_boundaries(&mut stages, &[12], 36);
        reprice_stages(&mut stages, &pkg);
        assert_eq!(stages[0].parameter_bytes, 12 * 100);
        assert_eq!(stages[1].parameter_bytes, 24 * 100);
    }

    #[test]
    fn repricing_falls_back_to_an_even_share_without_per_layer_weights() {
        let pkg = package(36, 3_600);
        let mut stages = vec![stage(0, 1, 0, 18), stage(1, 2, 18, 36)];
        apply_boundaries(&mut stages, &[12], 36);
        reprice_stages(&mut stages, &pkg);
        assert_eq!(stages[0].parameter_bytes, 12 * 100);
        assert_eq!(stages[1].parameter_bytes, 24 * 100);
    }

    const GIB: u64 = 1024 * 1024 * 1024;

    /// An overridden 12/24 cut of a 24-layer, 24 GiB model on two 20 GiB
    /// nodes: 12 GiB of weights per node leaves 6 GiB of usable budget for
    /// KV, compute reserve, and recurrent state.
    fn overridden_stages() -> (
        skippy::SkippyPackageIdentity,
        Vec<SplitParticipant>,
        Vec<RuntimeSliceStagePlan>,
    ) {
        let mut pkg = package(24, 24 * GIB);
        pkg.layer_weight_bytes = vec![GIB; 24];
        let participants = vec![participant(1, 20 * GIB), participant(2, 20 * GIB)];
        let mut stages = vec![stage(0, 1, 0, 12), stage(1, 2, 12, 24)];
        apply_boundaries(&mut stages, &[12], 24);
        reprice_stages(&mut stages, &pkg);
        (pkg, participants, stages)
    }

    fn capacity_model(
        kv_bytes_per_token: u64,
        recurrent_bytes_per_sequence_by_layer: Vec<u64>,
        context_length: u32,
        parallel_lanes: usize,
    ) -> SplitCapacityModel {
        SplitCapacityModel {
            kv_bytes_per_token,
            recurrent_bytes_per_sequence_by_layer,
            context_length,
            parallel_lanes,
            budgets_runtime_headroom: true,
        }
    }

    #[test]
    fn an_overridden_cut_that_fits_the_planner_budget_validates() {
        let (pkg, participants, stages) = overridden_stages();
        validate_split_capacity(
            "model-a",
            &pkg,
            &participants,
            &stages,
            &[],
            &capacity_model(128, Vec::new(), 65_536, 1),
        )
        .expect("12 GiB of weights plus negligible KV fits an 18 GiB usable budget");
    }

    #[test]
    fn an_override_that_fits_by_weights_but_exceeds_the_kv_budget_is_rejected() {
        let (pkg, participants, stages) = overridden_stages();
        // ~12.9 GB of context-scaled KV per stage on top of 12 GiB of weights
        // blows the 18 GiB usable budget even though the weights alone fit.
        let error = validate_split_capacity(
            "model-a",
            &pkg,
            &participants,
            &stages,
            &[],
            &capacity_model(4_000_000, Vec::new(), 65_536, 1),
        )
        .expect_err("weights that fit must not approve a cut exceeding the KV budget");
        assert!(
            error.to_string().contains("context-scaled KV"),
            "error should name the KV terms: {error}"
        );
    }

    #[test]
    fn an_override_that_exceeds_the_lane_scaled_recurrent_budget_is_rejected() {
        let (pkg, participants, stages) = overridden_stages();
        let recurrent = vec![256 * 1024 * 1024; 24];
        // One lane of recurrent state fits; four lanes of the same state do
        // not — recurrent cost scales with lanes, weights do not.
        validate_split_capacity(
            "model-a",
            &pkg,
            &participants,
            &stages,
            &[],
            &capacity_model(0, recurrent.clone(), 65_536, 1),
        )
        .expect("a single recurrent lane fits beside the weights");
        let error = validate_split_capacity(
            "model-a",
            &pkg,
            &participants,
            &stages,
            &[],
            &capacity_model(0, recurrent, 65_536, 4),
        )
        .expect_err("four recurrent lanes must exceed the budget the weights left");
        assert!(
            error.to_string().contains("recurrent state"),
            "error should name the recurrent terms: {error}"
        );
    }

    #[test]
    fn weights_that_only_fit_before_runtime_headroom_are_rejected() {
        // 13.5 GB of weights per stage on 13.9 GB nodes fit the old raw-VRAM
        // check; the planner's 10% headroom leaves 12.51 GB usable, so the
        // same placement is rejected.
        let weights_per_layer = 1_125_000_000u64;
        let mut pkg = package(24, weights_per_layer * 24);
        pkg.layer_weight_bytes = vec![weights_per_layer; 24];
        let participants = vec![
            participant(1, 13_900_000_000),
            participant(2, 13_900_000_000),
        ];
        let mut stages = vec![stage(0, 1, 0, 12), stage(1, 2, 12, 24)];
        apply_boundaries(&mut stages, &[12], 24);
        reprice_stages(&mut stages, &pkg);
        assert_eq!(stages[0].parameter_bytes, weights_per_layer * 12);
        let error = validate_split_capacity(
            "model-a",
            &pkg,
            &participants,
            &stages,
            &[],
            &capacity_model(0, Vec::new(), 1, 1),
        )
        .expect_err("weights sized against raw VRAM must fail once headroom is charged");
        assert!(
            error.to_string().contains("runtime headroom"),
            "error should name the headroom terms: {error}"
        );
    }

    #[test]
    fn a_fixed_split_ignores_the_override_entirely() {
        // The knob exists to exercise the controller. On a fixed split there
        // is no controller, so it must not move placement.
        let pkg = package(36, 3_600);
        let original = vec![stage(0, 1, 0, 18), stage(1, 2, 18, 36)];
        let mut stages = original.clone();
        apply_initial_cut_override(&mut stages, &pkg, false);
        assert_eq!(stages, original);
    }

    #[test]
    fn initial_cut_override_rewrites_boundaries_in_order() {
        let mut stages = vec![stage(0, 1, 0, 18), stage(1, 2, 18, 36)];
        apply_boundaries(&mut stages, &[12], 36);
        assert_eq!(
            stages
                .iter()
                .map(|s| (s.layer_start, s.layer_end))
                .collect::<Vec<_>>(),
            vec![(0, 12), (12, 36)]
        );
        // Nodes and order are untouched: only the cut moves.
        assert_eq!(stages[0].node_id, make_id(1));
        assert_eq!(stages[1].node_id, make_id(2));
    }

    #[test]
    fn initial_cut_override_ignores_a_cut_for_a_different_topology() {
        let original = vec![stage(0, 1, 0, 18), stage(1, 2, 18, 36)];
        for boundaries in [vec![12, 24], vec![36], vec![40]] {
            let mut stages = original.clone();
            apply_boundaries(&mut stages, &boundaries, 36);
            assert_eq!(
                stages, original,
                "boundaries {boundaries:?} should be ignored"
            );
        }
    }

    #[test]
    fn initial_cut_override_parses_only_sane_values() {
        assert_eq!(parse_initial_cut("12"), Some(vec![12]));
        assert_eq!(parse_initial_cut(" 12 , 24 "), Some(vec![12, 24]));
        assert_eq!(parse_initial_cut("0"), None, "a stage cannot be empty");
        assert_eq!(parse_initial_cut("24,12"), None, "must increase");
        assert_eq!(parse_initial_cut("memory"), None);
        assert_eq!(parse_initial_cut(""), None);
    }

    fn make_id(seed: u8) -> iroh::EndpointId {
        let mut bytes = [0u8; 32];
        bytes[0] = seed;
        SecretKey::from_bytes(&bytes).public()
    }

    fn package(layer_count: u32, source_model_bytes: u64) -> skippy::SkippyPackageIdentity {
        skippy::SkippyPackageIdentity {
            package_ref: "gguf:///models/qwen.gguf".to_string(),
            manifest_sha256: "manifest".to_string(),
            source_model_path: PathBuf::from("/models/qwen.gguf"),
            source_model_sha256: "source".to_string(),
            source_model_bytes,
            source_files: Vec::new(),
            layer_weight_bytes: Vec::new(),
            layer_count,
            activation_width: 896,
            tensor_count: 100,
            generation: None,
        }
    }

    fn participant(seed: u8, vram_bytes: u64) -> SplitParticipant {
        SplitParticipant::new(make_id(seed), vram_bytes, None)
    }

    fn participant_with_rtt(seed: u8, vram_bytes: u64, rtt_ms: u32) -> SplitParticipant {
        let mut participant = participant(seed, vram_bytes);
        participant.rtt_ms = Some(rtt_ms);
        participant
    }

    #[test]
    fn default_runtime_headroom_reserves_decode_margin() {
        // This fixed reserve is 1/10 (10%) of the advertised budget — the
        // context-independent half of the overhead model (OS/frontend margin +
        // advertised-vs-physical slack). The context-scaled half (compute-graph
        // buffers) is charged separately in the topology planner's KV term.
        // Regression guard for the zero-headroom bug that OOM'd stages / swapped
        // hosts.
        const GIB: u64 = 1024 * 1024 * 1024;
        assert_eq!(default_runtime_headroom_bytes(0), 0);
        assert_eq!(
            default_runtime_headroom_bytes(20 * GIB),
            2 * GIB,
            "1/10 of a 20 GiB budget should be reserved"
        );
        // A ~115 GB advertised budget should reserve ~11 GB of headroom.
        let budget = 115_000_000_000u64;
        let headroom = default_runtime_headroom_bytes(budget);
        assert!(
            headroom >= 11_000_000_000 && headroom < budget,
            "expected ~11 GB headroom under the budget, got {headroom}"
        );
        // Headroom must scale with the budget so bigger nodes reserve more.
        assert!(
            default_runtime_headroom_bytes(64 * GIB) < default_runtime_headroom_bytes(128 * GIB)
        );
    }

    #[test]
    fn selects_participants_in_stage_order() {
        let a = participant(1, 24_000_000_000);
        let b = participant(2, 24_000_000_000);
        let stages = vec![
            RuntimeSliceStagePlan {
                stage_id: "stage-0".to_string(),
                stage_index: 0,
                node_id: b.node_id,
                layer_start: 0,
                layer_end: 10,
                parameter_bytes: 10_000_000,
            },
            RuntimeSliceStagePlan {
                stage_id: "stage-1".to_string(),
                stage_index: 1,
                node_id: a.node_id,
                layer_start: 10,
                layer_end: 20,
                parameter_bytes: 10_000_000,
            },
        ];

        let selected = split_participants_for_stages(&[a, b], &stages);

        assert_eq!(
            selected
                .iter()
                .map(|participant| participant.node_id)
                .collect::<Vec<_>>(),
            vec![b.node_id, a.node_id]
        );
    }

    #[test]
    fn resource_planner_returns_runtime_stage_shape() {
        let participants = vec![
            participant(1, 42_000_000_000),
            participant(2, 42_000_000_000),
            participant(3, 42_000_000_000),
        ];

        let plan = plan_runtime_slice_topology_with_resources(
            "topology-test",
            "model-a",
            &package(30, 60_000_000_000),
            &participants,
            &[],
            SplitTopologyResourceInputs {
                native_context_length: 65_536,
                kv_bytes_per_token: 16 * 1024,
                recurrent_bytes_per_sequence_by_layer: Vec::new(),
                ctx_size_override: None,
                parallel_override: None,
                auto_balance: false,
            },
        )
        .expect("resource-aware topology");

        assert_eq!(plan.context_length, 65_536);
        assert_eq!(plan.stages.len(), 2);
        assert!(plan.slots > 0);
        assert_eq!(plan.stages.first().unwrap().layer_start, 0);
        assert_eq!(plan.stages.last().unwrap().layer_end, 30);
    }

    #[test]
    fn resource_planner_uses_exact_package_layer_weights() {
        const GIB: u64 = 1024 * 1024 * 1024;
        let participants = vec![participant(1, 12 * GIB), participant(2, 11 * GIB)];
        let mut package = package(4, 18 * GIB);
        package.layer_weight_bytes = vec![GIB / 8, GIB / 8, 9 * GIB, 8 * GIB];

        let plan = plan_runtime_slice_topology_with_resources(
            "topology-test",
            "model-a",
            &package,
            &participants,
            &[],
            SplitTopologyResourceInputs {
                native_context_length: 1,
                kv_bytes_per_token: 1,
                recurrent_bytes_per_sequence_by_layer: Vec::new(),
                ctx_size_override: Some(1),
                parallel_override: Some(1),
                auto_balance: false,
            },
        )
        .expect("resource-aware topology with exact layer weights");

        assert_eq!(
            plan.stages
                .iter()
                .map(|stage| (stage.layer_start, stage.layer_end))
                .collect::<Vec<_>>(),
            vec![(0, 3), (3, 4)]
        );
        assert_eq!(plan.stages[0].parameter_bytes, 9 * GIB + GIB / 4);
        assert_eq!(plan.stages[1].parameter_bytes, 8 * GIB);
    }

    #[test]
    fn resource_planner_assigns_mi300x_a_capacity_weighted_share() {
        const GIB: u64 = 1024 * 1024 * 1024;
        let mi300x = participant(1, 192 * GIB);
        let smaller_accelerator = participant(2, 48 * GIB);
        let participants = vec![mi300x, smaller_accelerator];
        let package = package(100, 200 * GIB);

        let plan = plan_runtime_slice_topology_with_resources(
            "mi300x-topology-test",
            "unsloth/Qwen3.5-122B-A10B-MTP-GGUF:UD-Q8_K_XL",
            &package,
            &participants,
            &[],
            SplitTopologyResourceInputs {
                native_context_length: 1,
                kv_bytes_per_token: 1,
                recurrent_bytes_per_sequence_by_layer: Vec::new(),
                ctx_size_override: Some(1),
                parallel_override: Some(1),
                auto_balance: false,
            },
        )
        .expect("MI300X and smaller accelerator should form a valid topology");

        assert_eq!(plan.stages.len(), 2);
        assert_eq!(plan.stages[0].node_id, mi300x.node_id);
        assert_eq!(plan.stages[1].node_id, smaller_accelerator.node_id);
        assert!(
            plan.stages[0].parameter_bytes >= plan.stages[1].parameter_bytes * 3,
            "the 192 GiB MI300X should receive most of the model weight: {:?}",
            plan.stages
        );
    }

    #[test]
    fn resource_planner_prefers_lower_tpot_stage_count_from_participant_rtt() {
        // Node VRAM is sized so the 40GB model fits on two nodes at the minimum
        // context (65536) but needs three at the native context (262144), so
        // latency-aware planning should prefer the two-stage/min-context shape.
        // The budget must clear the two-node fit *after* both halves of the
        // overhead model: the 1/10 fixed node headroom (see
        // default_runtime_headroom_bytes) and the topology planner's KV compute
        // reserve (KV billed at 100/85). 26GB usable ≈ 23.4GB fits 20 layers at
        // 65536 (~22.5GB) but not at 131072 (~25GB), preserving the invariant.
        let participants = vec![
            participant_with_rtt(1, 26_000_000_000, 10),
            participant_with_rtt(2, 26_000_000_000, 10),
            participant_with_rtt(3, 26_000_000_000, 10),
            participant_with_rtt(4, 26_000_000_000, 10),
        ];

        let plan = plan_runtime_slice_topology_with_resources(
            "topology-test",
            "model-a",
            &package(40, 40_000_000_000),
            &participants,
            &[],
            SplitTopologyResourceInputs {
                native_context_length: 262_144,
                kv_bytes_per_token: 64 * 1024,
                recurrent_bytes_per_sequence_by_layer: Vec::new(),
                ctx_size_override: None,
                parallel_override: None,
                auto_balance: false,
            },
        )
        .expect("latency-aware runtime topology");

        assert_eq!(plan.context_length, 65_536);
        assert_eq!(plan.stages.len(), 2);
        assert_eq!(plan.stages.first().unwrap().layer_start, 0);
        assert_eq!(plan.stages.last().unwrap().layer_end, 40);
    }

    #[test]
    fn capacity_report_includes_participants_and_exclusions() {
        let participants = vec![participant(1, 40_000_000_000)];
        let excluded = vec![SplitParticipantExclusion {
            node_id: make_id(2),
            reason: SplitParticipantExclusionReason::MissingVram,
        }];

        let message = format_aggregate_split_capacity_error(
            "model-a",
            100_000_000_000,
            40_000_000_000,
            &participants,
            &excluded,
        );

        assert!(message.contains("split_capacity_shortfall"));
        assert!(message.contains("model-a"));
        assert!(message.contains("short by 60.0GB"));
        assert!(message.contains("participants ["));
        assert!(message.contains("excluded ["));
        assert!(message.contains("missing_vram"));
    }

    #[test]
    fn topology_failure_reason_reports_floor_fit_capacity() {
        let participants = vec![participant(1, 8_000_000_000), participant(2, 8_000_000_000)];
        let excluded = vec![SplitParticipantExclusion {
            node_id: make_id(3),
            reason: SplitParticipantExclusionReason::MissingModelSource,
        }];

        let reason = split_topology_failure_reason(
            "model-a",
            &package(4, 40_000_000_000),
            &participants,
            &excluded,
            SplitTopologyResourceInputs {
                native_context_length: 131_072,
                kv_bytes_per_token: 1024,
                recurrent_bytes_per_sequence_by_layer: Vec::new(),
                ctx_size_override: None,
                parallel_override: None,
                auto_balance: false,
            },
        );

        assert!(reason.contains("model-a"));
        assert!(reason.contains("minimum_context=65536"));
        assert!(reason.contains("evaluated_context=65536"));
        assert!(reason.contains("evaluated_lanes=1"));
        assert!(reason.contains("max_placeable_layers_at_evaluated_shape=0/4"));
        assert!(reason.contains("participants ["));
        assert!(reason.contains("max_layers=0"));
        assert!(reason.contains("missing_model_source"));
    }
}
