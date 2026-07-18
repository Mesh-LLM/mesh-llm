use crate::inference::skippy;
use anyhow::{Context, Result};
use skippy_coordinator::topology::{
    TopologyNode, TopologyPlanningInput, TopologyStagePlan, minimum_valid_context, plan_topology,
};
use std::collections::HashMap;

use super::local::{SplitParticipant, SplitParticipantExclusion};

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
    pub(super) context_length_override: Option<u32>,
    pub(super) parallel_lanes_override: Option<usize>,
    pub(super) target_decode_tpot_ms: Option<u32>,
    pub(super) minimum_nodes: usize,
    pub(super) nodes: Vec<SplitTopologyPlanNode>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct SplitTopologyPlanNode {
    pub(super) node_id: String,
    pub(super) detected_vram_bytes: u64,
    pub(super) max_vram_bytes: Option<u64>,
    pub(super) runtime_headroom_bytes: u64,
    pub(super) stage_transfer_latency_ms: Option<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct SplitTopologyPlan {
    pub(super) context_length: u32,
    pub(super) parallel_lanes: usize,
    pub(super) estimated_decode_network_ms_per_token: Option<u32>,
    pub(super) decode_tpot_target_met: Option<bool>,
    pub(super) stages: Vec<TopologyStagePlan>,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct SplitTopologyResourceInputs {
    pub(super) native_context_length: u32,
    pub(super) kv_bytes_per_token: u64,
    pub(super) ctx_size_override: Option<u32>,
    pub(super) parallel_override: Option<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct PlannedRuntimeSliceTopology {
    pub(super) stages: Vec<RuntimeSliceStagePlan>,
    pub(super) context_length: u32,
    pub(super) slots: usize,
}

pub(super) fn plan_split_topology(input: SplitTopologyPlanInput) -> Result<SplitTopologyPlan> {
    let plan = plan_topology(&TopologyPlanningInput {
        native_context_length: input.native_context_length,
        layer_count: input.layer_count,
        model_weight_bytes: input.model_weight_bytes,
        layer_weight_bytes: input.layer_weight_bytes,
        kv_bytes_per_token: input.kv_bytes_per_token,
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
            })
            .collect(),
        context_length_override: input.context_length_override,
        parallel_lanes_override: input.parallel_lanes_override,
        target_decode_tpot_ms: input.target_decode_tpot_ms,
    })
    .context("plan skippy split topology")?;

    Ok(SplitTopologyPlan {
        context_length: plan.context_length,
        parallel_lanes: plan.parallel_lanes,
        estimated_decode_network_ms_per_token: plan.estimated_decode_network_ms_per_token,
        decode_tpot_target_met: plan.decode_tpot_target_met,
        stages: plan.stages,
    })
}

pub(super) fn default_runtime_headroom_bytes(vram_bytes: u64) -> u64 {
    vram_bytes
        .saturating_mul(RUNTIME_NODE_HEADROOM_NUMERATOR)
        .div_ceil(RUNTIME_NODE_HEADROOM_DENOMINATOR)
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
    tracing::info!(
        topology_id,
        model_ref,
        participants = ?split_participant_labels(participants),
        layer_count = package.layer_count,
        native_context_length = resources.native_context_length,
        "planning resource-aware split runtime topology"
    );

    let participant_by_id = participant_index_by_id(participants);
    let plan_input = runtime_slice_plan_input(package, participants, resources);
    let plan = plan_runtime_slice_topology_result(
        topology_id,
        model_ref,
        package,
        participants,
        excluded,
        resources,
        plan_input,
    )?;

    let mut stages = map_runtime_slice_stages(plan.stages, &participant_by_id)?;
    stages.sort_by_key(|stage| stage.stage_index);
    validate_split_capacity(model_ref, package, participants, &stages, excluded)?;
    tracing::info!(
        topology_id,
        model_ref,
        context_length = plan.context_length,
        slots = plan.parallel_lanes,
        estimated_decode_network_ms_per_token = plan.estimated_decode_network_ms_per_token,
        decode_tpot_target_met = plan.decode_tpot_target_met,
        stages = ?split_stage_plan_labels(&stages),
        "planned resource-aware split runtime topology"
    );
    Ok(PlannedRuntimeSliceTopology {
        stages,
        context_length: plan.context_length,
        slots: plan.parallel_lanes,
    })
}

fn plan_runtime_slice_topology_result(
    topology_id: &str,
    model_ref: &str,
    package: &skippy::SkippyPackageIdentity,
    participants: &[SplitParticipant],
    excluded: &[SplitParticipantExclusion],
    resources: SplitTopologyResourceInputs,
    plan_input: SplitTopologyPlanInput,
) -> Result<SplitTopologyPlan> {
    match plan_split_topology(plan_input) {
        Ok(plan) => Ok(plan),
        Err(err) => {
            let reason = split_topology_failure_reason(
                model_ref,
                package,
                participants,
                excluded,
                resources,
            );
            tracing::warn!(
                topology_id,
                model_ref,
                error = %err,
                reason = %reason,
                participants = ?split_participant_labels(participants),
                excluded = ?split_participant_exclusion_labels(excluded),
                "failed to plan resource-aware split runtime topology"
            );
            Err(err.context(reason))
        }
    }
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
            })
            .collect(),
    }
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

pub(super) fn validate_split_capacity(
    model_ref: &str,
    package: &skippy::SkippyPackageIdentity,
    participants: &[SplitParticipant],
    stages: &[RuntimeSliceStagePlan],
    excluded: &[SplitParticipantExclusion],
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
    for stage in stages {
        let node_vram = vram_by_node
            .get(&stage.node_id)
            .copied()
            .unwrap_or_default();
        // The topology planner already budgets VRAM including KV cache and
        // headroom.  Do not re-apply the solo-load 10% headroom here — it
        // double-counts and rejects topologies the planner approved.
        anyhow::ensure!(
            node_vram >= stage.parameter_bytes,
            "{} assigned to {} for {model_ref} requires {}, which exceeds node capacity {}",
            stage.stage_id,
            stage.node_id.fmt_short(),
            format_gb(stage.parameter_bytes),
            format_gb(node_vram)
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
                ctx_size_override: None,
                parallel_override: None,
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
                ctx_size_override: Some(1),
                parallel_override: Some(1),
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
                ctx_size_override: None,
                parallel_override: None,
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
                ctx_size_override: None,
                parallel_override: None,
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
