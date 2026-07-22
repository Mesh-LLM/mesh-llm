use super::coordinator::SplitTopologyGeneration;
use super::{RuntimeSliceStagePlan, SplitParticipant};
use crate::inference::skippy;
use crate::mesh;
use crate::runtime::local_package::SPLIT_DEFAULT_MIN_PARTICIPANTS;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum SplitLossRecoveryDecision {
    NoActiveStageLoss,
    ReplacementSplit,
    LocalFallback,
    Withdraw,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum SplitWithdrawGraceAction {
    Defer,
    Withdraw,
}

pub(super) fn split_withdraw_grace_action(
    first_seen: Option<Instant>,
    now: Instant,
    grace: Duration,
) -> SplitWithdrawGraceAction {
    match first_seen {
        Some(seen) if now.duration_since(seen) >= grace => SplitWithdrawGraceAction::Withdraw,
        _ => SplitWithdrawGraceAction::Defer,
    }
}

pub(super) fn split_loss_recovery_decision(
    active: &SplitTopologyGeneration,
    connected_node_ids: &[iroh::EndpointId],
    unavailable_stage_nodes: &[iroh::EndpointId],
    candidate: Option<&SplitTopologyGeneration>,
    local_model_fits: bool,
) -> SplitLossRecoveryDecision {
    if split_missing_active_stage_nodes(active, connected_node_ids).is_empty()
        && unavailable_stage_nodes.is_empty()
    {
        return SplitLossRecoveryDecision::NoActiveStageLoss;
    }
    if candidate.is_some_and(|candidate| {
        split_candidate_is_valid_replacement_split_after_loss(candidate, unavailable_stage_nodes)
    }) {
        return SplitLossRecoveryDecision::ReplacementSplit;
    }
    if local_model_fits {
        return SplitLossRecoveryDecision::LocalFallback;
    }
    SplitLossRecoveryDecision::Withdraw
}

pub(super) fn split_locked_loss_recovery_decision(
    active: &SplitTopologyGeneration,
    connected_node_ids: &[iroh::EndpointId],
    unavailable_stage_nodes: &[iroh::EndpointId],
) -> SplitLossRecoveryDecision {
    split_loss_recovery_decision(
        active,
        connected_node_ids,
        unavailable_stage_nodes,
        None,
        false,
    )
}

pub(super) fn split_candidate_is_valid_replacement_split(
    candidate: &SplitTopologyGeneration,
) -> bool {
    split_participants_meet_minimum(&candidate.participants)
        && split_stages_meet_minimum(&candidate.stages)
}

pub(super) fn split_candidate_is_valid_replacement_split_after_loss(
    candidate: &SplitTopologyGeneration,
    unavailable_stage_nodes: &[iroh::EndpointId],
) -> bool {
    split_candidate_is_valid_replacement_split(candidate)
        && !split_candidate_uses_unavailable_stage_node(candidate, unavailable_stage_nodes)
}

pub(super) fn split_candidate_uses_unavailable_stage_node(
    candidate: &SplitTopologyGeneration,
    unavailable_stage_nodes: &[iroh::EndpointId],
) -> bool {
    candidate
        .stages
        .iter()
        .any(|stage| unavailable_stage_nodes.contains(&stage.node_id))
}

pub(super) fn split_participants_meet_minimum(participants: &[SplitParticipant]) -> bool {
    participants.len() >= SPLIT_DEFAULT_MIN_PARTICIPANTS
}

pub(super) fn split_stages_meet_minimum(stages: &[RuntimeSliceStagePlan]) -> bool {
    stages.len() >= SPLIT_DEFAULT_MIN_PARTICIPANTS
}

pub(super) fn split_missing_active_stage_nodes(
    active: &SplitTopologyGeneration,
    connected_node_ids: &[iroh::EndpointId],
) -> Vec<iroh::EndpointId> {
    let mut missing = Vec::new();
    for stage in &active.stages {
        if connected_node_ids.contains(&stage.node_id) || missing.contains(&stage.node_id) {
            continue;
        }
        missing.push(stage.node_id);
    }
    missing
}

pub(super) fn split_unavailable_active_stage_nodes(
    active: &SplitTopologyGeneration,
    connected_node_ids: &[iroh::EndpointId],
    runtime_statuses: &[mesh::StageRuntimeStatus],
) -> Vec<iroh::EndpointId> {
    let mut unavailable = split_missing_active_stage_nodes(active, connected_node_ids);
    for status in runtime_statuses {
        if !matches!(
            status.state,
            skippy::StageRuntimeState::Failed
                | skippy::StageRuntimeState::Stopping
                | skippy::StageRuntimeState::Stopped
        ) || status.topology_id != active.topology_id
            || status.run_id != active.run_id
            || active
                .stages
                .iter()
                .all(|stage| stage.stage_id != status.stage_id)
        {
            continue;
        }
        let Some(node_id) = status.node_id else {
            continue;
        };
        if !unavailable.contains(&node_id) {
            unavailable.push(node_id);
        }
    }
    unavailable
}

pub(super) fn split_active_stage_nodes_pending_eligibility(
    active: &SplitTopologyGeneration,
    connected_node_ids: &[iroh::EndpointId],
    eligible_participants: &[SplitParticipant],
    unavailable_stage_nodes: &[iroh::EndpointId],
) -> Vec<iroh::EndpointId> {
    active
        .stages
        .iter()
        .filter_map(|stage| {
            let connected = connected_node_ids.contains(&stage.node_id);
            let eligible = eligible_participants
                .iter()
                .any(|participant| participant.node_id == stage.node_id);
            let unavailable = unavailable_stage_nodes.contains(&stage.node_id);
            (connected && !eligible && !unavailable).then_some(stage.node_id)
        })
        .collect()
}

pub(super) async fn split_connected_node_ids(node: &mesh::Node) -> Vec<iroh::EndpointId> {
    let mut node_ids = vec![node.id()];
    node_ids.extend(node.peers().await.into_iter().map(|peer| peer.id));
    node_ids.sort_by_key(ToString::to_string);
    node_ids.dedup();
    node_ids
}
