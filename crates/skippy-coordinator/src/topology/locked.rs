use super::{
    CandidatePlan, TopologyPlan, TopologyPlanError, TopologyPlanningInput, TopologyStagePlan,
    UsableNode, context_candidates, decode_tpot_target_met, estimate_decode_network_ms_per_token,
    layer_required_bytes, layer_weight_bytes, minimum_valid_context, parallel_lane_candidates,
    sum_u64, usable_nodes, validate_input,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LockedTopologyStage {
    pub node_id: String,
    pub layer_start: u32,
    pub layer_end: u32,
}

pub fn plan_locked_topology(
    input: &TopologyPlanningInput,
    locked_stages: &[LockedTopologyStage],
) -> Result<TopologyPlan, TopologyPlanError> {
    validate_input(input)?;
    validate_locked_stages(input, locked_stages)?;

    let minimum_context = minimum_valid_context(input.native_context_length);
    let context_candidates = context_candidates(
        input.native_context_length,
        minimum_context,
        input.context_length_override,
    )?;
    let lane_candidates = parallel_lane_candidates(input.parallel_lanes_override)?;
    let nodes = usable_nodes(&input.nodes);
    let locked_nodes = locked_stage_nodes(&nodes, locked_stages)?;

    for context_length in context_candidates {
        for parallel_lanes in lane_candidates.iter().copied() {
            if let Some(candidate) = fit_locked_candidate(
                input,
                locked_stages,
                &locked_nodes,
                context_length,
                parallel_lanes,
            ) {
                return Ok(candidate.plan);
            }
        }
    }

    Err(TopologyPlanError::LockedTopologyDoesNotFit { minimum_context })
}

fn validate_locked_stages(
    input: &TopologyPlanningInput,
    locked_stages: &[LockedTopologyStage],
) -> Result<(), TopologyPlanError> {
    let minimum = input.minimum_nodes.max(1);
    if locked_stages.len() < minimum {
        return Err(TopologyPlanError::LockedStageCount {
            minimum,
            actual: locked_stages.len(),
        });
    }

    let mut expected_start = 0;
    let mut seen_nodes = std::collections::HashSet::new();
    for (stage_index, stage) in locked_stages.iter().enumerate() {
        if stage.layer_end <= stage.layer_start {
            return Err(TopologyPlanError::LockedInvalidRange {
                stage_index,
                start: stage.layer_start,
                end: stage.layer_end,
            });
        }
        if stage.layer_start != expected_start {
            return Err(TopologyPlanError::LockedNonContiguousRange {
                stage_index,
                expected_start,
                actual_start: stage.layer_start,
            });
        }
        if !seen_nodes.insert(stage.node_id.as_str()) {
            return Err(TopologyPlanError::LockedDuplicateNode {
                node_id: stage.node_id.clone(),
            });
        }
        expected_start = stage.layer_end;
    }
    if expected_start != input.layer_count {
        return Err(TopologyPlanError::LockedIncompleteCoverage {
            actual_end: expected_start,
            layer_count: input.layer_count,
        });
    }
    Ok(())
}

fn locked_stage_nodes(
    nodes: &[UsableNode],
    locked_stages: &[LockedTopologyStage],
) -> Result<Vec<UsableNode>, TopologyPlanError> {
    locked_stages
        .iter()
        .map(|stage| {
            nodes
                .iter()
                .find(|node| node.node_id == stage.node_id)
                .cloned()
                .ok_or_else(|| TopologyPlanError::LockedUnknownNode {
                    node_id: stage.node_id.clone(),
                })
        })
        .collect()
}

fn fit_locked_candidate(
    input: &TopologyPlanningInput,
    locked_stages: &[LockedTopologyStage],
    nodes: &[UsableNode],
    context_length: u32,
    parallel_lanes: usize,
) -> Option<CandidatePlan> {
    let layer_weights = layer_weight_bytes(input);
    let kv_per_layer = input
        .kv_bytes_per_token
        .div_ceil(u64::from(input.layer_count));
    let layer_required_bytes =
        layer_required_bytes(&layer_weights, kv_per_layer, context_length, parallel_lanes)?;
    let mut stages = Vec::with_capacity(locked_stages.len());
    let mut minimum_remaining_vram = u64::MAX;
    let mut total_remaining_vram = 0u128;

    for (stage_index, (locked, node)) in locked_stages.iter().zip(nodes).enumerate() {
        let range = locked.layer_start as usize..locked.layer_end as usize;
        let parameter_bytes = sum_u64(&layer_weights[range.clone()]);
        let required_bytes = sum_u64(&layer_required_bytes[range]);
        if required_bytes > node.usable_vram_bytes {
            return None;
        }
        let remaining = node.usable_vram_bytes - required_bytes;
        minimum_remaining_vram = minimum_remaining_vram.min(remaining);
        total_remaining_vram += u128::from(remaining);
        stages.push(TopologyStagePlan {
            stage_id: format!("stage-{stage_index}"),
            stage_index: stage_index as u32,
            node_id: locked.node_id.clone(),
            layer_start: locked.layer_start,
            layer_end: locked.layer_end,
            parameter_bytes,
        });
    }

    let estimated_decode_network_ms_per_token = estimate_decode_network_ms_per_token(nodes);
    Some(CandidatePlan {
        plan: TopologyPlan {
            context_length,
            parallel_lanes,
            stages,
            estimated_decode_network_ms_per_token,
            decode_tpot_target_met: decode_tpot_target_met(
                estimated_decode_network_ms_per_token,
                input.target_decode_tpot_ms,
            ),
        },
        minimum_remaining_vram,
        total_remaining_vram,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::TopologyNode;

    const GIB: u64 = 1024 * 1024 * 1024;

    fn node(id: &str, gib: u64) -> TopologyNode {
        TopologyNode {
            node_id: id.to_string(),
            detected_vram_bytes: gib * GIB,
            max_vram_bytes: None,
            runtime_headroom_bytes: 0,
            stage_transfer_latency_ms: None,
        }
    }

    fn input(nodes: Vec<TopologyNode>) -> TopologyPlanningInput {
        TopologyPlanningInput {
            native_context_length: 65_536,
            layer_count: 40,
            model_weight_bytes: 40 * GIB,
            layer_weight_bytes: Vec::new(),
            kv_bytes_per_token: 64 * 1024,
            minimum_nodes: 2,
            nodes,
            context_length_override: None,
            parallel_lanes_override: None,
            target_decode_tpot_ms: None,
        }
    }

    fn stage(node_id: &str, layer_start: u32, layer_end: u32) -> LockedTopologyStage {
        LockedTopologyStage {
            node_id: node_id.to_string(),
            layer_start,
            layer_end,
        }
    }

    #[test]
    fn preserves_node_order_and_layer_ranges() {
        let request = input(vec![node("large", 48), node("small", 24)]);
        let locked = vec![stage("small", 0, 12), stage("large", 12, 40)];

        let plan = plan_locked_topology(&request, &locked).unwrap();

        assert_eq!(
            plan.stages
                .iter()
                .map(|stage| (stage.node_id.as_str(), stage.layer_start, stage.layer_end))
                .collect::<Vec<_>>(),
            vec![("small", 0, 12), ("large", 12, 40)]
        );
    }

    #[test]
    fn rejects_too_few_stages() {
        let request = input(vec![node("a", 80), node("b", 80)]);

        assert_eq!(
            plan_locked_topology(&request, &[stage("a", 0, 40)]),
            Err(TopologyPlanError::LockedStageCount {
                minimum: 2,
                actual: 1,
            })
        );
    }

    #[test]
    fn rejects_unknown_node() {
        let request = input(vec![node("a", 48), node("b", 48)]);
        let locked = vec![stage("a", 0, 20), stage("missing", 20, 40)];

        assert_eq!(
            plan_locked_topology(&request, &locked),
            Err(TopologyPlanError::LockedUnknownNode {
                node_id: "missing".to_string(),
            })
        );
    }

    #[test]
    fn rejects_duplicate_node() {
        let request = input(vec![node("a", 48), node("b", 48)]);
        let locked = vec![stage("a", 0, 20), stage("a", 20, 40)];

        assert_eq!(
            plan_locked_topology(&request, &locked),
            Err(TopologyPlanError::LockedDuplicateNode {
                node_id: "a".to_string(),
            })
        );
    }

    #[test]
    fn rejects_invalid_range() {
        let request = input(vec![node("a", 48), node("b", 48)]);
        let locked = vec![stage("a", 0, 20), stage("b", 20, 20)];

        assert_eq!(
            plan_locked_topology(&request, &locked),
            Err(TopologyPlanError::LockedInvalidRange {
                stage_index: 1,
                start: 20,
                end: 20,
            })
        );
    }

    #[test]
    fn rejects_non_contiguous_ranges() {
        let request = input(vec![node("a", 48), node("b", 48)]);
        let locked = vec![stage("a", 0, 19), stage("b", 20, 40)];

        assert_eq!(
            plan_locked_topology(&request, &locked),
            Err(TopologyPlanError::LockedNonContiguousRange {
                stage_index: 1,
                expected_start: 19,
                actual_start: 20,
            })
        );
    }

    #[test]
    fn rejects_incomplete_coverage() {
        let request = input(vec![node("a", 48), node("b", 48)]);
        let locked = vec![stage("a", 0, 20), stage("b", 20, 39)];

        assert_eq!(
            plan_locked_topology(&request, &locked),
            Err(TopologyPlanError::LockedIncompleteCoverage {
                actual_end: 39,
                layer_count: 40,
            })
        );
    }

    #[test]
    fn fails_when_pinned_stage_exceeds_node_capacity() {
        let request = input(vec![node("small", 12), node("large", 48)]);
        let locked = vec![stage("small", 0, 30), stage("large", 30, 40)];

        assert_eq!(
            plan_locked_topology(&request, &locked),
            Err(TopologyPlanError::LockedTopologyDoesNotFit {
                minimum_context: 65_536,
            })
        );
    }
}
