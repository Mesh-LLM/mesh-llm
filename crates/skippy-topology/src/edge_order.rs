use serde::{Deserialize, Serialize};

use crate::{
    DiagnosticSeverity, NodePlacementSignal, NodeSpec, PlanDiagnostic, PlanReasonCode, TopologyPlan,
};

const UNKNOWN_EDGE_RTT_MS: u64 = 10_000;
const MAX_EXHAUSTIVE_STAGE_COUNT: usize = 8;

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct StageEdgeSignal {
    pub source_node_id: String,
    pub target_node_id: String,
    #[serde(default)]
    pub rtt_ms: Option<u32>,
    #[serde(default)]
    pub large_frame_bytes_per_sec: Option<u64>,
    #[serde(default)]
    pub direct_prediction_return_supported: bool,
}

pub(crate) fn order_pipeline_nodes(
    nodes: Vec<(usize, NodeSpec)>,
    placement_signals: &[NodePlacementSignal],
    edge_signals: &[StageEdgeSignal],
) -> Vec<(usize, NodeSpec)> {
    if nodes.len() < 2 || edge_signals.is_empty() {
        return nodes;
    }
    if nodes.len() <= MAX_EXHAUSTIVE_STAGE_COUNT {
        return best_exhaustive_order(nodes, placement_signals, edge_signals);
    }
    greedy_order(nodes, placement_signals, edge_signals)
}

pub(crate) fn append_edge_diagnostics(plan: &mut TopologyPlan, edge_signals: &[StageEdgeSignal]) {
    if edge_signals.is_empty() {
        return;
    }

    for window in plan.stages.windows(2) {
        let source = &window[0];
        let target = &window[1];
        let Some(edge) = edge_signal_for(edge_signals, &source.node_id, &target.node_id) else {
            continue;
        };
        let Some(rtt_ms) = edge.rtt_ms else {
            continue;
        };
        plan.diagnostics.push(PlanDiagnostic {
            severity: DiagnosticSeverity::Info,
            code: PlanReasonCode::NetworkPipelineCost,
            message: format!(
                "pipeline edge {} -> {} after {} has measured rtt {} ms",
                source.node_id, target.node_id, source.stage_id, rtt_ms
            ),
        });
    }
}

fn best_exhaustive_order(
    nodes: Vec<(usize, NodeSpec)>,
    placement_signals: &[NodePlacementSignal],
    edge_signals: &[StageEdgeSignal],
) -> Vec<(usize, NodeSpec)> {
    let mut best_order = nodes.clone();
    let mut best_score = order_score(&best_order, placement_signals, edge_signals);
    let mut candidate = nodes;
    permute(0, &mut candidate, &mut |order| {
        let score = order_score(order, placement_signals, edge_signals);
        if score < best_score {
            best_score = score;
            best_order = order.to_vec();
        }
    });
    best_order
}

fn greedy_order(
    nodes: Vec<(usize, NodeSpec)>,
    placement_signals: &[NodePlacementSignal],
    edge_signals: &[StageEdgeSignal],
) -> Vec<(usize, NodeSpec)> {
    let mut remaining = nodes;
    let mut best_start = 0;
    let mut best_cost = u64::MAX;
    for (index, (_, node)) in remaining.iter().enumerate() {
        let cost = node_latency_cost(node, placement_signals);
        if cost < best_cost {
            best_start = index;
            best_cost = cost;
        }
    }

    let mut ordered = vec![remaining.remove(best_start)];
    while !remaining.is_empty() {
        let source_id = &ordered.last().expect("ordered has a start").1.node_id;
        let mut best_next = 0;
        let mut best_next_cost = u64::MAX;
        for (index, (_, candidate)) in remaining.iter().enumerate() {
            let cost = edge_cost(
                source_id,
                &candidate.node_id,
                placement_signals,
                edge_signals,
            );
            if cost < best_next_cost {
                best_next = index;
                best_next_cost = cost;
            }
        }
        ordered.push(remaining.remove(best_next));
    }
    ordered
}

fn permute(
    start: usize,
    values: &mut [(usize, NodeSpec)],
    visit: &mut impl FnMut(&[(usize, NodeSpec)]),
) {
    if start == values.len() {
        visit(values);
        return;
    }
    for index in start..values.len() {
        values.swap(start, index);
        permute(start + 1, values, visit);
        values.swap(start, index);
    }
}

fn order_score(
    order: &[(usize, NodeSpec)],
    placement_signals: &[NodePlacementSignal],
    edge_signals: &[StageEdgeSignal],
) -> (u64, Vec<usize>) {
    let edge_cost = order
        .windows(2)
        .map(|window| {
            edge_cost(
                &window[0].1.node_id,
                &window[1].1.node_id,
                placement_signals,
                edge_signals,
            )
        })
        .sum();
    let stable_order = order.iter().map(|(index, _)| *index).collect();
    (edge_cost, stable_order)
}

fn edge_cost(
    source: &str,
    target: &str,
    placement_signals: &[NodePlacementSignal],
    edge_signals: &[StageEdgeSignal],
) -> u64 {
    if let Some(edge) = edge_signal_for(edge_signals, source, target) {
        return edge.rtt_ms.map(u64::from).unwrap_or(UNKNOWN_EDGE_RTT_MS);
    }
    node_latency_cost_by_id(source, placement_signals)
        .saturating_add(node_latency_cost_by_id(target, placement_signals))
        .max(UNKNOWN_EDGE_RTT_MS)
}

fn edge_signal_for<'a>(
    edge_signals: &'a [StageEdgeSignal],
    source: &str,
    target: &str,
) -> Option<&'a StageEdgeSignal> {
    edge_signals
        .iter()
        .find(|edge| edge.source_node_id == source && edge.target_node_id == target)
}

fn node_latency_cost(node: &NodeSpec, placement_signals: &[NodePlacementSignal]) -> u64 {
    node_latency_cost_by_id(&node.node_id, placement_signals)
}

fn node_latency_cost_by_id(node_id: &str, placement_signals: &[NodePlacementSignal]) -> u64 {
    placement_signals
        .iter()
        .find(|signal| signal.node_id == node_id)
        .and_then(|signal| signal.rtt_ms)
        .map(u64::from)
        .unwrap_or(UNKNOWN_EDGE_RTT_MS)
}
