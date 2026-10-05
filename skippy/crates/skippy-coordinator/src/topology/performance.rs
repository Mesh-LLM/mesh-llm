//! Throughput-balanced layer placement.
//!
//! Memory-only placement fills the largest node first, which on a
//! heterogeneous split leaves the pipeline paced by whichever stage is slowest
//! while the faster stages idle. Under concurrent load a pipeline runs at the
//! rate of its slowest stage, so the throughput-optimal cut is the one that
//! minimises the maximum per-stage decode time.
//!
//! Decode is weight-bandwidth bound: a stage's per-token time is approximately
//! the weight bytes it holds divided by the rate its node streams weights. That
//! rate — `decode_bytes_per_second` on [`TopologyNode`](super::TopologyNode) —
//! is either estimated before load or measured from a running stage (weight
//! bytes resident on the stage divided by its observed per-token compute
//! time), so the same solver serves initial placement and runtime rebalancing.

use super::{TopologyStagePlan, UsableNode, sum_u64};

const NANOS_PER_SECOND: u128 = 1_000_000_000;

/// Per-stage decode estimate for a placement.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StageDecodeEstimate {
    pub stage_index: u32,
    pub node_id: String,
    pub layer_start: u32,
    pub layer_end: u32,
    /// Estimated compute time for one decode step on this stage.
    pub decode_nanos: u64,
}

/// Throughput view of a placement: per-stage times and the stage that paces
/// the pipeline.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ThroughputEstimate {
    pub stages: Vec<StageDecodeEstimate>,
    /// The largest per-stage time. At saturation the pipeline emits one token
    /// per lane every `bottleneck_decode_nanos`.
    pub bottleneck_decode_nanos: u64,
}

impl ThroughputEstimate {
    /// Fraction of the pipeline cycle each stage spends idle waiting for the
    /// bottleneck, in basis points (0 = never idle, 10_000 = always idle).
    pub fn idle_basis_points(&self) -> Vec<u32> {
        let bottleneck = u128::from(self.bottleneck_decode_nanos.max(1));
        self.stages
            .iter()
            .map(|stage| {
                let busy = u128::from(stage.decode_nanos).min(bottleneck);
                (((bottleneck - busy) * 10_000) / bottleneck) as u32
            })
            .collect()
    }
}

/// Estimate per-stage decode time for `stages` using each node's speed.
///
/// Returns `None` when any stage's node has no speed estimate, because a
/// partial estimate cannot identify the bottleneck.
pub(super) fn estimate_throughput(
    stages: &[TopologyStagePlan],
    nodes: &[UsableNode],
    layer_weights: &[u64],
) -> Option<ThroughputEstimate> {
    let mut estimates = Vec::with_capacity(stages.len());
    for stage in stages {
        let speed = node_speed(nodes, &stage.node_id)?;
        let bytes = sum_u64(&layer_weights[stage.layer_start as usize..stage.layer_end as usize]);
        estimates.push(StageDecodeEstimate {
            stage_index: stage.stage_index,
            node_id: stage.node_id.clone(),
            layer_start: stage.layer_start,
            layer_end: stage.layer_end,
            decode_nanos: decode_nanos(bytes, speed),
        });
    }
    let bottleneck_decode_nanos = estimates
        .iter()
        .map(|stage| stage.decode_nanos)
        .max()
        .unwrap_or_default();
    Some(ThroughputEstimate {
        stages: estimates,
        bottleneck_decode_nanos,
    })
}

/// Re-cut the layer boundaries of `stages` so the slowest stage is as fast as
/// possible, keeping the node order (and so stage 0) and each node's memory
/// limit.
///
/// Returns `None` when a node lacks a speed estimate or no feasible cut exists;
/// callers keep the memory-only placement in that case. The result is exact:
/// a dynamic program over contiguous partitions, `O(stages · layers²)`.
#[allow(
    clippy::needless_range_loop,
    reason = "the dynamic program indexes the cost, cut and prefix tables by the same layer boundary"
)]
pub(super) fn balance_stages(
    stages: &[TopologyStagePlan],
    nodes: &[UsableNode],
    layer_weights: &[u64],
    layer_required_bytes: &[u64],
) -> Option<Vec<TopologyStagePlan>> {
    let layer_count = layer_weights.len();
    let stage_count = stages.len();
    if stage_count == 0 || stage_count > layer_count || layer_required_bytes.len() != layer_count {
        return None;
    }
    let mut speeds = Vec::with_capacity(stage_count);
    let mut capacities = Vec::with_capacity(stage_count);
    for stage in stages {
        let node = nodes.iter().find(|node| node.node_id == stage.node_id)?;
        speeds.push(node.decode_bytes_per_second.filter(|speed| *speed > 0)?);
        capacities.push(node.usable_vram_bytes);
    }

    let weight_prefix = prefix_sums(layer_weights);
    let required_prefix = prefix_sums(layer_required_bytes);
    let range_weight = |start: usize, end: usize| weight_prefix[end] - weight_prefix[start];
    let range_required = |start: usize, end: usize| required_prefix[end] - required_prefix[start];

    // best[s][end]: minimal bottleneck placing layers 0..end on stages 0..=s,
    // with stage s ending at `end`. `cut[s][end]` records where stage s starts.
    let unreachable = u128::MAX;
    let mut best = vec![vec![unreachable; layer_count + 1]; stage_count];
    let mut cut = vec![vec![0usize; layer_count + 1]; stage_count];
    for end in 1..=layer_count - (stage_count - 1) {
        if range_required(0, end) <= u128::from(capacities[0]) {
            best[0][end] = stage_nanos(range_weight(0, end), speeds[0]);
        }
    }
    for stage in 1..stage_count {
        let later_stages = stage_count - 1 - stage;
        for end in stage + 1..=layer_count - later_stages {
            for start in stage..end {
                let previous = best[stage - 1][start];
                if previous == unreachable {
                    continue;
                }
                if range_required(start, end) > u128::from(capacities[stage]) {
                    continue;
                }
                let bottleneck = previous.max(stage_nanos(range_weight(start, end), speeds[stage]));
                // Ties go to the later start, which gives earlier stages more
                // layers; deterministic, and stable when stages are balanced.
                if bottleneck <= best[stage][end] {
                    best[stage][end] = bottleneck;
                    cut[stage][end] = start;
                }
            }
        }
    }
    if best[stage_count - 1][layer_count] == unreachable {
        return None;
    }

    let mut ends = vec![0usize; stage_count];
    let mut end = layer_count;
    for stage in (0..stage_count).rev() {
        ends[stage] = end;
        end = if stage == 0 { 0 } else { cut[stage][end] };
    }
    let mut start = 0usize;
    let balanced = stages
        .iter()
        .zip(ends)
        .map(|(stage, end)| {
            let plan = TopologyStagePlan {
                layer_start: start as u32,
                layer_end: end as u32,
                parameter_bytes: range_weight(start, end) as u64,
                ..stage.clone()
            };
            start = end;
            plan
        })
        .collect();
    Some(balanced)
}

fn node_speed(nodes: &[UsableNode], node_id: &str) -> Option<u64> {
    nodes
        .iter()
        .find(|node| node.node_id == node_id)?
        .decode_bytes_per_second
        .filter(|speed| *speed > 0)
}

fn decode_nanos(bytes: u64, bytes_per_second: u64) -> u64 {
    stage_nanos(u128::from(bytes), bytes_per_second).min(u128::from(u64::MAX)) as u64
}

fn stage_nanos(bytes: u128, bytes_per_second: u64) -> u128 {
    (bytes * NANOS_PER_SECOND).div_ceil(u128::from(bytes_per_second))
}

fn prefix_sums(values: &[u64]) -> Vec<u128> {
    let mut sums = Vec::with_capacity(values.len() + 1);
    sums.push(0u128);
    for value in values {
        let last = *sums.last().unwrap_or(&0);
        sums.push(last + u128::from(*value));
    }
    sums
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1_000_000_000;

    fn node(id: &str, usable: u64, speed: Option<u64>) -> UsableNode {
        UsableNode {
            node_id: id.to_string(),
            usable_vram_bytes: usable,
            stage_transfer_latency_ms: None,
            decode_bytes_per_second: speed,
        }
    }

    fn stage(index: u32, node_id: &str, start: u32, end: u32) -> TopologyStagePlan {
        TopologyStagePlan {
            stage_id: format!("stage-{index}"),
            stage_index: index,
            node_id: node_id.to_string(),
            layer_start: start,
            layer_end: end,
            parameter_bytes: 0,
        }
    }

    #[test]
    fn slow_first_stage_takes_fewer_layers() {
        // 36 equal layers; M1-class ~68 GB/s vs M4-class ~120 GB/s.
        let weights = vec![130_000_000u64; 36];
        let required = weights.clone();
        let nodes = [
            node("m1", 12 * GB, Some(68 * GB)),
            node("m4", 12 * GB, Some(120 * GB)),
        ];
        let stages = [stage(0, "m1", 0, 18), stage(1, "m4", 18, 36)];

        let balanced = balance_stages(&stages, &nodes, &weights, &required).unwrap();

        assert_eq!((balanced[0].layer_start, balanced[0].layer_end), (0, 13));
        assert_eq!((balanced[1].layer_start, balanced[1].layer_end), (13, 36));
        assert_eq!(balanced[0].node_id, "m1", "stage order is preserved");
        let before = estimate_throughput(&stages, &nodes, &weights).unwrap();
        let after = estimate_throughput(&balanced, &nodes, &weights).unwrap();
        assert!(after.bottleneck_decode_nanos < before.bottleneck_decode_nanos);
    }

    #[test]
    fn equal_nodes_split_evenly() {
        let weights = vec![100u64; 10];
        let nodes = [
            node("a", 10_000, Some(1_000)),
            node("b", 10_000, Some(1_000)),
        ];
        let stages = [stage(0, "a", 0, 9), stage(1, "b", 9, 10)];

        let balanced = balance_stages(&stages, &nodes, &weights, &weights).unwrap();

        assert_eq!(balanced[0].layer_end, 5);
    }

    #[test]
    fn memory_cap_binds_before_speed() {
        // The fast node would take 8 of 10 layers but only fits 6.
        let weights = vec![100u64; 10];
        let nodes = [
            node("slow", 10_000, Some(250)),
            node("fast", 600, Some(1_000)),
        ];
        let stages = [stage(0, "slow", 0, 5), stage(1, "fast", 5, 10)];

        let balanced = balance_stages(&stages, &nodes, &weights, &weights).unwrap();

        assert_eq!((balanced[1].layer_start, balanced[1].layer_end), (4, 10));
    }

    #[test]
    fn heavy_layers_count_by_bytes_not_by_index() {
        // The last layer carries the output head; bytes, not layer count, set cost.
        let mut weights = vec![100u64; 8];
        weights[7] = 800;
        let nodes = [
            node("a", 10_000, Some(1_000)),
            node("b", 10_000, Some(1_000)),
        ];
        let stages = [stage(0, "a", 0, 4), stage(1, "b", 4, 8)];

        let balanced = balance_stages(&stages, &nodes, &weights, &weights).unwrap();

        assert_eq!(
            balanced[0].layer_end, 7,
            "stage 1 keeps only the heavy output layer"
        );
    }

    #[test]
    fn missing_speed_leaves_placement_alone() {
        let weights = vec![100u64; 4];
        let nodes = [node("a", 10_000, Some(1_000)), node("b", 10_000, None)];
        let stages = [stage(0, "a", 0, 2), stage(1, "b", 2, 4)];

        assert!(balance_stages(&stages, &nodes, &weights, &weights).is_none());
        assert!(estimate_throughput(&stages, &nodes, &weights).is_none());
    }

    #[test]
    fn infeasible_memory_returns_none() {
        let weights = vec![100u64; 4];
        let nodes = [node("a", 150, Some(1_000)), node("b", 150, Some(1_000))];
        let stages = [stage(0, "a", 0, 2), stage(1, "b", 2, 4)];

        assert!(balance_stages(&stages, &nodes, &weights, &weights).is_none());
    }

    #[test]
    fn three_stages_every_stage_keeps_a_layer() {
        let weights = vec![100u64; 5];
        let nodes = [
            node("a", 10_000, Some(10_000)),
            node("b", 10_000, Some(1)),
            node("c", 10_000, Some(10_000)),
        ];
        let stages = [
            stage(0, "a", 0, 2),
            stage(1, "b", 2, 4),
            stage(2, "c", 4, 5),
        ];

        let balanced = balance_stages(&stages, &nodes, &weights, &weights).unwrap();

        assert_eq!(balanced[1].layer_end - balanced[1].layer_start, 1);
        assert!(
            balanced
                .iter()
                .all(|stage| stage.layer_end > stage.layer_start)
        );
    }

    #[test]
    fn idle_share_reports_the_waiting_stage() {
        let weights = vec![130_000_000u64; 36];
        let nodes = [
            node("m1", 12 * GB, Some(68 * GB)),
            node("m4", 12 * GB, Some(120 * GB)),
        ];
        let stages = [stage(0, "m1", 0, 18), stage(1, "m4", 18, 36)];

        let idle = estimate_throughput(&stages, &nodes, &weights)
            .unwrap()
            .idle_basis_points();

        assert_eq!(idle[0], 0, "the M1 stage is the bottleneck");
        assert!(
            (4_000..=4_500).contains(&idle[1]),
            "M4 idles ~43%: {}",
            idle[1]
        );
    }
}
