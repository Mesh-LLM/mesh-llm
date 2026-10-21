use super::{TopologyPlanningInput, TopologyStagePlan, UsableNode};
use std::time::Duration;

/// A fresh observation of outgoing payload acknowledgement latency on one path.
/// The caller owns freshness, admission and transport-selection validation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StageLink {
    pub source_node_id: String,
    pub destination_node_id: String,
    pub acknowledgement_latency: Duration,
}

pub(super) fn estimate_network_cost(
    input: &TopologyPlanningInput,
    nodes: &[UsableNode],
    stages: &[TopologyStagePlan],
) -> (Option<u32>, u64) {
    let fallback = nodes
        .iter()
        .filter_map(|node| node.stage_transfer_latency_ms)
        .max();
    if input.stage_links.is_empty() {
        let millis = fallback.map(|latency| latency.saturating_mul(nodes.len() as u32));
        return (millis, u64::from(millis.unwrap_or_default()) * 1_000_000);
    }
    // Legacy RTTs truncate to milliseconds; zero is not a free unmeasured path.
    let fallback = fallback.map(|millis| millis.max(1));
    if stages.len() < 2 {
        return (Some(0), 0);
    }
    let mut nanos = 0u128;
    for (source, destination) in stages
        .iter()
        .zip(stages.iter().cycle().skip(1))
        .take(stages.len())
    {
        let latency = input
            .stage_links
            .iter()
            .find(|link| {
                link.source_node_id == source.node_id
                    && link.destination_node_id == destination.node_id
                    && !link.acknowledgement_latency.is_zero()
            })
            .map(|link| link.acknowledgement_latency)
            .or_else(|| fallback.map(|latency| Duration::from_millis(u64::from(latency))));
        let Some(latency) = latency else {
            // An unmeasured edge must not make a partially measured candidate free.
            return (None, u64::MAX);
        };
        nanos = nanos.saturating_add(latency.as_nanos());
    }
    let millis = nanos.div_ceil(1_000_000).min(u128::from(u32::MAX)) as u32;
    (Some(millis), nanos.min(u128::from(u64::MAX)) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::{
        plan_topology,
        tests::{input, latency_node},
    };

    fn link(source: &str, destination: &str, micros: u64) -> StageLink {
        StageLink {
            source_node_id: source.into(),
            destination_node_id: destination.into(),
            acknowledgement_latency: Duration::from_micros(micros),
        }
    }

    #[test]
    fn chooses_measured_pair_with_submillisecond_precision_and_keeps_missing_edge_cost() {
        let mut request = input(vec![
            latency_node("a", 23, 10),
            latency_node("b", 23, 10),
            latency_node("c", 23, 10),
        ]);
        request.minimum_nodes = 2;
        request.stage_links = vec![link("a", "b", 900), link("a", "c", 100)];
        let plan = plan_topology(&request).unwrap();
        assert_eq!(
            plan.stages
                .iter()
                .map(|stage| stage.node_id.as_str())
                .collect::<Vec<_>>(),
            ["a", "c"]
        );
        assert_eq!(plan.estimated_decode_network_ms_per_token, Some(11));
        request.stage_links = vec![link("c", "a", 100), link("b", "a", 900)];
        let reversed = plan_topology(&request).unwrap();
        assert_eq!(
            reversed
                .stages
                .iter()
                .map(|stage| stage.node_id.as_str())
                .collect::<Vec<_>>(),
            ["a", "c"]
        );
    }

    #[test]
    fn prices_only_the_observed_direction_and_keeps_unknown_edges_unknown() {
        let mut request = input(vec![
            latency_node("a", 23, 10),
            latency_node("b", 23, 10),
            latency_node("c", 23, 10),
        ]);
        request.minimum_nodes = 3;
        request.stage_links = vec![link("a", "b", 100)];
        assert_eq!(
            plan_topology(&request)
                .unwrap()
                .estimated_decode_network_ms_per_token,
            Some(21)
        );
        request.stage_links = vec![link("b", "a", 100)];
        assert_eq!(
            plan_topology(&request)
                .unwrap()
                .estimated_decode_network_ms_per_token,
            Some(30)
        );
        for node in &mut request.nodes {
            node.stage_transfer_latency_ms = None;
        }
        assert_eq!(
            plan_topology(&request)
                .unwrap()
                .estimated_decode_network_ms_per_token,
            None
        );
    }

    #[test]
    fn rounded_zero_rtt_does_not_undercut_a_measured_submillisecond_link() {
        let mut request = input(vec![
            latency_node("a", 23, 0),
            latency_node("b", 23, 0),
            latency_node("c", 23, 0),
        ]);
        request.minimum_nodes = 2;
        request.stage_links = vec![link("a", "c", 150)];
        let plan = plan_topology(&request).unwrap();
        assert_eq!(
            plan.stages
                .iter()
                .map(|stage| stage.node_id.as_str())
                .collect::<Vec<_>>(),
            ["a", "c"]
        );
        assert_eq!(plan.estimated_decode_network_ms_per_token, Some(2));
        request.stage_links.clear();
        assert_eq!(
            plan_topology(&request)
                .unwrap()
                .estimated_decode_network_ms_per_token,
            Some(0)
        );
    }
}
