//! Scenario-driven placement simulator for the performance-aware topology
//! planner (`docs/design/PERFORMANCE_AWARE_TOPOLOGY_PLANNER.md`).
//!
//! A scenario is a TOML file describing nodes, directed links, a model
//! package, and workload intent. The simulator feeds the scenario into
//! [`skippy_coordinator::topology::plan_topology`] and scores the resulting
//! plan with the same cost model the planner uses, so planner decisions can
//! be asserted against expectations ("a 2x-bandwidth node receives ~2x the
//! layers", "a slow link rejects the TPOT target") in CI without a cluster.

use serde::Deserialize;
use skippy_coordinator::topology::{
    TopologyEdge, TopologyNode, TopologyPlanningInput, plan_topology,
};

/// One candidate node in a scenario.
#[derive(Clone, Debug, Deserialize)]
pub struct ScenarioNode {
    pub vram_bytes: u64,
    /// Sustained memory bandwidth in MiB/s (`None` = unreported signal).
    #[serde(default)]
    pub sustained_mem_bandwidth_mib_per_s: Option<u32>,
    /// Sustained fp16 compute in GFLOP/s (`None` = unreported signal).
    #[serde(default)]
    pub sustained_compute_gflop_per_s: Option<u32>,
}

/// One directed link between scenario nodes. Keys are `"<source> -> <target>"`.
#[derive(Clone, Debug, Deserialize)]
pub struct ScenarioLink {
    pub rtt_ms: u32,
    #[serde(default)]
    pub large_frame_mib_per_s: Option<u32>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ScenarioModel {
    pub layer_count: u32,
    pub weight_bytes_per_layer: u64,
    pub kv_bytes_per_token: u64,
    pub native_context_length: u32,
    #[serde(default)]
    pub activation_frame_bytes: u64,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct ScenarioWorkload {
    #[serde(default)]
    pub minimum_nodes: Option<usize>,
    #[serde(default)]
    pub target_decode_tpot_ms: Option<u32>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Scenario {
    pub nodes: std::collections::BTreeMap<String, ScenarioNode>,
    #[serde(default)]
    pub links: std::collections::BTreeMap<String, ScenarioLink>,
    pub model: ScenarioModel,
    #[serde(default)]
    pub workload: ScenarioWorkload,
}

#[derive(Debug, thiserror::Error)]
pub enum ScenarioError {
    #[error("scenario parse error: {0}")]
    Parse(#[from] toml::de::Error),
}

impl Scenario {
    pub fn from_toml(input: &str) -> Result<Self, ScenarioError> {
        Ok(toml::from_str(input)?)
    }

    /// Build the coordinator planning input for this scenario.
    pub fn planning_input(&self) -> TopologyPlanningInput {
        let nodes = self
            .nodes
            .iter()
            .map(|(id, node)| TopologyNode {
                node_id: id.clone(),
                detected_vram_bytes: node.vram_bytes,
                max_vram_bytes: None,
                runtime_headroom_bytes: node.vram_bytes / 10,
                stage_transfer_latency_ms: self.node_latency_ms(id),
                sustained_mem_bandwidth_mib_per_s: node.sustained_mem_bandwidth_mib_per_s,
                sustained_compute_gflop_per_s: node.sustained_compute_gflop_per_s,
            })
            .collect::<Vec<_>>();
        let edges = self
            .links
            .iter()
            .map(|(key, link)| {
                let (source, target) = parse_link_key(key);
                TopologyEdge {
                    source_node_id: source,
                    target_node_id: target,
                    rtt_ms: link.rtt_ms,
                    large_frame_mib_per_s: link.large_frame_mib_per_s,
                }
            })
            .collect::<Vec<_>>();
        TopologyPlanningInput {
            native_context_length: self.model.native_context_length,
            layer_count: self.model.layer_count,
            model_weight_bytes: self.model.weight_bytes_per_layer
                * u64::from(self.model.layer_count),
            layer_weight_bytes: Vec::new(),
            kv_bytes_per_token: self.model.kv_bytes_per_token,
            recurrent_bytes_per_sequence_by_layer: Vec::new(),
            reserved_sequence_ids: 16,
            minimum_nodes: self.workload.minimum_nodes.unwrap_or(1),
            nodes,
            context_length_override: None,
            parallel_lanes_override: None,
            target_decode_tpot_ms: self.workload.target_decode_tpot_ms,
            edges,
            activation_frame_bytes: self.model.activation_frame_bytes,
        }
    }

    /// Minimum observed RTT involving this node, used as the node's
    /// coordinator-RTT stand-in when links are present.
    fn node_latency_ms(&self, node_id: &str) -> Option<u32> {
        let mut best: Option<u32> = None;
        for (key, link) in &self.links {
            let (source, target) = parse_link_key(key);
            if source == node_id || target == node_id {
                best = Some(best.map_or(link.rtt_ms, |current| current.min(link.rtt_ms)));
            }
        }
        best
    }

    /// Plan and score the scenario, returning the chosen plan plus modeled
    /// per-stage service times for assertions.
    pub fn plan(&self) -> Result<skippy_coordinator::topology::TopologyPlan, String> {
        plan_topology(&self.planning_input()).map_err(|error| error.to_string())
    }
}

fn parse_link_key(key: &str) -> (String, String) {
    let mut parts = key.split("->");
    let source = parts.next().unwrap_or_default().trim().to_string();
    let target = parts.next().unwrap_or_default().trim().to_string();
    (source, target)
}

#[cfg(test)]
mod tests {
    use super::*;

    const HETEROGENEOUS_PAIR: &str = r#"
[nodes.alpha]
vram_bytes = 68719476736            # 64 GiB
sustained_mem_bandwidth_mib_per_s = 546000
sustained_compute_gflop_per_s = 34000

[nodes.beta]
vram_bytes = 51539607552
sustained_mem_bandwidth_mib_per_s = 273000
sustained_compute_gflop_per_s = 17000

["alpha -> beta"]
rtt_ms = 2

["beta -> alpha"]
rtt_ms = 2

[model]
layer_count = 40
weight_bytes_per_layer = 1610612736  # 1.5 GiB
kv_bytes_per_token = 4096
native_context_length = 65536

[workload]
minimum_nodes = 2
target_decode_tpot_ms = 33
"#;

    #[test]
    fn heterogeneous_bandwidth_pair_proportions_layers() {
        let scenario = Scenario::from_toml(HETEROGENEOUS_PAIR).expect("scenario");
        let plan = scenario.plan().expect("plan");
        assert_eq!(plan.stages.len(), 2);
        let alpha = plan
            .stages
            .iter()
            .find(|stage| stage.node_id == "alpha")
            .expect("alpha stage");
        let beta = plan
            .stages
            .iter()
            .find(|stage| stage.node_id == "beta")
            .expect("beta stage");
        let alpha_layers = alpha.layer_end - alpha.layer_start;
        let beta_layers = beta.layer_end - beta.layer_start;
        assert!(
            alpha_layers >= 2 * beta_layers - 2,
            "2x bandwidth should earn ~2x layers: alpha={alpha_layers} beta={beta_layers}"
        );
    }

    #[test]
    fn missing_signal_node_keeps_capacity_only_placement() {
        let scenario = Scenario::from_toml(HETEROGENEOUS_PAIR).expect("scenario");
        let mut input = scenario.planning_input();
        input.nodes[0].sustained_mem_bandwidth_mib_per_s = None;
        let plan = plan_topology(&input).expect("plan");
        let alpha = plan
            .stages
            .iter()
            .find(|stage| stage.node_id == "alpha")
            .expect("alpha stage");
        let beta = plan
            .stages
            .iter()
            .find(|stage| stage.node_id == "beta")
            .expect("beta stage");
        // Without signals the capacity-greedy walk fills the first node to
        // its memory ceiling (~38 layers) and hands the remainder (~2) to
        // the second — very different from the perf-aware ~2:1 split. What
        // must hold: full coverage and non-empty stages; and the split must
        // NOT match perf proportions, proving the fallback engaged.
        let alpha_layers = alpha.layer_end - alpha.layer_start;
        let beta_layers = beta.layer_end - beta.layer_start;
        assert_eq!(alpha_layers + beta_layers, 40, "all layers placed");
        assert!(alpha_layers > 0 && beta_layers > 0);
        assert!(
            beta_layers * 2 < alpha_layers,
            "capacity-only fallback packs the first node instead of balancing: alpha={alpha_layers} beta={beta_layers}"
        );
    }
}
