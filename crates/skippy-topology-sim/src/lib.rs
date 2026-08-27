//! Scenario-driven placement simulator for the performance-aware topology
//! planner (`docs/design/PERFORMANCE_AWARE_TOPOLOGY_PLANNER.md`).
//!
//! A scenario is a TOML file describing nodes, directed links, a model
//! package, and workload intent. The simulator feeds the scenario into
//! [`skippy_coordinator::topology::plan_topology`] and scores the resulting
//! plan with the same cost model the planner uses, so planner decisions can
//! be asserted against expectations ("a 2x-bandwidth node receives ~2x the
//! layers", "a slow link rejects the TPOT target") in CI without a cluster.
//!
//! The [`execution`](execution) layer adds a discrete pipeline model over a
//! chosen plan: per-stage service times from streamed weight bytes and
//! measured bandwidth, per-hop latency + activation transfer, serial vs
//! pipelined decode regimes, calibrated against `docs/BENCHMARKS.md`.

use serde::Deserialize;
use skippy_coordinator::topology::{
    TopologyEdge, TopologyNode, TopologyPlanningInput, plan_topology,
};

pub mod execution;

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
    /// Fraction of weight bytes actually streamed per token (MoE active
    /// experts / dense). 1.0 (default) = dense; the GLM-4.7-Flash anchor
    /// implies ~0.34. Only used by the execution layer, not placement.
    #[serde(default)]
    pub active_weight_fraction: Option<f64>,
    /// Calibration knob: fixed per-stage per-token software overhead
    /// (dispatch, kernel launch, sync) in milliseconds. Execution layer only.
    #[serde(default)]
    pub per_stage_overhead_ms: f64,
    /// Calibration knob: fixed per-hop per-token software overhead (QUIC
    /// stream, copies, scheduling) in milliseconds, on top of RTT and
    /// activation transfer. Execution layer only.
    #[serde(default)]
    pub per_hop_overhead_ms: f64,
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
    #[error(
        "unknown top-level scenario key `{key}` — links must be declared as `[links.\"a -> b\"]` tables, not top-level keys"
    )]
    UnknownTopLevelKey { key: String },
    #[error("malformed link key `{key}` — expected exactly one ` -> ` separating two node ids")]
    MalformedLinkKey { key: String },
}

impl Scenario {
    pub fn from_toml(input: &str) -> Result<Self, ScenarioError> {
        let value: toml::Value = toml::from_str(input)?;
        for key in value.as_table().into_iter().flat_map(|table| table.keys()) {
            if !matches!(key.as_str(), "nodes" | "links" | "model" | "workload") {
                return Err(ScenarioError::UnknownTopLevelKey { key: key.clone() });
            }
        }
        let scenario: Scenario = toml::from_str(input)?;
        // Validate link keys now so malformed edges fail at parse time,
        // not silently at planning time.
        for key in scenario.links.keys() {
            parse_link_key(key)?;
        }
        Ok(scenario)
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
                // Keys are validated at parse time; a malformed key here is
                // a programming error, not scenario content.
                let (source, target) = parse_link_key(key).expect("validated link key");
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
            active_weight_fraction_permil: ((self.model.active_weight_fraction.unwrap_or(1.0)
                * 1000.0)
                .round() as u32)
                .clamp(1, 1000),
            edges,
            activation_frame_bytes: self.model.activation_frame_bytes,
        }
    }

    /// Minimum observed RTT involving this node, used as the node's
    /// coordinator-RTT stand-in when links are present.
    fn node_latency_ms(&self, node_id: &str) -> Option<u32> {
        let mut best: Option<u32> = None;
        for (key, link) in &self.links {
            let (source, target) = parse_link_key(key).expect("validated link key");
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

fn parse_link_key(key: &str) -> Result<(String, String), ScenarioError> {
    // Accept exactly one "->" separating two non-empty node ids; anything
    // else is a malformed key that would silently produce unusable edges.
    let parts: Vec<&str> = key.split("->").collect();
    if parts.len() == 2 {
        let source = parts[0].trim();
        let target = parts[1].trim();
        if !source.is_empty()
            && !target.is_empty()
            && !source.contains(' ')
            && !target.contains(' ')
        {
            return Ok((source.to_string(), target.to_string()));
        }
    }
    Err(ScenarioError::MalformedLinkKey {
        key: key.to_string(),
    })
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

[links."alpha -> beta"]
rtt_ms = 2

[links."beta -> alpha"]
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
    fn link_tables_outside_links_fail_loudly() {
        // `["a -> b"]` at top level parses as a quoted *key* named
        // "a -> b", not a links entry — historically this silently
        // dropped the link. The schema now rejects unknown top-level
        // keys so misdeclared links cannot hide.
        let scenario = r#"
[nodes.alpha]
vram_bytes = 68719476736
sustained_mem_bandwidth_mib_per_s = 546000

[nodes.beta]
vram_bytes = 51539607552
sustained_mem_bandwidth_mib_per_s = 273000

["alpha -> beta"]
rtt_ms = 2

[model]
layer_count = 40
weight_bytes_per_layer = 1610612736
kv_bytes_per_token = 4096
native_context_length = 65536

[workload]
minimum_nodes = 2
"#;
        let error =
            Scenario::from_toml(scenario).expect_err("misdeclared top-level link must be rejected");
        assert!(error.to_string().contains("unknown top-level scenario key"));
    }

    #[test]
    fn malformed_link_keys_fail_loudly() {
        for key in ["alpha", "alpha -> beta -> gamma", " -> beta", "alpha -> "] {
            let scenario = format!(
                "[nodes.alpha]\nvram_bytes = 68719476736\n\
                 [nodes.beta]\nvram_bytes = 51539607552\n\
                 [links.\"{key}\"]\nrtt_ms = 2\n\
                 [model]\nlayer_count = 40\nweight_bytes_per_layer = 1610612736\n\
                 kv_bytes_per_token = 4096\nnative_context_length = 65536\n\
                 [workload]\nminimum_nodes = 2\n"
            );
            let error = Scenario::from_toml(&scenario)
                .err()
                .unwrap_or_else(|| panic!("malformed link key `{key}` must be rejected"));
            assert!(
                error.to_string().contains("malformed link key"),
                "key `{key}`: {error}"
            );
        }
    }

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
