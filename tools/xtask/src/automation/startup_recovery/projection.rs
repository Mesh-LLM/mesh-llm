use crate::process::retained::recovery::{Observation, Stage, Topology};
use serde::Deserialize;

#[derive(Deserialize, Default)]
pub(super) struct Status {
    #[serde(default)]
    pub token: String,
    #[serde(default)]
    pub node_id: String,
    #[serde(default)]
    pub peers: Vec<serde_json::Value>,
}
#[derive(Deserialize, Default)]
pub(super) struct Stages {
    #[serde(default)]
    pub topologies: Vec<WireTopology>,
}
#[derive(Deserialize)]
pub(super) struct WireTopology {
    #[serde(default)]
    pub run_id: String,
    #[serde(default)]
    pub stages: Vec<WireStage>,
}
#[derive(Deserialize)]
pub(super) struct WireStage {
    pub stage_index: u32,
    pub node_id: String,
}
#[derive(Deserialize, Default)]
pub(super) struct Models {
    #[serde(default)]
    pub data: Vec<Model>,
}
#[derive(Deserialize)]
pub(super) struct Model {
    pub id: String,
}

pub(super) fn observation(stages: Stages, models: Models) -> Observation {
    Observation {
        topologies: stages
            .topologies
            .into_iter()
            .map(|topology| Topology {
                run_id: topology.run_id,
                stages: topology
                    .stages
                    .into_iter()
                    .map(|stage| Stage {
                        index: stage.stage_index,
                        node_id: stage.node_id,
                    })
                    .collect(),
            })
            .collect(),
        model_count: models.data.len(),
    }
}
