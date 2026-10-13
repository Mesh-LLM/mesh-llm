use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Deserialize)]
pub(super) struct Document {
    pub config: Config,
    pub inputs: Inputs,
    pub builds: Vec<Build>,
    #[serde(default)]
    pub order: Vec<Order>,
    pub results: Vec<super::pooled_rows::ArmPass>,
    pub gates: Option<Gates>,
    pub completed_at: Option<String>,
}

#[derive(Deserialize)]
pub(super) struct Config {
    pub model: String,
    pub concurrency: Vec<u32>,
    pub warmup_turns: u64,
    #[serde(default = "all")]
    pub replay_mode: String,
    pub engine_config: Option<EngineConfig>,
    pub context_qualification: Option<String>,
}

fn all() -> String {
    "all".into()
}

#[derive(Deserialize)]
pub(super) struct EngineConfig {}

#[derive(Deserialize)]
pub(super) struct Inputs {
    pub kind: Option<String>,
    pub dataset: Option<Dataset>,
    pub manifest_sha256: String,
    pub cohorts: BTreeMap<String, Cohort>,
}

#[derive(Deserialize)]
pub(super) struct Dataset {
    pub revision: String,
}

#[derive(Deserialize)]
pub(super) struct Cohort {
    pub trajectory_count: u64,
    pub assistant_turns: u64,
    pub framework_trajectories: BTreeMap<String, u64>,
    pub framework_assistant_turns: BTreeMap<String, u64>,
}

#[derive(Deserialize)]
pub(super) struct Build {
    pub label: String,
    pub engine: Option<String>,
    pub version: Option<String>,
    #[serde(rename = "ref")]
    pub reference: Option<String>,
    pub commit: Option<String>,
}

#[derive(Deserialize)]
pub(super) struct Order {
    pub label: String,
}

#[derive(Deserialize)]
pub(super) struct Gates {
    #[serde(default = "evaluated")]
    pub evaluated: bool,
    pub passed: Option<bool>,
    #[serde(default)]
    pub checks: Vec<Check>,
    #[serde(default)]
    pub session_acceptance_failures: Vec<SessionFailure>,
}

fn evaluated() -> bool {
    true
}

#[derive(Deserialize)]
pub(super) struct Check {
    pub name: String,
    pub passed: bool,
    pub detail: String,
}

#[derive(Deserialize)]
pub(super) struct SessionFailure {
    pub passed: bool,
    #[serde(default)]
    #[serde(alias = "problems")]
    pub failures: Vec<String>,
}
