use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
};
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Pin {
    pub path: PathBuf,
    pub sha256: String,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Input {
    pub schema_version: u64,
    pub cache_root: PathBuf,
    #[serde(default)]
    pub cases: Vec<String>,
    #[serde(default)]
    pub use_cases: Vec<String>,
    pub corpus: Option<Pin>,
    #[serde(default)]
    pub prefix_sweep: Vec<u32>,
    pub prefix_tokens: Option<u32>,
    pub n_gpu_layers: Option<i32>,
    pub cache_hit_repeats: Option<u32>,
    pub runtime_lane_count: Option<u32>,
    pub serving_ctx_size: Option<u32>,
    pub concurrency: Vec<u32>,
    pub concurrent_requests: u32,
    pub concurrent_output_tokens: u32,
    pub llama_parallel: u32,
    pub llama_repeats: u32,
    pub ttft_slo_ms: u32,
    pub tpot_slo_ms: u32,
    pub skip_llama_server: bool,
    pub old_server: Option<Pin>,
    pub new_server: Option<Pin>,
    #[serde(default)]
    pub model_sha256: BTreeMap<String, String>,
}
pub(super) fn key(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 128
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-'))
}
pub(super) fn digest(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}
fn unique<T: Ord>(values: &[T]) -> bool {
    values.iter().collect::<BTreeSet<_>>().len() == values.len()
}
impl Input {
    pub(super) fn validate(&self) -> DynResult<()> {
        if self.schema_version != 1
            || !self.cache_root.is_absolute()
            || self.cases.len() > 14
            || self.use_cases.len() > 128
            || !unique(&self.cases)
            || !unique(&self.use_cases)
            || self.cases.iter().chain(&self.use_cases).any(|s| !key(s))
            || self.prefix_sweep.len() > 32
            || !unique(&self.prefix_sweep)
            || self.prefix_sweep.contains(&0)
            || self.prefix_tokens == Some(0)
            || self.n_gpu_layers.is_some_and(|n| n < -1)
            || self.cache_hit_repeats.is_some_and(|n| n == 0 || n > 1024)
            || self.runtime_lane_count.is_some_and(|n| n == 0 || n > 1024)
            || self.serving_ctx_size == Some(0)
            || self.concurrency.is_empty()
            || self.concurrency.len() > 32
            || !unique(&self.concurrency)
            || self.concurrency.iter().any(|n| *n == 0 || *n > 1024)
            || self.concurrent_requests == 0
            || self.concurrent_requests > 65536
            || self.concurrent_output_tokens == 0
            || self.concurrent_output_tokens > 65536
            || self.llama_parallel == 0
            || self.llama_parallel > 1024
            || self.llama_repeats == 0
            || self.llama_repeats > 1024
            || self.ttft_slo_ms == 0
            || self.tpot_slo_ms == 0
            || self.old_server.is_some() != self.new_server.is_some()
            || self.corpus.is_some() == self.use_cases.is_empty()
            || self.use_cases.iter().any(|s| s == "all") && self.use_cases.len() != 1
            || self.model_sha256.len() > 14
            || self.model_sha256.iter().any(|(k, v)| !key(k) || !digest(v))
        {
            return Err("invalid cache-family plan selection/profile".into());
        }
        for pin in [&self.corpus, &self.old_server, &self.new_server]
            .into_iter()
            .flatten()
        {
            if !pin.path.is_absolute() || !digest(&pin.sha256) {
                return Err("plan pins require absolute paths and SHA256".into());
            }
        }
        Ok(())
    }
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Case {
    pub key: String,
    pub family: String,
    pub model_id: String,
    pub snapshot_relative: String,
    pub payload: String,
    pub layer_end: u32,
    pub activation_width: u32,
    pub ctx_size: u32,
    pub n_gpu_layers: i32,
    pub prefix_tokens: u32,
    pub cache_hit_repeats: u32,
    pub stage_load_mode: String,
    pub state_layer_start: u32,
    pub state_layer_end: u32,
    pub state_stage_index: u32,
    pub resident_kv_bytes_per_token: Option<u64>,
    pub skip_llama_server_reason: Option<String>,
    pub revision: String,
    pub original_case_region_sha256: String,
}
#[derive(Clone, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Source {
    pub dataset: Option<String>,
    pub config: Option<String>,
    pub split: Option<String>,
    pub row_idx: Option<u64>,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct UseCase {
    pub key: String,
    pub label: String,
    pub prompt: String,
    #[serde(default = "prefix_default")]
    pub prefix_tokens: u32,
    #[serde(default)]
    pub source: Source,
}
fn prefix_default() -> u32 {
    128
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Corpus {
    pub version: u64,
    pub use_cases: Vec<UseCase>,
}
