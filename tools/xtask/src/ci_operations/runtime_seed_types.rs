use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub(super) const IMAGE: &str = "ghcr.io/mesh-llm/mesh-llm-cuda-runner@sha256:f499b79bc52dc7492d57397fdbec9f890c6f6bb1d8c1fcde9c1c97d45c0541a7";
pub(super) const EPOCH: &str =
    "mesh-llm-cuda-runner-sha256-f499b79bc52dc7492d57397fdbec9f890c6f6bb1d8c1fcde9c1c97d45c0541a7";
pub(super) const PREFIX: &str =
    "mesh-llm-sccache-seed-linux-x86_64-img-f499b79b-epoch-f499b79b-v3-";
pub(super) const BUILD: &str = ".deps/llama.cpp/build-stage-abi-dynamic-cpu";

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(super) enum Arm {
    Cold,
    Warm,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(super) struct Cache {
    pub id: u64,
    pub key: String,
    pub version: String,
    #[serde(rename = "ref")]
    pub reference: String,
    pub size_in_bytes: u64,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

impl Cache {
    pub fn validate(&self) -> DynResult<()> {
        if self.id == 0
            || self.id > (1 << 53) - 1
            || !hexadecimal(&self.version, 64)
            || self.reference != "refs/heads/main"
            || self.size_in_bytes == 0
            || self.size_in_bytes > 2 * 1024 * 1024 * 1024
        {
            return Err("seed metadata mismatch".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(super) struct Context {
    pub schema: u32,
    pub source: String,
    pub image: String,
    pub epoch: String,
    pub cache: Cache,
    pub pair: u8,
    pub arm: Arm,
    pub run_id: String,
    pub run_attempt: String,
    pub build_dir: String,
    pub initial_outputs_absent: bool,
    pub host_cpu: BTreeMap<String, String>,
    pub runner_class: String,
    pub kernel: String,
    pub runner_image_os: Option<String>,
    pub runner_image_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub restore_seconds: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_after_restore: Option<Cache>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_hit: Option<bool>,
}

#[derive(Debug, Deserialize, Serialize)]
pub(super) struct ResultEvidence {
    #[serde(flatten)]
    pub context: Context,
    pub action_seconds: f64,
    pub total_seconds: f64,
    pub native_preparation_and_build_seconds: Option<f64>,
    pub packaging_seconds: Option<f64>,
    pub phase_timing_note: String,
    pub classification: String,
    pub warm_floor_passed: bool,
    pub hit_rate: f64,
    pub native_hits: u64,
    pub native_cacheable_requests: u64,
    pub language_hits: BTreeMap<String, u64>,
    pub language_misses: BTreeMap<String, u64>,
    pub assembler_hits: u64,
    pub assembler_misses: u64,
    pub manifest_and_checksum_hashes: BTreeMap<String, String>,
    pub eligibility_changed: bool,
    pub verified: bool,
}

#[derive(Deserialize, Serialize)]
pub(super) struct Timestamp {
    pub monotonic: f64,
}

pub(super) fn hexadecimal(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

pub(super) fn elapsed(now: f64, start: f64) -> DynResult<f64> {
    if !now.is_finite() || !start.is_finite() || start < 0.0 || now < start {
        return Err("invalid timing".into());
    }
    Ok(now - start)
}
