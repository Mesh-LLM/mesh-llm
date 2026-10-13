use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct Config {
    pub model: String,
    pub backend: String,
    pub lifecycle_cohort: String,
    pub required_sources: Vec<String>,
    pub concurrency: Vec<usize>,
    pub prompt_min: u64,
    pub prompt_max: u64,
    pub cold_samples: usize,
    pub restart_samples: usize,
    pub identical_repeats: usize,
    pub max_output_tokens: u64,
    pub disk_budget: String,
    pub minimum_free: String,
    pub low_space_disk_budget: String,
    pub low_space_minimum_free: String,
    pub max_l3_ttft_ratio: f64,
    pub max_payload_write_amplification: f64,
    pub max_decode_p99_regression_pct: f64,
}
impl Config {
    pub fn validate(&self, execution: bool) -> DynResult<()> {
        if self.model.is_empty()
            || self.backend.is_empty()
            || self.lifecycle_cohort.is_empty()
            || self.prompt_min == 0
            || self.prompt_max < self.prompt_min
            || self.cold_samples == 0
            || self.restart_samples == 0
            || self.identical_repeats == 0
            || self.identical_repeats > 4096
            || self.max_output_tokens == 0
            || self.concurrency.is_empty()
            || self.concurrency.iter().any(|c| *c == 0 || *c > 4096)
            || self.concurrency.iter().collect::<BTreeSet<_>>().len() != self.concurrency.len()
            || self.required_sources.iter().collect::<BTreeSet<_>>().len()
                != self.required_sources.len()
            || self.required_sources.iter().any(String::is_empty)
            || (execution && self.required_sources.len() < 3)
        {
            return Err(
                "invalid disk-L3 cohort, sample, prompt or concurrency configuration".into(),
            );
        }
        if !self.max_l3_ttft_ratio.is_finite()
            || !(0.0..=1.0).contains(&self.max_l3_ttft_ratio)
            || self.max_l3_ttft_ratio == 0.0
            || !self.max_payload_write_amplification.is_finite()
            || self.max_payload_write_amplification < 1.0
            || !self.max_decode_p99_regression_pct.is_finite()
            || self.max_decode_p99_regression_pct < 0.0
        {
            return Err("invalid disk-L3 gate threshold".into());
        }
        if self.disk_budget != "auto" {
            size(&self.disk_budget)?;
        }
        for value in [
            &self.minimum_free,
            &self.low_space_disk_budget,
            &self.low_space_minimum_free,
        ] {
            size(value)?;
        }
        Ok(())
    }
}
fn size(value: &str) -> DynResult<u64> {
    for (suffix, power) in [("KiB", 1), ("MiB", 2), ("GiB", 3), ("TiB", 4)] {
        if let Some(number) = value.strip_suffix(suffix) {
            if number.starts_with('0') || !number.bytes().all(|b| b.is_ascii_digit()) {
                break;
            }
            return number
                .parse::<u64>()?
                .checked_mul(1024_u64.pow(power))
                .filter(|v| *v > 0)
                .ok_or_else(|| "disk size overflow".into());
        }
    }
    Err("disk size must be a positive IEC quantity".into())
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct Activity {
    pub fills: u64,
    pub hits: u64,
    pub misses: u64,
    pub writes: u64,
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub evictions: u64,
    pub corrupt_entries: u64,
}
impl Activity {
    pub fn delta(&self, before: &Self) -> DynResult<Self> {
        let subtract = |after: u64, before: u64| {
            after
                .checked_sub(before)
                .ok_or("disk counter reset within one server session")
        };
        Ok(Self {
            fills: subtract(self.fills, before.fills)?,
            hits: subtract(self.hits, before.hits)?,
            misses: subtract(self.misses, before.misses)?,
            writes: subtract(self.writes, before.writes)?,
            bytes_read: subtract(self.bytes_read, before.bytes_read)?,
            bytes_written: subtract(self.bytes_written, before.bytes_written)?,
            evictions: subtract(self.evictions, before.evictions)?,
            corrupt_entries: subtract(self.corrupt_entries, before.corrupt_entries)?,
        })
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct Usage {
    pub manifests: u64,
    pub used_bytes: u64,
    pub reserved_inflight_bytes: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct Effective {
    pub state: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct Status {
    pub version: u32,
    pub effective: Effective,
    pub usage: Option<Usage>,
    pub activity: Option<Activity>,
    #[serde(flatten)]
    pub evidence: BTreeMap<String, serde_json::Value>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct Operation {
    pub status: Status,
    pub freed_bytes: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct Request {
    pub request_id: String,
    pub session_id: String,
    pub source_dataset: String,
    pub assistant_turn: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttft_seconds: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(flatten)]
    pub evidence: BTreeMap<String, serde_json::Value>,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub(super) struct Phase {
    pub requests: Vec<Request>,
    pub status_before: Option<Status>,
    pub status_after: Option<Status>,
    pub activity_delta: Option<Activity>,
    #[serde(default)]
    pub activity_deltas: Vec<Activity>,
    pub prune: Option<Operation>,
    pub clear: Option<Operation>,
    pub final_clear: Option<Operation>,
    pub summary: Option<Summary>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct Summary {
    pub failed_requests: usize,
    pub decode_inter_token_p99_seconds: Option<f64>,
    pub content_sha256_by_request: BTreeMap<String, String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct Run {
    pub schema_version: u32,
    pub kind: String,
    pub config: Config,
    pub build: serde_json::Value,
    pub inputs: serde_json::Value,
    pub phases: BTreeMap<String, Phase>,
    pub completed_at: Option<String>,
    pub gates: Option<Gates>,
    #[serde(flatten)]
    pub evidence: BTreeMap<String, serde_json::Value>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct Check {
    pub name: String,
    pub passed: bool,
    pub detail: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct Gates {
    pub evaluated: bool,
    pub passed: bool,
    pub checks: Vec<Check>,
    pub cold_ttft_p50_seconds: Option<f64>,
    pub restart_l3_ttft_p50_seconds: Option<f64>,
    pub restart_l3_ttft_ratio: Option<f64>,
}
