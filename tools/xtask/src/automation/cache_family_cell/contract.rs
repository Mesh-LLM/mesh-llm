use crate::{
    automation::cache_family_measure::contract::{Cohort, Input as Measurement},
    command::DynResult,
};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::PathBuf};
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub(super) enum Host {
    NativeBaseline,
    SkippyOld,
    SkippyNew,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Input {
    pub schema_version: u64,
    pub host: Host,
    pub binary: PathBuf,
    pub binary_sha256: String,
    /// Caller-declared build source provenance, not derivable from a binary hash.
    pub source_commit: String,
    pub native_build: PathBuf,
    pub native_build_sha256: String,
    pub model: PathBuf,
    pub model_sha256: String,
    pub model_id: String,
    pub layer_end: u32,
    pub ctx_size: u32,
    pub lane_count: u32,
    pub n_gpu_layers: i32,
    pub port: u16,
    pub environment: BTreeMap<String, String>,
    pub worker: Measurement,
    pub startup_timeout_secs: u64,
    pub execution_timeout_secs: u64,
}
pub(super) fn digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}
pub(super) fn profile(name: &str, value: &str) -> bool {
    let explicit = [
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
    ];
    let denied = [
        "AUTH",
        "TOKEN",
        "SECRET",
        "PASSWORD",
        "CREDENTIAL",
        "PATH",
        "DIR",
        "ROOT",
        "KEY",
    ];
    (explicit.contains(&name) || name.starts_with("GGML_"))
        && !denied.iter().any(|part| name.contains(part))
        && !value.is_empty()
        && value.len() <= 256
        && !value.chars().any(char::is_control)
}
impl Input {
    pub(super) fn validate(&self) -> DynResult<()> {
        self.worker.validate()?;
        let expected_url = format!(
            "http://127.0.0.1:{}{}",
            self.port,
            if self.host == Host::NativeBaseline {
                "/"
            } else {
                "/v1"
            }
        );
        if self.schema_version != 1
            || !self.binary.is_absolute()
            || !self.native_build.is_absolute()
            || !self.model.is_absolute()
            || ![
                &self.binary_sha256,
                &self.native_build_sha256,
                &self.model_sha256,
            ]
            .iter()
            .all(|v| digest(v))
            || ![40, 64].contains(&self.source_commit.len())
            || !self
                .source_commit
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
            || self.model_id.is_empty()
            || self.model_id.len() > 4096
            || self.model_id.chars().any(char::is_whitespace)
            || self.port == 0
            || [self.layer_end, self.ctx_size, self.lane_count].contains(&0)
            || self.lane_count > 256
            || self.n_gpu_layers < -1
            || self.worker.base_url != expected_url
            || !(1..=3600).contains(&self.startup_timeout_secs)
            || !(15..=86400).contains(&self.execution_timeout_secs)
            || self.startup_timeout_secs >= self.execution_timeout_secs
            || self.environment.len() > 64
            || self.environment.iter().any(|(k, v)| !profile(k, v))
        {
            return Err("invalid cache cell artifact/profile/workload/deadline".into());
        }
        match (self.host, self.worker.cohort) {
            (Host::NativeBaseline, Cohort::NativeSerial | Cohort::NativeConcurrent) => {}
            (Host::SkippyOld | Host::SkippyNew, Cohort::OpenaiConcurrent)
                if self.worker.model_id.as_ref() == Some(&self.model_id) => {}
            _ => return Err("cache cell host/protocol/model cohort mismatch".into()),
        }
        Ok(())
    }
    pub(super) fn config(&self) -> serde_json::Value {
        serde_json::json!({"run_id":"cache-family-serving-cell","topology_id":"cache-family-single-stage",
            "model_id":self.model_id,"model_path":self.model,"source_model_sha256":self.model_sha256,
            "stage_id":"stage-0","stage_index":0,"layer_start":0,"layer_end":self.layer_end,
            "ctx_size":self.ctx_size,"lane_count":self.lane_count,"n_gpu_layers":self.n_gpu_layers,
            "load_mode":"runtime-slice","execution_contract":"","bind_addr":"127.0.0.1:0",
            "upstream":null,"downstream":null,"kv_server":null})
    }
}
