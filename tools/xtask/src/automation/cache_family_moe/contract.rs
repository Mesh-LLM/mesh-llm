use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::PathBuf;
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Input {
    pub schema_version: u64,
    pub inspector: PathBuf,
    pub inspector_sha256: String,
    pub inspector_source_commit: String,
    /// Existing correctness frontend inputs for current catalog families.
    pub cases: Vec<Case>,
    pub execution_seconds: u64,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Case {
    pub layer_end: u32,
    pub correctness: Value,
}
impl Input {
    pub(super) fn validate(&self) -> DynResult<()> {
        if self.schema_version != 1
            || !self.inspector.is_absolute()
            || self.inspector_sha256.len() != 64
            || !self
                .inspector_sha256
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
            || ![40, 64].contains(&self.inspector_source_commit.len())
            || !self
                .inspector_source_commit
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
            || self.cases.is_empty()
            || self.cases.len() > 14
            || !(30..=86400).contains(&self.execution_seconds)
            || self
                .cases
                .iter()
                .any(|c| c.layer_end < 3 || !c.correctness.is_object())
        {
            return Err("invalid bounded MoE supplied-input profile".into());
        }
        let mut keys = std::collections::BTreeSet::new();
        for case in &self.cases {
            let key = case.correctness["case_key"]
                .as_str()
                .ok_or("MoE case key absent")?;
            if !keys.insert(key) {
                return Err("duplicate MoE case".into());
            }
        }
        Ok(())
    }
    pub(super) fn admission(&self, case: &Case) -> DynResult<Value> {
        let profile = &case.correctness;
        Ok(
            serde_json::json!({"schema_version":1,"artifact":profile["artifact"],"toolkit_directories":profile.get("toolkit_directories").cloned().unwrap_or_else(||serde_json::json!({})),"host":"native-baseline","binary":self.inspector,"binary_sha256":self.inspector_sha256,"source_commit":self.inspector_source_commit,"native_build":profile["native_build"],"native_build_sha256":profile["native_build_sha256"],"model":profile["model"],"model_sha256":profile["model_sha256"],"model_id":profile["model_id"],"layer_end":case.layer_end,"ctx_size":profile["ctx_size"],"lane_count":profile["runtime_lane_count"],"n_gpu_layers":profile["n_gpu_layers"],"port":1,"environment":profile["settings"],"worker":{"schema_version":1,"cohort":"native-serial","base_url":"http://127.0.0.1:1/","model_id":null,"prompt":"identity admission only; never HTTP","requests":1,"concurrency":1,"output_tokens":1,"request_timeout_ms":1000,"execution_timeout_ms":1000},"startup_timeout_secs":1,"execution_timeout_secs":30}),
        )
    }
}
