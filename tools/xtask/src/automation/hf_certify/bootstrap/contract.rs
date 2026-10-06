use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(in crate::automation::hf_certify) struct Tool {
    pub name: String,
    pub path: PathBuf,
    pub sha256: String,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(in crate::automation::hf_certify) struct Input {
    pub schema_version: u32,
    pub mesh_commit: String,
    pub git_tree: String,
    pub llama_commit: String,
    pub upstream_file_sha256: String,
    pub image: String,
    pub native_profile: String,
    pub tools: Vec<Tool>,
    pub path_directories: Vec<PathBuf>,
    pub timeout_seconds: u64,
    pub cpu_plan_receipt_sha256: String,
    pub declared_estimate_usd: f64,
    pub max_cost_usd: f64,
}
pub(in crate::automation::hf_certify) fn hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
impl Input {
    pub(in crate::automation::hf_certify) fn validate(&self) -> DynResult<()> {
        if self.schema_version != 1
            || self.native_profile != "standalone-static-skippy-quantize-cpu"
            || !(30..=86400).contains(&self.timeout_seconds)
            || !hex(&self.mesh_commit, 40)
            || !hex(&self.git_tree, 40)
            || !hex(&self.llama_commit, 40)
            || !hex(&self.upstream_file_sha256, 64)
            || !hex(&self.cpu_plan_receipt_sha256, 64)
            || !self.declared_estimate_usd.is_finite()
            || self.declared_estimate_usd < 0.0
            || !self.max_cost_usd.is_finite()
            || self.max_cost_usd <= 0.0
            || self.declared_estimate_usd > self.max_cost_usd
        {
            return Err("bootstrap schema/source/profile/resource declaration refused".into());
        }
        let (name, pin) = self
            .image
            .rsplit_once("@sha256:")
            .ok_or("bootstrap requires digest-pinned image")?;
        if !hex(pin, 64)
            || name.is_empty()
            || name.len() > 256
            || !name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"/._-:".contains(&b))
        {
            return Err("bootstrap image reference refused".into());
        }
        let required = [
            "git", "just", "cargo", "rustc", "cmake", "c++", "ld.lld", "curl",
        ];
        if self.tools.len() != required.len()
            || required
                .iter()
                .any(|name| self.tools.iter().filter(|t| t.name == *name).count() != 1)
            || self
                .tools
                .iter()
                .any(|t| !t.path.is_absolute() || !hex(&t.sha256, 64))
            || self.path_directories.is_empty()
            || self.path_directories.len() > 16
            || self.path_directories.iter().any(|p| !p.is_absolute())
        {
            return Err(
                "bootstrap requires exact pinned local tool roster and absolute PATH directories"
                    .into(),
            );
        }
        Ok(())
    }
    pub(in crate::automation::hf_certify) fn tool(&self, name: &str) -> DynResult<&Tool> {
        self.tools
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| "bootstrap tool absent".into())
    }
}
