//! Closed family/payload declarations; provided local bytes are admitted separately.
use crate::command::DynResult;
use serde::{Deserialize, Serialize};
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub(super) enum Topology {
    OneStage,
    SplitStage0,
    SplitMiddle,
    SplitFinal,
    PackageStage1,
}
impl Topology {
    pub(super) fn range(self, layers: u32) -> DynResult<(u32, u32, u32)> {
        if layers < 3 {
            return Err("cache topology requires at least three layers".into());
        }
        let first = layers / 3;
        let second = 2 * (layers / 3) + (2 * (layers % 3)) / 3;
        Ok(match self {
            Self::OneStage => (0, layers, 0),
            Self::SplitStage0 => (0, first, 0),
            Self::SplitMiddle => (first, second, 1),
            Self::SplitFinal => (second, layers, 2),
            Self::PackageStage1 => return Err("requires admitted package profile".into()),
        })
    }
}
pub(super) fn family(key: &str) -> DynResult<(&'static str, &'static str)> {
    Ok(match key {
        "qwen3_dense" => ("Qwen3 dense", "resident-kv"),
        "llama" => ("Llama", "resident-kv"),
        "deepseek2" => ("DeepSeek2", "resident-kv"),
        "deepseek3" => ("DeepSeek3", "resident-kv"),
        "glm47_flash" => ("GLM-4.7 Flash", "resident-kv"),
        "glm4" => ("GLM4", "resident-kv"),
        "gemma4_a4b" => ("Gemma4 A4B", "resident-kv"),
        "gemma4_e4b" => ("Gemma4 E4B", "resident-kv"),
        "gemma3" => ("Gemma3", "resident-kv"),
        "gemma2" => ("Gemma2", "resident-kv"),
        "falcon_h1" => ("Falcon-H1", "kv-recurrent"),
        "olmo" => ("OLMo", "resident-kv"),
        "minimax_m27" => ("MiniMax M2.7", "resident-kv"),
        "qwen3next" => ("Qwen3Next", "kv-recurrent"),
        _ => return Err("unknown current cache-family catalog key".into()),
    })
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Input {
    pub schema_version: u64,
    #[serde(default)]
    pub artifact: Option<super::artifact::Artifact>,
    pub case_key: String,
    pub model_id: String,
    pub correctness: std::path::PathBuf,
    pub correctness_sha256: String,
    pub stage_server: std::path::PathBuf,
    pub stage_server_sha256: String,
    pub model: std::path::PathBuf,
    pub model_sha256: String,
    pub native_build: std::path::PathBuf,
    pub native_build_sha256: String,
    pub ctx_size: u32,
    pub prefix_tokens: u32,
    pub cache_hit_repeats: u32,
    pub runtime_lane_count: u32,
    pub source_port: u16,
    pub restore_port: u16,
    pub n_gpu_layers: i32,
    pub prompt: Option<String>,
    pub topologies: Vec<Topology>,
    pub borrow_resident_hits: bool,
    pub cache_decoded_result_hits: bool,
    pub execution_seconds: u64,
    pub cell_seconds: u64,
    pub settings: std::collections::BTreeMap<String, String>,
    #[serde(default)]
    pub toolkit_directories:
        std::collections::BTreeMap<String, crate::automation::cache_family_profile::Toolkit>,
}
impl Input {
    pub(super) fn range(&self, topology: Topology, layers: u32) -> DynResult<(u32, u32, u32)> {
        if matches!(topology, Topology::PackageStage1) {
            if self.case_key == "deepseek3"
                && layers == 61
                && self
                    .artifact
                    .as_ref()
                    .is_some_and(|a| a.kind == super::artifact::Kind::LayerPackage)
            {
                return Ok((3, 4, 1));
            }
            return Err("package range requires exact admitted DeepSeek3 profile".into());
        }
        topology.range(layers)
    }
    pub(super) fn validate(&self) -> DynResult<()> {
        family(&self.case_key)?;
        if let Some(artifact) = &self.artifact {
            artifact.validate(self)?;
        } else if self.case_key == "deepseek3" {
            return Err("DeepSeek3 requires explicit package admission".into());
        }
        let digest = |s: &str| {
            s.len() == 64
                && s.bytes()
                    .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
        };
        if self.schema_version != 1
            || self.model_id.is_empty()
            || self.model_id.len() > 4096
            || self.model_id.contains(['\0', '\n', '\r'])
            || self.source_port == 0
            || self.restore_port == 0
            || self.source_port == self.restore_port
            || self.ctx_size == 0
            || self.prefix_tokens == 0
            || self
                .prefix_tokens
                .checked_add(if self.case_key == "deepseek3" { 1 } else { 128 })
                .is_none_or(|v| v > self.ctx_size)
            || !(1..=1024).contains(&self.cache_hit_repeats)
            || !(1..=1024).contains(&self.runtime_lane_count)
            || self.n_gpu_layers < -1
            || self.topologies.is_empty()
            || self.topologies.len() > 4
            || !(4..=86400).contains(&self.execution_seconds)
            || !(1..=3600).contains(&self.cell_seconds)
            || [
                &self.correctness,
                &self.stage_server,
                &self.model,
                &self.native_build,
            ]
            .iter()
            .any(|p| !p.is_absolute())
            || [
                &self.correctness_sha256,
                &self.stage_server_sha256,
                &self.model_sha256,
                &self.native_build_sha256,
            ]
            .iter()
            .any(|s| !digest(s))
            || self
                .prompt
                .as_ref()
                .is_some_and(|s| s.len() > 65536 || s.contains('\0'))
        {
            return Err("invalid cache correctness input/pins/budgets".into());
        }
        let mut ranges = std::collections::BTreeSet::new();
        for t in &self.topologies {
            if !ranges.insert(self.range(*t, if self.case_key == "deepseek3" { 61 } else { 6 })?) {
                return Err("duplicate cache topology".into());
            }
        }
        crate::automation::cache_family_profile::validate(
            &self.settings,
            &self.toolkit_directories,
        )?;
        Ok(())
    }
}
