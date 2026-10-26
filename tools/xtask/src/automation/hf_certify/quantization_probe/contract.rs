//! Closed finite probe for a supplied quantizer; declarations do not approve a Job.
use super::super::admission::Artifact;
use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use std::path::{Component, Path, PathBuf};
#[derive(Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub(in crate::automation::hf_certify) enum ToolKind {
    SuppliedWindowQuantizer,
    CurrentNativeQuantizer,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub(in crate::automation::hf_certify) enum Workflow {
    Quantize,
    QuantizeAndPackage,
}
impl Workflow {
    pub(super) fn job_allowance_seconds(&self) -> u64 {
        match self {
            Self::Quantize => 259200,
            Self::QuantizeAndPackage => 345600,
        }
    }
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(in crate::automation::hf_certify) struct Input {
    pub schema_version: u32,
    pub workflow: Workflow,
    pub tool_kind: ToolKind,
    pub tool: Artifact,
    pub tool_source: Artifact,
    pub source_revision: String,
    pub profile_version: String,
    pub runtime: Artifact,
    pub loader: Artifact,
    pub manifest: Artifact,
    pub tensor_recipe: Artifact,
    pub source_root: PathBuf,
    pub target_root: PathBuf,
    pub work_root: PathBuf,
    pub source_prefix: String,
    pub target_prefix: String,
    pub basename: String,
    pub source_parts: Vec<Artifact>,
    pub golden_outputs: Vec<Artifact>,
    pub timeout_seconds: u64,
}
fn leaf(v: &str) -> bool {
    !v.is_empty()
        && v.len() <= 128
        && !matches!(v, "." | "..")
        && v.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
}
fn absolute(p: &Path) -> bool {
    p.is_absolute()
        && p.components()
            .all(|c| !matches!(c, Component::CurDir | Component::ParentDir))
}
fn hash(v: &str, n: usize) -> bool {
    v.len() == n
        && v.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
pub(super) fn disjoint(a: &Path, b: &Path) -> bool {
    !a.starts_with(b) && !b.starts_with(a)
}
impl Input {
    pub(super) fn validate(&self) -> DynResult<()> {
        if self.tool_kind == ToolKind::CurrentNativeQuantizer {
            return Err("current native quantizer refuses memory cap/partial windows; no full-model fallback".into());
        }
        if self.schema_version != 1
            || !hash(&self.source_revision, 40)
            || !leaf(&self.profile_version)
            || !leaf(&self.source_prefix)
            || !leaf(&self.target_prefix)
            || !leaf(&self.basename)
            || self.source_parts.len() != 2
            || self.golden_outputs.len() != 2
            || !(5..=600).contains(&self.timeout_seconds)
            || ![&self.source_root, &self.target_root, &self.work_root]
                .iter()
                .all(|p| absolute(p))
            || !disjoint(&self.source_root, &self.target_root)
            || !disjoint(&self.source_root, &self.work_root)
            || !disjoint(&self.target_root, &self.work_root)
        {
            return Err("finite supplied quantizer profile/source/window bounds refused".into());
        }
        let mut paths = std::collections::BTreeSet::new();
        for a in [
            &self.tool,
            &self.tool_source,
            &self.runtime,
            &self.loader,
            &self.manifest,
            &self.tensor_recipe,
        ]
        .into_iter()
        .chain(self.source_parts.iter())
        .chain(self.golden_outputs.iter())
        {
            if !absolute(&a.path) || !hash(&a.sha256, 64) || !paths.insert(&a.path) {
                return Err("finite quantizer pin/path/repeated artifact refused".into());
            }
        }
        for (i, a) in self.source_parts.iter().enumerate() {
            if a.path
                != self.source_root.join(&self.source_prefix).join(format!(
                    "{}-{:05}-of-00002.gguf",
                    self.basename,
                    i + 1
                ))
            {
                return Err("complete finite source shard roster refused".into());
            }
        }
        for (i, a) in self.golden_outputs.iter().enumerate() {
            if a.path
                != self.target_root.join(&self.target_prefix).join(format!(
                    "{}-{:05}-of-00002.gguf",
                    self.basename,
                    i + 1
                ))
            {
                return Err("independent golden output shard roster refused".into());
            }
        }
        for a in [
            &self.tool,
            &self.tool_source,
            &self.runtime,
            &self.loader,
            &self.manifest,
            &self.tensor_recipe,
        ]
        .into_iter()
        .chain(self.source_parts.iter())
        {
            if a.path.starts_with(&self.target_root) || a.path.starts_with(&self.work_root) {
                return Err("immutable probe artifact overlaps mutable workspace".into());
            }
        }
        Ok(())
    }
    pub(super) fn quant_args(&self) -> Vec<String> {
        vec![
            "run-quant".into(),
            "--manifest".into(),
            self.manifest.path.to_string_lossy().into(),
            "--backend".into(),
            "llama-api".into(),
            "--native-runtime-library".into(),
            self.runtime.path.to_string_lossy().into(),
            "--work-dir".into(),
            self.work_root.to_string_lossy().into(),
            "--max-memory".into(),
            "32G".into(),
            "--memory-policy".into(),
            "hard".into(),
            "--max-windows".into(),
            "1".into(),
            "--json".into(),
        ]
    }
}
