//! Closed one-window source, tool and publication authority.
use super::*;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(in crate::automation::hf_certify) struct Resume {
    pub commit: String,
    pub record: admission::Artifact,
    pub shard: admission::Artifact,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(in crate::automation::hf_certify) struct Input {
    pub schema_version: u32,
    pub tool_kind: String,
    pub profile_version: String,
    pub tool: admission::Artifact,
    pub tool_source: admission::Artifact,
    pub runtime: admission::Artifact,
    pub manifest: admission::Artifact,
    pub recipe: admission::Artifact,
    pub helper: admission::Artifact,
    pub helper_source: admission::Artifact,
    pub source_repo: String,
    pub source_revision: String,
    pub source_root: PathBuf,
    pub source_prefix: String,
    pub source_parts: Vec<admission::Artifact>,
    pub target_repo: String,
    pub target_root: PathBuf,
    pub target_prefix: String,
    pub basename: String,
    pub quant: String,
    pub expected_splits: u32,
    pub ordinal: u32,
    pub work_root: PathBuf,
    pub credential_file: PathBuf,
    pub publication_confirmed: bool,
    pub timeout_seconds: u64,
    pub resume: Option<Resume>,
}
pub(in crate::automation::hf_certify) fn disjoint(a: &Path, b: &Path) -> bool {
    !a.starts_with(b) && !b.starts_with(a)
}
fn leaf(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 128
        && s != "."
        && s != ".."
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
}
fn repo(s: &str) -> bool {
    let parts = s.split('/').collect::<Vec<_>>();
    parts.len() == 2 && parts.iter().all(|s| leaf(s))
}
impl Input {
    pub(in crate::automation::hf_certify) fn pins(&self) -> Vec<&admission::Artifact> {
        let mut pins = vec![
            &self.tool,
            &self.tool_source,
            &self.runtime,
            &self.manifest,
            &self.recipe,
            &self.helper,
            &self.helper_source,
        ];
        pins.extend(self.source_parts.iter());
        if let Some(r) = &self.resume {
            pins.extend([&r.record, &r.shard]);
        }
        pins
    }
    pub(in crate::automation::hf_certify) fn validate(&self) -> DynResult<()> {
        if self.schema_version != 1
            || self.tool_kind != "supplied-window-quantizer"
            || !self.publication_confirmed
            || !leaf(&self.profile_version)
            || !repo(&self.source_repo)
            || !repo(&self.target_repo)
            || !bootstrap::contract::hex(&self.source_revision, 40)
            || !(1..=1024).contains(&self.expected_splits)
            || !(1..=self.expected_splits).contains(&self.ordinal)
            || !(5..=86400).contains(&self.timeout_seconds)
            || !leaf(&self.source_prefix)
            || !leaf(&self.target_prefix)
            || !leaf(&self.basename)
            || !leaf(&self.quant)
            || self.source_parts.len() != self.expected_splits as usize
        {
            return Err(
                "quant window explicit supplied tool/authority/complete roster refused".into(),
            );
        }
        for p in [
            &self.source_root,
            &self.target_root,
            &self.work_root,
            &self.credential_file,
        ] {
            if !p.is_absolute() {
                return Err("quant window absolute paths required".into());
            }
        }
        if !disjoint(&self.source_root, &self.target_root)
            || !disjoint(&self.source_root, &self.work_root)
            || !disjoint(&self.target_root, &self.work_root)
            || self.credential_file.starts_with(&self.target_root)
            || self.credential_file.starts_with(&self.work_root)
        {
            return Err("quant window immutable/mutable ancestry refused".into());
        }
        let mut paths = std::collections::BTreeSet::new();
        for a in self.pins() {
            if !a.path.is_absolute()
                || !bootstrap::contract::hex(&a.sha256, 64)
                || !paths.insert(a.path.clone())
                || a.path.starts_with(&self.target_root)
                || a.path.starts_with(&self.work_root)
            {
                return Err("quant window immutable artifact identity/overlap refused".into());
            }
        }
        for (i, a) in self.source_parts.iter().enumerate() {
            if a.path
                != self.source_root.join(&self.source_prefix).join(format!(
                    "{}-{:05}-of-{:05}.gguf",
                    self.basename,
                    i + 1,
                    self.expected_splits
                ))
            {
                return Err("quant complete ordered source roster refused".into());
            }
        }
        if let Some(r) = &self.resume
            && !bootstrap::contract::hex(&r.commit, 40)
        {
            return Err("quant resume immutable commit required".into());
        }
        Ok(())
    }
    pub(in crate::automation::hf_certify) fn output_path(&self) -> PathBuf {
        self.target_root
            .join(&self.target_prefix)
            .join(self.shard_name())
    }
    fn shard_name(&self) -> String {
        format!(
            "{}-{:05}-of-{:05}.gguf",
            self.basename, self.ordinal, self.expected_splits
        )
    }
    pub(in crate::automation::hf_certify) fn remote_path(&self) -> String {
        format!("{}/{}", self.target_prefix, self.shard_name())
    }
    pub(in crate::automation::hf_certify) fn record_path(&self) -> String {
        format!(
            "window-records/{}-{:05}-of-{:05}.json",
            self.target_prefix, self.ordinal, self.expected_splits
        )
    }
    pub(in crate::automation::hf_certify) fn context_sha256(&self) -> DynResult<String> {
        Ok(admission::digest(&serde_json::to_vec(
            &json!({"schema_version":1,"source_repo":self.source_repo,"source_revision":self.source_revision,"source_sha256":self.source_parts.iter().map(|a|&a.sha256).collect::<Vec<_>>(),"manifest_sha256":self.manifest.sha256,"recipe_sha256":self.recipe.sha256,"tool_sha256":self.tool.sha256,"tool_source_sha256":self.tool_source.sha256,"runtime_sha256":self.runtime.sha256,"profile_version":self.profile_version,"target_repo":self.target_repo,"target_prefix":self.target_prefix,"basename":self.basename,"quant":self.quant,"expected_splits":self.expected_splits}),
        )?))
    }
    pub(in crate::automation::hf_certify) fn quant_args(&self) -> Vec<String> {
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
            "--first-split".into(),
            self.ordinal.to_string(),
            "--last-split".into(),
            self.ordinal.to_string(),
            "--keep-staged-source".into(),
            "--json".into(),
        ]
    }
}
