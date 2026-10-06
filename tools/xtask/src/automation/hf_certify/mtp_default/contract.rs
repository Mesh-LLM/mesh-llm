use super::super::bootstrap::contract::{Input as Bootstrap, Tool};
use crate::{automation::hf_certify::admission::Artifact, command::DynResult};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::PathBuf};
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Source {
    pub repo: String,
    pub revision: String,
    pub files: BTreeMap<String, String>,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Staging {
    pub schema_version: u32,
    pub checkpoint: Source,
    pub tokenizer_source: Source,
    pub tokenizer_profile: PathBuf,
    pub tokenizer_profile_sha256: String,
    pub output_directory: PathBuf,
    pub credential_file: Option<PathBuf>,
    pub timeout_seconds: u64,
    pub maximum_bytes: u64,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Sidecar {
    pub artifact: Artifact,
    pub path_in_repo: String,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Input {
    pub schema_version: u32,
    pub bootstrap: Bootstrap,
    pub staging_helper: Tool,
    pub checkpoint: Source,
    pub tokenizer_source: Source,
    pub tokenizer_profile: Artifact,
    pub credential_file: PathBuf,
    pub maximum_bytes: u64,
    pub target_parts: Vec<Artifact>,
    #[serde(default = "target_basename")]
    pub target_basename: String,
    #[serde(default = "composite_basename")]
    pub composite_basename: String,
    #[serde(default = "mtp_block")]
    pub mtp_block: u32,
    #[serde(default = "composite_repo")]
    pub composite_repo: String,
    pub sidecars: Vec<Sidecar>,
    pub repository_helper: Tool,
    pub repository_helper_source: Artifact,
    pub publisher_helper: Artifact,
    pub publisher_source: Artifact,
    pub overall_seconds: u64,
    pub publication_reserve_seconds: u64,
    pub dry_run: bool,
    pub confirm_publication: bool,
}
pub(super) fn pin(value: &str, n: usize) -> bool {
    value.len() == n
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || matches!(b, b'a'..=b'f'))
}
impl Input {
    pub(super) fn validate(&self) -> DynResult<()> {
        self.bootstrap.validate()?;
        if self.schema_version != 1
            || !(30..=86400).contains(&self.overall_seconds)
            || self.publication_reserve_seconds < 10
            || self.publication_reserve_seconds >= self.overall_seconds
            || !(1..=1_u64 << 40).contains(&self.maximum_bytes)
            || (!self.dry_run && !self.confirm_publication)
            || !self.credential_file.is_absolute()
            || self.target_parts.len() < 2
            || self.target_parts.len() > 128
            || self.sidecars.len() > 31
            || self.composite_repo.split('/').any(|p| p.len() > 96)
        {
            return Err("compose-default bounded explicit publication admission refused".into());
        }
        for source in [&self.checkpoint, &self.tokenizer_source] {
            if !pin(&source.revision, 40)
                || source.repo.split('/').count() != 2
                || source.repo.split('/').any(|part| {
                    part.is_empty()
                        || part.len() > 96
                        || !part
                            .bytes()
                            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
                })
                || source.files.is_empty()
                || source.files.len() > 1024
                || source.files.iter().any(|(n, h)| {
                    n.is_empty()
                        || n.contains('/')
                        || n.contains('\\')
                        || n == "."
                        || n == ".."
                        || n.chars().any(char::is_control)
                        || !pin(h, 64)
                })
            {
                return Err("immutable default checkpoint/tokenizer sources refused".into());
            }
        }
        if self
            .checkpoint
            .files
            .keys()
            .any(|n| !n.ends_with(".json") && !n.ends_with(".safetensors"))
            || !self.checkpoint.files.contains_key("config.json")
            || !self
                .checkpoint
                .files
                .keys()
                .any(|n| n.ends_with(".safetensors"))
            || self.tokenizer_source.files.keys().any(|n| {
                ![
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "chat_template.jinja",
                ]
                .contains(&n.as_str())
            })
        {
            return Err("default immutable staging roster refused".into());
        }
        for name in [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ] {
            if !self.checkpoint.files.contains_key(name)
                && !self.tokenizer_source.files.contains_key(name)
            {
                return Err("required tokenizer asset absent".into());
            }
        }
        for a in std::iter::once(&self.tokenizer_profile)
            .chain(self.target_parts.iter())
            .chain([
                &self.publisher_helper,
                &self.publisher_source,
                &self.repository_helper_source,
            ])
            .chain(self.sidecars.iter().map(|s| &s.artifact))
        {
            if !a.path.is_absolute() || !pin(&a.sha256, 64) {
                return Err("complete absolute compose artifact pin required".into());
            }
        }
        for tool in [&self.staging_helper, &self.repository_helper] {
            if !tool.path.is_absolute() || !pin(&tool.sha256, 64) {
                return Err("pinned helper required".into());
            }
        }
        let template = crate::automation::hf_mtp_compose::job_phase::Template {
            target_parts: self.target_parts.clone(),
            mtp: crate::automation::hf_mtp_compose::job_phase::MtpSource::SuppliedConverted {
                artifact: self.tokenizer_profile.clone(),
            },
            target_basename: self.target_basename.clone(),
            composite_basename: self.composite_basename.clone(),
            expected_parts: self.target_parts.len(),
            mtp_block: self.mtp_block,
            composite_repo: self.composite_repo.clone(),
        };
        template.validate()
    }
}

fn target_basename() -> String {
    "NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q4_K_XL".into()
}
fn composite_basename() -> String {
    "NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q4_K_XL-MTPv2".into()
}
fn mtp_block() -> u32 {
    88
}
fn composite_repo() -> String {
    "meshllm/NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q4_K_XL-MTPv2-GGUF".into()
}
