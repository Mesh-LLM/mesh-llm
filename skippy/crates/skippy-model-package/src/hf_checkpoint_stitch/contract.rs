use crate::competitive_acquisition::contract::{path, pin};
use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::PathBuf};
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Source {
    pub repo: String,
    pub revision: String,
    pub files: BTreeMap<String, String>,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
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
pub(super) const TOKENIZERS: [&str; 4] = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
];
pub(super) fn checkpoint_file(name: &str) -> bool {
    !name.contains('/') && (name.ends_with(".json") || name.ends_with(".safetensors"))
}
impl Source {
    fn validate(&self) -> Result<()> {
        if !path(&self.repo)
            || self.repo.split('/').count() != 2
            || !pin(&self.revision, 40)
            || self.files.is_empty()
            || self.files.len() > 1024
            || self.files.iter().any(|(n, h)| !path(n) || !pin(h, 64))
        {
            bail!("immutable complete source pins required");
        }
        Ok(())
    }
}
impl Request {
    pub(super) fn validate(&self) -> Result<()> {
        self.checkpoint.validate()?;
        self.tokenizer_source.validate()?;
        if self.schema_version != 1
            || !(1..=86400).contains(&self.timeout_seconds)
            || !(1..=1_u64 << 40).contains(&self.maximum_bytes)
            || !self.tokenizer_profile.is_absolute()
            || !pin(&self.tokenizer_profile_sha256, 64)
            || !self.output_directory.is_absolute()
        {
            bail!("checkpoint request bounds refused");
        }
        if self.checkpoint.files.keys().any(|n| !checkpoint_file(n))
            || !self.checkpoint.files.contains_key("config.json")
            || !self
                .checkpoint
                .files
                .keys()
                .any(|n| n.ends_with(".safetensors"))
            || self
                .tokenizer_source
                .files
                .keys()
                .any(|n| !TOKENIZERS.contains(&n.as_str()))
        {
            bail!("checkpoint/tokenizer roster refused");
        }
        for name in &TOKENIZERS[..3] {
            if !self.checkpoint.files.contains_key(*name)
                && !self.tokenizer_source.files.contains_key(*name)
            {
                bail!("required tokenizer source absent");
            }
        }
        let parent = self
            .output_directory
            .parent()
            .ok_or_else(|| anyhow::anyhow!("output parent"))?;
        if !parent.is_dir()
            || parent.canonicalize()? != parent
            || self.tokenizer_profile.starts_with(&self.output_directory)
            || self
                .credential_file
                .as_ref()
                .is_some_and(|p| p.starts_with(&self.output_directory))
        {
            bail!("canonical disjoint output parent required");
        }
        match std::fs::symlink_metadata(&self.output_directory) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => (),
            _ => bail!("fresh output required"),
        };
        Ok(())
    }
}
