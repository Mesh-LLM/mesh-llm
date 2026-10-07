//! Explicit tokenizer profile bound to every source file consumed for metadata.
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{fs, io::Read, path::Path};
#[derive(Clone, Copy, Deserialize)]
pub enum Pre {
    #[serde(rename = "llama-bpe")]
    LlamaBpe,
    #[serde(rename = "llama3")]
    Llama3,
    #[serde(rename = "qwen2")]
    Qwen2,
    #[serde(rename = "dbrx")]
    Dbrx,
}
impl Pre {
    pub(super) fn name(self) -> &'static str {
        match self {
            Self::LlamaBpe => "llama-bpe",
            Self::Llama3 => "llama3",
            Self::Qwen2 => "qwen2",
            Self::Dbrx => "dbrx",
        }
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Profile {
    pub schema_version: u32,
    pub config_sha256: String,
    pub tokenizer_sha256: String,
    pub tokenizer_config_sha256: Option<String>,
    pub chat_template_sha256: Option<String>,
    pub pre: Pre,
}
pub(super) struct Bound {
    pub config: Value,
    pub tokenizer: Value,
    pub tokenizer_config: Value,
    pub template: Option<String>,
    pub pre: Pre,
    pub profile_sha256: String,
}
fn read(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let path = path
        .canonicalize()
        .with_context(|| format!("metadata source absent: {}", path.display()))?;
    ensure!(
        fs::metadata(&path)?.is_file(),
        "metadata source must resolve to regular file"
    );
    let mut options = fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW);
    }
    let file = options.open(&path)?;
    let metadata = file.metadata()?;
    ensure!(
        metadata.is_file() && metadata.len() <= limit,
        "metadata source type/size refused"
    );
    let mut bytes = Vec::new();
    file.take(limit + 1).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= limit,
        "metadata source byte bound refused"
    );
    Ok(bytes)
}
fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
fn pinned(path: &Path, hash: &str, limit: u64) -> Result<Vec<u8>> {
    ensure!(
        hash.len() == 64
            && hash
                .bytes()
                .all(|b| b.is_ascii_digit() || matches!(b, b'a'..=b'f')),
        "metadata SHA256 pin must be complete lowercase hex"
    );
    let bytes = read(path, limit)?;
    ensure!(digest(&bytes) == hash, "metadata source byte pin mismatch");
    Ok(bytes)
}
fn optional(path: &Path, hash: Option<&str>, limit: u64) -> Result<Option<Vec<u8>>> {
    match hash {
        Some(hash) => pinned(path, hash, limit).map(Some),
        None => {
            ensure!(
                matches!(fs::symlink_metadata(path),Err(ref e) if e.kind()==std::io::ErrorKind::NotFound),
                "unbound optional metadata source refused"
            );
            Ok(None)
        }
    }
}
pub(super) fn load(source: &Path, profile: &Path) -> Result<Bound> {
    let bytes = read(profile, 16384)?;
    let profile_sha256 = digest(&bytes);
    let profile: Profile = serde_json::from_slice(&bytes)?;
    ensure!(
        profile.schema_version == 1,
        "tokenizer profile schema refused"
    );
    let config = serde_json::from_slice(&pinned(
        &source.join("config.json"),
        &profile.config_sha256,
        4 * 1024 * 1024,
    )?)?;
    let tokenizer = serde_json::from_slice(&pinned(
        &source.join("tokenizer.json"),
        &profile.tokenizer_sha256,
        128 * 1024 * 1024,
    )?)?;
    let tokenizer_config = optional(
        &source.join("tokenizer_config.json"),
        profile.tokenizer_config_sha256.as_deref(),
        4 * 1024 * 1024,
    )?
    .map(|bytes| serde_json::from_slice(&bytes))
    .transpose()?
    .unwrap_or_else(|| serde_json::json!({}));
    let template = optional(
        &source.join("chat_template.jinja"),
        profile.chat_template_sha256.as_deref(),
        4 * 1024 * 1024,
    )?
    .map(String::from_utf8)
    .transpose()?;
    Ok(Bound {
        config,
        tokenizer,
        tokenizer_config,
        template,
        pre: profile.pre,
        profile_sha256,
    })
}
