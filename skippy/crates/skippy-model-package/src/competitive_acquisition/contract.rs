use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::PathBuf, time::Instant};
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub schema_version: u32,
    pub config: PathBuf,
    pub config_sha256: String,
    #[serde(default)]
    pub model_manifest: Option<PathBuf>,
    #[serde(default)]
    pub model_manifest_sha256: Option<String>,
    #[serde(default)]
    pub model_keys: Vec<String>,
    pub output_directory: PathBuf,
    pub timeout_seconds: u64,
    pub maximum_bytes: u64,
    pub credential_file: Option<PathBuf>,
    #[serde(default)]
    pub skip_dataset: bool,
    #[serde(default)]
    pub skip_tokenizers: bool,
    #[serde(default)]
    pub skip_vllm_configs: bool,
    // Explicit accepted derived lineage; never modify the benchmark config implicitly.
    pub export_sha256: BTreeMap<String, String>,
    pub semantic_cases: BTreeMap<String, Vec<Case>>,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Case {
    pub text: String,
    pub add_special_tokens: bool,
    pub decode_ids: Vec<u32>,
    pub skip_special_tokens: bool,
    pub expected_ids: Vec<u32>,
    pub expected_decoded_sha256: String,
}
pub(crate) fn pin(s: &str, n: usize) -> bool {
    s.len() == n
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
pub(crate) fn path(s: &str) -> bool {
    !s.is_empty()
        && !s.starts_with('/')
        && !s.contains('\\')
        && s.split('/')
            .all(|p| !p.is_empty() && p != "." && p != ".." && !p.chars().any(char::is_control))
}
pub(crate) fn check(deadline: Instant) -> Result<()> {
    if Instant::now() >= deadline {
        bail!("acquisition deadline expired");
    }
    Ok(())
}
impl Request {
    pub(super) fn validate(&self) -> Result<()> {
        if self.schema_version != 1
            || !self.config.is_absolute()
            || !pin(&self.config_sha256, 64)
            || !self.output_directory.is_absolute()
            || !(1..=86400).contains(&self.timeout_seconds)
            || !(1..=1024 * 1024 * 1024 * 1024).contains(&self.maximum_bytes)
        {
            bail!("acquisition input refused");
        }
        match std::fs::symlink_metadata(&self.output_directory) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => (),
            _ => bail!("fresh output required"),
        }
        let parent = self
            .output_directory
            .parent()
            .ok_or_else(|| anyhow::anyhow!("output parent"))?;
        if !parent.is_dir()
            || parent.canonicalize()? != parent
            || self.config.starts_with(&self.output_directory)
        {
            bail!("canonical disjoint output parent required");
        }
        Ok(())
    }
}
pub(crate) fn digest(bytes: &[u8]) -> String {
    use sha2::Digest as _;
    sha2::Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
pub(super) fn tree(rows: &BTreeMap<String, String>) -> Result<String> {
    use sha2::Digest as _;
    let mut hash = sha2::Sha256::new();
    for (name, pin) in rows {
        hash.update((name.len() as u64).to_be_bytes());
        hash.update(name.as_bytes());
        for pair in pin.as_bytes().as_chunks::<2>().0 {
            hash.update([u8::from_str_radix(std::str::from_utf8(pair)?, 16)?]);
        }
    }
    Ok(hash.finalize().iter().map(|b| format!("{b:02x}")).collect())
}
