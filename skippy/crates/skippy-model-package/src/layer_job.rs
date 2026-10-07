//! Read-only HF source admission and published package-root projection.
//! Artifact uploads may have removed local shards; this is not byte/tensor certification.
mod card;
mod license;
mod workspace;
pub use license::License;
#[cfg(unix)]
mod card_frontdoor;
#[cfg(unix)]
mod catalog_frontdoor;
#[cfg(unix)]
mod cli;
mod projector;
mod source;
pub use projector::ProjectorIdentity;
#[cfg(unix)]
mod commit_frontdoor;
#[cfg(unix)]
mod projector_frontdoor;
#[cfg(test)]
mod tests;
#[cfg(unix)]
mod upload_frontdoor;
#[cfg(unix)]
mod verification_frontdoor;
#[cfg(unix)]
pub use cli::run;
#[cfg(not(unix))]
pub fn run(_output: &mut dyn std::io::Write) -> anyhow::Result<()> {
    anyhow::bail!("layer job input custody requires Unix");
}
use anyhow::{Result, bail};
use serde::Serialize;
use sha2::{Digest, Sha256};
use skippy_package_format::PackageManifest;
pub use source::{Source, SourceClient};
use std::collections::BTreeSet;

pub const MANIFEST_LIMIT: usize = 8 * 1024 * 1024;
pub const METADATA_NAMES: [&str; 5] = [
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "hf_quant_config.json",
];
pub(crate) fn repo(value: &str) -> Result<()> {
    let parts: Vec<_> = value.split('/').collect();
    if parts.len() != 2
        || parts.iter().any(|s| {
            s.is_empty()
                || *s == "."
                || *s == ".."
                || s.len() > 128
                || !s
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
        })
    {
        bail!("HF model coordinate refused");
    }
    Ok(())
}
pub(crate) fn revision(value: &str) -> Result<()> {
    if value.len() != 40
        || !value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        bail!("immutable source revision refused");
    }
    Ok(())
}
#[derive(Serialize)]
pub struct Projection {
    pub schema_version: u32,
    pub package_id: String,
    pub manifest_sha256: String,
    pub model_id: String,
    pub source_identity: String,
    pub source_repo: String,
    pub source_revision: String,
    pub layer_count: u32,
    pub total_bytes: u64,
    pub artifacts: Vec<skippy_package_format::Artifact>,
    pub experimental: bool,
    pub scope: &'static str,
}
pub fn project(
    bytes: &[u8],
    source_repo: &str,
    source_revision: &str,
    experimental: bool,
) -> Result<(PackageManifest, Projection)> {
    repo(source_repo)?;
    revision(source_revision)?;
    if bytes.len() > MANIFEST_LIMIT {
        bail!("package root byte bound exceeded");
    }
    let manifest: PackageManifest = serde_json::from_slice(bytes)?;
    manifest.validate_root()?;
    if manifest.source_model.repo.as_deref() != Some(source_repo)
        || manifest.source_model.revision.as_deref() != Some(source_revision)
    {
        bail!("package source identity does not match resolved source");
    }
    let mut paths = BTreeSet::new();
    let mut total = 0u64;
    for artifact in &manifest.artifact_catalog.entries {
        if artifact.byte_size == 0
            || !paths.insert(&artifact.path)
            || artifact.path == "model-package.json"
            || artifact.path == "README.md"
        {
            bail!("package artifact size/path roster refused");
        }
        total = total
            .checked_add(artifact.byte_size)
            .ok_or_else(|| anyhow::anyhow!("package artifact byte total overflow"))?;
    }
    let identity = manifest
        .source_model
        .canonical_ref
        .as_ref()
        .or(manifest.source_model.primary_file.as_ref())
        .cloned()
        .unwrap_or_else(|| manifest.source_model.sha256.clone());
    if identity.is_empty() || identity.len() > 4096 || identity.chars().any(char::is_control) {
        bail!("source summary identity refused");
    }
    let projection = Projection {
        schema_version: 1,
        package_id: manifest.package_id.clone(),
        manifest_sha256: Sha256::digest(bytes)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect(),
        model_id: manifest.model_id.clone(),
        source_identity: identity,
        source_repo: source_repo.into(),
        source_revision: source_revision.into(),
        layer_count: manifest.layer_count,
        total_bytes: total,
        artifacts: manifest.artifact_catalog.entries.clone(),
        experimental,
        scope: "validated package root and declared complete catalog; no artifact-byte, source tensor, hosted upload or runtime certification",
    };
    Ok((manifest, projection))
}
