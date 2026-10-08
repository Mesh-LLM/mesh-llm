use super::{ArtifactIdentity, lfs_transfer};
use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::{fs::File, path::PathBuf, time::Instant};
#[derive(Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum RepositoryKind {
    Model,
    Dataset,
}
impl RepositoryKind {
    pub(super) fn api(self) -> &'static str {
        match self {
            Self::Model => "models",
            Self::Dataset => "datasets",
        }
    }
    pub(super) fn prefix(self) -> &'static [&'static str] {
        match self {
            Self::Model => &[],
            Self::Dataset => &["datasets"],
        }
    }
}
pub struct Plan {
    pub repo: String,
    pub kind: RepositoryKind,
    pub revision: String,
    pub path: String,
    pub create_pr: bool,
    pub maximum_attempts: u8,
    pub expected_parent: Option<String>,
}
fn component(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 96
        && !matches!(s, "." | "..")
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
}
impl Plan {
    pub fn validate(&self) -> Result<()> {
        if self
            .expected_parent
            .as_ref()
            .is_some_and(|p| !crate::snapshot_promotion::regular_publication::contract::hex(p, 40))
        {
            bail!("expected catalog parent refused");
        }
        if self.repo.split('/').count() != 2
            || !self.repo.split('/').all(component)
            || self.path.len() > 512
            || !self.path.split('/').all(component)
            || self.revision.len() > 256
            || !self.revision.split('/').all(component)
            || !(1..=8).contains(&self.maximum_attempts)
            || (self.create_pr && (self.kind != RepositoryKind::Dataset || self.revision != "main"))
        {
            bail!("package upload repo/revision/path/attempt/PR policy refused");
        }
        Ok(())
    }
}
pub struct Artifact {
    pub file: File,
    pub identity: ArtifactIdentity,
    pub unlink_path: Option<PathBuf>,
}
impl Artifact {
    pub(super) fn verify(&mut self, until: Instant) -> Result<()> {
        lfs_transfer::Object {
            file: self.file.try_clone()?,
            oid: self.identity.sha256.clone(),
            size: self.identity.byte_size,
        }
        .verify(until)?;
        if let Some(path) = &self.unlink_path {
            if !path.is_absolute()
                || path
                    .parent()
                    .ok_or_else(|| anyhow::anyhow!("unlink parent"))?
                    .canonicalize()?
                    != path.parent().unwrap()
            {
                bail!("unlink path requires canonical parent");
            }
            let observed = std::fs::symlink_metadata(path)?;
            let owned = self.file.metadata()?;
            if !observed.is_file() || observed.len() != owned.len() {
                bail!("unlink path custody refused");
            }
            #[cfg(unix)]
            {
                use std::os::unix::fs::MetadataExt as _;
                if observed.dev() != owned.dev() || observed.ino() != owned.ino() {
                    bail!("unlink source replaced");
                }
            }
            #[cfg(not(unix))]
            bail!("successful unlink requires Unix file identity");
        }
        check(until)
    }
    pub(super) fn unlink(&mut self, until: Instant) -> Result<()> {
        self.verify(until)?;
        if let Some(path) = &self.unlink_path {
            std::fs::remove_file(path)?;
        }
        Ok(())
    }
}
pub(super) fn check(until: Instant) -> Result<()> {
    if Instant::now() >= until {
        bail!("package upload deadline");
    }
    Ok(())
}
