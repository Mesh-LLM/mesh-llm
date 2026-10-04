//! Parent-bound snapshot promotion policy, independent of the HF transport.
use anyhow::{Context, Result, bail};
use serde::Serialize;
use sha2::{Digest, Sha256};
use skippy_package_format::PackageManifest;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct SnapshotPlan {
    staging_revision: String,
    parent_commit: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct PromotionPlan {
    staging_revision: String,
    parent_commit: String,
    paths: Vec<String>,
    identities: BTreeMap<String, ArtifactIdentity>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ArtifactIdentity {
    pub byte_size: u64,
    pub sha256: String,
}

fn immutable_revision(value: &str) -> Result<()> {
    if value.len() != 40
        || !value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        bail!("snapshot revision must be a lowercase 40-character commit SHA");
    }
    Ok(())
}

pub fn prepare(source_revision: &str, token: &str, parent_commit: &str) -> Result<SnapshotPlan> {
    immutable_revision(source_revision)?;
    immutable_revision(parent_commit)?;
    let normalized: String = token
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-') {
                ch
            } else {
                '-'
            }
        })
        .collect();
    let normalized = normalized.trim_matches('-');
    if normalized.is_empty() {
        bail!("snapshot token is empty");
    }
    Ok(SnapshotPlan {
        staging_revision: format!(
            "automation/republish-{}-{normalized}",
            &source_revision[..12]
        ),
        parent_commit: parent_commit.to_owned(),
    })
}

fn artifact_path(path: &str) -> Result<()> {
    if path.is_empty()
        || path.contains('\\')
        || path.chars().any(char::is_control)
        || path
            .split('/')
            .any(|part| part.is_empty() || matches!(part, "." | "..") || part.contains(':'))
    {
        bail!("snapshot catalog path must be a safe repository-relative path");
    }
    Ok(())
}

pub fn promote(
    manifest: &[u8],
    staging_revision: &str,
    parent_commit: &str,
) -> Result<PromotionPlan> {
    immutable_revision(parent_commit)?;
    let suffix = staging_revision
        .strip_prefix("automation/republish-")
        .context("snapshot staging revision must use the republish namespace")?;
    if suffix.is_empty()
        || suffix
            .chars()
            .any(|ch| !ch.is_ascii_alphanumeric() && !matches!(ch, '.' | '_' | '-'))
    {
        bail!("snapshot staging revision is invalid");
    }
    let manifest_identity = ArtifactIdentity {
        byte_size: u64::try_from(manifest.len())?,
        sha256: Sha256::digest(manifest)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect(),
    };
    let manifest: PackageManifest =
        serde_json::from_slice(manifest).context("read snapshot package root")?;
    manifest
        .validate_root()
        .context("validate snapshot package root")?;
    if manifest.artifact_catalog.entries.is_empty() {
        bail!("manifest artifact catalog is empty");
    }
    let mut identities = BTreeMap::new();
    let mut paths = Vec::new();
    for entry in manifest.artifact_catalog.entries {
        if entry.sha256.len() != 64
            || !entry
                .sha256
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            bail!("snapshot artifact must have a lowercase SHA-256 identity");
        }
        paths.push(entry.path.clone());
        identities.insert(
            entry.path,
            ArtifactIdentity {
                byte_size: entry.byte_size,
                sha256: entry.sha256,
            },
        );
    }
    identities.insert("model-package.json".to_owned(), manifest_identity);
    paths.push("model-package.json".to_owned());
    let mut seen = BTreeSet::new();
    for path in &paths {
        artifact_path(path)?;
        if !seen.insert(path) {
            bail!("manifest snapshot paths are not unique");
        }
    }
    Ok(PromotionPlan {
        staging_revision: staging_revision.to_owned(),
        parent_commit: parent_commit.to_owned(),
        paths,
        identities,
    })
}

/// Transport must publish all listed paths in one commit against the exact parent.
/// A rejected commit leaves main unchanged. An uncertain network result retains
/// staging for reconciliation; it must not be reported as confirmed publication.
pub trait SnapshotTransport {
    fn main_revision(&mut self) -> Result<String>;
    fn create_staging(&mut self, plan: &SnapshotPlan) -> Result<()>;
    fn publish(&mut self, plan: &PromotionPlan) -> Result<()>;
    fn delete_staging(&mut self, staging_revision: &str) -> Result<()>;
}

pub fn prepare_with(
    transport: &mut impl SnapshotTransport,
    source: &str,
    token: &str,
) -> Result<SnapshotPlan> {
    immutable_revision(source)?;
    let parent = transport.main_revision()?;
    let plan = prepare(source, token, &parent)?;
    transport.create_staging(&plan)?;
    Ok(plan)
}

/// Successful publication remains successful if best-effort staging cleanup fails.
pub fn promote_with(
    transport: &mut impl SnapshotTransport,
    plan: &PromotionPlan,
) -> Result<Option<String>> {
    transport.publish(plan)?;
    Ok(transport
        .delete_staging(&plan.staging_revision)
        .err()
        .map(|error| error.to_string()))
}

impl SnapshotPlan {
    pub fn staging_revision(&self) -> &str {
        &self.staging_revision
    }
    pub fn parent_commit(&self) -> &str {
        &self.parent_commit
    }
}

impl PromotionPlan {
    pub fn staging_revision(&self) -> &str {
        &self.staging_revision
    }
    pub fn parent_commit(&self) -> &str {
        &self.parent_commit
    }
    pub fn paths(&self) -> &[String] {
        &self.paths
    }
    pub fn identity(&self, path: &str) -> Option<&ArtifactIdentity> {
        self.identities.get(path)
    }
}
