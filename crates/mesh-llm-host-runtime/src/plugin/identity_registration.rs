//! Resolve identity grants and installed artifact from the authenticated host registration.

use anyhow::{Result, bail};
use mesh_llm_identity::plugin_delegation::IdentityEvidenceStatus;
use sha2::{Digest, Sha256};
use std::path::Path;
use std::time::Duration;
use tokio::io::AsyncReadExt;

use super::PluginManager;
use super::identity_services::PluginIdentityGrants;

pub const EXCHANGE_SIGNING_SCOPE: &str = "mesh.openai.exchange.evidence.sign.v1";
const MAX_ARTIFACT_BYTES: u64 = 256 * 1024 * 1024;
pub(super) const IDENTITY_EXECUTION_DEADLINE: Duration = Duration::from_secs(20);

#[derive(Debug)]
pub(super) struct ArtifactInspectionTimeout;

impl std::fmt::Display for ArtifactInspectionTimeout {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("plugin artifact inspection deadline exceeded; retry setup")
    }
}

impl std::error::Error for ArtifactInspectionTimeout {}

pub(super) fn artifact_inspection_is_transient(error: &anyhow::Error) -> bool {
    error.downcast_ref::<ArtifactInspectionTimeout>().is_some()
}

fn artifact_inspection_deadline(bytes: u64) -> Duration {
    // Budget 16 MiB/s plus one second for I/O scheduling, capped at 17s.
    Duration::from_secs(1 + bytes.min(MAX_ARTIFACT_BYTES).div_ceil(16 * 1024 * 1024))
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PublicPluginArtifactMetadata {
    pub installed_version: String,
    pub source_repository: String,
    pub target_triple: String,
    pub downloaded_asset_name: String,
}

impl PluginManager {
    pub(crate) fn identity_plugin_metadata(
        &self,
        plugin_name: &str,
    ) -> Result<PublicPluginArtifactMetadata> {
        let metadata = self
            .inner
            .plugins
            .get(plugin_name)
            .and_then(|plugin| plugin.installed_metadata())
            .ok_or_else(|| anyhow::anyhow!("installed plugin metadata unavailable"))?;
        Ok(PublicPluginArtifactMetadata {
            installed_version: metadata.installed_version.clone(),
            source_repository: if metadata.source_repository.starts_with("local:") {
                "local_archive".into()
            } else {
                metadata.source_repository.clone()
            },
            target_triple: metadata.target_triple.clone(),
            downloaded_asset_name: metadata.downloaded_asset_name.clone(),
        })
    }

    pub(crate) async fn identity_grants_and_artifact(
        &self,
        plugin_name: &str,
    ) -> Result<(PluginIdentityGrants, Option<String>, IdentityEvidenceStatus)> {
        let plugin = self
            .inner
            .plugins
            .get(plugin_name)
            .ok_or_else(|| anyhow::anyhow!("authenticated plugin registration unavailable"))?;
        let grant = self
            .effective_exchange_grant(plugin_name)
            .ok_or_else(|| anyhow::anyhow!("plugin has no host identity grant"))?;
        let manifest = plugin
            .manifest_snapshot()
            .await
            .ok_or_else(|| anyhow::anyhow!("plugin manifest unavailable"))?;
        let mut declaration = manifest
            .openai_exchange_hook
            .ok_or_else(|| anyhow::anyhow!("plugin has no declared identity services"))?;
        // Startup already enforces required declarations. Live reductions clip
        // each service independently and cannot restore removed privileges.
        declaration.required = false;
        let grant = mesh_llm_plugin::openai_exchange::negotiate_openai_exchange(
            Some(&declaration),
            Some(&grant),
        )?
        .ok_or_else(|| anyhow::anyhow!("plugin has no negotiated identity services"))?;
        let grants = PluginIdentityGrants {
            read_identity_bundle: grant.read_identity_bundle,
            delegate_signing_key: grant.delegate_signing_key
                && grant
                    .signing_scopes
                    .iter()
                    .any(|scope| scope == EXCHANGE_SIGNING_SCOPE),
            max_delegation_lifetime_ms: grant.max_delegation_ttl_secs.saturating_mul(1000),
        };
        if !grants.read_identity_bundle && !grants.delegate_signing_key {
            bail!("plugin has no granted identity services");
        }
        let metadata = plugin.installed_metadata().ok_or_else(|| {
            anyhow::anyhow!("identity delegation requires a host-installed plugin artifact")
        })?;
        if metadata.name != plugin_name {
            bail!("installed plugin identity differs from authenticated connection");
        }
        let (digest, status) = inspect_artifact(
            &metadata.executable_path(),
            plugin.installed_artifact_sha256(),
        )
        .await?;
        Ok((grants, digest, status))
    }
}

pub(super) async fn artifact_sha256(path: &Path) -> Result<String> {
    let metadata = tokio::time::timeout(Duration::from_secs(1), tokio::fs::symlink_metadata(path))
        .await
        .map_err(|_| ArtifactInspectionTimeout)??;
    bounded_artifact_hash(
        hash_regular_artifact(path),
        artifact_inspection_deadline(metadata.len()),
    )
    .await
}

async fn bounded_artifact_hash(
    hash: impl std::future::Future<Output = Result<String>>,
    deadline: Duration,
) -> Result<String> {
    tokio::time::timeout(deadline, hash)
        .await
        .map_err(|_| ArtifactInspectionTimeout)?
}

async fn inspect_artifact(
    path: &Path,
    captured: Option<&str>,
) -> Result<(Option<String>, IdentityEvidenceStatus)> {
    artifact_evidence(artifact_sha256(path).await, captured)
}

fn artifact_evidence(
    result: Result<String>,
    captured: Option<&str>,
) -> Result<(Option<String>, IdentityEvidenceStatus)> {
    match result {
        Ok(digest) => {
            let status = if Some(digest.as_str()) == captured {
                IdentityEvidenceStatus::Verified
            } else {
                IdentityEvidenceStatus::Invalid
            };
            Ok((Some(digest), status))
        }
        Err(error) if artifact_inspection_is_transient(&error) => Err(error),
        Err(error) => {
            let missing = error
                .downcast_ref::<std::io::Error>()
                .is_some_and(|error| error.kind() == std::io::ErrorKind::NotFound);
            Ok((
                None,
                if missing {
                    IdentityEvidenceStatus::Missing
                } else {
                    IdentityEvidenceStatus::Invalid
                },
            ))
        }
    }
}

async fn hash_regular_artifact(path: &Path) -> Result<String> {
    let metadata = tokio::fs::symlink_metadata(path).await?;
    if !metadata.is_file() || metadata.len() > MAX_ARTIFACT_BYTES {
        bail!("plugin artifact must be a regular file at most 256 MiB");
    }
    let mut options = tokio::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    options.custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW);
    let mut file = options.open(path).await?;
    let metadata = file.metadata().await?;
    if !metadata.is_file() || metadata.len() > MAX_ARTIFACT_BYTES {
        bail!("plugin artifact changed during inspection");
    }
    let mut digest = Sha256::new();
    // Keep the I/O buffer off every enclosing host/router future's stack.
    let mut buffer = vec![0; 64 * 1024];
    let mut total = 0u64;
    loop {
        let len = file.read(&mut buffer).await?;
        if len == 0 {
            break;
        }
        total = total.saturating_add(len as u64);
        if total > MAX_ARTIFACT_BYTES {
            bail!("plugin artifact exceeds inspection size limit");
        }
        digest.update(&buffer[..len]);
    }
    Ok(hex::encode(digest.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn changed_and_missing_artifacts_remain_publicly_inspectable() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("observer");
        tokio::fs::write(&path, b"original").await.unwrap();
        let captured = artifact_sha256(&path).await.unwrap();
        assert_eq!(
            inspect_artifact(&path, Some(&captured)).await.unwrap().1,
            IdentityEvidenceStatus::Verified
        );
        tokio::fs::write(&path, b"changed").await.unwrap();
        let (actual, status) = inspect_artifact(&path, Some(&captured)).await.unwrap();
        assert_eq!(status, IdentityEvidenceStatus::Invalid);
        assert_eq!(actual.unwrap(), hex::encode(Sha256::digest(b"changed")));
        tokio::fs::remove_file(&path).await.unwrap();
        assert_eq!(
            inspect_artifact(&path, Some(&captured)).await.unwrap(),
            (None, IdentityEvidenceStatus::Missing)
        );
    }
    #[tokio::test]
    async fn inspection_timeouts_remain_transient_and_size_budget_is_bounded() {
        assert_eq!(artifact_inspection_deadline(1), Duration::from_secs(2));
        assert_eq!(
            artifact_inspection_deadline(64 * 1024 * 1024),
            Duration::from_secs(5)
        );
        assert_eq!(
            artifact_inspection_deadline(u64::MAX),
            Duration::from_secs(17)
        );
        assert!(artifact_inspection_deadline(MAX_ARTIFACT_BYTES) < IDENTITY_EXECUTION_DEADLINE);
        let error = bounded_artifact_hash(std::future::pending(), Duration::ZERO)
            .await
            .unwrap_err();
        assert!(artifact_inspection_is_transient(&error));
        let error = artifact_evidence(Err(error), Some("captured")).unwrap_err();
        assert!(artifact_inspection_is_transient(&error));
        assert!(!artifact_inspection_is_transient(&anyhow::anyhow!(
            "artifact changed"
        )));
    }
    #[tokio::test]
    async fn artifact_reads_reject_nonregular_and_oversized_inputs() {
        let root = tempfile::tempdir().unwrap();
        assert!(artifact_sha256(root.path()).await.is_err());
        let path = root.path().join("oversized");
        let file = std::fs::File::create(&path).unwrap();
        file.set_len(256 * 1024 * 1024 + 1).unwrap();
        assert!(artifact_sha256(&path).await.is_err());
        #[cfg(unix)]
        {
            let fifo = root.path().join("fifo");
            assert!(
                std::process::Command::new("mkfifo")
                    .arg(&fifo)
                    .status()
                    .unwrap()
                    .success()
            );
            assert!(
                tokio::time::timeout(
                    std::time::Duration::from_millis(200),
                    artifact_sha256(&fifo)
                )
                .await
                .unwrap()
                .is_err()
            );
            let link = root.path().join("link");
            std::os::unix::fs::symlink(&path, &link).unwrap();
            assert!(artifact_sha256(&link).await.is_err());
        }
    }
}
