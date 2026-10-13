//! One pinned layer-package artifact per commit, not whole-model certification.
pub mod catalog;
mod catalog_remote;
pub use catalog_remote::PreparedCatalog;
mod commit_verification;
mod exchange;
mod policy;
mod repository;
mod verification;
pub use commit_verification::{CommitArtifact, CommitReceipt, CommitRequest};
pub use repository::RepositoryReceipt;
pub use verification::VerificationReceipt;
#[cfg(test)]
mod tests;
use super::{lfs_transfer, policy::ArtifactIdentity};
use anyhow::{Result, anyhow};
use futures::{Future, FutureExt};
pub use policy::{Artifact, Plan, RepositoryKind};
use serde::Serialize;
use std::time::{Duration, Instant};

#[derive(Serialize)]
pub struct Attempt {
    pub ordinal: u8,
    pub parent_commit: Option<String>,
    pub object: Option<lfs_transfer::Receipt>,
    pub commit_attempted: bool,
    pub commit_oid: Option<String>,
    pub remote_verified: bool,
    pub error: Option<String>,
}
#[derive(Serialize)]
pub struct Receipt {
    pub schema_version: u32,
    pub repo: String,
    pub revision: String,
    pub path: String,
    pub identity: ArtifactIdentity,
    pub attempts: Vec<Attempt>,
    pub source_custody_verified: bool,
    pub unlink_requested: bool,
    pub unlinked: bool,
    pub completed: bool,
    pub error: Option<String>,
}
pub struct Publisher {
    client: lfs_transfer::Client,
}
impl Publisher {
    pub fn new(explicit_token: String) -> Result<Self> {
        Ok(Self {
            client: lfs_transfer::Client::new(explicit_token)?,
        })
    }
    /// One inherited deadline/cancellation scope covers hashing, attempts, delays and final custody.
    /// A PR's uncertain commit is never repeated: a duplicate PR cannot be safely reconciled.
    pub async fn upload_until<C: Future<Output = ()>>(
        &self,
        plan: &Plan,
        mut artifact: Artifact,
        until: Instant,
        cancellation: C,
        observer: &mut dyn FnMut(&Receipt) -> Result<()>,
    ) -> Receipt {
        let mut receipt = Receipt {
            schema_version: 1,
            repo: plan.repo.clone(),
            revision: plan.revision.clone(),
            path: plan.path.clone(),
            identity: artifact.identity.clone(),
            attempts: Vec::new(),
            source_custody_verified: false,
            unlink_requested: artifact.unlink_path.is_some(),
            unlinked: false,
            completed: false,
            error: None,
        };
        let mut cancel = Box::pin(cancellation.fuse());
        let result = tokio::time::timeout_at(
            until.into(),
            self.execute(
                plan,
                &mut artifact,
                until,
                cancel.as_mut(),
                &mut receipt,
                observer,
            ),
        )
        .await
        .unwrap_or_else(|_| {
            Err(anyhow!(
                "package upload deadline; mutation may be unconfirmed"
            ))
        });
        let result = if cancel.as_mut().now_or_never().is_some() {
            Err(anyhow!(
                "package upload cancelled; mutation may be unconfirmed"
            ))
        } else {
            policy::check(until).and(result)
        };
        match result {
            Ok(()) => receipt.completed = true,
            Err(error) => receipt.error = Some(error.to_string()),
        }
        receipt
    }
    async fn execute<C: Future<Output = ()>>(
        &self,
        plan: &Plan,
        artifact: &mut Artifact,
        until: Instant,
        mut cancel: std::pin::Pin<&mut C>,
        receipt: &mut Receipt,
        observer: &mut dyn FnMut(&Receipt) -> Result<()>,
    ) -> Result<()> {
        plan.validate()?;
        if cancel.as_mut().now_or_never().is_some() {
            return Err(anyhow!("package upload pre-cancelled"));
        }
        artifact.verify(until)?;
        for ordinal in 1..=plan.maximum_attempts {
            if cancel.as_mut().now_or_never().is_some() {
                return Err(anyhow!("package upload cancelled"));
            }
            policy::check(until)?;
            receipt.attempts.push(Attempt {
                ordinal,
                parent_commit: None,
                object: None,
                commit_attempted: false,
                commit_oid: None,
                remote_verified: false,
                error: None,
            });
            observer(receipt)?;
            let attempt = receipt
                .attempts
                .last_mut()
                .ok_or_else(|| anyhow!("attempt receipt absent"))?;
            let result = self
                .attempt(plan, artifact, until, cancel.as_mut(), attempt)
                .await;
            if let Err(error) = &result {
                attempt.error = Some(error.to_string());
            }
            observer(receipt)?;
            if result.is_ok() {
                artifact.verify(until)?;
                receipt.source_custody_verified = true;
                observer(receipt)?;
                policy::check(until)?;
                if cancel.as_mut().now_or_never().is_some() {
                    return Err(anyhow!("package upload cancelled before unlink"));
                }
                artifact.unlink(until)?;
                receipt.unlinked = artifact.unlink_path.is_some();
                return Ok(());
            }
            let last = receipt
                .attempts
                .last()
                .ok_or_else(|| anyhow!("attempt receipt absent"))?;
            if ordinal == plan.maximum_attempts || (plan.create_pr && last.commit_attempted) {
                return result;
            }
            artifact.verify(until)?;
            let delay = Duration::from_secs((10_u64 << (ordinal - 1)).min(300));
            let sleep = tokio::time::sleep(delay);
            match futures::future::select(cancel.as_mut(), Box::pin(sleep)).await {
                futures::future::Either::Left(_) => {
                    return Err(anyhow!("package upload cancelled during retry delay"));
                }
                futures::future::Either::Right(_) => policy::check(until)?,
            }
        }
        Err(anyhow!("package upload attempts exhausted"))
    }
}
