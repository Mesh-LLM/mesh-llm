//! Read-only exact immutable artifact verification for remote resume and final-commit custody.
use super::{Artifact, Plan, Publisher, RepositoryKind, policy};
use anyhow::{Result, anyhow};
use futures::{Future, FutureExt};
use serde::Serialize;
use std::time::Instant;
#[derive(Serialize)]
pub struct VerificationReceipt {
    pub repo: String,
    pub commit: String,
    pub path: String,
    pub identity: crate::snapshot_promotion::policy::ArtifactIdentity,
    pub local_custody_verified: bool,
    pub completed: bool,
    pub error: Option<String>,
}
impl Publisher {
    /// A supplied immutable commit is verified against an owned local byte identity.
    /// No head lookup, repository creation, upload, PR or unlink is performed.
    pub async fn verify_artifact_until<C: Future<Output = ()>>(
        &self,
        plan: &Plan,
        mut artifact: Artifact,
        until: Instant,
        cancellation: C,
    ) -> VerificationReceipt {
        let mut receipt = VerificationReceipt {
            repo: plan.repo.clone(),
            commit: plan.revision.clone(),
            path: plan.path.clone(),
            identity: artifact.identity.clone(),
            local_custody_verified: false,
            completed: false,
            error: None,
        };
        let mut cancel = Box::pin(cancellation.fuse());
        let pending = async {
            plan.validate()?;
            if plan.kind != RepositoryKind::Model
                || plan.create_pr
                || artifact.unlink_path.is_some()
                || !crate::snapshot_promotion::regular_publication::contract::hex(
                    &plan.revision,
                    40,
                )
            {
                return Err(anyhow!(
                    "read-only model resume requires immutable commit and preserved local file"
                ));
            }
            artifact.verify(until)?;
            self.client
                .verify_repository_until(
                    &plan.repo,
                    false,
                    &plan.revision,
                    &plan.path,
                    &artifact.identity,
                    until,
                )
                .await?;
            artifact.verify(until)?;
            Ok(())
        };
        let result: Result<()> = match futures::future::select(
            cancel.as_mut(),
            Box::pin(tokio::time::timeout_at(until.into(), pending)),
        )
        .await
        {
            futures::future::Either::Left(_) => {
                Err(anyhow!("immutable resume verification cancelled"))
            }
            futures::future::Either::Right((Err(_), _)) => {
                Err(anyhow!("immutable resume verification deadline"))
            }
            futures::future::Either::Right((Ok(result), _)) => result,
        };
        let result = if cancel.as_mut().now_or_never().is_some() {
            Err(anyhow!("immutable resume final cancellation"))
        } else {
            policy::check(until).and(result)
        };
        match result {
            Ok(()) => {
                receipt.local_custody_verified = true;
                receipt.completed = true;
            }
            Err(error) => receipt.error = Some(error.to_string()),
        }
        receipt
    }
}
