//! Explicit target repository provisioning; credential possession alone is not publication authority.
use super::{Plan, Publisher, RepositoryKind, policy};
use anyhow::{Result, anyhow, bail};
use futures::{Future, FutureExt};
use serde::Serialize;
use serde_json::{Value, json};
use std::time::Instant;
#[derive(Serialize)]
pub struct RepositoryReceipt {
    pub repo: String,
    pub mutation_attempted: bool,
    pub existing_conflict: bool,
    pub observed_parent: Option<String>,
    pub completed: bool,
    pub error: Option<String>,
}
impl Publisher {
    /// Mirrors create_repo(exist_ok=true), then requires an authenticated immutable HEAD observation.
    pub async fn ensure_model_repo_until<C: Future<Output = ()>>(
        &self,
        repo: &str,
        confirmed: bool,
        until: Instant,
        cancellation: C,
    ) -> RepositoryReceipt {
        let mut receipt = RepositoryReceipt {
            repo: repo.into(),
            mutation_attempted: false,
            existing_conflict: false,
            observed_parent: None,
            completed: false,
            error: None,
        };
        let mut cancel = Box::pin(cancellation.fuse());
        let result = {
            let pending = tokio::time::timeout_at(
                until.into(),
                self.ensure(repo, confirmed, until, &mut receipt),
            );
            match futures::future::select(cancel.as_mut(), Box::pin(pending)).await {
                futures::future::Either::Left(_) => Err(anyhow!(
                    "repository provisioning cancelled; mutation unconfirmed"
                )),
                futures::future::Either::Right((Err(_), _)) => Err(anyhow!(
                    "repository provisioning deadline; mutation unconfirmed"
                )),
                futures::future::Either::Right((Ok(result), _)) => result,
            }
        };
        let result = if cancel.as_mut().now_or_never().is_some() {
            Err(anyhow!("repository provisioning terminal cancellation"))
        } else {
            policy::check(until).and(result)
        };
        match result {
            Ok(()) => receipt.completed = true,
            Err(error) => receipt.error = Some(error.to_string()),
        };
        receipt
    }
    async fn ensure(
        &self,
        repo: &str,
        confirmed: bool,
        until: Instant,
        receipt: &mut RepositoryReceipt,
    ) -> Result<()> {
        if !confirmed {
            bail!("explicit repository creation confirmation required");
        }
        let plan = Plan {
            repo: repo.into(),
            kind: RepositoryKind::Model,
            revision: "main".into(),
            path: "model-package.json".into(),
            create_pr: false,
            maximum_attempts: 1,
            expected_parent: None,
        };
        plan.validate()?;
        policy::check(until)?;
        let (owner, name) = repo
            .split_once('/')
            .ok_or_else(|| anyhow!("repository identity"))?;
        let mut url = self.client.origin.clone();
        url.set_path("/api/repos/create");
        receipt.mutation_attempted = true;
        let response = self
            .client
            .http
            .post(url)
            .bearer_auth(&self.client.token)
            .json(&json!({"name":name,"organization":owner,"private":false,"type":"model"}))
            .send()
            .await
            .map_err(|_| anyhow!("repository create request unconfirmed"))?;
        if response.status() == reqwest::StatusCode::CONFLICT {
            receipt.existing_conflict = true;
        } else {
            let _: Value = serde_json::from_slice(&self.client.bounded(response, until).await?)
                .map_err(|_| anyhow!("repository create receipt malformed"))?;
        }
        policy::check(until)?;
        receipt.observed_parent = Some(self.head(&plan, until).await?);
        Ok(())
    }
}
