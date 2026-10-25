//! Optional catalog reads classify only an explicit missing entry; source identity is immutable HEAD.
use super::{Plan, Publisher, RepositoryKind, catalog, policy};
use anyhow::{Result, bail};
use serde::Serialize;
use std::time::Instant;
#[derive(Serialize)]
pub struct PreparedCatalog {
    pub observed_parent: String,
    pub entry_path: String,
    pub missing_entry: bool,
    pub bytes: Vec<u8>,
}
impl Publisher {
    pub async fn prepare_catalog_until(
        &self,
        input: &catalog::CatalogInput<'_>,
        until: Instant,
    ) -> Result<PreparedCatalog> {
        tokio::time::timeout_at(until.into(), self.prepare_catalog_inner(input, until))
            .await
            .map_err(|_| anyhow::anyhow!("catalog preparation deadline"))?
    }
    async fn prepare_catalog_inner(
        &self,
        input: &catalog::CatalogInput<'_>,
        until: Instant,
    ) -> Result<PreparedCatalog> {
        let (entry_path, _) = catalog::project(None, input)?;
        let plan = Plan {
            repo: "meshllm/catalog".into(),
            kind: RepositoryKind::Dataset,
            revision: "main".into(),
            path: entry_path.clone(),
            create_pr: false,
            maximum_attempts: 1,
            expected_parent: None,
        };
        let parent = self.head(&plan, until).await?;
        let mut tail = vec!["resolve", parent.as_str()];
        tail.extend(entry_path.split('/'));
        policy::check(until)?;
        let response = tokio::time::timeout_at(
            until.into(),
            self.client
                .http
                .get(self.url(&plan, &tail, false)?)
                .bearer_auth(&self.client.token)
                .send(),
        )
        .await
        .map_err(|_| anyhow::anyhow!("catalog read deadline"))?
        .map_err(|_| anyhow::anyhow!("catalog read failed"))?;
        let missing = response.status() == reqwest::StatusCode::NOT_FOUND
            && response
                .headers()
                .get("x-error-code")
                .is_some_and(|h| h == "EntryNotFound");
        let existing = if missing {
            None
        } else {
            if response.status() != reqwest::StatusCode::OK {
                bail!("catalog HTTP refusal; entry absence not confirmed");
            }
            Some(
                serde_json::from_slice(&bounded_catalog(response, until).await?)
                    .map_err(|_| anyhow::anyhow!("catalog JSON refused"))?,
            )
        };
        let (entry_path, projected) = catalog::project(existing, input)?;
        let bytes = serde_json::to_vec_pretty(&projected)?;
        if bytes.len() > 1024 * 1024 {
            bail!("projected catalog size refused");
        }
        policy::check(until)?;
        Ok(PreparedCatalog {
            observed_parent: parent,
            entry_path,
            missing_entry: missing,
            bytes,
        })
    }
}
async fn bounded_catalog(mut response: reqwest::Response, until: Instant) -> Result<Vec<u8>> {
    const CAP: usize = 1024 * 1024;
    if response.content_length().is_some_and(|n| n > CAP as u64) {
        bail!("catalog body size refused");
    }
    let mut bytes = Vec::new();
    while let Some(chunk) = tokio::time::timeout_at(until.into(), response.chunk())
        .await
        .map_err(|_| anyhow::anyhow!("catalog body deadline"))?
        .map_err(|_| anyhow::anyhow!("catalog body failed"))?
    {
        if chunk.len() > CAP.saturating_sub(bytes.len()) {
            bail!("catalog body size refused");
        }
        bytes.extend_from_slice(&chunk);
    }
    policy::check(until)?;
    Ok(bytes)
}
