use super::{Artifact, Attempt, Plan, Publisher, RepositoryKind, lfs_transfer, policy};
use crate::snapshot_promotion::{commit_payload, regular_publication};
use anyhow::{Result, anyhow, bail};
use base64::Engine as _;
use futures::{Future, FutureExt};
use serde_json::{Value, json};
use std::{
    io::{Read, Seek, Write},
    pin::Pin,
    time::Instant,
};
async fn owned<C: Future<Output = ()>, T>(
    operation: impl Future<Output = Result<T>>,
    until: Instant,
    mut cancel: Pin<&mut C>,
) -> Result<T> {
    policy::check(until)?;
    if cancel.as_mut().now_or_never().is_some() {
        return Err(anyhow!("package upload pre-cancelled"));
    }
    let future = tokio::time::timeout_at(until.into(), operation);
    match futures::future::select(cancel, Box::pin(future)).await {
        futures::future::Either::Left(_) => Err(anyhow!(
            "package upload cancelled; mutation may be unconfirmed"
        )),
        futures::future::Either::Right((Err(_), _)) => Err(anyhow!(
            "package upload deadline; mutation may be unconfirmed"
        )),
        futures::future::Either::Right((Ok(result), _)) => {
            policy::check(until)?;
            result
        }
    }
}
impl Publisher {
    pub(super) fn url(&self, plan: &Plan, tail: &[&str], api: bool) -> Result<reqwest::Url> {
        let mut url = self.client.origin.clone();
        {
            let mut parts = url
                .path_segments_mut()
                .map_err(|_| anyhow!("package origin refused"))?;
            parts.pop_if_empty();
            if api {
                parts.extend(["api", plan.kind.api()]);
            } else {
                parts.extend(plan.kind.prefix().iter().copied());
            }
            parts.extend(plan.repo.split('/'));
            parts.extend(tail.iter().copied());
        }
        Ok(url)
    }
    pub(super) async fn head(&self, plan: &Plan, until: Instant) -> Result<String> {
        let response = self
            .client
            .http
            .get(self.url(plan, &["revision", &plan.revision], true)?)
            .bearer_auth(&self.client.token)
            .send()
            .await
            .map_err(|_| anyhow!("package parent lookup failed"))?;
        let value: Value = serde_json::from_slice(&self.client.bounded(response, until).await?)
            .map_err(|_| anyhow!("package parent response malformed"))?;
        value["sha"]
            .as_str()
            .filter(|s| regular_publication::contract::hex(s, 40))
            .map(str::to_owned)
            .ok_or_else(|| anyhow!("package parent immutable identity absent"))
    }
    async fn classify(
        &self,
        plan: &Plan,
        artifact: &mut Artifact,
        until: Instant,
    ) -> Result<String> {
        artifact.file.rewind()?;
        let mut sample = [0; 512];
        let n = artifact.file.read(&mut sample)?;
        let response=self.client.http.post(self.url(plan,&["preupload",&plan.revision],true)?).bearer_auth(&self.client.token)
            .json(&json!({"files":[{"path":plan.path,"size":artifact.identity.byte_size,"sample":base64::engine::general_purpose::STANDARD.encode(&sample[..n])}]})).send().await.map_err(|_|anyhow!("package classification failed"))?;
        let value: Value = serde_json::from_slice(&self.client.bounded(response, until).await?)
            .map_err(|_| anyhow!("package classification malformed"))?;
        let rows = value["files"]
            .as_array()
            .filter(|v| v.len() == 1)
            .ok_or_else(|| anyhow!("package classification roster refused"))?;
        let row = &rows[0];
        if row["path"] != plan.path || row["shouldIgnore"].as_bool() == Some(true) {
            bail!("package classification path/refusal");
        }
        match row["uploadMode"].as_str() {
            Some("regular") if artifact.identity.byte_size <= 1024 * 1024 => Ok("regular".into()),
            Some("lfs") if plan.kind == RepositoryKind::Model => Ok("lfs".into()),
            _ => bail!("package upload mode/size unsupported; no large inline fallback"),
        }
    }
    pub(super) async fn attempt<C: Future<Output = ()>>(
        &self,
        plan: &Plan,
        artifact: &mut Artifact,
        until: Instant,
        mut cancel: Pin<&mut C>,
        receipt: &mut Attempt,
    ) -> Result<()> {
        let parent = owned(self.head(plan, until), until, cancel.as_mut()).await?;
        receipt.parent_commit = Some(parent.clone());
        // Reconcile a previous uncertain main/staging mutation by actual bytes at current immutable HEAD.
        if receipt.ordinal > 1 && !plan.create_pr {
            let result = owned(
                self.client.verify_repository_until(
                    &plan.repo,
                    plan.kind == RepositoryKind::Dataset,
                    &parent,
                    &plan.path,
                    &artifact.identity,
                    until,
                ),
                until,
                cancel.as_mut(),
            )
            .await;
            if result.is_ok() {
                receipt.commit_oid = Some(parent);
                receipt.remote_verified = true;
                return Ok(());
            }
            policy::check(until)?;
        }
        if plan
            .expected_parent
            .as_ref()
            .is_some_and(|expected| expected != &parent)
        {
            bail!("catalog parent changed before commit; reproject required");
        }
        let mode = owned(self.classify(plan, artifact, until), until, cancel.as_mut()).await?;
        if mode == "lfs" {
            let object = lfs_transfer::Object {
                file: artifact.file.try_clone()?,
                oid: artifact.identity.sha256.clone(),
                size: artifact.identity.byte_size,
            };
            let observed = self
                .client
                .upload_until(&plan.repo, object, until, cancel.as_mut())
                .await;
            let complete = observed.completed && observed.error.is_none();
            receipt.object = Some(observed);
            if !complete {
                bail!("package LFS object transfer incomplete");
            }
        }
        artifact.verify(until)?;
        let mut payload = tempfile::NamedTempFile::new()?;
        serde_json::to_writer(
            payload.as_file_mut(),
            &json!({"key":"header","value":{"summary":format!("Add package artifact {}",plan.path),"description":"","parentCommit":parent}}),
        )?;
        payload.write_all(b"\n")?;
        if mode == "lfs" {
            commit_payload::write_lfs_row(&plan.path, &artifact.identity, payload.as_file_mut())?;
        } else {
            artifact.file.rewind()?;
            commit_payload::write_regular_row(
                &plan.path,
                &artifact.identity,
                &mut artifact.file,
                payload.as_file_mut(),
            )?;
        }
        payload.flush()?;
        payload.as_file().sync_all()?;
        policy::check(until)?;
        let mut request = self
            .client
            .http
            .post(self.url(plan, &["commit", &plan.revision], true)?)
            .bearer_auth(&self.client.token)
            .header("Content-Type", "application/x-ndjson")
            .header(
                "Content-Length",
                payload.as_file().metadata()?.len().to_string(),
            )
            .body(regular_publication::exchange::file_body(payload.reopen()?));
        if plan.create_pr {
            request = request.query(&[("create_pr", "1")]);
        }
        receipt.commit_attempted = true;
        let operation = async {
            let response = request
                .send()
                .await
                .map_err(|_| anyhow!("package commit unconfirmed"))?;
            self.client.bounded(response, until).await
        };
        let value: Value = serde_json::from_slice(&owned(operation, until, cancel.as_mut()).await?)
            .map_err(|_| anyhow!("package commit receipt malformed; mutation unconfirmed"))?;
        let oid = value["commitOid"]
            .as_str()
            .filter(|s| regular_publication::contract::hex(s, 40))
            .ok_or_else(|| anyhow!("package commit identity absent; mutation unconfirmed"))?
            .to_owned();
        receipt.commit_oid = Some(oid.clone());
        owned(
            self.client.verify_repository_until(
                &plan.repo,
                plan.kind == RepositoryKind::Dataset,
                &oid,
                &plan.path,
                &artifact.identity,
                until,
            ),
            until,
            cancel.as_mut(),
        )
        .await?;
        receipt.remote_verified = true;
        artifact.verify(until)
    }
}
