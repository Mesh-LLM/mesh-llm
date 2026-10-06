//! Ordered shard+sidecar publication. Uploaded objects alone never constitute completion.
mod admission;
mod verification;
use super::{commit_payload, lfs_transfer, policy::ArtifactIdentity, regular_publication};
use anyhow::{Result, anyhow};
use futures::{Future, FutureExt};
use serde::Serialize;
use std::{
    io::{Seek, Write},
    time::Instant,
};
pub struct Shard {
    pub path_in_repo: String,
    pub object: lfs_transfer::Object,
}
pub struct Input {
    pub repo: String,
    pub parent_commit: String,
    pub shards: Vec<Shard>,
    pub sidecars: Vec<regular_publication::LocalFile>,
}
#[derive(Serialize)]
pub struct Receipt {
    pub schema_version: u32,
    pub repo: String,
    pub parent_commit: String,
    pub ordered_paths: Vec<String>,
    pub objects: Vec<lfs_transfer::Receipt>,
    pub object_attempted_paths: Vec<String>,
    pub commit_attempted: bool,
    pub commit_oid: Option<String>,
    pub remote_verified_paths: Vec<String>,
    pub final_source_custody_verified: bool,
    pub completed: bool,
    pub error: Option<String>,
}
pub struct Publisher {
    client: lfs_transfer::Client,
}
impl Publisher {
    pub fn new(token: String) -> Result<Self> {
        Ok(Self {
            client: lfs_transfer::Client::new(token)?,
        })
    }
    pub async fn publish_until<C: Future<Output = ()>>(
        &self,
        input: Input,
        until: Instant,
        cancellation: C,
    ) -> Receipt {
        self.publish_observed_until(input, until, cancellation, &mut |_| Ok(()))
            .await
    }
    pub fn admit_until(input: &mut Input, until: Instant) -> Result<()> {
        admission::validate(input, until)?;
        if !input.sidecars.is_empty() {
            let plan = regular_publication::Plan {
                repo: input.repo.clone(),
                parent_commit: input.parent_commit.clone(),
                paths: input
                    .sidecars
                    .iter()
                    .map(|file| file.path_in_repo.clone())
                    .collect(),
            };
            let copies = input
                .sidecars
                .iter()
                .map(|file| {
                    Ok(regular_publication::LocalFile {
                        path_in_repo: file.path_in_repo.clone(),
                        file: file.file.try_clone()?,
                        identity: ArtifactIdentity {
                            byte_size: file.identity.byte_size,
                            sha256: file.identity.sha256.clone(),
                        },
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let mut staged = regular_publication::contract::Staged::new(&plan, copies, until)?;
            staged.recheck(until)?;
        }
        for shard in &mut input.shards {
            shard.object.verify(until)?;
        }
        regular_publication::contract::check(until)
    }
    pub async fn publish_observed_until<C: Future<Output = ()>>(
        &self,
        input: Input,
        until: Instant,
        cancellation: C,
        observer: &mut dyn FnMut(&Receipt) -> Result<()>,
    ) -> Receipt {
        let mut receipt = Receipt {
            schema_version: 1,
            repo: input.repo.clone(),
            parent_commit: input.parent_commit.clone(),
            ordered_paths: input
                .shards
                .iter()
                .map(|s| s.path_in_repo.clone())
                .chain(input.sidecars.iter().map(|s| s.path_in_repo.clone()))
                .collect(),
            objects: Vec::new(),
            object_attempted_paths: Vec::new(),
            commit_attempted: false,
            commit_oid: None,
            remote_verified_paths: Vec::new(),
            final_source_custody_verified: false,
            completed: false,
            error: None,
        };
        let mut cancellation = Box::pin(cancellation.fuse());
        let result = {
            let operation = tokio::time::timeout_at(
                until.into(),
                self.execute(input, until, cancellation.as_mut(), &mut receipt, observer),
            );
            match operation.await {
                Ok(result) => result,
                Err(_) => Err(anyhow!(
                    "model publication deadline expired; mutation may be unconfirmed"
                )),
            }
        };
        let result = if cancellation.as_mut().now_or_never().is_some() {
            Err(anyhow!("model publication cancelled at terminal boundary"))
        } else {
            regular_publication::contract::check(until).and(result)
        };
        match result {
            Ok(()) => receipt.completed = true,
            Err(error) => receipt.error = Some(error.to_string()),
        };
        receipt
    }
    async fn execute<C: Future<Output = ()>>(
        &self,
        mut input: Input,
        until: Instant,
        mut cancellation: std::pin::Pin<&mut C>,
        receipt: &mut Receipt,
        observer: &mut dyn FnMut(&Receipt) -> Result<()>,
    ) -> Result<()> {
        if cancellation.as_mut().now_or_never().is_some() {
            return Err(anyhow!("model publication pre-cancelled"));
        }
        admission::validate(&mut input, until)?;
        let sidecar_plan = regular_publication::Plan {
            repo: input.repo.clone(),
            parent_commit: input.parent_commit.clone(),
            paths: input
                .sidecars
                .iter()
                .map(|f| f.path_in_repo.clone())
                .collect(),
        };
        let mut sidecars = regular_publication::contract::Staged::new(
            &sidecar_plan,
            std::mem::take(&mut input.sidecars),
            until,
        )?;
        for shard in &mut input.shards {
            let object = lfs_transfer::Object {
                file: shard.object.file.try_clone()?,
                oid: shard.object.oid.clone(),
                size: shard.object.size,
            };
            receipt
                .object_attempted_paths
                .push(shard.path_in_repo.clone());
            observer(receipt)?;
            let observed = self
                .client
                .upload_until(&input.repo, object, until, cancellation.as_mut())
                .await;
            let complete = observed.completed;
            receipt.objects.push(observed);
            observer(receipt)?;
            if !complete {
                return Err(anyhow!(
                    "model object transfer incomplete; repository commit not attempted"
                ));
            }
        }
        let mut payload = tempfile::NamedTempFile::new()?;
        serde_json::to_writer(
            payload.as_file_mut(),
            &serde_json::json!({"key":"header","value":{"summary":"Publish pinned ordered model shards and sidecars","description":"","parentCommit":input.parent_commit}}),
        )?;
        payload.write_all(b"\n")?;
        for shard in &input.shards {
            regular_publication::contract::check(until)?;
            commit_payload::write_lfs_row(
                &shard.path_in_repo,
                &ArtifactIdentity {
                    byte_size: shard.object.size,
                    sha256: shard.object.oid.clone(),
                },
                payload.as_file_mut(),
            )?;
        }
        for file in &sidecar_plan.paths {
            let artifact = sidecars
                .files
                .get_mut(file)
                .ok_or_else(|| anyhow!("sidecar staged roster incomplete"))?;
            artifact.spool.as_file_mut().rewind()?;
            commit_payload::write_regular_row(
                file,
                &artifact.identity,
                artifact.spool.as_file_mut(),
                payload.as_file_mut(),
            )?;
        }
        payload.flush()?;
        payload.as_file().sync_all()?;
        let bytes = payload.as_file().metadata()?.len();
        if bytes > 12 * 1024 * 1024 {
            return Err(anyhow!("model commit payload bound exceeded"));
        }
        let mut pending = Box::pin(async {
            regular_publication::contract::check(until)?;
            let request = self
                .client
                .http
                .post(self.url(&input.repo, &["commit", "main"], true)?)
                .bearer_auth(&self.client.token)
                .header("Content-Type", "application/x-ndjson")
                .header("Content-Length", bytes.to_string())
                .body(super::regular_publication::exchange::file_body(
                    payload.reopen()?,
                ));
            receipt.commit_attempted = true;
            observer(receipt)?;
            let response = request
                .send()
                .await
                .map_err(|_| anyhow!("model commit outcome unconfirmed"))?;
            let value: serde_json::Value =
                serde_json::from_slice(&self.client.bounded(response, until).await?)
                    .map_err(|_| anyhow!("model commit receipt malformed"))?;
            let oid = value["commitOid"]
                .as_str()
                .filter(|oid| regular_publication::contract::hex(oid, 40))
                .ok_or_else(|| anyhow!("model commit immutable identity invalid"))?
                .to_owned();
            receipt.commit_oid = Some(oid.clone());
            observer(receipt)?;
            for shard in &input.shards {
                self.verify(
                    &input.repo,
                    &oid,
                    &shard.path_in_repo,
                    &ArtifactIdentity {
                        byte_size: shard.object.size,
                        sha256: shard.object.oid.clone(),
                    },
                    until,
                )
                .await?;
                receipt
                    .remote_verified_paths
                    .push(shard.path_in_repo.clone());
                observer(receipt)?;
            }
            for path in &sidecar_plan.paths {
                self.verify(
                    &input.repo,
                    &oid,
                    path,
                    &sidecars.files[path].identity,
                    until,
                )
                .await?;
                receipt.remote_verified_paths.push(path.clone());
                observer(receipt)?;
            }
            for shard in &mut input.shards {
                shard.object.verify(until)?;
            }
            sidecars.recheck(until)?;
            regular_publication::contract::check(until)?;
            receipt.final_source_custody_verified = true;
            observer(receipt)?;
            Ok(())
        });
        match futures::future::select(cancellation.as_mut(), pending.as_mut()).await {
            futures::future::Either::Right((result, _)) => result,
            futures::future::Either::Left(_) => Err(anyhow!(
                "model publication cancelled; mutation may be unconfirmed"
            )),
        }
    }
    fn url(&self, repo: &str, tail: &[&str], api: bool) -> Result<reqwest::Url> {
        let mut url = self.client.origin.clone();
        {
            let mut parts = url
                .path_segments_mut()
                .map_err(|_| anyhow!("model publication origin refused"))?;
            parts.pop_if_empty();
            if api {
                parts.extend(["api", "models"]);
            }
            parts.extend(repo.split('/'));
            parts.extend(tail.iter().copied());
        }
        Ok(url)
    }
}

#[cfg(test)]
#[path = "model_publication/tests.rs"]
mod tests;
