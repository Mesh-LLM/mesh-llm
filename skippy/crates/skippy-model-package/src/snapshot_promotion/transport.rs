//! Immutable-source copy transport for parent-bound snapshot publication.
use super::{
    commit_payload::{self, CopyContent},
    policy::{PromotionPlan, SnapshotPlan, SnapshotTransport},
};
use anyhow::{Context, Result, bail};
use hf_hub::{HFClient, repository::RepoTreeEntry};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Write},
    path::{Path, PathBuf},
    time::Duration,
};

struct StagedContents {
    files: BTreeMap<String, CopyContent>,
    _directory: tempfile::TempDir,
}

fn payload_body(file: File) -> reqwest::Body {
    let stream = futures::stream::try_unfold(file, |mut file| async move {
        let mut buffer = vec![0_u8; 64 * 1024];
        let read = file.read(&mut buffer)?;
        if read == 0 {
            return Ok::<_, std::io::Error>(None);
        }
        buffer.truncate(read);
        Ok(Some((bytes::Bytes::from(buffer), file)))
    });
    reqwest::Body::wrap_stream(stream)
}

pub struct HubTransport {
    runtime: tokio::runtime::Runtime,
    client: HFClient,
    http: reqwest::Client,
    repo: String,
    token: Option<String>,
}

impl HubTransport {
    pub fn new(repo: String) -> Result<Self> {
        let parts = repo.split('/').collect::<Vec<_>>();
        if parts.len() != 2
            || parts.iter().any(|part| {
                part.is_empty()
                    || matches!(*part, "." | "..")
                    || part
                        .chars()
                        .any(|ch| !ch.is_ascii_alphanumeric() && !matches!(ch, '-' | '_' | '.'))
            })
        {
            bail!("promotion repository must be an owner/name coordinate");
        }
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        let client = crate::build_hf_client()?;
        let endpoint = reqwest::Url::parse(client.endpoint())?;
        if !endpoint.username().is_empty()
            || endpoint.password().is_some()
            || endpoint.query().is_some()
            || endpoint.fragment().is_some()
        {
            bail!("promotion endpoint cannot contain credentials, query or fragment");
        }
        let loopback = endpoint.host_str().is_some_and(|host| {
            host == "localhost"
                || host
                    .parse::<std::net::IpAddr>()
                    .is_ok_and(|ip| ip.is_loopback())
        });
        if endpoint.scheme() != "https" && !(endpoint.scheme() == "http" && loopback) {
            bail!("promotion endpoint must use HTTPS or loopback HTTP");
        }
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .redirect(reqwest::redirect::Policy::none())
            .build()?;
        Ok(Self {
            runtime,
            client,
            http,
            repo,
            token: skippy_model_hf::hf_token_override(),
        })
    }

    fn url(&self, api: bool, tail: &[&str]) -> Result<reqwest::Url> {
        let mut url = reqwest::Url::parse(self.client.endpoint())?;
        let mut segments = url
            .path_segments_mut()
            .map_err(|_| anyhow::anyhow!("invalid promotion endpoint"))?;
        segments.pop_if_empty();
        if api {
            segments.extend(["api", "models"]);
        }
        segments.extend(self.repo.split('/'));
        segments.extend(tail.iter().copied());
        drop(segments);
        Ok(url)
    }

    fn authorize(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.token {
            Some(token) => request.bearer_auth(token),
            None => request,
        }
    }

    async fn read_response(&self, mut url: reqwest::Url) -> Result<reqwest::Response> {
        let endpoint = reqwest::Url::parse(self.client.endpoint())?;
        for _ in 0..8 {
            if !url.username().is_empty() || url.password().is_some() {
                bail!("staged regular file redirect contains credentials");
            }
            if url.scheme() != "https" && url.origin() != endpoint.origin() {
                bail!("staged regular file redirect would downgrade transport");
            }
            let request = self.http.get(url.clone());
            let request = if url.origin() == endpoint.origin() {
                self.authorize(request)
            } else {
                request
            };
            let response = request
                .send()
                .await
                .map_err(|_| anyhow::anyhow!("staged regular file request failed"))?;
            if response.status().is_redirection() {
                let location = response
                    .headers()
                    .get(reqwest::header::LOCATION)
                    .context("staged regular file redirect lacks Location")?
                    .to_str()?;
                url = url.join(location)?;
                continue;
            }
            if !response.status().is_success() {
                bail!(
                    "staged regular file returned HTTP {}",
                    response.status().as_u16()
                );
            }
            return Ok(response);
        }
        bail!("staged regular file redirect limit exceeded")
    }

    async fn store_regular(
        &self,
        source: &str,
        path: &str,
        expected_size: u64,
        destination: &Path,
    ) -> Result<PathBuf> {
        let mut segments = vec!["resolve", source];
        segments.extend(path.split('/'));
        let mut response = self.read_response(self.url(false, &segments)?).await?;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(destination)?;
        let mut written = 0_u64;
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|_| anyhow::anyhow!("staged regular file stream failed"))?
        {
            written = written
                .checked_add(u64::try_from(chunk.len())?)
                .context("staged regular file size overflow")?;
            if written > expected_size {
                bail!("staged regular file exceeds declared size");
            }
            file.write_all(&chunk)?;
        }
        if written != expected_size {
            bail!("staged regular file differs from declared size");
        }
        file.flush()?;
        Ok(destination.to_owned())
    }

    async fn contents(&self, plan: &PromotionPlan) -> Result<StagedContents> {
        let model = self.client.model(
            self.repo.split_once('/').expect("validated coordinate").0,
            self.repo.split_once('/').expect("validated coordinate").1,
        );
        let source = model
            .info()
            .revision(plan.staging_revision())
            .send()
            .await?
            .sha
            .context("staging revision did not resolve to an immutable commit")?;
        if source.len() != 40
            || !source
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            bail!("staging revision is not an immutable SHA");
        }
        let mut contents = BTreeMap::new();
        let directory = tempfile::tempdir().context("create staging-content spool")?;
        for batch in plan.paths().chunks(500) {
            let entries = model
                .get_paths_info()
                .paths(batch.to_vec())
                .revision(source.clone())
                .send()
                .await?;
            for entry in entries {
                let RepoTreeEntry::File {
                    path, size, lfs, ..
                } = entry
                else {
                    bail!("staged catalog entry is not a file");
                };
                if !batch.contains(&path) {
                    bail!("Hub returned an unrequested staged path");
                }
                let content = if let Some(lfs) = lfs {
                    CopyContent::Lfs {
                        sha256: lfs.sha256.context("staged LFS file lacks identity")?,
                        byte_size: lfs.size.context("staged LFS file lacks byte size")?,
                    }
                } else {
                    let expected = plan
                        .identity(&path)
                        .context("catalog entry lacks identity")?;
                    if size != expected.byte_size {
                        bail!("staged file identity differs from local package catalog");
                    }
                    let destination = directory.path().join(contents.len().to_string());
                    CopyContent::File(
                        self.store_regular(&source, &path, size, &destination)
                            .await?,
                    )
                };
                if contents.insert(path, content).is_some() {
                    bail!("duplicate staged catalog path");
                }
            }
        }
        Ok(StagedContents {
            files: contents,
            _directory: directory,
        })
    }

    async fn publish_async(&self, plan: &PromotionPlan) -> Result<()> {
        let contents = self.contents(plan).await?;
        let mut payload = tempfile::NamedTempFile::new().context("create commit payload spool")?;
        commit_payload::write_payload(plan, &contents.files, payload.as_file_mut())?;
        let payload_size = payload.as_file().metadata()?.len();
        let body = payload_body(payload.reopen()?);
        let request = self
            .http
            .post(self.url(true, &["commit", "main"])?)
            .header("Content-Type", "application/x-ndjson")
            .header(reqwest::header::CONTENT_LENGTH, payload_size.to_string())
            .body(body);
        // Do not blindly retry an uncertain mutation. Retain staging on any
        // unconfirmed result, so a later operator can reconcile the main commit.
        let mut response = self.authorize(request).send().await.map_err(|_| {
            anyhow::anyhow!("promotion request outcome is unconfirmed; staging retained")
        })?;
        if !response.status().is_success() {
            bail!(
                "promotion returned HTTP {}; publication is unconfirmed and staging retained",
                response.status().as_u16()
            );
        }
        let mut receipt = Vec::new();
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|_| anyhow::anyhow!("promotion receipt stream failed; staging retained"))?
        {
            if receipt
                .len()
                .checked_add(chunk.len())
                .is_none_or(|size| size > 64 * 1024)
            {
                bail!("promotion receipt exceeds 64 KiB; staging retained");
            }
            receipt.extend_from_slice(&chunk);
        }
        let commit: serde_json::Value = serde_json::from_slice(&receipt)
            .context("promotion receipt is unconfirmed; staging retained")?;
        let oid = commit["commitOid"]
            .as_str()
            .context("promotion receipt lacks commitOid; staging retained")?;
        if oid.len() != 40
            || !oid
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            bail!("promotion receipt has invalid commit identity; staging retained");
        }
        Ok(())
    }
}

impl SnapshotTransport for HubTransport {
    fn main_revision(&mut self) -> Result<String> {
        self.runtime
            .block_on(
                self.client
                    .model(
                        self.repo.split_once('/').expect("validated coordinate").0,
                        self.repo.split_once('/').expect("validated coordinate").1,
                    )
                    .info()
                    .revision("main")
                    .send(),
            )?
            .sha
            .context("target main did not resolve to an immutable commit")
    }
    fn create_staging(&mut self, plan: &SnapshotPlan) -> Result<()> {
        self.runtime.block_on(
            self.client
                .model(
                    self.repo.split_once('/').expect("validated coordinate").0,
                    self.repo.split_once('/').expect("validated coordinate").1,
                )
                .create_branch()
                .branch(plan.staging_revision())
                .revision(plan.parent_commit())
                .send(),
        )?;
        Ok(())
    }
    fn publish(&mut self, plan: &PromotionPlan) -> Result<()> {
        self.runtime.block_on(self.publish_async(plan))
    }
    fn delete_staging(&mut self, staging_revision: &str) -> Result<()> {
        self.runtime.block_on(
            self.client
                .model(
                    self.repo.split_once('/').expect("validated coordinate").0,
                    self.repo.split_once('/').expect("validated coordinate").1,
                )
                .delete_branch()
                .branch(staging_revision)
                .send(),
        )?;
        Ok(())
    }
}

#[cfg(test)]
#[path = "transport_tests.rs"]
mod tests;
