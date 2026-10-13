//! Shared immutable repository byte transport; an optional file sink retains identical host/token/hash policy.
use super::Client;
use crate::snapshot_promotion::{policy::ArtifactIdentity, regular_publication};
use anyhow::{Result, anyhow, bail};
use sha2::{Digest, Sha256};
use std::{fs::File, io::Write, time::Instant};
pub(crate) struct RepositoryRead<'a> {
    pub repo: &'a str,
    pub dataset: bool,
    pub commit: &'a str,
    pub path: &'a str,
    pub size: u64,
    pub expected_sha256: Option<&'a str>,
}
impl Client {
    pub(in crate::snapshot_promotion) async fn verify_repository_until(
        &self,
        repo: &str,
        dataset: bool,
        oid: &str,
        path: &str,
        identity: &ArtifactIdentity,
        until: Instant,
    ) -> Result<()> {
        self.read_repository_until(
            RepositoryRead {
                repo,
                dataset,
                commit: oid,
                path,
                size: identity.byte_size,
                expected_sha256: Some(&identity.sha256),
            },
            None,
            until,
        )
        .await?;
        Ok(())
    }
    pub(crate) async fn acquire_repository_until(
        &self,
        input: RepositoryRead<'_>,
        file: &mut File,
        until: Instant,
    ) -> Result<ArtifactIdentity> {
        self.read_repository_until(input, Some(file), until).await
    }
    async fn read_repository_until(
        &self,
        input: RepositoryRead<'_>,
        sink: Option<&mut File>,
        until: Instant,
    ) -> Result<ArtifactIdentity> {
        super::custody::repo(input.repo)?;
        if !regular_publication::contract::hex(input.commit, 40)
            || input.size == 0
            || input.size > 1024_u64.pow(4)
            || input
                .expected_sha256
                .is_some_and(|s| !regular_publication::contract::hex(s, 64))
            || input.path.len() > 4096
            || input
                .path
                .split('/')
                .any(|s| s.is_empty() || matches!(s, "." | "..") || s.chars().any(char::is_control))
        {
            bail!("immutable repository read admission refused");
        }
        tokio::time::timeout_at(until.into(), self.read_repository(input, sink, until))
            .await
            .map_err(|_| anyhow!("immutable repository read deadline"))?
    }
    async fn read_repository(
        &self,
        input: RepositoryRead<'_>,
        mut sink: Option<&mut File>,
        until: Instant,
    ) -> Result<ArtifactIdentity> {
        let mut url = self.origin.clone();
        {
            let mut parts = url
                .path_segments_mut()
                .map_err(|_| anyhow!("verification origin refused"))?;
            parts.pop_if_empty();
            if input.dataset {
                parts.push("datasets");
            }
            parts.extend(input.repo.split('/'));
            parts.extend(["resolve", input.commit]);
            parts.extend(input.path.split('/'));
        }
        for _ in 0..8 {
            regular_publication::contract::check(until)?;
            url = self.address(url.as_str())?;
            let request = self.http.get(url.clone());
            let request = if url.origin() == self.origin.origin() {
                request.bearer_auth(&self.token)
            } else {
                request
            };
            let mut response = request
                .send()
                .await
                .map_err(|_| anyhow!("immutable model verification request failed"))?;
            if response.status().is_redirection() {
                let location = response
                    .headers()
                    .get(reqwest::header::LOCATION)
                    .and_then(|v| v.to_str().ok())
                    .filter(|s| s.len() <= 8192)
                    .ok_or_else(|| anyhow!("model redirect location refused"))?;
                url = url
                    .join(location)
                    .map_err(|_| anyhow!("model redirect malformed"))?;
                continue;
            }
            if response.status() != reqwest::StatusCode::OK {
                bail!(
                    "immutable model verification HTTP status {}",
                    response.status().as_u16()
                );
            }
            let mut hash = Sha256::new();
            let mut size = 0u64;
            while let Some(chunk) = response
                .chunk()
                .await
                .map_err(|_| anyhow!("immutable model verification stream failed"))?
            {
                regular_publication::contract::check(until)?;
                size = size
                    .checked_add(chunk.len() as u64)
                    .ok_or_else(|| anyhow!("model verification byte overflow"))?;
                if size > input.size {
                    bail!("immutable model verification exceeds expected size");
                }
                hash.update(&chunk);
                if let Some(file) = sink.as_mut() {
                    file.write_all(&chunk)?;
                }
            }
            let sha256: String = hash.finalize().iter().map(|b| format!("{b:02x}")).collect();
            if size != input.size
                || input
                    .expected_sha256
                    .is_some_and(|expected| expected != sha256)
            {
                bail!("immutable model verification byte identity mismatch");
            }
            if let Some(file) = sink.as_mut() {
                file.flush()?;
                file.sync_all()?;
            }
            regular_publication::contract::check(until)?;
            return Ok(ArtifactIdentity {
                byte_size: size,
                sha256,
            });
        }
        bail!("immutable model verification redirect count refused")
    }
}
