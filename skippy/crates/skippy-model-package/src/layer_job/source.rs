use super::{METADATA_NAMES, repo, revision};
use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, time::Instant};
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Source {
    pub repo: String,
    pub revision: String,
    pub metadata_sha256: BTreeMap<String, String>,
    pub missing: Vec<String>,
    pub license: Option<String>,
}
#[derive(Deserialize)]
struct Info {
    sha: String,
    #[serde(rename = "cardData", default)]
    card_data: serde_json::Value,
}
pub struct SourceClient {
    http: reqwest::Client,
    origin: String,
    token: Option<String>,
}
impl SourceClient {
    pub fn new(token: Option<String>) -> Result<Self> {
        if let Some(value) = &token {
            crate::jobs::transport::token(value)?;
        }
        Ok(Self {
            http: crate::jobs::transport::client()?,
            origin: crate::jobs::transport::origin("https://huggingface.co")?,
            token,
        })
    }
    #[cfg(test)]
    pub(super) fn fixture(origin: String) -> Result<Self> {
        Ok(Self {
            http: crate::jobs::transport::client()?,
            origin,
            token: None,
        })
    }
    pub(super) fn url(&self, parts: &[&str]) -> Result<reqwest::Url> {
        let mut url = reqwest::Url::parse(&self.origin)?;
        url.path_segments_mut()
            .map_err(|_| anyhow::anyhow!("source origin refused"))?
            .extend(parts.iter().copied());
        Ok(url)
    }
    pub(super) async fn request(
        &self,
        url: reqwest::Url,
        until: Instant,
    ) -> Result<reqwest::Response> {
        guard(until)?;
        let mut request = self.http.get(url);
        if let Some(token) = &self.token {
            request = request.bearer_auth(token);
        }
        let response = tokio::time::timeout_at(until.into(), request.send())
            .await
            .map_err(|_| anyhow::anyhow!("source request deadline"))?
            .map_err(|_| anyhow::anyhow!("source request failed"))?;
        guard(until)?;
        Ok(response)
    }
    pub async fn admit_until(
        &self,
        source_repo: &str,
        requested: &str,
        until: Instant,
    ) -> Result<(Source, BTreeMap<String, Vec<u8>>)> {
        repo(source_repo)?;
        if requested.is_empty()
            || requested.len() > 128
            || !requested
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"._-/".contains(&b))
        {
            bail!("source selector refused");
        }
        let coordinate: Vec<_> = source_repo.split('/').collect();
        let response = self
            .request(
                self.url(&[
                    "api",
                    "models",
                    coordinate[0],
                    coordinate[1],
                    "revision",
                    requested,
                ])?,
                until,
            )
            .await?;
        if response.status() != reqwest::StatusCode::OK {
            bail!("source info HTTP refusal");
        }
        let info: Info = serde_json::from_slice(&body(response, 262144, until).await?)
            .map_err(|_| anyhow::anyhow!("source info schema refused"))?;
        revision(&info.sha)?;
        if requested.len() == 40
            && requested.bytes().all(|b| b.is_ascii_hexdigit())
            && info.sha != requested
        {
            bail!("source immutable revision mismatch");
        }
        let license = info
            .card_data
            .get("license")
            .and_then(serde_json::Value::as_str)
            .filter(|s| !s.is_empty() && s.len() <= 256 && !s.chars().any(char::is_control))
            .map(str::to_owned);
        let mut source = Source {
            repo: source_repo.into(),
            revision: info.sha,
            metadata_sha256: BTreeMap::new(),
            missing: Vec::new(),
            license,
        };
        let mut files = BTreeMap::new();
        for name in METADATA_NAMES {
            if let Some(bytes) = self
                .metadata(source_repo, &source.revision, name, until)
                .await?
            {
                use sha2::Digest as _;
                source.metadata_sha256.insert(
                    name.into(),
                    sha2::Sha256::digest(&bytes)
                        .iter()
                        .map(|byte| format!("{byte:02x}"))
                        .collect(),
                );
                files.insert(name.into(), bytes);
            } else {
                source.missing.push(name.into());
            }
        }
        guard(until)?;
        Ok((source, files))
    }
    async fn metadata(
        &self,
        source_repo: &str,
        pin: &str,
        name: &str,
        until: Instant,
    ) -> Result<Option<Vec<u8>>> {
        let parts: Vec<_> = source_repo.split('/').collect();
        let mut url = self.url(&[parts[0], parts[1], "resolve", pin, name])?;
        for _ in 0..4 {
            let response = self.request(url.clone(), until).await?;
            if response.status() == reqwest::StatusCode::NOT_FOUND
                && response
                    .headers()
                    .get("x-error-code")
                    .is_some_and(|value| value == "EntryNotFound")
            {
                return Ok(None);
            }
            if response.status() == reqwest::StatusCode::OK {
                let bytes = body(response, 1024 * 1024, until).await?;
                if name.ends_with(".json") {
                    let _: serde_json::Value = serde_json::from_slice(&bytes)
                        .map_err(|_| anyhow::anyhow!("source metadata JSON refused"))?;
                }
                return Ok(Some(bytes));
            }
            if response.status().is_redirection() {
                let location = response
                    .headers()
                    .get(reqwest::header::LOCATION)
                    .and_then(|v| v.to_str().ok())
                    .filter(|v| v.len() <= 2048)
                    .ok_or_else(|| anyhow::anyhow!("metadata redirect refused"))?;
                let next = url
                    .join(location)
                    .map_err(|_| anyhow::anyhow!("metadata redirect refused"))?;
                let allowed = format!("/api/resolve-cache/models/{source_repo}/{pin}/{name}");
                if next.origin() != url.origin()
                    || !next.username().is_empty()
                    || next.password().is_some()
                    || next.fragment().is_some()
                    || next.path() != allowed
                {
                    bail!("metadata redirect origin/path refused");
                }
                url = next;
                continue;
            }
            bail!("optional metadata HTTP refusal");
        }
        bail!("metadata redirect budget refused")
    }
}
pub(super) fn guard(until: Instant) -> Result<()> {
    if Instant::now() >= until {
        bail!("source deadline expired");
    }
    Ok(())
}
pub(super) async fn body(
    mut response: reqwest::Response,
    cap: usize,
    until: Instant,
) -> Result<Vec<u8>> {
    if response.content_length().is_some_and(|n| n > cap as u64) {
        bail!("source response size refused");
    }
    let mut bytes = Vec::new();
    loop {
        let chunk = tokio::time::timeout_at(until.into(), response.chunk())
            .await
            .map_err(|_| anyhow::anyhow!("source body deadline"))?
            .map_err(|_| anyhow::anyhow!("source body failed"))?;
        let Some(chunk) = chunk else { break };
        if chunk.len() > cap.saturating_sub(bytes.len()) {
            bail!("source body size refused");
        }
        bytes.extend_from_slice(&chunk);
    }
    guard(until)?;
    Ok(bytes)
}
