use anyhow::{Context, Result, bail};
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};

use crate::source_ref::{GitHubPluginSource, PluginVersion};

pub const USER_AGENT: &str = "mesh-llm-plugin-manager";

#[derive(Debug, Clone)]
pub struct GitHubReleaseClient {
    client: Client,
}

impl GitHubReleaseClient {
    pub fn new() -> Result<Self> {
        Ok(Self {
            client: Client::builder()
                .user_agent(USER_AGENT)
                .build()
                .context("build GitHub release HTTP client")?,
        })
    }

    pub fn http_client(&self) -> &Client {
        &self.client
    }

    pub async fn resolve_release(
        &self,
        source: &GitHubPluginSource,
        version: Option<&PluginVersion>,
    ) -> Result<GitHubRelease> {
        match version {
            Some(version) => self.release_by_version(source, version).await,
            None => self.latest_release(source).await,
        }
    }

    async fn latest_release(&self, source: &GitHubPluginSource) -> Result<GitHubRelease> {
        let url = format!(
            "https://api.github.com/repos/{}/releases/latest",
            source.repo_slug()
        );
        self.get_release(&url)
            .await
            .with_context(|| format!("resolve latest GitHub release for {}", source.repo_slug()))
    }

    async fn release_by_version(
        &self,
        source: &GitHubPluginSource,
        version: &PluginVersion,
    ) -> Result<GitHubRelease> {
        let mut not_found = Vec::new();
        for segment in version.matching_segments() {
            let url = format!(
                "https://api.github.com/repos/{}/releases/tags/{}",
                source.repo_slug(),
                segment
            );
            match self.get_release(&url).await {
                Ok(release) => return Ok(release),
                Err(error) if error.to_string().contains("not found") => {
                    not_found.push(segment);
                }
                Err(error) => return Err(error),
            }
        }
        bail!(
            "GitHub release not found for {} with tag {}",
            source.repo_slug(),
            not_found.join(" or ")
        )
    }

    async fn get_release(&self, url: &str) -> Result<GitHubRelease> {
        let response = self
            .client
            .get(url)
            .send()
            .await
            .with_context(|| format!("request GitHub release {url}"))?;
        let status = response.status();
        if status == StatusCode::NOT_FOUND {
            bail!("GitHub release not found: {url}");
        }
        if !status.is_success() {
            bail!("GitHub release request failed: {status} {url}");
        }
        response
            .json::<GitHubRelease>()
            .await
            .with_context(|| format!("decode GitHub release {url}"))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GitHubRelease {
    pub tag_name: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub draft: bool,
    #[serde(default)]
    pub prerelease: bool,
    #[serde(default)]
    pub assets: Vec<GitHubReleaseAsset>,
}

impl GitHubRelease {
    pub fn asset_names(&self) -> Vec<String> {
        self.assets.iter().map(|asset| asset.name.clone()).collect()
    }

    pub fn asset_by_name(&self, name: &str) -> Option<&GitHubReleaseAsset> {
        self.assets.iter().find(|asset| asset.name == name)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GitHubReleaseAsset {
    pub name: String,
    pub browser_download_url: String,
    #[serde(default)]
    pub size: Option<u64>,
}
