//! Shared executable-provider host lifecycle for language SDK carriers.

use crate::embedded_node::MeshNode;
use anyhow::{Context, Result};
use serde_json::Value;
use std::net::TcpListener;
use std::path::PathBuf;
use std::time::Duration;

/// Typed provider-runtime resources supplied by a host-capable SDK package.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProviderHostConfig {
    pub bundle_roots: Vec<PathBuf>,
    pub release_manifest: Option<PathBuf>,
    pub cache_dir: Option<PathBuf>,
    pub allow_download: bool,
    pub startup_timeout: Duration,
}

impl ProviderHostConfig {
    pub fn new(bundle_roots: impl IntoIterator<Item = impl Into<PathBuf>>) -> Self {
        Self {
            bundle_roots: bundle_roots.into_iter().map(Into::into).collect(),
            startup_timeout: Duration::from_secs(30),
            ..Self::default()
        }
    }
}

/// A provider-only embedded host exposing the normal OpenAI-compatible REST API.
pub struct ProviderHost {
    node: MeshNode,
}

impl ProviderHost {
    pub async fn start(config: ProviderHostConfig) -> Result<Self> {
        anyhow::ensure!(
            !config.bundle_roots.is_empty() || config.release_manifest.is_some(),
            "provider host requires a bundle root or release manifest"
        );
        let api_listener =
            TcpListener::bind(("127.0.0.1", 0)).context("reserve provider host API port")?;
        let api_port = api_listener.local_addr()?.port();
        let console_listener =
            TcpListener::bind(("127.0.0.1", 0)).context("reserve provider host console port")?;
        let console_port = console_listener.local_addr()?.port();
        let mut builder = MeshNode::builder()
            .serve()
            .provider_runtime_roots(config.bundle_roots)
            .allow_provider_runtime_downloads(config.allow_download)
            .api_port(api_port)
            .console_port(console_port)
            .console_ui(false)
            .startup_timeout(config.startup_timeout);
        if let Some(path) = config.release_manifest {
            builder = builder.provider_runtime_release_manifest(path);
        }
        if let Some(path) = config.cache_dir {
            builder = builder.provider_runtime_cache_dir(path);
        }
        drop(api_listener);
        drop(console_listener);
        Ok(Self {
            node: builder.start().await?,
        })
    }

    pub fn api_base_url(&self) -> &str {
        self.node.api_base_url()
    }

    pub async fn status(&self) -> Result<Value> {
        Ok(self.node.status().await?.payload)
    }

    pub async fn stop(self) -> Result<()> {
        self.node.stop().await
    }
}

impl Default for ProviderHostConfig {
    fn default() -> Self {
        Self {
            bundle_roots: Vec::new(),
            release_manifest: None,
            cache_dir: None,
            allow_download: false,
            startup_timeout: Duration::from_secs(30),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_owns_carrier_paths_and_download_policy() {
        let config = ProviderHostConfig {
            bundle_roots: vec![PathBuf::from("/app/provider-runtimes/apple")],
            release_manifest: Some(PathBuf::from("/app/provider-runtimes.json")),
            cache_dir: Some(PathBuf::from("/cache/providers")),
            allow_download: true,
            startup_timeout: Duration::from_secs(45),
        };

        assert_eq!(
            config.bundle_roots,
            vec![PathBuf::from("/app/provider-runtimes/apple")]
        );
        assert!(config.allow_download);
        assert_eq!(config.startup_timeout, Duration::from_secs(45));
    }
}
