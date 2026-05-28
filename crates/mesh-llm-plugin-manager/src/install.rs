use std::{fs, io::Write, path::PathBuf};

use anyhow::{Context, Result, bail};
use futures_util::StreamExt;
use reqwest::Client;

use crate::{
    archive::extract_plugin_archive,
    catalog::PluginCatalog,
    github::{GitHubReleaseAsset, GitHubReleaseClient},
    select_plugin_asset,
    source_ref::{GitHubPluginSource, PluginInstallRef, PluginVersion, parse_install_ref},
    store::{InstalledPluginMetadata, PluginStore, default_store_root},
    target::PluginTarget,
};

pub const DEFAULT_CATALOG_URL: &str =
    "https://huggingface.co/datasets/meshllm/plugin-catalog/resolve/main/plugins.jsonl";

pub trait PluginProgressReporter {
    fn report(&mut self, event: PluginProgressEvent);
}

impl<F> PluginProgressReporter for F
where
    F: FnMut(PluginProgressEvent),
{
    fn report(&mut self, event: PluginProgressEvent) {
        self(event);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PluginProgressEvent {
    ResolvingCatalog {
        name: String,
    },
    ResolvingGitHub {
        repo: String,
    },
    SelectingAsset {
        target: String,
    },
    DownloadStarted {
        asset: String,
        total_bytes: Option<u64>,
    },
    DownloadProgress {
        downloaded_bytes: u64,
        total_bytes: Option<u64>,
    },
    DownloadFinished {
        asset: String,
    },
    Extracting {
        asset: String,
    },
    Installed {
        name: String,
        version: String,
    },
    Updated {
        name: String,
        from: String,
        to: String,
    },
    AlreadyCurrent {
        name: String,
        version: String,
    },
}

#[derive(Debug, Clone)]
pub struct PluginInstallOptions {
    pub store_root: PathBuf,
    pub install_root: PathBuf,
    pub catalog_url: String,
    pub target: PluginTarget,
}

impl PluginInstallOptions {
    pub fn from_env() -> Result<Self> {
        let store_root = default_store_root()?;
        let catalog_url = std::env::var("MESH_LLM_PLUGIN_CATALOG_URL")
            .unwrap_or_else(|_| DEFAULT_CATALOG_URL.to_string());
        Ok(Self {
            install_root: store_root.join("installed"),
            store_root,
            catalog_url,
            target: PluginTarget::current()?,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstallOutcome {
    pub metadata: InstalledPluginMetadata,
    pub changed: bool,
}

pub async fn install_plugin(
    reference: &str,
    options: &PluginInstallOptions,
    progress: &mut impl PluginProgressReporter,
) -> Result<InstallOutcome> {
    let parsed = parse_install_ref(reference)?;
    let resolved = resolve_install_source(parsed, options, progress).await?;
    install_resolved_plugin(resolved, options, progress, None).await
}

pub async fn update_plugin(
    name: &str,
    options: &PluginInstallOptions,
    progress: &mut impl PluginProgressReporter,
) -> Result<InstallOutcome> {
    let store = PluginStore::new(&options.store_root);
    let current = store.load(name)?;
    let source = GitHubPluginSource::from_url(&current.source_repository)?;
    let resolved = ResolvedInstallSource {
        plugin_name: current.name.clone(),
        source,
        version: None,
    };
    install_resolved_plugin(resolved, options, progress, Some(current)).await
}

struct ResolvedInstallSource {
    plugin_name: String,
    source: GitHubPluginSource,
    version: Option<PluginVersion>,
}

async fn resolve_install_source(
    parsed: PluginInstallRef,
    options: &PluginInstallOptions,
    progress: &mut impl PluginProgressReporter,
) -> Result<ResolvedInstallSource> {
    match parsed {
        PluginInstallRef::Catalog { name, version } => {
            progress.report(PluginProgressEvent::ResolvingCatalog { name: name.clone() });
            let client = Client::new();
            let catalog = PluginCatalog::fetch(&client, &options.catalog_url).await?;
            let entry = catalog
                .find_exact(&name)
                .with_context(|| format!("plugin '{name}' was not found in the catalog"))?;
            let source = GitHubPluginSource::from_url(&entry.github_url)?;
            Ok(ResolvedInstallSource {
                plugin_name: entry.name.clone(),
                source,
                version,
            })
        }
        PluginInstallRef::GitHub { source, version } => Ok(ResolvedInstallSource {
            plugin_name: source.repo.clone(),
            source,
            version,
        }),
    }
}

async fn install_resolved_plugin(
    resolved: ResolvedInstallSource,
    options: &PluginInstallOptions,
    progress: &mut impl PluginProgressReporter,
    current: Option<InstalledPluginMetadata>,
) -> Result<InstallOutcome> {
    let release_client = GitHubReleaseClient::new()?;
    progress.report(PluginProgressEvent::ResolvingGitHub {
        repo: resolved.source.repo_slug(),
    });
    let release = release_client
        .resolve_release(&resolved.source, resolved.version.as_ref())
        .await?;

    if let Some(current) = &current
        && current.installed_version == release.tag_name
    {
        progress.report(PluginProgressEvent::AlreadyCurrent {
            name: current.name.clone(),
            version: current.installed_version.clone(),
        });
        return Ok(InstallOutcome {
            metadata: current.clone(),
            changed: false,
        });
    }

    progress.report(PluginProgressEvent::SelectingAsset {
        target: options.target.triple().to_string(),
    });
    let asset_names = release.asset_names();
    let selected = select_plugin_asset(
        &resolved.plugin_name,
        Some(&PluginVersion::new(release.tag_name.clone())?),
        &options.target,
        &asset_names,
    )?;
    let asset = release
        .asset_by_name(&selected.name)
        .with_context(|| format!("selected asset '{}' missing from release", selected.name))?;
    let archive_path = download_asset(release_client.http_client(), asset, progress).await?;

    progress.report(PluginProgressEvent::Extracting {
        asset: asset.name.clone(),
    });
    let install_path = extract_plugin_archive(
        &archive_path,
        options.target.archive_ext(),
        &resolved.plugin_name,
        &options.install_root,
    )?;
    let _ = fs::remove_file(&archive_path);

    let metadata = InstalledPluginMetadata {
        name: resolved.plugin_name.clone(),
        source_repository: resolved.source.url(),
        installed_version: release.tag_name.clone(),
        target_triple: options.target.triple().to_string(),
        downloaded_asset_name: asset.name.clone(),
        install_path,
        enabled: current
            .as_ref()
            .map(|metadata| metadata.enabled)
            .unwrap_or(true),
        last_protocol_version: current
            .as_ref()
            .and_then(|metadata| metadata.last_protocol_version),
        last_status: current
            .as_ref()
            .and_then(|metadata| metadata.last_status.clone()),
        last_error: None,
    };
    PluginStore::new(&options.store_root).save(&metadata)?;

    if let Some(current) = current {
        progress.report(PluginProgressEvent::Updated {
            name: metadata.name.clone(),
            from: current.installed_version,
            to: metadata.installed_version.clone(),
        });
    } else {
        progress.report(PluginProgressEvent::Installed {
            name: metadata.name.clone(),
            version: metadata.installed_version.clone(),
        });
    }

    Ok(InstallOutcome {
        metadata,
        changed: true,
    })
}

async fn download_asset(
    client: &Client,
    asset: &GitHubReleaseAsset,
    progress: &mut impl PluginProgressReporter,
) -> Result<PathBuf> {
    progress.report(PluginProgressEvent::DownloadStarted {
        asset: asset.name.clone(),
        total_bytes: asset.size,
    });
    let response = client
        .get(&asset.browser_download_url)
        .header(reqwest::header::USER_AGENT, crate::github::USER_AGENT)
        .send()
        .await
        .with_context(|| format!("download plugin asset {}", asset.name))?;
    let status = response.status();
    if !status.is_success() {
        bail!("plugin asset download failed: {status} {}", asset.name);
    }

    let temp = tempfile::Builder::new()
        .prefix("mesh-plugin-asset-")
        .suffix(&format!("-{}", asset.name))
        .tempfile()
        .context("create plugin asset temp file")?;
    let (mut file, path) = temp.keep().context("persist plugin asset temp path")?;

    let mut downloaded = 0u64;
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.with_context(|| format!("read plugin asset {}", asset.name))?;
        downloaded += chunk.len() as u64;
        file.write_all(&chunk)
            .with_context(|| format!("write plugin asset temp file {}", path.display()))?;
        progress.report(PluginProgressEvent::DownloadProgress {
            downloaded_bytes: downloaded,
            total_bytes: asset.size,
        });
    }
    progress.report(PluginProgressEvent::DownloadFinished {
        asset: asset.name.clone(),
    });
    Ok(path)
}
