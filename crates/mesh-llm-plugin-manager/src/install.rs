use std::{fs, io::Write, path::PathBuf};

use anyhow::{Context, Result, bail};
use futures_util::StreamExt;
use reqwest::Client;

use crate::{
    archive::{ExtractedPluginArchive, extract_plugin_archive},
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

#[derive(Debug, Clone, PartialEq)]
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
    let extracted = extract_plugin_archive(
        &archive_path,
        options.target.archive_ext(),
        &resolved.plugin_name,
        &options.install_root,
    )?;
    let _ = fs::remove_file(&archive_path);

    let metadata = build_installed_metadata(
        &resolved,
        &release.tag_name,
        asset,
        &options.target,
        extracted,
        current.as_ref(),
    );
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

fn build_installed_metadata(
    resolved: &ResolvedInstallSource,
    release_tag: &str,
    asset: &GitHubReleaseAsset,
    target: &PluginTarget,
    extracted: ExtractedPluginArchive,
    current: Option<&InstalledPluginMetadata>,
) -> InstalledPluginMetadata {
    InstalledPluginMetadata {
        name: resolved.plugin_name.clone(),
        source_repository: resolved.source.url(),
        installed_version: release_tag.to_string(),
        target_triple: target.triple().to_string(),
        downloaded_asset_name: asset.name.clone(),
        install_path: extracted.install_path,
        enabled: current.map(|metadata| metadata.enabled).unwrap_or(true),
        manifest: extracted.manifest,
        last_protocol_version: current.and_then(|metadata| metadata.last_protocol_version),
        last_status: current.and_then(|metadata| metadata.last_status.clone()),
        last_error: None,
    }
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

#[cfg(test)]
mod tests {
    use std::path::Path;

    use flate2::{Compression, write::GzEncoder};
    use tempfile::TempDir;

    use super::*;
    use crate::ArchiveExt;
    use crate::store::{
        InstalledPluginApplyMode, InstalledPluginConditionOperator, InstalledPluginConditionValue,
        InstalledPluginConditionalDisable, InstalledPluginConfigSchema,
        InstalledPluginConflictRule, InstalledPluginConstraint, InstalledPluginControlAvailability,
        InstalledPluginControlAvailabilitySource, InstalledPluginControlBehavior,
        InstalledPluginControlCondition, InstalledPluginDisabledWritePolicy,
        InstalledPluginManifestMetadata, InstalledPluginNumericControl,
        InstalledPluginOptionsSource, InstalledPluginRestartScope, InstalledPluginSettingSchema,
        InstalledPluginTextFormat, InstalledPluginValueKind, InstalledPluginValueSchema,
        InstalledPluginVisibility, SUPPORTED_PLUGIN_SCHEMA_VERSION,
    };

    fn write_tar_gz(archive_path: &Path, plugin_name: &str, files: &[(&str, &[u8])]) -> Result<()> {
        let archive_file = fs::File::create(archive_path)?;
        let encoder = GzEncoder::new(archive_file, Compression::default());
        let mut archive = tar::Builder::new(encoder);
        for (relative_path, contents) in files {
            let path = format!("{plugin_name}/{relative_path}");
            let mut header = tar::Header::new_gnu();
            header.set_size(contents.len() as u64);
            header.set_mode(0o755);
            header.set_cksum();
            archive.append_data(&mut header, path, *contents)?;
        }
        archive.finish()?;
        archive.into_inner()?.finish()?;
        Ok(())
    }

    #[test]
    fn install_plugin_schema_roundtrip() {
        let temp = TempDir::new().unwrap();
        let install_root = temp.path().join("installed");
        let store_root = temp.path().join("store");
        let archive_path = temp.path().join("demo.tar.gz");
        let executable_name = format!("demo{}", std::env::consts::EXE_SUFFIX);
        let packaged_manifest = serde_json::to_vec_pretty(&InstalledPluginManifestMetadata {
            config_schema: Some(InstalledPluginConfigSchema {
                plugin_name: "demo".to_string(),
                schema_version: SUPPORTED_PLUGIN_SCHEMA_VERSION,
                allow_unvalidated_config: false,
                settings: vec![
                    InstalledPluginSettingSchema {
                        key: "retention_days".to_string(),
                        value_schema: InstalledPluginValueSchema {
                            kind: InstalledPluginValueKind::Integer,
                            enum_values: Vec::new(),
                            items: None,
                            object_properties: Vec::new(),
                            allow_additional_properties: false,
                        },
                        required: true,
                        default_json: Some("14".to_string()),
                        constraints: vec![InstalledPluginConstraint::Range {
                            min: Some("1".to_string()),
                            max: Some("365".to_string()),
                        }],
                        apply_mode: InstalledPluginApplyMode::DynamicValidationOnly,
                        restart_scope: InstalledPluginRestartScope::PluginProcess,
                        visibility: InstalledPluginVisibility::User,
                        description: Some("How long to retain entries.".to_string()),
                        presentation: Some(crate::store::InstalledPluginPresentationMetadata {
                            label: Some("Retention days".to_string()),
                            help: Some("How long to retain entries.".to_string()),
                            category_id: Some("retention".to_string()),
                            category_label: Some("Retention".to_string()),
                            category_summary: Some("Retention settings".to_string()),
                            category_order: Some(10),
                            setting_order: Some(20),
                            unit: Some("days".to_string()),
                            placeholder: None,
                            control_hint: Some("number".to_string()),
                            renderer_id: None,
                        }),
                        control_behavior: Some(InstalledPluginControlBehavior {
                            numeric: Some(InstalledPluginNumericControl {
                                min: Some(1.0),
                                max: Some(365.0),
                                step: Some(1.0),
                                soft_min: None,
                                soft_max: None,
                                unit: Some("days".to_string()),
                            }),
                            text_format: Some(InstalledPluginTextFormat::Path),
                            options_source: Some(
                                InstalledPluginOptionsSource::RuntimeInstalledPlugins,
                            ),
                            availability: Some(InstalledPluginControlAvailability {
                                enabled: false,
                                reason: Some("Waiting for runtime discovery".to_string()),
                                note: Some("The current value will be preserved.".to_string()),
                                source: InstalledPluginControlAvailabilitySource::Runtime,
                            }),
                            enable_when: vec![InstalledPluginControlCondition {
                                key: "peer_name".to_string(),
                                operator: InstalledPluginConditionOperator::Present,
                                values: Vec::new(),
                            }],
                            disable_when: vec![InstalledPluginConditionalDisable {
                                condition: InstalledPluginControlCondition {
                                    key: "mode".to_string(),
                                    operator: InstalledPluginConditionOperator::Equals,
                                    values: vec![InstalledPluginConditionValue::String(
                                        "strict".to_string(),
                                    )],
                                },
                                reason: "Strict mode disables retention edits".to_string(),
                                note: None,
                                write_policy: InstalledPluginDisabledWritePolicy::PreserveExisting,
                            }],
                            conflicts: vec![InstalledPluginConflictRule {
                                group: "retention-policy".to_string(),
                                condition: InstalledPluginControlCondition {
                                    key: "legacy_mode".to_string(),
                                    operator: InstalledPluginConditionOperator::Truthy,
                                    values: Vec::new(),
                                },
                                reason: "Legacy mode conflicts with retention controls".to_string(),
                                preferred_key: Some("retention_days".to_string()),
                            }],
                            write_policy: Some(
                                InstalledPluginDisabledWritePolicy::PreserveExisting,
                            ),
                        }),
                    },
                    InstalledPluginSettingSchema {
                        key: "endpoint_url".to_string(),
                        value_schema: InstalledPluginValueSchema {
                            kind: InstalledPluginValueKind::Url,
                            enum_values: Vec::new(),
                            items: None,
                            object_properties: Vec::new(),
                            allow_additional_properties: false,
                        },
                        required: false,
                        default_json: Some("\"https://example.invalid\"".to_string()),
                        constraints: vec![InstalledPluginConstraint::NonEmpty],
                        apply_mode: InstalledPluginApplyMode::DynamicValidationOnly,
                        restart_scope: InstalledPluginRestartScope::PluginProcess,
                        visibility: InstalledPluginVisibility::User,
                        description: Some("Plugin endpoint URL.".to_string()),
                        presentation: None,
                        control_behavior: Some(InstalledPluginControlBehavior {
                            text_format: Some(InstalledPluginTextFormat::Url),
                            ..InstalledPluginControlBehavior::default()
                        }),
                    },
                ],
            }),
        })
        .unwrap();
        write_tar_gz(
            &archive_path,
            "demo",
            &[
                ("plugin.toml", b"name = \"demo\""),
                (executable_name.as_str(), b""),
                ("plugin-manifest.json", packaged_manifest.as_slice()),
            ],
        )
        .unwrap();

        let extracted =
            extract_plugin_archive(&archive_path, ArchiveExt::TarGz, "demo", &install_root)
                .expect("archive should extract");
        let resolved = ResolvedInstallSource {
            plugin_name: "demo".to_string(),
            source: GitHubPluginSource::from_url("https://github.com/mesh-llm/demo").unwrap(),
            version: None,
        };
        let asset = GitHubReleaseAsset {
            name: "demo-v1.0.0-aarch64-apple-darwin.tar.gz".to_string(),
            browser_download_url: "https://example.invalid/demo.tar.gz".to_string(),
            size: Some(123),
        };

        let metadata = build_installed_metadata(
            &resolved,
            "v1.0.0",
            &asset,
            &PluginTarget::from_os_arch("macos", "aarch64").unwrap(),
            extracted,
            None,
        );
        let store = PluginStore::new(&store_root);
        store.save(&metadata).unwrap();
        let loaded = store.load("demo").unwrap();

        let schema = loaded
            .manifest
            .and_then(|manifest| manifest.config_schema)
            .expect("stored schema");
        assert_eq!(schema.schema_version, SUPPORTED_PLUGIN_SCHEMA_VERSION);
        assert_eq!(schema.settings[0].key, "retention_days");
        assert_eq!(schema.settings[0].default_json.as_deref(), Some("14"));
        assert_eq!(
            schema.settings[0].value_schema.kind,
            InstalledPluginValueKind::Integer
        );
        assert_eq!(
            schema.settings[1].value_schema.kind,
            InstalledPluginValueKind::Url
        );
        assert_eq!(
            schema.settings[0]
                .presentation
                .as_ref()
                .and_then(|presentation| presentation.label.as_deref()),
            Some("Retention days")
        );
        let control_behavior = schema.settings[0]
            .control_behavior
            .as_ref()
            .expect("control behavior should survive install/load");
        assert_eq!(
            control_behavior.text_format,
            Some(InstalledPluginTextFormat::Path)
        );
        assert_eq!(
            control_behavior.options_source,
            Some(InstalledPluginOptionsSource::RuntimeInstalledPlugins)
        );
        assert_eq!(control_behavior.enable_when.len(), 1);
        assert_eq!(control_behavior.disable_when.len(), 1);
        assert_eq!(control_behavior.conflicts.len(), 1);
        assert_eq!(
            schema.settings[1]
                .control_behavior
                .as_ref()
                .and_then(|behavior| behavior.text_format),
            Some(InstalledPluginTextFormat::Url)
        );
    }
}
