use std::{fs, io::Write, path::PathBuf};

use anyhow::{Context, Result, bail};
use futures_util::StreamExt;
use reqwest::Client;
use sha2::Digest;

use crate::{
    archive::{ExtractedPluginArchive, extract_plugin_archive},
    catalog::PluginCatalog,
    github::{GitHubReleaseAsset, GitHubReleaseClient},
    select_plugin_asset,
    source_ref::{
        GitHubPluginSource, PluginInstallRef, PluginVersion, is_valid_name, parse_install_ref,
    },
    store::{InstalledPluginMetadata, PluginStore, default_store_root},
    target::{ArchiveExt, PluginTarget},
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

pub fn install_plugin_archive(
    name: &str,
    version: &str,
    archive_path: &std::path::Path,
    options: &PluginInstallOptions,
    progress: &mut impl PluginProgressReporter,
) -> Result<InstallOutcome> {
    if !is_valid_name(name) {
        bail!(
            "local plugin name must use lowercase ASCII letters, numbers, and single '-' separators"
        );
    }
    if version.trim().is_empty() {
        bail!("local plugin version cannot be empty");
    }
    let archive_path = archive_path
        .canonicalize()
        .with_context(|| format!("open local plugin archive {}", archive_path.display()))?;
    let archive_ext = local_archive_ext(&archive_path)?;
    let asset_name = archive_path
        .file_name()
        .and_then(|value| value.to_str())
        .context("local plugin archive filename must be valid UTF-8")?
        .to_string();

    progress.report(PluginProgressEvent::Extracting {
        asset: asset_name.clone(),
    });
    let store = PluginStore::new(&options.store_root);
    let current = store.load_optional(name)?;
    let extracted =
        extract_plugin_archive(&archive_path, archive_ext, name, &options.install_root)?;
    let metadata = InstalledPluginMetadata {
        name: name.to_string(),
        source_repository: format!("local:{}", archive_path.display()),
        installed_version: version.to_string(),
        target_triple: options.target.triple().to_string(),
        downloaded_asset_name: asset_name,
        install_path: extracted.install_path,
        enabled: current.as_ref().map(|item| item.enabled).unwrap_or(true),
        manifest: extracted.manifest,
        last_protocol_version: current.as_ref().and_then(|item| item.last_protocol_version),
        last_status: current.as_ref().and_then(|item| item.last_status.clone()),
        last_error: None,
    };
    store.save(&metadata)?;
    progress.report(PluginProgressEvent::Installed {
        name: metadata.name.clone(),
        version: metadata.installed_version.clone(),
    });
    Ok(InstallOutcome {
        metadata,
        changed: true,
    })
}

fn local_archive_ext(path: &std::path::Path) -> Result<ArchiveExt> {
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .context("local plugin archive filename must be valid UTF-8")?;
    if name.ends_with(".tar.gz") {
        return Ok(ArchiveExt::TarGz);
    }
    if name.ends_with(".zip") {
        return Ok(ArchiveExt::Zip);
    }
    bail!("local plugin archive must end in .tar.gz or .zip")
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
    required_plugin_asset_sha256(asset)?;
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

    let mut temp = tempfile::Builder::new()
        .prefix("mesh-plugin-asset-")
        .suffix(&format!("-{}", asset.name))
        .tempfile()
        .context("create plugin asset temp file")?;

    let mut downloaded = 0u64;
    let mut hasher = sha2::Sha256::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.with_context(|| format!("read plugin asset {}", asset.name))?;
        downloaded += chunk.len() as u64;
        temp.write_all(&chunk)
            .with_context(|| format!("write plugin asset temp file {}", temp.path().display()))?;
        hasher.update(&chunk);
        progress.report(PluginProgressEvent::DownloadProgress {
            downloaded_bytes: downloaded,
            total_bytes: asset.size,
        });
    }
    temp.flush()
        .with_context(|| format!("flush plugin asset temp file {}", temp.path().display()))?;
    verify_plugin_asset_checksum(asset, hasher)?;
    let (_file, path) = temp.keep().context("persist plugin asset temp path")?;
    progress.report(PluginProgressEvent::DownloadFinished {
        asset: asset.name.clone(),
    });
    Ok(path)
}

fn verify_plugin_asset_checksum(asset: &GitHubReleaseAsset, hasher: sha2::Sha256) -> Result<()> {
    let expected = required_plugin_asset_sha256(asset)?;
    let actual = format!("{:x}", hasher.finalize());
    if actual != expected {
        bail!(
            "plugin asset checksum mismatch for {}: expected {expected}, got {actual}",
            asset.name
        );
    }
    Ok(())
}

fn required_plugin_asset_sha256(asset: &GitHubReleaseAsset) -> Result<String> {
    let digest = asset
        .digest
        .as_deref()
        .with_context(|| format!("plugin asset {} is missing a GitHub digest", asset.name))?;
    let Some(sha256) = digest.trim().strip_prefix("sha256:") else {
        bail!("plugin asset {} digest must use sha256", asset.name);
    };
    let sha256 = sha256.to_ascii_lowercase();
    if sha256.len() != 64 || !sha256.chars().all(|ch| ch.is_ascii_hexdigit()) {
        bail!("plugin asset {} has an invalid sha256 digest", asset.name);
    }
    Ok(sha256)
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
        InstalledPluginVisibility, InstalledPluginWebUiBundleMetadata,
        InstalledPluginWebUiConfigSectionMetadata, InstalledPluginWebUiMetadata,
        InstalledPluginWebUiPageMetadata, InstalledPluginWebUiValidation,
        InstalledPluginWebUiValidationStatus, SUPPORTED_PLUGIN_SCHEMA_VERSION,
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

    fn packaged_manifest_fixture() -> Vec<u8> {
        serde_json::to_vec_pretty(&InstalledPluginManifestMetadata {
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
            web_ui: Some(InstalledPluginWebUiMetadata {
                pages: vec![InstalledPluginWebUiPageMetadata {
                    id: "dashboard".to_string(),
                    label: "Dashboard".to_string(),
                    icon: None,
                    route: "dashboard".to_string(),
                    bundle_id: "main".to_string(),
                    entry_script: "assets/main.js".to_string(),
                }],
                config_sections: vec![InstalledPluginWebUiConfigSectionMetadata {
                    id: "settings".to_string(),
                    title: "Settings".to_string(),
                    entry_script: "assets/settings.js".to_string(),
                    parent_tab: Some("integrations".to_string()),
                    bundle_id: "main".to_string(),
                }],
                bundles: vec![InstalledPluginWebUiBundleMetadata {
                    id: "main".to_string(),
                    root_path: "web-ui".to_string(),
                }],
                asset_root: None,
                validation: InstalledPluginWebUiValidation {
                    status: InstalledPluginWebUiValidationStatus::Invalid,
                    reason: None,
                },
            }),
        })
        .unwrap()
    }

    fn assert_installed_web_ui(store: &PluginStore) {
        let loaded = store.load("demo").unwrap();
        let web_ui = loaded
            .manifest
            .as_ref()
            .and_then(|manifest| manifest.web_ui.as_ref())
            .expect("stored web UI metadata");
        assert_eq!(web_ui.asset_root.as_deref(), Some(Path::new("web-ui")));
        assert_eq!(
            web_ui.validation.status,
            InstalledPluginWebUiValidationStatus::Valid
        );
        assert_eq!(
            loaded.web_ui_asset_root_path(),
            Some(loaded.install_path.join("web-ui"))
        );
    }

    #[test]
    fn install_plugin_schema_roundtrip() {
        let temp = TempDir::new().unwrap();
        let install_root = temp.path().join("installed");
        let store_root = temp.path().join("store");
        let archive_path = temp.path().join("demo.tar.gz");
        let executable_name = format!("demo{}", std::env::consts::EXE_SUFFIX);
        let packaged_manifest = packaged_manifest_fixture();
        write_tar_gz(
            &archive_path,
            "demo",
            &[
                ("plugin.toml", b"name = \"demo\""),
                (executable_name.as_str(), b""),
                ("plugin-manifest.json", packaged_manifest.as_slice()),
                ("web-ui/assets/main.js", b"console.log('main')"),
                ("web-ui/assets/settings.js", b"console.log('settings')"),
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
            digest: None,
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
        assert_installed_web_ui(&store);
    }

    #[test]
    fn installs_local_archive_through_package_validation_boundary() {
        let temp = TempDir::new().unwrap();
        let archive_path = temp.path().join("demo-local.tar.gz");
        let executable_name = format!("demo{}", std::env::consts::EXE_SUFFIX);
        let manifest = serde_json::json!({
            "web_ui": {
                "pages": [{
                    "id": "overview",
                    "label": "Overview",
                    "route": "overview",
                    "bundle_id": "main",
                    "entry_script": "app.js"
                }],
                "bundles": [{"id": "main", "root_path": "bundle"}]
            }
        })
        .to_string();
        write_tar_gz(
            &archive_path,
            "demo",
            &[
                ("plugin.toml", b"name = \"demo\""),
                (&executable_name, b"executable"),
                ("plugin-manifest.json", manifest.as_bytes()),
                (
                    "bundle/app.js",
                    b"export const registerMeshPluginUi = () => ({ pages: {} });",
                ),
            ],
        )
        .unwrap();
        let options = PluginInstallOptions {
            store_root: temp.path().join("store"),
            install_root: temp.path().join("installed"),
            catalog_url: "unused".to_string(),
            target: PluginTarget::current().unwrap(),
        };
        let mut events = Vec::new();

        let outcome =
            install_plugin_archive("demo", "0.1.0-dev", &archive_path, &options, &mut |event| {
                events.push(event)
            })
            .unwrap();

        assert!(outcome.changed);
        assert_eq!(outcome.metadata.installed_version, "0.1.0-dev");
        assert!(outcome.metadata.source_repository.starts_with("local:"));
        assert_eq!(
            outcome
                .metadata
                .manifest
                .as_ref()
                .and_then(|manifest| manifest.web_ui.as_ref())
                .map(|web_ui| web_ui.validation.status),
            Some(InstalledPluginWebUiValidationStatus::Valid)
        );
        assert!(PluginStore::new(&options.store_root).load("demo").is_ok());
        assert!(matches!(
            events.last(),
            Some(PluginProgressEvent::Installed { name, version })
                if name == "demo" && version == "0.1.0-dev"
        ));
    }

    #[test]
    fn modified_plugin_asset_is_rejected_before_extraction() {
        let expected = format!("{:x}", sha2::Sha256::digest(b"expected plugin archive"));
        let asset = GitHubReleaseAsset {
            name: "demo.tar.gz".to_string(),
            browser_download_url: "https://example.invalid/demo.tar.gz".to_string(),
            size: None,
            digest: Some(format!("sha256:{expected}")),
        };
        let mut hasher = sha2::Sha256::new();
        hasher.update(b"modified plugin archive");

        let error = verify_plugin_asset_checksum(&asset, hasher)
            .expect_err("modified plugin asset must fail verification");
        assert!(error.to_string().contains("checksum mismatch"), "{error:?}");
    }

    #[test]
    fn plugin_assets_without_valid_github_digests_are_rejected() {
        let mut asset = GitHubReleaseAsset {
            name: "demo.tar.gz".to_string(),
            browser_download_url: "https://example.invalid/demo.tar.gz".to_string(),
            size: None,
            digest: None,
        };
        let missing = required_plugin_asset_sha256(&asset).unwrap_err();
        assert!(missing.to_string().contains("missing a GitHub digest"));

        asset.digest = Some("sha512:abc".to_string());
        let wrong_algorithm = required_plugin_asset_sha256(&asset).unwrap_err();
        assert!(wrong_algorithm.to_string().contains("must use sha256"));

        asset.digest = Some("sha256:not-a-digest".to_string());
        let malformed = required_plugin_asset_sha256(&asset).unwrap_err();
        assert!(malformed.to_string().contains("invalid sha256"));
    }
}
