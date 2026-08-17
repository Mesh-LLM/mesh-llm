use std::{
    fs,
    path::{Component, Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use flate2::read::GzDecoder;

use crate::{
    store::{
        InstalledPluginManifestMetadata, InstalledPluginWebUiMetadata,
        InstalledPluginWebUiValidation, InstalledPluginWebUiValidationStatus,
        SUPPORTED_PLUGIN_SCHEMA_VERSION,
    },
    target::ArchiveExt,
};

const PACKAGED_MANIFEST_FILE: &str = "plugin-manifest.json";

#[derive(Debug, Clone, PartialEq)]
pub struct ExtractedPluginArchive {
    pub install_path: PathBuf,
    pub manifest: Option<InstalledPluginManifestMetadata>,
}

pub fn extract_plugin_archive(
    archive_path: &Path,
    archive_ext: ArchiveExt,
    plugin_name: &str,
    install_dir: &Path,
) -> Result<ExtractedPluginArchive> {
    let staging = tempfile::Builder::new()
        .prefix("mesh-plugin-extract-")
        .tempdir()
        .context("create plugin extract staging directory")?;

    match archive_ext {
        ArchiveExt::TarGz => extract_tar_gz(archive_path, staging.path())?,
        ArchiveExt::Zip => extract_zip(archive_path, staging.path())?,
    }

    let extracted_root = find_plugin_root(staging.path(), plugin_name)?;
    validate_plugin_root(&extracted_root, plugin_name)?;
    let manifest = load_packaged_manifest(&extracted_root, plugin_name)?;
    let final_dir = install_dir.join(plugin_name);
    fs::create_dir_all(install_dir)
        .with_context(|| format!("create plugin install dir {}", install_dir.display()))?;
    replace_plugin_dir(&extracted_root, &final_dir, plugin_name)?;
    Ok(ExtractedPluginArchive {
        install_path: final_dir,
        manifest,
    })
}

fn load_packaged_manifest(
    plugin_dir: &Path,
    plugin_name: &str,
) -> Result<Option<InstalledPluginManifestMetadata>> {
    let manifest_path = plugin_dir.join(PACKAGED_MANIFEST_FILE);
    if !manifest_path.exists() {
        return Ok(None);
    }

    let contents = fs::read(&manifest_path)
        .with_context(|| format!("read packaged plugin manifest {}", manifest_path.display()))?;
    let mut manifest: InstalledPluginManifestMetadata = serde_json::from_slice(&contents)
        .with_context(|| format!("parse packaged plugin manifest {}", manifest_path.display()))?;
    validate_packaged_manifest(&manifest, plugin_name)?;
    validate_web_ui_assets(&mut manifest, plugin_dir);
    Ok(Some(manifest))
}

fn validate_web_ui_assets(manifest: &mut InstalledPluginManifestMetadata, plugin_dir: &Path) {
    let Some(web_ui) = manifest.web_ui.as_mut() else {
        return;
    };
    match validate_web_ui_asset_root(web_ui, plugin_dir) {
        Ok(asset_root) => {
            web_ui.asset_root = Some(asset_root);
            web_ui.validation = InstalledPluginWebUiValidation {
                status: InstalledPluginWebUiValidationStatus::Valid,
                reason: None,
            };
        }
        Err(reason) => {
            web_ui.asset_root = None;
            web_ui.validation = InstalledPluginWebUiValidation {
                status: InstalledPluginWebUiValidationStatus::Invalid,
                reason: Some(reason),
            };
        }
    }
}

fn validate_web_ui_asset_root(
    web_ui: &InstalledPluginWebUiMetadata,
    plugin_dir: &Path,
) -> std::result::Result<PathBuf, String> {
    let [bundle] = web_ui.bundles.as_slice() else {
        return Err("web UI v1 requires exactly one bundle root".to_string());
    };
    validate_web_ui_metadata(web_ui, bundle)?;
    validate_web_ui_relative_path(&bundle.root_path)?;
    let relative_root = PathBuf::from(&bundle.root_path);
    let plugin_root = plugin_dir
        .canonicalize()
        .map_err(|error| format!("canonicalize plugin root: {error}"))?;
    let asset_root = plugin_root.join(&relative_root);
    if !asset_root.exists() {
        return Err(format!(
            "web UI bundle root '{}' is missing",
            bundle.root_path
        ));
    }
    let canonical_asset_root = asset_root.canonicalize().map_err(|error| {
        format!(
            "canonicalize web UI bundle root '{}': {error}",
            bundle.root_path
        )
    })?;
    if !canonical_asset_root.starts_with(&plugin_root) {
        return Err(format!(
            "web UI bundle root '{}' escapes the plugin package",
            bundle.root_path
        ));
    }
    validate_web_ui_entry_scripts(web_ui, &canonical_asset_root)?;
    Ok(relative_root)
}

fn validate_web_ui_metadata(
    web_ui: &InstalledPluginWebUiMetadata,
    bundle: &crate::store::InstalledPluginWebUiBundleMetadata,
) -> std::result::Result<(), String> {
    validate_web_ui_non_empty("web UI bundle id", &bundle.id)?;

    for page in &web_ui.pages {
        validate_web_ui_non_empty("web UI page id", &page.id)?;
        validate_web_ui_non_empty("web UI page label", &page.label)?;
        validate_web_ui_route_slug(&page.route)?;
        validate_web_ui_non_empty("web UI page bundle_id", &page.bundle_id)?;
        if page.bundle_id != bundle.id {
            return Err(format!(
                "web UI page bundle_id must reference declared web UI bundle '{}', got '{}'",
                bundle.id, page.bundle_id
            ));
        }
        validate_web_ui_relative_path(&page.entry_script)?;
        if let Some(icon) = &page.icon {
            validate_web_ui_relative_path(icon)?;
        }
    }

    for section in &web_ui.config_sections {
        validate_web_ui_non_empty("web UI config section id", &section.id)?;
        validate_web_ui_non_empty("web UI config section title", &section.title)?;
        validate_web_ui_non_empty("web UI config section bundle_id", &section.bundle_id)?;
        if section.bundle_id != bundle.id {
            return Err(format!(
                "web UI config section bundle_id must reference declared web UI bundle '{}', got '{}'",
                bundle.id, section.bundle_id
            ));
        }
        validate_web_ui_relative_path(&section.entry_script)?;
        if let Some(parent_tab) = &section.parent_tab
            && parent_tab != "integrations"
        {
            return Err("web UI config section parent_tab must be `integrations`".to_string());
        }
    }
    Ok(())
}

fn validate_web_ui_entry_scripts(
    web_ui: &InstalledPluginWebUiMetadata,
    asset_root: &Path,
) -> std::result::Result<(), String> {
    let canonical_root = asset_root
        .canonicalize()
        .map_err(|error| format!("resolve web UI bundle root: {error}"))?;
    for entry_script in web_ui
        .pages
        .iter()
        .map(|page| page.entry_script.as_str())
        .chain(
            web_ui
                .config_sections
                .iter()
                .map(|section| section.entry_script.as_str()),
        )
    {
        let path = asset_root.join(entry_script);
        let canonical_path = path.canonicalize().map_err(|_| {
            format!("web UI entry script '{entry_script}' is missing from the bundle root")
        })?;
        if !canonical_path.starts_with(&canonical_root) {
            return Err(format!(
                "web UI entry script '{entry_script}' escapes the bundle root"
            ));
        }
        if !canonical_path.is_file() {
            return Err(format!(
                "web UI entry script '{entry_script}' is missing from the bundle root"
            ));
        }
    }
    Ok(())
}

fn validate_web_ui_non_empty(field_name: &str, value: &str) -> std::result::Result<(), String> {
    if value.trim().is_empty() {
        return Err(format!("{field_name} must be non-empty"));
    }
    Ok(())
}

fn validate_web_ui_route_slug(value: &str) -> std::result::Result<(), String> {
    validate_web_ui_non_empty("web UI page route", value)?;
    if value.starts_with("http://")
        || value.starts_with("https://")
        || value.starts_with("//")
        || value.contains("://")
    {
        return Err(format!(
            "web UI page route must be a slug, got URL-like value '{value}'"
        ));
    }
    if value.contains('/')
        || value.contains('\\')
        || value == "."
        || value == ".."
        || value.starts_with('.')
    {
        return Err(format!(
            "web UI page route must be a slug without path syntax '{value}'"
        ));
    }
    Ok(())
}

fn validate_web_ui_relative_path(value: &str) -> std::result::Result<(), String> {
    validate_web_ui_non_empty("web UI bundle path", value)?;
    if value.starts_with("http://") || value.starts_with("https://") || value.starts_with("//") {
        return Err(format!(
            "web UI bundle root must be a relative path, got remote URL '{value}'"
        ));
    }
    let path = Path::new(value);
    if path.is_absolute() {
        return Err(format!(
            "web UI bundle root must be a relative path, got absolute path '{value}'"
        ));
    }
    if path
        .components()
        .all(|component| matches!(component, Component::CurDir))
    {
        return Err(format!(
            "web UI bundle path must name a file or directory below the package root, got '{value}'"
        ));
    }
    if path
        .components()
        .any(|component| matches!(component, Component::ParentDir))
    {
        return Err(format!(
            "web UI bundle root must not contain traversal segments '{value}'"
        ));
    }
    Ok(())
}

fn validate_packaged_manifest(
    manifest: &InstalledPluginManifestMetadata,
    plugin_name: &str,
) -> Result<()> {
    let Some(schema) = &manifest.config_schema else {
        return Ok(());
    };
    if schema.plugin_name != plugin_name {
        bail!(
            "plugin manifest schema name '{}' does not match installed plugin '{}'",
            schema.plugin_name,
            plugin_name
        );
    }
    if schema.schema_version != SUPPORTED_PLUGIN_SCHEMA_VERSION {
        bail!(
            "plugin config schema version {} is unsupported for '{}'; supported version is {}",
            schema.schema_version,
            plugin_name,
            SUPPORTED_PLUGIN_SCHEMA_VERSION
        );
    }
    Ok(())
}

fn replace_plugin_dir(from: &Path, to: &Path, plugin_name: &str) -> Result<()> {
    if to.exists() {
        let backup_parent = tempfile::Builder::new()
            .prefix(&format!("{plugin_name}-previous-"))
            .tempdir_in(to.parent().unwrap_or_else(|| Path::new(".")))
            .with_context(|| format!("create plugin install backup for {}", to.display()))?;
        let backup_dir = backup_parent.path().join(plugin_name);
        move_dir(to, &backup_dir)
            .with_context(|| format!("backup previous plugin install {}", to.display()))?;
        if let Err(error) = move_dir(from, to) {
            let _ = move_dir(&backup_dir, to);
            return Err(error).with_context(|| format!("replace plugin install {}", to.display()));
        }
    } else {
        move_dir(from, to).with_context(|| format!("install plugin to {}", to.display()))?;
    }
    Ok(())
}

fn extract_tar_gz(archive_path: &Path, destination: &Path) -> Result<()> {
    let file = fs::File::open(archive_path)
        .with_context(|| format!("open plugin archive {}", archive_path.display()))?;
    let decoder = GzDecoder::new(file);
    let mut archive = tar::Archive::new(decoder);
    archive
        .unpack(destination)
        .with_context(|| format!("extract plugin archive {}", archive_path.display()))?;
    Ok(())
}

fn extract_zip(archive_path: &Path, destination: &Path) -> Result<()> {
    let file = fs::File::open(archive_path)
        .with_context(|| format!("open plugin archive {}", archive_path.display()))?;
    let mut archive = zip::ZipArchive::new(file)
        .with_context(|| format!("read plugin zip archive {}", archive_path.display()))?;
    for index in 0..archive.len() {
        let mut file = archive.by_index(index)?;
        let Some(enclosed) = file.enclosed_name() else {
            bail!("zip archive contains unsafe path: {}", file.name());
        };
        let output_path = destination.join(enclosed);
        if file.is_dir() {
            fs::create_dir_all(&output_path)
                .with_context(|| format!("create zip directory {}", output_path.display()))?;
        } else {
            if let Some(parent) = output_path.parent() {
                fs::create_dir_all(parent)
                    .with_context(|| format!("create zip parent directory {}", parent.display()))?;
            }
            let mut output = fs::File::create(&output_path)
                .with_context(|| format!("create zip output {}", output_path.display()))?;
            std::io::copy(&mut file, &mut output)
                .with_context(|| format!("write zip output {}", output_path.display()))?;
        }
    }
    Ok(())
}

fn find_plugin_root(staging: &Path, plugin_name: &str) -> Result<PathBuf> {
    let expected = staging.join(plugin_name);
    if expected.join("plugin.toml").exists() {
        return Ok(expected);
    }

    let mut matches = Vec::new();
    for entry in
        fs::read_dir(staging).with_context(|| format!("read staging dir {}", staging.display()))?
    {
        let entry = entry?;
        if entry.file_type()?.is_dir() && entry.path().join("plugin.toml").exists() {
            matches.push(entry.path());
        }
    }

    match matches.as_slice() {
        [path] => Ok(path.clone()),
        [] => bail!("plugin archive does not contain plugin.toml"),
        _ => bail!("plugin archive contains multiple plugin roots"),
    }
}

fn validate_plugin_root(plugin_dir: &Path, plugin_name: &str) -> Result<()> {
    if !plugin_dir.join("plugin.toml").exists() {
        bail!("installed plugin is missing plugin.toml");
    }
    let executable = plugin_dir.join(format!("{plugin_name}{}", std::env::consts::EXE_SUFFIX));
    if !executable.exists() {
        bail!(
            "installed plugin is missing executable {}",
            executable.display()
        );
    }
    Ok(())
}

fn copy_dir_and_remove(from: &Path, to: &Path) -> Result<()> {
    copy_dir(from, to)?;
    fs::remove_dir_all(from)
        .with_context(|| format!("remove copied plugin source {}", from.display()))?;
    Ok(())
}

fn move_dir(from: &Path, to: &Path) -> Result<()> {
    fs::rename(from, to).or_else(|_| copy_dir_and_remove(from, to))
}

fn copy_dir(from: &Path, to: &Path) -> Result<()> {
    fs::create_dir_all(to).with_context(|| format!("create directory {}", to.display()))?;
    for entry in fs::read_dir(from).with_context(|| format!("read directory {}", from.display()))? {
        let entry = entry?;
        let from_path = entry.path();
        let to_path = to.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir(&from_path, &to_path)?;
        } else {
            fs::copy(&from_path, &to_path).with_context(|| {
                format!("copy {} to {}", from_path.display(), to_path.display())
            })?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use flate2::{Compression, write::GzEncoder};
    use tempfile::TempDir;

    use super::*;
    use crate::store::{
        InstalledPluginApplyMode, InstalledPluginConfigSchema, InstalledPluginConstraint,
        InstalledPluginRestartScope, InstalledPluginSettingSchema, InstalledPluginValueKind,
        InstalledPluginValueSchema, InstalledPluginVisibility,
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

    fn web_ui_manifest(root_path: &str) -> Vec<u8> {
        serde_json::json!({
            "web_ui": {
                "pages": [{
                    "id": "dashboard",
                    "label": "Dashboard",
                    "route": "dashboard",
                    "bundle_id": "main",
                    "entry_script": "assets/main.js"
                }],
                "config_sections": [{
                    "id": "settings",
                    "title": "Settings",
                    "entry_script": "assets/settings.js",
                    "parent_tab": "integrations",
                    "bundle_id": "main"
                }],
                "bundles": [{
                    "id": "main",
                    "root_path": root_path
                }]
            }
        })
        .to_string()
        .into_bytes()
    }

    fn web_ui_manifest_without_bundles() -> Vec<u8> {
        serde_json::json!({
            "web_ui": {
                "pages": [{
                    "id": "dashboard",
                    "label": "Dashboard",
                    "route": "dashboard",
                    "bundle_id": "main",
                    "entry_script": "assets/main.js"
                }]
            }
        })
        .to_string()
        .into_bytes()
    }

    struct SymlinkArchive<'a> {
        archive_path: &'a Path,
        plugin_name: &'a str,
        link_path: &'a str,
        link_target: &'a Path,
        files: &'a [(&'a str, &'a [u8])],
    }

    fn write_tar_gz_with_symlink(fixture: SymlinkArchive<'_>) -> Result<()> {
        let archive_file = fs::File::create(fixture.archive_path)?;
        let encoder = GzEncoder::new(archive_file, Compression::default());
        let mut archive = tar::Builder::new(encoder);
        for (relative_path, contents) in fixture.files {
            let path = format!("{}/{relative_path}", fixture.plugin_name);
            let mut header = tar::Header::new_gnu();
            header.set_size(contents.len() as u64);
            header.set_mode(0o755);
            header.set_cksum();
            archive.append_data(&mut header, path, *contents)?;
        }
        let path = format!("{}/{}", fixture.plugin_name, fixture.link_path);
        let mut header = tar::Header::new_gnu();
        header.set_entry_type(tar::EntryType::Symlink);
        header.set_size(0);
        header.set_mode(0o755);
        header.set_link_name(fixture.link_target)?;
        header.set_cksum();
        archive.append_data(&mut header, path, std::io::empty())?;
        archive.finish()?;
        archive.into_inner()?.finish()?;
        Ok(())
    }

    #[test]
    fn invalid_archive_does_not_remove_existing_install() {
        let temp = TempDir::new().unwrap();
        let install_dir = temp.path().join("installed");
        let existing = install_dir.join("demo");
        fs::create_dir_all(&existing).unwrap();
        fs::write(existing.join("old-version.txt"), "keep me").unwrap();
        fs::write(existing.join("plugin.toml"), "name = \"demo\"").unwrap();
        fs::write(
            existing.join(format!("demo{}", std::env::consts::EXE_SUFFIX)),
            "",
        )
        .unwrap();

        let archive_path = temp.path().join("demo.tar.gz");
        write_tar_gz(
            &archive_path,
            "demo",
            &[("plugin.toml", b"name = \"demo\"")],
        )
        .unwrap();

        let err = extract_plugin_archive(&archive_path, ArchiveExt::TarGz, "demo", &install_dir)
            .expect_err("archive without executable should fail validation");

        assert!(err.to_string().contains("missing executable"));
        assert_eq!(
            fs::read_to_string(existing.join("old-version.txt")).unwrap(),
            "keep me"
        );
    }

    #[test]
    fn unsupported_plugin_schema_version() {
        let temp = TempDir::new().unwrap();
        let install_dir = temp.path().join("installed");
        let archive_path = temp.path().join("demo.tar.gz");
        let executable_name = format!("demo{}", std::env::consts::EXE_SUFFIX);
        let manifest = serde_json::to_vec_pretty(&InstalledPluginManifestMetadata {
            config_schema: Some(InstalledPluginConfigSchema {
                plugin_name: "demo".to_string(),
                schema_version: SUPPORTED_PLUGIN_SCHEMA_VERSION + 1,
                allow_unvalidated_config: false,
                settings: vec![InstalledPluginSettingSchema {
                    key: "mode".to_string(),
                    value_schema: InstalledPluginValueSchema {
                        kind: InstalledPluginValueKind::String,
                        enum_values: Vec::new(),
                        items: None,
                        object_properties: Vec::new(),
                        allow_additional_properties: false,
                    },
                    required: false,
                    default_json: Some("\"strict\"".to_string()),
                    constraints: vec![InstalledPluginConstraint::AllowedValues {
                        values: vec!["strict".to_string(), "relaxed".to_string()],
                    }],
                    apply_mode: InstalledPluginApplyMode::StaticOnLoad,
                    restart_scope: InstalledPluginRestartScope::PluginProcess,
                    visibility: InstalledPluginVisibility::User,
                    description: None,
                    presentation: None,
                    control_behavior: None,
                }],
            }),
            web_ui: None,
        })
        .unwrap();
        write_tar_gz(
            &archive_path,
            "demo",
            &[
                ("plugin.toml", b"name = \"demo\""),
                (executable_name.as_str(), b""),
                (PACKAGED_MANIFEST_FILE, manifest.as_slice()),
            ],
        )
        .unwrap();

        let error = extract_plugin_archive(&archive_path, ArchiveExt::TarGz, "demo", &install_dir)
            .expect_err("unsupported schema version should fail install-time extraction");

        assert!(error.to_string().contains("unsupported"));
    }

    #[test]
    fn old_packaged_manifest_without_web_ui_remains_accepted() {
        let temp = TempDir::new().unwrap();
        let install_dir = temp.path().join("installed");
        let archive_path = temp.path().join("demo.tar.gz");
        let executable_name = format!("demo{}", std::env::consts::EXE_SUFFIX);
        let manifest = serde_json::to_vec_pretty(&InstalledPluginManifestMetadata {
            config_schema: Some(InstalledPluginConfigSchema {
                plugin_name: "demo".to_string(),
                schema_version: SUPPORTED_PLUGIN_SCHEMA_VERSION,
                allow_unvalidated_config: false,
                settings: vec![InstalledPluginSettingSchema {
                    key: "mode".to_string(),
                    value_schema: InstalledPluginValueSchema {
                        kind: InstalledPluginValueKind::String,
                        enum_values: Vec::new(),
                        items: None,
                        object_properties: Vec::new(),
                        allow_additional_properties: false,
                    },
                    required: false,
                    default_json: None,
                    constraints: Vec::new(),
                    apply_mode: InstalledPluginApplyMode::StaticOnLoad,
                    restart_scope: InstalledPluginRestartScope::PluginProcess,
                    visibility: InstalledPluginVisibility::User,
                    description: None,
                    presentation: None,
                    control_behavior: None,
                }],
            }),
            web_ui: None,
        })
        .unwrap();

        write_tar_gz(
            &archive_path,
            "demo",
            &[
                ("plugin.toml", b"name = \"demo\""),
                (executable_name.as_str(), b""),
                (PACKAGED_MANIFEST_FILE, manifest.as_slice()),
            ],
        )
        .unwrap();

        let extracted =
            extract_plugin_archive(&archive_path, ArchiveExt::TarGz, "demo", &install_dir)
                .expect("old packaged manifest without web_ui should still install");
        let metadata = extracted.manifest.expect("packaged manifest should load");

        assert!(metadata.config_schema.is_some());
        assert!(metadata.web_ui.is_none());
    }

    #[test]
    fn package_with_web_ui_bundle_records_valid_asset_root() {
        let temp = TempDir::new().unwrap();
        let install_dir = temp.path().join("installed");
        let archive_path = temp.path().join("demo.tar.gz");
        let executable_name = format!("demo{}", std::env::consts::EXE_SUFFIX);
        let manifest = web_ui_manifest("web-ui");

        write_tar_gz(
            &archive_path,
            "demo",
            &[
                ("plugin.toml", b"name = \"demo\""),
                (executable_name.as_str(), b""),
                (PACKAGED_MANIFEST_FILE, manifest.as_slice()),
                ("web-ui/index.html", b"<div id=\"root\"></div>"),
                ("web-ui/assets/main.js", b"console.log('main')"),
                ("web-ui/assets/settings.js", b"console.log('settings')"),
            ],
        )
        .unwrap();

        let extracted =
            extract_plugin_archive(&archive_path, ArchiveExt::TarGz, "demo", &install_dir)
                .expect("archive should extract");
        let web_ui = extracted
            .manifest
            .and_then(|manifest| manifest.web_ui)
            .expect("web UI metadata should be stored");

        assert_eq!(web_ui.asset_root, Some(PathBuf::from("web-ui")));
        assert_eq!(
            web_ui.validation.status,
            InstalledPluginWebUiValidationStatus::Valid
        );
    }

    #[test]
    fn missing_web_ui_entry_script_records_invalid_ui_without_rejecting_plugin() {
        let temp = TempDir::new().unwrap();
        let install_dir = temp.path().join("installed");
        let archive_path = temp.path().join("demo.tar.gz");
        let executable_name = format!("demo{}", std::env::consts::EXE_SUFFIX);
        let manifest = web_ui_manifest("web-ui");

        write_tar_gz(
            &archive_path,
            "demo",
            &[
                ("plugin.toml", b"name = \"demo\""),
                (executable_name.as_str(), b""),
                (PACKAGED_MANIFEST_FILE, manifest.as_slice()),
                ("web-ui/assets/main.js", b"console.log('main')"),
            ],
        )
        .unwrap();

        let extracted =
            extract_plugin_archive(&archive_path, ArchiveExt::TarGz, "demo", &install_dir)
                .expect("missing web UI assets should not fail plugin installation");
        let web_ui = extracted
            .manifest
            .and_then(|manifest| manifest.web_ui)
            .expect("web UI metadata should be retained");

        assert_eq!(web_ui.asset_root, None);
        assert_eq!(
            web_ui.validation.status,
            InstalledPluginWebUiValidationStatus::Invalid
        );
        assert!(
            web_ui
                .validation
                .reason
                .as_deref()
                .is_some_and(|reason| reason.contains("settings.js"))
        );
    }

    #[test]
    fn missing_web_ui_bundle_entry_records_invalid_ui() {
        let temp = TempDir::new().unwrap();
        let install_dir = temp.path().join("installed");
        let archive_path = temp.path().join("demo.tar.gz");
        let executable_name = format!("demo{}", std::env::consts::EXE_SUFFIX);
        let manifest = web_ui_manifest_without_bundles();

        write_tar_gz(
            &archive_path,
            "demo",
            &[
                ("plugin.toml", b"name = \"demo\""),
                (executable_name.as_str(), b""),
                (PACKAGED_MANIFEST_FILE, manifest.as_slice()),
            ],
        )
        .unwrap();

        let extracted =
            extract_plugin_archive(&archive_path, ArchiveExt::TarGz, "demo", &install_dir)
                .expect("missing web UI assets should not fail plugin install");
        let web_ui = extracted
            .manifest
            .and_then(|manifest| manifest.web_ui)
            .unwrap();

        assert_eq!(web_ui.asset_root, None);
        assert_eq!(
            web_ui.validation.status,
            InstalledPluginWebUiValidationStatus::Invalid
        );
        assert!(web_ui.validation.reason.unwrap().contains("exactly one"));
    }

    #[test]
    fn traversal_web_ui_bundle_path_records_invalid_ui() {
        let temp = TempDir::new().unwrap();
        let install_dir = temp.path().join("installed");
        let archive_path = temp.path().join("demo.tar.gz");
        let executable_name = format!("demo{}", std::env::consts::EXE_SUFFIX);
        let manifest = web_ui_manifest("../outside");

        write_tar_gz(
            &archive_path,
            "demo",
            &[
                ("plugin.toml", b"name = \"demo\""),
                (executable_name.as_str(), b""),
                (PACKAGED_MANIFEST_FILE, manifest.as_slice()),
            ],
        )
        .unwrap();

        let extracted =
            extract_plugin_archive(&archive_path, ArchiveExt::TarGz, "demo", &install_dir)
                .expect("traversal web UI asset root should not fail plugin install");
        let web_ui = extracted
            .manifest
            .and_then(|manifest| manifest.web_ui)
            .unwrap();

        assert_eq!(
            web_ui.validation.status,
            InstalledPluginWebUiValidationStatus::Invalid
        );
        assert!(web_ui.validation.reason.unwrap().contains("traversal"));
    }

    #[test]
    fn remote_web_ui_bundle_path_records_invalid_ui() {
        let temp = TempDir::new().unwrap();
        let install_dir = temp.path().join("installed");
        let archive_path = temp.path().join("demo.tar.gz");
        let executable_name = format!("demo{}", std::env::consts::EXE_SUFFIX);
        let manifest = web_ui_manifest("https://example.invalid/plugin-ui");

        write_tar_gz(
            &archive_path,
            "demo",
            &[
                ("plugin.toml", b"name = \"demo\""),
                (executable_name.as_str(), b""),
                (PACKAGED_MANIFEST_FILE, manifest.as_slice()),
            ],
        )
        .unwrap();

        let extracted =
            extract_plugin_archive(&archive_path, ArchiveExt::TarGz, "demo", &install_dir)
                .expect("remote web UI asset root should not fail plugin install");
        let web_ui = extracted
            .manifest
            .and_then(|manifest| manifest.web_ui)
            .unwrap();

        assert_eq!(
            web_ui.validation.status,
            InstalledPluginWebUiValidationStatus::Invalid
        );
        assert!(web_ui.validation.reason.unwrap().contains("remote URL"));
    }

    #[test]
    fn symlink_escape_web_ui_bundle_path_records_invalid_ui() {
        let temp = TempDir::new().unwrap();
        let install_dir = temp.path().join("installed");
        let archive_path = temp.path().join("demo.tar.gz");
        let outside = temp.path().join("outside-ui");
        fs::create_dir_all(&outside).unwrap();
        let executable_name = format!("demo{}", std::env::consts::EXE_SUFFIX);
        let manifest = web_ui_manifest("web-ui");

        write_tar_gz_with_symlink(SymlinkArchive {
            archive_path: &archive_path,
            plugin_name: "demo",
            link_path: "web-ui",
            link_target: &outside,
            files: &[
                ("plugin.toml", b"name = \"demo\""),
                (executable_name.as_str(), b""),
                (PACKAGED_MANIFEST_FILE, manifest.as_slice()),
            ],
        })
        .unwrap();

        let extracted =
            extract_plugin_archive(&archive_path, ArchiveExt::TarGz, "demo", &install_dir)
                .expect("symlink escape should be recorded as web UI invalidity");
        let web_ui = extracted
            .manifest
            .and_then(|manifest| manifest.web_ui)
            .unwrap();

        assert_eq!(
            web_ui.validation.status,
            InstalledPluginWebUiValidationStatus::Invalid
        );
        assert!(web_ui.validation.reason.unwrap().contains("escapes"));
    }

    #[cfg(unix)]
    #[test]
    fn symlink_escape_web_ui_entry_script_records_invalid_ui() {
        let temp = TempDir::new().unwrap();
        let install_dir = temp.path().join("installed");
        let archive_path = temp.path().join("demo.tar.gz");
        let outside_script = temp.path().join("outside.js");
        fs::write(&outside_script, "export const outside = true;").unwrap();
        let executable_name = format!("demo{}", std::env::consts::EXE_SUFFIX);
        let manifest = web_ui_manifest("web-ui");

        write_tar_gz_with_symlink(SymlinkArchive {
            archive_path: &archive_path,
            plugin_name: "demo",
            link_path: "web-ui/assets/main.js",
            link_target: &outside_script,
            files: &[
                ("plugin.toml", b"name = \"demo\""),
                (executable_name.as_str(), b""),
                (PACKAGED_MANIFEST_FILE, manifest.as_slice()),
                ("web-ui/assets/settings.js", b"export {};"),
            ],
        })
        .unwrap();

        let extracted =
            extract_plugin_archive(&archive_path, ArchiveExt::TarGz, "demo", &install_dir)
                .expect("entry-script symlink escape should mark only the web UI invalid");
        let web_ui = extracted
            .manifest
            .and_then(|manifest| manifest.web_ui)
            .unwrap();

        assert_eq!(
            web_ui.validation.status,
            InstalledPluginWebUiValidationStatus::Invalid
        );
        assert!(web_ui.validation.reason.unwrap().contains("escapes"));
    }
}
