use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use flate2::read::GzDecoder;

use crate::{
    store::{InstalledPluginManifestMetadata, SUPPORTED_PLUGIN_SCHEMA_VERSION},
    target::ArchiveExt,
};

const PACKAGED_MANIFEST_FILE: &str = "plugin-manifest.json";

#[derive(Debug, Clone, PartialEq, Eq)]
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
    let manifest: InstalledPluginManifestMetadata = serde_json::from_slice(&contents)
        .with_context(|| format!("parse packaged plugin manifest {}", manifest_path.display()))?;
    validate_packaged_manifest(&manifest, plugin_name)?;
    Ok(Some(manifest))
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
                }],
            }),
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
}
