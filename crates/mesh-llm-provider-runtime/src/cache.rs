use crate::{PROVIDER_RUNTIME_MANIFEST_FILE, ProviderRuntimeManifest};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug)]
pub struct ProviderRuntimeCache {
    root: PathBuf,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct InstalledProviderRuntime {
    pub path: PathBuf,
    pub manifest: ProviderRuntimeManifest,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderRuntimeInstallStatus {
    AlreadyInstalled,
    Installed,
}

impl ProviderRuntimeCache {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn install_from_dir(
        &self,
        bundle: &Path,
    ) -> Result<(ProviderRuntimeInstallStatus, InstalledProviderRuntime)> {
        let manifest = ProviderRuntimeManifest::read_from_dir(bundle)?;
        let destination = self.artifact_path(&manifest);
        if destination.exists() {
            return existing_install(&destination, &manifest);
        }
        let parent = destination
            .parent()
            .context("provider runtime cache destination has no parent")?;
        fs::create_dir_all(parent)
            .with_context(|| format!("create provider runtime cache {}", parent.display()))?;
        let temporary = tempfile::Builder::new()
            .prefix(".install-")
            .tempdir_in(parent)
            .with_context(|| format!("create provider runtime staging in {}", parent.display()))?;
        copy_verified_bundle(bundle, temporary.path(), &manifest)
            .and_then(|()| ProviderRuntimeManifest::read_from_dir(temporary.path()))
            .and_then(|installed_manifest| {
                if installed_manifest != manifest {
                    bail!("installed provider runtime manifest changed during copy");
                }
                publish_install(temporary.path(), &destination, &manifest)
            })
    }

    pub fn find(
        &self,
        artifact_id: &str,
        version: &str,
        platform_key: &str,
    ) -> Result<Option<InstalledProviderRuntime>> {
        let path = self.root.join(artifact_id).join(version).join(platform_key);
        if !path.is_dir() {
            return Ok(None);
        }
        let manifest = ProviderRuntimeManifest::read_from_dir(&path)?;
        Ok(Some(InstalledProviderRuntime { path, manifest }))
    }

    pub fn list(&self) -> Result<Vec<InstalledProviderRuntime>> {
        if !self.root.is_dir() {
            return Ok(Vec::new());
        }
        let mut installed = Vec::new();
        for artifact_dir in child_directories(&self.root)? {
            for version_dir in child_directories(&artifact_dir)? {
                for platform_dir in child_directories(&version_dir)? {
                    if platform_dir.join(PROVIDER_RUNTIME_MANIFEST_FILE).is_file() {
                        let manifest = ProviderRuntimeManifest::read_from_dir(&platform_dir)?;
                        installed.push(InstalledProviderRuntime {
                            path: platform_dir,
                            manifest,
                        });
                    }
                }
            }
        }
        Ok(installed)
    }

    fn artifact_path(&self, manifest: &ProviderRuntimeManifest) -> PathBuf {
        self.root
            .join(&manifest.runtime.id)
            .join(&manifest.runtime.version)
            .join(manifest.runtime.platform_key())
    }
}

impl InstalledProviderRuntime {
    pub fn entrypoint(&self) -> PathBuf {
        self.path.join(&self.manifest.runtime.entrypoint)
    }
}

fn existing_install(
    destination: &Path,
    expected: &ProviderRuntimeManifest,
) -> Result<(ProviderRuntimeInstallStatus, InstalledProviderRuntime)> {
    let actual = ProviderRuntimeManifest::read_from_dir(destination)?;
    if &actual != expected {
        bail!(
            "provider runtime coordinate {}/{} already contains different metadata",
            expected.runtime.id,
            expected.runtime.version
        );
    }
    Ok((
        ProviderRuntimeInstallStatus::AlreadyInstalled,
        InstalledProviderRuntime {
            path: destination.to_path_buf(),
            manifest: actual,
        },
    ))
}

fn copy_verified_bundle(
    source: &Path,
    destination: &Path,
    manifest: &ProviderRuntimeManifest,
) -> Result<()> {
    for relative in manifest.runtime.files.keys() {
        let source_file = source.join(relative);
        let destination_file = destination.join(relative);
        if let Some(parent) = destination_file.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(&source_file, &destination_file).with_context(|| {
            format!(
                "copy provider runtime file {} to {}",
                source_file.display(),
                destination_file.display()
            )
        })?;
        fs::set_permissions(&destination_file, fs::metadata(&source_file)?.permissions())?;
    }
    fs::copy(
        source.join(PROVIDER_RUNTIME_MANIFEST_FILE),
        destination.join(PROVIDER_RUNTIME_MANIFEST_FILE),
    )?;
    Ok(())
}

fn publish_install(
    temporary: &Path,
    destination: &Path,
    manifest: &ProviderRuntimeManifest,
) -> Result<(ProviderRuntimeInstallStatus, InstalledProviderRuntime)> {
    match fs::rename(temporary, destination) {
        Ok(()) => Ok((
            ProviderRuntimeInstallStatus::Installed,
            InstalledProviderRuntime {
                path: destination.to_path_buf(),
                manifest: manifest.clone(),
            },
        )),
        Err(_error) if destination.exists() => {
            let _ = fs::remove_dir_all(temporary);
            existing_install(destination, manifest)
        }
        Err(error) => Err(error).with_context(|| {
            format!(
                "publish provider runtime cache install {}",
                destination.display()
            )
        }),
    }
}

fn child_directories(root: &Path) -> Result<Vec<PathBuf>> {
    let mut directories = Vec::new();
    for entry in fs::read_dir(root).with_context(|| format!("read {}", root.display()))? {
        let entry = entry?;
        if entry.file_type()?.is_dir() && !entry.file_name().to_string_lossy().starts_with('.') {
            directories.push(entry.path());
        }
    }
    directories.sort();
    Ok(directories)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        PROVIDER_RUNTIME_SCHEMA_VERSION, ProviderRuntimeArtifact, ProviderRuntimeModel,
        ProviderRuntimePlatform,
    };
    use sha2::{Digest, Sha256};
    use std::collections::{BTreeMap, BTreeSet};

    fn bundle(root: &Path, contents: &[u8]) -> ProviderRuntimeManifest {
        let binary = root.join("bin/provider");
        fs::create_dir_all(binary.parent().unwrap()).unwrap();
        fs::write(&binary, contents).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&binary, fs::Permissions::from_mode(0o755)).unwrap();
        }
        let digest = format!("{:x}", Sha256::digest(contents));
        let manifest = ProviderRuntimeManifest {
            schema_version: PROVIDER_RUNTIME_SCHEMA_VERSION,
            runtime: ProviderRuntimeArtifact {
                id: "fixture-provider".to_string(),
                version: "1.0.0".to_string(),
                provider_kind: "fixture".to_string(),
                protocol_version: "1.0".to_string(),
                platform: ProviderRuntimePlatform {
                    os: std::env::consts::OS.to_string(),
                    arch: std::env::consts::ARCH.to_string(),
                    target: None,
                    minimum_os_version: None,
                },
                entrypoint: "bin/provider".to_string(),
                models: vec![ProviderRuntimeModel {
                    id: "fixture/model".to_string(),
                    kind: "fixture".to_string(),
                }],
                features: BTreeSet::new(),
                files: BTreeMap::from([("bin/provider".to_string(), digest)]),
                build: BTreeMap::new(),
                signature: None,
                url: None,
                archive_sha256: None,
            },
        };
        manifest.write_to_dir(root).unwrap();
        manifest
    }

    #[test]
    fn installs_immutably_and_reuses_identical_coordinates() {
        let temp = tempfile::tempdir().unwrap();
        let bundle_dir = temp.path().join("bundle");
        fs::create_dir(&bundle_dir).unwrap();
        let manifest = bundle(&bundle_dir, b"runtime");
        let cache = ProviderRuntimeCache::new(temp.path().join("cache"));

        let (first_status, first) = cache.install_from_dir(&bundle_dir).unwrap();
        let (second_status, second) = cache.install_from_dir(&bundle_dir).unwrap();

        assert_eq!(first_status, ProviderRuntimeInstallStatus::Installed);
        assert_eq!(
            second_status,
            ProviderRuntimeInstallStatus::AlreadyInstalled
        );
        assert_eq!(first, second);
        assert_eq!(first.manifest, manifest);
        assert!(first.entrypoint().is_file());
    }

    #[test]
    fn rejects_different_bytes_at_an_existing_coordinate() {
        let temp = tempfile::tempdir().unwrap();
        let first_bundle = temp.path().join("first");
        let second_bundle = temp.path().join("second");
        fs::create_dir(&first_bundle).unwrap();
        fs::create_dir(&second_bundle).unwrap();
        bundle(&first_bundle, b"first");
        bundle(&second_bundle, b"second");
        let cache = ProviderRuntimeCache::new(temp.path().join("cache"));
        cache.install_from_dir(&first_bundle).unwrap();

        let error = cache.install_from_dir(&second_bundle).unwrap_err();
        assert!(
            error.to_string().contains("different metadata"),
            "{error:?}"
        );
    }
}
