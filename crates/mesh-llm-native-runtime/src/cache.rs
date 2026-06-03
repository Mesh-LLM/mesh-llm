use crate::{NativeRuntimeManifest, manifest::NATIVE_RUNTIME_MANIFEST_FILE};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NativeRuntimeCacheRoot {
    pub path: PathBuf,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct InstalledNativeRuntime {
    pub mesh_version: String,
    pub native_runtime_id: String,
    pub flavor: String,
    pub path: PathBuf,
    pub manifest: NativeRuntimeManifest,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeRuntimePruneMode {
    KeepActiveAndPrevious,
    ActiveOnly,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CachePrunePlan {
    #[serde(default)]
    pub remove_dirs: Vec<PathBuf>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeRuntimeCache {
    root: PathBuf,
}

impl NativeRuntimeCache {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn runtime_dir(&self, mesh_version: &str, native_runtime_id: &str) -> PathBuf {
        self.root.join(mesh_version).join(native_runtime_id)
    }

    pub fn installed(&self) -> Result<Vec<InstalledNativeRuntime>> {
        let mut installed = Vec::new();
        if !self.root.exists() {
            return Ok(installed);
        }
        for version_entry in fs::read_dir(&self.root)
            .with_context(|| format!("read native runtime cache {}", self.root.display()))?
        {
            let version_entry = version_entry?;
            if !version_entry.file_type()?.is_dir() {
                continue;
            }
            for runtime_entry in fs::read_dir(version_entry.path())? {
                let runtime_entry = runtime_entry?;
                if !runtime_entry.file_type()?.is_dir() {
                    continue;
                }
                if let Some(runtime) = installed_runtime_from_dir(&runtime_entry.path())? {
                    installed.push(runtime);
                }
            }
        }
        installed.sort_by(|left, right| {
            (&left.mesh_version, &left.native_runtime_id)
                .cmp(&(&right.mesh_version, &right.native_runtime_id))
        });
        Ok(installed)
    }

    pub fn find_installed(
        &self,
        mesh_version: &str,
        native_runtime_id: &str,
    ) -> Result<Option<InstalledNativeRuntime>> {
        let dir = self.runtime_dir(mesh_version, native_runtime_id);
        if !dir.join(NATIVE_RUNTIME_MANIFEST_FILE).exists() {
            return Ok(None);
        }
        installed_runtime_from_dir(&dir)
    }

    pub fn install_from_dir(&self, source_dir: &Path) -> Result<InstalledNativeRuntime> {
        let manifest = NativeRuntimeManifest::read_from_dir(source_dir)?;
        manifest.validate()?;
        let mesh_version = manifest
            .runtime
            .mesh_version
            .as_deref()
            .unwrap_or("unknown");
        let target = self.runtime_dir(mesh_version, manifest.runtime.native_runtime_id());
        if target.exists() {
            fs::remove_dir_all(&target)
                .with_context(|| format!("replace native runtime {}", target.display()))?;
        }
        copy_dir_recursive(source_dir, &target)?;
        installed_runtime_from_dir(&target)?.context("installed native runtime manifest missing")
    }

    pub fn remove(&self, mesh_version: &str, native_runtime_id: &str) -> Result<bool> {
        let dir = self.runtime_dir(mesh_version, native_runtime_id);
        if !dir.exists() {
            return Ok(false);
        }
        fs::remove_dir_all(&dir)
            .with_context(|| format!("remove native runtime {}", dir.display()))?;
        Ok(true)
    }

    pub fn prune_plan(
        &self,
        active_mesh_version: &str,
        mode: NativeRuntimePruneMode,
    ) -> Result<CachePrunePlan> {
        let mut versions = self.installed_versions()?;
        versions.sort();
        let previous = match mode {
            NativeRuntimePruneMode::ActiveOnly => None,
            NativeRuntimePruneMode::KeepActiveAndPrevious => versions
                .iter()
                .rfind(|version| version.as_str() != active_mesh_version)
                .cloned(),
        };
        let remove_dirs = versions
            .into_iter()
            .filter(|version| version != active_mesh_version)
            .filter(|version| Some(version) != previous.as_ref())
            .map(|version| self.root.join(version))
            .collect();
        Ok(CachePrunePlan { remove_dirs })
    }

    pub fn prune(
        &self,
        active_mesh_version: &str,
        mode: NativeRuntimePruneMode,
    ) -> Result<CachePrunePlan> {
        let plan = self.prune_plan(active_mesh_version, mode)?;
        for dir in &plan.remove_dirs {
            if dir.exists() {
                fs::remove_dir_all(dir)
                    .with_context(|| format!("remove native runtime cache {}", dir.display()))?;
            }
        }
        Ok(plan)
    }

    fn installed_versions(&self) -> Result<Vec<String>> {
        if !self.root.exists() {
            return Ok(Vec::new());
        }
        let mut versions = Vec::new();
        for entry in fs::read_dir(&self.root)
            .with_context(|| format!("read native runtime cache {}", self.root.display()))?
        {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                versions.push(entry.file_name().to_string_lossy().to_string());
            }
        }
        Ok(versions)
    }
}

pub fn native_runtime_cache_root(base_cache_dir: &Path) -> PathBuf {
    base_cache_dir.join("mesh-llm").join("native-runtimes")
}

fn installed_runtime_from_dir(dir: &Path) -> Result<Option<InstalledNativeRuntime>> {
    if !dir.join(NATIVE_RUNTIME_MANIFEST_FILE).exists() {
        return Ok(None);
    }
    let manifest = NativeRuntimeManifest::read_from_dir(dir)?;
    let mesh_version = manifest
        .runtime
        .mesh_version
        .clone()
        .unwrap_or_else(|| "unknown".to_string());
    Ok(Some(InstalledNativeRuntime {
        mesh_version,
        native_runtime_id: manifest.runtime.id.clone(),
        flavor: manifest.runtime.backend.kind.to_string(),
        path: dir.to_path_buf(),
        manifest,
    }))
}

fn copy_dir_recursive(source: &Path, target: &Path) -> Result<()> {
    fs::create_dir_all(target).with_context(|| format!("create {}", target.display()))?;
    for entry in fs::read_dir(source).with_context(|| format!("read {}", source.display()))? {
        let entry = entry?;
        let source_path = entry.path();
        let target_path = target.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir_recursive(&source_path, &target_path)?;
        } else {
            fs::copy(&source_path, &target_path).with_context(|| {
                format!(
                    "copy {} to {}",
                    source_path.display(),
                    target_path.display()
                )
            })?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        NativeRuntimeArtifact, NativeRuntimeBackend, NativeRuntimeManifest, NativeRuntimePlatform,
    };

    fn write_runtime(dir: &Path, version: &str, id: &str) {
        fs::create_dir_all(dir.join("lib")).unwrap();
        fs::write(dir.join("lib/libmeshllm_ffi.so"), b"native runtime").unwrap();
        let manifest = NativeRuntimeManifest {
            runtime: NativeRuntimeArtifact {
                id: id.to_string(),
                mesh_version: Some(version.to_string()),
                skippy_abi: "0.1.25".to_string(),
                platform: NativeRuntimePlatform {
                    os: "linux".to_string(),
                    arch: "x86_64".to_string(),
                    target: None,
                },
                backend: NativeRuntimeBackend::cpu(),
                rank: 0,
                libraries: vec!["lib/libmeshllm_ffi.so".to_string()],
                url: None,
                sha256: None,
                signature: None,
            },
        };
        manifest.write_to_dir(dir).unwrap();
    }

    #[test]
    fn installs_bundle_runtime_into_versioned_cache() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source");
        write_runtime(&source, "0.68.0", "meshllm-native-linux-x86_64-cpu");

        let cache = NativeRuntimeCache::new(temp.path().join("cache"));
        let installed = cache.install_from_dir(&source).unwrap();

        assert_eq!(installed.mesh_version, "0.68.0");
        assert!(installed.path.ends_with("meshllm-native-linux-x86_64-cpu"));
    }

    #[test]
    fn prune_keeps_active_and_previous_by_default() {
        let temp = tempfile::tempdir().unwrap();
        let cache = NativeRuntimeCache::new(temp.path().join("cache"));
        for version in ["0.67.0", "0.68.0", "0.69.0"] {
            write_runtime(
                &cache.runtime_dir(version, "meshllm-native-linux-x86_64-cpu"),
                version,
                "meshllm-native-linux-x86_64-cpu",
            );
        }

        let plan = cache
            .prune_plan("0.69.0", NativeRuntimePruneMode::KeepActiveAndPrevious)
            .unwrap();

        assert_eq!(plan.remove_dirs, vec![cache.root().join("0.67.0")]);
    }

    #[test]
    fn installed_runtime_exposes_load_plan() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source");
        write_runtime(&source, "0.68.0", "meshllm-native-linux-x86_64-cpu");

        let cache = NativeRuntimeCache::new(temp.path().join("cache"));
        let installed = cache.install_from_dir(&source).unwrap();
        let plan = installed.load_plan().unwrap();

        assert_eq!(plan.native_runtime_id, "meshllm-native-linux-x86_64-cpu");
        assert_eq!(
            plan.libraries,
            vec![
                cache
                    .runtime_dir("0.68.0", "meshllm-native-linux-x86_64-cpu")
                    .join("lib/libmeshllm_ffi.so")
            ]
        );
    }
}
