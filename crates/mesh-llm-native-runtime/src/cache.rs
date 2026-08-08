use crate::{NativeRuntimeManifest, manifest::NATIVE_RUNTIME_MANIFEST_FILE};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
};

/// Relative path of the hardware benchmark executable shipped by GPU runtimes.
///
/// The manifest checksum must include this path before the executable is
/// selected or run. CPU and Vulkan runtimes intentionally do not provide it.
#[cfg(target_os = "windows")]
pub const GPU_BENCHMARK_TOOL_PATH: &str = "tools/mesh-llm-gpu-benchmark.exe";
#[cfg(not(target_os = "windows"))]
pub const GPU_BENCHMARK_TOOL_PATH: &str = "tools/mesh-llm-gpu-benchmark";

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

/// A cache entry that could not be read as a valid installed runtime during a
/// lenient scan, together with the reason it was skipped.
#[derive(Clone, Debug)]
pub struct SkippedNativeRuntime {
    pub path: PathBuf,
    pub reason: String,
}

/// Result of leniently enumerating the runtime cache: every readable runtime
/// plus every entry that had to be skipped.
#[derive(Debug, Default)]
pub struct LenientInstalledScan {
    pub runtimes: Vec<InstalledNativeRuntime>,
    pub skipped: Vec<SkippedNativeRuntime>,
}

impl InstalledNativeRuntime {
    /// Resolve the manifest-verified GPU benchmark helper inside this runtime.
    pub fn gpu_benchmark_tool(&self) -> Result<PathBuf> {
        if !self
            .manifest
            .runtime
            .tools
            .contains_key(GPU_BENCHMARK_TOOL_PATH)
        {
            anyhow::bail!(
                "native runtime {} does not provide the GPU benchmark tool",
                self.native_runtime_id
            );
        }
        let path = self.path.join(GPU_BENCHMARK_TOOL_PATH);
        if !path.is_file() {
            anyhow::bail!(
                "native runtime GPU benchmark tool is missing: {}",
                path.display()
            );
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            if fs::metadata(&path)?.permissions().mode() & 0o111 == 0 {
                anyhow::bail!(
                    "native runtime GPU benchmark tool is not executable: {}",
                    path.display()
                );
            }
        }
        Ok(path)
    }
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
            installed.extend(installed_in_version_dir(&version_entry.path())?);
        }
        installed.sort_by(|left, right| {
            (&left.mesh_version, &left.native_runtime_id)
                .cmp(&(&right.mesh_version, &right.native_runtime_id))
        });
        Ok(installed)
    }

    /// Enumerates the whole cache leniently: entries whose manifests are
    /// missing, malformed, or fail checksum verification are collected as
    /// skipped instead of failing the scan.
    ///
    /// Startup paths must use this instead of [`Self::installed`] so a stale
    /// entry written by an older MeshLLM version (pre-checksum manifests,
    /// issue #1162) can never abort runtime resolution while a valid runtime
    /// is present or installable.
    pub fn installed_lenient(&self) -> Result<LenientInstalledScan> {
        let mut scan = LenientInstalledScan::default();
        if !self.root.exists() {
            return Ok(scan);
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
                let runtime_dir = runtime_entry.path();
                let manifest_path = runtime_dir.join(NATIVE_RUNTIME_MANIFEST_FILE);
                match fs::metadata(&manifest_path) {
                    Ok(metadata) if metadata.is_file() => {}
                    Ok(_) => {
                        scan.skipped.push(SkippedNativeRuntime {
                            path: runtime_dir,
                            reason: format!("{NATIVE_RUNTIME_MANIFEST_FILE} is not a regular file"),
                        });
                        continue;
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                        scan.skipped.push(SkippedNativeRuntime {
                            path: runtime_dir,
                            reason: format!("{NATIVE_RUNTIME_MANIFEST_FILE} is missing"),
                        });
                        continue;
                    }
                    Err(error) => {
                        scan.skipped.push(SkippedNativeRuntime {
                            path: runtime_dir,
                            reason: format!(
                                "read {NATIVE_RUNTIME_MANIFEST_FILE} metadata: {error}"
                            ),
                        });
                        continue;
                    }
                }
                match installed_runtime_from_dir(&runtime_dir) {
                    Ok(Some(runtime)) => scan.runtimes.push(runtime),
                    Ok(None) => {}
                    Err(error) => scan.skipped.push(SkippedNativeRuntime {
                        path: runtime_dir,
                        reason: format!("{error:#}"),
                    }),
                }
            }
        }
        scan.runtimes.sort_by(|left, right| {
            (&left.mesh_version, &left.native_runtime_id)
                .cmp(&(&right.mesh_version, &right.native_runtime_id))
        });
        Ok(scan)
    }

    /// Returns strictly validated runtimes for one MeshLLM version.
    ///
    /// TODO(issue #1162): Remove this resolver-specific compatibility boundary
    /// once the oldest supported upgrade path no longer contains pre-checksum
    /// runtime caches, if full-cache resolver enumeration is safe again.
    pub(crate) fn installed_for_version(
        &self,
        mesh_version: &str,
    ) -> Result<Vec<InstalledNativeRuntime>> {
        installed_in_version_dir(&self.root.join(mesh_version))
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

fn installed_in_version_dir(version_dir: &Path) -> Result<Vec<InstalledNativeRuntime>> {
    let mut installed = Vec::new();
    if !version_dir.is_dir() {
        return Ok(installed);
    }
    for runtime_entry in fs::read_dir(version_dir)
        .with_context(|| format!("read native runtime cache {}", version_dir.display()))?
    {
        let runtime_entry = runtime_entry?;
        if !runtime_entry.file_type()?.is_dir() {
            continue;
        }
        if let Some(runtime) = installed_runtime_from_dir(&runtime_entry.path())? {
            installed.push(runtime);
        }
    }
    installed.sort_by(|left, right| left.native_runtime_id.cmp(&right.native_runtime_id));
    Ok(installed)
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
                files: Default::default(),
                tools: Default::default(),
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
    fn installed_for_version_ignores_legacy_cache_versions() {
        let temp = tempfile::tempdir().unwrap();
        let cache = NativeRuntimeCache::new(temp.path().join("cache"));
        write_runtime(
            &cache.runtime_dir("0.75.0", "meshllm-native-linux-x86_64-cpu"),
            "0.75.0",
            "meshllm-native-linux-x86_64-cpu",
        );

        let legacy = cache.runtime_dir("0.74.0", "meshllm-native-linux-x86_64-cpu");
        fs::create_dir_all(legacy.join("lib")).unwrap();
        fs::write(legacy.join("lib/libmeshllm_ffi.so"), b"legacy runtime").unwrap();
        fs::write(
            legacy.join(NATIVE_RUNTIME_MANIFEST_FILE),
            r#"{
  "runtime": {
    "id": "meshllm-native-linux-x86_64-cpu",
    "mesh_version": "0.74.0",
    "skippy_abi": "0.1.25",
    "platform": {"os": "linux", "arch": "x86_64"},
    "backend": {"kind": "cpu"},
    "libraries": ["lib/libmeshllm_ffi.so"]
  }
}"#,
        )
        .unwrap();

        let installed = cache.installed_for_version("0.75.0").unwrap();

        assert_eq!(installed.len(), 1);
        assert_eq!(installed[0].mesh_version, "0.75.0");
        assert!(cache.installed().is_err());
    }

    #[test]
    fn installed_lenient_skips_legacy_manifest_and_keeps_valid_runtime() {
        let temp = tempfile::tempdir().unwrap();
        let cache = NativeRuntimeCache::new(temp.path().join("cache"));
        write_runtime(
            &cache.runtime_dir("0.75.0", "meshllm-native-linux-x86_64-cpu"),
            "0.75.0",
            "meshllm-native-linux-x86_64-cpu",
        );

        let legacy = cache.runtime_dir("0.74.0", "meshllm-native-linux-x86_64-cpu");
        fs::create_dir_all(legacy.join("lib")).unwrap();
        fs::write(legacy.join("lib/libmeshllm_ffi.so"), b"legacy runtime").unwrap();
        fs::write(
            legacy.join(NATIVE_RUNTIME_MANIFEST_FILE),
            r#"{
  "runtime": {
    "id": "meshllm-native-linux-x86_64-cpu",
    "mesh_version": "0.74.0",
    "skippy_abi": "0.1.25",
    "platform": {"os": "linux", "arch": "x86_64"},
    "backend": {"kind": "cpu"},
    "libraries": ["lib/libmeshllm_ffi.so"]
  }
}"#,
        )
        .unwrap();

        let scan = cache.installed_lenient().unwrap();

        assert_eq!(scan.runtimes.len(), 1);
        assert_eq!(scan.runtimes[0].mesh_version, "0.75.0");
        assert_eq!(scan.skipped.len(), 1);
        assert_eq!(scan.skipped[0].path, legacy);
        assert!(
            scan.skipped[0].reason.contains("checksum"),
            "unexpected skip reason: {}",
            scan.skipped[0].reason
        );
        // The strict scan still fails on the same cache; startup must not use it.
        assert!(cache.installed().is_err());
    }

    #[test]
    fn installed_lenient_reports_runtime_dir_without_manifest_as_skipped() {
        let temp = tempfile::tempdir().unwrap();
        let cache = NativeRuntimeCache::new(temp.path().join("cache"));
        write_runtime(
            &cache.runtime_dir("0.75.0", "meshllm-native-linux-x86_64-cpu"),
            "0.75.0",
            "meshllm-native-linux-x86_64-cpu",
        );

        let orphan = cache.runtime_dir("0.75.0", "meshllm-native-linux-x86_64-vulkan");
        fs::create_dir_all(orphan.join("lib")).unwrap();
        fs::write(orphan.join("lib/libmeshllm_ffi.so"), b"orphan runtime").unwrap();

        let scan = cache.installed_lenient().unwrap();

        assert_eq!(scan.runtimes.len(), 1);
        assert_eq!(
            scan.runtimes[0].native_runtime_id,
            "meshllm-native-linux-x86_64-cpu"
        );
        assert_eq!(scan.skipped.len(), 1);
        assert_eq!(scan.skipped[0].path, orphan);
        assert!(
            scan.skipped[0].reason.contains("missing"),
            "unexpected skip reason: {}",
            scan.skipped[0].reason
        );
    }

    #[test]
    fn installed_for_version_ignores_file_at_version_path() {
        let temp = tempfile::tempdir().unwrap();
        let cache = NativeRuntimeCache::new(temp.path().join("cache"));
        fs::create_dir_all(cache.root()).unwrap();
        fs::write(cache.root().join("0.75.0"), b"partial cache artifact").unwrap();

        let installed = cache.installed_for_version("0.75.0").unwrap();

        assert!(installed.is_empty());
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

    #[cfg(unix)]
    #[test]
    fn gpu_benchmark_tool_must_be_declared_and_executable() {
        use std::os::unix::fs::PermissionsExt;

        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source");
        write_runtime(&source, "0.68.0", "meshllm-native-linux-x86_64-cuda12");

        let cache = NativeRuntimeCache::new(temp.path().join("cache"));
        let mut installed = cache.install_from_dir(&source).unwrap();
        let tool = installed.path.join(GPU_BENCHMARK_TOOL_PATH);
        fs::create_dir_all(tool.parent().unwrap()).unwrap();
        fs::write(&tool, b"benchmark tool").unwrap();
        installed
            .manifest
            .runtime
            .tools
            .insert(GPU_BENCHMARK_TOOL_PATH.to_string(), "0".repeat(64));

        let error = installed.gpu_benchmark_tool().unwrap_err();
        assert!(error.to_string().contains("not executable"));

        let mut permissions = fs::metadata(&tool).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&tool, permissions).unwrap();
        assert_eq!(installed.gpu_benchmark_tool().unwrap(), tool);
    }
}
