use anyhow::{Context, Result, bail};
use mesh_llm_native_runtime::{
    InstalledNativeRuntime, NATIVE_RUNTIME_MANIFEST_FILE, NativeRuntimeCache, NativeRuntimeManifest,
};
use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

pub const NATIVE_RUNTIME_BUNDLE_DIR_ENV: &str = "MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR";

pub fn discover_native_runtime_bundle_dirs(explicit_dirs: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let environment_dirs = env::var_os(NATIVE_RUNTIME_BUNDLE_DIR_ENV)
        .map(|value| env::split_paths(&value).collect::<Vec<_>>())
        .unwrap_or_default();
    discover_native_runtime_bundle_dirs_from(
        explicit_dirs,
        &environment_dirs,
        env::current_exe().ok().as_deref(),
    )
}

pub fn discover_local_native_runtimes(
    explicit_dirs: &[PathBuf],
    cache: &NativeRuntimeCache,
) -> Result<Vec<InstalledNativeRuntime>> {
    let bundle_dirs = discover_native_runtime_bundle_dirs_lenient(explicit_dirs)?;
    let mut runtimes = Vec::new();
    let mut seen = BTreeSet::new();
    for path in bundle_dirs {
        if let Some(runtime) = read_installed_runtime_lenient(&path) {
            let identity = (
                runtime.mesh_version.clone(),
                runtime.native_runtime_id.clone(),
            );
            if seen.insert(identity) {
                runtimes.push(runtime);
            }
        }
    }
    append_cache_runtimes_lenient(cache, &mut runtimes, &mut seen)?;
    Ok(runtimes)
}

fn discover_native_runtime_bundle_dirs_lenient(explicit_dirs: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let environment_dirs = env::var_os(NATIVE_RUNTIME_BUNDLE_DIR_ENV)
        .map(|value| env::split_paths(&value).collect::<Vec<_>>())
        .unwrap_or_default();
    discover_native_runtime_bundle_dirs_from_with_policy(
        explicit_dirs,
        &environment_dirs,
        env::current_exe().ok().as_deref(),
        InvalidManifestPolicy::WarnAndSkip,
    )
}

fn append_cache_runtimes_lenient(
    cache: &NativeRuntimeCache,
    runtimes: &mut Vec<InstalledNativeRuntime>,
    seen: &mut BTreeSet<(String, String)>,
) -> Result<()> {
    if !cache.root().exists() {
        return Ok(());
    }
    for version_entry in fs::read_dir(cache.root())
        .with_context(|| format!("read native runtime cache {}", cache.root().display()))?
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
            if !runtime_dir.join(NATIVE_RUNTIME_MANIFEST_FILE).exists() {
                continue;
            }
            let Some(runtime) = read_installed_runtime_lenient(&runtime_dir) else {
                continue;
            };
            let identity = (
                runtime.mesh_version.clone(),
                runtime.native_runtime_id.clone(),
            );
            if seen.insert(identity) {
                runtimes.push(runtime);
            }
        }
    }
    Ok(())
}

fn read_installed_runtime_lenient(path: &Path) -> Option<InstalledNativeRuntime> {
    let manifest = match NativeRuntimeManifest::read_from_dir(path) {
        Ok(manifest) => manifest,
        Err(error) => {
            eprintln!(
                "warning: skipping malformed native runtime {}: {error:#}",
                path.display()
            );
            return None;
        }
    };
    let mesh_version = manifest
        .runtime
        .mesh_version
        .clone()
        .unwrap_or_else(|| "unknown".to_string());
    Some(InstalledNativeRuntime {
        mesh_version,
        native_runtime_id: manifest.runtime.id.clone(),
        flavor: manifest.runtime.backend.kind.to_string(),
        path: path.to_path_buf(),
        manifest,
    })
}

fn discover_native_runtime_bundle_dirs_from(
    explicit_dirs: &[PathBuf],
    environment_dirs: &[PathBuf],
    executable_path: Option<&Path>,
) -> Result<Vec<PathBuf>> {
    discover_native_runtime_bundle_dirs_from_with_policy(
        explicit_dirs,
        environment_dirs,
        executable_path,
        InvalidManifestPolicy::Reject,
    )
}

#[derive(Clone, Copy)]
enum InvalidManifestPolicy {
    Reject,
    WarnAndSkip,
}

fn discover_native_runtime_bundle_dirs_from_with_policy(
    explicit_dirs: &[PathBuf],
    environment_dirs: &[PathBuf],
    executable_path: Option<&Path>,
    invalid_manifest_policy: InvalidManifestPolicy,
) -> Result<Vec<PathBuf>> {
    let mut discovered = Vec::new();
    let mut seen = BTreeSet::new();
    for path in explicit_dirs {
        append_candidate(
            path,
            true,
            invalid_manifest_policy,
            &mut discovered,
            &mut seen,
        )?;
    }
    for path in environment_dirs {
        append_candidate(
            path,
            true,
            invalid_manifest_policy,
            &mut discovered,
            &mut seen,
        )?;
    }
    if let Some(executable_path) = executable_path {
        for path in executable_candidates(executable_path) {
            append_candidate(
                &path,
                false,
                invalid_manifest_policy,
                &mut discovered,
                &mut seen,
            )?;
        }
    }
    Ok(discovered)
}

fn executable_candidates(executable_path: &Path) -> Vec<PathBuf> {
    let executable_path = executable_path
        .canonicalize()
        .unwrap_or_else(|_| executable_path.to_path_buf());
    let Some(executable_dir) = executable_path.parent() else {
        return Vec::new();
    };
    let mut candidates = vec![executable_dir.join("native-runtimes")];
    if let Some(prefix) = executable_dir.parent() {
        candidates.push(
            prefix
                .join("lib")
                .join("mesh-llm")
                .join(crate::CURRENT_MESH_VERSION)
                .join("native-runtimes"),
        );
        candidates.push(prefix.join("lib").join("mesh-llm").join("native-runtimes"));
        candidates.push(prefix.join("libexec").join("native-runtimes"));
    }
    candidates
}

fn append_candidate(
    candidate: &Path,
    required: bool,
    invalid_manifest_policy: InvalidManifestPolicy,
    discovered: &mut Vec<PathBuf>,
    seen: &mut BTreeSet<PathBuf>,
) -> Result<()> {
    if !candidate.exists() {
        if required {
            bail!(
                "native runtime bundle directory does not exist: {}",
                candidate.display()
            );
        }
        return Ok(());
    }
    let candidate = candidate
        .canonicalize()
        .with_context(|| format!("canonicalize native runtime path {}", candidate.display()))?;
    if candidate.join(NATIVE_RUNTIME_MANIFEST_FILE).is_file() {
        append_runtime_dir(&candidate, invalid_manifest_policy, discovered, seen)?;
        return Ok(());
    }
    let root = if candidate.join("native-runtimes").is_dir() {
        candidate.join("native-runtimes")
    } else {
        candidate.clone()
    };
    let mut runtime_dirs = fs::read_dir(&root)
        .with_context(|| format!("read native runtime bundle root {}", root.display()))?
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            entry
                .file_type()
                .ok()
                .filter(|file_type| file_type.is_dir())
                .map(|_| entry.path())
        })
        .filter(|path| path.join(NATIVE_RUNTIME_MANIFEST_FILE).is_file())
        .collect::<Vec<_>>();
    runtime_dirs.sort();
    if runtime_dirs.is_empty() && required {
        bail!(
            "native runtime bundle path contains no runtime manifests: {}",
            candidate.display()
        );
    }
    for runtime_dir in runtime_dirs {
        append_runtime_dir(&runtime_dir, invalid_manifest_policy, discovered, seen)?;
    }
    Ok(())
}

fn append_runtime_dir(
    runtime_dir: &Path,
    invalid_manifest_policy: InvalidManifestPolicy,
    discovered: &mut Vec<PathBuf>,
    seen: &mut BTreeSet<PathBuf>,
) -> Result<()> {
    let runtime_dir = runtime_dir
        .canonicalize()
        .with_context(|| format!("canonicalize native runtime {}", runtime_dir.display()))?;
    if let Err(error) = NativeRuntimeManifest::read_from_dir(&runtime_dir) {
        match invalid_manifest_policy {
            InvalidManifestPolicy::Reject => {
                return Err(error)
                    .with_context(|| format!("validate native runtime {}", runtime_dir.display()));
            }
            InvalidManifestPolicy::WarnAndSkip => {
                eprintln!(
                    "warning: skipping malformed native runtime {}: {error:#}",
                    runtime_dir.display()
                );
                return Ok(());
            }
        }
    }
    if seen.insert(runtime_dir.clone()) {
        discovered.push(runtime_dir);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_llm_native_runtime::{
        NativeRuntimeArtifact, NativeRuntimeBackend, NativeRuntimePlatform,
    };

    fn write_runtime(path: &Path, id: &str) {
        fs::create_dir_all(path.join("lib")).unwrap();
        fs::write(path.join("lib/libllama.so"), b"runtime").unwrap();
        NativeRuntimeManifest {
            runtime: NativeRuntimeArtifact {
                id: id.to_string(),
                mesh_version: Some("0.75.0".to_string()),
                skippy_abi: "0.1.25".to_string(),
                platform: NativeRuntimePlatform {
                    os: "linux".to_string(),
                    arch: "x86_64".to_string(),
                    target: None,
                },
                backend: NativeRuntimeBackend::cpu(),
                rank: 0,
                libraries: vec!["lib/libllama.so".to_string()],
                files: Default::default(),
                tools: Default::default(),
                url: None,
                sha256: None,
                signature: None,
            },
        }
        .write_to_dir(path)
        .unwrap();
    }

    #[test]
    fn explicit_product_bundle_expands_runtime_children() {
        let temp = tempfile::tempdir().unwrap();
        let product = temp.path().join("mesh-bundle");
        let runtime = product.join("native-runtimes/runtime-a");
        write_runtime(&runtime, "runtime-a");

        let discovered = discover_native_runtime_bundle_dirs_from(&[product], &[], None).unwrap();

        assert_eq!(discovered, vec![runtime.canonicalize().unwrap()]);
    }

    #[test]
    fn explicit_and_environment_dirs_precede_installed_candidates() {
        let temp = tempfile::tempdir().unwrap();
        let explicit = temp.path().join("explicit");
        let environment = temp.path().join("environment");
        let prefix = temp.path().join("prefix");
        let executable = prefix.join("bin/mesh-llm");
        let adjacent = prefix.join("bin/native-runtimes/adjacent");
        let versioned_package = prefix.join(format!(
            "lib/mesh-llm/{}/native-runtimes/versioned-package",
            crate::CURRENT_MESH_VERSION
        ));
        let package = prefix.join("lib/mesh-llm/native-runtimes/package");
        let homebrew = prefix.join("libexec/native-runtimes/homebrew");
        fs::create_dir_all(executable.parent().unwrap()).unwrap();
        fs::write(&executable, b"host").unwrap();
        for (path, id) in [
            (&explicit, "explicit"),
            (&environment, "environment"),
            (&adjacent, "adjacent"),
            (&versioned_package, "versioned-package"),
            (&package, "package"),
            (&homebrew, "homebrew"),
        ] {
            write_runtime(path, id);
        }

        let discovered = discover_native_runtime_bundle_dirs_from(
            std::slice::from_ref(&explicit),
            std::slice::from_ref(&environment),
            Some(&executable),
        )
        .unwrap();

        assert_eq!(
            discovered,
            vec![
                explicit.canonicalize().unwrap(),
                environment.canonicalize().unwrap(),
                adjacent.canonicalize().unwrap(),
                versioned_package.canonicalize().unwrap(),
                package.canonicalize().unwrap(),
                homebrew.canonicalize().unwrap(),
            ]
        );
    }

    #[test]
    fn missing_explicit_bundle_is_rejected() {
        let temp = tempfile::tempdir().unwrap();
        let missing = temp.path().join("missing");

        let error = discover_native_runtime_bundle_dirs_from(&[missing], &[], None).unwrap_err();

        assert!(error.to_string().contains("does not exist"), "{error:?}");
    }

    #[test]
    fn duplicate_runtime_paths_are_returned_once() {
        let temp = tempfile::tempdir().unwrap();
        let runtime = temp.path().join("runtime");
        write_runtime(&runtime, "runtime");

        let discovered = discover_native_runtime_bundle_dirs_from(
            std::slice::from_ref(&runtime),
            std::slice::from_ref(&runtime),
            None,
        )
        .unwrap();

        assert_eq!(discovered, vec![runtime.canonicalize().unwrap()]);
    }

    #[test]
    fn local_runtime_listing_prefers_bundle_and_includes_distinct_cache_entries() {
        let temp = tempfile::tempdir().unwrap();
        let bundle = temp.path().join("bundle");
        let cached_source = temp.path().join("cached-source");
        write_runtime(&bundle, "runtime-a");
        write_runtime(&cached_source, "runtime-b");
        let cache = NativeRuntimeCache::new(temp.path().join("cache"));
        cache.install_from_dir(&bundle).unwrap();
        let cached = cache.install_from_dir(&cached_source).unwrap();

        let discovered =
            discover_local_native_runtimes(std::slice::from_ref(&bundle), &cache).unwrap();

        assert_eq!(discovered.len(), 2);
        assert_eq!(discovered[0].native_runtime_id, "runtime-a");
        assert_eq!(discovered[0].path, bundle.canonicalize().unwrap());
        assert_eq!(discovered[1], cached);
    }

    #[test]
    fn local_runtime_listing_skips_malformed_bundle_and_cache_candidates() {
        let temp = tempfile::tempdir().unwrap();
        let product = temp.path().join("mesh-bundle");
        let malformed_bundle = product.join("native-runtimes/malformed-bundle");
        let valid_bundle = product.join("native-runtimes/valid-bundle");
        fs::create_dir_all(&malformed_bundle).unwrap();
        fs::write(
            malformed_bundle.join(NATIVE_RUNTIME_MANIFEST_FILE),
            b"not json",
        )
        .unwrap();
        write_runtime(&valid_bundle, "valid-bundle");
        let cache = NativeRuntimeCache::new(temp.path().join("cache"));
        let malformed_cache = cache.root().join("0.75.0/malformed-cache");
        let valid_cache_source = temp.path().join("valid-cache-source");
        fs::create_dir_all(&malformed_cache).unwrap();
        fs::write(
            malformed_cache.join(NATIVE_RUNTIME_MANIFEST_FILE),
            b"not json",
        )
        .unwrap();
        write_runtime(&valid_cache_source, "valid-cache");
        let valid_cache = cache.install_from_dir(&valid_cache_source).unwrap();

        let discovered = discover_local_native_runtimes(std::slice::from_ref(&product), &cache)
            .expect("valid inventory should remain available");

        assert_eq!(discovered.len(), 2);
        assert_eq!(discovered[0].native_runtime_id, "valid-bundle");
        assert_eq!(discovered[0].path, valid_bundle.canonicalize().unwrap());
        assert_eq!(discovered[1], valid_cache);
    }
}
