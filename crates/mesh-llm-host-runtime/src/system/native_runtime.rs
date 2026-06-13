#[cfg(feature = "dynamic-native-runtime")]
mod dynamic {
    use crate::system::native_runtime_install::{
        NativeRuntimeInstallOptions, NativeRuntimeInstallOutcome,
    };
    use anyhow::{Context, Result};
    use mesh_llm_native_runtime::{
        HostRuntimeProfile, NativeRuntimeArtifact, NativeRuntimeCache, NativeRuntimeLoadPlan,
        NativeRuntimeReleaseManifest, RuntimeSelection, select_native_runtime,
    };
    use std::path::PathBuf;

    #[derive(Clone, Debug)]
    pub(crate) struct LoadedNativeRuntime {
        pub(crate) native_runtime_id: String,
        pub(crate) libraries: Vec<PathBuf>,
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub(crate) enum NativeRuntimePlanSource {
        CacheHit,
        PostInstall,
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    pub(crate) struct NativeRuntimeStartupLoadPlan {
        pub(crate) cache_mesh_version: String,
        pub(crate) native_runtime_id: String,
        pub(crate) root: PathBuf,
        pub(crate) selected_library_path: PathBuf,
        pub(crate) libraries: Vec<PathBuf>,
        pub(crate) source: NativeRuntimePlanSource,
    }

    pub(crate) fn try_load_installed_native_runtime() -> Result<Option<LoadedNativeRuntime>> {
        try_load_installed_native_runtime_with(
            skippy_runtime::native_runtime_loaded,
            default_native_runtime_cache,
            host_runtime_profile,
            default_install_options,
            default_install_executor,
            |libraries| {
                unsafe { skippy_runtime::load_native_runtime_libraries(libraries) }
                    .map_err(anyhow::Error::from)
            },
        )
    }

    fn try_load_installed_native_runtime_with<
        NativeRuntimeLoadedFn,
        CacheFn,
        ProfileFn,
        InstallOptionsFn,
        InstallExecutorFn,
        LoadLibrariesFn,
    >(
        native_runtime_loaded: NativeRuntimeLoadedFn,
        cache: CacheFn,
        profile: ProfileFn,
        install_options: InstallOptionsFn,
        install_executor: InstallExecutorFn,
        load_libraries: LoadLibrariesFn,
    ) -> Result<Option<LoadedNativeRuntime>>
    where
        NativeRuntimeLoadedFn: Fn() -> bool,
        CacheFn: Fn() -> Result<NativeRuntimeCache>,
        ProfileFn: Fn() -> HostRuntimeProfile,
        InstallOptionsFn: Fn() -> NativeRuntimeInstallOptions,
        InstallExecutorFn: Fn(NativeRuntimeInstallOptions) -> Result<NativeRuntimeInstallOutcome>,
        LoadLibrariesFn: Fn(&[PathBuf]) -> Result<()>,
    {
        if native_runtime_loaded() {
            return Ok(None);
        }
        let Some(plan) = resolve_startup_native_runtime_plan_with(
            cache,
            profile,
            install_options,
            install_executor,
        )?
        else {
            return Ok(None);
        };
        load_libraries(&plan.libraries).with_context(|| {
            format!(
                "load native runtime {} from {}",
                plan.native_runtime_id,
                plan.root.display()
            )
        })?;
        Ok(Some(LoadedNativeRuntime {
            native_runtime_id: plan.native_runtime_id,
            libraries: plan.libraries,
        }))
    }

    fn resolve_startup_native_runtime_plan_with<
        CacheFn,
        ProfileFn,
        InstallOptionsFn,
        InstallExecutorFn,
    >(
        cache: CacheFn,
        profile: ProfileFn,
        install_options: InstallOptionsFn,
        install_executor: InstallExecutorFn,
    ) -> Result<Option<NativeRuntimeStartupLoadPlan>>
    where
        CacheFn: Fn() -> Result<NativeRuntimeCache>,
        ProfileFn: Fn() -> HostRuntimeProfile,
        InstallOptionsFn: Fn() -> NativeRuntimeInstallOptions,
        InstallExecutorFn: Fn(NativeRuntimeInstallOptions) -> Result<NativeRuntimeInstallOutcome>,
    {
        let cache = cache()?;
        let profile = profile();
        if let Some(plan) = resolve_installed_native_runtime_plan(
            &cache,
            &profile,
            crate::BUILD_VERSION,
            crate::RELEASE_VERSION,
            &RuntimeSelection::Recommended,
        )? {
            return Ok(Some(plan));
        }

        let mut options = install_options();
        if options.cache_dir.is_none() {
            options.cache_dir = Some(cache.root().to_path_buf());
        }

        tracing::info!(
            cache_root = %cache.root().display(),
            mesh_version = %options.mesh_version,
            "No compatible installed MeshLLM native runtime found; attempting one-shot startup install"
        );

        let install_result = install_executor(options.clone());
        match install_result {
            Ok(outcome) => {
                let load_plan = outcome.runtime.load_plan()?;
                Ok(Some(startup_load_plan_from_installed(
                    outcome.runtime.mesh_version.clone(),
                    load_plan,
                    NativeRuntimePlanSource::PostInstall,
                )?))
            }
            Err(err) => {
                tracing::warn!(
                    error = %err,
                    cache_root = %cache.root().display(),
                    mesh_version = %options.mesh_version,
                    manifest_path = ?options.manifest_path,
                    manifest_url = ?options.manifest_url,
                    bundle_dirs = ?options.bundle_dirs,
                    allow_download = options.allow_download,
                    "Failed to install a compatible MeshLLM native runtime during startup; continuing without dynamic native runtime"
                );
                Ok(None)
            }
        }
    }

    fn resolve_installed_native_runtime_plan(
        cache: &NativeRuntimeCache,
        profile: &HostRuntimeProfile,
        build_version: &str,
        release_version: &str,
        selection: &RuntimeSelection,
    ) -> Result<Option<NativeRuntimeStartupLoadPlan>> {
        let installed = cache.installed()?;
        if installed.is_empty() {
            return Ok(None);
        }
        let initial_cache_version =
            startup_native_runtime_cache_version(build_version, release_version);
        let manifest = NativeRuntimeReleaseManifest {
            mesh_version: initial_cache_version.to_string(),
            skippy_abi: installed
                .first()
                .map(|runtime| runtime.manifest.runtime.skippy_abi.clone())
                .unwrap_or_default(),
            artifacts: installed
                .iter()
                .map(|runtime| runtime.manifest.runtime.clone())
                .collect(),
        };
        let Some(candidate) =
            select_native_runtime(&manifest, profile, initial_cache_version, selection)
        else {
            return Ok(None);
        };
        load_plan_from_candidate(cache, &manifest, candidate.artifact)
    }

    fn startup_native_runtime_cache_version<'a>(
        _build_version: &'a str,
        release_version: &'a str,
    ) -> &'a str {
        release_version
    }

    fn load_plan_from_candidate(
        cache: &NativeRuntimeCache,
        manifest: &NativeRuntimeReleaseManifest,
        artifact: NativeRuntimeArtifact,
    ) -> Result<Option<NativeRuntimeStartupLoadPlan>> {
        let cache_mesh_version = artifact
            .mesh_version_or(manifest.mesh_version.as_str())
            .to_string();
        let Some(installed) =
            cache.find_installed(&cache_mesh_version, artifact.native_runtime_id())?
        else {
            return Ok(None);
        };
        let load_plan = installed.load_plan()?;
        Ok(Some(startup_load_plan_from_installed(
            cache_mesh_version,
            load_plan,
            NativeRuntimePlanSource::CacheHit,
        )?))
    }

    fn startup_load_plan_from_installed(
        cache_mesh_version: String,
        load_plan: NativeRuntimeLoadPlan,
        source: NativeRuntimePlanSource,
    ) -> Result<NativeRuntimeStartupLoadPlan> {
        let selected_library_path = load_plan
            .libraries
            .first()
            .cloned()
            .context("native runtime load plan did not include a library path")?;
        Ok(NativeRuntimeStartupLoadPlan {
            cache_mesh_version,
            native_runtime_id: load_plan.native_runtime_id,
            root: load_plan.root,
            selected_library_path,
            libraries: load_plan.libraries,
            source,
        })
    }

    fn default_native_runtime_cache() -> Result<NativeRuntimeCache> {
        crate::system::native_runtime_install::default_native_runtime_cache()
    }

    fn host_runtime_profile() -> HostRuntimeProfile {
        crate::system::native_runtime_install::host_runtime_profile()
    }

    fn default_install_options() -> NativeRuntimeInstallOptions {
        NativeRuntimeInstallOptions {
            mesh_version: crate::RELEASE_VERSION.to_string(),
            selection: RuntimeSelection::Recommended,
            ..Default::default()
        }
    }

    fn default_install_executor(
        options: NativeRuntimeInstallOptions,
    ) -> Result<NativeRuntimeInstallOutcome> {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .context("build native runtime startup install executor")?
            .block_on(crate::system::native_runtime_install::install_native_runtime(options))
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use mesh_llm_native_runtime::{
            NativeRuntimeBackend, NativeRuntimeManifest, NativeRuntimePlatform,
        };
        use std::{
            fs,
            path::Path,
            sync::{Arc, Mutex},
        };

        fn write_runtime(dir: &Path, version: &str, id: &str) {
            let library_rel_path = test_library_rel_path();
            fs::create_dir_all(dir.join(library_rel_path.parent().unwrap())).unwrap();
            fs::write(dir.join(&library_rel_path), b"native runtime").unwrap();
            let manifest = NativeRuntimeManifest {
                runtime: NativeRuntimeArtifact {
                    id: id.to_string(),
                    mesh_version: Some(version.to_string()),
                    skippy_abi: "0.1.25".to_string(),
                    platform: NativeRuntimePlatform {
                        os: std::env::consts::OS.to_string(),
                        arch: std::env::consts::ARCH.to_string(),
                        target: None,
                    },
                    backend: NativeRuntimeBackend::cpu(),
                    rank: 0,
                    libraries: vec![library_rel_path.to_string_lossy().to_string()],
                    url: None,
                    sha256: None,
                    signature: None,
                },
            };
            manifest.write_to_dir(dir).unwrap();
        }

        fn test_library_rel_path() -> PathBuf {
            let file = if cfg!(target_os = "windows") {
                "meshllm_ffi.dll"
            } else if cfg!(target_os = "macos") {
                "libmeshllm_ffi.dylib"
            } else {
                "libmeshllm_ffi.so"
            };
            PathBuf::from("lib").join(file)
        }

        fn test_install_options() -> NativeRuntimeInstallOptions {
            NativeRuntimeInstallOptions {
                mesh_version: "0.68.0".to_string(),
                allow_download: false,
                ..Default::default()
            }
        }

        #[test]
        fn sha_build_uses_release_cache_identity_for_installed_runtime_lookup() {
            let temp = tempfile::tempdir().unwrap();
            let cache = NativeRuntimeCache::new(temp.path().join("cache"));
            let runtime_id = "meshllm-native-runtime-test-cpu";
            let release_version = "0.68.0";
            let sha_build_version = "0.68.0+gAB131C";
            let runtime_dir = cache.runtime_dir(release_version, runtime_id);
            write_runtime(&runtime_dir, release_version, runtime_id);

            let plan = resolve_installed_native_runtime_plan(
                &cache,
                &HostRuntimeProfile::current_without_gpu_probe(),
                sha_build_version,
                release_version,
                &RuntimeSelection::Recommended,
            )
            .unwrap()
            .expect("expected cached runtime plan");

            assert_eq!(plan.cache_mesh_version, release_version);
            assert_eq!(plan.native_runtime_id, runtime_id);
            assert_eq!(plan.source, NativeRuntimePlanSource::CacheHit);
            assert_eq!(
                plan.selected_library_path,
                runtime_dir.join(test_library_rel_path())
            );
            assert_eq!(
                plan.libraries,
                vec![runtime_dir.join(test_library_rel_path())]
            );
        }

        #[test]
        fn selected_artifact_mesh_version_wins_over_release_fallback() {
            let temp = tempfile::tempdir().unwrap();
            let cache = NativeRuntimeCache::new(temp.path().join("cache"));
            let runtime_id = "meshllm-native-runtime-test-cpu";
            let release_version = "0.68.0";
            let artifact_mesh_version = "0.69.0";
            let runtime_dir = cache.runtime_dir(artifact_mesh_version, runtime_id);
            write_runtime(&runtime_dir, artifact_mesh_version, runtime_id);

            let plan = resolve_installed_native_runtime_plan(
                &cache,
                &HostRuntimeProfile::current_without_gpu_probe(),
                "0.68.0+gAB131C.dirty",
                release_version,
                &RuntimeSelection::Recommended,
            )
            .unwrap()
            .expect("expected cached runtime plan");

            assert_eq!(plan.cache_mesh_version, artifact_mesh_version);
            assert_eq!(plan.root, runtime_dir);
            assert_eq!(plan.source, NativeRuntimePlanSource::CacheHit);
        }

        #[test]
        fn startup_plan_can_represent_post_install_source_without_loading() {
            let temp = tempfile::tempdir().unwrap();
            let runtime_id = "meshllm-native-runtime-test-cpu";
            let release_version = "0.68.0";
            let runtime_dir = temp.path().join(runtime_id);
            write_runtime(&runtime_dir, release_version, runtime_id);
            let load_plan = NativeRuntimeLoadPlan {
                mesh_version: release_version.to_string(),
                native_runtime_id: runtime_id.to_string(),
                root: runtime_dir.clone(),
                libraries: vec![runtime_dir.join(test_library_rel_path())],
            };

            let plan = startup_load_plan_from_installed(
                release_version.to_string(),
                load_plan,
                NativeRuntimePlanSource::PostInstall,
            )
            .unwrap();

            assert_eq!(plan.cache_mesh_version, release_version);
            assert_eq!(plan.root, runtime_dir);
            assert_eq!(plan.source, NativeRuntimePlanSource::PostInstall);
        }

        #[test]
        fn disappeared_cache_entry_is_treated_as_cache_miss() {
            let temp = tempfile::tempdir().unwrap();
            let cache = NativeRuntimeCache::new(temp.path().join("cache"));
            let runtime_id = "meshllm-native-runtime-test-cpu";
            let release_version = "0.68.0";
            let manifest = NativeRuntimeReleaseManifest {
                mesh_version: release_version.to_string(),
                skippy_abi: "0.1.25".to_string(),
                artifacts: Vec::new(),
            };
            let artifact = NativeRuntimeArtifact {
                id: runtime_id.to_string(),
                mesh_version: Some(release_version.to_string()),
                skippy_abi: "0.1.25".to_string(),
                platform: NativeRuntimePlatform {
                    os: std::env::consts::OS.to_string(),
                    arch: std::env::consts::ARCH.to_string(),
                    target: None,
                },
                backend: NativeRuntimeBackend::cpu(),
                rank: 0,
                libraries: vec![test_library_rel_path().to_string_lossy().to_string()],
                url: None,
                sha256: None,
                signature: None,
            };

            let plan = load_plan_from_candidate(&cache, &manifest, artifact).unwrap();

            assert!(plan.is_none());
        }

        #[test]
        fn cache_hit_skips_install_and_loads_cached_runtime_once() {
            let temp = tempfile::tempdir().unwrap();
            let cache = NativeRuntimeCache::new(temp.path().join("cache"));
            let runtime_id = "meshllm-native-runtime-test-cpu";
            let release_version = "0.68.0";
            let runtime_dir = cache.runtime_dir(release_version, runtime_id);
            write_runtime(&runtime_dir, release_version, runtime_id);

            let install_calls = Arc::new(Mutex::new(0_usize));
            let load_calls = Arc::new(Mutex::new(Vec::<Vec<PathBuf>>::new()));

            let runtime = try_load_installed_native_runtime_with(
                || false,
                || Ok(cache.clone()),
                HostRuntimeProfile::current_without_gpu_probe,
                test_install_options,
                {
                    let install_calls = Arc::clone(&install_calls);
                    move |_| {
                        *install_calls.lock().unwrap() += 1;
                        anyhow::bail!("install should not run on cache hit")
                    }
                },
                {
                    let load_calls = Arc::clone(&load_calls);
                    move |libraries| {
                        load_calls.lock().unwrap().push(libraries.to_vec());
                        Ok(())
                    }
                },
            )
            .unwrap()
            .expect("expected cached runtime to load");

            assert_eq!(*install_calls.lock().unwrap(), 0);
            assert_eq!(runtime.native_runtime_id, runtime_id);
            assert_eq!(
                runtime.libraries,
                vec![runtime_dir.join(test_library_rel_path())]
            );
            assert_eq!(load_calls.lock().unwrap().as_slice(), &[runtime.libraries]);
        }

        #[test]
        fn cache_miss_installs_once_and_loads_post_install_runtime() {
            let temp = tempfile::tempdir().unwrap();
            let cache = NativeRuntimeCache::new(temp.path().join("cache"));
            let bundle_dir = temp.path().join("bundle");
            let runtime_id = "meshllm-native-runtime-test-cpu";
            let manifest_mesh_version = "0.69.0";
            write_runtime(&bundle_dir, manifest_mesh_version, runtime_id);

            let install_calls = Arc::new(Mutex::new(Vec::<NativeRuntimeInstallOptions>::new()));
            let load_calls = Arc::new(Mutex::new(Vec::<Vec<PathBuf>>::new()));

            let runtime = try_load_installed_native_runtime_with(
                || false,
                || Ok(cache.clone()),
                HostRuntimeProfile::current_without_gpu_probe,
                test_install_options,
                {
                    let install_calls = Arc::clone(&install_calls);
                    let bundle_dir = bundle_dir.clone();
                    let cache = cache.clone();
                    move |mut options| {
                        install_calls.lock().unwrap().push(options.clone());
                        let source = options.bundle_dirs.pop().unwrap_or(bundle_dir.clone());
                        let runtime = cache.install_from_dir(&source)?;
                        Ok(NativeRuntimeInstallOutcome {
                            status: crate::system::native_runtime_install::NativeRuntimeInstallStatus::Installed,
                            runtime,
                            resolution: mesh_llm_native_runtime::NativeRuntimeResolution {
                                source: mesh_llm_native_runtime::NativeRuntimeSource::Bundle { path: source },
                                selected: NativeRuntimeManifest::read_from_dir(&bundle_dir)?.runtime,
                                evaluated: Vec::new(),
                            },
                        })
                    }
                },
                {
                    let load_calls = Arc::clone(&load_calls);
                    move |libraries| {
                        load_calls.lock().unwrap().push(libraries.to_vec());
                        Ok(())
                    }
                },
            )
            .unwrap()
            .expect("expected installed runtime to load");

            let recorded_options = install_calls.lock().unwrap();
            assert_eq!(recorded_options.len(), 1);
            assert_eq!(recorded_options[0].mesh_version, "0.68.0");
            assert_eq!(recorded_options[0].cache_dir.as_deref(), Some(cache.root()));
            assert_eq!(runtime.native_runtime_id, runtime_id);
            assert_eq!(
                runtime.libraries,
                vec![
                    cache
                        .runtime_dir(manifest_mesh_version, runtime_id)
                        .join(test_library_rel_path())
                ]
            );
            assert_eq!(load_calls.lock().unwrap().as_slice(), &[runtime.libraries]);
        }

        #[test]
        fn cache_miss_install_failure_returns_none_without_retry() {
            let temp = tempfile::tempdir().unwrap();
            let cache = NativeRuntimeCache::new(temp.path().join("cache"));
            let install_calls = Arc::new(Mutex::new(Vec::<NativeRuntimeInstallOptions>::new()));
            let load_calls = Arc::new(Mutex::new(0_usize));

            let runtime = try_load_installed_native_runtime_with(
                || false,
                || Ok(cache.clone()),
                HostRuntimeProfile::current_without_gpu_probe,
                test_install_options,
                {
                    let install_calls = Arc::clone(&install_calls);
                    move |options| {
                        install_calls.lock().unwrap().push(options);
                        anyhow::bail!(
                            "no compatible native runtime found for Skippy ABI 0.1.25 on test/test"
                        )
                    }
                },
                {
                    let load_calls = Arc::clone(&load_calls);
                    move |_| {
                        *load_calls.lock().unwrap() += 1;
                        Ok(())
                    }
                },
            )
            .unwrap();

            assert!(runtime.is_none());
            assert_eq!(install_calls.lock().unwrap().len(), 1);
            assert_eq!(*load_calls.lock().unwrap(), 0);
        }
    }
}

#[cfg(feature = "dynamic-native-runtime")]
pub(crate) use dynamic::*;

#[cfg(not(feature = "dynamic-native-runtime"))]
pub(crate) fn try_load_installed_native_runtime() -> anyhow::Result<Option<()>> {
    Ok(None)
}
