#[cfg(feature = "dynamic-native-runtime")]
mod dynamic {
    use anyhow::{Context, Result};
    use mesh_llm_native_runtime::{
        HostGpuProfile, HostRuntimeProfile, NativeRuntimeCache, NativeRuntimeFlavor,
        NativeRuntimeReleaseManifest, RuntimeSelection, select_native_runtime,
    };
    use std::collections::BTreeSet;
    use std::path::PathBuf;

    #[derive(Clone, Debug)]
    pub(crate) struct LoadedNativeRuntime {
        pub(crate) native_runtime_id: String,
        pub(crate) libraries: Vec<PathBuf>,
    }

    pub(crate) fn try_load_installed_native_runtime() -> Result<Option<LoadedNativeRuntime>> {
        if skippy_runtime::native_runtime_loaded() {
            return Ok(None);
        }
        let cache = default_native_runtime_cache()?;
        let installed = cache.installed()?;
        let profile = host_runtime_profile();
        let manifest = NativeRuntimeReleaseManifest {
            mesh_version: crate::VERSION.to_string(),
            artifacts: installed
                .iter()
                .map(|runtime| runtime.manifest.artifact.clone())
                .collect(),
        };
        let Some(candidate) = select_native_runtime(
            &manifest,
            &profile,
            crate::VERSION,
            &RuntimeSelection::Recommended,
        ) else {
            return Ok(None);
        };
        let installed = cache
            .find_installed(crate::VERSION, &candidate.artifact.native_runtime_id)?
            .with_context(|| {
                format!(
                    "selected native runtime {} disappeared from the cache",
                    candidate.artifact.native_runtime_id
                )
            })?;
        let plan = installed.load_plan()?;
        unsafe {
            skippy_runtime::load_native_runtime_libraries(&plan.libraries).with_context(|| {
                format!(
                    "load native runtime {} from {}",
                    plan.native_runtime_id,
                    plan.root.display()
                )
            })?;
        }
        Ok(Some(LoadedNativeRuntime {
            native_runtime_id: plan.native_runtime_id,
            libraries: plan.libraries,
        }))
    }

    fn default_native_runtime_cache() -> Result<NativeRuntimeCache> {
        let root = dirs::cache_dir()
            .or_else(|| dirs::home_dir().map(|home| home.join(".cache")))
            .context("cannot determine native runtime cache directory")?
            .join("mesh-llm")
            .join("native-runtimes");
        Ok(NativeRuntimeCache::new(root))
    }

    fn host_runtime_profile() -> HostRuntimeProfile {
        let survey = crate::system::hardware::survey();
        let gpus = survey
            .gpus
            .iter()
            .map(|gpu| HostGpuProfile {
                display_name: gpu.display_name.clone(),
                backend_device: gpu.backend_device.clone(),
                stable_id: gpu.stable_id.clone(),
                vram_bytes: Some(gpu.vram_bytes),
                unified_memory: gpu.unified_memory,
            })
            .collect::<Vec<_>>();
        HostRuntimeProfile {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            target_triple: option_env!("TARGET").map(str::to_string),
            available_flavors: detected_native_runtime_flavors(&survey.gpus),
            gpus,
        }
    }

    fn detected_native_runtime_flavors(
        gpus: &[crate::system::hardware::GpuFacts],
    ) -> BTreeSet<NativeRuntimeFlavor> {
        let mut flavors = BTreeSet::from([NativeRuntimeFlavor::Cpu]);
        if cfg!(target_os = "macos") {
            flavors.insert(NativeRuntimeFlavor::Metal);
        }
        for gpu in gpus {
            let label = format!(
                "{} {}",
                gpu.display_name,
                gpu.backend_device.as_deref().unwrap_or_default()
            )
            .to_ascii_lowercase();
            if label.contains("cuda") || label.contains("nvidia") {
                flavors.insert(NativeRuntimeFlavor::Cuda);
            }
            if label.contains("blackwell")
                || label.contains("gb200")
                || label.contains("b200")
                || label.contains("rtx 50")
            {
                flavors.insert(NativeRuntimeFlavor::CudaBlackwell);
            }
            if label.contains("rocm") || label.contains("hip") || label.contains("amd") {
                flavors.insert(NativeRuntimeFlavor::Rocm);
            }
            if label.contains("vulkan") {
                flavors.insert(NativeRuntimeFlavor::Vulkan);
            }
        }
        flavors
    }
}

#[cfg(feature = "dynamic-native-runtime")]
pub(crate) use dynamic::*;

#[cfg(not(feature = "dynamic-native-runtime"))]
pub(crate) fn try_load_installed_native_runtime() -> anyhow::Result<Option<()>> {
    Ok(None)
}
