//! Detected GPUs that the loaded native runtime leaves unused.
//!
//! Startup picks the best runtime that fits the host. When no runtime for a
//! detected GPU fits (a CUDA 13 runtime on a CUDA 12 driver, say), it falls
//! back to the CPU runtime and serves noticeably slower. The runtime is loaded
//! before the output surface exists, so the warnings are kept here and shown
//! once the node starts, like the external-provider fallback above.

use crate::system::native_runtime_install;
use mesh_llm_events::{OutputEvent, emit_event};
use skippy_native_runtime::{
    HostRuntimeProfile, InstalledNativeRuntime, NativeRuntimeArtifact, RuntimeSelection,
    describe_unused_gpus, has_startup_compatibility_metadata,
};
use std::sync::Mutex;

static WARNINGS: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Records a warning when the runtime `loaded_id` of release `loaded_release`
/// leaves a detected GPU unused. Uses the host profile the startup load already
/// detected (cached per process); any failure only means no warning, never a
/// failed startup.
pub(super) fn record(
    selection: &RuntimeSelection,
    mesh_version: &str,
    loaded_release: &str,
    loaded_id: &str,
) {
    record_with(
        selection,
        loaded_release,
        loaded_id,
        native_runtime_install::host_runtime_profile,
        || local_runtimes(mesh_version),
    );
}

fn record_with(
    selection: &RuntimeSelection,
    loaded_release: &str,
    loaded_id: &str,
    profile: impl FnOnce() -> HostRuntimeProfile,
    runtimes: impl FnOnce() -> anyhow::Result<Vec<InstalledNativeRuntime>>,
) {
    // An explicit backend or runtime is the user's choice: nothing to explain,
    // so don't spend time detecting anything.
    if *selection != RuntimeSelection::Recommended {
        return;
    }
    let runtimes = match runtimes() {
        Ok(runtimes) => runtimes,
        Err(error) => {
            tracing::debug!(%error, "Skipping the unused GPU check");
            return;
        }
    };
    let warnings = release_warnings(runtimes, &profile(), selection, loaded_release, loaded_id);
    *WARNINGS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = warnings;
}

/// Shows the recorded warnings once; called when the node starts.
pub(crate) fn emit_recorded() {
    let warnings = std::mem::take(
        &mut *WARNINGS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
    );
    for message in warnings {
        let _ = emit_event(OutputEvent::Warning {
            message,
            context: None,
        });
    }
}

/// The bundled and installed runtimes startup chose from, including those it
/// skips for missing compatibility metadata, so the warning can name them.
fn local_runtimes(mesh_version: &str) -> anyhow::Result<Vec<InstalledNativeRuntime>> {
    let bundle_dirs =
        native_runtime_install::discover_native_runtime_bundle_dirs_for_release(&[], mesh_version)?;
    let cache = native_runtime_install::default_native_runtime_cache()?;
    native_runtime_install::discover_local_native_runtimes(&bundle_dirs, &cache)
}

/// Judges only the runtimes of the loaded runtime's release. The cache keeps
/// older releases, whose runtimes share ids with the current ones; one of them
/// fitting says nothing about the release startup loaded.
fn release_warnings(
    runtimes: Vec<InstalledNativeRuntime>,
    profile: &HostRuntimeProfile,
    selection: &RuntimeSelection,
    release: &str,
    loaded_id: &str,
) -> Vec<String> {
    let artifacts = runtimes
        .into_iter()
        .filter(|runtime| runtime.release_version == release)
        .map(|runtime| runtime.manifest.runtime)
        .collect::<Vec<_>>();
    warnings(&artifacts, profile, selection, release, loaded_id)
}

fn warnings(
    artifacts: &[NativeRuntimeArtifact],
    profile: &HostRuntimeProfile,
    selection: &RuntimeSelection,
    release: &str,
    loaded_id: &str,
) -> Vec<String> {
    let Some(loaded) = artifacts.iter().find(|artifact| {
        artifact.id == loaded_id && has_startup_compatibility_metadata(artifact, profile)
    }) else {
        return Vec::new();
    };
    describe_unused_gpus(
        artifacts,
        profile,
        release,
        Some(&loaded.skippy_abi),
        selection,
        loaded,
    )
    .map(|description| {
        format!(
            "{description}; serving on the {} runtime instead. `mesh-llm runtime list --available` lists every runtime and why it doesn't fit.",
            loaded.backend.kind
        )
    })
    .into_iter()
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use skippy_native_runtime::{
        CudaRuntimeRequirements, HostCudaProfile, NativeRuntimeBackend, NativeRuntimeBackendKind,
        NativeRuntimeManifest, NativeRuntimePlatform,
    };
    use std::collections::BTreeSet;

    fn artifact(id: &str, backend: NativeRuntimeBackend) -> NativeRuntimeArtifact {
        NativeRuntimeArtifact {
            id: id.to_string(),
            release_version: Some("0.76.1".to_string()),
            skippy_abi: "0.1.71".to_string(),
            platform: NativeRuntimePlatform {
                os: "linux".to_string(),
                arch: "aarch64".to_string(),
                target: None,
                min_glibc: Some("2.17".to_string()),
            },
            backend,
            rank: 0,
            libraries: vec!["lib/libllama.so".to_string()],
            files: Default::default(),
            tools: Default::default(),
            url: None,
            sha256: None,
            signature: None,
        }
    }

    fn cuda(toolkit_major: u32) -> NativeRuntimeBackend {
        NativeRuntimeBackend {
            kind: NativeRuntimeBackendKind::Cuda,
            cuda: Some(CudaRuntimeRequirements {
                toolkit_major,
                min_driver: None,
                gpu_arches: Vec::new(),
            }),
            rocm: None,
            vulkan: None,
        }
    }

    fn cuda13() -> NativeRuntimeBackend {
        cuda(13)
    }

    /// A cached runtime `id` of `release`.
    fn installed(release: &str, id: &str, backend: NativeRuntimeBackend) -> InstalledNativeRuntime {
        let mut runtime = artifact(id, backend);
        runtime.release_version = Some(release.to_string());
        InstalledNativeRuntime {
            release_version: release.to_string(),
            native_runtime_id: id.to_string(),
            flavor: runtime.backend.kind.to_string(),
            path: format!("/cache/{release}/{id}").into(),
            manifest: NativeRuntimeManifest { runtime },
        }
    }

    /// A Jetson on JetPack 6: its driver supports CUDA 12.
    fn cuda12_host() -> HostRuntimeProfile {
        HostRuntimeProfile {
            os: "linux".to_string(),
            arch: "aarch64".to_string(),
            target_triple: None,
            glibc_version: None,
            available_flavors: BTreeSet::from([
                NativeRuntimeBackendKind::Cpu,
                NativeRuntimeBackendKind::Cuda,
            ]),
            gpus: Vec::new(),
            cuda: Some(HostCudaProfile {
                driver_max_major: Some(12),
                ..HostCudaProfile::default()
            }),
            rocm: None,
            vulkan: None,
        }
    }

    #[test]
    fn a_cpu_runtime_loaded_beside_a_rejected_cuda_runtime_is_reported() {
        let artifacts = [
            artifact("cpu", NativeRuntimeBackend::cpu()),
            artifact("cuda13", cuda13()),
        ];

        let warnings = warnings(
            &artifacts,
            &cuda12_host(),
            &RuntimeSelection::Recommended,
            "0.76.1",
            "cpu",
        );

        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(
            warnings[0].starts_with("cuda GPU detected, but its runtimes don't fit: cuda13 ("),
            "{}",
            warnings[0]
        );
        assert!(
            warnings[0].contains("CUDA driver too old"),
            "{}",
            warnings[0]
        );
        assert!(
            warnings[0].contains("serving on the cpu runtime instead"),
            "{}",
            warnings[0]
        );
    }

    #[test]
    fn a_cuda_runtime_without_startup_metadata_is_named() {
        let mut legacy = artifact("cuda13-legacy", cuda13());
        legacy.platform.min_glibc = None;
        let artifacts = [artifact("cpu", NativeRuntimeBackend::cpu()), legacy];

        let warnings = warnings(
            &artifacts,
            &cuda12_host(),
            &RuntimeSelection::Recommended,
            "0.76.1",
            "cpu",
        );

        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(
            warnings[0].contains("cuda13-legacy (no minimum glibc in its manifest"),
            "{}",
            warnings[0]
        );
    }

    #[test]
    fn an_unknown_loaded_runtime_reports_nothing() {
        let artifacts = [artifact("cuda13", cuda13())];

        assert!(
            warnings(
                &artifacts,
                &cuda12_host(),
                &RuntimeSelection::Recommended,
                "0.76.1",
                "cpu"
            )
            .is_empty()
        );
    }

    #[test]
    fn a_cached_older_release_does_not_hide_the_unused_gpu() {
        // The older release is enumerated first, and its CUDA 12 runtime would
        // fit; startup loaded the current release's CPU runtime.
        let runtimes = vec![
            installed("0.76.1", "cpu", NativeRuntimeBackend::cpu()),
            installed("0.76.1", "cuda", cuda(12)),
            installed("0.78.1", "cpu", NativeRuntimeBackend::cpu()),
            installed("0.78.1", "cuda", cuda13()),
        ];

        let warnings = release_warnings(
            runtimes,
            &cuda12_host(),
            &RuntimeSelection::Recommended,
            "0.78.1",
            "cpu",
        );

        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(
            warnings[0].starts_with("cuda GPU detected, but its runtimes don't fit: cuda ("),
            "{}",
            warnings[0]
        );
        assert!(
            warnings[0].contains("CUDA driver too old"),
            "{}",
            warnings[0]
        );
    }

    #[test]
    fn an_explicit_selection_detects_nothing() {
        record_with(
            &RuntimeSelection::Id("cpu".to_string()),
            "0.76.1",
            "cpu",
            || panic!("GPU detection ran for an explicit selection"),
            || panic!("runtime discovery ran for an explicit selection"),
        );
    }

    #[test]
    fn a_vulkan_icd_beside_an_unused_cuda_gpu_is_not_a_second_gpu() {
        let mut host = cuda12_host();
        host.available_flavors
            .insert(NativeRuntimeBackendKind::Vulkan);
        let artifacts = [artifact("cpu", NativeRuntimeBackend::cpu())];

        let warnings = warnings(
            &artifacts,
            &host,
            &RuntimeSelection::Recommended,
            "0.76.1",
            "cpu",
        );

        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(
            warnings[0].starts_with(
                "cuda GPU detected, but no cuda runtime is installed; serving on the cpu runtime instead."
            ),
            "{}",
            warnings[0]
        );
    }

    #[test]
    fn recorded_warnings_are_shown_once() {
        *WARNINGS.lock().unwrap() = vec!["first".to_string()];

        emit_recorded();

        assert!(WARNINGS.lock().unwrap().is_empty());
    }
}
