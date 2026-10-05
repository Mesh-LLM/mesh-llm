//! Local native-runtime selection shared by standalone and embedded startup.

use anyhow::Result;
use skippy_native_runtime::{
    HostRuntimeProfile, InstalledNativeRuntime, NativeRuntimeLoadPlan, RuntimeSelection,
    has_startup_compatibility_metadata, select_native_runtime_from_artifacts,
};

/// Select a verified local bundle/cache entry for an explicit runtime contract.
/// Input ordering retains discovery precedence. This does not download or load code.
pub fn select_local_native_runtime_plan(
    runtimes: &[InstalledNativeRuntime],
    profile: &HostRuntimeProfile,
    release: &str,
    required_abi: Option<&str>,
    selection: &RuntimeSelection,
) -> Result<Option<NativeRuntimeLoadPlan>> {
    let eligible = runtimes
        .iter()
        .filter(|runtime| has_startup_compatibility_metadata(&runtime.manifest.runtime, profile))
        .collect::<Vec<_>>();
    let artifacts = eligible
        .iter()
        .map(|runtime| runtime.manifest.runtime.clone())
        .collect::<Vec<_>>();
    let Some(candidate) =
        select_native_runtime_from_artifacts(&artifacts, profile, release, required_abi, selection)
    else {
        return Ok(None);
    };
    let selected_release = candidate.artifact.release_version_or(release);
    let Some(runtime) = eligible.into_iter().find(|runtime| {
        runtime.release_version == selected_release
            && runtime.native_runtime_id == candidate.artifact.native_runtime_id()
            && runtime.manifest.runtime.skippy_abi == candidate.artifact.skippy_abi
    }) else {
        return Ok(None);
    };
    runtime.load_plan().map(Some)
}
