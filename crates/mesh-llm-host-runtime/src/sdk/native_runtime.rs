//! Native runtime resolution and installation APIs for embedded MeshLLM clients.

pub use crate::system::native_runtime_install::{
    CURRENT_MESH_VERSION, NATIVE_RUNTIME_MANIFEST_URL_ENV, NativeRuntimeDownloadProgress,
    NativeRuntimeDownloadProgressCallback, NativeRuntimeInstallOptions,
    NativeRuntimeInstallOutcome, NativeRuntimeInstallStatus, NativeRuntimeManifestOptions,
    NativeRuntimeVerificationPolicy, default_native_runtime_cache, default_release_manifest_url,
    host_runtime_profile, install_native_runtime, load_release_manifest, native_runtime_cache,
};
pub use mesh_llm_native_runtime::{
    CachePrunePlan, CandidateEvaluation, CandidateRejection, HostGpuProfile, HostRuntimeProfile,
    InstalledNativeRuntime, NATIVE_RUNTIME_MANIFEST_FILE, NativeRuntimeArtifact,
    NativeRuntimeCache, NativeRuntimeCacheRoot, NativeRuntimeFlavor, NativeRuntimeFlavorParseError,
    NativeRuntimeLoadPlan, NativeRuntimeManifest, NativeRuntimePruneMode,
    NativeRuntimeReleaseManifest, NativeRuntimeResolution, NativeRuntimeResolver,
    NativeRuntimeSource, RuntimeSelection, native_runtime_cache_root, select_native_runtime,
};
