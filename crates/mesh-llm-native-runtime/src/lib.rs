//! Shared native runtime manifest, resolution, and cache policy.

mod cache;
mod flavor;
mod host;
mod load_plan;
mod manifest;
mod resolver;

pub use cache::{
    CachePrunePlan, InstalledNativeRuntime, NativeRuntimeCache, NativeRuntimeCacheRoot,
    NativeRuntimePruneMode, native_runtime_cache_root,
};
pub use flavor::{
    CudaRuntimeRequirements, NativeRuntimeBackend, NativeRuntimeBackendKind, NativeRuntimeFlavor,
    NativeRuntimeFlavorParseError, RocmRuntimeRequirements, VulkanRuntimeRequirements,
};
pub use host::{
    HostCudaProfile, HostGpuProfile, HostRocmProfile, HostRuntimeProfile, HostVulkanProfile,
};
pub use load_plan::NativeRuntimeLoadPlan;
pub use manifest::{
    NATIVE_RUNTIME_MANIFEST_FILE, NativeRuntimeArtifact, NativeRuntimeManifest,
    NativeRuntimePlatform, NativeRuntimeReleaseManifest,
};
pub use resolver::{
    CandidateEvaluation, CandidateRejection, NativeRuntimeResolution, NativeRuntimeResolver,
    NativeRuntimeSource, RuntimeSelection, select_native_runtime,
    select_native_runtime_for_skippy_abi,
};
