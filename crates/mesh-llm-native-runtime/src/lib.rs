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
pub use flavor::{NativeRuntimeFlavor, NativeRuntimeFlavorParseError};
pub use host::{HostGpuProfile, HostRuntimeProfile};
pub use load_plan::NativeRuntimeLoadPlan;
pub use manifest::{
    NATIVE_RUNTIME_MANIFEST_FILE, NativeRuntimeArtifact, NativeRuntimeManifest,
    NativeRuntimeReleaseManifest, NativeRuntimeRequirement,
};
pub use resolver::{
    CandidateEvaluation, CandidateRejection, NativeRuntimeResolution, NativeRuntimeResolver,
    NativeRuntimeSource, RuntimeSelection, select_native_runtime,
};
