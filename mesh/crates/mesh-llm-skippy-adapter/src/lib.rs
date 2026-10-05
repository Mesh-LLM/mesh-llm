//! Translate Mesh configuration and policy into Skippy preparation inputs.
//! Inference, model lifecycle and serving remain owned by Skippy.
mod checkpoint;
pub mod config;
pub mod hash_cache;
mod load_options;
pub mod package;
mod stage;
pub use load_options::{SkippyDeviceDescriptor, SkippyModelLoadOptions, SkippyTelemetryOptions};
pub use skippy_api::family_policy;
pub use skippy_api::family_policy::family_policy_for_model_path;
pub use skippy_api::kv_cache::{KvCachePolicy, KvCacheType};
pub use skippy_api::package::SkippyPackageIdentity;
pub use stage::single_stage_config;

/// Preserve Mesh's existing application cache location when calling Skippy.
pub fn mesh_cache_dir() -> std::path::PathBuf {
    skippy_model_hf::application_cache_dir()
}

pub fn synthetic_direct_gguf_package(
    model_id: &str,
    model_path: &std::path::Path,
) -> anyhow::Result<SkippyPackageIdentity> {
    skippy_api::source::synthetic_direct_gguf_package(
        model_id,
        model_path,
        hash_cache::open_default().as_ref(),
    )
}

pub use skippy_api::serving::readiness;
