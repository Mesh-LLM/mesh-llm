//! Mesh digest-cache policy for Skippy package verification.
use crate::{SkippyPackageIdentity, hash_cache};
pub use skippy_api::package::metadata::is_package_v2_ref;
use std::path::Path;

pub fn identity_from_package_v2(package_dir: &Path) -> anyhow::Result<SkippyPackageIdentity> {
    skippy_api::package::identity_from_package_v2(package_dir, hash_cache::open_default().as_ref())
}

pub fn identity_from_package_v2_metadata(
    package_ref: &str,
    local_ref: &str,
) -> anyhow::Result<SkippyPackageIdentity> {
    skippy_api::package::metadata::identity_from_package_v2_metadata(
        package_ref,
        local_ref,
        hash_cache::open_default().as_ref(),
    )
}

pub use skippy_api::package::inspection::{StagePackageInfo, StagePackageLayerInfo};

pub fn inspect_local_stage_package(
    package_ref: &str,
    local_ref: &str,
) -> anyhow::Result<StagePackageInfo> {
    skippy_api::package::inspection::inspect_local_stage_package(
        package_ref,
        local_ref,
        hash_cache::open_default().as_ref(),
    )
}

pub mod acquisition;
