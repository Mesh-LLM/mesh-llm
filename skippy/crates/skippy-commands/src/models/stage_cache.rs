//! Package certification and derived stage cache operations.
use anyhow::{Context, Result};
use skippy_model_hf::{huggingface_hub_cache_dir, remote_catalog};
use std::path::PathBuf;

pub(super) use acquisition::is_layer_package_ref;
pub(super) use certification::{CertificationGateStatus, SkippyCertificationRequest};
use skippy_api::{
    materialized_cache,
    package::{acquisition, certification},
};
fn acquisition() -> acquisition::remote::PackageAcquisition {
    acquisition::remote::PackageAcquisition::new(
        huggingface_hub_cache_dir(),
        skippy_model_hf::build_hf_sync_api,
    )
}
pub(super) fn materialized_stage_cache_dir() -> PathBuf {
    super::output::cache_root().join("skippy-stages")
}
pub(super) fn download_package_v2_to_local(package: &str) -> Result<PathBuf> {
    acquisition().download_package_v2_to_local(package)
}
pub(super) fn prune_unpinned_materialized_stages() -> Result<usize> {
    materialized_cache::prune_unpinned_materialized_stages(&materialized_stage_cache_dir())
}
pub(super) fn materialized_stages_for_sources(paths: &[PathBuf]) -> Result<Vec<PathBuf>> {
    materialized_cache::materialized_stages_for_sources(&materialized_stage_cache_dir(), paths)
}
pub(super) fn remove_materialized_stages_for_sources(paths: &[PathBuf]) -> Result<usize> {
    materialized_cache::remove_materialized_stages_for_sources(
        &materialized_stage_cache_dir(),
        paths,
    )
}
pub(super) async fn certify_layer_package(
    request: SkippyCertificationRequest,
) -> Result<certification::SkippyCertificationReport> {
    let package_ref = if let Ok(parsed) = acquisition::StagePackageRef::parse(&request.model_ref) {
        parsed
            .as_package_ref()
            .context("direct GGUF inputs are not layer-package certification targets")?
    } else {
        remote_catalog::find_layer_package(&request.model_ref)
            .with_context(|| format!("no layer package found for {:?}", request.model_ref))?
    };
    certification::certify_layer_package(request, package_ref, acquisition(), None).await
}
