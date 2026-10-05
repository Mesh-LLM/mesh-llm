use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::Result;
use skippy_model_hf::store::usage;
pub use skippy_model_hf::store::usage::{ModelCleanupPlan, ModelCleanupResult, ModelUsageRecord};

fn cache_root() -> PathBuf {
    crate::models::local::mesh_llm_cache_dir()
}

pub fn model_usage_cache_dir() -> PathBuf {
    usage::model_usage_cache_dir_in(&cache_root())
}

pub fn load_model_usage_record_for_path(path: &Path) -> Option<ModelUsageRecord> {
    usage::load_model_usage_record_for_path_in(path, &cache_root())
}

pub fn track_model_usage(
    path: &Path,
    display_name: Option<&str>,
    model_ref: Option<&str>,
    source: Option<&str>,
) -> Result<()> {
    usage::track_model_usage_in(&cache_root(), path, display_name, model_ref, source)
}

pub fn track_managed_model_usage(
    primary_path: &Path,
    managed_paths: &[PathBuf],
    display_name: &str,
    model_ref: Option<&str>,
    source: &str,
) -> Result<()> {
    usage::track_managed_model_usage_in(
        &cache_root(),
        primary_path,
        managed_paths,
        display_name,
        model_ref,
        source,
    )
}

pub fn plan_model_cleanup(unused_since: Option<Duration>) -> Result<ModelCleanupPlan> {
    usage::plan_model_cleanup_in(&cache_root(), unused_since)
}

pub fn execute_model_cleanup(unused_since: Option<Duration>) -> Result<ModelCleanupResult> {
    usage::execute_model_cleanup_in(&cache_root(), unused_since)
}
