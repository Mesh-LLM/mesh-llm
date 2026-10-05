//! Model-domain operations used by the shared command handlers.
use anyhow::Result;
pub(super) use skippy_model_hf::huggingface_hub_cache_dir;
pub(super) use skippy_model_hf::remote_catalog;
pub(super) use skippy_model_hf::store::delete::DeleteResult;
pub(super) use skippy_model_hf::store::installed::scan_installed_artifacts_in;
pub(super) use skippy_model_hf::store::local::{
    huggingface_identity_for_path, layered_package_layer_count_for_path,
    layered_package_total_bytes_for_path,
};
pub(super) use skippy_model_hf::store::usage::{
    ModelCleanupPlan, ModelCleanupResult, ModelUsageRecord,
};
use std::{
    path::{Path, PathBuf},
    time::Duration,
};

#[derive(Clone, Debug)]
pub(super) struct ResolvedModel {
    pub path: PathBuf,
    pub paths: Vec<PathBuf>,
    pub derived_stage_paths: Vec<PathBuf>,
    pub display_name: String,
    pub is_exact_path: bool,
    pub matched_records: Vec<ModelUsageRecord>,
}
pub(super) fn model_usage_cache_dir() -> PathBuf {
    skippy_model_hf::store::usage::model_usage_cache_dir_in(&super::output::cache_root())
}
pub(super) fn load_model_usage_record_for_path(path: &Path) -> Option<ModelUsageRecord> {
    skippy_model_hf::store::usage::load_model_usage_record_for_path_in(
        path,
        &super::output::cache_root(),
    )
}
pub(super) fn plan_model_cleanup(age: Option<Duration>) -> Result<ModelCleanupPlan> {
    skippy_model_hf::store::usage::plan_model_cleanup_in(&super::output::cache_root(), age)
}
pub(super) fn execute_model_cleanup(age: Option<Duration>) -> Result<ModelCleanupResult> {
    skippy_model_hf::store::usage::execute_model_cleanup_in(&super::output::cache_root(), age)
}
pub(super) fn format_size_bytes(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1}GB", bytes as f64 / 1e9)
    } else {
        format!("{:.0}MB", bytes as f64 / 1e6)
    }
}
pub(super) fn total_model_bytes(path: &Path) -> u64 {
    let Some(file) = path.file_name().and_then(|name| name.to_str()) else {
        return 0;
    };
    let Some(shard) = skippy_model_ref::split_gguf_shard_info(file) else {
        return path.metadata().map(|meta| meta.len()).unwrap_or(0);
    };
    if shard.part != "00001" {
        return path.metadata().map(|meta| meta.len()).unwrap_or(0);
    }
    (1..=shard.total.parse::<u32>().unwrap_or(0))
        .map(|index| format!("{}-{index:05}-of-{}.gguf", shard.prefix, shard.total))
        .map(|file| {
            path.with_file_name(file)
                .metadata()
                .map(|meta| meta.len())
                .unwrap_or(0)
        })
        .sum()
}
pub(super) mod delete {
    use super::*;
    pub(in crate::models) async fn resolve_model_identifier(model: &str) -> Result<Vec<PathBuf>> {
        skippy_model_hf::store::delete::resolve_model_identifier_with_catalog(
            model,
            &skippy_model_hf::store::delete::CuratedDeleteCatalog,
        )
        .await
    }
    pub(in crate::models) async fn delete_model_by_identifier(model: &str) -> Result<DeleteResult> {
        skippy_model_hf::store::delete::delete_model_by_identifier_with_catalog_in(
            model,
            &skippy_model_hf::store::delete::CuratedDeleteCatalog,
            &super::super::output::cache_root(),
        )
        .await
    }
}
