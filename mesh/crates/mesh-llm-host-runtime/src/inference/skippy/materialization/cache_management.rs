//! Mesh cache defaults for Skippy-owned materialized stage maintenance.
use std::path::PathBuf;

use anyhow::Result;
use skippy_api::materialized_cache;

pub fn prune_unpinned_materialized_stages() -> Result<usize> {
    materialized_cache::prune_unpinned_materialized_stages(&super::materialized_stage_cache_dir())
}

pub fn remove_materialized_stages_for_sources(sources: &[PathBuf]) -> Result<usize> {
    materialized_cache::remove_materialized_stages_for_sources(
        &super::materialized_stage_cache_dir(),
        sources,
    )
}

pub fn materialized_stages_for_sources(sources: &[PathBuf]) -> Result<Vec<PathBuf>> {
    materialized_cache::materialized_stages_for_sources(
        &super::materialized_stage_cache_dir(),
        sources,
    )
}
