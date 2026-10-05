//! Mesh presentation adapter for Skippy-owned model search.

use anyhow::Result;
use serde_json::Value;
use skippy_commands::models::{ModelCommandContext, lifecycle};
use skippy_model_hf::remote_catalog;

pub use lifecycle::{SearchArtifactFilter, SearchHit, SearchProgress, SearchSort};

fn model_context() -> ModelCommandContext {
    ModelCommandContext {
        program: "mesh-llm",
        cache_root: skippy_model_hf::application_cache_dir(),
        fit_budget_bytes: mesh_llm_system::capacity::local_fit_budget_bytes(
            &crate::system::hardware::survey(),
        ),
        terminal_progress: false,
        byte_progress: None,
        console_out: || Box::new(mesh_llm_events::console_out()),
        console_err: || Box::new(mesh_llm_events::console_err()),
        machine_out: || Box::new(mesh_llm_events::machine_out()),
    }
}

pub fn search_catalog_models(query: &str) -> Result<Vec<remote_catalog::RemoteCatalogModel>> {
    lifecycle::search_catalog_models(query)
}

pub fn search_catalog_json_payload(
    query: &str,
    filter: SearchArtifactFilter,
    sort: SearchSort,
    results: &[remote_catalog::RemoteCatalogModel],
    limit: usize,
) -> Value {
    lifecycle::search_catalog_json_payload(query, filter, sort, results, limit, model_context())
}

pub fn search_huggingface_json_payload(
    query: &str,
    filter: SearchArtifactFilter,
    sort: SearchSort,
    results: &[SearchHit],
) -> Value {
    lifecycle::search_huggingface_json_payload(query, filter, sort, results, model_context())
}

pub async fn search_huggingface<F>(
    query: &str,
    limit: usize,
    filter: SearchArtifactFilter,
    sort: SearchSort,
    progress: F,
) -> Result<Vec<SearchHit>>
where
    F: FnMut(SearchProgress),
{
    lifecycle::search_huggingface(query, limit, filter, sort, progress, model_context()).await
}
