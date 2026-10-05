//! Embedding API for model acquisition used by product command adapters.

use std::path::PathBuf;

use anyhow::{Context, Result};
use serde_json::Value;
use skippy_model_hf::remote_catalog;

pub use super::search::{SearchArtifactFilter, SearchHit, SearchProgress, SearchSort};
use super::{ModelCommandContext, acquisition, output};

#[derive(Clone, Debug)]
pub struct AcquiredModel {
    pub path: PathBuf,
    pub draft_ref: Option<String>,
}

#[derive(Clone, Debug)]
pub struct CatalogSummary {
    pub name: String,
    pub size: Option<String>,
    pub description: Option<String>,
}

pub fn canonical_catalog_ref(query: &str) -> String {
    remote_catalog::find_model_exact(query)
        .map(|model| model.exact_ref())
        .unwrap_or_else(|| query.to_string())
}

pub fn catalog_summaries() -> Result<Vec<CatalogSummary>> {
    remote_catalog::ensure_catalog().context("load model catalog")?;
    Ok(remote_catalog::loaded_models()
        .context("parse model catalog")?
        .into_iter()
        .map(|model| CatalogSummary {
            name: model.name,
            size: model.size,
            description: model.description,
        })
        .collect())
}

pub async fn acquire_model_ref(
    model_ref: &str,
    context: ModelCommandContext,
) -> Result<AcquiredModel> {
    let download = output::scope(
        context,
        acquisition::download_model_ref_with_progress_details(model_ref, true),
    )
    .await?;
    Ok(AcquiredModel {
        path: download.path,
        draft_ref: download.details.and_then(|details| details.draft),
    })
}

pub fn search_catalog_models(query: &str) -> Result<Vec<remote_catalog::RemoteCatalogModel>> {
    super::search::search_catalog_models(query)
}

pub fn search_catalog_json_payload(
    query: &str,
    filter: SearchArtifactFilter,
    sort: SearchSort,
    results: &[remote_catalog::RemoteCatalogModel],
    limit: usize,
    context: ModelCommandContext,
) -> Value {
    output::sync_scope(context, || {
        super::search::search_catalog_json_payload(query, filter, sort, results, limit)
    })
}

pub fn search_huggingface_json_payload(
    query: &str,
    filter: SearchArtifactFilter,
    sort: SearchSort,
    results: &[SearchHit],
    context: ModelCommandContext,
) -> Value {
    output::sync_scope(context, || {
        super::search::search_huggingface_json_payload(query, filter, sort, results)
    })
}

pub async fn search_huggingface<F>(
    query: &str,
    limit: usize,
    filter: SearchArtifactFilter,
    sort: SearchSort,
    progress: F,
    context: ModelCommandContext,
) -> Result<Vec<SearchHit>>
where
    F: FnMut(SearchProgress),
{
    output::scope(
        context,
        super::search::search_huggingface(query, limit, filter, sort, progress),
    )
    .await
}
