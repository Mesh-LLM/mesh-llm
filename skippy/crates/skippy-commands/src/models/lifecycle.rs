//! Embedding API for model acquisition used by product command adapters.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde_json::Value;
use skippy_model_hf::remote_catalog;
use skippy_model_hf::store::local::find_model_path;

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

/// Skippy's startup source selection. Product adapters own presentation and
/// usage reporting, while this ordering decides where a model comes from.
pub enum StartupModelSource {
    Passthrough(PathBuf),
    Existing(PathBuf),
    Catalog(remote_catalog::RemoteModelRef),
    Installed(PathBuf),
    Download {
        reference: String,
        bare_name_fallback: bool,
    },
    MissingBareName,
}

pub async fn resolve_startup_model_source(input: &Path) -> Result<StartupModelSource> {
    let raw = input.to_string_lossy();
    if raw.starts_with("hf://") {
        return Ok(StartupModelSource::Passthrough(input.to_path_buf()));
    }
    if input.exists() {
        return Ok(StartupModelSource::Existing(input.to_path_buf()));
    }
    if !raw.contains('/') {
        let name = raw.strip_suffix(".gguf").unwrap_or(&raw);
        let query = raw.to_string();
        if let Some(model) =
            tokio::task::spawn_blocking(move || remote_catalog::resolve_model_download(&query))
                .await
                .context("join remote catalog resolve task")?
        {
            return Ok(StartupModelSource::Catalog(model));
        }
        let installed = find_model_path(name);
        if installed.exists() {
            return Ok(StartupModelSource::Installed(installed));
        }
        if let Ok(canonical) = super::details::canonicalize_model_ref_input(&raw).await
            && canonical != raw
        {
            return Ok(StartupModelSource::Download {
                reference: canonical,
                bare_name_fallback: true,
            });
        }
        return Ok(StartupModelSource::MissingBareName);
    }
    let installed = find_model_path(&raw);
    if installed.exists() {
        return Ok(StartupModelSource::Installed(installed));
    }
    Ok(StartupModelSource::Download {
        reference: raw.into_owned(),
        bare_name_fallback: false,
    })
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
