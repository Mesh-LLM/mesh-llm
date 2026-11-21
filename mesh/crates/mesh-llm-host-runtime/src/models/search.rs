//! Mesh presentation adapter for Skippy-owned model search.
use anyhow::Result;
use serde_json::Value;
pub use skippy_model_hf::search::{SearchArtifactFilter, SearchHit, SearchProgress, SearchSort};
use skippy_model_hf::{remote_catalog, search};
use std::io::Write;

pub(crate) fn fit_budget_bytes() -> u64 {
    mesh_llm_system::capacity::local_fit_budget_bytes(&crate::system::hardware::survey())
}

pub fn search_catalog_models(query: &str) -> Result<Vec<remote_catalog::RemoteCatalogModel>> {
    search::search_catalog_models(query)
}

pub fn search_catalog_json_payload(
    query: &str,
    filter: SearchArtifactFilter,
    sort: SearchSort,
    results: &[remote_catalog::RemoteCatalogModel],
    limit: usize,
) -> Value {
    let mut payload = search::projection::search_catalog_json_payload(
        query,
        filter,
        sort,
        results,
        limit,
        fit_budget_bytes(),
    );
    add_command_hints(&mut payload);
    payload
}

pub fn search_huggingface_json_payload(
    query: &str,
    filter: SearchArtifactFilter,
    sort: SearchSort,
    results: &[SearchHit],
    fit_budget_bytes: u64,
) -> Value {
    let mut payload = search::projection::search_huggingface_json_payload(
        query,
        filter,
        sort,
        results,
        fit_budget_bytes,
    );
    add_command_hints(&mut payload);
    payload
}

struct MeshSearchObserver<F>(F);
impl<F: FnMut(SearchProgress) + Send> search::SearchObserver for MeshSearchObserver<F> {
    fn progress(&mut self, progress: SearchProgress) {
        (self.0)(progress);
    }
    fn inspection_warning(&mut self, error: &anyhow::Error) -> Result<()> {
        writeln!(
            mesh_llm_events::console_err(),
            "⚠️  Failed to inspect Hugging Face repo: {error:#}"
        )?;
        Ok(())
    }
}

pub async fn search_huggingface<F>(
    query: &str,
    limit: usize,
    filter: SearchArtifactFilter,
    sort: SearchSort,
    progress: F,
) -> Result<Vec<SearchHit>>
where
    F: FnMut(SearchProgress) + Send,
{
    search::search_huggingface(
        query,
        limit,
        filter,
        sort,
        &mut MeshSearchObserver(progress),
    )
    .await
}

fn add_command_hints(payload: &mut Value) {
    for result in payload["results"].as_array_mut().into_iter().flatten() {
        let model_ref = result["ref"].as_str().unwrap_or_default().to_string();
        result["show"] = format!("mesh-llm models show {model_ref}").into();
        result["download"] = format!("mesh-llm models download {model_ref}").into();
    }
}
