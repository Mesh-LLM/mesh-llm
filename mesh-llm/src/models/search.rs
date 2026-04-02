use super::resolve::{
    file_preference_score, matching_catalog_model_for_huggingface, merge_capabilities,
    remote_hf_size_label_with_api,
};
use super::ModelCapabilities;
use super::{build_hf_tokio_api, capabilities, catalog};
use anyhow::{Context, Result};
use hf_hub::api::tokio::Api as TokioApi;
use hf_hub::api::RepoSummary;
use hf_hub::RepoType;
use tokio::task::JoinSet;

#[derive(Clone, Debug)]
pub struct SearchHit {
    pub repo_id: String,
    pub file: String,
    pub exact_ref: String,
    pub size_label: Option<String>,
    pub downloads: Option<u64>,
    pub likes: Option<u64>,
    pub catalog: Option<&'static catalog::CatalogModel>,
    pub capabilities: ModelCapabilities,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SearchProgress {
    SearchingHub,
    InspectingRepos { completed: usize, total: usize },
}

pub fn search_catalog_models(query: &str) -> Vec<&'static catalog::CatalogModel> {
    let q = query.to_lowercase();
    let mut results: Vec<_> = catalog::MODEL_CATALOG
        .iter()
        .filter(|model| {
            model.name.to_lowercase().contains(&q)
                || model.file.to_lowercase().contains(&q)
                || model.description.to_lowercase().contains(&q)
        })
        .collect();
    results.sort_by(|left, right| left.name.cmp(&right.name));
    results
}

// Keep search custom for now. `hf-hub` handles cache and file transport well,
// but it does not expose a Hub search surface in this crate version.
pub async fn search_huggingface<F>(
    query: &str,
    limit: usize,
    mut progress: F,
) -> Result<Vec<SearchHit>>
where
    F: FnMut(SearchProgress),
{
    const SEARCH_CONCURRENCY: usize = 6;

    let repo_limit = limit.clamp(1, 100);
    progress(SearchProgress::SearchingHub);
    let api = build_hf_tokio_api(false)?;
    let repos = api
        .search(RepoType::Model)
        .with_query(query)
        .with_filter("gguf")
        .with_limit(repo_limit)
        .run()
        .await
        .context("Search Hugging Face")?;

    let total = repos.len();
    progress(SearchProgress::InspectingRepos {
        completed: 0,
        total,
    });

    let mut pending = repos.into_iter().enumerate();
    let mut join_set = JoinSet::new();
    for _ in 0..SEARCH_CONCURRENCY.min(total.max(1)) {
        if let Some((index, repo)) = pending.next() {
            let api = api.clone();
            join_set.spawn(async move { (index, build_search_hit(api, repo).await) });
        }
    }

    let mut completed = 0usize;
    let mut indexed_hits = Vec::new();
    while let Some(joined) = join_set.join_next().await {
        let (index, result) = joined.context("Join Hugging Face repo inspection task")?;
        completed += 1;
        progress(SearchProgress::InspectingRepos { completed, total });
        if let Some(hit) = result? {
            indexed_hits.push((index, hit));
        }
        if let Some((next_index, repo)) = pending.next() {
            let api = api.clone();
            join_set.spawn(async move { (next_index, build_search_hit(api, repo).await) });
        }
    }

    indexed_hits.sort_by_key(|(index, _)| *index);
    let mut hits: Vec<SearchHit> = indexed_hits
        .into_iter()
        .map(|(_, hit)| hit)
        .take(limit)
        .collect();
    if hits.len() > limit {
        hits.truncate(limit);
    }
    Ok(hits)
}

async fn build_search_hit(api: TokioApi, repo: RepoSummary) -> Result<Option<SearchHit>> {
    let detail = api
        .repo(repo.repo())
        .info()
        .await
        .with_context(|| format!("Fetch Hugging Face repo {}", repo.id))?;

    let repo_id = detail
        .id
        .clone()
        .or(detail.model_id.clone())
        .unwrap_or(repo.id.clone());
    let sibling_names: Vec<String> = detail
        .siblings
        .iter()
        .map(|sibling| sibling.rfilename.clone())
        .collect();
    let mut files: Vec<String> = detail
        .siblings
        .into_iter()
        .map(|sibling| sibling.rfilename)
        .filter(|file| file.ends_with(".gguf"))
        .collect();
    if files.is_empty() {
        return Ok(None);
    }
    files.sort_by(|left, right| {
        file_preference_score(left)
            .cmp(&file_preference_score(right))
            .then_with(|| left.cmp(right))
    });
    let Some(file) = files.into_iter().next() else {
        return Ok(None);
    };
    let catalog = matching_catalog_model_for_huggingface(&repo_id, None, &file);
    let size_label = match catalog {
        Some(model) => Some(model.size.to_string()),
        None => remote_hf_size_label_with_api(&api, &repo_id, None, &file).await,
    };
    let remote_caps =
        capabilities::infer_remote_hf_capabilities(&repo_id, None, &file, Some(&sibling_names))
            .await;
    let capabilities = match catalog {
        Some(model) => {
            let base = capabilities::infer_catalog_capabilities(model);
            merge_capabilities(base, remote_caps)
        }
        None => remote_caps,
    };
    Ok(Some(SearchHit {
        repo_id: repo_id.clone(),
        file: file.clone(),
        exact_ref: format!("{repo_id}/{file}"),
        size_label,
        downloads: detail.downloads.or(repo.downloads),
        likes: detail.likes.or(repo.likes),
        catalog,
        capabilities,
    }))
}
