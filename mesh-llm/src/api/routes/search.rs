use super::super::http::{respond_error, respond_json};
use crate::models::{
    capabilities, catalog, search_catalog_models, search_huggingface, SearchArtifactFilter,
    SearchHit, SearchSort,
};
use serde::Serialize;
use url::form_urlencoded;

const DEFAULT_LIMIT: usize = 20;

#[derive(Clone, Debug, Eq, PartialEq)]
struct SearchRequest {
    query: String,
    artifact: SearchArtifactFilter,
    catalog_only: bool,
    limit: usize,
    sort: SearchSort,
}

#[derive(Debug, Serialize)]
struct SearchResponse {
    query: String,
    artifact: &'static str,
    sort: &'static str,
    source: &'static str,
    limit: usize,
    results: Vec<SearchResult>,
}

#[derive(Debug, Serialize)]
struct SearchResult {
    model_ref: String,
    repo_id: Option<String>,
    repo_url: Option<String>,
    kind: &'static str,
    size: Option<String>,
    description: Option<String>,
    variant_count: Option<usize>,
    downloads: Option<u64>,
    likes: Option<u64>,
    capabilities: crate::models::ModelCapabilities,
}

pub(super) async fn handle(stream: &mut tokio::net::TcpStream, path: &str) -> anyhow::Result<()> {
    let request = match parse_request(path) {
        Ok(request) => request,
        Err(message) => return respond_error(stream, 400, &message).await,
    };

    if request.catalog_only {
        let results = search_catalog_models(&request.query)
            .into_iter()
            .filter(|model| catalog_model_matches_artifact(model, request.artifact))
            .take(request.limit)
            .map(catalog_result)
            .collect();
        let response = SearchResponse {
            query: request.query.clone(),
            artifact: artifact_name(request.artifact),
            sort: sort_name(request.sort),
            source: "catalog",
            limit: request.limit,
            results,
        };
        return respond_json(stream, 200, &response).await;
    }

    match search_huggingface(
        &request.query,
        request.limit,
        request.artifact,
        request.sort,
        |_| {},
    )
    .await
    {
        Ok(results) => {
            let response = SearchResponse {
                query: request.query.clone(),
                artifact: artifact_name(request.artifact),
                sort: sort_name(request.sort),
                source: "huggingface",
                limit: request.limit,
                results: results.iter().map(huggingface_result).collect(),
            };
            respond_json(stream, 200, &response).await
        }
        Err(err) => respond_error(stream, 502, &format!("Search failed: {err}")).await,
    }
}

fn parse_request(path: &str) -> Result<SearchRequest, String> {
    let mut query = None;
    let mut artifact = SearchArtifactFilter::Gguf;
    let mut catalog_only = false;
    let mut limit = DEFAULT_LIMIT;
    let mut sort = SearchSort::Trending;

    if let Some((_, raw_query)) = path.split_once('?') {
        for (key, value) in form_urlencoded::parse(raw_query.as_bytes()) {
            match key.as_ref() {
                "q" => query = Some(value),
                "artifact" => artifact = parse_artifact(&value)?,
                "catalog" => catalog_only = parse_bool(&value, "catalog")?,
                "limit" => limit = parse_limit(&value)?,
                "sort" => sort = parse_sort(&value)?,
                _ => {}
            }
        }
    }

    let query = query
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| "Missing required 'q' query parameter".to_string())?
        .to_string();

    Ok(SearchRequest {
        query,
        artifact,
        catalog_only,
        limit,
        sort,
    })
}

fn parse_artifact(value: &str) -> Result<SearchArtifactFilter, String> {
    match value {
        "gguf" => Ok(SearchArtifactFilter::Gguf),
        "mlx" => Ok(SearchArtifactFilter::Mlx),
        _ => Err(format!(
            "Invalid 'artifact' value '{value}'. Expected 'gguf' or 'mlx'"
        )),
    }
}

fn parse_bool(value: &str, field: &str) -> Result<bool, String> {
    match value {
        "true" | "1" | "yes" => Ok(true),
        "false" | "0" | "no" => Ok(false),
        _ => Err(format!(
            "Invalid '{field}' value '{value}'. Expected true or false"
        )),
    }
}

fn parse_limit(value: &str) -> Result<usize, String> {
    let limit = value
        .parse::<usize>()
        .map_err(|_| format!("Invalid 'limit' value '{value}'. Expected a positive integer"))?;
    if limit == 0 {
        return Err("Invalid 'limit' value '0'. Expected a positive integer".to_string());
    }
    Ok(limit)
}

fn parse_sort(value: &str) -> Result<SearchSort, String> {
    match value {
        "trending" => Ok(SearchSort::Trending),
        "downloads" => Ok(SearchSort::Downloads),
        "likes" => Ok(SearchSort::Likes),
        "created" => Ok(SearchSort::Created),
        "updated" => Ok(SearchSort::Updated),
        "most-parameters" | "parameters-desc" => Ok(SearchSort::ParametersDesc),
        "least-parameters" | "parameters-asc" => Ok(SearchSort::ParametersAsc),
        _ => Err(format!(
            "Invalid 'sort' value '{value}'. Expected one of: trending, downloads, likes, created, updated, most-parameters, least-parameters"
        )),
    }
}

fn artifact_name(filter: SearchArtifactFilter) -> &'static str {
    match filter {
        SearchArtifactFilter::Gguf => "gguf",
        SearchArtifactFilter::Mlx => "mlx",
    }
}

fn sort_name(sort: SearchSort) -> &'static str {
    match sort {
        SearchSort::Trending => "trending",
        SearchSort::Downloads => "downloads",
        SearchSort::Likes => "likes",
        SearchSort::Created => "created",
        SearchSort::Updated => "updated",
        SearchSort::ParametersDesc => "most-parameters",
        SearchSort::ParametersAsc => "least-parameters",
    }
}

fn catalog_model_matches_artifact(
    model: &catalog::CatalogModel,
    artifact: SearchArtifactFilter,
) -> bool {
    let is_mlx = model
        .source_file()
        .map(|file| {
            file.ends_with("model.safetensors") || file.ends_with("model.safetensors.index.json")
        })
        .unwrap_or(false)
        || model.url.contains("model.safetensors");
    match artifact {
        SearchArtifactFilter::Gguf => !is_mlx,
        SearchArtifactFilter::Mlx => is_mlx,
    }
}

fn catalog_result(model: &'static catalog::CatalogModel) -> SearchResult {
    SearchResult {
        model_ref: model.name.clone(),
        repo_id: model.source_repo().map(ToOwned::to_owned),
        repo_url: model
            .source_repo()
            .map(|repo_id| format!("https://huggingface.co/{repo_id}")),
        kind: if catalog_model_matches_artifact(model, SearchArtifactFilter::Mlx) {
            "mlx"
        } else {
            "gguf"
        },
        size: Some(model.size.clone()),
        description: Some(model.description.clone()),
        variant_count: None,
        downloads: None,
        likes: None,
        capabilities: capabilities::infer_catalog_capabilities(model),
    }
}

fn huggingface_result(hit: &SearchHit) -> SearchResult {
    SearchResult {
        model_ref: hit.exact_ref.clone(),
        repo_id: Some(hit.repo_id.clone()),
        repo_url: Some(format!("https://huggingface.co/{}", hit.repo_id)),
        kind: if hit.kind.to_ascii_lowercase().contains("mlx") {
            "mlx"
        } else {
            "gguf"
        },
        size: hit.size_label.clone(),
        description: hit.catalog.map(|model| model.description.clone()),
        variant_count: hit.variant_count,
        downloads: hit.downloads,
        likes: hit.likes,
        capabilities: hit.capabilities,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_request_requires_non_empty_query() {
        let err = parse_request("/api/search?artifact=gguf").unwrap_err();
        assert_eq!(err, "Missing required 'q' query parameter");

        let err = parse_request("/api/search?q=%20%20").unwrap_err();
        assert_eq!(err, "Missing required 'q' query parameter");
    }

    #[test]
    fn parse_request_accepts_cli_sort_names() {
        let request = parse_request(
            "/api/search?q=qwen&artifact=mlx&catalog=true&limit=7&sort=most-parameters",
        )
        .unwrap();
        assert_eq!(request.query, "qwen");
        assert_eq!(request.artifact, SearchArtifactFilter::Mlx);
        assert!(request.catalog_only);
        assert_eq!(request.limit, 7);
        assert_eq!(request.sort, SearchSort::ParametersDesc);
    }

    #[test]
    fn parse_request_rejects_invalid_values() {
        let err = parse_request("/api/search?q=qwen&artifact=onnx").unwrap_err();
        assert_eq!(
            err,
            "Invalid 'artifact' value 'onnx'. Expected 'gguf' or 'mlx'"
        );

        let err = parse_request("/api/search?q=qwen&limit=0").unwrap_err();
        assert_eq!(
            err,
            "Invalid 'limit' value '0'. Expected a positive integer"
        );

        let err = parse_request("/api/search?q=qwen&sort=random").unwrap_err();
        assert_eq!(
            err,
            "Invalid 'sort' value 'random'. Expected one of: trending, downloads, likes, created, updated, most-parameters, least-parameters"
        );
    }
}
