use super::super::http::{respond_error, respond_json};
use crate::models::search::fit_budget_bytes;
use crate::models::{
    SearchArtifactFilter, SearchSort, remote_catalog, search_catalog_json_payload,
    search_catalog_models, search_huggingface, search_huggingface_json_payload,
};
use url::form_urlencoded;

const DEFAULT_LIMIT: usize = 20;
const MAX_LIMIT: usize = 50;

#[derive(Clone, Debug, Eq, PartialEq)]
struct SearchRequest {
    query: String,
    artifact: SearchArtifactFilter,
    catalog_only: bool,
    limit: usize,
    sort: SearchSort,
}

pub(super) async fn handle(stream: &mut tokio::net::TcpStream, path: &str) -> anyhow::Result<()> {
    let request = match parse_request(path) {
        Ok(request) => request,
        Err(message) => return respond_error(stream, 400, &message).await,
    };

    if request.catalog_only {
        let results = match search_catalog_models(&request.query) {
            Ok(results) => results
                .into_iter()
                .filter(|model| catalog_model_matches_artifact(model, request.artifact))
                .collect::<Vec<_>>(),
            Err(err) => {
                return respond_error(stream, 502, &format!("Catalog search failed: {err}")).await;
            }
        };
        let response = search_catalog_json_payload(
            &request.query,
            request.artifact,
            request.sort,
            &results,
            request.limit,
        );
        return respond_json(stream, 200, &response).await;
    }

    let fit_budget = tokio::task::spawn_blocking(fit_budget_bytes).await?;
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
            let response = search_huggingface_json_payload(
                &request.query,
                request.artifact,
                request.sort,
                &results,
                fit_budget,
            );
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
    Ok(limit.min(MAX_LIMIT))
}

fn parse_sort(value: &str) -> Result<SearchSort, String> {
    match value {
        "trending" => Ok(SearchSort::Trending),
        "downloads" => Ok(SearchSort::Downloads),
        "likes" => Ok(SearchSort::Likes),
        "created" => Ok(SearchSort::Created),
        "updated" => Ok(SearchSort::Updated),
        "parameters-desc" => Ok(SearchSort::ParametersDesc),
        "parameters-asc" => Ok(SearchSort::ParametersAsc),
        _ => Err(format!(
            "Invalid 'sort' value '{value}'. Expected one of: trending, downloads, likes, created, updated, parameters-desc, parameters-asc"
        )),
    }
}

fn catalog_model_matches_artifact(
    model: &remote_catalog::RemoteCatalogModel,
    artifact: SearchArtifactFilter,
) -> bool {
    let is_mlx = model.source_file().ends_with("model.safetensors")
        || model
            .source_file()
            .ends_with("model.safetensors.index.json");
    match artifact {
        SearchArtifactFilter::Gguf => !is_mlx,
        SearchArtifactFilter::Mlx => is_mlx,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[serial_test::serial]
    async fn catalog_search_http_preserves_fields_hints_and_errors() {
        use tokio::io::AsyncReadExt;
        let entry = serde_json::from_value(serde_json::json!({
            "schema_version":1,"source_repo":"owner/demo","variants":{
                "FixtureSearch-Q4_K_M":{"source":{"repo":"owner/demo","file":"demo-Q4_K_M.gguf"},"curated":{"name":"FixtureSearch-Q4_K_M","size":"4GB","description":"Fixture model"}}
            }
        })).unwrap();
        let _catalog = remote_catalog::set_catalog_entries_for_test(vec![entry]);
        for (path, status) in [
            (
                "/api/search?q=FixtureSearch&catalog=true&limit=1",
                "HTTP/1.1 200 OK",
            ),
            ("/api/search?catalog=true", "HTTP/1.1 400 Bad Request"),
        ] {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let mut client = tokio::net::TcpStream::connect(listener.local_addr().unwrap())
                .await
                .unwrap();
            let (mut server, _) = listener.accept().await.unwrap();
            let handler = async {
                handle(&mut server, path).await.unwrap();
                drop(server);
            };
            let reader = async {
                let mut response = Vec::new();
                client.read_to_end(&mut response).await.unwrap();
                response
            };
            let (_, response) = tokio::time::timeout(std::time::Duration::from_secs(30), async {
                tokio::join!(handler, reader)
            })
            .await
            .unwrap();
            assert!(String::from_utf8_lossy(&response).starts_with(status));
            let body: serde_json::Value =
                serde_json::from_str(crate::api::http::http_body_text(&response)).unwrap();
            if path.contains("q=") {
                assert_eq!(body["source"], "catalog");
                assert_eq!(body["results"].as_array().unwrap().len(), 1);
                let model = &body["results"][0];
                assert_eq!(model["ref"], "owner/demo:Q4_K_M");
                assert_eq!(model["show"], "mesh-llm models show owner/demo:Q4_K_M");
                assert_eq!(
                    model["download"],
                    "mesh-llm models download owner/demo:Q4_K_M"
                );
                assert_eq!(model["size"], "4GB");
                assert_eq!(model["capabilities"]["text"], true);
            } else {
                assert_eq!(body["error"], "Missing required 'q' query parameter");
            }
        }
    }

    #[test]
    fn parse_request_requires_non_empty_query() {
        let err = parse_request("/api/search?artifact=gguf").unwrap_err();
        assert_eq!(err, "Missing required 'q' query parameter");

        let err = parse_request("/api/search?q=%20%20").unwrap_err();
        assert_eq!(err, "Missing required 'q' query parameter");
    }

    #[test]
    fn parse_request_accepts_canonical_sort_names_and_caps_limit() {
        let request = parse_request(
            "/api/search?q=qwen&artifact=mlx&catalog=true&limit=999&sort=parameters-desc",
        )
        .unwrap();
        assert_eq!(request.query, "qwen");
        assert_eq!(request.artifact, SearchArtifactFilter::Mlx);
        assert!(request.catalog_only);
        assert_eq!(request.limit, MAX_LIMIT);
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
            "Invalid 'sort' value 'random'. Expected one of: trending, downloads, likes, created, updated, parameters-desc, parameters-asc"
        );

        let err = parse_request("/api/search?q=qwen&sort=most-parameters").unwrap_err();
        assert_eq!(
            err,
            "Invalid 'sort' value 'most-parameters'. Expected one of: trending, downloads, likes, created, updated, parameters-desc, parameters-asc"
        );
    }
}
