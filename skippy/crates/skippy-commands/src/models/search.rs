//! Search presentation for the shared model commands.
use anyhow::Result;
use serde_json::Value;
use skippy_model_hf::search::{
    SearchArtifactFilter, SearchHit, SearchObserver, SearchProgress, SearchSort,
};
use skippy_model_hf::{remote_catalog, search};
use std::io::Write;

struct CommandSearchObserver<F>(F);
impl<F: FnMut(SearchProgress) + Send> SearchObserver for CommandSearchObserver<F> {
    fn progress(&mut self, progress: SearchProgress) {
        (self.0)(progress);
    }
    fn inspection_warning(&mut self, error: &anyhow::Error) -> Result<()> {
        writeln!(
            super::output::console_err(),
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
        &mut CommandSearchObserver(progress),
    )
    .await
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
        super::output::fit_budget_bytes(),
    );
    add_command_hints(&mut payload);
    payload
}

pub fn search_huggingface_json_payload(
    query: &str,
    filter: SearchArtifactFilter,
    sort: SearchSort,
    results: &[SearchHit],
) -> Value {
    let mut payload = search::projection::search_huggingface_json_payload(
        query,
        filter,
        sort,
        results,
        super::output::fit_budget_bytes(),
    );
    add_command_hints(&mut payload);
    payload
}

fn add_command_hints(payload: &mut Value) {
    for result in payload["results"].as_array_mut().into_iter().flatten() {
        let model_ref = result["ref"].as_str().unwrap_or_default().to_string();
        let program = super::output::program();
        result["show"] = format!("{program} models show {model_ref}").into();
        result["download"] = format!("{program} models download {model_ref}").into();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use skippy_model_hf::{capabilities, details::remote_catalog_model_ref};
    fn test_remote_catalog_model() -> remote_catalog::RemoteCatalogModel {
        remote_catalog::RemoteCatalogModel {
            name: "Qwen3-Coder-Next-Q4_K_M".to_string(),
            file: "Qwen3-Coder-Next-Q4_K_M.gguf".to_string(),
            repo: "Qwen/Qwen3-Coder-Next-GGUF".to_string(),
            revision: Some("main".to_string()),
            source_file: "Qwen3-Coder-Next-Q4_K_M.gguf".to_string(),
            size: Some("20GB".to_string()),
            description: Some("Coding model".to_string()),
            draft: None,
            extra_files: Vec::new(),
            mmproj: None,
        }
    }

    #[test]
    fn search_catalog_json_payload_uses_cli_contract_fields() {
        let model = test_remote_catalog_model();
        let payload = search_catalog_json_payload(
            "qwen",
            SearchArtifactFilter::Gguf,
            SearchSort::ParametersDesc,
            std::slice::from_ref(&model),
            1,
        );

        assert_eq!(payload["filter"], serde_json::json!("gguf"));
        assert_eq!(payload["sort"], serde_json::json!("parameters-desc"));
        assert!(payload.get("machine").is_some());
        let result = &payload["results"][0];
        let model_ref = remote_catalog_model_ref(&model);
        assert_eq!(result["ref"], serde_json::json!(model_ref));
        assert_eq!(result["type"], serde_json::json!("gguf"));
        assert_eq!(
            result["show"],
            serde_json::json!(format!(
                "{program} models show {model_ref}",
                program = crate::models::output::program()
            ))
        );
    }

    #[test]
    fn search_huggingface_json_payload_uses_cli_contract_fields() {
        let model = test_remote_catalog_model();
        let payload = search_huggingface_json_payload(
            "qwen",
            SearchArtifactFilter::Gguf,
            SearchSort::ParametersAsc,
            &[SearchHit {
                repo_id: "Qwen/Qwen3-Coder-Next-GGUF".to_string(),
                kind: "🦙 GGUF",
                exact_ref: "Qwen3-Coder-Next-Q4_K_M".to_string(),
                variant_count: Some(3),
                size_label: model.size.clone(),
                downloads: Some(42),
                likes: Some(7),
                catalog: Some(model.clone()),
                capabilities: capabilities::infer_remote_catalog_capabilities(&model),
            }],
        );

        assert_eq!(payload["filter"], serde_json::json!("gguf"));
        assert_eq!(payload["sort"], serde_json::json!("parameters-asc"));
        let result = &payload["results"][0];
        assert_eq!(result["ref"], serde_json::json!("Qwen3-Coder-Next-Q4_K_M"));
        assert_eq!(result["type"], serde_json::json!("gguf"));
        assert_eq!(
            result["repo_url"],
            serde_json::json!("https://huggingface.co/Qwen/Qwen3-Coder-Next-GGUF")
        );
        assert_eq!(
            result["catalog"]["name"],
            serde_json::json!("Qwen3-Coder-Next-Q4_K_M")
        );
    }
}
