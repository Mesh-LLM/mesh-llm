//! Shared search response fields, independent of command hints and output policy.
use super::{SearchArtifactFilter, SearchHit, SearchSort};
use crate::details::{remote_catalog_model_draft_ref, remote_catalog_model_ref};
use crate::{capabilities, remote_catalog};
use serde_json::{Value, json};
use skippy_model_artifact::capabilities::ModelCapabilities;

pub fn search_catalog_json_payload(
    query: &str,
    filter: SearchArtifactFilter,
    sort: SearchSort,
    results: &[remote_catalog::RemoteCatalogModel],
    limit: usize,
    fit_budget_bytes: u64,
) -> Value {
    let payload_results: Vec<Value> = results
        .iter()
        .take(limit)
        .map(|model| {
            let model_ref = remote_catalog_model_ref(model);
            json!({
                "name": model.name,
                "repo_id": model.source_repo(),
                "type": catalog_model_kind_code(model),
                "size": model.size,
                "description": model.description,
                "fit": model.size.as_deref().and_then(|size| fit_code_for_size_label(size, fit_budget_bytes)),
                "ref": model_ref,
                "draft": remote_catalog_model_draft_ref(model),
                "capabilities": capabilities_json(capabilities::infer_remote_catalog_capabilities(model)),
            })
        })
        .collect();
    json!({
        "query": query,
        "filter": search_filter_name(filter),
        "sort": search_sort_name(sort),
        "source": "catalog",
        "machine": local_capacity_json(fit_budget_bytes),
        "results": payload_results,
    })
}

pub fn search_huggingface_json_payload(
    query: &str,
    filter: SearchArtifactFilter,
    sort: SearchSort,
    results: &[SearchHit],
    fit_budget_bytes: u64,
) -> Value {
    let payload_results: Vec<Value> = results
        .iter()
        .map(|result| {
            json!({
                "repo_id": result.repo_id,
                "repo_url": huggingface_repo_url(&result.repo_id),
                "type": model_kind_code(result.kind),
                "variant_count": result.variant_count,
                "size": result.size_label,
                "downloads": result.downloads,
                "likes": result.likes,
                "fit": result
                    .size_label
                    .as_deref()
                    .and_then(|size| fit_code_for_size_label(size, fit_budget_bytes)),
                "ref": result.exact_ref,
                "capabilities": capabilities_json(result.capabilities),
                "catalog": result.catalog.as_ref().map(|model| {
                    json!({
                        "name": model.name,
                        "size": model.size,
                        "description": model.description,
                    })
                }),
            })
        })
        .collect();
    json!({
        "query": query,
        "filter": search_filter_name(filter),
        "sort": search_sort_name(sort),
        "source": "huggingface",
        "machine": local_capacity_json(fit_budget_bytes),
        "results": payload_results,
    })
}

fn search_filter_name(filter: SearchArtifactFilter) -> &'static str {
    match filter {
        SearchArtifactFilter::Gguf => "gguf",
        SearchArtifactFilter::Mlx => "mlx",
    }
}

fn search_sort_name(sort: SearchSort) -> &'static str {
    match sort {
        SearchSort::Trending => "trending",
        SearchSort::Downloads => "downloads",
        SearchSort::Likes => "likes",
        SearchSort::Created => "created",
        SearchSort::Updated => "updated",
        SearchSort::ParametersDesc => "parameters-desc",
        SearchSort::ParametersAsc => "parameters-asc",
    }
}

fn local_capacity_json(fit_budget_bytes: u64) -> Value {
    let vram_bytes = fit_budget_bytes;
    let vram_gb = vram_bytes as f64 / 1e9;
    json!({
        "vram_bytes": vram_bytes,
        "vram_gb": vram_gb,
    })
}

fn capabilities_json(caps: ModelCapabilities) -> Value {
    json!({
        "text": true,
        "multimodal": caps.multimodal_status(),
        "vision": caps.vision_status(),
        "audio": caps.audio_status(),
        "reasoning": caps.reasoning_status(),
        "tool_use": caps.tool_use_status(),
    })
}

fn fit_code_for_size_label(size_label: &str, fit_budget_bytes: u64) -> Option<&'static str> {
    let model_gb = skippy_model_artifact::sizes::parse_size_gb(size_label);
    let vram_gb = fit_budget_bytes as f64 / 1e9;
    if model_gb <= 0.0 || vram_gb <= 0.0 {
        return None;
    }

    let code = if model_gb <= vram_gb * 0.6 {
        "comfortable"
    } else if model_gb <= vram_gb * 0.9 {
        "tight"
    } else if model_gb <= vram_gb * 1.1 {
        "tradeoff"
    } else {
        "too_large"
    };
    Some(code)
}

fn huggingface_repo_url(repo_id: &str) -> String {
    format!("https://huggingface.co/{repo_id}")
}

fn model_kind_code(kind: &str) -> &'static str {
    if kind.to_ascii_lowercase().contains("mlx") {
        "mlx"
    } else {
        "gguf"
    }
}

fn catalog_model_kind_code(model: &remote_catalog::RemoteCatalogModel) -> &'static str {
    if model.source_file().ends_with("model.safetensors")
        || model
            .source_file()
            .ends_with("model.safetensors.index.json")
    {
        "mlx"
    } else {
        "gguf"
    }
}
