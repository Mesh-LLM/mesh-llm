use super::formatters::{
    DownloadRenderInput, InstalledRow, JsonFormatter, ModelsFormatter, SearchFormatter,
    capabilities_json, catalog_model_capabilities, catalog_model_kind_code,
    fit_code_for_size_label, format_installed_size, huggingface_cache_dir,
    installed_model_kind_code, local_capacity_json, model_kind_code, print_json,
};
use anyhow::Result;
use mesh_llm_host_runtime::command_support::models::{
    DeleteResult as CliDeleteResult, ResolvedModel as CliResolvedModel,
};
use mesh_llm_host_runtime::command_support::models::{
    ModelDetails, SearchArtifactFilter, SearchHit, SearchSort, remote_catalog,
    remote_catalog_model_draft_ref, remote_catalog_model_ref, search_catalog_json_payload,
    search_huggingface_json_payload,
};
use serde_json::{Value, json};
use std::path::Path;

fn show_payload(details: &ModelDetails, variants: Option<&[ModelDetails]>) -> Value {
    json!({
        "display_name": details.exact_ref,
        "ref": details.exact_ref,
        "type": model_kind_code(details.kind),
        "source": details.source,
        "size": details.size_label,
        "fit": details
            .size_label
            .as_deref()
            .and_then(fit_code_for_size_label),
        "description": details.description,
        "draft": details.draft,
        "capabilities": capabilities_json(details.capabilities),
        "download_url": details.download_url,
        "machine": local_capacity_json(),
        "variants": variants
            .unwrap_or_default()
            .iter()
            .map(|variant| {
                json!({
                    "display_name": variant.exact_ref,
                    "ref": variant.exact_ref,
                    "type": model_kind_code(variant.kind),
                    "source": variant.source,
                    "size": variant.size_label,
                    "fit": variant
                        .size_label
                        .as_deref()
                        .and_then(fit_code_for_size_label),
                    "download_url": variant.download_url,
                })
            })
            .collect::<Vec<_>>(),
    })
}

fn download_payload(input: &DownloadRenderInput<'_>) -> Value {
    let paths = input.all_paths();
    let mut payload = json!({
        "requested_ref": input.model_ref,
        "path": input.path,
        "type": input.details.as_ref().map(|d| model_kind_code(d.kind)),
        "resolved_ref": input.details.as_ref().map(|d| d.exact_ref.clone()),
    });
    if paths.len() > 1 {
        payload["part_count"] = json!(input.part_count());
        payload["paths"] = json!(paths);
    }
    if let Some(stats) = input.stats {
        let elapsed = stats.elapsed.as_secs_f64();
        payload["download"] = json!({
            "bytes": stats.bytes,
            "elapsed_seconds": elapsed,
            "bytes_per_second": stats.bytes_per_second,
            "avg_bytes_per_second": stats
                .bytes
                .and_then(|bytes| (elapsed > 0.0).then(|| (bytes as f64 / elapsed).round() as u64)),
        });
    }
    if input.include_draft {
        payload["draft"] = match input.draft {
            Some((name, draft_path)) => json!({
                "name": name,
                "path": draft_path,
            }),
            None => Value::Null,
        };
    }
    payload
}

impl SearchFormatter for JsonFormatter {
    fn is_json(&self) -> bool {
        true
    }

    fn render_catalog_empty(
        &self,
        query: &str,
        filter: SearchArtifactFilter,
        sort: SearchSort,
    ) -> Result<()> {
        print_json(search_catalog_json_payload(query, filter, sort, &[], 0))
    }

    fn render_catalog_results(
        &self,
        query: &str,
        filter: SearchArtifactFilter,
        results: &[remote_catalog::RemoteCatalogModel],
        limit: usize,
        sort: SearchSort,
    ) -> Result<()> {
        print_json(search_catalog_json_payload(
            query, filter, sort, results, limit,
        ))
    }

    fn render_hf_empty(
        &self,
        query: &str,
        filter: SearchArtifactFilter,
        sort: SearchSort,
    ) -> Result<()> {
        print_json(search_huggingface_json_payload(query, filter, sort, &[]))
    }

    fn render_hf_results(
        &self,
        query: &str,
        filter: SearchArtifactFilter,
        sort: SearchSort,
        results: &[SearchHit],
    ) -> Result<()> {
        print_json(search_huggingface_json_payload(
            query, filter, sort, results,
        ))
    }
}

impl ModelsFormatter for JsonFormatter {
    fn render_recommended(&self, models: &[remote_catalog::RemoteCatalogModel]) -> Result<()> {
        let results: Vec<Value> = models
            .iter()
            .map(|model| {
                let model_capabilities = catalog_model_capabilities(model);
                let model_ref = remote_catalog_model_ref(model);
                json!({
                    "name": model.name,
                    "size": model.size,
                    "description": model.description,
                    "draft": remote_catalog_model_draft_ref(model),
                    "type": catalog_model_kind_code(model),
                    "ref": model_ref,
                    "show": format!("mesh-llm models show {model_ref}"),
                    "download": format!("mesh-llm models download {model_ref}"),
                    "capabilities": capabilities_json(model_capabilities),
                })
            })
            .collect();
        print_json(json!({
            "source": "catalog",
            "results": results,
        }))
    }

    fn render_installed(&self, rows: &[InstalledRow]) -> Result<()> {
        let models: Vec<Value> = rows
            .iter()
            .map(|row| {
                json!({
                    "name": row.name,
                    "type": installed_model_kind_code(&row.path),
                    "size_bytes": row.size,
                    "size": row.size.map(super::formatters::format_installed_size),
                    "layer_count": row.layer_count,
                    "mesh_managed": row.managed_by_mesh,
                    "last_used_at": row.last_used_at,
                    "capabilities": capabilities_json(row.capabilities),
                    "ref": row.model_ref,
                    "show": row.show_command.as_deref(),
                    "download": row.download_command.as_deref(),
                    "delete": row.delete_command.as_str(),
                    "path": row.path,
                    "about": row.catalog_model.as_ref().and_then(|m| m.description.clone()),
                    "draft": row.catalog_model.as_ref().and_then(|m| m.draft.clone()),
                })
            })
            .collect();
        print_json(json!({
            "cache_dir": huggingface_cache_dir(),
            "delete_example": rows
                .first()
                .map(|row| row.delete_command.clone()),
            "results": models,
        }))
    }

    fn render_show(&self, details: &ModelDetails, variants: Option<&[ModelDetails]>) -> Result<()> {
        print_json(show_payload(details, variants))
    }

    fn render_download(&self, input: DownloadRenderInput<'_>) -> Result<()> {
        print_json(download_payload(&input))
    }

    fn render_layer_package_download(
        &self,
        model_ref: &str,
        package_ref: &str,
        path: &Path,
    ) -> Result<()> {
        print_json(json!({
            "requested_ref": model_ref,
            "type": "layer_package",
            "package_ref": package_ref,
            "path": path,
        }))
    }

    fn render_updates_status(&self, repo: Option<&str>, all: bool, check: bool) -> Result<()> {
        print_json(json!({
            "status": "ok",
            "mode": if check { "check" } else { "update" },
            "target": {
                "repo": repo,
                "all": all,
            },
        }))
    }

    fn render_delete_preview(&self, resolved: &CliResolvedModel) -> Result<()> {
        let file_size = resolved
            .paths
            .iter()
            .map(|path| std::fs::metadata(path).map(|m| m.len()).unwrap_or(0))
            .sum::<u64>();
        print_json(json!({
            "display_name": resolved.display_name,
            "path": resolved.path,
            "paths": resolved.paths,
            "file_count": resolved.paths.len(),
            "derived_stage_paths": resolved.derived_stage_paths,
            "derived_stage_file_count": resolved.derived_stage_paths.len(),
            "is_exact_path": resolved.is_exact_path,
            "file_size_bytes": file_size,
            "file_size_human": format_installed_size(file_size),
            "matched_records": resolved.matched_records.iter().map(|r| json!({
                "lookup_key": r.lookup_key,
                "display_name": r.display_name,
                "last_used_at": r.last_used_at,
            })).collect::<Vec<_>>(),
            "dry_run": true,
        }))
    }

    fn render_delete_result(&self, result: &CliDeleteResult) -> Result<()> {
        print_json(json!({
            "deleted_paths": result.deleted_paths.iter().map(|p| p.to_string_lossy().to_string()).collect::<Vec<_>>(),
            "reclaimed_bytes": result.reclaimed_bytes,
            "reclaimed_bytes_human": format_installed_size(result.reclaimed_bytes),
            "removed_metadata_files": result.removed_metadata_files,
            "removed_usage_records": result.removed_usage_records,
            "removed_derived_cache_files": result.removed_derived_cache_files,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::models::formatters::DownloadRenderInput;
    use mesh_llm_host_runtime::command_support::models::ModelCapabilities;
    use std::path::Path;

    #[test]
    fn show_payload_includes_variants_for_selected_gguf_ref() {
        let details = ModelDetails {
            display_name: "Qwen3.6-35B-A3B-BF16.gguf".to_string(),
            exact_ref: "unsloth/Qwen3.6-35B-A3B-GGUF:BF16".to_string(),
            source: "huggingface",
            kind: "🦙 GGUF",
            download_url: "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/BF16/Qwen3.6-35B-A3B-BF16-00001-of-00002.gguf".to_string(),
            size_label: Some("49.9GB".to_string()),
            description: None,
            draft: None,
            capabilities: ModelCapabilities::default(),
        };
        let variants = vec![
            ModelDetails {
                display_name: "Qwen3.6-35B-A3B-BF16.gguf".to_string(),
                exact_ref: "unsloth/Qwen3.6-35B-A3B-GGUF:BF16".to_string(),
                source: "huggingface",
                kind: "🦙 GGUF",
                download_url: "https://example.invalid/bf16.gguf".to_string(),
                size_label: Some("49.9GB".to_string()),
                description: None,
                draft: None,
                capabilities: ModelCapabilities::default(),
            },
            ModelDetails {
                display_name: "Qwen3.6-35B-A3B-Q4_K_M.gguf".to_string(),
                exact_ref: "unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_M".to_string(),
                source: "huggingface",
                kind: "🦙 GGUF",
                download_url: "https://example.invalid/q4_k_m.gguf".to_string(),
                size_label: Some("21.3GB".to_string()),
                description: None,
                draft: None,
                capabilities: ModelCapabilities::default(),
            },
        ];

        let payload = show_payload(&details, Some(&variants));
        let emitted_variants = payload["variants"].as_array().expect("variants array");

        assert_eq!(payload["ref"], "unsloth/Qwen3.6-35B-A3B-GGUF:BF16");
        assert_eq!(emitted_variants.len(), 2);
        assert_eq!(
            emitted_variants[0]["ref"],
            "unsloth/Qwen3.6-35B-A3B-GGUF:BF16"
        );
        assert_eq!(
            emitted_variants[1]["ref"],
            "unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_M"
        );
    }

    #[test]
    fn download_payload_includes_all_multipart_paths() {
        let paths = vec![
            Path::new("/cache/model-00001-of-00003.gguf"),
            Path::new("/cache/model-00002-of-00003.gguf"),
            Path::new("/cache/model-00003-of-00003.gguf"),
        ];
        let input = DownloadRenderInput {
            model_ref: "org/repo:model",
            path: paths[0],
            paths: &paths,
            details: None,
            stats: None,
            include_draft: false,
            draft: None,
        };

        let payload = download_payload(&input);

        assert_eq!(payload["path"], "/cache/model-00001-of-00003.gguf");
        assert_eq!(payload["part_count"], 3);
        assert_eq!(payload["paths"].as_array().expect("paths array").len(), 3);
        assert_eq!(payload["paths"][1], "/cache/model-00002-of-00003.gguf");
    }
}
