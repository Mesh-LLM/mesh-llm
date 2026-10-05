//! Verified model acquisition and transfer summaries for model commands.
use super::details::{ModelDetails, show_exact_model};
pub(super) use super::transfer::DownloadTransferStats;
use anyhow::{Context, Result};
use skippy_model_hf::huggingface_hub_cache_dir;
use std::path::PathBuf;
pub(super) struct ModelDownload {
    pub path: PathBuf,
    pub paths: Vec<PathBuf>,
    pub details: Option<ModelDetails>,
    pub transfer_stats: Option<DownloadTransferStats>,
}
pub(super) async fn download_model_ref_with_progress_details(
    model: &str,
    progress: bool,
) -> Result<ModelDownload> {
    download_model_ref_with_progress_details_direct(model, progress, false).await
}
pub(super) async fn download_model_ref_with_progress_details_direct(
    model: &str,
    progress: bool,
    direct: bool,
) -> Result<ModelDownload> {
    let details = show_exact_model(model).await.ok();
    let resolved = details
        .as_ref()
        .map(|detail| detail.download_url.as_str())
        .unwrap_or(model);
    let resolved = resolve_download_ref(resolved, direct);
    let download = super::download::download_model_direct(
        &huggingface_hub_cache_dir(),
        &resolved,
        None,
        None,
        progress,
    )
    .await?;
    let paths = download.report["files"]
        .as_array()
        .context("download report missing files")?
        .iter()
        .filter_map(|file| file["path"].as_str().map(PathBuf::from))
        .collect();
    Ok(ModelDownload {
        path: download.primary_path,
        paths,
        details,
        transfer_stats: download.transfer_stats,
    })
}

fn resolve_download_ref(resolved: &str, direct: bool) -> String {
    skippy_model_resolver::parse_hf_resolve_url(resolved)
        .map(|(repo, revision, file)| {
            if !direct
                && let Some(model) =
                    skippy_model_hf::remote_catalog::matching_primary_for_huggingface(
                        &repo,
                        revision.as_deref(),
                        &file,
                    )
            {
                return model.exact_ref();
            }
            skippy_model_ref::format_model_ref(&repo, revision.as_deref(), Some(&file))
        })
        .unwrap_or_else(|| resolved.to_string())
}
