//! Ordered startup discovery, separate from acquisition and presentation.
use crate::{details::canonicalize_model_ref_input, remote_catalog, store::local::find_model_path};
use anyhow::{Context, Result};
use skippy_model_resolver::startup::StartupModelSource;
use std::path::Path;

pub async fn resolve_startup_model_source(input: &Path) -> Result<StartupModelSource> {
    let raw = input.to_string_lossy();
    // Preserve package passthrough without probing the filesystem.
    if let Some(source) = StartupModelSource::explicit(input, || input.exists()) {
        return Ok(source);
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
        return Ok(StartupModelSource::missing(
            &raw,
            canonicalize_model_ref_input(&raw).await.ok(),
        ));
    }
    let installed = find_model_path(&raw);
    if installed.exists() {
        return Ok(StartupModelSource::Installed(installed));
    }
    Ok(StartupModelSource::missing(&raw, None))
}
