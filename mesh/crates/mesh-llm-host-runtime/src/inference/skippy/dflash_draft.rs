use std::path::PathBuf;

use anyhow::{Context, Result, ensure};

use super::resolver::DFlashPairing;

/// Fetches the DFlash draft the resolver chose by default into the Hugging
/// Face cache. A draft that cannot be fetched, verified or paired leaves the
/// model serving without speculation; it never fails the load.
pub(crate) async fn materialize_default_dflash_draft(resolved: &mut super::ResolvedSkippyConfig) {
    let Some(pairing) = resolved.speculative.default_dflash else {
        return;
    };
    match fetch_dflash_draft(pairing).await {
        Ok(path) => {
            tracing::info!(
                model = resolved.model_id,
                draft = %path.display(),
                max_draft_tokens = resolved.speculative.draft_max_tokens,
                "DFlash speculative decoding on by default"
            );
            resolved.speculative.set_default_dflash_draft(&path);
        }
        Err(error) => {
            tracing::warn!(
                model = resolved.model_id,
                draft = pairing.draft.repo,
                "DFlash draft unavailable, serving without speculation: {error:#}"
            );
            resolved.speculative.drop_default_dflash();
        }
    }
}

async fn fetch_dflash_draft(pairing: &'static DFlashPairing) -> Result<PathBuf> {
    let draft = &pairing.draft;
    let cached = draft.cache_path();
    let path = if cached.is_file() {
        cached
    } else {
        crate::models::catalog::download_hf_repo_file_with_progress_label(
            draft.repo,
            Some(draft.revision),
            draft.file,
            &format!("DFlash draft {}/{}", draft.repo, draft.file),
            true,
        )
        .await
        .with_context(|| format!("download {}/{}", draft.repo, draft.file))?
        .path
    };
    tokio::task::spawn_blocking(move || {
        let bytes = std::fs::metadata(&path)?.len();
        ensure!(
            bytes == draft.bytes,
            "{} is {bytes} bytes, expected {}",
            path.display(),
            draft.bytes
        );
        let sha256 = crate::models::artifact_transfer::file_sha256_hex(&path)?;
        ensure!(
            sha256 == draft.sha256,
            "{} has sha256 {sha256}, expected {}",
            path.display(),
            draft.sha256
        );
        pairing.check_draft(&path)?;
        Ok(path)
    })
    .await
    .context("join DFlash draft check")?
}
