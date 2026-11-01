use std::collections::HashSet;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result};

use super::resolver::DFlashPairing;

/// Whether the default DFlash draft is already in the Hugging Face cache.
///
/// A load never waits for the draft: a 1.38 GB download that stalled on a
/// Spark held a model down for 15 minutes. When the draft is missing it
/// downloads in the background, the caller loads and serves without
/// speculation (and plans no draft memory), and the next load of the model
/// picks the draft up.
pub(crate) fn default_dflash_draft_cached(pairing: &'static DFlashPairing) -> bool {
    draft_cached_or_fetch(
        &pairing.draft.cache_path(),
        pairing.draft.repo,
        fetch_dflash_draft(pairing),
    )
}

/// Checks the cached default DFlash draft before it is attached. A draft that
/// fails its size, digest, pairing or vocabulary check leaves the model
/// serving without speculation; it never fails the load.
pub(crate) async fn materialize_default_dflash_draft(resolved: &mut super::ResolvedSkippyConfig) {
    let Some(pairing) = resolved.speculative.default_dflash else {
        return;
    };
    let path = pairing.draft.cache_path();
    let target = resolved.model_path.clone();
    let checked = tokio::task::spawn_blocking(move || {
        let path = verify_dflash_draft(pairing, path)?;
        pairing.check_draft_vocabulary(&path, &target)?;
        Ok(path)
    })
    .await
    .context("join DFlash draft check")
    .and_then(|checked| checked);
    match checked {
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
            if error.downcast_ref::<CorruptDraft>().is_some() {
                discard_cached_draft(&pairing.draft.cache_path());
            }
            resolved.speculative.drop_default_dflash();
        }
    }
}

/// A cached draft whose size or digest is not the pinned one: a partial or
/// corrupted download, which a fresh download repairs.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
struct CorruptDraft(String);

/// Removes a corrupt cached draft, the snapshot entry and the blob it points
/// to, so the next load downloads it again instead of finding it cached. The
/// downloader returns an existing snapshot as is, so the blob has to go too.
fn discard_cached_draft(path: &Path) {
    let blob = std::fs::canonicalize(path).ok();
    for path in std::iter::once(path.to_path_buf()).chain(blob) {
        match std::fs::remove_file(&path) {
            Ok(()) => tracing::info!(
                path = %path.display(),
                "removed a corrupt DFlash draft; the next load downloads it again"
            ),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => tracing::warn!(
                path = %path.display(),
                "could not remove a corrupt DFlash draft: {error}"
            ),
        }
    }
}

/// `true` when `path` exists. Otherwise starts `fetch` in the background,
/// once per `key` at a time, and returns `false` without waiting for it.
fn draft_cached_or_fetch(
    path: &Path,
    key: &'static str,
    fetch: impl Future<Output = Result<PathBuf>> + Send + 'static,
) -> bool {
    if path.is_file() {
        return true;
    }
    static IN_FLIGHT: OnceLock<Mutex<HashSet<&'static str>>> = OnceLock::new();
    let in_flight = IN_FLIGHT.get_or_init(Default::default);
    let Ok(runtime) = tokio::runtime::Handle::try_current() else {
        return false;
    };
    if !in_flight.lock().is_ok_and(|mut keys| keys.insert(key)) {
        return false;
    }
    tracing::info!(
        draft = key,
        "DFlash draft is not cached; downloading it in the background and serving this load without speculation"
    );
    runtime.spawn(async move {
        match fetch.await {
            Ok(path) => tracing::info!(
                draft = %path.display(),
                "DFlash draft downloaded; the next load of its model speculates with it"
            ),
            Err(error) => tracing::warn!(draft = key, "DFlash draft download failed: {error:#}"),
        }
        if let Ok(mut keys) = in_flight.lock() {
            keys.remove(key);
        }
    });
    false
}

async fn fetch_dflash_draft(pairing: &'static DFlashPairing) -> Result<PathBuf> {
    let draft = &pairing.draft;
    let path = crate::models::catalog::download_hf_repo_file_with_progress_label(
        draft.repo,
        Some(draft.revision),
        draft.file,
        &format!("DFlash draft {}/{}", draft.repo, draft.file),
        false,
    )
    .await
    .with_context(|| format!("download {}/{}", draft.repo, draft.file))?
    .path;
    tokio::task::spawn_blocking(move || verify_dflash_draft(pairing, path))
        .await
        .context("join DFlash draft check")?
}

fn verify_dflash_draft(pairing: &'static DFlashPairing, path: PathBuf) -> Result<PathBuf> {
    let draft = &pairing.draft;
    let bytes = std::fs::metadata(&path)
        .with_context(|| format!("read {}", path.display()))?
        .len();
    if bytes != draft.bytes {
        return Err(CorruptDraft(format!(
            "{} is {bytes} bytes, expected {}",
            path.display(),
            draft.bytes
        ))
        .into());
    }
    let sha256 = crate::models::artifact_transfer::file_sha256_hex(&path)?;
    if sha256 != draft.sha256 {
        return Err(CorruptDraft(format!(
            "{} has sha256 {sha256}, expected {}",
            path.display(),
            draft.sha256
        ))
        .into());
    }
    pairing.check_draft(&path)?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    #[tokio::test]
    async fn a_missing_draft_downloads_in_the_background_without_delaying_the_load() {
        let missing = Path::new("/nonexistent/dflash-draft-test.gguf");
        let started = Arc::new(AtomicUsize::new(0));
        let stalled = |started: Arc<AtomicUsize>| async move {
            started.fetch_add(1, Ordering::SeqCst);
            std::future::pending::<Result<PathBuf>>().await
        };

        let begun = Instant::now();
        assert!(!draft_cached_or_fetch(
            missing,
            "test/stalled-draft",
            stalled(started.clone())
        ));
        // A second load while the first download is still running neither
        // waits nor starts another download.
        assert!(!draft_cached_or_fetch(
            missing,
            "test/stalled-draft",
            stalled(started.clone())
        ));
        assert!(begun.elapsed() < Duration::from_secs(1));

        tokio::task::yield_now().await;
        tokio::time::timeout(Duration::from_secs(5), async {
            while started.load(Ordering::SeqCst) == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("the download starts in the background");
        assert_eq!(started.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_corrupt_draft_is_removed_with_its_blob() {
        let cache = tempfile::tempdir().unwrap();
        let blob = cache.path().join("blobs").join("deadbeef");
        std::fs::create_dir_all(blob.parent().unwrap()).unwrap();
        std::fs::write(&blob, b"partial").unwrap();
        let snapshot = cache
            .path()
            .join("snapshots")
            .join("rev")
            .join("draft.gguf");
        std::fs::create_dir_all(snapshot.parent().unwrap()).unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(&blob, &snapshot).unwrap();
        #[cfg(not(unix))]
        std::fs::copy(&blob, &snapshot).unwrap();

        discard_cached_draft(&snapshot);

        assert!(std::fs::symlink_metadata(&snapshot).is_err());
        #[cfg(unix)]
        assert!(!blob.exists());
    }

    #[tokio::test]
    async fn a_cached_draft_is_used_without_fetching() {
        let cached = tempfile::NamedTempFile::new().unwrap();
        let fetched = Arc::new(AtomicUsize::new(0));
        let counter = fetched.clone();
        assert!(draft_cached_or_fetch(
            cached.path(),
            "test/cached-draft",
            async move {
                counter.fetch_add(1, Ordering::SeqCst);
                Ok(PathBuf::new())
            }
        ));
        tokio::task::yield_now().await;
        assert_eq!(fetched.load(Ordering::SeqCst), 0);
    }
}
