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
/// fails its size, digest or pairing check leaves the model serving without
/// speculation; it never fails the load.
pub(crate) async fn materialize_default_dflash_draft(resolved: &mut super::ResolvedSkippyConfig) {
    let Some(pairing) = resolved.speculative.default_dflash else {
        return;
    };
    let path = pairing.draft.cache_path();
    let checked = tokio::task::spawn_blocking(move || verify_dflash_draft(pairing, path))
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

/// The pinned source of a default DFlash draft, for a split plan to carry to
/// the final stage.
pub(crate) fn split_dflash_draft_source(
    pairing: &DFlashPairing,
) -> skippy_serving::split_dflash::SplitDFlashDraftSource {
    skippy_serving::split_dflash::SplitDFlashDraftSource {
        repo: pairing.draft.repo.to_string(),
        revision: pairing.draft.revision.to_string(),
        file: pairing.draft.file.to_string(),
        sha256: pairing.draft.sha256.to_string(),
    }
}

/// Points a split stage's catalog default draft at this node's own copy.
///
/// The planner's draft path is on the planner's node. A plan that names a
/// catalog draft is resolved here against the same pinned repo, revision,
/// file and SHA-256 the single-node default uses, in this node's Hugging Face
/// cache. As on one node, the load never waits for the draft: a missing draft
/// downloads in the background, this load serves without speculation, and the
/// next load of the model picks the draft up. A draft that fails its checks
/// also serves without speculation. A plan without a source (a configured
/// draft, or an older planner) loads its path as sent, and so does a source
/// this node's catalog does not pin.
pub(crate) async fn materialize_split_dflash_draft(
    plan: &mut skippy_serving::split_dflash::SplitDFlashStage,
) {
    let resolved = plan.clone();
    let resolved = tokio::task::spawn_blocking(move || {
        let mut plan = resolved;
        resolve_split_dflash_draft(&mut plan, default_dflash_draft_cached, verify_dflash_draft);
        plan
    })
    .await;
    match resolved {
        Ok(resolved) => *plan = resolved,
        Err(error) => {
            tracing::warn!("split DFlash draft check failed, serving without speculation: {error}");
            plan.draft_model_path = None;
        }
    }
}

fn resolve_split_dflash_draft(
    plan: &mut skippy_serving::split_dflash::SplitDFlashStage,
    cached_or_fetch: impl FnOnce(&'static DFlashPairing) -> bool,
    verify: impl FnOnce(&'static DFlashPairing, PathBuf) -> Result<PathBuf>,
) {
    let Some(pairing) = split_dflash_catalog_pairing(plan) else {
        return;
    };
    if !cached_or_fetch(pairing) {
        plan.draft_model_path = None;
        return;
    }
    plan.draft_model_path = match verify(pairing, pairing.draft.cache_path()) {
        Ok(path) => {
            tracing::info!(draft = %path.display(), "split DFlash draft verified on this node");
            Some(path)
        }
        Err(error) => {
            tracing::warn!(
                draft = pairing.draft.repo,
                "split DFlash draft unavailable, serving without speculation: {error:#}"
            );
            None
        }
    };
}

/// The catalog pairing a split plan's draft source names, or `None` when the
/// plan has no source or this node's catalog does not pin it.
fn split_dflash_catalog_pairing(
    plan: &skippy_serving::split_dflash::SplitDFlashStage,
) -> Option<&'static DFlashPairing> {
    let source = plan.draft_source.as_ref()?;
    let pairing =
        DFlashPairing::for_draft(&source.repo, &source.revision, &source.file, &source.sha256);
    if pairing.is_none() {
        tracing::warn!(
            draft = %format!("{}/{}@{}", source.repo, source.file, source.revision),
            "split DFlash draft is not in this node's catalog; loading the planner's path"
        );
    }
    pairing
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

    fn qwen36_27b() -> &'static DFlashPairing {
        DFlashPairing::for_draft(
            "AtomicChat/Qwen3.6-27B-DFlash-GGUF",
            "a3d4fb1eff6f268cbbdf808042f6902cc17f5458",
            "Qwen3.6-27B-DFlash.Q8_0.gguf",
            "b9be52167d0b4f5cbaffba6658bcb20dd814ad4e5846532353c165ccf97e47a1",
        )
        .expect("catalog pairing")
    }

    fn final_stage_plan(
        source: Option<skippy_serving::split_dflash::SplitDFlashDraftSource>,
    ) -> skippy_serving::split_dflash::SplitDFlashStage {
        skippy_serving::split_dflash::SplitDFlashStage {
            target_layers: vec![2, 62],
            draft_model_path: Some("/home/planner/.cache/draft.gguf".into()),
            draft_source: source,
            ..Default::default()
        }
    }

    #[test]
    fn a_final_stage_loads_its_own_copy_of_a_catalog_draft() {
        let pairing = qwen36_27b();
        let mut plan = final_stage_plan(Some(split_dflash_draft_source(pairing)));
        let verified = Arc::new(AtomicUsize::new(0));
        let counter = verified.clone();

        resolve_split_dflash_draft(
            &mut plan,
            |found| std::ptr::eq(found, pairing),
            |found, path| {
                assert!(std::ptr::eq(found, pairing));
                counter.fetch_add(1, Ordering::SeqCst);
                Ok(path)
            },
        );

        assert_eq!(verified.load(Ordering::SeqCst), 1);
        assert_eq!(plan.draft_model_path, Some(pairing.draft.cache_path()));
    }

    #[test]
    fn a_final_stage_without_the_catalog_draft_yet_serves_without_speculation() {
        let mut plan = final_stage_plan(Some(split_dflash_draft_source(qwen36_27b())));
        resolve_split_dflash_draft(&mut plan, |_| false, |_, _| unreachable!());
        assert_eq!(plan.draft_model_path, None);

        let mut plan = final_stage_plan(Some(split_dflash_draft_source(qwen36_27b())));
        resolve_split_dflash_draft(&mut plan, |_| true, |_, _| anyhow::bail!("sha256 mismatch"));
        assert_eq!(plan.draft_model_path, None);
    }

    #[test]
    fn a_plan_without_a_catalog_source_loads_the_path_as_sent() {
        let sent = final_stage_plan(None).draft_model_path;
        let mut plan = final_stage_plan(None);
        resolve_split_dflash_draft(&mut plan, |_| unreachable!(), |_, _| unreachable!());
        assert_eq!(plan.draft_model_path, sent);

        let mut unknown = split_dflash_draft_source(qwen36_27b());
        unknown.revision = "main".into();
        let mut plan = final_stage_plan(Some(unknown));
        resolve_split_dflash_draft(&mut plan, |_| unreachable!(), |_, _| unreachable!());
        assert_eq!(plan.draft_model_path, sent);
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
