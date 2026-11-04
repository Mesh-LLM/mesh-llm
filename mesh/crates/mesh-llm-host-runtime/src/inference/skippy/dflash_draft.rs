use std::collections::HashSet;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result};

use super::resolver::{
    DFlashDraft, DFlashPairingIndexPin, DefaultDFlash, check_split_dflash_draft,
    dflash_draft_cache_path, dflash_pairing_index_installed, install_dflash_pairing_index,
    load_dflash_pairing_index,
};

/// Brings in the DFlash pairing index this release pins, without waiting.
///
/// A cached index is read from the Hugging Face cache (no network, so it
/// works offline) and replaces the bootstrap table. A missing one downloads in
/// the background; this load chooses from the bootstrap table and the next
/// load from the index. Nothing happens until the index is published and
/// pinned.
pub(crate) fn refresh_dflash_pairing_index() {
    if dflash_pairing_index_installed() {
        return;
    }
    if let Some(pin) = DFlashPairingIndexPin::shipped() {
        refresh_cached_dflash_pairing_index(
            &pin.cache_path(),
            &format!("{}/{}@{}", pin.repo, pin.file, pin.revision),
            fetch_dflash_pairing_index(pin),
        );
    }
}

/// Installs the index cached at `path`, or fetches it in the background.
fn refresh_cached_dflash_pairing_index(
    path: &Path,
    key: &str,
    fetch: impl Future<Output = Result<PathBuf>> + Send + 'static,
) {
    if path.is_file() {
        let Err(error) = install_cached_dflash_pairing_index(path) else {
            return;
        };
        // A partial or corrupt download: remove it so a fresh one replaces it.
        tracing::warn!(
            index = %path.display(),
            "DFlash pairing index unusable, choosing from the bootstrap table: {error:#}"
        );
        discard_cached_artifact(path);
    }
    draft_cached_or_fetch(path, key, async move {
        let path = fetch.await?;
        install_cached_dflash_pairing_index(&path)?;
        Ok(path)
    });
}

fn install_cached_dflash_pairing_index(path: &Path) -> Result<()> {
    let index = load_dflash_pairing_index(path)?;
    tracing::info!(
        index = %path.display(),
        pairings = index.pairings.len(),
        "DFlash pairing index loaded"
    );
    install_dflash_pairing_index(index);
    Ok(())
}

async fn fetch_dflash_pairing_index(pin: DFlashPairingIndexPin) -> Result<PathBuf> {
    Ok(
        crate::models::catalog::download_hf_repo_file_with_progress_label(
            pin.repo,
            Some(pin.revision),
            pin.file,
            &format!("DFlash pairing index {}@{}", pin.repo, pin.revision),
            false,
        )
        .await
        .with_context(|| format!("download {}/{}@{}", pin.repo, pin.file, pin.revision))?
        .path,
    )
}

/// Whether the default DFlash draft is already in the Hugging Face cache.
///
/// A load never waits for the draft: a 1.38 GB download that stalled on a
/// Spark held a model down for 15 minutes. When the draft is missing it
/// downloads in the background, the caller loads and serves without
/// speculation (and plans no draft memory), and the next load of the model
/// picks the draft up.
pub(crate) fn default_dflash_draft_cached(chosen: &DefaultDFlash) -> bool {
    let chosen_for_fetch = chosen.clone();
    draft_cached_or_fetch(
        &dflash_draft_cache_path(&chosen.pairing.draft),
        &draft_fetch_key(&chosen.pairing.draft),
        async move {
            let draft = chosen_for_fetch.pairing.draft.clone();
            let path = fetch_dflash_draft(&draft).await?;
            tokio::task::spawn_blocking(move || {
                verify_dflash_draft(&draft, path, |path| chosen_for_fetch.check_draft(path))
            })
            .await
            .context("join DFlash draft check")?
        },
    )
}

/// Checks the cached default DFlash draft before it is attached. A draft that
/// fails its size, digest, pairing or vocabulary check leaves the model
/// serving without speculation; it never fails the load.
pub(crate) async fn materialize_default_dflash_draft(resolved: &mut super::ResolvedSkippyConfig) {
    let Some(chosen) = resolved.speculative.default_dflash.clone() else {
        return;
    };
    let cache_path = dflash_draft_cache_path(&chosen.pairing.draft);
    let repo = chosen.pairing.draft.repo.clone();
    let target = resolved.model_path.clone();
    let path = cache_path.clone();
    let checked = tokio::task::spawn_blocking(move || {
        let path =
            verify_dflash_draft(&chosen.pairing.draft, path, |path| chosen.check_draft(path))?;
        chosen.check_draft_vocabulary(&path, &target)?;
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
                draft = repo,
                "DFlash draft unavailable, serving without speculation: {error:#}"
            );
            if error.downcast_ref::<CorruptDraft>().is_some() {
                discard_cached_artifact(&cache_path);
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

/// Removes a corrupt cached draft or pairing index, the snapshot entry and the
/// blob it points to, so the next load downloads it again instead of finding
/// it cached. The downloader returns an existing snapshot as is, so the blob
/// has to go too.
fn discard_cached_artifact(path: &Path) {
    let blob = std::fs::canonicalize(path).ok();
    for path in std::iter::once(path.to_path_buf()).chain(blob) {
        match std::fs::remove_file(&path) {
            Ok(()) => tracing::info!(
                path = %path.display(),
                "removed a corrupt DFlash artifact; the next load downloads it again"
            ),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => tracing::warn!(
                path = %path.display(),
                "could not remove a corrupt DFlash artifact: {error}"
            ),
        }
    }
}

/// The pinned source of a default DFlash draft, for a split plan to carry to
/// the final stage.
pub(crate) fn split_dflash_draft_source(
    chosen: &DefaultDFlash,
) -> skippy_serving::split_dflash::SplitDFlashDraftSource {
    let draft = &chosen.pairing.draft;
    skippy_serving::split_dflash::SplitDFlashDraftSource {
        repo: draft.repo.clone(),
        revision: draft.revision.clone(),
        file: draft.file.clone(),
        sha256: draft.sha256.clone(),
        bytes: draft.bytes,
    }
}

/// Points a split stage's default draft at this node's own copy.
///
/// The planner's draft path is on the planner's node. A plan that names a
/// pinned draft (from the package, the pairing index or the bootstrap table)
/// is fetched here by that repo, revision and file into this node's Hugging
/// Face cache, and must match its SHA-256 and size. As on one node, the load
/// never waits for the draft: a missing draft downloads in the background,
/// this load serves without speculation, and the next load of the model picks
/// the draft up. A draft that fails its checks also serves without
/// speculation. A plan without a source (a configured draft, or an older
/// planner) loads its path as sent. A source that does not pin an immutable
/// commit, digest and size serves without speculation: the planner's path
/// names a file on the planner's node, which this node cannot check.
pub(crate) async fn materialize_split_dflash_draft(
    plan: &mut skippy_serving::split_dflash::SplitDFlashStage,
) {
    let resolved = plan.clone();
    let resolved = tokio::task::spawn_blocking(move || {
        let mut plan = resolved;
        resolve_split_dflash_draft(
            &mut plan,
            split_draft_cached_or_fetch,
            |draft, layers, path| {
                verify_dflash_draft(draft, path, |path| check_split_dflash_draft(path, layers))
            },
        );
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

fn split_draft_cached_or_fetch(draft: &DFlashDraft, target_layers: &[u32]) -> bool {
    let draft_for_fetch = draft.clone();
    let layers = target_layers.to_vec();
    draft_cached_or_fetch(
        &dflash_draft_cache_path(draft),
        &draft_fetch_key(draft),
        async move {
            let path = fetch_dflash_draft(&draft_for_fetch).await?;
            tokio::task::spawn_blocking(move || {
                verify_dflash_draft(&draft_for_fetch, path, |path| {
                    check_split_dflash_draft(path, &layers)
                })
            })
            .await
            .context("join DFlash draft check")?
        },
    )
}

fn resolve_split_dflash_draft(
    plan: &mut skippy_serving::split_dflash::SplitDFlashStage,
    cached_or_fetch: impl FnOnce(&DFlashDraft, &[u32]) -> bool,
    verify: impl FnOnce(&DFlashDraft, &[u32], PathBuf) -> Result<PathBuf>,
) {
    let Some(draft) = split_dflash_pinned_draft(plan) else {
        return;
    };
    if !cached_or_fetch(&draft, &plan.target_layers) {
        plan.draft_model_path = None;
        return;
    }
    let cache_path = dflash_draft_cache_path(&draft);
    plan.draft_model_path = match verify(&draft, &plan.target_layers, cache_path.clone()) {
        Ok(path) => {
            tracing::info!(draft = %path.display(), "split DFlash draft verified on this node");
            Some(path)
        }
        Err(error) => {
            tracing::warn!(
                draft = draft.repo,
                "split DFlash draft unavailable, serving without speculation: {error:#}"
            );
            if error.downcast_ref::<CorruptDraft>().is_some() {
                discard_cached_artifact(&cache_path);
            }
            None
        }
    };
}

/// The pinned draft a split plan's source names, or `None` when the plan has
/// no source. A source that does not pin an immutable commit, digest and size
/// is refused, and the stage serves without speculation.
fn split_dflash_pinned_draft(
    plan: &mut skippy_serving::split_dflash::SplitDFlashStage,
) -> Option<DFlashDraft> {
    let source = plan.draft_source.as_ref()?;
    let draft = DFlashDraft {
        repo: source.repo.clone(),
        revision: source.revision.clone(),
        file: source.file.clone(),
        sha256: source.sha256.clone(),
        bytes: source.bytes,
    };
    if !draft.is_pinned() {
        tracing::warn!(
            draft = %format!("{}/{}@{}", draft.repo, draft.file, draft.revision),
            "split DFlash draft source is not pinned to a commit, digest and size; serving without speculation"
        );
        plan.draft_model_path = None;
        return None;
    }
    Some(draft)
}

/// Names one draft download: two drafts in one repo fetch independently.
fn draft_fetch_key(draft: &DFlashDraft) -> String {
    format!("{}/{}@{}", draft.repo, draft.file, draft.revision)
}

/// `true` when `path` exists. Otherwise starts `fetch` in the background,
/// once per `key` at a time, and returns `false` without waiting for it.
fn draft_cached_or_fetch(
    path: &Path,
    key: &str,
    fetch: impl Future<Output = Result<PathBuf>> + Send + 'static,
) -> bool {
    if path.is_file() {
        return true;
    }
    static IN_FLIGHT: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    let in_flight = IN_FLIGHT.get_or_init(Default::default);
    let Ok(runtime) = tokio::runtime::Handle::try_current() else {
        return false;
    };
    let key = key.to_string();
    if !in_flight
        .lock()
        .is_ok_and(|mut keys| keys.insert(key.clone()))
    {
        return false;
    }
    tracing::info!(
        artifact = key,
        "DFlash artifact is not cached; downloading it in the background, and this load goes on without it"
    );
    runtime.spawn(async move {
        match fetch.await {
            Ok(path) => tracing::info!(
                artifact = %path.display(),
                "DFlash artifact downloaded; the next load uses it"
            ),
            Err(error) => {
                tracing::warn!(artifact = key, "DFlash artifact download failed: {error:#}")
            }
        }
        if let Ok(mut keys) = in_flight.lock() {
            keys.remove(&key);
        }
    });
    false
}

async fn fetch_dflash_draft(draft: &DFlashDraft) -> Result<PathBuf> {
    Ok(
        crate::models::catalog::download_hf_repo_file_with_progress_label(
            &draft.repo,
            Some(&draft.revision),
            &draft.file,
            &format!("DFlash draft {}/{}", draft.repo, draft.file),
            false,
        )
        .await
        .with_context(|| format!("download {}/{}", draft.repo, draft.file))?
        .path,
    )
}

/// Accepts `path` as `draft` only if its size and SHA-256 match
/// and `check` accepts its header.
fn verify_dflash_draft(
    draft: &DFlashDraft,
    path: PathBuf,
    check: impl FnOnce(&Path) -> Result<()>,
) -> Result<PathBuf> {
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
    if !sha256.eq_ignore_ascii_case(&draft.sha256) {
        return Err(CorruptDraft(format!(
            "{} has sha256 {sha256}, expected {}",
            path.display(),
            draft.sha256
        ))
        .into());
    }
    check(&path)?;
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

    fn pinned_source() -> skippy_serving::split_dflash::SplitDFlashDraftSource {
        skippy_serving::split_dflash::SplitDFlashDraftSource {
            repo: "meshllm/package-draft".into(),
            revision: "a3d4fb1eff6f268cbbdf808042f6902cc17f5458".into(),
            file: "draft-Q8_0.gguf".into(),
            sha256: "b9be52167d0b4f5cbaffba6658bcb20dd814ad4e5846532353c165ccf97e47a1".into(),
            bytes: 1_849_481_472,
        }
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
    fn a_final_stage_loads_its_own_copy_of_any_pinned_draft() {
        // Not in any catalog: a package-declared draft travels the same way.
        let source = pinned_source();
        let mut plan = final_stage_plan(Some(source.clone()));
        let verified = Arc::new(AtomicUsize::new(0));
        let counter = verified.clone();

        resolve_split_dflash_draft(
            &mut plan,
            |draft, layers| {
                assert_eq!(draft.repo, source.repo);
                assert_eq!(layers, [2, 62]);
                true
            },
            |draft, layers, path| {
                assert_eq!(draft.bytes, 1_849_481_472);
                assert_eq!(draft.sha256, source.sha256);
                assert_eq!(layers, [2, 62]);
                counter.fetch_add(1, Ordering::SeqCst);
                Ok(path)
            },
        );

        assert_eq!(verified.load(Ordering::SeqCst), 1);
        let path = plan.draft_model_path.expect("this node's copy");
        assert!(path.ends_with(format!("snapshots/{}/{}", source.revision, source.file)));
    }

    #[test]
    fn a_final_stage_without_the_draft_yet_serves_without_speculation() {
        let mut plan = final_stage_plan(Some(pinned_source()));
        resolve_split_dflash_draft(&mut plan, |_, _| false, |_, _, _| unreachable!());
        assert_eq!(plan.draft_model_path, None);

        let mut plan = final_stage_plan(Some(pinned_source()));
        resolve_split_dflash_draft(
            &mut plan,
            |_, _| true,
            |_, _, _| anyhow::bail!("sha256 mismatch"),
        );
        assert_eq!(plan.draft_model_path, None);
    }

    #[test]
    fn a_plan_without_a_source_loads_the_path_as_sent_and_an_unpinned_one_nothing() {
        let sent = final_stage_plan(None).draft_model_path;
        let mut plan = final_stage_plan(None);
        resolve_split_dflash_draft(&mut plan, |_, _| unreachable!(), |_, _, _| unreachable!());
        assert_eq!(plan.draft_model_path, sent);

        for unpin in [
            |source: &mut skippy_serving::split_dflash::SplitDFlashDraftSource| {
                source.revision = "main".into()
            },
            |source: &mut skippy_serving::split_dflash::SplitDFlashDraftSource| {
                source.sha256 = "00".into()
            },
            |source: &mut skippy_serving::split_dflash::SplitDFlashDraftSource| {
                source.file = "../escape.gguf".into()
            },
            |source: &mut skippy_serving::split_dflash::SplitDFlashDraftSource| source.bytes = 0,
            |source: &mut skippy_serving::split_dflash::SplitDFlashDraftSource| {
                source.repo = "owner/..".into()
            },
            |source: &mut skippy_serving::split_dflash::SplitDFlashDraftSource| {
                source.file = "C:\\draft.gguf".into()
            },
        ] {
            let mut source = pinned_source();
            unpin(&mut source);
            let mut plan = final_stage_plan(Some(source));
            resolve_split_dflash_draft(&mut plan, |_, _| unreachable!(), |_, _, _| unreachable!());
            assert_eq!(plan.draft_model_path, None);
        }
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

        discard_cached_artifact(&snapshot);

        assert!(std::fs::symlink_metadata(&snapshot).is_err());
        #[cfg(unix)]
        assert!(!blob.exists());
    }

    fn draft_with(bytes: &[u8]) -> (tempfile::NamedTempFile, DFlashDraft) {
        use std::io::Write;
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(bytes).unwrap();
        let sha256 = crate::models::artifact_transfer::file_sha256_hex(file.path()).unwrap();
        let draft = DFlashDraft {
            repo: "meshllm/draft".into(),
            revision: "a3d4fb1eff6f268cbbdf808042f6902cc17f5458".into(),
            file: "draft.gguf".into(),
            sha256,
            bytes: bytes.len() as u64,
        };
        (file, draft)
    }

    #[test]
    fn a_draft_with_a_bad_sha256_or_size_is_rejected() {
        let (file, draft) = draft_with(b"GGUF draft bytes");
        let path = file.path().to_path_buf();
        assert_eq!(
            verify_dflash_draft(&draft, path.clone(), |_| Ok(())).unwrap(),
            path
        );

        let mut bad_sha = draft.clone();
        bad_sha.sha256 = "0".repeat(64);
        let error = verify_dflash_draft(&bad_sha, path.clone(), |_| Ok(()))
            .unwrap_err()
            .to_string();
        assert!(error.contains("sha256"), "{error}");

        let mut bad_size = draft.clone();
        bad_size.bytes += 1;
        let error = verify_dflash_draft(&bad_size, path.clone(), |_| Ok(()))
            .unwrap_err()
            .to_string();
        assert!(error.contains("bytes"), "{error}");

        // The header check runs last and can refuse a draft that hashes right.
        assert!(
            verify_dflash_draft(&draft, path, |_| anyhow::bail!("not a dflash draft")).is_err()
        );
    }

    #[test]
    fn a_cached_pairing_index_is_installed_offline() {
        // The checked-in index is the bootstrap table, so installing it leaves
        // other tests' defaults as they were.
        let checked_in = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../skippy/evals/dflash-pairings/index.json");
        let cache = tempfile::tempdir().unwrap();
        let bad = cache.path().join("bad.json");
        std::fs::write(&bad, "{\"schema_version\": 1, \"pairings\": [{}]}").unwrap();
        // An unusable cached index is removed, so a later load fetches it again.
        refresh_cached_dflash_pairing_index(&bad, "test/bad-index", async {
            unreachable!("no runtime, so no fetch")
        });
        assert!(!bad.exists());
        let cached = cache.path().join("index.json");
        std::fs::copy(checked_in, &cached).unwrap();
        // No runtime and a fetch that must not run: the cache alone serves.
        refresh_cached_dflash_pairing_index(&cached, "test/index", async {
            unreachable!("a cached index is never fetched")
        });
        assert!(dflash_pairing_index_installed());
    }

    #[test]
    fn drafts_in_one_repo_fetch_independently() {
        let (_file, draft) = draft_with(b"draft");
        let other = DFlashDraft {
            file: "other.gguf".to_string(),
            ..draft.clone()
        };
        let newer = DFlashDraft {
            revision: "f".repeat(40),
            ..draft.clone()
        };
        assert_ne!(draft_fetch_key(&draft), draft_fetch_key(&other));
        assert_ne!(draft_fetch_key(&draft), draft_fetch_key(&newer));
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
