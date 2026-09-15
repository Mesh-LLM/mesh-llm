//! Content identity for a served GGUF: SHA-256 over the file BYTES actually
//! loaded, as opposed to `model_identity::identity_hash_for`'s hash of a
//! *reference string* (repo/revision/file for a Hugging Face source, or
//! nothing at all for a local path -- see `ServedModelIdentity::identity_hash`).
//! A served-model NAME is not proof of served BYTES: a proxy or a
//! mis-deployed node can serve a different file under the same name. This
//! digest makes the served bytes a fact any peer can check against the file
//! on disk. It does not stop a host from reporting a digest for a file it did
//! not load: this is a self-reported value, so it surfaces an honest node's
//! stale or swapped file, not a host that lies about what it loaded.
//!
//! **Known limitation, not closed by this module (TOCTOU):** the hash here is
//! read via an independent `File::open` of `path`, entirely separate from
//! whatever open the model runtime itself performs to actually serve the
//! file. Verified against `skippy_runtime::native::StageModel::open`: it
//! marshals only a C-string path across the FFI boundary to
//! `skippy_ffi::skippy_model_open` and gets back an opaque model handle --
//! the native (llama.cpp) loader owns that read and never returns the loaded
//! bytes, a file descriptor, or a streaming-hash hook to the Rust side. There
//! is no API to derive this digest from the same load. A file replaced
//! between the two opens can make this digest describe different bytes than
//! the ones served. [`file_fingerprint`] exists so a caller can narrow (not
//! eliminate) that window by re-checking the file's (size, mtime) after the
//! model has finished loading and discarding the digest on any mismatch (see
//! `runtime/local.rs::start_runtime_local_model`); a replacement that
//! preserves both size AND mtime exactly is the same documented blind spot
//! the cache key below already has, and remains undetected.
//!
//! Cached by (path, size, mtime): hashing an 8GB GGUF costs real wall-clock
//! time, and must happen once per file, never once per request. Concurrent
//! callers racing for the same (path, size, mtime) single-flight onto one
//! computation instead of each independently streaming the file -- the
//! cache's mutex is only ever held to read or install an entry, never across
//! the file read itself. The in-memory map only ever holds the most recent
//! entry per path (older states for a repeatedly-rewritten file are evicted
//! as the new one is installed), so it cannot grow without bound.
//!
//! The result is also persisted to a small JSON record under
//! `mesh_llm_cache_dir()/weights-digest/`, keyed by a hash of the path and
//! carrying the same (size, mtime) recipe, so a later process restart that
//! finds the file unchanged loads the digest from that record instead of
//! re-hashing the file -- the in-memory cache above is process-local and does
//! not survive a restart on its own. Persistence is best-effort: a failure to
//! read or write the record falls back to re-hashing, never to a fabricated
//! digest.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Instant, UNIX_EPOCH};

/// (path, size in bytes, mtime as nanos since the epoch) -- the same recipe
/// `model-hf`'s local-GGUF synthetic ref already uses to detect "this exact
/// file state," just applied to a cache key instead of a name.
type CacheKey = (PathBuf, u64, u128);

/// The outcome of one computation, shared by every caller racing for the
/// same `CacheKey`: the first caller in becomes the computer and publishes
/// its result here; every other caller waits on the condvar for it instead
/// of starting a redundant computation of its own.
struct PendingDigest {
    result: Mutex<Option<Option<String>>>,
    ready: Condvar,
}

enum CacheEntry {
    Ready(String),
    Pending(Arc<PendingDigest>),
}

/// A cache that single-flights concurrent misses for the same key: the
/// map's mutex is only ever held long enough to read or install an entry,
/// never across the (potentially multi-GB) computation itself.
struct SingleFlightCache {
    state: Mutex<SingleFlightState>,
}

struct SingleFlightState {
    entries: HashMap<CacheKey, (u64, CacheEntry)>,
    latest_generation_by_path: HashMap<PathBuf, u64>,
    next_generation: u64,
}

impl SingleFlightCache {
    fn new() -> Self {
        Self {
            state: Mutex::new(SingleFlightState {
                entries: HashMap::new(),
                latest_generation_by_path: HashMap::new(),
                next_generation: 0,
            }),
        }
    }

    /// Returns the cached value for `key`, joining an in-progress
    /// computation if one is already running, or running `compute` itself
    /// and publishing the result if it is the first caller for `key`.
    fn get_or_compute(
        &self,
        key: CacheKey,
        path: &Path,
        compute: impl FnOnce() -> Option<String>,
    ) -> Option<String> {
        enum Role {
            Cached(String),
            Wait(Arc<PendingDigest>),
            Compute(Arc<PendingDigest>),
        }

        let role = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            match state.entries.get(&key) {
                Some((_, CacheEntry::Ready(digest))) => Role::Cached(digest.clone()),
                Some((_, CacheEntry::Pending(pending))) => Role::Wait(pending.clone()),
                None => {
                    let pending = Arc::new(PendingDigest {
                        result: Mutex::new(None),
                        ready: Condvar::new(),
                    });
                    state.next_generation = state.next_generation.wrapping_add(1);
                    let generation = state.next_generation;
                    state
                        .latest_generation_by_path
                        .insert(key.0.clone(), generation);
                    state.entries.insert(
                        key.clone(),
                        (generation, CacheEntry::Pending(pending.clone())),
                    );
                    Role::Compute(pending)
                }
            }
        };

        match role {
            Role::Cached(digest) => {
                tracing::debug!(
                    path = %path.display(),
                    "weights_digest cache hit -- not re-hashed"
                );
                Some(digest)
            }
            Role::Wait(pending) => {
                tracing::debug!(
                    path = %path.display(),
                    "weights_digest already being computed by another caller -- waiting for it"
                );
                let mut result = pending
                    .result
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                while result.is_none() {
                    result = pending
                        .ready
                        .wait(result)
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                }
                result
                    .clone()
                    .expect("checked Some in the loop condition above")
            }
            Role::Compute(pending) => {
                // The map's mutex is not held across this call: `compute`
                // can be a multi-GB file read, and it must not block every
                // other caller checking a different (or even the same) key.
                let digest = compute();
                {
                    let mut state = self
                        .state
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    let generation = state.entries.remove(&key).map(|entry| entry.0);
                    let is_latest = generation.is_some_and(|generation| {
                        state.latest_generation_by_path.get(&key.0).copied() == Some(generation)
                    });
                    if is_latest {
                        // Keep older computations alive so their waiters are
                        // still single-flighted, but never let one that
                        // finishes late evict or replace this newer state.
                        state.entries.retain(|existing_key, (_, entry)| {
                            existing_key.0 != key.0 || matches!(entry, CacheEntry::Pending(_))
                        });
                        // Unreadable (`None`): never cache a fabricated
                        // absence, so a later retry can succeed.
                        if let (Some(generation), Some(computed)) = (generation, &digest) {
                            state
                                .entries
                                .insert(key, (generation, CacheEntry::Ready(computed.clone())));
                        }
                    }
                }
                *pending
                    .result
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(digest.clone());
                pending.ready.notify_all();
                digest
            }
        }
    }
}

fn cache() -> &'static SingleFlightCache {
    static CACHE: OnceLock<SingleFlightCache> = OnceLock::new();
    CACHE.get_or_init(SingleFlightCache::new)
}

/// (size in bytes, mtime as nanos since the epoch) for `path` -- the same
/// recipe the digest cache keys on, exposed so a caller can independently
/// confirm the file's state hasn't changed between reading this digest and
/// finishing whatever it loaded the file for (see the module TOCTOU note).
/// `None` when the file cannot be stat'd.
pub(crate) fn file_fingerprint(path: &Path) -> Option<(u64, u128)> {
    let metadata = std::fs::metadata(path).ok()?;
    let size = metadata.len();
    let mtime_nanos = metadata
        .modified()
        .ok()?
        .duration_since(UNIX_EPOCH)
        .ok()?
        .as_nanos();
    Some((size, mtime_nanos))
}

/// SHA-256 of `path`'s bytes, lowercase hex. `None` when the file cannot be
/// stat'd or read -- an honest absent fact, never a fabricated value (never a
/// `0`-repeat placeholder). A second call for the same (path, size, mtime)
/// returns the cached digest without re-reading the file, whether that call
/// lands in this process (in-memory single-flight cache) or a later restart
/// finds the same file state (persisted record). Concurrent calls for the
/// same (path, size, mtime) single-flight onto one read instead of each
/// streaming the file independently.
pub(crate) fn weights_digest_for_file(path: &Path) -> Option<String> {
    let (size, mtime_nanos) = file_fingerprint(path)?;
    let key: CacheKey = (path.to_path_buf(), size, mtime_nanos);

    let path_for_compute = path.to_path_buf();
    cache().get_or_compute(key, path, move || {
        if let Some(digest) = load_persisted_record(&path_for_compute, size, mtime_nanos) {
            tracing::debug!(
                path = %path_for_compute.display(),
                "weights_digest loaded from persisted record -- not re-hashed on this restart"
            );
            return Some(digest);
        }
        let started = Instant::now();
        let digest = hash_file_bytes(&path_for_compute)?;
        tracing::info!(
            path = %path_for_compute.display(),
            bytes = size,
            elapsed_ms = started.elapsed().as_millis() as u64,
            "computed weights_digest for served GGUF (one-time cost for this file)"
        );
        store_persisted_record(&path_for_compute, size, mtime_nanos, &digest);
        Some(digest)
    })
}

fn hash_file_bytes(path: &Path) -> Option<String> {
    let mut file = File::open(path).ok()?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer).ok()?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Some(hex::encode(hasher.finalize()))
}

/// One persisted (path, size, mtime) -> digest fact, one JSON file per
/// hashed path under [`weights_digest_cache_dir`].
#[derive(Debug, Serialize, Deserialize)]
struct PersistedDigestRecord {
    path: PathBuf,
    size: u64,
    mtime_nanos: u128,
    digest: String,
}

fn weights_digest_cache_dir() -> PathBuf {
    crate::models::mesh_llm_cache_dir().join("weights-digest")
}

/// Stable filename for `path`'s record -- a hash of the path itself, since
/// the path can contain characters that are not safe as a bare filename.
fn persisted_record_path(path: &Path) -> PathBuf {
    let mut hasher = Sha256::new();
    hasher.update(path.to_string_lossy().as_bytes());
    weights_digest_cache_dir().join(format!("{}.json", hex::encode(hasher.finalize())))
}

/// Best-effort: any read/parse/mismatch failure falls back to re-hashing,
/// never to a fabricated digest.
fn load_persisted_record(path: &Path, size: u64, mtime_nanos: u128) -> Option<String> {
    let contents = std::fs::read(persisted_record_path(path)).ok()?;
    let record: PersistedDigestRecord = serde_json::from_slice(&contents).ok()?;
    if record.path == path && record.size == size && record.mtime_nanos == mtime_nanos {
        Some(record.digest)
    } else {
        None
    }
}

/// Best-effort: a failure to persist is logged and otherwise ignored -- the
/// digest just computed is still returned to the caller and is correct for
/// this process, it simply will not be free on the next restart.
fn store_persisted_record(path: &Path, size: u64, mtime_nanos: u128, digest: &str) {
    let dir = weights_digest_cache_dir();
    if let Err(error) = std::fs::create_dir_all(&dir) {
        tracing::warn!(
            path = %dir.display(),
            %error,
            "cannot create weights_digest cache dir; digest will not survive a restart"
        );
        return;
    }
    let record = PersistedDigestRecord {
        path: path.to_path_buf(),
        size,
        mtime_nanos,
        digest: digest.to_string(),
    };
    let record_path = persisted_record_path(path);
    let tmp_path = record_path.with_extension("json.tmp");
    let write_result = serde_json::to_vec(&record)
        .map_err(std::io::Error::other)
        .and_then(|bytes| std::fs::write(&tmp_path, bytes))
        .and_then(|()| std::fs::rename(&tmp_path, &record_path));
    if let Err(error) = write_result {
        tracing::warn!(
            path = %record_path.display(),
            %error,
            "cannot persist weights_digest record; digest will not survive a restart"
        );
        let _ = std::fs::remove_file(&tmp_path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    fn temp_file(name: &str, contents: &[u8]) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "weights-digest-test-{}-{}",
            std::process::id(),
            name
        ));
        std::fs::create_dir_all(&dir).expect("mk temp dir");
        let path = dir.join("model.gguf");
        std::fs::write(&path, contents).expect("write temp file");
        path
    }

    /// The digest is a real SHA-256 of the bytes on disk -- recomputing it
    /// independently must agree exactly.
    #[test]
    fn digest_matches_independent_sha256_of_the_same_bytes() {
        let path = temp_file("matches", b"gguf-bytes-under-test");
        let digest = weights_digest_for_file(&path).expect("digest computed");

        let mut hasher = Sha256::new();
        hasher.update(b"gguf-bytes-under-test");
        let expected = hex::encode(hasher.finalize());

        assert_eq!(digest, expected);
        assert_eq!(digest.len(), 64);
        assert!(digest.chars().all(|c| c.is_ascii_hexdigit()));
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    /// A different quantization/file swapped in under the same path AND the
    /// same size changes the digest as soon as mtime moves -- the digest
    /// records the swap (it does not, by itself, prove which bytes actually
    /// ran).
    #[test]
    fn swapping_the_file_contents_changes_the_digest() {
        let path = temp_file("swap", b"quant-a-bytes-000000");
        let before = weights_digest_for_file(&path).expect("first digest");

        // Same length, different bytes, and force mtime forward so the cache
        // key changes -- otherwise a same-second rewrite could alias the
        // prior (path, size, mtime) key, which is the documented limitation
        // of this cache, not the case under test here.
        std::fs::write(&path, b"quant-b-bytes-111111").expect("rewrite file");
        let future = std::time::SystemTime::now() + std::time::Duration::from_secs(2);
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .expect("reopen for mtime bump");
        file.set_modified(future).expect("bump mtime");

        let after = weights_digest_for_file(&path).expect("second digest");
        assert_ne!(before, after, "swapped bytes must change the digest");
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    /// A second call for the SAME (path, size, mtime) is a cache hit: the
    /// digest is identical, and (implicitly) the file is not re-read -- the
    /// case above proves the cache key can change; this proves an unchanged
    /// key does not silently drift.
    #[test]
    fn second_call_for_unchanged_file_returns_the_same_cached_digest() {
        let path = temp_file("cache-hit", b"stable-bytes");
        let first = weights_digest_for_file(&path).expect("first digest");
        let second = weights_digest_for_file(&path).expect("second digest");
        assert_eq!(first, second);
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    /// A file that does not exist -- or cannot be stat'd -- yields `None`,
    /// never a fabricated digest.
    #[test]
    fn unreadable_file_yields_none_never_a_fabricated_digest() {
        let path = std::env::temp_dir().join(format!(
            "weights-digest-test-missing-{}.gguf",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        assert!(weights_digest_for_file(&path).is_none());
    }

    /// Item 1 (michaelneale): a persisted record surviving a process restart
    /// must be used instead of re-hashing. Simulated here by writing the
    /// record directly (bypassing `weights_digest_for_file`'s own in-memory
    /// cache, which a real restart would not have either) with a value that
    /// could not be a real SHA-256 of the file's contents, then confirming a
    /// fresh call for the SAME (path, size, mtime) returns that planted
    /// value rather than a freshly computed one.
    #[test]
    fn unchanged_file_loads_digest_from_a_persisted_record_without_rehashing() {
        let path = temp_file("persisted-hit", b"persisted-bytes-under-test");
        let (size, mtime_nanos) = file_fingerprint(&path).expect("fingerprint");
        store_persisted_record(&path, size, mtime_nanos, "not-a-real-sha256-digest");

        let digest = weights_digest_for_file(&path).expect("digest loaded");
        assert_eq!(
            digest, "not-a-real-sha256-digest",
            "a matching persisted record must be used instead of re-hashing the file"
        );
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
        let _ = std::fs::remove_file(persisted_record_path(&path));
    }

    /// A persisted record for a file's OLD (size, mtime) must be ignored
    /// once the file has detectably changed -- the persisted cache must not
    /// resurrect a stale digest for bytes that are no longer on disk.
    #[test]
    fn persisted_record_for_a_stale_file_state_is_ignored() {
        let path = temp_file("persisted-stale", b"original-bytes-0000000");
        let (size, mtime_nanos) = file_fingerprint(&path).expect("fingerprint");
        store_persisted_record(
            &path,
            size,
            mtime_nanos,
            "stale-digest-from-before-the-rewrite",
        );

        // Different length AND a forced later mtime, so the record above no
        // longer matches the file's current (size, mtime).
        std::fs::write(&path, b"rewritten-with-a-different-length").expect("rewrite file");
        let future = std::time::SystemTime::now() + std::time::Duration::from_secs(2);
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .expect("reopen for mtime bump");
        file.set_modified(future).expect("bump mtime");

        let digest = weights_digest_for_file(&path).expect("digest computed");
        assert_ne!(digest, "stale-digest-from-before-the-rewrite");
        let mut hasher = Sha256::new();
        hasher.update(b"rewritten-with-a-different-length");
        assert_eq!(digest, hex::encode(hasher.finalize()));
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
        let _ = std::fs::remove_file(persisted_record_path(&path));
    }

    /// Documents the known blind spot stated in the module doc comment: a
    /// file rewritten with content that changes but happens to preserve
    /// BOTH the exact size and the exact mtime aliases the cache key, so
    /// the stale digest is returned unchanged. This is not a bug to fix
    /// (the cache key is (path, size, mtime) by design, and mtime
    /// resolution/rewrite races are out of this module's control) -- it is
    /// a characterization test that pins the documented limitation so a
    /// future change to the cache key does not silently alter it un-noticed.
    #[test]
    fn same_size_and_mtime_rewrite_is_the_documented_blind_spot() {
        let path = temp_file("blind-spot", b"aaaaaaaaaaaaaaaaaaaa");
        let before = weights_digest_for_file(&path).expect("first digest");

        let metadata = std::fs::metadata(&path).expect("stat before rewrite");
        let original_mtime = metadata.modified().expect("mtime before rewrite");

        // Same length, different bytes, mtime forced back to the exact
        // original value -- the cache key is identical to the first call.
        std::fs::write(&path, b"bbbbbbbbbbbbbbbbbbbb").expect("rewrite file, same length");
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .expect("reopen for mtime pin");
        file.set_modified(original_mtime)
            .expect("pin mtime back to the original value");

        let after = weights_digest_for_file(&path).expect("second digest");
        assert_eq!(
            before, after,
            "a same-size/same-mtime rewrite is the documented blind spot: the stale digest is returned"
        );
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
        let _ = std::fs::remove_file(persisted_record_path(&path));
    }

    /// Concurrent callers racing for the SAME key must single-flight onto
    /// one computation, not each compute independently -- the exact defect
    /// this cache exists to prevent (each miss would otherwise stream a
    /// multi-GB GGUF on its own). This exercises `SingleFlightCache`
    /// directly with its own isolated instance and a custom slow `compute`
    /// closure, so it cannot interleave with the other tests in this file
    /// (which all go through the shared global cache).
    #[test]
    fn concurrent_misses_for_the_same_key_single_flight_onto_one_computation() {
        let cache = SingleFlightCache::new();
        let path = temp_file("single-flight", b"single-flight-bytes-under-test");
        let key: CacheKey = (path.clone(), 0, 0);
        let compute_calls = AtomicUsize::new(0);
        let barrier = Barrier::new(8);

        let results: Vec<Option<String>> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..8)
                .map(|_| {
                    let cache = &cache;
                    let barrier = &barrier;
                    let compute_calls = &compute_calls;
                    let key = key.clone();
                    let path = path.clone();
                    scope.spawn(move || {
                        barrier.wait();
                        cache.get_or_compute(key, &path, || {
                            compute_calls.fetch_add(1, Ordering::SeqCst);
                            std::thread::sleep(Duration::from_millis(150));
                            Some("single-flight-digest".to_string())
                        })
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        assert!(
            results
                .iter()
                .all(|digest| digest.as_deref() == Some("single-flight-digest")),
            "every caller must observe the single computed digest"
        );
        assert_eq!(
            compute_calls.load(Ordering::SeqCst),
            1,
            "concurrent callers racing for the same key must single-flight onto one computation"
        );

        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn older_completion_cannot_evict_a_newer_pending_or_ready_entry() {
        let cache = SingleFlightCache::new();
        let path = temp_file("out-of-order", b"out-of-order-bytes");
        let old_key: CacheKey = (path.clone(), 1, 1);
        let new_key: CacheKey = (path.clone(), 2, 2);
        let (old_started_tx, old_started_rx) = std::sync::mpsc::channel();
        let (release_old_tx, release_old_rx) = std::sync::mpsc::channel();

        std::thread::scope(|scope| {
            let old_path = path.clone();
            let cache_ref = &cache;
            let old = scope.spawn(move || {
                cache_ref.get_or_compute(old_key, &old_path, || {
                    old_started_tx.send(()).unwrap();
                    release_old_rx.recv().unwrap();
                    Some("old-digest".to_string())
                })
            });

            old_started_rx.recv().unwrap();
            assert_eq!(
                cache.get_or_compute(new_key.clone(), &path, || {
                    Some("new-digest".to_string())
                }),
                Some("new-digest".to_string())
            );
            release_old_tx.send(()).unwrap();
            assert_eq!(old.join().unwrap(), Some("old-digest".to_string()));

            let recomputes = AtomicUsize::new(0);
            assert_eq!(
                cache.get_or_compute(new_key, &path, || {
                    recomputes.fetch_add(1, Ordering::SeqCst);
                    Some("unexpected-recompute".to_string())
                }),
                Some("new-digest".to_string())
            );
            assert_eq!(recomputes.load(Ordering::SeqCst), 0);
        });

        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }
}
