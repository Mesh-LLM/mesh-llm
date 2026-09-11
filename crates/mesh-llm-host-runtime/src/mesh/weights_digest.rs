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
//! Cached by (path, size, mtime): hashing an 8GB GGUF costs real wall-clock
//! time, and must happen once per file, never once per request. Concurrent
//! callers racing for the same (path, size, mtime) single-flight onto one
//! computation instead of each independently streaming the file -- the
//! cache's mutex is only ever held to read or install an entry, never across
//! the file read itself.

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
    entries: Mutex<HashMap<CacheKey, CacheEntry>>,
}

impl SingleFlightCache {
    fn new() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
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
            let mut entries = self.entries.lock().expect("weights digest cache poisoned");
            match entries.get(&key) {
                Some(CacheEntry::Ready(digest)) => Role::Cached(digest.clone()),
                Some(CacheEntry::Pending(pending)) => Role::Wait(pending.clone()),
                None => {
                    let pending = Arc::new(PendingDigest {
                        result: Mutex::new(None),
                        ready: Condvar::new(),
                    });
                    entries.insert(key.clone(), CacheEntry::Pending(pending.clone()));
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
                    .expect("weights digest cache poisoned");
                while result.is_none() {
                    result = pending
                        .ready
                        .wait(result)
                        .expect("weights digest cache poisoned");
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
                    let mut entries = self.entries.lock().expect("weights digest cache poisoned");
                    match &digest {
                        Some(computed) => {
                            entries.insert(key, CacheEntry::Ready(computed.clone()));
                        }
                        // Unreadable: never cache a fabricated absence, so a
                        // later retry (e.g. once the file exists) can succeed.
                        None => {
                            entries.remove(&key);
                        }
                    }
                }
                *pending
                    .result
                    .lock()
                    .expect("weights digest cache poisoned") = Some(digest.clone());
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

/// SHA-256 of `path`'s bytes, lowercase hex. `None` when the file cannot be
/// stat'd or read -- an honest absent fact, never a fabricated value (never a
/// `0`-repeat placeholder). A second call for the same (path, size, mtime)
/// returns the cached digest without re-reading the file. Concurrent calls
/// for the same (path, size, mtime) single-flight onto one read instead of
/// each streaming the file independently.
pub(crate) fn weights_digest_for_file(path: &Path) -> Option<String> {
    let metadata = std::fs::metadata(path).ok()?;
    let size = metadata.len();
    let mtime_nanos = metadata
        .modified()
        .ok()?
        .duration_since(UNIX_EPOCH)
        .ok()?
        .as_nanos();
    let key: CacheKey = (path.to_path_buf(), size, mtime_nanos);

    let path_for_compute = path.to_path_buf();
    cache().get_or_compute(key, path, move || {
        let started = Instant::now();
        let digest = hash_file_bytes(&path_for_compute)?;
        tracing::info!(
            path = %path_for_compute.display(),
            bytes = size,
            elapsed_ms = started.elapsed().as_millis() as u64,
            "computed weights_digest for served GGUF (one-time cost for this file)"
        );
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
}
