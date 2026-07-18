//! Sidecar digest cache for direct GGUF source files.
//!
//! Building a synthetic direct-GGUF package identity requires a content digest
//! of every source shard. Recomputing those digests reads each shard end to
//! end, which costs seconds to minutes of sequential I/O on every model load
//! even when the files have not changed. This cache persists one small JSON
//! record per source file and reuses the stored digest when the file's size,
//! mtime, and ctime still match, skipping the full-file read entirely.
//!
//! The cache is advisory: it is a performance optimization for identity keys
//! that never leave Rust-side coordination (stage deduplication, cache keys,
//! split topology planning), not a security boundary. A `(size, mtime, ctime)`
//! match is treated as sufficient evidence that the content is unchanged. The
//! ctime check catches files replaced with same-size content by tooling that
//! restores mtime (rsync, tar, cp --preserve): userspace cannot reset ctime,
//! so any such replacement forces a recompute. Every failure mode (unreadable
//! entry, corrupt JSON, schema, algorithm, or digest-format mismatch,
//! unwritable cache directory) degrades to recomputing the digest. Non-UTF-8
//! paths and pre-epoch mtimes are simply not cached.
//!
//! Records live under a per-user cache directory (one file per source path)
//! rather than the runtime root, because runtime directories are per-instance
//! and may sit on tmpfs via `XDG_RUNTIME_DIR`, while digests stay valid across
//! reboots. Writes go through a uniquely named temp file and rename so
//! concurrent processes and threads never observe a torn record.

use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::UNIX_EPOCH,
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Bump when the record layout or digest algorithm changes so stale entries
/// from older builds miss cleanly instead of being misread.
const CACHE_SCHEMA_VERSION: u32 = 2;
const CACHE_DIGEST_ALGO: &str = "sha256";

/// Explicit cache directory override, mirroring `MESH_LLM_RUNTIME_ROOT`.
/// Useful for deployments without a home directory (e.g. systemd services).
const CACHE_DIR_ENV: &str = "MESH_LLM_HASH_CACHE_DIR";

#[derive(Serialize, Deserialize)]
struct CachedFileDigest {
    version: u32,
    algo: String,
    path: String,
    size: u64,
    mtime_nanos: u128,
    /// Inode change time; `None` on platforms that do not expose it, which
    /// fall back to validating on size and mtime alone.
    ctime_nanos: Option<i128>,
    digest: String,
}

/// Persistent map from `(path, size, mtime, ctime)` to a source-file content
/// digest.
pub(crate) struct SidecarDigestCache {
    dir: PathBuf,
}

impl SidecarDigestCache {
    /// Resolve the default cache location.
    ///
    /// Precedence:
    /// 1. `MESH_LLM_HASH_CACHE_DIR` environment variable
    /// 2. `~/.mesh-llm/cache/hashes`
    /// 3. `None` (caching disabled, digests are always recomputed)
    pub(crate) fn open_default() -> Option<Self> {
        if let Some(dir) = std::env::var_os(CACHE_DIR_ENV) {
            return Some(Self::open_in(PathBuf::from(dir)));
        }
        let home = dirs::home_dir()?;
        Some(Self::open_in(
            home.join(".mesh-llm").join("cache").join("hashes"),
        ))
    }

    /// Open a cache rooted at an explicit directory (used by tests).
    pub(crate) fn open_in(dir: PathBuf) -> Self {
        Self { dir }
    }

    /// Return the cached digest for `path` when the stored record matches the
    /// file's current size, mtime, and ctime and carries a well-formed digest.
    /// Any read, parse, or validation failure is a miss. Non-UTF-8 paths are
    /// never cached, so they always miss.
    pub(crate) fn lookup(
        &self,
        path: &Path,
        size: u64,
        mtime_nanos: u128,
        ctime_nanos: Option<i128>,
    ) -> Option<String> {
        let path_utf8 = path.to_str()?;
        let bytes = fs::read(self.entry_path(path_utf8)).ok()?;
        let record: CachedFileDigest = serde_json::from_slice(&bytes).ok()?;
        let valid = record.version == CACHE_SCHEMA_VERSION
            && record.algo == CACHE_DIGEST_ALGO
            && record.path == path_utf8
            && record.size == size
            && record.mtime_nanos == mtime_nanos
            && record.ctime_nanos == ctime_nanos
            && is_sha256_hex(&record.digest);
        valid.then_some(record.digest)
    }

    /// Persist a digest record for `path`. Best-effort: failures are logged at
    /// debug level and otherwise ignored so an unwritable cache directory can
    /// never fail a model load. Non-UTF-8 paths are skipped because their
    /// lossy string form could collide with a different path.
    pub(crate) fn store(
        &self,
        path: &Path,
        size: u64,
        mtime_nanos: u128,
        ctime_nanos: Option<i128>,
        digest: &str,
    ) {
        let Some(path_utf8) = path.to_str() else {
            tracing::debug!(
                path = %path.display(),
                "skipping GGUF source digest cache entry for non-UTF-8 path"
            );
            return;
        };
        let record = CachedFileDigest {
            version: CACHE_SCHEMA_VERSION,
            algo: CACHE_DIGEST_ALGO.to_string(),
            path: path_utf8.to_string(),
            size,
            mtime_nanos,
            ctime_nanos,
            digest: digest.to_string(),
        };
        if let Err(error) = self.write_record(path_utf8, &record) {
            tracing::debug!(
                path = %path.display(),
                cache_dir = %self.dir.display(),
                %error,
                "failed to persist GGUF source digest cache entry"
            );
        }
    }

    fn write_record(&self, path_utf8: &str, record: &CachedFileDigest) -> std::io::Result<()> {
        static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

        fs::create_dir_all(&self.dir)?;
        let bytes = serde_json::to_vec(record)?;
        let entry = self.entry_path(path_utf8);
        // Write-then-rename keeps concurrent readers and writers safe: readers
        // never see a partial record. The pid plus counter suffix keeps the
        // temp path unique across processes and threads, so writers never
        // interleave on the same temp file.
        let unique = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let tmp = entry.with_extension(format!("tmp.{}.{unique}", std::process::id()));
        // Remove the temp file on every failure path so failed writes (e.g. a
        // full disk) cannot leak partially written temp files into the cache.
        if let Err(error) = fs::write(&tmp, &bytes) {
            let _ = fs::remove_file(&tmp);
            return Err(error);
        }
        if let Err(error) = fs::rename(&tmp, &entry) {
            let _ = fs::remove_file(&tmp);
            return Err(error);
        }
        Ok(())
    }

    /// One record file per source path, named by a digest of the path so
    /// arbitrary absolute paths map to flat, filesystem-safe names.
    fn entry_path(&self, path_utf8: &str) -> PathBuf {
        let name = hex::encode(Sha256::digest(path_utf8.as_bytes()));
        self.dir.join(format!("{name}.json"))
    }
}

/// A well-formed SHA-256 digest as produced by this crate: exactly 64
/// lowercase hexadecimal characters. Anything else in a record is treated as
/// corruption and misses.
fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
}

/// Nanoseconds since the Unix epoch of the file's mtime, or `None` when the
/// platform reports no mtime or a pre-epoch time (such files are uncacheable).
pub(crate) fn file_mtime_nanos(metadata: &fs::Metadata) -> Option<u128> {
    let mtime = metadata.modified().ok()?;
    let since_epoch = mtime.duration_since(UNIX_EPOCH).ok()?;
    Some(since_epoch.as_nanos())
}

/// Nanoseconds since the Unix epoch of the file's inode change time (ctime).
/// Unlike mtime, ctime cannot be set from userspace, so it catches same-size
/// replacements made by tooling that restores mtime. Returns `None` on
/// platforms that do not expose it; their records validate on size and mtime
/// alone.
#[cfg(unix)]
pub(crate) fn file_ctime_nanos(metadata: &fs::Metadata) -> Option<i128> {
    use std::os::unix::fs::MetadataExt;

    Some(i128::from(metadata.ctime()) * 1_000_000_000 + i128::from(metadata.ctime_nsec()))
}

/// Non-Unix fallback: the change time is not exposed by the standard library,
/// so records validate on size and mtime alone.
#[cfg(not(unix))]
pub(crate) fn file_ctime_nanos(_metadata: &fs::Metadata) -> Option<i128> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const CTIME: Option<i128> = Some(5);

    fn cache_in_tempdir() -> (tempfile::TempDir, SidecarDigestCache) {
        let dir = tempfile::tempdir().unwrap();
        let cache = SidecarDigestCache::open_in(dir.path().join("hashes"));
        (dir, cache)
    }

    /// A syntactically valid SHA-256 fixture built from one hex character.
    fn digest_of(hex_char: char) -> String {
        hex_char.to_string().repeat(64)
    }

    #[test]
    fn lookup_returns_stored_digest_on_exact_match() {
        let (_dir, cache) = cache_in_tempdir();
        let path = Path::new("/models/model.gguf");

        cache.store(path, 42, 1_700_000_000_000_000_000, CTIME, &digest_of('a'));

        assert_eq!(
            cache.lookup(path, 42, 1_700_000_000_000_000_000, CTIME),
            Some(digest_of('a'))
        );
    }

    #[test]
    fn lookup_misses_when_size_or_mtime_changed() {
        let (_dir, cache) = cache_in_tempdir();
        let path = Path::new("/models/model.gguf");

        cache.store(path, 42, 1_700_000_000_000_000_000, CTIME, &digest_of('a'));

        assert_eq!(
            cache.lookup(path, 43, 1_700_000_000_000_000_000, CTIME),
            None
        );
        assert_eq!(
            cache.lookup(path, 42, 1_700_000_000_000_000_001, CTIME),
            None
        );
    }

    #[test]
    fn lookup_misses_when_ctime_changed() {
        let (_dir, cache) = cache_in_tempdir();
        let path = Path::new("/models/model.gguf");

        cache.store(path, 42, 1, Some(5), &digest_of('a'));

        assert_eq!(cache.lookup(path, 42, 1, Some(6)), None);
        assert_eq!(cache.lookup(path, 42, 1, None), None);
        assert_eq!(cache.lookup(path, 42, 1, Some(5)), Some(digest_of('a')));
    }

    #[test]
    fn lookup_misses_for_different_path_with_same_metadata() {
        let (_dir, cache) = cache_in_tempdir();

        cache.store(Path::new("/models/a.gguf"), 42, 1, CTIME, &digest_of('a'));

        assert_eq!(
            cache.lookup(Path::new("/models/b.gguf"), 42, 1, CTIME),
            None
        );
    }

    #[test]
    fn lookup_misses_on_corrupt_entry() {
        let (_dir, cache) = cache_in_tempdir();
        let path = Path::new("/models/model.gguf");

        cache.store(path, 42, 1, CTIME, &digest_of('a'));
        fs::write(cache.entry_path(path.to_str().unwrap()), b"not json").unwrap();

        assert_eq!(cache.lookup(path, 42, 1, CTIME), None);
    }

    #[test]
    fn lookup_misses_on_schema_or_algo_mismatch() {
        let (_dir, cache) = cache_in_tempdir();
        let path = Path::new("/models/model.gguf");
        let path_utf8 = path.to_str().unwrap();

        let stale = CachedFileDigest {
            version: CACHE_SCHEMA_VERSION + 1,
            algo: CACHE_DIGEST_ALGO.to_string(),
            path: path_utf8.to_string(),
            size: 42,
            mtime_nanos: 1,
            ctime_nanos: CTIME,
            digest: digest_of('a'),
        };
        cache.write_record(path_utf8, &stale).unwrap();
        assert_eq!(cache.lookup(path, 42, 1, CTIME), None);

        let wrong_algo = CachedFileDigest {
            version: CACHE_SCHEMA_VERSION,
            algo: "xxh3-128".to_string(),
            ..stale
        };
        cache.write_record(path_utf8, &wrong_algo).unwrap();
        assert_eq!(cache.lookup(path, 42, 1, CTIME), None);
    }

    #[test]
    fn lookup_misses_on_malformed_digest() {
        let (_dir, cache) = cache_in_tempdir();
        let path = Path::new("/models/model.gguf");
        let path_utf8 = path.to_str().unwrap();

        for malformed in ["abc123", "", &digest_of('a')[..63], &"A".repeat(64)] {
            let record = CachedFileDigest {
                version: CACHE_SCHEMA_VERSION,
                algo: CACHE_DIGEST_ALGO.to_string(),
                path: path_utf8.to_string(),
                size: 42,
                mtime_nanos: 1,
                ctime_nanos: CTIME,
                digest: malformed.to_string(),
            };
            cache.write_record(path_utf8, &record).unwrap();
            assert_eq!(
                cache.lookup(path, 42, 1, CTIME),
                None,
                "accepted {malformed:?}"
            );
        }
    }

    #[test]
    fn lookup_misses_when_cache_dir_does_not_exist() {
        let (_dir, cache) = cache_in_tempdir();

        assert_eq!(
            cache.lookup(Path::new("/models/model.gguf"), 42, 1, CTIME),
            None
        );
    }

    #[test]
    fn store_into_unwritable_dir_is_silent() {
        let file = tempfile::NamedTempFile::new().unwrap();
        // The cache dir path points at an existing regular file, so directory
        // creation fails; store must swallow the error.
        let cache = SidecarDigestCache::open_in(file.path().to_path_buf());

        cache.store(
            Path::new("/models/model.gguf"),
            42,
            1,
            CTIME,
            &digest_of('a'),
        );

        assert_eq!(
            cache.lookup(Path::new("/models/model.gguf"), 42, 1, CTIME),
            None
        );
    }

    #[test]
    fn store_overwrites_previous_record() {
        let (_dir, cache) = cache_in_tempdir();
        let path = Path::new("/models/model.gguf");

        cache.store(path, 42, 1, CTIME, &digest_of('a'));
        cache.store(path, 42, 2, CTIME, &digest_of('b'));

        assert_eq!(cache.lookup(path, 42, 1, CTIME), None);
        assert_eq!(cache.lookup(path, 42, 2, CTIME), Some(digest_of('b')));
    }

    #[cfg(unix)]
    #[test]
    fn non_utf8_paths_are_never_cached() {
        use std::{ffi::OsStr, os::unix::ffi::OsStrExt};

        let (_dir, cache) = cache_in_tempdir();
        // Two distinct invalid-byte paths whose lossy string forms collide.
        let first = Path::new(OsStr::from_bytes(b"/models/\xffmodel.gguf"));
        let second = Path::new(OsStr::from_bytes(b"/models/\xfemodel.gguf"));

        cache.store(first, 42, 1, CTIME, &digest_of('a'));

        assert_eq!(cache.lookup(first, 42, 1, CTIME), None);
        assert_eq!(cache.lookup(second, 42, 1, CTIME), None);
        // Nothing was written at all for the non-UTF-8 path.
        assert!(!cache.dir.exists());
    }

    #[test]
    fn concurrent_stores_publish_one_intact_record() {
        let (_dir, cache) = cache_in_tempdir();
        let cache = std::sync::Arc::new(cache);
        let path = Path::new("/models/model.gguf");
        let digests: Vec<String> = "abcdef".chars().map(digest_of).collect();

        std::thread::scope(|scope| {
            for digest in &digests {
                let cache = cache.clone();
                scope.spawn(move || {
                    for _ in 0..50 {
                        cache.store(path, 42, 1, CTIME, digest);
                    }
                });
            }
        });

        // Whichever writer renamed last, the published record must be one of
        // the stored digests, never torn or interleaved bytes.
        let found = cache.lookup(path, 42, 1, CTIME).unwrap();
        assert!(digests.contains(&found), "torn record: {found:?}");
    }

    #[test]
    fn file_mtime_nanos_reports_recent_files() {
        let file = tempfile::NamedTempFile::new().unwrap();
        let metadata = file.path().metadata().unwrap();

        let mtime = file_mtime_nanos(&metadata).unwrap();

        // Sanity: strictly after 2020-01-01 in nanoseconds.
        assert!(mtime > 1_577_836_800_000_000_000);
    }

    #[cfg(unix)]
    #[test]
    fn file_ctime_nanos_reports_recent_files() {
        let file = tempfile::NamedTempFile::new().unwrap();
        let metadata = file.path().metadata().unwrap();

        let ctime = file_ctime_nanos(&metadata).unwrap();

        // Sanity: strictly after 2020-01-01 in nanoseconds.
        assert!(ctime > 1_577_836_800_000_000_000);
    }
}
