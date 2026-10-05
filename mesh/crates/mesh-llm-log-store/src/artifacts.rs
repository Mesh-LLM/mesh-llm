//! File-backed artifact storage with transactional DB pointers.
//! Artifact content lives on disk; SQLite rows track metadata + checksums.

use crate::artifact_privacy::{
    ArtifactPrivacy, PlatformArtifactPrivacy, create_private_directory_tree,
};
use crate::error::LogStoreError;
use crate::maintenance::MaintenanceExecutionControl;
use crate::repositories::CascadeArtifactPointer;
use crate::store::{Clock, LogStore};
use crate::timestamps::canonical_persisted_timestamp;
use sha2::{Digest, Sha256};
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Redaction hook applied before every artifact write.
///
/// The storage crate intentionally does not define logging's privacy rules: that
/// policy belongs to the host. It does make the policy non-optional at this
/// boundary, so production callers cannot accidentally persist bytes without
/// passing them through their canonical redactor.
pub type ArtifactRedactor = Arc<dyn Fn(&[u8]) -> Vec<u8> + Send + Sync>;

#[cfg(test)]
type ArtifactFileRemover = Arc<dyn Fn(&Path) -> io::Result<()> + Send + Sync>;

/// Receipt returned after a successful artifact write (no filesystem paths exposed).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactWriteReceipt {
    pub artifact_id: String,
    pub bytes: usize,
    pub checksum: String, // lowercase hex sha256 of stored bytes
    pub version: u32,
    pub media_kind: Option<String>,
    pub redacted: bool,
    pub truncated: bool,
}

/// Artifact content returned by read (no filesystem paths exposed).
#[derive(Debug)]
pub struct ArtifactContent {
    pub artifact_id: String,
    pub bytes: Vec<u8>,
    pub checksum: String, // lowercase hex sha256 of stored bytes
    pub version: u32,
    pub media_kind: Option<String>,
    pub redacted: bool,
    pub truncated: bool,
    #[allow(dead_code)]
    pub kind: String,
}

/// Status enum for artifact health checks.
#[derive(Debug, Clone, PartialEq)]
pub enum ArtifactStatus {
    Ok { checksum: String },
    Missing,
    Corrupt,
}

/// File-backed store paired with a LogStore for transactional pointer rows.
pub struct ArtifactFileStore {
    root: PathBuf, // canonicalised at open time
    #[allow(dead_code)]
    clock: Arc<dyn Clock>, // reserved for stored_at timestamps in future work
    store: Arc<LogStore>, // shared DB connection (guarded by Mutex inside)
    redact: ArtifactRedactor,
    privacy: Arc<dyn ArtifactPrivacy>,
    maintenance_lock: Mutex<()>,
    #[cfg(test)]
    remove_file_for_test: ArtifactFileRemover,
}

/// Stable, path-free classification for a failed maintenance file removal.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CascadeArtifactDeleteFailure {
    Io,
    UnsafePath,
}

/// One transaction-selected artifact deletion outcome. This remains internal
/// to the maintenance layer so raw filesystem paths never become API data.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum CascadeArtifactDeleteResult {
    Removed(CascadeArtifactPointer),
    Missing(CascadeArtifactPointer),
    Failed {
        pointer: CascadeArtifactPointer,
        class: CascadeArtifactDeleteFailure,
    },
}

impl CascadeArtifactDeleteResult {
    pub(crate) const fn succeeded(&self) -> bool {
        matches!(self, Self::Removed(_) | Self::Missing(_))
    }

    pub(crate) fn pointer(&self) -> &CascadeArtifactPointer {
        match self {
            Self::Removed(pointer) | Self::Missing(pointer) | Self::Failed { pointer, .. } => {
                pointer
            }
        }
    }

    pub(crate) const fn failure_class(&self) -> Option<CascadeArtifactDeleteFailure> {
        match self {
            Self::Failed { class, .. } => Some(*class),
            Self::Removed(_) | Self::Missing(_) => None,
        }
    }
}

// ─── Path helpers ──────────────────────

/// Reject path segments containing / \ .. NUL or standalone ".".
fn sanitize_segment(segment: &str) -> Result<(), LogStoreError> {
    if segment.is_empty() || segment == "." || segment == ".." {
        return Err(LogStoreError::PathUnsafe {
            segment: segment.to_string(),
        });
    }
    for c in segment.chars() {
        match c {
            '/' | '\\' | '\0' => {
                return Err(LogStoreError::PathUnsafe {
                    segment: segment.to_string(),
                });
            }
            _ => {}
        }
    }
    Ok(())
}

/// Build the canonical file path for an artifact from request_id + artifact_id.
fn artifact_path(
    root: &Path,
    request_id: &str,
    artifact_id: &str,
) -> Result<PathBuf, LogStoreError> {
    sanitize_segment(request_id)?;
    sanitize_segment(artifact_id)?;

    require_directory(root)?;
    let request_dir = root.join(request_id);
    require_non_link_if_present(&request_dir)?;

    let path = request_dir.join(artifact_id);
    require_non_link_if_present(&path)?;
    Ok(path)
}

fn create_private_dir(path: &Path, privacy: &dyn ArtifactPrivacy) -> Result<(), LogStoreError> {
    create_private_directory_tree(path, privacy)
}

/// Creates one direct child of the already-confined artifact root. `create_dir`
/// deliberately replaces `create_dir_all` here: every parent has already been
/// inspected with `symlink_metadata`, so a request ID can never redirect a write
/// through a link outside the owned root.
fn create_private_child_dir(
    root: &Path,
    child: &str,
    privacy: &dyn ArtifactPrivacy,
) -> Result<PathBuf, LogStoreError> {
    sanitize_segment(child)?;
    require_directory(root)?;
    let path = root.join(child);
    match fs::symlink_metadata(&path) {
        Ok(_) => require_directory(&path)?,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            fs::create_dir(&path)?;
            require_directory(&path)?;
        }
        Err(error) => return Err(LogStoreError::IoError(error)),
    }
    privacy.prepare_directory(&path)?;
    Ok(path)
}

/// A component is safe only when it is a real directory, never a link. This is
/// intentionally based on `symlink_metadata`: `metadata` would follow a link
/// before we have established that it remains below the artifact root.
fn require_directory(path: &Path) -> Result<(), LogStoreError> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(path_unsafe());
    }
    Ok(())
}

/// Reject a link at an optional leaf without treating a missing file as an
/// error. Artifact reads/statuses need the latter to report typed Missing.
fn require_non_link_if_present(path: &Path) -> Result<(), LogStoreError> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => Err(path_unsafe()),
        Ok(_) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(LogStoreError::IoError(error)),
    }
}

fn safe_metadata(path: &Path) -> Result<Option<fs::Metadata>, LogStoreError> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => Err(path_unsafe()),
        Ok(metadata) => Ok(Some(metadata)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(LogStoreError::IoError(error)),
    }
}

fn pointer_has_no_stored_content(checksum: Option<&str>, bytes: i64) -> bool {
    checksum.is_none() && bytes == 0
}

fn path_unsafe() -> LogStoreError {
    LogStoreError::PathUnsafe {
        // Never include a storage path or attacker-controlled leaf in a
        // recoverable/loggable error. This is the same stable reason used by
        // the platform privacy adapter.
        segment: "symlink_not_allowed".to_string(),
    }
}

// ─── ArtifactFileStore implementation ──────────────

impl ArtifactFileStore {
    /// Open artifact storage at `artifact_root`, creating dirs with owner-only permissions.
    #[cfg(test)]
    pub fn open(
        artifact_root: PathBuf,
        clock: Arc<dyn Clock>,
        store: LogStore,
    ) -> Result<Self, LogStoreError> {
        Self::open_with_privacy_and_shared_store(
            artifact_root,
            clock,
            Arc::new(store),
            Arc::new(|content| content.to_vec()),
            Arc::new(PlatformArtifactPrivacy),
        )
    }

    /// Open with a redaction hook applied before every write.
    pub fn open_with_redactor(
        artifact_root: PathBuf,
        clock: Arc<dyn Clock>,
        store: LogStore,
        redact_fn: ArtifactRedactor,
    ) -> Result<Self, LogStoreError> {
        Self::open_with_privacy_and_shared_store(
            artifact_root,
            clock,
            Arc::new(store),
            redact_fn,
            Arc::new(PlatformArtifactPrivacy),
        )
    }

    pub(crate) fn open_with_privacy_and_shared_store(
        artifact_root: PathBuf,
        clock: Arc<dyn Clock>,
        store: Arc<LogStore>,
        redact: ArtifactRedactor,
        privacy: Arc<dyn ArtifactPrivacy>,
    ) -> Result<Self, LogStoreError> {
        create_private_dir(&artifact_root, privacy.as_ref())?;
        let canonical = artifact_root
            .canonicalize()
            .map_err(LogStoreError::IoError)?;
        create_private_dir(&canonical.join("tmp"), privacy.as_ref())?;

        let s = Self {
            root: canonical,
            clock,
            store,
            redact,
            privacy,
            maintenance_lock: Mutex::new(()),
            #[cfg(test)]
            remove_file_for_test: Arc::new(|path: &Path| fs::remove_file(path)),
        };
        s.recover_startup();
        Ok(s)
    }

    #[cfg(test)]
    pub(crate) fn open_with_privacy_for_test(
        artifact_root: PathBuf,
        clock: Arc<dyn Clock>,
        store: LogStore,
        privacy: Arc<dyn ArtifactPrivacy>,
    ) -> Result<Self, LogStoreError> {
        Self::open_with_privacy_and_shared_store(
            artifact_root,
            clock,
            Arc::new(store),
            Arc::new(|content| content.to_vec()),
            privacy,
        )
    }

    #[cfg(test)]
    pub(crate) fn set_remove_file_for_test(&mut self, remove_file: ArtifactFileRemover) {
        self.remove_file_for_test = remove_file;
    }

    pub(crate) fn maintenance_lock(&self) -> std::sync::MutexGuard<'_, ()> {
        self.maintenance_lock
            .lock()
            .expect("maintenance mutex poisoned")
    }

    // ─── Write ──────────────

    /// Write artifact content to disk with a transactional DB pointer.
    /// Rejects writes exceeding byte_limit or aggregate_limit before creating any file.
    #[allow(clippy::too_many_arguments)]
    pub fn write_artifact(
        &self,
        artifact_id: &str,
        request_id: &str,
        kind: &str,
        occurred_at: &str,
        content: &[u8],
        media_kind: Option<&str>,
        version: u32,
        _caller_claimed_redacted: bool,
        truncated_flag: bool,
        byte_limit: usize,
        aggregate_limit: usize,
    ) -> Result<ArtifactWriteReceipt, LogStoreError> {
        // Validate IDs.
        sanitize_segment(artifact_id)?;
        sanitize_segment(request_id)?;
        let occurred_at = canonical_persisted_timestamp(occurred_at)?;

        // Check for existing pointer before any disk work.
        let exists: bool = self
            .store
            .conn()
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM artifact_pointers WHERE artifact_id = ?)",
                rusqlite::params![artifact_id],
                |r| r.get::<_, i32>(0),
            )
            .map(|v| v != 0)
            .map_err(LogStoreError::Sqlite)?;

        if exists {
            return Err(LogStoreError::AlreadyExists {
                entity: format!("artifact_pointer {}", artifact_id),
            });
        }

        // Every write crosses the mandatory host-owned redactor before any
        // size check or disk I/O. The legacy caller flag is deliberately
        // ignored: it cannot assert that raw bytes were redacted.
        let processed = (self.redact)(content);
        let final_redacted = true;

        // Reject individual oversized artifacts before creating a request/tmp
        // directory, file, or pointer row. Storage never silently truncates.
        if processed.len() > byte_limit {
            return Err(LogStoreError::ArtifactLimitExceeded {
                artifact_id: artifact_id.to_string(),
                limit_bytes: byte_limit,
                kind: "byte".to_string(),
            });
        }
        let stored = processed;
        let final_truncated = truncated_flag;

        // Check aggregate limit for this request (existing bytes + new bytes).
        let existing_bytes = self.store.sum_artifact_bytes_for_request(request_id)?;
        if existing_bytes + stored.len() as i64 > aggregate_limit as i64 {
            return Err(LogStoreError::ArtifactLimitExceeded {
                artifact_id: artifact_id.to_string(),
                limit_bytes: aggregate_limit,
                kind: "aggregate".to_string(),
            });
        }

        // 5. Compute checksum over final stored bytes.
        let mut hasher = Sha256::new();
        hasher.update(&stored);
        let checksum_hex = hex::encode(hasher.finalize());

        // 6. Atomic write: tmp/<artifact_id>.part → rename to <request_id>/<artifact_id>.
        // Direct-child creation verifies the root and request component before
        // any artifact bytes exist. Never use create_dir_all for request IDs:
        // it follows an existing link before the privacy adapter can inspect it.
        let parent_dir = create_private_child_dir(&self.root, request_id, self.privacy.as_ref())?;
        let final_path = artifact_path(&self.root, request_id, artifact_id)?;
        debug_assert_eq!(final_path.parent(), Some(parent_dir.as_path()));
        if safe_metadata(&final_path)?.is_some() {
            return Err(path_unsafe());
        }

        let tmp_dir = create_private_child_dir(&self.root, "tmp", self.privacy.as_ref())?;

        let tmp_path = tmp_dir.join(format!("{}.part", artifact_id));

        {
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                // Recreate with mode. If file exists from previous crash, overwrite.
                if safe_metadata(&tmp_path)?.is_some() {
                    fs::remove_file(&tmp_path).map_err(LogStoreError::IoError)?;
                }

                let mut opts = OpenOptions::new();
                opts.mode(0o600).write(true).create_new(true);
                let mut f = opts.open(&tmp_path).map_err(|e| {
                    LogStoreError::IoError(io::Error::other(format!("open temp: {}", e)))
                })?;
                if let Err(error) = self.privacy.prepare_file(&tmp_path) {
                    let _ = fs::remove_file(&tmp_path);
                    return Err(error);
                }
                f.write_all(&stored).map_err(LogStoreError::IoError)?;
                f.sync_all().map_err(LogStoreError::IoError)?;
            }

            #[cfg(not(unix))]
            {
                if safe_metadata(&tmp_path)?.is_some() {
                    fs::remove_file(&tmp_path).map_err(LogStoreError::IoError)?;
                }
                let mut opts = OpenOptions::new();
                opts.write(true).create_new(true);
                let mut f = opts.open(&tmp_path).map_err(|e| {
                    LogStoreError::IoError(io::Error::other(format!("open temp: {}", e)))
                })?;
                if let Err(error) = self.privacy.prepare_file(&tmp_path) {
                    let _ = fs::remove_file(&tmp_path);
                    return Err(error);
                }
                f.write_all(&stored).map_err(LogStoreError::IoError)?;
                f.sync_all().map_err(LogStoreError::IoError)?;
            }
        }

        // Rename atomically.
        fs::rename(&tmp_path, &final_path)
            .map_err(|e| LogStoreError::IoError(io::Error::other(format!("rename: {}", e))))?;

        if let Err(error) = self.privacy.prepare_file(&final_path) {
            let _ = fs::remove_file(&final_path);
            return Err(error);
        }

        // Transactional DB INSERT + UPDATE of pointer row. File is already on disk; if txn fails, clean up the file.
        match self.store.txn(|tx| {
            tx.execute(
                "INSERT INTO artifact_pointers \
                 (artifact_id, request_id, occurred_at, kind) VALUES (?, ?, ?, ?)",
                rusqlite::params![artifact_id, request_id, occurred_at, kind],
            )
            .map_err(LogStoreError::Sqlite)?;

            tx.execute(
                "UPDATE artifact_pointers \
                 SET media_kind = ?, checksum = ?, bytes = ?, version = ?, \
                     redacted = ?, truncated = ? \
                 WHERE artifact_id = ?",
                rusqlite::params![
                    media_kind,
                    &checksum_hex,
                    stored.len() as i64,
                    version as i32,
                    final_redacted as i32,
                    final_truncated as i32,
                    artifact_id
                ],
            )
            .map_err(LogStoreError::Sqlite)?;

            Ok(())
        }) {
            Ok(()) => {}
            Err(e) => {
                // Best-effort cleanup: remove the file we just wrote since txn failed.
                let _ = fs::remove_file(&final_path);
                return Err(e);
            }
        }

        Ok(ArtifactWriteReceipt {
            artifact_id: artifact_id.to_string(),
            bytes: stored.len(),
            checksum: checksum_hex,
            version,
            media_kind: media_kind.map(String::from),
            redacted: final_redacted,
            truncated: final_truncated,
        })
    }

    // ─── Read ──────────────

    pub fn read_artifact(&self, artifact_id: &str) -> Result<ArtifactContent, LogStoreError> {
        sanitize_segment(artifact_id)?;

        let row = self
            .store
            .get_artifact_pointer(artifact_id)?
            .ok_or_else(|| LogStoreError::ArtifactMissing {
                artifact_id: artifact_id.to_string(),
            })?;

        // Check DB flags for missing/corrupt.
        if pointer_has_no_stored_content(row.checksum.as_deref(), row.bytes) {
            // Pointer exists but file was never written (pre-v2 or recovery gap).
            return Err(LogStoreError::ArtifactMissing {
                artifact_id: artifact_id.to_string(),
            });
        }

        let path = artifact_path(&self.root, &row.request_id, artifact_id)?;

        if safe_metadata(&path)?.is_none() {
            // Mark as missing in DB.
            let _ = self.store.update_artifact_pointer_missing(artifact_id);
            return Err(LogStoreError::ArtifactMissing {
                artifact_id: artifact_id.to_string(),
            });
        }

        let data = std::fs::read(&path)
            .map_err(|e| LogStoreError::IoError(io::Error::other(format!("read file: {}", e))))?;

        // Verify checksum.
        let mut hasher = Sha256::new();
        hasher.update(&data);
        let computed = hex::encode(hasher.finalize());

        let expected_checksum = row.checksum.as_deref().unwrap_or("");
        if !expected_checksum.is_empty() && computed != expected_checksum {
            // Mark as corrupt in DB.
            let _ = self.store.update_artifact_pointer_corrupt(artifact_id);
            return Err(LogStoreError::ArtifactCorrupt {
                artifact_id: artifact_id.to_string(),
            });
        }

        Ok(ArtifactContent {
            artifact_id: row.artifact_id,
            bytes: data,
            checksum: computed,
            version: row.version as u32,
            media_kind: row.media_kind,
            redacted: row.redacted,
            truncated: row.truncated,
            kind: row.kind,
        })
    }

    // ─── Delete single artifact ──────────────

    pub fn delete_artifact(&self, artifact_id: &str) -> Result<(), LogStoreError> {
        sanitize_segment(artifact_id)?;

        let row = self
            .store
            .get_artifact_pointer(artifact_id)?
            .ok_or_else(|| LogStoreError::ArtifactMissing {
                artifact_id: artifact_id.to_string(),
            })?;

        let path = artifact_path(&self.root, &row.request_id, artifact_id)?;
        // Validate confinement before deleting the durable pointer. A symlinked
        // request directory is an unsafe external target, never a best-effort
        // cleanup candidate.
        let _ = safe_metadata(&path)?;

        // Remove the backing file before its durable pointer. A failed
        // filesystem deletion leaves the pointer available for a retry.
        if safe_metadata(&path)?.is_some() {
            fs::remove_file(&path).map_err(LogStoreError::IoError)?;
        }
        self.store.delete_artifact_pointer_row(artifact_id)?;

        Ok(())
    }

    // ─── Delete all artifacts for a request ──────────────

    pub fn delete_artifacts_for_request(&self, request_id: &str) -> Result<u64, LogStoreError> {
        sanitize_segment(request_id)?;

        let rows = self.store.list_artifact_pointers_for_request(request_id)?;
        let count = rows.len() as u64;

        // Validate every pointer-owned path first. This avoids a partial
        // request delete when a later row is redirected through a symlink.
        let paths = rows
            .iter()
            .map(|row| artifact_path(&self.root, request_id, &row.artifact_id))
            .collect::<Result<Vec<_>, _>>()?;

        // Delete files first (best-effort), then DB rows in a transaction.
        for path in paths {
            if safe_metadata(&path)?.is_some() {
                fs::remove_file(path).map_err(LogStoreError::IoError)?;
            }
        }

        self.store
            .delete_artifact_pointer_rows_for_request(request_id)?;

        // Clean up empty request directory.
        let req_dir = self.root.join(request_id);
        if safe_metadata(&req_dir)?.is_some() && is_empty_dir(&req_dir) {
            let _ = fs::remove_dir(&req_dir);
        }

        Ok(count)
    }

    // ─── Status check ──────────────

    pub fn status(&self, artifact_id: &str) -> Result<ArtifactStatus, LogStoreError> {
        sanitize_segment(artifact_id)?;

        let row = match self.store.get_artifact_pointer(artifact_id)? {
            Some(r) => r,
            None => return Ok(ArtifactStatus::Missing),
        };

        // If DB flags it as missing or corrupt.
        if pointer_has_no_stored_content(row.checksum.as_deref(), row.bytes) {
            return Ok(ArtifactStatus::Missing);
        }

        let path = artifact_path(&self.root, &row.request_id, artifact_id)?;

        if safe_metadata(&path)?.is_none() {
            return Ok(ArtifactStatus::Missing);
        }

        // Verify checksum.
        match std::fs::read(&path) {
            Ok(data) => {
                let mut hasher = Sha256::new();
                hasher.update(&data);
                let computed = hex::encode(hasher.finalize());

                if let Some(ref expected) = row.checksum
                    && !expected.is_empty()
                    && computed != *expected
                {
                    return Ok(ArtifactStatus::Corrupt);
                }

                Ok(ArtifactStatus::Ok { checksum: computed })
            }
            Err(_) => Ok(ArtifactStatus::Missing),
        }
    }

    // ─── Startup recovery ──────────────

    /// Idempotent startup recovery. Called from `open()`.
    pub fn recover_startup(&self) {
        self.cleanup_orphan_temps();
        self.remove_unreferenced_files();
        self.mark_missing_pointers();
    }

    fn cleanup_orphan_temps(&self) {
        let tmp = self.root.join("tmp");
        if !matches!(safe_metadata(&tmp), Ok(Some(metadata)) if metadata.is_dir()) {
            return;
        }

        if let Ok(entries) = fs::read_dir(&tmp) {
            for entry in entries.flatten() {
                let path = entry.path();
                // A link is not stale content: recovery must never traverse or
                // remove anything selected through it.
                if matches!(safe_metadata(&path), Ok(Some(metadata)) if metadata.is_file()) {
                    // Both .part files and other regular files in tmp/ are
                    // stale. Directories and links are left untouched.
                    let _ = fs::remove_file(&path);
                }
            }
        }

        // Remove empty tmp dir if possible (it will be recreated on next write).
        let _ = fs::remove_dir(&tmp);
    }

    fn remove_unreferenced_files(&self) {
        // Scan all files under root/ (excluding tmp/) and check they have a pointer row.
        for entry in walk_top_level_dirs(&self.root) {
            let filename = match entry.file_name().and_then(|n| n.to_str()) {
                Some(n) => n,
                None => continue,
            };

            // Skip tmp/ contents.
            if let Ok(rel) = entry.strip_prefix(&self.root)
                && rel.starts_with("tmp")
            {
                continue;
            }

            // Check if this artifact_id has a pointer row.
            match self.store.get_artifact_pointer(filename) {
                Ok(Some(_)) => {} // Referenced — keep it.
                _ => {
                    // Unreferenced file — delete it.
                    // `walk_top_level_dirs` only yields regular files below
                    // non-link request directories. Re-check the leaf just
                    // before removal to avoid following a replacement link.
                    if matches!(safe_metadata(&entry), Ok(Some(metadata)) if metadata.is_file()) {
                        let _ = fs::remove_file(&entry);
                    }

                    // Clean up parent dir if empty.
                    if let Some(parent) = entry.parent() {
                        let _ = clean_empty_dir_up(parent, &self.root);
                    }
                }
            }
        }
    }

    fn mark_missing_pointers(&self) {
        let mut after_cursor: Option<String> = None;

        loop {
            match self
                .store
                .list_artifact_pointers(100, after_cursor.as_deref())
            {
                Ok(page) if page.items.is_empty() => break,
                Ok(page) => {
                    for row in &page.items {
                        if row.unavailable_reason.is_some() {
                            continue;
                        }
                        let path = artifact_path(&self.root, &row.request_id, &row.artifact_id);
                        if let Ok(p) = path
                            && let Ok(None) = safe_metadata(&p)
                        {
                            let _ = self.store.update_artifact_pointer_missing(&row.artifact_id);

                            if let Some(parent) = p.parent() {
                                let _ = clean_empty_dir_up(parent, &self.root);
                            }
                        }
                    }

                    match page.next_cursor {
                        Some(c) => after_cursor = Some(c),
                        None => break,
                    }
                }
                Err(_) => break, // Stop on error — recovery is best-effort.
            }
        }
    }

    /// Delete files named by durable post-commit work and acknowledge each
    /// queue row only after removal or missing-file reconciliation succeeds.
    pub fn delete_artifact_files(
        &self,
        pointers: &[CascadeArtifactPointer],
    ) -> Result<(), LogStoreError> {
        for pointer in pointers {
            let path = artifact_path(&self.root, &pointer.request_id, &pointer.artifact_id)?;
            match safe_metadata(&path)? {
                Some(metadata) if metadata.is_file() => {
                    fs::remove_file(&path).map_err(LogStoreError::IoError)?;
                }
                Some(_) => return Err(path_unsafe()),
                None => {}
            }

            self.store.acknowledge_artifact_deletion(pointer)?;
            if let Some(parent) = path.parent() {
                clean_empty_dir_up(parent, &self.root).map_err(LogStoreError::IoError)?;
            }
        }

        // Also check request dirs are empty after deletion.
        self.cleanup_empty_request_dirs()
    }

    /// Remove transaction-selected files and return one typed outcome per
    /// pointer. Failed entries retain their database pointers for a later
    /// same-operation retry; neither this result nor its failure class carries
    /// a filesystem path. Cancellation is checked around each potentially
    /// blocking removal so a timed-out operation never starts later files or
    /// reaches its durable reconciliation phase.
    pub(crate) fn delete_artifact_files_for_maintenance(
        &self,
        pointers: &[CascadeArtifactPointer],
        control: &dyn MaintenanceExecutionControl,
    ) -> Result<Vec<CascadeArtifactDeleteResult>, LogStoreError> {
        let mut results = Vec::with_capacity(pointers.len());
        for pointer in pointers {
            ensure_maintenance_active(control)?;
            let result = match artifact_path(&self.root, &pointer.request_id, &pointer.artifact_id)
            {
                Ok(path) => match safe_metadata(&path) {
                    Ok(None) => CascadeArtifactDeleteResult::Missing(pointer.clone()),
                    Ok(Some(metadata)) if metadata.is_file() => {
                        match self.remove_file_for_maintenance(&path) {
                            Ok(()) => CascadeArtifactDeleteResult::Removed(pointer.clone()),
                            Err(_) => CascadeArtifactDeleteResult::Failed {
                                pointer: pointer.clone(),
                                class: CascadeArtifactDeleteFailure::Io,
                            },
                        }
                    }
                    Ok(Some(_)) | Err(LogStoreError::PathUnsafe { .. }) => {
                        CascadeArtifactDeleteResult::Failed {
                            pointer: pointer.clone(),
                            class: CascadeArtifactDeleteFailure::UnsafePath,
                        }
                    }
                    Err(_) => CascadeArtifactDeleteResult::Failed {
                        pointer: pointer.clone(),
                        class: CascadeArtifactDeleteFailure::Io,
                    },
                },
                Err(LogStoreError::PathUnsafe { .. }) => CascadeArtifactDeleteResult::Failed {
                    pointer: pointer.clone(),
                    class: CascadeArtifactDeleteFailure::UnsafePath,
                },
                Err(_) => CascadeArtifactDeleteResult::Failed {
                    pointer: pointer.clone(),
                    class: CascadeArtifactDeleteFailure::Io,
                },
            };
            ensure_maintenance_active(control)?;
            results.push(result);
        }
        Ok(results)
    }

    fn remove_file_for_maintenance(&self, path: &Path) -> io::Result<()> {
        #[cfg(test)]
        {
            (self.remove_file_for_test)(path)
        }
        #[cfg(not(test))]
        {
            fs::remove_file(path)
        }
    }

    fn cleanup_empty_request_dirs(&self) -> Result<(), LogStoreError> {
        if let Ok(entries) = fs::read_dir(&self.root) {
            for entry in entries {
                let entry = match entry {
                    Ok(e) => e,
                    Err(_) => continue,
                };

                let path = entry.path();
                if entry.file_name().to_str() == Some("tmp")
                    || !matches!(safe_metadata(&path), Ok(Some(metadata)) if metadata.is_dir())
                {
                    continue;
                }

                let req_id = match entry.file_name().to_str() {
                    Some(r) => r.to_owned(),
                    None => continue,
                };

                let count: i64 = self
                    .store
                    .conn()
                    .query_row(
                        "SELECT COUNT(*) FROM artifact_pointers WHERE request_id = ?",
                        rusqlite::params![req_id],
                        |r| r.get::<_, i64>(0),
                    )
                    .map_err(LogStoreError::Sqlite)?;

                if count == 0 && is_empty_dir(&path) {
                    let _ = fs::remove_dir(path);
                }
            }
        }

        Ok(())
    }

    /// Expose root path for tests only.
    #[cfg(test)]
    pub fn root_path(&self) -> &Path {
        &self.root
    }

    /// Get the LogStore reference (for test access).
    pub(crate) fn store_ref(&self) -> &LogStore {
        self.store.as_ref()
    }
}

fn ensure_maintenance_active(
    control: &dyn MaintenanceExecutionControl,
) -> Result<(), LogStoreError> {
    if control.is_cancelled() {
        Err(LogStoreError::MaintenanceExecutionCancelled)
    } else {
        Ok(())
    }
}

// ─── Helpers ──────────────

fn is_empty_dir(path: &Path) -> bool {
    if let Ok(entries) = fs::read_dir(path) {
        entries.count() == 0
    } else {
        false
    }
}

/// Remove `dir` and walk up removing empty parent dirs, stopping at `stop_at`.
fn clean_empty_dir_up(dir: &Path, stop_at: &Path) -> io::Result<()> {
    let mut current = dir.to_path_buf();
    loop {
        if !current.starts_with(stop_at) || current == *stop_at {
            break;
        }

        match fs::remove_dir(&current) {
            Ok(()) => {}     // removed — try parent next.
            Err(_) => break, // not empty or other error — stop walking up.
        }

        if let Some(parent) = current.parent() {
            current = parent.to_path_buf();
        } else {
            break;
        }
    }
    Ok(())
}

/// Walk only regular files directly owned by request directories under `root`.
///
/// The artifact layout is exactly `<root>/<request-id>/<artifact-id>`. Do not
/// recurse generically: `Path::is_dir` follows links and would turn recovery
/// into a filesystem traversal primitive.
fn walk_top_level_dirs(root: &Path) -> Vec<PathBuf> {
    let mut result = Vec::new();

    if !matches!(safe_metadata(root), Ok(Some(metadata)) if metadata.is_dir()) {
        return result;
    }

    // Skip tmp/.
    for entry in fs::read_dir(root).ok().into_iter().flatten() {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };

        let path = entry.path();
        if path.file_name().is_some_and(|n| n == "tmp") {
            continue;
        }

        let metadata = match safe_metadata(&path) {
            Ok(Some(metadata)) => metadata,
            Ok(None) | Err(_) => continue,
        };
        if metadata.is_file() {
            result.push(path);
            continue;
        }

        if !metadata.is_dir() {
            continue;
        }

        if let Ok(children) = fs::read_dir(&path) {
            for child in children.flatten() {
                let child_path = child.path();
                if matches!(safe_metadata(&child_path), Ok(Some(metadata)) if metadata.is_file()) {
                    result.push(child_path);
                }
            }
        }
    }

    result
}
