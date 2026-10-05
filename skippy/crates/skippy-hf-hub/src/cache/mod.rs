//! Inspect the local Hugging Face cache.
//!
//! [`HFClient::scan_cache`] walks the cache directory and produces an
//! [`HFCacheInfo`] tree: cached repositories, their revisions, and the
//! individual files in each revision.
//!
//! On disk, downloads are content-addressed — files in
//! `<cache>/<repo_folder>/snapshots/<commit>/<filename>` are pointers (symlinks
//! on Unix, copies on Windows) to immutable blobs in
//! `<cache>/<repo_folder>/blobs/<etag>`. Multiple revisions of the same repo
//! often share the same blob, which is why repo-level sizes are deduplicated
//! while revision-level sizes are not — see [`CachedRepoInfo::size_on_disk`]
//! and [`CachedRevisionInfo::size_on_disk`].

use std::path::PathBuf;
use std::time::SystemTime;

use bon::bon;

use crate::client::HFClient;
use crate::error::HFResult;

pub(crate) mod storage;

/// A single file in a cached revision.
///
/// `file_path` is the pointer in the `snapshots/` tree (a symlink on Unix);
/// `blob_path` is the canonical location of the underlying content under
/// `blobs/`. Both paths refer to the same bytes.
#[derive(Debug, Clone)]
pub struct CachedFileInfo {
    /// Path of the file relative to its revision's snapshot root, including
    /// any subdirectories (e.g., `subdir/model.bin`).
    pub file_name: String,
    /// Absolute path of the pointer file inside the `snapshots/` tree.
    /// Symlink on Unix, regular file (a copy of the blob) on Windows.
    pub file_path: PathBuf,
    /// Absolute path of the actual blob under `blobs/`, after resolving
    /// `file_path`. Multiple revisions can point at the same blob.
    pub blob_path: PathBuf,
    /// Size of the blob in bytes.
    pub size_on_disk: u64,
    /// Last access time of the blob, from the filesystem.
    pub blob_last_accessed: SystemTime,
    /// Last modification time of the blob, from the filesystem.
    pub blob_last_modified: SystemTime,
}

/// A cached revision (commit) of a repository.
#[derive(Debug, Clone)]
pub struct CachedRevisionInfo {
    /// Full 40-character commit SHA.
    pub commit_hash: String,
    /// Directory of pointer files for this revision
    /// (`<cache>/<repo_folder>/snapshots/<commit_hash>/`).
    pub snapshot_path: PathBuf,
    /// Files belonging to this revision.
    pub files: Vec<CachedFileInfo>,
    /// Sum of [`CachedFileInfo::size_on_disk`] for every file in this
    /// revision. Blobs shared with other revisions of the same repo are
    /// counted here but not in [`CachedRepoInfo::size_on_disk`].
    pub size_on_disk: u64,
    /// Refs (branches, tags, `refs/pr/<n>`, …) that point at this commit.
    /// May be empty for revisions only reachable by SHA.
    pub refs: Vec<String>,
    /// Latest blob modification time across the files in this revision.
    pub last_modified: SystemTime,
}

/// A cached repository, with its revisions aggregated.
#[derive(Debug, Clone)]
pub struct CachedRepoInfo {
    /// Hub repo identifier (e.g., `gpt2`, `google/bert-base-uncased`).
    pub repo_id: String,
    /// Lowercase singular name of the repo kind (`"model"`, `"dataset"`, `"space"`, or `"kernel"`)
    /// — see [`crate::repository::RepoType::singular`].
    pub repo_type: &'static str,
    /// Absolute path of the repo's cache subfolder
    /// (`<cache>/<type>s--<owner>--<name>/`).
    pub repo_path: PathBuf,
    /// Cached revisions of this repo.
    pub revisions: Vec<CachedRevisionInfo>,
    /// Number of unique blobs stored for this repo. Two revisions sharing the
    /// same blob count once.
    pub nb_files: usize,
    /// Bytes used on disk for unique blobs, with shared blobs counted once.
    pub size_on_disk: u64,
    /// Latest blob access time across all revisions.
    pub last_accessed: SystemTime,
    /// Latest blob modification time across all revisions.
    pub last_modified: SystemTime,
}

/// Snapshot of the local Hugging Face cache directory.
///
/// Returned by [`HFClient::scan_cache`]; aggregates every cached repository
/// found at `cache_dir` along with total disk usage and any warnings emitted
/// during scanning.
#[derive(Debug, Clone)]
pub struct HFCacheInfo {
    /// Cache directory that was scanned.
    pub cache_dir: PathBuf,
    /// Cached repositories discovered under `cache_dir`.
    pub repos: Vec<CachedRepoInfo>,
    /// Sum of [`CachedRepoInfo::size_on_disk`] across all repos.
    pub size_on_disk: u64,
    /// Human-readable warnings for entries that could not be fully scanned —
    /// for example, snapshot pointers whose blobs are missing or unreadable.
    pub warnings: Vec<String>,
}

#[bon]
impl HFClient {
    /// Scan the configured cache directory and return a summary of all cached repositories,
    /// revisions, and files.
    ///
    /// If the cache directory does not exist, returns an [`HFCacheInfo`] with no repos and zero
    /// size — not an error. Unreadable blobs and dangling snapshot pointers are reported via
    /// [`HFCacheInfo::warnings`] rather than failing the scan.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn scan_cache(&self) -> HFResult<HFCacheInfo> {
        storage::scan_cache_dir(self.cache_dir()).await
    }
}

#[cfg(feature = "blocking")]
#[bon]
impl crate::blocking::HFClientSync {
    /// Blocking counterpart of [`HFClient::scan_cache`].
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn scan_cache(&self) -> HFResult<HFCacheInfo> {
        self.runtime.block_on(self.inner.scan_cache().send())
    }
}
