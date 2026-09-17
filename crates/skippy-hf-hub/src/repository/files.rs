//! Repository file listing, metadata, download, and upload APIs.
//!
//! Common entry points:
//!
//! - Use [`HFRepository::list_tree`] to stream file and directory entries, or [`HFRepository::get_paths_info`] /
//!   [`HFRepository::get_file_metadata`] for targeted lookups.
//! - Use [`HFRepository::download_file`] to write one file to disk, [`HFRepository::download_file_stream`] to stream
//!   bytes, [`HFRepository::download_file_to_bytes`] to collect a file into memory, and
//!   [`HFRepository::snapshot_download`] to fetch a whole revision.
//! - Use [`HFRepository::upload_file`] and [`HFRepository::upload_folder`] for convenience, or
//!   [`HFRepository::create_commit`] when you need an explicit set of add/delete operations in one commit.
//!
//! The public types in this module are shared across the listing, download, and
//! upload helpers implemented on [`HFRepository`].

#[cfg(not(target_family = "wasm"))]
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use bytes::Bytes;
use futures::stream::Stream;
use serde::{Deserialize, Serialize};

#[allow(unused_imports)] // used by intra-doc links
use super::HFRepository;
use crate::constants;
use crate::error::HFResult;

/// LFS metadata attached to a repository file, when the file is stored in Git LFS.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlobLfsInfo {
    /// Original file size in bytes, when reported by the Hub.
    pub size: Option<u64>,
    /// SHA-256 object id of the LFS payload.
    pub sha256: Option<String>,
    /// Size in bytes of the LFS pointer file stored in git.
    pub pointer_size: Option<u64>,
}

/// Summary of the last commit that touched a tree entry, included when expanded metadata is requested.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LastCommitInfo {
    /// Commit SHA, when available.
    pub id: Option<String>,
    /// Commit title/summary line.
    pub title: Option<String>,
    /// Commit timestamp in ISO 8601 format, when available.
    pub date: Option<String>,
}

/// Security-scan summary for a file in a repository.
///
/// Populated on [`RepoTreeEntry::File`] entries when the listing was requested with `expand=true`.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BlobSecurityInfo {
    /// Status string reported by the scanner (e.g., `"safe"`, `"unsafe"`, `"suspicious"`). The
    /// file is considered safe iff `status == "safe"`.
    pub status: String,
    /// Antivirus-scan details, when present.
    #[serde(default)]
    pub av_scan: Option<serde_json::Value>,
    /// Pickle-import-scan details, when present.
    #[serde(default)]
    pub pickle_import_scan: Option<serde_json::Value>,
}

/// Metadata returned from a HEAD request on a file's resolve URL.
///
/// Produced by [`HFRepository::get_file_metadata`]
/// and used internally by snapshot downloads.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileMetadataInfo {
    /// Path of the file within the repository.
    pub filename: String,
    /// ETag of the file content (normalized, with weak prefix and quotes stripped).
    pub etag: String,
    /// Commit hash the revision resolved to (from the `X-Repo-Commit` header).
    pub commit_hash: String,
    /// Xet content hash if the file is stored in Xet (from the `X-Xet-Hash` header).
    pub xet_hash: Option<String>,
    /// File size in bytes. Falls back to `0` if neither `X-Linked-Size` nor `Content-Length`
    /// is present on the response.
    pub file_size: u64,
    /// Final URL the HEAD request resolved to after redirects (Hub URL or CDN). `None` when no
    /// redirect was followed and the request URL itself was not preserved.
    pub location: Option<String>,
}

/// File or directory entry returned by repository tree/listing APIs.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum RepoTreeEntry {
    /// A file entry in the repository tree.
    File {
        /// Object id reported by the Hub for this entry.
        oid: String,
        /// File size in bytes.
        size: u64,
        /// Repository-relative path.
        path: String,
        /// LFS metadata, when the file is LFS-backed.
        lfs: Option<BlobLfsInfo>,
        /// Last-commit summary, only when expanded metadata is requested.
        #[serde(default, rename = "lastCommit")]
        last_commit: Option<LastCommitInfo>,
        /// Xet content hash, when the file is Xet-backed.
        #[serde(default, rename = "xetHash")]
        xet_hash: Option<String>,
        /// Security-scan summary for the file, only when expanded metadata is requested.
        #[serde(default, rename = "securityFileStatus")]
        security: Option<BlobSecurityInfo>,
    },
    /// A directory entry in the repository tree.
    Directory {
        /// Object id reported by the Hub for this entry.
        oid: String,
        /// Repository-relative path.
        path: String,
        /// Last-commit summary, only when expanded metadata is requested.
        #[serde(default, rename = "lastCommit")]
        last_commit: Option<LastCommitInfo>,
    },
}

/// Response body returned after creating a commit.
///
/// Includes URLs for the commit and any PR that was opened, along with the commit OID
/// when present. Returned by [`HFRepository::create_commit`]
/// and related upload/delete helpers.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CommitInfo {
    /// URL of the created commit on the Hub, when available.
    #[serde(default)]
    pub commit_url: Option<String>,
    /// Commit message recorded for the operation.
    #[serde(default)]
    pub commit_message: Option<String>,
    /// Commit description/body, when provided.
    #[serde(default)]
    pub commit_description: Option<String>,
    /// Commit SHA, when returned by the API.
    #[serde(default)]
    pub commit_oid: Option<String>,
    /// Pull-request URL, when `create_pr` was enabled and a PR was opened.
    #[serde(default)]
    pub pr_url: Option<String>,
    /// Pull-request number, when `create_pr` was enabled and a PR was opened.
    #[serde(default)]
    pub pr_num: Option<u64>,
}

/// File mutation included in [`HFRepository::create_commit`].
///
/// Use the constructors ([`add_file`](Self::add_file), [`add_bytes`](Self::add_bytes),
/// [`delete`](Self::delete)) — they read more linearly than the bare variants and
/// pick the right [`AddSource`] for you. See [`AddSource`] for the trade-off
/// between path-backed and in-memory uploads.
///
/// ```rust,no_run
/// use hf_hub::repository::CommitOperation;
///
/// let ops = vec![
///     CommitOperation::add_file("model.safetensors", "/local/path/model.safetensors"),
///     CommitOperation::add_bytes("config.json", br#"{"vocab_size":50257}"#.as_slice()),
///     CommitOperation::delete("old/checkpoint.bin"),
/// ];
/// # let _ = ops;
/// ```
#[derive(Debug, Clone)]
pub enum CommitOperation {
    /// Add or replace a file in the repository.
    Add {
        /// Destination path within the repository.
        path_in_repo: String,
        /// Source of the uploaded contents.
        source: AddSource,
    },
    /// Delete a file from the repository.
    Delete {
        /// Repository-relative path to remove.
        path_in_repo: String,
    },
}

impl CommitOperation {
    /// Add an operation backed by a local file path. Equivalent to
    /// `Add { path_in_repo, source: AddSource::file(source) }`. Prefer this
    /// for any non-trivial file size — see [`AddSource::File`] for the streaming
    /// behavior and memory cost.
    #[cfg(not(target_family = "wasm"))]
    pub fn add_file(path_in_repo: impl Into<String>, source: impl Into<PathBuf>) -> Self {
        CommitOperation::Add {
            path_in_repo: path_in_repo.into(),
            source: AddSource::file(source),
        }
    }

    /// Add operation backed by in-memory bytes. Equivalent to
    /// `Add { path_in_repo, source: AddSource::bytes(source) }`. The bytes are
    /// retained in the operation and re-read at commit time — see
    /// [`AddSource::Bytes`] for the memory cost.
    pub fn add_bytes(path_in_repo: impl Into<String>, source: impl Into<Bytes>) -> Self {
        CommitOperation::Add {
            path_in_repo: path_in_repo.into(),
            source: AddSource::bytes(source),
        }
    }

    /// Delete operation for a repository path.
    pub fn delete(path_in_repo: impl Into<String>) -> Self {
        CommitOperation::Delete {
            path_in_repo: path_in_repo.into(),
        }
    }
}

/// Stream of byte chunks produced by a [`StreamSource`].
///
/// `Send` on native targets; the bound is dropped on wasm, where streams
/// typically wrap thread-bound JS values.
#[cfg(not(target_family = "wasm"))]
pub type SourceByteStream = Pin<Box<dyn Stream<Item = HFResult<Bytes>> + Send>>;
#[cfg(target_family = "wasm")]
pub type SourceByteStream = Pin<Box<dyn Stream<Item = HFResult<Bytes>>>>;

/// Factory that produces a fresh [`SourceByteStream`] each time `open` is called.
///
/// The upload pipeline reads a source up to three times — a small "sample"
/// prefix for the preupload classification, the full content for the LFS SHA-256
/// hash, and the full content for the xet upload — so a one-shot stream isn't
/// enough. Implementations must be cheap to re-open and produce identical bytes
/// across opens.
///
/// The trait requires `Send + Sync` so that [`AddSource`] (and therefore
/// `CommitOperation::Add { source }`) can be moved between tasks.
pub trait StreamFactory: Send + Sync {
    /// Produce a fresh stream over the source's bytes.
    fn open(&self) -> SourceByteStream;
}

/// Source backed by a re-readable stream of byte chunks of known length.
///
/// Use this when the bytes are too large to materialize in memory at once.
/// The upload pipeline only ever holds one chunk in flight at a time.
#[derive(Clone)]
pub struct StreamSource {
    factory: Arc<dyn StreamFactory>,
    size: u64,
}

impl StreamSource {
    /// Construct a [`StreamSource`] from a factory and a known total byte length.
    ///
    /// `size` must equal the total number of bytes produced by `factory.open()`,
    /// which must be the same across every invocation.
    pub fn new(factory: Arc<dyn StreamFactory>, size: u64) -> Self {
        Self { factory, size }
    }

    /// Total number of bytes the stream will produce.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Open a fresh byte stream from the source.
    pub fn open(&self) -> SourceByteStream {
        self.factory.open()
    }
}

impl std::fmt::Debug for StreamSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamSource")
            .field("size", &self.size)
            .finish_non_exhaustive()
    }
}

/// Source of content for a [`CommitOperation::Add`] operation.
///
/// Use [`File`](Self::File) for content already on disk, especially large files:
/// the path is stored, but file contents are read when the commit runs and are
/// streamed where possible. Use [`Bytes`](Self::Bytes) for small in-memory content;
/// the buffer is owned by the operation and may be cloned for LFS/xet uploads.
///
/// `AddSource::File` is not a snapshot: the file must still exist, unchanged,
/// when [`HFRepository::create_commit`] runs.
#[derive(Debug, Clone)]
pub enum AddSource {
    /// Read file contents from this local path at commit time.
    #[cfg(not(target_family = "wasm"))]
    File(PathBuf),

    /// In-memory contents used as the file body. Stored as [`bytes::Bytes`] so the
    /// internal `.clone()` calls along the upload path are cheap refcount bumps
    /// rather than full memory copies.
    Bytes(Bytes),

    /// Lazily-read byte stream of known total length. Lets the upload pipeline
    /// process a source whose total size exceeds available memory: only one
    /// chunk is held in flight at any time. See [`StreamSource`] for the
    /// factory contract (must be re-readable to satisfy the sample, hash, and
    /// upload passes).
    Stream(StreamSource),
}

impl AddSource {
    /// Construct an [`AddSource::File`] from a local file path.
    #[cfg(not(target_family = "wasm"))]
    pub fn file(path: impl Into<PathBuf>) -> Self {
        AddSource::File(path.into())
    }

    /// Construct an [`AddSource::Bytes`] from in-memory contents.
    ///
    /// `bytes` accepts any type that converts to [`bytes::Bytes`]: `Vec<u8>` is
    /// taken by value with no copy (via `Bytes::from`), `&'static [u8]` /
    /// `&'static str` are wrapped without copying, and an existing [`Bytes`]
    /// passes through unchanged.
    pub fn bytes(bytes: impl Into<Bytes>) -> Self {
        AddSource::Bytes(bytes.into())
    }

    /// Construct an [`AddSource::Stream`] from a [`StreamSource`].
    ///
    /// The total length must be known up-front and the factory must produce
    /// identical bytes on every `open()` — see [`StreamSource::new`].
    pub fn stream(source: StreamSource) -> Self {
        AddSource::Stream(source)
    }
}

pub(super) fn extract_etag(response: &reqwest::Response) -> Option<String> {
    let headers = response.headers();
    let raw = headers
        .get(constants::HEADER_X_LINKED_ETAG)
        .or_else(|| headers.get(reqwest::header::ETAG))
        .and_then(|v| v.to_str().ok())?;
    let normalized = raw.strip_prefix("W/").unwrap_or(raw);
    Some(normalized.trim_matches('"').to_string())
}

pub(super) fn extract_commit_hash(response: &reqwest::Response) -> Option<String> {
    response
        .headers()
        .get(constants::HEADER_X_REPO_COMMIT)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
}

pub(crate) fn extract_file_size(response: &reqwest::Response) -> Option<u64> {
    let headers = response.headers();
    // `X-Linked-Size` is the authoritative size of the resolved file and is
    // safe to trust on any response, including redirects from the `resolve`
    // endpoint. `Content-Length`, however, describes the body of *this*
    // response — on a 3xx redirect that is the length of the redirect payload,
    // not the file. Falling back to it there yields a bogus size (e.g. 341 for
    // a redirect whose target file is 818 bytes), which later trips the
    // post-download `bytes_read != total_bytes` check. Only fall back to
    // `Content-Length` for non-redirect responses.
    if let Some(linked) = headers
        .get(constants::HEADER_X_LINKED_SIZE)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse().ok())
    {
        return Some(linked);
    }
    if response.status().is_redirection() {
        return None;
    }
    headers
        .get(reqwest::header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse().ok())
}

pub(crate) fn extract_xet_hash(response: &reqwest::Response) -> Option<String> {
    response
        .headers()
        .get(constants::HEADER_X_XET_HASH)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
}

/// Check if a path matches any of the given glob patterns using the `globset` crate.
pub(super) fn matches_any_glob(patterns: &[String], path: &str) -> bool {
    patterns.iter().any(|p| {
        globset::Glob::new(p)
            .ok()
            .map(|g| g.compile_matcher().is_match(path))
            .unwrap_or(false)
    })
}

#[cfg(test)]
mod tests {
    use super::{BlobSecurityInfo, CommitInfo, FileMetadataInfo, RepoTreeEntry, extract_file_size};

    fn response_with(status: u16, headers: &[(&str, &str)]) -> reqwest::Response {
        let mut builder = http::Response::builder().status(status);
        for (k, v) in headers {
            builder = builder.header(*k, *v);
        }
        reqwest::Response::from(builder.body("").unwrap())
    }

    #[test]
    fn extract_file_size_prefers_linked_size() {
        let resp = response_with(200, &[("x-linked-size", "818"), ("content-length", "341")]);
        assert_eq!(extract_file_size(&resp), Some(818));
    }

    #[test]
    fn extract_file_size_uses_content_length_for_non_redirect() {
        let resp = response_with(200, &[("content-length", "818")]);
        assert_eq!(extract_file_size(&resp), Some(818));
    }

    #[test]
    fn extract_file_size_ignores_content_length_on_redirect() {
        // A 307 from the `resolve` endpoint carries a `content-length` for the
        // redirect body, not the target file. Without `x-linked-size` we must
        // not trust it, otherwise the post-download size check fails.
        let resp = response_with(307, &[("content-length", "341")]);
        assert_eq!(extract_file_size(&resp), None);
    }

    #[test]
    fn extract_file_size_trusts_linked_size_on_redirect() {
        let resp = response_with(307, &[("x-linked-size", "818"), ("content-length", "341")]);
        assert_eq!(extract_file_size(&resp), Some(818));
    }

    #[test]
    fn test_repo_tree_entry_deserialize_file() {
        let json = r#"{"type":"file","oid":"abc123","size":100,"path":"test.txt"}"#;
        let entry: RepoTreeEntry = serde_json::from_str(json).unwrap();
        match entry {
            RepoTreeEntry::File {
                path,
                size,
                xet_hash,
                security,
                ..
            } => {
                assert_eq!(path, "test.txt");
                assert_eq!(size, 100);
                assert!(xet_hash.is_none());
                assert!(security.is_none());
            }
            _ => panic!("Expected File variant"),
        }
    }

    #[test]
    fn test_repo_tree_entry_file_expanded() {
        let json = r#"{
            "type":"file","oid":"abc123","size":100,"path":"weights.safetensors",
            "xetHash":"xet-deadbeef",
            "securityFileStatus":{"status":"safe","avScan":{"virusFound":false},"pickleImportScan":null},
            "lastCommit":{"id":"sha","title":"t","date":"2025-01-01T00:00:00Z"}
        }"#;
        let entry: RepoTreeEntry = serde_json::from_str(json).unwrap();
        match entry {
            RepoTreeEntry::File {
                xet_hash,
                security,
                last_commit,
                ..
            } => {
                assert_eq!(xet_hash.as_deref(), Some("xet-deadbeef"));
                let security = security.unwrap();
                assert_eq!(security.status, "safe");
                assert!(security.av_scan.is_some());
                assert!(security.pickle_import_scan.is_none());
                assert!(last_commit.is_some());
            }
            _ => panic!("Expected File variant"),
        }
    }

    #[test]
    fn test_repo_tree_entry_directory_with_last_commit() {
        let json = r#"{"type":"directory","oid":"def456","path":"src","lastCommit":{"id":"sha","title":"t","date":"2025-01-01T00:00:00Z"}}"#;
        let entry: RepoTreeEntry = serde_json::from_str(json).unwrap();
        match entry {
            RepoTreeEntry::Directory {
                path, last_commit, ..
            } => {
                assert_eq!(path, "src");
                assert!(last_commit.is_some());
            }
            _ => panic!("Expected Directory variant"),
        }
    }

    #[test]
    fn test_repo_tree_entry_deserialize_directory() {
        let json = r#"{"type":"directory","oid":"def456","path":"src"}"#;
        let entry: RepoTreeEntry = serde_json::from_str(json).unwrap();
        match entry {
            RepoTreeEntry::Directory {
                path, last_commit, ..
            } => {
                assert_eq!(path, "src");
                assert!(last_commit.is_none());
            }
            _ => panic!("Expected Directory variant"),
        }
    }

    #[test]
    fn test_blob_security_info_unsafe_status() {
        let json = r#"{"status":"suspicious","avScan":null,"pickleImportScan":{"matches":[]}}"#;
        let info: BlobSecurityInfo = serde_json::from_str(json).unwrap();
        assert_eq!(info.status, "suspicious");
        assert!(info.av_scan.is_none());
        assert!(info.pickle_import_scan.is_some());
    }

    #[test]
    fn test_commit_info_with_pr() {
        let json = r#"{
            "commitUrl":"https://huggingface.co/owner/repo/commit/abc123",
            "commitOid":"abc123",
            "prUrl":"https://huggingface.co/owner/repo/discussions/7",
            "prNum":7
        }"#;
        let info: CommitInfo = serde_json::from_str(json).unwrap();
        assert_eq!(info.commit_oid.as_deref(), Some("abc123"));
        assert_eq!(
            info.pr_url.as_deref(),
            Some("https://huggingface.co/owner/repo/discussions/7")
        );
        assert_eq!(info.pr_num, Some(7));
    }

    #[test]
    fn test_commit_info_no_pr() {
        let json =
            r#"{"commitUrl":"https://huggingface.co/owner/repo/commit/abc","commitOid":"abc"}"#;
        let info: CommitInfo = serde_json::from_str(json).unwrap();
        assert_eq!(
            info.commit_url.as_deref(),
            Some("https://huggingface.co/owner/repo/commit/abc")
        );
        assert!(info.pr_url.is_none());
        assert!(info.pr_num.is_none());
    }

    #[test]
    fn test_file_metadata_info_location_round_trip() {
        let original = FileMetadataInfo {
            filename: "config.json".into(),
            etag: "abc".into(),
            commit_hash: "deadbeef".into(),
            xet_hash: None,
            file_size: 100,
            location: Some("https://cdn-lfs.huggingface.co/repos/.../config.json".into()),
        };
        let json = serde_json::to_string(&original).unwrap();
        assert!(json.contains("location"));
        let round_tripped: FileMetadataInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(round_tripped.location, original.location);
    }

    #[test]
    fn test_file_metadata_info_location_optional() {
        let json = r#"{"filename":"f","etag":"e","commit_hash":"c","xet_hash":null,"file_size":0}"#;
        let info: FileMetadataInfo = serde_json::from_str(json).unwrap();
        assert!(info.location.is_none());
    }
}
