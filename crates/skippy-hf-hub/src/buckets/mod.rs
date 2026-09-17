//! Bucket handles and file operations for Hugging Face Hub buckets.
//!
//! Buckets are flat file stores that live under a namespace such as
//! `"user/my-bucket"`. Start by creating an [`HFBucket`] handle with
//! [`HFClient::bucket`], then use the handle to inspect metadata, browse the
//! tree, upload or download files, or delete entries.
//!
//! This module exposes both high-level helpers and the lower-level batch API:
//!
//! - Use [`HFBucket::info`], [`HFBucket::list_tree`], and [`HFBucket::get_paths_info`] to inspect bucket contents.
//! - Use [`HFBucket::upload_files`] and [`HFBucket::download_files`] for common transfer workflows.
//! - Use [`HFBucket::batch_operations`] when you already have xet hashes and want direct control over add, delete, or
//!   copy operations.
//! - Use [`sync`] for one-way directory mirroring between a local folder and a bucket prefix.

#[cfg(not(target_family = "wasm"))]
pub mod sync;

#[cfg(not(target_family = "wasm"))]
use std::path::PathBuf;

use bon::bon;
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use url::Url;

use crate::client::HFClient;
use crate::error::{HFError, HFResult, NotFoundContext};
use crate::progress::{DownloadEvent, EmitEvent, Progress, UploadEvent};
use crate::repository::download::{HFByteStream, wrap_stream_with_progress};
use crate::retry;

const BUCKET_BATCH_CHUNK_SIZE: usize = 1000;
const BUCKET_PATHS_INFO_BATCH_SIZE: usize = 1000;

/// A handle for a single bucket on the Hugging Face Hub.
///
/// `HFBucket` is created via [`HFClient::bucket`] and binds together the client,
/// owner (namespace), and bucket name. All bucket-scoped API operations are methods
/// on this type.
///
/// Cheap to clone — the inner [`HFClient`] is `Arc`-backed.
///
/// # Example
///
/// ```rust,no_run
/// # use futures::StreamExt;
/// # use hf_hub::HFClient;
/// # #[tokio::main] async fn main() -> hf_hub::HFResult<()> {
/// let client = HFClient::builder().build()?;
/// let bucket = client.bucket("my-org", "my-bucket");
///
/// let info = bucket.info().send().await?;
/// println!("bucket {} ({} files)", info.id, info.total_files);
///
/// let stream = bucket.list_tree().send()?;
/// futures::pin_mut!(stream);
/// while let Some(entry) = stream.next().await {
///     println!("{:?}", entry?);
/// }
/// # Ok(()) }
/// ```
#[derive(Clone)]
pub struct HFBucket {
    pub(crate) hf_client: HFClient,
    pub(super) owner: String,
    pub(super) name: String,
}

impl std::fmt::Debug for HFBucket {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HFBucket")
            .field("owner", &self.owner)
            .field("name", &self.name)
            .finish()
    }
}

impl HFBucket {
    /// Construct a new bucket handle. Prefer [`HFClient::bucket`] in most cases.
    pub fn new(client: HFClient, owner: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            hf_client: client,
            owner: owner.into(),
            name: name.into(),
        }
    }

    /// Return a reference to the underlying [`HFClient`].
    pub fn client(&self) -> &HFClient {
        &self.hf_client
    }

    /// The bucket owner (user or organization namespace).
    pub fn owner(&self) -> &str {
        &self.owner
    }

    /// The bucket name (without owner prefix).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The full `"owner/name"` bucket identifier used in Hub API calls.
    pub fn bucket_id(&self) -> String {
        format!("{}/{}", self.owner, self.name)
    }
}

#[bon]
impl HFBucket {
    /// Get metadata about this bucket.
    ///
    /// Endpoint: `GET /api/buckets/{bucket_id}`.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn info(&self) -> HFResult<BucketInfo> {
        let bucket_id = self.bucket_id();
        let url = self.hf_client.bucket_api_url(&bucket_id);

        let headers = self.hf_client.auth_headers();
        let response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client
                .http_client()
                .get(&url)
                .headers(headers.clone())
                .send()
        })
        .await?;

        let response = self
            .hf_client
            .check_response(response, Some(&bucket_id), NotFoundContext::Bucket)
            .await?;
        Ok(response.json().await?)
    }

    /// List files and directories in this bucket.
    ///
    /// This is the main API for browsing bucket contents. For targeted lookups of a known set of
    /// paths, prefer [`HFBucket::get_paths_info`].
    ///
    /// Endpoint: `GET /api/buckets/{bucket_id}/tree[/{prefix}]` (paginated).
    ///
    /// # Parameters
    ///
    /// - `prefix`: filter results to entries under this prefix.
    /// - `recursive` (default `false`): traverse subdirectories.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn list_tree(
        &self,
        /// Filter results to entries under this prefix.
        #[builder(into)]
        prefix: Option<String>,
        /// Traverse subdirectories.
        #[builder(default)]
        recursive: bool,
    ) -> HFResult<impl Stream<Item = HFResult<BucketTreeEntry>> + '_> {
        let bucket_id = self.bucket_id();
        let url_str = format!(
            "{}/api/buckets/{}/tree",
            self.hf_client.endpoint(),
            bucket_id
        );
        let mut url = Url::parse(&url_str)?;
        if let Some(ref prefix) = prefix {
            crate::client::append_path_segments(&mut url, prefix)?;
        }

        let mut query = vec![];
        if recursive {
            query.push(("recursive".to_string(), "true".to_string()));
        }

        Ok(self.hf_client.paginate(url, query, None))
    }

    /// Get info about specific paths in this bucket.
    ///
    /// This is useful when you already know which paths you care about and do not want to stream
    /// the full tree. Requests are automatically chunked in batches of 1000 paths.
    ///
    /// Endpoint: `POST /api/buckets/{bucket_id}/paths-info`.
    ///
    /// # Parameters
    ///
    /// - `paths` (required): paths to inspect.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn get_paths_info(
        &self,
        /// Paths to inspect.
        paths: Vec<String>,
    ) -> HFResult<Vec<BucketTreeEntry>> {
        let bucket_id = self.bucket_id();
        let url = format!(
            "{}/api/buckets/{}/paths-info",
            self.hf_client.endpoint(),
            bucket_id
        );

        let headers = self.hf_client.auth_headers();
        let mut all_entries = Vec::new();
        for chunk in paths.chunks(BUCKET_PATHS_INFO_BATCH_SIZE) {
            let body = serde_json::json!({ "paths": chunk });

            let response = retry::retry(self.hf_client.retry_config(), || {
                self.hf_client
                    .http_client()
                    .post(&url)
                    .headers(headers.clone())
                    .json(&body)
                    .send()
            })
            .await?;

            let response = self
                .hf_client
                .check_response(response, Some(&bucket_id), NotFoundContext::Bucket)
                .await?;
            let entries: Vec<BucketTreeEntry> = response.json().await?;
            all_entries.extend(entries);
        }

        Ok(all_entries)
    }

    /// Stream the bytes of a single file in this bucket without writing to disk.
    ///
    /// Parallel of
    /// [`HFRepository::download_file_stream`](crate::repository::HFRepository::download_file_stream) for buckets.
    /// Returns `(content_length, stream)` — `content_length` reflects the file size reported by `paths-info`.
    /// Xet-backed files dispatch through xet streaming; non-xet files fall back to a direct GET on the bucket's
    /// `resolve` URL.
    ///
    /// Endpoint (non-xet branch): `GET {endpoint}/buckets/{bucket_id}/resolve/{remote_path}`.
    ///
    /// # Parameters
    ///
    /// - `remote_path` (required): file path within the bucket.
    /// - `progress`: optional progress handler. `Start` is emitted before the stream is returned; `Progress` is emitted
    ///   as the caller polls each chunk; `Complete` is emitted when the stream is exhausted.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn download_file_stream(
        &self,
        /// File path within the bucket.
        #[builder(into)]
        remote_path: String,
        /// Progress handler. `Start` is emitted before the stream is returned; `Progress` is
        /// emitted as the caller polls each chunk; `Complete` is emitted when the stream is
        /// exhausted.
        #[builder(into)]
        progress: Option<Progress>,
    ) -> HFResult<(Option<u64>, HFByteStream)> {
        Box::pin(self.download_file_stream_impl(remote_path, progress)).await
    }

    /// Get metadata for a single file in this bucket via a HEAD request.
    ///
    /// Endpoint: `HEAD /buckets/{bucket_id}/resolve/{path}`.
    ///
    /// # Parameters
    ///
    /// - `remote_path` (required): file path within the bucket.
    #[cfg(not(target_family = "wasm"))]
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn get_file_metadata(
        &self,
        /// File path within the bucket.
        #[builder(into)]
        remote_path: String,
    ) -> HFResult<BucketFileMetadata> {
        let bucket_id = self.bucket_id();
        let mut url = Url::parse(&format!(
            "{}/buckets/{}/resolve",
            self.hf_client.endpoint(),
            bucket_id
        ))?;
        crate::client::append_path_segments(&mut url, &remote_path)?;

        let headers = self.hf_client.auth_headers();
        let response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client
                .no_redirect_client()
                .head(url.clone())
                .headers(headers.clone())
                .send()
        })
        .await?;

        let response = self
            .hf_client
            .check_response(
                response,
                Some(&bucket_id),
                NotFoundContext::Entry { path: remote_path },
            )
            .await?;

        let size = response
            .headers()
            .get(reqwest::header::CONTENT_LENGTH)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0);

        let xet_hash = response
            .headers()
            .get("x-xet-hash")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();

        Ok(BucketFileMetadata { size, xet_hash })
    }

    /// Execute batch file operations (add, delete, copy) on this bucket.
    ///
    /// This is the low-level file mutation API. `add` operations only register metadata on the
    /// bucket side; the file contents must already have been uploaded to xet so each entry has a
    /// valid [`BucketAddFile::xet_hash`].
    ///
    /// For simpler upload and download flows, prefer [`HFBucket::upload_files`] and
    /// [`HFBucket::download_files`].
    ///
    /// Endpoint: `POST /api/buckets/{bucket_id}/batch` (NDJSON). Operations are chunked at 1000
    /// entries per request.
    ///
    /// All paths in `add`, `delete`, and `copy` are **bucket-relative**, slash-separated paths
    /// (e.g., `"data/train/0001.bin"`) — no leading slash, forward slashes regardless of platform,
    /// no `bucket_id` prefix.
    ///
    /// # Parameters
    ///
    /// - `add_files`: files to register in the bucket. Each [`BucketAddFile`] requires:
    ///   - `path` (`String`): bucket-relative destination path.
    ///   - `xet_hash` (`String`): xet content hash from a prior xet upload — the bytes must already be in xet storage;
    ///     this call only registers metadata.
    ///   - `size` (`u64`): file size in bytes.
    ///   - `mtime` (`Option<u64>`): last modification time as a Unix timestamp in seconds.
    ///   - `content_type` (`Option<String>`): MIME type (e.g., `"text/plain"`).
    /// - `delete`: bucket-relative paths to remove from the bucket.
    /// - `copy`: server-side copies into this bucket. Each [`BucketCopyFile`] requires:
    ///   - `path` (`String`): bucket-relative destination path.
    ///   - `xet_hash` (`String`): xet content hash of the source bytes (already present in xet storage from another
    ///     repo or bucket — copies are by-hash, no data is transferred).
    ///   - `source_repo_type` ([`BucketCopySourceType`]): repo type of the source — one of `Bucket`, `Model`,
    ///     `Dataset`, or `Space`. Serializes to the lowercase wire string the Hub expects.
    ///   - `source_repo_id` (`String`): full source identifier in `"namespace/name"` form (e.g., `"user/my-bucket"`).
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn batch_operations(
        &self,
        /// Files to register in the bucket.
        #[builder(default)]
        add_files: Vec<BucketAddFile>,
        /// Bucket-relative paths to remove from the bucket.
        #[builder(default)]
        delete: Vec<String>,
        /// Server-side copies into this bucket.
        #[builder(default)]
        copy: Vec<BucketCopyFile>,
    ) -> HFResult<()> {
        let bucket_id = self.bucket_id();
        let url = format!(
            "{}/api/buckets/{}/batch",
            self.hf_client.endpoint(),
            bucket_id
        );

        let mut lines: Vec<String> = Vec::new();

        for add in &add_files {
            let mut obj = serde_json::json!({
                "type": "addFile",
                "path": add.path,
                "xetHash": add.xet_hash,
                "size": add.size,
            });
            if let Some(mtime) = add.mtime {
                obj["mtime"] = serde_json::Value::Number(mtime.into());
            }
            if let Some(ref ct) = add.content_type {
                obj["contentType"] = serde_json::Value::String(ct.clone());
            }
            lines.push(serde_json::to_string(&obj)?);
        }

        for path in &delete {
            let obj = serde_json::json!({
                "type": "deleteFile",
                "path": path,
            });
            lines.push(serde_json::to_string(&obj)?);
        }

        for copy in &copy {
            let mut obj = serde_json::json!({
                "type": "copyFile",
                "path": copy.path,
                "xetHash": copy.xet_hash,
                "sourceRepoType": copy.source_repo_type.as_str(),
                "sourceRepoId": copy.source_repo_id,
            });
            if let Some(size) = copy.size {
                obj["size"] = serde_json::Value::Number(size.into());
            }
            if let Some(mtime) = copy.mtime {
                obj["mtime"] = serde_json::Value::Number(mtime.into());
            }
            if let Some(ref ct) = copy.content_type {
                obj["contentType"] = serde_json::Value::String(ct.clone());
            }
            lines.push(serde_json::to_string(&obj)?);
        }

        let headers = self.hf_client.auth_headers();
        for chunk in lines.chunks(BUCKET_BATCH_CHUNK_SIZE) {
            let body = chunk.join("\n") + "\n";

            let response = retry::retry(self.hf_client.retry_config(), || {
                self.hf_client
                    .http_client()
                    .post(&url)
                    .headers(headers.clone())
                    .header(reqwest::header::CONTENT_TYPE, "application/x-ndjson")
                    .body(body.clone())
                    .send()
            })
            .await?;

            self.hf_client
                .check_response(response, Some(&bucket_id), NotFoundContext::Bucket)
                .await?;
        }

        Ok(())
    }

    /// Delete files from this bucket by path.
    ///
    /// Convenience wrapper around [`HFBucket::batch_operations`] that sends only `deleteFile` operations.
    ///
    /// # Parameters
    ///
    /// - `paths` (required): paths to delete from the bucket.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn delete_files(
        &self,
        /// Paths to delete from the bucket.
        paths: Vec<String>,
    ) -> HFResult<()> {
        self.batch_operations().delete(paths).send().await
    }

    /// Upload an arbitrary set of [`AddSource`](crate::repository::AddSource)-backed files to the bucket.
    ///
    /// More general analog of [`HFBucket::upload_files`] (which only reads from local
    /// paths): each pair can use any [`AddSource`](crate::repository::AddSource)
    /// variant — `Bytes` for in-memory content, `File` for a local path, or `Stream`
    /// for sources too large to materialize at once. For the common in-memory case,
    /// build entries with
    /// [`AddSource::bytes`](crate::repository::AddSource::bytes):
    ///
    /// ```rust,no_run
    /// # use hf_hub::repository::AddSource;
    /// # let _: Vec<(String, AddSource)> =
    /// vec![("logs/run-1.txt".into(), AddSource::bytes(b"hello".as_slice()))]
    /// # ;
    /// ```
    ///
    /// Goes through the same preupload + xet pipeline as [`HFBucket::upload_files`].
    ///
    /// # Parameters
    ///
    /// - `files` (required): list of `(remote_path, source)` pairs.
    /// - `progress`: optional progress handler.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn upload_source_files(
        &self,
        /// List of `(remote_path, source)` pairs.
        files: Vec<(String, crate::repository::AddSource)>,
        /// Progress handler.
        #[builder(into)]
        progress: Option<Progress>,
    ) -> HFResult<()> {
        Box::pin(self.upload_sources_impl(files, progress)).await
    }

    /// Upload local files to the bucket.
    ///
    /// File contents are uploaded to xet first, then registered in the bucket via the batch
    /// endpoint. Each entry pairs a local source with a bucket-relative destination via the
    /// [`BucketUpload`] struct (use [`BucketUpload::new`] to construct one).
    ///
    /// # Parameters
    ///
    /// - `files` (required): list of [`BucketUpload`] entries describing each `local` → `remote` mapping.
    /// - `progress`: optional progress handler.
    #[cfg(not(target_family = "wasm"))]
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn upload_files(
        &self,
        /// List of [`BucketUpload`] entries describing each `local` → `remote` mapping.
        files: Vec<BucketUpload>,
        /// Progress handler.
        #[builder(into)]
        progress: Option<Progress>,
    ) -> HFResult<()> {
        Box::pin(self.upload_files_impl(files, progress)).await
    }

    /// Download files from the bucket to local paths.
    ///
    /// The method first resolves xet metadata for each `remote` path, then downloads the file
    /// contents through xet. Directory entries are rejected. Each entry pairs a bucket-relative
    /// source with a local destination via the [`BucketDownload`] struct (use
    /// [`BucketDownload::new`] to construct one).
    ///
    /// # Parameters
    ///
    /// - `files` (required): list of [`BucketDownload`] entries describing each `remote` → `local` mapping.
    /// - `progress`: optional progress handler.
    #[cfg(not(target_family = "wasm"))]
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn download_files(
        &self,
        /// List of [`BucketDownload`] entries describing each `remote` → `local` mapping.
        files: Vec<BucketDownload>,
        /// Progress handler.
        #[builder(into)]
        progress: Option<Progress>,
    ) -> HFResult<()> {
        Box::pin(self.download_files_impl(files, progress)).await
    }
}

impl HFBucket {
    /// Body of [`HFBucket::download_file_stream`], extracted so the builder `send()` future only
    /// holds a boxed pointer to this state machine rather than embedding it inline.
    async fn download_file_stream_impl(
        &self,
        remote_path: String,
        progress: Option<Progress>,
    ) -> HFResult<(Option<u64>, HFByteStream)> {
        let bucket_id = self.bucket_id();

        let entries = self
            .get_paths_info()
            .paths(vec![remote_path.clone()])
            .send()
            .await?;
        let (xet_hash, file_size) = entries
            .into_iter()
            .find_map(|e| match e {
                BucketTreeEntry::File {
                    path,
                    size,
                    xet_hash,
                    ..
                } if path == remote_path => Some((xet_hash, size)),
                _ => None,
            })
            .ok_or_else(|| HFError::EntryNotFound {
                path: remote_path.clone(),
                repo_id: bucket_id.clone(),
                context: None,
            })?;

        if !xet_hash.is_empty() {
            let stream = self.xet_download_stream(&xet_hash, file_size).await?;
            progress.emit(DownloadEvent::Start {
                total_files: 1,
                total_bytes: file_size,
            });
            let boxed: HFByteStream = Box::new(Box::pin(stream));
            let wrapped = wrap_stream_with_progress(boxed, progress, remote_path, file_size);
            #[cfg(target_family = "wasm")]
            let wrapped = crate::repository::download::buffer_wasm_stream(wrapped);
            return Ok((Some(file_size), wrapped));
        }

        let mut url = Url::parse(&format!(
            "{}/buckets/{}/resolve",
            self.hf_client.endpoint(),
            bucket_id
        ))?;
        crate::client::append_path_segments(&mut url, &remote_path)?;
        let headers = self.hf_client.auth_headers();
        let response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client
                .http_client()
                .get(url.clone())
                .headers(headers.clone())
                .send()
        })
        .await?;
        let response = self
            .hf_client
            .check_response(
                response,
                Some(&bucket_id),
                NotFoundContext::Entry {
                    path: remote_path.clone(),
                },
            )
            .await?;

        let content_length = response.content_length().or(Some(file_size));
        let total_bytes = content_length.unwrap_or(file_size);
        let stream = response.bytes_stream().map(|r| r.map_err(HFError::from));
        progress.emit(DownloadEvent::Start {
            total_files: 1,
            total_bytes,
        });
        let boxed: HFByteStream = Box::new(Box::pin(stream));
        let wrapped = wrap_stream_with_progress(boxed, progress, remote_path, total_bytes);
        #[cfg(target_family = "wasm")]
        let wrapped = crate::repository::download::buffer_wasm_stream(wrapped);
        Ok((content_length, wrapped))
    }

    /// Body of [`HFBucket::upload_files`], extracted so the builder `send()` future only holds a
    /// boxed pointer to this state machine rather than embedding it inline.
    #[cfg(not(target_family = "wasm"))]
    async fn upload_files_impl(
        &self,
        files: Vec<BucketUpload>,
        progress: Option<Progress>,
    ) -> HFResult<()> {
        if files.is_empty() {
            return Ok(());
        }

        let total_bytes: u64 = files
            .iter()
            .filter_map(|f| std::fs::metadata(&f.local).ok())
            .map(|m| m.len())
            .sum();
        progress.emit(UploadEvent::Start {
            total_files: files.len(),
            total_bytes,
        });

        let xet_files: Vec<(String, crate::repository::AddSource)> = files
            .iter()
            .map(|f| {
                (
                    f.remote.clone(),
                    crate::repository::AddSource::file(f.local.clone()),
                )
            })
            .collect();

        let xet_infos = self.xet_upload(&xet_files, &progress).await?;

        let add_files: Vec<BucketAddFile> = files
            .iter()
            .zip(xet_infos.iter())
            .map(|(f, xet_info)| {
                let metadata = std::fs::metadata(&f.local).ok();
                let size = metadata
                    .as_ref()
                    .map(|m| m.len())
                    .or(xet_info.file_size)
                    .unwrap_or(0);
                let mtime = metadata
                    .as_ref()
                    .and_then(|m| m.modified().ok())
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| d.as_secs());

                BucketAddFile {
                    path: f.remote.clone(),
                    xet_hash: xet_info.hash.clone(),
                    size,
                    mtime,
                    content_type: None,
                }
            })
            .collect();

        self.batch_operations().add_files(add_files).send().await?;

        progress.emit(UploadEvent::Complete);
        Ok(())
    }

    /// Body of [`HFBucket::download_files`], extracted so the builder `send()` future only holds a
    /// boxed pointer to this state machine rather than embedding it inline.
    #[cfg(not(target_family = "wasm"))]
    async fn download_files_impl(
        &self,
        files: Vec<BucketDownload>,
        progress: Option<Progress>,
    ) -> HFResult<()> {
        if files.is_empty() {
            return Ok(());
        }

        let remote_paths: Vec<String> = files.iter().map(|f| f.remote.clone()).collect();
        let entries = self.get_paths_info().paths(remote_paths).send().await?;

        let entry_map: std::collections::HashMap<String, BucketTreeEntry> = entries
            .into_iter()
            .map(|e| {
                let path = match &e {
                    BucketTreeEntry::File { path, .. } => path.clone(),
                    BucketTreeEntry::Directory { path, .. } => path.clone(),
                };
                (path, e)
            })
            .collect();

        let mut xet_batch_files = Vec::new();
        let mut total_bytes: u64 = 0;

        for f in &files {
            match entry_map.get(&f.remote) {
                Some(BucketTreeEntry::File { xet_hash, size, .. }) => {
                    total_bytes += size;
                    xet_batch_files.push(crate::xet::XetBatchFile {
                        hash: xet_hash.clone(),
                        file_size: *size,
                        path: f.local.clone(),
                        filename: f.remote.clone(),
                    });
                }
                Some(BucketTreeEntry::Directory { path, .. }) => {
                    return Err(HFError::InvalidParameter(format!(
                        "Cannot download directory entry: {path}"
                    )));
                }
                None => {
                    return Err(HFError::EntryNotFound {
                        path: f.remote.clone(),
                        repo_id: self.bucket_id(),
                        context: None,
                    });
                }
            }
        }

        progress.emit(DownloadEvent::Start {
            total_files: xet_batch_files.len(),
            total_bytes,
        });

        self.xet_download_batch(&xet_batch_files, &progress).await?;

        progress.emit(DownloadEvent::Complete);
        Ok(())
    }

    /// Shared implementation: upload pre-built `(remote_path, AddSource)` pairs through xet
    /// and register them in the bucket. Used by [`HFBucket::upload_source_files`].
    async fn upload_sources_impl(
        &self,
        uploads: Vec<(String, crate::repository::AddSource)>,
        progress: Option<Progress>,
    ) -> HFResult<()> {
        if uploads.is_empty() {
            return Ok(());
        }

        let total_bytes: u64 = uploads
            .iter()
            .map(|(_, source)| match source {
                crate::repository::AddSource::Bytes(b) => b.len() as u64,
                crate::repository::AddSource::Stream(s) => s.size(),
                #[cfg(not(target_family = "wasm"))]
                crate::repository::AddSource::File(p) => {
                    std::fs::metadata(p).map(|m| m.len()).unwrap_or(0)
                }
            })
            .sum();
        progress.emit(UploadEvent::Start {
            total_files: uploads.len(),
            total_bytes,
        });

        let xet_infos = self.xet_upload(&uploads, &progress).await?;

        let add_files: Vec<BucketAddFile> = uploads
            .iter()
            .zip(xet_infos.iter())
            .map(|((remote, _source), xet_info)| BucketAddFile {
                path: remote.clone(),
                xet_hash: xet_info.hash.clone(),
                size: xet_info.file_size.unwrap_or(0),
                mtime: None,
                content_type: None,
            })
            .collect();

        self.batch_operations().add_files(add_files).send().await?;

        progress.emit(UploadEvent::Complete);
        Ok(())
    }
}

/// A file to register in a bucket via the batch endpoint.
///
/// Represents an `addFile` entry in the NDJSON batch payload.
/// The file content must have already been uploaded to xet to obtain the `xet_hash`.
/// Used as input to [`HFBucket::batch_operations`].
#[derive(Debug, Clone)]
pub struct BucketAddFile {
    /// Destination path in the bucket.
    pub path: String,
    /// Xet content hash from a prior upload.
    pub xet_hash: String,
    /// File size in bytes.
    pub size: u64,
    /// Last modification time as a Unix timestamp (seconds).
    pub mtime: Option<u64>,
    /// MIME content type (e.g., `"text/plain"`, `"application/octet-stream"`).
    pub content_type: Option<String>,
}

/// Source repo type for a [`BucketCopyFile`].
///
/// Tells the batch endpoint where the source bytes for a server-side copy live.
/// Serializes to the lowercase string the Hub expects (`"bucket"`, `"model"`,
/// `"dataset"`, `"space"`) so callers can't typo the field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BucketCopySourceType {
    /// Source bytes live in another bucket.
    Bucket,
    /// Source bytes live in a model repository.
    Model,
    /// Source bytes live in a dataset repository.
    Dataset,
    /// Source bytes live in a Space repository.
    Space,
}

impl BucketCopySourceType {
    /// Wire string sent to the Hub for this source type.
    pub fn as_str(&self) -> &'static str {
        match self {
            BucketCopySourceType::Bucket => "bucket",
            BucketCopySourceType::Model => "model",
            BucketCopySourceType::Dataset => "dataset",
            BucketCopySourceType::Space => "space",
        }
    }
}

/// A server-side copy operation for the batch endpoint.
///
/// Represents a `copyFile` entry in the NDJSON batch payload.
/// Copies are performed by xet hash — no data transfer occurs.
/// Used as input to [`HFBucket::batch_operations`].
#[derive(Debug, Clone)]
pub struct BucketCopyFile {
    /// Destination path in the bucket.
    pub path: String,
    /// Xet content hash to copy.
    pub xet_hash: String,
    /// Source repo type. See [`BucketCopySourceType`].
    pub source_repo_type: BucketCopySourceType,
    /// Source repo or bucket ID (e.g., `"user/my-bucket"`).
    pub source_repo_id: String,
    /// Source file size in bytes, when known. Forwarded to the batch endpoint as `size`.
    pub size: Option<u64>,
    /// Modification time in milliseconds since the Unix epoch, when set. Forwarded as `mtime`.
    /// Python's `_BucketCopyFile` defaults this to "now" at construction time; in Rust callers set
    /// it explicitly when they want it sent.
    pub mtime: Option<u64>,
    /// MIME content type, when known. Forwarded as `contentType`.
    pub content_type: Option<String>,
}

/// Metadata about a bucket on the Hugging Face Hub.
///
/// Returned by [`HFBucket::info`] and
/// [`HFClient::list_buckets`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BucketInfo {
    /// Full bucket identifier, e.g., `"namespace/bucket_name"`.
    pub id: String,
    /// Whether the bucket is private.
    pub private: bool,
    /// ISO 8601 creation timestamp.
    #[serde(rename = "createdAt")]
    pub created_at: String,
    /// Total size of all files in bytes.
    pub size: u64,
    /// Number of files in the bucket.
    #[serde(rename = "totalFiles")]
    pub total_files: u64,
}

/// URL returned after creating a bucket.
///
/// Returned by [`HFClient::create_bucket`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BucketUrl {
    /// Full URL to the bucket on the Hub.
    pub url: String,
}

/// A file or directory entry in a bucket tree listing.
///
/// Returned by [`HFBucket::list_tree`] and
/// [`HFBucket::get_paths_info`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum BucketTreeEntry {
    /// A file entry with content hash and size.
    File {
        /// Path within the bucket.
        path: String,
        /// File size in bytes.
        size: u64,
        /// Xet content-addressable hash.
        #[serde(rename = "xetHash")]
        xet_hash: String,
        /// Last modification time (ISO 8601), if available.
        mtime: Option<String>,
        /// Upload timestamp (ISO 8601), if available.
        uploaded_at: Option<String>,
    },
    /// A directory entry.
    Directory {
        /// Directory path within the bucket.
        path: String,
        /// Upload timestamp (ISO 8601), if available.
        uploaded_at: Option<String>,
    },
}

/// Metadata for a single file in a bucket, retrieved via HEAD request.
///
/// Returned by [`HFBucket::get_file_metadata`].
#[derive(Debug, Clone)]
pub struct BucketFileMetadata {
    /// File size in bytes.
    pub size: u64,
    /// Xet content-addressable hash.
    pub xet_hash: String,
}

/// One file to upload via [`HFBucket::upload_files`].
///
/// Pairs a local source path with the destination path inside the bucket.
/// Using a named struct (rather than a `(local, remote)` tuple) prevents the
/// two paths from being silently swapped at the call site, where a copy-paste
/// from [`BucketDownload`] would otherwise compile.
#[cfg(not(target_family = "wasm"))]
#[derive(Debug, Clone)]
pub struct BucketUpload {
    /// Path on the local filesystem (absolute or relative to the current
    /// working directory) of the file to upload.
    pub local: PathBuf,
    /// **Bucket-relative**, slash-separated destination path (e.g.,
    /// `"data/train/0001.bin"`) — no leading slash, forward slashes regardless
    /// of platform, and no `bucket_id` prefix.
    pub remote: String,
}

#[cfg(not(target_family = "wasm"))]
impl BucketUpload {
    /// Construct a [`BucketUpload`] from a local source path and a bucket-relative
    /// destination path.
    pub fn new(local: impl Into<PathBuf>, remote: impl Into<String>) -> Self {
        Self {
            local: local.into(),
            remote: remote.into(),
        }
    }
}

/// One file to download via [`HFBucket::download_files`].
///
/// Pairs a bucket-relative source path with the local destination path. Using a
/// named struct (rather than a `(remote, local)` tuple) prevents the two paths
/// from being silently swapped at the call site, where a copy-paste from
/// [`BucketUpload`] would otherwise compile.
#[cfg(not(target_family = "wasm"))]
#[derive(Debug, Clone)]
pub struct BucketDownload {
    /// **Bucket-relative**, slash-separated source path (e.g.,
    /// `"data/train/0001.bin"`) — no leading slash, forward slashes regardless
    /// of platform, and no `bucket_id` prefix. Must resolve to a file entry;
    /// directory entries return [`HFError::InvalidParameter`].
    pub remote: String,
    /// Destination path on the local filesystem. Parent directories are
    /// created as needed.
    pub local: PathBuf,
}

#[cfg(not(target_family = "wasm"))]
impl BucketDownload {
    /// Construct a [`BucketDownload`] from a bucket-relative source path and a
    /// local destination path.
    pub fn new(remote: impl Into<String>, local: impl Into<PathBuf>) -> Self {
        Self {
            remote: remote.into(),
            local: local.into(),
        }
    }
}

#[bon]
impl HFClient {
    /// Create an [`HFBucket`] handle for a bucket.
    ///
    /// The returned handle is cheap to clone and can be reused across multiple
    /// bucket-scoped operations.
    pub fn bucket(&self, owner: impl Into<String>, name: impl Into<String>) -> HFBucket {
        HFBucket::new(self.clone(), owner, name)
    }

    /// Create a new bucket on the Hub. Endpoint: `POST /api/buckets/{namespace}/{name}`.
    ///
    /// When `exist_ok` is `true`, an existing bucket is treated as success and a [`BucketUrl`] is
    /// synthesized locally.
    ///
    /// # Parameters
    ///
    /// - `namespace` (required): namespace (user or organization) that owns the bucket.
    /// - `name` (required): bucket name within the namespace.
    /// - `private` (default `false`): whether the bucket should be private.
    /// - `resource_group_id`: enterprise resource group ID.
    /// - `exist_ok` (default `false`): if `true`, do not error when the bucket already exists.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn create_bucket(
        &self,
        /// Namespace (user or organization) that owns the bucket.
        namespace: &str,
        /// Bucket name within the namespace.
        name: &str,
        /// Whether the bucket should be private.
        #[builder(default)]
        private: bool,
        /// Enterprise resource group ID.
        #[builder(into)]
        resource_group_id: Option<String>,
        /// If `true`, do not error when the bucket already exists.
        #[builder(default)]
        exist_ok: bool,
    ) -> HFResult<BucketUrl> {
        let url = format!("{}/api/buckets/{}/{}", self.endpoint(), namespace, name);

        let mut body = serde_json::json!({});
        if private {
            body["private"] = serde_json::Value::Bool(true);
        }
        if let Some(rg) = resource_group_id {
            body["resourceGroupId"] = serde_json::Value::String(rg);
        }

        let headers = self.auth_headers();
        let response = retry::retry(self.retry_config(), || {
            self.http_client()
                .post(&url)
                .headers(headers.clone())
                .json(&body)
                .send()
        })
        .await?;

        let bucket_id = format!("{}/{}", namespace, name);

        if response.status().as_u16() == 409 && exist_ok {
            return Ok(BucketUrl {
                url: format!("{}/buckets/{}", self.endpoint(), bucket_id),
            });
        }

        let response = self
            .check_response(response, Some(&bucket_id), NotFoundContext::Generic)
            .await?;
        Ok(response.json().await?)
    }

    /// Delete a bucket from the Hub. Endpoint: `DELETE /api/buckets/{bucket_id}`.
    ///
    /// # Parameters
    ///
    /// - `bucket_id` (required): bucket ID in `"owner/name"` format.
    /// - `missing_ok` (default `false`): if `true`, do not error when the bucket does not exist.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn delete_bucket(
        &self,
        /// Bucket ID in `"owner/name"` format.
        bucket_id: &str,
        /// If `true`, do not error when the bucket does not exist.
        #[builder(default)]
        missing_ok: bool,
    ) -> HFResult<()> {
        let url = self.bucket_api_url(bucket_id);

        let headers = self.auth_headers();
        let response = retry::retry(self.retry_config(), || {
            self.http_client()
                .delete(&url)
                .headers(headers.clone())
                .send()
        })
        .await?;

        if response.status().as_u16() == 404 && missing_ok {
            return Ok(());
        }

        self.check_response(response, Some(bucket_id), NotFoundContext::Bucket)
            .await?;
        Ok(())
    }

    /// List buckets in a namespace.
    ///
    /// Streams bucket metadata for all buckets owned by the given user or organization.
    ///
    /// Endpoint: `GET /api/buckets/{namespace}` (paginated).
    ///
    /// # Parameters
    ///
    /// - `namespace` (required): user or organization namespace to list buckets for.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn list_buckets(
        &self,
        /// User or organization namespace to list buckets for.
        #[builder(into)]
        namespace: String,
    ) -> HFResult<impl Stream<Item = HFResult<BucketInfo>> + '_> {
        let url = Url::parse(&format!("{}/api/buckets/{}", self.endpoint(), namespace))?;
        Ok(self.paginate(url, vec![], None))
    }

    /// Move (rename) a bucket.
    ///
    /// Endpoint: `POST /api/repos/move` with `type: "bucket"`.
    ///
    /// # Parameters
    ///
    /// - `from_id` (required): current bucket ID in `"owner/name"` format.
    /// - `to_id` (required): new bucket ID in `"owner/name"` format.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn move_bucket(
        &self,
        /// Current bucket ID in `"owner/name"` format.
        from_id: &str,
        /// New bucket ID in `"owner/name"` format.
        to_id: &str,
    ) -> HFResult<()> {
        let url = format!("{}/api/repos/move", self.endpoint());
        let body = serde_json::json!({
            "fromRepo": from_id,
            "toRepo": to_id,
            "type": "bucket",
        });

        let headers = self.auth_headers();
        let response = retry::retry(self.retry_config(), || {
            self.http_client()
                .post(&url)
                .headers(headers.clone())
                .json(&body)
                .send()
        })
        .await?;

        self.check_response(response, None, NotFoundContext::Generic)
            .await?;
        Ok(())
    }
}

#[cfg(feature = "blocking")]
#[bon]
impl crate::blocking::HFClientSync {
    /// Blocking counterpart of [`HFClient::create_bucket`]. See the async method for parameters
    /// and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn create_bucket(
        &self,
        namespace: &str,
        name: &str,
        #[builder(default)] private: bool,
        #[builder(into)] resource_group_id: Option<String>,
        #[builder(default)] exist_ok: bool,
    ) -> HFResult<BucketUrl> {
        self.runtime.block_on(
            self.inner
                .create_bucket()
                .namespace(namespace)
                .name(name)
                .private(private)
                .maybe_resource_group_id(resource_group_id)
                .exist_ok(exist_ok)
                .send(),
        )
    }

    /// Blocking counterpart of [`HFClient::delete_bucket`]. See the async method for parameters
    /// and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn delete_bucket(
        &self,
        bucket_id: &str,
        #[builder(default)] missing_ok: bool,
    ) -> HFResult<()> {
        self.runtime.block_on(
            self.inner
                .delete_bucket()
                .bucket_id(bucket_id)
                .missing_ok(missing_ok)
                .send(),
        )
    }

    /// Blocking counterpart of [`HFClient::list_buckets`]. Collects the stream into a
    /// `Vec<BucketInfo>`. See the async method for parameters and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn list_buckets(&self, #[builder(into)] namespace: String) -> HFResult<Vec<BucketInfo>> {
        self.runtime.block_on(async move {
            let stream = self.inner.list_buckets().namespace(namespace).send()?;
            futures::pin_mut!(stream);
            let mut items = Vec::new();
            while let Some(item) = stream.next().await {
                items.push(item?);
            }
            Ok(items)
        })
    }

    /// Blocking counterpart of [`HFClient::move_bucket`]. See the async method for parameters
    /// and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn move_bucket(&self, from_id: &str, to_id: &str) -> HFResult<()> {
        self.runtime.block_on(
            self.inner
                .move_bucket()
                .from_id(from_id)
                .to_id(to_id)
                .send(),
        )
    }
}

#[cfg(feature = "blocking")]
#[bon]
impl crate::blocking::HFBucketSync {
    /// Blocking counterpart of [`HFBucket::info`].
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn info(&self) -> HFResult<BucketInfo> {
        self.runtime.block_on(self.inner.info().send())
    }

    /// Blocking counterpart of [`HFBucket::list_tree`]. Collects the stream into a
    /// `Vec<BucketTreeEntry>`. See the async method for parameters and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn list_tree(
        &self,
        #[builder(into)] prefix: Option<String>,
        #[builder(default)] recursive: bool,
    ) -> HFResult<Vec<BucketTreeEntry>> {
        self.runtime.block_on(async move {
            let stream = self
                .inner
                .list_tree()
                .maybe_prefix(prefix)
                .recursive(recursive)
                .send()?;
            futures::pin_mut!(stream);
            let mut items = Vec::new();
            while let Some(item) = stream.next().await {
                items.push(item?);
            }
            Ok(items)
        })
    }

    /// Blocking counterpart of [`HFBucket::get_paths_info`]. See the async method for parameters
    /// and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn get_paths_info(&self, paths: Vec<String>) -> HFResult<Vec<BucketTreeEntry>> {
        self.runtime
            .block_on(self.inner.get_paths_info().paths(paths).send())
    }

    /// Blocking counterpart of [`HFBucket::get_file_metadata`]. See the async method for parameters
    /// and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn get_file_metadata(
        &self,
        #[builder(into)] remote_path: String,
    ) -> HFResult<BucketFileMetadata> {
        self.runtime.block_on(
            self.inner
                .get_file_metadata()
                .remote_path(remote_path)
                .send(),
        )
    }

    /// Blocking counterpart of [`HFBucket::batch_operations`]. See the async method for parameters and
    /// behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn batch_operations(
        &self,
        #[builder(default)] add_files: Vec<BucketAddFile>,
        #[builder(default)] delete: Vec<String>,
        #[builder(default)] copy: Vec<BucketCopyFile>,
    ) -> HFResult<()> {
        self.runtime.block_on(
            self.inner
                .batch_operations()
                .add_files(add_files)
                .delete(delete)
                .copy(copy)
                .send(),
        )
    }

    /// Blocking counterpart of [`HFBucket::delete_files`]. See the async method for parameters
    /// and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn delete_files(&self, paths: Vec<String>) -> HFResult<()> {
        self.runtime
            .block_on(self.inner.delete_files().paths(paths).send())
    }

    /// Blocking counterpart of [`HFBucket::upload_source_files`]. See the async method for
    /// parameters and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn upload_source_files(
        &self,
        files: Vec<(String, crate::repository::AddSource)>,
        #[builder(into)] progress: Option<Progress>,
    ) -> HFResult<()> {
        self.runtime.block_on(
            self.inner
                .upload_source_files()
                .files(files)
                .maybe_progress(progress)
                .send(),
        )
    }

    /// Blocking counterpart of [`HFBucket::upload_files`]. See the async method for parameters
    /// and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn upload_files(
        &self,
        files: Vec<BucketUpload>,
        #[builder(into)] progress: Option<Progress>,
    ) -> HFResult<()> {
        self.runtime.block_on(
            self.inner
                .upload_files()
                .files(files)
                .maybe_progress(progress)
                .send(),
        )
    }

    /// Blocking counterpart of [`HFBucket::download_files`]. See the async method for parameters
    /// and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn download_files(
        &self,
        files: Vec<BucketDownload>,
        #[builder(into)] progress: Option<Progress>,
    ) -> HFResult<()> {
        self.runtime.block_on(
            self.inner
                .download_files()
                .files(files)
                .maybe_progress(progress)
                .send(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{BucketCopyFile, BucketCopySourceType, HFBucket};

    #[test]
    fn test_bucket_accessors() {
        let client = crate::HFClient::builder().build().unwrap();
        let bucket = HFBucket::new(client, "my-org", "my-bucket");

        assert_eq!(bucket.owner(), "my-org");
        assert_eq!(bucket.name(), "my-bucket");
        assert_eq!(bucket.bucket_id(), "my-org/my-bucket");
    }

    #[test]
    fn test_bucket_copy_source_type_wire_strings() {
        assert_eq!(BucketCopySourceType::Bucket.as_str(), "bucket");
        assert_eq!(BucketCopySourceType::Model.as_str(), "model");
        assert_eq!(BucketCopySourceType::Dataset.as_str(), "dataset");
        assert_eq!(BucketCopySourceType::Space.as_str(), "space");
    }

    #[test]
    fn test_bucket_copy_file_optional_fields_construct() {
        let with_extras = BucketCopyFile {
            path: "p".into(),
            xet_hash: "x".into(),
            source_repo_type: BucketCopySourceType::Bucket,
            source_repo_id: "u/b".into(),
            size: Some(42),
            mtime: Some(1_700_000_000_000),
            content_type: Some("text/plain".into()),
        };
        assert_eq!(with_extras.size, Some(42));
        assert_eq!(with_extras.mtime, Some(1_700_000_000_000));
        assert_eq!(with_extras.content_type.as_deref(), Some("text/plain"));

        let bare = BucketCopyFile {
            path: "p".into(),
            xet_hash: "x".into(),
            source_repo_type: BucketCopySourceType::Bucket,
            source_repo_id: "u/b".into(),
            size: None,
            mtime: None,
            content_type: None,
        };
        assert!(bare.size.is_none() && bare.mtime.is_none() && bare.content_type.is_none());
    }
}
