//! Repository file and snapshot download builders.
//!
//! Builders on [`HFRepository`] for fetching file contents:
//!
//! - [`HFRepository::download_file`] — download one file to the cache or a local directory.
//! - [`HFRepository::download_file_stream`] — stream a file (or byte range) without buffering.
//! - [`HFRepository::download_file_to_bytes`] — read a file (or byte range) into memory.
//! - [`HFRepository::snapshot_download`] — download all files at a revision, optionally filtered by allow/ignore globs
//!   matched against repo-relative paths.
//!
//! Range parameters use Rust `std::ops::Range<u64>` semantics (start-inclusive, end-exclusive).
//! See each builder's docs for the exact path / range / glob format rules.

use std::collections::HashMap;
use std::io::{Seek, Write};
use std::path::{Path, PathBuf};

use bon::bon;
use futures::TryStreamExt;
use futures::stream::{Stream, StreamExt};
use reqwest::header::IF_NONE_MATCH;
use serde::Deserialize;

use super::files::{
    extract_commit_hash, extract_etag, extract_file_size, extract_xet_hash, matches_any_glob,
};
use super::{FileMetadataInfo, HFRepository, RepoTreeEntry, RepoType};
use crate::cache::storage as cache;
use crate::error::{HFError, HFResult};
use crate::progress::{DownloadEvent, EmitEvent, FileProgress, FileStatus, Progress};
use crate::{constants, retry};

pub(crate) type HFByteStream = Box<dyn Stream<Item = HFResult<bytes::Bytes>> + Send + Unpin>;

/// Internal options struct used by the file download helpers.
#[derive(Clone)]
struct DownloadFileParams {
    filename: String,
    local_dir: Option<PathBuf>,
    revision: Option<String>,
    force_download: bool,
    local_files_only: bool,
    progress: Option<Progress>,
}

/// Internal options struct used by the streaming download helpers.
struct DownloadFileStreamParams {
    filename: String,
    revision: Option<String>,
    range: Option<std::ops::Range<u64>>,
    progress: Option<Progress>,
}

/// Internal options struct used by `snapshot_download_impl`.
struct SnapshotDownloadParams {
    revision: Option<String>,
    allow_patterns: Option<Vec<String>>,
    ignore_patterns: Option<Vec<String>>,
    local_dir: Option<PathBuf>,
    force_download: bool,
    local_files_only: bool,
    max_workers: Option<usize>,
    progress: Option<Progress>,
}

impl<T: RepoType> HFRepository<T> {
    async fn download_file_impl(&self, params: DownloadFileParams) -> HFResult<PathBuf> {
        let result = self.download_file_inner(&params).await;
        if result.is_ok() {
            params.progress.emit(DownloadEvent::Complete);
        }
        result
    }

    async fn download_file_inner(&self, params: &DownloadFileParams) -> HFResult<PathBuf> {
        if params.local_dir.is_some() {
            self.download_file_to_local_dir(params).await
        } else {
            if !self.hf_client.cache_enabled() {
                return Err(HFError::CacheNotEnabled);
            }
            self.download_file_to_cache(params).await
        }
    }

    async fn download_file_stream_impl(
        &self,
        params: DownloadFileStreamParams,
    ) -> HFResult<(
        Option<u64>,
        Box<dyn Stream<Item = Result<bytes::Bytes, HFError>> + Send + Unpin>,
    )> {
        if let Some(ref range) = params.range
            && range.start >= range.end
        {
            return Err(HFError::InvalidParameter(format!(
                "range start ({}) must be less than end ({})",
                range.start, range.end
            )));
        }

        let revision = params
            .revision
            .as_deref()
            .unwrap_or(constants::DEFAULT_REVISION);
        let repo_path = self.repo_path();
        let url = self.hf_client.download_url(
            self.repo_type.url_prefix(),
            &repo_path,
            revision,
            &params.filename,
        )?;

        let headers = self.hf_client.auth_headers();
        let head_response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client
                .http_client()
                .head(&url)
                .headers(headers.clone())
                .send()
        })
        .await?;
        let head_response = self
            .hf_client
            .check_response(
                head_response,
                Some(&repo_path),
                crate::error::NotFoundContext::Entry {
                    path: params.filename.clone(),
                },
            )
            .await?;

        if let Some(xet_hash) = extract_xet_hash(&head_response) {
            let file_size: u64 = extract_file_size(&head_response).unwrap_or_else(|| {
                    tracing::warn!(url = %url, "missing or invalid Content-Length/X-Linked-Size header for xet file, defaulting file size to 0");
                    0
                });

            let content_length = params
                .range
                .as_ref()
                .map(|r| r.end.saturating_sub(r.start))
                .or(Some(file_size));

            let stream = self
                .xet_download_stream(revision, &xet_hash, file_size, params.range.clone())
                .await?;

            let total_bytes = content_length.unwrap_or(0);
            params.progress.emit(DownloadEvent::Start {
                total_files: 1,
                total_bytes,
            });
            let wrapped = wrap_stream_with_progress(
                Box::new(Box::pin(stream)),
                params.progress,
                params.filename,
                total_bytes,
            );
            return Ok((content_length, wrapped));
        }

        let range_header = params
            .range
            .as_ref()
            .map(|r| format!("bytes={}-{}", r.start, r.end.saturating_sub(1)));
        let response = retry::retry(self.hf_client.retry_config(), || {
            let mut req = self
                .hf_client
                .http_client()
                .get(&url)
                .headers(headers.clone());
            if let Some(ref range) = range_header {
                req = req.header(reqwest::header::RANGE, range);
            }
            req.send()
        })
        .await?;
        let response = self
            .hf_client
            .check_response(
                response,
                Some(&repo_path),
                crate::error::NotFoundContext::Entry {
                    path: params.filename.clone(),
                },
            )
            .await?;

        let content_length = extract_file_size(&response);
        let total_bytes = content_length.unwrap_or(0);
        let stream = response.bytes_stream().map(|r| r.map_err(HFError::from));
        params.progress.emit(DownloadEvent::Start {
            total_files: 1,
            total_bytes,
        });
        let wrapped = wrap_stream_with_progress(
            Box::new(Box::pin(stream)),
            params.progress,
            params.filename,
            total_bytes,
        );
        Ok((content_length, wrapped))
    }

    async fn download_file_to_bytes_impl(
        &self,
        params: DownloadFileStreamParams,
    ) -> HFResult<bytes::Bytes> {
        let (content_length, stream) = self.download_file_stream_impl(params).await?;
        futures::pin_mut!(stream);

        let capacity = content_length.unwrap_or(0) as usize;
        let mut buf = bytes::BytesMut::with_capacity(capacity);
        while let Some(chunk) = stream.next().await {
            buf.extend_from_slice(&chunk?);
        }
        Ok(buf.freeze())
    }

    async fn download_file_to_local_dir(&self, params: &DownloadFileParams) -> HFResult<PathBuf> {
        let revision = params
            .revision
            .as_deref()
            .unwrap_or(constants::DEFAULT_REVISION);
        let repo_path = self.repo_path();
        let url = self.hf_client.download_url(
            self.repo_type.url_prefix(),
            &repo_path,
            revision,
            &params.filename,
        )?;

        let headers = self.hf_client.auth_headers();
        let head_response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client
                .http_client()
                .head(&url)
                .headers(headers.clone())
                .send()
        })
        .await?;

        let head_response = self
            .hf_client
            .check_response(
                head_response,
                Some(&repo_path),
                crate::error::NotFoundContext::Entry {
                    path: params.filename.clone(),
                },
            )
            .await?;

        let file_size = extract_file_size(&head_response).unwrap_or(0);
        let has_xet_hash = head_response
            .headers()
            .get(constants::HEADER_X_XET_HASH)
            .is_some();

        params.progress.emit(DownloadEvent::Start {
            total_files: 1,
            total_bytes: file_size,
        });

        if has_xet_hash {
            let local_dir = params.local_dir.as_ref().unwrap();
            return self
                .xet_download_to_local_dir(
                    revision,
                    &params.filename,
                    local_dir,
                    &head_response,
                    &params.progress,
                )
                .await;
        }

        let local_dir = params.local_dir.as_ref().unwrap();
        std::fs::create_dir_all(local_dir)?;

        let dest_path = local_dir.join(&params.filename);
        if let Some(parent) = dest_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let resume_offset =
            match cache::partial_download_state(&dest_path, file_size, params.force_download)? {
                cache::PartialDownloadState::Complete => {
                    params.progress.emit(DownloadEvent::Progress {
                        files: vec![FileProgress {
                            filename: params.filename.clone(),
                            bytes_completed: file_size,
                            total_bytes: file_size,
                            status: FileStatus::Complete,
                        }],
                    });
                    return Ok(dest_path);
                }
                cache::PartialDownloadState::Resume(offset) => offset,
                cache::PartialDownloadState::Restart => 0,
            };
        let (response, write_offset) = self
            .http_download_response(
                &url,
                &repo_path,
                &params.filename,
                headers,
                resume_offset,
                file_size,
            )
            .await?;
        stream_response_to_file_with_progress(
            response,
            &dest_path,
            &params.progress,
            Some(&params.filename),
            file_size,
            write_offset,
        )
        .await?;
        params.progress.emit(DownloadEvent::Progress {
            files: vec![FileProgress {
                filename: params.filename.clone(),
                bytes_completed: file_size,
                total_bytes: file_size,
                status: FileStatus::Complete,
            }],
        });

        Ok(dest_path)
    }

    /// Resolve a file from the local cache without making network requests.
    /// Matches Python's `try_to_load_from_cache`: checks the snapshot pointer
    /// first, then consults `.no_exist` markers for negative cache hits.
    fn resolve_from_cache_only(
        &self,
        repo_folder: &str,
        revision: &str,
        filename: &str,
    ) -> HFResult<PathBuf> {
        let cache_dir = self.hf_client.cache_dir();

        let commit_hash = if cache::is_commit_hash(revision) {
            Some(revision.to_string())
        } else {
            let ref_path = cache::ref_path(cache_dir, repo_folder, revision);
            std::fs::read_to_string(&ref_path)
                .ok()
                .map(|s| s.trim().to_string())
        };

        if let Some(ref hash) = commit_hash {
            let snap = cache::snapshot_path(cache_dir, repo_folder, hash, filename);
            if snap.exists() {
                return Ok(snap);
            }
            if cache::no_exist_path(cache_dir, repo_folder, hash, filename).exists() {
                return Err(HFError::EntryNotFound {
                    path: filename.to_string(),
                    repo_id: String::new(),
                    context: None,
                });
            }
        }

        Err(HFError::LocalEntryNotFound {
            path: filename.to_string(),
        })
    }

    /// Resolve the cached etag for a file by reading the symlink target in snapshots/.
    /// If the snapshot is a regular file because symlinks are unavailable,
    /// `read_link` returns `None` and conditional requests are skipped.
    fn find_cached_etag(
        &self,
        repo_folder: &str,
        revision: &str,
        filename: &str,
    ) -> Option<String> {
        let cache_dir = self.hf_client.cache_dir();

        let commit_hash = if cache::is_commit_hash(revision) {
            Some(revision.to_string())
        } else {
            let ref_path = cache::ref_path(cache_dir, repo_folder, revision);
            std::fs::read_to_string(&ref_path)
                .ok()
                .map(|s| s.trim().to_string())
        };

        let hash = commit_hash?;
        let snap = cache::snapshot_path(cache_dir, repo_folder, &hash, filename);
        let target = std::fs::read_link(&snap).ok()?;
        target.file_name()?.to_str().map(|s| s.to_string())
    }

    async fn download_file_to_cache(&self, params: &DownloadFileParams) -> HFResult<PathBuf> {
        let revision = params
            .revision
            .as_deref()
            .unwrap_or(constants::DEFAULT_REVISION);
        let cache_dir = self.hf_client.cache_dir();
        let repo_folder = cache::repo_folder_name(&self.repo_path(), self.repo_type.plural());
        let force_download = params.force_download;

        if cache::is_commit_hash(revision) && !force_download {
            let snap = cache::snapshot_path(cache_dir, &repo_folder, revision, &params.filename);
            if snap.exists() {
                return Ok(snap);
            }
        }

        if params.local_files_only {
            return self.resolve_from_cache_only(&repo_folder, revision, &params.filename);
        }

        let result = self
            .download_file_to_cache_network(
                params,
                revision,
                cache_dir,
                &repo_folder,
                force_download,
            )
            .await;

        match &result {
            Err(e) if e.is_transient() && !force_download => self
                .resolve_from_cache_only(&repo_folder, revision, &params.filename)
                .or(result),
            _ => result,
        }
    }

    async fn download_file_to_cache_network(
        &self,
        params: &DownloadFileParams,
        revision: &str,
        cache_dir: &Path,
        repo_folder: &str,
        force_download: bool,
    ) -> HFResult<PathBuf> {
        let repo_path = self.repo_path();
        let url = self.hf_client.download_url(
            self.repo_type.url_prefix(),
            &repo_path,
            revision,
            &params.filename,
        )?;

        let cached_etag = if !force_download {
            self.find_cached_etag(repo_folder, revision, &params.filename)
        } else {
            None
        };

        let mut head_headers = self.hf_client.auth_headers();
        if let Some(ref etag_val) = cached_etag
            && let Ok(hv) = reqwest::header::HeaderValue::from_str(&format!("\"{etag_val}\""))
        {
            head_headers.insert(IF_NONE_MATCH, hv);
        }

        let head_response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client
                .no_redirect_client()
                .head(&url)
                .headers(head_headers.clone())
                .send()
        })
        .await?;

        let status = head_response.status();

        if status == reqwest::StatusCode::NOT_FOUND {
            return Err(mark_no_exist_and_return_error(
                cache_dir,
                repo_folder,
                revision,
                &head_response,
                &repo_path,
                &params.filename,
            )
            .await);
        }

        if status == reqwest::StatusCode::NOT_MODIFIED {
            let etag = cached_etag.ok_or_else(|| {
                HFError::malformed_response_at("304 Not Modified without cached ETag", url.clone())
            })?;
            let commit_hash = if cache::is_commit_hash(revision) {
                revision.to_string()
            } else {
                cache::read_ref(cache_dir, repo_folder, revision)
                    .await?
                    .ok_or_else(|| {
                        HFError::malformed_response_at(
                            "304 Not Modified without cached commit hash",
                            url.clone(),
                        )
                    })?
            };
            return finalize_cached_file(
                cache_dir,
                repo_folder,
                revision,
                &commit_hash,
                &params.filename,
                &etag,
                cache::PointerSource::ExistingBlob,
            )
            .await;
        }

        let etag = extract_etag(&head_response)
            .ok_or_else(|| HFError::malformed_response_at("missing ETag header", url.clone()));
        let commit_hash = extract_commit_hash(&head_response);
        let xet_hash = extract_xet_hash(&head_response);
        let has_xet_hash = xet_hash.is_some();
        let file_size: u64 = extract_file_size(&head_response).unwrap_or_else(|| {
            tracing::warn!(url = %url, "missing or invalid Content-Length/X-Linked-Size header, defaulting file size to 0");
            0
        });

        if !status.is_success() && !status.is_redirection() {
            self.hf_client
                .check_response(
                    head_response,
                    Some(&repo_path),
                    crate::error::NotFoundContext::Entry {
                        path: params.filename.clone(),
                    },
                )
                .await?;
        }

        let etag = etag?;
        let commit_hash = commit_hash.ok_or_else(|| {
            HFError::malformed_response_at("missing X-Repo-Commit header", url.clone())
        })?;

        params.progress.emit(DownloadEvent::Start {
            total_files: 1,
            total_bytes: file_size,
        });

        let snapshot_path =
            cache::snapshot_path(cache_dir, repo_folder, &commit_hash, &params.filename);
        if snapshot_path.exists() && !force_download {
            params.progress.emit(DownloadEvent::Progress {
                files: vec![FileProgress {
                    filename: params.filename.clone(),
                    bytes_completed: file_size,
                    total_bytes: file_size,
                    status: FileStatus::Complete,
                }],
            });
            return Ok(snapshot_path);
        }

        if has_xet_hash {
            let xet_hash = xet_hash.ok_or_else(|| {
                HFError::malformed_response_at("missing X-Xet-Hash header", url.clone())
            })?;
            let blob = cache::blob_path(cache_dir, repo_folder, &etag);
            let new_blob = !blob.exists();
            if new_blob || force_download {
                if let Some(parent) = blob.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                let _lock = cache::acquire_lock(cache_dir, repo_folder, &etag).await?;

                self.xet_download_to_blob(
                    revision,
                    crate::xet::XetBlobFile {
                        filename: &params.filename,
                        file_hash: &xet_hash,
                        file_size,
                        path: &blob,
                        force_download,
                    },
                    &params.progress,
                )
                .await?;
            }

            let source = if new_blob {
                cache::PointerSource::NewBlob
            } else {
                cache::PointerSource::ExistingBlob
            };
            return finalize_cached_file(
                cache_dir,
                repo_folder,
                revision,
                &commit_hash,
                &params.filename,
                &etag,
                source,
            )
            .await;
        }

        let blob = cache::blob_path(cache_dir, repo_folder, &etag);

        if blob.exists() && !force_download {
            params.progress.emit(DownloadEvent::Progress {
                files: vec![FileProgress {
                    filename: params.filename.clone(),
                    bytes_completed: file_size,
                    total_bytes: file_size,
                    status: FileStatus::Complete,
                }],
            });
            return finalize_cached_file(
                cache_dir,
                repo_folder,
                revision,
                &commit_hash,
                &params.filename,
                &etag,
                cache::PointerSource::ExistingBlob,
            )
            .await;
        }

        let _lock = cache::acquire_lock(cache_dir, repo_folder, &etag).await?;
        let new_blob = !blob.exists();
        let incomplete_path = cache::incomplete_path(&blob);
        if let Some(parent) = incomplete_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let resume_offset =
            match cache::partial_download_state(&incomplete_path, file_size, force_download)? {
                cache::PartialDownloadState::Complete => {
                    std::fs::rename(&incomplete_path, &blob)?;
                    params.progress.emit(DownloadEvent::Progress {
                        files: vec![FileProgress {
                            filename: params.filename.clone(),
                            bytes_completed: file_size,
                            total_bytes: file_size,
                            status: FileStatus::Complete,
                        }],
                    });
                    let source = if new_blob {
                        cache::PointerSource::NewBlob
                    } else {
                        cache::PointerSource::ExistingBlob
                    };
                    return finalize_cached_file(
                        cache_dir,
                        repo_folder,
                        revision,
                        &commit_hash,
                        &params.filename,
                        &etag,
                        source,
                    )
                    .await;
                }
                cache::PartialDownloadState::Resume(offset) => offset,
                cache::PartialDownloadState::Restart => 0,
            };

        let dl_headers = self.hf_client.auth_headers();
        let (response, write_offset) = self
            .http_download_response(
                &url,
                &repo_path,
                &params.filename,
                dl_headers,
                resume_offset,
                file_size,
            )
            .await?;
        stream_response_to_file_with_progress(
            response,
            &incomplete_path,
            &params.progress,
            Some(&params.filename),
            file_size,
            write_offset,
        )
        .await?;
        params.progress.emit(DownloadEvent::Progress {
            files: vec![FileProgress {
                filename: params.filename.clone(),
                bytes_completed: file_size,
                total_bytes: file_size,
                status: FileStatus::Complete,
            }],
        });
        std::fs::rename(&incomplete_path, &blob)?;

        let source = if new_blob {
            cache::PointerSource::NewBlob
        } else {
            cache::PointerSource::ExistingBlob
        };
        finalize_cached_file(
            cache_dir,
            repo_folder,
            revision,
            &commit_hash,
            &params.filename,
            &etag,
            source,
        )
        .await
    }

    async fn http_download_response(
        &self,
        url: &str,
        repo_path: &str,
        filename: &str,
        headers: reqwest::header::HeaderMap,
        resume_offset: u64,
        file_size: u64,
    ) -> HFResult<(reqwest::Response, u64)> {
        let response = self
            .http_download_response_once(url, repo_path, filename, headers.clone(), resume_offset)
            .await?;
        if resume_offset == 0 {
            return Ok((response, 0));
        }
        if response.status() == reqwest::StatusCode::PARTIAL_CONTENT
            && content_range_matches(response.headers(), resume_offset, file_size)
        {
            return Ok((response, resume_offset));
        }
        tracing::warn!(
            url,
            resume_offset,
            "server did not return a valid partial response; restarting download"
        );
        let response = self
            .http_download_response_once(url, repo_path, filename, headers, 0)
            .await?;
        Ok((response, 0))
    }

    async fn http_download_response_once(
        &self,
        url: &str,
        repo_path: &str,
        filename: &str,
        headers: reqwest::header::HeaderMap,
        resume_offset: u64,
    ) -> HFResult<reqwest::Response> {
        let response = retry::retry(self.hf_client.retry_config(), || {
            let mut request = self
                .hf_client
                .http_client()
                .get(url)
                .headers(headers.clone());
            if resume_offset > 0 {
                request = request.header(reqwest::header::RANGE, format!("bytes={resume_offset}-"));
            }
            request.send()
        })
        .await?;
        self.hf_client
            .check_response(
                response,
                Some(repo_path),
                crate::error::NotFoundContext::Entry {
                    path: filename.to_string(),
                },
            )
            .await
    }

    async fn resolve_commit_hash(&self, revision: &str) -> HFResult<String> {
        if cache::is_commit_hash(revision) {
            return Ok(revision.to_string());
        }
        #[derive(Deserialize)]
        struct ShaOnly {
            sha: Option<String>,
        }
        let repo_path = self.repo_path();
        let info: ShaOnly = self
            .fetch_repo_info(Some(revision.to_string()), None)
            .await?;
        info.sha.ok_or_else(|| {
            HFError::malformed_response(format!(
                "repo info for {}@{} returned no commit sha",
                repo_path, revision
            ))
        })
    }

    async fn list_filtered_files(
        &self,
        revision: &str,
        allow_patterns: Option<&Vec<String>>,
        ignore_patterns: Option<&Vec<String>>,
    ) -> HFResult<Vec<String>> {
        let stream = self
            .list_tree()
            .revision(revision.to_string())
            .recursive(true)
            .send()?;
        futures::pin_mut!(stream);

        let mut filenames: Vec<String> = Vec::new();
        while let Some(entry) = stream.next().await {
            let entry = entry?;
            if let RepoTreeEntry::File { path, .. } = entry {
                filenames.push(path);
            }
        }

        if let Some(allow) = allow_patterns {
            filenames.retain(|f| matches_any_glob(allow, f));
        }
        if let Some(ignore) = ignore_patterns {
            filenames.retain(|f| !matches_any_glob(ignore, f));
        }

        Ok(filenames)
    }

    async fn snapshot_download_impl(&self, params: SnapshotDownloadParams) -> HFResult<PathBuf> {
        if params.local_dir.is_none() && !self.hf_client.cache_enabled() {
            return Err(HFError::CacheNotEnabled);
        }
        let revision = params
            .revision
            .as_deref()
            .unwrap_or(constants::DEFAULT_REVISION);
        let max_workers = params.max_workers.unwrap_or(8);
        let repo_folder = cache::repo_folder_name(&self.repo_path(), self.repo_type.plural());
        let cache_dir = self.hf_client.cache_dir();

        if params.local_files_only {
            let commit_hash = if cache::is_commit_hash(revision) {
                revision.to_string()
            } else {
                cache::read_ref(cache_dir, &repo_folder, revision)
                    .await?
                    .ok_or_else(|| HFError::LocalEntryNotFound {
                        path: format!("{}/{}", repo_folder, revision),
                    })?
            };
            let snapshot_dir = cache_dir
                .join(&repo_folder)
                .join("snapshots")
                .join(&commit_hash);
            if snapshot_dir.exists() {
                return Ok(snapshot_dir);
            }
            return Err(HFError::LocalEntryNotFound {
                path: format!("{}/{}", repo_folder, commit_hash),
            });
        }

        let commit_hash = self.resolve_commit_hash(revision).await?;

        let mut filenames = self
            .list_filtered_files(
                &commit_hash,
                params.allow_patterns.as_ref(),
                params.ignore_patterns.as_ref(),
            )
            .await?;

        let total_files = filenames.len();
        let force = params.force_download;

        let mut cached_filenames = Vec::new();
        if !force && params.local_dir.is_none() {
            filenames.retain(|f| {
                if cache::snapshot_path(cache_dir, &repo_folder, &commit_hash, f).exists() {
                    cached_filenames.push(f.clone());
                    false
                } else {
                    true
                }
            });
        }

        let repo_path = self.repo_path();
        let head_futs = filenames.into_iter().map(|filename| {
                let commit_hash = commit_hash.clone();
                let url = self
                    .hf_client.download_url(self.repo_type.url_prefix(), &repo_path, &commit_hash, &filename);
                let auth = self.hf_client.auth_headers();
                let repo_folder = repo_folder.clone();
                async move {
                    let url = url?;
                    let resp = retry::retry(self.hf_client.retry_config(), || {
                        self.hf_client.no_redirect_client().head(&url).headers(auth.clone()).send()
                    })
                    .await?;
                    // Per-file 404 resilience: write a .no_exist marker and skip
                    // the file rather than aborting the entire snapshot download.
                    // This matches the Python huggingface_hub library behavior.
                    // Alternative: since the file list comes from list_repo_tree
                    // on a pinned commit, a 404 here is unexpected and could be
                    // treated as an error instead.
                    if resp.status() == reqwest::StatusCode::NOT_FOUND {
                        if let Some(commit) = extract_commit_hash(&resp) {
                            let no_exist = cache::no_exist_path(cache_dir, &repo_folder, &commit, &filename);
                            if let Some(parent) = no_exist.parent() {
                                let _ = std::fs::create_dir_all(parent);
                            }
                            let _ = std::fs::write(&no_exist, b"");
                        }
                        return Ok::<_, HFError>(None);
                    } else if !resp.status().is_success() && !resp.status().is_redirection() {
                        let context = Box::new(crate::error::HttpErrorContext::from_response(resp).await);
                        return Err(HFError::Http { context });
                    }
                    let etag = extract_etag(&resp).ok_or_else(|| {
                        HFError::malformed_response_at(format!("missing ETag header for {filename}"), url.clone())
                    })?;
                    let commit = extract_commit_hash(&resp).unwrap_or(commit_hash);
                    let xet_hash = extract_xet_hash(&resp);
                    let file_size: u64 = extract_file_size(&resp).unwrap_or_else(|| {
                        tracing::warn!(file = %filename, "missing or invalid Content-Length/X-Linked-Size header, defaulting file size to 0");
                        0
                    });
                    let location = Some(resp.url().to_string());
                    Ok::<_, HFError>(Some(FileMetadataInfo {
                        filename,
                        etag,
                        commit_hash: commit,
                        xet_hash,
                        file_size,
                        location,
                    }))
                }
            });

        let file_metas: Vec<FileMetadataInfo> = futures::stream::iter(head_futs)
            .buffer_unordered(max_workers)
            .try_collect::<Vec<Option<FileMetadataInfo>>>()
            .await?
            .into_iter()
            .flatten()
            .collect();

        let total_bytes: u64 = file_metas.iter().map(|m| m.file_size).sum();
        params.progress.emit(DownloadEvent::Start {
            total_files,
            total_bytes,
        });
        if !cached_filenames.is_empty() {
            params.progress.emit(DownloadEvent::Progress {
                files: cached_filenames
                    .iter()
                    .map(|f| FileProgress {
                        filename: f.clone(),
                        bytes_completed: 0,
                        total_bytes: 0,
                        status: FileStatus::Complete,
                    })
                    .collect(),
            });
        }

        let mut xet_metas = Vec::new();
        let mut non_xet_filenames = Vec::new();

        if let Some(ref local_dir) = params.local_dir {
            let mut local_cached = Vec::new();
            for meta in file_metas {
                let dest = local_dir.join(&meta.filename);
                if dest.exists() && !force {
                    local_cached.push(meta.filename);
                    continue;
                }
                if meta.xet_hash.is_some() {
                    xet_metas.push(meta);
                } else {
                    non_xet_filenames.push(meta.filename);
                }
            }
            if !local_cached.is_empty() {
                params.progress.emit(DownloadEvent::Progress {
                    files: local_cached
                        .iter()
                        .map(|f| FileProgress {
                            filename: f.clone(),
                            bytes_completed: 0,
                            total_bytes: 0,
                            status: FileStatus::Complete,
                        })
                        .collect(),
                });
            }

            let xet_batch_fut = async {
                if xet_metas.is_empty() {
                    return Ok::<_, HFError>(());
                }
                let batch_files: Vec<crate::xet::XetBatchFile> = xet_metas
                    .iter()
                    .map(|m| crate::xet::XetBatchFile {
                        hash: m.xet_hash.as_ref().unwrap().clone(),
                        file_size: m.file_size,
                        path: local_dir.join(&m.filename),
                        filename: m.filename.clone(),
                    })
                    .collect();
                self.xet_download_batch(
                    &commit_hash,
                    &batch_files,
                    params.force_download,
                    &params.progress,
                )
                .await?;
                Ok(())
            };

            let non_xet_dl_params = build_download_params(
                &repo_path,
                &non_xet_filenames,
                &commit_hash,
                params.force_download,
                Some(local_dir.clone()),
                &params.progress,
            );
            let non_xet_fut = async {
                download_concurrently(self, &non_xet_dl_params, max_workers).await?;
                Ok::<_, HFError>(())
            };

            tokio::try_join!(xet_batch_fut, non_xet_fut)?;
            params.progress.emit(DownloadEvent::Complete);
            return Ok(local_dir.clone());
        }

        // Cache mode
        let mut cached_progress: Vec<FileProgress> = Vec::new();
        let mut xet_pointer_sources = HashMap::new();
        for meta in file_metas {
            let blob = cache::blob_path(cache_dir, &repo_folder, &meta.etag);
            let pointer =
                cache::snapshot_path(cache_dir, &repo_folder, &meta.commit_hash, &meta.filename);
            if pointer.exists() && !force {
                cached_progress.push(FileProgress {
                    filename: meta.filename.clone(),
                    bytes_completed: meta.file_size,
                    total_bytes: meta.file_size,
                    status: FileStatus::Complete,
                });
                continue;
            }
            if blob.exists() && !force {
                cache::create_pointer_symlink(
                    cache_dir,
                    &repo_folder,
                    &meta.commit_hash,
                    &meta.filename,
                    &meta.etag,
                    cache::PointerSource::ExistingBlob,
                )
                .await?;
                cached_progress.push(FileProgress {
                    filename: meta.filename.clone(),
                    bytes_completed: meta.file_size,
                    total_bytes: meta.file_size,
                    status: FileStatus::Complete,
                });
                continue;
            }
            if meta.xet_hash.is_some() {
                xet_pointer_sources.insert(
                    meta.etag.clone(),
                    if blob.exists() {
                        cache::PointerSource::ExistingBlob
                    } else {
                        cache::PointerSource::NewBlob
                    },
                );
                xet_metas.push(meta);
            } else {
                non_xet_filenames.push(meta.filename);
            }
        }
        if !cached_progress.is_empty() {
            params.progress.emit(DownloadEvent::Progress {
                files: cached_progress,
            });
        }

        let xet_batch_fut = async {
            if xet_metas.is_empty() {
                return Ok::<_, HFError>(());
            }
            let mut locks = Vec::with_capacity(xet_metas.len());
            for m in &xet_metas {
                locks.push(cache::acquire_lock(cache_dir, &repo_folder, &m.etag).await?);
            }
            let batch_files: Vec<crate::xet::XetBatchFile> = xet_metas
                .iter()
                .map(|m| crate::xet::XetBatchFile {
                    hash: m.xet_hash.as_ref().unwrap().clone(),
                    file_size: m.file_size,
                    path: cache::blob_path(cache_dir, &repo_folder, &m.etag),
                    filename: m.filename.clone(),
                })
                .collect();
            self.xet_download_batch(
                &commit_hash,
                &batch_files,
                params.force_download,
                &params.progress,
            )
            .await?;
            for m in &xet_metas {
                cache::create_pointer_symlink(
                    cache_dir,
                    &repo_folder,
                    &m.commit_hash,
                    &m.filename,
                    &m.etag,
                    xet_pointer_sources
                        .get(&m.etag)
                        .copied()
                        .unwrap_or(cache::PointerSource::ExistingBlob),
                )
                .await?;
            }
            drop(locks);
            Ok(())
        };

        let non_xet_dl_params = build_download_params(
            &repo_path,
            &non_xet_filenames,
            &commit_hash,
            params.force_download,
            None,
            &params.progress,
        );
        let non_xet_fut = async {
            download_concurrently(self, &non_xet_dl_params, max_workers).await?;
            Ok::<_, HFError>(())
        };

        tokio::try_join!(xet_batch_fut, non_xet_fut)?;

        if !cache::is_commit_hash(revision) {
            cache::write_ref(cache_dir, &repo_folder, revision, &commit_hash).await?;
        }

        params.progress.emit(DownloadEvent::Complete);
        Ok(cache_dir
            .join(&repo_folder)
            .join("snapshots")
            .join(&commit_hash))
    }
}

async fn mark_no_exist_and_return_error(
    cache_dir: &Path,
    repo_folder: &str,
    revision: &str,
    response: &reqwest::Response,
    repo_id: &str,
    filename: &str,
) -> HFError {
    if let Some(commit_hash) = extract_commit_hash(response) {
        let no_exist = cache::no_exist_path(cache_dir, repo_folder, &commit_hash, filename);
        if let Some(parent) = no_exist.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(&no_exist, b"");
        if !cache::is_commit_hash(revision) {
            let _ = cache::write_ref(cache_dir, repo_folder, revision, &commit_hash).await;
        }
    }
    HFError::EntryNotFound {
        path: filename.to_string(),
        repo_id: repo_id.to_string(),
        context: None,
    }
}

async fn finalize_cached_file(
    cache_dir: &Path,
    repo_folder: &str,
    revision: &str,
    commit_hash: &str,
    filename: &str,
    etag: &str,
    source: cache::PointerSource,
) -> HFResult<PathBuf> {
    if !cache::is_commit_hash(revision) {
        cache::write_ref(cache_dir, repo_folder, revision, commit_hash).await?;
    }
    cache::create_pointer_symlink(cache_dir, repo_folder, commit_hash, filename, etag, source)
        .await?;
    Ok(cache::snapshot_path(
        cache_dir,
        repo_folder,
        commit_hash,
        filename,
    ))
}

fn build_download_params(
    _repo_id: &str,
    filenames: &[String],
    commit_hash: &str,
    force_download: bool,
    local_dir: Option<PathBuf>,
    progress: &Option<Progress>,
) -> Vec<DownloadFileParams> {
    filenames
        .iter()
        .map(|filename| DownloadFileParams {
            filename: filename.clone(),
            local_dir: local_dir.clone(),
            revision: Some(commit_hash.to_string()),
            force_download,
            local_files_only: false,
            progress: progress.clone(),
        })
        .collect()
}

async fn download_concurrently<T: RepoType>(
    api: &HFRepository<T>,
    params: &[DownloadFileParams],
    max_workers: usize,
) -> HFResult<Vec<PathBuf>> {
    futures::stream::iter(params.iter().cloned().map(|p| api.download_file_impl(p)))
        .buffer_unordered(max_workers)
        .try_collect()
        .await
}

async fn stream_response_to_file_with_progress(
    response: reqwest::Response,
    dest: &Path,
    handler: &Option<Progress>,
    filename: Option<&str>,
    total_bytes: u64,
    initial_bytes: u64,
) -> HFResult<()> {
    let mut file = if initial_bytes > 0 {
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(dest)?;
        file.seek(std::io::SeekFrom::End(0))?;
        file
    } else {
        std::fs::File::create(dest)?
    };
    let mut stream = response.bytes_stream();
    let mut bytes_read = initial_bytes;

    if let (Some(h), Some(filename)) = (handler, filename) {
        h.emit(DownloadEvent::Progress {
            files: vec![FileProgress {
                filename: filename.to_string(),
                bytes_completed: initial_bytes,
                total_bytes,
                status: FileStatus::Started,
            }],
        });
    }

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk)?;
        bytes_read += chunk.len() as u64;

        if let (Some(h), Some(filename)) = (handler, filename) {
            h.emit(DownloadEvent::Progress {
                files: vec![FileProgress {
                    filename: filename.to_string(),
                    bytes_completed: bytes_read,
                    total_bytes,
                    status: FileStatus::InProgress,
                }],
            });
        }
    }
    file.flush()?;
    if total_bytes > 0 && bytes_read != total_bytes {
        return Err(HFError::Other(format!(
            "downloaded {} bytes but expected {} bytes for {}",
            bytes_read,
            total_bytes,
            filename.unwrap_or("<unknown>")
        )));
    }
    Ok(())
}

fn content_range_matches(
    headers: &reqwest::header::HeaderMap,
    expected_start: u64,
    expected_total: u64,
) -> bool {
    let Some(raw) = headers
        .get(reqwest::header::CONTENT_RANGE)
        .and_then(|value| value.to_str().ok())
    else {
        return false;
    };
    let Some(range) = raw.strip_prefix("bytes ") else {
        return false;
    };
    let Some((bounds, total)) = range.split_once('/') else {
        return false;
    };
    let Some((start, end)) = bounds.split_once('-') else {
        return false;
    };
    let Ok(start) = start.parse::<u64>() else {
        return false;
    };
    let Ok(end) = end.parse::<u64>() else {
        return false;
    };
    let total_matches = total == "*"
        || total
            .parse::<u64>()
            .is_ok_and(|total| total == expected_total);
    start == expected_start && end.saturating_add(1) <= expected_total && total_matches
}

#[cfg(target_family = "wasm")]
pub(crate) fn buffer_wasm_stream(inner: HFByteStream) -> HFByteStream {
    use futures::SinkExt;
    use futures::channel::mpsc;

    const BUFFER_DEPTH: usize = 2;
    let (mut tx, rx) = mpsc::channel::<HFResult<bytes::Bytes>>(BUFFER_DEPTH);

    wasm_bindgen_futures::spawn_local(async move {
        let mut inner = inner;
        while let Some(item) = inner.next().await {
            let is_err = item.is_err();
            if tx.send(item).await.is_err() {
                return;
            }
            if is_err {
                return;
            }
        }
    });

    Box::new(Box::pin(rx))
}

pub(crate) fn wrap_stream_with_progress(
    stream: HFByteStream,
    progress: Option<Progress>,
    filename: String,
    total_bytes: u64,
) -> HFByteStream {
    if progress.is_none() {
        return stream;
    }
    let wrapped = futures::stream::unfold(
        (stream, 0u64, false),
        move |(mut inner, bytes_completed, ended)| {
            let progress = progress.clone();
            let filename = filename.clone();
            async move {
                if ended {
                    return None;
                }
                match inner.next().await {
                    Some(Ok(chunk)) => {
                        let bytes_completed = bytes_completed + chunk.len() as u64;
                        progress.emit(DownloadEvent::Progress {
                            files: vec![FileProgress {
                                filename,
                                bytes_completed,
                                total_bytes,
                                status: FileStatus::InProgress,
                            }],
                        });
                        Some((Ok(chunk), (inner, bytes_completed, false)))
                    }
                    Some(Err(e)) => Some((Err(e), (inner, bytes_completed, true))),
                    None => {
                        progress.emit(DownloadEvent::Complete);
                        None
                    }
                }
            }
        },
    );
    Box::new(Box::pin(wrapped))
}

#[cfg(test)]
mod fallback_tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    use crate::HFClientBuilder;
    use crate::cache::storage as cache;

    const TEST_COMMIT: &str = "0123456789012345678901234567890123456789";
    const TEST_ETAG: &str = "etag-http";

    #[tokio::test]
    async fn cache_download_reuses_regular_snapshot_when_blob_is_absent() {
        let get_count = Arc::new(AtomicUsize::new(0));
        let endpoint = start_metadata_server(Arc::clone(&get_count)).await;

        let dir = tempfile::tempdir().unwrap();
        let snapshot =
            cache::snapshot_path(dir.path(), "models--owner--repo", TEST_COMMIT, "model.bin");
        std::fs::create_dir_all(snapshot.parent().unwrap()).unwrap();
        std::fs::write(&snapshot, b"cached without symlink").unwrap();
        cache::write_ref(dir.path(), "models--owner--repo", "main", TEST_COMMIT)
            .await
            .unwrap();

        let client = HFClientBuilder::new()
            .endpoint(endpoint)
            .cache_dir(dir.path())
            .build()
            .unwrap();
        let path = client
            .model("owner", "repo")
            .download_file()
            .filename("model.bin")
            .revision("main")
            .send()
            .await
            .unwrap();

        assert_eq!(path, snapshot);
        assert_eq!(std::fs::read(path).unwrap(), b"cached without symlink");
        assert_eq!(get_count.load(Ordering::Relaxed), 0);
    }

    async fn start_metadata_server(get_count: Arc<AtomicUsize>) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    return;
                };
                let get_count = Arc::clone(&get_count);
                tokio::spawn(async move {
                    let mut request = vec![0; 4096];
                    let Ok(read) = socket.read(&mut request).await else {
                        return;
                    };
                    let request = String::from_utf8_lossy(&request[..read]);
                    let is_get = request.starts_with("GET ");
                    if is_get {
                        get_count.fetch_add(1, Ordering::Relaxed);
                    }
                    let body = if is_get {
                        b"downloaded".as_slice()
                    } else {
                        b"".as_slice()
                    };
                    let response = response_bytes(body);
                    let _ = socket.write_all(&response).await;
                });
            }
        });
        format!("http://{addr}")
    }

    fn response_bytes(body: &[u8]) -> Vec<u8> {
        format!(
            "HTTP/1.1 200 OK\r\n\
             ETag: \"{TEST_ETAG}\"\r\n\
             X-Repo-Commit: {TEST_COMMIT}\r\n\
             Content-Length: {}\r\n\
             Connection: close\r\n\
             \r\n",
            body.len()
        )
        .into_bytes()
        .into_iter()
        .chain(body.iter().copied())
        .collect()
    }
}

#[bon]
impl<T: RepoType> HFRepository<T> {
    /// Download a single file from a repository.
    ///
    /// When `local_dir` is `Some`, the file is downloaded directly to that directory
    /// (no caching). When `local_dir` is `None`, the HF cache system is used:
    /// blobs are stored by etag and symlinked from snapshots/{commit}/{filename}.
    ///
    /// Returns the local filesystem path of the downloaded or cached file. Use
    /// [`HFRepository::download_file_stream`] or
    /// [`HFRepository::download_file_to_bytes`] when you do not want to write to
    /// disk.
    ///
    /// Endpoint: `GET {endpoint}/{prefix}{repo_id}/resolve/{revision}/{filename}`.
    ///
    /// # Parameters
    ///
    /// - `filename` (required): path of the file to download within the repository.
    /// - `local_dir`: local directory to download the file into. When set, the file is saved with its repo path
    ///   structure.
    /// - `revision`: Git revision. Defaults to the main branch.
    /// - `force_download` (default `false`): re-download the file even if a cached copy exists.
    /// - `local_files_only` (default `false`): only return the file if cached locally; never make a network request.
    /// - `progress`: optional progress handler.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn download_file(
        &self,
        /// Path of the file to download within the repository.
        #[builder(into)]
        filename: String,
        /// Local directory to download the file into. When set, the file is saved with its repo path structure.
        #[builder(into)]
        local_dir: Option<PathBuf>,
        /// Git revision. Defaults to the main branch.
        #[builder(into)]
        revision: Option<String>,
        /// Re-download the file even if a cached copy exists.
        #[builder(default)]
        force_download: bool,
        /// Only return the file if cached locally; never make a network request.
        #[builder(default)]
        local_files_only: bool,
        /// Progress handler.
        #[builder(into)]
        progress: Option<Progress>,
    ) -> HFResult<PathBuf> {
        Box::pin(self.download_file_impl(DownloadFileParams {
            filename,
            local_dir,
            revision,
            force_download,
            local_files_only,
            progress,
        }))
        .await
    }

    /// Download a file and return a byte stream instead of writing to disk.
    ///
    /// Returns a `(content_length, stream)` tuple. `content_length` is `Some`
    /// when the server provides a `Content-Length` header.
    ///
    /// When `range` is set, only the specified byte range is fetched.
    ///
    /// # Parameters
    ///
    /// - `filename` (required): path of the file to stream within the repository.
    /// - `revision`: Git revision. Defaults to the main branch.
    /// - `range`: byte range to request, as a Rust `std::ops::Range<u64>`. The range follows standard Rust semantics —
    ///   `start` is **inclusive**, `end` is **exclusive** — so `0..1024` fetches the first 1024 bytes (offsets
    ///   `0..=1023`). Internally, this is converted to the HTTP `Range: bytes=<start>-<end-1>` header. `start` must be
    ///   strictly less than `end`; an empty or inverted range returns [`HFError::InvalidParameter`].
    /// - `progress`: optional progress handler. `Start` is emitted before the stream is returned; `Progress` is emitted
    ///   as the caller polls each chunk; `Complete` is emitted when the stream is exhausted.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn download_file_stream(
        &self,
        /// Path of the file to stream within the repository.
        #[builder(into)]
        filename: String,
        /// Git revision. Defaults to the main branch.
        #[builder(into)]
        revision: Option<String>,
        /// Byte range to request, as a Rust `std::ops::Range<u64>`. The range follows standard Rust semantics —
        /// `start` is **inclusive**, `end` is **exclusive** — so `0..1024` fetches the first 1024 bytes (offsets
        /// `0..=1023`). Internally, this is converted to the HTTP `Range: bytes=<start>-<end-1>` header. `start` must
        /// be strictly less than `end`; an empty or inverted range returns [`HFError::InvalidParameter`].
        range: Option<std::ops::Range<u64>>,
        /// Progress handler. `Start` is emitted before the stream is returned; `Progress` is emitted
        /// as the caller polls each chunk; `Complete` is emitted when the stream is exhausted.
        #[builder(into)]
        progress: Option<Progress>,
    ) -> HFResult<(
        Option<u64>,
        Box<dyn Stream<Item = Result<bytes::Bytes, HFError>> + Send + Unpin>,
    )> {
        Box::pin(self.download_file_stream_impl(DownloadFileStreamParams {
            filename,
            revision,
            range,
            progress,
        }))
        .await
    }

    /// Download a file (or byte range) into memory and return the contents as [`bytes::Bytes`].
    ///
    /// This is a convenience wrapper around
    /// [`download_file_stream`](Self::download_file_stream) that collects the entire stream into
    /// a single buffer. When `range` is set, only the specified byte range is fetched.
    ///
    /// # Parameters
    ///
    /// - `filename` (required): path of the file to download within the repository.
    /// - `revision`: Git revision. Defaults to the main branch.
    /// - `range`: byte range to request, as a Rust `std::ops::Range<u64>`. The range follows standard Rust semantics —
    ///   `start` is **inclusive**, `end` is **exclusive** — so `0..1024` fetches the first 1024 bytes (offsets
    ///   `0..=1023`). Internally, this is converted to the HTTP `Range: bytes=<start>-<end-1>` header. `start` must be
    ///   strictly less than `end`; an empty or inverted range returns [`HFError::InvalidParameter`].
    /// - `progress`: optional progress handler. Emits `Start`/`Progress`/`Complete` as the underlying stream is
    ///   drained, identically to [`download_file_stream`](Self::download_file_stream).
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn download_file_to_bytes(
        &self,
        /// Path of the file to download within the repository.
        #[builder(into)]
        filename: String,
        /// Git revision. Defaults to the main branch.
        #[builder(into)]
        revision: Option<String>,
        /// Byte range to request, as a Rust `std::ops::Range<u64>`. The range follows standard Rust semantics —
        /// `start` is **inclusive**, `end` is **exclusive** — so `0..1024` fetches the first 1024 bytes (offsets
        /// `0..=1023`). Internally, this is converted to the HTTP `Range: bytes=<start>-<end-1>` header. `start` must
        /// be strictly less than `end`; an empty or inverted range returns [`HFError::InvalidParameter`].
        range: Option<std::ops::Range<u64>>,
        /// Progress handler. Emits `Start`/`Progress`/`Complete` as the underlying stream is
        /// drained, identically to [`HFRepository::download_file_stream`].
        #[builder(into)]
        progress: Option<Progress>,
    ) -> HFResult<bytes::Bytes> {
        Box::pin(self.download_file_to_bytes_impl(DownloadFileStreamParams {
            filename,
            revision,
            range,
            progress,
        }))
        .await
    }

    /// Download all selected files for a resolved revision.
    ///
    /// When `local_dir` is `None`, files are stored in the HF cache, and the returned path is the
    /// cache snapshot directory for the resolved commit. When `local_dir` is `Some`, files are
    /// written directly under that directory.
    ///
    /// `allow_patterns` and `ignore_patterns` use [`globset`](https://docs.rs/globset) syntax
    /// (`*`, `?`, `**`, character classes, etc.). Both are matched against each candidate file's
    /// **repository path** — forward-slash-joined and relative to the repo root, e.g.,
    /// `tokenizer.json` or `weights/model-00001-of-00003.safetensors`.
    ///
    /// # Parameters
    ///
    /// - `revision`: Git revision. Defaults to the main branch.
    /// - `allow_patterns`: globs selecting which repository files to download. When set, only files whose repo path
    ///   matches at least one pattern are downloaded.
    /// - `ignore_patterns`: globs of repository files to skip. Matched against the same repo paths as `allow_patterns`.
    /// - `local_dir`: local directory to download into.
    /// - `force_download` (default `false`): re-download all files even if cached.
    /// - `local_files_only` (default `false`): resolve only from the local cache.
    /// - `max_workers`: maximum concurrent file downloads (default 8).
    /// - `progress`: optional progress handler.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn snapshot_download(
        &self,
        /// Git revision. Defaults to the main branch.
        #[builder(into)]
        revision: Option<String>,
        /// Globs selecting which repository files to download. When set, only files whose repo path
        /// matches at least one pattern are downloaded.
        allow_patterns: Option<Vec<String>>,
        /// Globs of repository files to skip. Matched against the same repo paths as `allow_patterns`.
        ignore_patterns: Option<Vec<String>>,
        /// Local directory to download into.
        #[builder(into)]
        local_dir: Option<PathBuf>,
        /// Re-download all files even if cached.
        #[builder(default)]
        force_download: bool,
        /// Resolve only from the local cache.
        #[builder(default)]
        local_files_only: bool,
        /// Maximum concurrent file downloads (default 8).
        max_workers: Option<usize>,
        /// Progress handler.
        #[builder(into)]
        progress: Option<Progress>,
    ) -> HFResult<PathBuf> {
        Box::pin(self.snapshot_download_impl(SnapshotDownloadParams {
            revision,
            allow_patterns,
            ignore_patterns,
            local_dir,
            force_download,
            local_files_only,
            max_workers,
            progress,
        }))
        .await
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::{Arc, Mutex};

    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    use super::content_range_matches;
    use crate::HFClientBuilder;
    use crate::cache::storage as cache;

    const TEST_COMMIT: &str = "0123456789012345678901234567890123456789";
    const TEST_ETAG: &str = "etag-http";

    #[test]
    fn content_range_validation_accepts_expected_range() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::CONTENT_RANGE,
            reqwest::header::HeaderValue::from_static("bytes 4-9/10"),
        );
        assert!(content_range_matches(&headers, 4, 10));
    }

    #[test]
    fn content_range_validation_rejects_wrong_start() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::CONTENT_RANGE,
            reqwest::header::HeaderValue::from_static("bytes 0-5/10"),
        );
        assert!(!content_range_matches(&headers, 4, 10));
    }

    #[tokio::test]
    async fn cache_download_resumes_http_incomplete_blob() {
        let body = Arc::new(b"abcdefghij".to_vec());
        let ranges = Arc::new(Mutex::new(Vec::new()));
        let endpoint =
            start_http_resume_server(Arc::clone(&body), Arc::clone(&ranges), false).await;

        let dir = tempfile::tempdir().unwrap();
        write_incomplete_blob(dir.path(), b"abcd");

        let client = HFClientBuilder::new()
            .endpoint(endpoint)
            .cache_dir(dir.path())
            .build()
            .unwrap();
        let path = client
            .model("owner", "repo")
            .download_file()
            .filename("model.bin")
            .revision("main")
            .send()
            .await
            .unwrap();

        assert_eq!(std::fs::read(path).unwrap(), body.as_slice());
        assert!(
            ranges
                .lock()
                .unwrap()
                .iter()
                .any(|range| range.as_deref() == Some("bytes=4-"))
        );
    }

    #[tokio::test]
    async fn cache_download_restarts_when_partial_response_is_invalid() {
        let body = Arc::new(b"abcdefghij".to_vec());
        let ranges = Arc::new(Mutex::new(Vec::new()));
        let endpoint = start_http_resume_server(Arc::clone(&body), Arc::clone(&ranges), true).await;

        let dir = tempfile::tempdir().unwrap();
        write_incomplete_blob(dir.path(), b"abcd");

        let client = HFClientBuilder::new()
            .endpoint(endpoint)
            .cache_dir(dir.path())
            .build()
            .unwrap();
        let path = client
            .model("owner", "repo")
            .download_file()
            .filename("model.bin")
            .revision("main")
            .send()
            .await
            .unwrap();

        assert_eq!(std::fs::read(path).unwrap(), body.as_slice());
        let ranges = ranges.lock().unwrap();
        assert!(
            ranges
                .iter()
                .any(|range| range.as_deref() == Some("bytes=4-"))
        );
        assert!(ranges.iter().any(Option::is_none));
    }

    #[tokio::test]
    async fn local_dir_download_resumes_http_partial_file() {
        let body = Arc::new(b"abcdefghij".to_vec());
        let ranges = Arc::new(Mutex::new(Vec::new()));
        let endpoint =
            start_http_resume_server(Arc::clone(&body), Arc::clone(&ranges), false).await;

        let cache_dir = tempfile::tempdir().unwrap();
        let local_dir = tempfile::tempdir().unwrap();
        std::fs::write(local_dir.path().join("model.bin"), b"abcd").unwrap();

        let client = HFClientBuilder::new()
            .endpoint(endpoint)
            .cache_dir(cache_dir.path())
            .build()
            .unwrap();
        let path = client
            .model("owner", "repo")
            .download_file()
            .filename("model.bin")
            .revision("main")
            .local_dir(local_dir.path())
            .send()
            .await
            .unwrap();

        assert_eq!(std::fs::read(path).unwrap(), body.as_slice());
        assert!(
            ranges
                .lock()
                .unwrap()
                .iter()
                .any(|range| range.as_deref() == Some("bytes=4-"))
        );
    }

    fn write_incomplete_blob(cache_dir: &Path, bytes: &[u8]) {
        let blob = cache::blob_path(cache_dir, "models--owner--repo", TEST_ETAG);
        let incomplete = cache::incomplete_path(&blob);
        std::fs::create_dir_all(incomplete.parent().unwrap()).unwrap();
        std::fs::write(incomplete, bytes).unwrap();
    }

    async fn start_http_resume_server(
        body: Arc<Vec<u8>>,
        ranges: Arc<Mutex<Vec<Option<String>>>>,
        invalid_partial: bool,
    ) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    return;
                };
                let body = Arc::clone(&body);
                let ranges = Arc::clone(&ranges);
                tokio::spawn(async move {
                    let mut request = vec![0; 4096];
                    let Ok(read) = socket.read(&mut request).await else {
                        return;
                    };
                    let request = String::from_utf8_lossy(&request[..read]);
                    let range = request
                        .lines()
                        .find_map(|line| {
                            line.strip_prefix("range: ")
                                .or_else(|| line.strip_prefix("Range: "))
                        })
                        .map(str::to_string);
                    let is_head = request.starts_with("HEAD ");
                    if !is_head {
                        ranges.lock().unwrap().push(range.clone());
                    }
                    let response = http_response(&body, is_head, range.as_deref(), invalid_partial);
                    let _ = socket.write_all(&response).await;
                });
            }
        });
        format!("http://{addr}")
    }

    fn http_response(
        body: &[u8],
        is_head: bool,
        range: Option<&str>,
        invalid_partial: bool,
    ) -> Vec<u8> {
        if is_head {
            return response_bytes("200 OK", body.len(), None, &[]);
        }
        if range == Some("bytes=4-") && !invalid_partial {
            return response_bytes(
                "206 Partial Content",
                body.len() - 4,
                Some("bytes 4-9/10"),
                &body[4..],
            );
        }
        if range == Some("bytes=4-") {
            return response_bytes("206 Partial Content", 6, Some("bytes 0-5/10"), b"xxxxxx");
        }
        response_bytes("200 OK", body.len(), None, body)
    }

    fn response_bytes(
        status: &str,
        content_length: usize,
        content_range: Option<&str>,
        body: &[u8],
    ) -> Vec<u8> {
        let content_range = content_range
            .map(|value| format!("Content-Range: {value}\r\n"))
            .unwrap_or_default();
        format!(
            "HTTP/1.1 {status}\r\n\
             ETag: \"{TEST_ETAG}\"\r\n\
             X-Repo-Commit: {TEST_COMMIT}\r\n\
             Content-Length: {content_length}\r\n\
             {content_range}\
             Connection: close\r\n\
             \r\n"
        )
        .into_bytes()
        .into_iter()
        .chain(body.iter().copied())
        .collect()
    }
}

#[cfg(feature = "blocking")]
#[bon]
impl<T: RepoType> crate::blocking::HFRepositorySync<T> {
    /// Blocking counterpart of [`HFRepository::download_file`]. See the async method for
    /// parameters and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn download_file(
        &self,
        #[builder(into)] filename: String,
        #[builder(into)] local_dir: Option<PathBuf>,
        #[builder(into)] revision: Option<String>,
        #[builder(default)] force_download: bool,
        #[builder(default)] local_files_only: bool,
        #[builder(into)] progress: Option<Progress>,
    ) -> HFResult<PathBuf> {
        self.runtime.block_on(
            self.inner
                .download_file()
                .filename(filename)
                .maybe_local_dir(local_dir)
                .maybe_revision(revision)
                .force_download(force_download)
                .local_files_only(local_files_only)
                .maybe_progress(progress)
                .send(),
        )
    }

    /// Blocking counterpart of [`HFRepository::download_file_to_bytes`]. See the async method for
    /// parameters and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn download_file_to_bytes(
        &self,
        #[builder(into)] filename: String,
        #[builder(into)] revision: Option<String>,
        range: Option<std::ops::Range<u64>>,
        #[builder(into)] progress: Option<Progress>,
    ) -> HFResult<bytes::Bytes> {
        self.runtime.block_on(
            self.inner
                .download_file_to_bytes()
                .filename(filename)
                .maybe_revision(revision)
                .maybe_range(range)
                .maybe_progress(progress)
                .send(),
        )
    }

    /// Blocking counterpart of [`HFRepository::snapshot_download`]. See the async method for
    /// parameters and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn snapshot_download(
        &self,
        #[builder(into)] revision: Option<String>,
        allow_patterns: Option<Vec<String>>,
        ignore_patterns: Option<Vec<String>>,
        #[builder(into)] local_dir: Option<PathBuf>,
        #[builder(default)] force_download: bool,
        #[builder(default)] local_files_only: bool,
        max_workers: Option<usize>,
        #[builder(into)] progress: Option<Progress>,
    ) -> HFResult<PathBuf> {
        self.runtime.block_on(
            self.inner
                .snapshot_download()
                .maybe_revision(revision)
                .maybe_allow_patterns(allow_patterns)
                .maybe_ignore_patterns(ignore_patterns)
                .maybe_local_dir(local_dir)
                .force_download(force_download)
                .local_files_only(local_files_only)
                .maybe_max_workers(max_workers)
                .maybe_progress(progress)
                .send(),
        )
    }
}
