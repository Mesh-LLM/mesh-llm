//! Xet component: crate-internal token fetching and all transfer plumbing
//! (session management, upload/download groups, progress pollers) used by
//! repositories and buckets for high-performance Xet transfers.

use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use futures::StreamExt;
use serde::Deserialize;
#[cfg(test)]
use xet::error::XetError;
use xet::xet_session::{
    Sha256Policy, XetFileDownload, XetFileInfo, XetFileUpload, XetStreamUpload,
};

use crate::cache::storage as cache;
use crate::client::HFClient;
use crate::error::{HFError, HFResult, XetOperation};
use crate::progress::{DownloadEvent, EmitEvent, FileProgress, FileStatus, Progress, UploadEvent};
use crate::repository::{AddSource, HFRepository, RepoType};
use crate::retry;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct XetTokenResponse {
    access_token: String,
    exp: u64,
    cas_url: String,
}

#[derive(Default)]
pub(crate) struct XetState {
    pub(crate) session: Option<xet::xet_session::XetSession>,
    pub(crate) generation: u64,
}

pub(crate) struct XetConnectionInfo {
    pub(crate) endpoint: String,
    pub(crate) access_token: String,
    pub(crate) expiration_unix_epoch: u64,
}

async fn fetch_xet_connection_info(
    client: &HFClient,
    token_url: &str,
    not_found_id: Option<&str>,
    not_found_ctx: crate::error::NotFoundContext,
) -> HFResult<XetConnectionInfo> {
    let headers = client.auth_headers();
    let response = retry::retry(client.retry_config(), || {
        client
            .http_client()
            .get(token_url)
            .headers(headers.clone())
            .send()
    })
    .await?;

    let response = client
        .check_response(response, not_found_id, not_found_ctx)
        .await?;

    let token_resp: XetTokenResponse = response.json().await?;
    Ok(XetConnectionInfo {
        endpoint: token_resp.cas_url,
        access_token: token_resp.access_token,
        expiration_unix_epoch: token_resp.exp,
    })
}

fn repo_xet_token_url(
    client: &HFClient,
    token_type: &str,
    repo_id: &str,
    api_segment: &str,
    revision: &str,
) -> String {
    format!(
        "{}/api/{}/{}/xet-{}-token/{}",
        client.endpoint(),
        api_segment,
        repo_id,
        token_type,
        revision
    )
}

pub(crate) fn bucket_xet_token_url(client: &HFClient, token_type: &str, bucket_id: &str) -> String {
    format!(
        "{}/api/buckets/{}/xet-{}-token",
        client.endpoint(),
        bucket_id,
        token_type
    )
}

/// Returns `true` if the error indicates the XetSession is permanently
/// poisoned and must be replaced before retrying.
#[cfg(test)]
fn is_session_poisoned(err: &XetError) -> bool {
    matches!(
        err,
        XetError::UserCancelled(_)
            | XetError::AlreadyCompleted
            | XetError::PreviousTaskError(_)
            | XetError::KeyboardInterrupt
    )
}

pub(crate) struct TrackedDownload {
    pub handle: XetFileDownload,
    pub filename: String,
    pub file_size: u64,
    pub complete_emitted: AtomicBool,
}

fn emit_remaining_completes(progress: &Option<Progress>, tracked: &[TrackedDownload]) {
    let files: Vec<FileProgress> = tracked
        .iter()
        .filter(|t| !t.complete_emitted.swap(true, Ordering::Relaxed))
        .map(|t| FileProgress {
            filename: t.filename.clone(),
            bytes_completed: t.file_size,
            total_bytes: t.file_size,
            status: FileStatus::Complete,
        })
        .collect();
    if !files.is_empty() {
        progress.emit(DownloadEvent::Progress { files });
    }
}

fn emit_xet_file_progress(
    progress: &Option<Progress>,
    filename: &str,
    bytes_completed: u64,
    total_bytes: u64,
    status: FileStatus,
) {
    progress.emit(DownloadEvent::Progress {
        files: vec![FileProgress {
            filename: filename.to_string(),
            bytes_completed,
            total_bytes,
            status,
        }],
    });
}

fn emit_xet_file_complete(progress: &Option<Progress>, filename: &str, file_size: u64) {
    emit_xet_file_progress(
        progress,
        filename,
        file_size,
        file_size,
        FileStatus::Complete,
    );
}

fn spawn_download_progress_poller(
    progress: &Option<Progress>,
    group: &xet::xet_session::XetFileDownloadGroup,
    tracked: Arc<Vec<TrackedDownload>>,
) -> Option<tokio::task::JoinHandle<()>> {
    let handler = progress.as_ref()?.clone();
    let group = group.clone();
    Some(tokio::spawn(async move {
        loop {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;

            let report = group.progress();
            handler.emit(DownloadEvent::AggregateProgress {
                bytes_completed: report.total_bytes_completed,
                total_bytes: report.total_bytes,
                bytes_per_sec: report.total_bytes_completion_rate,
            });

            let mut files = Vec::new();

            for t in tracked.iter() {
                if t.complete_emitted.load(Ordering::Relaxed) {
                    continue;
                }
                if t.handle.result().is_some() {
                    if !t.complete_emitted.swap(true, Ordering::Relaxed) {
                        files.push(FileProgress {
                            filename: t.filename.clone(),
                            bytes_completed: t.file_size,
                            total_bytes: t.file_size,
                            status: FileStatus::Complete,
                        });
                    }
                    continue;
                }
                if let Some(item) = t.handle.progress() {
                    let total = item.total_bytes.max(t.file_size);
                    if item.bytes_completed >= total && total > 0 {
                        if !t.complete_emitted.swap(true, Ordering::Relaxed) {
                            files.push(FileProgress {
                                filename: t.filename.clone(),
                                bytes_completed: total,
                                total_bytes: total,
                                status: FileStatus::Complete,
                            });
                        }
                    } else {
                        let status = if item.bytes_completed > 0 {
                            FileStatus::InProgress
                        } else {
                            FileStatus::Started
                        };
                        files.push(FileProgress {
                            filename: t.filename.clone(),
                            bytes_completed: item.bytes_completed,
                            total_bytes: total,
                            status,
                        });
                    }
                }
            }

            if !files.is_empty() {
                handler.emit(DownloadEvent::Progress { files });
            }
        }
    }))
}

pub(crate) struct XetBatchFile {
    pub hash: String,
    pub file_size: u64,
    pub path: PathBuf,
    pub filename: String,
}

enum NativeAnyHandle {
    File(XetFileUpload),
    Stream(XetStreamUpload),
}

impl NativeAnyHandle {
    fn progress(&self) -> Option<xet::xet_session::ItemProgressReport> {
        match self {
            Self::File(handle) => handle.progress(),
            Self::Stream(handle) => handle.progress(),
        }
    }
}

pub(crate) struct XetBlobFile<'a> {
    pub(crate) filename: &'a str,
    pub(crate) file_hash: &'a str,
    pub(crate) file_size: u64,
    pub(crate) path: &'a std::path::Path,
    pub(crate) force_download: bool,
}

struct XetResumeFile<'a> {
    revision: &'a str,
    file_hash: &'a str,
    file_size: u64,
    path: &'a std::path::Path,
    filename: &'a str,
    offset: u64,
}

/// Shared xet upload flow for both repositories and buckets.
///
/// Callers build the xet write-token URL with the appropriate variant
/// (repo + revision, or bucket id) and pass the corresponding `owner_id`
/// and `NotFoundContext`. `owner_kind` is used as a structured tracing
/// field value (`"repo"` or `"bucket"`).
async fn xet_upload_inner(
    hf_client: &HFClient,
    files: &[(String, AddSource)],
    token_url: String,
    owner_id: &str,
    not_found_ctx: crate::error::NotFoundContext,
    owner_kind: &'static str,
    progress: &Option<Progress>,
) -> HFResult<Vec<XetFileInfo>> {
    tracing::info!(owner_kind, owner = owner_id, "fetching xet write token");
    let conn =
        fetch_xet_connection_info(hf_client, &token_url, Some(owner_id), not_found_ctx).await?;
    tracing::info!(
        endpoint = conn.endpoint.as_str(),
        "xet write token obtained, building session"
    );

    tracing::info!("building xet upload commit");
    let (session, generation) = hf_client.xet_session()?;
    let commit = match session.new_upload_commit() {
        Ok(b) => b,
        Err(e) => {
            hf_client.replace_xet_session(generation, &e);
            hf_client
                .xet_session()?
                .0
                .new_upload_commit()
                .map_err(|e| HFError::xet(XetOperation::Upload, e))?
        }
    }
    .with_endpoint(conn.endpoint.clone())
    .with_token_info(conn.access_token.clone(), conn.expiration_unix_epoch)
    .with_token_refresh_url(token_url, hf_client.auth_headers())
    .build()
    .await
    .map_err(|e| HFError::xet(XetOperation::Upload, e))?;
    tracing::info!("xet upload commit built, queuing file uploads");

    enum TaskKey {
        Id(xet::xet_session::UniqueId),
        Stream,
    }
    let mut task_keys = Vec::with_capacity(files.len());
    let mut handles = Vec::with_capacity(files.len());
    let mut item_name_to_target_path: HashMap<String, String> = HashMap::with_capacity(files.len());
    let mut stream_tasks = Vec::new();

    for (index, (target_path, source)) in files.iter().enumerate() {
        tracing::info!(path = target_path.as_str(), "queuing xet upload");
        match source {
            AddSource::File(path) => {
                // Mimic xet-core's `std::path::absolute()` logic to derive the
                // item_name that will appear in ItemProgressReport.
                // See: xet-data upload_commit.rs XetUploadCommitInner::upload_from_path
                if let Ok(abs) = std::path::absolute(path) {
                    if let Some(s) = abs.to_str() {
                        item_name_to_target_path.insert(s.to_owned(), target_path.clone());
                    } else {
                        tracing::warn!(path = ?abs, "non-UTF-8 path; per-file progress unavailable");
                    }
                }
                let handle = commit
                    .upload_from_path(path.clone(), Sha256Policy::Compute)
                    .await
                    .map_err(|e| HFError::xet(XetOperation::Upload, e))?;
                task_keys.push(TaskKey::Id(handle.task_id()));
                handles.push(NativeAnyHandle::File(handle));
            }
            AddSource::Bytes(bytes) => {
                item_name_to_target_path.insert(target_path.clone(), target_path.clone());
                let handle = commit
                    .upload_bytes(
                        bytes.to_vec(),
                        Sha256Policy::Compute,
                        Some(target_path.clone()),
                    )
                    .await
                    .map_err(|e| HFError::xet(XetOperation::Upload, e))?;
                task_keys.push(TaskKey::Id(handle.task_id()));
                handles.push(NativeAnyHandle::File(handle));
            }
            AddSource::Stream(source) => {
                item_name_to_target_path.insert(target_path.clone(), target_path.clone());
                let handle = commit
                    .upload_stream(Some(target_path.clone()), Sha256Policy::Compute)
                    .await
                    .map_err(|e| HFError::xet(XetOperation::Upload, e))?;
                let task_handle = handle.clone();
                let source = source.clone();
                stream_tasks.push((
                    index,
                    tokio::spawn(async move {
                        let mut stream = source.open();
                        while let Some(chunk) = stream.next().await {
                            task_handle
                                .write(chunk?)
                                .await
                                .map_err(|e| HFError::xet(XetOperation::Upload, e))?;
                        }
                        let metadata = task_handle
                            .finish()
                            .await
                            .map_err(|e| HFError::xet(XetOperation::Upload, e))?;
                        Ok::<_, HFError>(metadata.xet_info)
                    }),
                ));
                task_keys.push(TaskKey::Stream);
                handles.push(NativeAnyHandle::Stream(handle));
            }
        }
    }

    tracing::info!(file_count = files.len(), "committing xet uploads");
    let shared_handles: Arc<Vec<NativeAnyHandle>> = Arc::new(handles);
    let shared_name_map: Arc<HashMap<String, String>> = Arc::new(item_name_to_target_path);

    let poll_handle = progress.as_ref().map(|handler| {
        let handler = handler.clone();
        let commit = commit.clone();
        let poll_handles = Arc::clone(&shared_handles);
        let poll_name_map = Arc::clone(&shared_name_map);
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                let report = commit.progress();
                let file_progress: Vec<FileProgress> = poll_handles
                    .iter()
                    .filter_map(|h| {
                        let item = h.progress()?;
                        let target_path = poll_name_map.get(&item.item_name)?;
                        let status =
                            if item.bytes_completed >= item.total_bytes && item.total_bytes > 0 {
                                FileStatus::Complete
                            } else if item.bytes_completed > 0 {
                                FileStatus::InProgress
                            } else {
                                FileStatus::Started
                            };
                        Some(FileProgress {
                            filename: target_path.clone(),
                            bytes_completed: item.bytes_completed,
                            total_bytes: item.total_bytes,
                            status,
                        })
                    })
                    .collect();
                handler.emit(UploadEvent::Progress {
                    bytes_completed: report.total_bytes_completed,
                    total_bytes: report.total_bytes,
                    bytes_per_sec: report.total_bytes_completion_rate,
                    transfer_bytes_completed: report.total_transfer_bytes_completed,
                    transfer_bytes: report.total_transfer_bytes,
                    transfer_bytes_per_sec: report.total_transfer_bytes_completion_rate,
                    files: file_progress,
                });
            }
        })
    });
    let mut stream_infos = HashMap::new();
    for (index, task) in stream_tasks {
        let info = task
            .await
            .map_err(|e| HFError::Other(format!("xet stream upload task panicked: {e}")))??;
        stream_infos.insert(index, info);
    }
    let results = commit
        .commit()
        .await
        .map_err(|e| HFError::xet(XetOperation::Upload, e))?;
    if let Some(h) = poll_handle {
        h.abort();
    }
    tracing::info!("xet upload commit complete");

    let final_files: Vec<FileProgress> = files
        .iter()
        .map(|(target_path, source)| {
            let size = match source {
                AddSource::Bytes(b) => b.len() as u64,
                AddSource::Stream(s) => s.size(),
                AddSource::File(p) => std::fs::metadata(p).map(|m| m.len()).unwrap_or(0),
            };
            FileProgress {
                filename: target_path.clone(),
                bytes_completed: size,
                total_bytes: size,
                status: FileStatus::Complete,
            }
        })
        .collect();

    progress.emit(UploadEvent::Progress {
        bytes_completed: results.progress.total_bytes_completed,
        total_bytes: results.progress.total_bytes,
        bytes_per_sec: results.progress.total_bytes_completion_rate,
        transfer_bytes_completed: results.progress.total_transfer_bytes_completed,
        transfer_bytes: results.progress.total_transfer_bytes,
        transfer_bytes_per_sec: results.progress.total_transfer_bytes_completion_rate,
        files: final_files,
    });

    let mut xet_file_infos = Vec::with_capacity(files.len());
    for (index, key) in task_keys.into_iter().enumerate() {
        let info = match key {
            TaskKey::Id(task_id) => results
                .uploads
                .get(&task_id)
                .ok_or_else(|| HFError::Other("Missing xet upload result for task".to_string()))?
                .xet_info
                .clone(),
            TaskKey::Stream => stream_infos.remove(&index).ok_or_else(|| {
                HFError::Other(format!(
                    "missing xet stream upload result for index {index}"
                ))
            })?,
        };
        xet_file_infos.push(info);
    }

    Ok(xet_file_infos)
}

impl<T: RepoType> HFRepository<T> {
    pub(crate) async fn xet_download_to_local_dir(
        &self,
        revision: &str,
        filename: &str,
        local_dir: &std::path::Path,
        head_response: &reqwest::Response,
        progress: &Option<Progress>,
    ) -> HFResult<PathBuf> {
        let repo_path = self.repo_path();
        let api_segment = self.repo_type.plural();
        let file_hash = crate::repository::extract_xet_hash(head_response)
            .ok_or_else(|| HFError::malformed_response("missing X-Xet-Hash header"))?;

        let file_size: u64 = crate::repository::extract_file_size(head_response).unwrap_or(0);

        let token_url =
            repo_xet_token_url(&self.hf_client, "read", &repo_path, api_segment, revision);
        let conn = fetch_xet_connection_info(
            &self.hf_client,
            &token_url,
            Some(&repo_path),
            crate::error::NotFoundContext::Repo,
        )
        .await?;

        std::fs::create_dir_all(local_dir)?;
        let dest_path = local_dir.join(filename);
        if let Some(parent) = dest_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        match cache::partial_download_state(&dest_path, file_size, false)? {
            cache::PartialDownloadState::Complete => {
                emit_xet_file_complete(progress, filename, file_size);
                return Ok(dest_path);
            }
            cache::PartialDownloadState::Resume(offset) => {
                self.resume_xet_file_to_path(
                    XetResumeFile {
                        revision,
                        file_hash: &file_hash,
                        file_size,
                        path: &dest_path,
                        filename,
                        offset,
                    },
                    progress,
                )
                .await?;
                emit_xet_file_complete(progress, filename, file_size);
                return Ok(dest_path);
            }
            cache::PartialDownloadState::Restart => {}
        }

        let (session, generation) = self.hf_client.xet_session()?;
        let group = match session.new_file_download_group() {
            Ok(b) => b,
            Err(e) => {
                self.hf_client.replace_xet_session(generation, &e);
                self.hf_client
                    .xet_session()?
                    .0
                    .new_file_download_group()
                    .map_err(|e| HFError::xet(XetOperation::Download, e))?
            }
        }
        .with_endpoint(conn.endpoint.clone())
        .with_token_info(conn.access_token.clone(), conn.expiration_unix_epoch)
        .with_token_refresh_url(token_url, self.hf_client.auth_headers())
        .build()
        .await
        .map_err(|e| HFError::xet(XetOperation::Download, e))?;

        let file_info = XetFileInfo::new(file_hash, file_size);

        let handle = group
            .download_file_to_path(file_info, dest_path.clone())
            .await
            .map_err(|e| HFError::xet(XetOperation::Download, e))?;

        let tracked = Arc::new(vec![TrackedDownload {
            handle,
            filename: filename.to_string(),
            file_size,
            complete_emitted: AtomicBool::new(false),
        }]);
        let poll_handle = spawn_download_progress_poller(progress, &group, Arc::clone(&tracked));

        let result = group.finish().await;
        if let Some(h) = poll_handle {
            h.abort();
        }
        result.map_err(|e| HFError::xet(XetOperation::Download, e))?;
        emit_remaining_completes(progress, &tracked);

        Ok(dest_path)
    }

    pub(crate) async fn xet_download_to_blob(
        &self,
        revision: &str,
        file: XetBlobFile<'_>,
        progress: &Option<Progress>,
    ) -> HFResult<()> {
        let repo_path = self.repo_path();
        let api_segment = self.repo_type.plural();
        let token_url =
            repo_xet_token_url(&self.hf_client, "read", &repo_path, api_segment, revision);
        let conn = fetch_xet_connection_info(
            &self.hf_client,
            &token_url,
            Some(&repo_path),
            crate::error::NotFoundContext::Repo,
        )
        .await?;

        if let Some(parent) = file.path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let incomplete_path = cache::incomplete_path(file.path);
        match cache::partial_download_state(&incomplete_path, file.file_size, file.force_download)?
        {
            cache::PartialDownloadState::Complete => {
                std::fs::rename(&incomplete_path, file.path)?;
                emit_xet_file_complete(progress, file.filename, file.file_size);
                return Ok(());
            }
            cache::PartialDownloadState::Resume(offset) => {
                self.resume_xet_file_to_path(
                    XetResumeFile {
                        revision,
                        file_hash: file.file_hash,
                        file_size: file.file_size,
                        path: &incomplete_path,
                        filename: file.filename,
                        offset,
                    },
                    progress,
                )
                .await?;
                std::fs::rename(&incomplete_path, file.path)?;
                emit_xet_file_complete(progress, file.filename, file.file_size);
                return Ok(());
            }
            cache::PartialDownloadState::Restart => {}
        }

        let (session, generation) = self.hf_client.xet_session()?;
        let group = match session.new_file_download_group() {
            Ok(b) => b,
            Err(e) => {
                self.hf_client.replace_xet_session(generation, &e);
                self.hf_client
                    .xet_session()?
                    .0
                    .new_file_download_group()
                    .map_err(|e| HFError::xet(XetOperation::Download, e))?
            }
        }
        .with_endpoint(conn.endpoint.clone())
        .with_token_info(conn.access_token.clone(), conn.expiration_unix_epoch)
        .with_token_refresh_url(token_url, self.hf_client.auth_headers())
        .build()
        .await
        .map_err(|e| HFError::xet(XetOperation::Download, e))?;

        let file_info = XetFileInfo::new(file.file_hash.to_string(), file.file_size);

        let handle = group
            .download_file_to_path(file_info, incomplete_path.clone())
            .await
            .map_err(|e| HFError::xet(XetOperation::Download, e))?;

        let tracked = Arc::new(vec![TrackedDownload {
            handle,
            filename: file.filename.to_string(),
            file_size: file.file_size,
            complete_emitted: AtomicBool::new(false),
        }]);
        let poll_handle = spawn_download_progress_poller(progress, &group, Arc::clone(&tracked));

        let result = group.finish().await;
        if let Some(h) = poll_handle {
            h.abort();
        }
        result.map_err(|e| HFError::xet(XetOperation::Download, e))?;
        emit_remaining_completes(progress, &tracked);

        std::fs::rename(&incomplete_path, file.path)?;
        Ok(())
    }

    pub(crate) async fn xet_download_batch(
        &self,
        revision: &str,
        files: &[XetBatchFile],
        force_download: bool,
        progress: &Option<Progress>,
    ) -> HFResult<()> {
        if files.is_empty() {
            return Ok(());
        }

        let repo_path = self.repo_path();
        let api_segment = self.repo_type.plural();
        let token_url =
            repo_xet_token_url(&self.hf_client, "read", &repo_path, api_segment, revision);
        let conn = fetch_xet_connection_info(
            &self.hf_client,
            &token_url,
            Some(&repo_path),
            crate::error::NotFoundContext::Repo,
        )
        .await?;

        let (session, generation) = self.hf_client.xet_session()?;
        let group = match session.new_file_download_group() {
            Ok(b) => b,
            Err(e) => {
                self.hf_client.replace_xet_session(generation, &e);
                self.hf_client
                    .xet_session()?
                    .0
                    .new_file_download_group()
                    .map_err(|e| HFError::xet(XetOperation::BatchDownload, e))?
            }
        }
        .with_endpoint(conn.endpoint.clone())
        .with_token_info(conn.access_token.clone(), conn.expiration_unix_epoch)
        .with_token_refresh_url(token_url, self.hf_client.auth_headers())
        .build()
        .await
        .map_err(|e| HFError::xet(XetOperation::BatchDownload, e))?;

        let mut tracked_vec = Vec::with_capacity(files.len());
        let mut incomplete_paths = Vec::with_capacity(files.len());
        for file in files {
            if let Some(parent) = file.path.parent() {
                std::fs::create_dir_all(parent)?;
            }

            let incomplete = cache::incomplete_path(&file.path);

            match cache::partial_download_state(&incomplete, file.file_size, force_download)? {
                cache::PartialDownloadState::Complete => {
                    std::fs::rename(&incomplete, &file.path)?;
                    emit_xet_file_complete(progress, &file.filename, file.file_size);
                    continue;
                }
                cache::PartialDownloadState::Resume(offset) => {
                    self.resume_xet_file_to_path(
                        XetResumeFile {
                            revision,
                            file_hash: &file.hash,
                            file_size: file.file_size,
                            path: &incomplete,
                            filename: &file.filename,
                            offset,
                        },
                        progress,
                    )
                    .await?;
                    std::fs::rename(&incomplete, &file.path)?;
                    emit_xet_file_complete(progress, &file.filename, file.file_size);
                    continue;
                }
                cache::PartialDownloadState::Restart => {}
            }

            let file_info = XetFileInfo::new(file.hash.clone(), file.file_size);

            let handle = group
                .download_file_to_path(file_info, incomplete.clone())
                .await
                .map_err(|e| HFError::xet(XetOperation::BatchDownload, e))?;

            tracked_vec.push(TrackedDownload {
                handle,
                filename: file.filename.clone(),
                file_size: file.file_size,
                complete_emitted: AtomicBool::new(false),
            });
            incomplete_paths.push((incomplete, file.path.clone()));
        }

        let tracked = Arc::new(tracked_vec);
        if tracked.is_empty() {
            return Ok(());
        }
        let poll_handle = spawn_download_progress_poller(progress, &group, Arc::clone(&tracked));

        let result = group.finish().await;
        if let Some(h) = poll_handle {
            h.abort();
        }
        result.map_err(|e| HFError::xet(XetOperation::BatchDownload, e))?;
        emit_remaining_completes(progress, &tracked);

        for (incomplete, final_path) in &incomplete_paths {
            std::fs::rename(incomplete, final_path)?;
        }

        Ok(())
    }

    async fn resume_xet_file_to_path(
        &self,
        file: XetResumeFile<'_>,
        progress: &Option<Progress>,
    ) -> HFResult<()> {
        let stream = self
            .xet_download_stream(
                file.revision,
                file.file_hash,
                file.file_size,
                Some(file.offset..file.file_size),
            )
            .await?;
        append_xet_resume_stream_to_path(
            stream,
            file.path,
            file.filename,
            file.offset,
            file.file_size,
            progress,
        )
        .await
    }

    /// Download a file (or byte range) via xet and return a byte stream.
    ///
    /// Uses `XetDownloadStreamGroup` which supports `Option<Range<u64>>` for partial downloads.
    pub(crate) async fn xet_download_stream(
        &self,
        revision: &str,
        file_hash: &str,
        file_size: u64,
        range: Option<std::ops::Range<u64>>,
    ) -> HFResult<impl futures::Stream<Item = HFResult<bytes::Bytes>> + use<T>> {
        let repo_path = self.repo_path();
        let api_segment = self.repo_type.plural();
        let token_url =
            repo_xet_token_url(&self.hf_client, "read", &repo_path, api_segment, revision);
        let conn = fetch_xet_connection_info(
            &self.hf_client,
            &token_url,
            Some(&repo_path),
            crate::error::NotFoundContext::Repo,
        )
        .await?;

        let (session, generation) = self.hf_client.xet_session()?;
        let group = match session.new_download_stream_group() {
            Ok(b) => b,
            Err(e) => {
                self.hf_client.replace_xet_session(generation, &e);
                self.hf_client
                    .xet_session()?
                    .0
                    .new_download_stream_group()
                    .map_err(|e| HFError::xet(XetOperation::StreamDownload, e))?
            }
        }
        .with_endpoint(conn.endpoint.clone())
        .with_token_info(conn.access_token.clone(), conn.expiration_unix_epoch)
        .with_token_refresh_url(token_url, self.hf_client.auth_headers())
        .build()
        .await
        .map_err(|e| HFError::xet(XetOperation::StreamDownload, e))?;

        let file_info = XetFileInfo::new(file_hash.to_string(), file_size);

        let mut stream = group
            .download_stream(file_info, range)
            .await
            .map_err(|e| HFError::xet(XetOperation::StreamDownload, e))?;

        stream.start();

        Ok(futures::stream::unfold(stream, |mut stream| async move {
            match stream.next().await {
                Ok(Some(bytes)) => Some((Ok(bytes), stream)),
                Ok(None) => None,
                Err(e) => Some((Err(HFError::xet(XetOperation::StreamDownload, e)), stream)),
            }
        }))
    }

    /// Upload files using the xet protocol.
    /// Fetches a write token and uses xet-session's UploadCommit.
    /// Returns the XetFileInfo (hash + size) for each uploaded file.
    pub(crate) async fn xet_upload(
        &self,
        files: &[(String, AddSource)],
        revision: &str,
        progress: &Option<Progress>,
    ) -> HFResult<Vec<XetFileInfo>> {
        let repo_path = self.repo_path();
        let token_url = repo_xet_token_url(
            &self.hf_client,
            "write",
            &repo_path,
            self.repo_type.plural(),
            revision,
        );
        xet_upload_inner(
            &self.hf_client,
            files,
            token_url,
            &repo_path,
            crate::error::NotFoundContext::Repo,
            "repo",
            progress,
        )
        .await
    }
}

impl crate::buckets::HFBucket {
    pub(crate) async fn xet_download_stream(
        &self,
        file_hash: &str,
        file_size: u64,
    ) -> HFResult<impl futures::Stream<Item = HFResult<bytes::Bytes>> + use<>> {
        let bucket_id = self.bucket_id();
        let token_url = bucket_xet_token_url(&self.hf_client, "read", &bucket_id);
        let conn = fetch_xet_connection_info(
            &self.hf_client,
            &token_url,
            Some(&bucket_id),
            crate::error::NotFoundContext::Bucket,
        )
        .await?;

        let (session, generation) = self.hf_client.xet_session()?;
        let group = match session.new_download_stream_group() {
            Ok(builder) => builder,
            Err(error) => {
                self.hf_client.replace_xet_session(generation, &error);
                self.hf_client
                    .xet_session()?
                    .0
                    .new_download_stream_group()
                    .map_err(|e| HFError::xet(XetOperation::StreamDownload, e))?
            }
        }
        .with_endpoint(conn.endpoint)
        .with_token_info(conn.access_token, conn.expiration_unix_epoch)
        .with_token_refresh_url(token_url, self.hf_client.auth_headers())
        .build()
        .await
        .map_err(|e| HFError::xet(XetOperation::StreamDownload, e))?;

        let mut stream = group
            .download_stream(XetFileInfo::new(file_hash.to_string(), file_size), None)
            .await
            .map_err(|e| HFError::xet(XetOperation::StreamDownload, e))?;
        stream.start();

        Ok(futures::stream::unfold(stream, |mut stream| async move {
            match stream.next().await {
                Ok(Some(bytes)) => Some((Ok(bytes), stream)),
                Ok(None) => None,
                Err(error) => Some((
                    Err(HFError::xet(XetOperation::StreamDownload, error)),
                    stream,
                )),
            }
        }))
    }

    pub(crate) async fn xet_upload(
        &self,
        files: &[(String, AddSource)],
        progress: &Option<Progress>,
    ) -> HFResult<Vec<XetFileInfo>> {
        let bucket_id = self.bucket_id();
        let token_url = bucket_xet_token_url(&self.hf_client, "write", &bucket_id);
        xet_upload_inner(
            &self.hf_client,
            files,
            token_url,
            &bucket_id,
            crate::error::NotFoundContext::Bucket,
            "bucket",
            progress,
        )
        .await
    }

    pub(crate) async fn xet_download_batch(
        &self,
        files: &[XetBatchFile],
        progress: &Option<Progress>,
    ) -> HFResult<()> {
        if files.is_empty() {
            return Ok(());
        }

        let bucket_id = self.bucket_id();
        tracing::info!(
            bucket = bucket_id.as_str(),
            file_count = files.len(),
            "fetching xet read token"
        );
        let token_url = bucket_xet_token_url(&self.hf_client, "read", &bucket_id);
        let conn = fetch_xet_connection_info(
            &self.hf_client,
            &token_url,
            Some(&bucket_id),
            crate::error::NotFoundContext::Bucket,
        )
        .await?;
        tracing::info!(
            endpoint = conn.endpoint.as_str(),
            "xet download session ready, queuing files"
        );

        let (session, generation) = self.hf_client.xet_session()?;
        let group = match session.new_file_download_group() {
            Ok(b) => b,
            Err(e) => {
                self.hf_client.replace_xet_session(generation, &e);
                self.hf_client
                    .xet_session()?
                    .0
                    .new_file_download_group()
                    .map_err(|e| HFError::xet(XetOperation::BucketBatchDownload, e))?
            }
        }
        .with_endpoint(conn.endpoint.clone())
        .with_token_info(conn.access_token.clone(), conn.expiration_unix_epoch)
        .with_token_refresh_url(token_url, self.hf_client.auth_headers())
        .build()
        .await
        .map_err(|e| HFError::xet(XetOperation::BucketBatchDownload, e))?;

        let mut tracked_vec = Vec::with_capacity(files.len());
        let mut incomplete_paths = Vec::with_capacity(files.len());
        for file in files {
            if let Some(parent) = file.path.parent() {
                std::fs::create_dir_all(parent)?;
            }

            let incomplete = PathBuf::from(format!("{}.incomplete", file.path.display()));

            let file_info = XetFileInfo::new(file.hash.clone(), file.file_size);

            let handle = group
                .download_file_to_path(file_info, incomplete.clone())
                .await
                .map_err(|e| HFError::xet(XetOperation::BucketBatchDownload, e))?;

            tracked_vec.push(TrackedDownload {
                handle,
                filename: file.filename.clone(),
                file_size: file.file_size,
                complete_emitted: AtomicBool::new(false),
            });
            incomplete_paths.push((incomplete, file.path.clone()));
        }

        let tracked = Arc::new(tracked_vec);
        let poll_handle = spawn_download_progress_poller(progress, &group, Arc::clone(&tracked));

        let result = group.finish().await;
        if let Some(h) = poll_handle {
            h.abort();
        }
        result.map_err(|e| HFError::xet(XetOperation::BucketBatchDownload, e))?;
        emit_remaining_completes(progress, &tracked);

        for (incomplete, final_path) in &incomplete_paths {
            std::fs::rename(incomplete, final_path)?;
        }

        Ok(())
    }
}

async fn append_xet_resume_stream_to_path<S>(
    stream: S,
    path: &std::path::Path,
    filename: &str,
    offset: u64,
    file_size: u64,
    progress: &Option<Progress>,
) -> HFResult<()>
where
    S: futures::Stream<Item = HFResult<bytes::Bytes>>,
{
    let mut stream = Box::pin(stream);
    let mut output = std::fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open(path)?;
    let mut completed = offset;
    emit_xet_file_progress(
        progress,
        filename,
        completed,
        file_size,
        FileStatus::Started,
    );
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        output.write_all(&chunk)?;
        completed += chunk.len() as u64;
        emit_xet_file_progress(
            progress,
            filename,
            completed,
            file_size,
            FileStatus::InProgress,
        );
    }
    output.flush()?;
    if completed != file_size {
        return Err(HFError::Other(format!(
            "downloaded {} bytes but expected {} bytes for {}",
            completed, file_size, filename
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use xet::error::XetError;

    use super::*;

    #[test]
    fn test_session_poisoned_positive() {
        assert!(is_session_poisoned(&XetError::UserCancelled("test".into())));
        assert!(is_session_poisoned(&XetError::AlreadyCompleted));
        assert!(is_session_poisoned(&XetError::PreviousTaskError(
            "err".into()
        )));
        assert!(is_session_poisoned(&XetError::KeyboardInterrupt));
    }

    #[test]
    fn test_session_poisoned_negative() {
        let non_poisoned = [
            XetError::Network("timeout".into()),
            XetError::Authentication("bad token".into()),
            XetError::Io("disk full".into()),
            XetError::Internal("bug".into()),
            XetError::Timeout("slow".into()),
            XetError::NotFound("missing".into()),
            XetError::DataIntegrity("corrupt".into()),
            XetError::Configuration("bad config".into()),
            XetError::Cancelled("cancelled".into()),
            XetError::WrongRuntimeMode("wrong mode".into()),
            XetError::TaskError("task failed".into()),
        ];
        for err in &non_poisoned {
            assert!(
                !is_session_poisoned(err),
                "{err:?} should NOT be classified as poisoned"
            );
        }
    }

    #[test]
    fn test_xet_error_message_preserved_in_hferror() {
        let xet_err = XetError::Network("connection reset by peer".into());
        let hf_err = HFError::xet(XetOperation::Download, xet_err);
        let msg = hf_err.to_string();
        assert!(msg.contains("Xet download failed"), "missing prefix: {msg}");
        assert!(
            msg.contains("connection reset by peer"),
            "missing original message: {msg}"
        );

        // Variant + operation are observable to callers without parsing the message.
        match hf_err {
            HFError::Xet { operation, .. } => assert_eq!(operation, XetOperation::Download),
            other => panic!("expected HFError::Xet, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn xet_resume_stream_appends_remaining_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf.incomplete");
        std::fs::write(&path, b"abc").unwrap();
        let chunks =
            futures::stream::iter([Ok(Bytes::from_static(b"de")), Ok(Bytes::from_static(b"f"))]);

        append_xet_resume_stream_to_path(chunks, &path, "model.gguf", 3, 6, &None)
            .await
            .unwrap();

        assert_eq!(std::fs::read(&path).unwrap(), b"abcdef");
    }

    #[tokio::test]
    async fn xet_resume_stream_rejects_short_download() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf.incomplete");
        std::fs::write(&path, b"abc").unwrap();
        let chunks = futures::stream::iter([Ok(Bytes::from_static(b"de"))]);

        let err = append_xet_resume_stream_to_path(chunks, &path, "model.gguf", 3, 6, &None)
            .await
            .unwrap_err();

        assert!(
            err.to_string()
                .contains("downloaded 5 bytes but expected 6 bytes")
        );
    }
}
