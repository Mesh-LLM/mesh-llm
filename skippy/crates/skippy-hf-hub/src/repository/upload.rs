//! Repository commit, upload, and delete builders.
//!
//! Builders on [`HFRepository`] for mutating repo contents. Every change goes through a single
//! commit:
//!
//! - [`HFRepository::create_commit`] — low-level: arbitrary mix of [`CommitOperation`] entries in one commit.
//! - [`HFRepository::upload_file`] — upload one file (bytes or local path) as a single-add commit.
//! - [`HFRepository::upload_folder`] — recursively upload a local folder, with allow/ignore globs matched against
//!   `folder_path`-relative paths and a `delete_patterns` glob matched against repo-root paths.
//! - [`HFRepository::delete_file`] / [`HFRepository::delete_folder`] — single-delete and recursive-delete commits.
//!
//! See each builder's docs for the exact path / glob format rules.

use std::collections::HashMap;
use std::fmt::Write as _;
#[cfg(not(target_family = "wasm"))]
use std::io::Read;
#[cfg(not(target_family = "wasm"))]
use std::path::{Path, PathBuf};

use base64::Engine;
use bon::bon;
use futures::stream::StreamExt;
use sha2::{Digest, Sha256};

#[cfg(not(target_family = "wasm"))]
use super::files::matches_any_glob;
use super::{AddSource, CommitInfo, CommitOperation, HFRepository, RepoTreeEntry, RepoType};
use crate::client::encode_ref;
#[cfg(not(target_family = "wasm"))]
use crate::error::HFError;
use crate::error::HFResult;
use crate::progress::{EmitEvent, Progress, UploadEvent};
use crate::{constants, retry};

/// Internal options struct for [`HFRepository::create_commit`]. Built by the bon-generated
/// `create_commit()` builder.
struct CreateCommitParams {
    operations: Vec<CommitOperation>,
    commit_message: String,
    commit_description: Option<String>,
    revision: Option<String>,
    create_pr: bool,
    parent_commit: Option<String>,
    progress: Option<Progress>,
}

/// Internal options struct for [`HFRepository::upload_file`].
struct UploadFileParams {
    source: AddSource,
    path_in_repo: String,
    revision: Option<String>,
    commit_message: Option<String>,
    commit_description: Option<String>,
    create_pr: bool,
    parent_commit: Option<String>,
    progress: Option<Progress>,
}

/// Internal options struct for [`HFRepository::upload_folder`].
#[cfg(not(target_family = "wasm"))]
struct UploadFolderParams {
    folder_path: PathBuf,
    path_in_repo: Option<String>,
    revision: Option<String>,
    commit_message: Option<String>,
    commit_description: Option<String>,
    create_pr: bool,
    allow_patterns: Option<Vec<String>>,
    ignore_patterns: Option<Vec<String>>,
    delete_patterns: Option<Vec<String>>,
    progress: Option<Progress>,
}

/// Internal options struct for [`HFRepository::delete_file`].
struct DeleteFileParams {
    path_in_repo: String,
    revision: Option<String>,
    commit_message: Option<String>,
    create_pr: bool,
}

/// Internal options struct for [`HFRepository::delete_folder`].
struct DeleteFolderParams {
    path_in_repo: String,
    revision: Option<String>,
    commit_message: Option<String>,
    create_pr: bool,
}

impl<T: RepoType> HFRepository<T> {
    async fn create_commit_impl(&self, params: CreateCommitParams) -> HFResult<CommitInfo> {
        let revision = params
            .revision
            .as_deref()
            .unwrap_or(constants::DEFAULT_REVISION);
        let url = format!(
            "{}/commit/{}",
            self.hf_client
                .api_url(self.repo_type.plural(), &self.repo_path()),
            encode_ref(revision)
        );

        let add_ops_count = params
            .operations
            .iter()
            .filter(|op| matches!(op, CommitOperation::Add { .. }))
            .count();
        let total_bytes: u64 = {
            let mut total = 0u64;
            for op in &params.operations {
                if let CommitOperation::Add { source, .. } = op {
                    total += match source {
                        AddSource::Bytes(b) => b.len() as u64,
                        AddSource::Stream(s) => s.size(),
                        #[cfg(not(target_family = "wasm"))]
                        AddSource::File(p) => std::fs::metadata(p).map(|m| m.len()).unwrap_or(0),
                    };
                }
            }
            total
        };

        params.progress.emit(UploadEvent::Start {
            total_files: add_ops_count,
            total_bytes,
        });

        // Determine which files should be uploaded via xet (LFS) vs. inline
        // (regular). Files uploaded via xet are referenced by their SHA256 OID
        // in the commit NDJSON.
        let lfs_uploaded: HashMap<String, (String, u64)> = self
            .preupload_and_upload_lfs_files(&params, revision)
            .await?;

        let mut ndjson_lines: Vec<Vec<u8>> = Vec::new();

        let mut header_value = serde_json::json!({
            "summary": params.commit_message,
            "description": params.commit_description.as_deref().unwrap_or(""),
        });
        if let Some(ref parent) = params.parent_commit {
            header_value["parentCommit"] = serde_json::Value::String(parent.clone());
        }
        let header_line = serde_json::json!({"key": "header", "value": header_value});
        ndjson_lines.push(serde_json::to_vec(&header_line)?);

        for op in &params.operations {
            let line = match op {
                CommitOperation::Add {
                    path_in_repo,
                    source,
                } => {
                    if let Some((oid, size)) = lfs_uploaded.get(path_in_repo) {
                        tracing::info!(
                            path = path_in_repo.as_str(),
                            oid = oid.as_str(),
                            size,
                            "adding lfsFile entry to commit"
                        );
                        serde_json::json!({
                            "key": "lfsFile",
                            "value": {
                                "path": path_in_repo,
                                "algo": "sha256",
                                "oid": oid,
                                "size": size,
                            }
                        })
                    } else {
                        tracing::info!(
                            path = path_in_repo.as_str(),
                            "adding inline base64 file entry to commit"
                        );
                        Self::inline_base64_entry(path_in_repo, source).await?
                    }
                }
                CommitOperation::Delete { path_in_repo } => {
                    serde_json::json!({
                        "key": "deletedFile",
                        "value": {"path": path_in_repo}
                    })
                }
            };
            ndjson_lines.push(serde_json::to_vec(&line)?);
        }

        let body: Vec<u8> = ndjson_lines
            .into_iter()
            .flat_map(|mut line| {
                line.push(b'\n');
                line
            })
            .collect();

        params.progress.emit(UploadEvent::Committing);

        let mut headers = self.hf_client.auth_headers();
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            "application/x-ndjson".parse().unwrap(),
        );

        let create_pr = params.create_pr;
        let response = retry::retry(self.hf_client.retry_config(), || {
            let mut req = self
                .hf_client
                .http_client()
                .post(&url)
                .headers(headers.clone())
                .body(body.clone());
            if create_pr {
                req = req.query(&[("create_pr", "1")]);
            }
            req.send()
        })
        .await?;
        let repo_path = self.repo_path();
        let response = self
            .hf_client
            .check_response(
                response,
                Some(&repo_path),
                crate::error::NotFoundContext::Repo,
            )
            .await?;

        params.progress.emit(UploadEvent::Complete);
        Ok(response.json().await?)
    }

    async fn inline_base64_entry(
        path_in_repo: &str,
        source: &AddSource,
    ) -> HFResult<serde_json::Value> {
        let content: bytes::Bytes = match source {
            #[cfg(not(target_family = "wasm"))]
            AddSource::File(path) => bytes::Bytes::from(std::fs::read(path)?),
            AddSource::Bytes(bytes) => bytes.clone(),
            AddSource::Stream(s) => {
                let mut stream = s.open();
                let mut buf = bytes::BytesMut::with_capacity(s.size() as usize);
                while let Some(chunk) = stream.next().await {
                    buf.extend_from_slice(&chunk?);
                }
                buf.freeze()
            }
        };
        let b64 = base64::engine::general_purpose::STANDARD.encode(&content);
        Ok(serde_json::json!({
            "key": "file",
            "value": {
                "content": b64,
                "path": path_in_repo,
                "encoding": "base64",
            }
        }))
    }

    async fn upload_file_impl(&self, params: UploadFileParams) -> HFResult<CommitInfo> {
        let commit_message = params
            .commit_message
            .clone()
            .unwrap_or_else(|| format!("Upload {}", params.path_in_repo));

        self.create_commit_impl(CreateCommitParams {
            operations: vec![CommitOperation::Add {
                path_in_repo: params.path_in_repo.clone(),
                source: params.source.clone(),
            }],
            commit_message,
            commit_description: params.commit_description.clone(),
            revision: params.revision.clone(),
            create_pr: params.create_pr,
            parent_commit: params.parent_commit.clone(),
            progress: params.progress.clone(),
        })
        .await
    }

    #[cfg(not(target_family = "wasm"))]
    async fn upload_folder_impl(&self, params: UploadFolderParams) -> HFResult<CommitInfo> {
        let mut operations = Vec::new();

        let folder = &params.folder_path;
        let base_repo_path = params.path_in_repo.as_deref().unwrap_or("");

        collect_files_recursive(
            folder,
            folder,
            base_repo_path,
            &params.allow_patterns,
            &params.ignore_patterns,
            &mut operations,
        )?;

        if let Some(ref delete_patterns) = params.delete_patterns {
            let revision = params
                .revision
                .as_deref()
                .unwrap_or(constants::DEFAULT_REVISION);
            let stream = self
                .list_tree()
                .revision(revision.to_string())
                .recursive(true)
                .send()?;
            futures::pin_mut!(stream);
            while let Some(entry) = stream.next().await {
                let entry = entry?;
                if let RepoTreeEntry::File { path, .. } = entry
                    && matches_any_glob(delete_patterns, &path)
                {
                    operations.push(CommitOperation::delete(path));
                }
            }
        }

        let commit_message = params
            .commit_message
            .clone()
            .unwrap_or_else(|| "Upload folder".to_string());

        self.create_commit_impl(CreateCommitParams {
            operations,
            commit_message,
            commit_description: params.commit_description.clone(),
            revision: params.revision.clone(),
            create_pr: params.create_pr,
            parent_commit: None,
            progress: params.progress.clone(),
        })
        .await
    }

    async fn delete_file_impl(&self, params: DeleteFileParams) -> HFResult<CommitInfo> {
        let commit_message = params
            .commit_message
            .clone()
            .unwrap_or_else(|| format!("Delete {}", params.path_in_repo));

        self.create_commit_impl(CreateCommitParams {
            operations: vec![CommitOperation::delete(params.path_in_repo.clone())],
            commit_message,
            commit_description: None,
            revision: params.revision.clone(),
            create_pr: params.create_pr,
            parent_commit: None,
            progress: None,
        })
        .await
    }

    async fn delete_folder_impl(&self, params: DeleteFolderParams) -> HFResult<CommitInfo> {
        let revision = params
            .revision
            .as_deref()
            .unwrap_or(constants::DEFAULT_REVISION);

        let stream = self
            .list_tree()
            .revision(revision.to_string())
            .recursive(true)
            .send()?;
        futures::pin_mut!(stream);

        let mut operations = Vec::new();
        let prefix = if params.path_in_repo.ends_with('/') {
            params.path_in_repo.clone()
        } else {
            format!("{}/", params.path_in_repo)
        };

        while let Some(entry) = stream.next().await {
            let entry = entry?;
            if let RepoTreeEntry::File { path, .. } = entry
                && (path.starts_with(&prefix) || path == params.path_in_repo)
            {
                operations.push(CommitOperation::delete(path));
            }
        }

        let commit_message = params
            .commit_message
            .clone()
            .unwrap_or_else(|| format!("Delete {}", params.path_in_repo));

        self.create_commit_impl(CreateCommitParams {
            operations,
            commit_message,
            commit_description: None,
            revision: Some(revision.to_string()),
            create_pr: params.create_pr,
            parent_commit: None,
            progress: None,
        })
        .await
    }

    /// Check upload modes for all files and upload LFS files via xet.
    ///
    /// Always calls the preupload endpoint to determine upload mode per file.
    ///
    /// Returns a map of path_in_repo -> (sha256_oid, size) for files that were
    /// uploaded via xet and should be referenced as lfsFile in the commit.
    async fn preupload_and_upload_lfs_files(
        &self,
        params: &CreateCommitParams,
        revision: &str,
    ) -> HFResult<HashMap<String, (String, u64)>> {
        let add_ops: Vec<(&String, &AddSource)> = params
            .operations
            .iter()
            .filter_map(|op| match op {
                CommitOperation::Add {
                    path_in_repo,
                    source,
                } => Some((path_in_repo, source)),
                _ => None,
            })
            .collect();

        if add_ops.is_empty() {
            return Ok(HashMap::new());
        }

        // Step 1: Gather file info (path, size, sample, sha) for preupload check.
        // Sample + SHA are computed in a single source-read pass — see prepare_source.
        let mut file_infos: Vec<(String, u64, Vec<u8>, String, &AddSource)> = Vec::new();
        for (path_in_repo, source) in &add_ops {
            let (size, sample, sha256_oid) = prepare_source(source).await?;
            file_infos.push(((*path_in_repo).clone(), size, sample, sha256_oid, source));
        }

        // Step 2: Call preupload endpoint to classify files as "lfs" or "regular"
        tracing::info!(
            "calling preupload endpoint to classify {} files",
            file_infos.len()
        );
        let upload_modes = self
            .fetch_upload_modes(
                &self.repo_path(),
                self.repo_type.plural(),
                revision,
                &file_infos
                    .iter()
                    .map(|(path, size, sample, _, _)| (path.as_str(), *size, sample.as_slice()))
                    .collect::<Vec<_>>(),
            )
            .await?;
        tracing::info!(?upload_modes, "preupload classification complete");

        // Step 3: Identify LFS files (empty files are always regular)
        let lfs_files: Vec<&(String, u64, Vec<u8>, String, &AddSource)> = file_infos
            .iter()
            .filter(|(path, size, _, _, _)| {
                *size > 0
                    && upload_modes
                        .get(path.as_str())
                        .map(|m| m == "lfs")
                        .unwrap_or(false)
            })
            .collect();

        if lfs_files.is_empty() {
            return Ok(HashMap::new());
        }

        tracing::info!(
            lfs_file_count = lfs_files.len(),
            lfs_files = ?lfs_files.iter().map(|(p, s, _, _, _)| (p.as_str(), *s)).collect::<Vec<_>>(),
            "files requiring LFS upload"
        );

        self.upload_lfs_files_via_xet(params, revision, &lfs_files)
            .await
    }

    /// Call the Hub preupload endpoint to determine the upload mode per file.
    /// Returns a map of path -> upload mode ("lfs" or "regular").
    async fn fetch_upload_modes(
        &self,
        repo_id: &str,
        api_segment: &str,
        revision: &str,
        files: &[(&str, u64, &[u8])],
    ) -> HFResult<HashMap<String, String>> {
        let url = format!(
            "{}/preupload/{}",
            self.hf_client.api_url(api_segment, repo_id),
            encode_ref(revision)
        );

        let files_payload: Vec<serde_json::Value> = files
            .iter()
            .map(|(path, size, sample)| {
                serde_json::json!({
                    "path": path,
                    "size": size,
                    "sample": base64::engine::general_purpose::STANDARD.encode(sample),
                })
            })
            .collect();

        let body = serde_json::json!({ "files": files_payload });

        let headers = self.hf_client.auth_headers();
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
            .check_response(response, Some(repo_id), crate::error::NotFoundContext::Repo)
            .await?;

        let preupload: PreuploadResponse = response.json().await?;

        Ok(preupload
            .files
            .into_iter()
            .map(|f| (f.path, f.upload_mode))
            .collect())
    }

    /// Compute SHA256, negotiate LFS batch transfer, and upload via xet.
    async fn upload_lfs_files_via_xet(
        &self,
        params: &CreateCommitParams,
        revision: &str,
        lfs_files: &[&(String, u64, Vec<u8>, String, &AddSource)],
    ) -> HFResult<HashMap<String, (String, u64)>> {
        // Step 4: SHA-256 was already computed alongside the preupload sample in
        // `prepare_source` — see the per-file `file_infos` tuple. We just
        // unpack it here instead of re-reading each source.
        tracing::info!(
            "collecting pre-computed SHA256 for {} LFS files",
            lfs_files.len()
        );
        let lfs_with_sha: Vec<(String, u64, String, &AddSource)> = lfs_files
            .iter()
            .map(|(path, size, _, sha256_oid, source)| {
                tracing::info!(
                    path = path.as_str(),
                    size = *size,
                    oid = sha256_oid.as_str(),
                    "SHA256 reused"
                );
                ((*path).clone(), *size, sha256_oid.clone(), *source)
            })
            .collect();

        // Step 5: Call LFS batch endpoint to negotiate transfer method
        let objects: Vec<(&str, u64)> = lfs_with_sha
            .iter()
            .map(|(_, size, oid, _)| (oid.as_str(), *size))
            .collect();

        let repo_path = self.repo_path();
        tracing::info!("calling LFS batch endpoint for transfer negotiation");
        let chosen_transfer = self
            .post_lfs_batch_info(&repo_path, self.repo_type.url_prefix(), revision, &objects)
            .await?;
        tracing::info!(?chosen_transfer, "LFS batch transfer negotiation complete");

        // Step 6: If server chose xet, upload via xet
        if chosen_transfer.as_deref() != Some("xet") {
            tracing::warn!(
                ?chosen_transfer,
                "LFS batch did not choose xet transfer; LFS files will fall through to inline upload"
            );
            return Ok(HashMap::new());
        }

        let xet_files: Vec<(String, AddSource)> = lfs_with_sha
            .iter()
            .map(|(path, _, _, source)| (path.clone(), (*source).clone()))
            .collect();

        self.xet_upload(&xet_files, revision, &params.progress)
            .await?;

        let result: HashMap<String, (String, u64)> = lfs_with_sha
            .into_iter()
            .map(|(path, size, oid, _)| (path, (oid, size)))
            .collect();

        Ok(result)
    }

    /// Call the LFS batch endpoint to negotiate the transfer method.
    /// Returns the chosen transfer (e.g., "xet", "basic", "multipart").
    async fn post_lfs_batch_info(
        &self,
        repo_id: &str,
        url_prefix: &str,
        revision: &str,
        objects: &[(&str, u64)],
    ) -> HFResult<Option<String>> {
        let url = format!(
            "{}/{}{}.git/info/lfs/objects/batch",
            self.hf_client.endpoint(),
            url_prefix,
            repo_id
        );

        let objects_payload: Vec<serde_json::Value> = objects
            .iter()
            .map(|(oid, size)| {
                serde_json::json!({
                    "oid": oid,
                    "size": size,
                })
            })
            .collect();

        let body = serde_json::json!({
            "operation": "upload",
            "transfers": ["basic", "multipart", "xet"],
            "objects": objects_payload,
            "hash_algo": "sha256",
            "ref": { "name": revision },
        });

        let mut headers = self.hf_client.auth_headers();
        headers.insert(
            reqwest::header::ACCEPT,
            "application/vnd.git-lfs+json".parse().unwrap(),
        );
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            "application/vnd.git-lfs+json".parse().unwrap(),
        );

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
            .check_response(response, Some(repo_id), crate::error::NotFoundContext::Repo)
            .await?;

        let batch: LfsBatchResponse = response.json().await?;
        Ok(batch.transfer)
    }
}

// --- Preupload and LFS upload integration ---

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct PreuploadFileInfo {
    path: String,
    upload_mode: String,
}

#[derive(Debug, serde::Deserialize)]
struct PreuploadResponse {
    files: Vec<PreuploadFileInfo>,
}

#[derive(Debug, serde::Deserialize)]
struct LfsBatchResponse {
    transfer: Option<String>,
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

const PREUPLOAD_SAMPLE_SIZE: usize = 512;

// Compute size, preupload sample, and SHA-256 in a single pass so a stream
// source is only read twice total: the metadata pass here and the xet upload.
async fn prepare_source(source: &AddSource) -> HFResult<(u64, Vec<u8>, String)> {
    match source {
        AddSource::Bytes(bytes) => {
            let sample = bytes[..std::cmp::min(bytes.len(), PREUPLOAD_SAMPLE_SIZE)].to_vec();
            let hash = Sha256::digest(bytes);
            Ok((bytes.len() as u64, sample, hex_encode(&hash)))
        }
        AddSource::Stream(s) => {
            let mut stream = s.open();
            let mut sample: Vec<u8> = Vec::with_capacity(PREUPLOAD_SAMPLE_SIZE);
            let mut hasher = Sha256::new();
            while let Some(chunk) = stream.next().await {
                let chunk = chunk?;
                hasher.update(&chunk);
                if sample.len() < PREUPLOAD_SAMPLE_SIZE {
                    let take = std::cmp::min(PREUPLOAD_SAMPLE_SIZE - sample.len(), chunk.len());
                    sample.extend_from_slice(&chunk[..take]);
                }
            }
            Ok((s.size(), sample, hex_encode(&hasher.finalize())))
        }
        #[cfg(not(target_family = "wasm"))]
        AddSource::File(path) => {
            let path = path.clone();
            tokio::task::spawn_blocking(move || -> HFResult<(u64, Vec<u8>, String)> {
                let mut file = std::fs::File::open(&path)?;
                let size = file.metadata()?.len();
                let mut hasher = Sha256::new();
                let mut sample: Vec<u8> = Vec::with_capacity(PREUPLOAD_SAMPLE_SIZE);
                let mut buf = vec![0u8; 64 * 1024];
                loop {
                    let n = file.read(&mut buf)?;
                    if n == 0 {
                        break;
                    }
                    hasher.update(&buf[..n]);
                    if sample.len() < PREUPLOAD_SAMPLE_SIZE {
                        let take = std::cmp::min(PREUPLOAD_SAMPLE_SIZE - sample.len(), n);
                        sample.extend_from_slice(&buf[..take]);
                    }
                }
                Ok((size, sample, hex_encode(&hasher.finalize())))
            })
            .await
            .map_err(|e| HFError::Other(format!("prepare_source task failed: {e}")))?
        }
    }
}

/// Recursively collect files from a directory into CommitOperation::Add entries.
/// Respects allow_patterns and ignore_patterns (glob-style).
#[cfg(not(target_family = "wasm"))]
fn collect_files_recursive(
    root: &Path,
    current: &Path,
    base_repo_path: &str,
    allow_patterns: &Option<Vec<String>>,
    ignore_patterns: &Option<Vec<String>>,
    operations: &mut Vec<CommitOperation>,
) -> HFResult<()> {
    for entry in std::fs::read_dir(current)? {
        let entry = entry?;
        let path = entry.path();
        let metadata = entry.metadata()?;

        if metadata.is_dir() {
            collect_files_recursive(
                root,
                &path,
                base_repo_path,
                allow_patterns,
                ignore_patterns,
                operations,
            )?;
        } else if metadata.is_file() {
            let relative = path.strip_prefix(root).map_err(|e| {
                HFError::InvalidParameter(format!(
                    "path {} is not under {}: {e}",
                    path.display(),
                    root.display()
                ))
            })?;
            let relative_str: String = relative
                .components()
                .filter_map(|c| match c {
                    std::path::Component::Normal(s) => Some(s.to_string_lossy().into_owned()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("/");

            if let Some(allow) = allow_patterns
                && !matches_any_glob(allow, &relative_str)
            {
                continue;
            }
            if let Some(ignore) = ignore_patterns
                && matches_any_glob(ignore, &relative_str)
            {
                continue;
            }

            let repo_path = if base_repo_path.is_empty() {
                relative_str
            } else {
                format!("{}/{}", base_repo_path.trim_end_matches('/'), relative_str)
            };

            operations.push(CommitOperation::add_file(repo_path, path));
        }
    }

    Ok(())
}

#[bon]
impl<T: RepoType> HFRepository<T> {
    /// Create a commit with multiple operations.
    ///
    /// This is the lowest-level public mutation API in the files' module. Use it when you need an
    /// explicit mix of add and delete operations in one commit. For one-shot workflows, prefer
    /// [`HFRepository::upload_file`], [`HFRepository::upload_folder`],
    /// [`HFRepository::delete_file`], or [`HFRepository::delete_folder`].
    ///
    /// Endpoint: `POST /api/{repo_type}s/{repo_id}/commit/{revision}`.
    ///
    /// # Parameters
    ///
    /// - `operations` (required): list of file operations to include in the commit.
    /// - `commit_message` (required): commit message.
    /// - `commit_description`: extended description for the commit.
    /// - `revision`: branch to commit to. Defaults to the main branch.
    /// - `create_pr` (default `false`): create a pull request instead of committing directly.
    /// - `parent_commit`: expected parent commit SHA. Fails if the branch head moved past it.
    /// - `progress`: optional progress handler.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn create_commit(
        &self,
        /// List of file operations to include in the commit.
        operations: Vec<CommitOperation>,
        /// Commit message.
        #[builder(into)]
        commit_message: String,
        /// Extended description for the commit.
        #[builder(into)]
        commit_description: Option<String>,
        /// Branch to commit to. Defaults to the main branch.
        #[builder(into)]
        revision: Option<String>,
        /// Create a pull request instead of committing directly.
        #[builder(default)]
        create_pr: bool,
        /// Expected parent commit SHA. Fails if the branch head moved past it.
        #[builder(into)]
        parent_commit: Option<String>,
        /// Progress handler.
        #[builder(into)]
        progress: Option<Progress>,
    ) -> HFResult<CommitInfo> {
        Box::pin(self.create_commit_impl(CreateCommitParams {
            operations,
            commit_message,
            commit_description,
            revision,
            create_pr,
            parent_commit,
            progress,
        }))
        .await
    }

    /// Upload a single file to a repository.
    ///
    /// Convenience wrapper around [`HFRepository::create_commit`]. If `commit_message` is
    /// omitted, a default `"Upload {path}"` message is used.
    ///
    /// # Parameters
    ///
    /// - `source` (required): file content source (bytes or local file path).
    /// - `path_in_repo` (required): destination path within the repository.
    /// - `revision`: branch to upload to. Defaults to the main branch.
    /// - `commit_message`, `commit_description`, `create_pr`, `parent_commit`, `progress`: same as
    ///   [`HFRepository::create_commit`]. `create_pr` defaults to `false`.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn upload_file(
        &self,
        /// File content source.
        source: AddSource,
        /// Destination path within the repository.
        #[builder(into)]
        path_in_repo: String,
        /// Branch to upload to. Defaults to the main branch.
        #[builder(into)]
        revision: Option<String>,
        /// Commit message. Same as [`HFRepository::create_commit`].
        #[builder(into)]
        commit_message: Option<String>,
        /// Extended description for the commit. Same as [`HFRepository::create_commit`].
        #[builder(into)]
        commit_description: Option<String>,
        /// Create a pull request instead of committing directly. Same as [`HFRepository::create_commit`].
        #[builder(default)]
        create_pr: bool,
        /// Expected parent commit SHA. Same as [`HFRepository::create_commit`].
        #[builder(into)]
        parent_commit: Option<String>,
        /// Progress handler. Same as [`HFRepository::create_commit`].
        #[builder(into)]
        progress: Option<Progress>,
    ) -> HFResult<CommitInfo> {
        Box::pin(self.upload_file_impl(UploadFileParams {
            source,
            path_in_repo,
            revision,
            commit_message,
            commit_description,
            create_pr,
            parent_commit,
            progress,
        }))
        .await
    }

    /// Upload a local folder to a repository.
    ///
    /// The folder is walked recursively and converted into add operations. When `delete_patterns`
    /// is set, matching remote files are also deleted in the same commit.
    ///
    /// All pattern arguments use [`globset`](https://docs.rs/globset) syntax (`*`, `?`, `**`,
    /// character classes, etc.). Path strings are forward-slash-joined regardless of platform.
    ///
    /// # Parameters
    ///
    /// - `folder_path` (required): local folder path to upload.
    /// - `path_in_repo`: destination directory within the repository (default: repo root).
    /// - `revision`: branch to upload to. Defaults to the main branch.
    /// - `commit_message`, `commit_description`: commit metadata.
    /// - `create_pr` (default `false`): create a pull request instead of committing directly.
    /// - `allow_patterns`: globs selecting which local files to include. Matched against each discovered file's path
    ///   relative to `folder_path` (e.g., `data/train.bin`, not the absolute path and not prefixed with
    ///   `path_in_repo`). When set, only files matching at least one pattern are uploaded.
    /// - `ignore_patterns`: globs of local files to skip. Matched against the same `folder_path`-relative paths as
    ///   `allow_patterns`.
    /// - `delete_patterns`: globs of *remote* files to delete in the same commit. Matched against each existing file's
    ///   full repository path (relative to repo root, **not** relative to `path_in_repo`) — e.g., `old/*.bin` to remove
    ///   every `.bin` directly under `old/` at the repo root.
    /// - `progress`: optional progress handler.
    #[cfg(not(target_family = "wasm"))]
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn upload_folder(
        &self,
        /// Local folder path to upload.
        #[builder(into)]
        folder_path: PathBuf,
        /// Destination directory within the repository (default: repo root).
        #[builder(into)]
        path_in_repo: Option<String>,
        /// Branch to upload to. Defaults to the main branch.
        #[builder(into)]
        revision: Option<String>,
        /// Commit message.
        #[builder(into)]
        commit_message: Option<String>,
        /// Extended description for the commit.
        #[builder(into)]
        commit_description: Option<String>,
        /// Create a pull request instead of committing directly.
        #[builder(default)]
        create_pr: bool,
        /// Globs selecting which local files to include. Matched against each discovered file's path
        /// relative to `folder_path` (e.g., `data/train.bin`, not the absolute path and not prefixed with
        /// `path_in_repo`). When set, only files matching at least one pattern are uploaded.
        allow_patterns: Option<Vec<String>>,
        /// Globs of local files to skip. Matched against the same `folder_path`-relative paths as
        /// `allow_patterns`.
        ignore_patterns: Option<Vec<String>>,
        /// Globs of *remote* files to delete in the same commit. Matched against each existing file's
        /// full repository path (relative to repo root, **not** relative to `path_in_repo`) — e.g., `old/*.bin` to
        /// remove every `.bin` directly under `old/` at the repo root.
        delete_patterns: Option<Vec<String>>,
        /// Progress handler.
        #[builder(into)]
        progress: Option<Progress>,
    ) -> HFResult<CommitInfo> {
        Box::pin(self.upload_folder_impl(UploadFolderParams {
            folder_path,
            path_in_repo,
            revision,
            commit_message,
            commit_description,
            create_pr,
            allow_patterns,
            ignore_patterns,
            delete_patterns,
            progress,
        }))
        .await
    }

    /// Delete a file from a repository.
    ///
    /// Convenience wrapper around [`HFRepository::create_commit`]. If `commit_message` is
    /// omitted, a default `"Delete {path}"` message is used.
    ///
    /// # Parameters
    ///
    /// - `path_in_repo` (required): path of the file to delete.
    /// - `revision`: branch to delete from. Defaults to the main branch.
    /// - `commit_message`: commit message.
    /// - `create_pr` (default `false`): create a pull request instead of committing directly.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn delete_file(
        &self,
        /// Path of the file to delete.
        #[builder(into)]
        path_in_repo: String,
        /// Branch to delete from. Defaults to the main branch.
        #[builder(into)]
        revision: Option<String>,
        /// Commit message.
        #[builder(into)]
        commit_message: Option<String>,
        /// Create a pull request instead of committing directly.
        #[builder(default)]
        create_pr: bool,
    ) -> HFResult<CommitInfo> {
        Box::pin(self.delete_file_impl(DeleteFileParams {
            path_in_repo,
            revision,
            commit_message,
            create_pr,
        }))
        .await
    }

    /// Delete all files under a repository path.
    ///
    /// The current tree is listed recursively and every file at or below `path_in_repo` is turned
    /// into a delete operation. Directories disappear as a consequence of deleting their contents.
    ///
    /// # Parameters
    ///
    /// - `path_in_repo` (required): folder path within the repository.
    /// - `revision`: branch to delete from. Defaults to the main branch.
    /// - `commit_message`: commit message.
    /// - `create_pr` (default `false`): create a pull request instead of committing directly.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn delete_folder(
        &self,
        /// Folder path within the repository.
        #[builder(into)]
        path_in_repo: String,
        /// Branch to delete from. Defaults to the main branch.
        #[builder(into)]
        revision: Option<String>,
        /// Commit message.
        #[builder(into)]
        commit_message: Option<String>,
        /// Create a pull request instead of committing directly.
        #[builder(default)]
        create_pr: bool,
    ) -> HFResult<CommitInfo> {
        Box::pin(self.delete_folder_impl(DeleteFolderParams {
            path_in_repo,
            revision,
            commit_message,
            create_pr,
        }))
        .await
    }
}

#[cfg(feature = "blocking")]
#[bon]
impl<T: RepoType> crate::blocking::HFRepositorySync<T> {
    /// Blocking counterpart of [`HFRepository::create_commit`]. See the async method for
    /// parameters and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn create_commit(
        &self,
        operations: Vec<CommitOperation>,
        #[builder(into)] commit_message: String,
        #[builder(into)] commit_description: Option<String>,
        #[builder(into)] revision: Option<String>,
        #[builder(default)] create_pr: bool,
        #[builder(into)] parent_commit: Option<String>,
        #[builder(into)] progress: Option<Progress>,
    ) -> HFResult<CommitInfo> {
        self.runtime.block_on(
            self.inner
                .create_commit()
                .operations(operations)
                .commit_message(commit_message)
                .maybe_commit_description(commit_description)
                .maybe_revision(revision)
                .create_pr(create_pr)
                .maybe_parent_commit(parent_commit)
                .maybe_progress(progress)
                .send(),
        )
    }

    /// Blocking counterpart of [`HFRepository::upload_file`]. See the async method for parameters
    /// and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn upload_file(
        &self,
        source: AddSource,
        #[builder(into)] path_in_repo: String,
        #[builder(into)] revision: Option<String>,
        #[builder(into)] commit_message: Option<String>,
        #[builder(into)] commit_description: Option<String>,
        #[builder(default)] create_pr: bool,
        #[builder(into)] parent_commit: Option<String>,
        #[builder(into)] progress: Option<Progress>,
    ) -> HFResult<CommitInfo> {
        self.runtime.block_on(
            self.inner
                .upload_file()
                .source(source)
                .path_in_repo(path_in_repo)
                .maybe_revision(revision)
                .maybe_commit_message(commit_message)
                .maybe_commit_description(commit_description)
                .create_pr(create_pr)
                .maybe_parent_commit(parent_commit)
                .maybe_progress(progress)
                .send(),
        )
    }

    /// Blocking counterpart of [`HFRepository::upload_folder`]. See the async method for
    /// parameters and behavior.
    #[cfg(not(target_family = "wasm"))]
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn upload_folder(
        &self,
        #[builder(into)] folder_path: PathBuf,
        #[builder(into)] path_in_repo: Option<String>,
        #[builder(into)] revision: Option<String>,
        #[builder(into)] commit_message: Option<String>,
        #[builder(into)] commit_description: Option<String>,
        #[builder(default)] create_pr: bool,
        allow_patterns: Option<Vec<String>>,
        ignore_patterns: Option<Vec<String>>,
        delete_patterns: Option<Vec<String>>,
        #[builder(into)] progress: Option<Progress>,
    ) -> HFResult<CommitInfo> {
        self.runtime.block_on(
            self.inner
                .upload_folder()
                .folder_path(folder_path)
                .maybe_path_in_repo(path_in_repo)
                .maybe_revision(revision)
                .maybe_commit_message(commit_message)
                .maybe_commit_description(commit_description)
                .create_pr(create_pr)
                .maybe_allow_patterns(allow_patterns)
                .maybe_ignore_patterns(ignore_patterns)
                .maybe_delete_patterns(delete_patterns)
                .maybe_progress(progress)
                .send(),
        )
    }

    /// Blocking counterpart of [`HFRepository::delete_file`]. See the async method for parameters
    /// and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn delete_file(
        &self,
        #[builder(into)] path_in_repo: String,
        #[builder(into)] revision: Option<String>,
        #[builder(into)] commit_message: Option<String>,
        #[builder(default)] create_pr: bool,
    ) -> HFResult<CommitInfo> {
        self.runtime.block_on(
            self.inner
                .delete_file()
                .path_in_repo(path_in_repo)
                .maybe_revision(revision)
                .maybe_commit_message(commit_message)
                .create_pr(create_pr)
                .send(),
        )
    }

    /// Blocking counterpart of [`HFRepository::delete_folder`]. See the async method for
    /// parameters and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn delete_folder(
        &self,
        #[builder(into)] path_in_repo: String,
        #[builder(into)] revision: Option<String>,
        #[builder(into)] commit_message: Option<String>,
        #[builder(default)] create_pr: bool,
    ) -> HFResult<CommitInfo> {
        self.runtime.block_on(
            self.inner
                .delete_folder()
                .path_in_repo(path_in_repo)
                .maybe_revision(revision)
                .maybe_commit_message(commit_message)
                .create_pr(create_pr)
                .send(),
        )
    }
}
