//! Repository content listing and metadata lookups.
//!
//! Builders on [`HFRepository`] for inspecting what's in a repo without downloading file
//! contents:
//!
//! - [`HFRepository::list_tree`] — paginated stream of [`RepoTreeEntry`] (files and directories), optionally recursive
//!   and prefix-filtered.
//! - [`HFRepository::get_paths_info`] — batched lookup of metadata for a known set of paths.
//! - [`HFRepository::get_file_metadata`] — HEAD-based metadata for one file (size, ETag, commit hash, xet hash).

use bon::bon;
use futures::stream::Stream;
use reqwest::Url;

use super::files::{extract_commit_hash, extract_etag, extract_file_size, extract_xet_hash};
use super::{FileMetadataInfo, HFRepository, RepoTreeEntry, RepoType};
use crate::client::encode_ref;
use crate::error::{HFError, HFResult};
use crate::{constants, retry};

#[bon]
impl<T: RepoType> HFRepository<T> {
    /// Stream file and directory entries in the repository tree.
    ///
    /// Returns `HFResult<impl Stream<Item = HFResult<RepoTreeEntry>>>`.
    ///
    /// Use [`HFRepository::get_paths_info`] when you already know the exact paths
    /// you want to inspect.
    ///
    /// # Parameters
    ///
    /// - `revision`: Git revision to list. Defaults to the main branch.
    /// - `path_in_repo`: repository-relative subdirectory to list. Defaults to the repo root. Leading, trailing, and
    ///   consecutive `/` separators are ignored; path segments are URL-encoded automatically.
    /// - `recursive` (default `false`): traverse subdirectories.
    /// - `expand` (default `false`): include per-file metadata such as size, LFS info, and last-commit summaries.
    /// - `limit`: cap the total number of entries yielded.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn list_tree(
        &self,
        /// Git revision to list. Defaults to the main branch.
        #[builder(into)]
        revision: Option<String>,
        /// Repository-relative subdirectory to list. Defaults to the repo root.
        /// Leading, trailing, and consecutive `/` separators are ignored;
        /// path segments are URL-encoded automatically.
        #[builder(into)]
        path_in_repo: Option<String>,
        /// Traverse subdirectories.
        #[builder(default)]
        recursive: bool,
        /// Include per-file metadata such as size, LFS info, and last-commit summaries.
        #[builder(default)]
        expand: bool,
        /// Cap the total number of entries yielded.
        limit: Option<usize>,
    ) -> HFResult<impl Stream<Item = HFResult<RepoTreeEntry>> + '_> {
        let revision = revision.as_deref().unwrap_or(constants::DEFAULT_REVISION);
        let url_str = format!(
            "{}/tree/{}",
            self.hf_client
                .api_url(self.repo_type.plural(), &self.repo_path()),
            encode_ref(revision)
        );
        let mut url = Url::parse(&url_str)?;
        if let Some(path) = path_in_repo.as_deref() {
            crate::client::append_path_segments(&mut url, path)?;
        }

        let mut query: Vec<(String, String)> = Vec::new();
        if recursive {
            query.push(("recursive".into(), "true".into()));
        }
        if expand {
            query.push(("expand".into(), "true".into()));
        }

        Ok(self.hf_client.paginate(url, query, limit))
    }

    /// Get info about specific paths in a repository.
    ///
    /// Prefer this over [`HFRepository::list_tree`] when you already know the
    /// small set of paths you want to inspect.
    ///
    /// Endpoint: `POST /api/{repo_type}s/{repo_id}/paths-info/{revision}`.
    ///
    /// # Parameters
    ///
    /// - `paths` (required): paths in the repository to fetch info for.
    /// - `revision`: Git revision. Defaults to the main branch.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn get_paths_info(
        &self,
        /// Paths in the repository to fetch info for.
        paths: Vec<String>,
        /// Git revision. Defaults to the main branch.
        #[builder(into)]
        revision: Option<String>,
    ) -> HFResult<Vec<RepoTreeEntry>> {
        let revision = revision.as_deref().unwrap_or(constants::DEFAULT_REVISION);
        let url = format!(
            "{}/paths-info/{}",
            self.hf_client
                .api_url(self.repo_type.plural(), &self.repo_path()),
            encode_ref(revision)
        );

        let body = serde_json::json!({ "paths": paths });

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

        let repo_path = self.repo_path();
        let response = self
            .hf_client
            .check_response(
                response,
                Some(&repo_path),
                crate::error::NotFoundContext::Entry {
                    path: paths.join(", "),
                },
            )
            .await?;
        Ok(response.json().await?)
    }

    /// Fetch metadata for a single file via a HEAD request on its resolve URL.
    ///
    /// Returns the resolved commit hash, ETag, file size, and (if the file is Xet-backed)
    /// the Xet content hash — without downloading the file contents.
    ///
    /// Endpoint: `HEAD {endpoint}/{prefix}{repo_id}/resolve/{revision}/{filepath}`.
    ///
    /// # Parameters
    ///
    /// - `filepath` (required): path of the file to inspect within the repository.
    /// - `revision`: Git revision. Defaults to the main branch.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn get_file_metadata(
        &self,
        /// Path of the file to inspect within the repository.
        #[builder(into)]
        filepath: String,
        /// Git revision. Defaults to the main branch.
        revision: Option<&str>,
    ) -> HFResult<FileMetadataInfo> {
        let filename = filepath;
        let revision = revision.unwrap_or(constants::DEFAULT_REVISION);
        let repo_path = self.repo_path();
        let url = self.hf_client.download_url(
            self.repo_type.url_prefix(),
            &repo_path,
            revision,
            &filename,
        )?;

        let headers = self.hf_client.auth_headers();
        let response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client
                .http_client()
                .head(&url)
                .headers(headers.clone())
                .send()
        })
        .await?;
        let response = self
            .hf_client
            .check_response(
                response,
                Some(&repo_path),
                crate::error::NotFoundContext::Entry {
                    path: filename.clone(),
                },
            )
            .await?;

        let etag = extract_etag(&response).ok_or_else(|| {
            HFError::malformed_response_at(
                format!("missing ETag header for {filename}"),
                url.to_string(),
            )
        })?;
        let commit_hash = extract_commit_hash(&response).ok_or_else(|| {
            HFError::malformed_response_at(
                format!("missing X-Repo-Commit header for {filename}"),
                url.to_string(),
            )
        })?;
        let xet_hash = extract_xet_hash(&response);
        let file_size = extract_file_size(&response).unwrap_or_else(|| {
            tracing::warn!(
                file = %filename,
                "missing or invalid Content-Length/X-Linked-Size header, defaulting file size to 0"
            );
            0
        });
        let location = Some(response.url().to_string());

        Ok(FileMetadataInfo {
            filename,
            etag,
            commit_hash,
            xet_hash,
            file_size,
            location,
        })
    }
}

#[cfg(feature = "blocking")]
use futures::stream::StreamExt as _;

#[cfg(feature = "blocking")]
#[bon]
impl<T: RepoType> crate::blocking::HFRepositorySync<T> {
    /// Blocking counterpart of [`HFRepository::list_tree`]. Returns the collected stream as a
    /// `Vec<RepoTreeEntry>`. See the async method for parameters and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn list_tree(
        &self,
        #[builder(into)] revision: Option<String>,
        #[builder(into)] path_in_repo: Option<String>,
        #[builder(default)] recursive: bool,
        #[builder(default)] expand: bool,
        limit: Option<usize>,
    ) -> HFResult<Vec<RepoTreeEntry>> {
        self.runtime.block_on(async move {
            let stream = self
                .inner
                .list_tree()
                .maybe_revision(revision)
                .maybe_path_in_repo(path_in_repo)
                .recursive(recursive)
                .expand(expand)
                .maybe_limit(limit)
                .send()?;
            futures::pin_mut!(stream);
            let mut items = Vec::new();
            while let Some(item) = stream.next().await {
                items.push(item?);
            }
            Ok(items)
        })
    }

    /// Blocking counterpart of [`HFRepository::get_paths_info`]. See the async method for
    /// parameters and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn get_paths_info(
        &self,
        paths: Vec<String>,
        #[builder(into)] revision: Option<String>,
    ) -> HFResult<Vec<RepoTreeEntry>> {
        self.runtime.block_on(
            self.inner
                .get_paths_info()
                .paths(paths)
                .maybe_revision(revision)
                .send(),
        )
    }

    /// Blocking counterpart of [`HFRepository::get_file_metadata`]. See the async method for
    /// parameters and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn get_file_metadata(
        &self,
        #[builder(into)] filepath: String,
        revision: Option<&str>,
    ) -> HFResult<FileMetadataInfo> {
        self.runtime.block_on(
            self.inner
                .get_file_metadata()
                .filepath(filepath)
                .maybe_revision(revision)
                .send(),
        )
    }
}
