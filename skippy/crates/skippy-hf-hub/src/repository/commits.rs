//! Commit history, refs, and revision comparison helpers for repositories.
//!
//! This module groups three related workflows:
//!
//! - Use [`HFRepository::list_commits`] to walk commit history and [`HFRepository::list_refs`] to inspect branches,
//!   tags, convert refs, and optional pull-request refs.
//! - Use [`HFRepository::get_commit_diff`] for the Hub's non-raw compare payload as text.
//! - Use [`HFRepository::get_raw_diff`] for the full raw diff text, or [`HFRepository::get_raw_diff_stream`] to parse
//!   that raw diff incrementally into [`HFFileDiff`] entries.
//!
//! The same module also hosts branch and tag creation/deletion helpers because
//! they operate on the same repository revision namespace.

use bon::bon;
use futures::TryStreamExt;
use futures::stream::{Stream, StreamExt};
use serde::Deserialize;
use url::Url;

use super::diff::{self, HFFileDiff};
use super::{HFRepository, RepoType};
use crate::client::encode_ref;
use crate::error::HFResult;
use crate::{constants, retry};

/// Author entry attached to a commit, as returned by the commit history endpoint.
///
/// All fields are optional because the Hub only surfaces the identifying fields it has
/// (a linked Hub user, or the raw git name/email).
#[derive(Debug, Clone, Deserialize)]
pub struct CommitAuthor {
    /// Hub username, when the commit author is linked to a Hub account.
    pub user: Option<String>,
    /// Git author name as recorded on the commit.
    pub name: Option<String>,
    /// Git author email as recorded on the commit.
    pub email: Option<String>,
}

/// A single commit entry returned by the commit history endpoint.
///
/// Returned by [`HFRepository::list_commits`].
#[derive(Debug, Clone, Deserialize)]
pub struct GitCommitInfo {
    /// Full commit SHA.
    pub id: String,
    /// Commit authors as returned by the Hub.
    pub authors: Vec<CommitAuthor>,
    /// Commit timestamp in ISO 8601 format, when available.
    pub date: Option<String>,
    /// Commit title/summary line.
    pub title: String,
    /// Full commit message.
    pub message: String,
    /// HTML-formatted commit title. Returned only when the request asks the Hub to format the
    /// message (e.g., `?formatted=true`).
    #[serde(default, rename = "formattedTitle")]
    pub formatted_title: Option<String>,
    /// HTML-formatted commit message. Returned only when the request asks the Hub to format the
    /// message (e.g., `?formatted=true`).
    #[serde(default, rename = "formattedMessage")]
    pub formatted_message: Option<String>,
    /// Parent commit SHAs.
    #[serde(default)]
    pub parents: Vec<String>,
}

/// A single git ref (branch, tag, convert, or pull-request ref) and the commit it points to.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GitRefInfo {
    /// Short ref name such as `"main"` or `"v1.0.0"`.
    pub name: String,
    /// Full git ref name such as `"refs/heads/main"`.
    #[serde(rename = "ref")]
    pub git_ref: String,
    /// Commit SHA the ref currently points to.
    pub target_commit: String,
}

/// All git refs on a repository — branches, tags, converts, and pull-request refs.
///
/// Returned by [`HFRepository::list_refs`].
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GitRefs {
    /// Branch refs on the repository.
    pub branches: Vec<GitRefInfo>,
    /// Tag refs on the repository.
    pub tags: Vec<GitRefInfo>,
    /// Convert refs exposed by the Hub, when present.
    #[serde(default)]
    pub converts: Vec<GitRefInfo>,
    /// Pull-request refs, only populated when requested.
    #[serde(default, rename = "pullRequests")]
    pub pull_requests: Vec<GitRefInfo>,
}

/// A single file entry in the Hub's non-raw compare payload.
///
/// This type is useful if you want to deserialize the response body returned by
/// [`HFRepository::get_commit_diff`] yourself.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DiffEntry {
    /// Destination path for the changed file, when present.
    pub path: Option<String>,
    /// Previous path for renames or moves, when present.
    pub old_path: Option<String>,
    /// Hub-provided status string for the change.
    pub status: Option<String>,
}

#[bon]
impl<T: RepoType> HFRepository<T> {
    /// Stream commit history for the repository at a given revision.
    ///
    /// Returns `HFResult<impl Stream<Item = HFResult<GitCommitInfo>>>`. Pagination is automatic.
    ///
    /// # Parameters
    ///
    /// - `revision`: Git revision (branch, tag, or commit SHA). Defaults to the main branch.
    /// - `limit`: maximum number of commits yielded.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn list_commits(
        &self,
        /// Git revision (branch, tag, or commit SHA). Defaults to the main branch.
        #[builder(into)]
        revision: Option<String>,
        /// Maximum number of commits yielded.
        limit: Option<usize>,
    ) -> HFResult<impl Stream<Item = HFResult<GitCommitInfo>> + '_> {
        let revision = revision.as_deref().unwrap_or(constants::DEFAULT_REVISION);
        let url_str = format!(
            "{}/commits/{}",
            self.hf_client
                .api_url(self.repo_type.plural(), &self.repo_path()),
            encode_ref(revision)
        );
        let url = Url::parse(&url_str)?;
        Ok(self.hf_client.paginate(url, vec![], limit))
    }

    /// Fetch all branches, tags, and optionally pull-request refs for the repository.
    ///
    /// Endpoint: `GET /api/{repo_type}s/{repo_id}/refs`.
    ///
    /// # Parameters
    ///
    /// - `include_pull_requests` (default `false`): include pull-request refs in the listing.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn list_refs(
        &self,
        /// Include pull-request refs in the listing.
        #[builder(default)]
        include_pull_requests: bool,
    ) -> HFResult<GitRefs> {
        let url = format!(
            "{}/refs",
            self.hf_client
                .api_url(self.repo_type.plural(), &self.repo_path())
        );
        let mut query: Vec<(&str, String)> = Vec::new();
        if include_pull_requests {
            query.push(("include_prs", "1".into()));
        }

        let headers = self.hf_client.auth_headers();
        let response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client
                .http_client()
                .get(&url)
                .headers(headers.clone())
                .query(&query)
                .send()
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
        Ok(response.json().await?)
    }

    /// Fetch the Hub's non-raw compare payload as text.
    ///
    /// This returns the response body from the standard `/compare/{compare}` endpoint. Use
    /// [`HFRepository::get_raw_diff`] for raw git-style diff text or
    /// [`HFRepository::get_raw_diff_stream`] for parsed [`HFFileDiff`] entries.
    ///
    /// Endpoint: `GET /api/{repo_type}s/{repo_id}/compare/{compare}`.
    ///
    /// # Parameters
    ///
    /// - `compare` (required): revision spec describing what to compare. Either:
    ///   - A single revision (branch name, tag, or commit SHA), compared against its parent (e.g., `"main"`, `"v1.0"`,
    ///     `"abc123…"`), or
    ///   - Two revisions in `<base>..<head>` form (two dots), comparing `base` to `head` (e.g., `"main..feature"`,
    ///     `"<sha1>..<sha2>"`).
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn get_commit_diff(
        &self,
        /// Revision spec describing what to compare. Either:
        /// - A single revision (branch name, tag, or commit SHA), compared against its parent (e.g., `"main"`,
        ///   `"v1.0"`, `"abc123…"`), or
        /// - Two revisions in `<base>..<head>` form (two dots), comparing `base` to `head` (e.g., `"main..feature"`,
        ///   `"<sha1>..<sha2>"`).
        compare: &str,
    ) -> HFResult<String> {
        let url = format!(
            "{}/compare/{}",
            self.hf_client
                .api_url(self.repo_type.plural(), &self.repo_path()),
            encode_ref(compare)
        );

        let headers = self.hf_client.auth_headers();
        let response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client
                .http_client()
                .get(&url)
                .headers(headers.clone())
                .send()
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
        Ok(response.text().await?)
    }

    /// Fetch the raw diff payload between two revisions as a string.
    ///
    /// Prefer [`HFRepository::get_raw_diff_stream`] when you want file-level metadata without
    /// buffering the entire diff response in memory.
    ///
    /// Endpoint: `GET /api/{repo_type}s/{repo_id}/compare/{compare}?raw=true`.
    ///
    /// # Parameters
    ///
    /// - `compare` (required): revision spec describing what to compare. Either:
    ///   - A single revision (branch name, tag, or commit SHA), compared against its parent (e.g., `"main"`, `"v1.0"`,
    ///     `"abc123…"`), or
    ///   - Two revisions in `<base>..<head>` form (two dots), comparing `base` to `head` (e.g., `"main..feature"`,
    ///     `"<sha1>..<sha2>"`).
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn get_raw_diff(
        &self,
        /// Revision spec describing what to compare. Either:
        /// - A single revision (branch name, tag, or commit SHA), compared against its parent (e.g., `"main"`,
        ///   `"v1.0"`, `"abc123…"`), or
        /// - Two revisions in `<base>..<head>` form (two dots), comparing `base` to `head` (e.g., `"main..feature"`,
        ///   `"<sha1>..<sha2>"`).
        compare: &str,
    ) -> HFResult<String> {
        let url = format!(
            "{}/compare/{}",
            self.hf_client
                .api_url(self.repo_type.plural(), &self.repo_path()),
            encode_ref(compare)
        );

        let headers = self.hf_client.auth_headers();
        let response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client
                .http_client()
                .get(&url)
                .headers(headers.clone())
                .query(&[("raw", "true")])
                .send()
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
        Ok(response.text().await?)
    }

    /// Fetch the raw diff between two revisions as a parsed stream of [`HFFileDiff`] entries.
    ///
    /// Each `Ok` item is one parsed diff entry; malformed lines are `Err` items.
    ///
    /// Endpoint: `GET /api/{repo_type}s/{repo_id}/compare/{compare}?raw=true`.
    ///
    /// # Parameters
    ///
    /// - `compare` (required): revision spec describing what to compare. Either:
    ///   - A single revision (branch name, tag, or commit SHA), compared against its parent (e.g., `"main"`, `"v1.0"`,
    ///     `"abc123…"`), or
    ///   - Two revisions in `<base>..<head>` form (two dots), comparing `base` to `head` (e.g., `"main..feature"`,
    ///     `"<sha1>..<sha2>"`).
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn get_raw_diff_stream(
        &self,
        /// Revision spec describing what to compare. Either:
        /// - A single revision (branch name, tag, or commit SHA), compared against its parent (e.g., `"main"`,
        ///   `"v1.0"`, `"abc123…"`), or
        /// - Two revisions in `<base>..<head>` form (two dots), comparing `base` to `head` (e.g., `"main..feature"`,
        ///   `"<sha1>..<sha2>"`).
        #[builder(into)]
        compare: String,
    ) -> HFResult<impl Stream<Item = HFResult<HFFileDiff>> + '_> {
        let url = format!(
            "{}/compare/{}",
            self.hf_client
                .api_url(self.repo_type.plural(), &self.repo_path()),
            encode_ref(&compare)
        );

        let headers = self.hf_client.auth_headers();
        let response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client
                .http_client()
                .get(&url)
                .headers(headers.clone())
                .query(&[("raw", "true")])
                .send()
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
        let byte_stream = response
            .bytes_stream()
            .map(|r| r.map_err(std::io::Error::other));
        Ok(diff::stream_raw_diff(byte_stream).map_err(Into::into))
    }

    /// Create a new branch, optionally starting from a specific revision.
    ///
    /// Endpoint: `POST /api/{repo_type}s/{repo_id}/branch/{branch}`.
    ///
    /// # Parameters
    ///
    /// - `branch` (required): name of the branch to create.
    /// - `revision`: revision to branch from. Defaults to the current main branch head.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn create_branch(
        &self,
        /// Name of the branch to create.
        branch: &str,
        /// Revision to branch from. Defaults to the current main branch head.
        #[builder(into)]
        revision: Option<String>,
    ) -> HFResult<()> {
        let url = format!(
            "{}/branch/{}",
            self.hf_client
                .api_url(self.repo_type.plural(), &self.repo_path()),
            encode_ref(branch)
        );

        let mut body = serde_json::Map::new();
        if let Some(rev) = revision {
            body.insert("startingPoint".into(), serde_json::Value::String(rev));
        }

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
        self.hf_client
            .check_response(
                response,
                Some(&repo_path),
                crate::error::NotFoundContext::Repo,
            )
            .await?;
        Ok(())
    }

    /// Delete a branch from the repository.
    ///
    /// Endpoint: `DELETE /api/{repo_type}s/{repo_id}/branch/{branch}`.
    ///
    /// # Parameters
    ///
    /// - `branch` (required): name of the branch to delete.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn delete_branch(
        &self,
        /// Name of the branch to delete.
        branch: &str,
    ) -> HFResult<()> {
        let url = format!(
            "{}/branch/{}",
            self.hf_client
                .api_url(self.repo_type.plural(), &self.repo_path()),
            encode_ref(branch)
        );

        let headers = self.hf_client.auth_headers();
        let response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client
                .http_client()
                .delete(&url)
                .headers(headers.clone())
                .send()
        })
        .await?;

        let repo_path = self.repo_path();
        self.hf_client
            .check_response(
                response,
                Some(&repo_path),
                crate::error::NotFoundContext::Repo,
            )
            .await?;
        Ok(())
    }

    /// Create a lightweight or annotated tag, optionally at a specific revision.
    ///
    /// Endpoint: `POST /api/{repo_type}s/{repo_id}/tag/{revision}`.
    ///
    /// # Parameters
    ///
    /// - `tag` (required): name of the tag to create.
    /// - `revision`: revision to tag. Defaults to the current main branch head.
    /// - `message`: annotation message for the tag.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn create_tag(
        &self,
        /// Name of the tag to create.
        tag: &str,
        /// Revision to tag. Defaults to the current main branch head.
        revision: Option<&str>,
        /// Annotation message for the tag.
        #[builder(into)]
        message: Option<String>,
    ) -> HFResult<()> {
        let revision = revision.unwrap_or(constants::DEFAULT_REVISION);
        let url = format!(
            "{}/tag/{}",
            self.hf_client
                .api_url(self.repo_type.plural(), &self.repo_path()),
            encode_ref(revision)
        );

        let mut body = serde_json::json!({ "tag": tag });
        if let Some(m) = message {
            body["message"] = serde_json::Value::String(m);
        }

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
        self.hf_client
            .check_response(
                response,
                Some(&repo_path),
                crate::error::NotFoundContext::Repo,
            )
            .await?;
        Ok(())
    }

    /// Delete a tag from the repository.
    ///
    /// Endpoint: `DELETE /api/{repo_type}s/{repo_id}/tag/{tag}`.
    ///
    /// # Parameters
    ///
    /// - `tag` (required): name of the tag to delete.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn delete_tag(
        &self,
        /// Name of the tag to delete.
        tag: &str,
    ) -> HFResult<()> {
        let url = format!(
            "{}/tag/{}",
            self.hf_client
                .api_url(self.repo_type.plural(), &self.repo_path()),
            encode_ref(tag)
        );

        let headers = self.hf_client.auth_headers();
        let response = retry::retry(self.hf_client.retry_config(), || {
            self.hf_client
                .http_client()
                .delete(&url)
                .headers(headers.clone())
                .send()
        })
        .await?;

        let repo_path = self.repo_path();
        self.hf_client
            .check_response(
                response,
                Some(&repo_path),
                crate::error::NotFoundContext::Repo,
            )
            .await?;
        Ok(())
    }
}

#[cfg(feature = "blocking")]
#[bon]
impl<T: RepoType> crate::blocking::HFRepositorySync<T> {
    /// Blocking counterpart of [`HFRepository::list_commits`]. Collects the stream into a
    /// `Vec<GitCommitInfo>`. See the async method for parameters and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn list_commits(
        &self,
        #[builder(into)] revision: Option<String>,
        limit: Option<usize>,
    ) -> HFResult<Vec<GitCommitInfo>> {
        self.runtime.block_on(async move {
            let stream = self
                .inner
                .list_commits()
                .maybe_revision(revision)
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

    /// Blocking counterpart of [`HFRepository::list_refs`]. See the async method for parameters
    /// and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn list_refs(&self, #[builder(default)] include_pull_requests: bool) -> HFResult<GitRefs> {
        self.runtime.block_on(
            self.inner
                .list_refs()
                .include_pull_requests(include_pull_requests)
                .send(),
        )
    }

    /// Blocking counterpart of [`HFRepository::get_commit_diff`]. See the async method for
    /// parameters and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn get_commit_diff(&self, compare: &str) -> HFResult<String> {
        self.runtime
            .block_on(self.inner.get_commit_diff().compare(compare).send())
    }

    /// Blocking counterpart of [`HFRepository::get_raw_diff`]. See the async method for
    /// parameters and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn get_raw_diff(&self, compare: &str) -> HFResult<String> {
        self.runtime
            .block_on(self.inner.get_raw_diff().compare(compare).send())
    }

    /// Blocking counterpart of [`HFRepository::get_raw_diff_stream`]. Collects the parsed stream
    /// into a `Vec<HFFileDiff>`. See the async method for parameters and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn get_raw_diff_stream(
        &self,
        #[builder(into)] compare: String,
    ) -> HFResult<Vec<HFFileDiff>> {
        self.runtime.block_on(async move {
            let stream = self
                .inner
                .get_raw_diff_stream()
                .compare(compare)
                .send()
                .await?;
            futures::pin_mut!(stream);
            let mut items = Vec::new();
            while let Some(item) = stream.next().await {
                items.push(item?);
            }
            Ok(items)
        })
    }

    /// Blocking counterpart of [`HFRepository::create_branch`]. See the async method for
    /// parameters and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn create_branch(
        &self,
        branch: &str,
        #[builder(into)] revision: Option<String>,
    ) -> HFResult<()> {
        self.runtime.block_on(
            self.inner
                .create_branch()
                .branch(branch)
                .maybe_revision(revision)
                .send(),
        )
    }

    /// Blocking counterpart of [`HFRepository::delete_branch`]. See the async method for
    /// parameters and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn delete_branch(&self, branch: &str) -> HFResult<()> {
        self.runtime
            .block_on(self.inner.delete_branch().branch(branch).send())
    }

    /// Blocking counterpart of [`HFRepository::create_tag`]. See the async method for parameters
    /// and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn create_tag(
        &self,
        tag: &str,
        revision: Option<&str>,
        #[builder(into)] message: Option<String>,
    ) -> HFResult<()> {
        self.runtime.block_on(
            self.inner
                .create_tag()
                .tag(tag)
                .maybe_revision(revision)
                .maybe_message(message)
                .send(),
        )
    }

    /// Blocking counterpart of [`HFRepository::delete_tag`]. See the async method for parameters
    /// and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn delete_tag(&self, tag: &str) -> HFResult<()> {
        self.runtime
            .block_on(self.inner.delete_tag().tag(tag).send())
    }
}

#[cfg(test)]
mod tests {
    use super::GitCommitInfo;

    #[test]
    fn test_git_commit_info_with_formatted_fields() {
        let json = r#"{
            "id":"abc123",
            "authors":[{"user":"u","name":"User","email":"u@x"}],
            "date":"2025-01-01T00:00:00Z",
            "title":"feat: add thing",
            "message":"feat: add thing\n\nLong body.",
            "formattedTitle":"<p>feat: add thing</p>",
            "formattedMessage":"<p>feat: add thing</p><p>Long body.</p>",
            "parents":["p1"]
        }"#;
        let info: GitCommitInfo = serde_json::from_str(json).unwrap();
        assert_eq!(
            info.formatted_title.as_deref(),
            Some("<p>feat: add thing</p>")
        );
        assert_eq!(
            info.formatted_message.as_deref(),
            Some("<p>feat: add thing</p><p>Long body.</p>")
        );
        assert_eq!(info.parents, vec!["p1"]);
    }

    #[test]
    fn test_git_commit_info_without_formatted_fields() {
        let json = r#"{"id":"abc","authors":[],"date":null,"title":"t","message":"m"}"#;
        let info: GitCommitInfo = serde_json::from_str(json).unwrap();
        assert!(info.formatted_title.is_none());
        assert!(info.formatted_message.is_none());
    }
}
