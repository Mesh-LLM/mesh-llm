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
        // As in huggingface_hub, only same-origin redirects are followed (a
        // renamed repo, `/api/resolve-cache/`). A Xet- or LFS-backed file
        // answers with a redirect to its CDN, and only that redirect carries
        // `X-Repo-Commit` and the linked ETag and size: the CDN's own answer
        // has none of them.
        let mut url = url;
        let mut followed = 0;
        let response = loop {
            let response = retry::retry(self.hf_client.retry_config(), || {
                head_client(&self.hf_client)
                    .head(&url)
                    .headers(headers.clone())
                    .send()
            })
            .await?;
            match same_host_redirect(&url, &response) {
                Some(next) if followed < MAX_SAME_HOST_REDIRECTS => {
                    url = next;
                    followed += 1;
                }
                _ => break response,
            }
        };
        let location = if response.status().is_redirection() {
            redirect_target(&url, &response)
        } else {
            Some(response.url().to_string())
        };
        let response = if response.status().is_redirection() {
            response
        } else {
            self.hf_client
                .check_response(
                    response,
                    Some(&repo_path),
                    crate::error::NotFoundContext::Entry {
                        path: filename.clone(),
                    },
                )
                .await?
        };

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

/// Same-host redirects [`HFRepository::get_file_metadata`] follows before it
/// reads the metadata.
const MAX_SAME_HOST_REDIRECTS: usize = 5;

/// The client for metadata HEADs, which leaves redirects to the caller. The
/// wasm backend follows them itself whatever it is told, so it has no such
/// client.
#[cfg(not(target_family = "wasm"))]
fn head_client(client: &crate::HFClient) -> &reqwest::Client {
    client.no_redirect_client()
}

#[cfg(target_family = "wasm")]
fn head_client(client: &crate::HFClient) -> &reqwest::Client {
    client.http_client()
}

/// Where `response` redirects to, if that keeps the scheme, host and port of
/// `url`: the request carries the Hub token, which no other origin may get.
/// `None` for any other answer, including a redirect to the Xet or LFS CDN.
/// The check is on the resolved URL, not the `Location` text: `\\host/path`
/// reads as relative but resolves to another host.
fn same_host_redirect(url: &str, response: &reqwest::Response) -> Option<String> {
    let current = Url::parse(url).ok()?;
    let next = current.join(location_header(response)?).ok()?;
    (next.origin() == current.origin()).then(|| next.into())
}

/// Where a redirect answered for `url` points.
fn redirect_target(url: &str, response: &reqwest::Response) -> Option<String> {
    let location = location_header(response)?;
    Url::parse(url).ok()?.join(location).ok().map(String::from)
}

fn location_header(response: &reqwest::Response) -> Option<&str> {
    if !response.status().is_redirection() {
        return None;
    }
    response
        .headers()
        .get(reqwest::header::LOCATION)?
        .to_str()
        .ok()
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

#[cfg(test)]
mod tests {
    use super::same_host_redirect;
    use crate::HFClientBuilder;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    const COMMIT: &str = "0123456789012345678901234567890123456789";

    /// Answers each request with the status line and headers mapped to its
    /// path, or 404, as the Hub and its CDN do for a HEAD.
    fn serve(listener: tokio::net::TcpListener, answers: Vec<(String, String)>) {
        tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    return;
                };
                let answers = answers.clone();
                tokio::spawn(async move {
                    let mut request = vec![0; 4096];
                    let Ok(read) = socket.read(&mut request).await else {
                        return;
                    };
                    let request = String::from_utf8_lossy(&request[..read]);
                    let path = request.split_whitespace().nth(1).unwrap_or_default();
                    let head = answers.iter().find(|(at, _)| at == path).map_or(
                        "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n",
                        |(_, head)| head,
                    );
                    let _ = socket
                        .write_all(format!("{head}Connection: close\r\n\r\n").as_bytes())
                        .await;
                });
            }
        });
    }

    async fn hub() -> (tokio::net::TcpListener, String) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        (listener, endpoint)
    }

    #[tokio::test]
    async fn metadata_comes_from_the_hub_redirect_not_the_cdn_answer() {
        let (listener, endpoint) = hub().await;
        // The same server under another origin, as the Xet CDN is.
        let cdn = format!(
            "{}/xet-bridge/blob",
            endpoint.replace("127.0.0.1", "localhost")
        );
        serve(
            listener,
            vec![
                (
                    "/owner/repo/resolve/main/model.gguf".into(),
                    format!(
                        "HTTP/1.1 302 Found\r\nLocation: {cdn}\r\nX-Repo-Commit: {COMMIT}\r\n\
                         X-Linked-ETag: \"sha\"\r\nX-Linked-Size: 5680522464\r\n\
                         X-Xet-Hash: xet\r\nContent-Length: 0\r\n"
                    ),
                ),
                // A Xet CDN answer carries none of the Hub's headers.
                (
                    "/xet-bridge/blob".into(),
                    "HTTP/1.1 200 OK\r\nContent-Length: 5680522464\r\n".into(),
                ),
            ],
        );
        let client = HFClientBuilder::new().endpoint(endpoint).build().unwrap();

        let info = client
            .model("owner", "repo")
            .get_file_metadata()
            .filepath("model.gguf")
            .send()
            .await
            .unwrap();

        assert_eq!(info.commit_hash, COMMIT);
        assert_eq!(info.etag, "sha");
        assert_eq!(info.file_size, 5_680_522_464);
        assert_eq!(info.xet_hash.as_deref(), Some("xet"));
        assert_eq!(info.location.as_deref(), Some(cdn.as_str()));
    }

    #[tokio::test]
    async fn a_same_host_redirect_is_followed_to_the_file() {
        let (listener, endpoint) = hub().await;
        let cached = format!("/api/resolve-cache/models/owner/repo/{COMMIT}/README.md");
        serve(
            listener,
            vec![
                // The redirect's Content-Length is its own body, not the file's.
                (
                    "/owner/repo/resolve/main/README.md".into(),
                    format!(
                        "HTTP/1.1 307 Temporary Redirect\r\nLocation: {cached}\r\n\
                         X-Repo-Commit: {COMMIT}\r\nX-Linked-ETag: \"readme\"\r\n\
                         Content-Length: 254\r\n"
                    ),
                ),
                (
                    cached.clone(),
                    format!(
                        "HTTP/1.1 200 OK\r\nETag: \"readme\"\r\nX-Repo-Commit: {COMMIT}\r\n\
                         Content-Length: 28503\r\n"
                    ),
                ),
            ],
        );
        let client = HFClientBuilder::new()
            .endpoint(endpoint.clone())
            .build()
            .unwrap();

        let info = client
            .model("owner", "repo")
            .get_file_metadata()
            .filepath("README.md")
            .send()
            .await
            .unwrap();

        assert_eq!(info.commit_hash, COMMIT);
        assert_eq!(info.etag, "readme");
        assert_eq!(info.file_size, 28_503);
        assert_eq!(info.location, Some(format!("{endpoint}{cached}")));
    }

    #[test]
    fn only_a_same_origin_redirect_is_followed() {
        let at = "https://huggingface.co/owner/repo/resolve/main/README.md";
        let redirect = |location: &str| {
            reqwest::Response::from(
                http::Response::builder()
                    .status(307)
                    .header("location", location)
                    .body("")
                    .unwrap(),
            )
        };
        assert_eq!(
            same_host_redirect(at, &redirect("/api/resolve-cache/x")).as_deref(),
            Some("https://huggingface.co/api/resolve-cache/x")
        );
        assert_eq!(
            same_host_redirect(at, &redirect("https://huggingface.co/api/x")).as_deref(),
            Some("https://huggingface.co/api/x")
        );
        // Each of these reaches another origin, which must not get the token;
        // `\\host` reads as relative but resolves to another host.
        for elsewhere in [
            "https://cdn-lfs.hf.co/x",
            "//cdn-lfs.hf.co/x",
            "\\\\evil.example/x",
            "http://huggingface.co/x",
            "https://huggingface.co:8443/x",
        ] {
            assert_eq!(
                same_host_redirect(at, &redirect(elsewhere)),
                None,
                "{elsewhere}"
            );
        }
    }

    #[tokio::test]
    async fn metadata_requests_keep_the_supplied_transport_settings() {
        let (listener, endpoint) = hub().await;
        serve(
            listener,
            vec![(
                "/owner/repo/resolve/main/model.gguf".into(),
                format!(
                    "HTTP/1.1 200 OK\r\nETag: \"e\"\r\nX-Repo-Commit: {COMMIT}\r\n\
                     Content-Length: 1\r\n"
                ),
            )],
        );
        // The local Hub is plain HTTP, so an HTTPS-only client must refuse it,
        // including for the metadata HEAD that bypasses the main client.
        let client = HFClientBuilder::new()
            .endpoint(endpoint)
            .retry_max_attempts(0)
            .client_builder(|| reqwest::Client::builder().https_only(true))
            .build()
            .unwrap();

        let result = client
            .model("owner", "repo")
            .get_file_metadata()
            .filepath("model.gguf")
            .send()
            .await;

        assert!(result.is_err(), "{result:?}");
    }
}
